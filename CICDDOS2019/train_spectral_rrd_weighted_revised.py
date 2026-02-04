import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

from data_loader_2019 import load_cicddos2019
from model_2019 import BiGRUTeacher, BiGRUStudent

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "spectral_rrd_weighted_learnable_model.pt")  # Changed name
LOG_PATH = os.path.join(BASE_DIR, "logs", "Feedback_KD", "spectral_rrd_weighted_learnable_history.json")  # Changed name
TEACHER_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "teacher_model.pt")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


# === Learning Rate Schedule ===
def lr_schedule20(epoch):
    if epoch < 5:
        return 1e-3
    elif epoch < 10:
        return 5e-4
    elif epoch < 15:
        return 1e-4
    elif epoch < 18:
        return 5e-5
    else:
        return 1e-5


# --- add imports ---
import os, random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# === Stable QR-based Weighted Spectral RRD ===
def weighted_spectral_project(embed, k=30):
    """
    Projects embeddings to dominant subspace using QR decomposition
    after a single power iteration. Avoids torch.svd() instability on ill-conditioned matrices.
    """
    embed = F.normalize(embed, p=2, dim=1)
    B, D = embed.shape
    V = torch.randn(D, k, device=embed.device)

    # One round power iteration (can be repeated if needed)
    V = embed.T @ (embed @ V)

    # QR orthonormalization
    V, _ = torch.linalg.qr(V)
    projected = embed @ V

    # Weighting: L2-norm of each component vector
    weights = projected.norm(p=2, dim=0)
    weights = weights / (weights.sum() + 1e-8)
    return projected, weights


def gram_k(Z):
    Zc = Z - Z.mean(dim=0, keepdim=True)
    G = Zc.T @ Zc
    G = G / (Zc.shape[0] + 1e-8)
    return G


def fro_norm(G):
    return G / (G.norm(p='fro') + 1e-8)


# === UPDATED: Learnable Weight Loss Function ===
def weighted_spectral_rrd_loss(student_embed, teacher_embed, learnable_w, alpha=0.7, k=30):
    B, D = student_embed.shape
    k_eff = max(1, min(k, D))

    try:
        s_proj, _ = weighted_spectral_project(student_embed, k_eff)
        t_proj, _ = weighted_spectral_project(teacher_embed, k_eff)

        if torch.any(torch.isnan(s_proj)) or torch.any(torch.isnan(t_proj)):
            raise RuntimeError("NaNs in spectral projection")

        # === LEARNABLE WEIGHTS WITH L2 NORMALIZATION ===
        # w = F.normalize(learnable_w[:k_eff], p=2, dim=0)  # L2 normalization
        # w = torch.abs(w) + 1e-8  # Ensure positivity
        w = torch.sigmoid(learnable_w[:k_eff]) + 1e-8  # [k_eff]
        # Apply weights
        s_w = s_proj * w
        t_w = t_proj * w

        Gs = fro_norm(gram_k(s_w))
        Gt = fro_norm(gram_k(t_w))

        # --- DEBUG: print once ---
        if not hasattr(weighted_spectral_rrd_loss, "_printed"):
            print(f"[learnable weighted] k={k} Gram shapes: student={Gs.shape}, teacher={Gt.shape}")
            print(f"Weight stats: mean={w.mean().item():.4f}, std={w.std().item():.4f}, "
                  f"min={w.min().item():.4f}, max={w.max().item():.4f}")
            weighted_spectral_rrd_loss._printed = True

        loss = torch.mean((Gs - Gt) ** 2)
        return alpha * loss, loss.detach().item()

    except Exception as e:
        print(f"[WARN][weighted_spectral_rrd_loss] Falling back to pairwise cosine: {e}")
        s_sim = F.cosine_similarity(student_embed.unsqueeze(1), student_embed.unsqueeze(0), dim=2)
        t_sim = F.cosine_similarity(teacher_embed.unsqueeze(1), teacher_embed.unsqueeze(0), dim=2)
        loss = F.mse_loss(s_sim, t_sim)
        return alpha * loss, loss.detach().item()


# === Evaluation ===
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total, total_loss / len(loader)


# === UPDATED: Train Function with Learnable Weights ===
def train_spectral_rrd_weighted_learnable(k: int = 30, seed: int = 1337, epoch=3):
    set_seed(seed)
    print(f" Training Weighted Spectral RRD Student (Learnable Weights)...k-{k}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_cicddos2019(batch_size=256)

    # === Load teacher ===
    teacher = BiGRUTeacher().to(device)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    teacher.eval()

    # === Init student ===
    student = BiGRUStudent(hidden_dim=64, num_classes=8).to(device)

    # === LEARNABLE WEIGHT PARAMETER ===
    learnable_w = nn.Parameter(torch.ones(k, device=device) / k)  # Initialize uniformly

    # === OPTIMIZER WITH L2 REGULARIZATION ON WEIGHTS ===
    optimizer = torch.optim.Adam([
        {'params': student.parameters()},
        {'params': [learnable_w], 'weight_decay': 1e-4}  # L2 regularization on weights
    ], lr=1e-3)

    criterion = torch.nn.CrossEntropyLoss()

    history = {
        "train_acc": [], "val_acc": [],
        "ce_loss": [], "weighted_rrd_loss": [], "total_loss": [],
        "val_loss": [], "lr_used": [],
        "epoch_time_sec": [], "gpu_peak_mem_mb": [],
        "weight_mean": [], "weight_std": [],  # Track weight statistics
        "time_sec": 0
    }

    start_time = time.time()
    print(f"🕒 Training started on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(epoch):
        lr = lr_schedule20(epoch)
        for g in optimizer.param_groups:
            g["lr"] = lr
        history["lr_used"].append(lr)
        epoch_start = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        student.train()
        total_loss = total_ce = total_wrrd = correct = total = 0

        for x, y in tqdm(train_loader, desc=f"Weighted Spectral Epoch {epoch + 1}/{epoch}", ncols=100):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            student_logits, student_embed = student(x)
            with torch.no_grad():
                teacher_embed = teacher.extract_features(x)

            ce = criterion(student_logits, y)

            # === PASS LEARNABLE WEIGHTS TO LOSS ===
            wrrd, wrrd_val = weighted_spectral_rrd_loss(
                student_embed, teacher_embed,
                learnable_w,  # ← PASS LEARNABLE WEIGHTS HERE
                alpha=0.7, k=k
            )

            loss = ce + wrrd
            loss.backward()
            optimizer.step()

            preds = torch.argmax(student_logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            total_loss += loss.item()
            total_ce += ce.item()
            total_wrrd += wrrd_val

        # === TRACK WEIGHT STATISTICS ===
        with torch.no_grad():
            w_actual = torch.sigmoid(learnable_w) + 1e-8  # Track actual used weights
            history["weight_mean"].append(w_actual.mean().item())
            history["weight_std"].append(w_actual.std().item())

        train_acc = correct / total
        val_acc, val_loss = evaluate_loss(student, test_loader, criterion, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            gpu_peak_mb = None
        history["gpu_peak_mem_mb"].append(gpu_peak_mb)
        history["epoch_time_sec"].append(time.time() - epoch_start)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["ce_loss"].append(total_ce / len(train_loader))
        history["weighted_rrd_loss"].append(total_wrrd / len(train_loader))
        history["total_loss"].append(total_loss / len(train_loader))
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch + 1} - Train Loss: {total_loss / len(train_loader):.4f} | "
              f"CE: {total_ce / len(train_loader):.4f} | WRRD: {total_wrrd / len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Weights: mean={history['weight_mean'][-1]:.4f}, std={history['weight_std'][-1]:.4f} | "
              f"LR: {lr:.6f}")

    history["time_sec"] = time.time() - start_time
    print(f"\n⏱️ Weighted Spectral RRD (Learnable Weights) training complete in {history['time_sec']:.2f}s.")

    # Save with learnable suffix
    model_path = MODEL_PATH.replace(".pt", f"_k{k}_learnable.pt")
    log_path = LOG_PATH.replace(".json", f"_k{k}_learnable.json")
    torch.save({
        'student_state_dict': student.state_dict(),
        'learnable_weights': learnable_w.detach().cpu()  # Save weights too
    }, model_path)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"✅ Saved Weighted Spectral RRD (Learnable) model to {model_path}")
    print(f"📁 Training log saved to {log_path}")


# === Run ===
if __name__ == "__main__":
    train_spectral_rrd_weighted_learnable(k=40, epoch=5)  # Test with 5 epochs


### older results
# D:\programs\Anaconda3\envs\IDS2025\python.exe "D:\Miftah_Data\IDS_Project -2019\train_variants\train_spectral_rrd_weighted_revised.py"
#  Training Weighted Spectral RRD Student (Learnable Weights)...k-40
# Weighted Spectral Epoch 1/0:   0%|                                        | 0/10745 [00:00<?, ?it/s]🕒 Training started on 2026-02-02 12:53:46
# D:\programs\Anaconda3\envs\IDS2025\lib\site-packages\torch\nn\modules\linear.py:116: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\Context.cpp:164.)
#   return F.linear(input, self.weight, self.bias)
# D:\Miftah_Data\IDS_Project -2019\train_variants\train_spectral_rrd_weighted_revised.py:66: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\Context.cpp:164.)
#   V = embed.T @ (embed @ V)
# D:\Miftah_Data\IDS_Project -2019\train_variants\train_spectral_rrd_weighted_revised.py:70: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\Context.cpp:164.)
#   projected = embed @ V
# [learnable weighted] k=40 Gram shapes: student=torch.Size([40, 40]), teacher=torch.Size([40, 40])
# D:\Miftah_Data\IDS_Project -2019\train_variants\train_spectral_rrd_weighted_revised.py:80: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\Context.cpp:164.)
#   G = Zc.T @ Zc
# Weight stats: mean=0.1581, std=0.0000, min=0.1581, max=0.1581
# D:\programs\Anaconda3\envs\IDS2025\lib\site-packages\torch\autograd\__init__.py:266: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\Context.cpp:164.)
#   Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
# Weighted Spectral Epoch 1/0: 100%|███████████████████████████| 10745/10745 [01:44<00:00, 102.98it/s]
# Epoch 1 - Train Loss: 0.2421 | CE: 0.2421 | WRRD: 0.0000 | Train Acc: 0.9151 | Val Loss: 0.0518 | Val Acc: 0.9820 | Weights: mean=0.0236, std=0.1583 | LR: 0.001000
# Weighted Spectral Epoch 2/1: 100%|███████████████████████████| 10745/10745 [01:44<00:00, 103.12it/s]
# Epoch 2 - Train Loss: 0.0679 | CE: 0.0679 | WRRD: 0.0000 | Train Acc: 0.9823 | Val Loss: 0.0285 | Val Acc: 0.9928 | Weights: mean=0.0250, std=0.1581 | LR: 0.001000
# Weighted Spectral Epoch 3/2: 100%|███████████████████████████| 10745/10745 [01:44<00:00, 102.88it/s]
# Weighted Spectral Epoch 4/3:   0%|                                        | 0/10745 [00:00<?, ?it/s]Epoch 3 - Train Loss: 0.0290 | CE: 0.0290 | WRRD: 0.0000 | Train Acc: 0.9924 | Val Loss: 0.0338 | Val Acc: 0.9881 | Weights: mean=0.0250, std=0.1581 | LR: 0.001000
# Weighted Spectral Epoch 4/3: 100%|███████████████████████████| 10745/10745 [01:42<00:00, 104.72it/s]
# Epoch 4 - Train Loss: 0.2031 | CE: 0.2031 | WRRD: 0.0000 | Train Acc: 0.9285 | Val Loss: 0.0860 | Val Acc: 0.9719 | Weights: mean=0.0231, std=0.1584 | LR: 0.001000
# Weighted Spectral Epoch 5/4: 100%|███████████████████████████| 10745/10745 [01:41<00:00, 106.26it/s]
# Epoch 5 - Train Loss: 0.0458 | CE: 0.0458 | WRRD: 0.0000 | Train Acc: 0.9877 | Val Loss: 0.0186 | Val Acc: 0.9938 | Weights: mean=0.0248, std=0.1581 | LR: 0.001000
#
# ⏱️ Weighted Spectral RRD (Learnable Weights) training complete in 525.52s.
# ✅ Saved Weighted Spectral RRD (Learnable) model to D:\Miftah_Data\IDS_Project -2019\models\Feedback_KD\spectral_rrd_weighted_learnable_model_k40_learnable.pt
# 📁 Training log saved to D:\Miftah_Data\IDS_Project -2019\logs\Feedback_KD\spectral_rrd_weighted_learnable_history_k40_learnable.json
#
# Process finished with exit code 0
