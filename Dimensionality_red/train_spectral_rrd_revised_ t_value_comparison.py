import os
import time
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import numpy as np
import random

from data_loader_2019 import load_cicddos2019
from model_2019 import BiGRUTeacher, BiGRUStudent

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEACHER_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "teacher_model.pt")


def set_seed(seed: int = 1337):
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


# === MODIFIED: spectral_project with t parameter ===
def spectral_project(embeddings, k=30, t=1, verbose=False):
    """
    Robust spectral projection with t iterations.
    """
    # Center first
    embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    embeddings_norm = F.normalize(embeddings_centered, p=2, dim=1)

    _, D = embeddings_norm.shape
    # Random orthonormal initialization
    V = torch.randn(D, k, device=embeddings_norm.device)
    V, _ = torch.linalg.qr(V)

    # Power iteration with QR at each step
    for i in range(t):
        V = embeddings_norm.T @ (embeddings_norm @ V)
        V, _ = torch.linalg.qr(V)

    # Optional verification
    if verbose:
        orth_error = torch.norm(V.T @ V - torch.eye(k, device=V.device))
        if orth_error > 1e-6:
            print(f"[t={t}] Orthogonality error: {orth_error:.2e}")

    return embeddings_norm @ V


def gram_k(Z):
    G = Z.T @ Z
    G = G / (Z.shape[0] + 1e-8)
    return G


def fro_norm(G):
    return G / (G.norm(p='fro') + 1e-8)


# === MODIFIED: spectral_rrd_loss with t parameter ===
def spectral_rrd_loss(student_embed, teacher_embed, alpha=0.7, k=30, t=1):
    B, D = student_embed.shape
    k_eff = max(1, min(k, D))

    s_proj = spectral_project(student_embed, k_eff, t)
    t_proj = spectral_project(teacher_embed, k_eff, t)

    Gs = fro_norm(gram_k(s_proj))
    Gt = fro_norm(gram_k(t_proj))

    loss = torch.mean((Gs - Gt) ** 2)
    return alpha * loss, loss.detach().item()


# === NEW: Function to test different t values ===
def test_t_values(k=40, test_epochs=3):
    """
    Compare different t values (1, 2, 3 iterations)
    """
    print("\n" + "=" * 60)
    print(f"TESTING POWER ITERATION VALUES (k={k}, epochs={test_epochs})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_cicddos2019(batch_size=256)

    # Load teacher
    teacher = BiGRUTeacher().to(device)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    teacher.eval()

    results = {}

    for t in [1, 2, 3]:
        print(f"\n🔬 Testing t = {t}")
        set_seed(1337)  # Same seed for fair comparison

        # Initialize fresh student
        student = BiGRUStudent(hidden_dim=64, num_classes=8).to(device)
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        train_times = []
        val_accs = []
        spectral_losses = []

        for epoch in range(test_epochs):
            epoch_start = time.time()
            student.train()
            total_loss = total_ce = total_spec = correct = total = 0

            for x, y in tqdm(train_loader, desc=f"t={t} Epoch {epoch + 1}/{test_epochs}", ncols=80, leave=False):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                student_logits, student_embed = student(x)
                with torch.no_grad():
                    teacher_embed = teacher.extract_features(x)

                ce = criterion(student_logits, y)
                spectral, spectral_val = spectral_rrd_loss(student_embed, teacher_embed,
                                                           alpha=0.7, k=k, t=t)
                loss = ce + spectral
                loss.backward()
                optimizer.step()

                preds = torch.argmax(student_logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                total_loss += loss.item()
                total_ce += ce.item()
                total_spec += spectral_val

            # Validation
            student.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits, _ = student(xb)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == yb).sum().item()
                    val_total += yb.size(0)

            train_acc = correct / total
            val_acc = val_correct / val_total
            epoch_time = time.time() - epoch_start

            train_times.append(epoch_time)
            val_accs.append(val_acc)
            spectral_losses.append(total_spec / len(train_loader))

            print(f"  Epoch {epoch + 1}: Val Acc={val_acc:.4f}, Time={epoch_time:.1f}s, "
                  f"Spectral Loss={spectral_losses[-1]:.6f}")

        # Store results
        results[t] = {
            'final_val_acc': val_accs[-1],
            'avg_epoch_time': np.mean(train_times),
            'avg_spectral_loss': np.mean(spectral_losses),
            'val_acc_history': val_accs
        }

    # Print comparison table
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'t':<4} {'Final Val Acc':<12} {'Avg Epoch Time':<14} {'Speed vs t=1':<12}")
    print("-" * 60)

    t1_time = results[1]['avg_epoch_time']
    for t in [1, 2, 3]:
        speedup = t1_time / results[t]['avg_epoch_time']
        print(f"{t:<4} {results[t]['final_val_acc']:.4f}       "
              f"{results[t]['avg_epoch_time']:.1f}s          "
              f"{speedup:.2f}x")

    return results


# === Main execution ===
if __name__ == "__main__":
    print("Starting t-value comparison test...")
    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Test with k=40 (your optimal)
    results = test_t_values(k=40, test_epochs=3)

    # Save results
    results_path = os.path.join(BASE_DIR, "logs", "Feedback_KD", "t_value_test.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {results_path}")

    # Quick analysis
    print("\n📊 ANALYSIS:")
    acc_diff_1_2 = abs(results[1]['final_val_acc'] - results[2]['final_val_acc'])
    acc_diff_1_3 = abs(results[1]['final_val_acc'] - results[3]['final_val_acc'])

    print(f"Accuracy difference t=1 vs t=2: {acc_diff_1_2:.4f}")
    print(f"Accuracy difference t=1 vs t=3: {acc_diff_1_3:.4f}")
    print(f"t=2 is {results[2]['avg_epoch_time'] / results[1]['avg_epoch_time']:.1f}x slower than t=1")
    print(f"t=3 is {results[3]['avg_epoch_time'] / results[1]['avg_epoch_time']:.1f}x slower than t=1")

# D:\programs\Anaconda3\envs\IDS2025\python.exe "D:\Miftah_Data\IDS_Project -2019\train_variants\train_spectral_rrd_revised_ t_value_comparison.py"
# Starting t-value comparison test...
# GPU Available: True
# GPU: NVIDIA RTX 6000 Ada Generation
#
# ============================================================
# TESTING POWER ITERATION VALUES (k=40, epochs=3)
# ============================================================
#
# 🔬 Testing t = 1
# t=1 Epoch 1/3:   0%|                                  | 0/10745 [00:00<?, ?it/s]D:\programs\Anaconda3\envs\IDS2025\lib\site-packages\torch\nn\modules\linear.py:116: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\Context.cpp:164.)
#   return F.linear(input, self.weight, self.bias)
# D:\Miftah_Data\IDS_Project -2019\train_variants\train_spectral_rrd_revised_ t_value_comparison.py:48: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\Context.cpp:164.)
#   V = embeddings_norm.T @ (embeddings_norm @ V)
# D:\Miftah_Data\IDS_Project -2019\train_variants\train_spectral_rrd_revised_ t_value_comparison.py:57: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\Context.cpp:164.)
#   return embeddings_norm @ V
# D:\Miftah_Data\IDS_Project -2019\train_variants\train_spectral_rrd_revised_ t_value_comparison.py:61: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\Context.cpp:164.)
#   G = Z.T @ Z
# D:\programs\Anaconda3\envs\IDS2025\lib\site-packages\torch\autograd\__init__.py:266: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\Context.cpp:164.)
#   Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
# t=1 Epoch 2/3:   0%|                                  | 0/10745 [00:00<?, ?it/s]  Epoch 1: Val Acc=0.9875, Time=100.9s, Spectral Loss=0.000419
# t=1 Epoch 3/3:   0%|                                  | 0/10745 [00:00<?, ?it/s]  Epoch 2: Val Acc=0.9923, Time=100.8s, Spectral Loss=0.000375
# t=2 Epoch 1/3:   0%|                                  | 0/10745 [00:00<?, ?it/s]  Epoch 3: Val Acc=0.9946, Time=99.6s, Spectral Loss=0.000356
#
# 🔬 Testing t = 2
# t=2 Epoch 2/3:   0%|                                  | 0/10745 [00:00<?, ?it/s]  Epoch 1: Val Acc=0.9783, Time=105.9s, Spectral Loss=0.000301
# t=2 Epoch 3/3:   0%|                                  | 0/10745 [00:00<?, ?it/s]  Epoch 2: Val Acc=0.9875, Time=104.1s, Spectral Loss=0.000266
# t=3 Epoch 1/3:   0%|                                  | 0/10745 [00:00<?, ?it/s]  Epoch 3: Val Acc=0.9947, Time=105.7s, Spectral Loss=0.000264
#
# 🔬 Testing t = 3
# t=3 Epoch 2/3:   0%|                                  | 0/10745 [00:00<?, ?it/s]  Epoch 1: Val Acc=0.3043, Time=113.9s, Spectral Loss=0.000634
# t=3 Epoch 3/3:   0%|                                  | 0/10745 [00:00<?, ?it/s]  Epoch 2: Val Acc=0.8003, Time=113.9s, Spectral Loss=0.000592
#   Epoch 3: Val Acc=0.9752, Time=111.2s, Spectral Loss=0.000337
#
# ============================================================
# RESULTS COMPARISON
# ============================================================
# t    Final Val Acc Avg Epoch Time Speed vs t=1
# ------------------------------------------------------------
# 1    0.9946       100.4s          1.00x
# 2    0.9947       105.3s          0.95x
# 3    0.9752       113.0s          0.89x
#
# ✅ Results saved to: D:\Miftah_Data\IDS_Project -2019\logs\Feedback_KD\t_value_test.json
#
# 📊 ANALYSIS:
# Accuracy difference t=1 vs t=2: 0.0002
# Accuracy difference t=1 vs t=3: 0.0193
# t=2 is 1.0x slower than t=1
# t=3 is 1.1x slower than t=1
#
# Process finished with exit code 0

