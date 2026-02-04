import os
import time
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

from data_loader_2019 import load_cicddos2019
from model_2019 import BiGRUTeacher, BiGRUStudent
import os, random
import numpy as np
# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "spectral_rrd_model.pt")
LOG_PATH = os.path.join(BASE_DIR, "logs", "Feedback_KD", "spectral_rrd_history.json")
TEACHER_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "teacher_model.pt")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Stronger determinism (may slow a bit):
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === LR Schedule ===
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


# === Stable Spectral Projection using Power Iteration + QR (instead of torch.svd) ===
def spectral_project(embeddings, k=30):
    """
    Robust spectral projection to avoid torch.svd instability.

    Explanation:
    torch.svd() can throw LinAlgError when the input matrix is ill-conditioned or
    has too many repeated singular values. Instead, we use a stable approximation:

    1. Power iteration: captures top-k dominant directions of the embedding space.
    2. QR decomposition: orthonormalizes those directions into a projection basis.

    This method avoids full SVD and is GPU-friendly.
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)
    _, D = embeddings.shape

    # Random basis
    V = torch.randn(D, k, device=embeddings.device)

    # One-step power iteration
    V = embeddings.T @ (embeddings @ V)

    # Orthonormalize via QR
    V, _ = torch.linalg.qr(V)

    # Project onto dominant subspace
    return embeddings @ V


def gram_k(Z):
    # Z: [B, k]
    Zc = Z - Z.mean(dim=0, keepdim=True)          # center features
    G = Zc.T @ Zc                                  # [k, k]
    G = G / (Zc.shape[0] + 1e-8)                   # normalize by batch
    return G

def fro_norm(G):
    return G / (G.norm(p='fro') + 1e-8)


# === Spectral RRD Loss using cosine similarity in projected space ===
def spectral_rrd_loss(student_embed, teacher_embed, alpha=0.7, k=30):
    B, D = student_embed.shape
    k_eff = max(1, min(k, D))  # clamp k

    try:
        s_proj = spectral_project(student_embed, k_eff)  # [B, k]
        t_proj = spectral_project(teacher_embed, k_eff)  # [B, k]

        if torch.any(torch.isnan(s_proj)) or torch.any(torch.isnan(t_proj)):
            raise RuntimeError("NaNs in spectral projection")

        Gs = fro_norm(gram_k(s_proj))  # [k, k]
        Gt = fro_norm(gram_k(t_proj))  # [k, k]

        # --- DEBUG: print once to confirm k×k ---
        if not hasattr(spectral_rrd_loss, "_printed"):
            print(f"[unweighted] k={k} Gram shapes: student={Gs.shape}, teacher={Gt.shape}")
            spectral_rrd_loss._printed = True
        # ---------------------------------------

        loss = torch.mean((Gs - Gt)**2)
        return alpha * loss, loss.detach().item()

    except Exception as e:
        print(f"[WARN][spectral_rrd_loss] Falling back to pairwise cosine (B×B): {e}")
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


#  Training Function
def train_spectral_rrd(k: int = 30,seed: int = 1337, epoch=3):
    set_seed(seed)
    print(f" Training Spectral RRD Student...k-{k}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_cicddos2019(batch_size=256)

    # === Load pretrained teacher ===
    teacher = BiGRUTeacher().to(device)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    teacher.eval()

    # === Initialize student ===
    student = BiGRUStudent(hidden_dim=64, num_classes=8).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    history = {
        "train_acc": [], "val_acc": [],
        "ce_loss": [], "spectral_loss": [], "total_loss": [],
        "val_loss": [], "lr_used": [],
        "epoch_time_sec": [], "gpu_peak_mem_mb": [],
        "time_sec": 0
    }

    start_time = time.time()
    print(f" Training started on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(epoch):
        lr = lr_schedule20(epoch)
        for g in optimizer.param_groups:
            g["lr"] = lr
        history["lr_used"].append(lr)
        # epoch_start = time.time()
        #
        # if torch.cuda.is_available():
        #     torch.cuda.reset_peak_memory_stats()

        epoch_start = time.time()
        if torch.cuda.is_available():
           torch.cuda.reset_peak_memory_stats()

        student.train()
        total_loss = total_ce = total_spec = correct = total = 0

        for x, y in tqdm(train_loader, desc=f"Spectral RRD Epoch {epoch + 1}/20", ncols=100):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            student_logits, student_embed = student(x)
            with torch.no_grad():
                teacher_embed = teacher.extract_features(x)

            ce = criterion(student_logits, y)
            spectral, spectral_val = spectral_rrd_loss(student_embed, teacher_embed, alpha=0.7, k=k)
            loss = ce + spectral

            loss.backward()
            optimizer.step()

            preds = torch.argmax(student_logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            total_loss += loss.item()
            total_ce += ce.item()
            total_spec += spectral_val

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
        history["spectral_loss"].append(total_spec / len(train_loader))
        history["total_loss"].append(total_loss / len(train_loader))
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch + 1} - Train Loss: {total_loss / len(train_loader):.4f} | "
              f"CE: {total_ce / len(train_loader):.4f} | Spectral: {total_spec / len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {lr:.6f}")

    history["time_sec"] = time.time() - start_time
    print(f"\n⏱️ Spectral RRD training complete in {history['time_sec']:.2f}s.")
    model_path = MODEL_PATH.replace(".pt", f"_k{k}.pt")
    log_path = LOG_PATH.replace(".json", f"_k{k}.json")
    torch.save(student.state_dict(), model_path)
    with open(log_path, "w") as f:
        json.dump(history, f)


    # torch.save(student.state_dict(), MODEL_PATH)
    # with open(LOG_PATH, "w") as f:
    #     json.dump(history, f)

    print(f"✅ Saved Spectral RRD model to {model_path}")
    print(f"📁 Training log saved to {log_path}")


# === Run ===
if __name__ == "__main__":
    set_seed(seed)
    train_spectral_rrd()
