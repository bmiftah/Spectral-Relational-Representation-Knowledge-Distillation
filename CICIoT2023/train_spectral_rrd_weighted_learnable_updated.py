import os
import time
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn  # Add if not present
from data_loader_2023 import load_cicIoT2023
from model_2019 import BiGRUTeacher, BiGRUStudent

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "spectral_rrd_weighted_model_learnable_k{k}.pt")
LOG_PATH = os.path.join(BASE_DIR, "logs", "Feedback_KD", "spectral_rrd_weighted_history_learnable{k}.json")
TEACHER_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "teacher_model.pt")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# === Learning Rate Schedule ===
# def lr_schedule20(epoch):
#     if epoch < 5:
#         return 1e-3
#     elif epoch < 10:
#         return 5e-4
#     elif epoch < 15:
#         return 1e-4
#     elif epoch < 18:
#         return 5e-5
#     else:
#         return 1e-5
def lr_schedule20(epoch):
    if epoch < 3:       # Initial high LR for fast progress
        return 1e-3
    elif epoch < 8:     # First reduction when progress slows
        return 5e-4
    elif epoch < 12:    # More gradual reduction
        return 2e-4
    elif epoch < 16:    # Fine-tuning phase
        return 5e-5
    else:               # Final very low LR
        return 1e-5





# === Stable QR-based Weighted Spectral RRD ===
def weighted_spectral_project(embed, k=50):
    """
    Projects embeddings to dominant subspace using QR decomposition
    after a single power iteration. Avoids torch.svd() instability on ill-conditioned matrices.

    Args:
        embed: Tensor [B, D] input features
        k: number of top singular directions to retain

    Returns:
        projected embedding: [B, k]
        singular values: [k] (approximated norm per projection basis)
    """
    embed = F.normalize(embed, p=2, dim=1)
    B, D = embed.shape
    V = torch.randn(D, k, device=embed.device)

    # One round power iteration (can be repeated if needed)
    V = embed.T @ (embed @ V)

    # QR orthonormalization
    V, _ = torch.linalg.qr(V)  # shape: [D, k]
    projected = embed @ V      # shape: [B, k]

    # Weighting: L2-norm of each component vector
    weights = projected.norm(p=2, dim=0)
    weights = weights / (weights.sum() + 1e-8)
    return projected, weights
#
# def weighted_spectral_rrd_loss(student_embed, teacher_embed, alpha=0.7, k=50):
#     """
#     Computes stable weighted spectral relational loss.
#     Avoids SVD-related convergence issues in torch on ill-conditioned batches.
#     """
#     try:
#         s_proj, s_weights = weighted_spectral_project(student_embed, k)
#         t_proj, t_weights = weighted_spectral_project(teacher_embed, k)
#
#         S_sim = (s_proj * s_weights) @ (s_proj * s_weights).T
#         T_sim = (t_proj * t_weights) @ (t_proj * t_weights).T
#
#         loss = F.mse_loss(S_sim, T_sim)
#         return alpha * loss, loss.item()
#
#     except RuntimeError as e:
#         print(f"[WARN] Spectral projection failed: {e}")
#         return torch.tensor(0.0, requires_grad=True, device=student_embed.device), 0.0
def _gram_k(Z: torch.Tensor) -> torch.Tensor:
    Zc = Z - Z.mean(dim=0, keepdim=True)
    G = (Zc.T @ Zc) / (Zc.shape[0] + 1e-8)
    return G / (G.norm(p='fro') + 1e-8)

# def weighted_spectral_rrd_loss(student_embed, teacher_embed, alpha=0.7, k=30):
#     Zs, ws = weighted_spectral_project(student_embed, k)   # [B,k], [k]
#     Zt, wt = weighted_spectral_project(teacher_embed, k)   # [B,k], [k]
#     Gs = _gram_k(Zs * ws)                                  # [k,k]
#     Gt = _gram_k(Zt * wt)                                  # [k,k]
#     loss = F.mse_loss(Gs, Gt)
#     return alpha * loss, loss.detach().item()
def weighted_spectral_rrd_loss(student_embed, teacher_embed, learnable_w, alpha=0.7, k=50):
    """
    Updated with learnable weights and sigmoid constraint
    """
    try:
        s_proj, _ = weighted_spectral_project(student_embed, k)
        t_proj, _ = weighted_spectral_project(teacher_embed, k)

        # === LEARNABLE WEIGHTS WITH SIGMOID ===
        w = torch.sigmoid(learnable_w[:k]) + 1e-8  # [k]

        Gs = _gram_k(s_proj * w)
        Gt = _gram_k(t_proj * w)

        loss = F.mse_loss(Gs, Gt)
        return alpha * loss, loss.detach().item()

    except Exception as e:
        print(f"[WARN] Spectral projection failed: {e}")
        return torch.tensor(0.0, requires_grad=True, device=student_embed.device), 0.0
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

# === Train Function ===
def train_spectral_rrd_weighted(k=30):
    print(" Training Weighted Spectral RRD Student with updated Learnable Weights ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_cicIoT2023(batch_size=1024)

    # === Load teacher ===
    teacher = BiGRUTeacher().to(device)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    teacher.eval()

    # === Init student ===
    student = BiGRUStudent(hidden_dim=64, num_classes=8).to(device)
    # === ADD LEARNABLE WEIGHT PARAMETER HERE ===
    learnable_w = nn.Parameter(torch.ones(k, device=device) / k)
    # optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam([
        {'params': student.parameters()},
        {'params': [learnable_w], 'weight_decay': 1e-4}  # L2 regularization on weights
    ], lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    history = {
        "train_acc": [], "val_acc": [],
        "ce_loss": [], "weighted_rrd_loss": [], "total_loss": [],
        "val_loss": [], "lr_used": [], "time_sec": 0,
        "weight_mean": [], "weight_std": []
    }

    start_time = time.time()
    print(" Using ",device)
    print(f"🕒 Training started on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(20):
        lr = lr_schedule20(epoch)
        for g in optimizer.param_groups:
            g["lr"] = lr
        history["lr_used"].append(lr)

        student.train()
        total_loss = total_ce = total_wrrd = correct = total = 0

        for x, y in tqdm(train_loader, desc=f"Weighted Spectral Epoch {epoch+1}/20", ncols=100):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            student_logits, student_embed = student(x)
            with torch.no_grad():
                teacher_embed = teacher.extract_features(x)

            ce = criterion(student_logits, y)
            wrrd, wrrd_val = weighted_spectral_rrd_loss(
                student_embed, teacher_embed,
                learnable_w,  # ← ADD THIS
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

        train_acc = correct / total
        val_acc, val_loss = evaluate_loss(student, test_loader, criterion, device)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["ce_loss"].append(total_ce / len(train_loader))
        history["weighted_rrd_loss"].append(total_wrrd / len(train_loader))
        history["total_loss"].append(total_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        with torch.no_grad():
            w_actual = torch.sigmoid(learnable_w) + 1e-8
            history["weight_mean"].append(w_actual.mean().item())
            history["weight_std"].append(w_actual.std().item())

        print(f"Epoch {epoch + 1} - Train Loss: {total_loss / len(train_loader):.4f} | "
              f"CE: {total_ce / len(train_loader):.4f} | WRRD: {total_wrrd / len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Weights: mean={history['weight_mean'][-1]:.4f}, std={history['weight_std'][-1]:.4f} | "  # ← ADD
              f"LR: {lr:.6f}")


    history["time_sec"] = time.time() - start_time
    print(f"\n⏱️ Weighted Spectral RRD training complete in {history['time_sec']:.2f}s.")

    # MODEL_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", f"spectral_rrd_weighted_model_k{k}.pt")
    # LOG_PATH = os.path.join(BASE_DIR, "logs", "Feedback_KD", f"spectral_rrd_weighted_history_{k}.json")

    MODEL_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", f"spectral_rrd_weighted_model_learnable_k{k}.pt")
    LOG_PATH = os.path.join(BASE_DIR, "logs", "Feedback_KD", f"spectral_rrd_weighted_history_learnable_k{k}.json")

    torch.save(student.state_dict(), MODEL_PATH)
    with open(LOG_PATH, "w") as f:
        json.dump(history, f)

    print(f"✅ Saved Weighted Spectral RRD model to {MODEL_PATH}")
    print(f"📁 Training log saved to {LOG_PATH}")

# === Run ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=30)
    args = parser.parse_args()
    train_spectral_rrd_weighted(k=args.k)
