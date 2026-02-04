import os
import time
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

from data_loader_2023 import load_cicIoT2023
from model_2019 import BiGRUTeacher, BiGRUStudent

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "spectral_rrd_model_k{k}.pt")
LOG_PATH = os.path.join(BASE_DIR, "logs", "Feedback_KD", "spectral_rrd_history_{k}.json")
TEACHER_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "teacher_model.pt")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


# === LR Schedule ===
def lr_schedule20(epoch):
    if epoch < 5:
        return 1e-3
    elif epoch < 10:
        return 5e-4
    elif epoch < 15:
        return 1e-4
    elif epoch < 18:
        return 5e-5       ## need to change this later ... accuracy is flatten ... revise this later after running is completed for the other two models.
    else:
        return 1e-5
import torch
import torch.nn.functional as F

def _spectral_project(embeddings: torch.Tensor, k: int):
    # Normalize rows to avoid scale issues
    Z = F.normalize(embeddings, p=2, dim=1)
    B, D = Z.shape
    k_eff = max(1, min(k, D, B - 1))  # never ask for more modes than rank
    # 1-step power iteration + QR (no SVD)
    V = torch.randn(D, k_eff, device=Z.device)
    V = Z.T @ (Z @ V)
    V, _ = torch.linalg.qr(V)                 # [D, k_eff]
    Zk = Z @ V                                # [B, k_eff]
    return Zk, k_eff

def _gram_k(Z: torch.Tensor) -> torch.Tensor:
    Zc = Z - Z.mean(dim=0, keepdim=True)
    G = (Zc.T @ Zc) / (Zc.shape[0] + 1e-8)    # [k, k]
    return G / (G.norm(p='fro') + 1e-8)

# === Stable Spectral Projection using Power Iteration + QR (instead of torch.svd) ===
def spectral_project(embed: torch.Tensor, k: int) -> torch.Tensor:
    embed = F.normalize(embed, p=2, dim=1)          # [B,D]
    B, D = embed.shape
    V = torch.randn(D, k, device=embed.device)      # [D,k]
    V = embed.T @ (embed @ V)                       # one power iteration
    V, _ = torch.linalg.qr(V)                       # [D,k]
    return embed @ V                                # [B,k]

def spectral_rrd_loss(student_embed, teacher_embed, alpha=0.7, k=30):
    Zs = spectral_project(student_embed, k)
    Zt = spectral_project(teacher_embed, k)
    Zs_c = Zs - Zs.mean(dim=0, keepdim=True)
    Zt_c = Zt - Zt.mean(dim=0, keepdim=True)
    Gs = (Zs_c.T @ Zs_c); Gs = Gs / (Gs.norm(p='fro') + 1e-8)
    Gt = (Zt_c.T @ Zt_c); Gt = Gt / (Gt.norm(p='fro') + 1e-8)
    loss = F.mse_loss(Gs, Gt)
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


# === Training Function ===
def train_spectral_rrd(k: int = 30):
    print(" Training Spectral RRD Student...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_cicIoT2023(batch_size=512)

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
        "val_loss": [], "lr_used": [], "time_sec": 0
    }

    start_time = time.time()
    print(f"🕒 Training started on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(20):
        lr = lr_schedule20(epoch)
        for g in optimizer.param_groups:
            g["lr"] = lr
        history["lr_used"].append(lr)

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

    # Save model and log with k suffix


    MODEL_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", f"spectral_rrd_model_k{k}.pt")
    LOG_PATH = os.path.join(BASE_DIR, "logs", "Feedback_KD", f"spectral_rrd_history_k{k}.json")

    torch.save(student.state_dict(), MODEL_PATH)
    with open(LOG_PATH, "w") as f:
        json.dump(history, f)

    print(f" Saved Spectral RRD model to {MODEL_PATH}")
    print(f" Training log saved to {LOG_PATH}")


# === Run ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=30)
    args = parser.parse_args()
    train_spectral_rrd(k=args.k)
