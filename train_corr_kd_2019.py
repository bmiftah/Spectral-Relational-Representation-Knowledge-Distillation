import os
import time
import json
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data_loader_2019 import load_cicddos2019
from model_2019 import BiGRUTeacher, BiGRUStudent

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "corr_kd_model.pt")
LOG_PATH   = os.path.join(BASE_DIR, "logs",   "Feedback_KD", "corr_kd_history.json")
TEACHER_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "teacher_model.pt")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

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

def _center(E: torch.Tensor) -> torch.Tensor:
    return E - E.mean(dim=0, keepdim=True)

def _cov(Ec: torch.Tensor) -> torch.Tensor:
    B = Ec.shape[0]
    denom = max(B - 1, 1)
    return (Ec.T @ Ec) / float(denom)

def _fro_norm(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return M / (M.norm(p="fro") + eps)

def corr_kd_loss(student_embed: torch.Tensor,
                 teacher_embed: torch.Tensor,
                 align: nn.Module,
                 alpha: float = 0.7) -> tuple[torch.Tensor, float]:
    """
    CorrKD with dimension alignment:
      student_embed: [B, Ds]
      teacher_embed: [B, Dt]
      align maps [B, Ds] -> [B, Dt]
    """
    # Align student to teacher dimension
    s_aligned = align(student_embed)          # [B, Dt]
    t_fixed   = teacher_embed                 # [B, Dt]

    # Center
    Es = _center(s_aligned)
    Et = _center(t_fixed)

    # Covariance in feature dimension (Dt×Dt)
    Cs = _fro_norm(_cov(Es))
    Ct = _fro_norm(_cov(Et))

    loss = torch.mean((Cs - Ct) ** 2)
    return alpha * loss, loss.detach().item()

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

def train_corr_kd(seed: int = 1337, epochs: int = 20, alpha: float = 0.7):
    set_seed(seed)
    print("🚀 Training Correlation-Alignment KD (CorrKD) Student...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_cicddos2019(batch_size=256)

    teacher = BiGRUTeacher().to(device)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    teacher.eval()

    student = BiGRUStudent(hidden_dim=64, num_classes=8).to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # We will create align layer lazily after seeing first batch dims
    align = None

    history = {
        "train_acc": [], "val_acc": [],
        "ce_loss": [], "corr_loss": [], "total_loss": [],
        "val_loss": [], "lr_used": [],
        "epoch_time_sec": [], "gpu_peak_mem_mb": [],
        "time_sec": 0
    }

    start_time = time.time()
    print(f"🕒 Training started on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(epochs):
        lr = lr_schedule20(epoch)
        for g in optimizer.param_groups:
            g["lr"] = lr
        history["lr_used"].append(lr)

        epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        student.train()
        total_loss = total_ce = total_corr = correct = total = 0

        for x, y in tqdm(train_loader, desc=f"CorrKD Epoch {epoch+1}/{epochs}", ncols=100):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            student_logits, student_embed = student(x)
            with torch.no_grad():
                teacher_embed = teacher.extract_features(x)

            # Create align layer once we know dimensions
            if align is None:
                Ds = student_embed.shape[1]
                Dt = teacher_embed.shape[1]
                align = nn.Linear(Ds, Dt, bias=False).to(device)
                # Add to optimizer so it learns jointly with student
                optimizer.add_param_group({"params": align.parameters()})
                print(f"✅ CorrKD alignment layer created: Ds={Ds} -> Dt={Dt}")

            ce = criterion(student_logits, y)
            corr, corr_val = corr_kd_loss(student_embed, teacher_embed, align, alpha=alpha)
            loss = ce + corr

            loss.backward()
            optimizer.step()

            preds = torch.argmax(student_logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            total_loss += loss.item()
            total_ce += ce.item()
            total_corr += corr_val

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
        history["corr_loss"].append(total_corr / len(train_loader))
        history["total_loss"].append(total_loss / len(train_loader))
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f} | "
            f"CE: {total_ce/len(train_loader):.4f} | Corr: {total_corr/len(train_loader):.4e} | "
            f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {lr:.6f}"
        )

    history["time_sec"] = time.time() - start_time
    print(f"\n⏱️ CorrKD training complete in {history['time_sec']:.2f}s.")

    # Save student+align together
    save_obj = {
        "student_state_dict": student.state_dict(),
        "align_state_dict": (align.state_dict() if align is not None else None),
        "meta": {"alpha": alpha}
    }
    torch.save(save_obj, MODEL_PATH)

    with open(LOG_PATH, "w") as f:
        json.dump(history, f)

    print(f"✅ Saved CorrKD model bundle to {MODEL_PATH}")
    print(f"📁 Training log saved to {LOG_PATH}")

if __name__ == "__main__":
    train_corr_kd()
