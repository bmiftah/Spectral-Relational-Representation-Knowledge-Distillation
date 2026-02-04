import os
import time
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

from data_loader_2023 import load_cicIoT2023
from model_2019 import BiGRUTeacher, BiGRUStudent

# === Paths (relative to project root) ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "vanilla_kd_model.pt")
LOG_PATH = os.path.join(BASE_DIR, "logs", "Feedback_KD", "vanilla_kd_history.json")
TEACHER_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "teacher_model.pt")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# === LR Schedule ===
def lr_schedule20(epoch):
    if epoch < 5:
        return 1e-3
    elif epoch < 10:
        return 4e-4
    elif epoch < 15:
        return 1e-4
    elif epoch < 18:
        return 1e-4
    else:
        return 1e-4

# === KD Loss ===
def kd_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=4.0):
    ce_loss = F.cross_entropy(student_logits, labels)
    kl_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    return alpha * kl_loss + (1 - alpha) * ce_loss, ce_loss.item(), kl_loss.item()

# === Evaluation ===
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)  # Student returns (logits, embed)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total, total_loss / len(loader)

# === Training Function ===
def train_vanilla_kd():
    print("🚀 Training Vanilla KD Student...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_cicIoT2023(batch_size=512)

    # === Load teacher model ===
    teacher = BiGRUTeacher().to(device)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    teacher.eval()

    # === Init student ===
    student = BiGRUStudent(hidden_dim=64, num_classes=8).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    history = {
        "train_acc": [], "val_acc": [],
        "ce_loss": [], "kl_loss": [], "total_loss": [],
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
        total_loss = total_ce = total_kl = correct = total = 0

        for x, y in tqdm(train_loader, desc=f"Vanilla KD Epoch {epoch+1}/20", ncols=100):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            student_logits, _ = student(x)  # Use logits only
            with torch.no_grad():
                teacher_logits = teacher(x)

            loss, ce_val, kl_val = kd_loss(student_logits, teacher_logits, y, alpha=0.5, temperature=4.0)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(student_logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            total_loss += loss.item()
            total_ce += ce_val
            total_kl += kl_val

        train_acc = correct / total
        val_acc, val_loss = evaluate_loss(student, test_loader, torch.nn.CrossEntropyLoss(), device)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["ce_loss"].append(total_ce / len(train_loader))
        history["kl_loss"].append(total_kl / len(train_loader))
        history["total_loss"].append(total_loss / len(train_loader))
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f} | "
              f"CE: {total_ce/len(train_loader):.4f} | KL: {total_kl/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {lr:.6f}")

    history["time_sec"] = time.time() - start_time
    print(f"\n⏱️ Vanilla KD training complete in {history['time_sec']:.2f}s.")

    # === Save ===
    torch.save(student.state_dict(), MODEL_PATH)
    with open(LOG_PATH, "w") as f:
        json.dump(history, f)

    print(f"✅ Saved Vanilla KD model to {MODEL_PATH}")
    print(f"📁 Training log saved to {LOG_PATH}")

# === Run ===
if __name__ == "__main__":
    train_vanilla_kd()
