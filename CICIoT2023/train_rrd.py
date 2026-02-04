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
MODEL_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "rrd_model.pt")
LOG_PATH = os.path.join(BASE_DIR, "logs", "Feedback_KD", "rrd_history.json")
TEACHER_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "teacher_model.pt")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# === LR Schedule ===
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
    if epoch < 3:
        return 1e-3
    elif epoch < 6:
        return 5e-4
    elif epoch < 10:
        return 1e-4
    elif epoch < 14:
        return 5e-5
    elif epoch < 17:
        return 2e-5
    else:
        return 5e-6

# === RRD Loss (Cosine Similarity of embeddings) ===
def rrd_loss(student_embed, teacher_embed, alpha=0.7):
    S = F.cosine_similarity(student_embed.unsqueeze(1), student_embed.unsqueeze(0), dim=2)
    T = F.cosine_similarity(teacher_embed.unsqueeze(1), teacher_embed.unsqueeze(0), dim=2)
    loss = F.mse_loss(S, T)
    return alpha * loss, loss.item()

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
def train_rrd_student():
    print("🚀 Training Vanilla RRD Student...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_cicIoT2023(batch_size=512)

    # === Load teacher ===
    teacher = BiGRUTeacher().to(device)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    teacher.eval()

    # === Init student ===
    student = BiGRUStudent(hidden_dim=64, num_classes=8).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    history = {
        "train_acc": [], "val_acc": [],
        "ce_loss": [], "rrd_loss": [], "total_loss": [],
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
        total_loss = total_ce = total_rrd = correct = total = 0

        for x, y in tqdm(train_loader, desc=f"RRD Epoch {epoch+1}/20", ncols=100):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            student_logits, student_embed = student(x)
            with torch.no_grad():
                teacher_embed = teacher.extract_features(x)

            ce = criterion(student_logits, y)
            rrd, rrd_val = rrd_loss(student_embed, teacher_embed, alpha=0.7)
            loss = ce + rrd

            loss.backward()
            optimizer.step()

            preds = torch.argmax(student_logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            total_loss += loss.item()
            total_ce += ce.item()
            total_rrd += rrd_val

        train_acc = correct / total
        val_acc, val_loss = evaluate_loss(student, test_loader, criterion, device)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["ce_loss"].append(total_ce / len(train_loader))
        history["rrd_loss"].append(total_rrd / len(train_loader))
        history["total_loss"].append(total_loss / len(train_loader))
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f} | "
              f"CE: {total_ce/len(train_loader):.4f} | RRD: {total_rrd/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {lr:.6f}")

    history["time_sec"] = time.time() - start_time
    print(f"\n⏱️ RRD training complete in {history['time_sec']:.2f}s.")

    torch.save(student.state_dict(), MODEL_PATH)
    with open(LOG_PATH, "w") as f:
        json.dump(history, f)

    print(f"✅ Saved RRD student model to {MODEL_PATH}")
    print(f"📁 Training log saved to {LOG_PATH}")

# === Run ===
if __name__ == "__main__":
    train_rrd_student()
