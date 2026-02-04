import os
import time
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

# Import for CICDDoS2019 (adjust import name as needed)
try:
    from data_loader_2019 import load_cicddos2019
except ImportError:
    from data_loader_2019 import load_cicddos2019

from model_2019 import BiGRUTeacher, BiGRUStudent

# === Paths (relative to project root) ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "vanilla_kd_2019_model.pt")
LOG_PATH = os.path.join(BASE_DIR, "logs", "Feedback_KD", "vanilla_kd_2019_history.json")
TEACHER_PATH = os.path.join(BASE_DIR, "models", "Feedback_KD", "teacher_2019_model.pt")  # Adjusted teacher path

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


# === LR Schedule (from RRD training for consistency) ===
def lr_schedule20(epoch):
    if epoch < 5:
        return 1e-3
    elif epoch < 10:
        return 4e-4
    elif epoch < 15:
        return 1e-4
    elif epoch < 18:
        return 5e-4
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
def train_vanilla_kd_2019():
    print("🚀 Training Vanilla KD Student for CICDDoS2019...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CICDDoS2019 data (same batch size as RRD training)
    train_loader, test_loader = load_cicddos2019(batch_size=256)  # Using RRD's batch size

    # === Load teacher model ===
    teacher = BiGRUTeacher().to(device)

    # Try multiple possible teacher paths for robustness
    possible_teacher_paths = [
        TEACHER_PATH,
        os.path.join(BASE_DIR, "models", "Feedback_KD", "teacher_model.pt"),
        os.path.join(BASE_DIR, "models", "teacher_model.pt")
    ]

    teacher_loaded = False
    for teacher_path in possible_teacher_paths:
        if os.path.exists(teacher_path):
            print(f"📂 Loading teacher from: {teacher_path}")
            teacher.load_state_dict(torch.load(teacher_path, map_location=device))
            teacher_loaded = True
            break

    if not teacher_loaded:
        raise FileNotFoundError(f"No teacher model found. Tried: {possible_teacher_paths}")

    teacher.eval()

    # === Init student (same architecture as RRD training) ===
    student = BiGRUStudent(hidden_dim=64, num_classes=8).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    history = {
        "train_acc": [], "val_acc": [],
        "ce_loss": [], "kl_loss": [], "total_loss": [],
        "val_loss": [], "lr_used": [],
        "epoch_time_sec": [], "gpu_peak_mem_mb": [],
        "time_sec": 0
    }

    start_time = time.time()
    print(f"🕒 Training started on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Dataset: CICDDoS2019 | Batch Size: 256")
    print(f"🎯 Teacher: {teacher_path.split('/')[-1]} | Student: BiGRU (hidden_dim=64)")

    for epoch in range(20):
        lr = lr_schedule20(epoch)
        for g in optimizer.param_groups:
            g["lr"] = lr
        history["lr_used"].append(lr)

        epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        student.train()
        total_loss = total_ce = total_kl = correct = total = 0

        for x, y in tqdm(train_loader, desc=f"Vanilla KD 2019 Epoch {epoch + 1}/20", ncols=100):
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

        # Track GPU memory and timing
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
        history["kl_loss"].append(total_kl / len(train_loader))
        history["total_loss"].append(total_loss / len(train_loader))
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch + 1:2d} - "
              f"Train Loss: {total_loss / len(train_loader):.4f} | "
              f"CE: {total_ce / len(train_loader):.4f} | KL: {total_kl / len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {lr:.6f}")

        if gpu_peak_mb:
            print(f"         GPU Peak Mem: {gpu_peak_mb:.1f} MB | Epoch Time: {time.time() - epoch_start:.1f}s")

    history["time_sec"] = time.time() - start_time
    print(f"\n⏱️ Vanilla KD training complete in {history['time_sec']:.2f}s.")
    print(f"📊 Final Validation Accuracy: {history['val_acc'][-1]:.4f}")

    # === Save ===
    torch.save(student.state_dict(), MODEL_PATH)
    with open(LOG_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"✅ Saved Vanilla KD model to {MODEL_PATH}")
    print(f"📁 Training log saved to {LOG_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("VANILLA KD CICDDoS2019 TRAINING SUMMARY")
    print("=" * 60)
    print(f"Final Train Acc: {history['train_acc'][-1]:.4f}")
    print(f"Final Val Acc:   {history['val_acc'][-1]:.4f}")
    print(f"Total Time:      {history['time_sec']:.2f}s")
    print(f"Avg Epoch Time:  {sum(history['epoch_time_sec']) / len(history['epoch_time_sec']):.2f}s")
    if history['gpu_peak_mem_mb'][-1]:
        print(f"Max GPU Memory:  {max([m for m in history['gpu_peak_mem_mb'] if m]):.1f} MB")
    print("=" * 60)


# === Run ===
if __name__ == "__main__":
    train_vanilla_kd_2019()