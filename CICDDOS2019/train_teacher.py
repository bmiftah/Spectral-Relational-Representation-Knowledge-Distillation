import os
import time
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data_loader_2019 import load_cicddos2019
from model_2019 import BiGRUTeacher

# ==== LR Schedule ====
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

# ==== Evaluate ====
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    val_loss = total_loss / len(loader)
    val_acc = correct / total
    return val_acc, val_loss

# ==== Train Function ====
def train_teacher():
    print("Starting TEACHER training exactly as in notebook...\n")

    # === Config ===
    # input_dim = 85
    # hidden_dim = 64
    # num_classes = 8
    batch_size = 256
    epochs = 20
    # model_save_path = "../models/Feedback_KD/Old/teacher_model.pt"
    # history_save_path = "../logs/Feedback_KD/teacher_history.json"
    # os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
    # Always save under the project root, regardless of CWD

    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]  # project root
    MODEL_DIR = ROOT / "models" / "Feedback_KD"
    LOG_DIR = ROOT / "logs" / "Feedback_KD"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = str(MODEL_DIR / "teacher_model.pt")
    history_save_path = str(LOG_DIR / "teacher_history.json")

    # === Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_cicddos2019(batch_size=batch_size)
    model = BiGRUTeacher().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
        "lr_used": [],
        "epoch_time_sec": [],
        "gpu_peak_mem_mb": [],
        "time_sec": 0
    }

    start_time = time.time()

    for epoch in range(epochs):
        current_lr = lr_schedule20(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        history["lr_used"].append(current_lr)
        epoch_start = time.time()
        if torch.cuda.is_available():
         torch.cuda.reset_peak_memory_stats()

        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for xb, yb in tqdm(train_loader, desc=f"Teacher Epoch {epoch+1}/{epochs}", ncols=100):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)
        val_acc, val_loss = evaluate_loss(model, test_loader, criterion, device)


        if torch.cuda.is_available():
           torch.cuda.synchronize()
           gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
           gpu_peak_mb = None
        history["gpu_peak_mem_mb"].append(gpu_peak_mb)
        history["epoch_time_sec"].append(time.time() - epoch_start)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

    history["time_sec"] = time.time() - start_time
    print(f"\n⏱️ Training completed in {history['time_sec']:.2f} seconds.")

    # === Save ===
    torch.save(model.state_dict(), model_save_path)
    with open(history_save_path, "w") as f:
        json.dump(history, f)

    print(f"✅ Teacher model saved to {model_save_path}")
    print(f"📁 Training history saved to {history_save_path}")

# ==== Run ====
if __name__ == "__main__":
    train_teacher()
