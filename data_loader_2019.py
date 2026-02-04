import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
#
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_cicddos2019(data_dir=None, batch_size=256):
    if data_dir is None:
        # Always resolve from project root (where this file lives)
        base_dir = Path(__file__).resolve().parent
        data_dir = base_dir / "data" / "train_test_data_cicddos2019"
    else:
        data_dir = Path(data_dir).resolve()

    X_train = np.load(str(data_dir / "X_train1.npy"))
    Y_train = np.load(str(data_dir / "Y_train.npy"))
    X_test = np.load(str(data_dir / "X_test1.npy"))
    Y_test = np.load(str(data_dir / "Y_test.npy"))

    X_train = np.squeeze(X_train)
    X_test = np.squeeze(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ========== SANITY CHECK (safe to delete later) ==========
if __name__ == "__main__":
    train_loader, test_loader = load_cicddos2019()
    for x_batch, y_batch in train_loader:
        print(f" X batch shape: {x_batch.shape}")   # Expected: [batch_size, 1, 85]
        print(f" Y batch shape: {y_batch.shape}")   # Expected: [batch_size]
        print(f" Unique labels in batch: {y_batch.unique().tolist()}")
        break
