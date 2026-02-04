import torch
import torch.nn as nn
import torch.nn.functional as F


# ======== Existing BiGRU Models (keep as reference) ========
class BiGRUTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.bigru1 = nn.GRU(input_size=1, hidden_size=64, batch_first=True, bidirectional=True)
        self.bigru2 = nn.GRU(input_size=128, hidden_size=128, batch_first=True, bidirectional=True)
        self.bigru3 = nn.GRU(input_size=256, hidden_size=256, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, 8)

    def forward(self, x):
        x, _ = self.bigru1(x)
        x, _ = self.bigru2(x)
        x, _ = self.bigru3(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)

    def extract_features(self, x):
        x, _ = self.bigru1(x)
        x, _ = self.bigru2(x)
        x, _ = self.bigru3(x)
        return x[:, -1, :]


class BiGRUStudent(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=8):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        embed = out[:, -1, :]
        x = torch.relu(self.fc1(embed))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits, embed


# ======== NEW: CNN Model ========
class CNNTeacher(nn.Module):
    """1D CNN for sequential data (treat as 1D image)"""

    def __init__(self):
        super().__init__()
        # Input: [batch, channels=1, seq_len=85]
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global pooling
        self.fc = nn.Linear(64, 8)

    def forward(self, x):
        # x shape: [B, 85, 1] -> transpose to [B, 1, 85]
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # [B, 64, 1]
        x = x.squeeze(-1)  # [B, 64]
        return self.fc(x)

    def extract_features(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.squeeze(-1)


class CNNStudent(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        embed = self.pool(x).squeeze(-1)  # [B, 32]
        x = torch.relu(self.fc1(embed))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits, embed


# ======== NEW: Transformer Model ========
class TransformerTeacher(nn.Module):
    """Simple Transformer for sequences"""

    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.embed = nn.Linear(1, d_model)  # Project 1-dim to d_model
        self.pos_encoder = nn.Parameter(torch.randn(1, 85, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, 8)

    def forward(self, x):
        # x: [B, 85, 1] -> [B, 85, d_model]
        x = self.embed(x)
        x = x + self.pos_encoder
        x = x.transpose(0, 1)  # [85, B, d_model]
        x = self.transformer(x)
        x = x.mean(dim=0)  # Global pooling
        return self.fc(x)

    def extract_features(self, x):
        x = self.embed(x)
        x = x + self.pos_encoder
        x = x.transpose(0, 1)
        x = self.transformer(x)
        return x.mean(dim=0)


class TransformerStudent(nn.Module):
    def __init__(self, d_model=32, nhead=4):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 85, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(d_model, 8)

    def forward(self, x):
        x = self.embed(x)
        x = x + self.pos_encoder
        x = x.transpose(0, 1)
        x = self.transformer(x)
        embed = x.mean(dim=0)  # [B, d_model]
        logits = self.fc(embed)
        return logits, embed