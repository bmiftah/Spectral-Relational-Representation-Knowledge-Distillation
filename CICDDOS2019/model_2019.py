import torch
import torch.nn as nn

# ======== Teacher Model (3-layer Bi-GRU) =========
class BiGRUTeacher(nn.Module):
    def __init__(self):
        super(BiGRUTeacher, self).__init__()
        self.bigru1 = nn.GRU(input_size=1, hidden_size=64, batch_first=True, bidirectional=True)
        self.bigru2 = nn.GRU(input_size=128, hidden_size=128, batch_first=True, bidirectional=True)
        self.bigru3 = nn.GRU(input_size=256, hidden_size=256, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, 8)  # 8-class output

    def forward(self, x):
        x, _ = self.bigru1(x)
        x, _ = self.bigru2(x)
        x, _ = self.bigru3(x)
        x = x[:, -1, :]  # Last timestep
        x = self.dropout(x)
        return self.fc(x)

    def extract_features(self, x):
        x, _ = self.bigru1(x)
        x, _ = self.bigru2(x)
        x, _ = self.bigru3(x)
        return x[:, -1, :]  # Embedding output

# ======== Student Model (for all student variants) =========
class BiGRUStudent(nn.Module):  # a.k.a RRDStudentModel
    def __init__(self, hidden_dim=64, num_classes=8):
        super(BiGRUStudent, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        embed = out[:, -1, :]  # Last timestep
        x = torch.relu(self.fc1(embed))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits, embed
