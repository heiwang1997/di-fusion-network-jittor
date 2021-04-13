import torch.nn as nn
import torch
import torch.nn.functional as F


class DIDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.utils.weight_norm(nn.Linear(32, 128))
        self.lin1 = nn.utils.weight_norm(nn.Linear(128, 128))
        self.lin2 = nn.utils.weight_norm(nn.Linear(128, 128 - 32))
        self.lin3 = nn.utils.weight_norm(nn.Linear(128, 128))
        self.lin4 = nn.utils.weight_norm(nn.Linear(128, 1))
        self.uncertainty_layer = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = [0, 1, 2, 3, 4, 5]
        self.th = nn.Tanh()

    def forward(self, ipt):
        x = self.lin0(ipt)
        x = self.relu(x)
        x = F.dropout(x, p=0.2, training=True)

        x = self.lin1(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.2, training=True)

        x = self.lin2(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.2, training=True)

        x = torch.cat([x, ipt], 1)
        x = self.lin3(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.2, training=True)

        std = self.uncertainty_layer(x)
        std = 0.05 + 0.5 * F.softplus(std)
        x = self.lin4(x)
        x = self.th(x)

        return x, std


class DIEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 256, kernel_size=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 29, kernel_size=1)
        )

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.mlp(x)     # (B, L, N)
        r = torch.mean(x, dim=-1)
        return r
