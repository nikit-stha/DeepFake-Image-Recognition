import torch
import torch.nn as nn
import torch.nn.functional as F

class QCO(nn.Module):
    def __init__(self, in_channels, levels=8):
        super(QCO, self).__init__()
        self.levels = levels
        self.mlp = nn.Sequential(
            nn.Linear(2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

    def forward(self, A):
        B, C, H, W = A.shape

        y = F.adaptive_avg_pool2d(A, 1)

        A_flat = A.view(B, C, -1)
        y_flat = y.view(B, C, 1)

        P = F.cosine_similarity(A_flat, y_flat, dim=1)

        P = P.view(B, H, W)

        P_min = P.amin(dim=(1, 2), keepdim=True)
        P_max = P.amax(dim=(1, 2), keepdim=True)

        P_norm = (P - P_min) / (P_max - P_min + 1e-8)

        levels = torch.arange(self.levels, device=A.device).float()

        step = 1.0 / self.levels
        U = step * levels

        U_flat = U.unsqueeze(0).expand(B, -1)

        P_flat = P_norm.flatten(1)
        P_expand = P_flat.unsqueeze(-1)

        U_expand = U_flat.unsqueeze(1)

        distance = U_expand - P_expand

        mask = ((distance >= -0.5 / self.levels) & (distance < 0.5 / self.levels))

        G = torch.where(mask, 1 - torch.abs(distance), torch.zeros_like(distance))

        count = G.sum(dim=1)
        count_total = count.sum(dim=1, keepdim=True)
        count = count / (count_total + 1e-8)

        T = torch.stack([U_flat, count], dim=-1)

        feature = self.mlp(T)

        y = y.view(B, C).unsqueeze(1).expand(-1, self.levels, -1)
        H = torch.cat([feature, y], dim=-1)

        return G, T, H