import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QCO(nn.Module):
    def __init__(self, in_channels, levels = 16, hidden_layers = 64):
        super(QCO, self).__init__()
        self.levels = levels
        self.hidden_layers = hidden_layers
        self.in_channels = in_channels

        self.MLP = nn.Sequential(
            nn.Linear(2, self.hidden_layers),
            nn.ReLU(),
            nn.Linear(self.hidden_layers, self.in_channels)
        )

    def forward(self, A):
        B, C, H, W = A.shape

        #Global Average Pooling
        y = F.adaptive_avg_pool2d(A, 1)

        #Cosine Similarity between A and y
        P = F.cosine_similarity(A, y, dim=1)

        #Flatten P
        P_flat = P.view(B, -1)

        #P_max and P_min in each batch
        P_max = P_flat.max(dim=1).values
        P_min = P_flat.min(dim=1).values

        #Quanrization to N Levels
        U = []
        for i in range(self.levels):
            value = (((P_max - P_min) / self.levels) * (i + 1)) + P_min
            U.append(value)

        U = torch.stack(U)
        U = U.T

        U_expand = U.unsqueeze(1)
        P_expand = P_flat.unsqueeze(-1)

        #Calculate Quantization Encoding Matrix
        distance = U_expand - P_expand

        mask = ((distance >= -0.5 / self.levels) & (distance < 0.5 / self.levels))
        G = torch.where(mask, 1 - torch.abs(distance), torch.zeros_like(distance))

        #Calculate Mean Matrix
        count = G.sum(dim=1)
        mean_matrix = count / (count.sum(dim=1, keepdim=True) + 1e-6)

        #Concatenate Mean_Matrix and U to get Quantization Count Map
        T = torch.stack([U, mean_matrix], dim = -1)

        #Pass T through MultiLayer Perceptron (MLP)
        X = self.MLP(T)

        #Concatenate X and y(Global Average Pooled Feature) without loosing either features
        y = y.view(B, C).unsqueeze(1).expand(-1, self.levels, -1)
        H = torch.cat([X, y], dim=-1)

        return H, G