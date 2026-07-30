import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from qco import QCO

class FEM(nn.Module):
    def __init__(self, in_channels):
        super(FEM, self).__init__()

        #Quantization and Counting Operator
        self.qco = QCO(in_channels)

        #Convolution Layers
        self.phi1 = nn.Sequential(
            nn.Conv1d(2 * in_channels, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.phi2 = nn.Sequential(
            nn.Conv1d(2 * in_channels, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.phi3 = nn.Sequential(
            nn.Conv1d(2 * in_channels, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, L):
        B, C, H, W = L.shape

        # Get Statistic Features(H) and Quantization Encoding Matrix(G) of Low-Level Features (L)
        H_qco, G_qco = self.qco(L)
        H_qco = H_qco.transpose(1,2)

        #Pass Statistic Features to 1x1 Conv Layers
        x1 = self.phi1(H_qco)
        x2 = self.phi2(H_qco)
        x3 = self.phi3(H_qco)

        #Softmax between x1 and x2
        X = F.softmax(
            torch.bmm(x1.transpose(1, 2), x2), dim=-1
        )

        #Final Set of Quantization Level (Y')
        Y = torch.bmm(x3, X)
        Y = Y.transpose(1, 2)

        #Final Output W
        W_prime = torch.bmm(G_qco, Y)
        W_prime = W_prime.transpose(1, 2)

        #Enhance Low Level Features (L_prime)
        L_prime = W_prime.reshape(B, C, H, W)

        return L_prime + L