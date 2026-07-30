import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cbam import CBAM
from fem import FEM
from qco import QCO

class QFENet(nn.Module):
    def __init__(self, shallow_stage, deep_stage, shallow_channels = 32, deep_channels = 1536, num_classes = 1000):
        super(QFENet, self).__init__()
        self.shallow_stage = shallow_stage
        self.deep_stage = deep_stage
        self.num_classes = num_classes

        self.fem = FEM(shallow_channels)
        self.cbam = CBAM(deep_channels)


        self.conv1x1 = nn.Conv2d(deep_channels,shallow_channels,1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(shallow_channels * 2 ,num_classes)
        )

    def forward(self, X):

        #Low Level Features
        L = self.shallow_stage(X)

        #Low Level Enhanced Features
        L_prime = self.fem(L)

        #High Level Features
        H = self.deep_stage(L_prime)

        #Convolution Block Attention Module (CBAM) applied to High Level Features in Image
        H_prime = self.cbam(H)

        #Pass through 1x1 Conv Block
        H_prime = self.conv1x1(H_prime)
        H_prime = self.pool(H_prime)

        #Lower Features for Fully Connected Layer
        L_pool = self.pool(L_prime).flatten(1)
        H_pool = self.pool(H_prime).flatten(1)

        #Features = Low Level Feature + High Level Feature 
        features = torch.cat([L_pool,H_pool],dim= 1)

        return self.classifier(features)
