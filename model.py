import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from timm.layers import cbam

class FeatureEnhancementModule(nn.Module):
    def __init__(self, in_channels: int, num_bins: int = 16):
        super(FeatureEnhancementModule, self).__init__()
        self.num_bins = num_bins
        self.in_channels = in_channels
        
        self.phi1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.phi2 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.phi3 = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def compute_qco(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        
        norm_x = F.normalize(x_flat, p=2, dim=1)
        global_feat = norm_x.mean(dim=-1, keepdim=True)
         
        sim = torch.bmm(global_feat.transpose(1, 2), norm_x) 
        sim = (sim + 1.0) / 2.0
        
        centers = torch.linspace(0.0, 1.0, self.num_bins, device=x.device).view(1, self.num_bins, 1)
        sigma = 1.0 / self.num_bins
        
        G = torch.exp(-((sim - centers) ** 2) / (2 * sigma ** 2))
        G = F.softmax(G, dim=1)
        
        H_feat = torch.bmm(x_flat, G.transpose(1, 2))
        
        return G, H_feat

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        B, C, H_spatial, W_spatial = L.shape
        
        G, H_feat = self.compute_qco(L) 
        
        proj1 = self.phi1(H_feat)
        proj2 = self.phi2(H_feat)
        
        energy = torch.bmm(proj1.transpose(1, 2), proj2)
        X = F.softmax(energy, dim=-1)
        
        proj3 = self.phi3(H_feat)
        Y_prime = torch.bmm(proj3, X)
        W = torch.bmm(Y_prime, G)
        
        L_prime = W.view(B, C, H_spatial, W_spatial)
        
        return L_prime


class DualAttentionEnhancedNetwork(nn.Module):
    def __init__(self, shallow_stage: nn.Module, deep_stage: nn.Module, shallow_channels: int = 32, deep_channels: int = 1536, num_classes: int = 1000):
        super(DualAttentionEnhancedNetwork, self).__init__()
        
        self.shallow_stage = shallow_stage
        self.deep_stage = deep_stage
        
        self.fem = FeatureEnhancementModule(in_channels=shallow_channels, num_bins=16)
        self.CbamModule = cbam.CbamModule(channels=deep_channels)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(shallow_channels + deep_channels, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = self.shallow_stage(x)
        L_prime = self.fem(L)
        
        H = self.deep_stage(L)
        H_prime = self.CbamModule(H)
        
        L_pool = self.global_pool(L_prime).flatten(1)
        H_pool = self.global_pool(H_prime).flatten(1)
        
        fused_features = torch.cat((L_pool, H_pool), dim=1)
        out = self.classifier(fused_features)
        
        return out