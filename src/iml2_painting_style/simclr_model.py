import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class SimCLR(nn.Module):
    def __init__(self, pretrained: bool = True,
                 projection_dim: int = 128,
                 hidden_dim: int = 256):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features  
        backbone.fc = nn.Identity()                 
        self.backbone = backbone

        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        h = self.backbone(x)                       
        z = self.projection_head(h)                
        z = F.normalize(z, dim=1)
        return h, z


class LinearClassifier(nn.Module):
    """
    Lightweight linear head for Stage 2 (linear probing) and Stage 3 (fine-tuning).
    """
    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, h):
        return self.fc(self.dropout(h))