import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ArtistStyleClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout=0.4):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.classifier(self.dropout(features))

        if return_features:
            return logits, features
        return logits