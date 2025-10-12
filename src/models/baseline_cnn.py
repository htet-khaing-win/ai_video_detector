# src/models/baseline_cnn.py
import torch
import torch.nn as nn
import torchvision.models as models

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, temporal_pool="avg"):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-2])  
        self.temporal_pool = temporal_pool
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # Now (B, T, C, H, W)
        
        # Reshape to process all frames at once
        x = x.reshape(B * T, C, H, W)
        feats = self.features(x)
        feats = self.head[0](feats)  # avgpool
        feats = feats.view(B, T, -1)

        if self.temporal_pool == "avg":
            pooled = feats.mean(dim=1)
        elif self.temporal_pool == "max":
            pooled = feats.max(dim=1)[0]
        else:
            pooled = feats[:, 0]

        logits = self.head[2](pooled)
        return logits