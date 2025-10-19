# baseline_cnn.py
import torch
import torch.nn as nn
import torchvision.models as models

class TemporalAttention(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.Tanh(),
            nn.Linear(feature_dim // 4, 1)
        )
    
    def forward(self, x):
        attn_weights = torch.softmax(self.attention(x), dim=1)
        return (x * attn_weights).sum(dim=1)

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, temporal_pool="attention", dropout=0.5):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-2])  
        self.temporal_pool = temporal_pool
        
        self.temporal_attention = TemporalAttention(feature_dim=512) if temporal_pool == "attention" else None
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape

        # --- TEMPORAL DROPOUT (during training only) ---
        if self.training and torch.rand(1) < 0.3:  # apply 30% of the time
            keep_ratio = 0.7  # keep ~70% of frames
            keep_idx = torch.randperm(T)[:int(T * keep_ratio)]
            keep_idx, _ = torch.sort(keep_idx)  # keep chronological order
            x = x[:, :, keep_idx, :, :]
            T = x.shape[2]
        # ------------------------------------------------

        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)
        
        feats = self.features(x)
        feats = self.head[0](feats)
        feats = self.head[1](feats)
        feats = feats.view(B, T, -1)

        if self.temporal_pool == "attention" and self.temporal_attention is not None:
            pooled = self.temporal_attention(feats)
        elif self.temporal_pool == "avg":
            pooled = feats.mean(dim=1)
        elif self.temporal_pool == "max":
            pooled = feats.max(dim=1)[0]
        else:
            pooled = feats[:, 0]

        logits = self.head[2:](pooled)
        return logits