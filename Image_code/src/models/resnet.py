import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(p_drop),
            nn.Linear(in_dim, out_dim, bias=False)
        )
    def forward(self, h):
        z = self.net(h)
        return F.normalize(z, dim=-1)

class VADHead(nn.Module):
    def __init__(self, in_dim, vad_dim=3, hidden=256, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, vad_dim)
        )
    def forward(self, h):
        return self.net(h)

class ResNetVAD(nn.Module):
    def __init__(self, vad_dim=3, emb_dim=128, backbone="resnet18", pretrained=True):
        super().__init__()
        assert backbone in ["resnet18","resnet34","resnet50","resnet101"]
        resnet_fn = getattr(torchvision.models, backbone)
        # weights = "IMAGENET1K_V1" if pretrained else None
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None   # 以 resnet18 为例
        base = resnet_fn(weights=weights)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base
        self.proj = ProjectionHead(feat_dim, emb_dim)
        self.vad_head = VADHead(feat_dim, vad_dim)

    def forward(self, x):
        h = self.backbone(x)    # [B, feat_dim]
        z = self.proj(h)        # [B, emb_dim], L2-normalized
        vad = self.vad_head(h)  # [B, vad_dim]
        return z, vad, h