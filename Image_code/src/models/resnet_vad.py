import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights

class EmbeddingHead(nn.Module):
    """Projects backbone features to embedding space."""
    def __init__(self, in_dim, out_dim=128, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(p_drop),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
        )
    def forward(self, h):
        return self.net(h)

class VADHead(nn.Module):
    """Predicts VAD from embeddings."""
    def __init__(self, in_dim, vad_dim=3, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(p_drop),
            nn.Linear(in_dim, vad_dim)
        )
    def forward(self, z):
        return self.net(z)

class ResNetVAD(nn.Module):
    def __init__(self, vad_dim=3, emb_dim=128, backbone="resnet18", pretrained=True, freeze_early_layers=False, dropout=0.3):
        super().__init__()
        assert backbone in ["resnet18","resnet34","resnet50","resnet101"]
        resnet_fn = getattr(torchvision.models, backbone)
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None 
        base = resnet_fn(weights=weights)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base
        self.emb_head = EmbeddingHead(feat_dim, emb_dim, p_drop=dropout) 
        self.vad_head = VADHead(emb_dim, vad_dim, p_drop=dropout)    
        
        if freeze_early_layers:
            for name, param in self.backbone.named_parameters():
                # Only keep layer4 trainable
                if not name.startswith('layer4'):
                    param.requires_grad = False

    def forward(self, x):
        h = self.backbone(x) 
        z = self.emb_head(h) 
        vad = self.vad_head(z)
        return z, vad, h