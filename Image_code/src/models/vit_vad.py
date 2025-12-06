import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import (
    vit_b_16, vit_b_32, vit_l_16, vit_l_32,
    ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights
)


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


class ViTVAD(nn.Module):
    """
    Vision Transformer based VAD prediction model.
    """
    
    VIT_CONFIGS = {
        "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1, 768, 12),
        "vit_b_32": (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1, 768, 12),
        "vit_l_16": (vit_l_16, ViT_L_16_Weights.IMAGENET1K_V1, 1024, 24),
        "vit_l_32": (vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1, 1024, 24),
    }
    
    def __init__(self, vad_dim=3, emb_dim=128, backbone="vit_b_16", pretrained=True, 
                 freeze_encoder=False, unfreeze_last_n_blocks=0, dropout=0.3):
        super().__init__()
        
        assert backbone in self.VIT_CONFIGS, f"Unknown backbone: {backbone}. Choose from {list(self.VIT_CONFIGS.keys())}"
        
        vit_fn, weights, feat_dim, num_blocks = self.VIT_CONFIGS[backbone]
        self.feat_dim = feat_dim
        self.num_blocks = num_blocks
        
        if pretrained:
            self.backbone = vit_fn(weights=weights)
        else:
            self.backbone = vit_fn(weights=None)

        self.backbone.heads = nn.Identity()
        
        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            if unfreeze_last_n_blocks > 0:                
                total_blocks = len(self.backbone.encoder.layers)
                start_unfreeze = total_blocks - unfreeze_last_n_blocks
                for i, block in enumerate(self.backbone.encoder.layers):
                    if i >= start_unfreeze:
                        for param in block.parameters():
                            param.requires_grad = True
                if hasattr(self.backbone.encoder, 'ln'):
                    for param in self.backbone.encoder.ln.parameters():
                        param.requires_grad = True
        
        self.emb_head = EmbeddingHead(feat_dim, emb_dim, p_drop=dropout)
        self.vad_head = VADHead(emb_dim, vad_dim, p_drop=dropout)

    def forward(self, x):
        h = self.backbone(x)
        z = self.emb_head(h)
        vad = self.vad_head(z)
        return z, vad, h


def get_vit_transform(backbone="vit_b_16", img_size=224):
    """Get appropriate transform for ViT models."""
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

