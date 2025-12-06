import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import clip
except ImportError:
    raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")


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


class CLIPVAD(nn.Module):
    """
    CLIP-based VAD prediction model.
    """

    CLIP_FEAT_DIMS = {
        "RN50": 1024,
        "RN101": 512,
        "RN50x4": 640,
        "RN50x16": 768,
        "RN50x64": 1024,
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        "ViT-L/14@336px": 768,
    }
    
    def __init__(self, vad_dim=3, emb_dim=128, clip_model="ViT-B/32", freeze_encoder=True, dropout=0.3):
        super().__init__()
        
        self.clip_model_name = clip_model
        
        self.clip, self.preprocess = clip.load(clip_model, device="cpu")
        
        if clip_model in self.CLIP_FEAT_DIMS:
            feat_dim = self.CLIP_FEAT_DIMS[clip_model]
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                feat = self.clip.encode_image(dummy)
                feat_dim = feat.shape[-1]
        
        self.feat_dim = feat_dim
        
        if freeze_encoder:
            for param in self.clip.parameters():
                param.requires_grad = False
        
        self.emb_head = EmbeddingHead(feat_dim, emb_dim, p_drop=dropout)
        self.vad_head = VADHead(emb_dim, vad_dim, p_drop=dropout)
    
    def encode_image(self, x):
        """Extract image features from CLIP encoder."""
        with torch.no_grad():
            h = self.clip.encode_image(x)
        return h.float() 
    
    def forward(self, x):
        h = self.encode_image(x) 
        z = self.emb_head(h)  
        vad = self.vad_head(z)
        return z, vad, h
    
    def get_preprocess(self):
        return self.preprocess


def get_clip_transform(clip_model="ViT-B/32"):
    _, preprocess = clip.load(clip_model, device="cpu")
    return preprocess

