"""
Unified training script supporting ResNet, CLIP, and ViT backbones for VAD prediction.
"""
import os
import math
import argparse
import yaml
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

from src.models.resnet_vad import ResNetVAD
from src.models.clip_vad import CLIPVAD, get_clip_transform
from src.models.vit_vad import ViTVAD, get_vit_transform
from src.data.image_data_loader import ImageVADDataset
from src.losses.mse_ccc_loss import mse_ccc_loss, mse_metric
from src.utils.utils import seed_everything, save_checkpoint, best_is_lower


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_transform(cfg):
    """Get appropriate transform based on model type."""
    model_type = cfg["model"].get("type", "resnet")
    img_size = cfg["data"]["img_size"]
    
    if model_type == "clip":
        return get_clip_transform(cfg["model"]["backbone"])
    elif model_type == "vit":
        return get_vit_transform(cfg["model"]["backbone"], img_size)
    else:  # resnet
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def build_model(cfg, device):
    """Build model based on config."""
    model_type = cfg["model"].get("type", "resnet")
    dropout = cfg["train"].get("dropout", 0.3)
    
    if model_type == "clip":
        model = CLIPVAD(
            vad_dim=cfg["data"]["vad_dim"],
            emb_dim=cfg["model"]["emb_dim"],
            clip_model=cfg["model"]["backbone"],
            freeze_encoder=cfg["model"].get("freeze_encoder", True),
            dropout=dropout
        )
        print(f"Built CLIP model: {cfg['model']['backbone']}, feat_dim={model.feat_dim}, encoder frozen, dropout={dropout}")
        
    elif model_type == "vit":
        freeze_encoder = cfg["model"].get("freeze_encoder", True)
        unfreeze_last_n = cfg["model"].get("unfreeze_last_n_blocks", 0)
        model = ViTVAD(
            vad_dim=cfg["data"]["vad_dim"],
            emb_dim=cfg["model"]["emb_dim"],
            backbone=cfg["model"]["backbone"],
            pretrained=cfg["model"].get("pretrained", True),
            freeze_encoder=freeze_encoder,
            unfreeze_last_n_blocks=unfreeze_last_n,
            dropout=dropout
        )
        print(f"Built ViT model: {cfg['model']['backbone']}, feat_dim={model.feat_dim}, "
              f"freeze={freeze_encoder}, unfreeze_last_n={unfreeze_last_n}, dropout={dropout}")
        
    else:  # resnet
        freeze_early = cfg["model"].get("freeze_early_layers", False)
        model = ResNetVAD(
            vad_dim=cfg["data"]["vad_dim"],
            emb_dim=cfg["model"]["emb_dim"],
            backbone=cfg["model"]["backbone"],
            pretrained=cfg["model"].get("pretrained", True),
            freeze_early_layers=freeze_early,
            dropout=dropout
        )
        print(f"Built ResNet model: {cfg['model']['backbone']}, freeze_early={freeze_early}, dropout={dropout}")
    
    return model.to(device)


def get_optimizer(model, cfg):
    """Get optimizer with appropriate parameter groups (only trainable params)."""
    model_type = cfg["model"].get("type", "resnet")
    lr = cfg["train"]["lr"]
    
    # Always include heads (stacked: emb_head → vad_head)
    params = [
        {"params": model.emb_head.parameters(), "lr": lr},
        {"params": model.vad_head.parameters(), "lr": lr},
    ]
    
    if model_type == "clip":
        # CLIP encoder is frozen, only train heads
        if not cfg["model"].get("freeze_encoder", True):
            params.append({"params": model.clip.parameters(), "lr": lr * 0.01})
            
    elif model_type == "vit":
        # Add only trainable backbone params (last N blocks if unfrozen)
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        if backbone_params:
            params.append({"params": backbone_params, "lr": lr * 0.1})
            
    else:  # resnet
        # Add only trainable backbone params (layer4 if freeze_early_layers=True)
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        if backbone_params:
            params.append({"params": backbone_params, "lr": lr * 0.1})
    
    return torch.optim.AdamW(params, weight_decay=cfg["train"]["weight_decay"])


def split_dataset(data_info, val_ratio, test_ratio, seed=42):
    N = len(data_info)
    n_val = max(1, int(N * val_ratio))
    n_test = max(1, int(N * test_ratio))
    n_train = max(1, N - n_val - n_test)
    gen = torch.Generator().manual_seed(seed)
    return random_split(data_info, [n_train, n_val, n_test], generator=gen)


def evaluate(model, loader, device, amp=False):
    model.eval()
    mse_sum, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                _, pred, _ = model(x)
                mse = F.mse_loss(pred, y, reduction="sum").item()
            mse_sum += mse
            count += y.numel()
    return mse_sum / count


def main():
    parser = argparse.ArgumentParser(description="Train VAD model with ResNet/CLIP/ViT")
    parser.add_argument("--config", type=str, default="../configs/train_resnet.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    seed_everything(cfg["train"]["seed"])
    
    model_type = cfg["model"].get("type", "resnet")
    print(f"Training {model_type.upper()} model...")

    # WandB
    wandb.init(
        project=cfg["project"],
        entity=cfg.get("entity") or None,
        name=cfg["run_name"],
        mode=cfg["wandb_mode"],
        config=cfg,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get transform
    transform = get_transform(cfg)
    
    # Data
    data_info = ImageVADDataset(
        tsv_path=cfg["data"]["tsv_path"],
        img_root=cfg["data"]["img_root"],
        vad_lexicon_path=cfg["data"]["vad_lexicon_path"],
        label_prefix="Art (image+title): ",
        img_size=cfg["data"]["img_size"],
        transform=transform
    )
    
    train_set, val_set, test_set = split_dataset(
        data_info,
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        seed=cfg["train"]["seed"]
    )

    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             num_workers=4, pin_memory=True)

    # Model
    model = build_model(cfg, device)

    # Optimizer
    optim = get_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")

    wandb.watch(model, log="all", log_freq=cfg["logging"]["log_every_n_steps"])

    # Train loop
    best_metric = math.inf if best_is_lower(cfg["logging"]["save_best_metric"]) else -math.inf
    for ep in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss, seen = 0.0, 0

        for step, (x, y) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg["train"]["amp"] and device.type == "cuda"):
                _, pred, _ = model(x)
                loss = mse_ccc_loss(pred, y, ccc_weight=cfg["train"]["ccc_weight"])

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

            if step % cfg["logging"]["log_every_n_steps"] == 0:
                wandb.log({"train/loss_step": loss.item()})

        train_loss = running_loss / seen
        val_mse = evaluate(model, val_loader, device, amp=cfg["train"]["amp"])

        wandb.log({
            "epoch": ep,
            "train/loss": train_loss,
            "val/mse": val_mse
        })

        # save ckpt
        is_better = (val_mse < best_metric) if best_is_lower(cfg["logging"]["save_best_metric"]) else (val_mse > best_metric)
        if is_better:
            best_metric = val_mse
        save_checkpoint({
            "epoch": ep,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "best_metric": best_metric,
            "config": cfg
        }, is_better, cfg["logging"]["save_dir"])

        print(f"[Epoch {ep:03d}] train/loss={train_loss:.6f}  val/mse={val_mse:.6f}")

    # Final test
    best_ckpt = os.path.join(cfg["logging"]["save_dir"], "best.pt")
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
        print(f"Loaded best checkpoint from {best_ckpt}")

    test_mse = evaluate(model, test_loader, device, amp=cfg["train"]["amp"])
    wandb.log({"test/mse": test_mse})
    print(f"[Test] mse={test_mse:.6f}")


if __name__ == "__main__":
    main()

