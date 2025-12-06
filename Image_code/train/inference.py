import os
import yaml
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, random_split, Subset
from PIL import Image
from torchvision import transforms

from src.models.resnet_vad import ResNetVAD
from src.models.clip_vad import CLIPVAD, get_clip_transform
from src.models.vit_vad import ViTVAD, get_vit_transform
from src.data.image_data_loader import ImageVADDataset
from src.utils.utils import seed_everything


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
        print(f"Built CLIP model: {cfg['model']['backbone']}, feat_dim={model.feat_dim}")
        
    elif model_type == "vit":
        model = ViTVAD(
            vad_dim=cfg["data"]["vad_dim"],
            emb_dim=cfg["model"]["emb_dim"],
            backbone=cfg["model"]["backbone"],
            pretrained=False,  # Will load from checkpoint
            freeze_encoder=cfg["model"].get("freeze_encoder", True),
            unfreeze_last_n_blocks=cfg["model"].get("unfreeze_last_n_blocks", 0),
            dropout=dropout
        )
        print(f"Built ViT model: {cfg['model']['backbone']}, feat_dim={model.feat_dim}")
        
    else:  # resnet
        model = ResNetVAD(
            vad_dim=cfg["data"]["vad_dim"],
            emb_dim=cfg["model"]["emb_dim"],
            backbone=cfg["model"]["backbone"],
            pretrained=False,  # Will load from checkpoint
            freeze_early_layers=cfg["model"].get("freeze_early_layers", False),
            dropout=dropout
        )
        print(f"Built ResNet model: {cfg['model']['backbone']}")
    
    return model.to(device)


class ImageVADDatasetWithPaths(ImageVADDataset):
    """Extended dataset that also returns image paths."""
    
    def __getitem__(self, idx):
        row = self.rows[idx]
        img_id = (row.get(self.id_col) or "").strip()
        if not img_id:
            raise KeyError(f"Row {idx} missing ID value.")
        
        img_path = os.path.join(self.img_root, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)
        
        # Compute VAD label
        vads = []
        for col in self.emotion_cols:
            raw = (row.get(col) or "").strip()
            if raw and raw != "0":
                emo_name = col[len(self.label_prefix):].strip().lower()
                if emo_name in self.vad_dict:
                    vads.append(torch.tensor(self.vad_dict[emo_name], dtype=torch.float32))
        
        if len(vads) == 0:
            y = torch.zeros(3, dtype=torch.float32)
        else:
            y = torch.stack(vads, dim=0).mean(dim=0)
        
        return x, y, img_path


def collate_fn(batch):
    """Custom collate function to handle paths."""
    xs, ys, paths = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(paths)


def split_dataset(dataset, val_ratio, test_ratio, seed=42):
    """Split dataset and return indices for reproducibility."""
    N = len(dataset)
    n_val = max(1, int(N * val_ratio))
    n_test = max(1, int(N * test_ratio))
    n_train = max(1, N - n_val - n_test)
    
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(N, generator=gen).tolist()
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return train_indices, val_indices, test_indices


def extract_features(model, loader, device, amp=False):
    """Extract features and VAD predictions from the model."""
    model.eval()
    
    all_paths = []
    all_features = []   
    all_backbone_features = []
    all_vad_preds = []
    all_targets = [] 
    
    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                z, vad_pred, h = model(x)
            
            all_features.append(z.float().cpu().numpy())
            all_backbone_features.append(h.float().cpu().numpy())
            all_vad_preds.append(vad_pred.float().cpu().numpy())
            all_targets.append(y.numpy())
            all_paths.extend(paths)
    
    features = np.concatenate(all_features, axis=0)
    backbone_features = np.concatenate(all_backbone_features, axis=0)
    vad_pred = np.concatenate(all_vad_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    paths = np.array(all_paths, dtype=object)
    
    mse = np.mean((vad_pred - targets) ** 2)
    rmse = np.sqrt(mse)
    
    mse_v = np.mean((vad_pred[:, 0] - targets[:, 0]) ** 2)
    mse_a = np.mean((vad_pred[:, 1] - targets[:, 1]) ** 2)
    mse_d = np.mean((vad_pred[:, 2] - targets[:, 2]) ** 2)
    
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mse_valence": mse_v,
        "mse_arousal": mse_a,
        "mse_dominance": mse_d,
    }
    
    print(f"\n{'='*50}")
    print(f"Loss Metrics:")
    print(f"  MSE (overall): {mse:.6f}")
    print(f"  RMSE (overall): {rmse:.6f}")
    print(f"  MSE (Valence):  {mse_v:.6f}")
    print(f"  MSE (Arousal):  {mse_a:.6f}")
    print(f"  MSE (Dominance): {mse_d:.6f}")
    print(f"{'='*50}")
    
    return paths, features, backbone_features, vad_pred, targets, metrics


def run_inference_on_split(model, dataset, indices, split_name, batch_size, device, amp):
    """Run inference on a specific split and return results."""
    subset = Subset(dataset, indices)
    print(f"\nProcessing {split_name} split with {len(subset)} samples")
    
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    paths, features, backbone_features, vad_pred, vad_gt, metrics = extract_features(
        model, loader, device, amp=amp
    )
    
    print(f"  paths shape: {paths.shape}")
    print(f"  features shape: {backbone_features.shape}")
    print(f"  vad_pred shape: {vad_gt.shape}")
    
    return paths, backbone_features, vad_gt, metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate NPZ file from trained image VAD model (ResNet/CLIP/ViT)")
    parser.add_argument("--config", type=str, default="../configs/train_resnet.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="outputs/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="outputs/features.npz",
                        help="Output NPZ file path")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--wandb_project", type=str, default="image_vad_eval",
                        help="WandB project name for logging")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    seed_everything(cfg["train"]["seed"])
    
    model_type = cfg["model"].get("type", "resnet")
    backbone = cfg["model"].get("backbone", "unknown")
    print(f"Model type: {model_type.upper()}, Backbone: {backbone}")
    
    run_name = f"{model_type}-{backbone}-eval".replace("/", "-")
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model_type": model_type,
            "backbone": backbone,
            "checkpoint": args.checkpoint,
            **cfg
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = get_transform(cfg)
    
    dataset = ImageVADDatasetWithPaths(
        tsv_path=cfg["data"]["tsv_path"],
        img_root=cfg["data"]["img_root"],
        vad_lexicon_path=cfg["data"]["vad_lexicon_path"],
        label_prefix="Art (image+title): ",
        img_size=cfg["data"]["img_size"],
        transform=transform
    )
    
    train_indices, val_indices, test_indices = split_dataset(
        dataset,
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        seed=cfg["train"]["seed"]
    )
    
    model = build_model(cfg, device)
    
    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        print(f"Loaded checkpoint from {args.checkpoint}")
        if "epoch" in state:
            print(f"  Checkpoint epoch: {state['epoch']}")
        if "best_metric" in state:
            print(f"  Best metric: {state['best_metric']:.6f}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    amp = cfg["train"].get("amp", False)
    
    train_paths, train_features, train_vad, train_metrics = run_inference_on_split(
        model, dataset, train_indices, "train", args.batch_size, device, amp
    )
    
    val_paths, val_features, val_vad, val_metrics = run_inference_on_split(
        model, dataset, val_indices, "val", args.batch_size, device, amp
    )
    
    test_paths, test_features, test_vad, test_metrics = run_inference_on_split(
        model, dataset, test_indices, "test", args.batch_size, device, amp
    )
    
    wandb.log({
        "train/mse": train_metrics["mse"],
        "train/rmse": train_metrics["rmse"],
        "train/mse_valence": train_metrics["mse_valence"],
        "train/mse_arousal": train_metrics["mse_arousal"],
        "train/mse_dominance": train_metrics["mse_dominance"],
        "val/mse": val_metrics["mse"],
        "val/rmse": val_metrics["rmse"],
        "val/mse_valence": val_metrics["mse_valence"],
        "val/mse_arousal": val_metrics["mse_arousal"],
        "val/mse_dominance": val_metrics["mse_dominance"],
        "test/mse": test_metrics["mse"],
        "test/rmse": test_metrics["rmse"],
        "test/mse_valence": test_metrics["mse_valence"],
        "test/mse_arousal": test_metrics["mse_arousal"],
        "test/mse_dominance": test_metrics["mse_dominance"],
    })
    
    wandb.run.summary["val_mse"] = val_metrics["mse"]
    wandb.run.summary["test_mse"] = test_metrics["mse"]
    wandb.run.summary["model"] = f"{model_type}-{backbone}"
    
    all_paths = np.concatenate([train_paths, val_paths, test_paths])
    all_features = np.concatenate([train_features, val_features, test_features])
    all_vad = np.concatenate([train_vad, val_vad, test_vad])
    
    split = np.array(['train'] * len(train_paths) + ['val'] * len(val_paths) + ['test'] * len(test_paths))
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        paths=all_paths,
        features=all_features,
        vad_pred=all_vad,
        split=split,
    )
    
    print(f"\n{'='*50}")
    print(f"Saved to {args.output}")
    print(f"  paths shape: {all_paths.shape}")
    print(f"  features shape: {all_features.shape}")
    print(f"  vad_pred shape: {all_vad.shape}")
    print(f"  split shape: {split.shape} (train: {len(train_paths)}, val: {len(val_paths)}, test: {len(test_paths)})")
    
    wandb.finish()


if __name__ == "__main__":
    main()
