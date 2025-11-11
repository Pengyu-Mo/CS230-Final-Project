import os
import math
import yaml
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from src.models.resnet import ResNetVAD
from src.data.image_data_loader import ImageVADDataset
from src.losses.mse_ccc_loss import mse_ccc_loss, mse_metric
from src.utils.utils import seed_everything, save_checkpoint, best_is_lower

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type=="cuda"):
                _, pred, _ = model(x)
                mse = F.mse_loss(pred, y, reduction="sum").item()
            mse_sum += mse
            count += y.numel()
    return mse_sum / count

def main():
    cfg = load_config("../configs/train_image.yaml")
    seed_everything(cfg["train"]["seed"])

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

    # Data
    data_info = ImageVADDataset(
        tsv_path=cfg["data"]["tsv_path"],
        img_root=cfg["data"]["img_root"],
        vad_lexicon_path=cfg["data"]["vad_lexicon_path"],
        label_prefix="Art (image+title): ",
        img_size=cfg["data"]["img_size"]
    )
    
    train_set, val_set, test_set = split_dataset(
                                    data_info, 
                                    val_ratio=cfg["data"]["val_ratio"], 
                                    test_ratio=cfg["data"]["test_ratio"], 
                                    seed=cfg["train"]["seed"]
                                )

    # For val/test use deterministic transform (already default), train can add augments if needed
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=False)
    val_loader   = DataLoader(val_set,   batch_size=cfg["train"]["batch_size"], shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=cfg["train"]["batch_size"], shuffle=False,
                              num_workers=4, pin_memory=True)

    # Model
    model = ResNetVAD(
        vad_dim=cfg["data"]["vad_dim"],
        emb_dim=cfg["model"]["emb_dim"],
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"]
    ).to(device)

    # Optim
    params = [
        {"params": model.proj.parameters(), "lr": cfg["train"]["lr"]},
        {"params": model.vad_head.parameters(), "lr": cfg["train"]["lr"]},
        {"params": model.backbone.parameters(), "lr": cfg["train"]["lr"] * 0.1},
    ]
    optim = torch.optim.AdamW(params, weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device.type=="cuda")

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
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg["train"]["amp"] and device.type=="cuda"):
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

#     # Optional: export 128-d embeddings for the test split
#     export_embeddings(model, test_loader, device, out_dir=cfg["logging"]["save_dir"])

# def export_embeddings(model, loader, device, out_dir):
#     import csv, numpy as np
#     os.makedirs(out_dir, exist_ok=True)
#     model.eval()
#     Z, rows = [], []
#     with torch.no_grad():
#         for x, y, paths in loader:
#             x = x.to(device, non_blocking=True)
#             z, pred, _ = model(x)
#             Z.append(z.cpu().numpy())
#             for i, p in enumerate(paths):
#                 row = {"image_path": p}
#                 vp = pred[i].cpu().numpy()
#                 names = ["valence","arousal","dominance"][:pred.shape[1]]
#                 for k, v in zip(names, vp): row[f"pred_{k}"] = float(v)
#                 rows.append(row)
#     Z = np.concatenate(Z, axis=0)
#     np.save(os.path.join(out_dir, "embeddings_test.npy"), Z)
#     with open(os.path.join(out_dir, "index_test.csv"), "w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#         w.writeheader(); w.writerows(rows)
#     wandb.save(os.path.join(out_dir, "embeddings_test.npy"))
#     wandb.save(os.path.join(out_dir, "index_test.csv"))
#     print(f"Saved embeddings_test.npy and index_test.csv to {out_dir}")

if __name__ == "__main__":
    main()