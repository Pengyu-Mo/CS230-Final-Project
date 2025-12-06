import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import wandb



class SharedVADModel(nn.Module):
    def __init__(self, dim_audio=768, dim_image=512, hidden_dim=128, vad_dim=3):
        super().__init__()
        # audio: 768 → 128
        self.proj_audio = nn.Sequential(
            nn.Linear(dim_audio, hidden_dim),
            nn.ReLU(),
        )
        # image: 512 → 128
        self.proj_image = nn.Sequential(
            nn.Linear(dim_image, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, vad_dim)

    def forward_audio(self, h_audio):
        z = self.proj_audio(h_audio)
        vad_pred = self.head(z)
        return vad_pred

    def forward_image(self, h_image):
        z = self.proj_image(h_image)
        vad_pred = self.head(z)
        return vad_pred



def evaluate_vad_mse(model, loader, device, modality="audio"):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if modality == "audio":
                pred = model.forward_audio(x)
            elif modality == "image":
                pred = model.forward_image(x)
            else:
                raise ValueError("modality must be 'audio' or 'image'")

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mse = ((all_preds - all_targets) ** 2).mean()
    return mse



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    audio_train_npz = np.load("hubert_trainset_features_768_backbone_train.npz", allow_pickle=True)
    audio_val_npz   = np.load("hubert_testset_features_768_backbone_test.npz", allow_pickle=True)  # 实际是 val

    a_train_feat = torch.from_numpy(audio_train_npz["features_768"]).float()  # [Na_train, 768]
    a_train_vad  = torch.from_numpy(audio_train_npz["vad_gt"]).float()        # [Na_train, vad_dim]

    a_val_feat = torch.from_numpy(audio_val_npz["features_768"]).float()      # [Na_val, 768]
    a_val_vad  = torch.from_numpy(audio_val_npz["vad_gt"]).float()            # [Na_val, vad_dim]

    img_train_npz = np.load("image_features_train.npz", allow_pickle=True)
    img_val_npz   = np.load("image_features_test.npz", allow_pickle=True)      # 实际是 val

    i_train_feat = torch.from_numpy(img_train_npz["features_512"]).float()    # [Ni_train, 512]
    i_train_vad  = torch.from_numpy(img_train_npz["vad_gt"]).float()          # [Ni_train, vad_dim]

    i_val_feat = torch.from_numpy(img_val_npz["features_512"]).float()        # [Ni_val, 512]
    i_val_vad  = torch.from_numpy(img_val_npz["vad_gt"]).float()              # [Ni_val, vad_dim]

    print("audio_train_feat:", a_train_feat.shape, "audio_val_feat:", a_val_feat.shape)
    print("image_train_feat:", i_train_feat.shape, "image_val_feat:", i_val_feat.shape)

    vad_dim = a_train_vad.shape[1]
    print("VAD dim:", vad_dim)

    batch_size = 32

    audio_train_ds = TensorDataset(a_train_feat, a_train_vad)
    img_train_ds   = TensorDataset(i_train_feat, i_train_vad)

    audio_val_ds = TensorDataset(a_val_feat, a_val_vad)
    img_val_ds   = TensorDataset(i_val_feat, i_val_vad)

    audio_train_loader = DataLoader(audio_train_ds, batch_size=batch_size, shuffle=True)
    img_train_loader   = DataLoader(img_train_ds,   batch_size=batch_size, shuffle=True)

    audio_val_loader = DataLoader(audio_val_ds, batch_size=batch_size, shuffle=False)
    img_val_loader   = DataLoader(img_val_ds,   batch_size=batch_size, shuffle=False)

    model = SharedVADModel(
        dim_audio=a_train_feat.shape[1],
        dim_image=i_train_feat.shape[1],
        hidden_dim=64,
        vad_dim=vad_dim,
    ).to(device)

    criterion = nn.MSELoss()
    lr_audio = 3e-5
    lr_image = 1e-4      
    lr_head  = 3e-4

    optimizer = torch.optim.Adam(
        [
            {"params": model.proj_audio.parameters(), "lr": lr_audio},
            {"params": model.proj_image.parameters(), "lr": lr_image},
            {"params": model.head.parameters(),       "lr": lr_head},
        ],
        weight_decay=1e-4,   
    )
    num_epochs = 10

    wandb.init(
        project="cs230-multimodal-vad",
        config={
            "batch_size": batch_size,
            "lr": 3e-4,
            "hidden_dim": 128,
            "num_epochs": num_epochs,
            "vad_dim": vad_dim,
        },
    )
    wandb.watch(model)

    best_val_score = float("inf")   
    best_epoch = -1
    best_ckpt_path = "best_vad_model.pt"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x, y in audio_train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model.forward_audio(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_cur = x.size(0)
            total_loss += loss.item() * batch_size_cur
            total_samples += batch_size_cur

        for x, y in img_train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model.forward_image(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_cur = x.size(0)
            total_loss += loss.item() * batch_size_cur
            total_samples += batch_size_cur

        avg_train_loss = total_loss / max(total_samples, 1)

        audio_mse = evaluate_vad_mse(model, audio_val_loader, device, modality="audio")
        img_mse   = evaluate_vad_mse(model, img_val_loader,   device, modality="image")

        val_loss = (audio_mse + img_mse) / 2.0

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"train_loss={avg_train_loss:.4f} "
            f"audio_val_mse={audio_mse:.4f} "
            f"img_val_mse={img_mse:.4f} "
            f"val_loss={val_loss:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "audio_val_mse": audio_mse,
            "img_val_mse": img_mse,
            "val_loss": val_loss,
        })


        if val_loss < best_val_score:
            best_val_score = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  ↳ New best model at epoch {best_epoch}, val_loss={val_loss:.4f}")

    print(f"Training finished. Best epoch = {best_epoch}, best val_loss = {best_val_score:.4f}")
    print(f"Best checkpoint saved to {best_ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
