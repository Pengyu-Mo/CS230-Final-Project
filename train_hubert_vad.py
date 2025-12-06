import os
import random
from pathlib import Path
from typing import List, Tuple

import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import HubertModel
import librosa
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# ==========================
# Config
# ==========================
AUDIO_DIR = Path("merged_audio")
LABEL_TSV = "autotagging_moodtheme_filtered_with_vad_clean.tsv"
TRAIN_RATIO = 0.85

SAMPLE_RATE = 16000  # HuBERT requires 16kHz PCM
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 10
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================
# Utils
# ==========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def robust_read_tracks_tsv(path: str):
    items = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            try:
                v = float(parts[-3])
                a = float(parts[-2])
                d = float(parts[-1])
            except ValueError:
                continue
            path_col = parts[3]
            items.append((path_col, (v, a, d)))
    return items


def resolve_audio_path(name: str) -> Path:
    p1 = AUDIO_DIR / name
    if p1.exists():
        return p1
    if name.lower().endswith(".mp3"):
        p2 = AUDIO_DIR / (name[:-4] + ".low.mp3")
        if p2.exists():
            return p2
    return p1


def split_train_val_pairs(pairs, train_ratio, seed):
    rng = random.Random(seed)
    idxs = list(range(len(pairs)))
    rng.shuffle(idxs)
    n_train = int(len(pairs) * train_ratio)
    train = [pairs[i] for i in idxs[:n_train]]
    val = [pairs[i] for i in idxs[n_train:]]
    return train, val


# ==========================
# Dataset for HuBERT
# ==========================
class VADDataset(Dataset):
    MAX_LEN = SAMPLE_RATE * 10   # 10 秒

    def __init__(self, items, train: bool = True):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, vad = self.items[idx]
        audio, sr = librosa.load(path, sr=SAMPLE_RATE)

        if len(audio) > self.MAX_LEN:
            if self.train:
                # 训练集：随机裁剪
                start = random.randint(0, len(audio) - self.MAX_LEN)
            else:
                # 验证集：固定裁剪（这里用“居中裁剪”）
                start = (len(audio) - self.MAX_LEN) // 2
            audio = audio[start:start + self.MAX_LEN]
        else:
            pad = self.MAX_LEN - len(audio)
            audio = np.pad(audio, (0, pad), mode='constant')

        audio = torch.tensor(audio, dtype=torch.float32)
        target = torch.tensor(vad, dtype=torch.float32)
        return audio, target, str(path)



def collate_fn(batch):
    audios, targets, paths = zip(*batch)

    max_len = max(a.shape[0] for a in audios)
    padded = []
    for a in audios:
        if a.shape[0] < max_len:
            pad = torch.zeros(max_len - a.shape[0])
            a = torch.cat([a, pad], dim=0)
        padded.append(a)

    audios = torch.stack(padded)
    targets = torch.stack(targets)
    return audios, targets, paths


# ==========================
# HuBERT + Head
# ==========================
class HuBERTRegression(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.backbone = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960",
            use_safetensors=True   # <── 加上这一行
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        d = self.backbone.config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )

    def forward(self, waveform):
        # waveform: [B, T]
        outputs = self.backbone(waveform).last_hidden_state  # [B, T', 768]
        x = outputs.mean(dim=1)
        return self.head(x)


# ==========================
# Training helpers
# ==========================
def train_one_epoch(model, loader, optim, loss_fn):
    model.train()
    total = 0
    iterator = tqdm(loader) if tqdm else loader

    for audio, y, _ in iterator:
        audio = audio.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(audio)
        loss = loss_fn(pred, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total += loss.item() * audio.size(0)
        if tqdm:
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn):
    model.eval()
    total = 0
    iterator = tqdm(loader) if tqdm else loader

    for audio, y, _ in iterator:
        audio = audio.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(audio)
        loss = loss_fn(pred, y)

        total += loss.item() * audio.size(0)
        if tqdm:
            iterator.set_postfix({"val_loss": f"{loss.item():.4f}"})

    return total / len(loader.dataset)


# ==========================
# Main
# ==========================
def main():
    set_seed(SEED)

    wandb.init(project="hubert-vad-regression")

    raw_items = robust_read_tracks_tsv(LABEL_TSV)
    pairs = []
    for rel, vad in raw_items:
        p = resolve_audio_path(rel)
        if p.exists():
            pairs.append((p, vad))

    print(f"Total samples: {len(pairs)}")

    train_items, val_items = split_train_val_pairs(pairs, TRAIN_RATIO, SEED)

    train_ds = VADDataset(train_items, train=True)
    val_ds   = VADDataset(val_items,   train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # model
    model = HuBERTRegression().to(DEVICE)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val = eval_epoch(model, val_loader, loss_fn)

        print(f"[{epoch}/{EPOCHS}] train={tr:.4f} | val={val:.4f}")
        wandb.log({"train_loss": tr, "val_loss": val, "epoch": epoch})

        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), "hubert_vad_best_without_validation_cropping_with_weightdecay.pt")

    print("Finished. Best val:", best_val)
    wandb.finish()


if __name__ == "__main__":
    print("Using:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    main()
