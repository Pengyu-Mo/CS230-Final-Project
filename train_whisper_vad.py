import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import whisper
from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim, N_FRAMES

# ===================== 新增数据配置（只改数据部分） =====================
AUDIO_DIR = Path("merged_audio")   # 你截图里的音频目录
LABEL_TSV = "autotagging_moodtheme_filtered_with_vad_clean.tsv"  # 刚刚处理后的 TSV（含 V/A/D）
TRAIN_RATIO = 0.85                 # 训练/验证划分比例（约两千条，85/15 较稳）
# ======================================================================

# ===================== 原有配置（不改） =====================
DATA_ROOT = Path("Data")          # 旧的目录（现不再使用，但保留变量）
WHISPER_MODEL_NAME = "base"       # 可选: "tiny"/"base"/"small"
SAMPLE_RATE = 16000
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 30
SEED = 42
TRAIN_PER_CLASS = 3               # 旧逻辑参数（现在仅在 wandb config 中记录，无实际作用）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================================


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- 仅用于读取 TSV 的小工具：健壮解析 ----------
def robust_read_tracks_tsv(path: str):
    """逐行读 TSV，最后三列必须是 V A D；返回列表 [ (path_str, (v,a,d)) , ... ]"""
    items = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()  # 丢掉表头
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            try:
                v = float(parts[-3]); a = float(parts[-2]); d = float(parts[-1])
            except ValueError:
                continue
            # 第4列通常是 PATH
            path_col = parts[3]
            items.append((path_col, (v, a, d)))
    return items


def resolve_audio_path(name: str) -> Path:
    """
    把 TSV 的 PATH 映射到 merged_audio 下真实文件：
    - 先找 <name>（如 '6606.mp3'）
    - 找不到再尝试把 '.mp3' 换成 '.low.mp3'（如 '6606.low.mp3'）
    - 都找不到就返回一个不存在的路径（后续会跳过）
    """
    p1 = AUDIO_DIR / name
    if p1.exists():
        return p1
    if name.lower().endswith(".mp3"):
        p2 = AUDIO_DIR / (name[:-4] + ".low.mp3")
        if p2.exists():
            return p2
    return p1  # 不存在；上层会过滤


# ---------- 数据划分：按比例随机切 ----------
def split_train_val_pairs(pairs: List[Tuple[Path, Tuple[float, float, float]]],
                          train_ratio: float, seed: int):
    rng = random.Random(seed)
    idxs = list(range(len(pairs)))
    rng.shuffle(idxs)
    n_train = int(len(pairs) * train_ratio)
    train = [pairs[i] for i in idxs[:n_train]]
    val = [pairs[i] for i in idxs[n_train:]]
    return train, val


# ===================== Dataset（只改成直接用 VAD 向量） =====================
class WhisperVADDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, Tuple[float, float, float]]]):
        """
        items: [(audio_path, (V, A, D)), ...]
        """
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, vad = self.items[idx]
        # 1) 读取并重采样到 32k
        audio = load_audio(str(path))                 # float32, 32kHz
        # 2) 固定到 30s（Whisper encoder 需要 3000 帧）
        audio = pad_or_trim(audio)                    # 保证长度匹配
        # 3) 计算 mel
        mel = log_mel_spectrogram(audio)              # [80, 3000]
        # 4) 目标
        target = torch.tensor(vad, dtype=torch.float32)  # [3]
        return mel, target, str(path), ""


# ===================== 原模型 & 训练流程（不改） =====================
class WhisperEncoderHead(nn.Module):
    """ 冻结 Whisper encoder，仅训练小头：
        mean-pool over time -> Linear(d,128)->ReLU->Linear(128,3)
    """
    def __init__(self, whisper_model: whisper.Whisper, hidden=128):
        super().__init__()
        self.wm = whisper_model
        # 冻结 encoder 参数
        for p in self.wm.parameters():
            p.requires_grad = False
        # 探测 encoder 输出维度
        with torch.no_grad():
            dummy = torch.zeros(1, 80, 3000)
            enc_out = self.wm.encoder(dummy.to(next(self.wm.parameters()).device))
            d = enc_out.shape[-1]
        self.head = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )

    def forward(self, mel: torch.Tensor):
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        enc = self.wm.encoder(mel)            # [B, T', d]
        x = enc.mean(dim=1)                   # [B, d]
        out = self.head(x)                    # [B, 3]
        return out


def collate_fn(batch):
    mels, targets, paths, labels = zip(*batch)
    max_T = max(m.shape[1] for m in mels)
    padded = []
    for m in mels:
        if m.shape[1] < max_T:
            pad = torch.zeros(80, max_T - m.shape[1], dtype=m.dtype)
            m = torch.cat([m, pad], dim=1)
        padded.append(m)
    mels = torch.stack(padded, dim=0)      # [B, 80, T]
    targets = torch.stack(targets, dim=0)  # [B, 3]
    return mels, targets, paths, labels


def train_one_epoch(model, loader, optim, loss_fn):
    model.train()
    total_loss = 0.0
    for mel, y, _, _ in loader:
        mel = mel.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(mel)
        loss = loss_fn(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * mel.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    for mel, y, _, _ in loader:
        mel = mel.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(mel)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * mel.size(0)
    return total_loss / len(loader.dataset)


def main():
    set_seed(SEED)

    # === 初始化 wandb（保持不变） ===
    wandb.init(
        project="whisper-vad-regression",
        config={
            "model": WHISPER_MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "train_per_class": TRAIN_PER_CLASS,
            "device": DEVICE,
            "train_ratio": TRAIN_RATIO,
            "audio_dir": str(AUDIO_DIR),
            "label_tsv": LABEL_TSV,
        },
        name=f"{WHISPER_MODEL_NAME}_head_lr{LR}_bs{BATCH_SIZE}",
    )
    config = wandb.config

    # ====== 新数据加载：从 TSV 取 V/A/D + merged_audio 下找文件 ======
    raw_items = robust_read_tracks_tsv(LABEL_TSV)  # [(path_str, (v,a,d)), ...]
    pairs = []
    missing = 0
    for rel, vad in raw_items:
        p = resolve_audio_path(rel)
        if p.exists():
            pairs.append((p, vad))
        else:
            missing += 1

    if not pairs:
        raise RuntimeError("未在 merged_audio 中找到可用音频，请检查路径或文件名后缀。")

    train_items, val_items = split_train_val_pairs(pairs, TRAIN_RATIO, SEED)

    print(f"总样本: {len(pairs)} | 缺失音频: {missing}")
    print(f"训练样本: {len(train_items)} | 验证样本: {len(val_items)}")

    train_ds = WhisperVADDataset(train_items)
    val_ds = WhisperVADDataset(val_items)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"加载 Whisper 模型: {WHISPER_MODEL_NAME} ...")
    wm = whisper.load_model(WHISPER_MODEL_NAME, device=DEVICE)
    model = WhisperEncoderHead(wm, hidden=128).to(DEVICE)

    optimizer = torch.optim.Adam(model.head.parameters(), lr=LR)  # 只训练 head
    loss_fn = nn.MSELoss()  # 回归到 V/A/D

    best_val = float("inf")
    best_path = "vad_head_best.pt"

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = eval_epoch(model, val_loader, loss_fn)

        print(f"[{epoch:02d}/{EPOCHS}] train={tr_loss:.4f} | val={val_loss:.4f}")
        wandb.log({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict()}, best_path)
            wandb.run.summary["best_val_loss"] = best_val

    print(f"训练完成。最佳验证 MSE: {best_val:.4f}  已保存到 {best_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
