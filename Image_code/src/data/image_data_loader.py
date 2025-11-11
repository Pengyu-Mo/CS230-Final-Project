import os, csv, re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageVADDataset(Dataset):
    def __init__(
        self,
        tsv_path: str,
        img_root: str,
        vad_lexicon_path: str,
        label_prefix: str = "Art (image+title): ",  
        img_size: int = 224
    ):
        self.img_root = img_root
        self.label_prefix = label_prefix

        self.vad_dict = {}
        with open(vad_lexicon_path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split("\t")
                term = parts[0].lower()
                mV = re.search(r"V\s*=\s*([+-]?\d+(?:\.\d+)?)", s, re.I)
                mA = re.search(r"A\s*=\s*([+-]?\d+(?:\.\d+)?)", s, re.I)
                mD = re.search(r"D\s*=\s*([+-]?\d+(?:\.\d+)?)", s, re.I)
                if mV and mA and mD:
                    V = float(mV.group(1)); A = float(mA.group(1)); D = float(mD.group(1))
                    self.vad_dict[term] = (V, A, D)

        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            self.rows = list(reader)
            fieldnames = reader.fieldnames or []

        self.emotion_cols = [c for c in fieldnames if c.startswith(self.label_prefix)]

        if "ID" not in fieldnames:
            raise KeyError("TSV 中未找到 'ID' 列，请确认表头。")
        self.id_col = "ID"

        if isinstance(img_size, int):
            size_tuple = (img_size, img_size)
        else:
            assert len(img_size) == 2, "img_size must be int or (H, W)"
            size_tuple = tuple(img_size)

        self.tf = transforms.Compose([
            transforms.Resize(size_tuple),
            transforms.CenterCrop(size_tuple),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        img_id = (row.get(self.id_col) or "").strip()
        if not img_id:
            raise KeyError(f"Row {idx} 缺少 ID 值。")
        img_path = os.path.join(self.img_root, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)

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

        return x, y
