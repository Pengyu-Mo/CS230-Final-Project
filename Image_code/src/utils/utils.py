import os, random
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def best_is_lower(metric_name: str) -> bool:
    return any(k in metric_name for k in ["loss","mse","mae"])

def save_checkpoint(state, is_best, save_dir, fname="last.pt", best_name="best.pt"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, fname))
    if is_best:
        torch.save(state, os.path.join(save_dir, best_name))