# infer.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class SharedVADModel(nn.Module):
    def __init__(self, dim_audio=768, dim_image=512, hidden_dim=128, vad_dim=3):
        super().__init__()
        self.proj_audio = nn.Sequential(
            nn.Linear(dim_audio, hidden_dim),
            nn.ReLU(),
        )
        self.proj_image = nn.Sequential(
            nn.Linear(dim_image, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, vad_dim)

    def encode_audio(self, h_audio):
        return self.proj_audio(h_audio)

    def encode_image(self, h_image):
        return self.proj_image(h_image)

    def forward_from_hidden(self, h_hidden):
        return self.head(h_hidden)


def run_inference(
    npz_path,
    ckpt_path="best_vad_model.pt",
    batch_size=64,
    dim_audio=768,
    dim_image=512,
    hidden_dim=128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data = np.load(npz_path, allow_pickle=True)
    print("keys in npz:", data.files)

    paths = data["paths"]

    if "features_512" in data.files:
        feats = data["features_512"].astype(np.float32)
    elif "features_768" in data.files:
        feats = data["features_768"].astype(np.float32)
    elif "features_128" in data.files:
        feats = data["features_128"].astype(np.float32)
    elif "features" in data.files:
        feats = data["features"].astype(np.float32)
    else:
        raise ValueError(f"Cannot find feature key in npz, got keys = {data.files}")

    if "vad_gt" in data.files:
        vad_gt = data["vad_gt"].astype(np.float32)
    elif "vad_pred" in data.files:
        vad_gt = data["vad_pred"].astype(np.float32)
    else:
        raise ValueError("Cannot find 'vad_gt' or 'vad_pred' in npz")

    N, D = feats.shape
    vad_dim = vad_gt.shape[1]
    print(f"Loaded {N} samples, feature dim = {D}, vad_dim = {vad_dim}")

    feats_t = torch.from_numpy(feats)
    vad_gt_t = torch.from_numpy(vad_gt)

    ds = TensorDataset(feats_t, vad_gt_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = SharedVADModel(
        dim_audio=dim_audio,
        dim_image=dim_image,
        hidden_dim=hidden_dim,
        vad_dim=vad_dim,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")

    if D == dim_audio:
        modality = "audio"
        print("Treating features as AUDIO (dim = 768), use encode_audio")
    elif D == dim_image:
        modality = "image"
        print("Treating features as IMAGE (dim = 512), use encode_image")
    elif D == hidden_dim:
        modality = "hidden"
        print("Treating features as already HIDDEN (dim = 128), skip encoder")
    else:
        raise ValueError(
            f"Feature dim {D} does not match audio({dim_audio}), "
            f"image({dim_image}) or hidden({hidden_dim})"
        )

    all_hidden = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if modality == "audio":
                z = model.encode_audio(x)
            elif modality == "image":
                z = model.encode_image(x)
            else:  # hidden
                z = x  

            pred = model.forward_from_hidden(z)

            all_hidden.append(z.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    hidden_128 = np.concatenate(all_hidden, axis=0)   # [N, 128]
    preds = np.concatenate(all_preds, axis=0)         # [N, 3]
    targets = np.concatenate(all_targets, axis=0)     # [N, 3]

    mse = ((preds - targets) ** 2).mean()
    mse_per_dim = ((preds - targets) ** 2).mean(axis=0)
    print("Overall MSE:", mse)
    print("MSE per dim:", mse_per_dim)

    for i in range(min(5, N)):
        print(f"[{i}] path: {paths[i]}")
        print("    gt_vad:      ", targets[i])
        print("    pred_vad:    ", preds[i])
        print("    hidden_128[:5]:", hidden_128[i][:5])

    out_path = npz_path.replace(".npz", "_with_model_pred.npz")
    np.savez(
        out_path,
        paths=paths,
        features=hidden_128,          
        vad_gt=targets,
        vad_pred=preds,
    )
    print("Saved predictions & hidden features to:", out_path)


if __name__ == "__main__":
    run_inference(
        npz_path="image_features_test.npz",
        ckpt_path="best_vad_model.pt",
    )
