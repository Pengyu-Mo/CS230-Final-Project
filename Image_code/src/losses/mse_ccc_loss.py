import torch
import torch.nn.functional as F

def mse_ccc_loss(pred, target, ccc_weight=0.0, eps=1e-8):
    loss = F.mse_loss(pred, target)
    if ccc_weight > 0:
        x, y = pred, target
        vx = x.var(dim=0, unbiased=False) + eps
        vy = y.var(dim=0, unbiased=False) + eps
        mx, my = x.mean(dim=0), y.mean(dim=0)
        cov = ((x - mx) * (y - my)).mean(dim=0)
        ccc = 2 * cov / (vx + vy + (mx - my) ** 2 + eps)
        loss += ccc_weight * (1 - ccc.mean())
    return loss

@torch.no_grad()
def mse_metric(pred, target):
    return F.mse_loss(pred, target, reduction="mean").item()