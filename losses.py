"""
GeoSR-X Loss Functions
========================
Two configurations of the same components:
  - "CNN (paper baseline)": MSE only
  - "GeoSR-X (ours)":        MSE + alpha * SAM + beta * NDVI_consistency

Band order is fixed: [B2, B3, B4, B8] (indices 0,1,2,3).
NDVI = (B8 - B4) / (B8 + B4 + eps)  -> uses indices 3 and 2.

IMPORTANT (per the ground-truth discussion): NDVI_consistency here is
SELF-consistency -- it compares NDVI computed from the model's SR output
against NDVI computed from the real HR target in the SAME Wald-protocol
training pair. It is never computed against GEHistoricalImagery output,
which has no NIR band and cannot support NDVI at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-6
RED_IDX = 2  # B4
NIR_IDX = 3  # B8


def compute_ndvi(x: torch.Tensor):
    """x: (B, 4, H, W) -> (ndvi: (B,H,W), valid_mask: (B,H,W))
    Band order: B2=0, B3=1, B4(Red)=2, B8(NIR)=3
    """
    red = x[:, RED_IDX, :, :]
    nir = x[:, NIR_IDX, :, :]
    denom = nir + red
    valid = (denom > 0.01)
    ndvi = torch.where(valid, (nir - red) / (denom + EPS), torch.zeros_like(nir))
    return ndvi, valid


def compute_ndwi(x: torch.Tensor):
    """
    NDWI = (Green - NIR) / (Green + NIR)
    Band order: B2=0, B3(Green)=1, B4=2, B8(NIR)=3
    High NDWI = water. Low NDWI = vegetation/urban.
    """
    green = x[:, 1, :, :]   # B3
    nir   = x[:, NIR_IDX, :, :]  # B8
    denom = green + nir
    valid = (denom > 0.01)
    ndwi  = torch.where(valid, (green - nir) / (denom + EPS), torch.zeros_like(green))
    return ndwi, valid


def spectral_angle_mapper(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    SAM, averaged over pixels and batch, in radians.
    pred, target: (B, C, H, W)
    """
    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])    # (N, C)
    target_flat = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])

    dot = (pred_flat * target_flat).sum(dim=1)
    norm_pred = pred_flat.norm(dim=1) + EPS
    norm_target = target_flat.norm(dim=1) + EPS

    cos_angle = torch.clamp(dot / (norm_pred * norm_target), -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    return angle.mean()


class GeoSRLoss(nn.Module):
    """
    total = MSE + alpha * SAM + beta * NDVI_consistency

    Set alpha=beta=0 to reproduce the plain-MSE paper baseline with the
    same loss class (keeps train.py identical for both runs, only the
    weights differ).

    alpha/beta can be mutated after construction (e.g. by a warm-up
    schedule in the training loop) since they're plain floats, not
    registered buffers.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mse_loss = self.mse(pred, target)

        sam_loss = spectral_angle_mapper(pred, target) if self.alpha > 0 else torch.tensor(0.0, device=pred.device)

        if self.beta > 0:
            # NDVI consistency
            ndvi_pred,   valid_ndvi_pred   = compute_ndvi(pred)
            ndvi_target, valid_ndvi_target = compute_ndvi(target)
            valid_ndvi = valid_ndvi_pred & valid_ndvi_target
            if valid_ndvi.any():
                ndvi_loss = torch.mean((ndvi_pred[valid_ndvi] - ndvi_target[valid_ndvi]) ** 2)
            else:
                ndvi_loss = torch.tensor(0.0, device=pred.device)

            # NDWI consistency (same beta weight — same class of spectral index)
            ndwi_pred,   valid_ndwi_pred   = compute_ndwi(pred)
            ndwi_target, valid_ndwi_target = compute_ndwi(target)
            valid_ndwi = valid_ndwi_pred & valid_ndwi_target
            if valid_ndwi.any():
                ndwi_loss = torch.mean((ndwi_pred[valid_ndwi] - ndwi_target[valid_ndwi]) ** 2)
            else:
                ndwi_loss = torch.tensor(0.0, device=pred.device)
        else:
            ndvi_loss = torch.tensor(0.0, device=pred.device)
            ndwi_loss = torch.tensor(0.0, device=pred.device)

        total = mse_loss + self.alpha * sam_loss + self.beta * (ndvi_loss + ndwi_loss)

        return total, {
            "mse":   mse_loss.item(),
            "sam":   sam_loss.item() if torch.is_tensor(sam_loss) else sam_loss,
            "ndvi":  ndvi_loss.item() if torch.is_tensor(ndvi_loss) else ndvi_loss,
            "ndwi":  ndwi_loss.item() if torch.is_tensor(ndwi_loss) else ndwi_loss,
            "total": total.item(),
        }


def warmup_weight(epoch: int, target_weight: float, warmup_epochs: int = 5) -> float:
    """
    Linearly ramps a loss weight from 0 -> target_weight over warmup_epochs.
    Used to delay full SAM/NDVI weight until the network has learned basic
    reconstruction, preventing the large early-epoch loss spikes observed
    in the first synthetic-data run (epoch 1 total loss ~650 vs ~0.006
    for plain MSE -- a direct symptom of SAM's arccos gradient being huge
    when predictions are still close to random).
    """
    if epoch >= warmup_epochs:
        return target_weight
    return target_weight * (epoch / warmup_epochs)


if __name__ == "__main__":
    # Smoke test: loss should be ~0 for identical pred/target, >0 otherwise
    torch.manual_seed(0)
    pred = torch.rand(2, 4, 32, 32)
    target = pred.clone()

    loss_fn = GeoSRLoss(alpha=0.1, beta=0.1)
    total, components = loss_fn(pred, target)
    print("Identical pred/target ->", components)
    assert abs(components["mse"]) < 1e-6
    assert abs(components["ndvi"]) < 1e-6

    pred_noisy = pred + torch.rand_like(pred) * 0.5
    total2, components2 = loss_fn(pred_noisy, target)
    print("Noisy pred/target     ->", components2)
    assert components2["mse"] > components["mse"]

    print("LOSS SMOKE TEST PASSED")