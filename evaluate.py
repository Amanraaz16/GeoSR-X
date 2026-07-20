"""
GeoSR-X Evaluation — Track 1: Wald Protocol Fidelity Metrics
==============================================================
Computes RMSE, PSNR, SSIM, SAM, ERGAS for:
  - Bicubic baseline
  - Lanczos baseline
  - CNN (plain MSE, paper-style)
  - GeoSR-X (spectral-preserving loss)

All evaluated on the same held-out Wald-protocol validation patches,
against the real Sentinel-2 HR target — the only honest ground truth
available without external HR reference data.

Run:
    python evaluate.py --patch-dir test_data/patches_real --scale-factor 2
"""

import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from model import ResidualSRNet
from baselines import bicubic_upsample, lanczos_upsample
from losses import compute_ndvi
from train import TorchPatchDataset


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse_val = np.mean((pred - target) ** 2)
    if mse_val == 0:
        return float("inf")
    return float(10 * np.log10((data_range ** 2) / mse_val))


def ssim_patch(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Per-band SSIM averaged across bands. Operates on (C, H, W) arrays."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    scores = []
    for b in range(pred.shape[0]):
        p, t = pred[b].astype(np.float64), target[b].astype(np.float64)
        mu_p, mu_t = p.mean(), t.mean()
        sig_p = p.std()
        sig_t = t.std()
        sig_pt = np.mean((p - mu_p) * (t - mu_t))
        num = (2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)
        den = (mu_p**2 + mu_t**2 + C1) * (sig_p**2 + sig_t**2 + C2)
        scores.append(num / den)
    return float(np.mean(scores))


def sam_metric(pred: np.ndarray, target: np.ndarray) -> float:
    """Spectral Angle Mapper in degrees, averaged over pixels."""
    pred_f = pred.reshape(pred.shape[0], -1).T      # (N, C)
    target_f = target.reshape(target.shape[0], -1).T
    dot = np.sum(pred_f * target_f, axis=1)
    norm_p = np.linalg.norm(pred_f, axis=1) + 1e-8
    norm_t = np.linalg.norm(target_f, axis=1) + 1e-8
    cos_a = np.clip(dot / (norm_p * norm_t), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)).mean())


def ergas(pred: np.ndarray, target: np.ndarray, scale_factor: int) -> float:
    """
    ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse).
    Lower is better. scale_factor = HR_resolution / LR_resolution.
    pred, target: (C, H, W)
    """
    n_bands = pred.shape[0]
    band_terms = []
    for b in range(n_bands):
        rmse_b = np.sqrt(np.mean((pred[b] - target[b]) ** 2))
        mean_b = np.mean(target[b])
        if mean_b > 1e-8:
            band_terms.append((rmse_b / mean_b) ** 2)
    return float(100 / scale_factor * np.sqrt(np.mean(band_terms)))


def ndvi_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Track 3: NDVI self-consistency.
    Mean absolute error between NDVI(pred) and NDVI(target).
    Uses the same validity mask as the training loss: skip pixels where
    nir+red < 0.01 to avoid numerical instability at boundary/nodata pixels.
    Band order: [B2, B3, B4(red), B8(nir)] -> red=idx2, nir=idx3
    """
    red_p, nir_p = pred[2], pred[3]
    red_t, nir_t = target[2], target[3]

    valid = (nir_p + red_p > 0.01) & (nir_t + red_t > 0.01)
    if not valid.any():
        return float("nan")

    ndvi_p = (nir_p - red_p) / (nir_p + red_p + 1e-6)
    ndvi_t = (nir_t - red_t) / (nir_t + red_t + 1e-6)
    return float(np.mean(np.abs(ndvi_p[valid] - ndvi_t[valid])))


def compute_all_metrics(pred: np.ndarray, target: np.ndarray, scale_factor: int) -> dict:
    return {
        "RMSE":     rmse(pred, target),
        "PSNR_dB":  psnr(pred, target),
        "SSIM":     ssim_patch(pred, target),
        "SAM_deg":  sam_metric(pred, target),
        "ERGAS":    ergas(pred, target, scale_factor),
        "NDVI_MAE": ndvi_mae(pred, target),
    }


# ---------------------------------------------------------------------------
# Evaluate a callable model over the validation loader
# ---------------------------------------------------------------------------

def evaluate_model(model_fn, val_patches, scale_factor: int) -> dict:
    """
    model_fn: callable (lr_np: ndarray (C,H,W)) -> pred_np: ndarray (C,H,W)
    val_patches: list of (lr, hr) numpy arrays
    """
    all_metrics = {k: [] for k in ["RMSE", "PSNR_dB", "SSIM", "SAM_deg", "ERGAS", "NDVI_MAE"]}

    for lr, hr in val_patches:
        pred = model_fn(lr)
        pred = np.clip(pred, 0.0, 1.0)
        m = compute_all_metrics(pred, hr, scale_factor)
        for k, v in m.items():
            if not np.isnan(v):
                all_metrics[k].append(v)

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


# ---------------------------------------------------------------------------
# Model wrappers (unified interface: np (C,H,W) -> np (C,H,W))
# ---------------------------------------------------------------------------

def make_cnn_fn(checkpoint_path: str, scale_factor: int,
                num_residual_blocks: int = 3):
    model = ResidualSRNet(in_channels=4, num_features=64,
                          num_residual_blocks=num_residual_blocks,
                          scale_factor=scale_factor)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True,
                                      map_location=torch.device('cpu')))
    model.eval()

    def fn(lr: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(lr).unsqueeze(0)
            out = model(t).squeeze(0).numpy()
        return out
    return fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", default="test_data/patches_real")
    parser.add_argument("--scale-factor", type=int, default=2)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    # --- Load patches, use same 80/20 split as training ---
    ds = TorchPatchDataset(args.patch_dir)
    n_val = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_val
    _, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(0)
    )
    val_patches = [(ds[i][0].numpy(), ds[i][1].numpy()) for i in val_ds.indices]
    print(f"Evaluating on {len(val_patches)} validation patches\n")

    # --- Define models ---
    models = {
        "Bicubic":    lambda lr: bicubic_upsample(lr, args.scale_factor),
        "Lanczos":    lambda lr: lanczos_upsample(lr, args.scale_factor),
        "CNN (MSE)":  make_cnn_fn(
                          os.path.join(args.checkpoint_dir, "baseline_mse.pt"),
                          args.scale_factor),
        "GeoSR-X":    make_cnn_fn(
                          os.path.join(args.checkpoint_dir, "geosrx.pt"),
                          args.scale_factor),
    }

    # --- Run evaluation ---
    results = {}
    for name, fn in models.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_model(fn, val_patches, args.scale_factor)

    # --- Print table (mirrors Massarelli Table 1 layout) ---
    metrics = ["RMSE", "PSNR_dB", "SSIM", "SAM_deg", "ERGAS", "NDVI_MAE"]
    header = f"{'Metric':<12}" + "".join(f"{n:<16}" for n in results)
    print("\n" + "=" * (12 + 16 * len(results)))
    print(header)
    print("=" * (12 + 16 * len(results)))
    for m in metrics:
        row = f"{m:<12}"
        for name in results:
            row += f"{results[name][m]:<16.5f}"
        print(row)
    print("=" * (12 + 16 * len(results)))

    # --- Save ---
    out_path = os.path.join(args.checkpoint_dir, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")