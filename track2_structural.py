"""
GeoSR-X Evaluation — Track 2: Structural Comparison vs Historical Reference
=============================================================================
Compares spatial sharpness and edge coherence of SR outputs against the
GEHistoricalImagery structural reference (Google Earth ~0.57m RGB).

Methodology
-----------
- Historical Google Earth imagery is used ONLY as a structural reference.
- No spectral metrics (SAM, ERGAS, NDVI, etc.) are computed because the
  reference is RGB while Sentinel-2 is multispectral.
- All structural metrics are computed on the SAME 1024×1024 centre crop for
  every method to ensure a fair comparison.
"""

import os
import argparse
import json
import numpy as np
import torch
from PIL import Image
import rasterio
from rasterio.windows import from_bounds

from model import ResidualSRNet
from baselines import bicubic_upsample, lanczos_upsample
from data_engine import read_geotiff, validate_bands, normalize_reflectance


def sobel_edge_density(arr: np.ndarray) -> float:
    """Mean Sobel gradient magnitude (higher = sharper)."""
    from scipy.ndimage import sobel

    scores = []
    for b in range(arr.shape[0]):
        band = arr[b].astype(np.float32)
        sx = sobel(band, axis=0)
        sy = sobel(band, axis=1)
        mag = np.sqrt(sx * sx + sy * sy)
        scores.append(float(mag.mean()))
    return float(np.mean(scores))


def laplacian_variance(arr: np.ndarray) -> float:
    """Variance of Laplacian (higher = sharper)."""
    from scipy.ndimage import laplace

    scores = []
    for b in range(arr.shape[0]):
        lap = laplace(arr[b].astype(np.float32))
        scores.append(float(lap.var()))
    return float(np.mean(scores))


def center_crop(arr: np.ndarray, crop_size: int = 1024) -> np.ndarray:
    """Return a centered crop, safely clipped to image size."""
    _, h, w = arr.shape
    crop_size = min(crop_size, h, w)

    cy = h // 2
    cx = w // 2
    half = crop_size // 2

    y0 = max(0, cy - half)
    x0 = max(0, cx - half)

    return arr[:, y0:y0 + crop_size, x0:x0 + crop_size]


def load_reference_crop(ref_path: str, sentinel_bounds) -> np.ndarray:
    """Crop historical reference to Sentinel bounds."""
    with rasterio.open(ref_path) as src:
        window = from_bounds(
            sentinel_bounds.left,
            sentinel_bounds.bottom,
            sentinel_bounds.right,
            sentinel_bounds.top,
            src.transform,
        )
        arr = src.read(window=window)

    return arr.astype(np.float32) / 255.0


def make_cnn_fn(checkpoint_path: str, scale_factor: int):
    model = ResidualSRNet(
        in_channels=4,
        num_features=64,
        num_residual_blocks=3,
        scale_factor=scale_factor,
    )

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    def fn(arr: np.ndarray):
        with torch.no_grad():
            tensor = torch.from_numpy(arr).unsqueeze(0)
            return model(tensor).squeeze(0).numpy()

    return fn


def save_rgb_crop(arr: np.ndarray, path: str):
    """Save RGB crop after percentile stretch."""
    p2, p98 = np.percentile(arr, (2, 98))
    if p98 > p2:
        arr = np.clip((arr - p2) / (p98 - p2), 0, 1)

    img = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentinel", default="test_data/nahargarh_real.tif")
    parser.add_argument("--reference", default="test_data/nahargarh_historical.tif")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--scale-factor", type=int, default=2)
    parser.add_argument("--out-dir", default="checkpoints/track2_visuals")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scale = args.scale_factor

    # --- Find overlap bounds ---
    with rasterio.open(args.sentinel) as s:
        sb = s.bounds
        sentinel_crs = s.crs
        sentinel_res = s.res[0]          # ~10m
        sentinel_transform = s.transform

    with rasterio.open(args.reference) as r:
        ref_res = r.res[0]               # ~0.57m

    overlap = rasterio.coords.BoundingBox(
        left=max(sb.left, sb.left),
        bottom=max(sb.bottom, sb.bottom),
        right=min(sb.right, sb.right),
        top=min(sb.top, sb.top),
    )

    # --- Pick a 500m x 500m patch well inside the overlap ---
    # Offset from bottom-left to avoid nodata polygon edges
    cx = overlap.left + (overlap.right  - overlap.left) * 0.45
    cy = overlap.bottom + (overlap.top - overlap.bottom) * 0.55
    half_m = 250   # 250m each side = 500m patch

    patch_bounds = rasterio.coords.BoundingBox(
        left=cx - half_m, bottom=cy - half_m,
        right=cx + half_m, top=cy + half_m
    )
    print(f"Patch centre: ({cx:.1f}, {cy:.1f}) UTM43N")
    print(f"Patch bounds: {patch_bounds}")

    # --- Load Sentinel patch (native 10m) ---
    def load_sentinel_patch(path, bounds):
        with rasterio.open(path) as src:
            window = rasterio.windows.from_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top,
                src.transform
            )
            arr = src.read(window=window).astype(np.float32)
        return arr

    sentinel_patch = load_sentinel_patch(args.sentinel, patch_bounds)
    selected, band_info = validate_bands(sentinel_patch, None)  # positional
    # Override: GEE preserved band descriptions so just use directly
    # band order: B2,B3,B4,B8
    reflectance = np.clip(sentinel_patch / 10000.0, 0, 1).astype(np.float32)
    print(f"Sentinel patch shape: {reflectance.shape}  "
          f"(expect ~50x50 at 10m for a 500m patch)")

    # Check we got real data not all nodata
    valid_frac = np.mean(reflectance > 0)
    print(f"Valid pixel fraction: {valid_frac:.2f}  (want > 0.7)")
    if valid_frac < 0.5:
        print("WARNING: too many nodata pixels in this patch — "
              "try adjusting cx/cy offsets above (0.45/0.55) "
              "to find a denser data region.")

    # --- Load historical reference patch at native ~0.57m ---
    def load_ref_patch(path, bounds):
        with rasterio.open(path) as src:
            window = rasterio.windows.from_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top,
                src.transform
            )
            arr = src.read(window=window).astype(np.float32) / 255.0
        return arr

    ref_patch = load_ref_patch(args.reference, patch_bounds)
    print(f"Reference patch shape: {ref_patch.shape}  "
          f"(expect ~875x875 at 0.57m for a 500m patch)")

    # --- Run SR models on the Sentinel patch ---
    models = {
        "Original Sentinel": reflectance[:3],
        "Bicubic":           bicubic_upsample(reflectance, scale)[:3],
        "Lanczos":           lanczos_upsample(reflectance, scale)[:3],
        "CNN (MSE)":         make_cnn_fn(
                                 os.path.join(args.checkpoint_dir, "baseline_mse.pt"),
                                 scale)(reflectance)[:3],
        "GeoSR-X":           make_cnn_fn(
                                 os.path.join(args.checkpoint_dir, "geosrx.pt"),
                                 scale)(reflectance)[:3],
    }

    # --- Metrics and visuals ---
    print("\n=== Track 2: Structural Sharpness Metrics ===")
    print("(Same 500m x 500m geographic patch, all methods)")
    print("(Reference is structural benchmark only — no spectral comparison)\n")

    ref_edge = sobel_edge_density(ref_patch)
    ref_lap  = laplacian_variance(ref_patch)
    print(f"{'Model':<22} {'Edge Density':>14} {'Laplacian Var':>15}")
    print("-" * 55)
    print(f"{'Historical Ref (RGB)':22} {ref_edge:>14.4f} "
          f"{ref_lap:>15.4f}  <- structural target")

    results = {}
    for name, arr in models.items():
        arr = np.clip(arr, 0.0, 1.0)
        edge = sobel_edge_density(arr)
        lap  = laplacian_variance(arr)
        results[name] = {"edge_density": edge, "laplacian_var": lap}
        print(f"{name:<22} {edge:>14.4f} {lap:>15.4f}")

        fname = (name.lower()
                 .replace(" ", "_")
                 .replace("(", "").replace(")", "") + ".png")
        save_rgb_crop(arr, os.path.join(args.out_dir, fname))

    # Save reference visual at same crop size
    save_rgb_crop(ref_patch, os.path.join(args.out_dir, "reference_historical.png"))

    print(f"\nVisual crops saved to: {args.out_dir}/")

    out = {
        "patch_bounds": list(patch_bounds),
        "patch_size_m": 500,
        "reference": {"edge_density": ref_edge, "laplacian_var": ref_lap},
        "models": results,
        "methodology_note": (
            "All metrics computed on the same 500m x 500m geographic patch. "
            "Historical RGB reference used as structural benchmark only. "
            "No spectral metrics computed due to sensor mismatch."
        )
    }
    with open(os.path.join(args.checkpoint_dir, "track2_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to checkpoints/track2_results.json")
