"""Smoke test: synthetic Sentinel-2-like GeoTIFF -> full data engine pipeline."""
import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_engine import (
    DatasetConfig, build_training_dataset, generate_metadata_report,
    SentinelPatchDataset,
)

TEST_DIR = "/home/claude/geosrx/test_data"
os.makedirs(TEST_DIR, exist_ok=True)
synthetic_path = os.path.join(TEST_DIR, "S2A_MSIL2A_20250618_synthetic.tif")
out_dir = os.path.join(TEST_DIR, "patches")

# --- Build a synthetic 4-band (B2,B3,B4,B8) Sentinel-2-like raster ---
H, W = 256, 256
rng = np.random.default_rng(42)

# Simulate plausible reflectance-like DN values with smooth spatial structure
def smooth_field(h, w, scale=20):
    base = rng.normal(0, 1, size=(h // scale + 2, w // scale + 2))
    from scipy.ndimage import zoom
    return zoom(base, scale, order=3)[:h, :w]

bands = []
for base_val in (1200, 1400, 1100, 2800):  # rough DN baselines for B2,B3,B4,B8
    field = base_val + smooth_field(H, W) * 300
    field = np.clip(field, 0, 10000).astype(np.float32)
    bands.append(field)
array = np.stack(bands, axis=0)

transform = from_origin(750000, 3050000, 10, 10)  # arbitrary UTM-like origin, 10m pixels
profile = {
    "driver": "GTiff",
    "height": H,
    "width": W,
    "count": 4,
    "dtype": "float32",
    "crs": "EPSG:32643",
    "transform": transform,
}

with rasterio.open(synthetic_path, "w", **profile) as dst:
    dst.write(array)
    dst.descriptions = ("B2", "B3", "B4", "B8")
    dst.update_tags(SENSOR="S2A", LEVEL="MSIL2A")

print("=== Metadata report ===")
report = generate_metadata_report(synthetic_path)
for k, v in report.items():
    if k != "raw_tags":
        print(f"  {k}: {v}")

print("\n=== Running build_training_dataset ===")
cfg = DatasetConfig(patch_size_hr=64, scale_factor=2, stride_hr=32)
manifest = build_training_dataset(synthetic_path, out_dir, cfg)
print("manifest:")
for k, v in manifest.items():
    print(f"  {k}: {v}")

print("\n=== Loading via SentinelPatchDataset ===")
ds = SentinelPatchDataset(out_dir)
print(f"  num patches: {len(ds)}")
lr0, hr0 = ds[0]
print(f"  patch 0 shapes -> lr: {lr0.shape}, hr: {hr0.shape}")
print(f"  lr dtype: {lr0.dtype}, hr dtype: {hr0.dtype}")
print(f"  hr value range: [{hr0.min():.4f}, {hr0.max():.4f}] (expect within [0,1])")

# Sanity: HR patch should be exactly scale_factor x larger than LR patch
assert hr0.shape[1] == lr0.shape[1] * cfg.scale_factor
assert hr0.shape[2] == lr0.shape[2] * cfg.scale_factor
assert hr0.shape[0] == 4 and lr0.shape[0] == 4  # 4 bands preserved

print("\nALL CHECKS PASSED")
