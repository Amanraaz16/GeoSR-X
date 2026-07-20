"""
Synthetic AOI generator with realistic spectral structure (not just smooth
noise). Vegetation regions get high-NIR/low-Red (high NDVI); urban/bare
regions get NIR ~ Red (low NDVI). This lets us actually test whether the
NDVI-consistency loss term does something, not just whether the code runs.

Still NOT real Sentinel-2 data. Useful for validating training mechanics
and the spectral loss behavior, not for producing a scientifically
meaningful trained model. Real Sentinel-2 data over the target AOI is
required before any reported result is meaningful.
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter


def generate_synthetic_aoi(size=512, seed=0):
    rng = np.random.default_rng(seed)

    # --- Land-cover mask: 1 = vegetation, 0 = urban/bare ---
    veg_mask = gaussian_filter(rng.normal(size=(size // 16, size // 16)), sigma=1.5)
    veg_mask = np.kron(veg_mask, np.ones((16, 16)))[:size, :size]
    veg_mask = (veg_mask > np.median(veg_mask)).astype(np.float32)

    # --- Add sharp rectangular "buildings" and linear "roads" (urban, low NDVI) ---
    urban_overlay = np.zeros((size, size), dtype=np.float32)
    for _ in range(25):
        x, y = rng.integers(0, size - 30, size=2)
        w, h = rng.integers(8, 25, size=2)
        urban_overlay[y:y + h, x:x + w] = 1.0
    for _ in range(4):
        if rng.random() > 0.5:
            row = rng.integers(0, size)
            urban_overlay[row:row + 3, :] = 1.0
        else:
            col = rng.integers(0, size)
            urban_overlay[:, col:col + 3] = 1.0

    is_urban = np.clip(urban_overlay + (1 - veg_mask) * 0.3, 0, 1)
    is_veg = np.clip(veg_mask * (1 - urban_overlay), 0, 1)

    # --- Assign reflectance per band based on land cover ---
    # Vegetation: low Red (~0.05), high NIR (~0.45) -> high NDVI
    # Urban/bare: Red ~ NIR (~0.20) -> low NDVI
    noise = lambda scale: rng.normal(0, scale, size=(size, size)).astype(np.float32)

    blue = 0.08 + is_urban * 0.10 + noise(0.01)
    green = 0.10 + is_veg * 0.08 + is_urban * 0.05 + noise(0.01)
    red = 0.05 + is_veg * 0.03 + is_urban * 0.15 + noise(0.01)
    nir = 0.10 + is_veg * 0.35 + is_urban * 0.10 + noise(0.015)

    bands = np.stack([blue, green, red, nir], axis=0)
    bands = np.clip(bands, 0.01, 0.6)

    # Convert reflectance back to DN scale (x10000) to match real-world ingestion path
    dn = (bands * 10000).astype(np.float32)
    return dn


def save_as_geotiff(dn_array, path):
    bands, h, w = dn_array.shape
    transform = from_origin(750000, 3050000, 10, 10)
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": bands,
        "dtype": "float32", "crs": "EPSG:32643", "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(dn_array)
        dst.descriptions = ("B2", "B3", "B4", "B8")
        dst.update_tags(SENSOR="S2A", LEVEL="MSIL2A")


if __name__ == "__main__":
    import os
    out_dir = "/home/claude/geosrx/test_data"
    os.makedirs(out_dir, exist_ok=True)
    dn = generate_synthetic_aoi(size=512, seed=1)
    path = os.path.join(out_dir, "synthetic_aoi_v2.tif")
    save_as_geotiff(dn, path)
    print(f"Saved synthetic AOI to {path}, shape {dn.shape}")

    # Quick sanity check on NDVI separation
    red, nir = dn[2] / 10000.0, dn[3] / 10000.0
    ndvi = (nir - red) / (nir + red + 1e-6)
    print(f"NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}], mean: {ndvi.mean():.3f}")
