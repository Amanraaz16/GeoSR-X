"""
GeoSR-X Data Engine
====================
Pipeline A (Training) only. This module NEVER touches GEHistoricalImagery
or any external structural-reference data — per the frozen spec, training
supervision comes exclusively from real Sentinel-2 reflectance via the
Wald protocol (degrade -> recover original). Structural-reference fetching
for visualization/comparison lives in a separate module (platform-side,
Pipeline B) and must not import from here.

Required bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR).
B8 is included specifically so NDVI/NDWI self-consistency can be computed
later in the loss function and evaluation — not for any resolution claim.
"""

from __future__ import annotations

import json
import os
import re
import dataclasses
from typing import Optional

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_BANDS = ["B2", "B3", "B4", "B8"]
SENTINEL2_REFLECTANCE_SCALE = 10000.0  # L2A DN -> reflectance divisor
DEFAULT_PATCH_SIZE_HR = 64
DEFAULT_SCALE_FACTOR = 2
DEFAULT_STRIDE = None  # defaults to non-overlapping (= patch size)


@dataclasses.dataclass
class DatasetConfig:
    patch_size_hr: int = DEFAULT_PATCH_SIZE_HR
    scale_factor: int = DEFAULT_SCALE_FACTOR
    stride_hr: Optional[int] = None
    gaussian_sigma: Optional[float] = None  # auto if None
    reflectance_scale: float = SENTINEL2_REFLECTANCE_SCALE
    nodata_fraction_threshold: float = 0.1  # skip patch if >10% invalid px


# ---------------------------------------------------------------------------
# 1. Read
# ---------------------------------------------------------------------------

def read_geotiff(path: str):
    """Read a Sentinel-2 GeoTIFF. Returns (array[bands,H,W] float32, profile, descriptions)."""
    with rasterio.open(path) as src:
        array = src.read().astype(np.float32)
        profile = src.profile
        descriptions = src.descriptions  # tuple, may contain None entries
        tags = src.tags()
    return array, profile, descriptions, tags


# ---------------------------------------------------------------------------
# 2. Validate & select bands
# ---------------------------------------------------------------------------

# All known Sentinel-2 band names for detection
ALL_S2_BANDS = [
    "B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"
]

# Alternative name mappings (GEE sometimes uses these)
BAND_ALIASES = {
    "B08": "B8", "B8A": "B8A", "B02": "B2", "B03": "B3",
    "B04": "B4", "B05": "B5", "B06": "B6", "B07": "B7",
    "B11": "B11", "B12": "B12", "B01": "B1", "B09": "B9",
}


def validate_bands(array: np.ndarray, descriptions: tuple,
                   required=REQUIRED_BANDS):
    """
    Flexible band validation:
    - Accepts any Sentinel-2 GeoTIFF regardless of band count
    - Detects all present bands from descriptions
    - Extracts required bands (B2,B3,B4,B8) if present
    - Falls back to positional assumption if descriptions are missing
    - Returns detailed band inventory for user-facing reporting
    - Never crashes on extra bands (12-band SNAP exports, etc.)
    """
    n_bands = array.shape[0]
    desc_raw = list(descriptions or [])

    # Normalise descriptions: strip whitespace, uppercase, resolve aliases
    def normalise(d):
        if not d:
            return None
        d = d.strip().upper()
        return BAND_ALIASES.get(d, d)

    desc_clean = [normalise(d) for d in desc_raw]

    # Build full inventory of detected bands
    detected = {}
    for i, name in enumerate(desc_clean):
        if name:
            detected[name] = i

    # Check which required bands are present
    found_required = {b: detected[b] for b in required if b in detected}
    missing = [b for b in required if b not in detected]
    extra   = [b for b in detected if b not in required]

    band_info = {
        "total_bands_in_file": n_bands,
        "bands_detected":      list(detected.keys()),
        "required_bands":      required,
        "found_required":      list(found_required.keys()),
        "missing_required":    missing,
        "extra_bands":         extra,
    }

    # Case 1: all required bands found by description
    if len(found_required) == len(required):
        order    = [found_required[b] for b in required]
        selected = array[order, :, :]
        band_info["mapping_source"] = "band_descriptions"
        band_info["mapping"]        = found_required
        return selected, band_info

    # Case 2: some required bands missing but file has enough bands
    # for positional fallback
    if missing and n_bands >= len(required):
        # Only use positional fallback if NO descriptions were found at all
        # (i.e. GEE export without band names)
        if not detected:
            selected = array[:len(required), :, :]
            band_info["mapping_source"] = "positional_fallback"
            band_info["mapping"]        = {b: i for i, b in enumerate(required)}
            band_info["warning"] = (
                f"No band descriptions found in GeoTIFF. Assumed first "
                f"{len(required)} bands are {required} in that order. "
                f"Verify band order if results look wrong."
            )
            return selected, band_info

        # Descriptions exist but required bands not all found
        raise ValueError(
            f"GeoTIFF has band descriptions but is missing required bands: "
            f"{missing}. Found bands: {list(detected.keys())}. "
            f"Required: {required} (B2=Blue, B3=Green, B4=Red, B8=NIR). "
            f"Please export these 4 bands from GEE or SNAP and re-upload."
        )

    # Case 3: not enough bands at all
    if n_bands < len(required):
        raise ValueError(
            f"GeoTIFF has only {n_bands} band(s). "
            f"GeoSR-X requires at minimum: {required} "
            f"(B2=Blue, B3=Green, B4=Red, B8=NIR at 10m resolution). "
            f"If you have a 3-band RGB-only GeoTIFF, NDVI computation "
            f"is not possible as NIR (B8) is missing."
        )

    selected = array[:len(required), :, :]
    band_info["mapping_source"] = "positional_fallback"
    band_info["mapping"]        = {b: i for i, b in enumerate(required)}
    return selected, band_info


# ---------------------------------------------------------------------------
# 3. Normalize
# ---------------------------------------------------------------------------

def normalize_reflectance(array: np.ndarray, scale: float = SENTINEL2_REFLECTANCE_SCALE) -> np.ndarray:
    """Convert Sentinel-2 L2A digital numbers to reflectance, clipped to [0, 1]."""
    arr = array.astype(np.float32) / scale
    return np.clip(arr, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 4. Wald-protocol degradation
# ---------------------------------------------------------------------------

def wald_degrade(hr_array: np.ndarray, scale_factor: int = DEFAULT_SCALE_FACTOR,
                  gaussian_sigma: Optional[float] = None) -> np.ndarray:
    """
    Simulate a lower-resolution Sentinel-2 acquisition from a real HR reflectance
    array. This is the ONLY source of "ground truth" in this pipeline: hr_array
    is real Sentinel-2 data, and lr_array is a synthetic degradation of it. The
    model is trained to recover hr_array from lr_array — i.e. "recover native
    Sentinel", not "verified 5m/1m". Do not relabel this output as a higher
    resolution than the sensor's native grid.

    Steps: per-band Gaussian blur (approximates sensor MTF) -> block-mean
    downsampling by scale_factor (more realistic than naive decimation).
    """
    if gaussian_sigma is None:
        gaussian_sigma = scale_factor / 2.0

    bands, h, w = hr_array.shape
    h_crop = (h // scale_factor) * scale_factor
    w_crop = (w // scale_factor) * scale_factor
    hr_cropped = hr_array[:, :h_crop, :w_crop]

    blurred = np.stack(
        [gaussian_filter(hr_cropped[b], sigma=gaussian_sigma) for b in range(bands)],
        axis=0,
    )

    lr_h, lr_w = h_crop // scale_factor, w_crop // scale_factor
    lr_array = blurred.reshape(bands, lr_h, scale_factor, lr_w, scale_factor).mean(axis=(2, 4))

    return lr_array.astype(np.float32), hr_cropped.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Patch extraction
# ---------------------------------------------------------------------------

def extract_patches(hr_array: np.ndarray, lr_array: np.ndarray, config: DatasetConfig):
    """
    Extract spatially aligned LR/HR patch pairs.

    hr_array: (bands, H, W) — already cropped to be divisible by scale_factor
    lr_array: (bands, H/scale, W/scale)

    Returns list of (lr_patch, hr_patch) tuples.
    """
    scale = config.scale_factor
    hr_patch = config.patch_size_hr
    lr_patch = hr_patch // scale
    stride_hr = config.stride_hr or hr_patch
    stride_lr = stride_hr // scale

    bands, lr_h, lr_w = lr_array.shape
    hr_bands, hr_h, hr_w = hr_array.shape

    patches = []
    for ly in range(0, lr_h - lr_patch + 1, stride_lr):
        for lx in range(0, lr_w - lr_patch + 1, stride_lr):
            hy, hx = ly * scale, lx * scale

            lr_p = lr_array[:, ly:ly + lr_patch, lx:lx + lr_patch]
            hr_p = hr_array[:, hy:hy + hr_patch, hx:hx + hr_patch]

            if lr_p.shape[1:] != (lr_patch, lr_patch) or hr_p.shape[1:] != (hr_patch, hr_patch):
                continue  # edge patch, incomplete

            invalid_frac = np.mean((hr_p <= 0) | np.isnan(hr_p))
            if invalid_frac > config.nodata_fraction_threshold:
                continue  # skip mostly-nodata patches

            patches.append((lr_p, hr_p))

    return patches


# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------

def save_patch_pairs(patches, out_dir: str, prefix: str = "patch"):
    os.makedirs(out_dir, exist_ok=True)
    for i, (lr, hr) in enumerate(patches):
        np.savez(os.path.join(out_dir, f"{prefix}_{i:05d}.npz"), lr=lr, hr=hr)
    return len(patches)


# ---------------------------------------------------------------------------
# 7. Metadata report (platform-facing, cheap, useful day-1 feature)
# ---------------------------------------------------------------------------

_DATE_PATTERN = re.compile(r"(20\d{2})(\d{2})(\d{2})")
_SENSOR_PATTERN = re.compile(r"(S2[AB])", re.IGNORECASE)
_LEVEL_PATTERN = re.compile(r"(MSIL2A|MSIL1C)", re.IGNORECASE)


def generate_metadata_report(path: str) -> dict:
    """Build a human-readable summary of an input Sentinel-2 GeoTIFF."""
    array, profile, descriptions, tags = read_geotiff(path)
    filename = os.path.basename(path)

    date_match = _DATE_PATTERN.search(filename)
    sensor_match = _SENSOR_PATTERN.search(filename)
    level_match = _LEVEL_PATTERN.search(filename)

    crs = profile.get("crs")
    transform = profile.get("transform")
    pixel_size = (abs(transform.a), abs(transform.e)) if transform else (None, None)

    band_names_detected = [d for d in (descriptions or []) if d]

    report = {
        "filename": filename,
        "sensor": sensor_match.group(1).upper() if sensor_match else "unknown",
        "level": level_match.group(1).upper() if level_match else "unknown",
        "date": (
            f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            if date_match else "unknown"
        ),
        "crs": str(crs) if crs else "unknown",
        "resolution_m": pixel_size,
        "band_count": array.shape[0],
        "band_descriptions_detected": band_names_detected or "none (positional assumption required)",
        "width": array.shape[2],
        "height": array.shape[1],
        "cloud_mask_available": any("SCL" in (d or "").upper() for d in (descriptions or [])),
        "raw_tags": tags,
    }
    return report


# ---------------------------------------------------------------------------
# 8. Orchestrator
# ---------------------------------------------------------------------------

def build_training_dataset(input_tif_path: str, out_dir: str, config: Optional[DatasetConfig] = None):
    """
    End-to-end: read -> validate bands -> normalize -> Wald-degrade ->
    extract patches -> save. Writes a manifest.json for reproducibility.

    This function touches ONLY the input Sentinel-2 GeoTIFF. No external
    structural-reference data is used here (see module docstring).
    """
    config = config or DatasetConfig()

    array, profile, descriptions, tags = read_geotiff(input_tif_path)
    selected, band_info = validate_bands(array, descriptions)
    reflectance = normalize_reflectance(selected, config.reflectance_scale)
    lr_array, hr_array = wald_degrade(reflectance, config.scale_factor, config.gaussian_sigma)
    patches = extract_patches(hr_array, lr_array, config)
    n_saved = save_patch_pairs(patches, out_dir)

    manifest = {
        "input_file": input_tif_path,
        "band_info": band_info,
        "config": dataclasses.asdict(config),
        "n_patches_saved": n_saved,
        "hr_shape_after_crop": list(hr_array.shape),
        "lr_shape": list(lr_array.shape),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest


# ---------------------------------------------------------------------------
# 9. PyTorch-facing Dataset (thin wrapper, used by training step next)
# ---------------------------------------------------------------------------

class SentinelPatchDataset:
    """
    Lazily loads saved LR/HR .npz patch pairs. Kept dependency-free of
    torch here so this module stays importable without torch installed;
    the training script will subclass/wrap this with torch.utils.data.Dataset.
    """

    def __init__(self, patch_dir: str):
        self.patch_dir = patch_dir
        self.files = sorted(
            f for f in os.listdir(patch_dir) if f.endswith(".npz")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.patch_dir, self.files[idx]))
        return data["lr"], data["hr"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GeoSR-X Data Engine (Pipeline A: training data only)")
    parser.add_argument("input_tif", help="Path to input Sentinel-2 GeoTIFF")
    parser.add_argument("out_dir", help="Output directory for patch pairs + manifest")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE_HR)
    parser.add_argument("--scale-factor", type=int, default=DEFAULT_SCALE_FACTOR)
    parser.add_argument("--stride", type=int, default=None)
    args = parser.parse_args()

    cfg = DatasetConfig(
        patch_size_hr=args.patch_size,
        scale_factor=args.scale_factor,
        stride_hr=args.stride,
    )
    result = build_training_dataset(args.input_tif, args.out_dir, cfg)
    print(json.dumps(result, indent=2, default=str))
