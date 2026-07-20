"""
GeoSR-X Baselines: Bicubic and Lanczos
========================================
Classical interpolation baselines. No training, no learned parameters.
These exist purely as comparison points in the metrics table, matching
the published methodology (Massarelli et al. used bicubic as the
deterministic baseline; we add Lanczos per the day-1 refinement since
it costs almost nothing and reviewers expect a stronger classical
comparison than bicubic alone).

Operates per-band on a (bands, H, W) float32 array in [0, 1] reflectance.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _resize_band(band: np.ndarray, target_shape: tuple, method: int) -> np.ndarray:
    """Resize a single 2D band using PIL (operates on float32 directly)."""
    img = Image.fromarray(band, mode="F")  # 32-bit float mode
    resized = img.resize((target_shape[1], target_shape[0]), resample=method)
    return np.array(resized, dtype=np.float32)


def bicubic_upsample(lr_array: np.ndarray, scale_factor: int) -> np.ndarray:
    """Upsample (bands, h, w) -> (bands, h*scale, w*scale) via bicubic interpolation."""
    bands, h, w = lr_array.shape
    target_shape = (h * scale_factor, w * scale_factor)
    out = np.stack(
        [_resize_band(lr_array[b], target_shape, Image.BICUBIC) for b in range(bands)],
        axis=0,
    )
    return out


def lanczos_upsample(lr_array: np.ndarray, scale_factor: int) -> np.ndarray:
    """Upsample (bands, h, w) -> (bands, h*scale, w*scale) via Lanczos interpolation."""
    bands, h, w = lr_array.shape
    target_shape = (h * scale_factor, w * scale_factor)
    out = np.stack(
        [_resize_band(lr_array[b], target_shape, Image.LANCZOS) for b in range(bands)],
        axis=0,
    )
    return out
