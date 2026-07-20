"""
GeoSR-X Model Architecture
============================
Single architecture, used for two training runs:
  - "CNN (paper baseline)": trained with plain MSE, mirrors Massarelli et al.
  - "GeoSR-X (ours)": same architecture, trained with MSE + SAM + NDVI loss

The architecture itself is NOT the contribution (per the frozen spec) —
it deliberately mirrors the published residual design (3x3 feature
extraction -> residual blocks -> reconstruction) rather than inventing
something new. Input/output channels are 4 (B2,B3,B4,B8) instead of the
paper's 3 (B2,B3,B4) so NDVI/NDWI can be computed downstream.

Upsampling uses sub-pixel convolution (PixelShuffle), a standard, stable
choice for this scale factor range -- not a novelty claim either.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, num_features: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class ResidualSRNet(nn.Module):
    """
    Layer count (matching the paper's "9-layer" framing):
      1 feature-extraction conv
      + 3 residual blocks x 2 convs = 6
      + 1 reconstruction conv
      = 8 conv layers, plus a PixelShuffle upsampling stage.
    """

    def __init__(self, in_channels: int = 4, num_features: int = 64,
                 num_residual_blocks: int = 6, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_residual_blocks)]
        )

        # Sub-pixel upsampling: conv to (scale^2 * features), then PixelShuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.ReLU(inplace=True),
        )

        self.reconstruction = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extraction(x)
        feat = self.residual_blocks(feat)
        feat = self.upsample(feat)
        out = self.reconstruction(feat)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test: verify shapes match what data_engine produces
    # (lr patch: 4x32x32 -> hr patch: 4x64x64, per the earlier pipeline test)
    model = ResidualSRNet(in_channels=4, num_features=64, num_residual_blocks=3, scale_factor=2)
    dummy_lr = torch.randn(2, 4, 32, 32)  # batch of 2
    out = model(dummy_lr)
    print(f"Input shape:  {dummy_lr.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected:     torch.Size([2, 4, 64, 64])")
    print(f"Parameter count: {model.count_parameters():,}")
    assert out.shape == (2, 4, 64, 64), "Shape mismatch!"
    print("SHAPE CHECK PASSED")
