import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from baselines import bicubic_upsample, lanczos_upsample
from model import ResidualSRNet
from data_engine import SentinelPatchDataset

ds = SentinelPatchDataset("/home/claude/geosrx/test_data/patches")
lr, hr = ds[0]
print(f"lr shape: {lr.shape}, hr shape: {hr.shape}")
scale_factor = hr.shape[1] // lr.shape[1]
print(f"inferred scale factor: {scale_factor}")

bicubic_out = bicubic_upsample(lr, scale_factor)
lanczos_out = lanczos_upsample(lr, scale_factor)
print(f"bicubic output shape: {bicubic_out.shape}")
print(f"lanczos output shape: {lanczos_out.shape}")
assert bicubic_out.shape == hr.shape
assert lanczos_out.shape == hr.shape

# RMSE vs real HR target (untrained CNN should be worse -- expected, it's untrained)
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

print(f"\nRMSE bicubic vs HR: {rmse(bicubic_out, hr):.4f}")
print(f"RMSE lanczos vs HR: {rmse(lanczos_out, hr):.4f}")

model = ResidualSRNet(in_channels=4, num_features=64, num_residual_blocks=3, scale_factor=scale_factor)
model.eval()
with torch.no_grad():
    lr_tensor = torch.from_numpy(lr).unsqueeze(0)  # add batch dim
    cnn_out = model(lr_tensor).squeeze(0).numpy()
print(f"CNN (untrained) output shape: {cnn_out.shape}")
print(f"RMSE CNN (untrained) vs HR: {rmse(cnn_out, hr):.4f}  <- expected to be poor, no training yet")

assert cnn_out.shape == hr.shape
print("\nALL BASELINE + ARCHITECTURE CHECKS PASSED")
