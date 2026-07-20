"""
GeoSR-X Training Loop
========================
Trains the SAME architecture (ResidualSRNet) under two loss configurations:
  - baseline:  alpha=0, beta=0   (plain MSE, mirrors the paper)
  - geosr_x:   alpha>0, beta>0   (MSE + SAM + NDVI consistency)

This script trains both, logs metrics per epoch, and reports whether the
spectral loss term actually reduces NDVI inconsistency on held-out
validation patches relative to the plain-MSE baseline -- the core
empirical question behind the project's one stated contribution.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from model import ResidualSRNet
from losses import GeoSRLoss, compute_ndvi, compute_ndwi, warmup_weight


class TorchPatchDataset(Dataset):
    def __init__(self, patch_dir: str):
        self.files = sorted(
            os.path.join(patch_dir, f) for f in os.listdir(patch_dir) if f.endswith(".npz")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        lr = torch.from_numpy(data["lr"]).float()
        hr = torch.from_numpy(data["hr"]).float()
        return lr, hr


def evaluate(model, loader, loss_fn, device):
    model.eval()
    totals = {"mse": 0.0, "sam": 0.0, "ndvi": 0.0, "ndwi": 0.0, "total": 0.0}
    n = 0
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            pred = model(lr)
            _, components = loss_fn(pred, hr)
            for k in totals:
                totals[k] += components[k] * lr.size(0)
            n += lr.size(0)
    return {k: v / n for k, v in totals.items()}


def train_one_config(patch_dir, alpha, beta, epochs, device, run_name, lr_rate=1e-3, seed=0,
                      warmup_epochs=5, grad_clip=1.0):
    torch.manual_seed(seed)

    full_ds = TorchPatchDataset(patch_dir)
    n_val = max(1, int(0.2 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    sample_lr, sample_hr = full_ds[0]
    scale_factor = sample_hr.shape[1] // sample_lr.shape[1]

    model = ResidualSRNet(in_channels=4, num_features=64, num_residual_blocks=3,
                           scale_factor=scale_factor).to(device)
    loss_fn = GeoSRLoss(alpha=alpha, beta=beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)
    
    history = []
    for epoch in range(1, epochs + 1):
        # Warm up alpha/beta over the first `warmup_epochs` so SAM's large
        # early-training gradients (arccos near +-1) don't destabilize MSE
        # convergence. alpha=beta=0 runs (the plain-MSE baseline) are
        # unaffected since warmup_weight(epoch, 0, ...) == 0 always.
        loss_fn.alpha = warmup_weight(epoch, alpha, warmup_epochs)
        loss_fn.beta = warmup_weight(epoch, beta, warmup_epochs)

        model.train()
        running = {"mse": 0.0, "sam": 0.0, "ndvi": 0.0, "ndwi": 0.0, "total": 0.0}
        n_seen = 0
        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            optimizer.zero_grad()
            pred = model(lr_batch)
            loss, components = loss_fn(pred, hr_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            for k in running:
                running[k] += components[k] * lr_batch.size(0)
            n_seen += lr_batch.size(0)

        train_metrics = {k: v / n_seen for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        history.append({"epoch": epoch, "alpha": loss_fn.alpha, "beta": loss_fn.beta,
                         "lr": scheduler.get_last_lr()[0],
                         "train": train_metrics, "val": val_metrics})
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{run_name}] epoch {epoch:3d} (a={loss_fn.alpha:.3f} b={loss_fn.beta:.3f}) | "
                  f"train_total={train_metrics['total']:.5f} | "
                  f"val_mse={val_metrics['mse']:.5f} val_ndvi={val_metrics['ndvi']:.5f}")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"{run_name}.pt")
    torch.save(model.state_dict(), ckpt_path)

    return model, history, val_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train baseline (MSE) and GeoSR-X (MSE+SAM+NDVI) and compare")
    parser.add_argument("input_tif", help="Path to input Sentinel-2 4-band GeoTIFF (B2,B3,B4,B8)")
    parser.add_argument("--patch-dir", default="test_data/patches_real")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--scale-factor", type=int, default=2)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    from data_engine import DatasetConfig, build_training_dataset
    cfg = DatasetConfig(patch_size_hr=args.patch_size, scale_factor=args.scale_factor, stride_hr=args.stride)
    manifest = build_training_dataset(args.input_tif, args.patch_dir, cfg)
    print(f"Band info: {manifest['band_info']}")
    print(f"Built {manifest['n_patches_saved']} patches for training\n")

    if manifest['n_patches_saved'] < 20:
        print("WARNING: very few patches. Consider a smaller --patch-size or --stride, "
              "or check that the input tile actually covers a useful area "
              "(see manifest.json nodata warnings).")

    EPOCHS = args.epochs

    print("=" * 60)
    print("Training BASELINE (plain MSE, alpha=0, beta=0)")
    print("=" * 60)
    _, hist_baseline, final_baseline = train_one_config(
        args.patch_dir, alpha=0.0, beta=0.0, epochs=EPOCHS, device=device,
        run_name="baseline_mse", warmup_epochs=args.warmup_epochs,
    )

    print("\n" + "=" * 60)
    print("Training GeoSR-X (MSE + 0.1*SAM + 0.1*NDVI, with warmup)")
    print("=" * 60)
    _, hist_geosrx, final_geosrx = train_one_config(
        args.patch_dir, alpha=0.1, beta=0.1, epochs=EPOCHS, device=device,
        run_name="geosrx", warmup_epochs=args.warmup_epochs,
    )

    print("\n" + "=" * 60)
    print("FAIR FINAL COMPARISON (re-evaluated under identical measurement loss)")
    print("=" * 60)
    from torch.utils.data import DataLoader, random_split
    measurement_loss = GeoSRLoss(alpha=0.1, beta=0.1)
    ds = TorchPatchDataset(args.patch_dir)
    n_val = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_val
    _, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    results = {}

    for name in ["baseline_mse", "geosrx"]:
        model = ResidualSRNet(
            in_channels=4,
            num_features=64,
            num_residual_blocks=3,
            scale_factor=args.scale_factor
        ).to(device)

        state_dict = torch.load(
            os.path.join("checkpoints", f"{name}.pt"),
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(state_dict)
        model.eval()

        results[name] = evaluate(model, val_loader, measurement_loss, device)


    print(f"{'Metric':<10}{'Baseline':<15}{'GeoSR-X':<15}{'Delta':<10}")
    for k in ["mse", "sam", "ndvi", "ndwi"]:
        b = results["baseline_mse"][k]
        g = results["geosrx"][k]
        print(f"{k:<10}{b:<15.6f}{g:<15.6f}{g-b:+.6f}")

    with open("checkpoints/training_summary.json", "w") as f:
        json.dump({
            "input_file": args.input_tif,
            "baseline_final_val": results["baseline_mse"],
            "geosrx_final_val": results["geosrx"],
            "baseline_history": hist_baseline,
            "geosrx_history": hist_geosrx,
        }, f, indent=2)

    print("\nSaved checkpoints/training_summary.json")