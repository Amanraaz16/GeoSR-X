import torch, json, os
from model import ResidualSRNet
from losses import GeoSRLoss
from train import TorchPatchDataset, evaluate
from torch.utils.data import DataLoader, random_split

ds = TorchPatchDataset('test_data/patches_real')
n_val = max(1, int(0.2 * len(ds)))
n_train = len(ds) - n_val
_, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

measurement_loss = GeoSRLoss(alpha=0.1, beta=0.1)

results = {}
for name in ['baseline_mse', 'geosrx']:
    model = ResidualSRNet(in_channels=4, num_features=64, num_residual_blocks=3, scale_factor=2)
    model.load_state_dict(torch.load(os.path.join('checkpoints', name + '.pt'), weights_only=True))
    results[name] = evaluate(model, val_loader, measurement_loss, 'cpu')

print(f"{'Metric':<10}{'Baseline':<15}{'GeoSR-X':<15}{'Delta':<10}")
for k in ['mse', 'sam', 'ndvi']:
    b, g = results['baseline_mse'][k], results['geosrx'][k]
    print(f'{k:<10}{b:<15.6f}{g:<15.6f}{g-b:+.6f}')