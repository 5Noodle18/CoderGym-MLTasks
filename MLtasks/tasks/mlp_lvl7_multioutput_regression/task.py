"""
MLP Multi-Output Regression
==============================
Trains a single MLP to simultaneously predict MULTIPLE continuous target
variables from the same input features.

Why this differs from single-output regression:
    A standard regression model produces one number.  Multi-output regression
    produces a vector:

        f(x) = [ŷ₁, ŷ₂, ŷ₃]

    All outputs share the same learned hidden representations.  This forces
    the network to find features that are jointly predictive of every target,
    acting as an implicit regulariser and often outperforming three separate
    single-output models, especially when targets are correlated.

Loss:
    L = (1/N) Σᵢ ||yᵢ - ŷᵢ||²   (mean over samples of sum-squared-error)

Metrics reported:
    • Overall MSE / R²  (flattened across all outputs)
    • Per-target MSE and R²  (so you can see which outputs are harder)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = 'output/tasks/mlp_lvl7_multioutput_regression'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Required interface functions
# ══════════════════════════════════════════════════════════════════════════════

def get_task_metadata() -> Dict[str, Any]:
    return {
        'task_id':     'mlp_lvl7_multioutput_regression',
        'task_name':   'MLP Multi-Output Regression',
        'series':      'Neural Networks (MLP)',
        'level':       7,
        'task_type':   'regression',
        'num_outputs': 3,
        'dataset':     'synthetic_multioutput',
        'metrics':     ['mse', 'r2', 'per_target_r2'],
        'thresholds':  {'val_r2_overall': 0.85, 'val_mse_overall': 1.0},
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination for a 1-D array."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def make_dataloaders(
    num_samples:  int   = 2000,
    num_features: int   = 10,
    num_outputs:  int   = 3,
    val_ratio:    float = 0.2,
    batch_size:   int   = 64,
    noise:        float = 0.3,
) -> Tuple[DataLoader, DataLoader]:
    """
    Synthetic multi-output dataset.

    Targets are correlated non-linear functions of the same inputs:
        y₁ = 2x₀ - x₁² + 0.5x₂          + ε
        y₂ = x₀x₁ + sin(x₂) - x₃        + ε
        y₃ = x₀² + x₁ - 2cos(x₂) + x₃  + ε

    Sharing inputs that contribute to multiple targets makes multi-output
    learning genuinely advantageous over independent single-output models.
    """
    set_seed(42)
    X = np.random.randn(num_samples, num_features).astype(np.float32)

    Y = np.zeros((num_samples, num_outputs), dtype=np.float32)
    Y[:, 0] = 2*X[:, 0] - X[:, 1]**2 + 0.5*X[:, 2]
    Y[:, 1] = X[:, 0]*X[:, 1] + np.sin(X[:, 2]) - X[:, 3]
    Y[:, 2] = X[:, 0]**2 + X[:, 1] - 2*np.cos(X[:, 2]) + X[:, 3]
    Y += np.random.randn(*Y.shape).astype(np.float32) * noise

    # Standardise inputs
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    perm  = np.random.permutation(num_samples)
    X, Y  = X[perm], Y[perm]
    split = int(num_samples * (1 - val_ratio))

    X_tr = torch.from_numpy(X[:split]);  Y_tr = torch.from_numpy(Y[:split])
    X_va = torch.from_numpy(X[split:]);  Y_va = torch.from_numpy(Y[split:])

    train_loader = DataLoader(TensorDataset(X_tr, Y_tr),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va, Y_va),
                              batch_size=batch_size, shuffle=False)

    print(f"Train: {split}  |  Val: {num_samples - split}  |  Outputs: {num_outputs}")
    return train_loader, val_loader


# ── Model ──────────────────────────────────────────────────────────────────────

class MultiOutputMLP(nn.Module):
    """
    MLP whose output layer has `num_outputs` units (no activation — raw
    regression values).  All hidden representations are shared across
    every output simultaneously.
    """
    def __init__(self, in_dim: int, hidden_dims: List[int],
                 num_outputs: int, dropout: float = 0.2):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_outputs))   # ← multi-output head
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)          # shape: (batch, num_outputs)


def build_model(
    in_dim:      int       = 10,
    hidden_dims: List[int] = [128, 64, 32],
    num_outputs: int       = 3,
    dropout:     float     = 0.2,
    lr:          float     = 1e-3,
) -> Tuple[nn.Module, optim.Optimizer]:
    device = get_device()
    model  = MultiOutputMLP(in_dim, hidden_dims, num_outputs, dropout).to(device)
    opt    = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    total  = sum(p.numel() for p in model.parameters())
    print(f"MultiOutputMLP | outputs={num_outputs} | params={total:,}")
    return model, opt


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    model:        nn.Module,
    train_loader: DataLoader,
    optimizer:    optim.Optimizer,
    criterion:    nn.Module,
    epochs:       int = 80,
    print_every:  int = 10,
) -> List[float]:
    device = get_device()
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), Y_b)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg = running / len(train_loader)
        losses.append(avg)
        if epoch % print_every == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  MSE={avg:.4f}")
    return losses


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module) -> Dict[str, Any]:
    """
    Returns:
        mse          - overall MSE (all outputs flattened)
        r2           - overall R²
        per_target   - list of {mse, r2} per output
        predictions  - numpy array (N, num_outputs)
        targets      - numpy array (N, num_outputs)
    """
    device = get_device()
    model.eval()
    preds_list, tgts_list = [], []
    total_loss = 0.0

    with torch.no_grad():
        for X_b, Y_b in loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            Y_hat = model(X_b)
            total_loss += criterion(Y_hat, Y_b).item()
            preds_list.append(Y_hat.cpu().numpy())
            tgts_list.append(Y_b.cpu().numpy())

    preds   = np.vstack(preds_list)    # (N, num_outputs)
    targets = np.vstack(tgts_list)

    # Overall metrics (flatten)
    overall_mse = float(np.mean((preds - targets) ** 2))
    overall_r2  = _r2(targets.ravel(), preds.ravel())

    # Per-target metrics
    per_target = []
    for k in range(targets.shape[1]):
        per_target.append({
            'mse': float(np.mean((preds[:, k] - targets[:, k]) ** 2)),
            'r2':  _r2(targets[:, k], preds[:, k]),
        })

    return {
        'loss':        total_loss / len(loader),
        'mse':         overall_mse,
        'r2':          overall_r2,
        'per_target':  per_target,
        'predictions': preds,
        'targets':     targets,
    }


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Return predicted output matrix (N, num_outputs)."""
    device = get_device()
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X.astype(np.float32)).to(device))
    return out.cpu().numpy()


# ── Artifacts ──────────────────────────────────────────────────────────────────

def save_artifacts(model: nn.Module, train_losses: List[float],
                   train_m: Dict, val_m: Dict) -> None:
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pt'))

    # Strip numpy arrays before serialising
    def strip(m):
        return {k: v for k, v in m.items()
                if not isinstance(v, np.ndarray)}

    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump({'train': strip(train_m), 'val': strip(val_m)}, f, indent=2)

    num_outputs = val_m['targets'].shape[1]
    fig, axes   = plt.subplots(1, num_outputs + 1,
                               figsize=(5 * (num_outputs + 1), 4))

    # Loss curve
    axes[0].plot(train_losses, color='steelblue')
    axes[0].set_title('Training Loss (MSE)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE')
    axes[0].grid(True)

    # Predicted vs True per target
    for k in range(num_outputs):
        ax  = axes[k + 1]
        tgt = val_m['targets'][:, k]
        prd = val_m['predictions'][:, k]
        ax.scatter(tgt, prd, alpha=0.4, s=10)
        lo, hi = tgt.min(), tgt.max()
        ax.plot([lo, hi], [lo, hi], 'r--')
        ax.set_title(f"Output {k+1}  R²={val_m['per_target'][k]['r2']:.3f}")
        ax.set_xlabel('True'); ax.set_ylabel('Predicted')
        ax.grid(True)

    plt.suptitle('Multi-Output Regression – Validation', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_lvl7_multioutput.png'),
                bbox_inches='tight')
    plt.close()
    print(f"Artifacts saved → {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("MLP Level 7 – Multi-Output Regression")
    print("=" * 60)

    set_seed(42)
    meta = get_task_metadata()
    print(f"Task  : {meta['task_name']}")
    print(f"Device: {get_device()}\n")

    NUM_FEATURES = 10
    NUM_OUTPUTS  = 3

    train_loader, val_loader = make_dataloaders(
        num_samples=2000, num_features=NUM_FEATURES,
        num_outputs=NUM_OUTPUTS, batch_size=64)

    model, optimizer = build_model(
        in_dim=NUM_FEATURES, hidden_dims=[128, 64, 32],
        num_outputs=NUM_OUTPUTS, dropout=0.2, lr=1e-3)

    criterion = nn.MSELoss()

    print("\nTraining …")
    train_losses = train(model, train_loader, optimizer, criterion,
                         epochs=80, print_every=10)

    print("\nEvaluating …")
    train_m = evaluate(model, train_loader, criterion)
    val_m   = evaluate(model, val_loader,   criterion)

    print(f"\nTrain  | MSE={train_m['mse']:.4f}  R²={train_m['r2']:.4f}")
    print(f"Val    | MSE={val_m['mse']:.4f}  R²={val_m['r2']:.4f}")
    print("\nPer-target validation R²:")
    for k, pt in enumerate(val_m['per_target']):
        print(f"  Output {k+1}: MSE={pt['mse']:.4f}  R²={pt['r2']:.4f}")

    save_artifacts(model, train_losses, train_m, val_m)

    # ── Assertions
    print("\n" + "=" * 60)
    print("Quality Assertions")
    print("=" * 60)

    exit_code = 0
    checks = [
        (val_m['r2'] > 0.85,
         f"Overall val R² > 0.85  →  {val_m['r2']:.4f}"),
        (val_m['mse'] < 1.0,
         f"Overall val MSE < 1.0  →  {val_m['mse']:.4f}"),
        (all(pt['r2'] > 0.70 for pt in val_m['per_target']),
         f"All per-target R² > 0.70  →  {[round(pt['r2'],3) for pt in val_m['per_target']]}"),
        (train_losses[-1] < train_losses[0],
         f"Loss decreased  {train_losses[0]:.4f} → {train_losses[-1]:.4f}"),
        (val_m['r2'] >= train_m['r2'] - 0.15,
         f"No severe overfit  (train R²={train_m['r2']:.4f}, val R²={val_m['r2']:.4f})"),
    ]
    for passed, msg in checks:
        mark = '|OOO|' if passed else '|XXX|'
        print(f"  {mark}  {msg}")
        if not passed:
            exit_code = 1

    print("\n" + ("PASS |OOO|" if exit_code == 0 else "FAIL |XXX|"))
    sys.exit(exit_code)
