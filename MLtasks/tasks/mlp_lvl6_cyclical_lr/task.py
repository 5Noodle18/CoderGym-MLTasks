"""
MLP with Cyclical Learning Rate (CLR)
========================================
Trains an MLP classifier using PyTorch's CyclicLR scheduler to demonstrate:
oscillating the learning rate can escape shallow local minima.

Cyclical LR schedule (triangular policy):
    lr(t) = lr_min + (lr_max - lr_min) * max(0, 1 - |t/step_size - 2k - 1|)
    where k = floor(t / (2 * step_size))

    +---- cycle 1 ----+---- cycle 2 ----+
    lr_max  *         *
           / |       / |
          /  |      /  |
    lr_min   *  *  *   *   ...
    t(ime) -->

REASON:
    A monotonically decaying LR will converge to whichever local minimum it
    finds first.  Periodically spiking the LR gives the optimiser enough
    energy to escape shallow (poor-quality) minima and settle in flatter,
    more general ones.

"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = 'output/tasks/mlp_lvl6_cyclical_lr'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Required interface functions
# ══════════════════════════════════════════════════════════════════════════════

def get_task_metadata() -> Dict[str, Any]:
    return {
        'task_id':    'mlp_lvl6_cyclical_lr',
        'task_name':  'MLP with Cyclical Learning Rate',
        'series':     'Neural Networks (MLP)',
        'level':      6,
        'task_type':  'classification',
        'dataset':    'synthetic_multiclass',
        'scheduler':  'CyclicLR (triangular)',
        'metrics':    ['accuracy', 'loss', 'f1_score'],
        'thresholds': {'val_accuracy': 0.75, 'val_f1': 0.70},
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(
    num_samples:  int   = 2000,
    num_features: int   = 30,
    num_classes:  int   = 4,
    val_ratio:    float = 0.2,
    batch_size:   int   = 64,
) -> Tuple[DataLoader, DataLoader]:
    """Synthetic class-conditional Gaussian data, standardised."""
    set_seed(42)
    Xs, ys = [], []
    n = num_samples // num_classes
    for c in range(num_classes):
        mu = np.random.randn(num_features) * 2.5
        Xs.append(np.random.randn(n, num_features) + mu)
        ys.append(np.full(n, c))

    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    perm  = np.random.permutation(len(X))
    X, y  = X[perm], y[perm]
    split = int(len(X) * (1 - val_ratio))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split])),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:])),
        batch_size=batch_size, shuffle=False)

    print(f"Train: {split} | Val: {len(X) - split}")
    return train_loader, val_loader


# ── Model ──────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Standard MLP with BatchNorm and Dropout."""
    def __init__(self, in_dim: int, hidden_dims: List[int],
                 num_classes: int, dropout: float = 0.3):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    in_dim:      int         = 30,
    hidden_dims: List[int]   = [128, 128, 64],
    num_classes: int         = 4,
    dropout:     float       = 0.3,
    lr_min:      float       = 1e-4,
    lr_max:      float       = 1e-2,
    step_size:   int         = 20,      # steps per half-cycle
) -> Tuple[nn.Module, optim.Optimizer, CyclicLR]:
    """Build model, SGD optimizer, and CyclicLR scheduler."""
    device = get_device()
    model  = MLP(in_dim, hidden_dims, num_classes, dropout).to(device)
    # SGD is preferred with CyclicLR (Adam's adaptive rates interact poorly)
    opt    = optim.SGD(model.parameters(), lr=lr_min,
                       momentum=0.9, weight_decay=1e-4)
    scheduler = CyclicLR(opt, base_lr=lr_min, max_lr=lr_max,
                         step_size_up=step_size, mode='triangular',
                         cycle_momentum=False)
    total = sum(p.numel() for p in model.parameters())
    print(f"MLP | hidden={hidden_dims} | params={total:,}")
    print(f"CyclicLR | lr=[{lr_min}, {lr_max}] | step_size={step_size}")
    return model, opt, scheduler


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    model:        nn.Module,
    train_loader: DataLoader,
    optimizer:    optim.Optimizer,
    criterion:    nn.Module,
    scheduler:    CyclicLR,
    epochs:       int = 80,
    print_every:  int = 10,
) -> Tuple[List[float], List[float]]:
    """
    Train with CyclicLR; return (per-epoch losses, per-batch lr history).
    The scheduler is stepped every BATCH (not every epoch) as recommended
    by Smith (2017).
    """
    device  = get_device()
    losses  = []
    lr_hist = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            scheduler.step()                        # ← step per batch
            running += loss.item()
            lr_hist.append(optimizer.param_groups[0]['lr'])

        avg = running / len(train_loader)
        losses.append(avg)
        if epoch % print_every == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg:.4f}  lr={cur_lr:.6f}")

    return losses, lr_hist


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module) -> Dict[str, float]:
    from sklearn.metrics import f1_score
    device = get_device()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits     = model(X_b)
            total_loss += criterion(logits, y_b).item()
            pred        = logits.argmax(1)
            correct    += (pred == y_b).sum().item()
            total      += y_b.size(0)
            preds_all.extend(pred.cpu().numpy())
            labels_all.extend(y_b.cpu().numpy())

    return {
        'loss':     total_loss / len(loader),
        'accuracy': correct / total,
        'f1':       float(f1_score(labels_all, preds_all, average='macro')),
    }


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = get_device()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)).to(device))
    return logits.argmax(1).cpu().numpy()


# ── Artifacts ──────────────────────────────────────────────────────────────────

def save_artifacts(model: nn.Module, train_losses: List[float],
                   lr_hist: List[float],
                   train_m: Dict, val_m: Dict) -> None:
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pt'))
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump({'train': train_m, 'val': val_m}, f, indent=2)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    # Loss curve — characteristic sawtooth visible
    axes[0].plot(train_losses, color='steelblue')
    axes[0].set_title('Training Loss (CyclicLR – sawtooth signature)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('CE Loss')
    axes[0].grid(True)

    # LR schedule
    axes[1].plot(lr_hist, color='darkorange', linewidth=0.8)
    axes[1].set_title('Learning Rate Schedule (per batch)')
    axes[1].set_xlabel('Batch step'); axes[1].set_ylabel('LR')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_lvl6_clr.png'))
    plt.close()
    print(f"Artifacts saved → {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("MLP Level 6 – Cyclical Learning Rate")
    print("=" * 60)

    set_seed(42)
    meta = get_task_metadata()
    print(f"Task  : {meta['task_name']}")
    print(f"Device: {get_device()}\n")

    NUM_FEATURES = 30
    NUM_CLASSES  = 4
    BATCH_SIZE   = 64
    EPOCHS       = 80
    # Step size ≈ 2–8× steps-per-epoch is recommended by Smith (2017)
    STEPS_PER_EPOCH = int(np.ceil(2000 * 0.8 / BATCH_SIZE))
    STEP_SIZE       = 2 * STEPS_PER_EPOCH

    train_loader, val_loader = make_dataloaders(
        num_samples=2000, num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES, batch_size=BATCH_SIZE)

    model, optimizer, scheduler = build_model(
        in_dim=NUM_FEATURES, hidden_dims=[128, 128, 64],
        num_classes=NUM_CLASSES, dropout=0.3,
        lr_min=1e-4, lr_max=1e-2, step_size=STEP_SIZE)

    criterion = nn.CrossEntropyLoss()

    print("\nTraining with CyclicLR …")
    train_losses, lr_hist = train(model, train_loader, optimizer,
                                  criterion, scheduler,
                                  epochs=EPOCHS, print_every=10)

    print("\nEvaluating …")
    train_m = evaluate(model, train_loader, criterion)
    val_m   = evaluate(model, val_loader,   criterion)

    print(f"\nTrain  | loss={train_m['loss']:.4f}  acc={train_m['accuracy']:.4f}  f1={train_m['f1']:.4f}")
    print(f"Val    | loss={val_m['loss']:.4f}  acc={val_m['accuracy']:.4f}  f1={val_m['f1']:.4f}")

    save_artifacts(model, train_losses, lr_hist, train_m, val_m)

    # ── Assertions
    print("\n" + "=" * 60)
    print("Quality Assertions")
    print("=" * 60)

    # Verify LR actually cycled (min and max both appear in history)
    lr_range = max(lr_hist) - min(lr_hist)

    exit_code = 0
    checks = [
        (val_m['accuracy'] > 0.75,
         f"Val accuracy > 0.75  →  {val_m['accuracy']:.4f}"),
        (val_m['f1'] > 0.70,
         f"Val macro-F1 > 0.70  →  {val_m['f1']:.4f}"),
        (train_losses[-1] < train_losses[0],
         f"Final loss < initial loss  {train_losses[0]:.4f} → {train_losses[-1]:.4f}"),
        (lr_range > 1e-5,
         f"LR actually cycled  (range={lr_range:.6f})"),
        (val_m['accuracy'] >= train_m['accuracy'] - 0.15,
         f"No severe overfit  (train={train_m['accuracy']:.4f}, val={val_m['accuracy']:.4f})"),
    ]
    for passed, msg in checks:
        mark = '|OOO|' if passed else '|XXX|'
        print(f"  {mark}  {msg}")
        if not passed:
            exit_code = 1

    print("\n" + ("PASS |OOO|" if exit_code == 0 else "FAIL |XXX|"))
    sys.exit(exit_code)
