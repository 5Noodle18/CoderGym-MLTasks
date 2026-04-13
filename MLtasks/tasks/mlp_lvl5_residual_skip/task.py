"""
MLP with Residual Skip Connections
====================================
Implements a multi-layer perceptron with residual connections on tabular data to use the residuals to create shortcuts

Architecture per residual block:
    out = Activation(BN(Linear(x))) + x   [when dims match] NORMAL CASE
    out = Activation(BN(Linear(x))) + proj(x)  [when dims differ, projection shortcut] SHORTCUT CASE

GOAL:
    In deep networks gradients vanish during backprop — they get multiplied by
    small numbers through many layers and shrink toward zero, starving early
    layers of signal.  The skip connection provides a direct gradient highway:

        dL/dx = dL/d(F(x)+x) * (dF/dx + I)
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

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = 'output/tasks/mlp_lvl5_residual_connections'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Required interface functions
# ══════════════════════════════════════════════════════════════════════════════

def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        'task_id':        'mlp_lvl5_residual_connections',
        'task_name':      'MLP with Residual Skip Connections',
        'series':         'Neural Networks (MLP)',
        'level':          5,
        'task_type':      'classification',
        'dataset':        'synthetic_multiclass',
        'num_classes':    4,
        'metrics':        ['accuracy', 'loss', 'f1_score'],
        'thresholds':     {'val_accuracy': 0.75, 'val_f1': 0.70},
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(
    num_samples: int = 2000,
    num_features: int = 32,
    num_classes: int = 4,
    val_ratio: float = 0.2,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """
    Generate a multiclass tabular data and return train/val loaders inside of Tuple[DataLoader, DataLoader].

    Data is drawn from class-conditional Gaussians with shared covariance,
    giving a moderately hard classification problem that benefits from depth.
    """
    set_seed(42)
    samples_per_class = num_samples // num_classes # divide based on inputted parameters
    X_list, y_list = [], [] # initialize empty arrays for X features, y predictions

    for c in range(num_classes):
        mean = np.random.randn(num_features) * 2 # generate a random 1-dim (num_features long) array [Gaussian], 2 is std. dev.
        X_c  = np.random.randn(samples_per_class, num_features) + mean # generate 2-dim array (samples n, num m) and offset by mean
        X_list.append(X_c)  #appends array
        y_list.append(np.full(samples_per_class, c)) #appends array

    X = np.vstack(X_list).astype(np.float32)    # reformat 
    y = np.concatenate(y_list).astype(np.int64) # reformat

    # Standardise
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    # Shuffle
    idx = np.random.permutation(len(X)) # generate an array of indices up to length of X (randomly)
    X, y = X[idx], y[idx] # reset lists with indices

    split = int(len(X) * (1 - val_ratio)) # calculate the int where the split will occur using length of X and val ratio
    X_train, y_train = torch.from_numpy(X[:split]),  torch.from_numpy(y[:split]) #create tensor using contents before :split 
    X_val,   y_val   = torch.from_numpy(X[split:]),  torch.from_numpy(y[split:]) #create tensor using contents after split:

    # call torch.utils.data to create dataloaders for each tensor set
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),
                              batch_size=batch_size, shuffle=False)

    print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples")
    return train_loader, val_loader


# ── Model ──────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Single residual block:
        out = Activation(BN(Linear(x))) + shortcut(x)

    If in_dim != out_dim a 1x1 linear projection aligns dimensions for the skip path; 
    otherwise the identity is used directly.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn     = nn.BatchNorm1d(out_dim)
        self.act    = nn.ReLU(inplace=True)
        self.drop   = nn.Dropout(dropout)

        # Projection shortcut when dimensions differ
        self.shortcut = (
            nn.Linear(in_dim, out_dim, bias=False) # derive shortcut here
            if in_dim != out_dim else nn.Identity() # don't if input/output are same dimension
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.bn(self.linear(x)))) 
        return out + self.shortcut(x)          # ← skip connection


class ResidualMLP(nn.Module):
    """
    Deep MLP where every pair of layers is wrapped in a ResidualBlock.

    Architecture:
        input (in_dim)
        → ResBlock(in_dim  → hidden)
        → ResBlock(hidden  → hidden)   [same dim → pure identity skip]
        → ResBlock(hidden  → hidden)
        → Linear(hidden → num_classes)
    """
    def __init__(self, in_dim: int, hidden: int, num_classes: int,
                 num_blocks: int = 3, dropout: float = 0.2):
        super().__init__()
        blocks = [ResidualBlock(in_dim, hidden, dropout)] # call the block class to make empty array of blocks
        for _ in range(num_blocks - 1):
            blocks.append(ResidualBlock(hidden, hidden, dropout)) # fill in array
        self.blocks = nn.Sequential(*blocks)
        self.head   = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x))


def build_model(
    in_dim:      int   = 32,
    hidden:      int   = 128,
    num_classes: int   = 4,
    num_blocks:  int   = 3,
    dropout:     float = 0.2,
    lr:          float = 1e-3,
) -> Tuple[nn.Module, optim.Optimizer]: # define a return of a Tuple[neural net, optimizer]
    """Instantiate ResidualMLP + Adam optimizer."""
    device = get_device()
    model  = ResidualMLP(in_dim, hidden, num_classes, num_blocks, dropout).to(device)
    opt    = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    total  = sum(p.numel() for p in model.parameters())
    print(f"ResidualMLP | blocks={num_blocks} | hidden={hidden} | params={total:,}")
    return model, opt


# ── Training ───────────────────────────────────────────────────────────────────

def train(
#   param         type
    model:        nn.Module,
    train_loader: DataLoader,
    optimizer:    optim.Optimizer,
    criterion:    nn.Module,
    epochs:       int = 60,
    print_every:  int = 10,
) -> List[float]:
    """Train model; return per-epoch average losses."""
    device = get_device()
    losses = []

    for epoch in range(1, epochs + 1):
        model.train()                               #set to training mode
        running = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device) #pass to device
            optimizer.zero_grad()                   #reset gradients
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()                        #take a step
            running += loss.item()
        avg = running / len(train_loader)
        losses.append(avg)
        if epoch % print_every == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    return losses


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module) -> Dict[str, float]:
    """Return loss, accuracy, and macro-F1 on the given loader."""
    from sklearn.metrics import f1_score

    device = get_device()
    model.eval()                            #set model to evaluation mdoe
    total_loss, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits    = model(X_b)
            total_loss += criterion(logits, y_b).item()
            pred       = logits.argmax(dim=1)
            correct   += (pred == y_b).sum().item()
            total     += y_b.size(0)
            preds_all.extend(pred.cpu().numpy())
            labels_all.extend(y_b.cpu().numpy())

    return {
        'loss':     total_loss / len(loader),
        'accuracy': correct / total,
        'f1':       float(f1_score(labels_all, preds_all, average='macro')),
    }


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Return predicted class indices for numpy array X."""
    device = get_device()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)).to(device))
    return logits.argmax(dim=1).cpu().numpy()


# ── Artifacts ──────────────────────────────────────────────────────────────────

def save_artifacts(model: nn.Module, train_losses: List[float],
                   train_metrics: Dict, val_metrics: Dict) -> None:
    """Save model weights, metrics JSON, and loss-curve plot."""
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pt'))

    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump({'train': train_metrics, 'val': val_metrics}, f, indent=2)

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('Cross-Entropy Loss')
    plt.title('ResidualMLP – Training Loss')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_lvl5_loss.png'))
    plt.close()
    print(f"Artifacts saved → {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("MLP Level 5 – Residual Skip Connections")
    print("=" * 60)

    set_seed(42)
    meta = get_task_metadata()
    print(f"Task : {meta['task_name']}")
    print(f"Device: {get_device()}\n")

    # ── Data
    NUM_FEATURES = 32
    NUM_CLASSES  = 4
    train_loader, val_loader = make_dataloaders(
        num_samples=2000, num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES, batch_size=64)

    # ── Model
    model, optimizer = build_model(
        in_dim=NUM_FEATURES, hidden=128, num_classes=NUM_CLASSES,
        num_blocks=3, dropout=0.2, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # ── Train
    print("\nTraining ResidualMLP …")
    train_losses = train(model, train_loader, optimizer, criterion,
                         epochs=60, print_every=10)

    # ── Evaluate
    print("\nEvaluating …")
    train_m = evaluate(model, train_loader, criterion)
    val_m   = evaluate(model, val_loader,   criterion)

    print(f"\nTrain  | loss={train_m['loss']:.4f}  acc={train_m['accuracy']:.4f}  f1={train_m['f1']:.4f}")
    print(f"Val    | loss={val_m['loss']:.4f}  acc={val_m['accuracy']:.4f}  f1={val_m['f1']:.4f}")

    # ── Save
    save_artifacts(model, train_losses, train_m, val_m)

    # ── Assertions
    print("\n" + "=" * 60)
    print("Quality Assertions")
    print("=" * 60)

    exit_code = 0
    checks = [
        (val_m['accuracy'] > 0.75,                                                              # pass
         f"Val accuracy > 0.75  →  {val_m['accuracy']:.4f}"),                                   # msg
        (val_m['f1'] > 0.70,                                                                    # pass
         f"Val macro-F1 > 0.70  →  {val_m['f1']:.4f}"),                                         # msg
        (train_losses[-1] < train_losses[0],                                                    # pass
         f"Loss decreased  {train_losses[0]:.4f} → {train_losses[-1]:.4f}"),                    # msg
        (val_m['accuracy'] >= train_m['accuracy'] - 0.15,                                       # pass
         f"No severe overfit  (train={train_m['accuracy']:.4f}, val={val_m['accuracy']:.4f})"), # msg
    ]
    for passed, msg in checks:
        mark = '|OOO|' if passed else '|XXX|'
        print(f"  {mark}  {msg}")
        if not passed:
            exit_code = 1

    print("\n" + ("PASS |OOO|" if exit_code == 0 else "FAIL |XXX|"))
    sys.exit(exit_code)
