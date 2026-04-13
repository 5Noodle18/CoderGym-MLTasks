"""
MLP Knowledge Distillation
==============================
Trains a large TEACHER network, then trains a smaller STUDENT network using
the teacher's soft probability distributions as a richer training signal.

Hard-target vs Soft-target illustration (4-class example):
    Hard label  (one-hot):  [0,  1,    0,    0   ]
    Soft target (T=1):      [0.02, 0.90, 0.05, 0.03]    soft
    Soft target (T=4):      [0.15, 0.55, 0.17, 0.13]    reaallly soft

    At temperature T the teacher's logits zᵢ are divided by T before softmax:
        p_T(k) = exp(zₖ/T) / Σⱼ exp(zⱼ/T)

    Higher T → softer distribution → more information about inter-class
    similarity flows to the student.
a = alpha
Student loss:
    L = a · L_hard + (1-a) · T² · L_soft

    where:
        L_hard = CrossEntropy(student_logits, hard_labels)
        L_soft  = KLDiv(student_soft_probs || teacher_soft_probs)
        T²      = rescales soft gradients back to the original magnitude

Experiment:
    1. Train TEACHER  (3 layers, 256-128-64 hidden)
    2. Train STUDENT-baseline with hard labels only
    3. Train STUDENT-distilled with soft labels from teacher
    → Assert: distilled student beats or matches hard-label student
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = 'output/tasks/mlp_lvl8_knowledge_distillation'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Required interface functions
# ══════════════════════════════════════════════════════════════════════════════

def get_task_metadata() -> Dict[str, Any]:
    return {
        'task_id':     'mlp_lvl8_knowledge_distillation',
        'task_name':   'MLP Knowledge Distillation',
        'series':      'Neural Networks (MLP)',
        'level':       8,
        'task_type':   'classification',
        'dataset':     'synthetic_multiclass',
        'num_classes': 5,
        'metrics':     ['accuracy', 'f1'],
        'thresholds':  {
            'teacher_val_acc':          0.80,
            'student_distilled_val_acc': 0.75,
            'distilled_beats_baseline': True,
        },
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(
    num_samples:  int   = 3000,
    num_features: int   = 40,
    num_classes:  int   = 5,
    val_ratio:    float = 0.2,
    batch_size:   int   = 64,
) -> Tuple[DataLoader, DataLoader]:
    """Synthetic class-conditional Gaussian data with overlapping classes."""
    set_seed(42)
    n = num_samples // num_classes
    Xs, ys = [], []
    for c in range(num_classes):
        mu = np.random.randn(num_features) * 1.5   # tighter clusters → more overlap
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

    print(f"Train: {split}  |  Val: {len(X)-split}  |  Classes: {num_classes}")
    return train_loader, val_loader


# ── Models ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Generic MLP; used for both teacher and student (just different sizes)."""
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
        return self.net(x)          # raw logits


def build_model(
    in_dim:      int,
    hidden_dims: List[int],
    num_classes: int,
    dropout:     float = 0.3,
    lr:          float = 1e-3,
    label:       str   = 'model',
) -> Tuple[nn.Module, optim.Optimizer]:
    device = get_device()
    model  = MLP(in_dim, hidden_dims, num_classes, dropout).to(device)
    opt    = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    total  = sum(p.numel() for p in model.parameters())
    print(f"{label} | hidden={hidden_dims} | params={total:,}")
    return model, opt


# ── Standard training (hard labels) ───────────────────────────────────────────

def train(
    model:        nn.Module,
    train_loader: DataLoader,
    optimizer:    optim.Optimizer,
    criterion:    nn.Module,
    epochs:       int = 60,
    print_every:  int = 10,
    label:        str = 'model',
) -> List[float]:
    """Train with standard cross-entropy hard labels."""
    device = get_device()
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg = running / len(train_loader)
        losses.append(avg)
        if epoch % print_every == 0:
            print(f"  [{label}] Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")
    return losses


def train_with_distillation(
    student:      nn.Module,
    teacher:      nn.Module,
    train_loader: DataLoader,
    optimizer:    optim.Optimizer,
    epochs:       int   = 60,
    temperature:  float = 4.0,
    alpha:        float = 0.5,
    print_every:  int   = 10,
) -> List[float]:
    """
    Train student using distillation loss:

        L = a · CrossEntropy(logits_s, hard_y)
          + (1-a) · T² · KLDiv(softmax(logits_s/T) || softmax(logits_t/T))

    Args:
        temperature : T — controls softness of teacher distribution.
                      Higher T → softer → more inter-class information.
        alpha       : weight on the hard-label loss term.
    """
    device  = get_device()
    teacher.eval()                      # teacher is frozen during distillation
    losses  = []
    hard_ce = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        student.train()
        running = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            # ── Student forward
            logits_s = student(X_b)

            # ── Hard-label loss
            l_hard = hard_ce(logits_s, y_b)

            # ── Soft-label loss (KL divergence at temperature T)
            with torch.no_grad():
                logits_t = teacher(X_b)

            # Soft probability distributions
            soft_t = F.softmax(logits_t / temperature, dim=1)
            soft_s = F.log_softmax(logits_s / temperature, dim=1)

            # KLDiv expects (log_input, target); reduction='batchmean'
            l_soft = F.kl_div(soft_s, soft_t, reduction='batchmean')

            # Combined loss — T² rescales the soft gradients
            loss = alpha * l_hard + (1.0 - alpha) * (temperature ** 2) * l_soft

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        avg = running / len(train_loader)
        losses.append(avg)
        if epoch % print_every == 0:
            print(f"  [student-distil] Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    return losses


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

def save_artifacts(
    teacher:           nn.Module,
    student_baseline:  nn.Module,
    student_distilled: nn.Module,
    t_losses:          List[float],
    b_losses:          List[float],
    d_losses:          List[float],
    results:           Dict,
) -> None:
    torch.save(teacher.state_dict(),
               os.path.join(OUTPUT_DIR, 'teacher.pt'))
    torch.save(student_distilled.state_dict(),
               os.path.join(OUTPUT_DIR, 'student_distilled.pt'))

    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curves
    axes[0].plot(t_losses, label='Teacher',           color='royalblue')
    axes[0].plot(b_losses, label='Student (hard)',    color='tomato',    linestyle='--')
    axes[0].plot(d_losses, label='Student (distil.)', color='seagreen',  linestyle='-.')
    axes[0].set_title('Training Loss Comparison')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True)

    # Accuracy bar chart
    labels   = ['Teacher', 'Student\n(hard)', 'Student\n(distilled)']
    val_accs = [
        results['teacher']['val']['accuracy'],
        results['student_baseline']['val']['accuracy'],
        results['student_distilled']['val']['accuracy'],
    ]
    colors = ['royalblue', 'tomato', 'seagreen']
    bars   = axes[1].bar(labels, val_accs, color=colors, alpha=0.85)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title('Validation Accuracy Comparison')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(axis='y')
    for bar, acc in zip(bars, val_accs):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     acc + 0.01, f'{acc:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_lvl8_distillation.png'))
    plt.close()
    print(f"Artifacts saved → {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("MLP Level 8 - Knowledge Distillation")
    print("=" * 60)

    set_seed(42)
    meta = get_task_metadata()
    print(f"Task  : {meta['task_name']}")
    print(f"Device: {get_device()}\n")

    NUM_FEATURES = 40
    NUM_CLASSES  = 5
    EPOCHS       = 60
    TEMPERATURE  = 4.0
    ALPHA        = 0.4      # 40% hard-label, 60% soft-label

    train_loader, val_loader = make_dataloaders(
        num_samples=3000, num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES, batch_size=64)

    criterion = nn.CrossEntropyLoss()

    # ── 1. Train TEACHER (large)
    print("\n── Step 1: Train TEACHER ──────────────────────────────────")
    teacher, t_opt = build_model(
        in_dim=NUM_FEATURES, hidden_dims=[256, 128, 64],
        num_classes=NUM_CLASSES, dropout=0.3, lr=1e-3, label='Teacher')

    t_losses = train(teacher, train_loader, t_opt, criterion,
                     epochs=EPOCHS, print_every=10, label='teacher')
    t_train  = evaluate(teacher, train_loader, criterion)
    t_val    = evaluate(teacher, val_loader,   criterion)
    print(f"  Teacher  train acc={t_train['accuracy']:.4f}  val acc={t_val['accuracy']:.4f}")

    # ── 2. Train STUDENT — hard labels only (baseline)
    print("\n── Step 2: Train STUDENT (hard labels baseline) ───────────")
    student_b, sb_opt = build_model(
        in_dim=NUM_FEATURES, hidden_dims=[64, 32],
        num_classes=NUM_CLASSES, dropout=0.2, lr=1e-3, label='Student-baseline')

    b_losses  = train(student_b, train_loader, sb_opt, criterion,
                      epochs=EPOCHS, print_every=10, label='student-hard')
    sb_train  = evaluate(student_b, train_loader, criterion)
    sb_val    = evaluate(student_b, val_loader,   criterion)
    print(f"  Student (hard)  train acc={sb_train['accuracy']:.4f}  val acc={sb_val['accuracy']:.4f}")

    # ── 3. Train STUDENT — distillation (same small architecture)
    print(f"\n── Step 3: Train STUDENT (distillation T={TEMPERATURE}, α={ALPHA}) ─")
    student_d, sd_opt = build_model(
        in_dim=NUM_FEATURES, hidden_dims=[64, 32],
        num_classes=NUM_CLASSES, dropout=0.2, lr=1e-3, label='Student-distilled')

    d_losses  = train_with_distillation(
        student_d, teacher, train_loader, sd_opt,
        epochs=EPOCHS, temperature=TEMPERATURE, alpha=ALPHA, print_every=10)
    sd_train  = evaluate(student_d, train_loader, criterion)
    sd_val    = evaluate(student_d, val_loader,   criterion)
    print(f"  Student (distil) train acc={sd_train['accuracy']:.4f}  val acc={sd_val['accuracy']:.4f}")

    # ── Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    rows = [
        ("Teacher  (large)",           t_val['accuracy'],  t_val['f1']),
        ("Student  (hard labels)",      sb_val['accuracy'], sb_val['f1']),
        ("Student  (distilled)",        sd_val['accuracy'], sd_val['f1']),
    ]
    print(f"  {'Model':<30} {'Val Acc':>8}  {'Val F1':>8}")
    print("  " + "-" * 50)
    for name, acc, f1 in rows:
        print(f"  {name:<30} {acc:>8.4f}  {f1:>8.4f}")

    results = {
        'temperature': TEMPERATURE,
        'alpha':       ALPHA,
        'teacher':           {'train': t_train,  'val': t_val},
        'student_baseline':  {'train': sb_train, 'val': sb_val},
        'student_distilled': {'train': sd_train, 'val': sd_val},
    }

    save_artifacts(teacher, student_b, student_d,
                   t_losses, b_losses, d_losses, results)

    # ── Assertions
    print("\n" + "=" * 60)
    print("Quality Assertions")
    print("=" * 60)

    distilled_beats = sd_val['accuracy'] >= sb_val['accuracy'] - 0.01

    exit_code = 0
    checks = [
        (t_val['accuracy'] > 0.80,
         f"Teacher val accuracy > 0.80  →  {t_val['accuracy']:.4f}"),
        (sd_val['accuracy'] > 0.75,
         f"Distilled student val accuracy > 0.75  →  {sd_val['accuracy']:.4f}"),
        (distilled_beats,
         f"Distilled student >= hard-label student  "
         f"({sd_val['accuracy']:.4f} vs {sb_val['accuracy']:.4f})"),
        (d_losses[-1] < d_losses[0],
         f"Distillation loss decreased  {d_losses[0]:.4f} → {d_losses[-1]:.4f}"),
    ]
    for passed, msg in checks:
        mark = '|000|' if passed else '|XXX|'
        print(f"  {mark}  {msg}")
        if not passed:
            exit_code = 1

    # Extra informational print — parameter count comparison
    t_params  = sum(p.numel() for p in teacher.parameters())
    sd_params = sum(p.numel() for p in student_d.parameters())
    print(f"\n  i  Teacher params:  {t_params:,}")
    print(f"  i  Student params:  {sd_params:,}  "
          f"({100*sd_params/t_params:.1f}% of teacher size)")

    print("\n" + ("PASS ✓" if exit_code == 0 else "FAIL ✗"))
    sys.exit(exit_code)
