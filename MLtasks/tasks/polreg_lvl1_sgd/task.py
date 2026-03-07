import torch
import numpy as np
import random
import time
from torch.utils.data import TensorDataset, DataLoader


# -----------------------------
# Metadata
# -----------------------------
def get_task_metadata():
    return {
        "task_id": "polreg_lvl1_sgd",
        "algorithm": "Polynomial Regression (Stochastic Gradient Descent)",
        "description": "SGD polynomial regression compared with closed-form solution"
    }


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------
# Device
# -----------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Data Generation
# -----------------------------
def make_dataloaders(batch_size=32):

    n_samples = 1000

    x = torch.linspace(-2, 2, n_samples).unsqueeze(1)

    noise = torch.randn(n_samples, 1) * 0.2

    # true polynomial relationship with noise (5, 2, -1 coefficients)
    y = 5 * x + 2 * x**2 - x**3 + noise

    # polynomial features
    X = torch.cat([x, x**2, x**3], dim=1)

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return loader, X, y


# -----------------------------
# Model
# -----------------------------
def build_model():

    # 3 polynomial features → 1 output
    model = torch.nn.Linear(3,1)

    return model


# -----------------------------
# R² Metric
# -----------------------------
def r2_score(y_true, y_pred):

    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum((y_true - torch.mean(y_true))**2)

    return 1 - ss_res/ss_tot


# -----------------------------
# Closed Form Solution
# -----------------------------
def closed_form_solution(X, y):

    X_np = X.numpy()
    y_np = y.numpy()

    theta = np.linalg.inv(X_np.T @ X_np) @ X_np.T @ y_np

    return theta


# -----------------------------
# Training (SGD)
# -----------------------------
def train(model, dataloader, device, epochs=100, lr=0.01):

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss_fn = torch.nn.MSELoss()

    loss_history = []

    start = time.time()

    for epoch in range(epochs):

        # learning rate decay
        for g in optimizer.param_groups:
            g['lr'] = lr / (1 + 0.01 * epoch)

        for X_batch, y_batch in dataloader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)

            loss = loss_fn(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = loss.item()

        epoch_loss /= len(dataloader)
        loss_history.append(epoch_loss)

    training_time = time.time() - start

    # extract the learned parameters (theta) from the model
    weights = model.weight.data.cpu().numpy()
    bias = model.bias.data.cpu().numpy()

    metrics = {
        "training_time_seconds": training_time,
        "final_loss": loss_history[-1],
        "sgd_weights": weights.tolist(),
        "sgd_bias": bias.tolist()
    }

    return metrics


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, X, y, device):

    model.eval()

    with torch.no_grad():

        X = X.to(device)
        y = y.to(device)

        preds = model(X)

        r2 = r2_score(y, preds)

    return r2.item(), preds.cpu()


# -----------------------------
# Predict
# -----------------------------
def predict(model, X, device):

    model.eval()

    with torch.no_grad():

        X = X.to(device)
        preds = model(X)

    return preds.cpu()


# -----------------------------
# Save Artifacts
# -----------------------------
def save_artifacts(model, metrics):

    artifacts = {
        "model_state_dict": model.state_dict(),
        "metrics": metrics
    }

    return artifacts


# -----------------------------
# Main Task Execution
# -----------------------------
if __name__ == "__main__":

    set_seed()

    device = get_device()

    loader, X, y = make_dataloaders()

    model = build_model()

    training_metrics = train(model, loader, device)

    r2_sgd, preds = evaluate(model, X, y, device)

    theta_closed = closed_form_solution(X, y)

    metrics = {
        "r2_sgd": r2_sgd,
        "closed_form_theta": theta_closed.tolist(),
    }

    metrics.update(training_metrics)

    artifacts = save_artifacts(model, metrics)

    print(metrics)