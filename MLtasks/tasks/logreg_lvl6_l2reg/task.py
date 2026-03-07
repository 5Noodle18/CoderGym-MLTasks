"""
Binary Logistic Regression: Compare Unregularized vs L2 Regularization
Trains two models side-by-side for direct comparison.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Seed & device
torch.manual_seed(42)
np.random.seed(42)
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Task metadata
def get_task_metadata():
    return {
        'task_type': 'binary_classification',
        'input_dim': 2,
        'output_dim': 1,
        'description': 'Binary logistic regression with optional L2 regularization (comparison)',
        'loss_type': 'binary_cross_entropy',
        'regularization': 'l2'
    }

# Synthetic dataset
def make_dataloaders(batch_size=32, val_ratio=0.2):
    n_samples = 400
    n_class = n_samples // 2
    cluster_0 = np.random.randn(n_class,2)*1.2 + np.array([-1,-1])
    cluster_1 = np.random.randn(n_class,2)*1.2 + np.array([1,1])
    X = np.vstack([cluster_0, cluster_1])
    y = np.array([0]*n_class + [1]*n_class)
    # Standardize
    X_mean, X_std = X.mean(0), X.std(0)
    X = (X - X_mean)/X_std
    # Shuffle and split
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    val_size = int(len(X)*val_ratio)
    X_val, X_train = X[:val_size], X[val_size:]
    y_val, y_train = y[:val_size], y[val_size:]
    # Convert tensors
    X_train_t, y_train_t = torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t, y_val_t = torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Logistic Regression Model
class LogisticRegressionL2(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.weights = torch.nn.Parameter(torch.randn(input_dim,1)*0.01)
        self.bias = torch.nn.Parameter(torch.zeros(1))
    def sigmoid(self,z):
        return 1/(1+torch.exp(-z))
    def forward(self,x):
        return self.sigmoid(torch.matmul(x,self.weights)+self.bias)
    def get_params(self):
        return self.weights.data.cpu().numpy(), self.bias.data.cpu().numpy()

# Gradients
def compute_gradients(model, X, y_true, l2_lambda=0.0, use_l2=False):
    batch_size = X.size(0)
    z = torch.matmul(X, model.weights) + model.bias
    y_pred = model.sigmoid(z)
    error = y_pred - y_true
    grad_w = torch.matmul(X.t(), error)/batch_size
    if use_l2:
        grad_w += l2_lambda*model.weights
    grad_b = torch.sum(error)/batch_size
    return grad_w, grad_b

# Loss (Comparing between unregularized and L2 regularized versions)
def compute_loss(model, X, y_true, l2_lambda=0.0, use_l2=False):
    eps = 1e-7
    y_pred = model.forward(X)
    bce_loss = -torch.mean(y_true*torch.log(y_pred+eps) + (1-y_true)*torch.log(1-y_pred+eps))
    if use_l2:
        bce_loss += (l2_lambda/2)*torch.sum(model.weights**2) #this is the key difference in loss when using L2 regularization
    return bce_loss

# Train
def train(model, train_loader, val_loader, device, learning_rate=0.1, epochs=500, use_l2=False, l2_lambda=0.01):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            grad_w, grad_b = compute_gradients(model, X_batch, y_batch, l2_lambda, use_l2)
            # Manual parameter update (gradient descent)
            with torch.no_grad():
                model.weights -= learning_rate*grad_w
                model.bias -= learning_rate*grad_b
    return model

# Evaluate loss (Comparing between unregularized and L2 regularized versions)
def evaluate_loss(model, data_loader, device, use_l2=False, l2_lambda=0.0):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            loss = compute_loss(model, X_batch, y_batch, l2_lambda, use_l2)

            total_loss += loss.item()
            count += 1

    return total_loss / count

# Evaluate accuracy
def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_targets = [], []
    # Logic: Since the model outputs probabilities, we threshold at 0.5 to get binary predictions
    #        y_pred comes from the sigmoid output in the model, y_true comes from the dataset labels. We then compare them to calculate accuracy.
    for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model.forward(X_batch)
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    y_pred_binary = (y_pred>=0.5).float() # Threshold at 0.5 for binary classification
    # Calculate accuracy
    accuracy = torch.mean((y_pred_binary==y_true).float()).item() 
    return accuracy

# Save artifacts
def save_artifacts(models_metrics, l2_lambda, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name, (model, metrics) in models_metrics.items():
        torch.save({'weights':model.weights.data.cpu(),'bias':model.bias.data.cpu()},
                   os.path.join(output_dir,f'{name}_model.pt'))
        np.save(os.path.join(output_dir,f'{name}_metrics.npy'), metrics)
        np.save(os.path.join(output_dir,f'{name}_l2_lambda.npy'),
                np.array([l2_lambda if 'l2' in name else 0.0]))
    print(f"Artifacts saved to {output_dir}")

# ---------------- Main ----------------
if __name__=="__main__":
    OUTPUT_DIR = '/Developer/AIserver/output/tasks/logreg_lvl6_l2reg_comparison'
    DEVICE = get_device()
    train_loader, val_loader = make_dataloaders()
    
    # Unregularized
    model_no_l2 = LogisticRegressionL2(input_dim=2)
    model_no_l2 = train(model_no_l2, train_loader, val_loader, DEVICE, use_l2=False)
    metrics_no_l2 = evaluate(model_no_l2, val_loader, DEVICE)
    
    # L2 Regularized
    L2_LAMBDA = 0.1
    model_l2 = LogisticRegressionL2(input_dim=2)
    model_l2 = train(model_l2, train_loader, val_loader, DEVICE, use_l2=True, l2_lambda=L2_LAMBDA)
    metrics_l2 = evaluate(model_l2, val_loader, DEVICE)
    
    # Collect metrics for comparison
    train_loss_no_l2 = evaluate_loss(model_no_l2, train_loader, DEVICE)
    val_loss_no_l2 = evaluate_loss(model_no_l2, val_loader, DEVICE)
    accuracy_no_l2 = evaluate(model_no_l2, val_loader, DEVICE)

    train_loss_l2 = evaluate_loss(model_l2, train_loader, DEVICE, True, L2_LAMBDA)
    val_loss_l2 = evaluate_loss(model_l2, val_loader, DEVICE, True, L2_LAMBDA)
    accuracy_l2 = evaluate(model_l2, val_loader, DEVICE)

    weight_norm_no_l2 = torch.norm(model_no_l2.weights).item()
    weight_norm_l2 = torch.norm(model_l2.weights).item()
    
    metrics_no_l2 = {
    "accuracy": accuracy_no_l2,
    "train_loss": train_loss_no_l2,
    "val_loss": val_loss_no_l2,
    "weight_norm": weight_norm_no_l2,
    "l2_reg_strength": 0.0
    }

    metrics_l2 = {
    "accuracy": accuracy_l2,
    "train_loss": train_loss_l2,
    "val_loss": val_loss_l2,
    "weight_norm": weight_norm_l2,
    "l2_reg_strength": L2_LAMBDA
    }
    
    print("Unregularized metrics:", metrics_no_l2)
    print("L2 Regularized metrics:", metrics_l2)