"""
Linear Regression using the Normal Equation (Analytical Solution)

Mathematical Formulation:
Theta = (X^T X)^(-1) X^T y

This computes the optimal parameters directly instead of using gradient descent. 
This means we can find the best fit line in one step, without needing to iterate through multiple epochs.
We will see that the normal equation is efficient for smaller featured datasets, 
BUT can become computationally expensive for larger feature sets due to the matrix inversion step.

Note: a feature is a weight in the linear regression model, otherwise known as a parameter (x_i in the equation).
"""

import os
import sys
import time
import numpy as np
import torch

OUTPUT_DIR = '/Developer/AIserver/output/tasks/linreg_lvl5_normal_equation'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        'task_name': 'linear_regression_normal_equation',
        'description': 'Linear regression solved analytically using the normal equation',
        'input_dim': 1,
        'output_dim': 1,
        'model_type': 'linear_regression',
        'loss_type': 'mse',
        'optimization': 'normal_equation'
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_device():
    """Get the appropriate device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataset(n_samples=200, noise_std=0.5):
    """
    Arbitrary, new dataset: y = 5x - 4 + feature-dimension noise
    """

    X = np.random.uniform(-3, 3, (n_samples))
    y = 5 * X - 4 + np.random.normal(0, noise_std, n_samples)

    X = torch.FloatTensor(X).unsqueeze(1)
    y = torch.FloatTensor(y).unsqueeze(1)

    return X, y

def make_dataset_features(n_samples=200, n_features=100, noise_std=0.5):
    """
    y = 5*x_1 + 0*x_2 + ... + 0*x_n - 4 + noise
    """
    # Create random data for all features
    X = np.random.uniform(-3, 3, (n_samples, n_features))
    
    # Define weights: First feature is 5, all others are 0
    weights = np.zeros((n_features, 1))
    weights[0] = 5 
    
    bias = -4
    
    # Matrix multiplication: y = Xw + b
    y = (X @ weights) + bias + np.random.normal(0, noise_std, (n_samples, 1))

    return torch.FloatTensor(X), torch.FloatTensor(y)


class NormalEquationLinearRegression:

    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        """
        Compute parameters using the Normal Equation:

        theta = (X^T X)^(-1) X^T y

        Note: there are no epochs or learning rates here since we are solving for the optimal parameters directly.
        """

        # Add bias column
        ones = torch.ones(X.shape[0], 1, device=X.device)
        X_b = torch.cat([ones, X], dim=1)

        XtX = X_b.T @ X_b
        XtX_inv = torch.inverse(XtX)

        # Compute theta using the normal equation
        self.theta = XtX_inv @ X_b.T @ y

    def predict(self, X):

        ones = torch.ones(X.shape[0], 1)
        X_b = torch.cat([ones, X], dim=1)

        return X_b @ self.theta

    def evaluate(self, X, y):

        y_pred = self.predict(X)

        mse = torch.mean((y_pred - y) ** 2).item()

        ss_res = torch.sum((y - y_pred) ** 2).item()
        ss_tot = torch.sum((y - torch.mean(y)) ** 2).item()

        # this results in the R-squared (coefficient of determination)
        r2 = 1 - (ss_res / ss_tot)

        return {
            "mse": mse,
            "r2": r2,
            "theta_0": self.theta[0].item(),
            "theta_1": self.theta[1].item(),
        }


def main():

    set_seed()

    print("\nTest 1: Small dataset with 1 feature")
    X, y = make_dataset()
    model = NormalEquationLinearRegression()
    start_time = time.time() 
    model.fit(X, y)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training completed in {duration:.4f} seconds")
    metrics = model.evaluate(X, y)
    print("Evaluation Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nTest 2: Same dataset but with 100 features")
    X, y = make_dataset_features()
    model = NormalEquationLinearRegression()
    start_time = time.time() 
    model.fit(X, y)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training completed in {duration:.4f} seconds")
    metrics = model.evaluate(X, y)
    print("Evaluation Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v}")


    # self-verification exit code
    if metrics["mse"] < 1.0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()