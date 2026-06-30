import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here

    # initialization
    N,d = X.shape
    w = np.zeros((d,))
    b = 0.0

    
    for iter in range(steps):
        # forward
        scores = X @ w + b

        y_hat = _sigmoid(scores)

        # backward
        error = y_hat - y           # (N,)
        grad_w = (X.T @ error) / N  # (d,)
        grad_b = error.sum() / N    # scalar

        # update
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b