import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    N = y_pred.shape[0]
    
    # pick the predicted prob of the true class for each row
    correct_logprobs = -np.log(y_pred[np.arange(N), y_true])
    return np.mean(correct_logprobs)