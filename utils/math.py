import numpy as np


def log_loss(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Loss function or log loss, for visualization

    mean(-(y * log(y_pred)) + (1 - y) * log(1 - y_pred))
    """
    res: float = -(y * np.log(y_pred)
                   + (1 - y) * np.log(1 - y_pred)).mean()
    return res


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function, turns any value to 0-1"""
    res: np.ndarray = 1 / (1 + np.exp(-x))
    return res
