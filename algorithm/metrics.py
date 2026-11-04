"""Evaluation metrics."""
import numpy as np


def theta_error(theta_hat, theta_star):
    """Squared Euclidean error of theta_hat against theta_star."""
    return float(np.sum((theta_hat - theta_star) ** 2))
