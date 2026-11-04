"""Logistic link."""
import numpy as np

EPS = 1e-12


def sig(u):
    """Logistic function, clipped input for numerical stability."""
    return 1 / (1 + np.exp(-np.clip(u, -30, 30)))
