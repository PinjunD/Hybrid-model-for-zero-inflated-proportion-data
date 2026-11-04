"""Data generation for the simulation study.

Model: mY | X ~ Bin(m, sigma(X'theta*) * I_{S*}(X)), X ~ N(0, I_2),
theta* = (1, 1). Three support configurations:
  case 1: S* = {||X|| <= 1}   (bounded)
  case 2: S* = {||X|| >= 1}   (unbounded)
  case 3: S* = R^2            (no structural zeros)
"""
import numpy as np

from algorithm.model import sig

THETA_STAR = np.array([1.0, 1.0])


def gen(rng, n, m, case):
    """Draw one dataset; returns (X, Y, inside)."""
    X = rng.standard_normal((n, 2))
    r = np.linalg.norm(X, axis=1)
    if case == 1:
        inside = r <= 1.0
    elif case == 2:
        inside = r >= 1.0
    else:
        inside = np.ones(n, dtype=bool)
    Y = rng.binomial(m, sig(X @ THETA_STAR) * inside).astype(float) / m
    return X, Y, inside
