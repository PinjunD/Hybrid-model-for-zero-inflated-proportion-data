"""Corrected-kernel support estimation.

corrected_target : the corrected outcome W = I(Y>0) / max(r, 0.05).
bw_theory        : rate-optimal bandwidth of Theorem 3 (simulation).
nw_bw            : leave-one-out cross-validated Gaussian bandwidth (real data).
nw_pred          : Nadaraya-Watson prediction.
support_error    : empirical P_X(S_hat delta S*).
"""
import numpy as np

from .model import EPS, sig


def bw_theory(n, alpha=1.0, C=1.0):
    """Rate-optimal bandwidth of Theorem `thm:support` (d=2).

    h_n = C * (log n / n)^{alpha/(2*alpha+1)} with alpha = 1, i.e.
    (log n / n)^{1/3}; used throughout the simulation study. The real
    data analysis keeps the LOO-CV bandwidth below.
    """
    return float(C * (np.log(n) / n) ** (alpha / (2 * alpha + 1)))


def corrected_target(X, Y, m, theta):
    """Corrected outcome W from a fitted theta."""
    r = 1 - (1 - np.clip(sig(X @ theta), EPS, 1 - EPS)) ** m
    return (Y > 0).astype(float) / np.maximum(r, 0.05)


def nw_bw(X, t, hs, n_sub=None, seed=0):
    """LOO cross-validated bandwidth of the Gaussian-kernel NW estimator.

    n_sub: optional subsample size for large datasets (the real-data
    analysis passes n_sub=1500).
    """
    if n_sub is not None and n_sub < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), n_sub, replace=False)
        X, t = X[idx], t[idx]
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    best_h, best_e = None, np.inf
    for h in hs:
        K = np.exp(-d2 / (2 * h * h))
        np.fill_diagonal(K, 0)
        q = (K @ t) / np.maximum(K.sum(1), EPS)
        e = float(((t - q) ** 2).sum())
        if e < best_e:
            best_h, best_e = h, e
    return best_h


def nw_pred(Xtr, t, Xte, h, chunk=8000):
    """Nadaraya-Watson prediction at Xte, chunked for memory."""
    out = np.empty(len(Xte))
    for a in range(0, len(Xte), chunk):
        b = min(a + chunk, len(Xte))
        d2 = ((Xte[a:b, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        K = np.exp(-d2 / (2 * h * h))
        out[a:b] = (K @ t) / np.maximum(K.sum(1), EPS)
    return out


def support_error(pred, inside):
    """Empirical P_X(S_hat delta S*) on the evaluation sample."""
    return float(np.mean(pred != inside))
