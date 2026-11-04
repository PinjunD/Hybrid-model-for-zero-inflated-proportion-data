"""Estimators of theta, all solved by Newton-Raphson with analytic
gradients and Hessians (finite-difference-free; converges to machine
precision in a handful of iterations).

The optimized objective equals m times the paper's quasi-likelihood
plus a constant (same argmax); gradients below therefore carry the
factor m where applicable.

fit_debiased  : our debiased MLE (paper eq:theta-hat).
fit_full      : full MLE on all observations, ignoring the support.
fit_pos       : MLE on positive observations, no debiasing.
fit_oracle    : MLE restricted to the true support (benchmark).
"""
import numpy as np

from .model import EPS, sig

TOL = 1e-12
MAXIT = 50


def _newton(grad, hess, d):
    th = np.zeros(d)
    for _ in range(MAXIT):
        g, H = grad(th), hess(th)
        step = np.linalg.solve(H + 1e-10 * np.eye(d), g)
        th = th - step
        if np.abs(step).max() < TOL:
            break
    return th


def fit_debiased(X, Y, m):
    """Debiased MLE: weighted conditional likelihood of the
    positives, weights m_i (paper eq:sample_loss). m may be an
    array (per-observation trials) or a scalar."""
    Xp, Yp = X[Y > 0], Y[Y > 0]
    mp = m if np.isscalar(m) else m[Y > 0]

    def grad(th):
        s = np.clip(sig(Xp @ th), EPS, 1 - EPS)
        r = np.clip(1 - (1 - s) ** mp, EPS, 1)
        coef = -mp * (Yp - s / r)
        return (Xp * coef[:, None]).sum(0)

    def hess(th):
        s = sig(Xp @ th)
        q = (1 - s) ** mp
        r = np.clip(1 - q, EPS, 1)
        d2 = mp * s * ((1 - s) * r - mp * s * q) / r ** 2
        return (Xp * d2[:, None]).T @ Xp

    return _newton(grad, hess, X.shape[1])


def _binom_fit(X, Y, m):
    def grad(th):
        s = sig(X @ th)
        coef = -m * (Y - s)
        return (X * coef[:, None]).sum(0)

    def hess(th):
        s = sig(X @ th)
        d2 = m * s * (1 - s)
        return (X * d2[:, None]).T @ X

    return _newton(grad, hess, X.shape[1])


def fit_full(X, Y, m):
    """Full MLE: all observations, support ignored."""
    return _binom_fit(X, Y, m)


def fit_pos(X, Y, m):
    """Positive MLE: positives only, no correction."""
    sel = Y > 0
    mfit = m if np.isscalar(m) else m[sel]
    return _binom_fit(X[sel], Y[sel], mfit)


def fit_oracle(X, Y, m, inside):
    """Oracle MLE: restricted to the true support (simulation only)."""
    mfit = m if np.isscalar(m) else m[inside]
    return _binom_fit(X[inside], Y[inside], mfit)
