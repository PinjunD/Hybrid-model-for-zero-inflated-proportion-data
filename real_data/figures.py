"""Full-sample support figures (paper Figures 1 and 4).

Task separation: this script does NO cross-validation. It fits the
debiased MLE once on the full sample of each dataset and draws the
support estimate. Performance comparisons live in rerun_all.py
(turtle CV) and full_sample.py (trawl full-sample CV), both of which
are cross-validation only.

Outputs: results/figure/fig_trawl_full.pdf, fig_turtle_support.pdf.
Run from the repository root:
    python real_data/figures.py
"""
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "real_data"))

import matplotlib
matplotlib.use("Agg")
import numpy as np

from algorithm.estimators import fit_debiased
from algorithm.kernel_support import corrected_target, nw_bw, nw_pred
from algorithm.plotting import paper_style, support_figure
from config import FIGD, HS_REAL, ensure_dirs
from full_sample import load_full_trawl
from load_data import load_turtle


def full_fit(loader):
    """Fit once on the full sample; return the grid of fitted values."""
    X, Y, m = loader["X"], loader["Y"], loader["m"]
    mu, sd = X.mean(0), X.std(0)
    Xs = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    theta = fit_debiased(Xs, Y, m)
    W = corrected_target(Xs, Y, m, theta)
    h = nw_bw(Xs, W, hs=HS_REAL, n_sub=1500)
    x1g = np.linspace(X[:, 0].min(), X[:, 0].max(), 120)
    x2g = np.linspace(X[:, 1].min(), X[:, 1].max(), 120)
    Xg = np.column_stack([
        np.array([[(a - mu[0]) / sd[0], (b - mu[1]) / sd[1]]
                  for a in x1g for b in x2g]),
        np.ones(len(x1g) * len(x2g))])
    Wg = nw_pred(Xs, W, Xg, h).reshape(len(x1g), len(x2g))
    return X, Y, Wg, x1g, x2g, theta, h


def main():
    ensure_dirs()
    paper_style()

    loader = load_full_trawl()
    X, Y, Wg, x1g, x2g, theta, h = full_fit(loader)
    fig = support_figure(
        X, Y, Wg, x1g, x2g,
        xlabel=r"Bottom temperature ($^\circ$C)", ylabel="Depth (m)",
        title_left=f"Observed data ($n = {len(X):,}$)")
    fig.savefig(str(FIGD / "fig_trawl_full.pdf"), dpi=300,
                bbox_inches="tight")
    print(f"saved {FIGD / 'fig_trawl_full.pdf'} "
          f"(theta_std={np.round(theta, 3)}, h={h:.3f})")

    loader = load_turtle()
    X, Y, Wg, x1g, x2g, theta, h = full_fit(loader)
    fig = support_figure(
        X, Y, Wg, x1g, x2g,
        xlabel=r"Nest temperature ($^\circ$C)", ylabel="Elevation (m)",
        title_left=f"Observed data ($n = {len(X)}$)", figsize=(9.2, 3.6))
    fig.savefig(str(FIGD / "fig_turtle_support.pdf"), dpi=300,
                bbox_inches="tight")
    print(f"saved {FIGD / 'fig_turtle_support.pdf'} "
          f"(theta_std={np.round(theta, 3)}, h={h:.3f})")


if __name__ == "__main__":
    main()
