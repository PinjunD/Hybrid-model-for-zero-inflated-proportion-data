"""Journal-style plotting helpers shared by the real-data figures.

`paper_style` switches Matplotlib to a serif (Times/STIX) look, and
`support_figure` draws the two-panel data + suitability-region figure
used for the trawl and turtle analyses. Positive outcomes are drawn in
the warm colour of the high end of the suitability scale, zeros in a
cool colour, so the two panels share one colour semantics.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def paper_style():
    """Serif fonts and restrained axes for publication figures."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
    })


def support_figure(X, Y, Wg, x1g, x2g, tau=0.5,
                   point_size=None, alpha=None,
                   xlabel="", ylabel="",
                   title_left="Observed data",
                   title_right="Estimated suitability region",
                   figsize=(9.2, 3.8)):
    """Two panels: observed data (left) and the fitted corrected outcome
    with the tau-level boundary (right). Both panels share their axes;
    depth is inverted (ocean convention). Returns the figure."""
    n = len(Y)
    if point_size is None:
        point_size = (5, 4) if n > 2000 else (18, 16)
    if alpha is None:
        alpha = (0.35, 0.07) if n > 2000 else (0.8, 0.7)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True,
                             sharey=True)
    ax = axes[0]
    ax.scatter(X[Y > 0, 0], X[Y > 0, 1], c="#B2182B", marker="o",
               s=point_size[0], alpha=alpha[0], rasterized=True,
               linewidths=0, label="positive")
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], c="#4393C3", marker="x",
               s=point_size[1], alpha=alpha[1], rasterized=True,
               linewidths=0.7 if n <= 2000 else 0.4, label="zero")
    ax.set_title(title_left)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="0.8", markerscale=2.2 if n > 2000 else 1.1)

    ax = axes[1]
    im = ax.contourf(x1g, x2g, Wg.T, levels=np.linspace(0, 1, 21),
                     cmap="viridis", vmin=0, vmax=1)
    ax.contour(x1g, x2g, Wg.T, levels=[tau], colors="black",
               linewidths=1.8, linestyles="--")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                      ticks=np.linspace(0, 1, 6))
    cb.ax.tick_params(labelsize=8)
    cb.set_label(r"$\hat Z(X)$", fontsize=9.5)
    cb.ax.plot([0, 1], [tau, tau], color="black", lw=1.2, ls="--")
    ax.set_title(title_right)

    for ax in axes:
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    for ax in axes:
        ax.invert_yaxis()
        ax.set_xlim(x1g[0], x1g[-1])
        ax.set_ylim(x2g[-1], x2g[0])
    fig.tight_layout(w_pad=2.2)
    return fig
