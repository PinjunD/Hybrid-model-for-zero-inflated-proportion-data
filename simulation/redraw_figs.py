"""Redraw the two simulation convergence figures in the shared serif style.

Reads the cached JSON produced by simulation/run_simulation.py and
re-emits fig_sim_theta.pdf and fig_sim_support_conv.pdf. No recomputation.
"""
import json
import sys
from pathlib import Path

PKG = Path(r"C:\Users\13299\Desktop\paper\hybrid_model_for_zero_inflated_proportion_data")
sys.path.insert(0, str(PKG))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithm.plotting import paper_style
from config import CASES, FIGD, MS, NS, TEMP

paper_style()

theta = json.loads((TEMP / "sim_theta.json").read_text())
support = json.loads((TEMP / "sim_support_conv.json").read_text())


def grid_figure(data, methods, ylabel, out_name, x_mode="n"):
    fig, axes = plt.subplots(
        3, len(MS), figsize=(2.6 * len(MS), 7.0), sharex=False,
        sharey=True)
    for i, case in enumerate(CASES):
        for j, m in enumerate(MS):
            ax = axes[i, j]
            xs = NS if x_mode == "n" else [n * m for n in NS]
            for key, c, ls, mk, lab in methods:
                ys = [data[f"c{case}_m{m}_n{n}"][key] for n in NS]
                ax.plot(xs, ys, ls, color=c, marker=mk, ms=3.5, lw=1.2,
                        label=lab, mfc="none")
            ax.set_yscale("log")
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            if i == 0:
                ax.set_title(f"$m$={m}", fontsize=10)
            ax.tick_params(labelsize=8)
            if i == 2:
                ax.set_xlabel("$M$" if x_mode == "M" else "$n$",
                              fontsize=10)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=10)
    for i, case in enumerate(CASES):
        axes[i, 0].annotate(f"Case {case}", xy=(-0.62, 0.5),
                            xycoords="axes fraction", fontsize=11,
                            rotation=90, va="center", ha="center",
                            fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0.06, 0, 1, 0.96])
    fig.savefig(str(FIGD / out_name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIGD / out_name)


grid_figure(
    theta,
    [("oracle", "k", "-", "o", "oracle"),
     ("full", "#0F4C81", "--", "^", "naive"),
     ("pos", "#4393C3", ":", "D", "uncorrected"),
     ("ours", "#B2182B", "-", "s", "ours")],
    r"$\|\hat\theta-\theta^*\|_2^2$",
    "fig_sim_theta.pdf",
    x_mode="M")

grid_figure(
    support,
    [("unc", "#4393C3", ":", "D", "uncorrected"),
     ("ours", "#B2182B", "-", "s", "ours")],
    r"$\mathcal{E}(\hat S)$",
    "fig_sim_support_conv.pdf")
