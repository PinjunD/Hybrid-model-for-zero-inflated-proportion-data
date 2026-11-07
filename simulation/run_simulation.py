"""Full simulation study: coefficient and support estimation.

- n grid: {200,400,600,800,1000}; tables at n=1000 (config.NTAB);
  m in {5,10,15}; 300 reps.
- Support estimation uses the theoretical bandwidth of Theorem thm:support:
  h_n = (log n / n)^{1/3}   (alpha=1, C=1, d=2).
- Outputs: sim_theta.json, sim_support_conv.json, fig_sim_theta.pdf,
  fig_sim_support_conv.pdf, and LaTeX-ready table rows with bold (best
  non-oracle) / underline (second best) marks.
- The paper's Figures 2-3 (serif style, M axis) are re-emitted from
  the cache by simulation/redraw_figs.py.

Run from hybrid_model_for_zero_inflated_proportion_data/.
"""
import json, sys, os, time
from pathlib import Path
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithm.estimators import fit_debiased, fit_full, fit_oracle, fit_pos
from algorithm.kernel_support import (bw_theory, corrected_target, nw_pred,
                                      support_error)
from algorithm.metrics import theta_error
from config import CASES, FIGD, MS, NREP, NS, NTAB, TEMP, ensure_dirs, sim_seed
from dgp import THETA_STAR, gen

t0 = time.time()
ensure_dirs()

# ============================================================
# 1. Theta simulation (full grid, 300 reps)
# ============================================================
print("=== Theta simulation ===", flush=True)
theta_res = {}
for case in CASES:
    for m in MS:
        for n in NS:
            acc = {k: [] for k in ("oracle", "full", "pos", "ours")}
            for sd in range(NREP):
                rng = np.random.default_rng(sim_seed(sd, n, m))
                X, Y, inside = gen(rng, n, m, case)
                fits = {
                    "oracle": fit_oracle(X, Y, m, inside),
                    "ours": fit_debiased(X, Y, m),
                    "full": fit_full(X, Y, m),
                    "pos": fit_pos(X, Y, m),
                }
                for k, th in fits.items():
                    acc[k].append(theta_error(th, THETA_STAR))
            key = f"c{case}_m{m}_n{n}"
            theta_res[key] = {k: float(np.mean(v)) for k, v in acc.items()}
            if n == NTAB:
                r = theta_res[key]
                print(f"c{case} m={m} n={n}: orc={r['oracle']:.4f} "
                      f"full={r['full']:.4f} pos={r['pos']:.4f} "
                      f"ours={r['ours']:.4f} ({time.time()-t0:.0f}s)", flush=True)

(TEMP / "sim_theta.json").write_text(json.dumps(theta_res, indent=2))

# ============================================================
# 2. Support simulation, theoretical bandwidth, full grid
# ============================================================
print("\n=== Support simulation (theoretical h_n) ===", flush=True)
supp_res = {}
for case in CASES:
    for m in MS:
        for n in NS:
            h = bw_theory(n)
            e_cor, e_unc = [], []
            for sd in range(NREP):
                rng = np.random.default_rng(sim_seed(sd, n, m))
                X, Y, inside = gen(rng, n, m, case)
                Z = (Y > 0).astype(float)
                theta = fit_debiased(X, Y, m)
                W = corrected_target(X, Y, m, theta)
                for target, acc in ((W, e_cor), (Z, e_unc)):
                    Wa = nw_pred(X, target, X, h)
                    acc.append(support_error(Wa > 0.5, inside))
            key = f"c{case}_m{m}_n{n}"
            supp_res[key] = {"ours": float(np.mean(e_cor)),
                             "unc": float(np.mean(e_unc))}
            if n == NTAB:
                r = supp_res[key]
                print(f"c{case} m={m} n={n}: ours={r['ours']:.4f} "
                      f"unc={r['unc']:.4f} ({time.time()-t0:.0f}s)", flush=True)

(TEMP / "sim_support_conv.json").write_text(json.dumps(supp_res, indent=2))

# ============================================================
# 3. LaTeX-ready table rows
# ============================================================
def fmt(v, nd=4):
    s = f"{v:.{nd}f}"
    return s[1:] if s.startswith("0") else s

def marks(vals, nd=4):
    """Bold = best (min) at displayed precision; underline = next distinct."""
    r = [round(v, nd) for v in vals]
    best = min(r)
    bold = {i for i, v in enumerate(r) if v == best}
    rest = sorted({v for i, v in enumerate(r) if i not in bold})
    und = set()
    if rest:
        und = {i for i, v in enumerate(r) if v == rest[0]}
    return bold, und

print("\n=== tab:main rows (n=%d) ===" % NTAB)
rows = {k: [] for k in ("oracle", "naive", "uncorrected", "ours")}
for case in CASES:
    for m in MS:
        r = theta_res[f"c{case}_m{m}_n{NTAB}"]
        rows["oracle"].append(fmt(r["oracle"]))
        vals = [r["full"], r["pos"], r["ours"]]   # naive, uncorrected, ours
        bold, und = marks(vals)
        for i, (name, v) in enumerate(zip(("naive", "uncorrected", "ours"), vals)):
            cell = fmt(v)
            if i in bold:
                cell = r"\textbf{" + cell + "}"
            elif i in und:
                cell = r"\underline{" + cell + "}"
            rows[name].append(cell)

for name, cells in rows.items():
    line = name + " & " + " & ".join(cells)
    print(line + r" \\")

print("\n=== tab:support rows (n=%d) ===" % NTAB)
sup = {"uncorrected": [], "ours": []}
extra = []
for case in CASES:
    for m in MS:
        r = supp_res[f"c{case}_m{m}_n{NTAB}"]
        u, o = round(r["unc"], 3), round(r["ours"], 3)
        b = "uncorrected" if u <= o else "ours"
        for name, v in (("uncorrected", u), ("ours", o)):
            cell = fmt(v, 3)
            if name == b:
                cell = r"\textbf{" + cell + "}"
            sup[name].append(cell)
        extra.append((case, m, r["unc"] / r["ours"]))

for name, cells in sup.items():
    print(name + " & " + " & ".join(cells) + r" \\")

print("\nuncorrected/ours ratios at n=%d:" % NTAB)
for case, m, ratio in extra:
    print(f"  Case {case} m={m}: {ratio:.2f}")

# ============================================================
# 4. Theta convergence figure
# ============================================================
print("\n=== Theta figure ===", flush=True)
STYLES = {
    "oracle": ("k", "-", "o", "oracle"),
    "full": ("#0072B2", "--", "^", "naive"),
    "pos": ("#009E73", ":", "D", "uncorrected"),
    "ours": ("#D55E00", "-", "s", "ours"),
}
fig, axes = plt.subplots(3, len(MS), figsize=(3 * len(MS), 7.5), sharex=True)
for i, case in enumerate(CASES):
    for j, m in enumerate(MS):
        ax = axes[i, j]
        for key, (c, ls, mk, lab) in STYLES.items():
            ys = [theta_res[f"c{case}_m{m}_n{n}"][key] for n in NS]
            ax.plot(NS, ys, ls, color=c, marker=mk, ms=4, lw=1.2,
                    label=lab, mfc="none")
        ax.set_yscale("log")
        if i == 2:
            ax.set_xlabel("$n$", fontsize=8)
        if j == 0:
            ax.set_ylabel(r"$\|\hat\theta-\theta^*\|^2$", fontsize=8)
        ax.set_title(f"$m$={m}", fontsize=9)
        ax.tick_params(labelsize=7)
for i, case in enumerate(CASES):
    axes[i, 0].annotate(f"Case {case}", xy=(-0.6, 0.5),
                        xycoords="axes fraction", fontsize=10, rotation=90,
                        va="center", ha="center", fontweight="bold")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=[0.05, 0, 1, 0.96])
fig.savefig(str(FIGD / "fig_sim_theta.pdf"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved fig_sim_theta.pdf")

# ============================================================
# 5. Support convergence figure
# ============================================================
print("\n=== Support convergence figure ===", flush=True)
fig, axes = plt.subplots(3, len(MS), figsize=(3 * len(MS), 7.5), sharex=True)
sup_styles = {
    "unc": ("#009E73", ":", "D", "uncorrected"),
    "ours": ("#D55E00", "-", "s", "ours"),
}
for i, case in enumerate(CASES):
    for j, m in enumerate(MS):
        ax = axes[i, j]
        for key, (c, ls, mk, lab) in sup_styles.items():
            ys = [supp_res[f"c{case}_m{m}_n{n}"][key] for n in NS]
            ax.plot(NS, ys, ls, color=c, marker=mk, ms=4, lw=1.2,
                    label=lab, mfc="none")
        ax.set_yscale("log")
        if i == 2:
            ax.set_xlabel("$n$", fontsize=8)
        if j == 0:
            ax.set_ylabel(r"$\mathcal{E}(\hat S)$", fontsize=8)
        ax.set_title(f"$m$={m}", fontsize=9)
        ax.tick_params(labelsize=7)
for i, case in enumerate(CASES):
    axes[i, 0].annotate(f"Case {case}", xy=(-0.6, 0.5),
                        xycoords="axes fraction", fontsize=10, rotation=90,
                        va="center", ha="center", fontweight="bold")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=[0.05, 0, 1, 0.96])
fig.savefig(str(FIGD / "fig_sim_support_conv.pdf"), dpi=180,
            bbox_inches="tight")
plt.close(fig)
print("saved fig_sim_support_conv.pdf")
