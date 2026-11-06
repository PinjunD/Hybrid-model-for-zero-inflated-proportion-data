"""Full-sample (unscreened) trawl analysis: the robustness check of
Section 5.2 and Figure 5 of the paper.

Sample: every haul with non-missing bottom temperature and depth and
m >= 1 (n = 30,131; no upper bound on m). Protocol identical to the
main analysis: five folds (seed 42), fold-standardized covariates plus
intercept, tau = 1/2, LOO-CV bandwidth subsampled to 1,500 points,
weighted AUC/MCC, unweighted MAE.

Outputs (results/temp/full_sample.json, results/figure/fig_trawl_full.pdf):

  - five-fold metrics of our method and the three parametric baselines,
  - the full-sample support figure,
  - optionally (pass --with-tabicl, ~69 min) the TabICLv2 rows.

Run from the repository root:
    python real_data/full_sample.py
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "real_data"))

import numpy as np
import pandas as pd

from algorithm.estimators import fit_debiased
from algorithm.kernel_support import corrected_target, nw_bw, nw_pred
from algorithm.model import sig
from config import HS_REAL, TEMP, ensure_dirs
from rerun_all import mae, wauc, wmcc
from run_experiment import R_CODE, RSCRIPT, folds_of


def load_full_trawl():
    """All hauls with observed covariates, m >= 1 (no upper bound)."""
    g = pd.read_csv(PKG / "data" / "trawl.csv")
    keep = g.btemp.notna() & g.btemp.gt(-2) & g.depth.notna() \
        & (g.n_total >= 1)
    g = g[keep].copy()
    g["y"] = g.k_cod / g.n_total
    return {
        "X": g[["btemp", "depth"]].values,
        "Y": g.y.values,
        "m": g.n_total.values.astype(float),
        "columns": ["bottom temperature (C)", "depth (m)"],
    }


def design(X, tr, te):
    """Fold-standardized covariates with an intercept column."""
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Xs_tr = np.column_stack([(X[tr] - mu) / sd, np.ones(len(tr))])
    Xs_te = np.column_stack([(X[te] - mu) / sd, np.ones(len(te))])
    return Xs_tr, Xs_te


def ours_cv(X, Y, m, folds):
    """Five-fold predictions of our method on the full sample.

    Returns (yhat, rhat, Doof): mean predictions, positivity scores,
    and the out-of-fold fitted corrected outcome used by the model
    check.
    """
    yhat_l, rhat_l, d_l = [], [], []
    for te in folds:
        tr = np.setdiff1d(np.arange(len(Y)), te)
        Xs_tr, Xs_te = design(X, tr, te)
        theta = fit_debiased(Xs_tr, Y[tr], m[tr])
        W = corrected_target(Xs_tr, Y[tr], m[tr], theta)
        h = nw_bw(Xs_tr, W, hs=HS_REAL, n_sub=1500)
        D = nw_pred(Xs_tr, W, Xs_te, h, chunk=4000)
        s = sig(Xs_te @ theta)
        r = 1 - (1 - s) ** m[te]
        yhat_l.append(s * (D > 0.5))
        rhat_l.append(r * D)
        d_l.append(D)
    return np.concatenate(yhat_l), np.concatenate(rhat_l), np.concatenate(d_l)


def r_baselines_cv(X, Y, m, folds):
    """Hurdle/ZIB/ZOIB via the same embedded R code as the main run."""
    work = TEMP / "full_sample_r"
    work.mkdir(exist_ok=True)
    (work / "folds.json").write_text(
        json.dumps({"trawl": [[int(i) for i in f] for f in folds]}))
    pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": Y, "m": m}).to_csv(
        work / "trawl.csv", index=False)
    src = (R_CODE.replace("RESULTS_TEMP", str(work).replace("\\", "/"))
                .replace('for (tag in c("trawl", "turtle")) {',
                         'for (tag in c("trawl")) {'))
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(src)
        script = fh.name
    res = subprocess.run([RSCRIPT, script], capture_output=True, text=True)
    Path(script).unlink()
    if res.returncode != 0:
        print(res.stderr[-1500:])
        raise SystemExit("R baseline fitting failed")
    base = pd.read_csv(work / "baseline_preds.csv")
    return {meth: base[base.method == meth].sort_values("fold")
            for meth in ("hurdle", "zib", "zoib")}


def tabicl_cv(loader, folds):
    """TabICLv2 rows (~69 min on the full sample; optional)."""
    from run_experiment import tabicl_rows
    return tabicl_rows("trawl", loader)


def metrics_with_se(y, w, pred, score, folds):
    """Weighted AUC/MCC and unweighted MAE with fold-level SEs."""
    pos = y > 0
    fl = [len(f) for f in folds]
    idx = np.cumsum([0] + fl)
    fold = lambda fn: [fn(y[idx[i]:idx[i+1]], pred[idx[i]:idx[i+1]],
                          score[idx[i]:idx[i+1]], w[idx[i]:idx[i+1]],
                          pos[idx[i]:idx[i+1]]) for i in range(5)]
    se = lambda v: float(np.std(v, ddof=1) / np.sqrt(5))
    f_mae = [mae(y[idx[i]:idx[i+1]], pred[idx[i]:idx[i+1]]) for i in range(5)]
    f_auc = [wauc(score[idx[i]:idx[i+1]], pos[idx[i]:idx[i+1]],
                  w[idx[i]:idx[i+1]]) for i in range(5)]
    f_mcc = [wmcc(score[idx[i]:idx[i+1]], pos[idx[i]:idx[i+1]],
                  w[idx[i]:idx[i+1]]) for i in range(5)]
    return {"mae": mae(y, pred), "auc": wauc(score, pos, w),
            "mcc": wmcc(score, pos, w),
            "mae_se": se(f_mae), "auc_se": se(f_auc), "mcc_se": se(f_mcc)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-tabicl", action="store_true",
                    help="also evaluate TabICLv2 (~69 minutes)")
    args = ap.parse_args()
    ensure_dirs()
    t0 = time.time()

    loader = load_full_trawl()
    X, Y, m = loader["X"], loader["Y"], loader["m"]
    folds = folds_of(len(Y))
    print(f"full sample n={len(Y):,}, "
          f"zeros={float((Y == 0).mean()):.3f}", flush=True)

    yhat, rhat, Doof = ours_cv(X, Y, m, folds)
    y_order = np.concatenate([Y[f] for f in folds])
    w_order = np.concatenate([m[f] for f in folds])
    pos_order = y_order > 0
    out = {"ours": metrics_with_se(y_order, w_order, yhat, rhat, folds)}
    check = {}
    for q in (0.05, 0.10, 0.20):
        thr = np.quantile(Doof, q)
        check[f"{int(q*100)}"] = (int(pos_order[Doof <= thr].sum()),
                                   int((Doof <= thr).sum()))
    out["ours"]["_diag"] = check
    print("ours:", {k: round(v, 4) for k, v in out["ours"].items()
                    if isinstance(v, float)}, "| diag:", check, flush=True)

    bl = r_baselines_cv(X, Y, m, folds)
    for meth, b in bl.items():
        out[meth] = metrics_with_se(y_order, w_order, b.p.values,
                                    b.score.values, folds)
        print(meth + ":", {k: round(v, 4)
                           for k, v in out[meth].items()}, flush=True)

    if args.with_tabicl:
        tab = tabicl_cv(loader, folds)
        assert np.allclose(y_order, tab.y.values)
        out["tabicl"] = metrics_with_se(y_order, w_order, tab.p.values,
                                        tab.score.values, folds)
    else:
        out["tabicl_skipped"] = ("pass --with-tabicl to evaluate "
                                 "(~69 minutes); paper values: "
                                 "MAE .0266, AUC .492, MCC -.020")

    (TEMP / "full_sample.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {TEMP / 'full_sample.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
