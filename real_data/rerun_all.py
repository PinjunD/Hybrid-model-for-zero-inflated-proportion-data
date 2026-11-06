"""Turtle CV: ours and baselines for Table 5 (loggerhead nests).

Five-fold cross-validation on the turtle data (n = 164): our method,
the R parametric baselines (hurdle, ZIB, ZOIB, from run_experiment.py),
weighted AUC/MCC and unweighted MAE with fold SEs, and the model-check
counts. The trawl analysis uses the full unscreened sample and lives in
full_sample.py; the support figures live in figures.py.

Run from the repository root:
    python real_data/run_experiment.py   # first: folds + R baselines
    python real_data/rerun_all.py
"""
import json
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algorithm.estimators import fit_debiased
from algorithm.kernel_support import corrected_target, nw_bw, nw_pred
from algorithm.model import sig
from config import HS_REAL, TEMP, FIGD, ensure_dirs
from load_data import load_turtle
from run_experiment import folds_of

EPS = 1e-12

# ---------- metric helpers ----------

def wauc(p, pos, w):
    W1, W0 = w[pos].sum(), w[~pos].sum()
    if W1 == 0 or W0 == 0:
        return np.nan
    sn, wn = p[~pos], w[~pos]
    o = np.argsort(sn)
    sn_s, wn_s = sn[o], wn[o]
    cw = np.cumsum(wn_s)
    sp, wp = p[pos], w[pos]
    ir = np.searchsorted(sn_s, sp, "right")
    il = np.searchsorted(sn_s, sp, "left")
    below = np.where(il > 0, cw[il - 1], 0.0)
    ties = cw[ir - 1] - below
    return float((wp * (below + 0.5 * ties)).sum() / (W1 * W0))


def wmcc(p, pos, w):
    tp = w[pos & (p >= .5)].sum()
    fp = w[~pos & (p >= .5)].sum()
    fn = w[pos & (p < .5)].sum()
    tn = w[~pos & (p < .5)].sum()
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den else np.nan


def wmse(y, p, w):
    return float((w * (y - p) ** 2).sum() / w.sum())


def mae(y, p):
    return float(np.abs(y - p).mean())


# ---------- support figure ----------

def make_figure(X, Y, m, theta, W, h, tau, columns, title, out_path):
    """Support figure in the shared journal style (algorithm/plotting)."""
    from algorithm.plotting import paper_style, support_figure

    mu, sd = X.mean(0), X.std(0)
    Xs = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    x1g = np.linspace(X[:, 0].min(), X[:, 0].max(), 120)
    x2g = np.linspace(X[:, 1].min(), X[:, 1].max(), 120)
    Xg = np.column_stack([
        np.array([[(a - mu[0]) / sd[0], (b - mu[1]) / sd[1]]
                  for a in x1g for b in x2g]),
        np.ones(len(x1g) * len(x2g))])
    Wg = nw_pred(Xs, W, Xg, h).reshape(len(x1g), len(x2g))

    paper_style()
    fig = support_figure(
        X, Y, Wg, x1g, x2g, tau=tau,
        xlabel=columns[0], ylabel=columns[1],
        title_left="Observed data")
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path} (tau={tau:.3f})")


# ---------- main ----------

def main():
    ensure_dirs()
    t0 = time.time()
    out = {}
    base = pd.read_csv(TEMP / "baseline_preds.csv")

    for tag, loader in (("turtle", load_turtle),):
        d = loader()
        X, Y, m = d["X"], d["Y"], d["m"]
        cols = d.get("columns", ["cov1", "cov2"])
        pos_all = Y > 0

        # ---- per-fold predictions ----
        yhat_l, rhat_l, zhat_l, sig_l, y_l, m_l, ins_l = [], [], [], [], [], [], []
        for j, te in enumerate(folds_of(len(Y)), 1):
            tr = np.setdiff1d(np.arange(len(Y)), te)
            mu, sd = X[tr].mean(0), X[tr].std(0)
            Xs_tr_raw, Xs_te_raw = (X[tr] - mu) / sd, (X[te] - mu) / sd
            Xs_tr = np.column_stack([Xs_tr_raw, np.ones(len(Xs_tr_raw))])
            Xs_te = np.column_stack([Xs_te_raw, np.ones(len(Xs_te_raw))])

            theta = fit_debiased(Xs_tr, Y[tr], m[tr])
            W = corrected_target(Xs_tr, Y[tr], m[tr], theta)
            h = nw_bw(Xs_tr, W, hs=HS_REAL, n_sub=1500)
            K = np.exp(-((Xs_tr[:, None, :] - Xs_tr[None, :, :]) ** 2).sum(-1)
                       / (2 * h * h))
            np.fill_diagonal(K, 0)
            Dl = (K @ W) / np.maximum(K.sum(1), EPS)
            tau = 0.5
            D = nw_pred(Xs_tr, W, Xs_te, h)
            inside = D > tau
            s = sig(Xs_te @ theta)
            r = 1 - (1 - s) ** m[te]

            yhat_l.append(s * inside)
            rhat_l.append(r * D)
            zhat_l.append(D)
            sig_l.append(s)
            ins_l.append(inside)
            y_l.append(Y[te])
            m_l.append(m[te])
            print(f"  {tag} fold {j} ({time.time()-t0:.0f}s)", flush=True)

        y = np.concatenate(y_l)
        w = np.concatenate(m_l)
        pos = y > 0
        yhat = np.concatenate(yhat_l)
        rhat = np.concatenate(rhat_l)
        zhat = np.concatenate(zhat_l)
        sig_raw = np.concatenate(sig_l)
        ins = np.concatenate(ins_l)

        # ---- full-fit for figure + interpretation ----
        Xs_full_raw = (X - X.mean(0)) / X.std(0)
        Xs_full = np.column_stack([Xs_full_raw, np.ones(len(Xs_full_raw))])
        th_full = fit_debiased(Xs_full, Y, m)
        W_full = corrected_target(Xs_full, Y, m, th_full)
        h_full = nw_bw(Xs_full, W_full, hs=HS_REAL, n_sub=1500)
        Kf = np.exp(-((Xs_full[:, None, :] - Xs_full[None, :, :]) ** 2).sum(-1)
                    / (2 * h_full * h_full))
        np.fill_diagonal(Kf, 0)
        Dl_full = (Kf @ W_full) / np.maximum(Kf.sum(1), EPS)
        tau_full = 0.5

        # ---- metrics ----
        b = base[base.dataset == tag]
        bl = {
            "hurdle": {
                "wmse": np.concatenate([b[(b.fold == j) & (b.method == "hurdle")].p.values for j in range(1, 6)]),
                "auc_score": np.concatenate([b[(b.fold == j) & (b.method == "hurdle")].score.values for j in range(1, 6)]),
            },
            "zib": {
                "wmse": np.concatenate([b[(b.fold == j) & (b.method == "zib")].p.values for j in range(1, 6)]),
                "auc_score": np.concatenate([b[(b.fold == j) & (b.method == "zib")].score.values for j in range(1, 6)]),
            },
            "zoib": {
                "wmse": np.concatenate([b[(b.fold == j) & (b.method == "zoib")].p.values for j in range(1, 6)]),
                "auc_score": np.concatenate([b[(b.fold == j) & (b.method == "zoib")].score.values for j in range(1, 6)]),
            },
        }

        res = {}
        # wMSE / MAE (mean prediction)
        res["ours"] = {"wmse": wmse(y, yhat, w), "mae": mae(y, yhat)}
        res["debiased"] = {"wmse": wmse(y, sig_raw, w),
                           "mae": mae(y, sig_raw)}
        for meth in bl:
            res[meth] = {"wmse": wmse(y, bl[meth]["wmse"], w),
                         "mae": mae(y, bl[meth]["wmse"])}

        # AUC (weighted)
        res["ours"]["auc"] = wauc(rhat, pos, w)
        res["debiased"]["auc"] = wauc(sig_raw, pos, w)
        for meth in bl:
            res[meth]["auc"] = wauc(bl[meth]["auc_score"], pos, w)

        # MCC (weighted)
        res["ours"]["mcc"] = wmcc(rhat, pos, w)
        res["debiased"]["mcc"] = wmcc(sig_raw, pos, w)
        for meth in bl:
            res[meth]["mcc"] = wmcc(bl[meth]["auc_score"], pos, w)

        # fold SEs
        fl = [len(f) for f in folds_of(len(Y))]
        idx = np.cumsum([0] + fl)
        for meth in ("ours", "debiased"):
            p_w = yhat if meth == "ours" else sig_raw
            p_a = rhat if meth == "ours" else sig_raw
            p_m = rhat if meth == "ours" else sig_raw
            ma = [mae(y[idx[i]:idx[i+1]], p_w[idx[i]:idx[i+1]]) for i in range(5)]
            as_ = [wauc(p_a[idx[i]:idx[i+1]], pos[idx[i]:idx[i+1]], w[idx[i]:idx[i+1]]) for i in range(5)]
            ms = [wmcc(p_m[idx[i]:idx[i+1]], pos[idx[i]:idx[i+1]], w[idx[i]:idx[i+1]]) for i in range(5)]
            res[meth]["mae_se"] = float(np.std(ma, ddof=1) / np.sqrt(5))
            res[meth]["auc_se"] = float(np.std(as_, ddof=1) / np.sqrt(5))
            res[meth]["mcc_se"] = float(np.std(ms, ddof=1) / np.sqrt(5))

        # diagnostic (cross-fitted: zhat is out-of-fold)
        res["_diag"] = {}
        for q in (0.05, 0.10, 0.20):
            thr = np.quantile(zhat, q)
            res["_diag"][f"T{int(q*100)}"] = int(pos[zhat <= thr].sum())
            res["_diag"][f"n{int(q*100)}"] = int((zhat <= thr).sum())

        # theta
        res["_theta"] = dict(zip(d.get("columns", ["x1", "x2"]) + ["intercept"],
                                  [round(v, 3) for v in th_full]))

        out[tag] = res
        print(f"\n--- {tag} ---")
        print(f"  theta: {res['_theta']}")
        print(f"  {'method':18s} {'MAE':>8} {'AUC':>7} {'MCC':>7}")
        for meth in ("ours", "debiased", "hurdle", "zib", "zoib"):
            r = res[meth]
            se_a = f"({r['auc_se']:.3f})" if 'auc_se' in r else ""
            se_m = f"({r['mcc_se']:.3f})" if 'mcc_se' in r else ""
            print(f"  {meth:18s} {r['mae']:>8.4f} "
                  f"{r['auc']:>7.3f}{se_a} {r['mcc']:>7.3f}{se_m}")
        print(f"  diag: {res['_diag']}")
        print()

    (TEMP / "turtle_cv.json").write_text(json.dumps(out, indent=2))
    print(f"Done ({time.time()-t0:.0f}s). saved -> {TEMP / 'turtle_cv.json'}")


if __name__ == "__main__":
    main()
