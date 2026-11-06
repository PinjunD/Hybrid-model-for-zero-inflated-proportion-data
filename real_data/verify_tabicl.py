"""TabICLv2 rows for the turtle data (Table 5).

The trawl (full-sample) TabICLv2 rows are produced by
real_data/full_sample.py --with-tabicl.
"""
import sys
from pathlib import Path
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "real_data"))
import numpy as np

from load_data import load_turtle
from rerun_all import mae, wauc, wmcc
from run_experiment import folds_of, tabicl_rows

for tag, loader in (("turtle", load_turtle),):
    d = loader()
    Y, m = d["Y"], d["m"]
    pos = Y > 0
    tab = tabicl_rows(tag, loader)
    ms, ys, poss = [], [], []
    for te in folds_of(len(Y)):
        ms.append(m[te])
        ys.append(Y[te])
        poss.append(pos[te])
    y = np.concatenate(ys)
    w = np.concatenate(ms)
    p = np.concatenate(poss)
    assert np.allclose(y, tab.y.values), "row order mismatch"
    score = tab.score.values
    pred = tab.p.values
    fl = [len(f) for f in folds_of(len(Y))]
    idx = np.cumsum([0] + fl)
    fold_mae = [mae(y[idx[i]:idx[i+1]], pred[idx[i]:idx[i+1]]) for i in range(5)]
    fold_auc = [wauc(score[idx[i]:idx[i+1]], p[idx[i]:idx[i+1]], w[idx[i]:idx[i+1]]) for i in range(5)]
    fold_mcc = [wmcc(score[idx[i]:idx[i+1]], p[idx[i]:idx[i+1]], w[idx[i]:idx[i+1]]) for i in range(5)]
    se = lambda v: float(np.std(v, ddof=1) / np.sqrt(5))
    print(f"{tag}: MAE {mae(y, pred):.4f} ({se(fold_mae):.3f})  "
          f"AUC {wauc(score, p, w):.3f} ({se(fold_auc):.3f})  "
          f"MCC {wmcc(score, p, w):.3f} ({se(fold_mcc):.3f})")
