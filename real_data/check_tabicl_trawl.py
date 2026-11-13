"""Reproduce the TabICLv2 row of Table 4 (trawl full sample) standalone.

Runs exactly the code path of `full_sample.py --with-tabicl`
(load_full_trawl -> folds_of -> tabicl_rows -> metrics_with_se) without
touching the cached full_sample.json. Results go to stdout and to
results/temp/full_sample_r/tabicl_trawl_check.json (gitignored).

Run from the repository root:
    python real_data/check_tabicl_trawl.py
"""
import json
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "real_data"))

import numpy as np

from full_sample import load_full_trawl, metrics_with_se
from run_experiment import folds_of, tabicl_rows

t0 = time.time()
loader = load_full_trawl()
Y, m = loader["Y"], loader["m"]
folds = folds_of(len(Y))
print(f"n={len(Y):,} zeros={float((Y == 0).mean()):.3f} "
      f"({time.time() - t0:.0f}s)", flush=True)

tab = tabicl_rows("trawl", load_full_trawl)

y_order = np.concatenate([Y[f] for f in folds])
w_order = np.concatenate([m[f] for f in folds])
assert np.allclose(y_order, tab.y.values), "row order mismatch"
res = metrics_with_se(y_order, w_order, tab.p.values, tab.score.values, folds)

print("\nTabICLv2 trawl (this run):")
print(json.dumps(res, indent=2))
print("paper Table 4 row        : MAE .0233 (.000)  AUC .870 (.004)  "
      "MCC .435 (.012)")
print(f"total {time.time() - t0:.0f}s")

out = PKG / "results" / "temp" / "full_sample_r"
out.mkdir(exist_ok=True)
(out / "tabicl_trawl_check.json").write_text(json.dumps(res, indent=2))
