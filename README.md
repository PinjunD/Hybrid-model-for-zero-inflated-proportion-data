# Hybrid model for zero-inflated proportion data

Code for the numerical study of "A hybrid model for zero-inflated
proportion data". Model: `mY | X ~ Bin(m, sigma(X'theta) * I_S(X))`.

## How to use

Python 3.9+ (numpy, pandas, matplotlib), R with jsonlite + glmmTMB +
gamlss (`Rscript` on the PATH), and `pip install tabicl==2.0.2`. Run
from this directory:

```bash
python simulation/run_simulation.py    # Tables 2-3 + Figures 2-3          (~8 min)
python simulation/redraw_figs.py       # re-style Figures 2-3 from cache   (instant)
python real_data/run_experiment.py     # folds + R baselines (turtle)      (~1 min)
python real_data/rerun_all.py          # Table 5 (turtle CV)               (~1 min)
python real_data/full_sample.py        # Table 4 (trawl full-sample CV)    (~2.5 min)
python real_data/figures.py            # Figures 1 and 4 (full-sample fits)(~1 min)
python real_data/verify_tabicl.py      # TabICLv2 rows, Table 5            (~2 min)
```

`full_sample.py --with-tabicl` adds the TabICLv2 row of Table 4
(~69 min). The cached JSON in `results/temp/` holds every number in
the paper's tables.

## Framework

```
data/            trawl.csv, turtle.csv          # the two analysis datasets
algorithm/       model, estimators, kernel_support, metrics, plotting
simulation/      dgp, run_simulation,           # Tables 2-3, Figures 2-3
                 redraw_figs
real_data/       load_data, run_experiment,     # Tables 4-5, Figures 1, 4
                 rerun_all, full_sample,
                 figures, verify_tabicl
results/         figure/ (paper figures), temp/ (cached JSON, reproducible)
```

| Piece | Where |
|---|---|
| Debiased MLE (Newton) | `algorithm/estimators.py::fit_debiased` |
| Corrected outcome + NW support, tau = 1/2 | `algorithm/kernel_support.py` |
| Bandwidth: `(log n/n)^(1/3)` (simulation), LOO-CV (real data) | `kernel_support.py::bw_theory` / `nw_bw` |
| Model check (bottom 5/10/20% counts, out-of-fold) | `real_data/rerun_all.py`, `real_data/full_sample.py` |

## Results

**Simulation** — n = 1000, 300 replications, m in {5,10,15}, tau = 1/2,
theoretical bandwidth (Tables 2-3).

| ![theta convergence](results/figure/png/fig_sim_theta.png) | ![support convergence](results/figure/png/fig_sim_support_conv.png) |
|---|---|
| *Left: mean squared error of the coefficient estimators vs M = nm; ours tracks the oracle and beats the uncorrected MLE wherever sigma is small (Cases 2-3).* | *Right: support error vs n; ours is 1.2-4.1 times more accurate in Cases 2-3.* |

**Real data** — five-fold cross-validation, same folds for every
method (Tables 4-5).

| ![trawl region](results/figure/png/fig_trawl_full.png) | ![turtle region](results/figure/png/fig_turtle_support.png) |
|---|---|
| *Trawl (n = 30,131, full sample): the recovered cod habitat traces the curved thermal front; the model check supports it (0, 4, 87 cod-containing hauls in the bottom 5/10/20%).* | *Turtle nests (n = 164): small sample, nearly linear boundary; best MAE on both datasets, AUC/MCC within one SE of TabICLv2.* |

| dataset | best MAE | ours | TabICLv2 |
|---|---|---|---|
| trawl  | ours .0167 | AUC .842 / MCC .375 | .492 / -.020 |
| turtle | ours .2116 | AUC .837 / MCC .456 | .845 / .472 |

On the full trawl sample m spans six orders of magnitude, so the
positivity probability is driven by m: our score models this dependence
explicitly and dominates, while positivity scores without m (the
parametric gates, TabICLv2) fall to the chance level (AUC ~ .49).

## Notes

- Seeds fixed (`config.sim_seed`, `default_rng(42)`): runs reproduce
  the cached JSON and the paper numbers bit-for-bit.
- Trawl full sample: hauls with observed `btemp` (> -2 C) and `depth`
  and `n_total >= 1` (n = 30,131); turtle nests with complete
  covariates (n = 164 of 170 recorded).
- All methods fit on fold-standardized covariates (R side included);
  bandwidth LOO-CV subsamples to at most 1,500 points; two stability
  guards (denominator floor 0.05, Newton ridge 1e-10) never activate.
- The three parametric baselines share one AUC by construction (same
  zero-indicator logit; no y = 1 in either dataset); their MAEs differ.
- `tabicl >= 2.0.0` ships TabICLv2 checkpoints; the paper uses 2.0.2.
