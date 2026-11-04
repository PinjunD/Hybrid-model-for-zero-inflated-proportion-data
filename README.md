# hybrid_model_for_zero_inflated_proportion_data

Numerical study for the zero-inflated binomial support paper.

## Data

Two CSV files, one per dataset, both loaded by `real_data/load_data.py`:

**Trawl** -- `data/trawl.csv` (34,608 hauls x 4 columns):

| column    | meaning                                 |
|-----------|-----------------------------------------|
| `btemp`   | bottom temperature (deg C)              |
| `depth`   | depth (m)                               |
| `n_total` | total catch in numbers (the trials $m$) |
| `k_cod`   | Atlantic cod in the catch               |

Source: Dryad doi:10.5061/dryad.bg08269 (Flanagan et al., 2018), one row
per haul. The analysis keeps hauls with non-missing `btemp` and
50 <= `n_total` <= 500, leaving n = 3,367 hauls (69% zeros).

**Turtle** -- `data/turtle.csv` (170 nests x 4 columns: `AvgTemp`,
`Elevation`, `Hatched`, `Eggs_in_Clutch`), from the Georgia Southern
University Digital Commons (Carroll et al., 2022); 164 nests have
complete covariates.

## Structure

```
data/               real datasets (trawl.csv, turtle.csv)
algorithm/          core methods
  model             sigmoid, binomial log-density
  estimators        theta-hat via Newton-Raphson (debiased/full/pos/oracle)
  kernel_support    corrected outcome, NW kernel, bw_theory (Theorem 3)
  metrics           AUC, AUPRC, theta error
simulation/         Section 5.1
  dgp               three support configurations
  run_simulation    single entry: Tables 2-3, Figures 2-3, both GIFs
real_data/          Section 5.2
  load_data         trawl + turtle loaders (data/ inside this repo)
  run_experiment    folds + R baselines (hurdle/ZIB/ZOIB) + TabICL folds
  rerun_all         ours rows of Tables 4-5, diagnostics, support figures
  verify_tabicl     TabICL rows of Tables 4-5
results/            generated outputs
  temp/             cached JSON/CSV (reproducible)
  figure/           all figures
```

## Reproduce

From this directory (Python 3.9+ with numpy/scipy/pandas/matplotlib/imageio;
R with glmmTMB + gamlss; `pip install tabicl==2.0.2` for the TabICLv2 rows --
note tabicl >= 2.0.0 ships the TabICLv2 checkpoints cited in the paper):

```bash
# Simulation: Tables 2-3 at n=1000 (300 reps), Figures 2-3, GIFs  (~8 min)
python simulation/run_simulation.py

# Real data: regenerate folds + R baselines + TabICL inference (~3 min)
python real_data/run_experiment.py

# Real data: ours/baseline rows of Tables 4-5, support figures (~1 min)
python real_data/rerun_all.py

# Real data: TabICL rows of Tables 4-5 (~2 min)
python real_data/verify_tabicl.py
```

Outputs land in `results/temp` (JSON) and `results/figure` (PDF/GIF).
`simulation/regen_support_conv_fig.py` redraws Figure 3 from the cached
JSON without recomputation.

## Settings (final, as in the manuscript)

- Simulation grid: m in {5,10,15}, n in {200,...,1000}, 300 replications,
  tables reported at n=1000.
- Threshold tau = 1/2 everywhere (simulation and real data); nothing is tuned.
- Simulation bandwidth: h_n = (log n / n)^{1/3}, the rate-optimal choice of
  Theorem 3 (alpha=1, d=2). Real data bandwidth: leave-one-out CV.
- Positivity score (AUC/MCC): pi_i = [1-(1-sigma(X_i'theta-hat))^m_i] Z-hat(X_i).

## Paper-to-output map

| Paper item | Output |
|---|---|
| Table 1 (cases) | manual |
| Table 2 (theta) | run_simulation.py stdout / results/temp/sim_theta.json |
| Table 3 (support) | run_simulation.py stdout / results/temp/sim_support_conv.json |
| Figure 1 (trawl) | results/figure/fig_trawl_support.pdf |
| Figure 2 (theta conv.) | results/figure/fig_sim_theta.pdf |
| Figure 3 (support conv.) | results/figure/fig_sim_support_conv.pdf |
| Figure 4 (turtle) | results/figure/fig_turtle_support.pdf |
| Table 4 (trawl) | rerun_all_intercept.json + verify_tabicl.py (trawl) |
| Table 5 (turtle) | rerun_all_intercept.json + verify_tabicl.py (turtle) |

## Remarks (implementation details)

The following details make the paper's descriptions exact:

1. **Standardized covariates for every method.** All methods --
   ours, TabICLv2, and the R baselines (hurdle, ZIB, ZOIB) -- fit on
   covariates standardized within each training fold, matching the
   paper's statement that all methods share the same folds and the
   same standardized covariates. For the R baselines this happens in
   the embedded R script (`real_data/run_experiment.py`); for logistic
   -type models the standardization is a reparametrization and does
   not change fitted values.

2. **Bandwidth LOO-CV subsampling.** On the real data the
   leave-one-out cross-validation for the kernel bandwidth runs on a
   training subsample of at most 1,500 observations
   (`algorithm/kernel_support.py::nw_bw`, argument `n_sub`); the turtle
   training folds (131 nests) are used in full.

3. **Numerical-stability guards.** Two small deviations from the
   paper's formulas are guards that are inactive on our data: the
   denominator of the corrected outcome is floored at 0.05 (the
   smallest fitted value is 0.448 on the trawl data and 1.0 on the
   turtle data, so the floor never binds), and the Newton solver adds
   a 1e-10 ridge to the Hessian (same argmax, stable steps).

## Determinism

All seeds are fixed (`config.sim_seed`; folds via `default_rng(42)`),
so every run reproduces the manuscript numbers bit-for-bit on the same
package versions.
