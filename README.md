# Hybrid model for zero-inflated proportion data

Code for the numerical study of the paper "A hybrid model for
zero-inflated proportion data": the simulation (Table 2, Figures
C.1-C.2 in the supplementary material) and the two real-data analyses
(Table 3, Figures 1-2). Every number and figure in the paper is
reproduced bit-for-bit by the four entry scripts below.

## 1. What the code does

The model is `mY | X ~ Bin(m, sigma(X'theta) * I_S(X))`: a binomial
regression for suitable covariates, truncated by a deterministic
unsuitable region S*. The code implements the three components of the
estimation procedure:

- **Coefficient** -- the debiased MLE, a Newton solver on the
  reweighted conditional likelihood of the positive observations
  (`algorithm/estimators.py::fit_debiased`), together with the naive,
  uncorrected, and oracle benchmarks used in Table 2.
- **Support** -- the corrected outcome
  `Z = I(Y>0) / [1-(1-sigma(X'theta_hat))^m]`, smoothed by a
  Nadaraya-Watson kernel and thresholded at tau = 1/2
  (`algorithm/kernel_support.py`). The bandwidth is the rate-optimal
  `(log n / n)^(1/3)` of Theorem 3 in the simulation and LOO-CV on the
  real data.
- **Model check** -- the count of positive observations in the bottom
  5/10/20 percent of the out-of-fold fitted corrected outcome; a count
  near zero supports the structural-zero premise.

**Structure.**

```
data/               the two analysis datasets (trawl.csv, turtle.csv)
algorithm/          model, estimators, kernel support, metrics
simulation/         dgp.py + run_simulation.py
real_data/          load_data.py, run_experiment.py, rerun_all.py, verify_tabicl.py
results/            figures and cached JSON (reproducible outputs)
```

**Entry scripts** (run from this directory; Python 3.9+ with
numpy/scipy/pandas/matplotlib/imageio, R with glmmTMB + gamlss,
`pip install tabicl==2.0.2` for the TabICLv2 rows):

```bash
python simulation/run_simulation.py     # Table 2 + Figures C.1-C.2 + GIFs (~8 min)
python real_data/run_experiment.py      # folds + R baselines (hurdle/ZIB/ZOIB) (~1 min)
python real_data/rerun_all.py           # Table 3 (ours/baselines) + Figures 1-2 (~1 min)
python real_data/verify_tabicl.py       # Table 3, TabICLv2 rows (~2 min)
```

**Data.** `data/trawl.csv` (34,608 hauls x 4 columns: `btemp`,
`depth`, `n_total`, `k_cod`) comes from the NEUS bottom-trawl survey
on Dryad (doi:10.5061/dryad.bg08269; Flanagan et al., 2018); the
analysis keeps hauls with non-missing `btemp` and
50 <= `n_total` <= 500, leaving n = 3,367 hauls (69% zeros).
`data/turtle.csv` (170 nests x 4 columns) comes from the Georgia
Southern University Digital Commons (Carroll et al., 2022); 164 nests
have complete covariates.

## 2. The numerical results in brief

**Simulation** (n = 1000, 300 replications, m in {5,10,15}, tau = 1/2,
theoretical bandwidth; Table 2). For the coefficient: our estimator is
best in most configurations. Where S* is bounded (Case 1), sigma is
bounded away from zero, the debiasing weights approach one as m grows,
and the uncorrected estimator matches us; where S* is unbounded
(Case 2), sigma is arbitrarily close to zero and the truncation bias
never disappears, so we lead by a factor of 3 to 11; with no structural
zeros (Case 3), the naive estimator coincides with the oracle and ours
is second best. For the support: the two outcomes are nearly identical
in Case 1, while ours is 1.2 to 4.1 times more accurate in Cases 2-3.
The convergence figures (C.1, C.2) confirm the decay with the effective
sample size M = nm.

**Real data** (five-fold CV, same folds for every method; Table 3).
On the trawl survey the procedure recovers the curved thermal front of
the cod habitat, the model check supports it (only 1, 8, and 35 of the
169, 337, and 674 hauls in the bottom 5/10/20 percent of the fitted
corrected outcome contain cod), and prediction is strong: the best MAE
(.0442), with AUC/MCC (.788/.347) within one standard error of the
tabular foundation model TabICLv2 (.793/.358) and far ahead of the
parametric baselines (.664/at most .085). On the turtle nests -- a
small sample (n = 164) with a nearly linear boundary -- our method
again attains the best MAE (.2116) and stays within one standard error
of TabICLv2 on AUC/MCC, clearly ahead of the parametric baselines.

## 3. Notes

- **Determinism.** All seeds are fixed (`config.sim_seed`; folds via
  `default_rng(42)`), so every run reproduces the cached JSON in
  `results/temp` and the paper numbers bit-for-bit on the same package
  versions.
- **Implementation details.** (i) All methods -- ours, TabICLv2, and
  the R baselines -- fit on covariates standardized within each
  training fold (the R side in the embedded script of
  `run_experiment.py`); for logistic-type models this is a
  reparametrization and does not change fitted values. (ii) The
  bandwidth LOO-CV on the real data runs on a training subsample of at
  most 1,500 observations (`nw_bw`, argument `n_sub`); the turtle
  folds (131 nests) are used in full. (iii) Two numerical-stability
  guards are inactive on our data: the corrected-outcome denominator
  is floored at 0.05 (smallest fitted value 0.448), and the Newton
  solver adds a 1e-10 ridge (same argmax).
- **Why the three parametric baselines share one AUC.** Not a bug: the
  zero-indicator part of the likelihood is the same Bernoulli logit in
  the hurdle and ZOIB models (and neither dataset has y = 1, so the
  ZOIB one-inflation part is absent), so their fitted gates coincide;
  the ZIB gate is a different estimator but ranks almost identically
  (Spearman 0.999 and 1 with the hurdle gate). The AUC depends only on
  the ranking. Their mean predictions differ, which is what separates
  them in the MAE.
- **TabICLv2 version.** `tabicl >= 2.0.0` ships the TabICLv2
  checkpoints; the paper reports TabICLv2 (Qu et al., 2026) through
  version 2.0.2.
