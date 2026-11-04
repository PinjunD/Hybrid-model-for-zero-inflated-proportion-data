"""Shared configuration for all numerical-study scripts.

Single source of truth for the simulation grid, RNG seeds, the
real-data bandwidth candidates, and output locations. Entry scripts insert the
hybrid_model_for_zero_inflated_proportion_data root on sys.path and then import this module
top-level, e.g. ``from config import MS, sim_seed``.

The arithmetic in :func:`sim_seed` is the reproducibility contract
behind every cached JSON in results/temp: keep it byte-identical.
"""
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent        # hybrid_model_for_zero_inflated_proportion_data/
TEMP = PKG / "results" / "temp"              # JSON outputs
FIGD = PKG / "results" / "figure"            # figure outputs

# Simulation grid (shared by the theta and support studies).
MS = [5, 10, 15]                      # binomial trials
NS = [200, 400, 600, 800, 1000]       # sample sizes (theta study)
NTAB = 1000                           # sample size of the tables
CASES = [1, 2, 3]                      # support configurations
NREP = 300                             # replications per configuration

# Bandwidth candidates for real-data kernel LOOCV (the simulation
# uses the theoretical rate of Theorem 3 instead; see
# algorithm/kernel_support.py::bw_theory).
HS_REAL = np.geomspace(0.1, 5.0, 15)


def sim_seed(sd, n, m):
    """Seed of replication ``sd`` at sample size ``n`` and trials ``m``."""
    return sd * 1000 + n * 7 + m


def ensure_dirs():
    """Create the output directories if they are missing."""
    TEMP.mkdir(parents=True, exist_ok=True)
    FIGD.mkdir(parents=True, exist_ok=True)
