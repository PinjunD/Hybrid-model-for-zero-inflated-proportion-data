"""Load the two real datasets.

Each loader returns a dict with the design matrix X, the outcome
proportion Y, the trial count m, and the covariate names.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def load_trawl():
    """NEUS bottom trawl: cod share of the catch, 50 <= m <= 500.

    Source: Dryad doi:10.5061/dryad.bg08269 (Flanagan et al., 2018),
    aggregated to one row per haul.
    """
    g = pd.read_csv(DATA / "trawl.csv")
    keep = g.btemp.notna() & g.btemp.gt(-2) \
        & g.n_total.between(50, 500)
    g = g[keep]
    return {
        "X": g[["btemp", "depth"]].values,
        "Y": (g.k_cod / g.n_total).values,
        "m": g.n_total.values.astype(float),
        "columns": ["bottom temperature (C)", "depth (m)"],
    }


def load_turtle():
    """Loggerhead nests: hatching proportion, Ossabaw Island.

    Source: Georgia Southern University Digital Commons
    (Carroll et al., 2022).
    """
    tu = pd.read_csv(DATA / "turtle.csv")
    tu = tu.dropna(subset=["AvgTemp", "Elevation", "Hatched"])
    return {
        "X": tu[["AvgTemp", "Elevation"]].values,
        "Y": (tu.Hatched / tu.Eggs_in_Clutch).values,
        "m": tu.Eggs_in_Clutch.values.astype(float),
        "columns": ["nest temperature (C)", "elevation (m)"],
    }
