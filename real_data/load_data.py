"""Load the turtle dataset (the trawl loader lives in full_sample.py).

Each loader returns a dict with the design matrix X, the outcome
proportion Y, the trial count m, and the covariate names.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


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
