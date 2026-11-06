"""Fit the external baselines for the real-data analysis.

Exports the fold structure and design matrices, fits the parametric
baselines (hurdle, ZIB, ZOIB) through an embedded R script, and writes
results/temp/baseline_preds.csv together with folds.json. The TabICL
rows of the paper's tables are produced by verify_tabicl.py, which
imports tabicl_rows below. Run from hybrid_model_for_zero_inflated_proportion_data/:

    python real_data/run_experiment.py
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

import numpy as np
import pandas as pd

from config import TEMP, ensure_dirs
from load_data import load_turtle

RSCRIPT = r"C:\Program Files\R\R-4.6.1\bin\x64\Rscript.exe"


def folds_of(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(n), k)


R_CODE = r"""
suppressMessages({library(jsonlite); library(glmmTMB); library(gamlss)})
tmp <- "RESULTS_TEMP"
folds <- fromJSON(paste(readLines(file.path(tmp, "folds.json")),
                        collapse = ""), simplifyVector = FALSE)
all <- list()
for (tag in c("trawl", "turtle")) {
  df <- read.csv(file.path(tmp, paste0(tag, ".csv")))
  fl <- folds[[tag]]; n <- nrow(df); out <- list()
  for (j in seq_len(5)) {
    te <- unlist(fl[[j]]) + 1
    tr <- setdiff(seq_len(n), te)
    dtr <- df[tr, ]; dte <- df[te, ]
    mtr <- dtr$m
    z <- as.integer(dtr$y > 0)
    for (v in c("x1", "x2")) {
      mu <- mean(dtr[[v]]); s <- sd(dtr[[v]])
      dtr[[v]] <- (dtr[[v]] - mu) / s
      dte[[v]] <- (dte[[v]] - mu) / s
    }

    f1 <- glm(y ~ x1 + x2, family = quasibinomial(), data = dtr)
    p1 <- predict(f1, dte, type = "response")

    g1 <- glm(z ~ x1 + x2, family = binomial(), data = dtr)
    g2 <- glm(y ~ x1 + x2, family = quasibinomial(),
              data = dtr[dtr$y > 0, ], weights = mtr[dtr$y > 0])
    h1 <- predict(g1, dte, type = "response")
    h2 <- predict(g2, dte, type = "response")

    fit <- glmmTMB(cbind(round(mtr * dtr$y), round(mtr * (1 - dtr$y)))
                   ~ x1 + x2, zi = ~ x1 + x2,
                   family = binomial(), data = dtr)
    zp <- predict(fit, dte, type = "zprob")
    pr <- predict(fit, dte, type = "response")

    bb <- try(gamlss(y ~ x1 + x2, sigma.formula = ~ 1,
                     nu.formula = ~ x1 + x2, tau.formula = ~ 1,
                     family = BEINF(), data = dtr, trace = FALSE),
              silent = TRUE)
    if (inherits(bb, "try-error"))
      bb <- gamlss(y ~ x1 + x2, sigma.formula = ~ 1, nu.formula = ~ 1,
                   tau.formula = ~ 1, family = BEINF(), data = dtr,
                   trace = FALSE)
    nu_te <- predict(bb, newdata = dte, what = "nu", type = "response")
    tau_te <- predict(bb, newdata = dte, what = "tau",
                      type = "response")
    mu_te <- predict(bb, newdata = dte, what = "mu", type = "response")
    p0 <- nu_te / (1 + nu_te + tau_te)
    pm <- (mu_te + tau_te) / (1 + nu_te + tau_te)

    for (m in c("fractional", "hurdle", "zib", "zoib")) {
      out[[length(out) + 1]] <- data.frame(
        dataset = tag, fold = j, method = m,
        z = as.integer(dte$y > 0), y = dte$y,
        score = switch(m, fractional = p1, hurdle = h1,
                       zib = 1 - zp, zoib = 1 - p0),
        p = switch(m, fractional = p1, hurdle = h1 * h2,
                   zib = pr, zoib = pm))
    }
  }
  all[[tag]] <- do.call(rbind, out)
}
write.csv(do.call(rbind, all), file.path(tmp, "baseline_preds.csv"),
          row.names = FALSE)
"""

def fit_baselines():
    """Export data + folds, run R script, return baseline preds."""
    ensure_dirs()
    folds = {}
    for tag, loader in [("turtle", load_turtle)]:
        d = loader()
        X, Y, m = d["X"], d["Y"], d["m"]
        pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1],
                      "y": Y, "m": m}).to_csv(
            TEMP / f"{tag}.csv", index=False)
        folds[tag] = [[int(i) for i in f]
                      for f in folds_of(len(Y))]
    (TEMP / "folds.json").write_text(json.dumps(folds))

    src = (R_CODE.replace("RESULTS_TEMP", str(TEMP).replace("\\", "/"))
            .replace('for (tag in c("trawl", "turtle")) {',
                     'for (tag in c("turtle")) {'))
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(src)
        script = fh.name
    res = subprocess.run([RSCRIPT, script], capture_output=True,
                         text=True)
    Path(script).unlink()
    if res.returncode != 0:
        print(res.stderr[-2000:])
        raise SystemExit("R baseline fitting failed")
    df = pd.read_csv(TEMP / "baseline_preds.csv")
    df["hard"] = (df["score"] > 0.5).astype(int)
    return df

# ---------- TabICL ----------

def tabicl_rows(tag, loader):
    try:
        from tabicl import TabICLClassifier, TabICLRegressor
    except ImportError:
        print("tabicl not installed; skipping")
        return None
    d = loader()
    X, Y = d["X"], d["Y"]
    Z = (Y > 0).astype(int)
    rows = []
    for j, te in enumerate(folds_of(len(Y)), start=1):
        tr = np.setdiff1d(np.arange(len(Y)), te)
        mu, sd = X[tr].mean(0), X[tr].std(0)
        Xs_tr, Xs_te = (X[tr] - mu) / sd, (X[te] - mu) / sd
        t0 = time.time()
        clf = TabICLClassifier().fit(Xs_tr, Z[tr])
        score = clf.predict_proba(Xs_te)[:, 1]
        reg = TabICLRegressor().fit(Xs_tr, Y[tr])
        p = np.clip(reg.predict(Xs_te), 0, 1)
        hard = (score > 0.5).astype(int)
        rows.append(pd.DataFrame(dict(
            dataset=tag, fold=j, method="tabicl",
            z=Z[te], y=Y[te], score=score, hard=hard, p=p)))
        print(f"  TabICL {tag} fold {j} ({time.time() - t0:.0f}s)",
              flush=True)
    return pd.concat(rows)


if __name__ == "__main__":
    fit_baselines()
    print("Done. Saved results/temp/baseline_preds.csv and folds.json")
