"""Simulate a borrower-month panel with irregular income and behaviour drift.

IMPORTANT: this is SYNTHETIC data. Real borrower histories are private, so the
simulator exists to let the project demonstrate drift detection and adaptation
end to end with a known ground truth. To use real data, produce a table with the
same columns (see FEATURES_FULL in config.py + `month` + `default`) and skip this
step.

What changes at SHOCK_MONTH (the drift we want to detect):
  * covariate drift : gig and seasonal workers earn less and have more zero-income
                      months; credit utilisation rises for everyone.
  * concept drift   : the same income volatility / utilisation now carries MORE
                      default risk than before (the relationship P(default | X)
                      itself changes), so a model trained earlier is miscalibrated.
"""
import os

import numpy as np
import pandas as pd

from config import (N_BORROWERS, N_MONTHS, SHOCK_MONTH, SEED, DATA_DIR,
                    FIRST_SNAPSHOT)
from src.features import build_feature_table

SEGMENTS = ["salaried", "gig", "seasonal"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate(seed=SEED):
    rng = np.random.default_rng(seed)
    n, m = N_BORROWERS, N_MONTHS
    months = np.arange(m)

    segment = rng.choice(SEGMENTS, size=n, p=[0.50, 0.35, 0.15])
    base_income = np.exp(rng.normal(10.3, 0.45, n))          # ~ 30k median per month
    credit_score = np.clip(rng.normal(680, 60, n), 450, 850)
    emi = base_income * np.clip(rng.lognormal(np.log(0.22), 0.35, n), 0.05, 0.7)
    frailty = rng.normal(0, 0.45, n)                          # hidden borrower risk
    util_mean = np.clip(rng.beta(2.2, 3.5, n), 0.02, 0.95)
    phase = rng.uniform(0, 2 * np.pi, n)

    income = np.zeros((n, m))
    util = np.zeros((n, m))
    pay_ratio = np.zeros((n, m))

    is_sal = segment == "salaried"
    is_gig = segment == "gig"
    is_sea = segment == "seasonal"

    u_prev = util_mean.copy()
    for t in months:
        shock = t >= SHOCK_MONTH

        noise_sal = 1 + rng.normal(0, 0.03, n)
        noise_gig = rng.lognormal(0, 0.35, n)
        noise_sea = np.clip(1 + rng.normal(0, 0.15, n), 0.3, None)
        season = 1 + 0.5 * np.sin(2 * np.pi * t / 12 + phase)

        inc = np.where(is_sal, base_income * noise_sal,
              np.where(is_gig, base_income * noise_gig,
                       base_income * season * noise_sea))

        gap_p = np.where(is_sal, 0.005, np.where(is_gig, 0.06, 0.03))
        if shock:
            inc = np.where(is_gig, inc * 0.80, np.where(is_sea, inc * 0.90, inc))
            gap_p = np.where(is_gig, 0.12, np.where(is_sea, 0.06, 0.01))
        gap = rng.random(n) < gap_p
        inc = np.where(gap, base_income * 0.1, inc)
        income[:, t] = inc

        u = 0.8 * u_prev + 0.2 * util_mean + rng.normal(0, 0.03, n)
        if shock:
            u = u + 0.012                       # utilisation creeps up each month
        u = np.clip(u, 0.0, 1.0)
        util[:, t] = u
        u_prev = u

        available = inc - 0.6 * inc             # income left after living costs
        pr = available / np.maximum(emi, 1.0) + rng.normal(0, 0.08, n)
        pay_ratio[:, t] = np.clip(pr, 0.0, 1.0)

    snaps, F = build_feature_table(income, pay_ratio, util, credit_score, emi)

    rows = []
    for j, t in enumerate(snaps):
        shock = t >= SHOCK_MONTH
        cv = F["income_cv"][:, j]
        u = F["utilisation"][:, j]
        dti = F["dti"][:, j]
        missed = F["missed_3m"][:, j]
        gaps = F["gap_months"][:, j]
        score = F["credit_score"][:, j]
        trend = F["income_trend"][:, j]

        # Concept drift: coefficients on volatility, utilisation and gaps grow after the shock
        w_cv = 4.0 if shock else 2.0
        w_util = 2.4 if shock else 1.2
        w_gap = 0.55 if shock else 0.30
        intercept = -4.15 + (0.35 if shock else 0.0)

        z = (intercept
             + 3.0 * (dti - 0.22)
             + w_cv * (cv - 0.20)
             + w_gap * gaps
             + 0.55 * missed
             + w_util * (u - 0.35)
             - 1.2 * trend
             - 0.010 * (score - 680)
             + frailty)
        p = _sigmoid(z)
        y = (rng.random(n) < p).astype(int)

        df = pd.DataFrame({k: v[:, j] for k, v in F.items()})
        df["borrower_id"] = np.arange(n)
        df["segment"] = segment
        df["month"] = t
        df["default"] = y
        df["true_pd"] = p
        rows.append(df)

    panel = pd.concat(rows, ignore_index=True)
    return panel


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    panel = simulate()
    path = os.path.join(DATA_DIR, "borrower_panel.csv")
    panel.to_csv(path, index=False)
    by_phase = panel.assign(phase=np.where(panel.month >= SHOCK_MONTH, "after shock", "before shock"))
    print(f"Saved {len(panel):,} rows -> {path}")
    print(by_phase.groupby("phase")["default"].mean().round(4).to_string())
    print(panel.groupby("segment")["default"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
