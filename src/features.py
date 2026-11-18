"""Feature engineering for irregular-income borrowers.

The same function is used for training (on simulated history) and inside the
Flask app (on the 6 monthly incomes a user types in), so there is no
training/serving mismatch.
"""
import numpy as np

from config import WINDOW


def build_features(incomes, payment_ratios, utilisation, credit_score, emi):
    """Return a dict of model features for ONE borrower snapshot.

    incomes         : last WINDOW monthly incomes, oldest first
    payment_ratios  : last 3+ monthly "share of EMI actually paid" values (0..1)
    utilisation     : current credit utilisation (0..1)
    """
    inc = np.asarray(incomes, dtype=float)
    if inc.shape[0] != WINDOW:
        raise ValueError(f"Need exactly {WINDOW} monthly incomes, got {inc.shape[0]}")

    mean = float(inc.mean())
    mean_safe = max(mean, 1.0)
    cv = float(inc.std() / mean_safe)
    median = float(np.median(inc))
    gap_months = int((inc < 0.5 * max(median, 1.0)).sum())

    x = np.arange(WINDOW)
    slope = float(np.polyfit(x, inc, 1)[0])
    income_trend = slope / mean_safe          # change per month, relative to the mean

    pr = np.asarray(payment_ratios, dtype=float)[-3:]
    payment_ratio_3m = float(pr.mean())
    missed_3m = int((pr < 0.5).sum())

    return {
        "income_mean": mean,
        "income_cv": cv,
        "gap_months": gap_months,
        "income_trend": income_trend,
        "last_income_ratio": float(inc[-1] / mean_safe),
        "dti": float(emi / mean_safe),
        "payment_ratio_3m": payment_ratio_3m,
        "missed_3m": missed_3m,
        "utilisation": float(utilisation),
        "credit_score": float(credit_score),
        "emi": float(emi),
    }


def build_feature_table(income, pay_ratio, util, score, emi):
    """Vectorised version for the whole simulated panel.

    income, pay_ratio, util : arrays of shape (n_borrowers, n_months)
    score, emi              : arrays of shape (n_borrowers,)
    Returns a dict of arrays shaped (n_borrowers, n_snapshots) for snapshot
    months WINDOW-1 .. n_months-1.
    """
    n, m = income.shape
    snaps = list(range(WINDOW - 1, m))
    cols = {k: [] for k in [
        "income_mean", "income_cv", "gap_months", "income_trend", "last_income_ratio",
        "dti", "payment_ratio_3m", "missed_3m", "utilisation", "credit_score", "emi"]}

    x = np.arange(WINDOW, dtype=float)
    xc = x - x.mean()
    denom = (xc ** 2).sum()

    for t in snaps:
        w = income[:, t - WINDOW + 1: t + 1]
        mean = w.mean(axis=1)
        mean_safe = np.maximum(mean, 1.0)
        cols["income_mean"].append(mean)
        cols["income_cv"].append(w.std(axis=1) / mean_safe)
        med = np.maximum(np.median(w, axis=1), 1.0)
        cols["gap_months"].append((w < 0.5 * med[:, None]).sum(axis=1))
        slope = (w * xc).sum(axis=1) / denom
        cols["income_trend"].append(slope / mean_safe)
        cols["last_income_ratio"].append(w[:, -1] / mean_safe)
        cols["dti"].append(emi / mean_safe)
        p3 = pay_ratio[:, t - 2: t + 1]
        cols["payment_ratio_3m"].append(p3.mean(axis=1))
        cols["missed_3m"].append((p3 < 0.5).sum(axis=1))
        cols["utilisation"].append(util[:, t])
        cols["credit_score"].append(score)
        cols["emi"].append(emi)

    return snaps, {k: np.stack(v, axis=1) for k, v in cols.items()}
