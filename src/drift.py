"""Drift detection utilities: PSI and KS per feature, plus a monthly PSI table."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

_EPS = 1e-4


def psi(reference, current, bins=10):
    """Population Stability Index between a reference and a current sample.

    Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 major shift.
    Bin edges come from the REFERENCE sample only (quantiles), so the comparison
    is always against what the model saw in training.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    uniq = np.unique(ref)
    if uniq.size <= bins:                      # discrete feature (e.g. missed payments)
        edges = np.concatenate(([-np.inf], (uniq[:-1] + uniq[1:]) / 2, [np.inf]))
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf

    r = np.histogram(ref, bins=edges)[0] / ref.size
    c = np.histogram(cur, bins=edges)[0] / cur.size
    r = np.clip(r, _EPS, None)
    c = np.clip(c, _EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def monthly_drift_table(panel, features, ref_months, months):
    """PSI and KS statistic of every feature, for every month in `months`,
    against the pooled reference months. Returns two DataFrames (features x months)."""
    ref = panel[panel["month"].isin(ref_months)]
    psi_rows, ks_rows, p_rows = {}, {}, {}
    for f in features:
        psi_rows[f], ks_rows[f], p_rows[f] = {}, {}, {}
        for m in months:
            cur = panel.loc[panel["month"] == m, f]
            psi_rows[f][m] = psi(ref[f], cur)
            res = ks_2samp(ref[f], cur)
            ks_rows[f][m] = float(res.statistic)
            p_rows[f][m] = float(res.pvalue)
    return (pd.DataFrame(psi_rows).T, pd.DataFrame(ks_rows).T,
            pd.DataFrame(p_rows).T)


def first_ks_alert_month(p_table, alpha=0.01):
    """First month in which any feature's KS test is significant, using a
    Bonferroni correction for the number of tests (features x months) so that
    a handful of false alarms are not expected by chance."""
    corrected = alpha / p_table.size
    for m in p_table.columns:
        if (p_table[m] < corrected).any():
            return int(m)
    return None


def first_alert_month(psi_table, threshold):
    """First month in which ANY feature crosses the PSI threshold (or None)."""
    for m in psi_table.columns:
        if (psi_table[m] > threshold).any():
            return int(m)
    return None


def feature_ranges(df, features, lo=0.01, hi=0.99):
    """1st-99th percentile of each feature - used by the app to warn when an
    input is outside the range the model was trained on."""
    return {f: [float(df[f].quantile(lo)), float(df[f].quantile(hi))] for f in features}
