"""Train, evaluate and monitor credit risk models under behaviour drift.

Run:  python -m src.train

What it does
  1. Loads (or simulates) the borrower-month panel.
  2. Splits strictly by TIME (never randomly): train -> validation -> post-shock test.
  3. Compares a basic feature set (income, score, EMI) with irregular-income features.
  4. Trains a STATIC model once, and an ADAPTIVE model that is retrained every month
     using only labels that would really be known by then (3-month label delay).
  5. Measures PSI/KS feature drift and monthly AUC / calibration after the shock.
  6. Saves metrics, tables, charts and the production model.
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from src.drift import (monthly_drift_table, first_alert_month, first_ks_alert_month,
                       feature_ranges, psi)
from src.simulate import simulate

# --- chart colours: reference categorical palette, slots 1-2 (see dataviz skill) ---
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))


def make_hgb():
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08,
                                          max_iter=150, random_state=C.SEED)


def months_df(panel, lo, hi):
    return panel[(panel["month"] >= lo) & (panel["month"] <= hi)]


def fit(make, df, features):
    m = make()
    m.fit(df[features], df["default"])
    return m


def score_month(model, df, features):
    p = model.predict_proba(df[features])[:, 1]
    y = df["default"].to_numpy()
    return {
        "auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "pred_rate": float(p.mean()),
        "obs_rate": float(y.mean()),
    }


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def line_chart(path, months, series, title, ylabel, shock_x, ylim=None, fmt=None):
    """series = [(label, values, colour, linestyle)]"""
    fig, ax = plt.subplots(figsize=(8.6, 4.9), facecolor=SURFACE)
    style_axes(ax)
    ax.axvline(shock_x, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    for label, vals, colour, ls in series:
        ax.plot(months, vals, color=colour, linewidth=2, linestyle=ls,
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    if ylim:
        ax.set_ylim(*ylim)
    ax.text(shock_x + 0.1, ax.get_ylim()[1], "shock starts", color=INK_2, fontsize=9, va="top")
    ax.set_xlabel("Month (snapshot date)", color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12, fontweight="bold")
    if fmt:
        ax.yaxis.set_major_formatter(fmt)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=len(series))
    ax.set_xticks(list(months))
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def psi_heatmap(path, psi_table):
    fig, ax = plt.subplots(figsize=(9.2, 5.0), facecolor=SURFACE)
    data = psi_table.copy()
    data.index = [C.FEATURE_LABELS[f] for f in data.index]
    im = ax.imshow(np.clip(data.values, 0, C.PSI_ALERT), aspect="auto", cmap="Blues",
                   vmin=0, vmax=C.PSI_ALERT)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, fontsize=8, color=INK_2)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index, fontsize=9, color=INK)
    ax.set_xlabel("Month (snapshot date)", color=INK_2, fontsize=9)
    ax.set_title("Feature drift (PSI) vs training period", color=INK, fontsize=12,
                 loc="left", pad=12, fontweight="bold")
    # mark cells above the alert threshold with a dot (not colour alone)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data.values[i, j] > C.PSI_ALERT:
                ax.plot(j, i, marker="o", markersize=4, color="white")
    shock_col = list(data.columns).index(C.SHOCK_MONTH) - 0.5
    ax.axvline(shock_col, color=ORANGE, linewidth=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(f"PSI (scale ends at the {C.PSI_ALERT} alert level; warn = {C.PSI_WARN})",
                 color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    csv = os.path.join(C.DATA_DIR, "borrower_panel.csv")
    if os.path.exists(csv):
        panel = pd.read_csv(csv)
    else:
        os.makedirs(C.DATA_DIR, exist_ok=True)
        panel = simulate()
        panel.to_csv(csv, index=False)

    F_full, F_basic = C.FEATURES_FULL, C.FEATURES_BASIC
    train = months_df(panel, C.FIRST_SNAPSHOT, C.TRAIN_END)
    valid = months_df(panel, C.TRAIN_END + 1, C.VALID_END)
    trainval = months_df(panel, C.FIRST_SNAPSHOT, C.VALID_END)

    # ------------------------------------------------------------ 1. model selection
    candidates = {
        "Logistic, basic features": (make_logreg, F_basic),
        "Logistic, irregular-income features": (make_logreg, F_full),
        "Gradient boosting, irregular-income features": (make_hgb, F_full),
    }
    selection = {}
    for name, (mk, feats) in candidates.items():
        m = fit(mk, train, feats)
        selection[name] = float(roc_auc_score(valid["default"], m.predict_proba(valid[feats])[:, 1]))
        print(f"validation AUC  {selection[name]:.4f}  {name}")

    best_name = max((n for n in selection if "irregular" in n), key=selection.get)
    best_make, best_feats = candidates[best_name]
    print(f"\nChosen model family: {best_name}")

    # ------------------------------------------------------------ 2. static vs adaptive after the shock
    static_model = fit(best_make, trainval, best_feats)           # trained once, never updated
    static_basic = fit(make_logreg, trainval, F_basic)            # the "old" 3-feature model

    # Label-free monitor: has the distribution of the model's SCORES moved?
    ref_scores = static_model.predict_proba(train[best_feats])[:, 1]

    rows = []
    for t in range(C.VALID_END + 1, C.TEST_END + 1):
        cur = panel[panel["month"] == t]

        s = score_month(static_model, cur, best_feats)
        b = score_month(static_basic, cur, F_basic)

        # Only labels known at month t: snapshot s is labelled once s + horizon <= t
        hi = t - C.LABEL_HORIZON
        lo = max(C.FIRST_SNAPSHOT, hi - C.ADAPTIVE_WINDOW + 1)
        adaptive = fit(best_make, months_df(panel, lo, hi), best_feats)
        a = score_month(adaptive, cur, best_feats)

        rows.append({
            "month": t,
            "static_auc": s["auc"], "adaptive_auc": a["auc"], "basic_auc": b["auc"],
            "static_brier": s["brier"], "adaptive_brier": a["brier"],
            "observed_rate": s["obs_rate"],
            "static_pred_rate": s["pred_rate"], "adaptive_pred_rate": a["pred_rate"],
            "score_psi": psi(ref_scores, static_model.predict_proba(cur[best_feats])[:, 1]),
            "adaptive_trained_on": f"{lo}-{hi}",
        })
    monthly = pd.DataFrame(rows)
    monthly.to_csv(os.path.join(C.RESULTS_DIR, "monthly_metrics.csv"), index=False)

    # ------------------------------------------------------------ 3. feature drift monitoring
    ref_months = list(range(C.FIRST_SNAPSHOT, C.TRAIN_END + 1))
    mon_months = list(range(C.TRAIN_END + 1, C.TEST_END + 1))
    psi_t, ks_t, ksp_t = monthly_drift_table(panel, F_full, ref_months, mon_months)
    psi_t.round(4).to_csv(os.path.join(C.RESULTS_DIR, "psi_by_month.csv"))
    ks_t.round(4).to_csv(os.path.join(C.RESULTS_DIR, "ks_by_month.csv"))

    ks_first = first_ks_alert_month(ksp_t)
    ks_pre_shock = first_ks_alert_month(ksp_t[[m for m in ksp_t.columns if m < C.SHOCK_MONTH]])
    pre_alert = first_alert_month(psi_t[[m for m in psi_t.columns if m < C.SHOCK_MONTH]], C.PSI_ALERT)
    first_psi_alert = first_alert_month(psi_t, C.PSI_ALERT)
    first_psi_warn = first_alert_month(psi_t, C.PSI_WARN)

    # performance monitoring: first month static model is clearly worse than on validation
    val_auc = selection[best_name]
    degr = monthly[monthly["static_auc"] < val_auc - 0.03]
    first_auc_drop = int(degr["month"].iloc[0]) if len(degr) else None
    calib = monthly[(monthly["observed_rate"] / monthly["static_pred_rate"]) > 1.5]
    first_calib_gap = int(calib["month"].iloc[0]) if len(calib) else None
    sw = monthly[monthly["score_psi"] > C.PSI_WARN]
    first_score_warn = int(sw["month"].iloc[0]) if len(sw) else None
    sa = monthly[monthly["score_psi"] > C.PSI_ALERT]
    first_score_alert = int(sa["month"].iloc[0]) if len(sa) else None

    # ------------------------------------------------------------ 4. charts
    months = monthly["month"].tolist()
    line_chart(os.path.join(C.RESULTS_DIR, "auc_over_time.png"), months,
               [("Static model (trained once)", monthly["static_auc"], BLUE, "-"),
                ("Adaptive model (retrained monthly)", monthly["adaptive_auc"], ORANGE, "-")],
               "Ranking quality (AUC) after the shock", "AUC", C.SHOCK_MONTH, ylim=(0.5, 1.0))
    pct = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    line_chart(os.path.join(C.RESULTS_DIR, "calibration_over_time.png"), months,
               [("Actual default rate", monthly["observed_rate"], INK, "-"),
                ("Static model: predicted", monthly["static_pred_rate"], BLUE, "-"),
                ("Adaptive model: predicted", monthly["adaptive_pred_rate"], ORANGE, "-")],
               "Predicted vs actual default rate", "Default rate", C.SHOCK_MONTH, ylim=(0, 0.14), fmt=pct)
    psi_heatmap(os.path.join(C.RESULTS_DIR, "psi_heatmap.png"), psi_t)

    # ------------------------------------------------------------ 5. production model
    # Retrain the chosen family on the latest labelled window and save it for the app.
    prod_lo = C.TEST_END - C.ADAPTIVE_WINDOW + 1
    prod_df = months_df(panel, prod_lo, C.TEST_END)
    prod_model = fit(best_make, prod_df, best_feats)
    joblib.dump({
        "model": prod_model,
        "features": best_feats,
        "model_name": best_name,
        "trained_on_months": [prod_lo, C.TEST_END],
        "feature_ranges": feature_ranges(prod_df, best_feats),
        "base_rate": float(prod_df["default"].mean()),
    }, C.MODEL_PATH)

    # ------------------------------------------------------------ 6. summary
    post = monthly
    summary = {
        "data": {
            "rows": int(len(panel)), "borrowers": int(panel["borrower_id"].nunique()),
            "default_rate_before_shock": float(panel[panel.month < C.SHOCK_MONTH]["default"].mean()),
            "default_rate_after_shock": float(panel[panel.month >= C.SHOCK_MONTH]["default"].mean()),
            "synthetic": True,
        },
        "validation_auc": selection,
        "chosen_model": best_name,
        "post_shock_mean": {
            "static_auc": float(post["static_auc"].mean()),
            "adaptive_auc": float(post["adaptive_auc"].mean()),
            "basic_features_auc": float(post["basic_auc"].mean()),
            "static_brier": float(post["static_brier"].mean()),
            "adaptive_brier": float(post["adaptive_brier"].mean()),
            "observed_rate": float(post["observed_rate"].mean()),
            "static_pred_rate": float(post["static_pred_rate"].mean()),
            "adaptive_pred_rate": float(post["adaptive_pred_rate"].mean()),
        },
        "last_3_months": {
            "static_auc": float(post["static_auc"].tail(3).mean()),
            "adaptive_auc": float(post["adaptive_auc"].tail(3).mean()),
            "observed_rate": float(post["observed_rate"].tail(3).mean()),
            "static_pred_rate": float(post["static_pred_rate"].tail(3).mean()),
            "adaptive_pred_rate": float(post["adaptive_pred_rate"].tail(3).mean()),
        },
        "monitoring": {
            "shock_month": C.SHOCK_MONTH,
            "psi_alert_before_shock_false_alarm_month": pre_alert,
            "first_psi_warn_month": first_psi_warn,
            "first_psi_alert_month": first_psi_alert,
            "first_month_static_auc_drops_0.03": first_auc_drop,
            "first_snapshot_month_actual_rate_1.5x_predicted": first_calib_gap,
            "that_gap_is_only_visible_from_month": (first_calib_gap + C.LABEL_HORIZON
                                                    if first_calib_gap is not None else None),
            "first_ks_significant_month_bonferroni": ks_first,
            "ks_false_alarm_before_shock_month": ks_pre_shock,
            "first_score_psi_warn_month": first_score_warn,
            "first_score_psi_alert_month": first_score_alert,
            "max_feature_psi_any_month": float(psi_t.values.max()),
            "max_score_psi_any_month": float(monthly["score_psi"].max()),
            "psi_thresholds": {"warn": C.PSI_WARN, "alert": C.PSI_ALERT},
        },
        "top_psi_last_month": psi_t[C.TEST_END].sort_values(ascending=False).head(5).round(3).to_dict(),
        "production_model_trained_on_months": [prod_lo, C.TEST_END],
    }
    with open(os.path.join(C.RESULTS_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== monthly post-shock ===")
    print(monthly.drop(columns=["adaptive_trained_on"]).round(3).to_string(index=False))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
