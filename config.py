"""Central settings for the behaviour-drift credit risk project."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DB_PATH = os.path.join(BASE_DIR, "database.db")

SEED = 42

# ---- simulation ----
N_BORROWERS = 3000
N_MONTHS = 36          # months 0..35
SHOCK_MONTH = 24       # macro shock: income stress + behaviour change starts here

# ---- features ----
WINDOW = 6             # months of history used to build features
LABEL_HORIZON = 3      # label = default within the next 3 months
                       # => a snapshot's label is only known 3 months later

# ---- time-based split (by snapshot month) ----
FIRST_SNAPSHOT = WINDOW - 1   # first month with a full 6-month history (5)
TRAIN_END = 17                # train:      months 5-17   (stable period)
VALID_END = 23                # validation: months 18-23  (still stable)
TEST_END = 33                 # test:       months 24-33  (after the shock)

# ---- drift monitoring ----
PSI_WARN = 0.10
PSI_ALERT = 0.25
ADAPTIVE_WINDOW = 9           # adaptive model retrains on the latest 9 labelled months

FEATURES_BASIC = ["income_mean", "credit_score", "emi"]

FEATURES_FULL = [
    "income_mean",
    "income_cv",
    "gap_months",
    "income_trend",
    "last_income_ratio",
    "dti",
    "payment_ratio_3m",
    "missed_3m",
    "utilisation",
    "credit_score",
    "emi",
]

FEATURE_LABELS = {
    "income_mean": "Average monthly income (6m)",
    "income_cv": "Income volatility (std/mean)",
    "gap_months": "Low-income months (of 6)",
    "income_trend": "Income trend",
    "last_income_ratio": "Latest income / average",
    "dti": "EMI / average income",
    "payment_ratio_3m": "Share of EMI paid (3m)",
    "missed_3m": "Missed payments (3m)",
    "utilisation": "Credit utilisation",
    "credit_score": "Credit score",
    "emi": "Monthly EMI",
}
