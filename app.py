"""Flask app: login, irregular-income risk prediction, drift monitoring page."""
import json
import os
import sqlite3

import joblib
import numpy as np
from flask import (Flask, render_template, request, redirect, session,
                   send_from_directory, abort)
from werkzeug.security import generate_password_hash, check_password_hash

import config as C
from src.features import build_features

app = Flask(__name__)
# Set SECRET_KEY as an environment variable in real use; the fallback is for local dev only
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")

if not os.path.exists(C.MODEL_PATH):
    raise SystemExit("model.pkl not found. Run `python -m src.train` first.")

bundle = joblib.load(C.MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
RANGES = bundle["feature_ranges"]

LOW_BAND, HIGH_BAND = 0.05, 0.12   # probability cut-offs for Low / Medium / High


# ----------------------------------------------------------------- database
def get_db():
    return sqlite3.connect(C.DB_PATH)


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT,
        income_mean REAL,
        income_cv REAL,
        credit_score INTEGER,
        emi REAL,
        pd REAL,
        result TEXT
    )""")
    conn.commit()
    conn.close()


init_db()


# ----------------------------------------------------------------- helpers
def risk_band(p):
    if p < LOW_BAND:
        return "Low Risk"
    if p < HIGH_BAND:
        return "Medium Risk"
    return "High Risk"


def parse_incomes(text):
    # accept commas (incl. full-width), semicolons, new lines, or plain spaces as separators
    for ch in ("，", "،", ";", "\n", "\r", "\t"):
        text = text.replace(ch, ",")
    if "," not in text:
        text = text.replace(" ", ",")
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    if len(parts) != C.WINDOW:
        raise ValueError(f"I found {len(parts)} value(s) in the incomes box, but need exactly "
                         f"{C.WINDOW} (one per month, oldest first, no commas inside numbers: "
                         f"write 32000, not 32,000).")
    values = [float(p) for p in parts]
    if any(v < 0 for v in values):
        raise ValueError("Incomes cannot be negative.")
    if sum(values) <= 0:
        raise ValueError("At least one month must have income above 0.")
    return values


def explain(x_row):
    """Top factors behind a prediction (only for the logistic model)."""
    if not hasattr(model, "named_steps"):
        return []
    scaler = model.named_steps["standardscaler"]
    lr = model.named_steps["logisticregression"]
    contrib = lr.coef_[0] * scaler.transform(x_row)[0]
    order = np.argsort(-np.abs(contrib))[:3]
    return [{"feature": C.FEATURE_LABELS[FEATURES[i]],
             "direction": "raises risk" if contrib[i] > 0 else "lowers risk",
             "strength": float(abs(contrib[i]))} for i in order]


def range_warnings(feats):
    out = []
    for name, (lo, hi) in RANGES.items():
        v = feats[name]
        if v < lo or v > hi:
            out.append(f"{C.FEATURE_LABELS[name]} = {v:,.2f} is outside the range the model "
                       f"was trained on ({lo:,.2f} to {hi:,.2f}); treat this prediction with caution.")
    return out


# ----------------------------------------------------------------- auth routes
@app.route("/")
def home():
    return redirect("/dashboard" if "user" in session else "/login")


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        raw = request.form.get("password", "")
        if not username or not raw:
            error = "Username and password are required."
        elif len(raw) < 6:
            error = "Password must be at least 6 characters."
        else:
            conn = get_db()
            try:
                conn.execute("INSERT INTO users(username,password) VALUES(?,?)",
                             (username, generate_password_hash(raw)))
                conn.commit()
            except sqlite3.IntegrityError:
                error = "That username is already taken."
            finally:
                conn.close()
            if error is None:
                return redirect("/login")
    return render_template("register.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session["user"] = username
            return redirect("/dashboard")
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")


# ----------------------------------------------------------------- prediction
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect("/login")

    result, error, warnings, form = None, None, [], {}

    if request.method == "POST":
        form = request.form
        try:
            incomes = parse_incomes(form.get("incomes", ""))
            credit_score = int(form["credit_score"])
            emi = float(form["emi"])
            util = float(form["utilisation"]) / 100.0
            missed = int(form["missed"])

            if not 300 <= credit_score <= 900:
                raise ValueError("Credit score must be between 300 and 900.")
            if emi <= 0:
                raise ValueError("EMI must be greater than 0.")
            if not 0 <= util <= 1:
                raise ValueError("Utilisation must be between 0 and 100 (%).")
            if not 0 <= missed <= 3:
                raise ValueError("Missed payments must be between 0 and 3.")
        except KeyError:
            error = "Please fill in every field."
        except ValueError as e:
            msg = str(e)
            error = msg if not msg.startswith("could not convert") and "invalid literal" not in msg \
                else "Please enter valid numbers in every field."

        if error is None:
            pay = [0.0] * missed + [1.0] * (3 - missed)     # missed EMIs -> 0% paid
            feats = build_features(incomes, pay, util, credit_score, emi)
            x = np.array([[feats[f] for f in FEATURES]])
            p = float(model.predict_proba(x)[0, 1])
            band = risk_band(p)
            result = {"pd": p, "band": band, "factors": explain(x),
                      "income_mean": feats["income_mean"], "income_cv": feats["income_cv"]}
            warnings = range_warnings(feats)

            conn = get_db()
            conn.execute(
                "INSERT INTO predictions(user,income_mean,income_cv,credit_score,emi,pd,result) "
                "VALUES(?,?,?,?,?,?,?)",
                (session["user"], feats["income_mean"], feats["income_cv"],
                 credit_score, emi, p, band))
            conn.commit()
            conn.close()

    conn = get_db()
    logs = conn.execute(
        "SELECT income_mean,income_cv,credit_score,emi,pd,result FROM predictions "
        "WHERE user=? ORDER BY id DESC LIMIT 20", (session["user"],)).fetchall()
    conn.close()

    return render_template("dashboard.html", result=result, error=error,
                           warnings=warnings, logs=logs, form=form,
                           window=C.WINDOW)


# ----------------------------------------------------------------- drift monitoring
@app.route("/drift")
def drift():
    if "user" not in session:
        return redirect("/login")
    path = os.path.join(C.RESULTS_DIR, "summary.json")
    if not os.path.exists(path):
        return render_template("drift.html", summary=None)
    with open(path) as fh:
        summary = json.load(fh)
    return render_template("drift.html", summary=summary,
                           trained_on=bundle["trained_on_months"],
                           shock=C.SHOCK_MONTH)


@app.route("/results/<name>")
def results_file(name):
    if "user" not in session:
        abort(403)
    if not name.endswith(".png"):
        abort(404)
    return send_from_directory(C.RESULTS_DIR, name)


if __name__ == "__main__":
    # debug=True is fine locally; never deploy with it on
    app.run(debug=True, port=5050)