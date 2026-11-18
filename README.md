# Behaviour Drift Modelling for Credit Risk under Irregular Income

A credit risk pipeline for borrowers whose income is irregular (gig workers,
seasonal workers), plus a monitoring layer that detects when the world changes
and the model goes stale. Includes a small Flask web app (login, prediction,
drift monitor page).

> **Data note:** real borrower histories are private, so this project runs on a
> **simulated** borrower-month panel (3,000 borrowers x 36 months) with a known,
> deliberately injected shock. Results below demonstrate that the pipeline and
> monitors work; they are **not** claims about performance on real lending data.
> To use real data, provide a table with the same columns (see `config.py`) and
> skip `src/simulate.py`.

## The problem

Most credit models see one income number. For irregular earners, the *pattern*
matters: volatility, months with almost no income, trend. On top of that, the
relationship between those patterns and default can shift (economic shock, new
lending behaviour), so a model trained once slowly becomes wrong.

## What is in here

| Part | File | What it does |
|---|---|---|
| Simulator | `src/simulate.py` | Salaried / gig / seasonal borrowers; at month 24 a shock lowers gig and seasonal income, raises utilisation, and makes volatility and utilisation **more** predictive of default (covariate + concept drift) |
| Features | `src/features.py` | Income mean, volatility (std/mean), low-income months, trend, latest/average income, EMI/income, payment behaviour, utilisation, score. Same code is used in training and in the web app |
| Drift detection | `src/drift.py` | PSI and KS test per feature, Bonferroni-corrected KS alerting |
| Training + evaluation | `src/train.py` | Strict time-based split, static vs adaptive models, monitoring, charts, production model |
| Web app | `app.py` | Register / login, prediction with main factors and out-of-range warnings, drift monitor page |

## Method

- **Time-based split, never random.** Train on months 5-17, validate on 18-23, test on 24-33 (after the shock).
- **Label delay is respected.** The label is "default within 3 months", so a month's label is only known 3 months later. The adaptive model is retrained every month using only labels that would have been available at that time (rolling 9-month window).
- **Static model:** trained once on months 5-23, never updated.
- **Monitors compared:** PSI on features, PSI on model scores, KS tests on features, and actual-vs-predicted default rate (needs labels).

## Results (simulated data)

Default rate: **3.4%** before the shock, **11.2%** after.

**1. Irregular-income features help.** Validation AUC: 0.692 with basic features
(income, score, EMI) vs 0.784 with the full feature set. *Caveat: the simulator
generates defaults from these behaviours, so this gap is expected by construction;
on real data the size of the gain would need to be measured.* A logistic model
(0.784) matched gradient boosting (0.782), as expected when the simulated
relationship is close to logistic.

**2. The shock showed up in calibration, not in ranking.** After the shock the
static model's AUC did not fall (mean 0.866); it stayed good at ranking borrowers.
But it badly under-predicted risk: it predicted 5.1% default on average while the
actual rate was 11.1%.

**3. Adaptive retraining recovers calibration, with a lag.** Mean Brier score
(lower is better) improved from 0.0817 (static) to 0.0735 (adaptive). In the last
3 months the adaptive model predicted 10.6% vs 11.8% actual, while the static
model was still at 5.5%. AUC gain was small (0.871 vs 0.866 mean). The adaptive
model needed several months to catch up because new labels arrive 3 months late.

**4. What detected the drift.**

| Monitor | Needs labels | Outcome |
|---|---|---|
| KS test on features (Bonferroni) | No | Significant from **month 24**, the first shock month; no false alarms before it |
| PSI on features (warn 0.10, alert 0.25) | No | Never warned; maximum 0.098 |
| PSI on model scores | No | Never warned; maximum 0.088 |
| Actual vs predicted default rate | Yes | Gap visible only from **month 27** (3-month label delay) |

The standard PSI rule-of-thumb thresholds did not fire on this gradual shift, while
a significance test did. Thresholds were not tuned to change this outcome.
Input-only monitors say that *inputs moved*; only label-based monitoring confirms
that *risk estimates are wrong*, so both are needed.

Charts are in `results/`: `calibration_over_time.png`, `auc_over_time.png`,
`psi_heatmap.png`; raw numbers in `monthly_metrics.csv`, `psi_by_month.csv`,
`ks_by_month.csv`, `summary.json`.

## Run it

```bash
python -m venv venv
# Windows: venv\Scripts\activate     Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

python -m src.train     # simulates data (if missing), trains, monitors, saves model.pkl + results/
python app.py           # http://127.0.0.1:5050
```

Run commands from the project root. If you change `requirements`, re-run
`python -m src.train` so `model.pkl` matches your scikit-learn version.

## Limitations

- Simulated data; no real-world validation.
- Rows from the same borrower are correlated across months (normal for behaviour scoring, but metrics are optimistic compared with a borrower-level holdout).
- Defaulted borrowers are not removed from later months (simplification).
- The web app has no CSRF protection and uses SQLite; suitable for a demo, not production.
- Risk bands (Low < 5%, Medium < 12%, High) are illustrative cut-offs.

## Possible next steps

Run the pipeline on a public credit dataset, add a borrower-level holdout, add a
labelled-drift detector (e.g. ADWIN on prediction errors), calibrate probabilities
(isotonic / Platt) after each retrain.
