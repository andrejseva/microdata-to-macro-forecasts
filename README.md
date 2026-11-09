# Microdata-to-macro forecasting of social-protection spending

Code for the study *From Microdata to Macro Forecasts: A Comparative Study of
Aggregation Strategies and Benchmark Models in Social Protection Spending*.

It forecasts monthly child-benefit expenditure from a short aggregate series
backed by transaction-level administrative microdata, under a strict
leakage-controlled, expanding-window design. Every predictor is constructed to
be observable at the forecast origin.

## What it does

- **Four aggregation strategies:** direct-aggregate (A), bottom-up (B),
  feature-enriched with microdata (C), and MinT reconciliation (D), plus a
  structural strategy (E) and transparent benchmarks (seasonal-dummy
  regression, forecast combinations).
- **Four anomaly treatments** for the recurring near-zero reset months: naive
  (T1), exclusion (T2), calendar adjustment (T3), regime-aware (T4), plus a
  deterministic January rule (TJAN).
- **Benchmarks and ML models:** seasonal naive, ETS, ARIMA, Theta, AR(1),
  seasonal AR(1)(1), LASSO, XGBoost, LightGBM.
- **Statistical comparison:** Diebold–Mariano (HLN correction, HAC variance,
  Holm multiplicity), Model Confidence Set, Friedman–Nemenyi, and a TOST
  equivalence check, all on the per-origin WAPE.

## Files

| File | Purpose |
|------|---------|
| `cb_data.py` | Data layer: point-in-time category attribution, monthly series, leakage-free anomaly labels, feature-enriched design, predictor audit. |
| `cb_models.py` | Forecasting models, cross-validation, metrics (WAPE, MASE, MAPE, RMSSE), MinT reconciliation, and the statistical tests (DM, MCS, Friedman–Nemenyi, TOST, Holm). |
| `cb_run.py` | Main runner: produces the headline accuracy tables and the statistical-comparison outputs. |
| `cb_robustness.py` | Additional robustness experiments: leaky Strategy C, structural Strategy E, ML target/transform variants, accuracy by horizon, cumulative error, interval coverage. |
| `cb_isolation.py` | Same-learner isolation of the marginal value of the microdata (with/without the lagged microdata features, seed-controlled). |
| `cb_entitlement.py` | Entitlement-basis (accrual) series reconstruction and its forecast accuracy. |
| `cb_figures.py` | Regenerates the figures from the saved result CSVs. |

## Data

The scripts expect two anonymised input files:

```
data/admin_clean.csv          # decision records (applicant id, category, decision dates, attributes)
data/transactions_clean.csv   # individual payment records (applicant id, amount, date, category)
```

The anonymised monthly aggregate and category series are included in this
repository. The transaction-level microdata are held under a data-sharing
agreement with the Public Child Protection Fund of Republika Srpska and cannot
be redistributed.

## Running

```bash
pip install -r requirements.txt

python cb_run.py data/admin_clean.csv data/transactions_clean.csv outputs
python cb_robustness.py          # writes to outputs/r3
python cb_isolation.py           # marginal-value-of-microdata experiment
python cb_entitlement.py         # accrual-series analysis
python cb_figures.py             # figures from the saved CSVs
```

Outputs (accuracy tables, DM/MCS/Friedman results, figures) are written under
`outputs/`.

## Notes

- The evaluation is expanding-window with a 60-month minimum training window,
  horizon `H = 6`, and 38 origins (26 for the feature-enriched Strategy C,
  which needs a full year of microdata as a seasonal warm-up).
- For any linear method the bottom-up sum equals the direct aggregate by
  construction, so identical A/B rows are an algebraic identity.
- The internal strategy key `A_TD` is kept for continuity with earlier output
  files; the display label is "direct-aggregate".
