import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cb_data as D, cb_models as M

A = "data/admin_clean.csv"
T = "data/transactions_clean.csv"

print("loading ...", flush=True)
panel, nb = D.load_attributed(A, T)
total, by_cat, grid = D.build_series(panel)
anom, thr = D.anomaly_labels(total)
micro = D.micro_aggregate(panel, grid)
design, FEAT, REG = D.build_design(total, micro, anom)

CAL_IDX = ["month_of_year", "quarter", "is_jan", "covid", "trend", "working_days",
           "step_2022", "post2022_ramp"]
YLAG  = [f"y_l{L}" for L in D.STD_LAGS]                     # y_l6, y_l12
MICRO = [f for f in FEAT if f not in CAL_IDX and f not in YLAG]
print(f"\n#CAL_IDX={len(CAL_IDX)}  #YLAG={len(YLAG)}  #MICRO={len(MICRO)}")
print("MICRO:", MICRO)

sets = {
    "Y+CAL":       CAL_IDX + YLAG,
    "Y+CAL+MICRO": CAL_IDX + YLAG + MICRO,
    "MICRO+CAL":   CAL_IDX + MICRO,
}

origins = None  # fe_cv builds its own expanding origins on the reduced design
scored = {}
for tag, cols in sets.items():
    fc = M.fe_cv(design, cols, origins, normalize=False)
    sc = M.score(fc, "C", "T2", total, anom)          # per-origin WAPE/MASE, exclusion
    scored[tag] = sc
    no = sc["origin_date"].nunique()
    print(f"\n=== {tag}  (#features={len(cols)}, #origins={no}) ===", flush=True)
    agg = sc.groupby("model").agg(WAPE=("WAPE", "mean"), MASE=("MASE", "mean"),
                                  n=("WAPE", "count")).round(4)
    print(agg.to_string())

# ---- per-learner delta and DM test: Y+CAL  vs  Y+CAL+MICRO ------------------
print("\n\n################  ISOLATION: adding microdata to Y+CAL  ################")
base = scored["Y+CAL"]; full = scored["Y+CAL+MICRO"]
for mdl in ["LASSO_FEA", "XGB_FEA", "LGBM_FEA"]:
    b = base[base.model == mdl][["origin_date", "WAPE"]].rename(columns={"WAPE": "wape_yonly"})
    f = full[full.model == mdl][["origin_date", "WAPE"]].rename(columns={"WAPE": "wape_micro"})
    j = b.merge(f, on="origin_date").sort_values("origin_date")
    l1 = j["wape_yonly"].values; l2 = j["wape_micro"].values   # loss per origin
    d_pp = (l2.mean() - l1.mean()) * 100                        # +ve = micro WORSE
    z, p = M.dm(l1, l2)                                         # HLN-corrected DM on WAPE loss
    sign = "micro WORSE" if d_pp > 0 else "micro better"
    print(f"{mdl:10s}: Y+CAL WAPE={l1.mean()*100:5.2f}%  +MICRO WAPE={l2.mean()*100:5.2f}%  "
          f"Δ={d_pp:+.2f}pp ({sign})  DM z={z:+.2f} p={p:.3f}  (n={len(j)} origins)")

# ---- classical ARIMAX(y; exog=lagged micro) vs plain SARIMA ----------------
print("\n\n################  CLASSICAL: SARIMA vs SARIMAX(+micro)  ################")
import statsmodels.api as sm
d = design.dropna(subset=(CAL_IDX + YLAG + MICRO) + ["y", "working_days"]).reset_index(drop=True)
dates = pd.to_datetime(d["month"])
n = len(d)
org = M.expanding_origins(n)
EXOG = ["n_recipients_l6", "mean_payment_l6", "new_recipients_l6", "exiting_recipients_l7"]
anom_ts = set(pd.to_datetime(list(anom)))
rows = []
for o in org:
    tr = d.iloc[:o]; te = d.iloc[o:o + D.H]
    if len(te) < D.H:
        continue
    ytr = tr["y"].values.astype(float)
    odate = dates.iloc[o - 1]
    for tag, exog_cols in [("SARIMA", None), ("SARIMAX+micro", EXOG)]:
        try:
            ex_tr = tr[exog_cols].values.astype(float) if exog_cols else None
            ex_te = te[exog_cols].values.astype(float) if exog_cols else None
            mod = sm.tsa.statespace.SARIMAX(ytr, exog=ex_tr, order=(1, 0, 0),
                                            seasonal_order=(1, 0, 0, 12),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            res = mod.fit(disp=0)
            fc = res.forecast(steps=D.H, exog=ex_te)
        except Exception as e:
            fc = np.full(D.H, np.nan)
        for k in range(D.H):
            rows.append((odate, tag, k + 1, dates.iloc[o + k], te["y"].values[k], fc[k]))
fcx = pd.DataFrame(rows, columns=["origin_date", "model", "h", "ds", "actual", "forecast"])
scx = M.score(fcx, "ARIMAX", "T2", total, anom)
aggx = scx.groupby("model").agg(WAPE=("WAPE", "mean"), MASE=("MASE", "mean"),
                                n=("WAPE", "count")).round(4)
print(aggx.to_string())
b = scx[scx.model == "SARIMA"][["origin_date", "WAPE"]].rename(columns={"WAPE": "w0"})
f = scx[scx.model == "SARIMAX+micro"][["origin_date", "WAPE"]].rename(columns={"WAPE": "w1"})
j = b.merge(f, on="origin_date")
if len(j):
    d_pp = (j["w1"].mean() - j["w0"].mean()) * 100
    z, p = M.dm(j["w0"].values, j["w1"].values)
    print(f"\nSARIMA WAPE={j.w0.mean()*100:.2f}%  SARIMAX+micro WAPE={j.w1.mean()*100:.2f}%  "
          f"Δ={d_pp:+.2f}pp  DM z={z:+.2f} p={p:.3f}  (n={len(j)})")
print("\nDONE")
