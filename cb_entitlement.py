"""


Writes to outputs/r3/.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import cb_data as D
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoETS, AutoARIMA, AutoTheta, ARIMA

OUT = "outputs/r3"
CATS = ["K1", "K2", "K3"]
START, END = pd.Timestamp("2017-06-01"), pd.Timestamp("2025-12-01")


def build_entitlement(path_admin):
    """Monthly accrual series: each positive decision contributes its monthly
    `amount` to every month in [right_from, effective_end] for its category;
    point-in-time dedupe keeps the most recent decision per (applicant, category,
    month) so renewals are not double-counted. Leakage-free in construction."""
    a = pd.read_csv(path_admin, low_memory=False)
    for c in ["right_from", "right_to", "document_date", "right_termination_date"]:
        a[c] = pd.to_datetime(a[c], errors="coerce")
    pos = a[(a.document_type == "Decision - positive") & (a.benefit_category.isin(CATS))
            & a.right_from.notna() & a.right_to.notna() & a.amount.notna()].copy()
    pos["end"] = pos["right_to"]
    m = pos["right_termination_date"].notna()
    pos.loc[m, "end"] = pos[["right_to", "right_termination_date"]].min(axis=1)[m]
    pos["start_m"] = pos["right_from"].values.astype("datetime64[M]")
    pos["end_m"] = pos["end"].values.astype("datetime64[M]")
    pos = pos[pos["end_m"] >= pos["start_m"]]
    pos["nmo"] = ((pos["end_m"].dt.year - pos["start_m"].dt.year) * 12
                  + (pos["end_m"].dt.month - pos["start_m"].dt.month) + 1).clip(upper=36)
    idx = np.repeat(pos.index.values, pos["nmo"].values)
    off = np.concatenate([np.arange(k) for k in pos["nmo"].values])
    ex = pos.loc[idx, ["applicant_id", "benefit_category", "amount", "document_date", "start_m"]].copy()
    ex["month"] = (ex["start_m"].values.astype("datetime64[M]")
                   + off.astype("timedelta64[M]")).astype("datetime64[ns]")
    ex = ex[(ex["month"] >= START) & (ex["month"] <= END)]
    ex = ex.sort_values("document_date").drop_duplicates(
        ["applicant_id", "benefit_category", "month"], keep="last")
    ent = ex.groupby(["month", "benefit_category"])["amount"].sum().rename("entitlement").reset_index()
    grid = pd.date_range(START, END, freq="MS")
    tot = ent.groupby("month")["entitlement"].sum().reindex(grid).fillna(0.0)
    return tot, ent


def _wape(a, f): a, f = np.asarray(a, float), np.asarray(f, float); return np.nansum(np.abs(a - f)) / np.nansum(np.abs(a))
def _mase(a, f, tr, m=12): sc = np.nanmean(np.abs(tr[m:] - tr[:-m])); return np.nanmean(np.abs(a - f)) / sc


def forecast_series(y, dates, label):
    n = len(y); H = D.H; NW = (n - H) - D.INIT_WINDOW + 1
    df = pd.DataFrame({"unique_id": "s", "ds": dates, "y": np.asarray(y, float)})
    sf = StatsForecast(models=[SeasonalNaive(season_length=12), AutoETS(season_length=12),
                               AutoARIMA(season_length=12), AutoTheta(season_length=12),
                               ARIMA(order=(1, 0, 0), alias="AR1")], freq="MS", n_jobs=1)
    cv = sf.cross_validation(df=df, h=H, step_size=1, n_windows=NW, refit=True)
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    cols = {"SNAIVE": "SeasonalNaive", "ETS": "AutoETS", "ARIMA": "AutoARIMA", "THETA": "AutoTheta", "AR1": "AR1"}
    yv = np.asarray(y, float); dts = pd.to_datetime(dates).values
    rows = []
    for nm, c in cols.items():
        per = [(_wape(g["y"], g[c]), _mase(g["y"].values, g[c].values, yv[dts <= np.datetime64(od)]))
               for od, g in cv.groupby("cutoff")]
        per = np.array(per)
        rows.append(dict(series=label, model=nm, WAPE=per[:, 0].mean(), MASE=per[:, 1].mean()))
    return pd.DataFrame(rows).sort_values("WAPE")


def main():
    os.makedirs(OUT, exist_ok=True)
    A, T = "data/admin_clean.csv", "data/transactions_clean.csv"
    ent, ent_cat = build_entitlement(A)
    panel, nb = D.load_attributed(A, T); cash, by_cat, grid = D.build_series(panel)
    cash_s = cash.set_index(pd.to_datetime(cash["month"]))["total_amount"].reindex(ent.index)

    calib = ent.iloc[:60].values; nm = np.median(calib[calib > np.quantile(calib, 0.25)])
    thr = 0.05 * nm; ent_anom = sorted(ent[ent < thr].index)
    anom, _ = D.anomaly_labels(cash)
    print(f"cash near-zero months: {len(anom)} | entitlement near-zero months: {len(ent_anom)}")
    print(f"CV cash={cash_s.std()/cash_s.mean():.3f}  CV entitlement={ent.std()/ent.mean():.3f}")

    cmp = pd.DataFrame({"month": ent.index, "cash": cash_s.values, "entitlement": ent.values})
    cmp.to_csv(f"{OUT}/entitlement_vs_cash.csv", index=False)

    acc = pd.concat([forecast_series(ent.values, ent.index, "entitlement"),
                     forecast_series(cash_s.values, cash_s.index, "cash_all_months")], ignore_index=True)
    acc.to_csv(f"{OUT}/entitlement_forecast.csv", index=False)
    print("\n=== forecast accuracy ===")
    print(acc.assign(WAPE=lambda d: (d.WAPE * 100).round(1), MASE=lambda d: d.MASE.round(3)).to_string(index=False))


if __name__ == "__main__":
    main()
