"""
Additional robustness experiments, isolated from the main pipeline.

Implements, all leakage-free unless explicitly the 'leaky' demonstration:
  * list of anomalous months + December->January timing context
  * LEAKY Strategy C (contemporaneous, un-lagged microdata) to quantify the
         leakage the leak-free design controls
  * C4a: Structural Strategy E = forecast beneficiaries per category x the
         per-recipient rate known at the origin, summed to the aggregate
  * ML variants on the aggregate -- log target, differenced target, direct
        multi-step, and a post-2022 subsample
  * accuracy by horizon, fiscal-year cumulative error, and 80/95% prediction-
        interval coverage for ETS/ARIMA

Writes everything to outputs/r3/ .
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import cb_data as D, cb_models as M
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA, SeasonalNaive

OUT = "outputs/r3"
H, W0 = D.H, D.INIT_WINDOW


def _wape(a, f):
    a = np.asarray(a, float); f = np.asarray(f, float)
    s = np.nansum(np.abs(a))
    return np.nan if s == 0 or not np.isfinite(f).all() else np.nansum(np.abs(a - f)) / s


def _mase(a, f, train, m=12):
    train = np.asarray(train, float)
    if len(train) <= m: return np.nan
    sc = np.nanmean(np.abs(train[m:] - train[:-m]))
    return np.nan if sc == 0 else np.nanmean(np.abs(np.asarray(a, float) - np.asarray(f, float))) / sc


# ---------------------------------------------------------------------------
def c2_anomaly_context(total, anom):
    am = sorted(pd.to_datetime(list(anom)))
    rows = []
    s = total.set_index(pd.to_datetime(total["month"]))["total_amount"]
    for m in am:
        prev = m - pd.DateOffset(months=1)
        rows.append(dict(month=m.strftime("%Y-%m"), is_january=(m.month == 1),
                         value=float(s.get(m, np.nan)),
                         prev_month_value=float(s.get(prev, np.nan)),
                         follows_spike=bool(s.get(prev, 0) > 1.5 * s.median())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def c4b_leaky_C(total, micro, anom, origins):
    """LEAKY Strategy C: microdata entered CONTEMPORANEOUSLY (features at the
    SAME month as the target), i.e. unknown at the origin. Demonstrates the
    accuracy inflation the leak-free (lagged) design removes."""
    d = total[["month", "working_days"]].copy()
    d["y"] = total["total_amount"].values
    d = d.merge(micro, on="month", how="left")
    mm = pd.to_datetime(d["month"])
    d["month_of_year"] = mm.dt.month
    feat = ["month_of_year", "working_days"] + D.EXO_STD + D.EXO_EXIT   # lag 0 == leak
    d = d.dropna(subset=feat + ["y"]).reset_index(drop=True)
    dates = pd.to_datetime(d["month"]); n = len(d)
    org = M.expanding_origins(n)
    rows = []
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV
    import xgboost as xgb, lightgbm as lgb
    for o in org:
        tr, te = d.iloc[:o], d.iloc[o:o + H]
        if len(te) < H: continue
        Xtr, Xte, ytr = tr[feat].values, te[feat].values, tr["y"].values
        for nm, mk in [("LASSO_LEAK", lambda: make_pipeline(StandardScaler(), LassoCV(cv=5, alphas=50, max_iter=100000))),
                       ("XGB_LEAK", lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, verbosity=0)),
                       ("LGBM_LEAK", lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5, subsample=0.9, verbosity=-1))]:
            try:
                fc = mk().fit(Xtr, ytr).predict(Xte)
            except Exception:
                fc = np.full(H, np.nan)
            for k in range(H):
                rows.append((dates.iloc[o - 1], nm, k + 1, dates.iloc[o + k], te["y"].values[k], fc[k]))
    fcdf = pd.DataFrame(rows, columns=["origin_date", "model", "h", "ds", "actual", "forecast"])
    return _score_T2(fcdf, total, anom)


# ---------------------------------------------------------------------------
def c4a_structural_E(panel, total, by_cat, anom, origins, cats=("K1", "K2", "K3")):
    """Structural model: expenditure = sum_c beneficiaries_c x rate_c.
    beneficiaries_c forecast by ETS on the recipient-count series (own history);
    rate_c = robust (12m median) per-recipient amount observed up to the origin
    (the legislated amount is a stable step function known in advance). Leak-free."""
    grid = pd.date_range(total["month"].min(), total["month"].max(), freq="MS")
    rec = (panel.groupby(["month", "category"])["applicant_id"].nunique()
           .rename("recipients").reset_index())
    recw = (pd.MultiIndex.from_product([grid, list(cats)], names=["month", "category"])
            .to_frame(index=False).merge(rec, on=["month", "category"], how="left").fillna(0.0))
    rec_s = {c: recw[recw.category == c].sort_values("month")["recipients"].values for c in cats}
    amt_s = {c: by_cat[by_cat.category == c].sort_values("month")["total_amount"].values for c in cats}
    dates = pd.to_datetime(total["month"]).values
    ytot = total["total_amount"].values
    rows = []
    for o in origins:
        od = dates[o - 1]
        tot_fc = np.zeros(H)
        ok = True
        for c in cats:
            r = rec_s[c][:o].astype(float)
            # rate known at origin: median per-recipient amount over last 12 months
            with np.errstate(divide="ignore", invalid="ignore"):
                rate_hist = np.where(rec_s[c][:o] > 0, amt_s[c][:o] / rec_s[c][:o], np.nan)
            rate = np.nanmedian(rate_hist[-12:]) if np.isfinite(rate_hist[-12:]).any() else np.nan
            try:
                sf = StatsForecast(models=[AutoETS(season_length=12)], freq="MS", n_jobs=1)
                df = pd.DataFrame({"unique_id": c, "ds": dates[:o], "y": r})
                fc = sf.forecast(df=df, h=H)
                rhat = np.clip(fc.iloc[:, -1].values, 0, None)
            except Exception:
                ok = False; break
            tot_fc += rhat * rate
        if not ok: continue
        for k in range(H):
            rows.append((od, "E_structural", k + 1, dates[o + k], ytot[o + k], tot_fc[k]))
    fcdf = pd.DataFrame(rows, columns=["origin_date", "model", "h", "ds", "actual", "forecast"])
    return _score_T2(fcdf, total, anom)


# ---------------------------------------------------------------------------
def c5_ml_variants(total, anom, origins):
    """ML robustness: log target, differenced target, direct multi-step,
    and a post-2022 subsample. XGBoost on the aggregate, leak-free."""
    import xgboost as xgb
    y = total["total_amount"].values.astype(float)
    dates = pd.to_datetime(total["month"]).values
    def xgbm(): return xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                        subsample=0.9, colsample_bytree=0.9, verbosity=0)
    rows = []

    def recursive_variant(name, transform, inverse):
        for o in origins:
            tr = y[:o]
            z = transform(tr)
            X = M._lagmat(z); okm = X.dropna()
            if len(okm) < 15: continue
            ztr = z[okm.index.values]
            try:
                m = xgbm().fit(okm.values, ztr)
                zf = M._recursive(lambda Z: m.predict(Z.values), z, H)
                fc = inverse(np.array(zf), tr)
            except Exception:
                fc = [np.nan] * H
            for k in range(H):
                rows.append((dates[o - 1], name, k + 1, dates[o + k], y[o + k], float(fc[k])))

    recursive_variant("XGB_log", lambda a: np.log1p(a), lambda zf, tr: np.expm1(zf))
    # differenced: model first difference, cumulate from last observed level
    def diff_inv(zf, tr):
        return tr[-1] + np.cumsum(zf)
    def diff_fwd(a):
        d = np.diff(a, prepend=a[0]); return d
    recursive_variant("XGB_diff", diff_fwd, diff_inv)

    # direct multi-step: a separate model per horizon on lagged design (leak-free)
    for o in origins:
        tr = y[:o]
        X = M._lagmat(tr); okm = X.dropna()
        if len(okm) < 15: continue
        idx = okm.index.values
        for k in range(1, H + 1):
            # target at +k for rows whose +k is within training
            valid = idx[idx + k < o]
            if len(valid) < 12:
                rows.append((dates[o - 1], "XGB_direct", k, dates[o + k - 1], y[o + k - 1], np.nan)); continue
            Xk = X.loc[valid].values; yk = tr[valid + k]
            # feature row at the origin (last complete lag row)
            xlast = X.loc[[idx[-1]]].values
            try:
                m = xgbm().fit(Xk, yk)
                fc = float(m.predict(xlast)[0])
            except Exception:
                fc = np.nan
            rows.append((dates[o - 1], "XGB_direct", k, dates[o + k - 1], y[o + k - 1], fc))

    fcdf = pd.DataFrame(rows, columns=["origin_date", "model", "h", "ds", "actual", "forecast"])
    full = _score_T2(fcdf, total, anom)

    # post-2022 subsample: origins from 2022-01 onward only
    post = [o for o in origins if pd.Timestamp(dates[o - 1]) >= pd.Timestamp("2022-01-01")]
    sub_rows = []
    for o in post:
        tr = y[:o]
        X = M._lagmat(tr); okm = X.dropna()
        if len(okm) < 15: continue
        try:
            m = xgbm().fit(okm.values, tr[okm.index.values])
            fc = M._recursive(lambda Z: m.predict(Z.values), tr, H)
        except Exception:
            fc = [np.nan] * H
        for k in range(H):
            sub_rows.append((dates[o - 1], "XGB_post2022", k + 1, dates[o + k], y[o + k], float(fc[k])))
    sub = _score_T2(pd.DataFrame(sub_rows, columns=["origin_date", "model", "h", "ds", "actual", "forecast"]), total, anom)
    return pd.concat([full, sub], ignore_index=True)


# ---------------------------------------------------------------------------
def c8_horizon_fiscal_pi(total, anom, origins):
    """Accuracy by horizon, fiscal-year cumulative error, and 80/95% PI coverage
    for ETS/ARIMA/SNAIVE. Uses statsforecast CV with prediction levels."""
    y = total["total_amount"].values.astype(float)
    dates = pd.to_datetime(total["month"])
    n = len(total); NW = (n - H) - W0 + 1
    df = pd.DataFrame({"unique_id": "agg", "ds": dates, "y": y})
    sf = StatsForecast(models=[SeasonalNaive(season_length=12), AutoETS(season_length=12),
                               AutoARIMA(season_length=12)], freq="MS", n_jobs=1)
    cv = sf.cross_validation(df=df, h=H, step_size=1, n_windows=NW, refit=True, level=[80, 95])
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    cv["ds"] = pd.to_datetime(cv["ds"]); cv["cutoff"] = pd.to_datetime(cv["cutoff"])
    cv["h"] = cv.groupby(["cutoff"]).cumcount() % H + 1
    anomset = set(pd.to_datetime(list(anom)))
    cv_na = cv[~cv["ds"].isin(anomset)].copy()      # exclude reset months from error metrics

    models = {"SNAIVE": "SeasonalNaive", "ETS": "AutoETS", "ARIMA": "AutoARIMA"}
    # by horizon (WAPE per h, non-anomalous)
    hz = []
    for nm, col in models.items():
        for k in range(1, H + 1):
            g = cv_na[cv_na.h == k]
            hz.append(dict(model=nm, h=k, WAPE=_wape(g["y"], g[col])))
    horizon = pd.DataFrame(hz)

    # fiscal-year cumulative error: sum forecast vs sum actual within calendar year,
    # per origin, then mean |pct error| (full year incl. all months)
    fisc = []
    for nm, col in models.items():
        rec = []
        for od, g in cv.groupby("cutoff"):
            g = g.copy(); g["year"] = g["ds"].dt.year
            for yr, gy in g.groupby("year"):
                a, f = gy["y"].sum(), gy[col].sum()
                if a > 0:
                    rec.append(abs(f - a) / a)
        fisc.append(dict(model=nm, mean_cumulative_abs_pct_error=float(np.mean(rec)) if rec else np.nan,
                         n_blocks=len(rec)))
    fiscal = pd.DataFrame(fisc)

    # PI coverage (non-anomalous months)
    cov = []
    for nm, col in models.items():
        g = cv_na
        for lv in [80, 95]:
            lo, hi = f"{col}-lo-{lv}", f"{col}-hi-{lv}"
            if lo in g and hi in g:
                inside = ((g["y"] >= g[lo]) & (g["y"] <= g[hi])).mean()
                cov.append(dict(model=nm, level=lv, empirical_coverage=float(inside)))
    coverage = pd.DataFrame(cov)
    return horizon, fiscal, coverage


def _score_T2(fcdf, total, anom):
    if fcdf is None or len(fcdf) == 0: return pd.DataFrame()
    dates = pd.to_datetime(total["month"]); yfull = total["total_amount"].values
    anom = set(pd.to_datetime(list(anom)))
    out = []
    for od, g0 in fcdf.groupby("origin_date"):
        od = pd.Timestamp(od); tr = yfull[dates.values <= np.datetime64(od)]
        for mdl, g in g0.groupby("model"):
            g = g.sort_values("h"); a = g["actual"].values.astype(float); f = g["forecast"].values.astype(float)
            ds = pd.to_datetime(g["ds"]); keep = ~ds.isin(anom).values
            a, f = a[keep], f[keep]
            if len(a) == 0: continue
            out.append(dict(model=mdl, origin_date=od, WAPE=_wape(a, f), MASE=_mase(a, f, tr)))
    r = pd.DataFrame(out)
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    A, T = "data/admin_clean.csv", "data/transactions_clean.csv"
    panel, nb = D.load_attributed(A, T)
    total, by_cat, grid = D.build_series(panel)
    anom, thr = D.anomaly_labels(total)
    micro = D.micro_aggregate(panel, grid)
    origins = M.expanding_origins(len(total))

    print("[C2] anomaly context"); ctx = c2_anomaly_context(total, anom); ctx.to_csv(f"{OUT}/anomaly_months.csv", index=False)
    print(ctx.to_string(index=False))

    print("\n[C4b] leaky vs leak-free Strategy C"); leaky = c4b_leaky_C(total, micro, anom, origins)
    leaky.groupby("model").agg(WAPE=("WAPE","mean"),MASE=("MASE","mean"),n=("WAPE","count")).to_csv(f"{OUT}/leaky_C.csv")
    print(leaky.groupby("model").WAPE.mean().round(4).to_string())

    print("\n[C4a] structural Strategy E"); E = c4a_structural_E(panel, total, by_cat, anom, origins)
    E.groupby("model").agg(WAPE=("WAPE","mean"),MASE=("MASE","mean"),n=("WAPE","count")).to_csv(f"{OUT}/structural_E.csv")
    print(E.groupby("model").agg(WAPE=("WAPE","mean"),MASE=("MASE","mean"),n=("WAPE","count")).round(4).to_string())

    print("\n[C5] ML variants"); mlv = c5_ml_variants(total, anom, origins)
    mlv.groupby("model").agg(WAPE=("WAPE","mean"),MASE=("MASE","mean"),n=("WAPE","count")).to_csv(f"{OUT}/ml_variants.csv")
    print(mlv.groupby("model").agg(WAPE=("WAPE","mean"),MASE=("MASE","mean"),n=("WAPE","count")).round(4).to_string())

    print("\n[C8] horizon / fiscal / PI coverage"); hz, fisc, cov = c8_horizon_fiscal_pi(total, anom, origins)
    hz.to_csv(f"{OUT}/accuracy_by_horizon.csv", index=False)
    fisc.to_csv(f"{OUT}/fiscal_year_cumulative.csv", index=False)
    cov.to_csv(f"{OUT}/pi_coverage.csv", index=False)
    print("by horizon (WAPE):\n", hz.pivot(index="h", columns="model", values="WAPE").round(4).to_string())
    print("fiscal-year cumulative abs % error:\n", fisc.round(4).to_string(index=False))
    print("PI coverage:\n", cov.round(3).to_string(index=False))
    print(f"\n[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
