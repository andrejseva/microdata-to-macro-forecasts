"""
Child-benefit forecasting pipeline -- MODELS LAYER.

Forecast generators all emit one long DataFrame with columns:
    origin_date, model, h, ds, actual, forecast
Scoring (metrics + anomaly treatments T1..T4, TJAN) is a SEPARATE, shared step so
the training-based MASE scale and the anomaly handling are applied identically to
every strategy.

Leakage guarantees:
  * statsforecast cross_validation uses only data up to each cutoff (expanding).
  * the feature-enriched design feeds only lagged (>= H) predictors.
  * the regime classifier (T4) uses only deterministic-calendar + lagged features.
  * MASE/RMSSE scale uses the training aggregate up to the origin only.

Notes:
  * dm(): HAC truncation lag now explicit and configurable (default H-1); a
    companion dm_detail() reports the effective sample size and settings, and a
    non_overlap flag gives the non-overlapping-block robustness check.
  * mcs(): can return per-model elimination p-values, not membership alone.
  * tost(): two one-sided tests for the ETS vs MinT equivalence claim.
  * holm(): Holm step-down adjustment for the pairwise DM family.
  * sdr_cv(): transparent seasonal-dummy + trend + step-2022 OLS benchmark;
    make_combinations(): simple forecast combinations.
  * score(): common-evaluation-sample support (restrict_ds) so every treatment
    can be scored on one shared sample, and a deterministic January-rule
    treatment "TJAN" to test whether the regime classifier (T4) beats a plain
    calendar rule.
  * mint_diagnostics(): exposes the base-error correlation matrix and the
    shrinkage intensity used by MinT.
  * regime_confusion_by_year(): per-year confusion matrix for the T4 classifier.
"""
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import lightgbm as lgb

from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoETS, AutoARIMA, AutoTheta, ARIMA

from scipy.stats import friedmanchisquare
from statsmodels.stats.libqsturng import qsturng, psturng
try:
    import scikit_posthocs as _sp
    _HAVE_SCIKIT_POSTHOCS = True
except ImportError:
    _HAVE_SCIKIT_POSTHOCS = False

from cb_data import H, INIT_WINDOW, STEP_DATE

N_JOBS = max(1, (os.cpu_count() or 2) - 1)


# ===========================================================================
# helpers
# ===========================================================================
def expanding_origins(n, init=INIT_WINDOW, h=H, step=1):
    """origin index o: train = months[0:o], test = months[o:o+h]."""
    return list(range(init, n - h + 1, step))


def _wape(a, f):
    a = np.asarray(a, float); f = np.asarray(f, float)
    if not np.isfinite(f).all():
        return np.nan
    s = np.nansum(np.abs(a))
    return np.nan if s == 0 else np.nansum(np.abs(a - f)) / s


def _mape(a, f):
    a = np.asarray(a, float); f = np.asarray(f, float)
    if not np.isfinite(f).all():
        return np.nan
    return np.nanmean(np.abs((a - f) / a))


def _mase(a, f, train, m=12):
    train = np.asarray(train, float); f = np.asarray(f, float)
    if not np.isfinite(f).all() or len(train) <= m:
        return np.nan
    sc = np.nanmean(np.abs(train[m:] - train[:-m]))
    if not np.isfinite(sc) or sc == 0:
        return np.nan
    return np.nanmean(np.abs(np.asarray(a, float) - np.asarray(f, float))) / sc


def _rmsse(a, f, train, m=12):
    train = np.asarray(train, float); f = np.asarray(f, float)
    if not np.isfinite(f).all() or len(train) <= m:
        return np.nan
    sc = np.nanmean((train[m:] - train[:-m]) ** 2)
    if not np.isfinite(sc) or sc == 0:
        return np.nan
    return np.sqrt(np.nanmean((np.asarray(a, float) - np.asarray(f, float)) ** 2) / sc)


# ===========================================================================
# univariate generators (Strategy A and per-category for B)
# ===========================================================================
AR1_ORDER = (1, 0, 0)
AR12_ORDER = (1, 0, 0)
AR12_SEASONAL_ORDER = (1, 0, 0)
AR12_SEASON_LENGTH = 12
print(f"[cb_models] AR1  benchmark fitted order: ARIMA{AR1_ORDER} (fixed, non-seasonal)")
print(f"[cb_models] AR12 benchmark fitted order: ARIMA{AR12_ORDER}x{AR12_SEASONAL_ORDER}_{AR12_SEASON_LENGTH} (fixed)")


def _sf_models_named():
    """(model, short_name) pairs. Built per call so each gets a fresh instance."""
    arima = (AutoARIMA(season_length=12, max_p=2, max_q=2, max_P=1, max_Q=1)
             if SF_FAST else AutoARIMA(season_length=12))
    ar1 = ARIMA(order=AR1_ORDER, alias="AR1")
    ar12 = ARIMA(order=AR12_ORDER, seasonal_order=AR12_SEASONAL_ORDER,
                 season_length=AR12_SEASON_LENGTH, alias="AR12")
    return [(SeasonalNaive(season_length=12), "SNAIVE"),
            (AutoETS(season_length=12), "ETS"),
            (arima, "ARIMA"),
            (AutoTheta(season_length=12), "THETA"),
            (ar1, "AR1"), (ar12, "AR12")]


def _sf_models():
    return [m for m, _ in _sf_models_named()]

_SF_NAMES = {"SeasonalNaive": "SNAIVE", "AutoETS": "ETS",
             "AutoARIMA": "ARIMA", "AutoTheta": "THETA",
             "AR1": "AR1", "AR12": "AR12"}

SF_REFIT = True
SF_FAST  = False


def sf_cv(y, dates, uid, n_windows):
    res = sf_cv_multi({uid: y}, dates, n_windows)
    return res.drop(columns="unique_id").reset_index(drop=True)


def _fcol(df):
    """name of the single forecast column in a statsforecast output frame."""
    return [c for c in df.columns if c not in ("unique_id", "ds", "cutoff", "y")][0]


def _cv_perwindow(model, uid, y, dates, origins):
    """Manual expanding-window CV for ONE model/series, catching per-window
    failures and writing NaN for the failing window only. Reproduces the
    cross_validation cutoffs exactly (train = y[:o], forecast H) so numbers
    match the batched path for windows that succeed."""
    rows = []
    for o in origins:
        od = dates[o - 1]
        try:
            tr = pd.DataFrame({"unique_id": uid, "ds": dates[:o], "y": np.asarray(y[:o], float)})
            fc = StatsForecast(models=[model], freq="MS", n_jobs=1).forecast(df=tr, h=H)
            vals = fc[_fcol(fc)].values
        except Exception:
            vals = [np.nan] * H
        for k in range(H):
            rows.append((uid, od, k + 1, dates[o + k], float(y[o + k]), float(vals[k])))
    return rows


def _cv_one_model(model, mname, series_map, dates, n_windows, origins):
    """CV one model over all series. Fast path: a single batched
    cross_validation call. On failure, fall back per series, then per window,
    so a degenerate (series, window) cell becomes NaN instead of aborting the
    whole run. Batched and per-series/per-window paths give identical numbers
    for the cells that succeed (statsforecast fits each series independently)."""
    try:
        df = pd.concat([pd.DataFrame({"unique_id": uid, "ds": dates, "y": np.asarray(y, float)})
                        for uid, y in series_map.items()], ignore_index=True)
        cv = StatsForecast(models=[model], freq="MS", n_jobs=min(N_JOBS, len(series_map))) \
            .cross_validation(df=df, h=H, step_size=1, n_windows=n_windows, refit=SF_REFIT)
        cv = cv.reset_index() if "unique_id" not in cv.columns else cv
        col = _fcol(cv)
        t = cv[["unique_id", "ds", "cutoff", "y", col]].rename(
            columns={"y": "actual", col: "forecast", "cutoff": "origin_date"})
        t["model"] = mname
        return t[["unique_id", "origin_date", "model", "ds", "actual", "forecast"]]
    except Exception as e:
        print(f"[sf_cv] batched '{mname}' failed ({str(e)[:60]}); per-series fallback")
        rows = []
        for uid, y in series_map.items():
            try:
                df1 = pd.DataFrame({"unique_id": uid, "ds": dates, "y": np.asarray(y, float)})
                cv = StatsForecast(models=[model], freq="MS", n_jobs=1) \
                    .cross_validation(df=df1, h=H, step_size=1, n_windows=n_windows, refit=SF_REFIT)
                cv = cv.reset_index() if "unique_id" not in cv.columns else cv
                col = _fcol(cv)
                for _, r in cv.iterrows():
                    rows.append((uid, r["cutoff"], None, r["ds"], r["y"], r[col]))
            except Exception as e2:
                print(f"[sf_cv]   series '{uid}' model '{mname}' per-window fallback ({str(e2)[:50]})")
                rows += _cv_perwindow(model, uid, np.asarray(y, float), dates, origins)
        t = pd.DataFrame(rows, columns=["unique_id", "origin_date", "_h", "ds", "actual", "forecast"])
        t["model"] = mname
        return t.drop(columns="_h")


def sf_cv_multi(series_map, dates, n_windows):
    """Robust statsforecast expanding-window CV for several series at once.
    Each model is run independently; a single degenerate (model, series, window)
    cell degrades to NaN (recorded as a failed origin downstream) instead of
    aborting the entire pipeline -- the failure mode that previously made the
    full-refit run unreproducible."""
    dates = pd.to_datetime(dates)
    origins = expanding_origins(len(dates))
    out = []
    for model, mname in _sf_models_named():
        t = _cv_one_model(model, mname, series_map, dates, n_windows, origins)
        out.append(t[["unique_id", "origin_date", "model", "ds", "actual", "forecast"]])
    res = pd.concat(out, ignore_index=True)
    res["origin_date"] = pd.to_datetime(res["origin_date"])
    res["ds"] = pd.to_datetime(res["ds"])
    res["h"] = res.groupby(["unique_id", "origin_date", "model"]).cumcount() + 1
    return res[["unique_id", "origin_date", "model", "h", "ds", "actual", "forecast"]]


# ---- recursive univariate ML (LASSO / XGB / LGBM), leak-free ---------------
def _lagmat(y):
    n = len(y); y = np.asarray(y, float)
    cols = {}
    for L in [1, 2, 3, 12, 13]:
        cols[f"lag{L}"] = np.concatenate([[np.nan] * L, y[:-L]]) if L < n else np.full(n, np.nan)
    roll = pd.Series(y).rolling(12).mean().shift(1).values
    cols["roll12"] = roll
    moy = np.arange(n) % 12
    X = pd.DataFrame(cols)
    for k in range(12):
        X[f"m{k}"] = (moy == k).astype(float)
    return X


def _recursive(predict, hist, h):
    hist = list(map(float, hist)); origin_len = len(hist)
    fc = []
    for k in range(h):
        nx = len(hist)
        if nx < 13:
            fc.append(float(np.mean(hist))); hist.append(fc[-1]); continue
        nm = (origin_len + k) % 12
        feat = {"lag1": hist[-1], "lag2": hist[-2], "lag3": hist[-3],
                "lag12": hist[-12], "lag13": hist[-13],
                "roll12": float(np.mean(hist[-12:]))}
        for j in range(12):
            feat[f"m{j}"] = 1.0 if j == nm else 0.0
        x = pd.DataFrame([feat])
        p = float(np.asarray(predict(x)).ravel()[0])
        lo, hi = 0.0, 3.0 * max(hist)
        if not np.isfinite(p): p = float(np.mean(hist))
        fc.append(min(max(p, lo), hi)); hist.append(fc[-1])
    return fc


def _fit_ml(name, X, y):
    if name == "LASSO":
        m = make_pipeline(StandardScaler(), LassoCV(cv=5, alphas=50, max_iter=100000)).fit(X.values, y)
        return lambda Z: m.predict(Z.values)
    if name == "XGB":
        m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, verbosity=0).fit(X.values, y)
        return lambda Z: m.predict(Z.values)
    if name == "LGBM":
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                              min_child_samples=5, subsample=0.9, verbosity=-1).fit(X.values, y)
        return lambda Z: m.predict(Z.values)


def _ml_cv_one_origin(o, y, dates, models):
    train = y[:o]
    Xall = _lagmat(train); ok = Xall.dropna()
    if len(ok) < 20:
        return []
    ytr = train[ok.index.values]
    rows = []
    for nm in models:
        try:
            pred = _fit_ml(nm, ok, ytr)
            fc = _recursive(pred, train, H)
        except Exception:
            fc = [np.nan] * H
        for k in range(H):
            rows.append((dates[o - 1], nm, k + 1, dates[o + k], y[o + k], fc[k]))
    return rows


def ml_cv(y, dates, origins, models=("LASSO", "XGB", "LGBM")):
    y = np.asarray(y, float); dates = pd.to_datetime(dates)
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_ml_cv_one_origin)(o, y, dates, models) for o in origins)
    rows = [r for res in results for r in res]
    return pd.DataFrame(rows, columns=["origin_date", "model", "h", "ds", "actual", "forecast"])


# ---- transparent seasonal-dummy + trend + step-2022 OLS benchmark ----------
#: the natural deterministic benchmark for a series with a fixed
# December peak and a legislated 2022 level shift. Leak-free: coefficients are
# estimated on the training window only; the design is deterministic so the
# H-step block is computable at the origin.
def _sdr_design(dates):
    dates = pd.to_datetime(pd.Series(list(dates))).reset_index(drop=True)
    n = len(dates)
    X = np.ones((n, 1))                                    # intercept
    moy = dates.dt.month.values
    dummies = np.zeros((n, 11))
    for j, mth in enumerate(range(2, 13)):                 # Feb..Dec (Jan is baseline)
        dummies[:, j] = (moy == mth).astype(float)
    trend = np.arange(n, dtype=float).reshape(-1, 1)
    step = (dates.values >= np.datetime64(STEP_DATE)).astype(float).reshape(-1, 1)
    return np.hstack([X, dummies, trend, step])


def _sdr_cv_one_origin(o, y, dates):
    Xall = _sdr_design(dates)
    Xtr, ytr = Xall[:o], y[:o]
    try:
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        fc = Xall[o:o + H] @ beta
    except Exception:
        fc = np.full(H, np.nan)
    return [(dates[o - 1], "SDR", k + 1, dates[o + k], y[o + k], float(fc[k]))
            for k in range(H)]


def sdr_cv(total, origins):
    y = total["total_amount"].values.astype(float)
    dates = pd.to_datetime(total["month"])
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_sdr_cv_one_origin)(o, y, dates) for o in origins)
    rows = [r for res in results for r in res]
    return pd.DataFrame(rows, columns=["origin_date", "model", "h", "ds", "actual", "forecast"])


def make_combinations(base_df, specs=(("ETS", "THETA"), ("ETS", "ARIMA"))):
    """Simple (equal-weight) forecast combinations.
    base_df: long df with the component models present. Returns the combined
    rows named e.g. COMB_ETS_THETA, aligned on (origin_date, h, ds)."""
    out = []
    for combo in specs:
        parts = [base_df[base_df.model == m][["origin_date", "h", "ds", "actual", "forecast"]]
                 .rename(columns={"forecast": f"f_{m}"}) for m in combo]
        merged = parts[0]
        for p in parts[1:]:
            merged = merged.merge(p.drop(columns="actual"), on=["origin_date", "h", "ds"])
        fcols = [f"f_{m}" for m in combo]
        merged["forecast"] = merged[fcols].mean(axis=1)
        merged["model"] = "COMB_" + "_".join(combo)
        out.append(merged[["origin_date", "model", "h", "ds", "actual", "forecast"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["origin_date", "model", "h", "ds", "actual", "forecast"])


# ===========================================================================
# Strategy C: feature-enriched direct multi-step (leak-free design)
# ===========================================================================
def _fe_cv_one_origin(o, d, dates, feat_cols, normalize):
    tr = d.iloc[:o]; te = d.iloc[o:o + H]
    if len(te) < H:
        return []
    Xtr = tr[feat_cols].values
    Xte = te[feat_cols].values
    ytr = (tr["y"] / tr["working_days"]).values if normalize else tr["y"].values
    wd = te["working_days"].values
    preds = {}
    try:
        preds["LASSO_FEA"] = make_pipeline(StandardScaler(), LassoCV(cv=5, alphas=50, max_iter=100000)).fit(Xtr, ytr).predict(Xte)
    except Exception:
        preds["LASSO_FEA"] = np.full(H, np.nan)
    try:
        preds["XGB_FEA"] = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, verbosity=0).fit(Xtr, ytr).predict(Xte)
    except Exception:
        preds["XGB_FEA"] = np.full(H, np.nan)
    try:
        preds["LGBM_FEA"] = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=5, subsample=0.9, verbosity=-1).fit(Xtr, ytr).predict(Xte)
    except Exception:
        preds["LGBM_FEA"] = np.full(H, np.nan)
    odate = dates.iloc[o - 1]
    rows = []
    for nm, fc in preds.items():
        fc = np.asarray(fc, float) * wd if normalize else np.asarray(fc, float)
        for k in range(H):
            rows.append((odate, nm, k + 1, dates.iloc[o + k], te["y"].values[k], fc[k]))
    return rows


def fe_cv(design, feat_cols, origins, normalize=False):
    d = design.dropna(subset=feat_cols + ["y", "working_days"]).reset_index(drop=True)
    dates = pd.to_datetime(d["month"])
    n = len(d)
    org = expanding_origins(n)
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_fe_cv_one_origin)(o, d, dates, feat_cols, normalize) for o in org)
    rows = [r for res in results for r in res]
    return pd.DataFrame(rows, columns=["origin_date", "model", "h", "ds", "actual", "forecast"])


# ===========================================================================
# Strategy D: MinT-shrink reconciliation (rolling), hand-rolled & coherent
# ===========================================================================
def _mint_shrink_P(residuals, S, return_diag=False):
    """MinT(shrink): P = (S'W^-1 S)^-1 S'W^-1 with W shrunk toward its diagonal
    (Schaefer-Strimmer intensity)."""
    R = residuals
    T = R.shape[0]
    W = np.cov(R, rowvar=False)
    d = np.diag(np.diag(W))
    Rc = R - R.mean(0)
    var_w = (Rc ** 2).T @ (Rc ** 2) / T - (W ** 2) * (T - 1) / T
    off = W - d
    denom = np.sum(off ** 2)
    lam = 1.0 if denom == 0 else np.clip(np.sum(np.abs(var_w - np.diag(np.diag(var_w)))) / max(denom, 1e-12), 0, 1)
    Wsh = lam * d + (1 - lam) * W
    Wi = np.linalg.pinv(Wsh)
    M = np.linalg.pinv(S.T @ Wi @ S) @ S.T @ Wi
    if return_diag:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = W / np.sqrt(np.outer(np.diag(W), np.diag(W)))
        return M, dict(W=W, corr=corr, shrink_lambda=float(lam), Wsh=Wsh)
    return M


def _mint_cv_one(model_name, sf_model, sf_fast, o, ytot, series, cats, dates, S):
    def mk():
        if sf_model is AutoARIMA and sf_fast:
            return AutoARIMA(season_length=12, max_p=2, max_q=2, max_P=1, max_Q=1)
        return sf_model(season_length=12)
    try:
        base_fc = []; resid = []
        node_series = [ytot[:o]] + [series[c][:o] for c in cats]
        for s in node_series:
            sf = StatsForecast(models=[mk()], freq="MS", n_jobs=1)
            df = pd.DataFrame({"unique_id": "x", "ds": dates.iloc[:o].values, "y": s})
            fc_df = sf.forecast(df=df, h=H, fitted=True)
            base_fc.append(fc_df.iloc[:, -1].values)
            fv = sf.forecast_fitted_values()
            r = (fv["y"] - fv.iloc[:, -1]).values
            resid.append(r[-min(len(r), o):])
        L = min(len(r) for r in resid)
        Rmat = np.column_stack([r[-L:] for r in resid])
        base_fc = np.array(base_fc)
        Mb = _mint_shrink_P(Rmat, S)
        rec_bottom = Mb @ base_fc
        rec_total = rec_bottom.sum(0)
        return [(dates.iloc[o - 1], model_name, k + 1, dates.iloc[o + k], ytot[o + k], rec_total[k])
                for k in range(H)]
    except Exception:
        return []


def mint_cv(by_cat, total, cats, origins, which=("ets", "arima")):
    dates = pd.to_datetime(total["month"])
    S = np.vstack([np.ones(len(cats)), np.eye(len(cats))])
    series = {c: by_cat[by_cat.category == c].sort_values("month")["total_amount"].values for c in cats}
    ytot = total["total_amount"].values
    spec = [(f"D_{nm}_mint", mdl) for nm, mdl in
            [("ets", AutoETS), ("arima", AutoARIMA)] if nm in which]
    jobs = [(model_name, sf_model, o) for model_name, sf_model in spec for o in origins]
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_mint_cv_one)(model_name, sf_model, SF_FAST, o, ytot, series, cats, dates, S)
        for model_name, sf_model, o in jobs)
    rows = [r for res in results for r in res]
    return pd.DataFrame(rows, columns=["origin_date", "model", "h", "ds", "actual", "forecast"])


def mint_diagnostics(by_cat, total, cats, origin_index, base="ets"):
    """Expose the base-forecast error correlation matrix and shrinkage intensity
    MinT uses at a representative origin. Leak-free: residuals use
    only data up to the origin."""
    dates = pd.to_datetime(total["month"])
    S = np.vstack([np.ones(len(cats)), np.eye(len(cats))])
    series = {c: by_cat[by_cat.category == c].sort_values("month")["total_amount"].values for c in cats}
    ytot = total["total_amount"].values
    o = origin_index
    mdl = AutoETS if base == "ets" else AutoARIMA
    node_series = [ytot[:o]] + [series[c][:o] for c in cats]
    resid = []
    for s in node_series:
        sf = StatsForecast(models=[mdl(season_length=12)], freq="MS", n_jobs=1)
        df = pd.DataFrame({"unique_id": "x", "ds": dates.iloc[:o].values, "y": s})
        sf.forecast(df=df, h=H, fitted=True)
        fv = sf.forecast_fitted_values()
        r = (fv["y"] - fv.iloc[:, -1]).values
        resid.append(r[-min(len(r), o):])
    L = min(len(r) for r in resid)
    Rmat = np.column_stack([r[-L:] for r in resid])
    _, diag = _mint_shrink_P(Rmat, S, return_diag=True)
    nodes = ["TOTAL"] + list(cats)
    diag["corr"] = pd.DataFrame(diag["corr"], index=nodes, columns=nodes)
    diag["W"] = pd.DataFrame(diag["W"], index=nodes, columns=nodes)
    diag["origin_date"] = dates.iloc[o - 1]
    diag["resid_rows"] = int(L)
    return diag


# ===========================================================================
# unified scoring with treatments T1..T4, TJAN, and common-sample support
# ===========================================================================
def fit_regime(design, regime_cols, origin_date):
    tr = design[pd.to_datetime(design["month"]) <= origin_date]
    tr = tr.dropna(subset=regime_cols + ["is_anomaly"])
    if len(tr) < 20 or tr["is_anomaly"].nunique() < 2:
        return None
    try:
        clf = LogisticRegressionCV(Cs=10, cv=5, penalty="l1", solver="liblinear", max_iter=2000)
        clf.fit(tr[regime_cols].values, tr["is_anomaly"].values)
        return clf
    except Exception:
        return None


def predict_regime(clf, design, regime_cols, test_dates):
    out = np.array(["normal"] * len(test_dates), dtype=object)
    if clf is None:
        return out
    dd = design.set_index(pd.to_datetime(design["month"]))
    for i, td in enumerate(pd.to_datetime(test_dates)):
        if td in dd.index:
            row = dd.loc[td, regime_cols]
            x = np.asarray(row.values, float).reshape(1, -1)
            if np.all(np.isfinite(x)):
                out[i] = "anomaly" if clf.predict(x)[0] == 1 else "normal"
    return out


def score(fc, strategy, treatment, total, anom, design=None, regime_cols=None,
          restrict_ds=None):
    """fc: long df [origin_date, model, h, ds, actual, forecast]. Returns per-origin metrics.

    treatment:
      T1   naive (score all target months, forecasts untouched)
      T2   exclusion (drop anomalous target months from scoring)
      T3   calendar-adjustment (applied upstream via normalised series)
      T4   regime-aware (substitute a pre-origin anomaly mean where the leak-free
           classifier predicts a reset)
      TJAN deterministic January rule (substitute the pre-origin anomaly mean on
           every January target month; no classifier): a simple deterministic
           benchmark to compare T4 against.

    restrict_ds (NEW): an iterable of target months; if given, scoring is
      restricted to those months AFTER the treatment is applied. This lets every
      treatment be scored on one common evaluation sample.
    """
    if fc is None or len(fc) == 0:
        return pd.DataFrame()
    dates = pd.to_datetime(total["month"]); yfull = total["total_amount"].values
    anom = set(pd.to_datetime(list(anom)))
    restrict = None if restrict_ds is None else set(pd.to_datetime(list(restrict_ds)))
    rows = []
    for od, g0 in fc.groupby("origin_date"):
        od = pd.Timestamp(od)
        train_scale = yfull[dates.values <= np.datetime64(od)]
        regime_pred = None; anomaly_mean = np.nan
        if treatment in ("T4", "TJAN"):
            av = yfull[(np.isin(dates.values, list(np.array(list(anom), dtype="datetime64[ns]"))))
                       & (dates.values <= np.datetime64(od))]
            anomaly_mean = av.mean() if len(av) else train_scale.mean() * 0.01
        if treatment == "T4":
            clf = fit_regime(design, regime_cols, od)
            tmonths = g0.sort_values("h")["ds"].drop_duplicates().tolist()
            regime_pred = dict(zip(pd.to_datetime(tmonths),
                                   predict_regime(clf, design, regime_cols, tmonths)))
        for mdl, g in g0.groupby("model"):
            g = g.sort_values("h")
            a = g["actual"].values.astype(float); f = g["forecast"].values.astype(float)
            ds = pd.to_datetime(g["ds"])
            if treatment == "T4" and regime_pred is not None:
                f = np.array([anomaly_mean if regime_pred.get(d, "normal") == "anomaly" else fi
                              for d, fi in zip(ds, f)], float)
            if treatment == "TJAN":
                f = np.array([anomaly_mean if d.month == 1 else fi
                              for d, fi in zip(ds, f)], float)
            keep = np.ones(len(ds), bool)
            if treatment == "T2":
                keep &= ~ds.isin(anom).values
            if restrict is not None:
                keep &= ds.isin(restrict).values
            a, f, ds = a[keep], f[keep], ds[keep]
            if len(a) == 0:
                continue
            rows.append(dict(strategy=strategy, treatment=treatment, origin_date=od, model=mdl,
                             MAPE=_mape(a, f), WAPE=_wape(a, f),
                             MASE=_mase(a, f, train_scale), RMSSE=_rmsse(a, f, train_scale)))
    return pd.DataFrame(rows)


def regime_confusion_by_year(regime_df):
    """Per-year TP/FP/FN/TN + precision/recall for the T4 classifier. regime_df columns: origin_date, ds, pred_anom, true_anom.
    One prediction per target month (collapse multiple origins by 'any flag')."""
    if regime_df is None or len(regime_df) == 0:
        return pd.DataFrame()
    g = (regime_df.groupby("ds")
         .agg(pred=("pred_anom", "max"), true=("true_anom", "max")).reset_index())
    g["year"] = pd.to_datetime(g["ds"]).dt.year
    out = []
    for yr, d in g.groupby("year"):
        tp = int(((d.pred) & (d.true)).sum()); fp = int(((d.pred) & (~d.true)).sum())
        fn = int(((~d.pred) & (d.true)).sum()); tn = int(((~d.pred) & (~d.true)).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        out.append(dict(year=int(yr), TP=tp, FP=fp, FN=fn, TN=tn,
                        precision=prec, recall=rec, n_true_resets=tp + fn))
    return pd.DataFrame(out)


# ===========================================================================
# Diebold-Mariano (HAC) + Model Confidence Set + TOST + Holm
# ===========================================================================
def dm(loss1, loss2, h=H, trunc=None):
    """Harvey-Leybourne-Newbold small-sample-corrected Diebold-Mariano.
    HAC long-run variance uses a Bartlett kernel truncated at `trunc` lags
    (default h-1, the standard choice for h-step forecasts). Returns (z, p)."""
    z, p, *_ = dm_detail(loss1, loss2, h=h, trunc=trunc)
    return z, p


def dm_detail(loss1, loss2, h=H, trunc=None):
    """As dm() but also returns n (overlapping blocks), n_eff (approx. independent
    blocks = n/h) and the truncation lag used."""
    d = np.asarray(loss1, float) - np.asarray(loss2, float)
    d = d[np.isfinite(d)]; n = len(d)
    if trunc is None:
        trunc = h - 1
    if n < 5:
        return np.nan, np.nan, n, np.nan, trunc
    db = d.mean(); g0 = np.mean((d - db) ** 2); lrv = g0
    for L in range(1, trunc + 1):
        if L >= n: break
        gl = np.mean((d[L:] - db) * (d[:-L] - db))
        lrv += 2 * (1 - L / (trunc + 1)) * gl
    if not np.isfinite(lrv) or lrv <= 0:
        lrv = g0
    stat = db / np.sqrt(lrv / n)
    hln = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-8))
    from scipy import stats
    z = stat * hln
    n_eff = n / h
    return z, 2 * stats.t.sf(abs(z), df=max(n - 1, 1)), n, n_eff, trunc


def dm_nonoverlap(per_h_df, key_a, key_b, h=H):
    """Non-overlapping-block DM robustness.
    per_h_df: long df with columns [origin_date, h, ds, model_key, loss]; here we
    take one loss per origin block (already block-aggregated WAPE) and keep only
    every h-th origin so the retained blocks do not overlap."""
    w = per_h_df.pivot_table(index="origin_date", columns="model_key", values="loss")
    w = w.sort_index().iloc[::h]            # keep every h-th origin -> disjoint blocks
    if key_a not in w or key_b not in w:
        return np.nan, np.nan, len(w)
    sub = w[[key_a, key_b]].dropna()
    z, p, n, _, _ = dm_detail(sub[key_a].values, sub[key_b].values, h=1, trunc=0)
    return z, p, n


def holm(pvals):
    """Holm step-down adjusted p-values. Input/returns dict or array."""
    if isinstance(pvals, dict):
        keys = list(pvals.keys()); arr = np.array([pvals[k] for k in keys], float)
        adj = _holm_arr(arr)
        return {k: v for k, v in zip(keys, adj)}
    return _holm_arr(np.asarray(pvals, float))


def _holm_arr(p):
    m = len(p); order = np.argsort(p); adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        run = max(run, val)
        adj[idx] = min(run, 1.0)
    return adj


def tost(diff, margin, alpha=0.10):
    """Two one-sided tests of equivalence on a paired difference series. diff = per-origin (loss_A - loss_B). H0: |mean diff| >= margin;
    rejecting both one-sided nulls => statistically equivalent within +/- margin.
    Returns dict with the two one-sided p-values and the decision."""
    from scipy import stats
    d = np.asarray(diff, float); d = d[np.isfinite(d)]; n = len(d)
    if n < 3:
        return dict(n=n, mean=np.nan, p_lower=np.nan, p_upper=np.nan, equivalent=None)
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        se = 1e-12
    t_lower = (m - (-margin)) / se      # H0: mean <= -margin
    t_upper = (m - margin) / se         # H0: mean >=  margin
    p_lower = stats.t.sf(t_lower, df=n - 1)         # P(T > t_lower)
    p_upper = stats.t.cdf(t_upper, df=n - 1)        # P(T < t_upper)
    return dict(n=n, mean=float(m), margin=float(margin),
                p_lower=float(p_lower), p_upper=float(p_upper),
                equivalent=bool(max(p_lower, p_upper) < alpha))


def mcs(loss_df, alpha=0.10, B=1000, blk=3, seed=1, return_pvals=False):
    """loss_df: rows=origins, cols=models, values=loss. Hansen MCS (Tmax,
    stationary bootstrap). With return_pvals=True also returns, per model, the
    MCS p-value = the test p at the step it was eliminated (survivors get the
    final non-rejection p), so membership is backed by a number."""
    L = loss_df.dropna(axis=1).values
    names = list(loss_df.dropna(axis=1).columns)
    T, M = L.shape
    if M < 2 or T < 10:
        return (names, {k: 1.0 for k in names}) if return_pvals else names
    rng = np.random.default_rng(seed)
    def boot_idx():
        idx = np.empty(T, int); i = 0
        while i < T:
            start = rng.integers(0, T); ln = rng.geometric(1 / blk)
            for j in range(ln):
                if i >= T: break
                idx[i] = (start + j) % T; i += 1
        return idx
    surv = list(range(M)); pval = {nm: np.nan for nm in names}; running = 0.0
    while len(surv) > 1:
        sub = L[:, surv]
        dbar = sub.mean(0)
        dij = dbar[:, None] - dbar[None, :]
        boots = np.array([sub[boot_idx()].mean(0) for _ in range(B)])
        var = boots.var(0) + 1e-12
        tstat = np.abs(dij) / np.sqrt(var[:, None] + var[None, :])
        Tmax = np.nanmax([tstat[i, j] for i in range(len(surv)) for j in range(len(surv)) if i != j])
        bd = []
        for b in boots:
            dd = b - b.mean()
            bd.append(np.max(np.abs(dd) / np.sqrt(var)))
        p = float(np.mean(np.array(bd) >= Tmax))
        running = max(running, p)            # MCS p-values are monotone non-decreasing
        if p > alpha:
            for i in surv:
                if np.isnan(pval[names[i]]):
                    pval[names[i]] = running
            break
        worst = surv[int(np.argmax(dbar))]
        pval[names[worst]] = running
        surv.remove(worst)
    if len(surv) == 1 and np.isnan(pval[names[surv[0]]]):
        pval[names[surv[0]]] = 1.0
    keep = [names[i] for i in surv]
    return (keep, pval) if return_pvals else keep


# ===========================================================================
# Friedman omnibus + Nemenyi post-hoc
# ===========================================================================
def _nemenyi_cd(k, n, alpha):
    q_alpha = qsturng(1 - alpha, k, np.inf) / np.sqrt(2)
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))


def friedman_nemenyi(per_origin_wape, alpha=0.10):
    df = per_origin_wape.copy()
    prev = None
    while prev != df.shape:
        prev = df.shape
        df = df.dropna(axis=0, how="any").dropna(axis=1, how="any")
    models = list(df.columns)
    n_origins, n_models = df.shape[0], df.shape[1]
    dropped_models = [c for c in per_origin_wape.columns if c not in models]

    out = dict(models=models, n_models=n_models, n_origins=n_origins,
               origins=list(df.index), dropped_models=dropped_models,
               dropped_origins=int(per_origin_wape.shape[0] - n_origins),
               chi2=np.nan, dof=np.nan, p=np.nan, ranks=None, nemenyi=None,
               cd=np.nan, posthoc_method=None)
    if n_models < 3 or n_origins < 5:
        return out

    chi2, p = friedmanchisquare(*[df[c].values for c in models])
    dof = n_models - 1
    mean_rank = df.rank(axis=1, method="average").mean(axis=0)
    out.update(chi2=float(chi2), dof=dof, p=float(p), ranks=mean_rank.to_dict())

    if p < alpha:
        if _HAVE_SCIKIT_POSTHOCS:
            pmat = _sp.posthoc_nemenyi_friedman(df.values)
            pmat.index = models; pmat.columns = models
            method = "scikit-posthocs (posthoc_nemenyi_friedman)"
        else:
            se = np.sqrt(n_models * (n_models + 1) / (6.0 * n_origins))
            pmat = pd.DataFrame(np.eye(n_models), index=models, columns=models)
            for i, mi in enumerate(models):
                for j, mj in enumerate(models):
                    if i == j:
                        continue
                    q = abs(mean_rank[mi] - mean_rank[mj]) / se * np.sqrt(2)
                    pmat.iloc[i, j] = float(np.atleast_1d(psturng(q, n_models, np.inf))[0])
            method = "manual studentised-range (statsmodels qsturng/psturng)"
        cd = _nemenyi_cd(n_models, n_origins, alpha)
        pairs = {(models[i], models[j]): float(pmat.iloc[i, j])
                 for i in range(n_models) for j in range(i + 1, n_models)}
        out.update(nemenyi=pairs, cd=float(cd), posthoc_method=method)
    return out
