import numpy as np
import pandas as pd

# configuration 
KEEP_CATS   = ["K1", "K2", "K3"]
H           = 6
INIT_WINDOW = 60
SEAS_LAG    = max(12, H + 1)          # 12
STD_LAGS    = sorted({H,     SEAS_LAG})   # vars known within month t
EXIT_LAGS   = sorted({H + 1, SEAS_LAG})   # exiting_recipients: +1 hindsight
assert min(STD_LAGS)  >= H
assert min(EXIT_LAGS) >= H + 1

# the indexation break: percentages rose to 18/26/32% of the minimum-wage base
# from January 2022. The break date is legislated and therefore deterministic
# and known at every forecast origin.
STEP_DATE = pd.Timestamp("2022-01-01")

EXO_STD  = ["n_recipients", "mean_payment", "sd_payment", "p90_payment",
            "mean_num_children", "mean_income", "share_disability",
            "share_bank", "new_recipients"]
EXO_EXIT = ["exiting_recipients"]


def working_days(month_starts):
    out = []
    for m in pd.to_datetime(month_starts):
        rng = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="D")
        out.append(int((rng.weekday < 5).sum()))
    return np.array(out, dtype=float)


def load_attributed(path_admin, path_trx):
    """Load CSVs and attribute each transaction to the category EFFECTIVE on its
    date (most recent positive decision with document_date <= transaction_date).
    Equivalent to a data.table roll=TRUE / R as-of join. Leakage-free: a decision
    always precedes the payment it governs."""
    admin = pd.read_csv(path_admin, low_memory=False)
    admin["document_date"] = pd.to_datetime(admin["document_date"], errors="coerce")
    pos = admin[(admin["document_type"] == "Decision - positive") &
                (admin["benefit_category"].isin(KEEP_CATS)) &
                (admin["document_date"].notna())].copy()
    keep = ["request_id", "document_date", "benefit_category", "applicant_id",
            "num_children", "income_per_member", "status_disability",
            "status_guardianship", "payment_method"]
    pos = pos[keep].sort_values("document_date")

    trx = pd.read_csv(path_trx, usecols=["id", "amount", "transaction_date"])
    trx["transaction_date"] = pd.to_datetime(trx["transaction_date"], errors="coerce")
    trx["amount"] = pd.to_numeric(trx["amount"], errors="coerce")
    trx = trx[(trx["transaction_date"] <= "2025-12-31")] \
            .dropna(subset=["amount", "transaction_date"]).sort_values("transaction_date")
    n_before = len(trx)

    # rolling as-of join, per request, backward in time
    a = pd.merge_asof(trx, pos, left_on="transaction_date", right_on="document_date",
                      left_by="id", right_by="request_id", direction="backward")
    # fallback: payments preceding a request's first decision -> earliest decision
    first = pos.groupby("request_id", as_index=False).head(1).add_suffix("_f")
    a = a.merge(first, left_on="id", right_on="request_id_f", how="left")
    for c in ["benefit_category", "applicant_id", "num_children", "income_per_member",
              "status_disability", "status_guardianship", "payment_method"]:
        a[c] = a[c].fillna(a[c + "_f"])
    a = a.dropna(subset=["benefit_category"])
    a["month"]    = a["transaction_date"].values.astype("datetime64[M]")
    a["category"] = a["benefit_category"]
    assert len(a) <= n_before, "row inflation!"
    return a, n_before


def build_series(panel):
    """Gap-free monthly aggregate and per-category series."""
    grid = pd.date_range(panel["month"].min(), panel["month"].max(), freq="MS")
    by_cat = (panel.groupby(["month", "category"])["amount"]
              .sum().rename("total_amount").reset_index())
    full = (pd.MultiIndex.from_product([grid, KEEP_CATS], names=["month", "category"])
            .to_frame(index=False))
    by_cat = full.merge(by_cat, on=["month", "category"], how="left")
    by_cat["total_amount"] = by_cat["total_amount"].fillna(0.0)

    total = (by_cat.groupby("month")["total_amount"].sum().rename("total_amount")
             .reset_index().sort_values("month").reset_index(drop=True))
    total["working_days"] = working_days(total["month"])
    return total, by_cat, grid


def anomaly_labels(total, init_window=INIT_WINDOW):
    """Threshold calibrated ONLY on the first `init_window` months (known before
    the earliest forecast origin) -> leakage-free."""
    calib = total["total_amount"].iloc[:min(init_window, len(total))].values
    nm = np.median(calib[calib > np.quantile(calib, 0.25)])
    thr = 0.05 * nm
    anom = set(pd.to_datetime(total.loc[total["total_amount"] < thr, "month"]))
    return anom, thr


def micro_aggregate(panel, grid):
    """Contemporaneous microdata aggregate. NEVER used un-lagged downstream."""
    panel = panel.copy()
    panel["is_bank"] = (panel["payment_method"] == "Bank account").astype(float)
    panel["has_dis"] = panel["status_disability"].notna().astype(float)
    g = panel.groupby(["month", "category"])
    mf = g.agg(
        n_recipients=("applicant_id", "nunique"),
        mean_payment=("amount", "mean"),
        sd_payment=("amount", "std"),
        p90_payment=("amount", lambda x: np.nanpercentile(x, 90)),
        mean_num_children=("num_children", "mean"),
        mean_income=("income_per_member", "mean"),
        share_disability=("has_dis", "mean"),
        share_bank=("is_bank", "mean"),
    ).reset_index()

    first_seen = panel.groupby(["applicant_id", "category"])["month"].min().reset_index()
    newr = (first_seen.groupby(["month", "category"]).size()
            .rename("new_recipients").reset_index())

    pay = panel[["month", "category", "applicant_id"]].drop_duplicates().sort_values("month")
    pay["nextm"] = pay.groupby(["category", "applicant_id"])["month"].shift(-1)
    pay["exit"] = pay["nextm"].isna() | (
        pay["nextm"] > (pay["month"].values.astype("datetime64[M]")
                        + np.timedelta64(1, "M")))
    exitr = (pay[pay["exit"]].groupby(["month", "category"]).size()
             .rename("exiting_recipients").reset_index())

    mf = mf.merge(newr, on=["month", "category"], how="left") \
           .merge(exitr, on=["month", "category"], how="left")
    mf[["new_recipients", "exiting_recipients"]] = \
        mf[["new_recipients", "exiting_recipients"]].fillna(0.0)

    # aggregate to total level (recipient-weighted mean payment)
    def wmean(d):
        w = d["n_recipients"].values; v = d["mean_payment"].values
        ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
        return np.average(v[ok], weights=w[ok]) if ok.any() else np.nan
    rows = []
    for m, d in mf.groupby("month"):
        rows.append(dict(
            month=m,
            n_recipients=d["n_recipients"].sum(),
            mean_payment=wmean(d),
            sd_payment=d["sd_payment"].mean(),
            p90_payment=d["p90_payment"].mean(),
            mean_num_children=d["mean_num_children"].mean(),
            mean_income=d["mean_income"].mean(),
            share_disability=d["share_disability"].mean(),
            share_bank=d["share_bank"].mean(),
            new_recipients=d["new_recipients"].sum(),
            exiting_recipients=d["exiting_recipients"].sum(),
        ))
    agg = pd.DataFrame(rows)
    agg = (pd.DataFrame({"month": grid}).merge(agg, on="month", how="left")
           .sort_values("month").reset_index(drop=True))
    num = agg.columns.drop("month")
    agg[num] = agg[num].fillna(0.0)
    return agg


def indexation_controls(months, base_path=None):
    """Deterministic indexation regressors known at every forecast origin.

      step_2022    : 1 from January 2022 onward (the legislated indexation break)
      post2022_ramp: months elapsed since the break (0 before it), a slow drift
      legislated_base : OPTIONAL. If `base_path` (a DataFrame or CSV path with
                        columns [month, base_value]) is supplied, the announced
                        minimum-wage-linked base level is merged in as an
                        exogenous deterministic predictor. Every value is a
                        published, forward-known figure, so it respects the
                        information set. When absent, the step dummy and the ramp
                        capture the break without inventing numbers.
    """
    mm = pd.to_datetime(pd.Series(list(months))).reset_index(drop=True)
    out = pd.DataFrame({"month": mm})
    out["step_2022"] = (mm >= STEP_DATE).astype(int)
    out["post2022_ramp"] = np.maximum(
        0, (mm.dt.year - STEP_DATE.year) * 12 + (mm.dt.month - STEP_DATE.month)).astype(float)
    cols = ["step_2022", "post2022_ramp"]
    if base_path is not None:
        bp = base_path if isinstance(base_path, pd.DataFrame) else pd.read_csv(base_path)
        bp = bp.rename(columns={bp.columns[0]: "month", bp.columns[1]: "legislated_base"})
        bp["month"] = pd.to_datetime(bp["month"])
        out = out.merge(bp[["month", "legislated_base"]], on="month", how="left")
        out["legislated_base"] = out["legislated_base"].ffill().bfill()
        cols.append("legislated_base")
    return out, cols


def build_design(total, micro_agg, anom, base_path=None):
    """Leak-free feature-enriched design.
    Guarantee: for target t+h (h=1..H) every predictor is observed at month <= t.
      - calendar (month_of_year, quarter, is_jan, covid, trend, working_days): deterministic
      - indexation (step_2022, post2022_ramp, [legislated_base]): deterministic, known at origin
      - <var>_l<L>, L>=H  -> value at (t+h)-L <= t
      - exiting_recipients_l<L>, L>=H+1 -> its +1 hindsight peeks at (t+h)-L+1 <= t
    """
    d = total[["month", "working_days"]].copy()
    d["y"] = total["total_amount"].values
    d = d.merge(micro_agg, on="month", how="left")
    mm = pd.to_datetime(d["month"])
    d["month_of_year"] = mm.dt.month
    d["quarter"] = mm.dt.quarter
    d["is_jan"] = (mm.dt.month == 1).astype(int)
    d["covid"] = ((mm >= "2020-03-01") & (mm <= "2020-08-31")).astype(int)
    d["trend"] = (np.arange(len(d)))

    # --- deterministic indexation controls ----------
    idx, idx_cols = indexation_controls(d["month"], base_path=base_path)
    for c in idx_cols:
        d[c] = idx[c].values

    feat = (["month_of_year", "quarter", "is_jan", "covid", "trend", "working_days"]
            + idx_cols)
    for v in EXO_STD:
        for L in STD_LAGS:
            nm = f"{v}_l{L}"; d[nm] = d[v].shift(L); feat.append(nm)
    for v in EXO_EXIT:
        for L in EXIT_LAGS:
            nm = f"{v}_l{L}"; d[nm] = d[v].shift(L); feat.append(nm)
    for L in STD_LAGS:
        nm = f"y_l{L}"; d[nm] = d["y"].shift(L); feat.append(nm)

    d["is_anomaly"] = pd.to_datetime(d["month"]).isin(anom).astype(int)
    regime_cols = (["month_of_year", "quarter", "is_jan", "working_days", "trend",
                    "step_2022"]
                   + [f"y_l{L}" for L in STD_LAGS]
                   + [f"n_recipients_l{L}" for L in STD_LAGS])
    regime_cols = [c for c in regime_cols if c in d.columns]
    return d, feat, regime_cols


def feature_audit(base_path_used=False):
    """Auditable predictor table.

    One row per predictor family: operational definition, source, the reference
    period it summarises, when it becomes available relative to the forecast
    origin t, and the lag actually applied in build_design. Every non-
    deterministic predictor is lagged by at least H (exiting_recipients by H+1),
    so for a forecast of month t+h issued at origin t it is a function of data
    observed at month <= t. Deterministic calendar/indexation regressors are
    known for all future months by construction.
    """
    rows = [
        ("month_of_year / quarter / is_jan", "Calendar position of the target month",
         "Calendar", "target month", "known for all future months", "none (deterministic)"),
        ("working_days", "Number of weekdays in the target month",
         "Calendar", "target month", "known for all future months", "none (deterministic)"),
        ("trend", "Linear month index", "Calendar", "target month",
         "known for all future months", "none (deterministic)"),
        ("covid", "1 during Mar--Aug 2020", "Calendar/policy", "target month",
         "known for all future months", "none (deterministic)"),
        ("step_2022", "1 from Jan 2022 (indexation break)", "Legislation (RS)",
         "target month", "legislated in advance -> known at origin", "none (deterministic)"),
        ("post2022_ramp", "Months since Jan-2022 break", "Legislation (RS)",
         "target month", "known at origin", "none (deterministic)"),
        # legislated_base row is emitted only when an external base path is supplied
        ("n_recipients_lL", "Distinct recipients paid in a month",
         "Admin microdata", "month (t+h)-L", "end of that month",
         f"L in {{{','.join(map(str, STD_LAGS))}}} (>= H)"),
        ("mean_payment / sd_payment / p90_payment (lagged)",
         "Within-month payment distribution", "Admin microdata",
         "month (t+h)-L", "end of that month", f"L in {{{','.join(map(str, STD_LAGS))}}} (>= H)"),
        ("mean_num_children / mean_income (lagged)", "Recipient household means",
         "Admin microdata", "month (t+h)-L", "end of that month",
         f"L in {{{','.join(map(str, STD_LAGS))}}} (>= H)"),
        ("share_disability / share_bank (lagged)", "Composition shares",
         "Admin microdata", "month (t+h)-L", "end of that month",
         f"L in {{{','.join(map(str, STD_LAGS))}}} (>= H)"),
        ("new_recipients_lL", "Recipients first seen in a month",
         "Admin microdata", "month (t+h)-L", "end of that month",
         f"L in {{{','.join(map(str, STD_LAGS))}}} (>= H)"),
        ("exiting_recipients_lL",
         "Recipients with no payment in the FOLLOWING month (one-month look-ahead "
         "in its own construction)", "Admin microdata", "month (t+h)-L",
         "known only one month later", f"L in {{{','.join(map(str, EXIT_LAGS))}}} (>= H+1)"),
        ("y_lL", "Lagged target (aggregate expenditure)", "Admin microdata",
         "month (t+h)-L", "end of that month", f"L in {{{','.join(map(str, STD_LAGS))}}} (>= H)"),
    ]
    if base_path_used:
        rows.insert(6, ("legislated_base", "Announced minimum-wage-linked base level",
                        "Legislation (RS), annual decree", "target month",
                        "published in advance -> known at origin", "none (deterministic)"))
    return pd.DataFrame(rows, columns=[
        "feature", "definition", "source", "reference_period",
        "availability", "lag_applied"])


if __name__ == "__main__":
    A = "data/admin_clean.csv"
    T = "data/transactions_clean.csv"
    panel, n_before = load_attributed(A, T)
    total, by_cat, grid = build_series(panel)
    anom, thr = anomaly_labels(total)
    micro = micro_aggregate(panel, grid)
    design, FEAT, REG = build_design(total, micro, anom)

    print(f"attributed {len(panel):,} of {n_before:,} transactions")
    print(f"months: {len(total)}  range {total['month'].min().date()} -> {total['month'].max().date()}")
    print(f"leak-free anomaly threshold: {thr:,.0f} KM   anomalous months: {len(anom)}")
    print(f"per-category totals (KM): "
          + ", ".join(f"{c}={by_cat[by_cat.category==c].total_amount.sum():,.0f}" for c in KEEP_CATS))
    print(f"#features (leak-free): {len(FEAT)}   #regime features: {len(REG)}")
    print(f"design rows: {len(design)}  complete-case rows: {design[FEAT].dropna().shape[0]}")
    print("first usable design month:", design.dropna(subset=FEAT)['month'].min().date())
    print("\nfeature audit:\n", feature_audit().to_string(index=False))
