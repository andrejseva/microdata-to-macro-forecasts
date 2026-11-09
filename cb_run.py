"""
Child-benefit forecasting pipeline -- RUNNER.

Runs four aggregation strategies (A direct-aggregate, B bottom-up,
C feature-enriched, D MinT reconciliation) under anomaly treatments
(T1 naive, T2 exclusion, T3 calendar-adjustment, T4 regime-aware), all
leakage-free, then builds the headline tables and the statistical-comparison
package.

NOTE on labelling: Strategy A forecasts the aggregate DIRECTLY
(it is not a proportions-based top-down split); for any LINEAR method (e.g.
seasonal naive) the bottom-up sum equals the direct aggregate by construction,
so identical A/B rows are an algebraic identity, not an empirical finding. The
internal key "A_TD" is kept for continuity with earlier outputs; the display
label is "A direct-aggregate".

Added for the revision:
  * SDR (seasonal-dummy + trend + step-2022 OLS) and forecast combinations
    (ETS+Theta, ETS+ARIMA) as transparent benchmarks.
  * a COMMON-evaluation-sample table: every strategy scored on the 26 common
    origins and the identical non-anomalous target months.
  * TJAN (deterministic January rule) alongside T4.
  * DM with HAC bandwidth + effective n + non-overlapping robustness + Holm;
    MCS elimination p-values; TOST equivalence for ETS vs MinT.
  * regime confusion matrix by year; MinT error-correlation + shrinkage; feature
    audit.

Usage:
    python cb_run.py <admin_csv> <transactions_csv> [out_dir]
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import cb_data as D
import cb_models as M

DISPLAY = {"A_TD": "A direct-aggregate", "B_BU": "B bottom-up",
           "C_FEA": "C feature-enriched", "D_HYB": "D MinT"}


def per_cat_base(by_cat, total, cats, n_windows, origins, normalize=False):
    wd = total.set_index("month")["working_days"]
    cat_df = {c: by_cat[by_cat.category == c].sort_values("month") for c in cats}
    dts = cat_df[cats[0]]["month"].values
    assert all((cat_df[c]["month"].values == dts).all() for c in cats), \
        "per_cat_base assumes every category shares the same month grid"
    w = wd.reindex(pd.to_datetime(dts)).values
    series_map = {c: (cat_df[c]["total_amount"].values / w if normalize
                       else cat_df[c]["total_amount"].values) for c in cats}
    sf_all = M.sf_cv_multi(series_map, dts, n_windows)
    parts = []
    for c in cats:
        y = cat_df[c]["total_amount"].values
        sf = sf_all[sf_all["unique_id"] == c].drop(columns="unique_id").reset_index(drop=True)
        if normalize:
            ml = M.ml_cv(y / w, dts, origins)
            base = pd.concat([sf, ml], ignore_index=True)
            wmap = dict(zip(pd.to_datetime(dts), w))
            base["forecast"] = base["forecast"] * base["ds"].map(wmap)
            base["actual"]   = base["actual"]   * base["ds"].map(wmap)
        else:
            ml = M.ml_cv(y, dts, origins)
            base = pd.concat([sf, ml], ignore_index=True)
        parts.append(base)
    allc = pd.concat(parts, ignore_index=True)
    summed = (allc.groupby(["origin_date", "model", "h", "ds"], as_index=False)
              .agg(actual=("actual", lambda s: s.sum(skipna=False)),
                   forecast=("forecast", lambda s: s.sum(skipna=False))))
    return summed


def agg_base(total, n_windows, origins, normalize=False):
    y = total["total_amount"].values; dts = total["month"].values
    if normalize:
        w = total["working_days"].values
        sf = M.sf_cv(y / w, dts, "agg", n_windows)
        ml = M.ml_cv(y / w, dts, origins)
        base = pd.concat([sf, ml], ignore_index=True)
        wmap = dict(zip(pd.to_datetime(dts), w))
        base["forecast"] = base["forecast"] * base["ds"].map(wmap)
        base["actual"]   = base["actual"]   * base["ds"].map(wmap)
        return base
    sf = M.sf_cv(y, dts, "agg", n_windows)
    ml = M.ml_cv(y, dts, origins)
    base = pd.concat([sf, ml], ignore_index=True)
    # transparent deterministic benchmark + simple combinations
    sdr = M.sdr_cv(total, origins)
    comb = M.make_combinations(base, specs=(("ETS", "THETA"), ("ETS", "ARIMA")))
    return pd.concat([base, sdr, comb], ignore_index=True)


def _headline(all_res):
    return (all_res.groupby(["strategy", "treatment", "model"])
            .agg(MAPE=("MAPE", "mean"), WAPE=("WAPE", "mean"), MASE=("MASE", "mean"),
                 RMSSE=("RMSSE", "mean"), n=("WAPE", "count")).reset_index()
            .sort_values(["treatment", "WAPE"]))


def compute(path_admin, path_trx, fast=False, do_stats=True):
    if fast:
        M.SF_REFIT = False
        M.SF_FAST = True
    panel, nb = D.load_attributed(path_admin, path_trx)
    total, by_cat, grid = D.build_series(panel)
    anom, thr = D.anomaly_labels(total)
    micro = D.micro_aggregate(panel, grid)
    design, FEAT, REG = D.build_design(total, micro, anom)
    cats = D.KEEP_CATS
    n = len(total); origins = M.expanding_origins(n); NW = (n - D.H) - D.INIT_WINDOW + 1

    A_raw = agg_base(total, NW, origins, False); A_norm = agg_base(total, NW, origins, True)
    B_raw = per_cat_base(by_cat, total, cats, NW, origins, False)
    B_norm = per_cat_base(by_cat, total, cats, NW, origins, True)
    C_raw = M.fe_cv(design, FEAT, origins, False); C_norm = M.fe_cv(design, FEAT, origins, True)
    D_raw = M.mint_cv(by_cat, total, cats, origins, which=(("ets",) if fast else ("ets", "arima")))

    # ---- score every (strategy, treatment), incl. TJAN for A --------------
    res = [
        M.score(A_raw, "A_TD", "T1", total, anom), M.score(A_raw, "A_TD", "T2", total, anom),
        M.score(A_norm, "A_TD", "T3", total, anom),
        M.score(A_raw, "A_TD", "T4", total, anom, design, REG),
        M.score(A_raw, "A_TD", "TJAN", total, anom),
        M.score(B_raw, "B_BU", "T1", total, anom), M.score(B_raw, "B_BU", "T2", total, anom),
        M.score(B_norm, "B_BU", "T3", total, anom),
        M.score(B_raw, "B_BU", "T4", total, anom, design, REG),
        M.score(C_raw, "C_FEA", "T1", total, anom), M.score(C_raw, "C_FEA", "T2", total, anom),
        M.score(C_norm, "C_FEA", "T3", total, anom),
        M.score(C_raw, "C_FEA", "T4", total, anom, design, REG),
        M.score(D_raw, "D_HYB", "T1", total, anom), M.score(D_raw, "D_HYB", "T2", total, anom),
    ]
    all_res = pd.concat([r for r in res if len(r)], ignore_index=True)
    head = _headline(all_res)

    # ---- COMMON evaluation sample: 26 common origins, non-anomalous months -
    #. Every strategy scored on the identical sample.
    common_origins = set(pd.to_datetime(C_raw["origin_date"].unique()))
    non_anom = [m for m in pd.to_datetime(total["month"]) if m not in anom]
    def _restrict(fc):
        return fc[pd.to_datetime(fc["origin_date"]).isin(common_origins)]
    res_c = [
        M.score(_restrict(A_raw), "A_TD", "COMMON", total, anom, restrict_ds=non_anom),
        M.score(_restrict(B_raw), "B_BU", "COMMON", total, anom, restrict_ds=non_anom),
        M.score(_restrict(C_raw), "C_FEA", "COMMON", total, anom, restrict_ds=non_anom),
        M.score(_restrict(D_raw), "D_HYB", "COMMON", total, anom, restrict_ds=non_anom),
    ]
    all_common = pd.concat([r for r in res_c if len(r)], ignore_index=True)
    head_common = _headline(all_common)

    out = dict(panel=panel, total=total, by_cat=by_cat, grid=grid, anom=anom, thr=thr,
               micro=micro, design=design, FEAT=FEAT, REG=REG, origins=origins,
               A_raw=A_raw, A_norm=A_norm, B_raw=B_raw, B_norm=B_norm,
               C_raw=C_raw, C_norm=C_norm, D_raw=D_raw,
               all_res=all_res, head=head, all_common=all_common, head_common=head_common,
               common_origins=sorted(common_origins), non_anom=non_anom)

    if not do_stats:
        return out

    # ---- Diebold-Mariano (HAC + HLN) vs best, with detail + Holm ----------
    dm_rows = []
    for trt, g in all_res.groupby("treatment"):
        wide = g.assign(key=g.strategy + "_" + g.model).pivot_table(
            index="origin_date", columns="key", values="WAPE").dropna(axis=1)
        if wide.shape[1] < 2 or wide.shape[0] < 5:
            continue
        best = wide.mean().idxmin()
        prow = {}
        for k in wide.columns:
            if k == best: continue
            z, p, nn, neff, trunc = M.dm_detail(wide[k].values, wide[best].values, trunc=D.H - 1)
            z12, p12, *_ = M.dm_detail(wide[k].values, wide[best].values, trunc=12)
            dm_rows.append(dict(treatment=trt, model=k, vs_best=best, DM=z, p=p,
                                n=nn, n_eff=neff, trunc=trunc, DM_bw12=z12, p_bw12=p12))
            prow[k] = p
        holm_adj = M.holm(prow)
        for r in dm_rows:
            if r["treatment"] == trt and r["model"] in holm_adj:
                r["p_holm"] = holm_adj[r["model"]]
    dm_tab = pd.DataFrame(dm_rows)

    # non-overlapping robustness on T2
    dmno_rows = []
    g2 = all_res[all_res.treatment == "T2"].assign(key=lambda x: x.strategy + "_" + x.model)
    if len(g2):
        wide2 = g2.pivot_table(index="origin_date", columns="key", values="WAPE").dropna(axis=1)
        best2 = wide2.mean().idxmin()
        longform = wide2.reset_index().melt(id_vars="origin_date", var_name="model_key", value_name="loss")
        for k in wide2.columns:
            if k == best2: continue
            z, p, nb_ = M.dm_nonoverlap(
                longform[longform.model_key.isin([k, best2])], k, best2, h=D.H)
            dmno_rows.append(dict(treatment="T2", model=k, vs_best=best2,
                                  DM_nonoverlap=z, p_nonoverlap=p, n_blocks=nb_))
    dm_nonoverlap_tab = pd.DataFrame(dmno_rows)

    # ---- MCS with elimination p-values ------------------------------------
    mcs_rows = []
    for trt, g in all_res.groupby("treatment"):
        wide = g.assign(key=g.strategy + "_" + g.model).pivot_table(
            index="origin_date", columns="key", values="WAPE")
        keep, pvals = M.mcs(wide, alpha=0.10, B=(300 if fast else 800), return_pvals=True)
        for k, pv in pvals.items():
            mcs_rows.append(dict(treatment=trt, model=k, mcs_p=pv, in_MCS=(k in keep)))
    mcs_tab = pd.DataFrame(mcs_rows)

    # ---- TOST: ETS vs best MinT equivalence (T2 and COMMON) ----------------
    tost_rows = []
    for label, frame in [("T2", all_res[all_res.treatment == "T2"]),
                         ("COMMON", all_common)]:
        w = frame.assign(key=frame.strategy + "_" + frame.model).pivot_table(
            index="origin_date", columns="key", values="WAPE")
        a_key = "A_TD_ETS"
        mint_keys = [c for c in w.columns if c.startswith("D_HYB_")]
        if a_key in w and mint_keys:
            bestmint = w[mint_keys].mean().idxmin()
            sub = w[[a_key, bestmint]].dropna()
            margin = 0.02 * sub[a_key].mean()        # equivalence margin = 2% of ETS WAPE level
            tr = M.tost((sub[bestmint] - sub[a_key]).values, margin=margin)
            tr.update(sample=label, model_a=a_key, model_b=bestmint)
            tost_rows.append(tr)
    tost_tab = pd.DataFrame(tost_rows)

    # ---- Friedman + Nemenyi (+ explicit mean-rank table) ------------------
    friedman_rows = []; ranks_rows = []; nemenyi_rows = []; friedman_detail = {}
    for trt, g in all_res.groupby("treatment"):
        wide = g.assign(key=g.strategy + "_" + g.model).pivot_table(
            index="origin_date", columns="key", values="WAPE")
        fres = M.friedman_nemenyi(wide, alpha=0.10)
        friedman_detail[trt] = fres
        friedman_rows.append(dict(
            treatment=trt, chi2=fres["chi2"], dof=fres["dof"], p=fres["p"],
            n_models=fres["n_models"], n_origins=fres["n_origins"], cd=fres["cd"],
            dropped_models=";".join(fres["dropped_models"]), posthoc_method=fres["posthoc_method"]))
        if fres["ranks"] is not None:
            best_rank = min(fres["ranks"].values())
            best_key = min(fres["ranks"], key=fres["ranks"].get)
            for mk, r in fres["ranks"].items():
                ranks_rows.append(dict(treatment=trt, model=mk, mean_rank=r,
                                       rank_gap_vs_best=r - best_rank,
                                       within_CD_of_best=(r - best_rank) <= (fres["cd"] or np.inf),
                                       best_model=best_key))
        if fres["nemenyi"] is not None:
            for (mi, mj), pv in fres["nemenyi"].items():
                nemenyi_rows.append(dict(treatment=trt, model_i=mi, model_j=mj, p_value=pv))
    friedman_tab = pd.DataFrame(friedman_rows)
    ranks_tab = pd.DataFrame(ranks_rows)
    nemenyi_tab = pd.DataFrame(nemenyi_rows)

    # ---- regime diagnostic + confusion by year ----------------------------
    rrows = []; aset = set(pd.to_datetime(list(anom)))
    for o in origins:
        od = pd.to_datetime(total["month"]).iloc[o - 1]
        clf = M.fit_regime(design, REG, od)
        tdates = pd.to_datetime(total["month"]).iloc[o:o + D.H].tolist()
        pred = M.predict_regime(clf, design, REG, tdates)
        for dt, pr in zip(tdates, pred):
            rrows.append(dict(origin_date=od, ds=dt, pred_anom=(pr == "anomaly"),
                              true_anom=(pd.Timestamp(dt) in aset)))
    regime = pd.DataFrame(rrows)
    confusion = M.regime_confusion_by_year(regime)

    # ---- per-origin-year WAPE table (T2) ----------------------------------
    t2 = all_res[all_res.treatment == "T2"].copy()
    t2["year"] = pd.to_datetime(t2["origin_date"]).dt.year
    t2["key"] = t2.strategy + "·" + t2.model
    byyear = (t2.pivot_table(index="key", columns="year", values="WAPE", aggfunc="mean"))
    byyear["ALL"] = t2.groupby("key")["WAPE"].mean()

    # ---- MinT transparency + feature audit --------------------------------
    mint_diag = M.mint_diagnostics(by_cat, total, cats, origins[-1], base="ets")
    audit = D.feature_audit(base_path_used=("legislated_base" in FEAT))

    out.update(dm=dm_tab, dm_nonoverlap=dm_nonoverlap_tab, mcs=mcs_tab, tost=tost_tab,
               friedman=friedman_tab, ranks=ranks_tab, nemenyi=nemenyi_tab,
               friedman_detail=friedman_detail, regime=regime, confusion=confusion,
               byyear=byyear, mint_diag=mint_diag, audit=audit)
    return out


def main(path_admin, path_trx, out_dir="outputs", fast=False):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[mode] {'FAST' if fast else 'FULL (refit on)'}")
    R = compute(path_admin, path_trx, fast=fast, do_stats=True)

    # write every table
    R["all_res"].to_csv(f"{out_dir}/all_results.csv", index=False)
    R["head"].to_csv(f"{out_dir}/accuracy_table.csv", index=False)
    R["all_common"].to_csv(f"{out_dir}/all_results_common.csv", index=False)
    R["head_common"].to_csv(f"{out_dir}/accuracy_table_common.csv", index=False)
    R["dm"].to_csv(f"{out_dir}/dm_tests.csv", index=False)
    R["dm_nonoverlap"].to_csv(f"{out_dir}/dm_nonoverlap.csv", index=False)
    R["mcs"].to_csv(f"{out_dir}/mcs_set.csv", index=False)
    R["tost"].to_csv(f"{out_dir}/tost_equivalence.csv", index=False)
    R["friedman"].to_csv(f"{out_dir}/friedman.csv", index=False)
    R["ranks"].to_csv(f"{out_dir}/friedman_ranks.csv", index=False)
    R["nemenyi"].to_csv(f"{out_dir}/nemenyi_pairs.csv", index=False)
    R["regime"].to_csv(f"{out_dir}/regime.csv", index=False)
    R["confusion"].to_csv(f"{out_dir}/regime_confusion_by_year.csv", index=False)
    R["byyear"].to_csv(f"{out_dir}/wape_by_origin_year.csv")
    R["audit"].to_csv(f"{out_dir}/feature_audit.csv", index=False)
    md = R["mint_diag"]
    md["corr"].to_csv(f"{out_dir}/mint_error_correlation.csv")
    pd.DataFrame([{"origin_date": md["origin_date"], "shrink_lambda": md["shrink_lambda"],
                   "resid_rows": md["resid_rows"]}]).to_csv(
        f"{out_dir}/mint_shrinkage.csv", index=False)

    # print summary
    print("\n=== HEADLINE per treatment (mean WAPE) ===")
    for trt in ["T1", "T2", "T3", "T4", "TJAN"]:
        sub = R["head"][R["head"].treatment == trt].nsmallest(6, "WAPE")
        if len(sub):
            print(f"\n-- {trt} --")
            print(sub[["strategy", "model", "WAPE", "MASE", "n"]].to_string(index=False))
    print("\n=== COMMON SAMPLE (26 common origins, non-anomalous months) ===")
    print(R["head_common"].nsmallest(10, "WAPE")[["strategy", "model", "WAPE", "MASE", "n"]].to_string(index=False))
    print("\n=== T4 vs TJAN (regime classifier vs plain January rule), Strategy A ===")
    for trt in ["T1", "T4", "TJAN"]:
        row = R["head"][(R["head"].treatment == trt) & (R["head"].strategy == "A_TD") & (R["head"].model == "ETS")]
        if len(row):
            print(f"  A·ETS {trt}: WAPE {row.WAPE.values[0]:.3f}  MASE {row.MASE.values[0]:.3f}")
    print("\n=== regime confusion by year ===")
    print(R["confusion"].to_string(index=False))
    print("\n=== MinT error correlation (last origin), shrink lambda =", round(md["shrink_lambda"], 3), "===")
    print(md["corr"].round(3).to_string())
    print("\n=== MCS membership (alpha=0.10) ===")
    for trt in ["T1", "T2", "T3", "T4"]:
        keep = R["mcs"][(R["mcs"].treatment == trt) & (R["mcs"].in_MCS)]["model"].tolist()
        print(f"{trt}: n={len(keep)} -> {keep}")
    print("\n=== TOST (ETS vs best MinT) ===")
    print(R["tost"].to_string(index=False))
    print(f"\nOutputs written to {out_dir}")
    return R


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "data/admin_clean.csv"
    t = sys.argv[2] if len(sys.argv) > 2 else "data/transactions_clean.csv"
    o = sys.argv[3] if len(sys.argv) > 3 else "outputs"
    fast = "--fast" in sys.argv
    main(a, t, o, fast=fast)
