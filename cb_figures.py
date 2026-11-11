import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.colors import LinearSegmentedColormap

PO="outputs"; OUT="outputs/figs_revised"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi":130,"savefig.dpi":300,"figure.constrained_layout.use":True,
    "font.size":10.5,"axes.titlesize":12,"axes.titleweight":"semibold","axes.titlepad":8,
    "axes.labelsize":10,"axes.labelcolor":"#33414B","text.color":"#33414B",
    "axes.edgecolor":"#C9D2D8","axes.linewidth":0.9,"axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.16,"grid.linewidth":0.7,"xtick.labelsize":9.5,"ytick.labelsize":9.5,
    "xtick.color":"#5B6770","ytick.color":"#5B6770","legend.frameon":False,"legend.fontsize":9.5,
    "axes.axisbelow":True,"lines.linewidth":1.7})
INK="#26323B"; BLUE="#2E6F95"; ORANGE="#E08A3C"; RED="#C0392B"; GREEN="#3C8D5B"; GREY="#9AA7B0"; PURPLE="#6C5B9E"
STRAT_C={"A_TD":BLUE,"B_BU":GREEN,"C_FEA":PURPLE,"D_HYB":ORANGE}
SLAB={"A_TD":"DA","B_BU":"BU","C_FEA":"FEA","D_HYB":"MinT"}
TLAB={"T1":"T1 naive","T2":"T2 exclusion","T3":"T3 calendar-adj","T4":"T4 regime-aware","TJAN":"TJAN January-rule"}
MLAB={"ETS":"ETS","Theta":"Theta","THETA":"Theta","ARIMA":"ARIMA","LASSO":"LASSO","XGB":"XGBoost",
      "LGBM":"LightGBM","SNAIVE":"SNAIVE","AR1":"AR(1)","AR12":"AR(1)(1)12",
      "COMB_ETS_THETA":"ETS+Theta","COMB_ETS_ARIMA":"ETS+ARIMA","SDR":"SDR",
      "D_ets_mint":"ETS","D_arima_mint":"ARIMA","LASSO_FEA":"LASSO","XGB_FEA":"XGBoost","LGBM_FEA":"LightGBM"}
def key(s,m): return f"{SLAB.get(s,s)}·{MLAB.get(m,m)}"
def yPct(ax): ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v*100:.0f}%")); ax.yaxis.set_major_locator(MaxNLocator(5))
def yM(ax): ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v/1e6:.1f}M")); ax.yaxis.set_major_locator(MaxNLocator(5))
def years(ax): ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); ax.margins(x=0.01)
def save(n): plt.gcf().savefig(f"{OUT}/{n}.png", dpi=300, bbox_inches="tight"); plt.close()

head=pd.read_csv(f"{PO}/accuracy_table.csv")
allr=pd.read_csv(f"{PO}/all_results.csv")

# --- 08 heatmap (blue) ---
h=head.copy(); h["key"]=[key(s,m) for s,m in zip(h.strategy,h.model)]
piv=h.pivot(index="key",columns="treatment",values="WAPE")
cols=[c for c in ["T1","T2","T3","T4","TJAN"] if c in piv.columns]; piv=piv[cols]
piv=piv.loc[piv.mean(axis=1).sort_values().index]
fig,ax=plt.subplots(figsize=(6.8,0.34*len(piv)+1.2))
cmap=LinearSegmentedColormap.from_list("cb",["#FAFCFD",BLUE,INK]); vmax=np.nanpercentile(piv.values,95)
im=ax.imshow(piv.values,aspect="auto",cmap=cmap,vmin=np.nanmin(piv.values),vmax=vmax)
ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([TLAB.get(c,c) for c in piv.columns],rotation=12,ha="right")
ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v=piv.values[i,j]
        if np.isfinite(v): ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=7,color="white" if v>0.6*vmax else INK)
ax.tick_params(length=0); cb=fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02,label="mean WAPE"); cb.outline.set_visible(False)
ax.set_title("WAPE by strategy·model and treatment"); save("08_wape_heatmap")

# --- 09 treatment bars ---
best=head.groupby(["strategy","treatment"]).WAPE.min().reset_index()
tres=["T1","T2","T3","T4"]; x=np.arange(len(tres)); w=0.2
fig,ax=plt.subplots(figsize=(8,3.4))
for k,st in enumerate(["A_TD","B_BU","C_FEA","D_HYB"]):
    vals=[best[(best.strategy==st)&(best.treatment==t)].WAPE.values for t in tres]
    vals=[v[0] if len(v) else np.nan for v in vals]
    ax.bar(x+(k-1.5)*w,vals,w,label=SLAB[st],color=STRAT_C[st])
ax.set_xticks(x); ax.set_xticklabels([TLAB[t] for t in tres]); yPct(ax)
ax.set_ylabel("best-model WAPE"); ax.set_title("Best model per strategy × treatment",pad=30)
ax.legend(ncol=4,loc="lower center",bbox_to_anchor=(0.5,1.005)); save("09_treatment_bars")

# --- 11 box ---
sub=allr[allr.treatment=="T2"].copy(); sub["key"]=[key(s,m) for s,m in zip(sub.strategy,sub.model)]
k2strat=dict(zip(sub.key,sub.strategy))
order=sub.groupby("key").WAPE.median().sort_values().index
data=[sub[sub.key==k].WAPE.dropna().values for k in order]
scol=[STRAT_C.get(k2strat.get(k),GREY) for k in order]
fig,ax=plt.subplots(figsize=(9,3.8))
bp=ax.boxplot(data,vert=True,patch_artist=True,widths=0.62,showfliers=False,medianprops=dict(color=INK,lw=1.3))
for patch,c in zip(bp["boxes"],scol): patch.set_facecolor(c); patch.set_alpha(0.65); patch.set_edgecolor("none")
ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels(order,rotation=40,ha="right")
yPct(ax); ax.set_ylabel("WAPE per origin"); ax.set_title("Per-origin WAPE spread (T2)"); save("11_box")

# --- 12 mase (T4) ---
s=head[head.treatment=="T4"].copy(); s["key"]=[key(a,m) for a,m in zip(s.strategy,s.model)]
s=s[s.MASE>0.05].sort_values("MASE")
fig,ax=plt.subplots(figsize=(8,0.32*len(s)+1))
ax.barh(s.key,s.MASE,color=[STRAT_C[a] for a in s.strategy],alpha=0.85)
ax.axvline(1.0,color=RED,ls="--",lw=1.2); ax.invert_yaxis()
ax.set_xlabel("MASE  (< 1: errors below the in-sample seasonal-naive scale)"); ax.set_title("Forecast skill under T4"); save("12_mase")

# --- 10 overlay (best T2 model = DA ETS) ---
fA=pd.read_csv("data/forecasts_A.csv",parse_dates=["origin_date","ds"])
ec=pd.read_csv(f"{PO.replace('py_outputs','py_outputs_r3')}/entitlement_vs_cash.csv",parse_dates=["month"])
tot=ec.set_index("month")["cash"].sort_index()          # full monthly cash series (2017-06 -> )
anom=set(pd.read_csv(f"{PO}/regime.csv",parse_dates=["ds"]).query("true_anom").ds)
cand=[o for o in sorted(fA.origin_date.unique()) if set(fA[fA.origin_date==o].ds)&anom]
chosen=(cand or sorted(fA.origin_date.unique()))[:2]
fig,axes=plt.subplots(1,len(chosen),figsize=(9,3.1),sharey=True); axes=np.atleast_1d(axes)
for ax,od in zip(axes,chosen):
    od=pd.Timestamp(od)
    g=fA[(fA.origin_date==od)&(fA.model=="ETS")].sort_values("h")
    win=tot[(tot.index>=od-pd.DateOffset(months=14))&(tot.index<=g.ds.max())]   # continuous actual
    ax.plot(win.index,win.values,color=INK,lw=1.5,label="actual")
    ax.plot(g.ds,g.forecast,color=ORANGE,lw=2.0,marker="o",ms=3.5,label="forecast")
    ax.axvline(od,color=GREY,ls=":",lw=1); years(ax); yM(ax); ax.set_title(str(od.date()),fontsize=10)
axes[0].set_ylabel("KM"); axes[0].legend(loc="upper left")
fig.suptitle("6-step forecasts — best T2 model (DA·ETS)",fontsize=12,fontweight="semibold"); save("10_overlay")

# --- 15 CD diagram (Demšar style, matching submitted layout) ---
fr=pd.read_csv(f"{PO}/friedman_ranks.csv"); fr=fr[fr.treatment=="T2"].copy()
def splitm(full):
    for p in ("A_TD","B_BU","C_FEA","D_HYB"):
        if full.startswith(p+"_"): return p, full[len(p)+1:]
    return full, full
pm=fr.model.apply(splitm)
fr["strat"]=[a for a,_ in pm]; fr["mdl"]=[b for _,b in pm]
fr["key"]=[key(s,m) for s,m in zip(fr["strat"],fr["mdl"])]
fr=fr.sort_values("mean_rank").reset_index(drop=True)
CD=7.34; r=fr.mean_rank.values; n=len(r)
# maximal cliques: consecutive models within CD of each other
cl=[]
for i in range(n):
    j=i
    while j<n-1 and r[j+1]-r[i]<=CD: j+=1
    cl.append((i,j))
cl=[c for c in cl if c[1]>c[0]]
cl=[c for c in cl if not any(d!=c and d[0]<=c[0] and d[1]>=c[1] for d in cl)]
fig,ax=plt.subplots(figsize=(8.4,0.34*n+1.6))
ydot=-np.arange(n)                      # best at top (y=0), worst at bottom
for i in range(n):
    ax.scatter(r[i],ydot[i],s=46,color="#1F6F8B",zorder=3)
    ax.text(r[i]+0.18,ydot[i],f" {fr.key.iloc[i]}",va="center",ha="left",fontsize=10,color=INK)
# orange clique bars stacked above the dots
for row,(i,j) in enumerate(cl):
    yb=1.1+0.9*row
    ax.plot([r[i]-0.12,r[j]+0.12],[yb,yb],color=ORANGE,lw=7,solid_capstyle="round",zorder=2)
ax.set_ylim(ydot.min()-1.2, 1.1+0.9*len(cl)+0.6)
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlim(r.min()-1.3, r.max()+6.5)
ax.set_xlabel("mean rank across common origins (lower = better)")
ax.set_title(f"Critical-difference diagram — T2  (Friedman $\\chi^2$=231.7, $p$=2.1e-35; Nemenyi CD={CD:.2f}, $\\alpha$=0.10)",
             fontsize=11)
ax.grid(axis="y",visible=False); save("15_cd_diagram")

print("done:", os.listdir(OUT))
