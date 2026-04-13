"""
Project: Flood Inequality Across Brazil
Module: 13_flood_inequality_map.py
Version: v4.0 — One figure per panel, no composite layout

Each panel is saved as an independent 500 DPI PNG + PDF.

Figure 1 panels (spatial):
  fig1a_flood_inequality_score_map.png
  fig1b_quadrant_map.png
  fig1c_score_by_region.png
  fig1d_quadrant_composition_by_region.png

Figure 2 panels (mechanisms):
  fig2a_forest_plot_m4.png
  fig2b_moderation_scatter.png
  fig2c_marginal_effects.png
  fig2d_violin_by_quadrant.png
  fig2e_cohens_d_pairwise.png
  fig2f_coefficient_stability.png

Also saves: flood_inequality_brazil.shp

Author: Enner H. de Alcantara
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import mannwhitneyu, kruskal

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# =========================================================
# PATHS
# =========================================================
BASE = Path("/content/drive/MyDrive/Brazil/flood_inequality_project")

SPATIAL_PATH = BASE / "04_integrated" / "hazard_social_disaster_municipal_summary_brazil.geoparquet"
MOD_PATH     = BASE / "05_modeling"   / "moderation_results.csv"
SPA_PATH     = BASE / "05_modeling"   / "spatial_regression_results.csv"

OUTPUT_DIR = BASE / "04_integrated"
FIG_DIR    = BASE / "06_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SHP = OUTPUT_DIR / "flood_inequality_brazil.shp"

TARGET   = "disaster_observed_index"
HAZARD   = "hazard_recent_extremes_index"
SOCIAL   = "social_inequality_index"
CAPACITY = "adaptive_capacity_index"
TREND    = "hazard_trend_index"
COMPOUND = "hazard_social_disaster_compound_index"

# =========================================================
# FONT SIZES  (Nature baseline × 1.5)
# =========================================================
FS = dict(
    title    = 11.0,
    label    = 10.0,
    tick     =  9.0,
    legend   =  8.5,
    annot    =  8.5,
    annot_sm =  7.5,
    panel    = 14.0,
    caption  =  8.0,
    cb_label =  9.5,
    cb_tick  =  8.5,
    sig      =  9.5,
    bar_val  =  8.0,
)

def set_style():
    matplotlib.rcParams.update({
        "font.family"        : "serif",
        "font.serif"         : ["Times New Roman", "DejaVu Serif", "Times"],
        "font.size"          : FS["label"],
        "axes.linewidth"     : 0.8,
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.labelsize"     : FS["label"],
        "axes.titlesize"     : FS["title"],
        "axes.titleweight"   : "bold",
        "axes.titlepad"      : 8,
        "xtick.major.width"  : 0.6,
        "ytick.major.width"  : 0.6,
        "xtick.major.size"   : 3.0,
        "ytick.major.size"   : 3.0,
        "xtick.labelsize"    : FS["tick"],
        "ytick.labelsize"    : FS["tick"],
        "legend.fontsize"    : FS["legend"],
        "legend.frameon"     : True,
        "legend.framealpha"  : 0.92,
        "legend.edgecolor"   : "#C0C0C0",
        "legend.handlelength": 1.2,
        "figure.dpi"         : 72,
        "savefig.dpi"        : 500,
        "pdf.fonttype"       : 42,
        "ps.fonttype"        : 42,
    })

C = dict(
    bg="#FFFFFF", panel="#F6F6F6", text="#111122", sub="#555566",
    border="#C8C8C8", blue="#1B4F8A", red="#8B1A1A", teal="#1E6B5A",
    amber="#A0521A", gray="#888888", gold="#B8860B",
    HH="#8B1A1A", HL="#D4703A", LH="#3A7AB8", LL="#1E6B5A",
)

QUAD_FULL = {
    "HH": "High hazard, high inequality",
    "HL": "High hazard, low inequality",
    "LH": "Low hazard, high inequality",
    "LL": "Low hazard, low inequality",
}

REGION_C = {
    "North": "#4A90D9", "Northeast": "#D4803A",
    "Center-West": "#4CAF76", "Southeast": "#B03040",
    "South": "#7B5EA7",
}

FI_CMAP = LinearSegmentedColormap.from_list(
    "flood_ineq",
    ["#1E6B5A", "#74B8A2", "#F5F0D8", "#D4703A", "#8B1A1A"],
    N=512,
)

def grd(ax, axis="both"):
    ax.grid(axis=axis, linewidth=0.25, color=C["border"],
            alpha=0.85, zorder=0, linestyle="--")

def sax(ax):
    ax.set_facecolor(C["bg"])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(C["border"])

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True); sd = s.std(skipna=True, ddof=1)
    return (s - mu) / sd if (sd and sd > 0) else s * 0.0

def savep(fig, stem):
    """Save PNG + PDF and display in Colab."""
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=500, bbox_inches="tight", facecolor=C["bg"])
    fig.savefig(pdf, bbox_inches="tight", facecolor=C["bg"])
    try:
        from IPython.display import display; display(fig)
    except Exception:
        plt.show()
    plt.close(fig)
    print(f"  -> {png.name}")

def region_data(gdf_in, col, min_n=5):
    all_r = [r for r in ["North","Northeast","Center-West","Southeast","South"]
             if r in gdf_in["region_en"].values]
    out = []
    for r in all_r:
        v = gdf_in.loc[gdf_in["region_en"]==r, col].dropna().values
        if len(v) >= min_n:
            out.append((r, v, REGION_C.get(r, C["gray"])))
    return out

# =========================================================
# LOAD DATA
# =========================================================
print("Loading data ...")
gdf = gpd.read_parquet(SPATIAL_PATH)
if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)

for col in [TARGET, HAZARD, SOCIAL, CAPACITY, TREND, COMPOUND]:
    gdf[col] = pd.to_numeric(gdf.get(col), errors="coerce")

gdf = gdf.dropna(subset=[HAZARD, SOCIAL]).reset_index(drop=True)

for col in [TARGET, HAZARD, SOCIAL, CAPACITY, TREND]:
    gdf[col+"_std"] = zscore(gdf[col])
gdf["hazard_x_social"] = gdf[HAZARD+"_std"] * gdf[SOCIAL+"_std"]

if "quadrant" not in gdf.columns:
    hm = gdf[HAZARD].median(); sm = gdf[SOCIAL].median()
    gdf["quadrant"] = np.select(
        [gdf[HAZARD].ge(hm) & gdf[SOCIAL].ge(sm),
         gdf[HAZARD].ge(hm) & gdf[SOCIAL].lt(sm),
         gdf[HAZARD].lt(hm) & gdf[SOCIAL].ge(sm)],
        ["HH","HL","LH"], default="LL")

comps, wts = [], []
for col, w in [(HAZARD,0.40),(SOCIAL,0.35),(TARGET,0.25)]:
    if gdf[col].notna().sum() > 100:
        comps.append(zscore(gdf[col])*w); wts.append(w)
if comps:
    raw = sum(comps)/sum(wts)
    lo = raw.quantile(0.01); hi = raw.quantile(0.99)
    gdf["fi_score"] = ((raw-lo)/(hi-lo)*100).clip(0,100)
else:
    gdf["fi_score"] = np.nan

# Region detection with fallback to uf_sigla
_reg_candidates = [c for c in gdf.columns
                   if any(k in c.lower() for k in
                          ["region","regiao","macro","grande_reg"])]
_reg_col = None
for _c in _reg_candidates:
    if 2 <= gdf[_c].nunique() <= 10:
        _reg_col = _c; break

if _reg_col:
    _rm = {"Norte":"North","Nordeste":"Northeast",
           "Centro-Oeste":"Center-West","Sudeste":"Southeast","Sul":"South",
           "North":"North","Northeast":"Northeast",
           "Center-West":"Center-West","Southeast":"Southeast","South":"South"}
    gdf["region_en"] = (gdf[_reg_col].astype(str)
                        .map(_rm).fillna(gdf[_reg_col].astype(str)))
    print(f"  Region col: {_reg_col} | {sorted(gdf['region_en'].dropna().unique())}")
else:
    _uf2reg = {
        "AC":"North","AM":"North","AP":"North","PA":"North",
        "RO":"North","RR":"North","TO":"North",
        "AL":"Northeast","BA":"Northeast","CE":"Northeast","MA":"Northeast",
        "PB":"Northeast","PE":"Northeast","PI":"Northeast",
        "RN":"Northeast","SE":"Northeast",
        "DF":"Center-West","GO":"Center-West","MS":"Center-West","MT":"Center-West",
        "ES":"Southeast","MG":"Southeast","RJ":"Southeast","SP":"Southeast",
        "PR":"South","RS":"South","SC":"South",
    }
    if "uf_sigla" in gdf.columns:
        gdf["region_en"] = gdf["uf_sigla"].map(_uf2reg).fillna("Unknown")
        print(f"  Region from uf_sigla | {sorted(gdf['region_en'].dropna().unique())}")
    else:
        gdf["region_en"] = "Brazil"
        print("  WARNING: no region column found")

mod_df = pd.read_csv(MOD_PATH) if MOD_PATH.exists() else pd.DataFrame()
spa_df = pd.read_csv(SPA_PATH) if SPA_PATH.exists() else pd.DataFrame()
print(f"  {len(gdf):,} municipalities | fi mean={gdf['fi_score'].mean():.1f}")

# =========================================================
# SAVE SHAPEFILE
# =========================================================
print("Saving shapefile ...")
SHP_MAP = {"mun_code":"mun_code","mun_name":"mun_name","uf_sigla":"uf_sigla",
           HAZARD:"haz_idx", SOCIAL:"soc_idx", TARGET:"dis_idx",
           COMPOUND:"compound", "fi_score":"fi_score",
           "quadrant":"quadrant", "region_en":"region"}
keep  = {k:v for k,v in SHP_MAP.items() if k in gdf.columns}
shpdf = gdf[[*keep.keys(),"geometry"]].rename(columns=keep).copy()
shpdf["geometry"] = shpdf.geometry.buffer(0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    shpdf.to_file(OUTPUT_SHP, driver="ESRI Shapefile", encoding="utf-8")
print(f"  Saved -> {OUTPUT_SHP.name}")

# =========================================================
# M4 MODEL  (computed once, reused across panels)
# =========================================================
df_m = gdf[[TARGET+"_std", HAZARD+"_std", SOCIAL+"_std",
            CAPACITY+"_std", TREND+"_std", "hazard_x_social",
            TARGET, HAZARD, SOCIAL, "quadrant"]].copy()
df_m = df_m.dropna(subset=[TARGET+"_std", HAZARD+"_std", SOCIAL+"_std"])

if mod_df.empty:
    import statsmodels.api as _sm
    _y = df_m[TARGET+"_std"]
    _X = _sm.add_constant(df_m[[HAZARD+"_std", SOCIAL+"_std",
                                  CAPACITY+"_std", TREND+"_std",
                                  "hazard_x_social"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _m4 = _sm.OLS(_y, _X).fit(cov_type="HC3")
    m4_params = _m4.params; m4_p = _m4.pvalues
    m4_ci     = _m4.conf_int()
    m4_r2     = _m4.rsquared; m4_r2adj = _m4.rsquared_adj
    m4_n      = int(_m4.nobs)
else:
    _sub      = mod_df[mod_df["model"]=="M4 Interaction"].set_index("variable")
    m4_params = _sub["coef"]; m4_p = _sub["p"]
    m4_ci     = _sub[["ci_low","ci_high"]].rename(columns={"ci_low":0,"ci_high":1})
    m4_r2     = float(_sub["r2"].iloc[0])
    m4_r2adj  = float(_sub["r2_adj"].iloc[0])
    m4_n      = int(_sub["n"].iloc[0])

b_h  = float(m4_params.get(HAZARD+"_std",   0))
b_hx = float(m4_params.get("hazard_x_social",0))
b_s  = float(m4_params.get(SOCIAL+"_std",   0))
b0   = float(m4_params.get("const",          0))

quads = ["HH","HL","LH","LL"]
qcols = [C[q] for q in quads]
dists = {q: gdf.loc[gdf["quadrant"]==q, TARGET].dropna() for q in quads}

print("Model M4 ready.")

# =========================================================
# ── FIG 1a  Flood inequality score map ───────────────────
# =========================================================
def fig1a():
    set_style()
    fig, ax = plt.subplots(figsize=(8, 9))
    fig.patch.set_facecolor(C["bg"]); ax.set_facecolor(C["bg"])

    vmin = gdf["fi_score"].quantile(0.02)
    vmax = gdf["fi_score"].quantile(0.98)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    gdf.plot(column="fi_score", cmap=FI_CMAP, norm=norm,
             linewidth=0.03, edgecolor="white", ax=ax,
             missing_kwds={"color":"#DDDDDD"}, legend=False)

    if "triple_burden_flag" in gdf.columns:
        hot = gdf[gdf["triple_burden_flag"]==1]
        if len(hot): hot.plot(ax=ax, color="none", edgecolor="#111122",
                              linewidth=0.7, zorder=5)

    sm = plt.cm.ScalarMappable(cmap=FI_CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                      fraction=0.032, pad=0.02, shrink=0.80,
                      ticks=[0,25,50,75,100])
    cb.set_label("Flood inequality score (0–100)",
                 fontsize=FS["cb_label"], labelpad=5)
    cb.ax.set_xticklabels(["0\nVery low","25","50","75","100\nVery high"],
                           fontsize=FS["cb_tick"])
    cb.ax.tick_params(width=0.6, length=3)

    n_hot = int((gdf["fi_score"]>=75).sum())
    n_tot = int(gdf["fi_score"].notna().sum())
    ax.text(0.97, 0.03,
            f"Score \u2265 75:  {n_hot:,} municipalities ({n_hot/n_tot*100:.1f}%)\n"
            f"Outlined = triple burden hotspots",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS["annot_sm"], linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="white",
                      ec=C["border"], lw=0.6, alpha=0.94))

    ax.axis("off")
    ax.set_title("Flood inequality score across Brazilian municipalities",
                 fontsize=FS["title"], pad=10)
    fig.text(0.5, 0.01,
             "Weighted composite: hazard (40%), social inequality (35%), "
             "disaster impact (25%)  |  n = 5,573 municipalities",
             ha="center", fontsize=FS["caption"], color=C["sub"])
    savep(fig, "fig1a_flood_inequality_score_map")


# =========================================================
# ── FIG 1b  Quadrant map ─────────────────────────────────
# =========================================================
def fig1b():
    set_style()
    fig, ax = plt.subplots(figsize=(8, 9))
    fig.patch.set_facecolor(C["bg"]); ax.set_facecolor(C["bg"])

    gdf["_qc"] = gdf["quadrant"].map({q:C[q] for q in quads}).fillna("#DDDDDD")
    gdf.plot(color=gdf["_qc"], ax=ax, linewidth=0.03, edgecolor="white")

    lp = [mpatches.Patch(color=C[q],
                          label=f"{q}  {QUAD_FULL[q]}  "
                                f"(n={int((gdf['quadrant']==q).sum()):,})")
          for q in quads]
    ax.legend(handles=lp, fontsize=FS["legend"], loc="lower left",
              framealpha=0.94, edgecolor=C["border"],
              handlelength=1.3, borderpad=0.9, labelspacing=0.6)
    ax.axis("off")
    ax.set_title("Hazard \u00d7 social inequality quadrant classification",
                 fontsize=FS["title"], pad=10)
    fig.text(0.5, 0.01,
             "Quadrant based on median split of hazard and social inequality indices  |  "
             "n = 5,573 municipalities",
             ha="center", fontsize=FS["caption"], color=C["sub"])
    savep(fig, "fig1b_quadrant_map")


# =========================================================
# ── FIG 1c  Score by region (boxplot + strip) ────────────
# =========================================================
def fig1c():
    set_style()
    rd = region_data(gdf, "fi_score", min_n=5)
    if not rd:
        print("  fig1c: no regional data — skipping")
        return

    regs  = [r for r,v,c in rd]
    vdata = [v for r,v,c in rd]
    rcols = [c for r,v,c in rd]
    rng   = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(C["bg"]); sax(ax)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)

    bp = ax.boxplot(vdata, positions=range(len(regs)),
                     patch_artist=True, notch=False, widths=0.50,
                     showfliers=True,
                     flierprops=dict(marker=".", markersize=2.5,
                                     color=C["gray"], alpha=0.30),
                     medianprops=dict(color=C["text"], lw=1.8),
                     whiskerprops=dict(color=C["gray"], lw=1.0),
                     capprops=dict(color=C["gray"], lw=1.0))
    for patch, col in zip(bp["boxes"], rcols):
        patch.set_facecolor(mcolors.to_rgba(col, 0.42))
        patch.set_edgecolor(col); patch.set_linewidth(1.1)
    for i, (vals, col) in enumerate(zip(vdata, rcols)):
        ax.scatter(i + rng.uniform(-0.10, 0.10, len(vals)),
                   vals, s=1.5, alpha=0.12, color=col,
                   linewidths=0, zorder=2)

    ax.set_xticks(range(len(regs)))
    ax.set_xticklabels(regs, fontsize=FS["tick"])
    ax.set_ylabel("Flood inequality score", fontsize=FS["label"], labelpad=8)
    ax.set_ylim(-5, 125)
    ax.set_title("Flood inequality score by macro-region",
                 fontsize=FS["title"], pad=10)
    grd(ax, "y")

    kw_stat, kw_p = kruskal(*vdata)
    p_str = "< 0.001" if kw_p<0.001 else f"= {kw_p:.3f}"
    ax.text(0.97, 0.97,
            f"Kruskal–Wallis\nH = {kw_stat:.1f},  p {p_str}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FS["annot"], linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", fc=C["panel"],
                      ec=C["border"], lw=0.6))
    savep(fig, "fig1c_score_by_region")


# =========================================================
# ── FIG 1d  Quadrant composition by region (stacked bar) ─
# =========================================================
def fig1d():
    set_style()
    rd2 = region_data(gdf, "fi_score", min_n=1)
    if not rd2:
        print("  fig1d: no regional data — skipping")
        return

    regs2  = [r for r,v,c in rd2]
    x_r    = np.arange(len(regs2))
    bottom = np.zeros(len(regs2))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(C["bg"]); sax(ax)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)

    for q, col in zip(quads, qcols):
        vals = np.array([
            (gdf.loc[gdf["region_en"]==r,"quadrant"]==q).sum() /
            max((gdf["region_en"]==r).sum(), 1)
            for r in regs2])
        ax.bar(x_r, vals, bottom=bottom, color=col,
               alpha=0.88, edgecolor="white", lw=0.4,
               label=q, width=0.62)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.07:
                ax.text(xi, b+v/2, f"{v*100:.0f}%",
                        ha="center", va="center",
                        fontsize=FS["bar_val"],
                        color="white", fontweight="bold")
        bottom += vals

    ax.set_xticks(x_r)
    ax.set_xticklabels(regs2, fontsize=FS["tick"])
    ax.set_ylabel("Fraction of municipalities",
                  fontsize=FS["label"], labelpad=8)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x,_: f"{x*100:.0f}%"))
    ax.set_ylim(0, 1.08)
    ax.set_title("Quadrant composition by macro-region",
                 fontsize=FS["title"], pad=10)
    grd(ax, "y")
    lp = [mpatches.Patch(color=C[q], label=f"{q}  {QUAD_FULL[q]}") for q in quads]
    ax.legend(handles=lp, fontsize=FS["legend"]-1, loc="upper right",
              ncol=1, framealpha=0.93)
    savep(fig, "fig1d_quadrant_composition_by_region")


# =========================================================
# ── FIG 2a  Forest plot M4 ───────────────────────────────
# =========================================================
def fig2a():
    set_style()
    VAR_LABELS = {
        HAZARD+"_std"  : "Hydroclimatic hazard",
        SOCIAL+"_std"  : "Social inequality",
        CAPACITY+"_std": "Adaptive capacity (\u2212)",
        TREND+"_std"   : "Hazard trend",
        "hazard_x_social": "Hazard \u00d7 Inequality  (interaction term)",
    }
    PLOT_VARS = [HAZARD+"_std", SOCIAL+"_std",
                 CAPACITY+"_std", TREND+"_std", "hazard_x_social"]

    rows = []
    for v in PLOT_VARS:
        if v in m4_params.index:
            ci = m4_ci.loc[v]
            rows.append({"label":VAR_LABELS.get(v,v),
                          "coef":float(m4_params[v]),
                          "ci_low":float(ci.iloc[0]),
                          "ci_hi":float(ci.iloc[1]),
                          "p":float(m4_p[v])})
    fdf = pd.DataFrame(rows).sort_values("coef").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(C["bg"]); sax(ax)
    fig.subplots_adjust(left=0.30, right=0.88, top=0.85, bottom=0.18)

    for i, row in fdf.iterrows():
        col = C["red"] if row["p"]<0.05 else C["gray"]
        ax.barh(i, row["coef"], color=col,
                alpha=0.82, height=0.52, edgecolor="white", lw=0.35)
        ax.errorbar(row["coef"], i,
                    xerr=[[row["coef"]-row["ci_low"]],
                          [row["ci_hi"]-row["coef"]]],
                    fmt="none", ecolor=col,
                    elinewidth=1.2, capsize=4, capthick=1.2)
        sig = ("***" if row["p"]<0.001 else "**" if row["p"]<0.01
               else "*" if row["p"]<0.05 else "ns")
        ax.text(1.02, i, sig, transform=ax.get_yaxis_transform(),
                va="center", fontsize=FS["sig"], clip_on=False,
                color=C["red"] if sig!="ns" else C["gray"])

    ax.set_yticks(range(len(fdf)))
    ax.set_yticklabels(fdf["label"].values, fontsize=FS["label"])
    ax.axvline(0, color=C["text"], lw=1.0, ls="--", zorder=4)
    ax.set_xlabel(
        "Standardized coefficient (\u03b2)  with 95% CI  [HC3-robust SE]",
        fontsize=FS["label"], labelpad=8)
    ax.set_title(
        f"Interaction model M4   "
        f"(R\u00b2 = {m4_r2:.3f},  R\u00b2adj = {m4_r2adj:.3f},  "
        f"n = {m4_n:,})",
        fontsize=FS["title"], pad=10)
    grd(ax, "x")
    fig.text(0.04, 0.02,
             "*** p < 0.001   ** p < 0.01   * p < 0.05   ns not significant",
             fontsize=FS["annot_sm"], color=C["gray"])
    savep(fig, "fig2a_forest_plot_m4")


# =========================================================
# ── FIG 2b  Moderation scatter ───────────────────────────
# =========================================================
def fig2b():
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor(C["bg"]); sax(ax)
    fig.subplots_adjust(left=0.13, right=0.88, top=0.88, bottom=0.14)

    sc = ax.scatter(df_m[HAZARD+"_std"], df_m[TARGET+"_std"],
                    c=df_m[SOCIAL+"_std"], cmap="RdYlBu_r",
                    s=3, alpha=0.35, linewidths=0, zorder=3)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.03)
    cb.set_label("Social inequality (std)",
                 fontsize=FS["cb_label"], labelpad=6)
    cb.ax.tick_params(labelsize=FS["cb_tick"])

    h_rng = np.linspace(df_m[HAZARD+"_std"].min(),
                        df_m[HAZARD+"_std"].max(), 150)
    for lbl, sq, col, ls in [
        ("Low ineq. (P10)",  df_m[SOCIAL+"_std"].quantile(0.10), C["teal"], "-"),
        ("Med. ineq. (P50)", df_m[SOCIAL+"_std"].quantile(0.50), C["gold"], "--"),
        ("High ineq. (P90)", df_m[SOCIAL+"_std"].quantile(0.90), C["red"],  "-"),
    ]:
        ax.plot(h_rng, b0+b_h*h_rng+b_s*sq+b_hx*h_rng*sq,
                color=col, lw=2.0, ls=ls, label=lbl, zorder=5)

    ax.legend(fontsize=FS["legend"], loc="upper left",
              framealpha=0.93, edgecolor=C["border"])
    ax.set_xlabel("Hydroclimatic hazard (std)",
                  fontsize=FS["label"], labelpad=7)
    ax.set_ylabel("Disaster impact (std)",
                  fontsize=FS["label"], labelpad=7)
    ax.set_title("Moderation: hazard effect by inequality level",
                 fontsize=FS["title"], pad=10)
    grd(ax)
    savep(fig, "fig2b_moderation_scatter")


# =========================================================
# ── FIG 2c  Marginal effects ─────────────────────────────
# =========================================================
def fig2c():
    set_style()
    lvls = {"Low (P10)" : df_m[SOCIAL+"_std"].quantile(0.10),
            "Med. (P50)": df_m[SOCIAL+"_std"].quantile(0.50),
            "High (P90)": df_m[SOCIAL+"_std"].quantile(0.90)}
    me_vals = {lbl: b_h+b_hx*sq for lbl,sq in lvls.items()}

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    fig.patch.set_facecolor(C["bg"]); sax(ax)
    fig.subplots_adjust(left=0.18, right=0.82, top=0.85, bottom=0.18)

    bars = ax.barh(list(me_vals.keys()), list(me_vals.values()),
                   color=[C["teal"], C["gold"], C["red"]],
                   alpha=0.88, edgecolor="white", lw=0.4, height=0.42)
    ax.axvline(0, color=C["text"], lw=0.9, ls="--")

    xmax = max(abs(v) for v in me_vals.values())
    ax.set_xlim(-xmax*0.20, xmax*1.60)

    for bar, v in zip(bars, me_vals.values()):
        ax.text(v + xmax*0.06,
                bar.get_y()+bar.get_height()/2,
                f"{v:+.3f}", va="center", ha="left",
                fontsize=FS["annot"], fontweight="bold")

    ax.set_xlabel("Marginal effect of hazard on disaster impact",
                  fontsize=FS["label"], labelpad=8)
    ax.set_title("Marginal effects of hazard\nat three inequality levels",
                 fontsize=FS["title"], pad=10)
    ax.tick_params(axis="y", labelsize=FS["tick"])
    grd(ax, "x")

    p_int = float(m4_p.get("hazard_x_social", np.nan))
    b_int = float(m4_params.get("hazard_x_social", np.nan))
    p_str = "< 0.001" if p_int<0.001 else f"= {p_int:.3f}"
    ax.text(0.97, 0.06,
            f"Interaction \u03b2 = {b_int:.3f}\np {p_str}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS["annot"], linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", fc=C["panel"],
                      ec=C["border"], lw=0.6))
    savep(fig, "fig2c_marginal_effects")


# =========================================================
# ── FIG 2d  Violin by quadrant ───────────────────────────
# =========================================================
def fig2d():
    set_style()
    vdata = [dists[q].values for q in quads]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(C["bg"]); sax(ax)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.20)

    parts = ax.violinplot(vdata, positions=range(4),
                           showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], qcols):
        pc.set_facecolor(mcolors.to_rgba(col, 0.45))
        pc.set_edgecolor(col); pc.set_linewidth(1.1)
    parts["cmedians"].set_color(C["text"])
    parts["cmedians"].set_linewidth(1.8)

    rng2 = np.random.default_rng(42)
    for i, (q, col) in enumerate(zip(quads, qcols)):
        v = dists[q].values
        ax.scatter(i+rng2.uniform(-0.09,0.09,len(v)),
                   v, s=1.2, alpha=0.10, color=col,
                   linewidths=0, zorder=2)

    ax.set_xticks(range(4))
    ax.set_xticklabels(
        ["HH\nHigh hazard\nHigh ineq.",
         "HL\nHigh hazard\nLow ineq.",
         "LH\nLow hazard\nHigh ineq.",
         "LL\nLow hazard\nLow ineq."],
        fontsize=FS["tick"]-1)
    ax.set_ylabel("Disaster impact index",
                  fontsize=FS["label"], labelpad=8)
    ax.set_title("Disaster impact by hazard \u00d7 inequality quadrant",
                 fontsize=FS["title"], pad=10)

    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi*1.30)
    grd(ax, "y")

    kw_stat, kw_p = kruskal(*[v for v in vdata if len(v)>=5])
    p_str = "< 0.001" if kw_p<0.001 else f"= {kw_p:.3f}"
    ax.text(0.97, 0.97,
            f"Kruskal–Wallis\nH = {kw_stat:.1f},  p {p_str}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FS["annot"], linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", fc=C["panel"],
                      ec=C["border"], lw=0.6))
    savep(fig, "fig2d_violin_by_quadrant")


# =========================================================
# ── FIG 2e  Cohen's d pairwise ───────────────────────────
# =========================================================
def fig2e():
    set_style()
    pairs = [("HH vs LL","HH","LL"),("HH vs HL","HH","HL"),
             ("LH vs LL","LH","LL"),("HL vs LL","HL","LL")]
    ef_rows = []
    for lbl, q1, q2 in pairs:
        a, b_ = dists[q1].values, dists[q2].values
        pool  = np.sqrt((a.std()**2+b_.std()**2)/2)
        d     = (a.mean()-b_.mean())/pool if pool>0 else np.nan
        _, p  = mannwhitneyu(a, b_, alternative="two-sided")
        ef_rows.append({"label":lbl, "d":d, "p":p})
    edf = pd.DataFrame(ef_rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(C["bg"]); sax(ax)
    fig.subplots_adjust(left=0.18, right=0.78, top=0.88, bottom=0.18)

    bcols = [C["red"] if p<0.05 else C["gray"] for p in edf["p"]]
    bars  = ax.barh(edf["label"], edf["d"], color=bcols,
                    alpha=0.85, edgecolor="white", lw=0.4, height=0.42)
    ax.axvline(0, color=C["text"], lw=0.9, ls="--")

    xmax_d = edf["d"].max()
    ax.set_xlim(-0.04, xmax_d*1.65)

    for thresh, lbl_ in [(0.2,"small"),(0.5,"medium"),(0.8,"large")]:
        ax.axvline(thresh, color=C["border"], lw=0.7, ls=":", zorder=1)
        ax.text(thresh, -0.62, lbl_, ha="center",
                fontsize=FS["annot_sm"]-0.5, color=C["gray"])

    for bar, row in zip(bars, edf.itertuples()):
        sig = ("***" if row.p<0.001 else "**" if row.p<0.01
               else "*" if row.p<0.05 else "ns")
        ax.text(row.d + xmax_d*0.06,
                bar.get_y()+bar.get_height()/2,
                f"d = {row.d:.2f}  {sig}",
                va="center", fontsize=FS["annot"],
                color=C["red"] if sig!="ns" else C["gray"])

    ax.set_xlabel("Cohen's d  (effect size)",
                  fontsize=FS["label"], labelpad=8)
    ax.set_title("Pairwise effect sizes  (Mann–Whitney U)",
                 fontsize=FS["title"], pad=10)
    ax.tick_params(axis="y", labelsize=FS["tick"])
    grd(ax, "x")

    _, p_amp = mannwhitneyu(dists["HH"].values, dists["HL"].values,
                             alternative="greater")
    p_amp_str = "< 0.001" if p_amp<0.001 else f"= {p_amp:.4f}"
    ax.text(0.97, 0.97,
            f"Social amplification (HH > HL, one-sided):\np {p_amp_str}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FS["annot"], color=C["red"], linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="white",
                      ec=C["border"], lw=0.6))
    savep(fig, "fig2e_cohens_d_pairwise")


# =========================================================
# ── FIG 2f  Coefficient stability OLS/SLM/SEM ────────────
# =========================================================
def fig2f():
    if spa_df.empty:
        print("  fig2f: spatial results not found — skipping")
        return
    set_style()
    vmap_s = {HAZARD+"_std":"Hazard", SOCIAL+"_std":"Social",
              "hazard_x_social":"H\u00d7I"}
    pvars_s = [v for v in vmap_s if v in spa_df["variable"].values]
    if not pvars_s:
        print("  fig2f: no matching variables in spatial results — skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor(C["bg"]); sax(ax)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.18)

    mn_list = ["OLS","SLM","SEM"]
    mc_list = [C["blue"], C["amber"], C["teal"]]
    xb      = np.arange(len(pvars_s)); w = 0.22

    for mi, (mn, mc) in enumerate(zip(mn_list, mc_list)):
        sub = spa_df[spa_df["model"]==mn]
        cs  = [sub.loc[sub["variable"]==v,"coef"].values[0]
               if len(sub.loc[sub["variable"]==v]) else np.nan
               for v in pvars_s]
        ses = [sub.loc[sub["variable"]==v,"se"].values[0]
               if len(sub.loc[sub["variable"]==v]) else np.nan
               for v in pvars_s]
        cs = np.array(cs); ses = np.array(ses)
        ax.bar(xb+(mi-1)*w, cs, w, label=mn, color=mc,
               alpha=0.85, edgecolor="white", lw=0.35)
        ax.errorbar(xb+(mi-1)*w, cs, yerr=1.96*ses,
                    fmt="none", ecolor=mc, elinewidth=1.0, capsize=3)

    ax.axhline(0, color=C["text"], lw=0.8, ls="--")
    ax.set_xticks(xb)
    ax.set_xticklabels([vmap_s[v] for v in pvars_s], fontsize=FS["tick"])
    ax.set_ylabel("Standardized coefficient",
                  fontsize=FS["label"], labelpad=8)
    ax.set_title("Coefficient stability across spatial models",
                 fontsize=FS["title"], pad=10)
    ax.legend(fontsize=FS["legend"], loc="upper right", framealpha=0.93)
    grd(ax, "y")
    savep(fig, "fig2f_coefficient_stability")


# =========================================================
# RUN ALL
# =========================================================
print("\n" + "="*65)
print("  Module 13 v4.0 — Individual panel figures")
print("="*65 + "\n")

print("Figure 1 panels:")
fig1a()
fig1b()
fig1c()
fig1d()

print("\nFigure 2 panels:")
fig2a()
fig2b()
fig2c()
fig2d()
fig2e()
fig2f()

print("\n" + "="*65)
print("  All panels saved to:", FIG_DIR)
print("="*65 + "\n")
