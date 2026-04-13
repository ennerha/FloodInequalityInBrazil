"""
Project: Flood Inequality Across Brazil

Module: 11_model_hazard_inequality_disaster.py
Version: v3.0 — Scientific inference edition

Scientific question:
  Does social inequality amplify the impact of hydroclimatic hazard
  on observed disaster outcomes across Brazilian municipalities?

Analytical strategy (publication-grade):
  1. Moderation analysis — OLS with hazard × social_inequality
     interaction term as the central test of amplification
  2. Spatial regression — Spatial Lag Model (SLM) and Spatial Error
     Model (SEM) via PySAL/spreg to correct for spatial autocorrelation
     and produce robust associative coefficients
  3. Quadrant disparity analysis — comparing disaster impact distributions
     across hazard × inequality quadrants (high/low × high/low)
     to isolate the social amplification effect

Publication figures (500 DPI, Nature/Science style):
  fig11a_moderation_analysis.png     — 5-panel
  fig11b_spatial_regression.png      — 5-panel
  fig11c_quadrant_disparity.png      — 5-panel

Author: Enner H. de Alcântara
Language: English
"""

# =========================================================
# 1. IMPORTS
# =========================================================
import os
import sys
import json
import logging
import warnings
import subprocess
from datetime import datetime
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal

import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

def _ensure(pkg, install_name=None):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q",
             install_name or pkg])

_ensure("tqdm")
_ensure("libpysal", "libpysal")
_ensure("esda",     "esda")
_ensure("spreg",    "spreg")

from tqdm.auto import tqdm
import libpysal
from libpysal.weights import Queen, KNN
import esda
from esda.moran import Moran, Moran_Local
import spreg

# =========================================================
# 2. PATHS AND CONSTANTS
# =========================================================
BASE_PATH = Path("/content/drive/MyDrive/Brazil/flood_inequality_project")

LOG_PATH     = BASE_PATH / "07_logs"    / "11_model_hazard_inequality_disaster.log"
CATALOG_PATH = BASE_PATH / "08_catalog" / "catalog.csv"
CONFIG_PATH  = BASE_PATH / "00_config"  / "config.json"

INPUT_PATH = (
    BASE_PATH / "04_integrated"
    / "hazard_social_disaster_municipal_summary_brazil.geoparquet"
)

OUTPUT_DIR = BASE_PATH / "05_modeling"
FIG_DIR    = BASE_PATH / "06_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

OUT_MODERATION = OUTPUT_DIR / "moderation_results.csv"
OUT_SPATIAL    = OUTPUT_DIR / "spatial_regression_results.csv"
OUT_QUADRANT   = OUTPUT_DIR / "quadrant_disparity_results.csv"
OUT_META       = OUTPUT_DIR / "11_model_hazard_inequality_disaster.meta.json"

FIG_A_PNG = FIG_DIR / "fig11a_moderation_analysis.png"
FIG_A_PDF = FIG_DIR / "fig11a_moderation_analysis.pdf"
FIG_B_PNG = FIG_DIR / "fig11b_spatial_regression.png"
FIG_B_PDF = FIG_DIR / "fig11b_spatial_regression.pdf"
FIG_C_PNG = FIG_DIR / "fig11c_quadrant_disparity.png"
FIG_C_PDF = FIG_DIR / "fig11c_quadrant_disparity.pdf"

TARGET   = "disaster_observed_index"
HAZARD   = "hazard_recent_extremes_index"
SOCIAL   = "social_inequality_index"
CAPACITY = "adaptive_capacity_index"
TREND    = "hazard_trend_index"
RANDOM_STATE = 42

# =========================================================
# 3. LOGGING
# =========================================================
logging.basicConfig(
    filename=str(LOG_PATH), filemode="a",
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
)
_logger = logging.getLogger("module_11_v3")

def log(msg, level="INFO"):
    _logger.info(f"[{level}] {msg}")
    if level in ("WARNING", "ERROR", "SUMMARY"):
        print(f"[{level}] {msg}")

# =========================================================
# 4. PUBLICATION STYLE
# =========================================================
def set_pub_style():
    matplotlib.rcParams.update({
        "font.family"      : "serif",
        "font.serif"       : ["Times New Roman", "DejaVu Serif", "Times"],
        "font.size"        : 7,
        "axes.linewidth"   : 0.6,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.labelsize"   : 7,
        "axes.titlesize"   : 7.5,
        "axes.titleweight" : "bold",
        "axes.titlepad"    : 5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size" : 2.5,
        "ytick.major.size" : 2.5,
        "xtick.labelsize"  : 6,
        "ytick.labelsize"  : 6,
        "legend.fontsize"  : 5.5,
        "legend.frameon"   : True,
        "legend.framealpha": 0.9,
        "legend.edgecolor" : "#CCCCCC",
        "figure.dpi"       : 72,
        "savefig.dpi"      : 500,
        "pdf.fonttype"     : 42,
        "ps.fonttype"      : 42,
    })

C = {
    "bg"    : "#FFFFFF",
    "panel" : "#F7F7F7",
    "text"  : "#1A1A2E",
    "sub"   : "#555577",
    "border": "#CCCCCC",
    "blue"  : "#1D4E89",
    "red"   : "#9B2226",
    "teal"  : "#2A7F6F",
    "amber" : "#B5621B",
    "purple": "#5C4374",
    "gray"  : "#888888",
    "HH"    : "#9B2226",
    "HL"    : "#E07B39",
    "LH"    : "#4A7CB5",
    "LL"    : "#2A7F6F",
}

def plabel(ax, letter, x=0.03, y=0.97):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top",
            color=C["text"], fontfamily="serif")

def grid_style(ax, axis="both"):
    ax.grid(axis=axis, linewidth=0.22, color=C["border"],
            alpha=0.8, zorder=0, linestyle="--")

def save_fig(fig, png_path, pdf_path):
    fig.savefig(png_path, dpi=500, bbox_inches="tight", facecolor=C["bg"])
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=C["bg"])
    try:
        from IPython.display import display
        display(fig)
    except Exception:
        plt.show()
    plt.close(fig)

# =========================================================
# 5. LOAD AND PREPARE
# =========================================================
def load_and_prepare() -> gpd.GeoDataFrame:
    log("Loading input ...", "SUMMARY")
    gdf = gpd.read_parquet(INPUT_PATH)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    needed = [TARGET, HAZARD, SOCIAL, CAPACITY, TREND]
    for col in needed:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
        else:
            gdf[col] = np.nan

    gdf = gdf.dropna(subset=[TARGET, HAZARD, SOCIAL]).reset_index(drop=True)

    for col in [TARGET, HAZARD, SOCIAL, CAPACITY, TREND]:
        mu = gdf[col].mean(skipna=True)
        sd = gdf[col].std(skipna=True, ddof=1)
        gdf[f"{col}_std"] = (gdf[col] - mu) / sd if (sd and sd > 0) else 0.0

    gdf["hazard_x_social"] = gdf[f"{HAZARD}_std"] * gdf[f"{SOCIAL}_std"]

    h_med = gdf[HAZARD].median()
    s_med = gdf[SOCIAL].median()
    h, s  = gdf[HAZARD], gdf[SOCIAL]
    gdf["quadrant"] = np.select(
        [h.ge(h_med) & s.ge(s_med),
         h.ge(h_med) & s.lt(s_med),
         h.lt(h_med) & s.ge(s_med)],
        ["HH", "HL", "LH"],
        default="LL",
    )

    reg_col = next((c for c in gdf.columns
                    if "region" in c.lower() and gdf[c].dtype == object), None)
    if reg_col:
        mapping = {
            "Norte": "North", "Nordeste": "Northeast",
            "Centro-Oeste": "Center-West",
            "Sudeste": "Southeast", "Sul": "South",
        }
        gdf["region_en"] = gdf[reg_col].map(mapping).fillna(gdf[reg_col])
    else:
        gdf["region_en"] = "Brazil"

    log(f"Ready: {len(gdf):,} municipalities | "
        f"quadrants: {gdf['quadrant'].value_counts().to_dict()}", "SUMMARY")
    return gdf

# =========================================================
# 6. MODERATION ANALYSIS
# =========================================================
def run_moderation(gdf: gpd.GeoDataFrame) -> dict:
    log("Moderation analysis ...", "SUMMARY")

    cols = [f"{TARGET}_std", f"{HAZARD}_std", f"{SOCIAL}_std",
            f"{CAPACITY}_std", f"{TREND}_std", "hazard_x_social"]
    df   = pd.DataFrame(gdf[cols].dropna())
    y    = df[f"{TARGET}_std"]

    specs = {
        "M1 Hazard only" : [f"{HAZARD}_std"],
        "M2 Social only" : [f"{SOCIAL}_std"],
        "M3 Additive"    : [f"{HAZARD}_std", f"{SOCIAL}_std",
                            f"{CAPACITY}_std", f"{TREND}_std"],
        "M4 Interaction" : [f"{HAZARD}_std", f"{SOCIAL}_std",
                            f"{CAPACITY}_std", f"{TREND}_std",
                            "hazard_x_social"],
    }

    results, rows = {}, []
    for name, preds in specs.items():
        X = sm.add_constant(df[preds])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(y, X).fit(cov_type="HC3")
        results[name] = res
        for var in preds + ["const"]:
            ci = res.conf_int()
            rows.append({
                "model"  : name, "variable": var,
                "coef"   : res.params.get(var, np.nan),
                "se"     : res.bse.get(var, np.nan),
                "t"      : res.tvalues.get(var, np.nan),
                "p"      : res.pvalues.get(var, np.nan),
                "ci_low" : ci.loc[var, 0] if var in ci.index else np.nan,
                "ci_high": ci.loc[var, 1] if var in ci.index else np.nan,
                "r2"     : res.rsquared,
                "r2_adj" : res.rsquared_adj,
                "n"      : int(res.nobs),
            })

    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(OUT_MODERATION, index=False)

    m4   = results["M4 Interaction"]
    b_h  = m4.params.get(f"{HAZARD}_std", 0)
    b_hx = m4.params.get("hazard_x_social", 0)
    marginal = {
        "Low inequality (P10)"   : b_h + b_hx * df[f"{SOCIAL}_std"].quantile(0.10),
        "Medium inequality (P50)": b_h + b_hx * df[f"{SOCIAL}_std"].quantile(0.50),
        "High inequality (P90)"  : b_h + b_hx * df[f"{SOCIAL}_std"].quantile(0.90),
    }

    log(f"M4 R²={m4.rsquared:.3f} | "
        f"interaction p={m4.pvalues.get('hazard_x_social', np.nan):.4f}", "SUMMARY")
    return {"results": results, "coef_df": coef_df,
            "marginal": marginal, "m4": m4, "df": df}

# =========================================================
# 7. SPATIAL REGRESSION
# =========================================================
def run_spatial_regression(gdf: gpd.GeoDataFrame) -> dict:
    log("Building spatial weights ...", "SUMMARY")
    gdf_proj = gdf.to_crs("EPSG:5880").copy()

    cols    = [f"{TARGET}_std", f"{HAZARD}_std", f"{SOCIAL}_std",
               f"{CAPACITY}_std", f"{TREND}_std", "hazard_x_social"]
    mask    = gdf[cols].notna().all(axis=1)
    gdf_v   = gdf.iloc[np.where(mask)[0]].reset_index(drop=True)
    gdf_p_v = gdf_proj.iloc[np.where(mask)[0]].reset_index(drop=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            W = Queen.from_dataframe(gdf_p_v, silence_warnings=True)
        except Exception:
            W = KNN.from_dataframe(gdf_p_v, k=6, silence_warnings=True)
    W.transform = "r"

    y      = gdf_v[f"{TARGET}_std"].values
    X_cols = [f"{HAZARD}_std", f"{SOCIAL}_std",
              f"{CAPACITY}_std", f"{TREND}_std", "hazard_x_social"]
    X      = gdf_v[X_cols].values

    log("Running OLS, SLM, SEM ...", "SUMMARY")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = spreg.OLS(y.reshape(-1,1), X, w=W,
                        name_y=TARGET, name_x=X_cols,
                        spat_diag=True, moran=True)
        slm = spreg.ML_Lag(y.reshape(-1,1), X, w=W,
                           name_y=TARGET, name_x=X_cols)
        sem = spreg.ML_Error(y.reshape(-1,1), X, w=W,
                             name_y=TARGET, name_x=X_cols)

    moran_y = Moran(y, W)
    moran_r = Moran(ols.u.flatten(), W)
    lisa    = Moran_Local(y, W, permutations=499, seed=RANDOM_STATE)

    rows, var_labels = [], ["const"] + X_cols
    for mname, res in [("OLS", ols), ("SLM", slm), ("SEM", sem)]:
        betas = res.betas.flatten()
        ses   = np.sqrt(np.diag(res.vm)) if res.vm is not None else [np.nan]*len(betas)
        for i, (b, s) in enumerate(zip(betas, ses)):
            lbl = var_labels[i] if i < len(var_labels) else "spatial_param"
            rows.append({
                "model"   : mname, "variable": lbl,
                "coef"    : float(b), "se": float(s),
                "z"       : float(b/s) if s > 0 else np.nan,
                "p"       : float(2*(1 - stats.norm.cdf(abs(b/s)))) if s > 0 else np.nan,
            })

    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(OUT_SPATIAL, index=False)

    log(f"SLM ρ={slm.betas[-1][0]:.4f} | SEM λ={sem.betas[-1][0]:.4f} | "
        f"Moran I(resid)={moran_r.I:.4f} p={moran_r.p_sim:.4f}", "SUMMARY")

    return {"W": W, "results": {"OLS": ols, "SLM": slm, "SEM": sem},
            "coef_df": coef_df, "moran_y": moran_y, "moran_r": moran_r,
            "lisa": lisa, "gdf_v": gdf_v, "y": y, "X_cols": X_cols,
            "slm": slm, "sem": sem, "ols": ols}

# =========================================================
# 8. QUADRANT DISPARITY
# =========================================================
def run_quadrant_disparity(gdf: gpd.GeoDataFrame) -> dict:
    log("Quadrant disparity analysis ...", "SUMMARY")
    quads = ["HH", "HL", "LH", "LL"]
    ql    = {"HH": "High H.\nHigh Ineq.", "HL": "High H.\nLow Ineq.",
             "LH": "Low H.\nHigh Ineq.", "LL": "Low H.\nLow Ineq."}
    dists = {q: gdf.loc[gdf["quadrant"] == q, TARGET].dropna() for q in quads}

    global_rows = []
    for q in quads:
        s = dists[q]
        global_rows.append({
            "quadrant": q, "label": ql[q], "n": len(s),
            "mean": s.mean(), "median": s.median(), "std": s.std(),
            "p25": s.quantile(0.25), "p75": s.quantile(0.75),
            "p90": s.quantile(0.90),
        })

    mw_rows = []
    for q1, q2 in combinations(quads, 2):
        a, b  = dists[q1].values, dists[q2].values
        u, p  = mannwhitneyu(a, b, alternative="two-sided")
        pool  = np.sqrt((a.std()**2 + b.std()**2) / 2)
        d     = (a.mean() - b.mean()) / pool if pool > 0 else np.nan
        mw_rows.append({"comparison": f"{q1} vs {q2}", "U": u, "p": p, "cohens_d": d})

    kw_stat, kw_p = kruskal(*[dists[q].values for q in quads])
    _, hh_hl_p    = mannwhitneyu(dists["HH"].values, dists["HL"].values,
                                  alternative="greater")

    reg_rows = []
    for reg in gdf["region_en"].dropna().unique():
        rgdf = gdf[gdf["region_en"] == reg]
        for q in quads:
            s = rgdf.loc[rgdf["quadrant"] == q, TARGET].dropna()
            if len(s) >= 5:
                reg_rows.append({"region": reg, "quadrant": q, "n": len(s),
                                  "mean": s.mean(), "median": s.median(),
                                  "p90": s.quantile(0.90)})

    global_df = pd.DataFrame(global_rows)
    mw_df     = pd.DataFrame(mw_rows)
    region_df = pd.DataFrame(reg_rows)
    pd.concat([global_df, mw_df], axis=1).to_csv(OUT_QUADRANT, index=False)

    log(f"KW H={kw_stat:.2f} p={kw_p:.2e} | "
        f"HH vs HL (amplification) p={hh_hl_p:.4f}", "SUMMARY")
    return {"global_df": global_df, "mw_df": mw_df, "region_df": region_df,
            "kw_stat": kw_stat, "kw_p": kw_p, "hh_hl_p": hh_hl_p,
            "dists": dists, "ql": ql}

# =========================================================
# 9. FIGURE A — MODERATION
# =========================================================
def make_figure_a(mod: dict, gdf: gpd.GeoDataFrame) -> str:
    set_pub_style()
    fig = plt.figure(figsize=(7.2, 9.0))
    fig.patch.set_facecolor(C["bg"])
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.10, right=0.97, top=0.93, bottom=0.07,
                           hspace=0.54, wspace=0.38)

    ax_r2   = fig.add_subplot(gs[0, 0])
    ax_me   = fig.add_subplot(gs[0, 1])
    ax_coef = fig.add_subplot(gs[1, :])
    ax_sc   = fig.add_subplot(gs[2, 0])
    ax_res  = fig.add_subplot(gs[2, 1])

    for ax in [ax_r2, ax_me, ax_coef, ax_sc, ax_res]:
        ax.set_facecolor(C["bg"])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color(C["border"])

    results = mod["results"]
    m4, df  = mod["m4"], mod["df"]

    # a) R² comparison
    names = list(results.keys())
    r2v   = [results[m].rsquared     for m in names]
    r2av  = [results[m].rsquared_adj for m in names]
    xlbls = ["M1\nHazard", "M2\nSocial", "M3\nAdditive", "M4\nInteract."]
    x     = np.arange(len(names))
    ax_r2.bar(x-0.18, r2v,  0.32, label="R²",     color=C["blue"], alpha=0.85, edgecolor="w", lw=0.4)
    ax_r2.bar(x+0.18, r2av, 0.32, label="R²adj.", color=C["teal"], alpha=0.78, edgecolor="w", lw=0.4)
    ax_r2.set_xticks(x); ax_r2.set_xticklabels(xlbls, fontsize=6)
    ax_r2.set_ylabel("Coefficient of determination"); ax_r2.legend(fontsize=5.5)
    ax_r2.set_title("Model fit comparison"); grid_style(ax_r2, "y")
    for i, (v1, v2) in enumerate(zip(r2v, r2av)):
        ax_r2.text(i-0.18, v1+0.002, f"{v1:.3f}", ha="center", fontsize=4.8, color=C["blue"])
        ax_r2.text(i+0.18, v2+0.002, f"{v2:.3f}", ha="center", fontsize=4.8, color=C["teal"])
    plabel(ax_r2, "a")

    # b) Marginal effects
    me   = mod["marginal"]
    lbls = list(me.keys()); vals = list(me.values())
    bars = ax_me.barh(lbls, vals, color=[C["teal"], C["amber"], C["red"]],
                      alpha=0.87, edgecolor="w", lw=0.4, height=0.5)
    ax_me.axvline(0, color=C["text"], lw=0.7, ls="--")
    ax_me.set_xlabel("Marginal effect of hazard on disaster impact")
    ax_me.set_title("Social inequality as moderator"); grid_style(ax_me, "x")
    for bar, v in zip(bars, vals):
        ax_me.text(v+(0.003 if v >= 0 else -0.003),
                   bar.get_y()+bar.get_height()/2,
                   f"{v:+.3f}", va="center",
                   ha="left" if v >= 0 else "right", fontsize=5.5)
    p_int = m4.pvalues.get("hazard_x_social", np.nan)
    b_int = m4.params.get("hazard_x_social", np.nan)
    ax_me.text(0.97, 0.05,
               f"Interaction β={b_int:.3f}\n"
               f"p {'<0.001' if p_int < 0.001 else f'={p_int:.3f}'}",
               transform=ax_me.transAxes, ha="right", va="bottom", fontsize=5.5,
               bbox=dict(boxstyle="round,pad=0.3", fc=C["panel"], ec=C["border"], lw=0.5))
    plabel(ax_me, "b")

    # c) Forest plot M4
    var_map = {
        f"{HAZARD}_std"  : "Hydroclimatic hazard",
        f"{SOCIAL}_std"  : "Social inequality",
        f"{CAPACITY}_std": "Adaptive capacity",
        f"{TREND}_std"   : "Hazard trend",
        "hazard_x_social": "Hazard × Inequality (interaction)",
    }
    m4c = mod["coef_df"][
        (mod["coef_df"]["model"] == "M4 Interaction") &
        (mod["coef_df"]["variable"] != "const")
    ].copy()
    m4c["label"] = m4c["variable"].map(var_map).fillna(m4c["variable"])
    m4c = m4c.sort_values("coef")
    yp  = np.arange(len(m4c))
    fc  = [C["red"] if p < 0.05 else C["gray"] for p in m4c["p"]]
    # barh não aceita lista no ecolor — separar barras e error bars
    ax_coef.barh(yp, m4c["coef"].values, color=fc,
                 edgecolor="w", lw=0.3, height=0.5, alpha=0.87)
    for i, (_, row) in enumerate(m4c.iterrows()):
        ec = C["red"] if row["p"] < 0.05 else C["gray"]
        ax_coef.errorbar(row["coef"], i,
                         xerr=[[row["coef"] - row["ci_low"]],
                               [row["ci_high"] - row["coef"]]],
                         fmt="none", ecolor=ec, elinewidth=0.8,
                         capsize=2.5, capthick=0.8)
    ax_coef.axvline(0, color=C["text"], lw=0.8, ls="--")
    ax_coef.set_yticks(yp)
    ax_coef.set_yticklabels(m4c["label"].values, fontsize=6.5)
    ax_coef.set_xlabel("Standardized coefficient (β) with 95% CI  [HC3-robust SE]")
    ax_coef.set_title(
        f"M4 interaction model  (R²={m4.rsquared:.3f}, "
        f"R²adj={m4.rsquared_adj:.3f}, n={int(m4.nobs):,})")
    grid_style(ax_coef, "x")
    for i, (_, row) in enumerate(m4c.iterrows()):
        sig = ("***" if row["p"]<0.001 else "**" if row["p"]<0.01
               else "*" if row["p"]<0.05 else "ns")
        ax_coef.text(ax_coef.get_xlim()[1]*0.98, i, sig, ha="right", va="center",
                     fontsize=6, color=C["red"] if sig != "ns" else C["gray"])
    ax_coef.text(0.99, -0.11, "*** p<0.001  ** p<0.01  * p<0.05  ns not significant",
                 transform=ax_coef.transAxes, ha="right", fontsize=5, color=C["gray"])
    plabel(ax_coef, "c")

    # d) Moderation scatter
    sc = ax_sc.scatter(df[f"{HAZARD}_std"], df[f"{TARGET}_std"],
                       c=df[f"{SOCIAL}_std"], cmap="RdYlBu_r",
                       s=3, alpha=0.38, linewidths=0, zorder=3)
    plt.colorbar(sc, ax=ax_sc, shrink=0.75, pad=0.02, label="Social inequality (std)")
    h_range = np.linspace(df[f"{HAZARD}_std"].min(), df[f"{HAZARD}_std"].max(), 100)
    b_h  = m4.params.get(f"{HAZARD}_std", 0)
    b_hx = m4.params.get("hazard_x_social", 0)
    b_s  = m4.params.get(f"{SOCIAL}_std", 0)
    b0   = m4.params.get("const", 0)
    for lbl, sq, col in [
        ("Low inequality (P10)",  df[f"{SOCIAL}_std"].quantile(0.10), C["teal"]),
        ("High inequality (P90)", df[f"{SOCIAL}_std"].quantile(0.90), C["red"]),
    ]:
        ax_sc.plot(h_range, b0+b_h*h_range+b_s*sq+b_hx*h_range*sq,
                   color=col, lw=1.2, label=lbl, zorder=4)
    ax_sc.legend(fontsize=5, loc="upper left")
    ax_sc.set_xlabel("Hydroclimatic hazard (std)")
    ax_sc.set_ylabel("Disaster impact (std)")
    ax_sc.set_title("Moderation: hazard effect by inequality level")
    grid_style(ax_sc); plabel(ax_sc, "d")

    # e) Residuals
    fitted = m4.fittedvalues; resid = m4.resid
    ax_res.scatter(fitted, resid, s=2.5, alpha=0.3, color=C["blue"], linewidths=0, zorder=3)
    ax_res.axhline(0, color=C["red"], lw=0.8, ls="--")
    try:
        sm_ = lowess(resid, fitted, frac=0.3)
        ax_res.plot(sm_[:, 0], sm_[:, 1], color=C["amber"], lw=1.0, zorder=4)
    except Exception:
        pass
    ax_res.set_xlabel("Fitted values"); ax_res.set_ylabel("Residuals")
    ax_res.set_title("Residual diagnostics (M4)")
    grid_style(ax_res); plabel(ax_res, "e")

    fig.suptitle(
        "Figure 11a  |  Moderation analysis: social inequality amplifies "
        "hydroclimatic hazard impacts across Brazilian municipalities",
        fontsize=8, fontweight="bold", color=C["text"], y=0.975)
    fig.text(0.5, 0.013,
             f"OLS with HC3-robust SE | standardized variables | n={int(m4.nobs):,} municipalities",
             ha="center", fontsize=5.5, color=C["sub"])

    save_fig(fig, FIG_A_PNG, FIG_A_PDF)
    log(f"Figure A saved: {FIG_A_PNG}", "SUMMARY")
    return str(FIG_A_PNG)

# =========================================================
# 10. FIGURE B — SPATIAL REGRESSION
# =========================================================
def make_figure_b(spa: dict, gdf: gpd.GeoDataFrame) -> str:
    set_pub_style()
    fig = plt.figure(figsize=(7.2, 9.2))
    fig.patch.set_facecolor(C["bg"])
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.07, right=0.97, top=0.93, bottom=0.07,
                           hspace=0.54, wspace=0.30)

    ax_mi   = fig.add_subplot(gs[0, 0])
    ax_lisa = fig.add_subplot(gs[0, 1])
    ax_coef = fig.add_subplot(gs[1, :])
    ax_slm  = fig.add_subplot(gs[2, 0])
    ax_diag = fig.add_subplot(gs[2, 1])

    for ax in [ax_mi, ax_lisa, ax_coef, ax_slm, ax_diag]:
        ax.set_facecolor(C["bg"])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color(C["border"])

    W, y    = spa["W"], spa["y"]
    gdf_v   = spa["gdf_v"].copy()
    moran_y = spa["moran_y"]; moran_r = spa["moran_r"]
    lisa    = spa["lisa"]; coef_df = spa["coef_df"]
    X_cols  = spa["X_cols"]
    slm_r   = spa["slm"]; sem_r = spa["sem"]; ols_r = spa["ols"]

    # a) Moran scatterplot
    Wy = libpysal.weights.lag_spatial(W, y)
    ax_mi.scatter(y, Wy, s=2.5, alpha=0.28, color=C["blue"], linewidths=0, zorder=3)
    zf = np.polyfit(y, Wy, 1); xl = np.linspace(y.min(), y.max(), 100)
    ax_mi.plot(xl, np.poly1d(zf)(xl), color=C["red"], lw=1.2, zorder=4)
    ax_mi.axhline(0, color=C["border"], lw=0.4); ax_mi.axvline(0, color=C["border"], lw=0.4)
    ax_mi.set_xlabel("Disaster impact (std)"); ax_mi.set_ylabel("Spatial lag")
    ax_mi.set_title(f"Global Moran's I = {moran_y.I:.4f}  (p = {moran_y.p_sim:.3f})")
    grid_style(ax_mi); plabel(ax_mi, "a")

    # b) LISA cluster map
    lc  = {"HH": C["HH"], "LL": C["LL"], "LH": C["LH"], "HL": C["HL"], "ns": "#DDDDDD"}
    sig = lisa.p_sim < 0.05; q = lisa.q
    cat = np.where(~sig, "ns", np.where(q==1, "HH", np.where(q==3, "LL", np.where(q==2, "LH", "HL"))))
    gdf_v["lisa_color"] = [lc[c] for c in cat]
    gdf_v.plot(color=gdf_v["lisa_color"], ax=ax_lisa, linewidth=0.05, edgecolor="white")
    ax_lisa.axis("off"); ax_lisa.set_title("LISA cluster map (Local Moran's I)")
    lp = [mpatches.Patch(color=lc[k], label=k) for k in ["HH","HL","LH","LL","ns"]]
    ax_lisa.legend(handles=lp, fontsize=5, loc="lower left", ncol=2,
                   framealpha=0.9, edgecolor=C["border"])
    plabel(ax_lisa, "b")

    # c) Coefficient comparison OLS vs SLM vs SEM
    vmap   = {f"{HAZARD}_std": "Hazard", f"{SOCIAL}_std": "Social ineq.",
              f"{CAPACITY}_std": "Adapt. capacity", f"{TREND}_std": "Hazard trend",
              "hazard_x_social": "Hazard×Ineq."}
    pvars  = [v for v in X_cols if v in vmap]
    mnames = ["OLS", "SLM", "SEM"]; mcols = [C["blue"], C["amber"], C["teal"]]
    xb     = np.arange(len(pvars)); w = 0.22
    for mi, (mn, mc) in enumerate(zip(mnames, mcols)):
        sub = coef_df[coef_df["model"] == mn]
        cs  = [sub.loc[sub["variable"]==v, "coef"].values[0]
               if len(sub.loc[sub["variable"]==v]) else np.nan for v in pvars]
        ses = [sub.loc[sub["variable"]==v, "se"].values[0]
               if len(sub.loc[sub["variable"]==v]) else np.nan for v in pvars]
        cs = np.array(cs); ses = np.array(ses)
        ax_coef.bar(xb+(mi-1)*w, cs, w, label=mn, color=mc, alpha=0.83, edgecolor="w", lw=0.3)
        ax_coef.errorbar(xb+(mi-1)*w, cs, yerr=1.96*ses, fmt="none",
                         ecolor=mc, elinewidth=0.7, capsize=2)
    ax_coef.axhline(0, color=C["text"], lw=0.7, ls="--")
    ax_coef.set_xticks(xb); ax_coef.set_xticklabels([vmap[v] for v in pvars], fontsize=6.5)
    ax_coef.set_ylabel("Coefficient (standardized)")
    ax_coef.set_title("Coefficient stability: OLS vs Spatial Lag (SLM) vs Spatial Error (SEM)")
    ax_coef.legend(fontsize=5.5, loc="upper right"); grid_style(ax_coef, "y")
    ax_coef.text(0.01, 0.97,
                 f"SLM ρ={slm_r.betas[-1][0]:.4f}  |  SEM λ={sem_r.betas[-1][0]:.4f}  |  "
                 f"Moran I(OLS resid)={moran_r.I:.4f}  p={moran_r.p_sim:.3f}",
                 transform=ax_coef.transAxes, va="top", fontsize=5.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc=C["panel"], ec=C["border"], lw=0.4))
    plabel(ax_coef, "c")

    # d) SLM predicted map
    gdf_v["slm_pred"] = slm_r.predy.flatten()
    gdf_v.plot(column="slm_pred", cmap="YlOrRd", ax=ax_slm, legend=True,
               linewidth=0.05, edgecolor="white",
               legend_kwds={"shrink": 0.6, "label": "Predicted (std)"})
    ax_slm.axis("off"); ax_slm.set_title("SLM predicted disaster impact"); plabel(ax_slm, "d")

    # e) Diagnostics table
    ax_diag.axis("off")
    diag = []
    for mn, res in [("OLS", ols_r), ("SLM", slm_r), ("SEM", sem_r)]:
        r2  = getattr(res, "r2",    getattr(res, "pr2",   None))
        aic = getattr(res, "aic",   None)
        ll  = getattr(res, "logll", None)
        diag.append([mn, f"{r2:.4f}" if r2 is not None else "—",
                     f"{aic:.1f}" if aic is not None else "—",
                     f"{ll:.1f}"  if ll  is not None else "—"])
    tbl = ax_diag.table(cellText=diag, colLabels=["Model", "R²/pseudo-R²", "AIC", "Log-L"],
                         cellLoc="center", loc="center", bbox=[0.0, 0.25, 1.0, 0.60])
    tbl.auto_set_font_size(False); tbl.set_fontsize(6.5)
    for (r, c_), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["border"]); cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor(C["blue"]); cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(C["panel"] if r%2==0 else C["bg"])
    ax_diag.set_title("Model fit diagnostics"); plabel(ax_diag, "e")

    fig.suptitle(
        "Figure 11b  |  Spatial regression: correcting for spatial "
        "autocorrelation in flood inequality associations",
        fontsize=8, fontweight="bold", color=C["text"], y=0.975)
    fig.text(0.5, 0.013,
             "Queen contiguity weights | row-standardized | ML estimation | 499 permutations (LISA)",
             ha="center", fontsize=5.5, color=C["sub"])

    save_fig(fig, FIG_B_PNG, FIG_B_PDF)
    log(f"Figure B saved: {FIG_B_PNG}", "SUMMARY")
    return str(FIG_B_PNG)

# =========================================================
# 11. FIGURE C — QUADRANT DISPARITY
# =========================================================
def make_figure_c(qd: dict, gdf: gpd.GeoDataFrame) -> str:
    set_pub_style()
    fig = plt.figure(figsize=(7.2, 9.2))
    fig.patch.set_facecolor(C["bg"])
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.10, right=0.97, top=0.93, bottom=0.07,
                           hspace=0.54, wspace=0.36)

    ax_map  = fig.add_subplot(gs[0, :])
    ax_viol = fig.add_subplot(gs[1, 0])
    ax_eff  = fig.add_subplot(gs[1, 1])
    ax_reg  = fig.add_subplot(gs[2, 0])
    ax_p90  = fig.add_subplot(gs[2, 1])

    for ax in [ax_map, ax_viol, ax_eff, ax_reg, ax_p90]:
        ax.set_facecolor(C["bg"])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color(C["border"])

    quads  = ["HH", "HL", "LH", "LL"]
    qcols  = [C["HH"], C["HL"], C["LH"], C["LL"]]
    ql     = qd["ql"]; dists = qd["dists"]; gdf_df = qd["global_df"]

    # a) Quadrant map
    gdf["_qc"] = gdf["quadrant"].map(
        {"HH": C["HH"], "HL": C["HL"], "LH": C["LH"], "LL": C["LL"]}
    ).fillna("#DDDDDD")
    gdf.plot(color=gdf["_qc"], ax=ax_map, linewidth=0.04, edgecolor="white")
    ax_map.axis("off")
    ax_map.set_title("Municipality quadrant classification  (Hydroclimatic hazard × Social inequality)")
    lp = [mpatches.Patch(color=C[q],
                          label=f"{q}  {ql[q].replace(chr(10),' ')}  "
                                f"(n={int(gdf_df.loc[gdf_df['quadrant']==q,'n'].values[0]):,})")
          for q in quads]
    ax_map.legend(handles=lp, fontsize=5.5, loc="lower left", ncol=2,
                  framealpha=0.92, edgecolor=C["border"])
    plabel(ax_map, "a")

    # b) Violin distributions
    vdata = [dists[q].values for q in quads]
    parts = ax_viol.violinplot(vdata, positions=range(4), showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], qcols):
        pc.set_facecolor(mcolors.to_rgba(col, 0.42)); pc.set_edgecolor(col); pc.set_linewidth(0.8)
    parts["cmedians"].set_color(C["text"]); parts["cmedians"].set_linewidth(1.2)
    rng = np.random.default_rng(42)
    for i, (q, col) in enumerate(zip(quads, qcols)):
        jit = rng.uniform(-0.08, 0.08, len(dists[q]))
        ax_viol.scatter(i+jit, dists[q].values, s=0.9, alpha=0.12, color=col, linewidths=0, zorder=2)
    ax_viol.set_xticks(range(4)); ax_viol.set_xticklabels([ql[q] for q in quads], fontsize=6)
    ax_viol.set_ylabel("Disaster impact index")
    ax_viol.set_title("Disaster impact distribution by quadrant")
    grid_style(ax_viol, "y")
    kw_p = qd["kw_p"]
    ax_viol.text(0.97, 0.97,
                 f"Kruskal-Wallis\nH={qd['kw_stat']:.2f}\n"
                 f"p {'<0.001' if kw_p<0.001 else f'={kw_p:.3f}'}",
                 transform=ax_viol.transAxes, ha="right", va="top", fontsize=5.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc=C["panel"], ec=C["border"], lw=0.5))
    plabel(ax_viol, "b")

    # c) Effect sizes
    pairs = [("HH vs LL","HH","LL"),("HH vs HL","HH","HL"),
             ("LH vs LL","LH","LL"),("HL vs LL","HL","LL")]
    ef_rows = []
    for lbl, q1, q2 in pairs:
        a, b_ = dists[q1].values, dists[q2].values
        pool  = np.sqrt((a.std()**2+b_.std()**2)/2)
        d     = (a.mean()-b_.mean())/pool if pool > 0 else np.nan
        _, p  = mannwhitneyu(a, b_, alternative="two-sided")
        ef_rows.append({"label": lbl, "d": d, "p": p})
    edf   = pd.DataFrame(ef_rows)
    bcols = [C["red"] if p<0.05 else C["gray"] for p in edf["p"]]
    bars  = ax_eff.barh(edf["label"], edf["d"], color=bcols, alpha=0.85, edgecolor="w", lw=0.4, height=0.5)
    ax_eff.axvline(0, color=C["text"], lw=0.7, ls="--")
    for thresh, lbl_ in [(0.2,"small"),(0.5,"medium"),(0.8,"large")]:
        ax_eff.axvline(thresh, color=C["border"], lw=0.5, ls=":")
        ax_eff.text(thresh+0.01, -0.6, lbl_, fontsize=4.5, color=C["gray"])
    for bar, row in zip(bars, edf.itertuples()):
        sig = "***" if row.p<0.001 else "**" if row.p<0.01 else "*" if row.p<0.05 else "ns"
        ax_eff.text(row.d+0.01, bar.get_y()+bar.get_height()/2,
                    f"d={row.d:.2f} {sig}", va="center", fontsize=5.5)
    ax_eff.set_xlabel("Cohen's d (effect size)")
    ax_eff.set_title("Pairwise effect sizes (Mann-Whitney U)")
    grid_style(ax_eff, "x"); plabel(ax_eff, "c")

    # d) Regional breakdown
    reg_df  = qd["region_df"]; regions = sorted(reg_df["region"].unique())
    xr      = np.arange(len(regions)); wr = 0.18
    for qi, (q, col) in enumerate(zip(quads, qcols)):
        vals = [reg_df.loc[(reg_df["region"]==r)&(reg_df["quadrant"]==q), "mean"].values[0]
                if len(reg_df.loc[(reg_df["region"]==r)&(reg_df["quadrant"]==q)]) else np.nan
                for r in regions]
        ax_reg.bar(xr+(qi-1.5)*wr, vals, wr, label=q, color=col, alpha=0.83, edgecolor="w", lw=0.3)
    ax_reg.set_xticks(xr)
    ax_reg.set_xticklabels(regions, fontsize=5.5, rotation=15, ha="right")
    ax_reg.set_ylabel("Mean disaster impact"); ax_reg.set_title("Regional breakdown by quadrant")
    ax_reg.legend(fontsize=5, loc="upper right", ncol=2)
    grid_style(ax_reg, "y"); plabel(ax_reg, "d")

    # e) P90 + social amplification
    p90v  = [gdf_df.loc[gdf_df["quadrant"]==q,"p90"].values[0]  for q in quads]
    meanv = [gdf_df.loc[gdf_df["quadrant"]==q,"mean"].values[0] for q in quads]
    nv    = [gdf_df.loc[gdf_df["quadrant"]==q,"n"].values[0]    for q in quads]
    xp    = np.arange(4)
    ax_p90.bar(xp, p90v, color=qcols, alpha=0.85, edgecolor="w", lw=0.4, label="P90")
    ax_p90.scatter(xp, meanv, color=C["text"], s=20, zorder=5, marker="D", label="Mean")
    ax_p90.set_xticks(xp); ax_p90.set_xticklabels([ql[q] for q in quads], fontsize=6)
    ax_p90.set_ylabel("Disaster impact index"); ax_p90.set_title("90th percentile vs mean impact")
    ax_p90.legend(fontsize=5.5, loc="upper right"); grid_style(ax_p90, "y")
    for i, (v, n) in enumerate(zip(p90v, nv)):
        ax_p90.text(i, v+0.001, f"n={int(n):,}", ha="center", fontsize=4.8, color=C["text"])
    diff = p90v[0] - p90v[1]
    if diff > 0:
        ax_p90.annotate("", xy=(0, p90v[0]+0.007), xytext=(1, p90v[1]+0.007),
                         arrowprops=dict(arrowstyle="<->", color=C["red"], lw=1.0))
        ax_p90.text(0.5, max(p90v[0], p90v[1])+0.013,
                    f"Social amplification\n+{diff:.4f}",
                    ha="center", fontsize=5.5, color=C["red"], fontweight="bold")
    plabel(ax_p90, "e")

    hh_hl_p = qd["hh_hl_p"]
    fig.suptitle(
        "Figure 11c  |  Quadrant disparity: social inequality amplifies "
        "flood disaster impacts across Brazilian municipalities",
        fontsize=8, fontweight="bold", color=C["text"], y=0.975)
    fig.text(0.5, 0.013,
             f"Social amplification (HH vs HL, Mann-Whitney one-sided): "
             f"p {'<0.001' if hh_hl_p<0.001 else f'={hh_hl_p:.4f}'}  |  "
             "Effect size thresholds: small d=0.2, medium d=0.5, large d=0.8",
             ha="center", fontsize=5.5, color=C["sub"])

    save_fig(fig, FIG_C_PNG, FIG_C_PDF)
    log(f"Figure C saved: {FIG_C_PNG}", "SUMMARY")
    return str(FIG_C_PNG)

# =========================================================
# 12. SAVE META
# =========================================================
def save_meta(mod, spa, qd, gdf):
    m4 = mod["m4"]
    meta = {
        "project"          : "Flood Inequality Across Brazil",
        "module"           : "11_model_hazard_inequality_disaster.py",
        "version"          : "v3.0",
        "status"           : "completed",
        "created_at"       : datetime.now().isoformat(),
        "n_municipalities" : int(len(gdf)),
        "target"           : TARGET,
        "approach"         : ["moderation_ols_hc3", "spatial_lag_model",
                              "spatial_error_model", "quadrant_disparity_mannwhitney"],
        "m4_r2"            : float(m4.rsquared),
        "m4_r2_adj"        : float(m4.rsquared_adj),
        "m4_interaction_p" : float(m4.pvalues.get("hazard_x_social", np.nan)),
        "moran_i_target"   : float(spa["moran_y"].I),
        "moran_p_residuals": float(spa["moran_r"].p_sim),
        "kruskal_wallis_p" : float(qd["kw_p"]),
        "social_amplif_p"  : float(qd["hh_hl_p"]),
        "fig_a"            : str(FIG_A_PNG),
        "fig_b"            : str(FIG_B_PNG),
        "fig_c"            : str(FIG_C_PNG),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    row = pd.DataFrame([{
        "stage": "11_model_hazard_inequality_disaster", "tile_id": "ALL",
        "period": "2013_2022", "status": "completed",
        "output_path": str(OUT_MODERATION),
        "timestamp": datetime.now().isoformat(),
    }])
    if CATALOG_PATH.exists():
        try:
            cat = pd.read_csv(CATALOG_PATH)
            cat = cat[~((cat["stage"]=="11_model_hazard_inequality_disaster")&
                        (cat["tile_id"]=="ALL"))]
            pd.concat([cat, row], ignore_index=True).to_csv(CATALOG_PATH, index=False)
        except Exception:
            row.to_csv(CATALOG_PATH, index=False)
    else:
        row.to_csv(CATALOG_PATH, index=False)
    log("Meta and catalog saved.", "SUMMARY")

# =========================================================
# 13. MAIN
# =========================================================
def main():
    print("\n" + "=" * 68)
    print("  Module 11 v3.0 — Flood Inequality in Brazil")
    print("  Moderation + Spatial Regression + Quadrant Disparity")
    print("=" * 68 + "\n")

    gdf = load_and_prepare()

    print("  [1/6] Moderation analysis ...")
    mod = run_moderation(gdf)

    print("  [2/6] Spatial regression ...")
    spa = run_spatial_regression(gdf)

    print("  [3/6] Quadrant disparity ...")
    qd  = run_quadrant_disparity(gdf)

    print("  [4/6] Rendering Figure A — moderation ...")
    make_figure_a(mod, gdf)

    print("  [5/6] Rendering Figure B — spatial regression ...")
    make_figure_b(spa, gdf)

    print("  [6/6] Rendering Figure C — quadrant disparity ...")
    make_figure_c(qd, gdf)

    save_meta(mod, spa, qd, gdf)

    m4 = mod["m4"]
    print("\n" + "=" * 68)
    print("  ✓ Module 11 v3.0 completed.")
    print(f"\n  ── Key findings ──────────────────────────────────")
    print(f"  n                  = {len(gdf):,} municipalities")
    print(f"  M4 R²              = {m4.rsquared:.3f}  (R²adj={m4.rsquared_adj:.3f})")
    p_int = m4.pvalues.get("hazard_x_social", np.nan)
    b_int = m4.params.get("hazard_x_social", np.nan)
    print(f"  Interaction β      = {b_int:.4f}  (p={p_int:.4f})")
    print(f"  Moran I (target)   = {spa['moran_y'].I:.4f}  (p={spa['moran_y'].p_sim:.3f})")
    print(f"  Kruskal-Wallis p   = {qd['kw_p']:.2e}")
    print(f"  Social amplif. p   = {qd['hh_hl_p']:.4f}  (HH vs HL, one-sided)")
    print(f"\n  ── Figures ───────────────────────────────────────")
    print(f"  {FIG_A_PNG}")
    print(f"  {FIG_B_PNG}")
    print(f"  {FIG_C_PNG}")
    print("=" * 68 + "\n")


main()
