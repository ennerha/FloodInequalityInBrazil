"""
Project: Flood Inequality Across Brazil

Module: 12_make_publication_figures.py
Version: v4.0 — Publication-grade figures + GIS export

Purpose:
  Generate publication-ready composite figures (500 DPI, Nature/Science style,
  English) from the integrated hazard-social-disaster dataset and modeling outputs.

Figures:
  FIG_01 — National map composite
  FIG_02 — Model performance + EBM importance + partial effects composite
  FIG_03 — Hazard-inequality-disaster scatter composite
  FIG_04 — Main synthesis figure

GIS exports:
  - Figure 1 map layers as SHP and GPKG
  - Figure 4 synthesis layers as SHP and GPKG

All figures:
  - 500 DPI PNG + vector PDF
  - Times New Roman serif, panel labels (a)(b)(c)...
  - Consistent color palette across all panels
  - Caption-ready subtitles

Author: Enner H. de Alcântara
Version: v4.0
Language: English
"""

# =========================================================
# 1. IMPORTS
# =========================================================
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 2. PATHS
# =========================================================
BASE = Path("/content/drive/MyDrive/Brazil/flood_inequality_project")

SPATIAL    = BASE / "04_integrated" / "hazard_social_disaster_municipal_summary_brazil.geoparquet"
IMPORTANCE = BASE / "05_modeling"   / "feature_importance_ebm.csv"
PARTIALS   = BASE / "05_modeling"   / "partial_effects_ebm.parquet"
METRICS    = BASE / "05_modeling"   / "model_metrics.csv"
META_11    = BASE / "05_modeling"   / "11_model_hazard_inequality_disaster.meta.json"

MOD_RESULTS = BASE / "05_modeling" / "moderation_results.csv"
SPA_RESULTS = BASE / "05_modeling" / "spatial_regression_results.csv"
QUA_RESULTS = BASE / "05_modeling" / "quadrant_disparity_results.csv"

OUT = BASE / "06_figures"
OUT.mkdir(parents=True, exist_ok=True)

GIS_OUT = OUT / "gis_exports"
GIS_OUT.mkdir(parents=True, exist_ok=True)

OUT_FIG1 = OUT / "FIG_01_national_maps_composite.png"
OUT_FIG2 = OUT / "FIG_02_model_interpretation_composite.png"
OUT_FIG3 = OUT / "FIG_03_hazard_inequality_scatter_composite.png"
OUT_FIG4 = OUT / "FIG_04_main_synthesis_composite.png"

OUT_FIG1_PDF = OUT / "FIG_01_national_maps_composite.pdf"
OUT_FIG2_PDF = OUT / "FIG_02_model_interpretation_composite.pdf"
OUT_FIG3_PDF = OUT / "FIG_03_hazard_inequality_scatter_composite.pdf"
OUT_FIG4_PDF = OUT / "FIG_04_main_synthesis_composite.pdf"

# =========================================================
# 3. GLOBAL PUBLICATION STYLE
# =========================================================
def set_pub_style():
    matplotlib.rcParams.update({
        "font.family"       : "serif",
        "font.serif"        : ["Times New Roman", "DejaVu Serif", "Times"],
        "font.size"         : 7,
        "axes.linewidth"    : 0.6,
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "axes.labelsize"    : 7,
        "axes.titlesize"    : 7.5,
        "axes.titleweight"  : "bold",
        "axes.titlepad"     : 5,
        "xtick.major.width" : 0.5,
        "ytick.major.width" : 0.5,
        "xtick.major.size"  : 2.5,
        "ytick.major.size"  : 2.5,
        "xtick.labelsize"   : 6,
        "ytick.labelsize"   : 6,
        "legend.fontsize"   : 5.5,
        "legend.frameon"    : True,
        "legend.framealpha" : 0.9,
        "legend.edgecolor"  : "#CCCCCC",
        "figure.dpi"        : 72,
        "savefig.dpi"       : 500,
        "pdf.fonttype"      : 42,
        "ps.fonttype"       : 42,
        "svg.fonttype"      : "none",
    })

C = {
    "bg"     : "#FFFFFF",
    "panel"  : "#F7F7F7",
    "text"   : "#1A1A2E",
    "sub"    : "#555577",
    "border" : "#CCCCCC",
    "blue"   : "#1D4E89",
    "red"    : "#9B2226",
    "teal"   : "#2A7F6F",
    "amber"  : "#B5621B",
    "purple" : "#5C4374",
    "gray"   : "#888888",
    "gold"   : "#C8963E",
    "HH"     : "#9B2226",
    "HL"     : "#E07B39",
    "LH"     : "#4A7CB5",
    "LL"     : "#2A7F6F",
}

REGION_COLORS = {
    "North"       : "#4A90D9",
    "Northeast"   : "#E8A838",
    "Center-West" : "#6DB56D",
    "Southeast"   : "#C0504D",
    "South"       : "#9B59B6",
}

def plabel(ax, letter, x=0.03, y=0.97, size=10):
    ax.text(
        x, y, letter, transform=ax.transAxes,
        fontsize=size, fontweight="bold", va="top",
        color=C["text"], fontfamily="serif"
    )

def grid_style(ax, axis="both"):
    ax.grid(axis=axis, linewidth=0.22, color=C["border"],
            alpha=0.8, zorder=0, linestyle="--")

def style_ax(ax):
    ax.set_facecolor(C["bg"])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color(C["border"])

def save_fig(fig, png_path, pdf_path):
    fig.savefig(png_path, dpi=500, bbox_inches="tight", facecolor=C["bg"])
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=C["bg"])
    try:
        from IPython.display import display
        display(fig)
    except Exception:
        plt.show()
    plt.close(fig)
    print(f"  Saved: {png_path.name}")

# =========================================================
# 4. HELPERS
# =========================================================
def clean_name(name: str, max_len: int = 38) -> str:
    if pd.isna(name):
        return ""
    s = str(name)
    MAP = {
        "annual_prcp_mm"                     : "Annual precipitation (mm)",
        "wet_days_n"                         : "Wet days yr⁻¹",
        "heavy_rain_days_20mm_n"             : "Heavy-rain days (≥20 mm)",
        "rx1day_mm"                          : "Rx1day (mm)",
        "rx3day_mm"                          : "Rx3day (mm)",
        "rx5day_mm"                          : "Rx5day (mm)",
        "income_pc"                          : "Per capita income",
        "illiteracy_rate"                    : "Illiteracy rate (%)",
        "water_supply_adequate_pct"          : "Water supply coverage (%)",
        "sewerage_adequate_pct"              : "Sewerage coverage (%)",
        "population_total"                   : "Population",
        "population_density"                 : "Population density",
        "urbanization_proxy_pct"             : "Urbanization (%)",
        "hazard_recent_extremes_index"       : "Recent hazard extremes",
        "hazard_trend_index"                 : "Hazard trend",
        "social_inequality_index"            : "Social inequality",
        "adaptive_capacity_index"            : "Adaptive capacity",
        "hazard_social_disaster_compound_index": "Compound burden index",
        "hazard_inequality_coupling_index"   : "Flood inequality score",
        "disaster_observed_index"            : "Disaster impact index",
        "s2id_feat_flood_freq_total"         : "Flood event frequency",
        "s2id_feat_flood_trend_slope"        : "Flood trend (slope)",
        "s2id_feat_flood_acceleration"       : "Flood acceleration",
        "s2id_feat_flood_freq_rate"          : "Flood frequency rate",
    }
    if s in MAP:
        return MAP[s]
    s = s.replace("_", " ").strip()
    s = " ".join(s.split())
    s = s[0].upper() + s[1:] if s else s
    return s if len(s) <= max_len else s[:max_len - 1] + "…"

def safe_read_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def choose_map_col(gdf: gpd.GeoDataFrame) -> str:
    for c in [
        "hazard_inequality_coupling_index",
        "hazard_social_disaster_compound_index",
        "disaster_observed_index",
        "social_inequality_index",
        "hazard_recent_extremes_index",
    ]:
        if c in gdf.columns:
            return c
    raise RuntimeError("No suitable map column found.")

def choose_metric_col(metrics: pd.DataFrame) -> str:
    for c in ["cv_r2_mean", "oof_r2", "cv_rmse_mean", "oof_rmse"]:
        if c in metrics.columns:
            return c
    raise RuntimeError("No metric column found.")

def numeric_x(series: pd.Series) -> tuple[np.ndarray, bool]:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() >= max(5, int(0.4 * len(series))):
        return x.to_numpy(), True
    return np.arange(len(series)), False

def write_vector(gdf_out: gpd.GeoDataFrame, stem: str):
    """
    Save vector layers as both GPKG and ESRI Shapefile.
    Renames long columns for SHP compatibility (<= 10 chars).
    """
    gdf_out = gdf_out.copy()
    gdf_out = gdf_out[gdf_out.geometry.notna()].copy()
    if gdf_out.empty:
        print(f"  Skipping vector export for {stem}: empty GeoDataFrame")
        return

    gpkg_path = GIS_OUT / f"{stem}.gpkg"
    shp_path  = GIS_OUT / f"{stem}.shp"

    # Save GPKG with original column names
    gdf_out.to_file(gpkg_path, driver="GPKG")

    # Shapefile-safe copy
    shp_gdf = gdf_out.copy()
    rename_map = {}
    used = set()

    for col in shp_gdf.columns:
        if col == "geometry":
            continue
        short = col[:10]
        if short in used:
            base = short[:8]
            k = 1
            short = f"{base}{k:02d}"
            while short in used:
                k += 1
                short = f"{base}{k:02d}"
        used.add(short)
        if short != col:
            rename_map[col] = short

    shp_gdf = shp_gdf.rename(columns=rename_map)
    shp_gdf.to_file(shp_path, driver="ESRI Shapefile")

    print(f"  Vector saved: {gpkg_path.name}")
    print(f"  Vector saved: {shp_path.name}")

def make_hotspot_flag(series: pd.Series, threshold: float = 75.0) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return (x >= threshold).astype(int)

# =========================================================
# 5. LOAD DATA
# =========================================================
print("Loading data ...")

if not SPATIAL.exists():
    raise FileNotFoundError(f"Spatial dataset not found: {SPATIAL}")
if not IMPORTANCE.exists():
    raise FileNotFoundError(f"Importance file not found: {IMPORTANCE}")
if not PARTIALS.exists():
    raise FileNotFoundError(f"Partial effects file not found: {PARTIALS}")
if not METRICS.exists():
    raise FileNotFoundError(f"Metrics file not found: {METRICS}")

gdf      = gpd.read_parquet(SPATIAL)
imp      = pd.read_csv(IMPORTANCE)
partials = pd.read_parquet(PARTIALS)
metrics  = pd.read_csv(METRICS)
meta11   = safe_read_meta(META_11)

mod_df = pd.read_csv(MOD_RESULTS) if MOD_RESULTS.exists() else pd.DataFrame()
qua_df = pd.read_csv(QUA_RESULTS) if QUA_RESULTS.exists() else pd.DataFrame()
spa_df = pd.read_csv(SPA_RESULTS) if SPA_RESULTS.exists() else pd.DataFrame()

for name, df in [("Spatial", gdf), ("Importance", imp), ("Partials", partials), ("Metrics", metrics)]:
    if len(df) == 0:
        raise RuntimeError(f"{name} dataset is empty.")

if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4674", allow_override=True)

map_col    = choose_map_col(gdf)
metric_col = choose_metric_col(metrics)

imp_clean = imp.copy()
imp_clean["label"] = imp_clean["feature"].apply(clean_name)
imp_top15 = imp_clean.head(15).sort_values("importance", ascending=True).reset_index(drop=True)
imp_top4  = imp_clean.head(4)["feature"].tolist()

metric_label = {
    "cv_r2_mean"  : "Cross-validated R²",
    "oof_r2"      : "Out-of-fold R²",
    "cv_rmse_mean": "Cross-validated RMSE",
    "oof_rmse"    : "Out-of-fold RMSE",
}.get(metric_col, metric_col)

if metric_col in ["cv_rmse_mean", "oof_rmse"]:
    metrics_plot = metrics.sort_values(metric_col, ascending=False).reset_index(drop=True)
else:
    metrics_plot = metrics.sort_values(metric_col, ascending=True).reset_index(drop=True)

# Ensure quadrant field
if "quadrant" not in gdf.columns:
    haz_col = "hazard_recent_extremes_index"
    soc_col = "social_inequality_index"
    if haz_col in gdf.columns and soc_col in gdf.columns:
        h_med = pd.to_numeric(gdf[haz_col], errors="coerce").median()
        s_med = pd.to_numeric(gdf[soc_col], errors="coerce").median()
        h = pd.to_numeric(gdf[haz_col], errors="coerce")
        s = pd.to_numeric(gdf[soc_col], errors="coerce")
        gdf["quadrant"] = np.select(
            [h.ge(h_med) & s.ge(s_med), h.ge(h_med) & s.lt(s_med), h.lt(h_med) & s.ge(s_med)],
            ["HH", "HL", "LH"],
            default="LL"
        )

# English region
reg_col = next((c for c in gdf.columns if "region" in c.lower() and gdf[c].dtype == object), None)
if reg_col:
    mapping = {
        "Norte": "North",
        "Nordeste": "Northeast",
        "Centro-Oeste": "Center-West",
        "Sudeste": "Southeast",
        "Sul": "South",
    }
    gdf["region_en"] = gdf[reg_col].map(mapping).fillna(gdf[reg_col])

# Build hotspot flags if absent
if "triple_burden_flag" not in gdf.columns:
    gdf["triple_burden_flag"] = make_hotspot_flag(gdf[map_col], threshold=75.0)

# =========================================================
# 5.1 VECTOR EXPORTS FOR MAPS
# =========================================================
def export_map_layers():
    print("Exporting GIS layers ...")

    # Figure 1 — main burden map
    fig1_cols = ["geometry", map_col, "triple_burden_flag"]
    if "mun_code" in gdf.columns:
        fig1_cols.insert(0, "mun_code")
    if "mun_name" in gdf.columns:
        fig1_cols.insert(1 if "mun_code" in fig1_cols else 0, "mun_name")

    fig1_gdf = gdf[[c for c in fig1_cols if c in gdf.columns]].copy()
    write_vector(fig1_gdf, "fig01_burden_map")

    # Figure 1 hotspots only
    hot = gdf[gdf["triple_burden_flag"] == 1].copy()
    if len(hot):
        hot_cols = ["geometry", map_col, "triple_burden_flag"]
        if "mun_code" in hot.columns:
            hot_cols.insert(0, "mun_code")
        if "mun_name" in hot.columns:
            hot_cols.insert(1 if "mun_code" in hot_cols else 0, "mun_name")
        hot_gdf = hot[[c for c in hot_cols if c in hot.columns]].copy()
        write_vector(hot_gdf, "fig01_hotspots")

    # Figure 4 — quadrant map
    if "quadrant" in gdf.columns:
        quad_cols = ["geometry", "quadrant", map_col]
        if "hazard_recent_extremes_index" in gdf.columns:
            quad_cols.append("hazard_recent_extremes_index")
        if "social_inequality_index" in gdf.columns:
            quad_cols.append("social_inequality_index")
        if "mun_code" in gdf.columns:
            quad_cols.insert(0, "mun_code")
        if "mun_name" in gdf.columns:
            quad_cols.insert(1 if "mun_code" in quad_cols else 0, "mun_name")
        quad_gdf = gdf[[c for c in quad_cols if c in gdf.columns]].copy()
        write_vector(quad_gdf, "fig04_quadrants_map")

# =========================================================
# 6. FIGURE 1 — NATIONAL MAPS COMPOSITE
# =========================================================
def make_fig1():
    set_pub_style()
    fig = plt.figure(figsize=(7.2, 9.5))
    fig.patch.set_facecolor(C["bg"])

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.04, right=0.97, top=0.93, bottom=0.05,
        hspace=0.12, wspace=0.08
    )

    ax_comp = fig.add_subplot(gs[0, :])
    ax_haz  = fig.add_subplot(gs[1, 0])
    ax_soc  = fig.add_subplot(gs[1, 1])

    for ax in [ax_comp, ax_haz, ax_soc]:
        ax.set_facecolor(C["bg"])

    # a) Main compound/coupling index
    x = pd.to_numeric(gdf[map_col], errors="coerce")
    vmin = x.quantile(0.02)
    vmax = x.quantile(0.98)
    vcenter = (vmin + vmax) / 2.0

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    gdf.plot(
        column=map_col, cmap="RdYlBu_r", norm=norm,
        linewidth=0.04, edgecolor="white", ax=ax_comp,
        missing_kwds={"color": "#DDDDDD"}, legend=False
    )
    sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_comp, orientation="vertical", fraction=0.018, pad=0.01, shrink=0.85)
    cb.set_label("Index value", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    ax_comp.axis("off")
    ax_comp.set_title("Flood inequality burden index", fontsize=8, fontweight="bold", pad=6)

    hot = gdf[gdf["triple_burden_flag"] == 1]
    if len(hot):
        hot.plot(ax=ax_comp, color="none", edgecolor="#111111", linewidth=0.25, zorder=5)
        ax_comp.text(
            0.97, 0.04,
            f"Hotspots (score ≥ 75)\n n = {len(hot):,}",
            transform=ax_comp.transAxes, ha="right", va="bottom",
            fontsize=5.5, color=C["text"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["border"], lw=0.5, alpha=0.92)
        )
    plabel(ax_comp, "a")

    # b) Hazard map
    haz_col = "hazard_recent_extremes_index"
    if haz_col in gdf.columns:
        hz = pd.to_numeric(gdf[haz_col], errors="coerce")
        hnorm = mcolors.Normalize(vmin=hz.quantile(0.02), vmax=hz.quantile(0.98))
        gdf.plot(
            column=haz_col, cmap="YlOrRd", linewidth=0.04,
            edgecolor="white", ax=ax_haz, legend=False,
            missing_kwds={"color": "#DDDDDD"}
        )
        sm2 = plt.cm.ScalarMappable(cmap="YlOrRd", norm=hnorm)
        sm2.set_array([])
        cb2 = fig.colorbar(sm2, ax=ax_haz, orientation="vertical", fraction=0.035, pad=0.01, shrink=0.85)
        cb2.set_label("Index value", fontsize=5.5)
        cb2.ax.tick_params(labelsize=5)
    ax_haz.axis("off")
    ax_haz.set_title("Hydroclimatic hazard extremes", fontsize=7.5, fontweight="bold", pad=5)
    plabel(ax_haz, "b")

    # c) Social inequality map
    soc_col = "social_inequality_index"
    if soc_col in gdf.columns:
        sc = pd.to_numeric(gdf[soc_col], errors="coerce")
        snorm = mcolors.Normalize(vmin=sc.quantile(0.02), vmax=sc.quantile(0.98))
        gdf.plot(
            column=soc_col, cmap="PuBu", linewidth=0.04,
            edgecolor="white", ax=ax_soc, legend=False,
            missing_kwds={"color": "#DDDDDD"}
        )
        sm3 = plt.cm.ScalarMappable(cmap="PuBu", norm=snorm)
        sm3.set_array([])
        cb3 = fig.colorbar(sm3, ax=ax_soc, orientation="vertical", fraction=0.035, pad=0.01, shrink=0.85)
        cb3.set_label("Index value", fontsize=5.5)
        cb3.ax.tick_params(labelsize=5)
    ax_soc.axis("off")
    ax_soc.set_title("Social inequality", fontsize=7.5, fontweight="bold", pad=5)
    plabel(ax_soc, "c")

    fig.suptitle(
        "Figure 1 | Spatial distribution of flood inequality burden across Brazil",
        fontsize=8.5, fontweight="bold", color=C["text"], y=0.975
    )
    fig.text(
        0.5, 0.015,
        "National municipal layer | publication export at 500 DPI | hotspot threshold = 75",
        ha="center", fontsize=5.5, color=C["sub"]
    )

    save_fig(fig, OUT_FIG1, OUT_FIG1_PDF)

# =========================================================
# 7. FIGURE 2 — MODEL INTERPRETATION COMPOSITE
# =========================================================
def make_fig2():
    set_pub_style()
    fig = plt.figure(figsize=(7.2, 9.2))
    fig.patch.set_facecolor(C["bg"])

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        left=0.10, right=0.97, top=0.93, bottom=0.07,
        hspace=0.52, wspace=0.38
    )

    ax_met = fig.add_subplot(gs[0, 0])
    ax_imp = fig.add_subplot(gs[0:2, 1])
    ax_pe1 = fig.add_subplot(gs[1, 0])
    ax_pe2 = fig.add_subplot(gs[2, 0])
    ax_pe3 = fig.add_subplot(gs[2, 1])

    for ax in [ax_met, ax_imp, ax_pe1, ax_pe2, ax_pe3]:
        style_ax(ax)

    # a) Model performance
    bar_cols = [C["blue"], C["amber"], C["teal"], C["purple"]][:len(metrics_plot)]
    ax_met.bar(
        metrics_plot["model"], metrics_plot[metric_col],
        color=bar_cols, edgecolor="white", linewidth=0.4, alpha=0.87
    )
    ax_met.set_ylabel(metric_label)
    ax_met.set_title("Model performance")
    grid_style(ax_met, "y")
    for i, v in enumerate(metrics_plot[metric_col]):
        ax_met.text(i, v + metrics_plot[metric_col].max() * 0.02, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=5.5, color=C["text"])
    plabel(ax_met, "a")

    # b) EBM importance
    imp_colors = [C["blue"] if i >= len(imp_top15) - 5 else C["teal"] for i in range(len(imp_top15))]
    ax_imp.barh(
        imp_top15["label"], imp_top15["importance"],
        color=imp_colors, edgecolor="white", linewidth=0.3, alpha=0.87
    )
    ax_imp.set_xlabel("Importance")
    ax_imp.set_title("EBM feature importance (top 15)")
    grid_style(ax_imp, "x")
    ax_imp.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_imp.legend(
        handles=[
            mpatches.Patch(color=C["blue"], label="Top 5"),
            mpatches.Patch(color=C["teal"], label="Top 6–15"),
        ],
        fontsize=5, loc="lower right"
    )
    plabel(ax_imp, "b")

    # c/d/e) Top three partial effects
    pe_axes = [ax_pe1, ax_pe2, ax_pe3]
    pe_feats = imp_top4[:3]
    pe_cols = [C["blue"], C["red"], C["teal"]]
    pe_lbls = ["c", "d", "e"]

    for ax, feat, col, lbl in zip(pe_axes, pe_feats, pe_cols, pe_lbls):
        dfp = partials[partials["feature"] == feat].copy()
        if dfp.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=7, color=C["gray"], transform=ax.transAxes)
            ax.axis("off")
            plabel(ax, lbl)
            continue

        xv, is_num = numeric_x(dfp["x"])
        yv = pd.to_numeric(dfp["effect"], errors="coerce").to_numpy()
        mask = np.isfinite(xv) & np.isfinite(yv)
        xv = xv[mask]
        yv = yv[mask]

        if len(xv) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                    fontsize=7, color=C["gray"], transform=ax.transAxes)
            ax.axis("off")
            plabel(ax, lbl)
            continue

        order = np.argsort(xv)
        xv = xv[order]
        yv = yv[order]

        ax.plot(xv, yv, color=col, lw=1.2, zorder=3)
        step = max(1, len(xv) // 30)
        ax.scatter(xv[::step], yv[::step], s=6, color=col, linewidths=0, zorder=4, alpha=0.7)
        ax.axhline(0, color=C["gray"], lw=0.6, ls="--", zorder=2)
        ax.fill_between(xv, 0, yv, where=yv > 0, alpha=0.08, color=col)
        ax.fill_between(xv, 0, yv, where=yv <= 0, alpha=0.08, color=C["gray"])
        ax.set_xlabel("Feature value" if is_num else "Bins", fontsize=6)
        ax.set_ylabel("EBM effect", fontsize=6)
        ax.set_title(clean_name(feat))
        grid_style(ax)
        plabel(ax, lbl)

    fig.suptitle(
        "Figure 2 | Model performance and EBM interpretation of flood inequality drivers",
        fontsize=8.5, fontweight="bold", color=C["text"], y=0.975
    )
    fig.text(
        0.5, 0.013,
        "EBM = Explainable Boosting Machine | partial effects represent marginal predictor contribution",
        ha="center", fontsize=5.5, color=C["sub"]
    )

    save_fig(fig, OUT_FIG2, OUT_FIG2_PDF)

# =========================================================
# 8. FIGURE 3 — HAZARD × INEQUALITY × DISASTER SCATTER
# =========================================================
def make_fig3():
    set_pub_style()
    fig = plt.figure(figsize=(7.2, 9.2))
    fig.patch.set_facecolor(C["bg"])

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        left=0.10, right=0.97, top=0.93, bottom=0.07,
        hspace=0.52, wspace=0.38
    )

    ax_sc1 = fig.add_subplot(gs[0, 0])
    ax_sc2 = fig.add_subplot(gs[0, 1])
    ax_box = fig.add_subplot(gs[1, 0])
    ax_reg = fig.add_subplot(gs[1, 1])
    ax_den = fig.add_subplot(gs[2, 0])
    ax_cor = fig.add_subplot(gs[2, 1])

    for ax in [ax_sc1, ax_sc2, ax_box, ax_reg, ax_den, ax_cor]:
        style_ax(ax)

    haz_col = "hazard_recent_extremes_index"
    soc_col = "social_inequality_index"
    dis_col = "disaster_observed_index"
    cap_col = "adaptive_capacity_index"

    available = [c for c in [haz_col, soc_col, dis_col, cap_col] if c in gdf.columns]
    base_cols = available.copy()
    if "quadrant" in gdf.columns:
        base_cols.append("quadrant")
    if "region_en" in gdf.columns:
        base_cols.append("region_en")

    df = gdf[base_cols].copy()
    for c in available:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if haz_col in df.columns and soc_col in df.columns:
        df = df.dropna(subset=[haz_col, soc_col])

    # a) Hazard vs Disaster (color = social)
    if dis_col in df.columns and soc_col in df.columns:
        sc = ax_sc1.scatter(
            df[haz_col], df[dis_col],
            c=df[soc_col], cmap="RdYlBu_r",
            s=3, alpha=0.4, linewidths=0, zorder=3
        )
        cb = plt.colorbar(sc, ax=ax_sc1, shrink=0.75, pad=0.02)
        cb.set_label("Social inequality", fontsize=6)
        cb.ax.tick_params(labelsize=5)
        ax_sc1.set_xlabel("Hazard extremes index")
        ax_sc1.set_ylabel("Disaster impact index")
        ax_sc1.set_title("Hazard vs disaster (color = social)")
        grid_style(ax_sc1)
    else:
        ax_sc1.axis("off")
    plabel(ax_sc1, "a")

    # b) Social vs Disaster (color = hazard)
    if dis_col in df.columns and haz_col in df.columns:
        sc2 = ax_sc2.scatter(
            df[soc_col], df[dis_col],
            c=df[haz_col], cmap="YlOrRd",
            s=3, alpha=0.4, linewidths=0, zorder=3
        )
        cb2 = plt.colorbar(sc2, ax=ax_sc2, shrink=0.75, pad=0.02)
        cb2.set_label("Hazard extremes", fontsize=6)
        cb2.ax.tick_params(labelsize=5)
        ax_sc2.set_xlabel("Social inequality index")
        ax_sc2.set_ylabel("Disaster impact index")
        ax_sc2.set_title("Social inequality vs disaster (color = hazard)")
        grid_style(ax_sc2)
    else:
        ax_sc2.axis("off")
    plabel(ax_sc2, "b")

    # c) Boxplot by quadrant
    if "quadrant" in df.columns and dis_col in df.columns:
        quads = ["HH", "HL", "LH", "LL"]
        qcols = [C["HH"], C["HL"], C["LH"], C["LL"]]
        vdata = [df.loc[df["quadrant"] == q, dis_col].dropna().values for q in quads]
        bp = ax_box.boxplot(
            vdata, patch_artist=True, notch=False, widths=0.55, showfliers=True,
            flierprops=dict(marker=".", markersize=1.5, color=C["gray"], alpha=0.4),
            medianprops=dict(color=C["text"], lw=1.2),
            whiskerprops=dict(color=C["gray"], lw=0.7),
            capprops=dict(color=C["gray"], lw=0.7)
        )
        for patch, col in zip(bp["boxes"], qcols):
            patch.set_facecolor(mcolors.to_rgba(col, 0.35))
            patch.set_edgecolor(col)
            patch.set_linewidth(0.8)
        ax_box.set_xticks(range(1, 5))
        ax_box.set_xticklabels(quads, fontsize=6)
        ax_box.set_ylabel("Disaster impact index")
        ax_box.set_title("Disaster impact by hazard × inequality quadrant")
        grid_style(ax_box, "y")
        ax_box.legend(
            handles=[mpatches.Patch(color=C[q], label=q) for q in quads],
            fontsize=5, ncol=2, loc="upper right"
        )
    else:
        ax_box.axis("off")
    plabel(ax_box, "c")

    # d) Mean disaster by region and quadrant
    if "region_en" in df.columns and "quadrant" in df.columns and dis_col in df.columns:
        quads = ["HH", "HL", "LH", "LL"]
        qcols = [C["HH"], C["HL"], C["LH"], C["LL"]]
        regions = sorted(df["region_en"].dropna().unique())
        xr = np.arange(len(regions))
        wr = 0.18
        for qi, (q, col) in enumerate(zip(quads, qcols)):
            vals = [df.loc[(df["region_en"] == r) & (df["quadrant"] == q), dis_col].mean() for r in regions]
            ax_reg.bar(xr + (qi - 1.5) * wr, vals, wr, label=q, color=col,
                       alpha=0.83, edgecolor="w", lw=0.3)
        ax_reg.set_xticks(xr)
        ax_reg.set_xticklabels(regions, fontsize=5.5, rotation=15, ha="right")
        ax_reg.set_ylabel("Mean disaster impact")
        ax_reg.set_title("Regional disaster impact by quadrant")
        ax_reg.legend(fontsize=5, ncol=2, loc="upper right")
        grid_style(ax_reg, "y")
    else:
        ax_reg.axis("off")
    plabel(ax_reg, "d")

    # e) Disaster density by social tercile
    if dis_col in df.columns and soc_col in df.columns:
        t33 = df[soc_col].quantile(0.33)
        t67 = df[soc_col].quantile(0.67)
        groups = {
            "Low inequality (P0–33)"    : df.loc[df[soc_col] <= t33, dis_col].dropna(),
            "Medium inequality (P33–67)": df.loc[(df[soc_col] > t33) & (df[soc_col] <= t67), dis_col].dropna(),
            "High inequality (P67–100)" : df.loc[df[soc_col] > t67, dis_col].dropna(),
        }
        dens_cols = [C["teal"], C["amber"], C["red"]]
        for (lbl, vals), col in zip(groups.items(), dens_cols):
            ax_den.hist(vals, bins=40, density=True, color=col,
                        alpha=0.55, edgecolor="white", lw=0.2, label=lbl)
        ax_den.set_xlabel("Disaster impact index")
        ax_den.set_ylabel("Density")
        ax_den.set_title("Disaster impact by social inequality tercile")
        ax_den.legend(fontsize=5, loc="upper right")
        grid_style(ax_den, "y")
    else:
        ax_den.axis("off")
    plabel(ax_den, "e")

    # f) Correlation matrix
    corr_cols = [c for c in [haz_col, soc_col, dis_col, cap_col, "hazard_trend_index"] if c in df.columns]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr().values
        short = {
            haz_col: "Hazard",
            soc_col: "Social",
            dis_col: "Disaster",
            cap_col: "Capacity",
            "hazard_trend_index": "Trend",
        }
        lbls = [short.get(c, c[:8]) for c in corr_cols]
        im = ax_cor.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax_cor, shrink=0.8, pad=0.02).ax.tick_params(labelsize=5)
        ax_cor.set_xticks(range(len(corr_cols)))
        ax_cor.set_yticks(range(len(corr_cols)))
        ax_cor.set_xticklabels(lbls, fontsize=6, rotation=30, ha="right")
        ax_cor.set_yticklabels(lbls, fontsize=6)
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                ax_cor.text(
                    j, i, f"{corr[i, j]:.2f}",
                    ha="center", va="center", fontsize=4.5,
                    color="white" if abs(corr[i, j]) > 0.5 else C["text"]
                )
        ax_cor.set_title("Index correlation matrix")
    else:
        ax_cor.axis("off")
    plabel(ax_cor, "f")

    fig.suptitle(
        "Figure 3 | Hazard–inequality–disaster associations across Brazilian municipalities",
        fontsize=8.5, fontweight="bold", color=C["text"], y=0.975
    )
    fig.text(
        0.5, 0.013,
        "HH = high hazard / high inequality | HL = high hazard / low inequality | "
        "LH = low hazard / high inequality | LL = low hazard / low inequality",
        ha="center", fontsize=5.5, color=C["sub"]
    )

    save_fig(fig, OUT_FIG3, OUT_FIG3_PDF)

# =========================================================
# 9. FIGURE 4 — MAIN SYNTHESIS FIGURE
# =========================================================
def make_fig4():
    set_pub_style()
    fig = plt.figure(figsize=(7.2, 8.8))
    fig.patch.set_facecolor(C["bg"])

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.06, right=0.97, top=0.93, bottom=0.07,
        hspace=0.22, wspace=0.18
    )

    ax_map  = fig.add_subplot(gs[:, 0])
    ax_quad = fig.add_subplot(gs[0, 1])
    ax_bar  = fig.add_subplot(gs[1, 1])

    for ax in [ax_map, ax_quad, ax_bar]:
        style_ax(ax)

    # a) Main map
    x = pd.to_numeric(gdf[map_col], errors="coerce")
    norm = mcolors.Normalize(vmin=x.quantile(0.02), vmax=x.quantile(0.98))
    gdf.plot(
        column=map_col, cmap="RdYlBu_r", linewidth=0.04,
        edgecolor="white", ax=ax_map, legend=False,
        missing_kwds={"color": "#DDDDDD"}
    )
    hot = gdf[gdf["triple_burden_flag"] == 1]
    if len(hot):
        hot.plot(ax=ax_map, color="none", edgecolor="#111111", linewidth=0.25, zorder=5)
    sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_map, orientation="horizontal", fraction=0.04, pad=0.02)
    cb.set_label("Flood inequality score", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    ax_map.axis("off")
    ax_map.set_title("National hotspots of flood inequality", fontsize=7.8, fontweight="bold")
    plabel(ax_map, "a")

    # b) Quadrant map
    if "quadrant" in gdf.columns:
        quad_order = ["HH", "HL", "LH", "LL"]
        quad_colors = [C[q] for q in quad_order]
        cmap = mcolors.ListedColormap(quad_colors)
        quad_code = {q: i for i, q in enumerate(quad_order)}
        plot_gdf = gdf.copy()
        plot_gdf["quad_code"] = plot_gdf["quadrant"].map(quad_code)

        plot_gdf.plot(
            column="quad_code", cmap=cmap, linewidth=0.04,
            edgecolor="white", ax=ax_quad, legend=False,
            missing_kwds={"color": "#DDDDDD"}
        )
        ax_quad.axis("off")
        ax_quad.set_title("Hazard × inequality regimes", fontsize=7.8, fontweight="bold")
        ax_quad.legend(
            handles=[mpatches.Patch(color=C[q], label=q) for q in quad_order],
            loc="lower left", fontsize=5.5, ncol=2
        )
    else:
        ax_quad.text(0.5, 0.5, "Quadrant data unavailable", ha="center", va="center", transform=ax_quad.transAxes)
        ax_quad.axis("off")
    plabel(ax_quad, "b")

    # c) Counts by quadrant
    if "quadrant" in gdf.columns:
        counts = gdf["quadrant"].value_counts().reindex(["HH", "HL", "LH", "LL"]).fillna(0)
        ax_bar.bar(counts.index, counts.values, color=[C[q] for q in counts.index],
                   edgecolor="white", linewidth=0.4, alpha=0.9)
        for i, v in enumerate(counts.values):
            ax_bar.text(i, v + max(counts.values) * 0.02, f"{int(v):,}", ha="center", va="bottom", fontsize=6)
        ax_bar.set_ylabel("Municipalities")
        ax_bar.set_title("Municipalities by regime")
        grid_style(ax_bar, "y")
    else:
        ax_bar.axis("off")
    plabel(ax_bar, "c")

    fig.suptitle(
        "Figure 4 | Main synthesis of spatial flood inequality across Brazil",
        fontsize=8.5, fontweight="bold", color=C["text"], y=0.975
    )
    fig.text(
        0.5, 0.013,
        "Synthesis figure for manuscript-ready presentation and GIS export",
        ha="center", fontsize=5.5, color=C["sub"]
    )

    save_fig(fig, OUT_FIG4, OUT_FIG4_PDF)

# =========================================================
# 10. MAIN
# =========================================================
def main():
    print("\n" + "=" * 72)
    print(" Module 12 v4.0 — Publication Figures + SHP/GPKG Export")
    print("=" * 72)

    print(f"Spatial layer loaded: {len(gdf):,} municipalities")
    print(f"Selected map column: {map_col}")
    print(f"Selected metric column: {metric_col}")

    export_map_layers()

    print("\nRendering Figure 1 ...")
    make_fig1()

    print("Rendering Figure 2 ...")
    make_fig2()

    print("Rendering Figure 3 ...")
    make_fig3()

    print("Rendering Figure 4 ...")
    make_fig4()

    print("\nOutputs:")
    print(f"  {OUT_FIG1}")
    print(f"  {OUT_FIG2}")
    print(f"  {OUT_FIG3}")
    print(f"  {OUT_FIG4}")
    print(f"  GIS exports: {GIS_OUT}")

    print("\nDone.")

if __name__ == "__main__":
    main()
