"""
Project: Flood Inequality Across Brazil
Module: 10_integrate_s2id_with_hazard_social.py
Version: v3.0

Purpose:
  Integrate S2ID disaster panel with hazard and social inequality spatial
  data to produce the final municipal summary dataset for modeling (Module 11).

Key improvements over v2.1:
  - Richer target (disaster_observed_index): composite of afetados,
    mortos, desabrigados and desalojados — all normalized per capita
    and combined via z-score weighted mean
  - S2ID temporal features: historical event frequency, linear trend,
    and recent acceleration (2018–2022 vs 2013–2017) per municipality
  - These S2ID-derived features are exposed to Module 11 (not blacklisted)
    because they reflect structural disaster exposure, not the target itself
  - Output geoparquet contains all columns needed by Module 11, including
    disaster_observed_index and all s2id_feat_* columns

Main outputs:
  - 04_integrated/hazard_social_disaster_municipal_summary_brazil.geoparquet
  - 04_integrated/hazard_social_disaster_municipal_summary_brazil.parquet (no geom)
  - 04_integrated/hazard_social_disaster_municipal_summary_brazil.meta.json
  - 06_figures/fig10_compound_burden.png  (500 DPI)
  - 06_figures/fig10_compound_burden.pdf

Author: Enner H. de Alcantara
Language: English
"""

# ============================================================
# 1. IMPORTS
# ============================================================
import os
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec

from tqdm.auto import tqdm

# ============================================================
# 2. PATHS
# ============================================================
BASE_PATH = Path("/content/drive/MyDrive/Brazil/flood_inequality_project")

HAZARD_SOCIAL_PATH = (
    BASE_PATH / "04_integrated"
    / "hazard_social_inequality_municipal_brazil.geoparquet"
)
S2ID_PANEL_PATH = (
    BASE_PATH / "04_integrated"
    / "s2id_municipal_annual_brazil.parquet"
)

OUTPUT_DIR = BASE_PATH / "04_integrated"
FIG_DIR    = BASE_PATH / "06_figures"
LOG_PATH   = BASE_PATH / "07_logs" / "10_integrate_s2id_with_hazard_social.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

OUTPUT_GEOPARQUET = (
    OUTPUT_DIR / "hazard_social_disaster_municipal_summary_brazil.geoparquet"
)
OUTPUT_PARQUET = (
    OUTPUT_DIR / "hazard_social_disaster_municipal_summary_brazil.parquet"
)
OUTPUT_META = (
    OUTPUT_DIR / "hazard_social_disaster_municipal_summary_brazil.meta.json"
)
OUTPUT_FIG_PNG = FIG_DIR / "fig10_compound_burden.png"
OUTPUT_FIG_PDF = FIG_DIR / "fig10_compound_burden.pdf"

START_YEAR  = 2013
END_YEAR    = 2022
RECENT_CUT  = 2018   # splits period into early (2013-2017) vs recent (2018-2022)
GEOGRAPHIC_CRS = "EPSG:4326"

# ============================================================
# 3. LOGGING
# ============================================================
logging.basicConfig(
    filename=str(LOG_PATH),
    filemode="a",
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
_logger = logging.getLogger(__name__)

def log(msg: str, level: str = "INFO") -> None:
    _logger.info(f"[{level}] {msg}")
    if level in ("WARNING", "ERROR", "SUMMARY"):
        print(f"[{level}] {msg}")

# ============================================================
# 4. HELPERS
# ============================================================
def zscore_series(s: pd.Series) -> pd.Series:
    """Robust z-score: returns NaN where std == 0 or all-NaN."""
    s  = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=1)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd

def safe_divide(a, b, mult=1.0):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where((b > 0) & b.notna(), (a / b) * mult, np.nan)

def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Simple OLS slope; returns NaN if fewer than 2 valid points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan
    xm = x.mean()
    denom = np.sum((x - xm) ** 2)
    return float(np.sum((x - xm) * (y - y.mean())) / denom) if denom > 0 else np.nan

def minmax_01(s: pd.Series) -> pd.Series:
    """Min-max normalization to [0, 1]; returns NaN if constant."""
    s   = pd.to_numeric(s, errors="coerce")
    lo  = s.min(skipna=True)
    hi  = s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)

def std_region(x) -> str:
    if pd.isna(x):
        return ""
    m = {
        "Norte": "North",      "Nordeste": "Northeast",
        "Centro-Oeste": "Center-West", "Sudeste": "Southeast",
        "Sul": "South",
        "N": "North",  "NE": "Northeast", "CO": "Center-West",
        "SE": "Southeast", "S": "South",
    }
    return m.get(str(x).strip(), str(x).strip())

# ============================================================
# 5. LOAD INPUTS
# ============================================================
def load_inputs() -> tuple:
    log("Loading inputs ...", "SUMMARY")
    with tqdm(total=2, desc="Reading files",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        spatial = gpd.read_parquet(HAZARD_SOCIAL_PATH)
        pbar.update(1)
        annual = pd.read_parquet(S2ID_PANEL_PATH)
        pbar.update(1)

    spatial["mun_code"] = spatial["mun_code"].astype(str)
    annual["mun_code"]  = annual["mun_code"].astype(str)
    annual["year"]      = pd.to_numeric(annual["year"], errors="coerce")

    log(f"Spatial: {len(spatial):,} municipalities | "
        f"Annual S2ID: {len(annual):,} rows", "SUMMARY")
    return spatial, annual

# ============================================================
# 6. BUILD DISASTER OBSERVED INDEX (rich target)
# ============================================================
def build_disaster_target(annual: pd.DataFrame,
                           spatial_tab: pd.DataFrame) -> pd.DataFrame:
    """
    Construct disaster_observed_index as a composite of four impact
    dimensions, each normalized per capita before aggregation:

      1. affected_per_1000  → people affected per 1,000 inhabitants
      2. deaths_per_100k    → deaths per 100,000 inhabitants
      3. homeless_per_1000  → homeless per 1,000 inhabitants
      4. displaced_per_1000 → displaced per 1,000 inhabitants

    Each dimension is z-scored nationally, then averaged across
    dimensions with weights [0.4, 0.3, 0.2, 0.1] reflecting
    their relative severity. The result is re-z-scored for the
    final index, then min-max normalized to [0,1] for
    interpretability.

    A raw version (disaster_observed_raw) is also retained.
    """
    log("Building disaster_observed_index ...", "SUMMARY")

    # Merge population for per-capita normalization
    pop = spatial_tab[["mun_code", "population_total"]].copy()
    df  = annual.merge(pop, on="mun_code", how="left")

    pop_col = pd.to_numeric(df["population_total"], errors="coerce")

    df["affected_per_1000"]  = safe_divide(df["s2id_people_affected_sum"],  pop_col, 1_000)
    df["deaths_per_100k"]    = safe_divide(df["s2id_deaths_sum"],            pop_col, 100_000)
    df["homeless_per_1000"]  = safe_divide(df["s2id_homeless_sum"],          pop_col, 1_000)
    df["displaced_per_1000"] = safe_divide(df["s2id_displaced_sum"],         pop_col, 1_000)

    # National z-score of each dimension (pooled across mun × year)
    for col in ["affected_per_1000", "deaths_per_100k",
                "homeless_per_1000", "displaced_per_1000"]:
        df[f"{col}_z"] = zscore_series(df[col])

    WEIGHTS = {
        "affected_per_1000_z" : 0.40,
        "deaths_per_100k_z"   : 0.30,
        "homeless_per_1000_z" : 0.20,
        "displaced_per_1000_z": 0.10,
    }

    # Weighted mean across dimensions (skip NaN)
    def weighted_row_mean(row):
        vals, ws = [], []
        for col, w in WEIGHTS.items():
            v = row[col]
            if np.isfinite(v):
                vals.append(v * w)
                ws.append(w)
        return sum(vals) / sum(ws) if ws else np.nan

    df["disaster_composite_annual"] = df[list(WEIGHTS.keys())].apply(
        weighted_row_mean, axis=1
    )

    # Aggregate to municipality level (mean across years)
    target = (
        df.groupby("mun_code", as_index=False)
        .agg(
            disaster_observed_raw=(
                "disaster_composite_annual", "mean"),
            annual_disaster_observed_index_mean=(
                "disaster_composite_annual", "mean"),
            annual_disaster_observed_index_max=(
                "disaster_composite_annual", "max"),
        )
    )

    # Final z-score → min-max [0,1]
    target["disaster_observed_index"] = minmax_01(
        zscore_series(target["disaster_observed_raw"])
    )

    log(f"Target built: {target['disaster_observed_index'].notna().sum():,} "
        f"non-null values | "
        f"mean={target['disaster_observed_index'].mean():.4f} | "
        f"std={target['disaster_observed_index'].std():.4f}", "SUMMARY")
    return target

# ============================================================
# 7. BUILD S2ID TEMPORAL FEATURES (structural predictors)
# ============================================================
def build_s2id_features(annual: pd.DataFrame) -> pd.DataFrame:
    """
    Build municipality-level features from the S2ID annual panel.
    These are STRUCTURAL features (not the target) and will be
    exposed to Module 11 as legitimate predictors:

      s2id_feat_flood_freq_total       : total flood-like records 2013–2022
      s2id_feat_hydro_freq_total       : total hydrological records
      s2id_feat_flood_freq_rate        : fraction of years with ≥1 flood event
      s2id_feat_flood_trend_slope      : OLS slope of annual flood count
      s2id_feat_flood_acceleration     : recent (2018–2022) mean minus
                                         early (2013–2017) mean flood count
      s2id_feat_deaths_total           : total deaths 2013–2022
      s2id_feat_affected_total         : total people affected
      s2id_feat_peak_year_flood        : year with highest flood count
      s2id_feat_any_flood              : 1 if municipality had ≥1 flood event
    """
    log("Building S2ID temporal features ...", "SUMMARY")
    n_years = END_YEAR - START_YEAR + 1

    records = []
    mun_groups = list(annual.groupby("mun_code", sort=False))

    with tqdm(total=len(mun_groups), desc="S2ID temporal features",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for mun_code, g in mun_groups:
            g = g.sort_values("year")

            flood  = g["s2id_flood_like_records_n"].fillna(0).values
            hydro  = g["s2id_hydrological_records_n"].fillna(0).values
            years  = g["year"].values.astype(float)

            early  = g.loc[g["year"] < RECENT_CUT, "s2id_flood_like_records_n"].fillna(0).values
            recent = g.loc[g["year"] >= RECENT_CUT, "s2id_flood_like_records_n"].fillna(0).values

            records.append({
                "mun_code": mun_code,
                "s2id_feat_flood_freq_total"  : float(flood.sum()),
                "s2id_feat_hydro_freq_total"  : float(hydro.sum()),
                "s2id_feat_flood_freq_rate"   : float((flood > 0).sum()) / n_years,
                "s2id_feat_flood_trend_slope" : ols_slope(years, flood),
                "s2id_feat_flood_acceleration": (
                    float(recent.mean()) - float(early.mean())
                    if len(recent) > 0 and len(early) > 0
                    else np.nan
                ),
                "s2id_feat_deaths_total"      : float(g["s2id_deaths_sum"].fillna(0).sum()),
                "s2id_feat_affected_total"    : float(g["s2id_people_affected_sum"].fillna(0).sum()),
                "s2id_feat_peak_year_flood"   : (
                    float(g.loc[g["s2id_flood_like_records_n"].idxmax(), "year"])
                    if flood.sum() > 0 else np.nan
                ),
                "s2id_feat_any_flood"         : float((flood.sum() > 0)),
            })
            pbar.update(1)

    feats = pd.DataFrame(records)

    # Z-score all continuous features for comparability
    for col in [
        "s2id_feat_flood_freq_total",
        "s2id_feat_hydro_freq_total",
        "s2id_feat_flood_trend_slope",
        "s2id_feat_flood_acceleration",
        "s2id_feat_deaths_total",
        "s2id_feat_affected_total",
    ]:
        feats[f"{col}_z"] = zscore_series(feats[col])

    log(f"S2ID features built: {len(feats):,} municipalities | "
        f"{len(feats.columns)-1} feature columns", "SUMMARY")
    return feats

# ============================================================
# 8. BUILD COMPOUND INDEX AND QUADRANT
# ============================================================
def build_compound_index(summary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compound index = z-score( hazard_extremes + social_inequality + disaster )
    Quadrant based on medians of hazard and social dimensions.
    """
    hz = summary["hazard_recent_extremes_index"]
    sc = summary["social_inequality_index"]
    ds = summary["disaster_observed_index"]

    # All three are already z-scored or [0,1]; standardize before summing
    hz_z = zscore_series(hz)
    sc_z = zscore_series(sc)
    ds_z = zscore_series(ds)

    summary = summary.copy()
    summary["hazard_social_disaster_compound_raw"]   = hz_z + sc_z + ds_z
    summary["hazard_social_disaster_compound_index"] = zscore_series(
        summary["hazard_social_disaster_compound_raw"]
    )

    # Quartile-based compound class
    q25, q75 = (summary["hazard_social_disaster_compound_index"].quantile(0.25),
                summary["hazard_social_disaster_compound_index"].quantile(0.75))
    summary["compound_class_q"] = pd.cut(
        summary["hazard_social_disaster_compound_index"],
        bins=[-np.inf, q25, q75, np.inf],
        labels=["low", "medium", "high"],
    )

    # Hazard × social quadrant
    hz_med = hz.median(skipna=True)
    sc_med = sc.median(skipna=True)
    summary["hazard_social_quadrant"] = np.select(
        [
            hz.isna() | sc.isna(),
            hz.ge(hz_med) & sc.ge(sc_med),
            hz.ge(hz_med) & sc.lt(sc_med),
            hz.lt(hz_med) & sc.ge(sc_med),
        ],
        [
            "missing",
            "high_hazard_high_inequality",
            "high_hazard_low_inequality",
            "low_hazard_high_inequality",
        ],
        default="low_hazard_low_inequality",
    )

    # Triple burden flag: above median in all three dimensions
    ds_med = ds.median(skipna=True)
    summary["triple_burden_flag"] = (
        hz.ge(hz_med) & sc.ge(sc_med) & ds.ge(ds_med)
    ).astype(int)

    return summary

# ============================================================
# 9. MAIN BUILD FUNCTION
# ============================================================
def build_summary(spatial: gpd.GeoDataFrame,
                   annual: pd.DataFrame) -> gpd.GeoDataFrame:

    spatial_tab = pd.DataFrame(spatial.drop(columns="geometry"))

    # --- Target ---
    target = build_disaster_target(annual, spatial_tab)

    # --- S2ID structural features ---
    s2id_feats = build_s2id_features(annual)

    # --- Merge everything onto spatial base ---
    log("Merging all components onto spatial base ...", "SUMMARY")
    with tqdm(total=3, desc="Merging",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        summary = spatial.merge(target,     on="mun_code", how="left")
        pbar.update(1)
        summary = summary.merge(s2id_feats, on="mun_code", how="left")
        pbar.update(1)
        summary = build_compound_index(summary)
        pbar.update(1)

    log(f"Final summary: {len(summary):,} municipalities | "
        f"{len(summary.columns)} columns", "SUMMARY")
    return summary

# ============================================================
# 10. SAVE OUTPUTS
# ============================================================
def save_outputs(summary: gpd.GeoDataFrame) -> None:
    log("Saving outputs ...", "SUMMARY")
    summary = summary.sort_values(
        ["uf_sigla", "mun_name"] if "mun_name" in summary.columns
        else ["mun_code"]
    ).reset_index(drop=True)

    tmps = {
        "geo" : str(OUTPUT_GEOPARQUET) + ".tmp",
        "flat": str(OUTPUT_PARQUET)    + ".tmp",
        "meta": str(OUTPUT_META)       + ".tmp",
    }
    for p in tmps.values():
        if os.path.exists(p):
            os.remove(p)

    with tqdm(total=3, desc="Writing files",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        summary.to_parquet(tmps["geo"], index=False)
        pbar.update(1)

        flat = pd.DataFrame(summary.drop(columns="geometry"))
        flat.to_parquet(tmps["flat"], index=False)
        pbar.update(1)

        meta = {
            "project"          : "Flood Inequality Across Brazil",
            "module"           : "10_integrate_s2id_with_hazard_social.py",
            "version"          : "v3.0",
            "status"           : "completed",
            "created_at"       : datetime.now().isoformat(),
            "source_spatial"   : str(HAZARD_SOCIAL_PATH),
            "source_s2id"      : str(S2ID_PANEL_PATH),
            "start_year"       : START_YEAR,
            "end_year"         : END_YEAR,
            "recent_cutoff"    : RECENT_CUT,
            "n_municipalities" : int(len(summary)),
            "target_col"       : "disaster_observed_index",
            "target_construction": (
                "Weighted z-score composite of afetados (0.4), mortos (0.3), "
                "desabrigados (0.2), desalojados (0.1) — per capita — "
                "aggregated to municipality level, then min-max [0,1]"
            ),
            "s2id_feature_prefix": "s2id_feat_",
            "output_geoparquet": str(OUTPUT_GEOPARQUET),
            "output_parquet"   : str(OUTPUT_PARQUET),
            "columns"          : list(summary.columns),
        }
        with open(tmps["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
        pbar.update(1)

    os.replace(tmps["geo"],  OUTPUT_GEOPARQUET)
    os.replace(tmps["flat"], OUTPUT_PARQUET)
    os.replace(tmps["meta"], OUTPUT_META)

    log(f"GeoParquet : {OUTPUT_GEOPARQUET}", "SUMMARY")
    log(f"Parquet    : {OUTPUT_PARQUET}",    "SUMMARY")
    log(f"Meta       : {OUTPUT_META}",       "SUMMARY")

# ============================================================
# 11. FIGURE 10
# ============================================================
def make_figure(summary: gpd.GeoDataFrame, show: bool = True) -> str:
    """
    6-panel publication-grade composite (500 DPI):
      a) Map: compound burden index
      b) Map: triple burden hotspots
      c) Scatter: hazard vs disaster (color = social inequality)
      d) Distribution: disaster_observed_index
      e) Bar: triple burden share by region
      f) Correlation heatmap: hazard × social × disaster
    """
    matplotlib.rcParams.update({
        "font.family"      : "sans-serif",
        "font.sans-serif"  : ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size"        : 7,
        "axes.linewidth"   : 0.5,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "figure.dpi"       : 72,
        "pdf.fonttype"     : 42,
        "ps.fonttype"      : 42,
    })

    C = {
        "bg"     : "#FAFAF8",
        "text_hd": "#111827",
        "text_sm": "#6B7280",
        "border" : "#D1D5DB",
        "accent" : "#B45309",
        "blue"   : "#2166AC",
        "teal"   : "#1B9E77",
        "red"    : "#C0504D",
        "gray"   : "#6B7280",
    }

    REGION_COLORS = {
        "North"      : "#4A90D9",
        "Northeast"  : "#E8A838",
        "Center-West": "#6DB56D",
        "Southeast"  : "#C0504D",
        "South"      : "#9B59B6",
    }

    gdf = summary.copy()
    df  = pd.DataFrame(summary.drop(columns="geometry"))

    reg_col = next((c for c in df.columns
                    if "region" in c.lower() and df[c].dtype == object), None)
    if reg_col:
        df["_region"]  = df[reg_col].apply(std_region)
        gdf["_region"] = df["_region"]

    fig = plt.figure(figsize=(7.2, 9.3))
    fig.patch.set_facecolor(C["bg"])
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        left=0.06, right=0.97, top=0.93, bottom=0.06,
        hspace=0.45, wspace=0.30,
    )

    # a) Compound index map
    ax = fig.add_subplot(gs[0, 0])
    gdf.plot(column="hazard_social_disaster_compound_index",
             cmap="RdBu_r", ax=ax, legend=True,
             legend_kwds={"shrink": 0.5, "label": "z"})
    ax.set_title("Compound burden index", fontsize=7, color=C["text_hd"], pad=4)
    ax.axis("off")
    ax.text(0.03, 0.97, "a", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", color=C["text_hd"])

    # b) Triple burden map
    ax = fig.add_subplot(gs[0, 1])
    gdf.plot(color=gdf["triple_burden_flag"].map({0: "#D0D0D0", 1: "#7f0000"}),
             ax=ax, linewidth=0.05, edgecolor="white")
    ax.set_title("Triple burden hotspots", fontsize=7, color=C["text_hd"], pad=4)
    ax.axis("off")
    ax.text(0.03, 0.97, "b", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    n_triple = int(gdf["triple_burden_flag"].sum())
    ax.text(0.97, 0.03,
            f"{n_triple:,} municipalities\n({n_triple/len(gdf)*100:.1f}%)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.5, color="#7f0000",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=C["border"], lw=0.5, alpha=0.9))

    # c) Scatter: hazard vs disaster
    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(
        df["hazard_recent_extremes_index"],
        df["disaster_observed_index"],
        c=df["social_inequality_index"],
        cmap="viridis", s=3, alpha=0.45, linewidths=0,
    )
    plt.colorbar(sc, ax=ax, shrink=0.7,
                 label="Social inequality (z)")
    ax.set_xlabel("Hazard extremes index (z)", fontsize=6, color=C["text_sm"])
    ax.set_ylabel("Disaster observed index [0-1]", fontsize=6, color=C["text_sm"])
    ax.set_title("Hazard vs observed disaster", fontsize=7, color=C["text_hd"], pad=4)
    ax.grid(linewidth=0.18, color=C["border"], alpha=0.7)
    ax.text(0.03, 0.97, "c", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", color=C["text_hd"])

    # d) Distribution of disaster_observed_index
    ax = fig.add_subplot(gs[1, 1])
    vals = df["disaster_observed_index"].dropna().values
    ax.hist(vals, bins=40, color=C["blue"], edgecolor="white",
            linewidth=0.3, alpha=0.85, zorder=3)
    ax.axvline(np.median(vals), color=C["accent"], lw=0.9, ls="--", zorder=4)
    ax.set_xlabel("Disaster observed index [0-1]", fontsize=6, color=C["text_sm"])
    ax.set_ylabel("Municipalities", fontsize=6, color=C["text_sm"])
    ax.set_title("Target distribution", fontsize=7, color=C["text_hd"], pad=4)
    ax.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7)
    skew_val = pd.Series(vals).skew()
    ax.text(0.97, 0.97, f"n={len(vals):,}\nmedian={np.median(vals):.3f}\nskew={skew_val:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5, color=C["text_sm"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=C["border"], lw=0.5, alpha=0.9))
    ax.text(0.03, 0.97, "d", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", color=C["text_hd"])

    # e) Triple burden share by region
    ax = fig.add_subplot(gs[2, 0])
    REG_ORDER = ["North", "Northeast", "Center-West", "Southeast", "South"]
    if reg_col and "_region" in df.columns:
        shares = []
        for reg in REG_ORDER:
            sub = df[df["_region"] == reg]
            shares.append(sub["triple_burden_flag"].mean() if len(sub) > 0 else 0)
        rcols = [REGION_COLORS.get(r, C["gray"]) for r in REG_ORDER]
        ax.barh(REG_ORDER, shares, color=rcols,
                edgecolor="white", linewidth=0.3, alpha=0.87)
        for i, v in enumerate(shares):
            ax.text(v + 0.003, i, f"{v*100:.1f}%",
                    va="center", fontsize=5.5, color=C["text_sm"])
    else:
        ax.barh(["Brazil"], [df["triple_burden_flag"].mean()])
    ax.set_xlabel("Share of municipalities", fontsize=6, color=C["text_sm"])
    ax.set_title("Triple burden share by region", fontsize=7, color=C["text_hd"], pad=4)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", linewidth=0.18, color=C["border"], alpha=0.7)
    ax.text(0.03, 0.97, "e", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", color=C["text_hd"])

    # f) Correlation heatmap
    ax = fig.add_subplot(gs[2, 1])
    corr_cols = [
        "hazard_recent_extremes_index",
        "social_inequality_index",
        "disaster_observed_index",
        "hazard_trend_index",
        "adaptive_capacity_index",
    ]
    valid_cols = [c for c in corr_cols if c in df.columns]
    short = {
        "hazard_recent_extremes_index": "Hazard",
        "social_inequality_index"     : "Social",
        "disaster_observed_index"     : "Disaster",
        "hazard_trend_index"          : "Trend",
        "adaptive_capacity_index"     : "Capacity",
    }
    corr = df[valid_cols].corr()
    im   = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7)
    labels = [short.get(c, c) for c in valid_cols]
    ax.set_xticks(range(len(valid_cols)))
    ax.set_yticks(range(len(valid_cols)))
    ax.set_xticklabels(labels, fontsize=5.5, rotation=30, ha="right")
    ax.set_yticklabels(labels, fontsize=5.5)
    for i in range(len(valid_cols)):
        for j in range(len(valid_cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=4.5,
                    color="white" if abs(v) > 0.5 else C["text_hd"])
    ax.set_title("Index correlation matrix", fontsize=7, color=C["text_hd"], pad=4)
    ax.text(0.03, 0.97, "f", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", color=C["text_hd"])

    fig.suptitle(
        "Integrated flood inequality burden in Brazil",
        fontsize=8.5, color=C["text_hd"], y=0.975,
    )
    fig.text(
        0.5, 0.02,
        "Compound index = z(hazard) + z(social) + z(disaster) | "
        "Triple burden = above median in all three dimensions | "
        "Target: weighted per-capita impact composite",
        ha="center", va="center", fontsize=5.5, color=C["text_sm"],
    )

    fig.savefig(OUTPUT_FIG_PNG, dpi=500, bbox_inches="tight",
                facecolor=C["bg"])
    fig.savefig(OUTPUT_FIG_PDF, bbox_inches="tight",
                facecolor=C["bg"])

    if show:
        try:
            from IPython.display import display
            display(fig)
        except Exception:
            plt.show()

    plt.close(fig)
    return str(OUTPUT_FIG_PNG)

# ============================================================
# 12. MAIN
# ============================================================
def main():
    print("\n" + "=" * 65)
    print("  Module 10 — S2ID × Hazard × Social Integration")
    print("  Version: v3.0 | Rich target + S2ID features + publication figure")
    print("=" * 65 + "\n")

    spatial, annual = load_inputs()
    summary = build_summary(spatial, annual)
    save_outputs(summary)

    print("\n  Rendering publication-grade figure ...")
    fig_path = make_figure(summary, show=True)
    print(f"  Figure saved: {fig_path}")

    print("\n" + "=" * 65)
    print("  ✓ Module 10 completed successfully.")
    print(f"  Output : {OUTPUT_GEOPARQUET}")
    print(f"  Target : disaster_observed_index")
    print(f"  S2ID features prefix : s2id_feat_")
    print("=" * 65)
    print()
    print("  ⚠  IMPORTANT — Module 11 adjustment required:")
    print("  Remove 's2id_feat_' prefix from blacklist in choose_features()")
    print("  Keep 's2id_' blocked (raw counts) but allow 's2id_feat_' through.")
    print()

# Colab: chama main() diretamente (if __name__ não dispara em células)
main()
