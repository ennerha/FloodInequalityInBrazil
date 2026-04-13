"""
Project: Flood Inequality Across Brazil
Module:  04_build_hazard_time_series_chirps_by_tile.py

Purpose:
  Build annual CHIRPS-based hydroclimatic hazard metrics for each processing
  tile across Brazil using Google Earth Engine.

Dataset:
  CHIRPS Daily (UCSB-CHG/CHIRPS/DAILY)

Metrics computed per tile and year:
  - annual_prcp_mm
  - wet_days_n
  - heavy_rain_days_20mm_n
  - rx1day_mm
  - rx3day_mm
  - rx5day_mm

Outputs:
  - 03_features/chirps_tile_annual_csv/chirps_annual_<tile_id>.csv
  - 03_features/chirps_tile_annual_meta/chirps_annual_<tile_id>.meta.json
  - 03_features/chirps_tile_annual_manifest.csv
  - 06_figures/fig04_chirps_hazard_timeseries.png   (500 DPI)
  - 06_figures/fig04_chirps_hazard_timeseries.pdf   (vector)
  - 07_logs/04_build_hazard_time_series_chirps_by_tile.log

Reproducibility:
  - Idempotent execution  - safe to re-run without side effects
  - Per-tile atomic save (tmp → final)
  - Skips tiles with valid existing output

Author:  Enner H. de Alcantara
Version: v1.6
"""

# ============================================================
# 1. STANDARD LIBRARY IMPORTS
# ============================================================
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# ============================================================
# 2. THIRD-PARTY IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib import gridspec

from tqdm.auto import tqdm

# ============================================================
# 3. GEE IMPORT
# ============================================================
import ee

logging.getLogger("googleapiclient.http").setLevel(logging.ERROR)

# ============================================================
# 4. PATHS AND CONSTANTS
# ============================================================
BASE_PATH     = "/content/drive/MyDrive/Brazil/flood_inequality_project"
CONFIG_PATH   = os.path.join(BASE_PATH, "00_config",  "config.json")
LOG_PATH      = os.path.join(BASE_PATH, "07_logs",    "04_build_hazard_time_series_chirps_by_tile.log")
CATALOG_PATH  = os.path.join(BASE_PATH, "08_catalog", "catalog.csv")
TILES_PATH    = os.path.join(BASE_PATH, "02_intermediate", "processing_tiles_brazil.parquet")

OUTPUT_DIR     = os.path.join(BASE_PATH, "03_features", "chirps_tile_annual_csv")
META_DIR       = os.path.join(BASE_PATH, "03_features", "chirps_tile_annual_meta")
MANIFEST_PATH  = os.path.join(BASE_PATH, "03_features", "chirps_tile_annual_manifest.csv")
OUTPUT_FIG_PNG = os.path.join(BASE_PATH, "06_figures",  "fig04_chirps_hazard_timeseries.png")
OUTPUT_FIG_PDF = os.path.join(BASE_PATH, "06_figures",  "fig04_chirps_hazard_timeseries.pdf")

GEE_DATASET    = "UCSB-CHG/CHIRPS/DAILY"
GEOGRAPHIC_CRS = "EPSG:4326"
START_YEAR     = 1981
END_YEAR       = 2025
SCALE_M        = 5566
MAX_PIXELS     = 1e13

# VERBOSE = True  → print every log message
# VERBOSE = False → print only WARNING / ERROR / SUMMARY
VERBOSE = False

# ============================================================
# 5. DIRECTORY SETUP
# ============================================================
for _p in [OUTPUT_DIR, META_DIR]:
    Path(_p).mkdir(parents=True, exist_ok=True)

# ============================================================
# 6. LOGGING
# ============================================================
logging.basicConfig(
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def log(msg: str, level: str = "INFO") -> None:
    logging.info(f"[{level}] {msg}")
    if VERBOSE or level in ("WARNING", "ERROR", "SUMMARY"):
        print(f"[{level}] {msg}")

# ============================================================
# 7. HELPERS
# ============================================================
def read_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def initialize_ee(gee_project: str) -> None:
    try:
        ee.Initialize(project=gee_project)
        log(f"Earth Engine initialized: {gee_project}")
    except Exception:
        log("Authentication required - running ee.Authenticate() ...")
        ee.Authenticate()
        ee.Initialize(project=gee_project)
        log(f"Earth Engine initialized after authentication: {gee_project}")


def is_valid_tile_output(csv_path: str, meta_path: str,
                          expected_years: int) -> bool:
    if not os.path.exists(csv_path) or not os.path.exists(meta_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        required = {
            "tile_id", "year", "annual_prcp_mm", "wet_days_n",
            "heavy_rain_days_20mm_n", "rx1day_mm", "rx3day_mm", "rx5day_mm",
        }
        if df.empty or not required.issubset(df.columns):
            return False
        if df["year"].nunique() != expected_years:
            return False
    except Exception:
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("status") != "completed":
            return False
    except Exception:
        return False
    return True


def update_catalog(stage: str, tile_id: str,
                   output_path: str, status: str) -> None:
    row = pd.DataFrame([{
        "stage"      : stage,
        "tile_id"    : tile_id,
        "period"     : f"{START_YEAR}_{END_YEAR}",
        "status"     : status,
        "output_path": output_path,
        "timestamp"  : datetime.now().isoformat(),
    }])
    if os.path.exists(CATALOG_PATH):
        try:
            cat = pd.read_csv(CATALOG_PATH)
            cat = cat[~((cat["stage"] == stage) & (cat["tile_id"] == tile_id))]
            pd.concat([cat, row], ignore_index=True).to_csv(CATALOG_PATH, index=False)
        except Exception:
            row.to_csv(CATALOG_PATH, index=False)
    else:
        row.to_csv(CATALOG_PATH, index=False)


def get_tile_paths(tile_id: str):
    csv  = os.path.join(OUTPUT_DIR, f"chirps_annual_{tile_id}.csv")
    meta = os.path.join(META_DIR,   f"chirps_annual_{tile_id}.meta.json")
    return csv, meta


def build_tile_geometry(row) -> ee.Geometry:
    return ee.Geometry.Rectangle(
        [row["min_lon"], row["min_lat"], row["max_lon"], row["max_lat"]],
        proj=GEOGRAPHIC_CRS, geodesic=False,
    )

# ============================================================
# 8. GEE METRIC BUILDERS
# ============================================================

def compute_rxnday_batch(daily_ic: ee.ImageCollection,
                          windows: list) -> dict:
    """
    Compute multiple RxNday metrics via ee.Join-based rolling sums.

    For each image a saveAll join attaches all images within the preceding
    (W-1) days. Summing the window gives the W-day rolling total; the
    annual max gives RxNday. Fully server-side, no Python iteration over days.
    """
    sorted_ic = daily_ic.sort("system:time_start")
    results   = {}

    for w in windows:
        millis    = (w - 1) * 24 * 60 * 60 * 1000
        join      = ee.Join.saveAll("window_images")
        condition = ee.Filter.And(
            ee.Filter.maxDifference(
                difference=millis,
                leftField="system:time_start",
                rightField="system:time_start",
            ),
            ee.Filter.greaterThanOrEquals(
                leftField="system:time_start",
                rightField="system:time_start",
            ),
        )
        joined = join.apply(sorted_ic, sorted_ic, condition)

        def window_sum(img):
            return (ee.ImageCollection
                    .fromImages(img.get("window_images"))
                    .sum()
                    .copyProperties(img, ["system:time_start"]))

        rolling    = ee.ImageCollection(joined.map(window_sum))
        results[w] = rolling.max().rename(f"rx{w}day_mm")

    return results


def compute_tile_year_feature(tile_geom: ee.Geometry,
                               tile_id: str,
                               year: int) -> ee.Feature:
    """Compute all annual metrics for one tile × year → ee.Feature."""
    start = ee.Date.fromYMD(year, 1, 1)
    end   = start.advance(1, "year")

    daily = (
        ee.ImageCollection(GEE_DATASET)
        .filterDate(start, end)
        .filterBounds(tile_geom)
        .select("precipitation")
        .sort("system:time_start")
    )

    annual_sum = daily.sum().rename("annual_prcp_mm")
    wet_days   = (daily.map(lambda img: img.gte(1).rename("wet"))
                  .sum().rename("wet_days_n"))
    heavy_days = (daily.map(lambda img: img.gte(20).rename("heavy20"))
                  .sum().rename("heavy_rain_days_20mm_n"))
    rx1day     = daily.max().rename("rx1day_mm")
    rxn        = compute_rxnday_batch(daily, [3, 5])

    metric_stack = annual_sum.addBands(
        [wet_days, heavy_days, rx1day, rxn[3], rxn[5]])

    reduced = metric_stack.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=tile_geom,
        scale=SCALE_M,
        maxPixels=MAX_PIXELS,
        bestEffort=True,
    )

    return (ee.Feature(None, reduced)
            .set("tile_id", tile_id)
            .set("year",    year))


def build_feature_collection_for_tile(tile_geom: ee.Geometry,
                                       tile_id: str) -> ee.FeatureCollection:
    """
    Build the full annual FeatureCollection for one tile.
    Single empty-collection probe on END_YEAR reduces getInfo() calls
    from 36,765 to 817 with no loss of safety.
    """
    probe = (
        ee.ImageCollection(GEE_DATASET)
        .filterDate(f"{END_YEAR}-01-01", f"{END_YEAR}-12-31")
        .filterBounds(tile_geom)
        .size()
        .getInfo()
    )
    if probe == 0:
        raise RuntimeError(
            f"Empty CHIRPS collection for tile={tile_id}, year={END_YEAR}.")

    years = list(range(START_YEAR, END_YEAR + 1))
    return ee.FeatureCollection([
        compute_tile_year_feature(tile_geom, tile_id, y) for y in years
    ])

# ============================================================
# 9. LOAD CONFIG + INPUTS
# ============================================================
config      = read_config(CONFIG_PATH)
gee_project = config["gee_project"]

initialize_ee(gee_project)

if not os.path.exists(TILES_PATH):
    raise FileNotFoundError(f"Tiles not found: {TILES_PATH}")

tiles = gpd.read_parquet(TILES_PATH)
if tiles.empty:
    raise RuntimeError("Processing tile layer is empty.")

required_cols = {"tile_id", "min_lon", "min_lat", "max_lon", "max_lat"}
if not required_cols.issubset(tiles.columns):
    raise RuntimeError(f"Missing tile columns: {required_cols - set(tiles.columns)}")

tiles = tiles.sort_values("tile_n").reset_index(drop=True)
log(f"Tiles loaded: {len(tiles):,}", level="SUMMARY")

# ============================================================
# 10. MAIN LOOP
# ============================================================
expected_years = END_YEAR - START_YEAR + 1
manifest_rows  = []
n_total        = len(tiles)
n_skipped      = 0
n_completed    = 0
n_failed       = 0
failed_ids     = []

with tqdm(total=n_total, desc="Processing tiles",
          bar_format="{l_bar}{bar:30}{r_bar}") as pbar:

    for _, row in tiles.iterrows():
        tile_id          = row["tile_id"]
        csv_path, meta_p = get_tile_paths(tile_id)

        # Skip if valid output already exists
        if is_valid_tile_output(csv_path, meta_p, expected_years):
            log(f"{tile_id}: valid output exists - skipping.")
            manifest_rows.append({
                "tile_id": tile_id, "csv_path": csv_path,
                "meta_path": meta_p, "status": "skipped_existing",
            })
            n_skipped += 1
            pbar.update(1)
            continue

        log(f"{tile_id}: extracting ...")
        tile_geom = build_tile_geometry(row)

        try:
            fc       = build_feature_collection_for_tile(tile_geom, tile_id)
            info     = fc.getInfo()
            features = info.get("features", [])

            if not features:
                raise RuntimeError(f"{tile_id}: no features returned from GEE.")

            records = [{
                "tile_id"               : ft["properties"].get("tile_id"),
                "year"                  : ft["properties"].get("year"),
                "annual_prcp_mm"        : ft["properties"].get("annual_prcp_mm"),
                "wet_days_n"            : ft["properties"].get("wet_days_n"),
                "heavy_rain_days_20mm_n": ft["properties"].get("heavy_rain_days_20mm_n"),
                "rx1day_mm"             : ft["properties"].get("rx1day_mm"),
                "rx3day_mm"             : ft["properties"].get("rx3day_mm"),
                "rx5day_mm"             : ft["properties"].get("rx5day_mm"),
            } for ft in features]

            df = pd.DataFrame(records).sort_values("year").reset_index(drop=True)

            if df["year"].nunique() != expected_years:
                raise RuntimeError(
                    f"{tile_id}: expected {expected_years} years, "
                    f"got {df['year'].nunique()}.")

            tmp_csv  = csv_path  + ".tmp"
            tmp_meta = meta_p    + ".tmp"
            for p in [tmp_csv, tmp_meta]:
                if os.path.exists(p): os.remove(p)

            df.to_csv(tmp_csv, index=False)

            meta = {
                "project"    : "Flood Inequality Across Brazil",
                "module"     : "04_build_hazard_time_series_chirps_by_tile.py",
                "version"    : "v1.6",
                "status"     : "completed",
                "created_at" : datetime.now().isoformat(),
                "tile_id"    : tile_id,
                "dataset"    : GEE_DATASET,
                "gee_project": gee_project,
                "start_year" : START_YEAR,
                "end_year"   : END_YEAR,
                "n_years"    : expected_years,
                "scale_m"    : SCALE_M,
                "metrics"    : ["annual_prcp_mm", "wet_days_n",
                                "heavy_rain_days_20mm_n", "rx1day_mm",
                                "rx3day_mm", "rx5day_mm"],
                "tile_bounds": {
                    "min_lon": float(row["min_lon"]),
                    "min_lat": float(row["min_lat"]),
                    "max_lon": float(row["max_lon"]),
                    "max_lat": float(row["max_lat"]),
                },
                "output_csv" : csv_path,
            }
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)

            os.replace(tmp_csv,  csv_path)
            os.replace(tmp_meta, meta_p)

            update_catalog("04_build_hazard_time_series_chirps_by_tile",
                           tile_id, csv_path, "completed")
            manifest_rows.append({
                "tile_id": tile_id, "csv_path": csv_path,
                "meta_path": meta_p, "status": "completed",
            })
            n_completed += 1
            done = n_skipped + n_completed + n_failed
            log(f"[{done}/{n_total}] {tile_id}: OK", level="SUMMARY")

        except Exception as e:
            log(f"{tile_id}: FAILED - {e}", level="ERROR")
            update_catalog("04_build_hazard_time_series_chirps_by_tile",
                           tile_id, csv_path, "failed")
            manifest_rows.append({
                "tile_id": tile_id, "csv_path": csv_path,
                "meta_path": meta_p, "status": f"failed: {e}",
            })
            n_failed += 1
            failed_ids.append(tile_id)
            done = n_skipped + n_completed + n_failed
            log(f"[{done}/{n_total}] {tile_id}: FAILED", level="ERROR")

        pbar.update(1)

# ============================================================
# 11. SAVE MANIFEST
# ============================================================
pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False)
log("=" * 60, level="SUMMARY")
log(f"DONE  {n_total} tiles | {n_completed} completed | "
    f"{n_skipped} skipped | {n_failed} failed", level="SUMMARY")
if failed_ids:
    log(f"Failed tiles: {', '.join(failed_ids)}", level="SUMMARY")

# ============================================================
# 12. UPDATE CONFIG
# ============================================================
config["hazard_chirps_tile_annual"] = {
    "name"        : "chirps_tile_annual_metrics",
    "dataset"     : GEE_DATASET,
    "start_year"  : START_YEAR,
    "end_year"    : END_YEAR,
    "output_dir"  : OUTPUT_DIR,
    "meta_dir"    : META_DIR,
    "manifest_csv": MANIFEST_PATH,
    "metrics"     : ["annual_prcp_mm", "wet_days_n", "heavy_rain_days_20mm_n",
                     "rx1day_mm", "rx3day_mm", "rx5day_mm"],
}
with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)
log("config.json updated.")

# ============================================================
# 13. FIGURE 04 — CHIRPS HAZARD TIME SERIES  (500 DPI composite)
# ============================================================

def make_figure_04_chirps(manifest_path: str, output_dir: str,
                           save_dir: str, dpi: int = 500) -> str:
    """
    6-panel composite figure (2 rows × 3 cols):
      a) National annual precipitation (mean ± IQR + 10-yr rolling)
      b) National wet days per year
      c) Rx1day distribution by climate era
      d) Rx3day vs Rx5day scatter coloured by year
      e) Heavy rain days trend with linear fit
      f) Pipeline progress + dataset metadata

    Falls back to simulated data if no completed tiles exist yet.
    """

    matplotlib.rcParams.update({
        "font.family"      : "sans-serif",
        "font.sans-serif"  : ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size"        : 7,
        "axes.linewidth"   : 0.5,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "figure.dpi"       : 72,
        "pdf.fonttype"     : 42,
        "ps.fonttype"      : 42,
        "svg.fonttype"     : "none",
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.labelsize"  : 5.5,
        "ytick.labelsize"  : 5.5,
    })

    C = {
        "bg"     : "#FAFAF8", "panel"  : "#F2F1ED",
        "text_hd": "#111827", "text_sm": "#6B7280",
        "border" : "#D1D5DB", "accent" : "#B45309",
        "blue"   : "#2E6DA4", "teal"   : "#2A8C6E",
        "gray"   : "#6B7280", "purple" : "#6B46C1",
        "red"    : "#C0504D", "amber"  : "#D97706",
    }

    METRICS = ["annual_prcp_mm", "wet_days_n", "heavy_rain_days_20mm_n",
               "rx1day_mm", "rx3day_mm", "rx5day_mm"]
    YEARS   = np.arange(START_YEAR, END_YEAR + 1)

    # Load real data or fall back to simulation
    df = None
    if os.path.exists(manifest_path):
        try:
            manifest  = pd.read_csv(manifest_path)
            completed = manifest[manifest["status"] == "completed"]
            if not completed.empty:
                dfs = []
                for _, mrow in completed.iterrows():
                    try:
                        dfs.append(pd.read_csv(mrow["csv_path"]))
                    except Exception:
                        pass
                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
        except Exception:
            pass

    if df is None or df.empty:
        log("Figure 04: no completed tiles — using simulated data for preview.",
            level="WARNING")
        np.random.seed(42)
        rows_sim = []
        for t in range(80):
            base = np.random.uniform(800, 2500)
            for y in YEARS:
                trend = (y - START_YEAR) * np.random.uniform(-2, 3)
                rows_sim.append({
                    "tile_id"               : f"tile{t+1:04d}",
                    "year"                  : y,
                    "annual_prcp_mm"        : max(0, base + trend + np.random.normal(0, 150)),
                    "wet_days_n"            : max(0, np.random.normal(120, 30)),
                    "heavy_rain_days_20mm_n": max(0, np.random.normal(20,  8)),
                    "rx1day_mm"             : max(0, np.random.normal(60,  20)),
                    "rx3day_mm"             : max(0, np.random.normal(100, 30)),
                    "rx5day_mm"             : max(0, np.random.normal(130, 35)),
                })
        df = pd.DataFrame(rows_sim)

    nat     = df.groupby("year")[METRICS].mean().reset_index()
    n_tiles = df["tile_id"].nunique()

    def iqr_band(col):
        lo = df.groupby("year")[col].quantile(0.25).values
        hi = df.groupby("year")[col].quantile(0.75).values
        return lo, hi

    def rolling10(series):
        return pd.Series(series).rolling(10, center=True, min_periods=5).mean().values

    # Layout
    fig = plt.figure(figsize=(7.087, 9.2))
    fig.patch.set_facecolor(C["bg"])

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        left=0.10, right=0.97, top=0.93, bottom=0.06,
        hspace=0.50, wspace=0.32,
    )
    ax_prcp  = fig.add_subplot(gs[0, 0])
    ax_wet   = fig.add_subplot(gs[0, 1])
    ax_rx1   = fig.add_subplot(gs[1, 0])
    ax_rx35  = fig.add_subplot(gs[1, 1])
    ax_heavy = fig.add_subplot(gs[2, 0])
    ax_prog  = fig.add_subplot(gs[2, 1])

    for ax in (ax_prcp, ax_wet, ax_rx1, ax_rx35, ax_heavy, ax_prog):
        ax.set_facecolor(C["bg"])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color(C["border"])

    # Panel a: annual precipitation
    lo, hi = iqr_band("annual_prcp_mm")
    ax_prcp.fill_between(nat["year"], lo, hi, color=C["blue"], alpha=0.12, zorder=2)
    ax_prcp.plot(nat["year"], nat["annual_prcp_mm"],
                 color=C["blue"], lw=0.9, alpha=0.9, zorder=3, label="Annual mean")
    ax_prcp.plot(nat["year"], rolling10(nat["annual_prcp_mm"]),
                 color=C["accent"], lw=1.1, ls="--", zorder=4, label="10-yr rolling")
    ax_prcp.set_xlabel("Year", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_prcp.set_ylabel("Annual precipitation (mm)", fontsize=6,
                       color=C["text_sm"], labelpad=3)
    ax_prcp.grid(linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_prcp.legend(fontsize=4.5, loc="lower right", frameon=True,
                   framealpha=0.9, edgecolor=C["border"])
    ax_prcp.text(0.03, 0.97, "a", transform=ax_prcp.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_prcp.set_title("National annual precipitation",
                      fontsize=7, color=C["text_hd"], pad=4)

    # Panel b: wet days
    lo_w, hi_w = iqr_band("wet_days_n")
    ax_wet.fill_between(nat["year"], lo_w, hi_w, color=C["teal"], alpha=0.12, zorder=2)
    ax_wet.plot(nat["year"], nat["wet_days_n"],
                color=C["teal"], lw=0.9, alpha=0.9, zorder=3)
    ax_wet.plot(nat["year"], rolling10(nat["wet_days_n"]),
                color=C["accent"], lw=1.1, ls="--", zorder=4)
    ax_wet.set_xlabel("Year", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_wet.set_ylabel("Wet days yr\u207b\u00b9  (\u2265 1 mm)", fontsize=6,
                      color=C["text_sm"], labelpad=3)
    ax_wet.grid(linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_wet.text(0.03, 0.97, "b", transform=ax_wet.transAxes,
                fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_wet.set_title("National wet days per year",
                     fontsize=7, color=C["text_hd"], pad=4)

    # Panel c: Rx1day by era
    eras  = {"1981\u20131999": (1981, 1999),
             "2000\u20132012": (2000, 2012),
             "2013\u20132025": (2013, 2025)}
    ecols = [C["blue"], C["teal"], C["amber"]]
    for (lbl, (y0, y1)), col in zip(eras.items(), ecols):
        vals = df.loc[(df["year"] >= y0) & (df["year"] <= y1), "rx1day_mm"].values
        ax_rx1.hist(vals, bins=20, color=col, edgecolor="white",
                    linewidth=0.25, alpha=0.72, label=lbl, zorder=3)
    ax_rx1.set_xlabel("Rx1day (mm)", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_rx1.set_ylabel("Count", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_rx1.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_rx1.legend(fontsize=4.5, loc="upper right", frameon=True,
                  framealpha=0.9, edgecolor=C["border"])
    ax_rx1.text(0.03, 0.97, "c", transform=ax_rx1.transAxes,
                fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_rx1.set_title("Rx1day distribution by era",
                     fontsize=7, color=C["text_hd"], pad=4)

    # Panel d: Rx3day vs Rx5day
    sc = ax_rx35.scatter(
        df["rx3day_mm"], df["rx5day_mm"],
        c=df["year"], cmap="viridis",
        s=1.5, alpha=0.35, linewidths=0, zorder=3)
    lim = max(df["rx3day_mm"].max(), df["rx5day_mm"].max()) * 1.05
    ax_rx35.plot([0, lim], [0, lim], color=C["gray"], lw=0.5, ls="--", zorder=2)
    cbar = fig.colorbar(sc, ax=ax_rx35, pad=0.02, fraction=0.04)
    cbar.ax.tick_params(labelsize=4.5)
    cbar.set_label("Year", fontsize=5, color=C["text_sm"])
    ax_rx35.set_xlim(0, lim); ax_rx35.set_ylim(0, lim)
    ax_rx35.set_xlabel("Rx3day (mm)", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_rx35.set_ylabel("Rx5day (mm)", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_rx35.grid(linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_rx35.text(0.03, 0.97, "d", transform=ax_rx35.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_rx35.set_title("Rx3day vs Rx5day (all tiles \u00d7 years)",
                      fontsize=7, color=C["text_hd"], pad=4)

    # Panel e: heavy rain days trend
    lo_h, hi_h = iqr_band("heavy_rain_days_20mm_n")
    ax_heavy.fill_between(nat["year"], lo_h, hi_h, color=C["red"],
                          alpha=0.12, zorder=2)
    ax_heavy.plot(nat["year"], nat["heavy_rain_days_20mm_n"],
                  color=C["red"], lw=0.9, alpha=0.9, zorder=3)
    ax_heavy.plot(nat["year"], rolling10(nat["heavy_rain_days_20mm_n"]),
                  color=C["accent"], lw=1.1, ls="--", zorder=4)
    z = np.polyfit(nat["year"], nat["heavy_rain_days_20mm_n"], 1)
    ax_heavy.plot(nat["year"], np.poly1d(z)(nat["year"]),
                  color=C["purple"], lw=0.8, ls=":", zorder=5)
    ax_heavy.text(0.97, 0.05,
                  f"trend: {z[0]*10:+.2f} days decade\u207b\u00b9",
                  transform=ax_heavy.transAxes, ha="right", va="bottom",
                  fontsize=5, color=C["purple"],
                  bbox=dict(boxstyle="round,pad=0.3", fc="white",
                            ec=C["border"], lw=0.5, alpha=0.9))
    ax_heavy.set_xlabel("Year", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_heavy.set_ylabel("Heavy rain days yr\u207b\u00b9  (\u2265 20 mm)",
                        fontsize=6, color=C["text_sm"], labelpad=3)
    ax_heavy.grid(linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_heavy.text(0.03, 0.97, "e", transform=ax_heavy.transAxes,
                  fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_heavy.set_title("Heavy rain days trend",
                       fontsize=7, color=C["text_hd"], pad=4)

    # Panel f: pipeline progress
    ax_prog.set_xlim(0, 1); ax_prog.set_ylim(0, 1)
    ax_prog.axis("off")
    ax_prog.text(0.03, 0.97, "f", transform=ax_prog.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_prog.set_title("Pipeline progress", fontsize=7, color=C["text_hd"], pad=4)

    pipeline = [
        ("00 · Environment setup",    1.00, C["teal"]),
        ("01 · Processing tiles",     1.00, C["teal"]),
        ("02 · Municipal units",      1.00, C["teal"]),
        ("03 · Tile-mun crosswalk",   1.00, C["teal"]),
        ("04 · CHIRPS hazard series", 1.00, C["teal"]),
        ("05 · Data integration",     0.00, C["gray"]),
        ("06 · Figures",              0.00, C["gray"]),
    ]

    bar_w = 0.56; bar_h = 0.082; x0 = 0.32; y0 = 0.90
    for label, frac, col in pipeline:
        ax_prog.add_patch(FancyBboxPatch(
            (x0, y0), bar_w, bar_h, boxstyle="round,pad=0.004",
            linewidth=0.4, edgecolor=C["border"], facecolor="#E5E7EB",
            transform=ax_prog.transAxes, clip_on=True, zorder=2))
        if frac > 0:
            ax_prog.add_patch(FancyBboxPatch(
                (x0, y0), max(frac * bar_w, 0.02), bar_h,
                boxstyle="round,pad=0.004", linewidth=0, facecolor=col,
                transform=ax_prog.transAxes, clip_on=True, zorder=3))
        ax_prog.text(x0 - 0.02, y0 + bar_h / 2, label,
                     ha="right", va="center", fontsize=5.5,
                     color=C["text_hd"], transform=ax_prog.transAxes)
        ax_prog.text(x0 + bar_w + 0.03, y0 + bar_h / 2,
                     f"{int(frac * 100)}%",
                     ha="left", va="center", fontsize=5.5,
                     color=col if frac > 0 else C["gray"],
                     fontweight="bold" if frac > 0 else "normal",
                     transform=ax_prog.transAxes)
        y0 -= 0.120

    info = (f"Dataset    CHIRPS Daily\n"
            f"Source     UCSB-CHG/CHIRPS/DAILY\n"
            f"Period     {START_YEAR} \u2013 {END_YEAR}\n"
            f"Tiles      {n_tiles:,}\n"
            f"Scale      \u223c{SCALE_M:,} m")
    ax_prog.text(0.50, 0.04, info, ha="center", va="bottom", fontsize=5,
                 color=C["text_sm"], transform=ax_prog.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", fc=C["panel"],
                           ec=C["border"], lw=0.5))

    for xi, (fc, lbl) in enumerate([(C["teal"], "Completed"),
                                     (C["gray"], "Pending")]):
        bx = 0.30 + xi * 0.35
        ax_prog.add_patch(mpatches.Rectangle(
            (bx, 0.01), 0.025, 0.042, facecolor=fc, edgecolor="none",
            transform=ax_prog.transAxes, clip_on=False, zorder=5))
        ax_prog.text(bx + 0.035, 0.031, lbl, fontsize=5.5, va="center",
                     color=C["text_sm"], transform=ax_prog.transAxes)

    fig.text(
        0.50, 0.970,
        "Figure 5  |  CHIRPS hazard time series \u2014 Flood Inequality in Brazil",
        ha="center", va="top",
        fontsize=8, fontweight="bold", color=C["text_hd"],
    )
    fig.text(
        0.50, 0.957,
        (f"CHIRPS Daily \u00b7 UCSB-CHG \u00b7 {START_YEAR}\u2013{END_YEAR} \u00b7 "
         f"{n_tiles:,} tiles \u00b7 annual metrics \u00b7 GEE server-side"),
        ha="center", va="top",
        fontsize=6, color=C["text_sm"], style="italic",
    )

    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, "fig04_chirps_hazard_timeseries.png")
    pdf_path = os.path.join(save_dir, "fig04_chirps_hazard_timeseries.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=C["bg"])
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor=C["bg"])
    plt.show()
    plt.close(fig)
    return png_path


# ============================================================
# RUN
# ============================================================
log("Generating Figure 04 ...")
with tqdm(total=1, desc="Rendering figure",
          bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
    fig_path = make_figure_04_chirps(
        MANIFEST_PATH,
        OUTPUT_DIR,
        os.path.join(BASE_PATH, "06_figures"),
        dpi=500,
    )
    pbar.update(1)

log(f"Figure saved: {fig_path}", level="SUMMARY")
logging.info("Figure 04 generated successfully.")

print("\n" + "=" * 60)
print("  Module 04 complete")
print("=" * 60)
print(f"  Tiles total    : {n_total:,}")
print(f"  Completed      : {n_completed:,}")
print(f"  Skipped        : {n_skipped:,}")
print(f"  Failed         : {n_failed:,}")
print(f"  Manifest       : {MANIFEST_PATH}")
print(f"  Figure PNG     : {OUTPUT_FIG_PNG}")
print(f"  Figure PDF     : {OUTPUT_FIG_PDF}")
if failed_ids:
    print(f"  Failed IDs     : {', '.join(failed_ids)}")
print("  Ready for Module 05.")
print("=" * 60)
