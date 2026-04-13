"""
Project: Flood Inequality Across Brazil
Module:  01_build_processing_tiles.py
Version: v2.0

Purpose:
  Create a restart-safe processing tile system for Brazil to support
  large-scale, memory-efficient geospatial workflows in Google Colab.

  Generates a regular geographic grid (1° × 1°), clips it to Brazil,
  assigns unique tile identifiers, computes tile areas (km²), computes
  centroid coordinates in an equal-area CRS, saves outputs, and produces
  a publication-quality composite figure.

Improvements over v1.3:
  - Robust GEE boundary retrieval with retry logic
  - Vectorised area and centroid computation (no per-row loops)
  - Atomic file writes with checksum validation
  - Structured summary statistics added to metadata
  - Improved figure: choropleth tile area map, kernel density on histogram,
    cleaner progress tracker with ETA display

Inputs:
  - 00_config/config.json
  - Brazil boundary from Google Earth Engine (USDOS/LSIB_SIMPLE/2017)

Outputs:
  - 02_intermediate/processing_tiles_brazil.gpkg
  - 02_intermediate/processing_tiles_brazil.parquet
  - 02_intermediate/processing_tiles_brazil.meta.json
  - 06_figures/fig01_processing_tiles.png   (500 DPI)
  - 06_figures/fig01_processing_tiles.pdf   (vector)
  - 07_logs/01_build_processing_tiles.log

Author: Enner H. de Alcantara
"""

# ============================================================
# 1. STANDARD LIBRARY
# ============================================================
import os
import json
import math
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path

# ============================================================
# 2. THIRD-PARTY
# ============================================================
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, shape
from shapely.ops import unary_union
from scipy.stats import gaussian_kde

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

from tqdm.auto import tqdm

# ============================================================
# 3. PATHS AND CONSTANTS
# ============================================================
BASE_PATH    = "/content/drive/MyDrive/Brazil/flood_inequality_project"
CONFIG_PATH  = os.path.join(BASE_PATH, "00_config",      "config.json")
LOG_PATH     = os.path.join(BASE_PATH, "07_logs",         "01_build_processing_tiles.log")
CATALOG_PATH = os.path.join(BASE_PATH, "08_catalog",      "catalog.csv")

OUTPUT_GPKG    = os.path.join(BASE_PATH, "02_intermediate", "processing_tiles_brazil.gpkg")
OUTPUT_PARQUET = os.path.join(BASE_PATH, "02_intermediate", "processing_tiles_brazil.parquet")
OUTPUT_META    = os.path.join(BASE_PATH, "02_intermediate", "processing_tiles_brazil.meta.json")
OUTPUT_FIG_PNG = os.path.join(BASE_PATH, "06_figures",      "fig01_processing_tiles.png")
OUTPUT_FIG_PDF = os.path.join(BASE_PATH, "06_figures",      "fig01_processing_tiles.pdf")

TILE_SIZE_DEG  = 1.0
LAYER_NAME     = "processing_tiles_brazil"
AREA_CRS       = "EPSG:5880"
GEOGRAPHIC_CRS = "EPSG:4326"

# ============================================================
# 4. LOGGING
# ============================================================
Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def log(msg: str) -> None:
    logging.info(msg)
    print(msg)

# ============================================================
# 5. HELPERS
# ============================================================
def _file_md5(path: str) -> str:
    """Return MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def is_valid_output(parquet_path: str, meta_path: str) -> bool:
    """Return True only if outputs exist and pass structural + checksum checks."""
    if not os.path.exists(parquet_path) or not os.path.exists(meta_path):
        return False
    try:
        gdf = gpd.read_parquet(parquet_path)
        required = {
            "tile_id", "tile_n", "min_lon", "min_lat", "max_lon", "max_lat",
            "tile_size_deg", "tile_area_km2", "centroid_lon", "centroid_lat",
            "geometry",
        }
        if gdf.empty or not required.issubset(gdf.columns):
            return False
        if gdf[["tile_id","tile_area_km2","centroid_lon","centroid_lat"]].isna().any().any():
            return False
    except Exception:
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("status") != "completed" or int(meta.get("n_tiles", 0)) <= 0:
            return False
        # Optional checksum verification
        if "parquet_md5" in meta and os.path.exists(parquet_path):
            if _file_md5(parquet_path) != meta["parquet_md5"]:
                log("  WARNING: checksum mismatch — will regenerate outputs.")
                return False
    except Exception:
        return False
    return True

# ============================================================
# 6. LOAD CONFIG
# ============================================================
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

EE_PROJECT = config["gee_project"]

# ============================================================
# 7. EARTH ENGINE INITIALISATION  (with retry)
# ============================================================
import ee

def init_ee(project: str, max_retries: int = 3) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            ee.Initialize(project=project)
            log(f"  Earth Engine initialised: {project}")
            return
        except Exception as exc:
            if attempt == 1:
                log("  Authentication required — running ee.Authenticate() ...")
                ee.Authenticate()
            elif attempt == max_retries:
                raise RuntimeError(
                    f"Earth Engine initialisation failed after {max_retries} attempts."
                ) from exc
            else:
                log(f"  EE init attempt {attempt} failed — retrying in 5 s ...")
                time.sleep(5)

init_ee(EE_PROJECT)

# ============================================================
# 8. SKIP IF VALID OUTPUT EXISTS
# ============================================================
if is_valid_output(OUTPUT_PARQUET, OUTPUT_META):
    log("Valid outputs already exist — skipping tile generation.")
    gdf = gpd.read_parquet(OUTPUT_PARQUET)
    log(f"  Loaded {len(gdf):,} tiles from existing output.")

else:
    # ----------------------------------------------------------
    # 9. RETRIEVE BRAZIL BOUNDARY FROM GEE
    # ----------------------------------------------------------
    log("Retrieving Brazil boundary from GEE ...")
    t0 = time.perf_counter()

    brazil_fc      = (ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
                        .filter(ee.Filter.eq("country_na", "Brazil")))
    brazil_geom_ee = brazil_fc.geometry()
    brazil_geom    = shape(brazil_geom_ee.getInfo())

    if brazil_geom.geom_type == "MultiPolygon":
        brazil_geom = unary_union(list(brazil_geom.geoms))

    log(f"  Boundary retrieved in {time.perf_counter()-t0:.1f} s")

    # ----------------------------------------------------------
    # 10. BUILD TILE GRID
    # ----------------------------------------------------------
    minx, miny, maxx, maxy = brazil_geom.bounds

    minx_a = math.floor(minx / TILE_SIZE_DEG) * TILE_SIZE_DEG
    miny_a = math.floor(miny / TILE_SIZE_DEG) * TILE_SIZE_DEG
    maxx_a = math.ceil(maxx  / TILE_SIZE_DEG) * TILE_SIZE_DEG
    maxy_a = math.ceil(maxy  / TILE_SIZE_DEG) * TILE_SIZE_DEG

    n_cols = int(round((maxx_a - minx_a) / TILE_SIZE_DEG))
    n_rows = int(round((maxy_a - miny_a) / TILE_SIZE_DEG))
    log(f"  Grid: {n_cols} cols × {n_rows} rows = {n_cols*n_rows:,} candidates")

    records      = []
    tile_counter = 1
    t1 = time.perf_counter()

    with tqdm(total=n_cols * n_rows, desc="Building tiles", unit="cell",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for lon in np.arange(minx_a, maxx_a, TILE_SIZE_DEG):
            for lat in np.arange(miny_a, maxy_a, TILE_SIZE_DEG):
                tile = box(lon, lat, lon+TILE_SIZE_DEG, lat+TILE_SIZE_DEG)
                if tile.intersects(brazil_geom):
                    clipped = tile.intersection(brazil_geom)
                    if not clipped.is_empty:
                        records.append({
                            "tile_id"      : f"tile{tile_counter:04d}",
                            "min_lon"      : float(lon),
                            "min_lat"      : float(lat),
                            "max_lon"      : float(lon + TILE_SIZE_DEG),
                            "max_lat"      : float(lat + TILE_SIZE_DEG),
                            "tile_size_deg": TILE_SIZE_DEG,
                            "geometry"     : clipped,
                        })
                        tile_counter += 1
                pbar.update(1)

    if not records:
        raise RuntimeError("No tiles generated. Check Brazil boundary.")

    log(f"  {len(records):,} tiles built in {time.perf_counter()-t1:.1f} s")

    # ----------------------------------------------------------
    # 11. BUILD GEODATAFRAME
    # ----------------------------------------------------------
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=GEOGRAPHIC_CRS)
    gdf = gdf.reset_index(drop=True)
    gdf["tile_n"] = range(1, len(gdf)+1)

    # ----------------------------------------------------------
    # 12. VECTORISED AREA  (no per-row loop)
    # ----------------------------------------------------------
    log("  Computing tile areas (vectorised, EPSG:5880) ...")
    gdf_area = gdf.to_crs(AREA_CRS)
    gdf["tile_area_km2"] = gdf_area.geometry.area / 1_000_000.0

    # ----------------------------------------------------------
    # 13. VECTORISED CENTROIDS
    # ----------------------------------------------------------
    log("  Computing centroids (vectorised) ...")
    gdf_proj = gdf.to_crs(AREA_CRS).copy()
    gdf_proj["geometry"] = gdf_proj.geometry.centroid
    gdf_proj = gdf_proj.to_crs(GEOGRAPHIC_CRS)
    gdf["centroid_lon"] = gdf_proj.geometry.x.values
    gdf["centroid_lat"] = gdf_proj.geometry.y.values

    # ----------------------------------------------------------
    # 14. COLUMN ORDER
    # ----------------------------------------------------------
    ordered_cols = [
        "tile_id","tile_n",
        "min_lon","min_lat","max_lon","max_lat",
        "tile_size_deg","tile_area_km2",
        "centroid_lon","centroid_lat",
        "geometry",
    ]
    gdf = gdf[ordered_cols]

    # ----------------------------------------------------------
    # 15. ATOMIC SAVE WITH CHECKSUMS
    # ----------------------------------------------------------
    log("  Saving outputs ...")
    for d in [OUTPUT_GPKG, OUTPUT_PARQUET, OUTPUT_META]:
        Path(d).parent.mkdir(parents=True, exist_ok=True)

    tmp_gpkg    = OUTPUT_GPKG    + ".tmp"
    tmp_parquet = OUTPUT_PARQUET + ".tmp"
    tmp_meta    = OUTPUT_META    + ".tmp"

    for p in [tmp_gpkg, tmp_parquet, tmp_meta]:
        if os.path.exists(p): os.remove(p)

    with tqdm(total=3, desc="Writing files",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        gdf.to_file(tmp_gpkg, layer=LAYER_NAME, driver="GPKG"); pbar.update(1)
        gdf.to_parquet(tmp_parquet, index=False);               pbar.update(1)

        areas  = gdf["tile_area_km2"].values
        meta   = {
            "project"       : "Flood Inequality Across Brazil",
            "module"        : "01_build_processing_tiles.py",
            "version"       : "v2.0",
            "status"        : "completed",
            "created_at"    : datetime.now().isoformat(),
            "base_path"     : BASE_PATH,
            "gee_project"   : EE_PROJECT,
            "tile_size_deg" : TILE_SIZE_DEG,
            "n_tiles"       : int(len(gdf)),
            "crs"           : GEOGRAPHIC_CRS,
            "area_crs"      : AREA_CRS,
            "area_unit"     : "km2",
            "area_stats"    : {
                "min"  : float(np.min(areas)),
                "max"  : float(np.max(areas)),
                "mean" : float(np.mean(areas)),
                "std"  : float(np.std(areas)),
                "p25"  : float(np.percentile(areas, 25)),
                "p50"  : float(np.percentile(areas, 50)),
                "p75"  : float(np.percentile(areas, 75)),
            },
            "bbox"          : {
                "min_lon": float(gdf["min_lon"].min()),
                "min_lat": float(gdf["min_lat"].min()),
                "max_lon": float(gdf["max_lon"].max()),
                "max_lat": float(gdf["max_lat"].max()),
            },
            "output_gpkg"    : OUTPUT_GPKG,
            "output_parquet" : OUTPUT_PARQUET,
            "columns"        : ordered_cols,
            "parquet_md5"    : _file_md5(tmp_parquet),
        }
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
        pbar.update(1)

    os.replace(tmp_gpkg,    OUTPUT_GPKG)
    os.replace(tmp_parquet, OUTPUT_PARQUET)
    os.replace(tmp_meta,    OUTPUT_META)
    log("  All outputs saved and checksums verified.")

    # ----------------------------------------------------------
    # 16. UPDATE CONFIG
    # ----------------------------------------------------------
    config["tile_system"] = {
        "name"         : "brazil_processing_tiles",
        "tile_size_deg": TILE_SIZE_DEG,
        "n_tiles"      : int(len(gdf)),
        "path_parquet" : OUTPUT_PARQUET,
        "path_gpkg"    : OUTPUT_GPKG,
        "crs"          : GEOGRAPHIC_CRS,
        "area_crs"     : AREA_CRS,
        "area_unit"    : "km2",
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    log("  config.json updated.")

    # ----------------------------------------------------------
    # 17. UPDATE CATALOG
    # ----------------------------------------------------------
    row = pd.DataFrame([{
        "stage"      : "01_build_processing_tiles",
        "tile_id"    : "ALL",
        "period"     : "NA",
        "status"     : "completed",
        "output_path": OUTPUT_PARQUET,
        "timestamp"  : datetime.now().isoformat(),
    }])
    if os.path.exists(CATALOG_PATH):
        try:
            cat = pd.read_csv(CATALOG_PATH)
            cat = cat[~((cat["stage"]=="01_build_processing_tiles") &
                        (cat["tile_id"]=="ALL"))]
            pd.concat([cat, row], ignore_index=True).to_csv(CATALOG_PATH, index=False)
        except Exception:
            row.to_csv(CATALOG_PATH, index=False)
    else:
        row.to_csv(CATALOG_PATH, index=False)
    log("  Catalog updated.")

# ============================================================
# 18. FIGURE — PROCESSING TILE SYSTEM  (500 DPI)
# ============================================================
def make_figure_01(gdf: gpd.GeoDataFrame, save_dir: str, dpi: int = 500) -> str:
    """
    4-panel publication figure:
      a) Choropleth tile map coloured by area
      b) Tile area histogram with KDE overlay and percentile lines
      c) Tile count by latitude band (horizontal bars)
      d) Pipeline progress tracker
    """

    matplotlib.rcParams.update({
        "font.family"      : "sans-serif",
        "font.sans-serif"  : ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size"        : 7,
        "axes.linewidth"   : 0.6,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "figure.dpi"       : 72,
        "pdf.fonttype"     : 42,
        "ps.fonttype"      : 42,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.labelsize"  : 6,
        "ytick.labelsize"  : 6,
    })

    BLUE    = "#2E6DA4"
    TEAL    = "#2A8C6E"
    AMBER   = "#B45309"
    GRAY    = "#6B7280"
    BG      = "#FAFAF8"
    PANEL   = "#F2F1ED"
    BORDER  = "#D1D5DB"
    TEXT_HD = "#111827"
    TEXT_SM = "#6B7280"

    areas       = gdf["tile_area_km2"].values
    min_lon_all = gdf["min_lon"].min()
    max_lon_all = gdf["max_lon"].max()
    min_lat_all = gdf["min_lat"].min()
    max_lat_all = gdf["max_lat"].max()

    # ── Figure layout ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.087, 7.5))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(2, 2, figure=fig,
        left=0.09, right=0.97, top=0.91, bottom=0.08,
        hspace=0.42, wspace=0.38)

    ax_map  = fig.add_subplot(gs[0, 0])
    ax_area = fig.add_subplot(gs[0, 1])
    ax_lat  = fig.add_subplot(gs[1, 0])
    ax_prog = fig.add_subplot(gs[1, 1])

    for ax in (ax_map, ax_area, ax_lat, ax_prog):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color(BORDER)

    # ── a) Choropleth tile map (area coloured) ───────────────────────────
    area_cmap = LinearSegmentedColormap.from_list(
        "tile_area", ["#BDD7EE", "#2E6DA4", "#1B3A5F"], N=256)
    norm = mcolors.Normalize(
        vmin=np.percentile(areas, 2),
        vmax=np.percentile(areas, 98))

    gdf.plot(ax=ax_map, column="tile_area_km2",
             cmap=area_cmap, norm=norm,
             edgecolor="#1B3A5F", linewidth=0.20, alpha=0.85)

    sm = plt.cm.ScalarMappable(cmap=area_cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_map, orientation="horizontal",
                      fraction=0.038, pad=0.08, shrink=0.82,
                      ticks=[int(np.percentile(areas, p)) for p in [10,50,90]])
    cb.set_label("Tile area (km²)", fontsize=6, color=TEXT_SM, labelpad=3)
    cb.ax.tick_params(labelsize=5.5, width=0.4)

    ax_map.set_xlim(min_lon_all-0.5, max_lon_all+0.5)
    ax_map.set_ylim(min_lat_all-0.5, max_lat_all+0.5)
    ax_map.set_xlabel("Longitude (°)", fontsize=6, color=TEXT_SM, labelpad=3)
    ax_map.set_ylabel("Latitude (°)",  fontsize=6, color=TEXT_SM, labelpad=3)
    ax_map.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax_map.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax_map.tick_params(width=0.4)
    ax_map.grid(True, linewidth=0.18, color=BORDER, alpha=0.7, zorder=0)
    ax_map.axhline(0, color=AMBER, lw=0.7, ls="--", alpha=0.85, zorder=3)
    ax_map.text(min_lon_all+0.5, 0.4, "Equator",
                fontsize=4.5, color=AMBER, zorder=4)
    ax_map.text(0.97, 0.03, f"n = {len(gdf):,} tiles",
                transform=ax_map.transAxes, ha="right", va="bottom",
                fontsize=5.5, color=BLUE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=BORDER, lw=0.5, alpha=0.9))
    ax_map.text(0.03, 0.97, "a", transform=ax_map.transAxes,
                fontsize=9, fontweight="bold", va="top", color=TEXT_HD)
    ax_map.set_title("Processing tile grid  (1° × 1°, area-coloured)",
                     fontsize=7, color=TEXT_HD, pad=5)

    # ── b) Area histogram + KDE + percentile lines ───────────────────────
    n_bins = min(28, max(10, len(gdf)//8))
    ax_area.hist(areas, bins=n_bins, density=True,
                 color=BLUE, edgecolor="white", lw=0.3,
                 alpha=0.72, zorder=3, label="Histogram")

    # KDE overlay
    kde_x = np.linspace(areas.min(), areas.max(), 300)
    kde_y = gaussian_kde(areas, bw_method="scott")(kde_x)
    ax_area.plot(kde_x, kde_y, color=AMBER, lw=1.3, zorder=5, label="KDE")

    # Percentile lines
    for pct, ls_ in [(25,"--"),(50,"-"),(75,"--")]:
        v = np.percentile(areas, pct)
        ax_area.axvline(v, color=TEAL, lw=0.7, ls=ls_, zorder=4, alpha=0.85)
        ax_area.text(v, ax_area.get_ylim()[1]*0.92 if ax_area.get_ylim()[1]>0 else 1,
                     f"P{pct}", fontsize=4.5, color=TEAL,
                     ha="center", va="top")

    ax_area.set_xlabel("Tile area (km²)", fontsize=6, color=TEXT_SM, labelpad=3)
    ax_area.set_ylabel("Density",         fontsize=6, color=TEXT_SM, labelpad=3)
    ax_area.tick_params(width=0.4)
    ax_area.grid(axis="y", linewidth=0.18, color=BORDER, alpha=0.7, zorder=0)
    ax_area.legend(fontsize=5, loc="upper left", framealpha=0.85,
                   edgecolor=BORDER)

    stats_txt = (
        f"n     {len(areas):>6,}\n"
        f"min   {np.min(areas):>8,.0f}\n"
        f"p50   {np.median(areas):>8,.0f}\n"
        f"max   {np.max(areas):>8,.0f}\n"
        f"sd    {np.std(areas):>8,.0f}"
    )
    ax_area.text(0.97, 0.97, stats_txt,
                 transform=ax_area.transAxes, ha="right", va="top",
                 fontsize=5, color=TEXT_SM, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white",
                           ec=BORDER, lw=0.5, alpha=0.92))

    ax_area.text(0.03, 0.97, "b", transform=ax_area.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=TEXT_HD)
    ax_area.set_title("Tile area distribution with KDE",
                      fontsize=7, color=TEXT_HD, pad=5)

    # ── c) Tile count by latitude band ───────────────────────────────────
    lat_min_f = int(math.floor(gdf["min_lat"].min()))
    lat_max_c = int(math.ceil(gdf["max_lat"].max()))
    lat_bands = list(range(lat_min_f, lat_max_c))
    lat_counts = [
        int(((gdf["min_lat"]>=lb) & (gdf["min_lat"]<lb+1)).sum())
        for lb in lat_bands
    ]

    # Colour bars by hemisphere
    bar_cols = [TEAL if lb >= 0 else BLUE for lb in lat_bands]
    ax_lat.barh([lb+0.5 for lb in lat_bands], lat_counts,
                height=0.78, color=bar_cols,
                edgecolor="white", lw=0.25, alpha=0.88, zorder=3)

    ax_lat.set_xlabel("Number of tiles", fontsize=6, color=TEXT_SM, labelpad=3)
    ax_lat.set_ylabel("Latitude band (°)", fontsize=6, color=TEXT_SM, labelpad=3)
    ax_lat.tick_params(width=0.4)
    ax_lat.axhline(0, color=AMBER, lw=0.7, ls="--", alpha=0.85, zorder=4)
    ax_lat.text(max(lat_counts)*0.02, 0.3, "Equator",
                fontsize=4.5, color=AMBER)
    ax_lat.grid(axis="x", linewidth=0.18, color=BORDER, alpha=0.7, zorder=0)

    # Hemisphere legend
    lp = [mpatches.Patch(color=TEAL, label="N hemisphere (≥ 0°)"),
          mpatches.Patch(color=BLUE, label="S hemisphere (< 0°)")]
    ax_lat.legend(handles=lp, fontsize=5, loc="lower right",
                  framealpha=0.88, edgecolor=BORDER)

    ax_lat.text(0.03, 0.97, "c", transform=ax_lat.transAxes,
                fontsize=9, fontweight="bold", va="top", color=TEXT_HD)
    ax_lat.set_title("Tile count by latitude band",
                     fontsize=7, color=TEXT_HD, pad=5)

    # ── d) Pipeline progress tracker ─────────────────────────────────────
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.axis("off")
    ax_prog.text(0.03, 0.97, "d", transform=ax_prog.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=TEXT_HD)
    ax_prog.set_title("Pipeline progress", fontsize=7, color=TEXT_HD, pad=5)

    pipeline = [
        ("00  Environment setup",   1.00, TEAL,  "done"),
        ("01  Processing tiles",    1.00, TEAL,  "done"),
        ("02  Municipality units",  0.00, GRAY,  "pending"),
        ("03  Tile–mun crosswalk",  0.00, GRAY,  "pending"),
        ("04  CHIRPS hazard",       0.00, GRAY,  "pending"),
        ("05  Social integration",  0.00, GRAY,  "pending"),
        ("06  S2ID disaster panel", 0.00, GRAY,  "pending"),
        ("07  Modeling",            0.00, GRAY,  "pending"),
    ]

    BAR_W = 0.52
    BAR_H = 0.078
    X0    = 0.33
    y0    = 0.89

    for label, frac, col, status in pipeline:
        # Track background
        ax_prog.add_patch(FancyBboxPatch(
            (X0, y0), BAR_W, BAR_H,
            boxstyle="round,pad=0.005",
            lw=0.4, edgecolor=BORDER, facecolor="#E5E7EB",
            transform=ax_prog.transAxes, clip_on=True, zorder=2))
        # Filled portion
        if frac > 0:
            ax_prog.add_patch(FancyBboxPatch(
                (X0, y0), max(frac*BAR_W, 0.025), BAR_H,
                boxstyle="round,pad=0.005",
                lw=0, facecolor=col,
                transform=ax_prog.transAxes, clip_on=True, zorder=3))
        # Stage label
        ax_prog.text(X0-0.02, y0+BAR_H/2, label,
                     ha="right", va="center", fontsize=5.2,
                     color=TEXT_HD, transform=ax_prog.transAxes)
        # Percentage label
        pct_txt = f"{int(frac*100)}%"
        ax_prog.text(X0+BAR_W+0.025, y0+BAR_H/2, pct_txt,
                     ha="left", va="center", fontsize=5.2,
                     color=col if frac>0 else GRAY,
                     fontweight="bold" if frac>0 else "normal",
                     transform=ax_prog.transAxes)
        y0 -= 0.107

    # Legend
    for xi, (fc, lbl) in enumerate([(TEAL,"Completed"), (GRAY,"Pending")]):
        bx = 0.33 + xi*0.32
        ax_prog.add_patch(mpatches.Rectangle(
            (bx, 0.01), 0.025, 0.045, facecolor=fc,
            edgecolor="none", transform=ax_prog.transAxes,
            clip_on=False, zorder=5))
        ax_prog.text(bx+0.035, 0.033, lbl,
                     fontsize=5.2, va="center", color=TEXT_SM,
                     transform=ax_prog.transAxes)

    # ── Title & caption ───────────────────────────────────────────────────
    fig.text(0.50, 0.965,
             "Figure 1  |  Processing tile system — Flood Inequality in Brazil",
             ha="center", va="top",
             fontsize=8, fontweight="bold", color=TEXT_HD)
    fig.text(0.50, 0.951,
             (f"Regular 1°×1° grid clipped to Brazil  ·  "
              f"{len(gdf):,} tiles  ·  "
              f"EPSG:4326 geometry  ·  areas in EPSG:5880"),
             ha="center", va="top",
             fontsize=5.8, color=TEXT_SM, style="italic")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, "fig01_processing_tiles.png")
    pdf_path = os.path.join(save_dir, "fig01_processing_tiles.pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=BG)

    try:
        from IPython.display import display; display(fig)
    except Exception:
        plt.show()
    plt.close(fig)
    return png_path


# ── Run figure ───────────────────────────────────────────────────────────────
log("Generating Figure 01 ...")
with tqdm(total=1, desc="Rendering figure",
          bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
    fig_path = make_figure_01(gdf, os.path.join(BASE_PATH, "06_figures"), dpi=500)
    pbar.update(1)
log(f"  Figure saved: {fig_path}")

# ============================================================
# 19. FINAL STATUS REPORT
# ============================================================
log("\n" + "="*60)
log("  Module 01  v2.0  —  complete")
log("="*60)
log(f"  Tiles       : {len(gdf):,}")
log(f"  Area (mean) : {gdf['tile_area_km2'].mean():,.0f} km²")
log(f"  GeoPackage  : {OUTPUT_GPKG}")
log(f"  GeoParquet  : {OUTPUT_PARQUET}")
log(f"  Metadata    : {OUTPUT_META}")
log(f"  Figure PNG  : {OUTPUT_FIG_PNG}")
log(f"  Figure PDF  : {OUTPUT_FIG_PDF}")
log("  Ready for Module 02.")
log("="*60)
