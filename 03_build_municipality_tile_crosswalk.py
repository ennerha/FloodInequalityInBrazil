"""
Project: Flood Inequality Across Brazil
Module:  03_build_municipality_tile_crosswalk.py

Purpose:
  Build a robust spatial crosswalk between the Brazil processing tile system
  (Module 01) and the municipal analysis units (Module 02).

  Identifies all valid tile-municipality intersections and computes:
    - intersection area in km²
    - fraction of each municipality covered by each tile
    - fraction of each tile occupied by each municipality

Critical methodological note:
  Tile geometries are reconstructed as full regular grid cells from bounding
  coordinates (min_lon, min_lat, max_lon, max_lat), NOT from the Brazil-clipped
  geometries in Module 01. This is required so that mun_fraction_covered sums
  correctly to 1.0. A post-hoc normalization (v1.4) guarantees closure for
  coastal and border municipalities.

Inputs:
  - 02_intermediate/processing_tiles_brazil.parquet
  - 02_intermediate/analysis_units_municipal_brazil.parquet

Outputs:
  - 03_features/municipality_tile_crosswalk.parquet
  - 03_features/municipality_tile_crosswalk.gpkg
  - 03_features/municipality_tile_crosswalk.meta.json
  - 03_features/municipality_tile_crosswalk_failed_pairs.csv
  - 06_figures/fig03_municipality_tile_crosswalk.png   (500 DPI)
  - 06_figures/fig03_municipality_tile_crosswalk.pdf   (vector)
  - 07_logs/03_build_municipality_tile_crosswalk.log

Reproducibility:
  - Idempotent execution  - safe to re-run without side effects
  - Metadata and logging enabled

Author:  Enner H. de Alcantara
Version: v1.4
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
import shapely
from shapely.geometry import box
from shapely.errors import GEOSException

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib import gridspec

from tqdm.auto import tqdm

# ============================================================
# 3. PATHS AND CONSTANTS
# ============================================================
BASE_PATH    = "/content/drive/MyDrive/Brazil/flood_inequality_project"
CONFIG_PATH  = os.path.join(BASE_PATH, "00_config",  "config.json")
LOG_PATH     = os.path.join(BASE_PATH, "07_logs",    "03_build_municipality_tile_crosswalk.log")
CATALOG_PATH = os.path.join(BASE_PATH, "08_catalog", "catalog.csv")

TILES_PATH = os.path.join(BASE_PATH, "02_intermediate", "processing_tiles_brazil.parquet")
MUNI_PATH  = os.path.join(BASE_PATH, "02_intermediate", "analysis_units_municipal_brazil.parquet")

OUTPUT_DIR     = os.path.join(BASE_PATH, "03_features")
OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "municipality_tile_crosswalk.parquet")
OUTPUT_GPKG    = os.path.join(OUTPUT_DIR, "municipality_tile_crosswalk.gpkg")
OUTPUT_META    = os.path.join(OUTPUT_DIR, "municipality_tile_crosswalk.meta.json")
OUTPUT_FAILED  = os.path.join(OUTPUT_DIR, "municipality_tile_crosswalk_failed_pairs.csv")
OUTPUT_FIG_PNG = os.path.join(BASE_PATH, "06_figures", "fig03_municipality_tile_crosswalk.png")
OUTPUT_FIG_PDF = os.path.join(BASE_PATH, "06_figures", "fig03_municipality_tile_crosswalk.pdf")

LAYER_NAME     = "municipality_tile_crosswalk"
GEOGRAPHIC_CRS = "EPSG:4326"
AREA_CRS       = "EPSG:5880"

MIN_INTERSECTION_AREA_KM2     = 1e-8
MAX_ACCEPTABLE_FRACTION_ERROR = 0.01

# ============================================================
# 4. DIRECTORY SETUP
# ============================================================
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================
# 5. LOGGING
# ============================================================
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
# 6. OUTPUT VALIDATION
# ============================================================
def is_valid_output(parquet_path: str, meta_path: str) -> bool:
    if not os.path.exists(parquet_path) or not os.path.exists(meta_path):
        return False
    try:
        gdf = gpd.read_parquet(parquet_path)
        required = {
            "tile_id", "mun_code", "intersection_area_km2",
            "mun_area_km2_geom", "tile_area_km2_geom",
            "mun_fraction_covered", "tile_fraction_occupied", "geometry",
        }
        if gdf.empty or not required.issubset(gdf.columns):
            return False
        if gdf[["tile_id", "mun_code"]].isna().any().any():
            return False
        if (gdf["intersection_area_km2"] <= 0).any():
            return False
    except Exception:
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("status") != "completed":
            return False
        if int(meta.get("n_intersections", 0)) <= 0:
            return False
        if float(meta.get("max_municipality_fraction_closure_error", 999)) > MAX_ACCEPTABLE_FRACTION_ERROR:
            return False
    except Exception:
        return False
    return True

# ============================================================
# 7. GEOMETRY HELPERS
# ============================================================
def make_geometry_valid(geom):
    if geom is None or geom.is_empty:
        return geom
    try:
        if geom.is_valid:
            return geom
    except Exception:
        pass
    for fn in [lambda g: shapely.make_valid(g), lambda g: g.buffer(0)]:
        try:
            r = fn(geom)
            if r is not None and not r.is_empty:
                return r
        except Exception:
            pass
    return geom


def safe_make_valid_gdf(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    n_invalid = (~gdf.geometry.is_valid).sum()
    log(f"{label}: invalid geometries = {n_invalid}")
    if n_invalid > 0:
        gdf["geometry"] = gdf.geometry.apply(make_geometry_valid)
        log(f"{label}: invalid after repair = {(~gdf.geometry.is_valid).sum()}")
    return gdf


def safe_intersection(geom_a, geom_b):
    strategies = [
        lambda a, b: a.intersection(b),
        lambda a, b: make_geometry_valid(a).intersection(make_geometry_valid(b)),
        lambda a, b: (shapely.set_precision(make_geometry_valid(a), 0.01)
                      .intersection(shapely.set_precision(make_geometry_valid(b), 0.01))),
        lambda a, b: (shapely.set_precision(make_geometry_valid(a), 0.1).buffer(0)
                      .intersection(shapely.set_precision(make_geometry_valid(b), 0.1).buffer(0))),
    ]
    for strategy in strategies:
        try:
            return strategy(geom_a, geom_b)
        except (GEOSException, Exception):
            continue
    return None

# ============================================================
# 8. LOAD CONFIGURATION
# ============================================================
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

# ============================================================
# 9. SKIP IF VALID OUTPUT EXISTS
# ============================================================
if is_valid_output(OUTPUT_PARQUET, OUTPUT_META):
    log("Valid crosswalk outputs already exist - skipping.")
    crosswalk = gpd.read_parquet(OUTPUT_PARQUET)
    log(f"Loaded {len(crosswalk):,} intersections from existing output.")

else:
    for path, label in [(TILES_PATH, "Tile input"), (MUNI_PATH, "Municipality input")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # ----------------------------------------------------------
    # 10. LOAD INPUT DATA
    # ----------------------------------------------------------
    log("Loading input data ...")
    with tqdm(total=2, desc="Reading inputs",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        tiles_raw = gpd.read_parquet(TILES_PATH); pbar.update(1)
        muni      = gpd.read_parquet(MUNI_PATH);  pbar.update(1)

    if tiles_raw.empty: raise RuntimeError("Tile dataset is empty.")
    if muni.empty:      raise RuntimeError("Municipality dataset is empty.")
    log(f"Tiles: {len(tiles_raw):,}  |  Municipalities: {len(muni):,}")

    # ----------------------------------------------------------
    # 11. COLUMN SELECTION
    # ----------------------------------------------------------
    tile_req = ["tile_id", "tile_n", "min_lon", "min_lat", "max_lon", "max_lat"]
    muni_req = ["mun_code", "mun_name", "uf_code", "uf_sigla", "geometry"]

    for col in tile_req:
        if col not in tiles_raw.columns:
            raise RuntimeError(f"Missing tile column: {col}")
    for col in muni_req:
        if col not in muni.columns:
            raise RuntimeError(f"Missing municipality column: {col}")

    tiles_raw = tiles_raw[tile_req].copy()
    muni      = muni[muni_req].copy()

    # ----------------------------------------------------------
    # 12. REBUILD FULL TILE GEOMETRIES FROM BOUNDS
    # ----------------------------------------------------------
    log("Rebuilding full tile geometries from bounding coordinates ...")
    with tqdm(total=1, desc="Building tile grid",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        tiles_raw["geometry"] = tiles_raw.apply(
            lambda r: box(r["min_lon"], r["min_lat"], r["max_lon"], r["max_lat"]),
            axis=1,
        )
        tiles = gpd.GeoDataFrame(tiles_raw, geometry="geometry", crs=GEOGRAPHIC_CRS)
        pbar.update(1)

    # ----------------------------------------------------------
    # 13. STANDARDIZE CRS
    # ----------------------------------------------------------
    if muni.crs is None:
        raise RuntimeError("Municipality CRS is missing.")

    with tqdm(total=2, desc="Standardizing CRS",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        muni  = muni.to_crs(GEOGRAPHIC_CRS);  pbar.update(1)
        tiles = tiles.to_crs(GEOGRAPHIC_CRS); pbar.update(1)

    # ----------------------------------------------------------
    # 14. GEOMETRY REPAIR (geographic)
    # ----------------------------------------------------------
    log("Repairing geometries ...")
    with tqdm(total=2, desc="Geometry repair (geo)",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        tiles = safe_make_valid_gdf(tiles, "Tiles");         pbar.update(1)
        muni  = safe_make_valid_gdf(muni,  "Municipalities"); pbar.update(1)

    # ----------------------------------------------------------
    # 15. PROJECT TO EQUAL-AREA CRS
    # ----------------------------------------------------------
    log(f"Projecting to {AREA_CRS} ...")
    with tqdm(total=2, desc="Projecting to EPSG:5880",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        tiles_area = tiles.to_crs(AREA_CRS); pbar.update(1)
        muni_area  = muni.to_crs(AREA_CRS);  pbar.update(1)

    # ----------------------------------------------------------
    # 16. GEOMETRY REPAIR (projected) + INTERNAL AREAS
    # ----------------------------------------------------------
    with tqdm(total=2, desc="Geometry repair (proj)",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        tiles_area = safe_make_valid_gdf(tiles_area, "Tiles projected");  pbar.update(1)
        muni_area  = safe_make_valid_gdf(muni_area,  "Muni projected");   pbar.update(1)

    log("Computing internal areas ...")
    tiles_area = tiles_area.copy()
    muni_area  = muni_area.copy()
    tiles_area["tile_area_km2_geom"] = tiles_area.geometry.area / 1_000_000.0
    muni_area["mun_area_km2_geom"]   = muni_area.geometry.area  / 1_000_000.0

    if (tiles_area["tile_area_km2_geom"] <= 0).any():
        raise RuntimeError("One or more tiles have non-positive area.")
    if (muni_area["mun_area_km2_geom"] <= 0).any():
        raise RuntimeError("One or more municipalities have non-positive area.")

    # ----------------------------------------------------------
    # 17. SPATIAL JOIN — CANDIDATE PAIRS
    # ----------------------------------------------------------
    log("Spatial join for candidate tile-municipality pairs ...")
    with tqdm(total=1, desc="Spatial join",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        candidates = gpd.sjoin(
            muni_area, tiles_area, how="inner", predicate="intersects"
        ).reset_index(drop=True)
        pbar.update(1)

    if candidates.empty:
        raise RuntimeError("Spatial join produced no candidates.")
    log(f"Candidate pairs: {len(candidates):,}")

    # ----------------------------------------------------------
    # 18. LOOKUP TABLES
    # ----------------------------------------------------------
    muni_map = muni_area.set_index("mun_code")[
        ["geometry", "mun_area_km2_geom", "mun_name", "uf_code", "uf_sigla"]]
    tile_map = tiles_area.set_index("tile_id")[
        ["geometry", "tile_area_km2_geom", "tile_n"]]
    candidates = candidates[["mun_code", "tile_id"]].drop_duplicates().reset_index(drop=True)

    # ----------------------------------------------------------
    # 19. EXACT INTERSECTION LOOP
    # ----------------------------------------------------------
    log(f"Computing exact intersections for {len(candidates):,} candidate pairs ...")
    records      = []
    failed_pairs = []
    total        = len(candidates)

    with tqdm(total=total, desc="Computing intersections",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for idx, row in candidates.iterrows():
            mun_code  = row["mun_code"]
            tile_id   = row["tile_id"]
            mun_row   = muni_map.loc[mun_code]
            tile_row  = tile_map.loc[tile_id]
            mun_geom  = mun_row["geometry"]
            tile_geom = tile_row["geometry"]

            try:
                if not mun_geom.intersects(tile_geom):
                    pbar.update(1); continue
            except Exception as e:
                failed_pairs.append({"mun_code": mun_code, "tile_id": tile_id,
                                     "step": "intersects", "error": str(e)})
                pbar.update(1); continue

            inter_geom = safe_intersection(mun_geom, tile_geom)
            if inter_geom is None:
                failed_pairs.append({"mun_code": mun_code, "tile_id": tile_id,
                                     "step": "intersection", "error": "returned None"})
                pbar.update(1); continue

            try:
                if inter_geom.is_empty:
                    pbar.update(1); continue
            except Exception as e:
                failed_pairs.append({"mun_code": mun_code, "tile_id": tile_id,
                                     "step": "is_empty", "error": str(e)})
                pbar.update(1); continue

            try:
                inter_area_km2 = inter_geom.area / 1_000_000.0
            except Exception as e:
                failed_pairs.append({"mun_code": mun_code, "tile_id": tile_id,
                                     "step": "area", "error": str(e)})
                pbar.update(1); continue

            if inter_area_km2 <= MIN_INTERSECTION_AREA_KM2:
                pbar.update(1); continue

            mun_area_km2  = float(mun_row["mun_area_km2_geom"])
            tile_area_km2 = float(tile_row["tile_area_km2_geom"])

            records.append({
                "tile_id"               : tile_id,
                "tile_n"                : int(tile_row["tile_n"]),
                "mun_code"              : mun_code,
                "mun_name"              : mun_row["mun_name"],
                "uf_code"               : mun_row["uf_code"],
                "uf_sigla"              : mun_row["uf_sigla"],
                "intersection_area_km2" : inter_area_km2,
                "mun_area_km2_geom"     : mun_area_km2,
                "tile_area_km2_geom"    : tile_area_km2,
                "mun_fraction_covered"  : inter_area_km2 / mun_area_km2  if mun_area_km2  > 0 else None,
                "tile_fraction_occupied": inter_area_km2 / tile_area_km2 if tile_area_km2 > 0 else None,
                "geometry"              : inter_geom,
            })
            pbar.update(1)

    if not records:
        raise RuntimeError("No valid intersections produced.")
    log(f"Valid intersections: {len(records):,}  |  Failed pairs: {len(failed_pairs):,}")

    # ----------------------------------------------------------
    # 20. BUILD GEODATAFRAME
    # ----------------------------------------------------------
    desired_cols = [
        "tile_id", "tile_n", "mun_code", "mun_name", "uf_code", "uf_sigla",
        "intersection_area_km2", "mun_area_km2_geom", "tile_area_km2_geom",
        "mun_fraction_covered", "tile_fraction_occupied", "geometry",
    ]

    crosswalk = gpd.GeoDataFrame(records, geometry="geometry", crs=AREA_CRS)
    crosswalk = crosswalk.to_crs(GEOGRAPHIC_CRS)
    crosswalk = crosswalk.sort_values(["tile_n", "uf_sigla", "mun_name"]).reset_index(drop=True)
    crosswalk = crosswalk[desired_cols]

    # ----------------------------------------------------------
    # 21. NORMALIZE mun_fraction_covered (guarantee closure)
    # ----------------------------------------------------------
    log("Normalizing mun_fraction_covered to enforce per-municipality closure ...")
    with tqdm(total=1, desc="Normalizing fractions",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        mun_sum = crosswalk.groupby("mun_code")["intersection_area_km2"].transform("sum")
        crosswalk = crosswalk.copy()
        crosswalk["mun_fraction_covered"] = crosswalk["intersection_area_km2"] / mun_sum
        raw_sums = (crosswalk.groupby("mun_code")["intersection_area_km2"].sum()
                    / crosswalk.groupby("mun_code")["mun_area_km2_geom"].first())
        n_affected = int((raw_sums - 1.0).abs().gt(0.001).sum())
        log(f"Municipalities with raw closure error > 0.1% (before norm): {n_affected}")
        pbar.update(1)

    # ----------------------------------------------------------
    # 22. QUALITY CHECKS
    # ----------------------------------------------------------
    log("Running quality checks ...")
    with tqdm(total=4, desc="Quality checks",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        if crosswalk["intersection_area_km2"].isna().any():
            raise RuntimeError("Missing intersection areas.")
        pbar.update(1)
        if (crosswalk["intersection_area_km2"] <= 0).any():
            raise RuntimeError("Non-positive intersection areas.")
        pbar.update(1)
        if (crosswalk["mun_fraction_covered"] <= 0).any():
            raise RuntimeError("Non-positive municipality coverage fractions.")
        pbar.update(1)
        if (crosswalk["tile_fraction_occupied"] <= 0).any():
            raise RuntimeError("Non-positive tile occupancy fractions.")
        pbar.update(1)

    frac_check  = (crosswalk.groupby("mun_code")["mun_fraction_covered"]
                   .sum().reset_index(name="frac_sum"))
    max_frac_err = float((frac_check["frac_sum"] - 1.0).abs().max())
    log(f"Max mun_fraction closure error after normalization: {max_frac_err:.2e}")

    if max_frac_err > MAX_ACCEPTABLE_FRACTION_ERROR:
        raise RuntimeError(
            f"Closure error too high after normalization: {max_frac_err:.2e}")

    # ----------------------------------------------------------
    # 23. SAVE FAILED PAIRS
    # ----------------------------------------------------------
    pd.DataFrame(failed_pairs).to_csv(OUTPUT_FAILED, index=False)
    log(f"Failed pairs saved: {OUTPUT_FAILED}")

    # ----------------------------------------------------------
    # 24. SAVE OUTPUTS  (atomic via temp files)
    # ----------------------------------------------------------
    log("Saving outputs ...")
    tmp_parquet = OUTPUT_PARQUET + ".tmp"
    tmp_gpkg    = OUTPUT_GPKG    + ".tmp"
    tmp_meta    = OUTPUT_META    + ".tmp"

    for p in [tmp_parquet, tmp_gpkg, tmp_meta]:
        if os.path.exists(p): os.remove(p)

    with tqdm(total=3, desc="Writing files",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        crosswalk.to_parquet(tmp_parquet, index=False);                pbar.update(1)
        crosswalk.to_file(tmp_gpkg, layer=LAYER_NAME, driver="GPKG"); pbar.update(1)
        meta = {
            "project"                               : "Flood Inequality Across Brazil",
            "module"                                : "03_build_municipality_tile_crosswalk.py",
            "version"                               : "v1.4",
            "status"                                : "completed",
            "created_at"                            : datetime.now().isoformat(),
            "base_path"                             : BASE_PATH,
            "tile_input"                            : TILES_PATH,
            "municipal_input"                       : MUNI_PATH,
            "tile_geometry_mode"                    : "full_grid_cell_from_bounds",
            "mun_fraction_covered_normalization"    : "normalized_by_intersection_sum",
            "n_tiles"                               : int(tiles["tile_id"].nunique()),
            "n_municipalities"                      : int(muni["mun_code"].nunique()),
            "n_intersections"                       : int(len(crosswalk)),
            "n_failed_pairs"                        : int(len(failed_pairs)),
            "crs_output"                            : GEOGRAPHIC_CRS,
            "area_crs"                              : AREA_CRS,
            "area_unit"                             : "km2",
            "max_municipality_fraction_closure_error": max_frac_err,
            "failed_pairs_csv"                      : OUTPUT_FAILED,
            "output_parquet"                        : OUTPUT_PARQUET,
            "output_gpkg"                           : OUTPUT_GPKG,
            "columns"                               : desired_cols,
        }
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=4)
        pbar.update(1)

    os.replace(tmp_parquet, OUTPUT_PARQUET)
    os.replace(tmp_gpkg,    OUTPUT_GPKG)
    os.replace(tmp_meta,    OUTPUT_META)
    log("All outputs saved.")

    # ----------------------------------------------------------
    # 25. UPDATE CONFIG
    # ----------------------------------------------------------
    config["municipality_tile_crosswalk"] = {
        "name"                               : "brazil_municipality_tile_crosswalk",
        "path_parquet"                       : OUTPUT_PARQUET,
        "path_gpkg"                          : OUTPUT_GPKG,
        "failed_pairs_csv"                   : OUTPUT_FAILED,
        "tile_geometry_mode"                 : "full_grid_cell_from_bounds",
        "mun_fraction_covered_normalization" : "normalized_by_intersection_sum",
        "area_unit"                          : "km2",
        "area_crs"                           : AREA_CRS,
        "key_fields"                         : ["tile_id", "mun_code"],
        "n_intersections"                    : int(len(crosswalk)),
        "n_failed_pairs"                     : int(len(failed_pairs)),
        "max_municipality_fraction_closure_error": max_frac_err,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    log("config.json updated.")

    # ----------------------------------------------------------
    # 26. UPDATE CATALOG
    # ----------------------------------------------------------
    row = pd.DataFrame([{
        "stage"      : "03_build_municipality_tile_crosswalk",
        "tile_id"    : "ALL",
        "period"     : "NA",
        "status"     : "completed",
        "output_path": OUTPUT_PARQUET,
        "timestamp"  : datetime.now().isoformat(),
    }])
    if os.path.exists(CATALOG_PATH):
        try:
            cat = pd.read_csv(CATALOG_PATH)
            cat = cat[~((cat["stage"] == "03_build_municipality_tile_crosswalk") &
                        (cat["tile_id"] == "ALL"))]
            pd.concat([cat, row], ignore_index=True).to_csv(CATALOG_PATH, index=False)
        except Exception:
            row.to_csv(CATALOG_PATH, index=False)
    else:
        row.to_csv(CATALOG_PATH, index=False)
    log("Catalog updated.")

# ============================================================
# 27. FIGURE 03 — CROSSWALK DIAGNOSTICS  (500 DPI composite)
# ============================================================

def make_figure_03_crosswalk(crosswalk: gpd.GeoDataFrame,
                              save_dir: str,
                              dpi: int = 500) -> str:
    """
    4-panel publication-quality composite figure:
      a) Tiles per municipality histogram
      b) mun_fraction_covered distribution stacked by macro-region
      c) Fraction closure error distribution (before normalization)
      d) Pipeline progress tracker + key metrics
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
        "gray"   : "#6B7280", "red"    : "#C0504D",
    }

    REGION_COLORS = {
        "North"       : "#4A90D9",
        "Northeast"   : "#E8A838",
        "Center-West" : "#6DB56D",
        "Southeast"   : "#C0504D",
        "South"       : "#9B59B6",
    }

    tiles_per_mun = crosswalk.groupby("mun_code")["tile_id"].nunique()
    frac_vals     = crosswalk["mun_fraction_covered"].values
    n_intersect   = len(crosswalk)
    n_mun         = crosswalk["mun_code"].nunique()
    n_tiles       = crosswalk["tile_id"].nunique()

    raw_sums = (crosswalk.groupby("mun_code")["intersection_area_km2"].sum()
                / crosswalk.groupby("mun_code")["mun_area_km2_geom"].first())
    closure_errors = (raw_sums - 1.0).abs()

    region_col = next((c for c in crosswalk.columns if "region" in c.lower()), None)

    fig = plt.figure(figsize=(7.087, 7.8))
    fig.patch.set_facecolor(C["bg"])

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.09, right=0.97, top=0.91, bottom=0.07,
        hspace=0.44, wspace=0.34,
    )
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_frac = fig.add_subplot(gs[0, 1])
    ax_clos = fig.add_subplot(gs[1, 0])
    ax_prog = fig.add_subplot(gs[1, 1])

    for ax in (ax_hist, ax_frac, ax_clos, ax_prog):
        ax.set_facecolor(C["bg"])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color(C["border"])

    # Panel a: tiles per municipality
    vals  = tiles_per_mun.values
    max_v = int(vals.max())
    ax_hist.hist(vals, bins=range(1, max_v + 2), color=C["blue"],
                 edgecolor="white", linewidth=0.4, alpha=0.85,
                 align="left", zorder=3)
    med_v = np.median(vals)
    ax_hist.axvline(med_v, color=C["accent"], lw=0.9, ls="--", zorder=4)
    ax_hist.text(med_v + 0.1, ax_hist.get_ylim()[1] * 0.88,
                 f"median\n{med_v:.1f}", fontsize=5, color=C["accent"])
    stats = (f"n_mun  {n_mun:,}\n"
             f"min    {vals.min()}\n"
             f"max    {vals.max()}\n"
             f"mean   {vals.mean():.1f}")
    ax_hist.text(0.97, 0.97, stats, transform=ax_hist.transAxes,
                 ha="right", va="top", fontsize=5, family="monospace",
                 color=C["text_sm"],
                 bbox=dict(boxstyle="round,pad=0.4", fc="white",
                           ec=C["border"], lw=0.5, alpha=0.9))
    ax_hist.set_xlabel("Tiles per municipality", fontsize=6,
                       color=C["text_sm"], labelpad=3)
    ax_hist.set_ylabel("Count", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_hist.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_hist.text(0.03, 0.97, "a", transform=ax_hist.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_hist.set_title("Tiles per municipality", fontsize=7,
                      color=C["text_hd"], pad=4)

    # Panel b: mun_fraction_covered stacked by region
    bins = np.linspace(0, 1, 22)
    if region_col:
        bottom = np.zeros(len(bins) - 1)
        for reg, col in REGION_COLORS.items():
            v = crosswalk.loc[crosswalk[region_col] == reg,
                              "mun_fraction_covered"].values
            if len(v):
                h, _ = np.histogram(v, bins=bins)
                ax_frac.bar(bins[:-1], h, width=np.diff(bins),
                            bottom=bottom, color=col, edgecolor="white",
                            linewidth=0.2, alpha=0.82, align="edge",
                            label=reg, zorder=3)
                bottom += h
        leg = [mpatches.Patch(facecolor=c, edgecolor="none", label=r)
               for r, c in REGION_COLORS.items()]
        ax_frac.legend(handles=leg, fontsize=4.5, loc="upper right",
                       frameon=True, framealpha=0.9, edgecolor=C["border"])
    else:
        ax_frac.hist(frac_vals, bins=bins, color=C["blue"],
                     edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)

    med_f = np.median(frac_vals)
    ax_frac.axvline(med_f, color=C["accent"], lw=0.8, ls="--", zorder=4)
    ax_frac.text(med_f + 0.02, ax_frac.get_ylim()[1] * 0.88,
                 f"median\n{med_f:.2f}", fontsize=5, color=C["accent"])
    ax_frac.set_xlabel("Municipality fraction covered per intersection",
                       fontsize=6, color=C["text_sm"], labelpad=3)
    ax_frac.set_ylabel("Count", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_frac.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_frac.text(0.03, 0.97, "b", transform=ax_frac.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_frac.set_title("Municipality fraction covered distribution",
                      fontsize=7, color=C["text_hd"], pad=4)

    # Panel c: closure error (log scale, before normalization)
    err_vals = closure_errors.values
    log_err  = np.log10(err_vals + 1e-15)
    ax_clos.hist(log_err, bins=26, color=C["teal"],
                 edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)
    thr_log = np.log10(0.01)
    ax_clos.axvline(thr_log, color=C["red"], lw=0.9, ls="--", zorder=4)
    ax_clos.text(thr_log + 0.05, ax_clos.get_ylim()[1] * 0.85,
                 "1% threshold\n(pre-normalization)",
                 fontsize=4.5, color=C["red"], va="top")
    n_above = int((err_vals > 0.01).sum())
    ax_clos.text(0.97, 0.60,
                 f"Above 1%:\n{n_above:,} municipalities\n(fixed by normalization)",
                 transform=ax_clos.transAxes, ha="right", va="top",
                 fontsize=5, color=C["red"],
                 bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec=C["border"], lw=0.5, alpha=0.9))
    ax_clos.set_xlabel("Closure error  log\u2081\u2080(|\u03a3 \u2212 1|)",
                       fontsize=6, color=C["text_sm"], labelpad=3)
    ax_clos.set_ylabel("Number of municipalities", fontsize=6,
                       color=C["text_sm"], labelpad=3)
    ax_clos.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_clos.text(0.03, 0.97, "c", transform=ax_clos.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_clos.set_title("Fraction closure error (before normalization)",
                      fontsize=7, color=C["text_hd"], pad=4)

    # Panel d: pipeline progress
    ax_prog.set_xlim(0, 1); ax_prog.set_ylim(0, 1)
    ax_prog.axis("off")
    ax_prog.text(0.03, 0.97, "d", transform=ax_prog.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_prog.set_title("Pipeline progress", fontsize=7, color=C["text_hd"], pad=4)

    pipeline = [
        ("00 · Environment setup",  1.00, C["teal"]),
        ("01 · Processing tiles",   1.00, C["teal"]),
        ("02 · Municipal units",    1.00, C["teal"]),
        ("03 · Tile-mun crosswalk", 1.00, C["teal"]),
        ("04 · Data integration",   0.00, C["gray"]),
        ("05 · Modeling",           0.00, C["gray"]),
        ("06 · Figures",            0.00, C["gray"]),
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

    metrics = (f"Intersections   {n_intersect:,}\n"
               f"Municipalities  {n_mun:,}\n"
               f"Tiles           {n_tiles:,}\n"
               f"Tile geometry   full grid cell\n"
               f"Closure fix     normalized")
    ax_prog.text(0.50, 0.06, metrics, ha="center", va="bottom",
                 fontsize=5, color=C["text_sm"],
                 transform=ax_prog.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", fc=C["panel"],
                           ec=C["border"], lw=0.5))

    for xi, (fc, lbl) in enumerate([(C["teal"], "Completed"),
                                     (C["gray"], "Pending")]):
        bx = 0.30 + xi * 0.35
        ax_prog.add_patch(mpatches.Rectangle(
            (bx, 0.02), 0.025, 0.045, facecolor=fc, edgecolor="none",
            transform=ax_prog.transAxes, clip_on=False, zorder=5))
        ax_prog.text(bx + 0.035, 0.043, lbl, fontsize=5.5, va="center",
                     color=C["text_sm"], transform=ax_prog.transAxes)

    fig.text(
        0.50, 0.966,
        "Figure 4  |  Municipality\u2013tile crosswalk \u2014 Flood Inequality in Brazil",
        ha="center", va="top",
        fontsize=8, fontweight="bold", color=C["text_hd"],
    )
    fig.text(
        0.50, 0.952,
        (f"Spatial crosswalk: {n_tiles:,} tiles \u00d7 {n_mun:,} municipalities "
         f"\u00b7 {n_intersect:,} intersections \u00b7 EPSG:5880 "
         f"\u00b7 mun_fraction normalized"),
        ha="center", va="top",
        fontsize=6, color=C["text_sm"], style="italic",
    )

    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, "fig03_municipality_tile_crosswalk.png")
    pdf_path = os.path.join(save_dir, "fig03_municipality_tile_crosswalk.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=C["bg"])
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor=C["bg"])
    plt.show()
    plt.close(fig)
    return png_path


# ============================================================
# RUN
# ============================================================
log("Generating Figure 03 ...")
with tqdm(total=1, desc="Rendering figure",
          bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
    fig_path = make_figure_03_crosswalk(
        crosswalk,
        os.path.join(BASE_PATH, "06_figures"),
        dpi=500,
    )
    pbar.update(1)

log(f"Figure saved: {fig_path}")
logging.info("Figure 03 generated successfully.")

print("\n" + "=" * 60)
print("  Module 03 complete")
print("=" * 60)
print(f"  Intersections  : {len(crosswalk):,}")
print(f"  GeoParquet     : {OUTPUT_PARQUET}")
print(f"  GeoPackage     : {OUTPUT_GPKG}")
print(f"  Metadata       : {OUTPUT_META}")
print(f"  Failed pairs   : {OUTPUT_FAILED}")
print(f"  Figure PNG     : {OUTPUT_FIG_PNG}")
print(f"  Figure PDF     : {OUTPUT_FIG_PDF}")
print("  Ready for Module 04.")
print("=" * 60)
