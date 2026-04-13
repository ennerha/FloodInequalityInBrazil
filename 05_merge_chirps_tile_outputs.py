"""
Project: Flood Inequality Across Brazil
Module:  05_merge_chirps_tile_outputs.py

Purpose:
  Merge all valid tile-level annual CHIRPS hazard outputs (Module 04) into
  a single national long-format panel dataset. Validates structure and year
  coverage per tile, concatenates all valid outputs, and saves a unified
  national table for downstream aggregation to municipalities and modeling.

Inputs:
  - 03_features/chirps_tile_annual_csv/      (one CSV per tile)
  - 03_features/chirps_tile_annual_meta/     (one JSON per tile)
  - 02_intermediate/processing_tiles_brazil.parquet

Outputs:
  - 04_integrated/chirps_tile_annual_brazil.parquet
  - 04_integrated/chirps_tile_annual_brazil.csv
  - 04_integrated/chirps_tile_annual_brazil.meta.json
  - 04_integrated/chirps_tile_annual_merge_issues.csv
  - 06_figures/fig05_chirps_merged_panel.png   (500 DPI)
  - 06_figures/fig05_chirps_merged_panel.pdf   (vector)
  - 07_logs/05_merge_chirps_tile_outputs.log

Scientific rationale:
  The merged dataset is the national hazard panel at the tile-year level,
  supporting spatial-temporal analysis and tile-to-municipality aggregation
  via the crosswalk built in Module 03.

Reproducibility:
  - Idempotent execution  - safe to re-run without side effects
  - Atomic save (tmp → final)
  - Full validation before and after merge

Author:  Enner H. de Alcantara
Version: v1.1
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
import matplotlib.ticker as mticker
from matplotlib import gridspec

from tqdm.auto import tqdm

# ============================================================
# 3. PATHS AND CONSTANTS
# ============================================================
BASE_PATH    = "/content/drive/MyDrive/Brazil/flood_inequality_project"
CONFIG_PATH  = os.path.join(BASE_PATH, "00_config",  "config.json")
LOG_PATH     = os.path.join(BASE_PATH, "07_logs",    "05_merge_chirps_tile_outputs.log")
CATALOG_PATH = os.path.join(BASE_PATH, "08_catalog", "catalog.csv")
TILES_PATH   = os.path.join(BASE_PATH, "02_intermediate", "processing_tiles_brazil.parquet")

INPUT_DIR      = os.path.join(BASE_PATH, "03_features", "chirps_tile_annual_csv")
INPUT_META_DIR = os.path.join(BASE_PATH, "03_features", "chirps_tile_annual_meta")
INPUT_MANIFEST = os.path.join(BASE_PATH, "03_features", "chirps_tile_annual_manifest.csv")

OUTPUT_DIR     = os.path.join(BASE_PATH, "04_integrated")
OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "chirps_tile_annual_brazil.parquet")
OUTPUT_CSV     = os.path.join(OUTPUT_DIR, "chirps_tile_annual_brazil.csv")
OUTPUT_META    = os.path.join(OUTPUT_DIR, "chirps_tile_annual_brazil.meta.json")
OUTPUT_ISSUES  = os.path.join(OUTPUT_DIR, "chirps_tile_annual_merge_issues.csv")
OUTPUT_FIG_PNG = os.path.join(BASE_PATH, "06_figures", "fig05_chirps_merged_panel.png")
OUTPUT_FIG_PDF = os.path.join(BASE_PATH, "06_figures", "fig05_chirps_merged_panel.pdf")

START_YEAR     = 1981
END_YEAR       = 2025
EXPECTED_YEARS = END_YEAR - START_YEAR + 1

REQUIRED_COLS = [
    "tile_id", "year",
    "annual_prcp_mm", "wet_days_n", "heavy_rain_days_20mm_n",
    "rx1day_mm", "rx3day_mm", "rx5day_mm",
]
METRIC_COLS = REQUIRED_COLS[2:]

VERBOSE = False

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

def log(msg: str, level: str = "INFO") -> None:
    logging.info(f"[{level}] {msg}")
    if VERBOSE or level in ("WARNING", "ERROR", "SUMMARY"):
        print(f"[{level}] {msg}")

# ============================================================
# 6. HELPERS
# ============================================================
def read_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_final_output(parquet_path: str, meta_path: str,
                           expected_tiles: int) -> bool:
    """Return True only if merged outputs exist and pass structural checks."""
    if not os.path.exists(parquet_path) or not os.path.exists(meta_path):
        return False
    try:
        df = pd.read_parquet(parquet_path)
        if df.empty or not set(REQUIRED_COLS).issubset(df.columns):
            return False
        if df["tile_id"].nunique() != expected_tiles:
            return False
        if df["year"].nunique() != EXPECTED_YEARS:
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
    """Upsert one row into the processing catalog CSV."""
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


def validate_tile_csv(csv_path: str, tile_id_expected: str):
    """
    Read and validate one tile CSV. Returns (DataFrame | None, list[str]).
    None means the file is unusable; issues is a list of problem strings.
    """
    issues = []
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, [f"read_error: {e}"]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, [f"missing_columns: {','.join(missing)}"]
    if df.empty:
        return None, ["empty_dataframe"]

    if df["tile_id"].isna().any():
        issues.append("tile_id_contains_na")

    unique_ids = df["tile_id"].dropna().unique().tolist()
    if len(unique_ids) != 1 or unique_ids[0] != tile_id_expected:
        issues.append(f"tile_id_mismatch: found={unique_ids}")

    if df["year"].nunique() != EXPECTED_YEARS:
        issues.append(f"unexpected_year_count: {df['year'].nunique()}")

    expected_yrs = set(range(START_YEAR, END_YEAR + 1))
    observed_yrs = set(df["year"].dropna().astype(int).tolist())
    missing_yrs  = sorted(expected_yrs - observed_yrs)
    extra_yrs    = sorted(observed_yrs - expected_yrs)
    if missing_yrs: issues.append(f"missing_years: {missing_yrs}")
    if extra_yrs:   issues.append(f"extra_years: {extra_yrs}")

    for col in METRIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[METRIC_COLS].isna().all(axis=None):
        issues.append("all_numeric_metrics_are_na")

    df = df.sort_values("year").reset_index(drop=True)
    return df, issues

# ============================================================
# 7. LOAD CONFIG + TILES
# ============================================================
config = read_config(CONFIG_PATH)

if not os.path.exists(TILES_PATH):
    raise FileNotFoundError(f"Tiles not found: {TILES_PATH}")

tiles = gpd.read_parquet(TILES_PATH)
if tiles.empty:
    raise RuntimeError("Processing tiles layer is empty.")

expected_tile_ids = tiles.sort_values("tile_n")["tile_id"].tolist()
expected_n_tiles  = len(expected_tile_ids)
log(f"Expected tiles: {expected_n_tiles:,}", level="SUMMARY")

# ============================================================
# 8. SKIP IF VALID OUTPUT EXISTS
# ============================================================
if is_valid_final_output(OUTPUT_PARQUET, OUTPUT_META, expected_n_tiles):
    log("Valid merged output already exists - skipping.", level="SUMMARY")
    merged = pd.read_parquet(OUTPUT_PARQUET)
    log(f"Loaded merged panel: {len(merged):,} rows, "
        f"{merged['tile_id'].nunique():,} tiles.", level="SUMMARY")

else:
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    # ----------------------------------------------------------
    # 9. READ AND VALIDATE EACH TILE CSV
    # ----------------------------------------------------------
    log("Scanning and validating tile-level CSV files ...")
    merged_frames = []
    issue_rows    = []
    n_completed   = 0
    n_failed      = 0
    n_missing     = 0

    with tqdm(total=expected_n_tiles, desc="Reading tile CSVs",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for tile_id in expected_tile_ids:
            csv_path  = os.path.join(INPUT_DIR,      f"chirps_annual_{tile_id}.csv")
            meta_path = os.path.join(INPUT_META_DIR, f"chirps_annual_{tile_id}.meta.json")

            if not os.path.exists(csv_path):
                issue_rows.append({"tile_id": tile_id,
                                   "issue_type": "missing_csv",
                                   "details": csv_path})
                n_missing += 1
                pbar.update(1); continue

            if not os.path.exists(meta_path):
                issue_rows.append({"tile_id": tile_id,
                                   "issue_type": "missing_meta",
                                   "details": meta_path})
                n_missing += 1
                pbar.update(1); continue

            df, issues = validate_tile_csv(csv_path, tile_id)

            for iss in issues:
                issue_rows.append({"tile_id": tile_id,
                                   "issue_type": "validation_issue",
                                   "details": iss})

            if df is None:
                n_failed += 1
            else:
                merged_frames.append(df)
                n_completed += 1

            pbar.update(1)

    log(f"Scan complete: valid={n_completed:,} | "
        f"failed={n_failed:,} | missing={n_missing:,}", level="SUMMARY")

    if not merged_frames:
        raise RuntimeError("No valid tile-level CHIRPS outputs available for merging.")

    # ----------------------------------------------------------
    # 10. CONCATENATE
    # ----------------------------------------------------------
    log("Concatenating tile frames ...")
    with tqdm(total=1, desc="Concatenating",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        merged = pd.concat(merged_frames, ignore_index=True)
        merged = merged.sort_values(["tile_id","year"]).reset_index(drop=True)
        pbar.update(1)

    # ----------------------------------------------------------
    # 11. NATIONAL VALIDATION
    # ----------------------------------------------------------
    log("Validating merged national panel ...")
    with tqdm(total=4, desc="Validation checks",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        merged_tile_count = merged["tile_id"].nunique()
        merged_year_count = merged["year"].nunique()
        pbar.update(1)

        if merged_year_count != EXPECTED_YEARS:
            raise RuntimeError(
                f"Year count mismatch: got {merged_year_count}, "
                f"expected {EXPECTED_YEARS}.")
        pbar.update(1)

        if merged_tile_count != expected_n_tiles:
            missing_tiles = sorted(
                set(expected_tile_ids) - set(merged["tile_id"].unique()))
            for t in missing_tiles:
                issue_rows.append({"tile_id": t,
                                   "issue_type": "missing_from_merged",
                                   "details": "absent after merge"})
            raise RuntimeError(
                f"Tile count mismatch: got {merged_tile_count}, "
                f"expected {expected_n_tiles}. "
                f"Missing: {missing_tiles[:10]}")
        pbar.update(1)

        duplicates = merged.duplicated(subset=["tile_id","year"]).sum()
        if duplicates > 0:
            raise RuntimeError(
                f"Merged panel contains {duplicates} duplicate tile-year rows.")

        expected_rows = expected_n_tiles * EXPECTED_YEARS
        if len(merged) != expected_rows:
            raise RuntimeError(
                f"Row count mismatch: got {len(merged):,}, "
                f"expected {expected_rows:,}.")
        pbar.update(1)

    log(f"Panel validated: {merged_tile_count:,} tiles × "
        f"{merged_year_count} years = {len(merged):,} rows.", level="SUMMARY")

    # ----------------------------------------------------------
    # 12. SAVE ISSUE TABLE
    # ----------------------------------------------------------
    pd.DataFrame(issue_rows).to_csv(OUTPUT_ISSUES, index=False)
    log(f"Issues CSV: {OUTPUT_ISSUES}  ({len(issue_rows)} entries)")

    # ----------------------------------------------------------
    # 13. SAVE OUTPUTS  (atomic via temp files)
    # ----------------------------------------------------------
    log("Saving merged panel outputs ...")
    tmp_parquet = OUTPUT_PARQUET + ".tmp"
    tmp_csv     = OUTPUT_CSV     + ".tmp"
    tmp_meta    = OUTPUT_META    + ".tmp"

    for p in [tmp_parquet, tmp_csv, tmp_meta]:
        if os.path.exists(p): os.remove(p)

    with tqdm(total=3, desc="Writing files",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        merged.to_parquet(tmp_parquet, index=False); pbar.update(1)
        merged.to_csv(tmp_csv, index=False);         pbar.update(1)

        meta = {
            "project"        : "Flood Inequality Across Brazil",
            "module"         : "05_merge_chirps_tile_outputs.py",
            "version"        : "v1.1",
            "status"         : "completed",
            "created_at"     : datetime.now().isoformat(),
            "dataset"        : "UCSB-CHG/CHIRPS/DAILY",
            "start_year"     : START_YEAR,
            "end_year"       : END_YEAR,
            "expected_years" : EXPECTED_YEARS,
            "expected_tiles" : expected_n_tiles,
            "merged_tiles"   : int(merged_tile_count),
            "merged_rows"    : int(len(merged)),
            "n_failed"       : n_failed,
            "n_missing"      : n_missing,
            "issues_csv"     : OUTPUT_ISSUES,
            "input_dir"      : INPUT_DIR,
            "input_meta_dir" : INPUT_META_DIR,
            "input_manifest" : INPUT_MANIFEST,
            "output_parquet" : OUTPUT_PARQUET,
            "output_csv"     : OUTPUT_CSV,
            "columns"        : list(merged.columns),
        }
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
        pbar.update(1)

    os.replace(tmp_parquet, OUTPUT_PARQUET)
    os.replace(tmp_csv,     OUTPUT_CSV)
    os.replace(tmp_meta,    OUTPUT_META)
    log("All outputs saved.")

    # ----------------------------------------------------------
    # 14. UPDATE CATALOG
    # ----------------------------------------------------------
    update_catalog("05_merge_chirps_tile_outputs", "ALL",
                   OUTPUT_PARQUET, "completed")
    log("Catalog updated.")

    # ----------------------------------------------------------
    # 15. UPDATE CONFIG
    # ----------------------------------------------------------
    config["hazard_chirps_tile_annual_merged"] = {
        "name"           : "chirps_tile_annual_brazil",
        "dataset"        : "UCSB-CHG/CHIRPS/DAILY",
        "start_year"     : START_YEAR,
        "end_year"       : END_YEAR,
        "path_parquet"   : OUTPUT_PARQUET,
        "path_csv"       : OUTPUT_CSV,
        "path_meta"      : OUTPUT_META,
        "path_issues_csv": OUTPUT_ISSUES,
        "n_tiles"        : int(merged_tile_count),
        "n_rows"         : int(len(merged)),
        "metrics"        : METRIC_COLS,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    log("config.json updated.")

# ============================================================
# 16. FIGURE 05 — MERGED CHIRPS PANEL QA  (500 DPI composite)
# ============================================================

def make_figure_05_merged(df: pd.DataFrame,
                           issue_rows: list,
                           save_dir: str,
                           dpi: int = 500) -> str:
    """
    6-panel publication-quality composite figure:
      a) Data coverage heatmap (metric × year, % non-null)
      b) Tile temporal completeness bar chart
      c) Annual precipitation boxplot by decade
      d) Extreme rainfall indices violin plot (Rx1/3/5day)
      e) Inter-metric Pearson correlation heatmap
      f) Merge QA summary table

    Parameters
    ----------
    df         : Merged national CHIRPS panel DataFrame.
    issue_rows : List of issue dicts from the merge scan.
    save_dir   : Directory for PNG and PDF outputs.
    dpi        : Output resolution (default 500).

    Returns
    -------
    str : Path to saved PNG file.
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
        "bg"    : "#FAFAF8", "panel" : "#F2F1ED",
        "text_hd": "#111827","text_sm": "#6B7280",
        "border": "#D1D5DB", "accent": "#B45309",
        "blue"  : "#2E6DA4", "teal"  : "#2A8C6E",
        "gray"  : "#6B7280", "purple": "#6B46C1",
        "red"   : "#C0504D", "amber" : "#D97706",
    }

    SHORT    = ["Prcp","WetD","HvyD","Rx1d","Rx3d","Rx5d"]
    YEARS    = np.arange(START_YEAR, END_YEAR + 1)
    n_tiles  = df["tile_id"].nunique()
    n_rows   = len(df)
    n_issues = len(issue_rows)
    dup      = int(df.duplicated(subset=["tile_id","year"]).sum())
    n_missing_tiles = int(expected_n_tiles - n_tiles)

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.087, 9.4))
    fig.patch.set_facecolor(C["bg"])

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        left=0.10, right=0.97, top=0.93, bottom=0.06,
        hspace=0.50, wspace=0.34,
    )
    ax_cov  = fig.add_subplot(gs[0, 0])
    ax_comp = fig.add_subplot(gs[0, 1])
    ax_prcp = fig.add_subplot(gs[1, 0])
    ax_viol = fig.add_subplot(gs[1, 1])
    ax_corr = fig.add_subplot(gs[2, 0])
    ax_qa   = fig.add_subplot(gs[2, 1])

    for ax in (ax_cov, ax_comp, ax_prcp, ax_viol, ax_corr, ax_qa):
        ax.set_facecolor(C["bg"])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color(C["border"])

    # ── Panel a: Coverage heatmap ────────────────────────────────────────────
    coverage = np.array([
        [df.loc[df["year"] == y, col].notna().mean() for col in METRIC_COLS]
        for y in YEARS
    ])
    im = ax_cov.imshow(coverage.T, aspect="auto", cmap="RdYlGn",
                       vmin=0.5, vmax=1.0, interpolation="nearest")
    step = max(1, len(YEARS) // 9)
    ax_cov.set_xticks(range(0, len(YEARS), step))
    ax_cov.set_xticklabels(YEARS[::step], rotation=45, ha="right", fontsize=5)
    ax_cov.set_yticks(range(len(SHORT)))
    ax_cov.set_yticklabels(SHORT, fontsize=5.5)
    cbar = fig.colorbar(im, ax=ax_cov, pad=0.02, fraction=0.04)
    cbar.ax.tick_params(labelsize=4.5)
    cbar.set_label("Coverage", fontsize=5, color=C["text_sm"])
    ax_cov.set_xlabel("Year", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_cov.text(0.03, 0.97, "a", transform=ax_cov.transAxes,
                fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_cov.set_title("Data coverage by metric × year",
                     fontsize=7, color=C["text_hd"], pad=4)

    # ── Panel b: Tile temporal completeness ──────────────────────────────────
    yrs_per_tile = df.groupby("tile_id")["year"].nunique().values
    bins_c    = [0, 10, 20, 30, 40, EXPECTED_YEARS + 1]
    labels_c  = ["<10","10–19","20–29","30–39",f"40–{EXPECTED_YEARS}"]
    counts_c  = [np.sum((yrs_per_tile >= bins_c[i]) & (yrs_per_tile < bins_c[i+1]))
                 for i in range(len(bins_c)-1)]
    bar_cols  = [C["red"], C["amber"], C["amber"], C["teal"], C["teal"]]
    ax_comp.bar(range(len(labels_c)), counts_c, color=bar_cols,
                edgecolor="white", linewidth=0.3, alpha=0.87, zorder=3)
    for i, cnt in enumerate(counts_c):
        ax_comp.text(i, cnt + 0.2, str(cnt), ha="center",
                     fontsize=5.5, color=C["text_sm"])
    ax_comp.set_xticks(range(len(labels_c)))
    ax_comp.set_xticklabels(labels_c, fontsize=5.5)
    ax_comp.set_xlabel("Years with data", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_comp.set_ylabel("Number of tiles", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_comp.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_comp.text(0.03, 0.97, "b", transform=ax_comp.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_comp.set_title("Tile temporal completeness",
                      fontsize=7, color=C["text_hd"], pad=4)

    # ── Panel c: Annual precipitation boxplot by decade ──────────────────────
    decades = {
        "1981–90": (1981,1990), "1991–00": (1991,2000),
        "2001–10": (2001,2010), "2011–20": (2011,2020),
        "2021–25": (2021,2025),
    }
    box_data = [
        df.loc[(df["year"] >= y0) & (df["year"] <= y1),
               "annual_prcp_mm"].dropna().values
        for y0, y1 in decades.values()
    ]
    bp = ax_prcp.boxplot(
        box_data, patch_artist=True, notch=False, widths=0.55,
        showfliers=True,
        flierprops=dict(marker=".", markersize=1.5, color=C["gray"], alpha=0.4),
        medianprops=dict(color=C["accent"], lw=1.2),
        whiskerprops=dict(color=C["blue"], lw=0.6),
        capprops=dict(color=C["blue"], lw=0.6),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(matplotlib.colors.to_rgba(C["blue"], 0.15))
        patch.set_edgecolor(C["blue"]); patch.set_linewidth(0.6)
    ax_prcp.set_xticklabels(list(decades.keys()), fontsize=5.5)
    ax_prcp.set_xlabel("Decade", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_prcp.set_ylabel("Annual precipitation (mm)", fontsize=6,
                       color=C["text_sm"], labelpad=3)
    ax_prcp.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_prcp.text(0.03, 0.97, "c", transform=ax_prcp.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_prcp.set_title("Annual precipitation by decade",
                      fontsize=7, color=C["text_hd"], pad=4)

    # ── Panel d: Extreme indices violin ──────────────────────────────────────
    ext_m   = ["rx1day_mm","rx3day_mm","rx5day_mm"]
    ext_lbl = ["Rx1day","Rx3day","Rx5day"]
    ext_col = [C["blue"], C["teal"], C["purple"]]
    ext_dat = [df[m].dropna().values for m in ext_m]
    parts   = ax_viol.violinplot(ext_dat, positions=range(len(ext_m)),
                                  showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], ext_col):
        pc.set_facecolor(matplotlib.colors.to_rgba(col, 0.35))
        pc.set_edgecolor(col); pc.set_linewidth(0.6)
    parts["cmedians"].set_color(C["accent"])
    parts["cmedians"].set_linewidth(1.2)
    ax_viol.set_xticks(range(len(ext_m)))
    ax_viol.set_xticklabels(ext_lbl, fontsize=6)
    ax_viol.set_ylabel("Precipitation (mm)", fontsize=6,
                       color=C["text_sm"], labelpad=3)
    ax_viol.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_viol.text(0.03, 0.97, "d", transform=ax_viol.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_viol.set_title("Extreme rainfall indices distribution",
                      fontsize=7, color=C["text_hd"], pad=4)

    # ── Panel e: Inter-metric correlation heatmap ────────────────────────────
    corr = df[METRIC_COLS].corr().values
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_disp = np.where(mask, np.nan, corr)
    im2 = ax_corr.imshow(corr_disp, cmap="RdBu_r", vmin=-1, vmax=1,
                          aspect="auto", interpolation="nearest")
    ax_corr.set_xticks(range(len(SHORT))); ax_corr.set_yticks(range(len(SHORT)))
    ax_corr.set_xticklabels(SHORT, fontsize=5.5)
    ax_corr.set_yticklabels(SHORT, fontsize=5.5)
    for i in range(len(SHORT)):
        for j in range(len(SHORT)):
            if not mask[i, j]:
                v = corr[i, j]
                ax_corr.text(j, i, f"{v:.2f}", ha="center", va="center",
                             fontsize=4.5,
                             color="white" if abs(v) > 0.6 else C["text_hd"])
    cbar2 = fig.colorbar(im2, ax=ax_corr, pad=0.02, fraction=0.04)
    cbar2.ax.tick_params(labelsize=4.5)
    cbar2.set_label("r", fontsize=5, color=C["text_sm"])
    ax_corr.text(0.03, 0.97, "e", transform=ax_corr.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_corr.set_title("Inter-metric Pearson correlation",
                      fontsize=7, color=C["text_hd"], pad=4)

    # ── Panel f: Merge QA summary ────────────────────────────────────────────
    ax_qa.set_xlim(0, 1); ax_qa.set_ylim(0, 1)
    ax_qa.axis("off")
    ax_qa.text(0.03, 0.97, "f", transform=ax_qa.transAxes,
               fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_qa.set_title("Merge QA summary", fontsize=7, color=C["text_hd"], pad=4)

    qa_items = [
        ("Tiles expected",       f"{expected_n_tiles:,}"),
        ("Tiles merged",         f"{n_tiles:,}"),
        ("Tiles missing",        str(n_missing_tiles)),
        ("Years per tile",       f"{EXPECTED_YEARS}  ({START_YEAR}–{END_YEAR})"),
        ("Total rows",           f"{n_rows:,}"),
        ("Duplicate tile-years", str(dup)),
        ("Columns",              str(len(df.columns))),
        ("Issue entries",        str(n_issues)),
        ("Output format",        "Parquet + CSV"),
    ]

    y0_qa = 0.88
    for key, val in qa_items:
        is_ok = val in (f"{expected_n_tiles:,}", f"{n_tiles:,}",
                        "0", f"{n_rows:,}",
                        f"{EXPECTED_YEARS}  ({START_YEAR}–{END_YEAR})",
                        str(len(df.columns)), "Parquet + CSV")
        color_val = C["teal"] if is_ok else C["red"]
        ax_qa.text(0.04, y0_qa, key, ha="left", va="center",
                   fontsize=6, color=C["text_sm"],
                   transform=ax_qa.transAxes)
        ax_qa.text(0.96, y0_qa, val, ha="right", va="center",
                   fontsize=6, color=color_val, fontweight="bold",
                   transform=ax_qa.transAxes)
        ax_qa.plot([0.02, 0.98], [y0_qa - 0.04, y0_qa - 0.04],
                   color=C["border"], lw=0.3,
                   transform=ax_qa.transAxes)
        y0_qa -= 0.096

    # ── Title and caption ──────────────────────────────────────────────────
    fig.text(
        0.50, 0.970,
        "Figure 6  |  Merged CHIRPS tile panel \u2014 Flood Inequality in Brazil",
        ha="center", va="top",
        fontsize=8, fontweight="bold", color=C["text_hd"],
    )
    fig.text(
        0.50, 0.956,
        (f"CHIRPS Daily \u00b7 {n_tiles:,} tiles \u00b7 "
         f"{START_YEAR}\u2013{END_YEAR} \u00b7 "
         f"{n_rows:,} tile-year records \u00b7 national panel validated"),
        ha="center", va="top",
        fontsize=6, color=C["text_sm"], style="italic",
    )

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, "fig05_chirps_merged_panel.png")
    pdf_path = os.path.join(save_dir, "fig05_chirps_merged_panel.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=C["bg"])
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor=C["bg"])
    plt.show()
    plt.close(fig)
    return png_path


# ── Run figure ────────────────────────────────────────────────────────────────
log("Generating Figure 05 ...")
with tqdm(total=1, desc="Rendering figure",
          bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
    fig_path = make_figure_05_merged(
        merged,
        issue_rows if "issue_rows" in dir() else [],
        os.path.join(BASE_PATH, "06_figures"),
        dpi=500,
    )
    pbar.update(1)

log(f"Figure saved: {fig_path}", level="SUMMARY")
logging.info("Figure 05 generated successfully.")

# ============================================================
# 17. FINAL STATUS REPORT
# ============================================================
print("\n" + "=" * 60)
print("  Module 05 complete")
print("=" * 60)
print(f"  Tiles merged   : {merged['tile_id'].nunique():,}")
print(f"  Total rows     : {len(merged):,}")
print(f"  Parquet        : {OUTPUT_PARQUET}")
print(f"  CSV            : {OUTPUT_CSV}")
print(f"  Metadata       : {OUTPUT_META}")
print(f"  Issues CSV     : {OUTPUT_ISSUES}")
print(f"  Figure PNG     : {OUTPUT_FIG_PNG}")
print(f"  Figure PDF     : {OUTPUT_FIG_PDF}")
print("  Ready for Module 06.")
print("=" * 60)
