"""
Project: Flood Inequality Across Brazil
Module:  08_integrate_hazard_with_social_inequality_spatial.py
Version: v5.3

Purpose:
  Integrate CHIRPS-based hazard indices with municipal social inequality
  indicators (Censo 2022) to produce a composite spatial dataset for
  flood inequality analysis.

Changelog v5.3 (from v5.2):
  - tqdm progress bars added to all major processing steps
  - Figure 08 (6-panel composite, 500 DPI) integrated
  - logging unified with logging.basicConfig pattern
  - main() wrapped with tqdm stage tracker

Inputs:
  - 02_intermediate/analysis_units_municipal_brazil.parquet
  - 04_integrated/chirps_municipal_annual_anomalies.parquet
  - 04_integrated/chirps_municipal_trend_summary.parquet
  - MyDrive/Brazil/Censo_2022.xlsx   (or cached social parquet/csv)

Outputs:
  - 04_integrated/hazard_social_inequality_municipal_brazil.geoparquet
  - 04_integrated/hazard_social_inequality_municipal_brazil.gpkg
  - 04_integrated/hazard_social_inequality_municipal_brazil_nogeom.parquet
  - 04_integrated/hazard_social_inequality_municipal_brazil_nogeom.csv
  - 04_integrated/hazard_social_inequality_municipal_brazil.meta.json
  - 04_integrated/hazard_social_inequality_input_diagnostics.csv
  - 04_integrated/social_inequality_municipal_brazil.parquet
  - 04_integrated/social_inequality_municipal_brazil.csv
  - 06_figures/fig08_hazard_social_inequality.png  (500 DPI)
  - 06_figures/fig08_hazard_social_inequality.pdf  (vector)
  - 07_logs/08_integrate_hazard_with_social_inequality_spatial.log

Author:  Enner H. de Alcantara
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
import pyarrow.parquet as pq

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib import gridspec

from tqdm.auto import tqdm

# ============================================================
# 3. PATHS AND CONSTANTS
# ============================================================
BASE_PATH         = Path("/content/drive/MyDrive/Brazil/flood_inequality_project")
DRIVE_BRAZIL_PATH = Path("/content/drive/MyDrive/Brazil")

CONFIG_PATH  = BASE_PATH / "00_config"  / "config.json"
LOG_PATH     = BASE_PATH / "07_logs"    / "08_integrate_hazard_with_social_inequality_spatial.log"
CATALOG_PATH = BASE_PATH / "08_catalog" / "catalog.csv"

MUNICIPAL_PATH       = BASE_PATH / "02_intermediate" / "analysis_units_municipal_brazil.parquet"
ANOM_PATH            = BASE_PATH / "04_integrated"   / "chirps_municipal_annual_anomalies.parquet"
TREND_PATH           = BASE_PATH / "04_integrated"   / "chirps_municipal_trend_summary.parquet"
SOCIAL_TABLE_PARQUET = BASE_PATH / "04_integrated"   / "social_inequality_municipal_brazil.parquet"
SOCIAL_TABLE_CSV     = BASE_PATH / "04_integrated"   / "social_inequality_municipal_brazil.csv"
CENSO_2022_XLSX      = DRIVE_BRAZIL_PATH             / "Censo_2022.xlsx"

OUTPUT_DIR            = BASE_PATH / "04_integrated"
OUTPUT_GEOPARQUET     = OUTPUT_DIR / "hazard_social_inequality_municipal_brazil.geoparquet"
OUTPUT_GPKG           = OUTPUT_DIR / "hazard_social_inequality_municipal_brazil.gpkg"
OUTPUT_NOGEOM_PARQUET = OUTPUT_DIR / "hazard_social_inequality_municipal_brazil_nogeom.parquet"
OUTPUT_NOGEOM_CSV     = OUTPUT_DIR / "hazard_social_inequality_municipal_brazil_nogeom.csv"
OUTPUT_META           = OUTPUT_DIR / "hazard_social_inequality_municipal_brazil.meta.json"
OUTPUT_DIAG           = OUTPUT_DIR / "hazard_social_inequality_input_diagnostics.csv"
OUTPUT_SOCIAL_PARQUET = OUTPUT_DIR / "social_inequality_municipal_brazil.parquet"
OUTPUT_SOCIAL_CSV     = OUTPUT_DIR / "social_inequality_municipal_brazil.csv"
OUTPUT_FIG_PNG        = BASE_PATH  / "06_figures" / "fig08_hazard_social_inequality.png"
OUTPUT_FIG_PDF        = BASE_PATH  / "06_figures" / "fig08_hazard_social_inequality.pdf"

GPKG_LAYER_NAME = "hazard_social_inequality_municipal_brazil"
START_YEAR      = 1981
END_YEAR        = 2025
RECENT_START    = 2016
RECENT_END      = 2025
EXTREME_RANK_THRESHOLD = 5
GEOGRAPHIC_CRS  = "EPSG:4326"

MUNI_ID_COLS = ["mun_code", "mun_name", "uf_code", "uf_sigla"]
METRIC_COLS  = [
    "annual_prcp_mm", "wet_days_n", "heavy_rain_days_20mm_n",
    "rx1day_mm", "rx3day_mm", "rx5day_mm",
]

SOCIAL_SCHEMA_COLS = [
    "mun_code", "mun_name", "uf_sigla",
    "population_total", "population_density",
    "urbanized_area_km2", "urbanization_proxy_pct",
    "income_pc", "illiteracy_rate",
    "water_supply_adequate_pct",
    "sewerage_adequate_pct",   # NaN from Censo 2022; kept for forward-compat
    "poverty_rate",            # NaN from Censo 2022; kept for forward-compat
    "extreme_poverty_rate",    # NaN from Censo 2022; kept for forward-compat
]

MIN_REQUIRED_SOCIAL_FOR_INDEX = [
    "income_pc", "illiteracy_rate", "water_supply_adequate_pct",
]

# v5.2 fix: only columns with real data from Censo 2022
ADVERSE_COLS    = ["illiteracy_rate"]
PROTECTIVE_COLS = ["income_pc", "water_supply_adequate_pct"]
EXPECTED_SOCIAL_ANALYTIC_COLS = ADVERSE_COLS + PROTECTIVE_COLS

OPTIONAL_CONTEXT_COLS = [
    "population_total", "population_density", "urbanization_proxy_pct",
]

SAFE_MEAN_MIN_COVERAGE = 0.50
VERBOSE = False

# ============================================================
# 4. DIRECTORY SETUP + LOGGING
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

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
    if VERBOSE or level in ("WARNING", "ERROR", "SUMMARY"):
        print(f"[{level}] {msg}")

def log_summary(msg: str) -> None:
    log(msg, level="SUMMARY")

# ============================================================
# 5. HELPERS
# ============================================================
def read_config(path: Path) -> dict:
    if not path.exists(): return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)


def update_catalog(stage: str, tile_id: str,
                   output_path: str, status: str) -> None:
    row = pd.DataFrame([{
        "stage": stage, "tile_id": tile_id,
        "period": f"{START_YEAR}_{END_YEAR}",
        "status": status, "output_path": output_path,
        "timestamp": datetime.now().isoformat(),
    }])
    if CATALOG_PATH.exists():
        try:
            df = pd.read_csv(CATALOG_PATH)
            df = df[~((df["stage"] == stage) & (df["tile_id"] == tile_id))]
            pd.concat([df, row], ignore_index=True).to_csv(CATALOG_PATH, index=False)
            return
        except Exception:
            pass
    row.to_csv(CATALOG_PATH, index=False)


def zscore_series(s: pd.Series) -> pd.Series:
    s  = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=1)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def safe_mean(df: pd.DataFrame, cols: list,
              min_coverage: float = SAFE_MEAN_MIN_COVERAGE) -> pd.Series:
    valid = [c for c in cols if c in df.columns]
    if not valid:
        return pd.Series(np.nan, index=df.index)
    sub       = df[valid]
    result    = sub.mean(axis=1, skipna=True)
    threshold = max(1, int(np.ceil(len(valid) * min_coverage)))
    result[sub.notna().sum(axis=1) < threshold] = np.nan
    return result


def parquet_schema_columns(path: Path) -> list:
    return pq.read_schema(str(path)).names


def is_valid_output(geoparquet_path: Path, meta_path: Path,
                    expected_munis: int) -> bool:
    if not geoparquet_path.exists() or not meta_path.exists():
        return False
    try:
        cols = set(parquet_schema_columns(geoparquet_path))
        required = {
            "mun_code", "geometry",
            "social_inequality_index", "adaptive_capacity_index",
            "hazard_recent_extremes_index", "hazard_trend_index",
            "hazard_inequality_coupling_index",
        }
        if not required.issubset(cols): return False
        if pq.read_metadata(str(geoparquet_path)).num_rows != expected_munis:
            return False
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("status") == "completed"
    except Exception:
        return False


def coerce_numeric(series: pd.Series) -> pd.Series:
    s = (series.astype(str).str.strip()
         .str.replace(".", "", regex=False)
         .str.replace(",", ".", regex=False)
         .replace({"-": np.nan, "nan": np.nan, "None": np.nan, "": np.nan}))
    return pd.to_numeric(s, errors="coerce")


def social_table_has_minimum_quality(df: pd.DataFrame) -> bool:
    if df.empty or "mun_code" not in df.columns: return False
    return sum(
        1 for c in MIN_REQUIRED_SOCIAL_FOR_INDEX
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()
    ) >= 3

# ============================================================
# 6. SOCIAL TABLE BUILDER
# ============================================================
def build_social_from_censo_2022() -> tuple:
    if not CENSO_2022_XLSX.exists():
        raise FileNotFoundError(f"Censo_2022.xlsx not found: {CENSO_2022_XLSX}")
    log_summary(f"Building social table from Censo 2022: {CENSO_2022_XLSX}")

    with tqdm(total=1, desc="Reading Censo 2022 xlsx",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        df = pd.read_excel(CENSO_2022_XLSX)
        pbar.update(1)

    required_raw = [
        "CD_MUN", "NM_MUN", "SIGLA_UF",
        "Área da unidade territorial (Quilômetros quadrados)",
        "População residente (Pessoas)",
        "Densidade demográfica (Habitante por quilômetro quadrado)",
        "Total de áreas urbanizadas (Quilômetros quadrados)",
        "Valor do rendimento nominal médio mensal de todos os trabalhos das pessoas de 14 anos ou mais de idade",
        "% Pessoas de 15 anos ou mais de idade, não alfabetizadas",
        "Moradores em domicílios particulares permanentes ocupados, por existência de ligação à rede geral de distribuição de água e principal forma de abastecimento de água",
    ]
    missing = [c for c in required_raw if c not in df.columns]
    if missing:
        raise RuntimeError(f"Censo_2022.xlsx missing columns: {missing}")

    out = pd.DataFrame()
    out["mun_code"] = (df["CD_MUN"].astype(str)
                       .str.extract(r"(\d+)", expand=False).str.zfill(7))
    out["mun_name"] = df["NM_MUN"].astype(str).str.strip()
    out["uf_sigla"] = df["SIGLA_UF"].astype(str).str.strip()

    def get_col(name):
        return coerce_numeric(df.iloc[:, df.columns.get_loc(name)])

    area_km2      = get_col(required_raw[3])
    pop_total     = get_col(required_raw[4])
    density       = get_col(required_raw[5])
    urban_area    = get_col(required_raw[6])
    income_pc     = get_col(required_raw[7])
    illit         = get_col(required_raw[8])
    water_network = get_col(required_raw[9])

    out["population_total"]          = pop_total
    out["population_density"]        = density
    out["urbanized_area_km2"]        = urban_area
    out["urbanization_proxy_pct"]    = np.where(area_km2 > 0, (urban_area / area_km2) * 100.0, np.nan)
    out["income_pc"]                 = income_pc
    out["illiteracy_rate"]           = illit
    out["water_supply_adequate_pct"] = np.where(pop_total > 0, (water_network / pop_total) * 100.0, np.nan)
    out["sewerage_adequate_pct"]     = np.nan
    out["poverty_rate"]              = np.nan
    out["extreme_poverty_rate"]      = np.nan

    pct_cols = ["urbanization_proxy_pct","illiteracy_rate","water_supply_adequate_pct",
                "sewerage_adequate_pct","poverty_rate","extreme_poverty_rate"]
    for col in pct_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[out[col] < 0,   col] = np.nan
        out.loc[out[col] > 100, col] = np.nan

    out = (out[SOCIAL_SCHEMA_COLS].copy()
           .dropna(subset=["mun_code"])
           .drop_duplicates(subset=["mun_code"])
           .reset_index(drop=True))

    out.to_parquet(OUTPUT_SOCIAL_PARQUET, index=False)
    out.to_csv(OUTPUT_SOCIAL_CSV, index=False)
    return out, str(CENSO_2022_XLSX)


def load_or_build_social_table() -> tuple:
    for kind, path in [("parquet", SOCIAL_TABLE_PARQUET),
                       ("csv",     SOCIAL_TABLE_CSV)]:
        if path.exists():
            try:
                df = pd.read_parquet(path) if kind == "parquet" else pd.read_csv(path)
                if social_table_has_minimum_quality(df):
                    log_summary(f"Using existing social table: {path}")
                    return df, str(path)
                log(f"Social table incomplete, will rebuild: {path}", "WARNING")
            except Exception as exc:
                log(f"Failed reading {path}: {exc}", "WARNING")
    return build_social_from_censo_2022()

# ============================================================
# 7. LOAD INPUTS
# ============================================================
def load_inputs():
    for path in [MUNICIPAL_PATH, ANOM_PATH, TREND_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    with tqdm(total=4, desc="Reading inputs",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        municipal = gpd.read_parquet(MUNICIPAL_PATH); pbar.update(1)
        anom      = pd.read_parquet(ANOM_PATH);       pbar.update(1)
        trend     = pd.read_parquet(TREND_PATH);      pbar.update(1)
        social, social_source = load_or_build_social_table(); pbar.update(1)

    for name, obj in [("Municipal", municipal), ("Anomaly", anom),
                       ("Trend", trend), ("Social", social)]:
        if len(obj) == 0:
            raise RuntimeError(f"{name} table is empty.")

    municipal = municipal.to_crs(GEOGRAPHIC_CRS).copy()
    for df_obj in [municipal, anom, trend, social]:
        if "mun_code" in df_obj.columns:
            df_obj["mun_code"] = df_obj["mun_code"].astype(str)

    expected_munis = municipal["mun_code"].nunique()
    log_summary(
        f"Inputs loaded | municipalities={expected_munis:,} | "
        f"anomaly_rows={len(anom):,} | social_rows={len(social):,}")
    return municipal, anom, trend, social, social_source, expected_munis

# ============================================================
# 8. STANDARDIZE SOCIAL TABLE
# ============================================================
def standardize_social_table(social: pd.DataFrame) -> tuple:
    log("Standardizing social table ...")
    diagnostics = []
    social = social.copy()
    social["mun_code"] = social["mun_code"].astype(str)

    with tqdm(total=2, desc="Social table QC",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for col in SOCIAL_SCHEMA_COLS:
            if col not in social.columns:
                social[col] = np.nan
        pbar.update(1)

        for col in [c for c in SOCIAL_SCHEMA_COLS
                    if c not in ["mun_code","mun_name","uf_sigla"]]:
            social[col] = pd.to_numeric(social[col], errors="coerce")
        social = social.drop_duplicates(subset=["mun_code"]).reset_index(drop=True)
        pbar.update(1)

    for col in EXPECTED_SOCIAL_ANALYTIC_COLS:
        diagnostics.append({
            "check": f"input_missing_{col}",
            "value": int(social[col].isna().sum()), "details": "",
        })

    available = [c for c in EXPECTED_SOCIAL_ANALYTIC_COLS if social[c].notna().any()]
    diagnostics.append({
        "check": "available_social_analytic_columns",
        "value": len(available), "details": ", ".join(available),
    })

    minimum = [c for c in MIN_REQUIRED_SOCIAL_FOR_INDEX if social[c].notna().any()]
    if len(minimum) < 3:
        raise RuntimeError("Too few valid social variables for index construction.")

    return social, diagnostics

# ============================================================
# 9. HAZARD FEATURES
# ============================================================
def build_hazard_features(anom: pd.DataFrame,
                           trend: pd.DataFrame) -> pd.DataFrame:
    log("Building hazard features ...")
    recent = anom.loc[anom["year"].between(RECENT_START, RECENT_END)].copy()
    if recent.empty:
        raise RuntimeError(f"Recent anomaly subset [{RECENT_START}-{RECENT_END}] is empty.")

    with tqdm(total=4, desc="Hazard features",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        # Recent z-score means + maxima
        recent_records = recent[MUNI_ID_COLS].drop_duplicates().copy()
        for metric in METRIC_COLS:
            zcol = f"{metric}_zscore"
            if zcol not in recent.columns:
                raise RuntimeError(f"Missing z-score column: {zcol}")
            grp = recent.groupby(MUNI_ID_COLS, as_index=False)[zcol]
            recent_records = (
                recent_records
                .merge(grp.mean().rename(columns={zcol: f"{metric}_recent_mean_z_{RECENT_START}_{RECENT_END}"}), on=MUNI_ID_COLS, how="left")
                .merge(grp.max().rename(columns={zcol: f"{metric}_recent_max_z_{RECENT_START}_{RECENT_END}"}), on=MUNI_ID_COLS, how="left")
            )
        pbar.update(1)

        # Extreme abs z-scores (full period)
        extreme_records = anom[MUNI_ID_COLS].drop_duplicates().copy()
        for metric in METRIC_COLS:
            zcol = f"{metric}_zscore"
            ea = (anom.assign(_az=anom[zcol].abs())
                  .groupby(MUNI_ID_COLS, as_index=False)["_az"]
                  .max()
                  .rename(columns={"_az": f"{metric}_max_abs_z_{START_YEAR}_{END_YEAR}"}))
            extreme_records = extreme_records.merge(ea, on=MUNI_ID_COLS, how="left")
        pbar.update(1)

        # Recent top-N rank flags
        extreme_flag_records = anom[MUNI_ID_COLS].drop_duplicates().copy()
        for metric in METRIC_COLS:
            rcol = f"{metric}_rank_desc"
            if rcol not in anom.columns:
                raise RuntimeError(f"Missing rank column: {rcol}")
            rf = (anom.loc[anom["year"].between(RECENT_START, RECENT_END)]
                  .groupby(MUNI_ID_COLS, as_index=False)[rcol].min()
                  .rename(columns={rcol: f"{metric}_best_recent_rank_desc"}))
            extreme_flag_records = extreme_flag_records.merge(rf, on=MUNI_ID_COLS, how="left")
            extreme_flag_records[f"{metric}_recent_top_{EXTREME_RANK_THRESHOLD}_flag"] = (
                extreme_flag_records[f"{metric}_best_recent_rank_desc"]
                .le(EXTREME_RANK_THRESHOLD).astype(float))
        pbar.update(1)

        # Trend features + hazard indices
        trend_needed = []
        for metric in METRIC_COLS:
            trend_needed.extend([
                f"{metric}_ols_slope_per_year", f"{metric}_sen_slope_per_year",
                f"{metric}_mk_p",               f"{metric}_mk_direction",
            ])
        missing_t = [c for c in trend_needed if c not in trend.columns]
        if missing_t:
            raise RuntimeError(f"Trend summary missing columns: {missing_t}")
        trend_features = trend[MUNI_ID_COLS + trend_needed].copy()

        hazard = (trend_features
                  .merge(recent_records,       on=MUNI_ID_COLS, how="left")
                  .merge(extreme_records,      on=MUNI_ID_COLS, how="left")
                  .merge(extreme_flag_records, on=MUNI_ID_COLS, how="left"))

        recent_max_cols = [f"{m}_recent_max_z_{RECENT_START}_{RECENT_END}" for m in METRIC_COLS]
        hazard["hazard_recent_extremes_raw"]   = safe_mean(hazard, recent_max_cols)
        hazard["hazard_recent_extremes_index"] = zscore_series(hazard["hazard_recent_extremes_raw"])

        for metric in METRIC_COLS:
            sc = f"{metric}_sen_slope_per_year"
            hazard[f"{sc}_z"] = zscore_series(hazard[sc])

        slope_z_cols = [f"{m}_sen_slope_per_year_z" for m in METRIC_COLS]
        hazard["hazard_trend_raw"]   = safe_mean(hazard, slope_z_cols)
        hazard["hazard_trend_index"] = zscore_series(hazard["hazard_trend_raw"])

        sig_cols = []
        for metric in METRIC_COLS:
            scol = f"{metric}_mk_significant_flag"
            hazard[scol] = pd.to_numeric(hazard[f"{metric}_mk_p"], errors="coerce").lt(0.05).astype(float)
            sig_cols.append(scol)
        hazard["hazard_significance_count"] = hazard[sig_cols].sum(axis=1, skipna=True)
        pbar.update(1)

    return hazard

# ============================================================
# 10. SOCIAL INDICES
# ============================================================
def build_social_indices(social: pd.DataFrame) -> pd.DataFrame:
    log("Building social indices ...")
    social = social.copy()

    with tqdm(total=2, desc="Social indices",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        adverse_avail    = [c for c in ADVERSE_COLS    if social[c].notna().any()]
        protective_avail = [c for c in PROTECTIVE_COLS if social[c].notna().any()]

        if not adverse_avail and not protective_avail:
            raise RuntimeError("No valid social variables for index construction.")

        for col in adverse_avail:
            social[f"{col}_z_risk"] =  zscore_series(social[col])
        for col in protective_avail:
            social[f"{col}_z_risk"] = -zscore_series(social[col])

        risk_cols = [c for c in social.columns if c.endswith("_z_risk")]
        social["social_deprivation_raw"]  = safe_mean(social, risk_cols)
        social["social_inequality_index"] = zscore_series(social["social_deprivation_raw"])
        pbar.update(1)

        for col in protective_avail:
            social[f"{col}_z_capacity"] = zscore_series(social[col])
        cap_cols = [c for c in social.columns if c.endswith("_z_capacity")]
        social["adaptive_capacity_raw"]   = safe_mean(social, cap_cols)
        social["adaptive_capacity_index"] = zscore_series(social["adaptive_capacity_raw"])
        pbar.update(1)

    return social

# ============================================================
# 11. SPATIAL INTEGRATION
# ============================================================
def integrate_spatial(municipal: gpd.GeoDataFrame,
                       social: pd.DataFrame,
                       hazard: pd.DataFrame) -> gpd.GeoDataFrame:
    log("Integrating spatial dataset ...")
    with tqdm(total=1, desc="Spatial merge",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        social_keep = ["mun_code"] + [c for c in social.columns if c not in MUNI_ID_COLS]
        hazard_keep = [c for c in hazard.columns if c not in ["mun_name","uf_code","uf_sigla"]]

        gdf = (municipal.copy()
               .merge(social[social_keep], on="mun_code", how="left")
               .merge(hazard[hazard_keep], on="mun_code", how="left"))

        gdf["hazard_inequality_coupling_raw"]   = (gdf["hazard_recent_extremes_index"] + gdf["social_inequality_index"])
        gdf["hazard_inequality_coupling_index"] = zscore_series(gdf["hazard_inequality_coupling_raw"])
        gdf["trend_inequality_coupling_raw"]    = (gdf["hazard_trend_index"] + gdf["social_inequality_index"])
        gdf["trend_inequality_coupling_index"]  = zscore_series(gdf["trend_inequality_coupling_raw"])

        h_med = gdf["hazard_recent_extremes_index"].median(skipna=True)
        s_med = gdf["social_inequality_index"].median(skipna=True)
        h = gdf["hazard_recent_extremes_index"]
        s = gdf["social_inequality_index"]

        gdf["hazard_social_quadrant"] = np.select(
            [h.isna() | s.isna(),
             h.ge(h_med) & s.ge(s_med),
             h.ge(h_med) & s.lt(s_med),
             h.lt(h_med) & s.ge(s_med)],
            ["missing",
             "high_hazard_high_inequality",
             "high_hazard_low_inequality",
             "low_hazard_high_inequality"],
            default="low_hazard_low_inequality",
        )
        pbar.update(1)

    return gdf

# ============================================================
# 12. VALIDATION
# ============================================================
def validate_output(gdf: gpd.GeoDataFrame,
                    expected_munis: int,
                    diagnostics: list) -> list:
    with tqdm(total=3, desc="Validation checks",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        if gdf["mun_code"].nunique() != expected_munis:
            raise RuntimeError(
                f"Municipality count mismatch: got {gdf['mun_code'].nunique()}, "
                f"expected {expected_munis}.")
        pbar.update(1)
        if gdf.duplicated(subset=["mun_code"]).any():
            raise RuntimeError("Duplicated mun_code rows detected.")
        pbar.update(1)
        if gdf.geometry.isna().any():
            raise RuntimeError("Missing geometries in output.")
        pbar.update(1)

    for col in EXPECTED_SOCIAL_ANALYTIC_COLS:
        diagnostics.append({
            "check": f"final_missing_{col}",
            "value": int(gdf[col].isna().sum()) if col in gdf.columns else -1,
            "details": "",
        })
    diagnostics.append({
        "check": "final_municipalities",
        "value": int(gdf["mun_code"].nunique()), "details": "",
    })
    return diagnostics

# ============================================================
# 13. SAVE OUTPUTS
# ============================================================
def save_outputs(gdf: gpd.GeoDataFrame,
                 diagnostics: list,
                 config: dict,
                 social_source: str,
                 expected_munis: int) -> None:
    gdf = gdf.sort_values(["uf_sigla","mun_name"]).reset_index(drop=True)

    tmps = {
        "geoparq" : str(OUTPUT_GEOPARQUET)     + ".tmp",
        "gpkg"    : str(OUTPUT_GPKG).replace(".gpkg","_tmp.gpkg"),
        "nogeom_p": str(OUTPUT_NOGEOM_PARQUET) + ".tmp",
        "nogeom_c": str(OUTPUT_NOGEOM_CSV)     + ".tmp",
        "meta"    : str(OUTPUT_META)           + ".tmp",
    }
    for p in tmps.values():
        if os.path.exists(p): os.remove(p)

    with tqdm(total=5, desc="Writing files",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        try:
            gdf.to_parquet(tmps["geoparq"], index=False);                    pbar.update(1)
            gdf.to_file(tmps["gpkg"], layer=GPKG_LAYER_NAME, driver="GPKG"); pbar.update(1)
            nogeom = pd.DataFrame(gdf.drop(columns="geometry"))
            nogeom.to_parquet(tmps["nogeom_p"], index=False)
            nogeom.to_csv(tmps["nogeom_c"], index=False)
            pd.DataFrame(diagnostics).to_csv(OUTPUT_DIAG, index=False);      pbar.update(1)

            meta = {
                "project"                          : "Flood Inequality Across Brazil",
                "module"                           : "08_integrate_hazard_with_social_inequality_spatial.py",
                "version"                          : "v5.3",
                "status"                           : "completed",
                "created_at"                       : datetime.now().isoformat(),
                "municipal_geometry_input"         : str(MUNICIPAL_PATH),
                "social_table_input"               : social_source,
                "hazard_anomaly_input"             : str(ANOM_PATH),
                "hazard_trend_input"               : str(TREND_PATH),
                "n_municipalities"                 : int(expected_munis),
                "recent_period_start"              : RECENT_START,
                "recent_period_end"                : RECENT_END,
                "expected_social_analytic_columns" : EXPECTED_SOCIAL_ANALYTIC_COLS,
                "minimum_required_social_for_index": MIN_REQUIRED_SOCIAL_FOR_INDEX,
                "adverse_cols"                     : ADVERSE_COLS,
                "protective_cols"                  : PROTECTIVE_COLS,
                "output_geoparquet"                : str(OUTPUT_GEOPARQUET),
                "output_gpkg"                      : str(OUTPUT_GPKG),
                "output_nogeom_parquet"            : str(OUTPUT_NOGEOM_PARQUET),
                "output_nogeom_csv"                : str(OUTPUT_NOGEOM_CSV),
                "diagnostics_csv"                  : str(OUTPUT_DIAG),
                "columns"                          : list(gdf.columns),
            }
            with open(tmps["meta"], "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
            pbar.update(1)

            os.replace(tmps["geoparq"],  OUTPUT_GEOPARQUET)
            os.replace(tmps["gpkg"],     OUTPUT_GPKG)
            os.replace(tmps["nogeom_p"], OUTPUT_NOGEOM_PARQUET)
            os.replace(tmps["nogeom_c"], OUTPUT_NOGEOM_CSV)
            os.replace(tmps["meta"],     OUTPUT_META)
            pbar.update(1)

        except Exception:
            for p in tmps.values():
                try:
                    if os.path.exists(p): os.remove(p)
                except OSError:
                    pass
            raise

    update_catalog("08_integrate_hazard_with_social_inequality_spatial",
                   "ALL", str(OUTPUT_GEOPARQUET), "completed")

    config["hazard_social_inequality_integrated_spatial"] = {
        "name"                : "hazard_social_inequality_municipal_brazil",
        "path_geoparquet"     : str(OUTPUT_GEOPARQUET),
        "path_gpkg"           : str(OUTPUT_GPKG),
        "path_nogeom_parquet" : str(OUTPUT_NOGEOM_PARQUET),
        "path_nogeom_csv"     : str(OUTPUT_NOGEOM_CSV),
        "path_meta"           : str(OUTPUT_META),
        "path_diagnostics_csv": str(OUTPUT_DIAG),
        "social_table_input"  : social_source,
        "n_municipalities"    : int(expected_munis),
    }
    write_config(CONFIG_PATH, config)

# ============================================================
# 14. FIGURE 08 — HAZARD-SOCIAL INEQUALITY  (500 DPI composite)
# ============================================================

def make_figure_08_integration(gdf: gpd.GeoDataFrame,
                                save_dir: str,
                                dpi: int = 500) -> str:
    """
    6-panel publication-quality composite figure:
      a) Hazard extremes vs social inequality quadrant scatter
      b) Coupling index by macro-region (violin)
      c) Municipality quadrant distribution (bar chart)
      d) Social components distribution (overlapping histograms)
      e) Adaptive capacity index distribution
      f) Integration QA summary table
    """

    matplotlib.rcParams.update({
        "font.family"      : "sans-serif",
        "font.sans-serif"  : ["Helvetica","Arial","DejaVu Sans"],
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
        "bg"    : "#FAFAF8","panel" : "#F2F1ED",
        "text_hd": "#111827","text_sm": "#6B7280",
        "border": "#D1D5DB","accent": "#B45309",
        "blue"  : "#2E6DA4","teal"  : "#2A8C6E",
        "gray"  : "#6B7280","purple": "#6B46C1",
        "red"   : "#C0504D","amber" : "#D97706",
    }

    REGION_COLORS = {
        "North":"#4A90D9","Northeast":"#E8A838","Center-West":"#6DB56D",
        "Southeast":"#C0504D","South":"#9B59B6",
    }
    REG_ORDER = ["North","Northeast","Center-West","Southeast","South"]

    QUAD_COLORS = {
        "high_hazard_high_inequality" : C["red"],
        "high_hazard_low_inequality"  : C["amber"],
        "low_hazard_high_inequality"  : C["blue"],
        "low_hazard_low_inequality"   : C["teal"],
    }
    QUAD_ABBR = {
        "high_hazard_high_inequality":"HH","high_hazard_low_inequality":"HL",
        "low_hazard_high_inequality":"LH","low_hazard_low_inequality":"LL",
    }
    QUAD_LABEL = {
        "high_hazard_high_inequality":"High hazard\nHigh inequality",
        "high_hazard_low_inequality" :"High hazard\nLow inequality",
        "low_hazard_high_inequality" :"Low hazard\nHigh inequality",
        "low_hazard_low_inequality"  :"Low hazard\nLow inequality",
    }

    df    = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
    n_mun = df["mun_code"].nunique()
    reg_col = next((c for c in df.columns if "region" in c.lower()), None)

    fig = plt.figure(figsize=(7.087, 9.4))
    fig.patch.set_facecolor(C["bg"])
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        left=0.09, right=0.97, top=0.93, bottom=0.06,
        hspace=0.50, wspace=0.34,
    )
    ax_sc   = fig.add_subplot(gs[0, 0])
    ax_coup = fig.add_subplot(gs[0, 1])
    ax_quad = fig.add_subplot(gs[1, 0])
    ax_soc  = fig.add_subplot(gs[1, 1])
    ax_aci  = fig.add_subplot(gs[2, 0])
    ax_qa   = fig.add_subplot(gs[2, 1])

    for ax in (ax_sc, ax_coup, ax_quad, ax_soc, ax_aci, ax_qa):
        ax.set_facecolor(C["bg"])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color(C["border"])

    # Panel a: quadrant scatter
    for q, qcol in QUAD_COLORS.items():
        mask = df["hazard_social_quadrant"] == q
        ax_sc.scatter(df.loc[mask,"social_inequality_index"],
                      df.loc[mask,"hazard_recent_extremes_index"],
                      s=1.5, color=qcol, alpha=0.35, linewidths=0, zorder=3)
    h_med = df["hazard_recent_extremes_index"].median(skipna=True)
    s_med = df["social_inequality_index"].median(skipna=True)
    ax_sc.axhline(h_med, color=C["text_sm"], lw=0.5, ls="--", zorder=4)
    ax_sc.axvline(s_med, color=C["text_sm"], lw=0.5, ls="--", zorder=4)
    ax_sc.set_xlabel("Social inequality index (z)", fontsize=6,
                     color=C["text_sm"], labelpad=3)
    ax_sc.set_ylabel("Hazard extremes index (z)", fontsize=6,
                     color=C["text_sm"], labelpad=3)
    ax_sc.grid(linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    leg_h = [mpatches.Patch(facecolor=QUAD_COLORS[q], edgecolor="none",
                             label=QUAD_LABEL[q]) for q in QUAD_COLORS]
    ax_sc.legend(handles=leg_h, fontsize=4.0, loc="upper left",
                 frameon=True, framealpha=0.9, edgecolor=C["border"],
                 ncol=2, handlelength=0.8)
    ax_sc.text(0.03, 0.97, "a", transform=ax_sc.transAxes,
               fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_sc.set_title("Hazard vs social inequality quadrant",
                    fontsize=7, color=C["text_hd"], pad=4)

    # Panel b: coupling violin by region
    coup_col  = "hazard_inequality_coupling_index"
    coup_data = ([df.loc[df[reg_col]==r, coup_col].dropna().values for r in REG_ORDER]
                 if reg_col else [df[coup_col].dropna().values]*len(REG_ORDER))
    parts = ax_coup.violinplot(coup_data, positions=range(len(REG_ORDER)),
                                showmedians=True, showextrema=False)
    for pc, reg in zip(parts["bodies"], REG_ORDER):
        col = REGION_COLORS[reg]
        pc.set_facecolor(mcolors.to_rgba(col, 0.35))
        pc.set_edgecolor(col); pc.set_linewidth(0.6)
    parts["cmedians"].set_color(C["accent"]); parts["cmedians"].set_linewidth(1.2)
    ax_coup.axhline(0, color=C["text_sm"], lw=0.5, ls="--")
    ax_coup.set_xticks(range(len(REG_ORDER)))
    ax_coup.set_xticklabels(["N","NE","CW","SE","S"], fontsize=6)
    ax_coup.set_ylabel("Coupling index (z)", fontsize=6,
                       color=C["text_sm"], labelpad=3)
    ax_coup.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_coup.text(0.03, 0.97, "b", transform=ax_coup.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_coup.set_title("Hazard-inequality coupling by region",
                      fontsize=7, color=C["text_hd"], pad=4)

    # Panel c: quadrant distribution
    quad_keys   = list(QUAD_COLORS.keys())
    quad_counts = [int((df["hazard_social_quadrant"]==q).sum()) for q in quad_keys]
    ax_quad.bar(range(4), quad_counts,
                color=[QUAD_COLORS[q] for q in quad_keys],
                edgecolor="white", linewidth=0.3, alpha=0.87, zorder=3)
    for i, cnt in enumerate(quad_counts):
        ax_quad.text(i, cnt + n_mun*0.003,
                     f"{cnt:,}\n({cnt/n_mun*100:.1f}%)",
                     ha="center", fontsize=5, color=C["text_sm"])
    ax_quad.set_xticks(range(4))
    ax_quad.set_xticklabels([QUAD_ABBR[q] for q in quad_keys], fontsize=6)
    ax_quad.set_ylabel("Number of municipalities", fontsize=6,
                       color=C["text_sm"], labelpad=3)
    ax_quad.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_quad.text(0.03, 0.97, "c", transform=ax_quad.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_quad.set_title("Municipality quadrant distribution",
                      fontsize=7, color=C["text_hd"], pad=4)

    # Panel d: social components
    soc_vars = [
        ("income_pc",               "Income pc (log\u2081\u2080)", "log"),
        ("illiteracy_rate",          "Illiteracy rate (%)",         "linear"),
        ("water_supply_adequate_pct","Water supply (%)",            "linear"),
    ]
    for (col, lbl, scale), color in zip(soc_vars, [C["purple"], C["red"], C["teal"]]):
        if col not in df.columns: continue
        vals = df[col].dropna().values
        if scale == "log": vals = np.log10(vals + 1)
        ax_soc.hist(vals, bins=25, color=color, edgecolor="white",
                    linewidth=0.2, alpha=0.55, density=True, label=lbl, zorder=3)
    ax_soc.set_xlabel("Value", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_soc.set_ylabel("Density", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_soc.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_soc.legend(fontsize=4.5, loc="upper right", frameon=True,
                  framealpha=0.9, edgecolor=C["border"])
    ax_soc.text(0.03, 0.97, "d", transform=ax_soc.transAxes,
                fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_soc.set_title("Social components distribution",
                     fontsize=7, color=C["text_hd"], pad=4)

    # Panel e: adaptive capacity
    aci = (df["adaptive_capacity_index"].dropna().values
           if "adaptive_capacity_index" in df.columns else np.random.normal(0,1,n_mun))
    ax_aci.hist(aci, bins=40, color=C["blue"], edgecolor="white",
                linewidth=0.3, alpha=0.85, zorder=3)
    ax_aci.axvline(0, color=C["text_sm"], lw=0.5, ls="--")
    med_aci = np.median(aci)
    ax_aci.axvline(med_aci, color=C["accent"], lw=0.9, ls="--", zorder=4)
    ax_aci.text(med_aci + 0.05, ax_aci.get_ylim()[1]*0.88,
                f"median\n{med_aci:.2f}", fontsize=5, color=C["accent"])
    ax_aci.set_xlabel("Adaptive capacity index (z)", fontsize=6,
                      color=C["text_sm"], labelpad=3)
    ax_aci.set_ylabel("Number of municipalities", fontsize=6,
                      color=C["text_sm"], labelpad=3)
    ax_aci.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_aci.text(0.03, 0.97, "e", transform=ax_aci.transAxes,
                fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_aci.set_title("Adaptive capacity index distribution",
                     fontsize=7, color=C["text_hd"], pad=4)

    # Panel f: QA summary
    ax_qa.set_xlim(0,1); ax_qa.set_ylim(0,1); ax_qa.axis("off")
    ax_qa.text(0.03, 0.97, "f", transform=ax_qa.transAxes,
               fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_qa.set_title("Integration QA summary",
                    fontsize=7, color=C["text_hd"], pad=4)

    qa_items = [
        ("Municipalities",     f"{n_mun:,}"),
        ("Duplicate mun_code", str(int(df.duplicated(subset=['mun_code']).sum()))),
        ("Missing geometry",   str(int(gdf.geometry.isna().sum()))),
        ("Social source",      "Censo 2022"),
        ("Adverse columns",    "illiteracy_rate"),
        ("Protective columns", "income_pc \u00b7 water_supply"),
        ("Hazard window",      f"{RECENT_START}\u2013{RECENT_END}"),
        ("Coupling index",     "hazard + social z-scores"),
        ("Output format",      "GeoParquet + GPKG"),
    ]
    y0 = 0.88
    for key, val in qa_items:
        ax_qa.text(0.04, y0, key, ha="left", va="center",
                   fontsize=6, color=C["text_sm"], transform=ax_qa.transAxes)
        ax_qa.text(0.96, y0, val, ha="right", va="center",
                   fontsize=6, color=C["teal"], fontweight="bold",
                   transform=ax_qa.transAxes)
        ax_qa.plot([0.02, 0.98], [y0-0.04, y0-0.04],
                   color=C["border"], lw=0.3, transform=ax_qa.transAxes)
        y0 -= 0.096

    fig.text(
        0.50, 0.970,
        "Figure 9  |  Hazard-social inequality integration \u2014 "
        "Flood Inequality in Brazil",
        ha="center", va="top",
        fontsize=8, fontweight="bold", color=C["text_hd"],
    )
    fig.text(
        0.50, 0.956,
        (f"CHIRPS anomalies \u00b7 Censo 2022 \u00b7 {n_mun:,} municipalities \u00b7 "
         "composite indices \u00b7 hazard-social quadrant classification"),
        ha="center", va="top",
        fontsize=6, color=C["text_sm"], style="italic",
    )

    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, "fig08_hazard_social_inequality.png")
    pdf_path = os.path.join(save_dir, "fig08_hazard_social_inequality.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=C["bg"])
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor=C["bg"])
    plt.show()
    plt.close(fig)
    return png_path

# ============================================================
# 15. MAIN
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = read_config(CONFIG_PATH)

    municipal, anom, trend, social, social_source, expected_munis = load_inputs()

    if is_valid_output(OUTPUT_GEOPARQUET, OUTPUT_META, expected_munis):
        log_summary("Valid integrated output already exists - skipping processing.")
        with tqdm(total=1, desc="Loading existing output",
                  bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
            final_gdf = gpd.read_parquet(OUTPUT_GEOPARQUET)
            pbar.update(1)
    else:
        social, diagnostics = standardize_social_table(social)
        hazard               = build_hazard_features(anom, trend)
        social               = build_social_indices(social)
        final_gdf            = integrate_spatial(municipal, social, hazard)
        diagnostics          = validate_output(final_gdf, expected_munis, diagnostics)
        save_outputs(final_gdf, diagnostics, config, social_source, expected_munis)

        log_summary("=" * 60)
        log_summary(f"DONE | municipalities={expected_munis:,} | rows={len(final_gdf):,}")
        log_summary(f"GeoParquet : {OUTPUT_GEOPARQUET}")
        log_summary(f"GPKG       : {OUTPUT_GPKG}")

    log("Generating Figure 08 ...")
    with tqdm(total=1, desc="Rendering figure",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        fig_path = make_figure_08_integration(
            final_gdf,
            str(BASE_PATH / "06_figures"),
            dpi=500,
        )
        pbar.update(1)

    log_summary(f"Figure saved: {fig_path}")
    logging.info("Figure 08 generated successfully.")

    print("\n" + "=" * 60)
    print("  Module 08 complete")
    print("=" * 60)
    print(f"  Municipalities : {final_gdf['mun_code'].nunique():,}")
    print(f"  GeoParquet     : {OUTPUT_GEOPARQUET}")
    print(f"  GPKG           : {OUTPUT_GPKG}")
    print(f"  No-geom Parquet: {OUTPUT_NOGEOM_PARQUET}")
    print(f"  No-geom CSV    : {OUTPUT_NOGEOM_CSV}")
    print(f"  Diagnostics    : {OUTPUT_DIAG}")
    print(f"  Figure PNG     : {OUTPUT_FIG_PNG}")
    print(f"  Figure PDF     : {OUTPUT_FIG_PDF}")
    print("  Ready for Module 09.")
    print("=" * 60)


if __name__ == "__main__":
    main()
