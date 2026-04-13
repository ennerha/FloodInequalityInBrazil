"""
Project: Flood Inequality Across Brazil
Module:  02_build_analysis_units_municipal.py

Purpose:
  Build the official municipal analysis units for Brazil using the IBGE
  municipal mesh (2024) and the IBGE Localidades API. Creates the main
  analytical spatial layer for subsequent integration with disaster,
  population, socioeconomic, and environmental datasets.

Why this module matters:
  - Municipalities are the primary analytical unit for nationwide linkage
    with disaster records (S2iD), demographic indicators, and
    administrative statistics.
  - Module 01 produces a computational tile system; this module produces
    the scientific unit of analysis.

Data sources:
  - IBGE national municipal mesh  (geoftp.ibge.gov.br, year 2024)
  - IBGE Localidades API v1       (servicodados.ibge.gov.br)

Outputs:
  - 02_intermediate/analysis_units_municipal_brazil.gpkg
  - 02_intermediate/analysis_units_municipal_brazil.parquet
  - 02_intermediate/analysis_units_municipal_brazil.meta.json
  - 06_figures/fig02_municipal_analysis_units.png   (500 DPI)
  - 06_figures/fig02_municipal_analysis_units.pdf   (vector)
  - 07_logs/02_build_analysis_units_municipal.log

Reproducibility:
  - Idempotent execution  - safe to re-run without side effects
  - No re-download if raw files already exist
  - No reprocessing if valid outputs already exist

Author:  Enner H. de Alcantara
Version: v1.1
"""

# ============================================================
# 1. STANDARD LIBRARY IMPORTS
# ============================================================
import os
import re
import json
import zipfile
import logging
from datetime import datetime
from pathlib import Path

# ============================================================
# 2. THIRD-PARTY IMPORTS
# ============================================================
import numpy as np
import requests
import pandas as pd
import geopandas as gpd

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
LOG_PATH     = os.path.join(BASE_PATH, "07_logs",    "02_build_analysis_units_municipal.log")
CATALOG_PATH = os.path.join(BASE_PATH, "08_catalog", "catalog.csv")

RAW_DIR         = os.path.join(BASE_PATH, "01_raw", "ibge_municipal_mesh")
RAW_ZIP_DIR     = os.path.join(RAW_DIR, "zip")
RAW_EXTRACT_DIR = os.path.join(RAW_DIR, "extracted")
RAW_API_DIR     = os.path.join(RAW_DIR, "api_cache")
OUTPUT_DIR      = os.path.join(BASE_PATH, "02_intermediate")

OUTPUT_GPKG    = os.path.join(OUTPUT_DIR, "analysis_units_municipal_brazil.gpkg")
OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "analysis_units_municipal_brazil.parquet")
OUTPUT_META    = os.path.join(OUTPUT_DIR, "analysis_units_municipal_brazil.meta.json")
OUTPUT_FIG_PNG = os.path.join(BASE_PATH, "06_figures", "fig02_municipal_analysis_units.png")
OUTPUT_FIG_PDF = os.path.join(BASE_PATH, "06_figures", "fig02_municipal_analysis_units.pdf")

LAYER_NAME     = "analysis_units_municipal_brazil"
GEOGRAPHIC_CRS = "EPSG:4326"
AREA_CRS       = "EPSG:5880"

MESH_YEAR = 2024
MESH_URL  = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/"
    "malhas_territoriais/malhas_municipais/"
    f"municipio_{MESH_YEAR}/Brasil/BR_Municipios_{MESH_YEAR}.zip"
)
RAW_ZIP_PATH = os.path.join(RAW_ZIP_DIR, f"BR_Municipios_{MESH_YEAR}.zip")

LOCALIDADES_MUNICIPIOS_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
LOCALIDADES_ESTADOS_URL    = "https://servicodados.ibge.gov.br/api/v1/localidades/estados"
RAW_MUNICIPIOS_JSON = os.path.join(RAW_API_DIR, "ibge_localidades_municipios.json")
RAW_ESTADOS_JSON    = os.path.join(RAW_API_DIR, "ibge_localidades_estados.json")

EXPECTED_MIN_MUNICIPALITIES = 5500

# ============================================================
# 4. DIRECTORY SETUP
# ============================================================
for path in [RAW_DIR, RAW_ZIP_DIR, RAW_EXTRACT_DIR, RAW_API_DIR, OUTPUT_DIR]:
    Path(path).mkdir(parents=True, exist_ok=True)

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
            "mun_code", "mun_name", "uf_code", "uf_name", "uf_sigla",
            "region_name", "region_sigla", "mesh_year",
            "area_km2", "centroid_lon", "centroid_lat", "geometry",
        }
        if gdf.empty or not required.issubset(gdf.columns):
            return False
        if gdf["mun_code"].nunique() < EXPECTED_MIN_MUNICIPALITIES:
            return False
        if gdf[["mun_code", "geometry"]].isna().any().any():
            return False
    except Exception:
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("status") != "completed":
            return False
        if int(meta.get("n_municipalities", 0)) < EXPECTED_MIN_MUNICIPALITIES:
            return False
    except Exception:
        return False
    return True

# ============================================================
# 7. HELPERS
# ============================================================
def download_file(url: str, output_path: str, timeout: int = 300,
                  desc: str = "") -> None:
    if os.path.exists(output_path):
        log(f"File already exists - skipping: {output_path}")
        return
    log(f"Downloading: {url}")
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    tmp   = output_path + ".tmp"
    with open(tmp, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=desc or Path(output_path).name,
        bar_format="{l_bar}{bar:30}{r_bar}",
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    os.replace(tmp, output_path)
    log(f"Download complete: {output_path}")


def download_json(url: str, output_path: str, timeout: int = 300,
                  desc: str = ""):
    if os.path.exists(output_path):
        log(f"JSON cache found - loading: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    log(f"Downloading JSON: {url}")
    with tqdm(total=1, desc=desc or Path(output_path).name,
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        pbar.update(1)
    tmp = output_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_path)
    log(f"JSON cached: {output_path}")
    return data


def extract_zip(zip_path: str, extract_dir: str) -> None:
    marker = os.path.join(extract_dir, f".extracted_{Path(zip_path).stem}.ok")
    if os.path.exists(marker):
        log(f"Already extracted - skipping: {zip_path}")
        return
    log(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        with tqdm(total=len(members), desc="Extracting files",
                  bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
            for member in members:
                zf.extract(member, extract_dir)
                pbar.update(1)
    Path(marker).write_text(datetime.now().isoformat())
    log(f"Extraction complete: {extract_dir}")


def find_vector_file(extract_dir: str) -> str:
    gpkg, shp, geo = [], [], []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            p = os.path.join(root, f); l = f.lower()
            if l.endswith(".gpkg"):              gpkg.append(p)
            elif l.endswith(".shp"):             shp.append(p)
            elif l.endswith((".geojson",".json")): geo.append(p)
    candidates = gpkg + shp + geo
    if not candidates:
        raise FileNotFoundError("No vector file found after extraction.")
    preferred   = [p for p in candidates if "municip" in os.path.basename(p).lower()]
    vector_path = (preferred or candidates)[0]
    log(f"Selected vector file: {vector_path}")
    return vector_path


def find_column(columns, candidates):
    colmap = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in colmap:
            return colmap[c.lower()]
    return None


def normalize_text(value):
    if pd.isna(value): return None
    return str(value).strip()


def normalize_code(value, width=None):
    if pd.isna(value): return None
    text = re.sub(r"\.0$", "", str(value).strip())
    text = re.sub(r"\D", "", text)
    if width and text: text = text.zfill(width)
    return text or None


def flatten_estado(rec: dict) -> dict:
    return {
        "uf_code"     : normalize_code(rec.get("id"), width=2),
        "uf_sigla"    : normalize_text(rec.get("sigla")),
        "uf_name"     : normalize_text(rec.get("nome")),
        "region_code" : normalize_code((rec.get("regiao") or {}).get("id")),
        "region_sigla": normalize_text((rec.get("regiao") or {}).get("sigla")),
        "region_name" : normalize_text((rec.get("regiao") or {}).get("nome")),
    }


def flatten_municipio(rec: dict) -> dict:
    micro  = rec.get("microrregiao") or {}
    meso   = micro.get("mesorregiao") or {}
    uf     = meso.get("UF") or {}
    regiao = uf.get("regiao") or {}
    return {
        "mun_code"         : normalize_code(rec.get("id"), width=7),
        "mun_name_official": normalize_text(rec.get("nome")),
        "uf_code"          : normalize_code(uf.get("id"), width=2),
        "uf_sigla"         : normalize_text(uf.get("sigla")),
        "uf_name"          : normalize_text(uf.get("nome")),
        "region_code"      : normalize_code(regiao.get("id")),
        "region_sigla"     : normalize_text(regiao.get("sigla")),
        "region_name"      : normalize_text(regiao.get("nome")),
        "micro_code"       : normalize_code(micro.get("id")),
        "micro_name"       : normalize_text(micro.get("nome")),
        "meso_code"        : normalize_code(meso.get("id")),
        "meso_name"        : normalize_text(meso.get("nome")),
    }

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
    log("Valid outputs already exist - skipping processing.")
    gdf_final = gpd.read_parquet(OUTPUT_PARQUET)
    log(f"Loaded {len(gdf_final):,} municipalities from existing output.")

else:
    # ----------------------------------------------------------
    # 10. DOWNLOAD RAW DATA
    # ----------------------------------------------------------
    download_file(MESH_URL, RAW_ZIP_PATH, desc="IBGE municipal mesh")
    extract_zip(RAW_ZIP_PATH, RAW_EXTRACT_DIR)

    municipios_json = download_json(
        LOCALIDADES_MUNICIPIOS_URL, RAW_MUNICIPIOS_JSON,
        desc="IBGE Localidades municipios")
    estados_json = download_json(
        LOCALIDADES_ESTADOS_URL, RAW_ESTADOS_JSON,
        desc="IBGE Localidades estados")

    # ----------------------------------------------------------
    # 11. READ VECTOR LAYER
    # ----------------------------------------------------------
    vector_path = find_vector_file(RAW_EXTRACT_DIR)
    log("Reading municipal vector layer ...")
    with tqdm(total=1, desc="Reading vector file",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        gdf_raw = gpd.read_file(vector_path)
        pbar.update(1)

    if gdf_raw.empty:
        raise RuntimeError("Municipal vector layer is empty.")
    log(f"Features loaded: {len(gdf_raw):,}  |  Columns: {list(gdf_raw.columns)}")

    # ----------------------------------------------------------
    # 12. STANDARDIZE CRS
    # ----------------------------------------------------------
    if gdf_raw.crs is None:
        log("No CRS detected - assigning EPSG:4674 (SIRGAS 2000).")
        gdf_raw = gdf_raw.set_crs("EPSG:4674", allow_override=True)

    with tqdm(total=1, desc="Reprojecting to EPSG:4326",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        gdf = gdf_raw.to_crs(GEOGRAPHIC_CRS).copy()
        pbar.update(1)

    # ----------------------------------------------------------
    # 13. DETECT ATTRIBUTE COLUMNS
    # ----------------------------------------------------------
    col_mun_code = find_column(gdf.columns,
        ["CD_MUN","CD_GEOCMU","GEOCODIGO","CODIGO","ID","code_muni"])
    col_mun_name = find_column(gdf.columns,
        ["NM_MUN","NM_MUNICIP","NM_MUNICIPIO","NOME","name"])
    col_uf_code  = find_column(gdf.columns, ["CD_UF","UF","CODUF"])
    col_uf_sigla = find_column(gdf.columns, ["SIGLA_UF","UF_SIGLA"])
    col_uf_name  = find_column(gdf.columns, ["NM_UF","NOME_UF"])

    if col_mun_code is None:
        raise RuntimeError("Municipality code column not found in mesh.")

    # ----------------------------------------------------------
    # 14. BUILD STANDARDIZED GEOMETRY TABLE
    # ----------------------------------------------------------
    log("Standardizing geometry attributes ...")
    gdf_std = gdf.copy()
    gdf_std["mun_code"]      = gdf_std[col_mun_code].apply(lambda x: normalize_code(x, 7))
    gdf_std["mun_name_mesh"] = gdf_std[col_mun_name].apply(normalize_text) if col_mun_name else None
    gdf_std["uf_code_mesh"]  = (gdf_std[col_uf_code].apply(lambda x: normalize_code(x, 2))
                                 if col_uf_code else gdf_std["mun_code"].str[:2])
    gdf_std["uf_sigla_mesh"] = gdf_std[col_uf_sigla].apply(normalize_text) if col_uf_sigla else None
    gdf_std["uf_name_mesh"]  = gdf_std[col_uf_name].apply(normalize_text)  if col_uf_name  else None

    gdf_std = (gdf_std[["mun_code","mun_name_mesh","uf_code_mesh",
                         "uf_sigla_mesh","uf_name_mesh","geometry"]]
               .dropna(subset=["mun_code"])
               .drop_duplicates(subset=["mun_code"])
               .reset_index(drop=True))

    if gdf_std["mun_code"].nunique() < EXPECTED_MIN_MUNICIPALITIES:
        raise RuntimeError(
            f"Too few municipalities in mesh: {gdf_std['mun_code'].nunique()}")
    log(f"Standardized geometries: {gdf_std['mun_code'].nunique():,}")

    # ----------------------------------------------------------
    # 15. FLATTEN LOCALIDADES TABLES
    # ----------------------------------------------------------
    log("Processing IBGE Localidades data ...")
    with tqdm(total=2, desc="Flattening API records",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        df_mun = pd.DataFrame([flatten_municipio(r) for r in municipios_json])
        pbar.update(1)
        df_uf  = pd.DataFrame([flatten_estado(r) for r in estados_json])
        pbar.update(1)

    df_mun = df_mun.merge(df_uf, how="left", on="uf_code", suffixes=("","_uf"))
    for col in ["uf_sigla","uf_name","region_code","region_sigla","region_name"]:
        fb = col + "_uf"
        if fb in df_mun.columns:
            df_mun[col] = df_mun[col].fillna(df_mun[fb])
    df_mun = df_mun.drop(columns=[c for c in df_mun.columns if c.endswith("_uf")])
    df_mun = df_mun.drop_duplicates(subset=["mun_code"]).reset_index(drop=True)

    if df_mun["mun_code"].nunique() < EXPECTED_MIN_MUNICIPALITIES:
        raise RuntimeError(
            f"Too few municipalities in Localidades API: {df_mun['mun_code'].nunique()}")
    log(f"Official municipality records: {df_mun['mun_code'].nunique():,}")

    # ----------------------------------------------------------
    # 16. MERGE GEOMETRY WITH OFFICIAL ATTRIBUTES
    # ----------------------------------------------------------
    log("Merging geometry with official attributes ...")
    with tqdm(total=1, desc="Merging tables",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        gdf_final = gdf_std.merge(df_mun, how="left", on="mun_code", validate="1:1")
        pbar.update(1)

    gdf_final["mun_name"]  = gdf_final["mun_name_official"].fillna(gdf_final["mun_name_mesh"])
    gdf_final["uf_code"]   = gdf_final["uf_code"].fillna(gdf_final["uf_code_mesh"])
    gdf_final["uf_sigla"]  = gdf_final["uf_sigla"].fillna(gdf_final["uf_sigla_mesh"])
    gdf_final["uf_name"]   = gdf_final["uf_name"].fillna(gdf_final["uf_name_mesh"])
    gdf_final["mesh_year"] = MESH_YEAR

    missing = gdf_final["mun_name"].isna().sum()
    if missing:
        log(f"Warning: {missing} municipalities with missing names after merge.")

    # ----------------------------------------------------------
    # 17. AREA AND CENTROIDS
    # ----------------------------------------------------------
    log("Computing area (EPSG:5880) and centroids ...")
    with tqdm(total=2, desc="Geometric attributes",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        gdf_proj = gdf_final.to_crs(AREA_CRS)
        gdf_final["area_km2"] = gdf_proj.geometry.area / 1_000_000.0
        pbar.update(1)

        gdf_ctr = gdf_proj.copy()
        gdf_ctr["geometry"] = gdf_ctr.geometry.centroid
        gdf_ctr = gdf_ctr.to_crs(GEOGRAPHIC_CRS)
        gdf_final["centroid_lon"] = gdf_ctr.geometry.x
        gdf_final["centroid_lat"] = gdf_ctr.geometry.y
        pbar.update(1)

    # ----------------------------------------------------------
    # 18. FINAL COLUMN ORDER AND QUALITY CHECKS
    # ----------------------------------------------------------
    desired_cols = [
        "mun_code","mun_name",
        "uf_code","uf_name","uf_sigla",
        "region_code","region_sigla","region_name",
        "meso_code","meso_name","micro_code","micro_name",
        "mesh_year","area_km2","centroid_lon","centroid_lat","geometry",
    ]
    for col in desired_cols:
        if col not in gdf_final.columns:
            gdf_final[col] = None

    gdf_final = gpd.GeoDataFrame(
        gdf_final[desired_cols], geometry="geometry", crs=GEOGRAPHIC_CRS
    ).sort_values(["uf_sigla","mun_name"]).reset_index(drop=True)

    n_mun = gdf_final["mun_code"].nunique()
    if n_mun < EXPECTED_MIN_MUNICIPALITIES:
        raise RuntimeError(f"Final layer has too few municipalities: {n_mun}")
    if gdf_final.geometry.isna().any():
        raise RuntimeError("Final layer contains missing geometries.")
    if not gdf_final.geometry.is_valid.all():
        log("Warning: invalid geometries - applying buffer(0) repair.")
        gdf_final["geometry"] = gdf_final.geometry.buffer(0)

    log(f"Final municipalities: {n_mun:,}")

    # ----------------------------------------------------------
    # 19. SAVE OUTPUTS  (atomic via temp files)
    # ----------------------------------------------------------
    log("Saving outputs ...")
    tmp_gpkg    = OUTPUT_GPKG    + ".tmp"
    tmp_parquet = OUTPUT_PARQUET + ".tmp"
    tmp_meta    = OUTPUT_META    + ".tmp"

    for p in [tmp_gpkg, tmp_parquet, tmp_meta]:
        if os.path.exists(p): os.remove(p)

    with tqdm(total=3, desc="Writing files",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        gdf_final.to_file(tmp_gpkg, layer=LAYER_NAME, driver="GPKG"); pbar.update(1)
        gdf_final.to_parquet(tmp_parquet, index=False);                pbar.update(1)
        meta = {
            "project"                   : "Flood Inequality Across Brazil",
            "module"                    : "02_build_analysis_units_municipal.py",
            "status"                    : "completed",
            "created_at"                : datetime.now().isoformat(),
            "base_path"                 : BASE_PATH,
            "mesh_year"                 : MESH_YEAR,
            "mesh_url"                  : MESH_URL,
            "n_municipalities"          : int(n_mun),
            "crs"                       : GEOGRAPHIC_CRS,
            "area_crs"                  : AREA_CRS,
            "area_unit"                 : "km2",
            "raw_zip_path"              : RAW_ZIP_PATH,
            "vector_source_path"        : vector_path,
            "localidades_municipios_url": LOCALIDADES_MUNICIPIOS_URL,
            "localidades_estados_url"   : LOCALIDADES_ESTADOS_URL,
            "output_gpkg"               : OUTPUT_GPKG,
            "output_parquet"            : OUTPUT_PARQUET,
            "columns"                   : desired_cols,
        }
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=4)
        pbar.update(1)

    os.replace(tmp_gpkg,    OUTPUT_GPKG)
    os.replace(tmp_parquet, OUTPUT_PARQUET)
    os.replace(tmp_meta,    OUTPUT_META)
    log("All outputs saved.")

    # ----------------------------------------------------------
    # 20. UPDATE CONFIG
    # ----------------------------------------------------------
    config["analysis_units"] = {
        "name"        : "brazil_municipal_analysis_units",
        "source"      : "IBGE municipal mesh + IBGE Localidades API",
        "mesh_year"   : MESH_YEAR,
        "n_units"     : int(n_mun),
        "path_parquet": OUTPUT_PARQUET,
        "path_gpkg"   : OUTPUT_GPKG,
        "crs"         : GEOGRAPHIC_CRS,
        "area_crs"    : AREA_CRS,
        "area_unit"   : "km2",
        "key_field"   : "mun_code",
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    log("config.json updated.")

    # ----------------------------------------------------------
    # 21. UPDATE CATALOG
    # ----------------------------------------------------------
    row = pd.DataFrame([{
        "stage"      : "02_build_analysis_units_municipal",
        "tile_id"    : "ALL",
        "period"     : "NA",
        "status"     : "completed",
        "output_path": OUTPUT_PARQUET,
        "timestamp"  : datetime.now().isoformat(),
    }])
    if os.path.exists(CATALOG_PATH):
        try:
            cat = pd.read_csv(CATALOG_PATH)
            cat = cat[~((cat["stage"] == "02_build_analysis_units_municipal") &
                        (cat["tile_id"] == "ALL"))]
            pd.concat([cat, row], ignore_index=True).to_csv(CATALOG_PATH, index=False)
        except Exception:
            row.to_csv(CATALOG_PATH, index=False)
    else:
        row.to_csv(CATALOG_PATH, index=False)
    log("Catalog updated.")

# ============================================================
# 22. FIGURE 02 — MUNICIPAL ANALYSIS UNITS  (500 DPI composite)
# ============================================================

def make_figure_02_municipal(gdf: gpd.GeoDataFrame,
                              save_dir: str,
                              dpi: int = 500) -> str:
    """
    4-panel composite figure:
      a) Map of municipal units coloured by macro-region
      b) Municipal area distribution (log-scale histogram)
      c) Municipality count and median area by macro-region
      d) Pipeline progress tracker + data source annotation
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
        "gray"   : "#6B7280",
    }

    REGION_COLORS = {
        "North"       : "#4A90D9",
        "Northeast"   : "#E8A838",
        "Center-West" : "#6DB56D",
        "Southeast"   : "#C0504D",
        "South"       : "#9B59B6",
    }
    region_order = ["North", "Northeast", "Center-West", "Southeast", "South"]
    areas = gdf["area_km2"].values

    fig = plt.figure(figsize=(7.087, 7.8))
    fig.patch.set_facecolor(C["bg"])

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.09, right=0.97,
        top=0.91,  bottom=0.07,
        hspace=0.42, wspace=0.32,
    )
    ax_map  = fig.add_subplot(gs[0, 0])
    ax_area = fig.add_subplot(gs[0, 1])
    ax_reg  = fig.add_subplot(gs[1, 0])
    ax_prog = fig.add_subplot(gs[1, 1])

    for ax in (ax_map, ax_area, ax_reg, ax_prog):
        ax.set_facecolor(C["bg"])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color(C["border"])

    # Panel a: map
    for region, col in REGION_COLORS.items():
        sub = gdf[gdf["region_name"] == region]
        if not sub.empty:
            sub.plot(ax=ax_map, facecolor=col, edgecolor="white",
                     linewidth=0.10, alpha=0.82)

    bounds = gdf.total_bounds
    pad = 0.5
    ax_map.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax_map.set_ylim(bounds[1] - pad, bounds[3] + pad)
    ax_map.set_xlabel("Longitude (°)", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_map.set_ylabel("Latitude (°)",  fontsize=6, color=C["text_sm"], labelpad=3)
    ax_map.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax_map.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax_map.tick_params(width=0.4)
    ax_map.grid(True, linewidth=0.18, color=C["border"], alpha=0.6, zorder=0)
    ax_map.axhline(0, color=C["accent"], lw=0.6, ls="--", alpha=0.8)
    ax_map.text(bounds[0] + 0.5, 0.4, "Equator", fontsize=4.5, color=C["accent"])

    leg = [mpatches.Patch(facecolor=c, edgecolor="none", label=r)
           for r, c in REGION_COLORS.items()]
    ax_map.legend(handles=leg, fontsize=4.5, loc="lower left",
                  frameon=True, framealpha=0.9, edgecolor=C["border"],
                  handlelength=1.0, handleheight=0.8)
    ax_map.text(0.97, 0.03, f"n = {len(gdf):,}",
                transform=ax_map.transAxes, ha="right", va="bottom",
                fontsize=5.5, color=C["blue"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=C["border"], lw=0.5, alpha=0.9))
    ax_map.text(0.03, 0.97, "a", transform=ax_map.transAxes,
                fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_map.set_title("Municipal units by macro-region",
                     fontsize=7, color=C["text_hd"], pad=4)

    # Panel b: area histogram (log scale)
    log_areas = np.log10(areas + 1)
    n_bins    = min(28, max(12, len(gdf) // 250))
    ax_area.hist(log_areas, bins=n_bins, color=C["blue"],
                 edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)

    mu_log = np.median(log_areas)
    ymax   = ax_area.get_ylim()[1]
    ax_area.axvline(mu_log, color=C["accent"], lw=0.9, ls="--", zorder=4)
    ax_area.text(mu_log + 0.05, ymax * 0.82,
                 f"median\n{10**mu_log:,.0f} km²",
                 fontsize=5, color=C["accent"], va="top")

    for pct, ls in [(10, "dotted"), (90, "dotted")]:
        v = np.percentile(log_areas, pct)
        ax_area.axvline(v, color=C["gray"], lw=0.5, ls=ls, zorder=3)
        ax_area.text(v + 0.02, ymax * 0.45, f"P{pct}",
                     fontsize=4.5, color=C["gray"])

    stats_txt = (f"n      {len(areas):,}\n"
                 f"min    {np.min(areas):,.0f}\n"
                 f"max    {np.max(areas):,.0f}\n"
                 f"sd     {np.std(areas):,.0f}")
    ax_area.text(0.97, 0.97, stats_txt, transform=ax_area.transAxes,
                 ha="right", va="top", fontsize=5, family="monospace",
                 color=C["text_sm"],
                 bbox=dict(boxstyle="round,pad=0.4", fc="white",
                           ec=C["border"], lw=0.5, alpha=0.9))
    ax_area.set_xlabel("Municipal area  log\u2081\u2080(km\u00b2)", fontsize=6,
                       color=C["text_sm"], labelpad=3)
    ax_area.set_ylabel("Number of municipalities", fontsize=6,
                       color=C["text_sm"], labelpad=3)
    ax_area.tick_params(width=0.4)
    ax_area.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_area.text(0.03, 0.97, "b", transform=ax_area.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_area.set_title("Municipal area distribution",
                      fontsize=7, color=C["text_hd"], pad=4)

    # Panel c: count + median area per region
    r_counts = gdf["region_name"].value_counts()
    counts   = [r_counts.get(r, 0) for r in region_order]
    r_colors = [REGION_COLORS[r] for r in region_order]

    ax_reg.barh(range(len(region_order)), counts, height=0.6,
                color=r_colors, edgecolor="white", linewidth=0.25,
                alpha=0.88, zorder=3)

    x_max = max(counts) if counts else 1
    for i, (cnt, reg) in enumerate(zip(counts, region_order)):
        ax_reg.text(cnt + x_max * 0.01, i, f"{cnt:,}",
                    va="center", fontsize=5.5, color=C["text_sm"])
        sub_a = gdf.loc[gdf["region_name"] == reg, "area_km2"]
        if len(sub_a):
            ax_reg.text(x_max * 1.06, i,
                        f"med\n{np.median(sub_a):,.0f} km²",
                        va="center", ha="left", fontsize=4.5, color=C["gray"])

    ax_reg.set_yticks(range(len(region_order)))
    ax_reg.set_yticklabels(region_order, fontsize=6)
    ax_reg.set_xlabel("Number of municipalities", fontsize=6,
                      color=C["text_sm"], labelpad=3)
    ax_reg.set_xlim(0, x_max * 1.30)
    ax_reg.tick_params(width=0.4)
    ax_reg.grid(axis="x", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_reg.text(0.03, 0.97, "c", transform=ax_reg.transAxes,
                fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_reg.set_title("Municipalities per macro-region",
                     fontsize=7, color=C["text_hd"], pad=4)

    # Panel d: pipeline progress
    ax_prog.set_xlim(0, 1); ax_prog.set_ylim(0, 1)
    ax_prog.axis("off")
    ax_prog.text(0.03, 0.97, "d", transform=ax_prog.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_prog.set_title("Pipeline progress", fontsize=7, color=C["text_hd"], pad=4)

    pipeline = [
        ("00 · Environment setup",   1.00, C["teal"]),
        ("01 · Processing tiles",    1.00, C["teal"]),
        ("02 · Municipal units",     1.00, C["teal"]),
        ("03 · Feature engineering", 0.00, C["gray"]),
        ("04 · Data integration",    0.00, C["gray"]),
        ("05 · Modeling",            0.00, C["gray"]),
        ("06 · Figures",             0.00, C["gray"]),
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
                     f"{int(frac*100)}%",
                     ha="left", va="center", fontsize=5.5,
                     color=col if frac > 0 else C["gray"],
                     fontweight="bold" if frac > 0 else "normal",
                     transform=ax_prog.transAxes)
        y0 -= 0.120

    src = ("Data sources\n"
           f"IBGE municipal mesh  {MESH_YEAR}\n"
           "IBGE Localidades API  v1\n"
           "Area: EPSG:5880 (Brazil Albers)")
    ax_prog.text(0.50, 0.06, src, ha="center", va="bottom",
                 fontsize=5, color=C["text_sm"],
                 transform=ax_prog.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", fc=C["panel"],
                           ec=C["border"], lw=0.5))

    for xi, (fc, lbl) in enumerate([(C["teal"],"Completed"),
                                     (C["gray"],"Pending")]):
        bx = 0.30 + xi * 0.35
        ax_prog.add_patch(mpatches.Rectangle(
            (bx, 0.02), 0.025, 0.045, facecolor=fc, edgecolor="none",
            transform=ax_prog.transAxes, clip_on=False, zorder=5))
        ax_prog.text(bx + 0.035, 0.043, lbl, fontsize=5.5, va="center",
                     color=C["text_sm"], transform=ax_prog.transAxes)

    fig.text(
        0.50, 0.966,
        "Figure 3  |  Municipal analysis units — Flood Inequality in Brazil",
        ha="center", va="top",
        fontsize=8, fontweight="bold", color=C["text_hd"],
    )
    fig.text(
        0.50, 0.952,
        (f"IBGE municipal mesh {MESH_YEAR} · IBGE Localidades API v1 · "
         f"EPSG:4326 geometry · area in EPSG:5880 (Brazil Albers Equal Area)"),
        ha="center", va="top",
        fontsize=6, color=C["text_sm"], style="italic",
    )

    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, "fig02_municipal_analysis_units.png")
    pdf_path = os.path.join(save_dir, "fig02_municipal_analysis_units.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=C["bg"])
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor=C["bg"])
    plt.show()
    plt.close(fig)
    return png_path


# ============================================================
# RUN
# ============================================================
log("Generating Figure 02 ...")
with tqdm(total=1, desc="Rendering figure",
          bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
    fig_path = make_figure_02_municipal(
        gdf_final,
        os.path.join(BASE_PATH, "06_figures"),
        dpi=500,
    )
    pbar.update(1)

log(f"Figure saved: {fig_path}")
logging.info("Figure 02 generated successfully.")

print("\n" + "=" * 60)
print("  Module 02 complete")
print("=" * 60)
print(f"  Municipalities : {len(gdf_final):,}")
print(f"  GeoPackage     : {OUTPUT_GPKG}")
print(f"  GeoParquet     : {OUTPUT_PARQUET}")
print(f"  Metadata       : {OUTPUT_META}")
print(f"  Figure PNG     : {OUTPUT_FIG_PNG}")
print(f"  Figure PDF     : {OUTPUT_FIG_PDF}")
print("  Ready for Module 03.")
print("=" * 60)
