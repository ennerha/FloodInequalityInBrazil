"""
Project: Flood Inequality Across Brazil
Module:  09_build_disaster_s2id.py
Version: v4.3

Purpose:
  Read locally available annual S2ID Excel files (.xls/.xlsx), standardize
  them, link municipalities to official IBGE codes using municipality name +
  UF when IBGE code is absent, and build a municipality-year disaster panel.

Search location:
  /content/drive/MyDrive/Brazil  (recursive)

Changelog v4.3 (from v4.2):
  - tqdm progress bars added to all major steps
  - Figure 09 (6-panel composite, 500 DPI) integrated
  - logging unified with logging.basicConfig pattern
  - setup_logger() replaced by logging.basicConfig

Outputs:
  - 04_integrated/s2id_raw_concat_brazil.parquet
  - 04_integrated/s2id_municipal_annual_brazil.parquet
  - 04_integrated/s2id_municipal_annual_brazil.csv
  - 04_integrated/s2id_municipal_annual_brazil.meta.json
  - 04_integrated/s2id_input_diagnostics.csv
  - 06_figures/fig09_s2id_disaster_panel.png   (500 DPI)
  - 06_figures/fig09_s2id_disaster_panel.pdf   (vector)
  - 07_logs/09_build_disaster_s2id.log

Reproducibility:
  - Idempotent execution  - safe to re-run without side effects
  - Atomic save (tmp -> final)

Author:  Enner H. de Alcantara
"""

# ============================================================
# 1. STANDARD LIBRARY IMPORTS
# ============================================================
import os
import re
import sys
import json
import unicodedata
import logging
import subprocess
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
from matplotlib import gridspec

from tqdm.auto import tqdm

# ============================================================
# 3. OPTIONAL EXCEL ENGINES
# ============================================================
def ensure_excel_engines() -> None:
    needed = []
    try:
        import xlrd  # noqa
    except Exception:
        needed.append("xlrd>=2.0.1")
    try:
        import openpyxl  # noqa
    except Exception:
        needed.append("openpyxl")
    if needed:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + needed)

ensure_excel_engines()

# ============================================================
# 4. PATHS AND CONSTANTS
# ============================================================
BASE_PATH         = Path("/content/drive/MyDrive/Brazil/flood_inequality_project")
DRIVE_BRAZIL_PATH = Path("/content/drive/MyDrive/Brazil")

CONFIG_PATH  = BASE_PATH / "00_config"  / "config.json"
LOG_PATH     = BASE_PATH / "07_logs"    / "09_build_disaster_s2id.log"
CATALOG_PATH = BASE_PATH / "08_catalog" / "catalog.csv"
MUNICIPAL_PATH = BASE_PATH / "02_intermediate" / "analysis_units_municipal_brazil.parquet"

OUTPUT_DIR           = BASE_PATH / "04_integrated"
RAW_CONCAT_PARQUET   = OUTPUT_DIR / "s2id_raw_concat_brazil.parquet"
OUTPUT_PANEL_PARQUET = OUTPUT_DIR / "s2id_municipal_annual_brazil.parquet"
OUTPUT_PANEL_CSV     = OUTPUT_DIR / "s2id_municipal_annual_brazil.csv"
OUTPUT_META          = OUTPUT_DIR / "s2id_municipal_annual_brazil.meta.json"
OUTPUT_DIAG          = OUTPUT_DIR / "s2id_input_diagnostics.csv"
OUTPUT_FIG_PNG       = BASE_PATH  / "06_figures" / "fig09_s2id_disaster_panel.png"
OUTPUT_FIG_PDF       = BASE_PATH  / "06_figures" / "fig09_s2id_disaster_panel.pdf"

START_YEAR       = 2013
END_YEAR         = 2022
SEARCH_DIRS      = [DRIVE_BRAZIL_PATH]
ALLOWED_SUFFIXES = {".xls", ".xlsx"}
VERBOSE          = False

# ============================================================
# 5. DIRECTORY SETUP + LOGGING
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
# 6. COLUMN ALIASES
# ============================================================
MUN_CODE_ALIASES = [
    "cod_ibge_mun","codigo_ibge","cod_municipio","codigo_municipio",
    "cd_mun","mun_code","ibge","municipio_ibge",
]
MUN_NAME_ALIASES    = ["municipio","nome_municipio","nm_mun","mun_name"]
UF_ALIASES          = ["uf","sigla_uf","estado"]
DATE_ALIASES        = ["data_ocorrencia","dt_ocorrencia","data","data_desastre","data_registro"]
COBRADE_ALIASES     = ["cobrade","cod_cobrade","codigo_cobrade"]
DISASTER_NAME_ALIASES = ["desastre","tipo_desastre","descricao_tipologia","tipologia"]
PEOPLE_AFFECTED_ALIASES = [
    "afetados","pessoas_afetadas","total_afetados",
    "dh_outros afetados","dh_outros_afetados",
]
HOMELESS_ALIASES    = ["desabrigados","pessoas_desabrigadas","dh_desabrigados"]
DISPLACED_ALIASES   = ["desalojados","pessoas_desalojadas","dh_desalojados"]
DEATHS_ALIASES      = ["obitos","mortos","nr_obitos","dh_mortos"]

# ============================================================
# 7. GENERIC HELPERS
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


def parquet_schema_columns(path: Path) -> list:
    return pq.read_schema(str(path)).names


def is_valid_output(panel_path: Path, meta_path: Path,
                    expected_munis: int) -> bool:
    if not panel_path.exists() or not meta_path.exists():
        return False
    try:
        cols = set(parquet_schema_columns(panel_path))
        required = {
            "mun_code","year",
            "s2id_event_records_n",
            "s2id_hydrological_records_n",
            "s2id_flood_like_records_n",
        }
        if not required.issubset(cols): return False
        if pq.read_metadata(str(panel_path)).num_rows < expected_munis: return False
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("status") == "completed"
    except Exception:
        return False


def normalize_code(x, width: int = 7):
    if pd.isna(x): return None
    txt = re.sub(r"\.0$", "", str(x).strip())
    txt = re.sub(r"\D", "", txt)
    return txt.zfill(width) if txt else None


def lower_map(columns: list) -> dict:
    return {str(c).strip().lower(): c for c in columns}


def find_col(columns: list, aliases: list):
    cmap = lower_map(columns)
    for alias in aliases:
        if alias.lower() in cmap:
            return cmap[alias.lower()]
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    s = (series.astype(str).str.strip()
         .str.replace(".", "", regex=False)
         .str.replace(",", ".", regex=False)
         .replace({"-": np.nan, "nan": np.nan, "None": np.nan, "": np.nan}))
    return pd.to_numeric(s, errors="coerce")


def normalize_text(x) -> str:
    if pd.isna(x): return ""
    txt = unicodedata.normalize("NFKD", str(x).strip().upper())
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = re.sub(r"['`´^~]", "", txt)
    txt = re.sub(r"[^A-Z0-9 ]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

# ============================================================
# 8. MUNICIPAL REFERENCE
# ============================================================
def load_municipal_reference() -> tuple:
    if not MUNICIPAL_PATH.exists():
        raise FileNotFoundError(f"Municipal base not found: {MUNICIPAL_PATH}")

    with tqdm(total=1, desc="Loading municipal reference",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        municipal = gpd.read_parquet(MUNICIPAL_PATH)
        pbar.update(1)

    municipal["mun_code"] = municipal["mun_code"].astype(str)
    municipal["mun_name"] = municipal["mun_name"].astype(str)
    municipal["uf_sigla"] = municipal["uf_sigla"].astype(str)

    ref = municipal[["mun_code","mun_name","uf_sigla"]].copy()
    ref["mun_name_norm"] = ref["mun_name"].apply(normalize_text)
    ref["uf_sigla_norm"] = ref["uf_sigla"].str.upper().str.strip()
    return municipal, ref

# ============================================================
# 9. FILE DISCOVERY
# ============================================================
def discover_local_files() -> list:
    log("Scanning for local S2ID Excel files ...")
    files = []

    with tqdm(total=1, desc="Discovering files",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for directory in SEARCH_DIRS:
            if not directory.exists(): continue
            for p in sorted(directory.rglob("*")):
                if not p.is_file(): continue
                if p.suffix.lower() not in ALLOWED_SUFFIXES: continue
                m = re.search(r"(20\d{2})", p.name)
                if not m: continue
                year = int(m.group(1))
                if START_YEAR <= year <= END_YEAR:
                    files.append({"year": year, "path": p,
                                  "name": p.name, "suffix": p.suffix.lower()})
        pbar.update(1)

    if not files:
        raise RuntimeError(
            f"No S2ID Excel files found in: {SEARCH_DIRS}")

    dedup = {}
    for item in sorted(files, key=lambda x: (x["year"], str(x["path"]))):
        if item["year"] not in dedup:
            dedup[item["year"]] = item

    result  = [dedup[y] for y in sorted(dedup)]
    missing = [y for y in range(START_YEAR, END_YEAR + 1)
               if y not in [r["year"] for r in result]]
    if missing:
        log(f"Missing years in local S2ID files: {missing}", "WARNING")

    log_summary(f"Found {len(result)} S2ID files: "
                f"{result[0]['year']}\u2013{result[-1]['year']}")
    return result

# ============================================================
# 10. FILE READER
# ============================================================
def read_excel_flexible(path: Path) -> pd.DataFrame:
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    xls    = pd.ExcelFile(path, engine=engine)
    best_df, best_score = None, -1

    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet)
        except Exception:
            continue
        cols  = [str(c).lower() for c in df.columns]
        score = sum(
            1 for tok in ["mun","ibge","data","desastre","cobrade",
                           "afet","obito","desabrig","desaloj"]
            if any(tok in c for c in cols))
        if len(df) > 0 and score > best_score:
            best_score, best_df = score, df.copy()

    if best_df is None or best_df.empty:
        raise RuntimeError(f"No usable sheet in: {path.name}")
    return best_df

# ============================================================
# 11. STANDARDIZATION
# ============================================================
def standardize_one_raw(df_raw: pd.DataFrame, source_year: int,
                         municipal_ref: pd.DataFrame) -> tuple:
    cols = list(df_raw.columns)

    mun_code_col     = find_col(cols, MUN_CODE_ALIASES)
    mun_name_col     = find_col(cols, MUN_NAME_ALIASES)
    uf_col           = find_col(cols, UF_ALIASES)
    date_col         = find_col(cols, DATE_ALIASES)
    cobrade_col      = find_col(cols, COBRADE_ALIASES)
    disaster_name_col = find_col(cols, DISASTER_NAME_ALIASES)

    out = pd.DataFrame()

    if mun_code_col is not None:
        out["mun_code"] = df_raw[mun_code_col].apply(normalize_code)
    else:
        if mun_name_col is None or uf_col is None:
            raise RuntimeError(
                f"Cannot detect municipality identifier. Columns: {cols}")
        tmp = pd.DataFrame()
        tmp["mun_name_norm"] = df_raw[mun_name_col].astype(str).str.strip().apply(normalize_text)
        tmp["uf_sigla_norm"] = df_raw[uf_col].astype(str).str.strip().str.upper()
        tmp = tmp.merge(
            municipal_ref[["mun_code","mun_name_norm","uf_sigla_norm"]],
            on=["mun_name_norm","uf_sigla_norm"], how="left")
        out["mun_code"] = tmp["mun_code"]

    out["mun_name_s2id"] = df_raw[mun_name_col].astype(str).str.strip() if mun_name_col else np.nan
    out["uf_sigla_s2id"] = df_raw[uf_col].astype(str).str.strip()       if uf_col       else np.nan

    if date_col is not None:
        parsed = pd.to_datetime(df_raw[date_col], errors="coerce", dayfirst=True)
        out["event_date"] = parsed
        out["year"]       = parsed.dt.year.fillna(source_year).astype("Int64")
    else:
        out["event_date"] = pd.NaT
        out["year"]       = source_year

    out["cobrade"]       = df_raw[cobrade_col].astype(str).str.strip()       if cobrade_col       else np.nan
    out["disaster_name"] = df_raw[disaster_name_col].astype(str).str.strip() if disaster_name_col else np.nan

    def _num(aliases):
        col = find_col(cols, aliases)
        return coerce_numeric(df_raw[col]) if col else pd.Series(np.nan, index=df_raw.index)

    out["people_affected"] = _num(PEOPLE_AFFECTED_ALIASES)
    out["homeless"]        = _num(HOMELESS_ALIASES)
    out["displaced"]       = _num(DISPLACED_ALIASES)
    out["deaths"]          = _num(DEATHS_ALIASES)
    out["source_year"]     = source_year

    total   = len(out)
    matched = int(out["mun_code"].notna().sum())
    out     = out.dropna(subset=["mun_code"]).copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")

    diag = {
        "rows_input"                  : int(total),
        "rows_matched_municipality"   : matched,
        "rows_unmatched_municipality" : int(total - matched),
        "columns_detected"            : ", ".join(cols),
    }
    return out, diag


def classify_hydrological(row: pd.Series) -> bool:
    text = " ".join([str(row.get("cobrade","")),
                     str(row.get("disaster_name",""))]).lower()
    return any(k in text for k in
               ["hidrol","inund","enxurr","alag","chuva",
                "corrida","tromba","eros"])


def classify_flood_like(row: pd.Series) -> bool:
    text = " ".join([str(row.get("cobrade","")),
                     str(row.get("disaster_name",""))]).lower()
    return any(k in text for k in ["inund","enxurr","alag"])

# ============================================================
# 12. BUILD RAW CONCATENATED TABLE
# ============================================================
def build_raw_concat(resources: list,
                     municipal_ref: pd.DataFrame) -> tuple:
    diagnostics = []
    frames      = []

    with tqdm(total=len(resources), desc="Reading S2ID files",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for r in resources:
            try:
                df_raw = read_excel_flexible(r["path"])
                std, diag = standardize_one_raw(df_raw, r["year"], municipal_ref)
                std["resource_name"]   = r["name"]
                std["resource_path"]   = str(r["path"])
                std["resource_suffix"] = r["suffix"]
                frames.append(std)
                diagnostics += [
                    {"check": f"resource_{r['year']}_rows_standardized",
                     "value": int(len(std)), "details": r["name"]},
                    {"check": f"resource_{r['year']}_rows_input",
                     "value": diag["rows_input"], "details": r["name"]},
                    {"check": f"resource_{r['year']}_rows_unmatched",
                     "value": diag["rows_unmatched_municipality"], "details": r["name"]},
                ]
            except Exception as exc:
                log(f"{r['name']}: FAILED - {exc}", "ERROR")
                diagnostics.append({
                    "check": f"resource_{r['year']}_read_error",
                    "value": -1, "details": str(exc),
                })
            pbar.update(1)

    if not frames:
        raise RuntimeError("No S2ID files successfully standardized.")

    with tqdm(total=2, desc="Classifying records",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        raw = pd.concat(frames, ignore_index=True)
        pbar.update(1)
        raw["is_hydrological"] = raw.apply(classify_hydrological, axis=1).astype(int)
        raw["is_flood_like"]   = raw.apply(classify_flood_like,   axis=1).astype(int)
        pbar.update(1)

    return raw, diagnostics

# ============================================================
# 13. BUILD MUNICIPALITY-YEAR PANEL
# ============================================================
def build_municipal_annual_panel(raw: pd.DataFrame,
                                  municipal_codes: pd.Series) -> pd.DataFrame:
    if raw["year"].isna().all():
        raise RuntimeError("All S2ID records have missing year.")

    with tqdm(total=3, desc="Building panel",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        grouped = raw.groupby(["mun_code","year"], as_index=False).agg(
            s2id_event_records_n       =("mun_code",        "size"),
            s2id_hydrological_records_n=("is_hydrological", "sum"),
            s2id_flood_like_records_n  =("is_flood_like",   "sum"),
            s2id_people_affected_sum   =("people_affected", "sum"),
            s2id_homeless_sum          =("homeless",        "sum"),
            s2id_displaced_sum         =("displaced",       "sum"),
            s2id_deaths_sum            =("deaths",          "sum"),
        )
        pbar.update(1)

        years = list(range(START_YEAR, END_YEAR + 1))
        grid  = pd.MultiIndex.from_product(
            [sorted(municipal_codes.astype(str).unique()), years],
            names=["mun_code","year"],
        ).to_frame(index=False)
        pbar.update(1)

        panel = grid.merge(grouped, on=["mun_code","year"], how="left")
        fill_cols = [
            "s2id_event_records_n","s2id_hydrological_records_n",
            "s2id_flood_like_records_n","s2id_people_affected_sum",
            "s2id_homeless_sum","s2id_displaced_sum","s2id_deaths_sum",
        ]
        for col in fill_cols:
            panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0)
        pbar.update(1)

    return panel.sort_values(["mun_code","year"]).reset_index(drop=True)

# ============================================================
# 14. FIGURE 09 — S2ID DISASTER PANEL  (500 DPI composite)
# ============================================================

def make_figure_09_s2id(panel: pd.DataFrame,
                         save_dir: str,
                         dpi: int = 500) -> str:
    """
    6-panel publication-quality composite figure:
      a) National annual disaster records (hydrological + flood-like)
      b) Hydrological records by macro-region (line chart)
      c) Municipality flood record frequency distribution
      d) Annual reported deaths time series
      e) Flood-like / hydrological ratio by macro-region
      f) S2ID integration QA summary table

    Falls back gracefully if region column is absent.

    Parameters
    ----------
    panel    : Municipality-year S2ID panel DataFrame.
    save_dir : Directory for PNG and PDF outputs.
    dpi      : Output resolution (default 500).

    Returns
    -------
    str : Path to saved PNG file.
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

    n_mun    = panel["mun_code"].nunique()
    n_years  = panel["year"].nunique()
    nat      = (panel.groupby("year")[["s2id_hydrological_records_n",
                                       "s2id_flood_like_records_n",
                                       "s2id_deaths_sum"]]
                .sum().reset_index())
    mun_tot  = panel.groupby("mun_code")["s2id_flood_like_records_n"].sum()
    n_ever   = int((mun_tot > 0).sum())
    reg_col  = next((c for c in panel.columns if "region" in c.lower()), None)

    if reg_col:
        reg_annual = (panel.groupby(["year",reg_col])
                      ["s2id_hydrological_records_n"].sum().reset_index())
        reg_tot = (panel.groupby(reg_col)[["s2id_hydrological_records_n",
                                            "s2id_flood_like_records_n"]].sum())
    else:
        reg_annual = None
        reg_tot    = None

    # Layout
    fig = plt.figure(figsize=(7.087, 9.4))
    fig.patch.set_facecolor(C["bg"])
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        left=0.10, right=0.97, top=0.93, bottom=0.06,
        hspace=0.50, wspace=0.34,
    )
    ax_nat  = fig.add_subplot(gs[0, 0])
    ax_reg  = fig.add_subplot(gs[0, 1])
    ax_mun  = fig.add_subplot(gs[1, 0])
    ax_dead = fig.add_subplot(gs[1, 1])
    ax_type = fig.add_subplot(gs[2, 0])
    ax_qa   = fig.add_subplot(gs[2, 1])

    for ax in (ax_nat, ax_reg, ax_mun, ax_dead, ax_type, ax_qa):
        ax.set_facecolor(C["bg"])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color(C["border"])

    # Panel a: national annual records
    ax_nat.bar(nat["year"], nat["s2id_hydrological_records_n"],
               color=C["blue"], width=0.7, alpha=0.82, zorder=3,
               label="Hydrological")
    ax_nat.bar(nat["year"], nat["s2id_flood_like_records_n"],
               color=C["red"],  width=0.7, alpha=0.72, zorder=4,
               label="Flood-like")
    ax_nat.set_xlabel("Year", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_nat.set_ylabel("Disaster records (n)", fontsize=6,
                      color=C["text_sm"], labelpad=3)
    ax_nat.legend(fontsize=4.5, loc="upper left", frameon=True,
                  framealpha=0.9, edgecolor=C["border"])
    ax_nat.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_nat.text(0.03, 0.97, "a", transform=ax_nat.transAxes,
                fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_nat.set_title("National annual disaster records (S2ID)",
                     fontsize=7, color=C["text_hd"], pad=4)

    # Panel b: by region
    if reg_annual is not None:
        for reg in REG_ORDER:
            sub = reg_annual[reg_annual[reg_col]==reg]
            ax_reg.plot(sub["year"], sub["s2id_hydrological_records_n"],
                        color=REGION_COLORS[reg], lw=0.9,
                        marker="o", markersize=2.5, alpha=0.9, label=reg)
        ax_reg.legend(fontsize=4.5, loc="upper right", frameon=True,
                      framealpha=0.9, edgecolor=C["border"])
    else:
        ax_reg.plot(nat["year"], nat["s2id_hydrological_records_n"],
                    color=C["blue"], lw=0.9, marker="o", markersize=2.5)
    ax_reg.set_xlabel("Year", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_reg.set_ylabel("Hydrological records (n)", fontsize=6,
                      color=C["text_sm"], labelpad=3)
    ax_reg.grid(linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_reg.text(0.03, 0.97, "b", transform=ax_reg.transAxes,
                fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_reg.set_title("Hydrological records by macro-region",
                     fontsize=7, color=C["text_hd"], pad=4)

    # Panel c: municipality flood frequency
    vals = mun_tot.values
    if vals.max() > 0:
        ax_mun.hist(vals[vals > 0], bins=range(1, int(vals.max())+2),
                    color=C["teal"], edgecolor="white", linewidth=0.3,
                    alpha=0.85, align="left", zorder=3)
    ax_mun.text(0.97, 0.97,
                f"{n_ever:,} municipalities\nwith \u22651 flood record\n"
                f"({n_ever/n_mun*100:.1f}%)",
                transform=ax_mun.transAxes, ha="right", va="top",
                fontsize=5, color=C["teal"],
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=C["border"], lw=0.5, alpha=0.9))
    ax_mun.set_xlabel(f"Total flood-like records  "
                      f"{START_YEAR}\u2013{END_YEAR}",
                      fontsize=6, color=C["text_sm"], labelpad=3)
    ax_mun.set_ylabel("Number of municipalities", fontsize=6,
                      color=C["text_sm"], labelpad=3)
    ax_mun.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_mun.text(0.03, 0.97, "c", transform=ax_mun.transAxes,
                fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_mun.set_title("Municipality flood record frequency",
                     fontsize=7, color=C["text_hd"], pad=4)

    # Panel d: deaths time series
    ax_dead.bar(nat["year"], nat["s2id_deaths_sum"],
                color=C["red"], width=0.7, alpha=0.82, zorder=3)
    roll = (pd.Series(nat["s2id_deaths_sum"].values)
            .rolling(3, min_periods=2).mean())
    ax_dead.plot(nat["year"], roll, color=C["accent"],
                 lw=1.0, ls="--", zorder=4, label="3-yr rolling")
    ax_dead.set_xlabel("Year", fontsize=6, color=C["text_sm"], labelpad=3)
    ax_dead.set_ylabel("Reported deaths (n)", fontsize=6,
                       color=C["text_sm"], labelpad=3)
    ax_dead.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_dead.legend(fontsize=4.5, loc="upper right", frameon=True,
                   framealpha=0.9, edgecolor=C["border"])
    ax_dead.text(0.03, 0.97, "d", transform=ax_dead.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_dead.set_title("Annual deaths from disasters (S2ID)",
                      fontsize=7, color=C["text_hd"], pad=4)

    # Panel e: flood-like / hydro ratio by region
    if reg_tot is not None:
        ratio = (reg_tot["s2id_flood_like_records_n"] /
                 reg_tot["s2id_hydrological_records_n"].replace(0, np.nan)
                 ).fillna(0)
        rvals = [float(ratio.get(r, 0)) for r in REG_ORDER]
        rcols = [REGION_COLORS[r] for r in REG_ORDER]
        rlbls = ["N","NE","CW","SE","S"]
    else:
        total_hydro = nat["s2id_hydrological_records_n"].sum()
        total_flood = nat["s2id_flood_like_records_n"].sum()
        rvals = [total_flood/total_hydro if total_hydro > 0 else 0] * 5
        rcols = [C["blue"]] * 5
        rlbls = REG_ORDER

    ax_type.barh(range(len(REG_ORDER)), rvals, color=rcols,
                 edgecolor="white", linewidth=0.3, alpha=0.87, height=0.55)
    ax_type.axvline(np.mean(rvals), color=C["accent"], lw=0.8, ls="--", zorder=4)
    ax_type.set_yticks(range(len(REG_ORDER)))
    ax_type.set_yticklabels(rlbls, fontsize=6)
    ax_type.set_xlabel("Flood-like / hydrological ratio",
                       fontsize=6, color=C["text_sm"], labelpad=3)
    ax_type.set_xlim(0, 1)
    ax_type.grid(axis="x", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_type.text(0.03, 0.97, "e", transform=ax_type.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_type.set_title("Flood-like / hydrological ratio by region",
                      fontsize=7, color=C["text_hd"], pad=4)

    # Panel f: QA summary
    ax_qa.set_xlim(0,1); ax_qa.set_ylim(0,1); ax_qa.axis("off")
    ax_qa.text(0.03, 0.97, "f", transform=ax_qa.transAxes,
               fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_qa.set_title("S2ID integration QA summary",
                    fontsize=7, color=C["text_hd"], pad=4)

    qa_items = [
        ("Source",                  "S2ID Excel files"),
        ("Period",                  f"{START_YEAR}\u2013{END_YEAR}"),
        ("Municipalities",          f"{n_mun:,}"),
        ("Panel rows",              f"{n_mun*n_years:,}"),
        ("Years in panel",          str(n_years)),
        ("Hydrological records",    f"{int(nat['s2id_hydrological_records_n'].sum()):,}"),
        ("Flood-like records",      f"{int(nat['s2id_flood_like_records_n'].sum()):,}"),
        ("Municipalities w/ flood", f"{n_ever:,}  ({n_ever/n_mun*100:.1f}%)"),
        ("Output format",           "Parquet + CSV"),
    ]
    y0 = 0.88
    for key, val in qa_items:
        ax_qa.text(0.04, y0, key, ha="left", va="center",
                   fontsize=6, color=C["text_sm"],
                   transform=ax_qa.transAxes)
        ax_qa.text(0.96, y0, val, ha="right", va="center",
                   fontsize=6, color=C["teal"], fontweight="bold",
                   transform=ax_qa.transAxes)
        ax_qa.plot([0.02, 0.98], [y0-0.04, y0-0.04],
                   color=C["border"], lw=0.3,
                   transform=ax_qa.transAxes)
        y0 -= 0.096

    # Title and caption
    fig.text(
        0.50, 0.970,
        "Figure 10  |  S2ID disaster panel \u2014 Flood Inequality in Brazil",
        ha="center", va="top",
        fontsize=8, fontweight="bold", color=C["text_hd"],
    )
    fig.text(
        0.50, 0.956,
        (f"S2ID \u00b7 {n_mun:,} municipalities \u00b7 "
         f"{START_YEAR}\u2013{END_YEAR} \u00b7 "
         "annual municipality-year panel \u00b7 "
         "hydrological + flood-like classification"),
        ha="center", va="top",
        fontsize=6, color=C["text_sm"], style="italic",
    )

    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, "fig09_s2id_disaster_panel.png")
    pdf_path = os.path.join(save_dir, "fig09_s2id_disaster_panel.pdf")
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

    municipal, municipal_ref = load_municipal_reference()
    municipal["mun_code"]    = municipal["mun_code"].astype(str)
    expected_munis           = municipal["mun_code"].nunique()

    if is_valid_output(OUTPUT_PANEL_PARQUET, OUTPUT_META, expected_munis):
        log_summary("Valid S2ID panel already exists - skipping processing.")
        with tqdm(total=1, desc="Loading existing panel",
                  bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
            panel = pd.read_parquet(OUTPUT_PANEL_PARQUET)
            pbar.update(1)
        log_summary(f"Loaded {len(panel):,} rows from existing panel.")
    else:
        config    = read_config(CONFIG_PATH)
        resources = discover_local_files()
        raw, diagnostics = build_raw_concat(resources, municipal_ref)

        with tqdm(total=1, desc="Saving raw concat",
                  bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
            raw.to_parquet(RAW_CONCAT_PARQUET, index=False)
            pbar.update(1)

        panel = build_municipal_annual_panel(raw, municipal["mun_code"])

        expected_rows = expected_munis * (END_YEAR - START_YEAR + 1)
        if len(panel) != expected_rows:
            raise RuntimeError(
                f"Panel has {len(panel):,} rows; expected {expected_rows:,}.")

        diagnostics += [
            {"check":"raw_concat_rows",       "value":int(len(raw)),  "details":""},
            {"check":"panel_rows",            "value":int(len(panel)),"details":""},
            {"check":"panel_municipalities",  "value":int(panel["mun_code"].nunique()),"details":""},
            {"check":"panel_years",           "value":int(panel["year"].nunique()),"details":""},
            {"check":"hydrological_total",    "value":int(panel["s2id_hydrological_records_n"].sum()),"details":""},
            {"check":"flood_like_total",      "value":int(panel["s2id_flood_like_records_n"].sum()),"details":""},
        ]
        pd.DataFrame(diagnostics).to_csv(OUTPUT_DIAG, index=False)

        # Atomic save
        tmp_p = str(OUTPUT_PANEL_PARQUET) + ".tmp"
        tmp_c = str(OUTPUT_PANEL_CSV)     + ".tmp"
        tmp_m = str(OUTPUT_META)          + ".tmp"
        for p in [tmp_p, tmp_c, tmp_m]:
            if os.path.exists(p): os.remove(p)

        with tqdm(total=3, desc="Writing files",
                  bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
            panel.to_parquet(tmp_p, index=False); pbar.update(1)
            panel.to_csv(tmp_c, index=False);     pbar.update(1)
            meta = {
                "project"              : "Flood Inequality Across Brazil",
                "module"               : "09_build_disaster_s2id.py",
                "version"              : "v4.3",
                "status"               : "completed",
                "created_at"           : datetime.now().isoformat(),
                "search_dirs"          : [str(p) for p in SEARCH_DIRS],
                "resource_years"       : [r["year"] for r in resources],
                "resource_files"       : [str(r["path"]) for r in resources],
                "raw_concat_parquet"   : str(RAW_CONCAT_PARQUET),
                "output_panel_parquet" : str(OUTPUT_PANEL_PARQUET),
                "output_panel_csv"     : str(OUTPUT_PANEL_CSV),
                "diagnostics_csv"      : str(OUTPUT_DIAG),
                "n_municipalities"     : int(panel["mun_code"].nunique()),
                "n_rows"               : int(len(panel)),
                "columns"              : list(panel.columns),
            }
            with open(tmp_m, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
            pbar.update(1)

        os.replace(tmp_p, OUTPUT_PANEL_PARQUET)
        os.replace(tmp_c, OUTPUT_PANEL_CSV)
        os.replace(tmp_m, OUTPUT_META)

        update_catalog("09_build_disaster_s2id", "ALL",
                       str(OUTPUT_PANEL_PARQUET), "completed")

        config["s2id_municipal_annual_panel"] = {
            "name"               : "s2id_municipal_annual_brazil",
            "path_parquet"       : str(OUTPUT_PANEL_PARQUET),
            "path_csv"           : str(OUTPUT_PANEL_CSV),
            "path_meta"          : str(OUTPUT_META),
            "path_diag"          : str(OUTPUT_DIAG),
            "raw_concat_parquet" : str(RAW_CONCAT_PARQUET),
            "resource_years"     : [r["year"] for r in resources],
            "resource_files"     : [str(r["path"]) for r in resources],
            "n_municipalities"   : int(panel["mun_code"].nunique()),
            "n_rows"             : int(len(panel)),
        }
        write_config(CONFIG_PATH, config)

        log_summary("=" * 60)
        log_summary(f"DONE | municipalities={panel['mun_code'].nunique():,} | "
                    f"rows={len(panel):,}")

    # Figure
    log("Generating Figure 09 ...")
    with tqdm(total=1, desc="Rendering figure",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        fig_path = make_figure_09_s2id(
            panel,
            str(BASE_PATH / "06_figures"),
            dpi=500,
        )
        pbar.update(1)

    log_summary(f"Figure saved: {fig_path}")
    logging.info("Figure 09 generated successfully.")

    print("\n" + "=" * 60)
    print("  Module 09 complete")
    print("=" * 60)
    print(f"  Municipalities : {panel['mun_code'].nunique():,}")
    print(f"  Panel rows     : {len(panel):,}")
    print(f"  Panel Parquet  : {OUTPUT_PANEL_PARQUET}")
    print(f"  Panel CSV      : {OUTPUT_PANEL_CSV}")
    print(f"  Diagnostics    : {OUTPUT_DIAG}")
    print(f"  Figure PNG     : {OUTPUT_FIG_PNG}")
    print(f"  Figure PDF     : {OUTPUT_FIG_PDF}")
    print("  Ready for Module 10.")
    print("=" * 60)


if __name__ == "__main__":
    main()
