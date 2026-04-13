"""
Project: Flood Inequality Across Brazil
Module:  00_setup_project_environment.py
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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib import gridspec

# ============================================================
# 3. MOUNT GOOGLE DRIVE  (idempotent)
# ============================================================
from google.colab import drive

if not os.path.exists("/content/drive/MyDrive"):
    drive.mount("/content/drive")
else:
    print("Drive already mounted - skipping.")

# ============================================================
# 4. PROJECT CONSTANTS
# ============================================================
BASE_PATH  = "/content/drive/MyDrive/Brazil/flood_inequality_project"
EE_PROJECT = "ee-enneralcantara2"

DIRS = [
    "00_config",
    "01_raw",
    "02_intermediate",
    "03_features",
    "04_integrated",
    "05_modeling",
    "06_figures",
    "07_logs",
    "08_catalog",
]

# ============================================================
# 5. DIRECTORY STRUCTURE  (idempotent)
# ============================================================
for d in DIRS:
    Path(BASE_PATH, d).mkdir(parents=True, exist_ok=True)
print("Directory structure ready.")

# ============================================================
# 6. CONFIGURATION FILE  (created only once)
# ============================================================
config = {
    "project_name": "Flood Inequality Brazil",
    "version"     : "v1",
    "created_at"  : str(datetime.now()),
    "base_path"   : BASE_PATH,
    "gee_project" : EE_PROJECT,
    "year_start"  : 1981,
    "year_end"    : 2025,
    "spatial_unit": "to_define",
    "tile_system" : "to_define",
    "crs"         : "EPSG:4326",
}

config_path = Path(BASE_PATH, "00_config", "config.json")
if not config_path.exists():
    config_path.write_text(json.dumps(config, indent=4))
    print("Config file created.")
else:
    print("Config file already exists - skipping.")

# ============================================================
# 7. LOGGING SYSTEM
# ============================================================
log_path = Path(BASE_PATH, "07_logs", "setup.log")
logging.basicConfig(
    filename=str(log_path),
    filemode="a",
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logging.info("Setup executed successfully.")
print("Logging initialized.")

# ============================================================
# 8. PROCESSING CATALOG  (created only once)
# ============================================================
catalog_path = Path(BASE_PATH, "08_catalog", "catalog.csv")
if not catalog_path.exists():
    catalog_path.write_text(
        "stage,tile_id,period,status,output_path,timestamp\n"
    )
    print("Catalog initialized.")
else:
    print("Catalog already exists - skipping.")

# ============================================================
# 9. EARTH ENGINE INITIALIZATION
# ============================================================
print("\nInitializing Google Earth Engine ...")
import ee

try:
    ee.Initialize(project=EE_PROJECT)
    print("Earth Engine initialized successfully.")
except Exception:
    print("Authentication required - running ee.Authenticate() ...")
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)
    print("Earth Engine initialized after authentication.")

# ============================================================
# 10. FIGURE 00 - PROJECT PIPELINE ARCHITECTURE
#     Nature / Science style · 500 DPI · composite layout
# ============================================================

def make_figure_00_pipeline(save_dir: str, dpi: int = 500) -> str:
    """
    Generate a publication-quality pipeline architecture figure.

    Panels:
      a) Main pipeline  : stages 00-06 (vertical flow with arrows)
      b) Support stages : 07 logs + 08 catalog (horizontal)
      c) Metadata strip : project, coverage, CRS, GEE project, resolution
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
        "svg.fonttype"     : "none",
    })

    C = {
        "bg"      : "#FAFAF8",
        "panel"   : "#F2F1ED",
        "infra"   : "#6B7280",
        "data"    : "#2E6DA4",
        "feat"    : "#2A8C6E",
        "model"   : "#6B46C1",
        "arrow"   : "#4B5563",
        "text_hd" : "#111827",
        "text_bd" : "#374151",
        "text_sm" : "#6B7280",
        "border"  : "#D1D5DB",
    }

    stages = [
        ("00", "Environment setup",   "Drive · config.json · EE init",           "infra", "INIT"),
        ("01", "Raw data ingestion",  "CHIRPS · ERA5 · DEM · LULC · census",     "data",  "RAW"),
        ("02", "Preprocessing",       "Reproject · resample · QC mask",          "data",  "PROC"),
        ("03", "Feature engineering", "Flood index · SPI · TWI · exposure",      "feat",  "FEAT"),
        ("04", "Data integration",    "Socioeconomic × environmental join",       "feat",  "INT"),
        ("05", "Modeling",            "Regression · spatial stats · ML",         "model", "MOD"),
        ("06", "Figures",             "Maps · composites · 500 DPI",             "model", "FIG"),
        ("07", "Logs",                "Timestamped execution history",            "infra", "LOG"),
        ("08", "Catalog",             "stage · tile_id · status · path",         "infra", "CAT"),
    ]

    fig = plt.figure(figsize=(7.087, 9.0))
    fig.patch.set_facecolor(C["bg"])

    gs_outer = gridspec.GridSpec(
        3, 1,
        height_ratios=[7, 1.2, 0.6],
        hspace=0.06,
        left=0.08, right=0.97,
        top=0.96,  bottom=0.03,
    )

    ax_main = fig.add_subplot(gs_outer[0])
    ax_supp = fig.add_subplot(gs_outer[1])
    ax_meta = fig.add_subplot(gs_outer[2])

    for ax in (ax_main, ax_supp, ax_meta):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(C["bg"])

    BOX_H   = 0.092
    BOX_PAD = 0.012
    Y_START = 0.97

    def draw_stage(ax, x, y, w, h, label, sublabel,
                   stage_id, color, tag, fs_lbl=7.5, fs_sub=6):
        ec = color
        fc = matplotlib.colors.to_rgba(color, alpha=0.10)

        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008",
            linewidth=0.8, edgecolor=ec, facecolor=fc,
            transform=ax.transAxes, clip_on=False, zorder=3,
        ))
        ax.add_patch(FancyBboxPatch(
            (x + 0.010, y + h/2 - 0.025), 0.058, 0.050,
            boxstyle="round,pad=0.005",
            linewidth=0, facecolor=color,
            transform=ax.transAxes, clip_on=False, zorder=4,
        ))
        ax.text(x + 0.039, y + h/2, stage_id,
                ha="center", va="center", fontsize=7, fontweight="bold",
                color="white", transform=ax.transAxes, zorder=5)
        ax.add_patch(FancyBboxPatch(
            (x + w - 0.097, y + h/2 - 0.022), 0.090, 0.044,
            boxstyle="round,pad=0.005",
            linewidth=0.5, edgecolor=ec,
            facecolor=matplotlib.colors.to_rgba(color, 0.18),
            transform=ax.transAxes, clip_on=False, zorder=4,
        ))
        ax.text(x + w - 0.052, y + h/2, tag,
                ha="center", va="center", fontsize=5.5, fontweight="bold",
                color=color, transform=ax.transAxes, zorder=5)
        ax.text(x + 0.085, y + h * 0.64, label,
                ha="left", va="center", fontsize=fs_lbl, fontweight="bold",
                color=C["text_hd"], transform=ax.transAxes, zorder=5)
        ax.text(x + 0.085, y + h * 0.30, sublabel,
                ha="left", va="center", fontsize=fs_sub, color=C["text_sm"],
                transform=ax.transAxes, zorder=5)

    # Panel a: stages 00-06
    main_stages = stages[:7]
    y_positions = []
    for i, (sid, lbl, sub, ckey, tag) in enumerate(main_stages):
        y = Y_START - (i + 1) * (BOX_H + BOX_PAD)
        y_positions.append(y)
        draw_stage(ax_main, 0.02, y, 0.96, BOX_H,
                   lbl, sub, sid, C[ckey], tag)

    for i in range(len(main_stages) - 1):
        ax_main.annotate(
            "",
            xy    =(0.50, y_positions[i + 1] + BOX_H + 0.001),
            xytext=(0.50, y_positions[i]               - 0.001),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                            lw=0.8, mutation_scale=7),
            zorder=2,
        )

    ax_main.text(0.01, 0.995, "a", ha="left", va="top",
                 fontsize=9, fontweight="bold", color=C["text_hd"],
                 transform=ax_main.transAxes)

    # Panel b: stages 07-08
    for j, (sid, lbl, sub, ckey, tag) in enumerate(stages[7:]):
        draw_stage(ax_supp, 0.02 + j * 0.50, 0.10, 0.46, 0.75,
                   lbl, sub, sid, C[ckey], tag, fs_lbl=7, fs_sub=5.5)

    for xp in [0.245, 0.745]:
        ax_supp.annotate(
            "", xy=(xp, 0.87), xytext=(xp, 1.04),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                            lw=0.7, mutation_scale=6),
            zorder=2, clip_on=False,
        )

    ax_supp.text(0.01, 0.995, "b", ha="left", va="top",
                 fontsize=9, fontweight="bold", color=C["text_hd"],
                 transform=ax_supp.transAxes)

    legend_items = [
        (C["infra"], "Infrastructure"),
        (C["data"],  "Data acquisition"),
        (C["feat"],  "Feature engineering"),
        (C["model"], "Analysis & output"),
    ]
    lx, ly = 0.97, 0.92
    for color, label in legend_items:
        ax_supp.add_patch(mpatches.Circle(
            (lx, ly), 0.025,
            transform=ax_supp.transAxes,
            color=color, clip_on=False, zorder=6,
        ))
        ax_supp.text(lx + 0.035, ly, label,
                     ha="left", va="center", fontsize=5.5, color=C["text_bd"],
                     transform=ax_supp.transAxes)
        ly -= 0.22

    # Panel c: metadata
    meta_items = [
        ("Project",     "Flood Inequality - Brazil"),
        ("Coverage",    "1981 - 2025"),
        ("CRS",         "EPSG:4326"),
        ("GEE project", "ee-enneralcantara2"),
        ("Resolution",  "500 DPI  (Nature/Science)"),
    ]

    ax_meta.set_facecolor(C["panel"])
    ax_meta.add_patch(FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.0",
        linewidth=0.5, edgecolor=C["border"], facecolor=C["panel"],
        transform=ax_meta.transAxes, clip_on=False, zorder=0,
    ))

    col_w = 1.0 / len(meta_items)
    for k, (key, val) in enumerate(meta_items):
        cx = col_w * k + col_w / 2
        ax_meta.text(cx, 0.72, key,
                     ha="center", va="center", fontsize=5.5,
                     color=C["text_sm"], transform=ax_meta.transAxes)
        ax_meta.text(cx, 0.28, val,
                     ha="center", va="center", fontsize=6.0,
                     fontweight="bold", color=C["text_hd"],
                     transform=ax_meta.transAxes)
        if k < len(meta_items) - 1:
            xd = col_w * (k + 1)
            ax_meta.plot([xd, xd], [0.05, 0.95],
                         color=C["border"], linewidth=0.5,
                         transform=ax_meta.transAxes, clip_on=True)

    fig.text(
        0.50, 0.975,
        "Figure 1  |  Project pipeline - Flood Inequality in Brazil",
        ha="center", va="top",
        fontsize=8, fontweight="bold", color=C["text_hd"],
    )
    fig.text(
        0.50, 0.963,
        ("Pipeline from environment initialization to publication-quality "
         "figures. Arrows indicate data flow. Tags denote stage type."),
        ha="center", va="top",
        fontsize=6, color=C["text_sm"], style="italic",
    )

    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, "fig00_pipeline_architecture.png")
    pdf_path = os.path.join(save_dir, "fig00_pipeline_architecture.pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight",
                facecolor=C["bg"], format="png")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight",
                facecolor=C["bg"], format="pdf")

    plt.show()
    plt.close(fig)
    print(f"\nFigure 00 saved:\n  PNG -> {png_path}\n  PDF -> {pdf_path}")
    return png_path


# ============================================================
# RUN
# ============================================================
fig_dir = os.path.join(BASE_PATH, "06_figures")
make_figure_00_pipeline(fig_dir, dpi=500)
logging.info("Figure 00 generated successfully.")

print("\n" + "=" * 60)
print("  Setup complete")
print("=" * 60)
print(f"  Project path : {BASE_PATH}")
print(f"  GEE project  : {EE_PROJECT}")
print(f"  Config       : {config_path}")
print(f"  Log          : {log_path}")
print(f"  Catalog      : {catalog_path}")
print(f"  Figures      : {fig_dir}")
print("  Ready for Module 01.")
print("=" * 60)
