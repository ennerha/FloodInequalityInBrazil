"""
Project: Flood Inequality Across Brazil

Module: 11_model_hazard_inequality_disaster.py

Purpose:
Model the relationship between hydroclimatic hazard, social inequality,
and observed disaster impacts across Brazilian municipalities.

Main target:
- disaster_observed_index

Models:
- Explainable Boosting Regressor (main model)
- Elastic Net (baseline linear regularized)
- Random Forest Regressor (baseline nonlinear)

Outputs:
- model_matrix.parquet
- model_metrics.csv
- feature_importance_ebm.csv
- feature_importance_rf.csv
- partial_effects_ebm.parquet
- model_predictions.parquet
- fig11_model_performance_and_interpretation.png
- fig11_model_performance_and_interpretation.pdf
- 11_model_hazard_inequality_disaster.meta.json
- 11_model_hazard_inequality_disaster.log

Changelog v2.1 (from v2.0):
- choose_features() updated: s2id_feat_* columns are now ALLOWED as
  predictors (structural disaster exposure features from Module 10).
  Only raw s2id_ annual counts remain blocked (target leakage risk).
- delete stale checkpoints before re-run if target changed (v3.0 of M10)

Author: Enner H. de Alcântara
Version: v2.1
Language: English
"""

# =========================================================
# 1. IMPORTS
# =========================================================
import os
import sys
import json
import logging
import warnings
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

def _ensure_tqdm():
    try:
        import tqdm  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tqdm"])

_ensure_tqdm()
from tqdm.auto import tqdm

# =========================================================
# 2. OPTIONAL DEPENDENCY — interpret
# =========================================================
def ensure_interpret():
    try:
        from interpret.glassbox import ExplainableBoostingRegressor  # noqa
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "interpret"])

ensure_interpret()
from interpret.glassbox import ExplainableBoostingRegressor

# =========================================================
# 3. PATHS AND CONSTANTS
# =========================================================
BASE_PATH = Path("/content/drive/MyDrive/Brazil/flood_inequality_project")

CONFIG_PATH  = BASE_PATH / "00_config"  / "config.json"
LOG_PATH     = BASE_PATH / "07_logs"    / "11_model_hazard_inequality_disaster.log"
CATALOG_PATH = BASE_PATH / "08_catalog" / "catalog.csv"

INPUT_SUMMARY_PATH = (
    BASE_PATH / "04_integrated"
    / "hazard_social_disaster_municipal_summary_brazil.geoparquet"
)

OUTPUT_DIR     = BASE_PATH / "05_modeling"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIG_DIR        = BASE_PATH / "06_figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MATRIX         = OUTPUT_DIR / "model_matrix.parquet"
OUTPUT_METRICS        = OUTPUT_DIR / "model_metrics.csv"
OUTPUT_EBM_IMPORTANCE = OUTPUT_DIR / "feature_importance_ebm.csv"
OUTPUT_RF_IMPORTANCE  = OUTPUT_DIR / "feature_importance_rf.csv"
OUTPUT_PARTIALS       = OUTPUT_DIR / "partial_effects_ebm.parquet"
OUTPUT_PREDICTIONS    = OUTPUT_DIR / "model_predictions.parquet"
OUTPUT_META           = OUTPUT_DIR / "11_model_hazard_inequality_disaster.meta.json"

OUTPUT_FIG_PNG = FIG_DIR / "fig11_model_performance_and_interpretation.png"
OUTPUT_FIG_PDF = FIG_DIR / "fig11_model_performance_and_interpretation.pdf"

TARGET_COL        = "disaster_observed_index"
CV_FOLDS          = 10
RANDOM_STATE      = 42
VERBOSE           = False
KEEP_CHECKPOINTS  = False
SHOW_FIG_IN_COLAB = True

# =========================================================
# 4. LOGGING
# =========================================================
def setup_logger(log_path: Path, verbose: bool) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("module_11")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

log = setup_logger(LOG_PATH, VERBOSE)

def log_summary(message: str) -> None:
    log.warning("SUMMARY | " + message)

# =========================================================
# 5. HELPERS
# =========================================================
def read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def write_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)

def update_catalog(stage: str, tile_id: str, output_path: str, status: str) -> None:
    row = pd.DataFrame([{
        "stage":       stage,
        "tile_id":     tile_id,
        "period":      "2013_2022",
        "status":      status,
        "output_path": output_path,
        "timestamp":   datetime.now().isoformat(),
    }])
    if CATALOG_PATH.exists():
        try:
            df = pd.read_csv(CATALOG_PATH)
            df = df[~((df["stage"] == stage) & (df["tile_id"] == tile_id))]
            df = pd.concat([df, row], ignore_index=True)
            df.to_csv(CATALOG_PATH, index=False)
            return
        except Exception:
            pass
    row.to_csv(CATALOG_PATH, index=False)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_imputed_feature_names(pipeline: Pipeline, input_features: list) -> list:
    imputer = pipeline.named_steps["imputer"]
    try:
        return list(imputer.get_feature_names_out(input_features))
    except AttributeError:
        n_out = len(imputer.statistics_)
        if n_out == len(input_features):
            return input_features
        return input_features[:n_out]

def checkpoint_path(model_name: str) -> Path:
    return CHECKPOINT_DIR / f"checkpoint_{model_name}.pkl"

def checkpoint_meta_path(model_name: str) -> Path:
    return CHECKPOINT_DIR / f"checkpoint_{model_name}_meta.json"

def save_checkpoint(model_name: str, pipeline: Pipeline, oof_pred: np.ndarray,
                    metrics_row: dict) -> None:
    joblib.dump(pipeline, checkpoint_path(model_name))
    with open(checkpoint_meta_path(model_name), "w", encoding="utf-8") as f:
        json.dump({**metrics_row, "oof_pred": oof_pred.tolist()}, f)
    log_summary(f"Checkpoint saved: {model_name}")

def load_checkpoint(model_name: str):
    cp  = checkpoint_path(model_name)
    cpm = checkpoint_meta_path(model_name)
    if not cp.exists() or not cpm.exists():
        return None
    pipeline = joblib.load(cp)
    with open(cpm, encoding="utf-8") as f:
        meta = json.load(f)
    oof_pred = np.array(meta.pop("oof_pred"))
    log_summary(f"Checkpoint loaded (skipping training): {model_name}")
    return pipeline, oof_pred, meta

def remove_checkpoints(model_names: list) -> None:
    for name in model_names:
        for p in [checkpoint_path(name), checkpoint_meta_path(name)]:
            if p.exists():
                p.unlink()
    log_summary("Checkpoints removed.")

def purge_stale_checkpoints() -> None:
    """
    Remove all existing checkpoints before a new run.
    Required when the target (disaster_observed_index) was rebuilt
    in Module 10 v3.0 — old checkpoints trained on the v2.1 target
    would produce misleading results.
    """
    removed = 0
    for p in CHECKPOINT_DIR.glob("checkpoint_*"):
        p.unlink()
        removed += 1
    if removed:
        print(f"  ⚠  Purged {removed} stale checkpoint(s) from previous run.")
        log_summary(f"Purged {removed} stale checkpoints.")

# =========================================================
# 6. LOAD DATA
# =========================================================
def load_input() -> pd.DataFrame:
    if not INPUT_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_SUMMARY_PATH}")
    gdf = gpd.read_parquet(INPUT_SUMMARY_PATH)
    if gdf.empty:
        raise RuntimeError("Input summary dataset is empty.")
    df = pd.DataFrame(gdf.drop(columns="geometry").copy())
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"Target column not found: {TARGET_COL}")
    return df

# =========================================================
# 7. FEATURE SELECTION  (v2.1 — s2id_feat_* now ALLOWED)
# =========================================================
def choose_features(df: pd.DataFrame) -> list:
    """
    Select numeric predictor columns for modeling.

    Blacklist:
      - Municipality identifiers
      - Target column and its direct derivatives
      - Raw s2id_ annual event counts (prefix "s2id_" without "s2id_feat_")
        → these are aggregated into the target; keeping them would be leakage
      - Compound index columns (built from the target; leakage)
      - Categorical classification columns

    Allowed (intentionally):
      - s2id_feat_* → structural S2ID features built in Module 10
        (historical flood frequency, trend slope, acceleration, etc.)
        These reflect long-run disaster exposure and are legitimate predictors.
    """
    blacklist = {
        "mun_code",
        "mun_name",
        "uf_code",
        "uf_sigla",
        # Target and direct derivatives
        TARGET_COL,
        "disaster_observed_raw",
        "annual_disaster_observed_index_mean",
        "annual_disaster_observed_index_max",
        # Compound index (built using the target — leakage)
        "hazard_social_disaster_compound_raw",
        "hazard_social_disaster_compound_index",
        # Categorical
        "triple_burden_flag",
        "hazard_social_quadrant",
        "compound_class_q",
    }

    features = []
    for col in df.columns:
        if col in blacklist:
            continue
        # Block raw S2ID annual counts but ALLOW s2id_feat_* structural features
        if col.startswith("s2id_") and not col.startswith("s2id_feat_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)

    return sorted(features)

# =========================================================
# 8. BUILD MODEL MATRIX
# =========================================================
def build_model_matrix(df: pd.DataFrame):
    features = choose_features(df)

    X = df[features].copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    if len(X) == 0:
        raise RuntimeError("No valid rows remained after filtering target.")

    all_nan_cols = [c for c in features if X[c].isna().all()]
    if all_nan_cols:
        log_summary(f"Columns with ALL-NaN values (dropped): {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)
        features = [c for c in features if c not in all_nan_cols]

    matrix = X.copy()
    matrix[TARGET_COL] = y

    return X, y, features, matrix

# =========================================================
# 9. MODEL DEFINITIONS
# =========================================================
def build_models() -> dict:
    ebm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExplainableBoostingRegressor(
            random_state=RANDOM_STATE,
            interactions=10,
            outer_bags=8,
            inner_bags=0,
            learning_rate=0.03,
            max_rounds=5000,
            min_samples_leaf=4,
            max_leaves=3,
        )),
    ])

    enet = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0],
            alphas=np.logspace(-3, 1, 30),
            cv=5,
            random_state=RANDOM_STATE,
            max_iter=50_000,
        )),
    ])

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    return {"EBM": ebm, "ElasticNet": enet, "RandomForest": rf}

# =========================================================
# 10. CROSS-VALIDATION WITH PROGRESS BAR
# =========================================================
def _cv_with_progress(model, X, y, cv, model_name: str):
    r2s, maes, rmses = [], [], []
    oof_pred = np.full(len(y), np.nan)

    folds = list(cv.split(X, y))

    with tqdm(folds, desc=f"  CV folds [{model_name}]",
              unit="fold", leave=False, colour="cyan") as pbar:
        for train_idx, val_idx in pbar:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model.fit(X_tr, y_tr)

            preds = model.predict(X_val)
            oof_pred[val_idx] = preds

            fold_r2   = r2_score(y_val, preds)
            fold_mae  = mean_absolute_error(y_val, preds)
            fold_rmse = rmse(y_val, preds)

            r2s.append(fold_r2)
            maes.append(fold_mae)
            rmses.append(fold_rmse)

            pbar.set_postfix(r2=f"{fold_r2:.3f}", mae=f"{fold_mae:.4f}")

    return {
        "cv_r2_mean":   np.mean(r2s),
        "cv_r2_std":    np.std(r2s),
        "cv_mae_mean":  np.mean(maes),
        "cv_mae_std":   np.std(maes),
        "cv_rmse_mean": np.mean(rmses),
        "cv_rmse_std":  np.std(rmses),
        "oof_r2":       r2_score(y, oof_pred),
        "oof_mae":      mean_absolute_error(y, oof_pred),
        "oof_rmse":     rmse(y, oof_pred),
    }, oof_pred

# =========================================================
# 11. MODEL EVALUATION WITH CHECKPOINTING
# =========================================================
def evaluate_models(models: dict, X: pd.DataFrame, y: pd.Series):
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    metrics_rows  = []
    fitted_models = {}

    def _flush_metrics():
        (
            pd.DataFrame(metrics_rows)
            .sort_values("cv_r2_mean", ascending=False)
            .reset_index(drop=True)
            .to_csv(OUTPUT_METRICS, index=False)
        )

    model_names = list(models.keys())

    with tqdm(model_names, desc="Models", unit="model",
              colour="green", position=0) as outer_bar:
        for name in outer_bar:
            outer_bar.set_description(f"Model: {name}")
            model = models[name]

            cached = load_checkpoint(name)
            if cached is not None:
                fitted_pipeline, oof_pred, metrics_row = cached
                metrics_row["model"] = name
                log_summary(f"[{name}] Loaded from checkpoint.")
                tqdm.write(f"  ✓ {name}: loaded from checkpoint "
                           f"(OOF R²={metrics_row['oof_r2']:.4f})")
            else:
                log_summary(f"Starting CV: {name}")
                cv_metrics, oof_pred = _cv_with_progress(model, X, y, cv, name)

                tqdm.write(f"  → Fitting {name} on full dataset…")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model.fit(X, y)

                fitted_pipeline = model
                metrics_row = {"model": name, **cv_metrics}
                save_checkpoint(name, fitted_pipeline, oof_pred, metrics_row)

                tqdm.write(
                    f"  ✓ {name}: OOF R²={metrics_row['oof_r2']:.4f} | "
                    f"CV R²={metrics_row['cv_r2_mean']:.4f} "
                    f"(±{metrics_row['cv_r2_std']:.4f})"
                )

            metrics_rows.append(metrics_row)
            fitted_models[name] = {
                "pipeline": fitted_pipeline,
                "oof_pred": oof_pred,
            }

            _flush_metrics()
            log_summary(f"Metrics saved after: {name}")

    metrics_df = (
        pd.DataFrame(metrics_rows)
        .sort_values("cv_r2_mean", ascending=False)
        .reset_index(drop=True)
    )
    return metrics_df, fitted_models

# =========================================================
# 12. INTERPRETATION OUTPUTS
# =========================================================
def extract_ebm_outputs(fitted_pipeline: Pipeline,
                        X: pd.DataFrame,
                        y: pd.Series,
                        input_features: list):
    ebm = fitted_pipeline.named_steps["model"]
    global_exp = ebm.explain_global()
    data = global_exp.data()

    names  = data["names"]
    scores = data["scores"]

    imp_df = (
        pd.DataFrame({"feature": names, "importance": scores})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    partial_rows = []
    with tqdm(enumerate(names), desc="  EBM partial effects",
              total=len(names), unit="feat", leave=False, colour="yellow") as pbar:
        for i, feat in pbar:
            feat_data = global_exp.data(i)
            xs = feat_data.get("names", [])
            ys = feat_data.get("scores", [])
            for xv, yv in zip(xs, ys):
                partial_rows.append({"feature": feat, "x": str(xv), "effect": yv})

    partial_df = pd.DataFrame(partial_rows)
    return imp_df, partial_df

def extract_rf_importance(fitted_pipeline: Pipeline, input_features: list):
    rf_step      = fitted_pipeline.named_steps["model"]
    actual_names = get_imputed_feature_names(fitted_pipeline, input_features)

    n_expected = len(rf_step.feature_importances_)
    if len(actual_names) != n_expected:
        log_summary(
            f"WARNING: feature name count ({len(actual_names)}) != "
            f"RF importances length ({n_expected}). Truncating/padding."
        )
        actual_names = actual_names[:n_expected]
        while len(actual_names) < n_expected:
            actual_names.append(f"unknown_{len(actual_names)}")

    imp_df = (
        pd.DataFrame({"feature": actual_names,
                      "importance": rf_step.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return imp_df

# =========================================================
# 13. PREDICTIONS TABLE
# =========================================================
def build_predictions_table(df: pd.DataFrame, y: pd.Series,
                             fitted_models: dict):
    pred = pd.DataFrame({
        "mun_code": df.loc[y.index, "mun_code"].astype(str).values,
        "observed_disaster_observed_index": y.values,
    })
    for name, obj in fitted_models.items():
        pred[f"pred_{name}"] = obj["oof_pred"]
    return pred

# =========================================================
# 14. FIGURE GENERATION
# =========================================================
def _truncate_label(text: str, max_len: int = 38) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[:max_len - 1] + "…"

def make_figure_11_modeling(metrics: pd.DataFrame,
                             ebm_importance: pd.DataFrame,
                             rf_importance: pd.DataFrame,
                             partials: pd.DataFrame,
                             predictions: pd.DataFrame,
                             dpi: int = 500,
                             show_in_colab: bool = True) -> str:
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
        "svg.fonttype"     : "none",
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.labelsize"  : 5.5,
        "ytick.labelsize"  : 5.5,
    })

    C = {
        "bg"     : "#FAFAF8",
        "text_hd": "#111827",
        "text_sm": "#6B7280",
        "border" : "#D1D5DB",
        "blue"   : "#2166AC",
        "teal"   : "#1B9E77",
        "amber"  : "#D97706",
        "red"    : "#C0504D",
        "purple" : "#7B3294",
        "gray"   : "#6B7280",
    }

    fig = plt.figure(figsize=(7.2, 9.2))
    fig.patch.set_facecolor(C["bg"])

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        left=0.08, right=0.98, top=0.94, bottom=0.07,
        hspace=0.48, wspace=0.32,
    )

    ax_r2      = fig.add_subplot(gs[0, 0])
    ax_rmse    = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_ebm     = fig.add_subplot(gs[1, 1])
    ax_rf      = fig.add_subplot(gs[2, 0])
    ax_pe      = fig.add_subplot(gs[2, 1])

    for ax in [ax_r2, ax_rmse, ax_scatter, ax_ebm, ax_rf, ax_pe]:
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color(C["border"])

    metrics_plot = metrics.copy()
    order = metrics_plot["model"].tolist()
    palette = [C["blue"], C["amber"], C["teal"]][:len(order)]

    # a) OOF R²
    ax_r2.bar(order, metrics_plot["oof_r2"], color=palette,
              edgecolor="white", linewidth=0.4)
    ax_r2.set_ylabel("Out-of-fold R²", fontsize=6, color=C["text_sm"])
    ax_r2.set_title("Predictive performance (R²)", fontsize=7.2,
                    color=C["text_hd"], pad=4)
    ax_r2.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_r2.text(0.03, 0.97, "a", transform=ax_r2.transAxes,
               fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    for i, v in enumerate(metrics_plot["oof_r2"]):
        ax_r2.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                   fontsize=5.3, color=C["text_sm"])

    # b) OOF RMSE
    ax_rmse.bar(order, metrics_plot["oof_rmse"], color=palette,
                edgecolor="white", linewidth=0.4)
    ax_rmse.set_ylabel("Out-of-fold RMSE", fontsize=6, color=C["text_sm"])
    ax_rmse.set_title("Predictive performance (RMSE)", fontsize=7.2,
                      color=C["text_hd"], pad=4)
    ax_rmse.grid(axis="y", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_rmse.text(0.03, 0.97, "b", transform=ax_rmse.transAxes,
                 fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    for i, v in enumerate(metrics_plot["oof_rmse"]):
        ax_rmse.text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom",
                     fontsize=5.3, color=C["text_sm"])

    # c) Observed vs predicted
    best_model = metrics_plot.iloc[0]["model"]
    pred_col   = f"pred_{best_model}"
    obs  = pd.to_numeric(predictions["observed_disaster_observed_index"],
                         errors="coerce")
    pred = pd.to_numeric(predictions[pred_col], errors="coerce")
    valid = obs.notna() & pred.notna()

    ax_scatter.scatter(obs[valid], pred[valid],
                       s=5, alpha=0.35, linewidths=0, color=C["purple"])
    if valid.any():
        vmin = min(obs[valid].min(), pred[valid].min())
        vmax = max(obs[valid].max(), pred[valid].max())
        ax_scatter.plot([vmin, vmax], [vmin, vmax],
                        ls="--", lw=0.7, color=C["gray"])

    ax_scatter.set_xlabel("Observed disaster burden index",
                          fontsize=6, color=C["text_sm"])
    ax_scatter.set_ylabel(f"Predicted ({best_model})",
                          fontsize=6, color=C["text_sm"])
    ax_scatter.set_title("Observed vs predicted values",
                         fontsize=7.2, color=C["text_hd"], pad=4)
    ax_scatter.grid(linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_scatter.text(0.03, 0.97, "c", transform=ax_scatter.transAxes,
                    fontsize=9, fontweight="bold", va="top", color=C["text_hd"])
    ax_scatter.text(
        0.97, 0.05,
        f"Best: {best_model}\nOOF R² = {metrics_plot.iloc[0]['oof_r2']:.3f}",
        transform=ax_scatter.transAxes, ha="right", va="bottom",
        fontsize=5.4, color=C["text_hd"],
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec=C["border"], lw=0.5),
    )

    # d) EBM importance
    top_ebm = ebm_importance.head(10).iloc[::-1].copy()
    top_ebm["label"] = top_ebm["feature"].apply(_truncate_label)
    ax_ebm.barh(top_ebm["label"], top_ebm["importance"],
                color=C["blue"], edgecolor="white", linewidth=0.4)
    ax_ebm.set_xlabel("Importance", fontsize=6, color=C["text_sm"])
    ax_ebm.set_title("Top EBM features", fontsize=7.2,
                     color=C["text_hd"], pad=4)
    ax_ebm.grid(axis="x", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_ebm.text(0.03, 0.97, "d", transform=ax_ebm.transAxes,
                fontsize=9, fontweight="bold", va="top", color=C["text_hd"])

    # e) RF importance
    top_rf = rf_importance.head(10).iloc[::-1].copy()
    top_rf["label"] = top_rf["feature"].apply(_truncate_label)
    ax_rf.barh(top_rf["label"], top_rf["importance"],
               color=C["teal"], edgecolor="white", linewidth=0.4)
    ax_rf.set_xlabel("Importance", fontsize=6, color=C["text_sm"])
    ax_rf.set_title("Top Random Forest features", fontsize=7.2,
                    color=C["text_hd"], pad=4)
    ax_rf.grid(axis="x", linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)
    ax_rf.text(0.03, 0.97, "e", transform=ax_rf.transAxes,
               fontsize=9, fontweight="bold", va="top", color=C["text_hd"])

    # f) EBM partial effects (top feature)
    if partials.empty:
        ax_pe.text(0.5, 0.5, "No partial effects available",
                   ha="center", va="center", fontsize=7, color=C["text_sm"])
        ax_pe.set_xticks([]); ax_pe.set_yticks([])
    else:
        feat_to_plot = ebm_importance.iloc[0]["feature"]
        pe1 = partials.loc[partials["feature"] == feat_to_plot].copy()
        if pe1.empty:
            ax_pe.text(0.5, 0.5, "No plottable partial effects",
                       ha="center", va="center", fontsize=7, color=C["text_sm"])
            ax_pe.set_xticks([]); ax_pe.set_yticks([])
        else:
            pe1["x_num"] = pd.to_numeric(pe1["x"], errors="coerce")
            if pe1["x_num"].notna().sum() >= 3:
                pe1 = pe1.sort_values("x_num")
                ax_pe.plot(pe1["x_num"], pe1["effect"],
                           lw=1.2, color=C["red"])
                ax_pe.scatter(pe1["x_num"], pe1["effect"],
                              s=10, color=C["red"], linewidths=0)
                ax_pe.set_xlabel(_truncate_label(feat_to_plot, 30),
                                 fontsize=6, color=C["text_sm"])
            else:
                pe1 = pe1.reset_index(drop=True)
                ax_pe.plot(np.arange(len(pe1)), pe1["effect"],
                           lw=1.2, color=C["red"])
                ax_pe.scatter(np.arange(len(pe1)), pe1["effect"],
                              s=10, color=C["red"], linewidths=0)
                ax_pe.set_xlabel("Binned feature levels",
                                 fontsize=6, color=C["text_sm"])
            ax_pe.set_ylabel("EBM effect", fontsize=6, color=C["text_sm"])
            ax_pe.grid(linewidth=0.18, color=C["border"], alpha=0.7, zorder=0)

    ax_pe.set_title("EBM partial effect (top feature)", fontsize=7.2,
                    color=C["text_hd"], pad=4)
    ax_pe.text(0.03, 0.97, "f", transform=ax_pe.transAxes,
               fontsize=9, fontweight="bold", va="top", color=C["text_hd"])

    fig.suptitle(
        "Model performance and interpretation — municipal disaster burden (Brazil)",
        fontsize=8.5, color=C["text_hd"], y=0.975,
    )
    fig.text(
        0.5, 0.02,
        "Target: weighted per-capita disaster impact index (Module 10 v3.0) | "
        "Predictors include s2id_feat_* structural exposure features",
        ha="center", va="center", fontsize=5.6, color=C["text_sm"],
    )

    fig.savefig(OUTPUT_FIG_PNG, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_FIG_PDF, bbox_inches="tight",
                facecolor=fig.get_facecolor())

    if show_in_colab:
        try:
            from IPython.display import display
            display(fig)
        except Exception:
            plt.show()

    plt.close(fig)
    return str(OUTPUT_FIG_PNG)

# =========================================================
# 15. SAVE OUTPUTS
# =========================================================
def save_outputs(matrix, metrics, ebm_importance, rf_importance,
                 partials, predictions, features, n_obs):
    with tqdm(total=6, desc="Saving outputs", unit="file",
              colour="magenta", leave=False) as pbar:
        matrix.to_parquet(OUTPUT_MATRIX, index=False);           pbar.update(1)
        metrics.to_csv(OUTPUT_METRICS, index=False);             pbar.update(1)
        ebm_importance.to_csv(OUTPUT_EBM_IMPORTANCE, index=False); pbar.update(1)
        rf_importance.to_csv(OUTPUT_RF_IMPORTANCE, index=False); pbar.update(1)
        partials.to_parquet(OUTPUT_PARTIALS, index=False);       pbar.update(1)
        predictions.to_parquet(OUTPUT_PREDICTIONS, index=False); pbar.update(1)

    meta = {
        "project"               : "Flood Inequality Across Brazil",
        "module"                : "11_model_hazard_inequality_disaster.py",
        "version"               : "v2.1",
        "status"                : "completed",
        "created_at"            : datetime.now().isoformat(),
        "input_summary_path"    : str(INPUT_SUMMARY_PATH),
        "target_col"            : TARGET_COL,
        "n_observations"        : int(n_obs),
        "n_features"            : int(len(features)),
        "features"              : features,
        "cv_folds"              : CV_FOLDS,
        "output_matrix"         : str(OUTPUT_MATRIX),
        "output_metrics"        : str(OUTPUT_METRICS),
        "output_ebm_importance" : str(OUTPUT_EBM_IMPORTANCE),
        "output_rf_importance"  : str(OUTPUT_RF_IMPORTANCE),
        "output_partials"       : str(OUTPUT_PARTIALS),
        "output_predictions"    : str(OUTPUT_PREDICTIONS),
        "output_figure_png"     : str(OUTPUT_FIG_PNG),
        "output_figure_pdf"     : str(OUTPUT_FIG_PDF),
    }
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

# =========================================================
# 16. MAIN
# =========================================================
def main() -> None:
    print("\n" + "=" * 65)
    print("  Module 11 — Hazard × Inequality → Disaster Impact Modeling")
    print("  Version: v2.1 | s2id_feat_* features enabled")
    print("=" * 65 + "\n")

    # Purge stale checkpoints (target changed in Module 10 v3.0)
    purge_stale_checkpoints()

    # Load data
    with tqdm(total=1, desc="Loading input data", unit="file",
              colour="blue", leave=False) as pbar:
        df = load_input()
        pbar.update(1)

    config = read_config(CONFIG_PATH)

    # Build model matrix
    with tqdm(total=1, desc="Building model matrix", unit="step",
              colour="blue", leave=False) as pbar:
        X, y, features, matrix = build_model_matrix(df)
        pbar.update(1)

    # Report s2id_feat_ count
    s2id_feat_cols = [f for f in features if f.startswith("s2id_feat_")]
    log_summary(
        f"Model matrix | obs={len(X):,} | features={len(features)} "
        f"(incl. {len(s2id_feat_cols)} s2id_feat_*) | target={TARGET_COL}"
    )
    print(f"  Observations   : {len(X):,}")
    print(f"  Features total : {len(features)}")
    print(f"  → s2id_feat_*  : {len(s2id_feat_cols)}")
    print(f"  Target         : {TARGET_COL}")
    print(f"  Target mean    : {y.mean():.6f}  std: {y.std():.6f}  "
          f"skew: {y.skew():.3f}\n")

    # Train / evaluate
    models = build_models()
    metrics, fitted_models = evaluate_models(models, X, y)

    print("\n  ── Cross-validation results ──")
    print(metrics[["model", "cv_r2_mean", "cv_r2_std",
                   "oof_r2", "oof_mae", "oof_rmse"]].to_string(index=False))

    # Interpretation
    print("\n  Extracting EBM interpretation outputs…")
    ebm_importance, partials = extract_ebm_outputs(
        fitted_models["EBM"]["pipeline"], X, y, features
    )

    print("  Extracting RF feature importances…")
    rf_importance = extract_rf_importance(
        fitted_models["RandomForest"]["pipeline"], features
    )

    # Predictions table
    predictions = build_predictions_table(df, y, fitted_models)

    # Save outputs
    print("\n  Saving final outputs…")
    save_outputs(
        matrix=matrix, metrics=metrics,
        ebm_importance=ebm_importance, rf_importance=rf_importance,
        partials=partials, predictions=predictions,
        features=features, n_obs=len(X),
    )

    # Figure
    print("\n  Rendering publication-grade composite figure…")
    with tqdm(total=1, desc="Rendering figure", unit="fig",
              colour="magenta", leave=False) as pbar:
        fig_path = make_figure_11_modeling(
            metrics=metrics,
            ebm_importance=ebm_importance,
            rf_importance=rf_importance,
            partials=partials,
            predictions=predictions,
            dpi=500,
            show_in_colab=SHOW_FIG_IN_COLAB,
        )
        pbar.update(1)

    print(f"  Figure saved: {fig_path}")

    # Checkpoints
    if not KEEP_CHECKPOINTS:
        remove_checkpoints(list(fitted_models.keys()))

    # Catalog + config
    update_catalog(
        stage="11_model_hazard_inequality_disaster",
        tile_id="ALL",
        output_path=str(OUTPUT_METRICS),
        status="completed",
    )

    config["modeling_module_11"] = {
        "version"              : "v2.1",
        "name"                 : "hazard_inequality_disaster_models",
        "target"               : TARGET_COL,
        "metrics_csv"          : str(OUTPUT_METRICS),
        "ebm_importance_csv"   : str(OUTPUT_EBM_IMPORTANCE),
        "rf_importance_csv"    : str(OUTPUT_RF_IMPORTANCE),
        "partials_parquet"     : str(OUTPUT_PARTIALS),
        "predictions_parquet"  : str(OUTPUT_PREDICTIONS),
        "figure_png"           : str(OUTPUT_FIG_PNG),
        "figure_pdf"           : str(OUTPUT_FIG_PDF),
        "meta_json"            : str(OUTPUT_META),
        "n_observations"       : int(len(X)),
        "n_features"           : int(len(features)),
        "n_s2id_feat_features" : int(len(s2id_feat_cols)),
    }
    write_config(CONFIG_PATH, config)

    log_summary("=" * 60)
    log_summary(f"DONE | obs={len(X):,} | features={len(features)}")
    log_summary("Module 11 v2.1 completed.")

    print("\n" + "=" * 65)
    print("  ✓ Module 11 completed successfully.")
    print(f"  Outputs : {OUTPUT_DIR}")
    print(f"  Figure  : {OUTPUT_FIG_PNG}")
    print("=" * 65 + "\n")


# Colab: chama main() diretamente
main()
