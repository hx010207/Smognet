"""
Smog Net Evaluation, Interpretability, and Reporting
===================================================
Computes:
  - MAE, RMSE, R², MAPE on untouched test windows (Overall and Seasonal).
  - SHAP TreeExplainer analysis: beeswarm summary plot, interaction plots, and raw CSV.
  - XGBoost feature importance ranking (feature_importance.csv).
  - Robustness split side-by-side comparison (table1_model_comparison.md).
  - High-resolution publication Figure 1 (fig1_smognet_predictions.png).
  - Structured results.json with research_framing metadata.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import shap
import shap.explainers._tree as shap_tree
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# Ensure seamless compatibility between XGBoost 3.2+ string format and SHAP TreeExplainer
_orig_decode = shap_tree.decode_ubjson_buffer
def _safe_decode_ubjson_buffer(fd):
    jmodel = _orig_decode(fd)
    try:
        lmp = jmodel["learner"]["learner_model_param"]
        if "base_score" in lmp and isinstance(lmp["base_score"], str):
            lmp["base_score"] = float(lmp["base_score"].strip("[]"))
    except Exception:
        pass
    return jmodel

shap_tree.decode_ubjson_buffer = _safe_decode_ubjson_buffer

from features import SplitDataset
from models import SmogNetPipeline


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, R2, MAPE (%), and sMAPE (%) metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    # Clip near-zero true values to avoid division blowup in raw MAPE
    clipped_true = np.maximum(y_true, 1.0)
    mape = float(np.mean(np.abs((y_true - y_pred) / clipped_true)) * 100.0)
    
    # Symmetric MAPE (sMAPE): bounded 0-200%, immune to right-skewed zero-denominator artifact
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.maximum(denom, 1.0)
    smape = float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
        "sMAPE": round(smape, 2),
    }


def evaluate_all_models_on_split(
    pipeline: SmogNetPipeline,
    split: SplitDataset,
    split_name: str = "primary"
) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    """
    Evaluate Persistence, Linear Regression, RF, GBR, XGBoost, TabularMLP,
    SequenceLSTM, BiLSTM_Attn, CNN_BiLSTM_Attn, and the stacked Smog Net ensemble.
    """
    all_models = [
        "Persistence",
        "LinearRegression",
        "RandomForest",
        "GradientBoosting",
        "XGBoost",
        "TabularMLP",
        "SequenceLSTM",
    ]
    if pipeline.use_advanced_branches:
        all_models.extend(["BiLSTM_Attn", "CNN_BiLSTM_Attn"])
    all_models.append("SmogNet")

    dates_test = pd.to_datetime(split.dates_test)
    is_winter = dates_test.dt.month.isin([10, 11, 12, 1, 2]).values
    is_rest = ~is_winter

    predictions = {}
    for model_name in all_models:
        if model_name == "SmogNet":
            y_pred = pipeline.predict_smognet(split, subset="test")
        else:
            y_pred = pipeline.predict_base_learner(model_name, split, subset="test")
        predictions[model_name] = y_pred

    metrics_dict = {}
    for model_name, y_pred in predictions.items():
        overall = compute_metrics(split.y_test, y_pred)
        winter = compute_metrics(split.y_test[is_winter], y_pred[is_winter]) if np.any(is_winter) else {}
        rest = compute_metrics(split.y_test[is_rest], y_pred[is_rest]) if np.any(is_rest) else {}

        metrics_dict[model_name] = {
            "overall": overall,
            "winter_post_monsoon": winter,
            "rest_of_year": rest,
        }

    return metrics_dict, predictions


def plot_figure_1_predictions(
    dates: pd.Series,
    y_true: np.ndarray,
    y_smognet: np.ndarray,
    y_persistence: np.ndarray,
    output_path: Path
):
    """
    Generate publication-quality Figure 1: Predicted-vs-actual test-set line chart
    (Smog Net vs. Persistence baseline vs. Ground Truth).
    """
    dates_dt = pd.to_datetime(dates)
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

    # Plot lines
    ax.plot(dates_dt, y_true, label="Ground Truth (CPCB Observed)", color="#111827", linewidth=2.0, alpha=0.9, zorder=4)
    ax.plot(dates_dt, y_smognet, label="Smog Net (Hybrid Stacked Ensemble)", color="#dc2626", linewidth=1.8, linestyle="-", alpha=0.9, zorder=5)
    ax.plot(dates_dt, y_persistence, label="Persistence Baseline (Yesterday's PM2.5)", color="#9ca3af", linewidth=1.2, linestyle="--", alpha=0.8, zorder=3)

    # Highlight post-monsoon stubble burning period (Oct 15 - Nov 15)
    stubble_start = pd.Timestamp("2019-10-15")
    stubble_end = pd.Timestamp("2019-11-15")
    if stubble_start >= dates_dt.min() and stubble_end <= dates_dt.max():
        ax.axvspan(stubble_start, stubble_end, color="#fbbf24", alpha=0.25, label="Stubble Burning Window (Oct 15 - Nov 15)", zorder=1)
        ax.text(stubble_start + pd.Timedelta(days=1), ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 400,
                "Crop Burning Peak", fontsize=9, color="#b45309", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7", ec="#f59e0b", lw=0.8))

    ax.set_title("Smog Net: Next-Day PM2.5 Forecast Performance in Delhi (Held-Out Test Window)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Forecast Target Date", fontsize=11, labelpad=10)
    ax.set_ylabel("PM2.5 Concentration (µg/m³)", fontsize=11, labelpad=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#d1d5db", fontsize=9.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Evaluate] Figure 1 saved to {output_path}")


def run_shap_analysis(
    pipeline: SmogNetPipeline,
    split: SplitDataset,
    output_dir: Path
) -> Tuple[pd.DataFrame, float, float]:
    """
    Run SHAP TreeExplainer on fitted XGBoost model over the test set.
    Generates:
      - shap_summary.png (beeswarm summary plot of top 15 features)
      - shap_values.csv (raw SHAP values)
      - shap_interactions.png (joint interaction effect between stubble burning, wind speed, humidity)
    Computes:
      - mean absolute SHAP value for is_stubble_burning
      - counterfactual delta PM2.5 attributable to stubble burning
    """
    xgb_model = pipeline.models["XGBoost"]
    X_test_df = split.X_test

    print("[SHAP] Computing TreeExplainer SHAP values over test set...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test_df)

    # Save raw SHAP values to CSV
    shap_df = pd.DataFrame(shap_values.values, columns=split.feature_names)
    shap_csv_path = output_dir / "shap_values.csv"
    shap_df.to_csv(shap_csv_path, index=False)
    print(f"[SHAP] Raw SHAP values saved to {shap_csv_path}")

    # Generate beeswarm summary plot (top 15 features)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    shap.summary_plot(shap_values.values, X_test_df, max_display=15, show=False)
    plt.title("SHAP Beeswarm Summary: Top 15 Predictors of Next-Day PM2.5 (XGBoost)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    summary_path = output_dir / "shap_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Beeswarm summary saved to {summary_path}")

    # Feature importances CSV from XGBoost
    importances = xgb_model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": split.feature_names,
        "importance": importances,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_path = output_dir / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    print(f"[SHAP] Feature importance table saved to {fi_path}")

    # Interaction / Joint effect plot (is_stubble_burning, wind_speed, humidity)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    stubble_idx = split.feature_names.index("is_stubble_burning")
    ws_idx = split.feature_names.index("wind_speed")
    hum_idx = split.feature_names.index("humidity")
    
    # Left: Stubble burning SHAP colored by Wind Speed
    sc1 = ax1.scatter(
        X_test_df["is_stubble_burning"],
        shap_values.values[:, stubble_idx],
        c=X_test_df["wind_speed"],
        cmap="coolwarm_r",
        alpha=0.8,
        s=45,
        edgecolors="none"
    )
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label("Wind Speed WS10M (m/s)", fontsize=9.5)
    ax1.set_title("Stubble Burning Impact vs. Wind Dispersion", fontsize=11, fontweight="bold")
    ax1.set_xlabel("is_stubble_burning Flag (0 = No, 1 = Oct 15-Nov 15)", fontsize=10)
    ax1.set_ylabel("SHAP Value for is_stubble_burning (µg/m³)", fontsize=10)
    ax1.set_xticks([0, 1])
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Right: Stubble burning SHAP colored by Humidity
    sc2 = ax2.scatter(
        X_test_df["is_stubble_burning"],
        shap_values.values[:, stubble_idx],
        c=X_test_df["humidity"],
        cmap="YlGnBu",
        alpha=0.8,
        s=45,
        edgecolors="none"
    )
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label("Relative Humidity RH2M (%)", fontsize=9.5)
    ax2.set_title("Stubble Burning Impact vs. Moisture Accumulation", fontsize=11, fontweight="bold")
    ax2.set_xlabel("is_stubble_burning Flag (0 = No, 1 = Oct 15-Nov 15)", fontsize=10)
    ax2.set_ylabel("SHAP Value for is_stubble_burning (µg/m³)", fontsize=10)
    ax2.set_xticks([0, 1])
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Atmospheric Modulation of Crop-Residue Burning: SHAP Joint Dynamics", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    interaction_path = output_dir / "shap_interactions.png"
    plt.savefig(interaction_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Interaction plot saved to {interaction_path}")

    # Compute exact scientific metrics for research framing
    mean_abs_shap_stubble = float(np.abs(shap_values.values[:, stubble_idx]).mean())

    # Counterfactual PM2.5 delta: predicted PM2.5 with is_stubble_burning=1 vs 0, holding other features at median
    median_profile = X_test_df.median().to_dict()
    profile_stubble_1 = pd.DataFrame([median_profile])
    profile_stubble_0 = pd.DataFrame([median_profile])
    profile_stubble_1["is_stubble_burning"] = 1
    profile_stubble_0["is_stubble_burning"] = 0

    pred_1 = float(xgb_model.predict(profile_stubble_1)[0])
    pred_0 = float(xgb_model.predict(profile_stubble_0)[0])
    counterfactual_delta = round(pred_1 - pred_0, 2)

    return fi_df, round(mean_abs_shap_stubble, 2), counterfactual_delta


def generate_table1_markdown(
    primary_metrics: Dict[str, Dict],
    split_2019_metrics: Dict[str, Dict],
    split_2020_metrics: Dict[str, Dict],
    output_path: Path
):
    """
    Generate formatted Table 1: Model Comparison across Primary and Robustness Splits.
    """
    models = [
        ("Persistence Baseline", "Persistence"),
        ("Linear Regression", "LinearRegression"),
        ("Random Forest", "RandomForest"),
        ("Gradient Boosting", "GradientBoosting"),
        ("XGBoost", "XGBoost"),
        ("Sequence LSTM (2-Layer)", "SequenceLSTM"),
        ("Tabular MLP", "TabularMLP"),
        ("**Smog Net (Stacked Ensemble)**", "SmogNet"),
    ]

    lines = [
        "# Table 1: Comparative Next-Day PM2.5 Forecasting Performance in Delhi\n",
        "Evaluated on untouched test sets: Primary chronological split (2019–2020), 2019 full held-out calendar year, and 2020 COVID/monsoon window.\n",
        "| Model Architecture | Primary Test MAE | Primary Test RMSE | Primary Test R² | Primary Test MAPE (%) | 2019 Held-Out MAE | 2019 Held-Out RMSE | 2020 Held-Out MAE | 2020 Held-Out RMSE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for display_name, key in models:
        p_m = primary_metrics[key]["overall"]
        m19 = split_2019_metrics[key]["overall"]
        m20 = split_2020_metrics[key]["overall"]

        line = (
            f"| {display_name} | {p_m['MAE']:.2f} | {p_m['RMSE']:.2f} | {p_m['R2']:.4f} | {p_m['MAPE']:.2f}% | "
            f"{m19['MAE']:.2f} | {m19['RMSE']:.2f} | {m20['MAE']:.2f} | {m20['RMSE']:.2f} |"
        )
        lines.append(line)

    lines.append("\n### Seasonal Breakdown on Primary Test Set (Oct–Feb Winter/Post-Monsoon vs. Mar–Sep Rest of Year)\n")
    lines.append("| Model Architecture | Winter MAE | Winter RMSE | Winter R² | Rest-of-Year MAE | Rest-of-Year RMSE | Rest-of-Year R² |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for display_name, key in models:
        w_m = primary_metrics[key]["winter_post_monsoon"]
        r_m = primary_metrics[key]["rest_of_year"]
        line = (
            f"| {display_name} | {w_m.get('MAE', 0.0):.2f} | {w_m.get('RMSE', 0.0):.2f} | {w_m.get('R2', 0.0):.4f} | "
            f"{r_m.get('MAE', 0.0):.2f} | {r_m.get('RMSE', 0.0):.2f} | {r_m.get('R2', 0.0):.4f} |"
        )
        lines.append(line)

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Evaluate] Table 1 markdown saved to {output_path}")
