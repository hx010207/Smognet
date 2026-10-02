"""
generate_all_publication_results.py
===================================
Consolidates all experimental results into a structured results/ directory:
  results/
  ├── 01_model_comparison/
  ├── 02_feature_analysis/
  ├── 03_forecast_visualization/
  ├── 04_spatial_generalization/
  ├── 05_robustness/
  ├── 06_literature_context/
  ├── 07_computational/
  └── README.md

Generates high-resolution, publication-ready figures (>= 300 DPI) using a unified
visual style, coherent color palette, and clear axis labels, titles, and legends.
Saves all underlying data alongside each figure as CSV / JSON.
"""

import os
import shutil
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# -----------------------------------------------------------------------------
# 1. UNIFIED PUBLICATION STYLE & COLOR PALETTE
# -----------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.linestyle": ":",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 1.0,
})

PALETTE = {
    "smognet": "#dc2626",          # Crimson / Coral for Smog Net
    "persistence": "#64748b",      # Slate Gray for baseline
    "ground_truth": "#0f172a",     # Near Black
    "xgboost": "#059669",          # Forest Green
    "rf": "#2563eb",               # Royal Blue
    "gbr": "#0891b2",              # Cyan / Blue
    "tabular_mlp": "#d97706",      # Amber
    "bilstm": "#7c3aed",           # Purple
    "cnn_bilstm": "#9333ea",       # Violet
    "seq_lstm": "#c026d3",         # Magenta
    "linear": "#ea580c",           # Orange
    "sarima": "#475569",           # Dark Slate
    "prophet": "#6b7280",          # Medium Gray
    "stubble_gold": "#f59e0b",     # Stubble burning gold
    "stubble_fill": "#fef3c7",     # Soft yellow highlight
    "box_bg": "#f8fafc",
    "box_edge": "#cbd5e1",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# 2. CATEGORY 01: MODEL COMPARISON
# -----------------------------------------------------------------------------
def generate_01_model_comparison(output_dir: Path):
    print("Generating 01_model_comparison...")
    d = ensure_dir(output_dir / "01_model_comparison")

    # Table 1 Expanding-Window Walk-Forward Cross-Validation Data (2016-2020)
    data = [
        {"Category": "Heuristic Baseline", "Model": "Persistence", "2016_RMSE": 44.9, "2016_MAE": 29.8, "2017_RMSE": 46.9, "2017_MAE": 28.4, "2018_RMSE": 37.5, "2018_MAE": 23.4, "2019_RMSE": 46.4, "2019_MAE": 27.9, "2020_RMSE": 29.4, "2020_MAE": 19.8, "Mean_RMSE": 41.02, "Std_RMSE": 7.48, "Mean_MAE": 25.85, "Std_MAE": 4.17, "Mean_R2": 0.7408, "Std_R2": 0.0365, "Mean_sMAPE": 23.61, "Std_sMAPE": 2.32},
        {"Category": "Classical Time Series", "Model": "SARIMA (seasonal)", "2016_RMSE": 118.8, "2016_MAE": 104.5, "2017_RMSE": 115.2, "2017_MAE": 80.9, "2018_RMSE": 109.7, "2018_MAE": 80.6, "2019_RMSE": 86.5, "2019_MAE": 59.2, "2020_RMSE": 55.5, "2020_MAE": 41.6, "Mean_RMSE": 97.14, "Std_RMSE": 26.45, "Mean_MAE": 73.35, "Std_MAE": 23.91, "Mean_R2": -0.4384, "Std_R2": 0.4635, "Mean_sMAPE": 66.92, "Std_sMAPE": 14.33},
        {"Category": "Classical Time Series", "Model": "Meta Prophet", "2016_RMSE": 207.8, "2016_MAE": 112.0, "2017_RMSE": 81.9, "2017_MAE": 50.7, "2018_RMSE": 67.2, "2018_MAE": 44.5, "2019_RMSE": 53.6, "2019_MAE": 36.4, "2020_RMSE": 41.6, "2020_MAE": 34.2, "Mean_RMSE": 90.41, "Std_RMSE": 67.30, "Mean_MAE": 55.57, "Std_MAE": 32.22, "Mean_R2": -0.3818, "Std_R2": 1.6590, "Mean_sMAPE": 42.59, "Std_sMAPE": 7.58},
        {"Category": "Linear Baseline", "Model": "LinearRegression", "2016_RMSE": 233.2, "2016_MAE": 68.3, "2017_RMSE": 48.5, "2017_MAE": 29.4, "2018_RMSE": 38.1, "2018_MAE": 24.4, "2019_RMSE": 44.4, "2019_MAE": 27.7, "2020_RMSE": 29.7, "2020_MAE": 22.8, "Mean_RMSE": 78.78, "Std_RMSE": 86.64, "Mean_MAE": 34.51, "Std_MAE": 19.06, "Mean_R2": -0.3119, "Std_R2": 2.3185, "Mean_sMAPE": 26.69, "Std_sMAPE": 3.51},
        {"Category": "Tree Ensemble", "Model": "RandomForest", "2016_RMSE": 68.2, "2016_MAE": 41.0, "2017_RMSE": 54.6, "2017_MAE": 30.3, "2018_RMSE": 35.9, "2018_MAE": 21.8, "2019_RMSE": 43.2, "2019_MAE": 26.7, "2020_RMSE": 25.4, "2020_MAE": 17.1, "Mean_RMSE": 45.48, "Std_RMSE": 16.59, "Mean_MAE": 27.39, "Std_MAE": 9.12, "Mean_R2": 0.6942, "Std_R2": 0.1109, "Mean_sMAPE": 23.99, "Std_sMAPE": 3.22},
        {"Category": "Tree Ensemble", "Model": "GradientBoosting", "2016_RMSE": 66.9, "2016_MAE": 41.1, "2017_RMSE": 57.3, "2017_MAE": 31.9, "2018_RMSE": 34.3, "2018_MAE": 21.6, "2019_RMSE": 40.3, "2019_MAE": 25.6, "2020_RMSE": 23.8, "2020_MAE": 16.7, "Mean_RMSE": 44.53, "Std_RMSE": 17.43, "Mean_MAE": 27.38, "Std_MAE": 9.48, "Mean_R2": 0.7060, "Std_R2": 0.1254, "Mean_sMAPE": 24.44, "Std_sMAPE": 3.38},
        {"Category": "Tree Ensemble", "Model": "Tuned XGBoost", "2016_RMSE": 68.3, "2016_MAE": 39.9, "2017_RMSE": 58.1, "2017_MAE": 32.3, "2018_RMSE": 33.6, "2018_MAE": 20.9, "2019_RMSE": 39.0, "2019_MAE": 25.3, "2020_RMSE": 23.7, "2020_MAE": 16.2, "Mean_RMSE": 44.53, "Std_RMSE": 18.25, "Mean_MAE": 26.91, "Std_MAE": 9.35, "Mean_R2": 0.7049, "Std_R2": 0.1378, "Mean_sMAPE": 23.92, "Std_sMAPE": 3.30},
        {"Category": "Deep Learning", "Model": "TabularMLP", "2016_RMSE": 93.8, "2016_MAE": 55.2, "2017_RMSE": 56.3, "2017_MAE": 34.6, "2018_RMSE": 35.0, "2018_MAE": 23.1, "2019_RMSE": 40.1, "2019_MAE": 26.1, "2020_RMSE": 27.0, "2020_MAE": 21.3, "Mean_RMSE": 50.45, "Std_RMSE": 26.50, "Mean_MAE": 32.06, "Std_MAE": 13.92, "Mean_R2": 0.6104, "Std_R2": 0.2858, "Mean_sMAPE": 28.73, "Std_sMAPE": 5.67},
        {"Category": "Deep Learning", "Model": "SequenceLSTM", "2016_RMSE": 94.6, "2016_MAE": 56.8, "2017_RMSE": 61.2, "2017_MAE": 37.1, "2018_RMSE": 41.6, "2018_MAE": 27.2, "2019_RMSE": 41.1, "2019_MAE": 27.0, "2020_RMSE": 29.6, "2020_MAE": 21.2, "Mean_RMSE": 53.63, "Std_RMSE": 25.56, "Mean_MAE": 33.86, "Std_MAE": 14.02, "Mean_R2": 0.5635, "Std_R2": 0.2728, "Mean_sMAPE": 30.29, "Std_sMAPE": 5.12},
        {"Category": "Deep Learning", "Model": "BiLSTM + Attention", "2016_RMSE": 61.8, "2016_MAE": 36.4, "2017_RMSE": 56.2, "2017_MAE": 34.1, "2018_RMSE": 37.6, "2018_MAE": 25.1, "2019_RMSE": 39.8, "2019_MAE": 26.1, "2020_RMSE": 27.1, "2020_MAE": 20.2, "Mean_RMSE": 44.49, "Std_RMSE": 14.19, "Mean_MAE": 28.36, "Std_MAE": 6.71, "Mean_R2": 0.7045, "Std_R2": 0.0856, "Mean_sMAPE": 26.63, "Std_sMAPE": 2.55},
        {"Category": "Deep Learning", "Model": "CNN-BiLSTM + Attention", "2016_RMSE": 63.6, "2016_MAE": 41.2, "2017_RMSE": 62.9, "2017_MAE": 41.4, "2018_RMSE": 37.4, "2018_MAE": 24.0, "2019_RMSE": 41.7, "2019_MAE": 26.5, "2020_RMSE": 29.0, "2020_MAE": 22.9, "Mean_RMSE": 46.93, "Std_RMSE": 15.62, "Mean_MAE": 31.20, "Std_MAE": 9.30, "Mean_R2": 0.6692, "Std_R2": 0.1143, "Mean_sMAPE": 29.06, "Std_sMAPE": 6.40},
        {"Category": "Proposed Hybrid", "Model": "Smog Net", "2016_RMSE": 52.7, "2016_MAE": 34.4, "2017_RMSE": 53.7, "2017_MAE": 31.9, "2018_RMSE": 33.4, "2018_MAE": 20.6, "2019_RMSE": 38.4, "2019_MAE": 24.2, "2020_RMSE": 23.8, "2020_MAE": 17.4, "Mean_RMSE": 40.40, "Std_RMSE": 12.80, "Mean_MAE": 25.68, "Std_MAE": 7.28, "Mean_R2": 0.7565, "Std_R2": 0.0751, "Mean_sMAPE": 23.85, "Std_sMAPE": 3.63},
    ]
    df = pd.DataFrame(data)
    df.to_csv(d / "model_comparison_table.csv", index=False)

    # 1. Line Chart: walk_forward_rmse_by_year.png
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    years = [2016, 2017, 2018, 2019, 2020]

    model_colors = {
        "Smog Net": (PALETTE["smognet"], "o", 3.0, 8, 10),
        "Persistence": (PALETTE["persistence"], "s", 1.8, 6, 8),
        "Tuned XGBoost": (PALETTE["xgboost"], "D", 1.6, 6, 7),
        "RandomForest": (PALETTE["rf"], "^", 1.4, 5, 5),
        "GradientBoosting": (PALETTE["gbr"], "v", 1.4, 5, 5),
        "BiLSTM + Attention": (PALETTE["bilstm"], "p", 1.5, 6, 6),
        "CNN-BiLSTM + Attention": (PALETTE["cnn_bilstm"], "<", 1.3, 5, 4),
        "TabularMLP": (PALETTE["tabular_mlp"], "h", 1.3, 5, 4),
        "SequenceLSTM": (PALETTE["seq_lstm"], "x", 1.3, 5, 3),
        "LinearRegression": (PALETTE["linear"], "*", 1.2, 5, 2),
        "SARIMA (seasonal)": (PALETTE["sarima"], "--", 1.2, 4, 1),
        "Meta Prophet": (PALETTE["prophet"], "-.", 1.2, 4, 1),
    }

    for row in data:
        m = row["Model"]
        if m not in model_colors:
            continue
        c, marker, lw, ms, z = model_colors[m]
        vals = [row[f"{yr}_RMSE"] for yr in years]
        ls = "--" if marker in ["--", "-."] else "-"
        m_style = "o" if marker in ["--", "-."] else marker
        ax.plot(years, vals, label=m, color=c, marker=m_style, linestyle=ls, linewidth=lw, markersize=ms, zorder=z)

    ax.set_title("Walk-Forward Test RMSE Across 5 Expanding-Window Test Years (2016–2020)", pad=12)
    ax.set_xlabel("Held-Out Annual Test Regimes")
    ax.set_ylabel("Root Mean Squared Error (µg/m³)")
    ax.set_xticks(years)
    ax.set_ylim(15, 130)  # zoom on meaningful region, note outliers in text
    ax.text(2016.05, 122, "Note: Linear (233.2) & Prophet (207.8) extend off-scale in 2016 shock", fontsize=8.5, color="#64748b", fontstyle="italic", bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1", lw=0.6))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")
    plt.tight_layout()
    fig.savefig(d / "walk_forward_rmse_by_year.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Bar Chart: model_comparison_bars.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    plot_models = [r for r in data if r["Model"] not in ["LinearRegression", "SARIMA (seasonal)", "Meta Prophet"]]  # drop extreme divergees for clean bar view
    plot_models.append([r for r in data if r["Model"] == "LinearRegression"][0])
    
    # Sort by Mean RMSE
    plot_models = sorted(plot_models, key=lambda x: x["Mean_RMSE"])
    m_names = [r["Model"] for r in plot_models]
    rmse_means = [r["Mean_RMSE"] for r in plot_models]
    rmse_stds = [r["Std_RMSE"] for r in plot_models]
    mae_means = [r["Mean_MAE"] for r in plot_models]
    mae_stds = [r["Std_MAE"] for r in plot_models]
    r2_means = [r["Mean_R2"] for r in plot_models]
    r2_stds = [r["Std_R2"] for r in plot_models]

    colors = [PALETTE["smognet"] if m == "Smog Net" else ("#94a3b8" if m == "Persistence" else "#3b82f6") for m in m_names]

    y_pos = np.arange(len(m_names))
    ax1.barh(y_pos, rmse_means, xerr=rmse_stds, color=colors, alpha=0.85, edgecolor="#1e293b", linewidth=0.8, capsize=3.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(m_names)
    ax1.set_xlabel("Mean Test RMSE ± Std (µg/m³)")
    ax1.set_title("5-Fold Cross-Validation: Mean RMSE (Lower is Better)")
    ax1.invert_yaxis()
    for idx, (val, std) in enumerate(zip(rmse_means, rmse_stds)):
        ax1.text(val + std + 1.2, idx, f"{val:.1f}±{std:.1f}", va="center", fontsize=8.5, fontweight="bold" if m_names[idx] == "Smog Net" else "normal")

    ax2.barh(y_pos, r2_means, xerr=r2_stds, color=colors, alpha=0.85, edgecolor="#1e293b", linewidth=0.8, capsize=3.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(["" for _ in m_names])
    ax2.set_xlabel("Mean Test R² ± Std (Higher is Better)")
    ax2.set_title("5-Fold Cross-Validation: Mean R²")
    ax2.set_xlim(-0.4, 0.9)
    ax2.invert_yaxis()
    for idx, (val, std) in enumerate(zip(r2_means, r2_stds)):
        ax2.text(val + std + 0.02, idx, f"{val:.3f}", va="center", fontsize=8.5, fontweight="bold" if m_names[idx] == "Smog Net" else "normal")

    plt.suptitle("Smog Net vs. Competitive Baselines: Expanding Walk-Forward Benchmark (2016–2020)", y=1.02)
    plt.tight_layout()
    fig.savefig(d / "model_comparison_bars.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Radar Chart: model_ranking_radar.png
    radar_models = ["Smog Net", "Persistence", "Tuned XGBoost", "BiLSTM + Attention", "RandomForest", "TabularMLP"]
    categories = ["RMSE Metric", "MAE Metric", "R² Metric", "sMAPE Metric", "Inference Speed"]
    
    # Latency map in ms
    latencies = {"Smog Net": 1.42, "Persistence": 0.05, "Tuned XGBoost": 0.45, "BiLSTM + Attention": 4.10, "RandomForest": 0.85, "TabularMLP": 1.10}

    # Normalize metrics 0 to 1 (1.0 = best across models)
    radar_raw = {}
    for rm in radar_models:
        row = [r for r in data if r["Model"] == rm][0]
        radar_raw[rm] = {
            "RMSE": row["Mean_RMSE"],
            "MAE": row["Mean_MAE"],
            "R2": row["Mean_R2"],
            "sMAPE": row["Mean_sMAPE"],
            "Latency": latencies[rm],
        }

    all_rmse = [v["RMSE"] for v in radar_raw.values()]
    all_mae = [v["MAE"] for v in radar_raw.values()]
    all_r2 = [v["R2"] for v in radar_raw.values()]
    all_smape = [v["sMAPE"] for v in radar_raw.values()]
    all_lat = [v["Latency"] for v in radar_raw.values()]

    radar_scores = {}
    for rm, v in radar_raw.items():
        score_rmse = (max(all_rmse) - v["RMSE"]) / (max(all_rmse) - min(all_rmse) + 1e-6)
        score_mae = (max(all_mae) - v["MAE"]) / (max(all_mae) - min(all_mae) + 1e-6)
        score_r2 = (v["R2"] - min(all_r2)) / (max(all_r2) - min(all_r2) + 1e-6)
        score_smape = (max(all_smape) - v["sMAPE"]) / (max(all_smape) - min(all_smape) + 1e-6)
        score_lat = (max(all_lat) - v["Latency"]) / (max(all_lat) - min(all_lat) + 1e-6)
        radar_scores[rm] = [score_rmse, score_mae, score_r2, score_smape, score_lat]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)
    plt.xticks(angles[:-1], categories, color="#1e293b", size=10.5, fontweight="bold")
    ax.set_rlabel_position(30)
    plt.yticks([0.25, 0.50, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00 (Best)"], color="#64748b", size=8.5)
    plt.ylim(0, 1.05)

    r_colors = {
        "Smog Net": (PALETTE["smognet"], 2.8, 0.25),
        "Persistence": (PALETTE["persistence"], 1.8, 0.05),
        "Tuned XGBoost": (PALETTE["xgboost"], 1.8, 0.05),
        "BiLSTM + Attention": (PALETTE["bilstm"], 1.5, 0.05),
        "RandomForest": (PALETTE["rf"], 1.5, 0.05),
        "TabularMLP": (PALETTE["tabular_mlp"], 1.5, 0.05),
    }

    for rm in radar_models:
        vals = radar_scores[rm]
        vals += vals[:1]
        col, lw, alpha = r_colors[rm]
        ax.plot(angles, vals, linewidth=lw, linestyle="solid", label=rm, color=col)
        ax.fill(angles, vals, color=col, alpha=alpha)

    plt.title("Multi-Objective Performance Radar: Trade-offs Across Accuracy, Error Metrics & Latency\n(Normalized 0.0 to 1.0; Outer Perimeter Represents Optimal)", size=12, fontweight="bold", pad=25)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=True, edgecolor="#cbd5e1")
    plt.tight_layout()
    fig.savefig(d / "model_ranking_radar.png", bbox_inches="tight")
    plt.close(fig)

    print("  [OK] 01_model_comparison files generated successfully.")


# -----------------------------------------------------------------------------
# 3. CATEGORY 02: FEATURE ANALYSIS
# -----------------------------------------------------------------------------
def generate_02_feature_analysis(output_dir: Path):
    print("Generating 02_feature_analysis...")
    d = ensure_dir(output_dir / "02_feature_analysis")

    # 1. Feature Group Ablation Donut Chart: feature_group_ablation_pie.png
    ablation_data = [
        {"Group": "Crop-Residue Stubble Burning & Calendar Flags", "Delta_RMSE": 1.96, "Features": 5, "Share_Pct": 53.7, "Description": "is_stubble_burning, upwind_transport, stubble_x_ws, diwali/lockdown"},
        {"Group": "Multi-Day Lags & Autoregressive Rolling Stats", "Delta_RMSE": 0.91, "Features": 7, "Share_Pct": 24.9, "Description": "pm25_roll3_mean/std, roll7_mean/std, lag2, lag3, lag7"},
        {"Group": "Surface Meteorology & Thermal Inversion Proxies", "Delta_RMSE": 0.78, "Features": 7, "Share_Pct": 21.4, "Description": "wind_speed, wind_dir, humidity, temp, temp_inversion_proxy"},
    ]
    df_abl = pd.DataFrame(ablation_data)
    df_abl.to_csv(d / "feature_group_ablations.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    pie_colors = ["#dc2626", "#0284c7", "#10b981"]
    wedges, texts, autotexts = ax.pie(
        df_abl["Share_Pct"],
        labels=df_abl["Group"],
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        colors=pie_colors,
        wedgeprops=dict(width=0.45, edgecolor="#ffffff", linewidth=2.5),
        textprops=dict(color="#0f172a", fontsize=9.5, fontweight="bold"),
    )
    for at in autotexts:
        at.set_color("#ffffff")
        at.set_fontsize(10.5)
        at.set_fontweight("bold")

    ax.set_title("Share of Explainable Error Reduction by Feature Group\n(Denominator: Total Explained ΔRMSE = 3.65 µg/m³; Sum = 100%)", pad=15)
    plt.tight_layout()
    fig.savefig(d / "feature_group_ablation_pie.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Horizontal Feature Importance Bar: feature_importance_bar.png
    fi_orig = pd.read_csv("feature_importance.csv")
    fi_top15 = fi_orig.head(15).copy()
    fi_top15.to_csv(d / "top15_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    
    def get_color(name):
        n = name.lower()
        if "stubble" in n or "diwali" in n:
            return "#dc2626"
        elif "wind" in n or "temp" in n or "humid" in n:
            return "#0284c7"
        elif "gurugram" in n or "spatial" in n:
            return "#d97706"
        else:
            return "#10b981"

    bar_colors = [get_color(f) for f in fi_top15["feature"]][::-1]
    y_pos = np.arange(len(fi_top15))[::-1]

    bars = ax.barh(y_pos, fi_top15["importance"] * 100, color=bar_colors, edgecolor="#1e293b", linewidth=0.7, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fi_top15["feature"][::-1], fontsize=9)
    ax.set_xlabel("Relative Feature Importance (% of Total Split Gain)")
    ax.set_title("Top 15 Most Influential Features in Smog Net's Gradient Boosting Branch")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=8.5, fontweight="bold", color="#1e293b")

    # Domain legend
    handles = [
        mpatches.Patch(color="#10b981", label="Chemical Telemetry & Autoregressive Lags"),
        mpatches.Patch(color="#dc2626", label="Agricultural Stubble Burning / Calendar"),
        mpatches.Patch(color="#0284c7", label="Synoptic Surface Meteorology"),
        mpatches.Patch(color="#d97706", label="Spatial Airshed Neighbor (Gurugram)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, edgecolor="#cbd5e1", fontsize=9)
    plt.tight_layout()
    fig.savefig(d / "feature_importance_bar.png", bbox_inches="tight")
    plt.close(fig)

    # Copy SHAP plots if present
    for src, dst in [
        ("shap_summary.png", "shap_summary_beeswarm.png"),
        ("shap_interactions.png", "shap_interactions_stubble_wind_humidity.png"),
        ("shap_firms_summary.png", "shap_firms_summary.png"),
    ]:
        if Path(src).exists():
            shutil.copy(src, d / dst)
            print(f"  Copied {src} -> {d / dst}")

    print("  [OK] 02_feature_analysis files generated successfully.")


# -----------------------------------------------------------------------------
# 4. CATEGORY 03: FORECAST VISUALIZATION
# -----------------------------------------------------------------------------
def generate_03_forecast_visualization(output_dir: Path):
    print("Generating 03_forecast_visualization...")
    d = ensure_dir(output_dir / "03_forecast_visualization")

    # Copy test_window_forecast_vs_actual.png from fig1_smognet_predictions.png if present
    if Path("fig1_smognet_predictions.png").exists():
        shutil.copy("fig1_smognet_predictions.png", d / "test_window_forecast_vs_actual.png")

    # 1. Multi-Horizon Forecast Curve: multi_horizon_forecast_curve.png
    horizon_data = [
        {"Horizon": "t+1 (Next-Day)", "Days": 1, "Persistence_RMSE": 41.02, "Persistence_R2": 0.7408, "XGBoost_RMSE": 44.53, "XGBoost_R2": 0.7049, "SmogNet_RMSE": 43.34, "SmogNet_R2": 0.7212, "SmogNet_MAE": 26.58, "SmogNet_sMAPE": 23.91, "Outcome": "Persistence wins (Autocorrelation r ≈ 0.86 dominates)"},
        {"Horizon": "t+3 (3 Days Ahead)", "Days": 3, "Persistence_RMSE": 61.74, "Persistence_R2": 0.4098, "XGBoost_RMSE": 55.10, "XGBoost_R2": 0.5401, "SmogNet_RMSE": 59.76, "SmogNet_R2": 0.4631, "SmogNet_MAE": 39.18, "SmogNet_sMAPE": 34.24, "Outcome": "Smog Net outperforms Persistence by +1.98 µg/m³"},
        {"Horizon": "t+7 (7 Days Ahead)", "Days": 7, "Persistence_RMSE": 72.93, "Persistence_R2": 0.1813, "XGBoost_RMSE": 57.63, "XGBoost_R2": 0.4812, "SmogNet_RMSE": 58.09, "SmogNet_R2": 0.4957, "SmogNet_MAE": 38.28, "SmogNet_sMAPE": 33.82, "Outcome": "Smog Net outperforms Persistence by +14.84 µg/m³"},
    ]
    df_hor = pd.DataFrame(horizon_data)
    df_hor.to_csv(d / "multi_horizon_forecast_metrics.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    days = [1, 3, 7]

    # Panel A: RMSE vs Horizon
    ax1.plot(days, df_hor["Persistence_RMSE"], marker="s", color=PALETTE["persistence"], linewidth=2.0, markersize=8, label="Persistence Baseline")
    ax1.plot(days, df_hor["XGBoost_RMSE"], marker="^", color=PALETTE["xgboost"], linewidth=2.0, markersize=8, label="Single XGBoost")
    ax1.plot(days, df_hor["SmogNet_RMSE"], marker="o", color=PALETTE["smognet"], linewidth=2.5, markersize=9, label="Smog Net (Stacked Ensemble)")
    ax1.set_title("Forecast Error (RMSE) vs. Horizon Progression", pad=10)
    ax1.set_xlabel("Forecast Horizon Lead Time (Days)")
    ax1.set_ylabel("Root Mean Squared Error (µg/m³)")
    ax1.set_xticks(days)
    ax1.set_xticklabels(["t+1 Day", "t+3 Days", "t+7 Days"])
    ax1.legend(loc="upper left", frameon=True, edgecolor="#cbd5e1")

    # Annotate t+1 vs t+7 delta
    ax1.annotate("Persistence wins at t+1\n(Autocorrelation r≈0.86)", xy=(1, 41.02), xytext=(1.4, 33),
                 arrowprops=dict(arrowstyle="->", color="#475569", lw=1.0), fontsize=8.5, fontweight="bold", color="#475569")
    ax1.annotate("Smog Net buffer expands:\n+14.84 µg/m³ at t+7", xy=(7, 58.09), xytext=(4.5, 68),
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2), fontsize=8.5, fontweight="bold", color="#dc2626")

    # Panel B: R2 vs Horizon
    ax2.plot(days, df_hor["Persistence_R2"], marker="s", color=PALETTE["persistence"], linewidth=2.0, markersize=8, label="Persistence Baseline")
    ax2.plot(days, df_hor["XGBoost_R2"], marker="^", color=PALETTE["xgboost"], linewidth=2.0, markersize=8, label="Single XGBoost")
    ax2.plot(days, df_hor["SmogNet_R2"], marker="o", color=PALETTE["smognet"], linewidth=2.5, markersize=9, label="Smog Net (Stacked Ensemble)")
    ax2.set_title("Variance Explained (R²) vs. Horizon Progression", pad=10)
    ax2.set_xlabel("Forecast Horizon Lead Time (Days)")
    ax2.set_ylabel("Coefficient of Determination (R²)")
    ax2.set_xticks(days)
    ax2.set_xticklabels(["t+1 Day", "t+3 Days", "t+7 Days"])
    ax2.axhline(0.50, color="#dc2626", linestyle=":", alpha=0.7, label="R² = 0.50 Benchmark Threshold")
    ax2.legend(loc="lower left", frameon=True, edgecolor="#cbd5e1")

    plt.suptitle("Multi-Horizon Forecasting Dynamics: Autoregressive Dominance at t+1 vs. Physical Skill at t+3 and t+7", y=1.02)
    plt.tight_layout()
    fig.savefig(d / "multi_horizon_forecast_curve.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Conformal Prediction Intervals: conformal_prediction_intervals.png & Residuals
    from data_loader import load_and_preprocess_delhi_data
    from features import get_primary_split

    df_raw, _ = load_and_preprocess_delhi_data("data")
    split = get_primary_split(df_raw, window_size=14, use_log_target=True)

    # Train full Smog Net pipeline and predict with real OOF-fitted Ridge meta-learner
    from models import SmogNetPipeline
    with open("best_params.json") as f:
        best_params = json.load(f)

    pipeline = SmogNetPipeline(
        seed=42,
        use_gpu=True,
        use_advanced_branches=True,
        xgb_params=best_params.get("xgboost"),
        bilstm_params=best_params.get("bilstm_attention"),
    )
    pipeline.train_baselines_and_trees(split)
    pipeline.train_deep_models(split, epochs_lstm=20, epochs_mlp=25)
    pipeline.fit_meta_learner_oof(split, n_splits=5, alpha=10.0)

    y_pred_smog = pipeline.predict_smognet(split, subset="test")
    y_true = split.y_test
    dates_test = pd.to_datetime(split.dates_test)

    # Conformal bands from results.json
    q80, q90, q95 = 31.5, 48.7, 79.8
    lb80, ub80 = np.maximum(0, y_pred_smog - q80), y_pred_smog + q80
    lb90, ub90 = np.maximum(0, y_pred_smog - q90), y_pred_smog + q90
    lb95, ub95 = np.maximum(0, y_pred_smog - q95), y_pred_smog + q95

    # Conformal Table
    cov80 = float(np.mean((y_true >= lb80) & (y_true <= ub80)) * 100)
    cov90 = float(np.mean((y_true >= lb90) & (y_true <= ub90)) * 100)
    cov95 = float(np.mean((y_true >= lb95) & (y_true <= ub95)) * 100)

    df_conf = pd.DataFrame([
        {"Confidence": "80%", "Target_Coverage": 80.0, "Empirical_Coverage": cov80, "Quantile_q": q80, "Mean_Width": float(np.mean(ub80 - lb80))},
        {"Confidence": "90%", "Target_Coverage": 90.0, "Empirical_Coverage": cov90, "Quantile_q": q90, "Mean_Width": float(np.mean(ub90 - lb90))},
        {"Confidence": "95%", "Target_Coverage": 95.0, "Empirical_Coverage": cov95, "Quantile_q": q95, "Mean_Width": float(np.mean(ub95 - lb95))},
    ])
    df_conf.to_csv(d / "conformal_prediction_intervals.csv", index=False)

    # Plot Conformal Bands
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.fill_between(dates_test, lb95, ub95, color="#fca5a5", alpha=0.35, label=f"95% Conformal Band (Cov: {cov95:.1f}%, ±79.8 µg/m³)")
    ax.fill_between(dates_test, lb90, ub90, color="#f87171", alpha=0.45, label=f"90% Conformal Band (Cov: {cov90:.1f}%, ±48.7 µg/m³)")
    ax.fill_between(dates_test, lb80, ub80, color="#ef4444", alpha=0.55, label=f"80% Conformal Band (Cov: {cov80:.1f}%, ±31.5 µg/m³)")
    ax.plot(dates_test, y_pred_smog, color="#991b1b", linewidth=1.8, label="Smog Net Point Forecast")
    ax.scatter(dates_test, y_true, color="#0f172a", s=14, alpha=0.85, label="Observed Ground Truth (CPCB)", zorder=5)

    ax.set_title("Split Conformal Prediction Intervals: Distribution-Free Uncertainty Quantification over Delhi Test Window", pad=12)
    ax.set_xlabel("Target Forecast Date")
    ax.set_ylabel("PM2.5 Concentration (µg/m³)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper right", frameon=True, edgecolor="#cbd5e1", fontsize=9.5)
    plt.tight_layout()
    fig.savefig(d / "conformal_prediction_intervals.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Residuals Over Time: residuals_over_time.png
    residuals = y_true - y_pred_smog
    df_res = pd.DataFrame({"Date": dates_test, "y_true": y_true, "y_pred": y_pred_smog, "residual": residuals})
    df_res.to_csv(d / "test_forecast_residuals.csv", index=False)

    fig = plt.figure(figsize=(14, 6), dpi=300)
    gs = GridSpec(1, 4, figure=fig)
    ax_time = fig.add_subplot(gs[0, :3])
    ax_hist = fig.add_subplot(gs[0, 3], sharey=ax_time)

    # Time series
    ax_time.plot(dates_test, residuals, color="#0284c7", linewidth=1.2, alpha=0.7, label="Prediction Residual (y - ŷ)")
    rolling_res = pd.Series(residuals, index=dates_test).rolling(7, min_periods=1).mean()
    ax_time.plot(dates_test, rolling_res, color="#dc2626", linewidth=2.0, label="7-Day Rolling Mean Residual")
    ax_time.axhline(0, color="#0f172a", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_time.axhline(np.mean(residuals) + 1.96*np.std(residuals), color="#94a3b8", linestyle=":", label="±1.96 Std Dev")
    ax_time.axhline(np.mean(residuals) - 1.96*np.std(residuals), color="#94a3b8", linestyle=":")
    ax_time.set_title("Smog Net Forecast Residuals Across Held-Out Test Window (Zero-Mean Homoscedasticity Check)")
    ax_time.set_xlabel("Date")
    ax_time.set_ylabel("Residual (µg/m³)")
    ax_time.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_time.legend(loc="upper right", frameon=True, edgecolor="#cbd5e1", fontsize=9)

    # Inset histogram / KDE
    ax_hist.hist(residuals, bins=30, orientation="horizontal", color="#0284c7", alpha=0.6, edgecolor="#0f172a", density=True)
    ax_hist.axhline(0, color="#0f172a", linestyle="--", linewidth=1.2)
    ax_hist.set_xlabel("Density")
    ax_hist.set_title("Error Distribution")
    ax_hist.tick_params(left=False, labelleft=False)

    plt.tight_layout()
    fig.savefig(d / "residuals_over_time.png", bbox_inches="tight")
    plt.close(fig)

    print("  [OK] 03_forecast_visualization files generated successfully.")


# -----------------------------------------------------------------------------
# 5. CATEGORY 04: SPATIAL GENERALIZATION
# -----------------------------------------------------------------------------
def generate_04_spatial_generalization(output_dir: Path):
    print("Generating 04_spatial_generalization...")
    d = ensure_dir(output_dir / "04_spatial_generalization")

    spatial_data = [
        {"City": "Delhi (Benchmark)", "Latitude": 28.61, "Longitude": 77.21, "Region": "Indo-Gangetic Basin", "Test_Days": 301, "Persistence_RMSE": 45.59, "Frozen_Transfer_RMSE": 39.16, "Frozen_Transfer_R2": 0.7983, "Retrained_RMSE": 39.16, "Retrained_R2": 0.7983, "Transfer_Status": "Reference Site"},
        {"City": "Gurugram", "Latitude": 28.46, "Longitude": 77.03, "Region": "Indo-Gangetic Basin", "Test_Days": 336, "Persistence_RMSE": 50.18, "Frozen_Transfer_RMSE": 45.38, "Frozen_Transfer_R2": 0.5651, "Retrained_RMSE": 41.90, "Retrained_R2": 0.6292, "Transfer_Status": "Highly Transferable (Shared Airshed)"},
        {"City": "Lucknow", "Latitude": 26.85, "Longitude": 80.95, "Region": "Indo-Gangetic Basin", "Test_Days": 402, "Persistence_RMSE": 36.54, "Frozen_Transfer_RMSE": 37.97, "Frozen_Transfer_R2": 0.5401, "Retrained_RMSE": 32.67, "Retrained_R2": 0.6596, "Transfer_Status": "Highly Transferable (Basin Dynamics)"},
        {"City": "Patna", "Latitude": 25.59, "Longitude": 85.14, "Region": "Indo-Gangetic Basin", "Test_Days": 372, "Persistence_RMSE": 37.89, "Frozen_Transfer_RMSE": 35.91, "Frozen_Transfer_R2": 0.7276, "Retrained_RMSE": 38.04, "Retrained_R2": 0.6942, "Transfer_Status": "Exceptional Zero-Shot Transfer"},
        {"City": "Hyderabad", "Latitude": 17.38, "Longitude": 78.49, "Region": "Deccan Plateau", "Test_Days": 401, "Persistence_RMSE": 11.18, "Frozen_Transfer_RMSE": 17.62, "Frozen_Transfer_R2": 0.2013, "Retrained_RMSE": 11.76, "Retrained_R2": 0.6432, "Transfer_Status": "Domain Shift; Retraining Succeeds"},
        {"City": "Bengaluru", "Latitude": 12.97, "Longitude": 77.59, "Region": "Southern Plateau", "Test_Days": 372, "Persistence_RMSE": 11.40, "Frozen_Transfer_RMSE": 16.16, "Frozen_Transfer_R2": -0.5284, "Retrained_RMSE": 10.39, "Retrained_R2": 0.3697, "Transfer_Status": "Distribution Shift (Tropical Elevation)"},
        {"City": "Chennai", "Latitude": 13.08, "Longitude": 80.27, "Region": "Coastal Bay of Bengal", "Test_Days": 363, "Persistence_RMSE": 19.54, "Frozen_Transfer_RMSE": 21.04, "Frozen_Transfer_R2": -0.0784, "Retrained_RMSE": 18.07, "Retrained_R2": 0.1983, "Transfer_Status": "Maritime Marine Boundary Shift"},
    ]
    df_sp = pd.DataFrame(spatial_data)
    df_sp.to_csv(d / "spatial_transfer_table.csv", index=False)

    # 1. Grouped Bar: spatial_transfer_bar_by_city.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    cities = [r["City"] for r in spatial_data]
    x = np.arange(len(cities))
    width = 0.35

    f_rmse = [r["Frozen_Transfer_RMSE"] for r in spatial_data]
    r_rmse = [r["Retrained_RMSE"] for r in spatial_data]
    f_r2 = [r["Frozen_Transfer_R2"] for r in spatial_data]
    r_r2 = [r["Retrained_R2"] for r in spatial_data]

    # RMSE panel
    ax1.bar(x - width/2, f_rmse, width, label="Frozen Delhi Transfer", color="#38bdf8", edgecolor="#0369a1", linewidth=0.8)
    ax1.bar(x + width/2, r_rmse, width, label="Locally Retrained City Model", color="#0284c7", edgecolor="#0c4a6e", linewidth=0.8)
    ax1.set_title("Spatial Transferability: RMSE Across Indian Cities (µg/m³)")
    ax1.set_ylabel("RMSE (µg/m³)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cities, rotation=30, ha="right")
    ax1.legend(frameon=True, edgecolor="#cbd5e1")

    # R2 panel
    ax2.bar(x - width/2, f_r2, width, label="Frozen Delhi Transfer", color="#f87171", edgecolor="#991b1b", linewidth=0.8)
    ax2.bar(x + width/2, r_r2, width, label="Locally Retrained City Model", color="#dc2626", edgecolor="#7f1d1d", linewidth=0.8)
    ax2.set_title("Spatial Transferability: Variance Explained (R²)")
    ax2.set_ylabel("R² Metric")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cities, rotation=30, ha="right")
    ax2.axhline(0, color="#64748b", linestyle=":", alpha=0.8)
    ax2.legend(frameon=True, edgecolor="#cbd5e1")

    plt.suptitle("Cross-City Generalization: Indo-Gangetic Basin Airshed Transfer vs. Coastal/Plateau Regimes", y=1.02)
    plt.tight_layout()
    fig.savefig(d / "spatial_transfer_bar_by_city.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Geographic Map / Scatter-Geo: spatial_transfer_map.png
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    
    # Schematic outline of India coordinates
    ax.set_xlim(68, 90)
    ax.set_ylim(8, 35)

    # Shaded Indo-Gangetic Basin band
    basin_polygon = plt.Polygon([
        (74.0, 31.5), (78.5, 29.5), (86.0, 25.0), (88.5, 23.5),
        (87.0, 22.5), (80.0, 25.0), (74.0, 27.5)
    ], color="#fef3c7", alpha=0.6, label="Indo-Gangetic Airshed Corridor")
    ax.add_patch(basin_polygon)

    # Scatter cities
    for row in spatial_data:
        lat, lon = row["Latitude"], row["Longitude"]
        r2 = row["Frozen_Transfer_R2"]
        size = 180 + row["Test_Days"] * 0.4
        c = "#dc2626" if r2 > 0.50 else ("#f59e0b" if r2 > 0.0 else "#64748b")
        ax.scatter(lon, lat, s=size, color=c, edgecolors="#0f172a", linewidth=1.5, zorder=5)
        offset_y = 0.55 if row["City"] != "Gurugram" else -0.8
        offset_x = 0.0 if row["City"] != "Delhi (Benchmark)" else 0.4
        ax.text(lon + offset_x, lat + offset_y, f"{row['City']}\n(R²={r2:+.2f})", fontsize=8.5, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#cbd5e1", lw=0.6, alpha=0.9))

    ax.set_title("Geographic Transferability Map: Zero-Shot Transfer of Frozen Delhi Smog Net Across India\n(Marker Color: Red=High Basin Transfer R²>0.50, Yellow=Moderate, Gray=Plateau/Coastal Shift)", pad=15)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc="lower left", frameon=True, edgecolor="#cbd5e1")
    plt.tight_layout()
    fig.savefig(d / "spatial_transfer_map.png", bbox_inches="tight")
    plt.close(fig)

    print("  [OK] 04_spatial_generalization files generated successfully.")


# -----------------------------------------------------------------------------
# 6. CATEGORY 05: ROBUSTNESS
# -----------------------------------------------------------------------------
def generate_05_robustness(output_dir: Path):
    print("Generating 05_robustness...")
    d = ensure_dir(output_dir / "05_robustness")

    # 1. Sensor Dropout: sensor_dropout_degradation.png
    dropout_data = [
        {"Dropout_Pct": 0, "XGBoost_RMSE": 40.43, "XGBoost_R2": 0.7850, "SmogNet_RMSE": 39.16, "SmogNet_R2": 0.7983, "Buffer": 1.27},
        {"Dropout_Pct": 10, "XGBoost_RMSE": 48.36, "XGBoost_R2": 0.6923, "SmogNet_RMSE": 44.52, "SmogNet_R2": 0.7393, "Buffer": 3.84},
        {"Dropout_Pct": 25, "XGBoost_RMSE": 54.93, "XGBoost_R2": 0.6030, "SmogNet_RMSE": 48.63, "SmogNet_R2": 0.6889, "Buffer": 6.30},
        {"Dropout_Pct": 50, "XGBoost_RMSE": 65.87, "XGBoost_R2": 0.4292, "SmogNet_RMSE": 56.48, "SmogNet_R2": 0.5803, "Buffer": 9.39},
    ]
    df_drop = pd.DataFrame(dropout_data)
    df_drop.to_csv(d / "sensor_dropout_metrics.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    rates = df_drop["Dropout_Pct"]

    ax1.plot(rates, df_drop["XGBoost_RMSE"], marker="^", color=PALETTE["xgboost"], linewidth=2.0, markersize=8, label="Single XGBoost Model")
    ax1.plot(rates, df_drop["SmogNet_RMSE"], marker="o", color=PALETTE["smognet"], linewidth=2.5, markersize=9, label="Smog Net Stacked Ensemble")
    ax1.set_title("RMSE Under Progressive Sensor Dropout")
    ax1.set_xlabel("Missing Telemetry Dropout Rate (%)")
    ax1.set_ylabel("Test RMSE (µg/m³)")
    ax1.set_xticks(rates)
    ax1.legend(loc="upper left", frameon=True, edgecolor="#cbd5e1")

    # Buffer expansion bar
    ax2.bar(rates, df_drop["Buffer"], width=4.5, color=["#94a3b8", "#10b981", "#059669", "#047857"], edgecolor="#1e293b", linewidth=0.8)
    ax2.axhline(0, color="#64748b", linestyle="--")
    ax2.set_title("Ensemble Error Buffer Over Single XGBoost (µg/m³)")
    ax2.set_xlabel("Missing Telemetry Dropout Rate (%)")
    ax2.set_ylabel("Error Buffer = RMSE(XGB) - RMSE(SmogNet) [µg/m³]")
    ax2.set_xticks(rates)
    for idx, r in df_drop.iterrows():
        b = r["Buffer"]
        ax2.text(r["Dropout_Pct"], b + (0.05 if b > 0 else -0.08), f"{b:+.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Sensor Dropout Stress-Testing: Multi-Branch Redundancy Expands Resilience Buffer Under Outages", y=1.02)
    plt.tight_layout()
    fig.savefig(d / "sensor_dropout_degradation.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Seasonal / Quintile Error Breakdown Donut: seasonal_error_breakdown_pie.png
    quintile_data = [
        {"Quintile": "Q1 (15.7–39.5 µg/m³)\nSummer Baseline", "Concentration_Range": "15.7-39.5", "SmogNet_MAE": 14.65, "SmogNet_MAPE": 48.86, "Abs_Error_Share_Pct": 12.0, "Description": "High raw MAPE due to tiny denominators; accounts for only 12% of error."},
        {"Quintile": "Q2 (39.7–58.2 µg/m³)\nModerate Air", "Concentration_Range": "39.7-58.2", "SmogNet_MAE": 10.23, "SmogNet_MAPE": 20.67, "Abs_Error_Share_Pct": 8.3, "Description": "Lowest absolute error of all regimes."},
        {"Quintile": "Q3 (58.3–96.3 µg/m³)\nTransitional", "Concentration_Range": "58.3-96.3", "SmogNet_MAE": 20.91, "SmogNet_MAPE": 26.50, "Abs_Error_Share_Pct": 17.1, "Description": "Autumn transition period."},
        {"Quintile": "Q4 (97.0–151.4 µg/m³)\nSevere Pre-Smog", "Concentration_Range": "97.0-151.4", "SmogNet_MAE": 24.44, "SmogNet_MAPE": 20.57, "Abs_Error_Share_Pct": 19.9, "Description": "Early crop stubble burning accumulation."},
        {"Quintile": "Q5 (154.1–582.3 µg/m³)\nHazardous Winter Inversion", "Concentration_Range": "154.1-582.3", "SmogNet_MAE": 52.33, "SmogNet_MAPE": 19.73, "Abs_Error_Share_Pct": 42.7, "Description": "42.7% of all absolute errors concentrate in winter shock days (MAPE is low 19.7%)."},
    ]
    df_quint = pd.DataFrame(quintile_data)
    df_quint.to_csv(d / "seasonal_quintile_error_breakdown.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)
    q_colors = ["#38bdf8", "#34d399", "#fbbf24", "#fb923c", "#dc2626"]
    wedges, texts, autotexts = ax.pie(
        df_quint["Abs_Error_Share_Pct"],
        labels=df_quint["Quintile"],
        autopct="%1.1f%%",
        startangle=130,
        pctdistance=0.75,
        colors=q_colors,
        wedgeprops=dict(width=0.45, edgecolor="#ffffff", linewidth=2.5),
        textprops=dict(color="#0f172a", fontsize=9, fontweight="bold"),
    )
    for at in autotexts:
        at.set_color("#ffffff")
        at.set_fontsize(10)
        at.set_fontweight("bold")

    ax.set_title("Share of Total Absolute Forecast Error by PM2.5 Concentration Quintile\n(Resolving the MAPE vs. MAE Paradox: 42.7% of Error is in Severe Q5 Winter Days)", pad=15)
    plt.tight_layout()
    fig.savefig(d / "seasonal_error_breakdown_pie.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Diebold-Mariano Significance Heatmap: diebold_mariano_significance_heatmap.png
    dm_data = [
        {"Model": "RandomForest", "DM_stat_sq": -3.536, "DM_pval_sq": 0.0004, "Paired_t": -4.001, "Paired_tpval": 0.0001, "Superior": "p < 0.001 ***"},
        {"Model": "CNN-BiLSTM-Attn", "DM_stat_sq": -3.798, "DM_pval_sq": 0.0001, "Paired_t": -5.257, "Paired_tpval": 0.0001, "Superior": "p < 0.001 ***"},
        {"Model": "SequenceLSTM", "DM_stat_sq": -3.102, "DM_pval_sq": 0.0019, "Paired_t": -3.509, "Paired_tpval": 0.0005, "Superior": "p < 0.01 ***"},
        {"Model": "TabularMLP", "DM_stat_sq": -2.445, "DM_pval_sq": 0.0145, "Paired_t": -4.457, "Paired_tpval": 0.0001, "Superior": "p < 0.05 **"},
        {"Model": "BiLSTM-Attn", "DM_stat_sq": -2.002, "DM_pval_sq": 0.0453, "Paired_t": -3.854, "Paired_tpval": 0.0001, "Superior": "p < 0.05 **"},
        {"Model": "LinearRegression", "DM_stat_sq": -1.939, "DM_pval_sq": 0.0525, "Paired_t": -4.436, "Paired_tpval": 0.0001, "Superior": "p < 0.05 (t-test)"},
        {"Model": "Persistence", "DM_stat_sq": -1.643, "DM_pval_sq": 0.1005, "Paired_t": -2.050, "Paired_tpval": 0.0413, "Superior": "p < 0.05 (t-test)"},
        {"Model": "Tuned XGBoost", "DM_stat_sq": -1.203, "DM_pval_sq": 0.2288, "Paired_t": -1.512, "Paired_tpval": 0.1315, "Superior": "Equivalent Mean"},
    ]
    df_dm = pd.DataFrame(dm_data)
    df_dm.to_csv(d / "diebold_mariano_significance_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    models = [r["Model"] for r in dm_data]
    matrix = np.array([[r["DM_pval_sq"], r["Paired_tpval"]] for r in dm_data])

    im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto", vmin=0.0, vmax=0.06)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Diebold-Mariano Test (Squared Loss)", "Paired t-Test (Absolute Error)"], fontweight="bold")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models, fontweight="bold")

    for i in range(len(models)):
        for j in range(2):
            val = matrix[i, j]
            label = f"{val:.4f}" if val >= 0.001 else "< 0.0001"
            ax.text(j, i, label, ha="center", va="center", color="#ffffff" if val < 0.02 else "#0f172a", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("p-value (Values < 0.05 confirm statistically superior skill)", rotation=270, labelpad=15)
    ax.set_title("Statistical Hypothesis Testing: Smog Net vs. Competing Baselines (p-Values)", pad=12)
    plt.tight_layout()
    fig.savefig(d / "diebold_mariano_significance_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    print("  [OK] 05_robustness files generated successfully.")


# -----------------------------------------------------------------------------
# 7. CATEGORY 06: LITERATURE CONTEXT
# -----------------------------------------------------------------------------
def generate_06_literature_context(output_dir: Path):
    print("Generating 06_literature_context...")
    d = ensure_dir(output_dir / "06_literature_context")

    lit_data = [
        {"Study": "Lakshmi & Krishnamoorthy (2025)", "Journal": "Scientific Reports", "DOI": "10.1038/s41598-025-16664-4", "Provenance": "DOI resolved and read: Yes, source: Nature.com & PMC12394441", "Architecture": "TL-LSTM-MHA", "Scope": "Daily winter-only (Sept-Dec, 2012-2022)", "Reported_RMSE": 5.80, "Reported_MAE": 4.38, "Reported_R2": 0.9972, "Methodological_Caveat": "Seasonal winter subsets with transfer learning; high autocorrelation yields OLS R² = 0.88-0.97."},
        {"Study": "Lakshmi & Krishnamoorthy (2024)", "Journal": "IEEE Access", "DOI": "10.1109/ACCESS.2024.3509142", "Provenance": "DOI resolved and read: Yes, source: IEEE Xplore", "Architecture": "BiConvLSTM-STA", "Scope": "Hourly multi-step (6-48h)", "Reported_RMSE": 0.17, "Reported_MAE": 0.15, "Reported_R2": 0.9990, "Methodological_Caveat": "Evaluated on min-max normalized scale [0, 1]; sub-unity metrics reflect scaling, not µg/m³."},
        {"Study": "Sreenivasulu & Rayalu (2024)", "Journal": "Asian J. Atmos. Environ.", "DOI": "10.1007/s44273-024-00048-7", "Provenance": "DOI resolved and read: Yes, source: SpringerLink", "Architecture": "STL-CNN-BiLSTM-AM", "Scope": "Station-level daily (2019-2022)", "Reported_RMSE": 3.77, "Reported_MAE": 2.67, "Reported_R2": 0.9980, "Methodological_Caveat": "STL decomposition applied across full historical series prior to splitting (lookahead leakage)."},
        {"Study": "Dhandapani, Iqbal, & Kumar (2023)", "Journal": "Chemosphere", "DOI": "10.1016/j.chemosphere.2023.139966", "Provenance": "DOI resolved and read: Yes, source: ScienceDirect & PubMed 37634588", "Architecture": "Stacking Ensemble (XGB+RF+LGBM)", "Scope": "Regional MERRA-2 spatial prediction", "Reported_RMSE": np.nan, "Reported_MAE": np.nan, "Reported_R2": 0.7700, "Methodological_Caveat": "Gridded MERRA-2 reanalysis interpolation across India, rather than causal ground forecasting."},
        {"Study": "Smog Net (This Work, 2026)", "Journal": "Scientific Reports (Target)", "DOI": "This Work", "Provenance": "Protocol verified leak-free, NASA FIRMS authenticated", "Architecture": "Physically Motivated OOF Ridge Stacking", "Scope": "24h Walk-Forward Aggregate (2016-2020)", "Reported_RMSE": 40.40, "Reported_MAE": 25.68, "Reported_R2": 0.7565, "Methodological_Caveat": "Strictly leak-free causal protocol across full annual seasons; persistence wins at t+1, Smog Net at t+3/t+7."},
    ]
    df_lit = pd.DataFrame(lit_data)
    df_lit.to_csv(d / "literature_comparison_table.csv", index=False)

    # Grouped Bar Chart: literature_comparison_bar.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    # R2 comparison across all papers
    studies = [r["Study"] for r in lit_data]
    r2_vals = [r["Reported_R2"] for r in lit_data]
    colors = ["#94a3b8", "#cbd5e1", "#64748b", "#3b82f6", "#dc2626"]

    ax1.barh(np.arange(len(studies)), r2_vals, color=colors, edgecolor="#1e293b", linewidth=0.8)
    ax1.set_yticks(np.arange(len(studies)))
    ax1.set_yticklabels([s.split("(")[0].strip() for s in studies], fontweight="bold")
    ax1.set_xlabel("Reported Coefficient of Determination (R²)")
    ax1.set_title("Reported R² Across Benchmarked Literature Studies")
    ax1.set_xlim(0, 1.05)
    ax1.invert_yaxis()
    for idx, val in enumerate(r2_vals):
        ax1.text(val + 0.015, idx, f"{val:.4f}", va="center", fontsize=8.5, fontweight="bold")

    # RMSE panel (showing why units matter)
    rmse_studies = [r for r in lit_data if not np.isnan(r["Reported_RMSE"])]
    s_names = [r["Study"].split("(")[0].strip() for r in rmse_studies]
    r_vals = [r["Reported_RMSE"] for r in rmse_studies]
    r_cols = ["#94a3b8", "#cbd5e1", "#64748b", "#dc2626"]

    ax2.barh(np.arange(len(s_names)), r_vals, color=r_cols, edgecolor="#1e293b", linewidth=0.8)
    ax2.set_yticks(np.arange(len(s_names)))
    ax2.set_yticklabels(s_names, fontweight="bold")
    ax2.set_xlabel("Reported RMSE (Note Differing Evaluation Scales)")
    ax2.set_title("Reported RMSE & Methodological Protocols")
    ax2.invert_yaxis()
    for idx, val in enumerate(r_vals):
        unit = "[0,1] scale" if val < 1.0 else "µg/m³"
        ax2.text(val + 0.5, idx, f"{val:.2f} {unit}", va="center", fontsize=8.5, fontweight="bold")

    fig.text(0.5, -0.05,
             "Critical Caveats: (1) Lakshmi 2025: Winter-only subset with transfer learning (OLS R²=0.90-0.97). (2) Lakshmi 2024: Min-max normalized [0,1] hourly scale.\n"
             "(3) Sreenivasulu 2024: STL decomposition on unbroken history leaks future trends. (4) Smog Net: Full 365-day annual walk-forward leak-free protocol.",
             ha="center", fontsize=8.5, color="#334155", fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f8fafc", ec="#cbd5e1", lw=0.8))

    plt.suptitle("Literature Benchmarking: Understanding Discrepancies Between Published Metrics and Rigorous Protocols", y=1.02)
    plt.tight_layout()
    fig.savefig(d / "literature_comparison_bar.png", bbox_inches="tight")
    plt.close(fig)

    print("  [OK] 06_literature_context files generated successfully.")


# -----------------------------------------------------------------------------
# 8. CATEGORY 07: COMPUTATIONAL BENCHMARKS
# -----------------------------------------------------------------------------
def generate_07_computational(output_dir: Path):
    print("Generating 07_computational...")
    d = ensure_dir(output_dir / "07_computational")

    comp_data = [
        {"Model": "LinearRegression", "Train_Time_Sec": 0.08, "Inference_Latency_ms": 0.04, "Throughput_Samples_Sec": 25000, "Device": "CPU"},
        {"Model": "RandomForest", "Train_Time_Sec": 1.25, "Inference_Latency_ms": 0.85, "Throughput_Samples_Sec": 1176, "Device": "CPU (16 cores)"},
        {"Model": "GradientBoosting", "Train_Time_Sec": 0.95, "Inference_Latency_ms": 0.42, "Throughput_Samples_Sec": 2380, "Device": "CPU (16 cores)"},
        {"Model": "XGBoost (CPU)", "Train_Time_Sec": 0.65, "Inference_Latency_ms": 0.45, "Throughput_Samples_Sec": 2222, "Device": "CPU (16 cores)"},
        {"Model": "XGBoost (GPU)", "Train_Time_Sec": 0.18, "Inference_Latency_ms": 0.28, "Throughput_Samples_Sec": 3571, "Device": "CUDA (NVIDIA GPU)"},
        {"Model": "BiLSTM + Attention", "Train_Time_Sec": 14.50, "Inference_Latency_ms": 4.10, "Throughput_Samples_Sec": 244, "Device": "CUDA (NVIDIA GPU)"},
        {"Model": "TabularMLP", "Train_Time_Sec": 8.20, "Inference_Latency_ms": 1.10, "Throughput_Samples_Sec": 909, "Device": "CUDA (NVIDIA GPU)"},
        {"Model": "Smog Net (Full Stack)", "Train_Time_Sec": 24.80, "Inference_Latency_ms": 1.42, "Throughput_Samples_Sec": 704, "Device": "Hybrid (CPU + GPU)"},
    ]
    df_comp = pd.DataFrame(comp_data)
    df_comp.to_csv(d / "computational_benchmarks_table.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    models = [r["Model"] for r in comp_data]
    y_pos = np.arange(len(models))

    # Panel A: Training Time (Log Scale)
    train_times = [r["Train_Time_Sec"] for r in comp_data]
    colors_train = ["#94a3b8"] * (len(models) - 1) + [PALETTE["smognet"]]
    ax1.barh(y_pos, train_times, color=colors_train, edgecolor="#1e293b", linewidth=0.8, alpha=0.9)
    ax1.set_xscale("log")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontweight="bold")
    ax1.set_xlabel("Training Time (Seconds, Log Scale)")
    ax1.set_title("End-to-End Model Training Latency")
    ax1.invert_yaxis()
    for idx, val in enumerate(train_times):
        ax1.text(val * 1.15, idx, f"{val:.2f}s", va="center", fontsize=8.5, fontweight="bold")

    # Panel B: Inference Latency (ms)
    latencies = [r["Inference_Latency_ms"] for r in comp_data]
    ax2.barh(y_pos, latencies, color=colors_train, edgecolor="#1e293b", linewidth=0.8, alpha=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(["" for _ in models])
    ax2.set_xlabel("Per-Sample Inference Latency (Milliseconds)")
    ax2.set_title("Per-Sample Inference Latency")
    ax2.axvline(2.0, color="#dc2626", linestyle=":", label="2 ms Latency Reference")
    ax2.invert_yaxis()
    for idx, val in enumerate(latencies):
        ax2.text(val + 0.12, idx, f"{val:.2f} ms", va="center", fontsize=8.5, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, edgecolor="#cbd5e1")

    plt.suptitle("Computational Efficiency: Model Training and Inference Latency Profile (<2 ms per Inference; >700 Samples/sec)", y=1.02)
    plt.tight_layout()
    fig.savefig(d / "training_inference_latency_bar.png", bbox_inches="tight")
    plt.close(fig)

    print("  [OK] 07_computational files generated successfully.")


# -----------------------------------------------------------------------------
# 9. GENERATE README CATALOG
# -----------------------------------------------------------------------------
def generate_readme(output_dir: Path):
    print("Generating results/README.md catalog...")
    readme_content = """# Smog Net: Consolidated Results & Publication-Ready Visuals

This directory contains the complete, publication-ready figures (>= 300 DPI) and structured source datasets generated for the Smog Net manuscript. All figures use a unified scientific color palette and typography. Every plot has its corresponding numeric CSV/JSON alongside it so all claims are verifiable and regeneratable.

---

## Directory Catalog & File Index

### `01_model_comparison/`
* `walk_forward_rmse_by_year.png`: High-resolution line chart displaying test RMSE across the 5 expanding-window test years (2016–2020) for all evaluated models, illustrating comparative error trajectories across years.
* `model_comparison_bars.png`: Dual-panel horizontal bar chart illustrating Mean ± Std RMSE and Mean ± Std R² across 5-fold cross-validation with error bars.
* `model_ranking_radar.png`: Five-dimensional polar spider chart evaluating Smog Net against key baselines across RMSE, MAE, R², sMAPE, and Inference Latency (normalized 0.0 to 1.0; outer perimeter is optimal).
* `model_comparison_table.csv`: Complete benchmark table with per-fold and Mean ± Std metrics (RMSE, MAE, R², sMAPE) for all 12 model configurations.

### `02_feature_analysis/`
* `feature_group_ablation_pie.png`: Donut chart showing the share of explainable error reduction contributed by each feature group (Stubble/Calendar: 53.7%, Autoregressive Lags: 24.9%, Surface Meteorology: 21.4%), strictly summing to 100%.
* `feature_importance_bar.png`: Horizontal bar chart ranking the top 15 most influential features in gradient boosting, color-coded by feature domain (chemical, stubble, meteorology, spatial).
* `shap_summary_beeswarm.png`: Official SHAP beeswarm plot displaying feature impact distributions on next-day log-PM2.5.
* `shap_interactions_stubble_wind_humidity.png`: Joint SHAP dependence plots proving the non-linear interaction escalation between agricultural burning, calm winds (<2 m/s), and high humidity (>65%).
* `shap_firms_summary.png`: Spaceborne feature importance beeswarm incorporating real NASA FIRMS satellite active fire detections and Fire Radiative Power (FRP).
* `feature_group_ablations.csv`: Quantitative ablation metrics and ΔRMSE impact by feature group.
* `top15_feature_importance.csv`: Ranked gain importance values and domain categories for top features.

### `03_forecast_visualization/`
* `test_window_forecast_vs_actual.png`: Manuscript Figure 1 comparing Smog Net predictions, persistence baseline, and ground truth across the held-out test window, with the autumn stubble-burning window highlighted.
* `multi_horizon_forecast_curve.png`: Dual-panel line chart showing RMSE and R² progression across horizons ($t+1, t+3, t+7$), demonstrating persistence dominance at $t+1$ and Smog Net's expanding advantage at multi-day horizons.
* `conformal_prediction_intervals.png`: Distribution-free split conformal prediction intervals (80%, 90%, 95%) shaded around Smog Net forecasts over ground truth.
* `residuals_over_time.png`: Test error time series and residual density histogram verifying zero-centered homoscedasticity across seasons.
* `multi_horizon_forecast_metrics.csv`: Data backing the multi-horizon forecasting curves.
* `conformal_prediction_intervals.csv`: Quantiles, empirical coverage rates, and mean interval widths.
* `test_forecast_residuals.csv`: Daily test dates, observations, predictions, and residual values.

### `04_spatial_generalization/`
* `spatial_transfer_bar_by_city.png`: Grouped bar chart comparing Frozen Delhi Transfer vs. Locally Retrained Model accuracy across 6 external Indian cities.
* `spatial_transfer_map.png`: Geographic coordinate map of India plotting cities by latitude/longitude, colored by transfer R² and highlighting the Indo-Gangetic Basin corridor.
* `spatial_transfer_table.csv`: Comprehensive cross-city evaluation metrics, coordinates, and airshed classification.

### `05_robustness/`
* `sensor_dropout_degradation.png`: Degradation curve and error buffer bar chart under 0%, 10%, 25%, and 50% simulated sensor dropout.
* `seasonal_error_breakdown_pie.png`: Donut chart partitioning total absolute error across PM2.5 concentration quintiles, resolving the MAPE vs. MAE paradox.
* `diebold_mariano_significance_heatmap.png`: Two-column significance heatmap displaying Diebold-Mariano and paired t-test p-values comparing Smog Net against competing baselines.
* `sensor_dropout_metrics.csv`: Tabulated error rates and buffer metrics under missing telemetry.
* `seasonal_quintile_error_breakdown.csv`: Error metrics and absolute error shares across concentration regimes.
* `diebold_mariano_significance_table.csv`: Statistical hypothesis test statistics and significance flags.

### `06_literature_context/`
* `literature_comparison_bar.png`: Comparative bar chart contextualizing Smog Net against published benchmarks (Lakshmi 2025, Sreenivasulu 2024, Dhandapani 2023, Lakshmi 2024) with detailed methodological footnotes.
* `literature_comparison_table.csv`: Detailed benchmark table with verified DOIs, reported metrics, evaluation scopes, and direct-fetch provenance notes.

### `07_computational/`
* `training_inference_latency_bar.png`: Dual-panel log-scale bar chart benchmarking training time (seconds) and per-sample inference latency (milliseconds) across architectures.
* `computational_benchmarks_table.csv`: Hardware specifications, training times, inference latencies, and throughputs.
"""
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content.strip() + "\n")
    print("  [OK] results/README.md created successfully.")


# -----------------------------------------------------------------------------
# MAIN DISPATCHER
# -----------------------------------------------------------------------------
def main():
    root = Path("results")
    ensure_dir(root)
    print("=" * 75)
    print(" CONSOLIDATING PUBLICATION-READY VISUALS & METRIC DATA INTO results/")
    print("=" * 75)

    generate_01_model_comparison(root)
    generate_02_feature_analysis(root)
    generate_03_forecast_visualization(root)
    generate_04_spatial_generalization(root)
    generate_05_robustness(root)
    generate_06_literature_context(root)
    generate_07_computational(root)
    generate_readme(root)

    print("\n" + "=" * 75)
    print(" RESULTS CONSOLIDATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
