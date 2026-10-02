import sys
sys.path.insert(0, ".")
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from data_loader import load_and_preprocess_delhi_data
from features import get_primary_split, TABULAR_FEATURE_COLS
from evaluate import compute_metrics

df_raw, _ = load_and_preprocess_delhi_data("data")
primary_split = get_primary_split(df_raw, window_size=14, use_log_target=True)

with open("best_params.json", "r") as f:
    best_params = json.load(f)
best_xgb_params = best_params["xgboost"]

feature_groups = {
    "Full_Model_38_Features": TABULAR_FEATURE_COLS,
    "Ablation_Without_Surface_Meteorology": [
        col for col in TABULAR_FEATURE_COLS
        if col not in ["wind_speed", "wind_dir_sin", "wind_dir_cos", "humidity", "temp", "temp_lag1", "temp_inversion_proxy"]
    ],
    "Ablation_Without_Spatial_Neighbor_Gurugram": [
        col for col in TABULAR_FEATURE_COLS
        if not col.startswith("gurugram_") and not col.startswith("spatial_grad_")
    ],
    "Ablation_Without_Crop_Stubble_And_Calendar": [
        col for col in TABULAR_FEATURE_COLS
        if col not in ["is_stubble_burning", "is_diwali_window", "is_covid_lockdown", "upwind_stubble_transport", "stubble_x_windspeed"]
    ],
    "Ablation_Without_Rolling_Stats_And_MultiDay_Lags": [
        col for col in TABULAR_FEATURE_COLS
        if col not in ["pm25_roll3_mean", "pm25_roll3_std", "pm25_roll7_mean", "pm25_roll7_std", "pm25_lag2", "pm25_lag3", "pm25_lag7"]
    ],
}

print("=== CPU (Default) Ablation Results ===")
base_rmse_cpu = None
for name, feats in feature_groups.items():
    m = xgb.XGBRegressor(**best_xgb_params, random_state=42)
    m.fit(primary_split.X_train[feats].values, primary_split.y_train_target)
    pred = np.expm1(m.predict(primary_split.X_test[feats].values))
    metrics = compute_metrics(primary_split.y_test, pred)
    if base_rmse_cpu is None:
        base_rmse_cpu = metrics["RMSE"]
    delta = metrics["RMSE"] - base_rmse_cpu
    print(f"CPU  | {name:45s} | Feats: {len(feats):2d} | RMSE: {metrics['RMSE']:.2f} (delta: {delta:+.2f}) | R2: {metrics['R2']:.4f}")

print("\n=== GPU (hist/cuda matching pipeline) Ablation Results ===")
base_rmse_gpu = None
for name, feats in feature_groups.items():
    m = xgb.XGBRegressor(**best_xgb_params, tree_method="hist", device="cuda", random_state=42)
    m.fit(primary_split.X_train[feats].values, primary_split.y_train_target)
    pred = np.expm1(m.predict(primary_split.X_test[feats].values))
    metrics = compute_metrics(primary_split.y_test, pred)
    if base_rmse_gpu is None:
        base_rmse_gpu = metrics["RMSE"]
    delta = metrics["RMSE"] - base_rmse_gpu
    print(f"GPU  | {name:45s} | Feats: {len(feats):2d} | RMSE: {metrics['RMSE']:.2f} (delta: {delta:+.2f}) | R2: {metrics['R2']:.4f}")
