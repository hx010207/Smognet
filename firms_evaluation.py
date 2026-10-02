"""
NASA FIRMS Satellite Active Fire Evaluation Script
==================================================
Evaluates the predictive utility of real MODIS_SP active fire telemetry
(Fire Count, Fire Radiative Power [FRP], and Northwestern Transport Index)
against the calendar stubble burning flag on the Primary Test Benchmark.

Generates SHAP explanations, counterfactual analysis, and comparison metrics.
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import load_and_preprocess_delhi_data
from features import engineer_features, TABULAR_FEATURE_COLS, create_split_dataset, SplitDataset


def patch_shap_xgboost():
    """Patches shap.explainers._tree.decode_ubjson_buffer to handle bracketed base_score strings."""
    try:
        from shap.explainers import _tree
        original_decode = _tree.decode_ubjson_buffer

        def patched_decode(buffer):
            tree_info = original_decode(buffer)
            if "learner" in tree_info and "learner_model_param" in tree_info["learner"]:
                param = tree_info["learner"]["learner_model_param"]
                if "base_score" in param and isinstance(param["base_score"], str):
                    val_str = param["base_score"].strip("[]")
                    param["base_score"] = val_str
            return tree_info

        _tree.decode_ubjson_buffer = patched_decode
    except Exception as e:
        print(f"[SHAP Patch] Warning: {e}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    clipped_true = np.maximum(y_true, 1.0)
    mape = float(np.mean(np.abs((y_true - y_pred) / clipped_true)) * 100.0)
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-6
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
        "sMAPE": round(smape, 2),
    }


def add_firms_features(df_feat: pd.DataFrame, df_firms: pd.DataFrame) -> pd.DataFrame:
    df_firms["Date"] = pd.to_datetime(df_firms["Date"])
    df_feat["Date"] = pd.to_datetime(df_feat["Date"])
    
    merged = df_feat.merge(df_firms, on="Date", how="left")
    merged["firms_fire_count"] = merged["firms_fire_count"].fillna(0).astype(float)
    merged["firms_frp_total"] = merged["firms_frp_total"].fillna(0.0).astype(float)

    # Lags (yesterday's active fire detections in Punjab & Haryana)
    for lag in [1, 2, 3]:
        merged[f"firms_fire_count_lag{lag}"] = merged["firms_fire_count"].shift(lag).fillna(0.0)
        merged[f"firms_frp_lag{lag}"] = merged["firms_frp_total"].shift(lag).fillna(0.0)

    # Northwestern transport alignment:
    # Punjab/Haryana is located NW of Delhi (~305 degrees bearing).
    # When WD10M blows from ~270 to 330, smoke is advected directly towards Delhi.
    wd = merged["WD10M"].fillna(270.0)
    angle_diff = (wd - 305.0) % 360.0
    angle_diff = np.where(angle_diff > 180.0, 360.0 - angle_diff, angle_diff)
    nw_weight = np.maximum(0.0, np.cos(np.radians(angle_diff)))
    
    ws = merged["WS10M"].fillna(2.5)
    # Log-scaled transport intensity
    frp_lag1 = merged["firms_frp_lag1"]
    merged["firms_transport_index"] = np.log1p(frp_lag1) * nw_weight * ws
    merged["firms_log_frp_lag1"] = np.log1p(frp_lag1)
    
    return merged


def run_firms_comparison():
    patch_shap_xgboost()
    
    df_raw, _ = load_and_preprocess_delhi_data("data")
    firms_file = Path("data/firms_punjab_haryana_daily_2015_2020.csv")
    if not firms_file.exists():
        raise FileNotFoundError("FIRMS daily CSV not found yet! Ensure downloader completes.")
    
    df_firms = pd.read_csv(firms_file)
    print(f"[FIRMS Eval] Loaded {len(df_firms)} daily records from {firms_file}")

    # Build base features
    df_feat = engineer_features(df_raw, window_size=14)
    df_augmented = add_firms_features(df_feat, df_firms)
    df_augmented = df_augmented.dropna(subset=["target_pm25"]).reset_index(drop=True)

    with open("best_params.json", "r") as f:
        best_params = json.load(f)
    xgb_params = best_params["xgboost"]

    # Train/Val/Test Split (70% / 15% / 15%)
    n_total = len(df_augmented)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, n_total)

    y_train_log = np.log1p(df_augmented["target_pm25"].values[train_slice])
    y_test_real = df_augmented["target_pm25"].values[test_slice]

    # Model 1: Calendar Only (Standard Smog Net Tabular Features with is_stubble_burning)
    cols_calendar = [c for c in TABULAR_FEATURE_COLS if c in df_augmented.columns]
    X_tr_cal = df_augmented[cols_calendar].iloc[train_slice]
    X_te_cal = df_augmented[cols_calendar].iloc[test_slice]

    m_cal = xgb.XGBRegressor(**xgb_params, random_state=42)
    m_cal.fit(X_tr_cal, y_train_log)
    p_cal = np.expm1(m_cal.predict(X_te_cal))
    metrics_cal = compute_metrics(y_test_real, p_cal)

    # Model 2: Satellite Only (Replaces is_stubble_burning with FIRMS active fire features)
    firms_cols = ["firms_log_frp_lag1", "firms_transport_index", "firms_fire_count_lag1"]
    cols_sat_only = [c for c in cols_calendar if c != "is_stubble_burning"] + firms_cols
    X_tr_sat = df_augmented[cols_sat_only].iloc[train_slice]
    X_te_sat = df_augmented[cols_sat_only].iloc[test_slice]

    m_sat = xgb.XGBRegressor(**xgb_params, random_state=42)
    m_sat.fit(X_tr_sat, y_train_log)
    p_sat = np.expm1(m_sat.predict(X_te_sat))
    metrics_sat = compute_metrics(y_test_real, p_sat)

    # Model 3: Combined (Both Calendar Flag AND FIRMS Satellite Features)
    cols_comb = cols_calendar + firms_cols
    X_tr_comb = df_augmented[cols_comb].iloc[train_slice]
    X_te_comb = df_augmented[cols_comb].iloc[test_slice]

    m_comb = xgb.XGBRegressor(**xgb_params, random_state=42)
    m_comb.fit(X_tr_comb, y_train_log)
    p_comb = np.expm1(m_comb.predict(X_te_comb))
    metrics_comb = compute_metrics(y_test_real, p_comb)

    print("\n" + "=" * 75)
    print(" FIRMS Satellite Fire Telemetry Evaluation vs. Calendar Baseline")
    print("=" * 75)
    print(f"  Model A [Calendar Flag Only] : RMSE = {metrics_cal['RMSE']:5.2f} µg/m³ | MAE = {metrics_cal['MAE']:5.2f} | R² = {metrics_cal['R2']:.4f} | sMAPE = {metrics_cal['sMAPE']}%")
    print(f"  Model B [Satellite FIRMS Only]: RMSE = {metrics_sat['RMSE']:5.2f} µg/m³ | MAE = {metrics_sat['MAE']:5.2f} | R² = {metrics_sat['R2']:.4f} | sMAPE = {metrics_sat['sMAPE']}%")
    print(f"  Model C [Combined Cal + Sat] : RMSE = {metrics_comb['RMSE']:5.2f} µg/m³ | MAE = {metrics_comb['MAE']:5.2f} | R² = {metrics_comb['R2']:.4f} | sMAPE = {metrics_comb['sMAPE']}%")

    # SHAP Analysis on Combined Model
    explainer = shap.TreeExplainer(m_comb)
    shap_vals_log = explainer.shap_values(X_te_comb)
    
    mean_abs_shap = pd.Series(np.mean(np.abs(shap_vals_log), axis=0), index=cols_comb).sort_values(ascending=False)
    
    # Counterfactual analysis:
    # Holding all test features at their medians, evaluate predicted next-day PM2.5 at FRP = 0 vs FRP at peak burning (90th percentile of active fire season)
    test_medians = X_te_comb.median().to_frame().T
    row_clean = test_medians.copy()
    row_clean["firms_log_frp_lag1"] = 0.0
    row_clean["firms_fire_count_lag1"] = 0.0
    row_clean["firms_transport_index"] = 0.0
    row_clean["is_stubble_burning"] = 0.0

    row_fire = test_medians.copy()
    # 90th percentile during Oct-Nov active fires
    peak_frp = float(df_augmented.loc[df_augmented["is_stubble_burning"] == 1, "firms_frp_lag1"].quantile(0.90))
    peak_count = float(df_augmented.loc[df_augmented["is_stubble_burning"] == 1, "firms_fire_count_lag1"].quantile(0.90))
    nw_wind_speed = 3.5  # moderate advective transport
    
    row_fire["firms_log_frp_lag1"] = np.log1p(peak_frp)
    row_fire["firms_fire_count_lag1"] = peak_count
    row_fire["firms_transport_index"] = np.log1p(peak_frp) * 1.0 * nw_wind_speed
    row_fire["is_stubble_burning"] = 1.0

    pred_clean_log = float(m_comb.predict(row_clean)[0])
    pred_fire_log = float(m_comb.predict(row_fire)[0])
    pred_clean_real = float(np.expm1(pred_clean_log))
    pred_fire_real = float(np.expm1(pred_fire_log))
    cf_delta = pred_fire_real - pred_clean_real

    print(f"\n[FIRMS SHAP] Peak FRP used in counterfactual: {peak_frp:.1f} MW ({peak_count:.0f} fire counts)")
    print(f"[FIRMS SHAP] Baseline prediction (Zero Fires): {pred_clean_real:.2f} ug/m3")
    print(f"[FIRMS SHAP] Peak Satellite Fire prediction : {pred_fire_real:.2f} ug/m3")
    print(f"[FIRMS SHAP] Satellite-Informed Counterfactual Delta PM2.5: +{cf_delta:.2f} ug/m3")

    # Plot SHAP Summary for FIRMS model
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_log, X_te_comb, max_display=15, show=False)
    plt.title("TreeSHAP Summary: Smog Net with NASA FIRMS Satellite Telemetry", fontsize=12)
    plt.tight_layout()
    plt.savefig("shap_firms_summary.png", dpi=300)
    plt.close()

    results_firms = {
        "firms_api_query_metadata": {
            "api_endpoint": "https://firms.modaps.eosdis.nasa.gov/api/area/csv/[MAP_KEY]/MODIS_SP/73.8,27.5,77.6,32.5/5/[DATE]",
            "source_instrument": "MODIS Standard Processing (MODIS_SP on Terra and Aqua)",
            "bounding_box_coordinates": "73.8°E, 27.5°N, 77.6°E, 32.5°N (Punjab & Haryana)",
            "day_range_per_request": 5,
            "date_range": "2015-01-01 to 2020-07-01",
            "is_synthetic": False
        },
        "model_comparison": {
            "Model_A_Calendar_Only": metrics_cal,
            "Model_B_Satellite_FIRMS_Only": metrics_sat,
            "Model_C_Combined_Calendar_And_Satellite": metrics_comb
        },
        "shap_and_counterfactual": {
            "satellite_informed_counterfactual_delta_pm25_ug_m3": round(cf_delta, 2),
            "baseline_zero_fire_pred_ug_m3": round(pred_clean_real, 2),
            "peak_satellite_fire_pred_ug_m3": round(pred_fire_real, 2),
            "peak_fire_radiative_power_mw": round(peak_frp, 1),
            "peak_fire_pixel_count": round(peak_count, 1),
            "firms_transport_index_mean_abs_shap_log": round(float(mean_abs_shap.get("firms_transport_index", 0.0)), 4),
            "firms_log_frp_lag1_mean_abs_shap_log": round(float(mean_abs_shap.get("firms_log_frp_lag1", 0.0)), 4),
            "is_stubble_burning_mean_abs_shap_log": round(float(mean_abs_shap.get("is_stubble_burning", 0.0)), 4)
        }
    }

    with open("firms_results.json", "w") as f:
        json.dump(results_firms, f, indent=2)
    print("\n[FIRMS Eval] Saved results to firms_results.json and figure to shap_firms_summary.png")
    return results_firms


if __name__ == "__main__":
    run_firms_comparison()
