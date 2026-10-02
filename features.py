"""
Smog Net Advanced Feature Engineering & Expanding-Window Splits
==============================================================
Constructs:
  - Calendar cyclics, Diwali firework windows, COVID-19 natural experiment lock-downs.
  - Stubble burning indicators, physical upwind transport vectors, and explicit interactions.
  - Spatial lags from neighboring NCR stations (Gurugram) and regional pollution gradients.
  - Multi-day episode persistence flags (is_prev_day_severe).
  - Pollutant lags and shifted rolling statistics.
  - 14-day sliding sequence windows for deep GPU sequence branches.
  - Leak-free chronological train/val/test splits, expanding-window walk-forward folds,
    and out-of-fold training masks.
  - Supports both raw-scale target and log1p-scale target modeling.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


TABULAR_FEATURE_COLS = [
    # Calendar & Domain Events
    "month",
    "day_of_year_sin",
    "day_of_year_cos",
    "day_of_week",
    "is_weekend",
    "is_stubble_burning",
    "is_diwali_window",
    "is_covid_lockdown",
    # Regime Persistence
    "is_prev_day_severe",
    # Pollutant Lags
    "pm25_lag1",
    "pm25_lag2",
    "pm25_lag3",
    "pm25_lag7",
    "pm10_lag1",
    "pm10_lag3",
    "no2_lag1",
    "no2_lag3",
    # Rolling Statistics (shifted by 1 to prevent leakage)
    "pm25_roll3_mean",
    "pm25_roll3_std",
    "pm25_roll7_mean",
    "pm25_roll7_std",
    # Meteorology & Inversion
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "humidity",
    "temp",
    "temp_lag1",
    "temp_inversion_proxy",
    # Spatial Neighbor (Gurugram) Lags & Regional Gradients
    "gurugram_pm25_lag1",
    "gurugram_pm10_lag1",
    "gurugram_no2_lag1",
    "spatial_grad_pm25",
    "spatial_grad_pm10",
    "spatial_grad_no2",
    # Explicit Auditable Physical Interactions
    "stubble_x_windspeed",
    "humidity_x_tempdrop",
    "upwind_stubble_transport",
]

SEQUENCE_RAW_COLS = [
    "PM2.5",
    "PM10",
    "NO2",
    "NOx",
    "CO",
    "SO2",
    "O3",
    "day_of_year_sin",
    "day_of_year_cos",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "humidity",
    "temp",
    "temp_inversion_proxy",
    "GURUGRAM_PM2.5",
    "GURUGRAM_PM10",
    "upwind_stubble_transport",
]

DIWALI_DATES = [
    pd.Timestamp("2015-11-11"),
    pd.Timestamp("2016-10-30"),
    pd.Timestamp("2017-10-19"),
    pd.Timestamp("2018-11-07"),
    pd.Timestamp("2019-10-27"),
]


@dataclass
class SplitDataset:
    """Holds tabular and sequence datasets for train, validation, and test splits."""
    dates_train: pd.Series
    dates_val: pd.Series
    dates_test: pd.Series
    
    # Tabular features (X) and Target (y)
    X_train: pd.DataFrame
    y_train: np.ndarray          # Real-scale PM2.5 (for baseline & inverse checks)
    y_train_target: np.ndarray   # Actual training target (either real or log1p)
    X_val: pd.DataFrame
    y_val: np.ndarray
    y_val_target: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    y_test_target: np.ndarray
    
    # Standardized Tabular features
    X_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    X_test_scaled: np.ndarray
    
    # Sequence inputs (N, seq_len=14, channels=18) standardized on train fold statistics
    X_seq_train: np.ndarray
    X_seq_val: np.ndarray
    X_seq_test: np.ndarray
    
    # Target transformation flag
    use_log_target: bool
    
    # Scalers and medians for auditing
    tabular_scaler: StandardScaler
    sequence_scaler: StandardScaler
    feature_names: List[str]


def engineer_features(df: pd.DataFrame, window_size: int = 14, forecast_horizon: int = 1) -> pd.DataFrame:
    """
    Generate all calendar, stubble burning, spatial, interaction, lag, rolling,
    and meteorological features for Delhi next-day (or t+h) PM2.5 forecasting.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)
    
    # Target: PM2.5 at forecast horizon t+h
    df["target_pm25"] = df["PM2.5"].shift(-forecast_horizon)
    df["target_pm25_log"] = np.log1p(df["target_pm25"])
    df["target_date"] = df["Date"].shift(-forecast_horizon)
    
    target_dt = pd.to_datetime(df["target_date"])
    df["month"] = target_dt.dt.month
    doy = target_dt.dt.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["day_of_week"] = target_dt.dt.dayofweek
    df["is_weekend"] = (target_dt.dt.dayofweek >= 5).astype(int)
    
    # Stubble burning window: Oct 15 to Nov 15 (Punjab/Haryana crop residue burning)
    month = target_dt.dt.month
    day = target_dt.dt.day
    is_stubble = ((month == 10) & (day >= 15)) | ((month == 11) & (day <= 15))
    df["is_stubble_burning"] = is_stubble.astype(int)
    
    # Festival / Natural Experiment Calendar Flags
    is_diwali = pd.Series(False, index=df.index)
    for d_date in DIWALI_DATES:
        window_start = d_date - pd.Timedelta(days=2)
        window_end = d_date + pd.Timedelta(days=2)
        is_diwali = is_diwali | ((target_dt >= window_start) & (target_dt <= window_end))
    df["is_diwali_window"] = is_diwali.astype(int)
    
    # COVID-19 Lockdown (2020-03-25 to 2020-06-08)
    lockdown_start = pd.Timestamp("2020-03-25")
    lockdown_end = pd.Timestamp("2020-06-08")
    df["is_covid_lockdown"] = ((target_dt >= lockdown_start) & (target_dt <= lockdown_end)).astype(int)
    
    # Multi-Day Episode Persistence (was yesterday Severe or Very Poor AQI?)
    prev_aqi = df["AQI_Bucket"].astype(str)
    df["is_prev_day_severe"] = prev_aqi.isin(["Severe", "Very Poor"]).astype(int)
    
    # Pollutant Lags (at day t)
    df["pm25_lag1"] = df["PM2.5"]
    df["pm25_lag2"] = df["PM2.5"].shift(1)
    df["pm25_lag3"] = df["PM2.5"].shift(2)
    df["pm25_lag7"] = df["PM2.5"].shift(6)
    
    df["pm10_lag1"] = df["PM10"]
    df["pm10_lag3"] = df["PM10"].shift(2)
    
    df["no2_lag1"] = df["NO2"]
    df["no2_lag3"] = df["NO2"].shift(2)
    
    # Shifted rolling statistics (avoiding target leakage)
    df["pm25_roll3_mean"] = df["PM2.5"].rolling(3).mean()
    df["pm25_roll3_std"] = df["PM2.5"].rolling(3).std()
    df["pm25_roll7_mean"] = df["PM2.5"].rolling(7).mean()
    df["pm25_roll7_std"] = df["PM2.5"].rolling(7).std()
    
    # Surface Meteorology (at day t)
    df["wind_speed"] = df["WS10M"]
    wd_rad = np.radians(df["WD10M"] % 360.0)
    df["wind_dir_sin"] = np.sin(wd_rad)
    df["wind_dir_cos"] = np.cos(wd_rad)
    df["humidity"] = df["RH2M"]
    df["temp"] = df["T2M"]
    df["temp_lag1"] = df["T2M"].shift(1)
    
    t_min_thresh = df["T2M_MIN"].quantile(0.25)
    ws_thresh = df["WS10M"].quantile(0.25)
    df["temp_inversion_proxy"] = (
        (df["T2M_MIN"] <= t_min_thresh) & (df["WS10M"] <= ws_thresh)
    ).astype(int)
    
    # Spatial Neighbor (Gurugram) Lags & Gradients
    # Prior to Gurugram monitoring coming online in 2016, fall back to Delhi's regional value
    g_pm25 = df["GURUGRAM_PM2.5"].fillna(df["PM2.5"])
    g_pm10 = df["GURUGRAM_PM10"].fillna(df["PM10"])
    g_no2 = df["GURUGRAM_NO2"].fillna(df["NO2"])
    
    df["gurugram_pm25_lag1"] = g_pm25
    df["gurugram_pm10_lag1"] = g_pm10
    df["gurugram_no2_lag1"] = g_no2
    
    df["spatial_grad_pm25"] = df["gurugram_pm25_lag1"] - df["pm25_lag1"]
    df["spatial_grad_pm10"] = df["gurugram_pm10_lag1"] - df["pm10_lag1"]
    df["spatial_grad_no2"] = df["gurugram_no2_lag1"] - df["no2_lag1"]
    
    # Explicit Auditable Physical Interactions
    # 1. Stubble burning under calm air
    df["stubble_x_windspeed"] = df["is_stubble_burning"] * df["wind_speed"]
    # 2. Moisture and diurnal temperature drop (inversion strength proxy)
    df["humidity_x_tempdrop"] = df["humidity"] * (df["T2M_MAX"] - df["T2M_MIN"])
    # 3. Upwind transport vector (North-West wind directed from Punjab/Haryana toward Delhi)
    # Peak transport angle is ~315 degrees
    upwind_factor = np.maximum(0.0, np.cos(np.radians(df["WD10M"] - 315.0)))
    df["upwind_stubble_transport"] = df["is_stubble_burning"] * upwind_factor * df["wind_speed"]
    
    return df


def build_sequence_windows(df: pd.DataFrame, valid_indices: np.ndarray, window_size: int = 14) -> np.ndarray:
    """
    Construct 3D sequence array of shape (N, window_size=14, num_channels)
    covering [idx - window_size + 1, ..., idx].
    """
    raw_mat = df[SEQUENCE_RAW_COLS].values
    sequences = []
    for idx in valid_indices:
        start_idx = idx - window_size + 1
        seq = raw_mat[start_idx : idx + 1]
        sequences.append(seq)
    return np.array(sequences, dtype=np.float32)


def create_split_dataset(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    window_size: int = 14,
    use_log_target: bool = True,
) -> SplitDataset:
    """
    Create leak-free SplitDataset with optional log1p-target scaling.
    Median imputation and scalers are fitted exclusively on the train split.
    """
    valid_flag = np.ones(len(df), dtype=bool)
    valid_flag[:window_size - 1] = False
    valid_flag[df["target_pm25"].isna()] = False
    
    train_idx = np.where(train_mask & valid_flag)[0]
    val_idx = np.where(val_mask & valid_flag)[0]
    test_idx = np.where(test_mask & valid_flag)[0]
    
    dates_train = df.loc[train_idx, "target_date"].reset_index(drop=True)
    dates_val = df.loc[val_idx, "target_date"].reset_index(drop=True)
    dates_test = df.loc[test_idx, "target_date"].reset_index(drop=True)
    
    # Targets: real-world µg/m³
    y_train = df.loc[train_idx, "target_pm25"].values.astype(np.float32)
    y_val = df.loc[val_idx, "target_pm25"].values.astype(np.float32)
    y_test = df.loc[test_idx, "target_pm25"].values.astype(np.float32)
    
    # Model target (log1p if requested, else raw)
    if use_log_target:
        y_train_target = np.log1p(y_train)
        y_val_target = np.log1p(y_val)
        y_test_target = np.log1p(y_test)
    else:
        y_train_target = y_train.copy()
        y_val_target = y_val.copy()
        y_test_target = y_test.copy()
    
    # Tabular features
    X_train_df = df.loc[train_idx, TABULAR_FEATURE_COLS].copy().reset_index(drop=True)
    X_val_df = df.loc[val_idx, TABULAR_FEATURE_COLS].copy().reset_index(drop=True)
    X_test_df = df.loc[test_idx, TABULAR_FEATURE_COLS].copy().reset_index(drop=True)
    
    # Recalibrate temperature inversion threshold strictly on train split
    t_min_train_q = df.loc[train_idx, "T2M_MIN"].quantile(0.25)
    ws_train_q = df.loc[train_idx, "WS10M"].quantile(0.25)
    for sub_df, idxs in [(X_train_df, train_idx), (X_val_df, val_idx), (X_test_df, test_idx)]:
        sub_df["temp_inversion_proxy"] = (
            (df.loc[idxs, "T2M_MIN"].values <= t_min_train_q) &
            (df.loc[idxs, "WS10M"].values <= ws_train_q)
        ).astype(int)
    
    # Leak-free median imputation fitted on train split only
    train_medians = X_train_df.median()
    X_train_df = X_train_df.fillna(train_medians)
    X_val_df = X_val_df.fillna(train_medians)
    X_test_df = X_test_df.fillna(train_medians)
    
    # Standardize tabular features
    tab_scaler = StandardScaler()
    X_train_scaled = tab_scaler.fit_transform(X_train_df.values).astype(np.float32)
    X_val_scaled = tab_scaler.transform(X_val_df.values).astype(np.float32)
    X_test_scaled = tab_scaler.transform(X_test_df.values).astype(np.float32)
    
    # Sequence median imputation and standardization
    seq_df = df.copy()
    seq_df["GURUGRAM_PM2.5"] = seq_df["GURUGRAM_PM2.5"].fillna(seq_df["PM2.5"])
    seq_df["GURUGRAM_PM10"] = seq_df["GURUGRAM_PM10"].fillna(seq_df["PM10"])
    
    seq_medians = seq_df.loc[train_idx, SEQUENCE_RAW_COLS].median().fillna(0.0)
    seq_df[SEQUENCE_RAW_COLS] = seq_df[SEQUENCE_RAW_COLS].fillna(seq_medians)
    
    seq_train = build_sequence_windows(seq_df, train_idx, window_size)
    seq_val = build_sequence_windows(seq_df, val_idx, window_size)
    seq_test = build_sequence_windows(seq_df, test_idx, window_size)
    
    N_tr, S, C = seq_train.shape
    seq_scaler = StandardScaler()
    seq_train_flat = seq_scaler.fit_transform(seq_train.reshape(-1, C))
    seq_val_flat = seq_scaler.transform(seq_val.reshape(-1, C))
    seq_test_flat = seq_scaler.transform(seq_test.reshape(-1, C))
    
    seq_train_norm = np.nan_to_num(seq_train_flat.reshape(N_tr, S, C).astype(np.float32))
    seq_val_norm = np.nan_to_num(seq_val_flat.reshape(seq_val.shape[0], S, C).astype(np.float32))
    seq_test_norm = np.nan_to_num(seq_test_flat.reshape(seq_test.shape[0], S, C).astype(np.float32))
    
    X_train_scaled = np.nan_to_num(X_train_scaled)
    X_val_scaled = np.nan_to_num(X_val_scaled)
    X_test_scaled = np.nan_to_num(X_test_scaled)
    
    return SplitDataset(
        dates_train=dates_train,
        dates_val=dates_val,
        dates_test=dates_test,
        X_train=X_train_df,
        y_train=y_train,
        y_train_target=y_train_target,
        X_val=X_val_df,
        y_val=y_val,
        y_val_target=y_val_target,
        X_test=X_test_df,
        y_test=y_test,
        y_test_target=y_test_target,
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        X_test_scaled=X_test_scaled,
        X_seq_train=seq_train_norm,
        X_seq_val=seq_val_norm,
        X_seq_test=seq_test_norm,
        use_log_target=use_log_target,
        tabular_scaler=tab_scaler,
        sequence_scaler=seq_scaler,
        feature_names=TABULAR_FEATURE_COLS,
    )


def get_primary_split(df_raw: pd.DataFrame, window_size: int = 14, use_log_target: bool = True) -> SplitDataset:
    """Primary 70% Train / 15% Validation / 15% Test chronological split."""
    df_feat = engineer_features(df_raw, window_size)
    n_total = len(df_feat)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    
    train_mask = np.zeros(n_total, dtype=bool)
    val_mask = np.zeros(n_total, dtype=bool)
    test_mask = np.zeros(n_total, dtype=bool)
    
    train_mask[:n_train] = True
    val_mask[n_train : n_train + n_val] = True
    test_mask[n_train + n_val :] = True
    
    return create_split_dataset(df_feat, train_mask, val_mask, test_mask, window_size, use_log_target)


def get_walk_forward_folds(df_raw: pd.DataFrame, window_size: int = 14, use_log_target: bool = True, forecast_horizon: int = 1) -> List[Tuple[str, SplitDataset]]:
    """
    Generate 5 expanding-window walk-forward cross-validation folds:
      - Fold 1: Train 2015 -> Test 2016
      - Fold 2: Train 2015-16 -> Test 2017
      - Fold 3: Train 2015-17 -> Test 2018
      - Fold 4: Train 2015-18 -> Test 2019
      - Fold 5: Train 2015-19 -> Test 2020 (Jan-Jul)
    """
    df_feat = engineer_features(df_raw, window_size, forecast_horizon=forecast_horizon)
    target_dt = pd.to_datetime(df_feat["target_date"])
    years = [2016, 2017, 2018, 2019, 2020]
    folds = []
    
    for test_year in years:
        fold_name = f"Train 2015-{test_year-1} -> Test {test_year}"
        train_val_mask = (target_dt.dt.year < test_year).values
        test_mask = (target_dt.dt.year == test_year).values
        
        # Chronological tail (15% of historical data) allocated for fold-specific meta-validation
        tv_indices = np.where(train_val_mask)[0]
        n_tv = len(tv_indices)
        n_val = max(int(0.15 * n_tv), 30)
        val_indices = set(tv_indices[n_tv - n_val :])
        
        train_mask = np.array([i in tv_indices and i not in val_indices for i in range(len(df_feat))])
        val_mask = np.array([i in val_indices for i in range(len(df_feat))])
        
        split_data = create_split_dataset(df_feat, train_mask, val_mask, test_mask, window_size, use_log_target)
        folds.append((fold_name, split_data))
        
    return folds
