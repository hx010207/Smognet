import sys
sys.path.insert(0, ".")
import json
import numpy as np
import pandas as pd
from dataclasses import replace
from data_loader import load_and_preprocess_delhi_data
from features import get_primary_split, TABULAR_FEATURE_COLS
from models import SmogNetPipeline
from evaluate import compute_metrics

def main():
    df_raw, _ = load_and_preprocess_delhi_data("data")
    primary_split = get_primary_split(df_raw, window_size=14, use_log_target=True)

    with open("best_params.json", "r") as f:
        best_params = json.load(f)
    best_xgb = best_params["xgboost"]
    best_bilstm = best_params["bilstm_attention"]

    pipeline = SmogNetPipeline(
        seed=42, use_gpu=True, use_advanced_branches=True,
        xgb_params=best_xgb, bilstm_params=best_bilstm
    )
    pipeline.train_baselines_and_trees(primary_split)
    pipeline.train_deep_models(primary_split, epochs_lstm=25, epochs_mlp=30)
    pipeline.fit_meta_learner_oof(primary_split, n_splits=5, alpha=10.0)

    # Clean predictions
    p_smog_clean = pipeline.predict_smognet(primary_split, subset="test")
    xgb_model = pipeline.models["XGBoost"]
    p_xgb_clean = np.expm1(xgb_model.predict(primary_split.X_test))

    print(f"Clean SmogNet RMSE: {compute_metrics(primary_split.y_test, p_smog_clean)['RMSE']:.2f}")
    print(f"Clean XGBoost RMSE: {compute_metrics(primary_split.y_test, p_xgb_clean)['RMSE']:.2f}")

    # Test XGBoost native NaN handling
    X_test_orig = primary_split.X_test.copy().values
    train_medians = primary_split.X_train.median().values
    
    np.random.seed(42)
    mask = np.random.rand(*X_test_orig.shape) < 0.25
    
    # 1. Median imputed
    X_test_imp = X_test_orig.copy()
    for col_idx in range(X_test_imp.shape[1]):
        X_test_imp[mask[:, col_idx], col_idx] = train_medians[col_idx]
    df_imp = pd.DataFrame(X_test_imp, columns=primary_split.feature_names)
    p_xgb_imp = np.expm1(xgb_model.predict(df_imp))
    print(f"25% drop - XGBoost (Median Imputed) RMSE: {compute_metrics(primary_split.y_test, p_xgb_imp)['RMSE']:.2f}")

    # 2. Native NaN
    X_test_nan = X_test_orig.copy().astype(float)
    X_test_nan[mask] = np.nan
    df_nan = pd.DataFrame(X_test_nan, columns=primary_split.feature_names)
    p_xgb_nan = np.expm1(xgb_model.predict(df_nan))
    print(f"25% drop - XGBoost (Native NaN) RMSE: {compute_metrics(primary_split.y_test, p_xgb_nan)['RMSE']:.2f}")

    # SmogNet on imputed
    corrupt_scaled = primary_split.tabular_scaler.transform(df_imp.values).astype(np.float32)
    corrupt_split = replace(primary_split, X_test=df_imp, X_test_scaled=corrupt_scaled)
    p_smog_imp = pipeline.predict_smognet(corrupt_split, subset="test")
    print(f"25% drop - SmogNet RMSE: {compute_metrics(primary_split.y_test, p_smog_imp)['RMSE']:.2f}")

if __name__ == "__main__":
    main()
