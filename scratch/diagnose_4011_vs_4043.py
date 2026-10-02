import sys
sys.path.insert(0, ".")
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from data_loader import load_and_preprocess_delhi_data
from features import get_primary_split, TABULAR_FEATURE_COLS
from models import SmogNetPipeline
from evaluate import compute_metrics

df_raw, _ = load_and_preprocess_delhi_data("data")
primary_split = get_primary_split(df_raw, window_size=14, use_log_target=True)

with open("best_params.json", "r") as f:
    best_params = json.load(f)
best_xgb = best_params["xgboost"]
best_bilstm = best_params["bilstm_attention"]

# Method 1: SmogNetPipeline (which produced 40.43)
pipe = SmogNetPipeline(
    seed=42, use_gpu=True, use_advanced_branches=True,
    xgb_params=best_xgb, bilstm_params=best_bilstm
)
pipe.train_baselines_and_trees(primary_split)
p_xgb_pipe = np.expm1(pipe.models["XGBoost"].predict(primary_split.X_test))
m_pipe = compute_metrics(primary_split.y_test, p_xgb_pipe)
print(f"Pipeline XGBoost: RMSE = {m_pipe['RMSE']:.4f}, params = {pipe.models['XGBoost'].get_params()}")

# Method 2: Ablation loop code (lines 516-520 of phase_b_and_baselines.py)
model_ablation = xgb.XGBRegressor(**best_xgb_params if 'best_xgb_params' in locals() else best_xgb, random_state=42)
X_tr_sub = primary_split.X_train[TABULAR_FEATURE_COLS].values
X_te_sub = primary_split.X_test[TABULAR_FEATURE_COLS].values
model_ablation.fit(X_tr_sub, primary_split.y_train_target)
p_log_abl = model_ablation.predict(X_te_sub)
p_real_abl = np.expm1(p_log_abl)
m_abl = compute_metrics(primary_split.y_test, p_real_abl)
print(f"Ablation XGBoost: RMSE = {m_abl['RMSE']:.4f}, params = {model_ablation.get_params()}")

# What if we pass DataFrame vs numpy array?
# What if we use tree_method='hist', device='cuda' vs CPU?
model_cpu = xgb.XGBRegressor(**best_xgb, random_state=42)
model_cpu.fit(primary_split.X_train, primary_split.y_train_target)
p_cpu = np.expm1(model_cpu.predict(primary_split.X_test))
print(f"CPU DataFrame XGBoost: RMSE = {compute_metrics(primary_split.y_test, p_cpu)['RMSE']:.4f}")

# What if tree_method='hist' device='cuda'?
model_gpu = xgb.XGBRegressor(**best_xgb, tree_method="hist", device="cuda", random_state=42)
model_gpu.fit(primary_split.X_train, primary_split.y_train_target)
p_gpu = np.expm1(model_gpu.predict(primary_split.X_test))
print(f"GPU DataFrame XGBoost: RMSE = {compute_metrics(primary_split.y_test, p_gpu)['RMSE']:.4f}")
