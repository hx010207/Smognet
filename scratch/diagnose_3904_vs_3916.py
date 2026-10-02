import sys
sys.path.insert(0, ".")
import json
import joblib
import numpy as np
import pandas as pd
from data_loader import load_and_preprocess_delhi_data
from features import get_primary_split
from evaluate import compute_metrics
from models import SmogNetPipeline

df_raw, _ = load_and_preprocess_delhi_data("data")
primary_split = get_primary_split(df_raw, window_size=14, use_log_target=True)

# 1. Load cached primary_pipeline.pkl
pipe_cached = joblib.load("results/primary_pipeline.pkl")
p_cached = pipe_cached.predict_smognet(primary_split, subset="test")
m_cached = compute_metrics(primary_split.y_test, p_cached)
print("--- Cached Pipeline (results/primary_pipeline.pkl) ---")
print(f"RMSE: {m_cached['RMSE']:.4f}, MAE: {m_cached['MAE']:.4f}, R2: {m_cached['R2']:.4f}")
print(f"Meta-learner coefs: {pipe_cached.meta_learner.coef_}")
print(f"Meta-learner intercept: {pipe_cached.meta_learner.intercept_}")

# Check base learner predictions from cached pipeline
for name in pipe_cached.base_models_order:
    p_b = pipe_cached.predict_base_learner(name, primary_split, subset="test")
    m_b = compute_metrics(primary_split.y_test, p_b)
    print(f"  Base {name:20s}: RMSE={m_b['RMSE']:.4f}, R2={m_b['R2']:.4f}")

# 2. What about the run_enhanced_dropout_experiments 0% dropout logic?
# In run_enhanced_dropout_experiments.py lines 145-155:
# df_imp = pd.DataFrame(primary_split.X_test.copy().values, columns=primary_split.feature_names)
# imp_scaled = primary_split.tabular_scaler.transform(df_imp.values).astype(np.float32)
# corrupt_split = replace(primary_split, X_test=df_imp, X_test_scaled=imp_scaled)
from dataclasses import replace
df_clean = pd.DataFrame(primary_split.X_test.copy().values, columns=primary_split.feature_names)
clean_scaled = primary_split.tabular_scaler.transform(df_clean.values).astype(np.float32)
clean_split = replace(primary_split, X_test=df_clean, X_test_scaled=clean_scaled)
p_clean_split = pipe_cached.predict_smognet(clean_split, subset="test")
m_clean_split = compute_metrics(primary_split.y_test, p_clean_split)
print("\n--- Cached Pipeline on corrupt_split (0% dropout) ---")
print(f"RMSE: {m_clean_split['RMSE']:.4f}, MAE: {m_clean_split['MAE']:.4f}, R2: {m_clean_split['R2']:.4f}")

# 3. What about Phase B results? Where did 39.16 come from?
with open("phase_b_results.json", "r") as f:
    pb = json.load(f)
print("\n--- phase_b_results.json sensor_dropout_robustness 0% ---")
print(pb["sensor_dropout_robustness"]["dropout_0%"]["SmogNet_stacked_ensemble"])
