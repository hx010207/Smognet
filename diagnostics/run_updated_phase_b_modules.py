import json
import numpy as np
import pandas as pd
from data_loader import load_and_preprocess_delhi_data
from features import get_primary_split
from models import SmogNetPipeline
from phase_b_and_baselines import run_multi_horizon_forecasting, run_sensor_dropout_robustness

print("=== Running Updated Phase B Modules (True Meta-Learner) ===")
df_raw, _ = load_and_preprocess_delhi_data("data")
primary_split = get_primary_split(df_raw, window_size=14, use_log_target=True)

with open("best_params.json", "r") as f:
    best_params = json.load(f)
best_xgb = best_params["xgboost"]
best_bilstm = best_params["bilstm_attention"]

primary_pipeline = SmogNetPipeline(
    seed=42, use_gpu=True, use_advanced_branches=True,
    xgb_params=best_xgb, bilstm_params=best_bilstm
)
primary_pipeline.train_baselines_and_trees(primary_split)
primary_pipeline.train_deep_models(primary_split, epochs_lstm=20, epochs_mlp=25)
primary_pipeline.fit_meta_learner_oof(primary_split, n_splits=5, alpha=10.0)

print("\n--- 1. Sensor Dropout Robustness with True Meta-Learner ---")
dropout_res = run_sensor_dropout_robustness(primary_pipeline, primary_split)

print("\n--- 2. Multi-Horizon Forecasting with True Meta-Learner ---")
multi_horizon_res = run_multi_horizon_forecasting(df_raw, best_xgb)

# Update phase_b_results.json
with open("phase_b_results.json", "r") as f:
    pb = json.load(f)

pb["sensor_dropout_robustness"] = dropout_res
pb["multi_horizon_forecasting"] = multi_horizon_res

with open("phase_b_results.json", "w") as f:
    json.dump(pb, f, indent=2)
print("\n[OK] Updated phase_b_results.json with true meta-learner numbers.")

# Update results.json
with open("results.json", "r") as f:
    res = json.load(f)

res["phase_b_experimental_rigor"]["sensor_dropout_robustness"] = dropout_res
res["phase_b_experimental_rigor"]["multi_horizon_forecasting"] = multi_horizon_res

with open("results.json", "w") as f:
    json.dump(res, f, indent=2)
print("[OK] Updated results.json with true meta-learner numbers.")
