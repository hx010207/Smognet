import sys
sys.path.insert(0, ".")
import json
import joblib
import numpy as np
import pandas as pd
import torch
from data_loader import load_and_preprocess_delhi_data
from features import get_primary_split
from evaluate import compute_metrics
from models import SmogNetPipeline

df_raw, _ = load_and_preprocess_delhi_data("data")
primary_split = get_primary_split(df_raw, window_size=14, use_log_target=True)

with open("best_params.json", "r") as f:
    best_params = json.load(f)
best_xgb = best_params["xgboost"]
best_bilstm = best_params["bilstm_attention"]

print("1. Testing loaded results/primary_pipeline.pkl...")
pipe_cached = joblib.load("results/primary_pipeline.pkl")
p_cached = pipe_cached.predict_smognet(primary_split, subset="test")
m_cached = compute_metrics(primary_split.y_test, p_cached)
print(f"Cached Pipeline: RMSE={m_cached['RMSE']:.4f}, MAE={m_cached['MAE']:.4f}, R2={m_cached['R2']:.4f}")
print("Meta-learner coefs (cached):", dict(zip(pipe_cached.meta_feature_names, pipe_cached.meta_learner.coef_)))
print("Meta-learner intercept (cached):", pipe_cached.meta_learner.intercept_)

# Now let's train fresh with epochs_lstm=20, epochs_mlp=25 (as in generate_all_publication_results.py)
print("\n2. Training fresh with epochs_lstm=20, epochs_mlp=25...")
pipe_20 = SmogNetPipeline(
    seed=42, use_gpu=True, use_advanced_branches=True,
    xgb_params=best_xgb, bilstm_params=best_bilstm
)
pipe_20.train_baselines_and_trees(primary_split)
pipe_20.train_deep_models(primary_split, epochs_lstm=20, epochs_mlp=25)
pipe_20.fit_meta_learner_oof(primary_split, n_splits=5, alpha=10.0)
p_20 = pipe_20.predict_smognet(primary_split, subset="test")
m_20 = compute_metrics(primary_split.y_test, p_20)
print(f"Fresh (20/25 epochs): RMSE={m_20['RMSE']:.4f}, MAE={m_20['MAE']:.4f}, R2={m_20['R2']:.4f}")
print("Meta-learner coefs (20/25):", dict(zip(pipe_20.meta_feature_names, pipe_20.meta_learner.coef_)))
print("Meta-learner intercept (20/25):", pipe_20.meta_learner.intercept_)

# Now let's train fresh with epochs_lstm=25, epochs_mlp=30 (as in run_enhanced_dropout_experiments.py and phase_b_and_baselines.py)
print("\n3. Training fresh with epochs_lstm=25, epochs_mlp=30...")
pipe_25 = SmogNetPipeline(
    seed=42, use_gpu=True, use_advanced_branches=True,
    xgb_params=best_xgb, bilstm_params=best_bilstm
)
pipe_25.train_baselines_and_trees(primary_split)
pipe_25.train_deep_models(primary_split, epochs_lstm=25, epochs_mlp=30)
pipe_25.fit_meta_learner_oof(primary_split, n_splits=5, alpha=10.0)
p_25 = pipe_25.predict_smognet(primary_split, subset="test")
m_25 = compute_metrics(primary_split.y_test, p_25)
print(f"Fresh (25/30 epochs): RMSE={m_25['RMSE']:.4f}, MAE={m_25['MAE']:.4f}, R2={m_25['R2']:.4f}")
print("Meta-learner coefs (25/30):", dict(zip(pipe_25.meta_feature_names, pipe_25.meta_learner.coef_)))
print("Meta-learner intercept (25/30):", pipe_25.meta_learner.intercept_)
