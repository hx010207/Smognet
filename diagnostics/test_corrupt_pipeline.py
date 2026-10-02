import json
from dataclasses import replace
import numpy as np
import pandas as pd
from data_loader import load_and_preprocess_delhi_data
from features import get_primary_split
from models import SmogNetPipeline

df, _ = load_and_preprocess_delhi_data("data")
split = get_primary_split(df, window_size=14, use_log_target=True)

with open("best_params.json") as f:
    best_params = json.load(f)

pipe = SmogNetPipeline(
    seed=42,
    use_gpu=True,
    use_advanced_branches=True,
    xgb_params=best_params.get("xgb_params"),
    bilstm_params=best_params.get("bilstm_params"),
)
pipe.train_baselines_and_trees(split)
pipe.train_deep_models(split, epochs_lstm=15, epochs_mlp=20)
pipe.fit_meta_learner_oof(split, n_splits=5, alpha=10.0)

# Test clean prediction
clean_pred = pipe.predict_smognet(split, "test")
print("Clean prediction shape:", clean_pred.shape, "mean:", clean_pred.mean())

# Test corrupted tabular features
X_corrupt = split.X_test.copy().values
mask = np.random.rand(*X_corrupt.shape) < 0.25
medians = split.X_train.median().values
for j in range(X_corrupt.shape[1]):
    X_corrupt[mask[:, j], j] = medians[j]

df_corrupt = pd.DataFrame(X_corrupt, columns=split.feature_names)
corrupt_scaled = split.tabular_scaler.transform(df_corrupt.values).astype(np.float32)
corrupt_split = replace(split, X_test=df_corrupt, X_test_scaled=corrupt_scaled)

corrupt_pred = pipe.predict_smognet(corrupt_split, "test")
print("Corrupt prediction shape:", corrupt_pred.shape, "mean:", corrupt_pred.mean())
