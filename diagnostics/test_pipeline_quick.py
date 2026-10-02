import json
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
pipe.train_deep_models(split, epochs_lstm=20, epochs_mlp=25)
weights = pipe.fit_meta_learner_oof(split, n_splits=5, alpha=10.0)

print("Trained meta-learner features:", pipe.meta_feature_names)
print("Trained meta-weights:", weights)
print("Meta-learner coef_:", pipe.meta_learner.coef_)
print("Meta-learner intercept_:", pipe.meta_learner.intercept_)
