"""
=============================================================================
 Smog Net: Multimodal Stacking Ensemble for Day-Ahead Delhi PM2.5 Forecasting
=============================================================================
 Single Documented Reproducibility Entrypoint (run_all.py)
 
 This script reproduces the entire experimental pipeline end-to-end:
   1. Environment & Hardware Diagnostics (Python 3.10+, CUDA detection)
   2. Data Ingestion & Preprocessing (CPCB, NASA POWER, NASA FIRMS active fire)
   3. Feature Engineering & Walk-Forward CV Splits (37 causal features)
   4. Phase 1 & 2 Pipeline (5-fold expanding walk-forward cross validation,
      base learners, deep sequence/attention models, and 5-fold OOF Ridge stacking)
   5. Classical Baselines & Experimental Rigor (SARIMA, Prophet, Diebold-Mariano
      significance tests, conformal prediction, spatial transfer, sensor dropout)
   6. Provenance & Benchmark Consolidation (results.json)
   7. Publication-Quality Figure & Table Generation (results/ directory, >= 300 DPI)

 Expected Runtime:
   - GPU (NVIDIA CUDA acceleration): ~3 - 5 minutes
   - CPU (16 cores): ~15 - 20 minutes

 Random Seed Architecture:
   - Global deterministic seed: 42 (fixed across all data splits, tree models,
     PyTorch initializations, and cross-validation folds for exact reproducibility).
   - Deliberate exception: 5-seed stability test in Phase 2 robustness check
     evaluates seeds [42, 123, 456, 789, 999] specifically to measure model
     variance across random weight initializations (reported as mean ± std).
=============================================================================
"""

import sys
import os
import time
import json
from pathlib import Path

# Step 0: Environment Diagnostics
print("=" * 80)
print(" SMOG NET: FULL REPRODUCIBILITY PIPELINE")
print("=" * 80)
print(f"[Environment] Python Version: {sys.version.split()[0]}")

try:
    import torch
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
    print(f"[Hardware] PyTorch {torch.__version__} | CUDA Available: {cuda_avail} ({device_name})")
except ImportError:
    print("[Hardware] PyTorch not detected or CPU fallback.")

start_time = time.time()

# Step 1: Data Ingestion & Alignment
print("\n" + "=" * 80)
print(" [Step 1/5] Ingesting and Preprocessing Telemetry Data")
print("=" * 80)
from data_loader import load_and_preprocess_delhi_data
df_raw, has_met = load_and_preprocess_delhi_data("data")
print(f"  [OK] Ingested {len(df_raw)} daily observations from 2015-01-01 to 2020-07-01.")

# Step 2: Feature Engineering & Primary Split
print("\n" + "=" * 80)
print(" [Step 2/5] Engineering 37 Causal Features & Constructing Splits")
print("=" * 80)
from features import engineer_features, get_primary_split, get_walk_forward_folds
split_primary = get_primary_split(df_raw, window_size=14, use_log_target=True)
print(f"  [OK] Engineered feature set shape: {split_primary.X_train.shape[1]} features.")
print(f"  [OK] Primary benchmark split: Train={len(split_primary.y_train)}, Test={len(split_primary.y_test)} rows.")

# Step 3: Core Cross-Validation & Meta-Learner Stacking Pipeline
print("\n" + "=" * 80)
print(" [Step 3/5] Running 5-Fold Walk-Forward Cross-Validation & OOF Stacking")
print("=" * 80)
from phase2_pipeline import run_phase2_full
phase2_metrics, best_params = run_phase2_full()
print("  [OK] Walk-forward cross-validation completed across 5 annual folds (2016-2020).")

# Step 4: Experimental Rigor, Spatial Transfer, & Robustness Modules
print("\n" + "=" * 80)
print(" [Step 4/5] Executing Rigor Baselines, Multi-Horizon, & Robustness Testing")
print("=" * 80)
from phase_b_and_baselines import run_all_phase_b
phase_b_results = run_all_phase_b()
print("  [OK] Baselines, multi-horizon, spatial transfer, and robustness testing completed.")

# Step 5: Consolidate Results & Generate Publication Visuals
print("\n" + "=" * 80)
print(" [Step 5/5] Consolidating Results & Rendering Publication Deliverables")
print("=" * 80)
from consolidate_final_results import consolidate
consolidate()

from generate_all_publication_results import generate_all_results
generate_all_results(Path("results"))

total_elapsed = time.time() - start_time
print("\n" + "=" * 80)
print(f" SMOG NET PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {total_elapsed/60:.2f} MINUTES")
print(" Authoritative results saved to results.json and results/ directory.")
print("=" * 80)
