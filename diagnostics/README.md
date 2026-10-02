# Smog Net Diagnostic & Validation Scripts

This directory houses internal diagnostic scripts, ablation sanity checks, and unit-level hardware/verification tests developed during the experimental workflow.

### Files Catalog
- `phase1_diagnostics.py`: Initial baseline check and target distribution diagnostics.
- `diagnose_lr.py`: Linear regression stability investigation across walk-forward annual regimes.
- `test_pipeline_quick.py`: Fast GPU execution and non-negative Ridge OOF meta-learner verification harness.
- `test_corrupt_pipeline.py`: Sensor dropout telemetry degradation simulation test harness.
- `run_updated_phase_b_modules.py`: Execution harness for evaluating multi-horizon forecasting and sensor dropout using the true 5-fold OOF non-negative Ridge meta-learner.

All primary end-to-end experimental benchmarks are executed via the single reproducible entrypoint `run_all.py` at the project root.
