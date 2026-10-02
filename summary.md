# Smog Net: Project Orientation & Summary

## 1. Overview
Smog Net forecasts next-day PM2.5 in the Delhi-NCR airshed (Indo-Gangetic Basin), driven by autumn post-monsoon agricultural crop-residue burning and winter thermal inversions. It uses a physically motivated stacked hybrid ensemble: gradient-boosted trees capture non-linear chemistry, deep sequence models (BiLSTM with attention, Tabular MLP) learn temporal patterns, and a non-negative Ridge meta-learner blends their predictions. Training uses authentic Central Pollution Control Board (CPCB) air monitors, NASA POWER surface weather, and NASA FIRMS satellite active fire data.

## 2. What's in the Pipeline
- **Data Scope**: 2,009 daily records (2015-01-01 to 2020-07-01) from CPCB monitors, NASA POWER weather, and 113,557 NASA FIRMS fire detections in Punjab/Haryana. Framed strictly as a historical benchmarking study of the 2015–2020 extreme pollution regime (PATH B).
- **Features**: 37 core tabular ground predictors (calendar cycles, pollutant lags/rolling stats, surface weather, and upwind Gurugram spatial neighbor), expanding to **40 predictors** when augmented with the 3 NASA FIRMS satellite fire features (`firms_log_frp_lag1`, `firms_fire_count`, `firms_transport_index`).
- **Base Models**: Tuned XGBoost, Random Forest, Gradient Boosting, Tabular MLP, Sequence LSTM, BiLSTM with attention, and CNN-BiLSTM with attention.
- **Meta-Learner**: Non-negative Ridge regression (alpha = 10.0) trained on 5-fold Out-Of-Fold (OOF) predictions to prevent single-model weight inflation.
- **Evaluation**: 5-fold expanding walk-forward cross-validation by year (2016–2020) without future lookahead leakage, backed by a unified [`evaluation_ledger.csv`](file:///c:/Users/workh/OneDrive/Desktop/smognet/evaluation_ledger.csv).

## 3. Headline Results
- **Primary Finding — Telemetry Degradation Resilience**: Evaluated across 50 Monte Carlo random masks per level, the ensemble's error buffer over median-imputed XGBoost widens from **$+1.39\ \mu\text{g/m}^3$** (0% dropout) to **$+3.92 \pm 0.23\ \mu\text{g/m}^3$** (10% dropout, $p = 3.93 \times 10^{-35}$), **$+6.40 \pm 0.23\ \mu\text{g/m}^3$** (25% dropout, $p = 1.15 \times 10^{-45}$), and **$+9.28 \pm 0.16\ \mu\text{g/m}^3$** (50% catastrophic dropout, $p = 4.63 \times 10^{-61}$). Over native-NaN XGBoost, the buffer expands up to **$+43.06 \pm 1.13\ \mu\text{g/m}^3$**.
- **Point Forecasts Comparable to Tuned XGBoost on Clean Data**: On clean telemetry, Smog Net achieves **40.40 ± 12.80 µg/m³** 5-fold walk-forward RMSE ($R^2 = 0.7565 \pm 0.0751$, MAE = **25.68 ± 7.28 µg/m³**) and **39.16 µg/m³** on the fixed single-split holdout (2019-09-02 to 2020-07-01). Tuned XGBoost achieves 40.43 µg/m³; Diebold-Mariano ($p = 0.2288$) and paired $t$-test ($p = 0.1315$) confirm statistical equivalence on clean point forecasts. Point superiority is explicitly not claimed.
- **Classical Baselines Fail**: Classical models break down on smog shocks (SARIMA RMSE: **97.14 ± 26.45 µg/m³**, $R^2 = -0.4384$; Prophet RMSE: **90.41 ± 67.30 µg/m³**, $R^2 = -0.3818$).
- **Multi-Horizon Disclosure**: Persistence wins at day 1 ($t+1$ RMSE 41.02 vs. 43.34 µg/m³) from strong autocorrelation ($r \approx 0.86$). Smog Net wins at 3 days (**59.76 vs. 61.74 µg/m³**) and 7 days (**58.09 vs. 72.93 µg/m³**, holding $R^2$ at 0.4957 vs. 0.1813).
- **Fire Counterfactual**: Satellite telemetry refines test RMSE to **39.13 µg/m³** ($R^2 = 0.7980$). TreeSHAP isolates a **+17.80 µg/m³** next-day surge from harvest burning.
- **Airshed Transferability**: The frozen Delhi model transfers across the Indo-Gangetic Basin (Patna $R^2 = \mathbf{0.73}$, Gurugram $R^2 = \mathbf{0.57}$, Lucknow $R^2 = \mathbf{0.54}$), but fails in peninsular/coastal climates (Bengaluru $R^2 = -0.53$, Chennai $R^2 = -0.08$), confirming that the model's physical priors are tied to northern winter boundary layer meteorology.

## 4. Reconciled RMSE Ledger
- **40.40 ± 12.80 µg/m³** ($R^2 = 0.7565 \pm 0.0751$, MAE = **25.68 ± 7.28 µg/m³**): 5-fold expanding walk-forward cross-validation mean across test years 2016–2020.
- **39.16 µg/m³** ($R^2 = 0.7983$, MAE = 24.54): Fixed single-split benchmark under standard 20/25 epoch neural schedule (test dates 2019-09-02 to 2020-07-01, 304 holdout days). Compares with 39.04 µg/m³ ($R^2 = 0.7995$) under extended 25/30 epoch schedule saved in primary_pipeline.pkl for 50-mask sensor dropout experiments.
- **40.43 µg/m³** ($R^2 = 0.7850$): Standalone Tuned XGBoost fixed-split benchmark evaluated via GPU pipeline checkpoint (`tree_method="hist", device="cuda"`).
- **40.11 µg/m³** ($R^2 = 0.7883$): Single-split ablation reference baseline (Standalone Tuned XGBoost trained on CPU on 37 tabular features, 2019-09-02 to 2020-07-01 holdout; differs from 40.43 solely by CPU exact splitting vs. GPU histogram binning under identical seed and hyperparameters).
- **39.13 µg/m³** ($R^2 = 0.7980$): Fixed single-split benchmark augmented with 3 NASA FIRMS satellite active-fire features (40 features total).

## 5. What Was Fixed Along the Way
- **Core Claim Reframing**: Reframed away from point-forecast superiority over Tuned XGBoost ($p = 0.1315$, n.s.) to focus on statistically validated resilience under telemetry degradation ($p < 10^{-34}$) and multi-day lead times ($t+3, t+7$).
- **50-Mask Monte Carlo Dropout Upgrade**: Replaced 4-row point-estimate table with 50 Monte Carlo masks reporting mean ± 95% CIs, paired $t$-tests, Wilcoxon tests, conformal coverage calibration, and 5 realistic structured outage scenarios. Wilcoxon signed-rank p-values saturates at the floor $7.55 \times 10^{-10}$ across 10%, 25%, and 50% levels because Smog Net strictly beats XGBoost in 50/50 masks at every level ($W=0$, saturating test floor).
- **Metric Ledger Concordance**: Resolved the 40.43 vs. 40.11 baseline divergence, 39.16 holdout number, and 37 vs. 40 feature counts in [`evaluation_ledger.csv`](file:///c:/Users/workh/OneDrive/Desktop/smognet/evaluation_ledger.csv).
- **Terminology & Scope**: Replaced "physics-informed" with "physically motivated" and constrained geographic scope to the Indo-Gangetic Basin.
- **Data Recency (PATH B)**: Stripped speculative operational claims; reframed as a historical benchmark (2015–2020) of extreme pollution episodes.

## 6. File Map
- **[`run_all.py`](file:///c:/Users/workh/OneDrive/Desktop/smognet/run_all.py)**: Master entry point script reproducing the full pipeline (~3-5 min GPU, ~15-20 min CPU).
- **[`run_enhanced_dropout_experiments.py`](file:///c:/Users/workh/OneDrive/Desktop/smognet/run_enhanced_dropout_experiments.py)**: Upgraded 50-mask Monte Carlo sensor degradation and realistic outage simulation script.
- **[`evaluation_ledger.csv`](file:///c:/Users/workh/OneDrive/Desktop/smognet/evaluation_ledger.csv)**: Unified evaluation ledger mapping every reported metric to its exact date range, feature count, protocol, and script.
- **[`table1_model_comparison.md`](file:///c:/Users/workh/OneDrive/Desktop/smognet/table1_model_comparison.md)**: Reconciled benchmark table matching Table 1 of the manuscript.
- **[`manuscript_summary.md`](file:///c:/Users/workh/OneDrive/Desktop/smognet/manuscript_summary.md)**: Full academic research summary prepared for *Scientific Reports* (includes Data and Code Availability statements with **`[TODO before submission: 10.5281/zenodo.XXXXXXX]`**).
- **[`results/`](file:///c:/Users/workh/OneDrive/Desktop/smognet/results)**: High-resolution figures and CSV data across categories `01` to `07`.
