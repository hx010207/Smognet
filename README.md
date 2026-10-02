# Smog Net: Multimodal Physically Motivated Ensemble for Day-Ahead PM2.5 Forecasting in the Delhi-NCR Airshed

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Overview

**Smog Net** is a physically motivated stacked hybrid forecasting ensemble designed for day-ahead particulate matter ($\text{PM}_{2.5}$) prediction in the Delhi National Capital Region (NCR), Indo-Gangetic Basin. The architecture combines gradient-boosted decision trees (Tuned XGBoost, Random Forest, Gradient Boosting) for non-linear atmospheric chemical relationships, deep sequence neural networks (BiLSTM with temporal attention, Tabular MLP) for autoregressive memory and multi-day stagnation, and a 5-fold Out-Of-Fold (OOF) non-negative Ridge meta-learner that blends base model predictions without single-model weight inflation. Evaluated across a 2,009-day historical record (2015-01-01 to 2020-07-01) capturing natural shock experiments (the 2016 Great Smog and the 2020 national COVID-19 lockdown), Smog Net achieves an expanding 5-fold walk-forward RMSE of **$40.40 \pm 12.80\ \mu\text{g/m}^3$** ($R^2 = 0.7565 \pm 0.0751$, $\text{MAE} = 25.68 \pm 7.28\ \mu\text{g/m}^3$), which is statistically indistinguishable from a standalone tuned XGBoost baseline ($40.43\ \mu\text{g/m}^3$; $p = 0.1315$ paired $t$-test, $p = 0.2288$ Diebold-Mariano). Rather than claiming point-forecast superiority on pristine inputs, Smog Net's primary empirical contribution is its **proven resilience under sensor degradation and telemetry outages**: across 50 Monte Carlo random corruption masks per level, the ensemble's error buffer over median-imputed XGBoost widens from **$+1.39\ \mu\text{g/m}^3$** (0% dropout) to **$+3.92 \pm 0.23\ \mu\text{g/m}^3$** (10% dropout, $p = 3.93 \times 10^{-35}$), **$+6.40 \pm 0.23\ \mu\text{g/m}^3$** (25% dropout, $p = 1.15 \times 10^{-45}$), and **$+9.28 \pm 0.16\ \mu\text{g/m}^3$** at 50% catastrophic dropout ($p = 4.63 \times 10^{-61}$), expanding up to **$+43.06 \pm 1.13\ \mu\text{g/m}^3$** over native missing value routing.

---

## Key Experimental Results

| Metric Dimension | Smog Net Multimodal Ensemble | Comparator Baseline | Statistical Test & Significance |
| :--- | :--- | :--- | :--- |
| **Clean 5-Fold Walk-Forward** | **$\text{RMSE} = 40.40 \pm 12.80\ \mu\text{g/m}^3$**<br>$\text{MAE} = 25.68 \pm 7.28\ \mu\text{g/m}^3$<br>$R^2 = 0.7565 \pm 0.0751$ | Tuned XGBoost: $40.43\ \mu\text{g/m}^3$<br>Random Forest: $43.91\ \mu\text{g/m}^3$<br>BiLSTM-Attn: $48.24\ \mu\text{g/m}^3$ | Diebold-Mariano $p = 0.2288$ (n.s.)<br>Paired $t$-test $p = 0.1315$ (n.s.)<br>*Point superiority is disclaimed* |
| **Sensor Dropout (10%)** | **$\text{RMSE} = 42.11 \pm 0.49\ \mu\text{g/m}^3$** | XGBoost (Imputed): $46.03 \pm 0.68$<br>XGBoost (NaN): $55.15 \pm 0.84$ | Buffer: **$+3.92 \pm 0.23\ \mu\text{g/m}^3$** ($p = 3.93 \times 10^{-35}$)<br>Wilcoxon $p = 7.55 \times 10^{-10}$ |
| **Sensor Dropout (25%)** | **$\text{RMSE} = 46.67 \pm 0.52\ \mu\text{g/m}^3$** | XGBoost (Imputed): $53.08 \pm 0.66$<br>XGBoost (NaN): $74.22 \pm 1.11$ | Buffer: **$+6.40 \pm 0.23\ \mu\text{g/m}^3$** ($p = 1.15 \times 10^{-45}$)<br>Wilcoxon $p = 7.55 \times 10^{-10}$ |
| **Sensor Dropout (50%)** | **$\text{RMSE} = 56.03 \pm 0.63\ \mu\text{g/m}^3$** | XGBoost (Imputed): $65.31 \pm 0.71$<br>XGBoost (NaN): $99.09 \pm 0.89$ | Buffer: **$+9.28 \pm 0.16\ \mu\text{g/m}^3$** ($p = 4.63 \times 10^{-61}$)<br>NaN Buffer: **$+43.06 \pm 1.13\ \mu\text{g/m}^3$** |
| **Multi-Horizon Disclosure** | $t+1$: $43.34\ \mu\text{g/m}^3$<br>**$t+3$: $59.76\ \mu\text{g/m}^3$** ($R^2 = 0.46$)<br>**$t+7$: $58.09\ \mu\text{g/m}^3$** ($R^2 = 0.50$) | $t+1$ Persistence: **$41.02\ \mu\text{g/m}^3$**<br>$t+3$ Persistence: $61.74\ \mu\text{g/m}^3$<br>$t+7$ Persistence: $72.93\ \mu\text{g/m}^3$ ($R^2 = 0.18$) | **Honest disclosure**: Persistence wins at $t+1$ ($r \approx 0.86$). Smog Net dominates at multi-day leads ($+14.84\ \mu\text{g/m}^3$ at $t+7$). |
| **Spaceborne Fire Attribution** | Ground: $39.16\ \mu\text{g/m}^3$<br>**Ground + Satellite: $39.13\ \mu\text{g/m}^3$** | 113,557 NASA FIRMS active fire pixels (Punjab/Haryana MODIS FRP) | TreeSHAP isolates **$+17.80\ \mu\text{g/m}^3$** next-day surge during peak harvest fires. |
| **Regional Transferability** | **Gurugram**: $R^2 = 0.5651$ (336 days)<br>**Lucknow**: $R^2 = 0.5401$ (402 days)<br>**Patna**: $R^2 = 0.7276$ (372 days) | **Bengaluru**: $R^2 = -0.5284$ (Shift)<br>**Chennai**: $R^2 = -0.0784$ (Shift)<br>**Hyderabad**: $R^2 = 0.2013$ | Positive physical boundary finding: Frozen Delhi model transfers across Indo-Gangetic Basin, fails outside it. |

---

## Repository Structure

```text
smognet/
├── data_loader.py                 # CPCB & NASA POWER data pipeline and chronological split generators
├── features.py                    # Tabular (37), FIRMS satellite (40), and 3D sequence feature engineering
├── models.py                      # Tree, neural sequence (BiLSTM-Attn, TabularMLP), and OOF Ridge meta-learner
├── evaluate.py                    # Statistical metrics, split conformal prediction, and TreeSHAP explainability
├── phase2_pipeline.py             # Optuna hyperparameter optimization routines across base learners
├── phase_b_and_baselines.py       # Baselines (SARIMA, Prophet), multi-horizon (t+1..t+7), and spatial transfer
├── run_enhanced_dropout_experiments.py # 50-mask Monte Carlo sensor degradation suite
├── firms_evaluation.py            # NASA FIRMS MODIS active fire evaluation and counterfactual attribution
├── generate_all_publication_results.py # Generates 300 DPI publication figures and structured CSV tables
├── generate_evaluation_ledger.py  # Builds the master evaluation_ledger.csv single source of truth
├── run_all.py                     # Master execution runner reproducing full experimental suite
├── diagnose_4011_vs_4043.py       # Diagnostic script reconciling standalone CPU vs GPU XGBoost baseline
├── diagnose_3904_vs_3916.py       # Diagnostic script reconciling 20/25 vs 25/30 neural training epochs
├── scripts/
│   └── verify_metric_consistency.py # Automated pre-submission audit asserting cross-file concordance
├── results/                       # Publication-ready figures (300 DPI) and structured CSV/JSON data
│   ├── 01_model_comparison/       # Walk-forward charts, radar ranking, comparison tables
│   ├── 02_feature_analysis/       # Feature group ablations, TreeSHAP beeswarm, joint interactions
│   ├── 03_forecast_visualization/ # Time series forecasts, multi-horizon curves, conformal intervals
│   ├── 04_spatial_generalization/ # Zero-shot cross-city transfer bars, Indo-Gangetic coordinate map
│   ├── 05_robustness/             # 50-mask Monte Carlo degradation curves, realistic outage scenarios
│   ├── 06_literature_context/     # Audited literature benchmark tables and verified DOI citations
│   ├── 07_computational/          # Training times, per-sample inference latencies, hardware specs
│   └── primary_pipeline.pkl       # Saved trained pipeline checkpoint (31.5 MB) for instant evaluation
├── evaluation_ledger.csv          # Comprehensive ledger of all 21 manuscript metrics (M01-M18, V01-V03)
├── table1_model_comparison.md     # Manuscript Table 1 and full multi-table breakdown
├── manuscript_summary.md          # Complete research paper summary prepared for Scientific Reports
├── summary.md                     # Executive orientation summary for the project
└── requirements.txt               # Pinned Python package dependencies
```

---

## Quickstart & Reproducibility

### 1. Environment Setup

Clone this repository and install the dependencies into a clean Python 3.10 virtual environment:

```bash
git clone https://github.com/hx010207/Smognet.git
cd Smognet
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. NASA FIRMS API Key Configuration (Optional for Satellite Refetching)

Precomputed NASA FIRMS daily active fire aggregations (113,557 fire points) are pre-packaged. If you wish to refetch raw MODIS chunks from the NASA FIRMS Area API:

```bash
cp .env.example .env
# Edit .env and set your free key from https://firms.modaps.eosdis.nasa.gov/api/map_key/
export FIRMS_MAP_KEY="your_key_here"
```

### 3. Data Sources

Raw historical air quality records are provided by the [Central Pollution Control Board (CPCB)](https://cpcb.nic.in/) and surface meteorology by the [NASA POWER API](https://power.larc.nasa.gov/). Raw data files are placed in the `data/` directory:
* `data/city_day.csv`: Multi-city daily air quality records across India.
* `data/nasa_power_delhi.csv`: MERRA-2 surface weather assimilation grid for Delhi (28.61°N, 77.21°E).
* `data/firms_punjab_haryana_daily_2015_2020.csv`: Daily satellite active fire counts and Fire Radiative Power (FRP).

### 4. Running the Full Pipeline

To reproduce all models, degradation experiments, satellite counterfactuals, and generate publication-ready visual assets:

```bash
python run_all.py
```

* **Expected Runtime**: ~3–5 minutes with NVIDIA CUDA GPU acceleration (`torch==2.7.1+cu118`, `xgboost==3.2.0`), or ~15–20 minutes on multi-core CPU.
* **Instant Evaluation**: Because the trained model checkpoint is committed as [`results/primary_pipeline.pkl`](results/primary_pipeline.pkl), you can run downstream evaluations, diagnostic checks, and figure generations in seconds without retraining from scratch:
  ```bash
  python run_enhanced_dropout_experiments.py
  python firms_evaluation.py
  python generate_all_publication_results.py
  ```

### 5. Automated Pre-Submission Concordance Verification

Run the automated consistency verification script to verify 100% numerical and methodological concordance across all repository files:

```bash
python scripts/verify_metric_consistency.py
```

Expected output:
```text
==============================================================================
 SMOG NET PRE-SUBMISSION METRIC CONCORDANCE AUDIT
==============================================================================
[OK] Loaded evaluation_ledger.csv (21 records)
[OK] M01 Walk-Forward cross-validation matches model_comparison_table.csv exactly
[OK] M02 (39.16) and M06 (39.04) fixed benchmark values and notes verified
[OK] M03 (40.43 GPU) and M04 (40.11 CPU) baseline values and notes verified
[OK] Audit verification record V01 present in ledger
[OK] Audit verification record V02 present in ledger
[OK] Audit verification record V03 present in ledger
[OK] M07-M09 Monte Carlo degradation metrics verified against enhanced_sensor_dropout_metrics.csv
[OK] M13-M17 Spatial transfer metrics verified against spatial_transfer_table.csv
[OK] M10-M12 Multi-horizon forecasting metrics verified across files
[OK] M05 NASA FIRMS satellite evaluation metrics verified across files
[OK] M18 Conformal prediction interval metrics verified across files
[OK] Markdown documentation files cross-checked and validated

==============================================================================
 AUDIT RESULT: 0 ERRORS, 0 WARNINGS
==============================================================================

>>> ALL AUDITED METRICS ARE 100% CONCORDANT ACROSS EVERY REPOSITORY FILE! <<<
```

---

## Manuscript & Documentation Links

* **Manuscript Summary**: [`manuscript_summary.md`](manuscript_summary.md) provides the full academic research paper prepared for submission to *Scientific Reports*.
* **Table 1 Model Comparisons**: [`table1_model_comparison.md`](table1_model_comparison.md) details the primary walk-forward benchmark table, multi-horizon dynamics, spatial transfer table, 50-mask degradation metrics, and audited literature comparison table.
* **Evaluation Ledger**: [`evaluation_ledger.csv`](evaluation_ledger.csv) is the single source of truth mapping all 21 reported metrics to their exact train/val/test date ranges, feature counts, seeds, and generating scripts.
* **Orientation Guide**: [`summary.md`](summary.md) provides a high-level orientation summary for researchers navigating the repository.
* **Data & Artifact Archive**: Permanent Zenodo archive repository `10.5281/zenodo.XXXXXXX` [TODO before submission: mint and replace with minted DOI].

---

## License & Attribution

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. Ground observation data is distributed under open access via the Central Pollution Control Board (CPCB) of India. Surface meteorology data is provided courtesy of the NASA Langley Research Center POWER Project. Active fire data is provided courtesy of NASA FIRMS / LANCE ESDIS.
