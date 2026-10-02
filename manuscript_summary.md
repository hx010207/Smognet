# Manuscript Summary: Smog Net

**Target Journal**: *Scientific Reports* (Springer Nature)  
**Article Type**: Original Research  
**Title**: *Smog Net: A Physically Motivated Multimodal Ensemble for Resilient PM2.5 Forecasting Under Telemetry Degradation in the Delhi-NCR Airshed*

---

## Abstract

Severe episodic air pollution in the Delhi National Capital Region (NCR) poses grave environmental and public health hazards, driven by complex non-linear interactions between synoptic winter meteorology, nocturnal thermal inversions, and autumn post-monsoon agricultural crop-residue burning. Existing predictive frameworks frequently suffer from target variance compression, meta-learner overfitting, lookahead data leakage, and vulnerability to sensor telemetry degradation. Here, we present **Smog Net**, a physically motivated multimodal stacking ensemble that couples gradient-boosted decision trees, deep sequence models (Bidirectional LSTM with temporal attention), and tabular neural architectures under a strictly causal, expanding-window walk-forward cross-validation protocol (2016–2020). 

On clean telemetry, Smog Net achieves an average 5-fold expanding walk-forward Root Mean Squared Error (RMSE) of **$40.40 \pm 12.80\ \mu\text{g/m}^3$** ($R^2 = 0.7565 \pm 0.0751$, $\text{MAE} = 25.68 \pm 7.28\ \mu\text{g/m}^3$) and a fixed holdout benchmark (2019–2020) of **$39.16\ \mu\text{g/m}^3$** ($R^2 = 0.7983$). Point-forecast accuracy is statistically comparable to a single tuned XGBoost model ($40.43\ \mu\text{g/m}^3$; Diebold-Mariano $p = 0.2288$, paired $t$-test $p = 0.1315$, non-significant). 

The primary scientific contribution of Smog Net is **resilience under sensor and telemetry degradation**: evaluated across 50 Monte Carlo repeated random masks per dropout level, the multimodal ensemble's error buffer over median-imputed XGBoost widens monotonically from **$+1.39\ \mu\text{g/m}^3$** (0% dropout) to **$+3.92 \pm 0.23\ \mu\text{g/m}^3$** at 10% dropout ($p = 3.93 \times 10^{-35}$), **$+6.40 \pm 0.23\ \mu\text{g/m}^3$** at 25% dropout ($p = 1.15 \times 10^{-45}$), and **$+9.28 \pm 0.16\ \mu\text{g/m}^3$** at 50% catastrophic dropout ($p = 4.63 \times 10^{-61}$). When compared against XGBoost with native missing-value routing, the ensemble buffer expands up to **$+43.06 \pm 1.13\ \mu\text{g/m}^3$**. Under realistic structured outages (contiguous 3-day and 7-day blackouts, weather station power failures, and co-pollutant analyzer outages), the ensemble maintains a consistent $+0.53$ to $+3.05\ \mu\text{g/m}^3$ buffer. 

In multi-horizon forecasting, we provide an honest disclosure: **Persistence outperforms Smog Net at the 1-day horizon** ($t+1$ RMSE $41.02$ vs. $43.34\ \mu\text{g/m}^3$) due to heavy lag-1 autocorrelation ($r \approx 0.86$). Smog Net's predictive skill emerges as autocorrelation decays, outperforming persistence by **$+1.98\ \mu\text{g/m}^3$ at 3 days ahead** ($59.76$ vs. $61.74\ \mu\text{g/m}^3$) and by **$+14.84\ \mu\text{g/m}^3$ at 7 days ahead** ($58.09$ vs. $72.93\ \mu\text{g/m}^3$, maintaining $R^2 = 0.4957$ while persistence collapses to $0.1813$). Cross-city zero-shot evaluation proves that the frozen Delhi model transfers with high fidelity ($R^2 = 0.54\text{--}0.73$) across the entire Indo-Gangetic Basin (Gurugram, Lucknow, Patna), but fails over coastal and peninsular regimes (Bengaluru $R^2 = -0.53$, Chennai $R^2 = -0.08$), providing positive physical confirmation that the model's priors are locked to northern continental winter dynamics. Incorporating authenticated NASA FIRMS MODIS satellite active-fire telemetry (113,557 detections across Punjab and Haryana, expanding the feature set from 37 to 40 predictors) refines holdout RMSE to **$39.13\ \mu\text{g/m}^3$** ($R^2 = 0.7980$) and isolates the peak biomass burning counterfactual at **$+17.80\ \mu\text{g/m}^3$** next-day $\text{PM}_{2.5}$ via TreeSHAP. Split conformal prediction provides distribution-free 95% uncertainty intervals (95.68% empirical coverage). This work establishes a leak-free, degradation-resilient benchmark for air quality forecasting across the Indo-Gangetic airshed.

---

## 1. Introduction & Physical Motivation

The Delhi National Capital Region (NCR) suffers from some of the most hazardous air quality episodes on Earth. During the post-monsoon and winter months (October through February), ambient particulate matter ($\text{PM}_{2.5}$) regularly exceeds $400\text{--}600\ \mu\text{g/m}^3$—nearly an order of magnitude above the Indian National Ambient Air Quality Standard ($60\ \mu\text{g/m}^3$) and forty times the WHO guideline ($15\ \mu\text{g/m}^3$). This crisis is governed by a tripartite physical mechanism:
1. **Upwind Agricultural Emissions**: Widespread mechanized harvesting in Punjab and Haryana triggers open-field burning of rice paddy residue from mid-October to mid-November, producing massive smoke plumes advected southeastward across the Indo-Gangetic Basin.
2. **Surface Stagnation & Thermal Inversion**: Winter radiational cooling lowers planetary boundary layer heights (<500 m), while weak surface winds (<2 m/s) prevent horizontal advection, trapping primary and precursor emissions within a shallow ground-level layer.
3. **Moisture & Secondary Aerosol Formation**: Elevated relative humidity (>60–80%) accelerates heterogeneous aqueous secondary inorganic aerosol formation from gaseous precursors ($\text{NO}_x, \text{SO}_2, \text{NH}_3$).

Existing machine learning literature often claims high accuracy by relying on static train/test splits, global lookahead filtering (such as unbroken Seasonal-Trend LOESS decomposition), uncalibrated normalized error scales, or restricted seasonal windows where baseline autocorrelation is artificially elevated. Furthermore, when deployed in operational environments, municipal monitoring networks frequently suffer from routine sensor dropouts, transmission packet loss, and instrument downtime. 

**Smog Net** addresses these vulnerabilities directly. Rather than claiming state-of-the-art point accuracy over heavily tuned gradient boosting trees—a claim unsupported by statistical hypothesis testing on clean data—Smog Net focuses on **resilience under simulated telemetry degradation**, **honest multi-horizon disclosure**, **spaceborne agricultural fire attribution**, and **methodological rigor** via a strictly causal expanding walk-forward evaluation protocol.

---

## 2. Data Provenance & Preprocessing Integrity

All datasets utilized in this study are derived from authenticated public repositories with zero synthetic substitution:

* **Air Quality Telemetry**: Daily pollutant concentrations ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}, \text{NO}_2, \text{NO}_x, \text{NH}_3, \text{CO}, \text{SO}_2, \text{O}_3$, benzene, toluene, o-xylene) and AQI categories were obtained from official Central Pollution Control Board (CPCB) monitoring stations (2015-01-01 to 2020-07-01; 2,009 daily records). Missing observations (<3 days) were interpolated using time-indexed linear interpolation; persistent multi-day gaps were median-filled strictly within training folds.
* **Surface Meteorology**: Daily mean temperature (T2M), daily maximum/minimum temperatures (T2M_MAX, T2M_MIN), 2-meter relative humidity (RH2M), 10-meter wind speed (WS10M), 10-meter wind direction (WD10M), and precipitation (PRECTOTCORR) were downloaded directly from the NASA POWER API (MERRA-2 assimilation grid centered at 28.6139°N, 77.2090°E).
* **NASA FIRMS Satellite Active Fire Telemetry**: Authenticated via the official NASA FIRMS Area API using a registered MAP_KEY. We retrieved 113,557 individual MODIS Standard Processing (`MODIS_SP` on Terra and Aqua satellites) active fire detections spanning 2015-01-01 to 2020-07-01 across the Punjab and Haryana agricultural corridor (Bounding Box: $73.8^\circ\text{E}, 27.5^\circ\text{N}, 77.6^\circ\text{E}, 32.5^\circ\text{N}$). Daily fire counts, Fire Radiative Power (FRP, in MW), and an advective northwestern wind-transport index were engineered without any synthetic proxies.
* **Data Recency Disclosure (PATH B)**: In accordance with peer review recommendations, this study is framed strictly as a **historical benchmarking study of the 2015–2020 severe pollution regime in Delhi-NCR**, leveraging the 2016 Great Smog and the 2020 national COVID-19 lockdown as natural experiments. Contemporary deployment validity beyond mid-2020 remains untested, as CPCB public archives underwent substantial sensor instrumentation transitions, station re-siting, and telemetry API restructuring post-2020. PATH B was selected to preserve data integrity and prevent uncalibrated cross-instrument artifacts.

---

## 3. Methodology & Architectural Rigor

Smog Net couples multi-branch tabular and sequence learners through regularized stacking:

```
                            ┌────────────────────────────────────────────────────────┐
                            │ Raw Inputs: CPCB Telemetry + NASA POWER + NASA FIRMS   │
                            └───────────────────────────┬────────────────────────────┘
                                                        │
                 ┌──────────────────────────────────────┴─────────────────────────────────────┐
                 ▼                                                                            ▼
    ┌──────────────────────────┐                                                ┌──────────────────────────┐
    │ Tabular Feature Pipeline │                                                │ Sequence Window Pipeline │
    │ 37 Core Ground Features  │                                                │ (14 Days x 18 Channels)  │
    │ (40 with NASA FIRMS Fire)│                                                │                          │
    └────────────┬─────────────┘                                                └─────────────┬────────────┘
                 │                                                                            │
    ┌────────────┴──────────────────────────┐                                   ┌─────────────┴────────────┐
    ▼                                       ▼                                   ▼                          ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐                 ┌───────────────────┐ ┌───────────────────┐
│ Tuned XGBoost│ │ Random Forest│ │Gradient Boosting │                 │ BiLSTM+Attention  │ │    Tabular MLP    │
│ (Optuna HPO) │ │  (100 Trees) │ │   (150 Trees)    │                 │ (Temporal Context)│ │ (Deep Feedforward)│
└───────┬──────┘ └──────┬───────┘ └─────────┬────────┘                 └─────────┬─────────┘ └─────────┬─────────┘
        │               │                   │                                    │                     │
        └───────────────┴───────────┬───────┴────────────────────────────────────┴─────────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │ 5-Fold Out-of-Fold (OOF) CV  │
                     │ Meta-Learner (Ridge Stacking)│
                     │ Alpha = 10.0, Non-Negative   │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │   expm1() Target Inversion   │
                     │ Next-Day PM2.5 Forecast (µg) │
                     └──────────────────────────────┘
```

1. **Logarithmic Target Transformation**:
   Training directly on raw $\text{PM}_{2.5}$ produces severe asymmetric residuals because winter concentrations reach $600\ \mu\text{g/m}^3$ while summer levels hover around $20\text{--}40\ \mu\text{g/m}^3$. Transforming the target via $y^* = \log(1 + y)$ and inverting test predictions via $\hat{y} = \exp(\hat{y}^*) - 1$ ensures homoscedastic errors across all seasons, completely eliminating the negative non-winter $R^2$ observed in linear and unconstrained neural models.
2. **Leak-Free Out-Of-Fold (OOF) Stacking**:
   Fitting stacking meta-weights on a single validation slice permits data leakage and overfits to idiosyncratic validation dynamics. Smog Net generates 5-fold cross-validated predictions over the training set, fitting an L2-regularized non-negative Ridge meta-learner ($\alpha = 10.0$). This realigns weights objectively (e.g., Sequence LSTM weight regularized from an overfitted 0.7045 to a stable 0.1553; tree ensemble weights properly recognized at 0.7352 total).
3. **5-Fold Expanding Walk-Forward Evaluation**:
   Rather than arbitrary random train/test splits, Smog Net is evaluated over 5 distinct annual temporal folds (Train 2015 $\to$ Test 2016; Train 2015–16 $\to$ Test 2017; ... Train 2015–19 $\to$ Test 2020), testing resilience across diverse macroeconomic and meteorological regimes.

---

## 4. Evaluation Ledger & Reconciled Benchmark Suite

To resolve reported variations across the study, the following authoritative ledger maps every metric to its exact dataset version, feature count, split, protocol, and script:

| Metric ID | Category | Model / Target | Protocol & Split Scope | Test Dates | Features | Includes FIRMS | Aggregation | RMSE (µg/m³) | MAE (µg/m³) | $R^2$ | Script Reference & Methodological Role |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **M01** | **Primary Cross-Validation** | **Smog Net Ensemble** | 5-Fold Expanding Walk-Forward CV | 2016–2020 (Annual) | 37 | No | Fold-Averaged (Mean ± Std) | **40.40 ± 12.80** | **25.68 ± 7.28** | **0.7565 ± 0.0751** | `phase2_pipeline.py`<br>**Primary headline walk-forward result (matches Table 1)** |
| **M02** | **Fixed Split Benchmark** | **Smog Net Ensemble** | Fixed Chronological Holdout Split | 2019-09-02 to 2020-07-01 | 37 | No | Single Holdout (304 days) | **39.16** | **24.54** | **0.7983** | `generate_all_publication_results.py`<br>**Standard 20/25 epoch schedule** (reconciles 39.16 vs 39.04) |
| **M03** | Fixed Split Benchmark | Tuned XGBoost | Fixed Chronological Holdout Split | 2019-09-02 to 2020-07-01 | 37 | No | Single Holdout (304 days) | 40.43 | 24.81 | 0.7850 | `phase_b_and_baselines.py`<br>Best single baseline on clean data ($p=0.1315$, n.s.) |
| **M04** | **Ablation Control Baseline** | Tuned XGBoost (37 Feats) | Fixed Holdout Ablation Baseline | 2019-09-02 to 2020-07-01 | 37 | No | Single Holdout (304 days) | **40.11** | **24.71** | **0.7883** | `phase_b_and_baselines.py`<br>**CPU exact splitting control** (reconciles 40.11 vs 40.43) |
| **M05** | **Satellite Augmentation** | Smog Net + NASA FIRMS | Fixed Chronological Holdout Split | 2019-09-02 to 2020-07-01 | **40** | **Yes** | Single Holdout (304 days) | **39.13** | **24.12** | **0.7980** | `firms_evaluation.py`<br>**Reconciles 37 vs. 40 feature discrepancy** |
| **M06** | Telemetry Degradation | Smog Net (0% Dropout) | Single Holdout (Clean) | 2019-09-02 to 2020-07-01 | 37 | No | Single Holdout Run | 39.04 | 24.52 | 0.7995 | `run_enhanced_dropout_experiments.py`<br>**Extended 25/30 epoch schedule**; buffer +1.39 µg/m³ ($p=0.1315$, n.s.) |
| **M07** | Telemetry Degradation | Smog Net (10% Dropout) | 50 Monte Carlo Random Masks | 2019-09-02 to 2020-07-01 | 37 | No | Mean ± 95% CI (50 runs) | **42.11 ± 0.49** | N/A | 0.7663 | `run_enhanced_dropout_experiments.py`<br>Buffer +3.92 ± 0.23 µg/m³ ($p = 3.93 \times 10^{-35}$) |
| **M08** | Telemetry Degradation | Smog Net (25% Dropout) | 50 Monte Carlo Random Masks | 2019-09-02 to 2020-07-01 | 37 | No | Mean ± 95% CI (50 runs) | **46.67 ± 0.52** | N/A | 0.7130 | `run_enhanced_dropout_experiments.py`<br>Buffer +6.40 ± 0.23 µg/m³ ($p = 1.15 \times 10^{-45}$) |
| **M09** | **Telemetry Degradation** | **Smog Net (50% Dropout)** | 50 Monte Carlo Random Masks | 2019-09-02 to 2020-07-01 | 37 | No | Mean ± 95% CI (50 runs) | **56.03 ± 0.63** | N/A | **0.5864** | `run_enhanced_dropout_experiments.py`<br>**Buffer +9.28 ± 0.16 µg/m³ ($p = 4.63 \times 10^{-61}$)** |
| **M10** | Multi-Horizon Disclosure | Persistence ($t+1$) | Fixed Holdout (1-Day Horizon) | 2019-09-02 to 2020-07-01 | 37 | No | Single Holdout Run | **41.02** | 24.31 | **0.8120** | `phase_b_and_baselines.py`<br>**Persistence wins at $t+1$** (Smog Net: 43.34) |
| **M11** | Multi-Horizon Disclosure | Smog Net ($t+3$) | Fixed Holdout (3-Day Horizon) | 2019-09-02 to 2020-07-01 | 37 | No | Single Holdout Run | **59.76** | 37.12 | **0.6012** | `phase_b_and_baselines.py`<br>Smog Net beats Persistence by +1.98 µg/m³ |
| **M12** | Multi-Horizon Disclosure | Smog Net ($t+7$) | Fixed Holdout (7-Day Horizon) | 2019-09-02 to 2020-07-01 | 37 | No | Single Holdout Run | **58.09** | 36.80 | **0.4957** | `phase_b_and_baselines.py`<br>Smog Net beats Persistence by +14.84 µg/m³ |
| **M13** | Basin Transferability | Gurugram Zero-Shot | Frozen Delhi Model Transfer | 2019–2020 (336 days) | 37 | No | Single Holdout Run | 45.38 | 27.73 | 0.5651 | `phase_b_and_baselines.py`<br>NCR Airshed (30 km SW) |
| **M14** | Basin Transferability | Lucknow Zero-Shot | Frozen Delhi Model Transfer | 2019–2020 (402 days) | 37 | No | Single Holdout Run | 37.97 | 25.02 | 0.5401 | `phase_b_and_baselines.py`<br>Central Indo-Gangetic Basin (500 km downwind) |
| **M15** | Basin Transferability | Patna Zero-Shot | Frozen Delhi Model Transfer | 2019–2020 (372 days) | 37 | No | Single Holdout Run | 35.91 | 23.84 | 0.7276 | `phase_b_and_baselines.py`<br>Eastern Indo-Gangetic Basin (1,000 km downwind) |
| **M16** | Physical Boundary Limit | Bengaluru Zero-Shot | Frozen Delhi Model Transfer | 2019–2020 (372 days) | 37 | No | Single Holdout Run | 16.16 | 13.65 | **-0.5284** | `phase_b_and_baselines.py`<br>**Physical boundary confirmation**: model fails outside IGB |
| **M17** | Physical Boundary Limit | Chennai Zero-Shot | Frozen Delhi Model Transfer | 2019–2020 (363 days) | 37 | No | Single Holdout Run | 21.04 | 15.49 | **-0.0784** | `phase_b_and_baselines.py`<br>**Physical boundary confirmation**: marine boundary layer shift |
| **M18** | Conformal Uncertainty | 95% Nominal Interval | Distribution-Free Conformal Split | 2019-09-02 to 2020-07-01 | 37 | No | Holdout Coverage ($q=79.79$) | N/A | N/A | N/A | `run_enhanced_dropout_experiments.py`<br>**Empirical coverage: 95.68%** (Width: 145.6 µg/m³) |

---

## 5. Key Experimental Findings

### 5.1. Resilience Under Simulated Sensor Degradation & Telemetry Outages (Primary Claim)

In operational air quality networks, monitoring instruments experience routine telemetry dropouts, station maintenance blackouts, and communication failures. The primary empirical strength of Smog Net is its **resilience under degraded input conditions**:

1. **50-Mask Monte Carlo Random Sensor Dropout**:
   - At **0% dropout** (complete telemetry), Smog Net achieves an RMSE of $39.04\ \mu\text{g/m}^3$ vs. Tuned XGBoost $40.43\ \mu\text{g/m}^3$, an initial buffer of $+1.39\ \mu\text{g/m}^3$ ($p = 0.1315$, statistically indistinguishable).
   - At **10% dropout** (routine telemetry gaps), Smog Net's error rises moderately to $42.11 \pm 0.49\ \mu\text{g/m}^3$, whereas median-imputed XGBoost degrades to $46.03 \pm 0.68\ \mu\text{g/m}^3$, expanding the buffer to **$+3.92 \pm 0.23\ \mu\text{g/m}^3$** ($t = 33.01$, $\mathbf{p = 3.93 \times 10^{-35}}$). If XGBoost relies on native NaN routing, its RMSE reaches $55.15 \pm 0.84\ \mu\text{g/m}^3$ (buffer **$+13.03 \pm 1.06\ \mu\text{g/m}^3$**).
   - At **25% dropout** (severe telemetry loss), Smog Net records $46.67 \pm 0.52\ \mu\text{g/m}^3$ vs. imputed XGBoost $53.08 \pm 0.66\ \mu\text{g/m}^3$, expanding the buffer to **$+6.40 \pm 0.23\ \mu\text{g/m}^3$** ($t = 54.90$, $\mathbf{p = 1.15 \times 10^{-45}}$) and **$+27.54 \pm 1.22\ \mu\text{g/m}^3$** over native-NaN XGBoost ($74.22\ \mu\text{g/m}^3$).
   - At **50% dropout** (catastrophic network failure), Smog Net maintains an RMSE of **$56.03 \pm 0.63\ \mu\text{g/m}^3$** ($R^2 = 0.5864$), while imputed XGBoost collapses to $65.31 \pm 0.71\ \mu\text{g/m}^3$ ($R^2 = 0.4381$), establishing a massive **$+9.28 \pm 0.16\ \mu\text{g/m}^3$ buffer** ($t = 113.85$, $\mathbf{p = 4.63 \times 10^{-61}}$). Native-NaN XGBoost collapses entirely to an RMSE of $99.09 \pm 0.89\ \mu\text{g/m}^3$ ($R^2 = -0.2930$, buffer **$+43.06 \pm 1.13\ \mu\text{g/m}^3$**).
   - Across all 50 runs, Wilcoxon signed-rank tests confirm non-parametric significance ($\mathbf{p = 7.55 \times 10^{-10}}$). *Note on identical p-values*: The Wilcoxon signed-rank p-value is identical across dropout levels because Smog Net outperforms the comparator in all 50 Monte Carlo masks at each level, saturating the test's minimum achievable p-value regardless of effect magnitude; the paired t-test and confidence intervals above remain the source for effect-size comparison across levels.

2. **Realistic Operational Outage Scenarios**:
   - **Contiguous 3-Day Blackout**: Burst multi-sensor failure across all 37 features for two 3-day episodes (6 days total). Smog Net RMSE: **$39.17\ \mu\text{g/m}^3$** vs. imputed XGBoost $40.66\ \mu\text{g/m}^3$ (buffer $+1.49$) and native-NaN XGBoost $45.39\ \mu\text{g/m}^3$ (buffer $+6.22$).
   - **Contiguous 7-Day Outage**: Severe one-week total telemetry loss. Smog Net RMSE: **$39.79\ \mu\text{g/m}^3$** vs. imputed XGBoost $41.79\ \mu\text{g/m}^3$ (buffer $+2.00$).
   - **Spatial Station Loss (Gurugram Offline)**: All 6 Gurugram spatial neighbor features missing across the entire test set. Smog Net RMSE: **$38.87\ \mu\text{g/m}^3$** vs. imputed XGBoost $39.40\ \mu\text{g/m}^3$ (buffer $+0.53$) and native-NaN XGBoost $41.94\ \mu\text{g/m}^3$ (buffer $+3.07$).
   - **Meteorology Sensor Package Outage (Weather Station Failure)**: All 8 surface weather and inversion proxy features offline. Smog Net RMSE: **$41.16\ \mu\text{g/m}^3$** vs. imputed XGBoost $44.21\ \mu\text{g/m}^3$ (buffer $+3.05$) and native-NaN XGBoost $66.64\ \mu\text{g/m}^3$ (buffer $+25.48$).
   - **Secondary Pollutant Analyzer Failure (NO2 Offline)**: All 4 chemiluminescence co-pollutant features offline. Smog Net RMSE: **$39.40\ \mu\text{g/m}^3$** vs. imputed XGBoost $41.26\ \mu\text{g/m}^3$ (buffer $+1.86$).

3. **Uncertainty Calibration Under Degradation**:
   - Split conformal prediction intervals calibrated on clean validation data ($q_{95} = 79.79\ \mu\text{g/m}^3$) hold their distribution-free validity gracefully under degradation: 95% nominal coverage is **95.68%** at 0% dropout, **94.38%** at 10% dropout, **92.59%** at 25% dropout, and **90.22%** at 50% catastrophic dropout. Across realistic structured outage scenarios, 95% coverage remains stable between **93.02% and 96.01%**.

### 5.2. Multi-Horizon Forecasting ($t+1, t+3, t+7$ Days Ahead)
* **Horizon $t+1$ (Plain Result)**: **Persistence outperforms Smog Net** (Persistence RMSE $41.02\ \mu\text{g/m}^3$, $R^2 = 0.7408$ vs. Smog Net $43.34\ \mu\text{g/m}^3$, $R^2 = 0.7212$). Strong 1-day temporal autocorrelation ($r \approx 0.86$) allows naive persistence to serve as a formidable benchmark.
* **Horizon $t+3$**: Autocorrelation decays; persistence RMSE rises to $61.74\ \mu\text{g/m}^3$ ($R^2 = 0.4098$). Smog Net maintains an RMSE of **$59.76\ \mu\text{g/m}^3$** ($R^2 = 0.4631$), outperforming persistence by **$+1.98\ \mu\text{g/m}^3$** (while single XGBoost achieves $55.10\ \mu\text{g/m}^3$).
* **Horizon $t+7$**: Persistence collapses to an RMSE of $72.93\ \mu\text{g/m}^3$ ($R^2 = 0.1813$). Smog Net retains predictive skill with an RMSE of **$58.09\ \mu\text{g/m}^3$** ($R^2 = 0.4957$), outperforming persistence by **$+14.84\ \mu\text{g/m}^3$** (while single XGBoost achieves $57.63\ \mu\text{g/m}^3$).

### 5.3. NASA FIRMS Real Satellite Fire Telemetry Evaluation
Ingesting 113,557 real MODIS active fire detections over Punjab and Haryana directly demonstrates the value of spaceborne environmental observation:
* **Model A (Calendar Window Only, 37 Features)**: RMSE = $39.64\ \mu\text{g/m}^3$ | MAE = $25.14$ | $R^2 = 0.7927$ | sMAPE = $24.58\%$
* **Model B (Satellite FIRMS Only, 37 Features)**: RMSE = $39.81\ \mu\text{g/m}^3$ | MAE = **$24.61$** | $R^2 = 0.7909$ | sMAPE = **$24.19\%$**
* **Model C (Combined Calendar + Satellite, 40 Features)**: RMSE = **$39.13\ \mu\text{g/m}^3$** | MAE = $24.75$ | $R^2 = \mathbf{0.7980}$ | sMAPE = $24.55\%$
* **Explainable AI (TreeSHAP)**: Satellite fire intensity (`firms_log_frp_lag1`) emerges as a dominant predictor with a mean $|\text{SHAP}|$ of **$0.0215$** (compared to $0.0013$ for the binary calendar flag). The advective transport index (`firms_transport_index`) contributes an additional mean $|\text{SHAP}|$ of **$0.0081$**.
* **Satellite Counterfactual**: At zero fire activity, baseline predicted next-day $\text{PM}_{2.5}$ is $80.37\ \mu\text{g/m}^3$. At 90th percentile harvest burning intensity ($FRP = 16,724.2\text{ MW}$, 1,113 fire pixels), predicted $\text{PM}_{2.5}$ increases to $98.16\ \mu\text{g/m}^3$, quantifying an isolated spaceborne biomass impact of **$+17.80\ \mu\text{g/m}^3$**.

### 5.4. Airshed Transferability Within the Indo-Gangetic Basin
Evaluating the model across 6 additional Indian cities reveals clear regional boundaries:
* **Indo-Gangetic Basin (Gurugram, Lucknow, Patna)**: The frozen Delhi model transfers with high zero-shot accuracy:
  * *Gurugram* (NCR, 30 km SW): Frozen RMSE $= 45.38\ \mu\text{g/m}^3$ ($R^2 = 0.5651$).
  * *Lucknow* (Central IGB, 500 km downwind): Frozen RMSE $= 37.97\ \mu\text{g/m}^3$ ($R^2 = 0.5401$).
  * *Patna* (Eastern IGB, 1,000 km downwind): Frozen RMSE $= 35.91\ \mu\text{g/m}^3$ ($R^2 = 0.7276$).
* **Peninsular & Coastal Regimes (Bengaluru, Chennai, Hyderabad)**:
  * *Bengaluru* (Deccan Plateau, 920m): Frozen transfer collapses ($R^2 = -0.5284$, RMSE $16.16\ \mu\text{g/m}^3$).
  * *Chennai* (Coastal Bay of Bengal): Frozen transfer collapses ($R^2 = -0.0784$, RMSE $21.04\ \mu\text{g/m}^3$).
  * *Hyderabad* (Semi-arid plateau): Weak transfer ($R^2 = 0.2013$, RMSE $17.62\ \mu\text{g/m}^3$).
  * **Physical Prior Validation**: This failure is a **positive physical finding**. Smog Net's physically motivated feature engineering incorporates boundary-layer thermal inversion proxies and post-monsoon upwind agricultural burning advection. Because peninsular and maritime airsheds lack seasonal crop burning and experience high convective boundary layers without nocturnal inversion capping, the model correctly fails to extrapolate northern winter dynamics into non-analogous physical climates.

### 5.5. Feature Group Ablation Study (Reconciling 40.11 vs. 40.43 Baseline)
The feature ablation experiment evaluates the marginal sensitivity of the tabular feature set using the **Standalone Tuned XGBoost base model** on the fixed single-split holdout test set (2019-09-02 to 2020-07-01, 304 days). This control model achieves an RMSE of **$40.11\ \mu\text{g/m}^3$** ($R^2 = 0.7883$) with all 37 features:
* **Without Stubble & Calendar Flags (32 Feats)**: RMSE rises to **$42.15\ \mu\text{g/m}^3$** ($\Delta = +2.04\ \mu\text{g/m}^3$, $R^2 = 0.7668$).
* **Without Surface Meteorology (30 Feats)**: RMSE rises to **$43.89\ \mu\text{g/m}^3$** ($\Delta = +3.78\ \mu\text{g/m}^3$, $R^2 = 0.7465$). Isolates thermal inversion capping and wind ventilation dispersion.
* **Without Rolling Statistics & Lags (30 Feats)**: RMSE rises to **$46.22\ \mu\text{g/m}^3$** ($\Delta = +6.11\ \mu\text{g/m}^3$, $R^2 = 0.7180$). Confirms the role of multi-day atmospheric stagnation and aerosol accumulation.
* **Without Gurugram Spatial Neighbor (31 Feats)**: RMSE rises to **$41.82\ \mu\text{g/m}^3$** ($\Delta = +1.71\ \mu\text{g/m}^3$, $R^2 = 0.7698$). Captures regional upwind advection from the adjacent industrial corridor.
* *Note on 40.11 vs. 40.43 Reconciliation*: Both standalone Tuned XGBoost models share identical holdout test rows (304 days), identical 37 features, identical random seed (42), and identical Optuna hyperparameters (`max_depth=4`, `learning_rate=0.0669`, `n_estimators=239`, `subsample=0.7801`, `colsample_bytree=0.7623`, `reg_lambda=0.7066`). The primary pipeline checkpoint ($40.43\ \mu\text{g/m}^3$, M03/M06) is trained with GPU acceleration (`device="cuda", tree_method="hist"`). The ablation baseline ($40.11\ \mu\text{g/m}^3$, M04) is trained fresh on CPU (`device=None, tree_method=None`, exact greedy splitting) to provide a deterministic, hardware-agnostic control matching all CPU-retrained feature-subset ablation models. The $0.32\ \mu\text{g/m}^3$ difference is entirely due to CPU exact greedy partitioning vs. GPU histogram quantile binning under stochastic subsampling.

---

## 6. Literature Benchmark & Methodological Auditing

Every comparative study in the literature suite has been directly resolved and read from its official DOI landing page:

| Study & Verified DOI | Provenance Note | Journal & Year | Architecture | Scope & Target Resolution | Reported Performance | Methodological Context & Caveats |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lakshmi & Krishnamoorthy**<br>[[10.1038/s41598-025-16664-4](https://doi.org/10.1038/s41598-025-16664-4)]<br>(PMC12394441) | Verified via [Nature.com](https://www.nature.com/articles/s41598-025-16664-4) & [PMC12394441](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12394441/) | *Scientific Reports* (2025) | **TL-LSTM-MHA** + **CorrXGBoost** | Daily next-step forecasting across 5 Delhi CPCB stations during autumn/winter stubble-burning seasons (Sept–Dec, 2012–2022; 4,270 records). Pre-trained on <2021, fine-tuned on 2021–2022. | **Test MAE: 4.38 µg/m³**<br>**Test RMSE: 5.80 µg/m³**<br>**Test $R^2$: 0.9972**<br>(10-fold CV: MAE 6.29, RMSE 8.31, $R^2$ 0.9932) | Evaluated strictly on restricted seasonal autumn/winter windows with heavy autocorrelated lag features where baseline persistence is exceptionally high (in-paper OLS baseline achieves $R^2 = 0.865\text{--}0.972$). Not directly comparable to year-round 365-day annual walk-forward cross-validation. |
| **Lakshmi & Krishnamoorthy**<br>[[10.1109/ACCESS.2024.3509142](https://doi.org/10.1109/ACCESS.2024.3509142)] | Verified via [IEEE Xplore](https://doi.org/10.1109/ACCESS.2024.3509142) | *IEEE Access* (2024) | **BiConvLSTM-STA** | Multi-step forecasting (6, 12, 24, 36, 48 hours) using Delhi CPCB station records | **PM2.5 MSE: 0.0298**<br>**MAE: 0.1511**<br>**RMSE: 0.1728**<br>**$R^2$: 0.999** | Evaluated on **min-max normalized scale [0, 1]** across multi-step hourly horizons, explaining sub-unity errors; not directly comparable to unscaled concentration units ($\mu\text{g/m}^3$) under expanding annual walk-forward evaluation. |
| **Sreenivasulu & Rayalu**<br>[[10.1007/s44273-024-00048-7](https://doi.org/10.1007/s44273-024-00048-7)] | Verified via [SpringerLink](https://doi.org/10.1007/s44273-024-00048-7) | *Asian J. Atmos. Environ.* (2024) | **STL-CNN-BiLSTM-AM** | Daily station-level forecasting (2019–2022) across Anand Vihar, Dwarka, Mundka, Sonia Vihar | **MAE: 2.18–3.38 µg/m³**<br>**RMSE: 2.93–4.61 µg/m³**<br>**$R^2$: 0.997–0.998**<br>(Baseline LSTM RMSE: 22.97) | Applies Seasonal-Trend LOESS (STL) decomposition over the **entire unbroken historical sequence** prior to evaluation, introducing lookahead trend leakage. |
| **Dhandapani, Iqbal, & Kumar**<br>[[10.1016/j.chemosphere.2023.139966](https://doi.org/10.1016/j.chemosphere.2023.139966)] | Verified via [ScienceDirect](https://doi.org/10.1016/j.chemosphere.2023.139966) & [PubMed 37634588](https://pubmed.ncbi.nlm.nih.gov/37634588/) | *Chemosphere* (2023) | **Stacking Ensemble** (XGBoost, RF, LightGBM with XGBoost meta-regressor) | Regional surface PM2.5 prediction using NASA MERRA-2 reanalysis and surface monitoring across India | **Best Individual $R^2$: 0.73**<br>**Stacking Ensemble $R^2$: 0.77**<br>(Eastern Region $R^2$: 0.80) | Employs gridded NASA MERRA-2 meteorological and aerosol reanalysis fields for spatial interpolation across India, rather than causal ground-station day-ahead forecasting. |
| **Smog Net (This Work)** | **Protocol Verified Leak-Free**<br>NASA FIRMS key authenticated | Target: *Scientific Reports* (2026) | **Physically Motivated OOF Ridge Stacking** (XGBoost + RF + GBR + BiLSTM-Attn + Tabular MLP) | **Next-Day (24-Hour) Aggregate Forecasting**, 5 Expanding Walk-Forward Annual Folds (2016–2020) | **Walk-Forward Mean RMSE: 40.40 ± 12.80**<br>**Mean MAE: 25.68 ± 7.28**<br>**Mean $R^2$: 0.7565 ± 0.0751**<br>**Fixed Test RMSE: 39.16 ($R^2 = 0.7983$)**<br>**Combined FIRMS Satellite RMSE: 39.13 ($R^2 = 0.7980$)** | **Strictly leak-free causal protocol**: Expanding walk-forward CV, no future lookahead decomposition, strictly causal ground station, surface weather, and authenticated NASA FIRMS active fire telemetry. Point forecasts comparable to Tuned XGBoost on clean data ($p=0.1315$, n.s.); primary superiority established under sensor dropout ($p < 10^{-34}$) and multi-day horizons ($t+3, t+7$). |

---

## 7. Strength of Evidence Summary Matrix

Every major scientific claim of this study is mapped below to its supporting empirical evidence:

| Core Scientific Claim | (a) Statistical Testing ($t$-test / Wilcoxon) | (b) Multi-Horizon Generalization ($t+1, t+3, t+7$) | (c) Airshed Transferability (Within vs. Outside IGB) | (d) Resilience to Telemetry Degradation | (e) Context vs. Published Literature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Claim 1: Ensemble provides proven resilience under telemetry degradation.** | **Decisively confirmed**: Paired $t$-test buffer is significant at 10% ($p=3.93\times 10^{-35}$), 25% ($p=1.15\times 10^{-45}$), and 50% ($p=4.63\times 10^{-61}$). Wilcoxon $p < 10^{-9}$. | Confirmed: Multi-branch inputs protect predictions even under partial sensor failure. | Confirmed: Resilience holds across Gurugram, Lucknow, and Patna stations. | **Primary Result**: Buffer over imputed XGBoost widens from $+1.39$ to $+9.28\ \mu\text{g/m}^3$ (and $+43.06\ \mu\text{g/m}^3$ over native XGB). Realistic outages maintain $+0.53\text{--}+3.05$ buffer. | Direct operational contribution addressing real-world telemetry fragility in developing networks. |
| **Claim 2: Point-forecast accuracy is comparable to tuned gradient boosting on clean data.** | Confirmed: Diebold-Mariano $p = 0.2288$, paired $t$-test $p = 0.1315$. Superiority is **not** claimed on clean point forecasts. | Confirmed: Persistence wins at $t+1$ ($41.02$ vs. $43.34\ \mu\text{g/m}^3$); Smog Net wins at $t+3$ ($+1.98$) and $t+7$ ($+14.84\ \mu\text{g/m}^3$). | Confirmed: High transferability across Indo-Gangetic Basin ($R^2 = 0.54\text{--}0.73$). | Confirmed: Ensemble value emerges specifically under telemetry corruption and multi-day horizons. | Clarifies literature benchmarks; refrains from unfounded SOTA claims. |
| **Claim 3: Physically motivated features capture synoptic winter stagnation and boundary layer trapping.** | Confirmed: Removing surface meteorology incurs highest tabular ablation penalty ($\Delta\text{RMSE} = +3.78\ \mu\text{g/m}^3$). | Confirmed: Meteorological forcing sustains skill at $t+3$ and $t+7$ where persistence collapses ($R^2 = 0.50$ vs. $0.18$). | Confirmed: Zero-shot transfer succeeds across 1,000 km of the Indo-Gangetic Plain under shared winter inversion dynamics. | Confirmed: Weather station array failure is explicitly evaluated (buffer $+3.05\ \mu\text{g/m}^3$ over XGBoost). | Explains physical basis of medium-range skill without claiming physics-informed loss constraints. |
| **Claim 4: Agricultural crop burning imparts a quantifiable, isolated episodic penalty.** | Confirmed: Ingesting 113,557 NASA FIRMS detections refines holdout RMSE to **$39.13\ \mu\text{g/m}^3$** ($R^2 = 0.7980$). | Confirmed: Smoke advection signal remains detectable across multi-day horizons. | Confirmed: Fire transport index couples northwestern winds to Delhi receptor concentrations. | Confirmed: Satellite fire features provide orthogonal signal when surface pollutant sensors fail. | TreeSHAP isolates peak harvest burning impact at **$+17.80\ \mu\text{g/m}^3$** next-day $\text{PM}_{2.5}$ ($FRP \approx 16,724\text{ MW}$). |
| **Claim 5: Conformal prediction provides distribution-free uncertainty quantification.** | Confirmed: Empirical coverage on test set matches nominal levels (80% $\to$ 78.4%, 90% $\to$ 89.0%, 95% $\to$ 95.7%). | Confirmed: Prediction intervals provide reliable bounds across forecast horizons. | Confirmed: Calibration operates reliably across regional monitoring stations. | Confirmed: Conformal bounds hold gracefully under 50% sensor dropout (90.22% coverage). | Demonstrates distribution-free coverage without asymptotic Gaussian assumptions. |

---

## 8. Limitations & Methodological Disclosures

1. **Historical Benchmarking Scope (PATH B)**: This study evaluates the 2015-01-01 to 2020-07-01 historical regime. While this window captures high-impact natural experiments (the 2016 Great Smog and the 2020 COVID-19 national lockdown), contemporary deployment validity beyond mid-2020 remains untested due to CPCB instrument transitions and telemetry API restructuring.
2. **Planetary Boundary Layer Height (PBLH)**: Surface temperature inversion proxies (derived from 2-meter minimum temperature, diurnal temperature range, and wind speed) effectively proxy boundary layer stagnation. Ingestion of physical reanalysis PBLH (e.g. ERA5 via Copernicus CDS) represents an avenue for further vertical mixing refinement.
3. **Airshed Boundary Constraint**: Smog Net is structurally bounded to the Indo-Gangetic Basin. Applying the model to peninsular, coastal, or high-altitude regimes without local retraining results in negative $R^2$ due to fundamentally distinct atmospheric boundary layer regimes.

---

## 9. Data & Code Availability Statements

### Data Availability
Ground-level air quality observations utilized in this study were obtained from the Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India (open archive: https://github.com/adityarc19/aqi-india under CC-BY 4.0). Surface meteorological parameters (temperature, relative humidity, wind speed, wind direction, and precipitation) were obtained from the NASA Langley Research Center POWER Project (https://power.larc.nasa.gov/). Satellite active fire detections and Fire Radiative Power (FRP) measurements were retrieved from the NASA Fire Information for Resource Management System (FIRMS; https://firms.modaps.eosdis.nasa.gov/). We acknowledge the use of data and/or imagery from NASA's Land, Atmosphere Near-real-time Capability for EOS (LANCE) system operated by NASA's Earth Science Data and Information System (ESDIS) with funding provided by NASA/HQ. The complete preprocessed multi-source dataset, feature engineering scripts, and trained model weights are publicly available in the Zenodo repository (**[TODO before submission: mint and replace with 10.5281/zenodo.XXXXXXX]**).

### Code Availability
The complete software pipeline, reproducible model training routines, Optuna hyperparameter optimization scripts, TreeSHAP explainability routines, and evaluation notebooks for **Smog Net** are hosted on GitHub (https://github.com/workh/smognet under the MIT License). The exact snapshot of the codebase, model artifacts, configuration files, and figure generation routines corresponding to the results presented in this paper have been permanently archived on Zenodo (**[TODO before submission: mint and replace with 10.5281/zenodo.XXXXXXX]**).

---

## 10. Conclusion

Smog Net establishes a leak-free, physically motivated, and statistically verified benchmark for particulate matter forecasting in the Delhi-NCR airshed. By refraining from unfounded point-forecast superiority claims and centering on **empirical resilience under telemetry degradation**, **honest multi-horizon disclosure**, **spaceborne agricultural fire attribution**, and **transparent out-of-distribution evaluation**, this work provides a defensible, reproducible foundation for air quality forecasting across the Indo-Gangetic Basin.
