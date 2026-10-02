# Smog Net: Consolidated Results & Publication-Ready Visuals

This directory contains the complete, publication-ready figures (>= 300 DPI) and structured source datasets generated for the Smog Net manuscript. All figures use a unified scientific color palette and typography. Every plot has its corresponding numeric CSV/JSON alongside it so all claims are verifiable and regeneratable.

---

## Directory Catalog & File Index

### `01_model_comparison/`
* `walk_forward_rmse_by_year.png`: High-resolution line chart displaying test RMSE across the 5 expanding-window test years (2016–2020) for all evaluated models, illustrating comparative error trajectories across years.
* `model_comparison_bars.png`: Dual-panel horizontal bar chart illustrating Mean ± Std RMSE and Mean ± Std R² across 5-fold cross-validation with error bars.
* `model_ranking_radar.png`: Five-dimensional polar spider chart evaluating Smog Net against key baselines across RMSE, MAE, R², sMAPE, and Inference Latency (normalized 0.0 to 1.0; outer perimeter is optimal).
* `model_comparison_table.csv`: Complete benchmark table with per-fold and Mean ± Std metrics (RMSE, MAE, R², sMAPE) for all 12 model configurations.

### `02_feature_analysis/`
* `feature_group_ablation_pie.png`: Donut chart showing the share of explainable error reduction contributed by each feature group (Stubble/Calendar: 53.7%, Autoregressive Lags: 24.9%, Surface Meteorology: 21.4%), strictly summing to 100%.
* `feature_importance_bar.png`: Horizontal bar chart ranking the top 15 most influential features in gradient boosting, color-coded by feature domain (chemical, stubble, meteorology, spatial).
* `shap_summary_beeswarm.png`: Official SHAP beeswarm plot displaying feature impact distributions on next-day log-PM2.5.
* `shap_interactions_stubble_wind_humidity.png`: Joint SHAP dependence plots proving the non-linear interaction escalation between agricultural burning, calm winds (<2 m/s), and high humidity (>65%).
* `shap_firms_summary.png`: Spaceborne feature importance beeswarm incorporating real NASA FIRMS satellite active fire detections and Fire Radiative Power (FRP).
* `feature_group_ablations.csv`: Quantitative ablation metrics and ΔRMSE impact by feature group.
* `top15_feature_importance.csv`: Ranked gain importance values and domain categories for top features.

### `03_forecast_visualization/`
* `test_window_forecast_vs_actual.png`: Manuscript Figure 1 comparing Smog Net predictions, persistence baseline, and ground truth across the held-out test window, with the autumn stubble-burning window highlighted.
* `multi_horizon_forecast_curve.png`: Dual-panel line chart showing RMSE and R² progression across horizons ($t+1, t+3, t+7$), demonstrating persistence dominance at $t+1$ and Smog Net's expanding advantage at multi-day horizons.
* `conformal_prediction_intervals.png`: Distribution-free split conformal prediction intervals (80%, 90%, 95%) shaded around Smog Net forecasts over ground truth.
* `residuals_over_time.png`: Test error time series and residual density histogram verifying zero-centered homoscedasticity across seasons.
* `multi_horizon_forecast_metrics.csv`: Data backing the multi-horizon forecasting curves.
* `conformal_prediction_intervals.csv`: Quantiles, empirical coverage rates, and mean interval widths.
* `test_forecast_residuals.csv`: Daily test dates, observations, predictions, and residual values.

### `04_spatial_generalization/`
* `spatial_transfer_bar_by_city.png`: Grouped bar chart comparing Frozen Delhi Transfer vs. Locally Retrained Model accuracy across 6 external Indian cities.
* `spatial_transfer_map.png`: Geographic coordinate map of India plotting cities by latitude/longitude, colored by transfer R² and highlighting the Indo-Gangetic Basin corridor.
* `spatial_transfer_table.csv`: Comprehensive cross-city evaluation metrics, coordinates, and airshed classification.

### `05_robustness/`
* `sensor_dropout_degradation.png`: 50-mask Monte Carlo degradation curves (mean ± 95% CI) and buffer expansion bar chart comparing Smog Net vs. Median-Imputed XGBoost and Native-NaN XGBoost across 0%, 10%, 25%, and 50% dropout.
* `realistic_outage_scenarios.png`: Grouped comparative bar chart evaluating Smog Net vs. XGBoost variants across 5 realistic operational outage patterns (contiguous blackouts, weather power failures, station loss).
* `conformal_coverage_under_degradation.png`: Dual-panel calibration curve tracking split conformal prediction coverage (80%, 90%, 95%) and interval widths across degradation levels and outage scenarios.
* `seasonal_error_breakdown_pie.png`: Donut chart partitioning total absolute error across PM2.5 concentration quintiles, resolving the MAPE vs. MAE paradox.
* `diebold_mariano_significance_heatmap.png`: Two-column significance heatmap displaying Diebold-Mariano and paired t-test p-values comparing Smog Net against competing baselines.
* `enhanced_sensor_dropout_metrics.csv`: 50-run Monte Carlo summary metrics (means, 95% CIs, stds, paired t-tests, Wilcoxon p-values, conformal coverage) across 0%, 10%, 25%, and 50% dropout.
* `enhanced_sensor_dropout_50runs_raw.csv`: Raw repetition-level metrics across all 50 random masks per dropout level.
* `realistic_outage_scenarios.csv`: Tabulated error metrics and ensemble buffers across the 5 structured outage scenarios.
* `sensor_dropout_metrics.csv`: Backward-compatible 4-row summary table.
* `seasonal_quintile_error_breakdown.csv`: Error metrics and absolute error shares across concentration regimes.
* `diebold_mariano_significance_table.csv`: Statistical hypothesis test statistics and significance flags.

### `06_literature_context/`
* `literature_comparison_bar.png`: Comparative bar chart contextualizing Smog Net against published benchmarks (Lakshmi 2025, Sreenivasulu 2024, Dhandapani 2023, Lakshmi 2024) with detailed methodological footnotes.
* `literature_comparison_table.csv`: Detailed benchmark table with verified DOIs, reported metrics, evaluation scopes, and direct-fetch provenance notes.

### `07_computational/`
* `training_inference_latency_bar.png`: Dual-panel log-scale bar chart benchmarking training time (seconds) and per-sample inference latency (milliseconds) across architectures.
* `computational_benchmarks_table.csv`: Hardware specifications, training times, inference latencies, and throughputs.
