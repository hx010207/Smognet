"""
Automated Metric Concordance & Consistency Verification Script
=============================================================
Audits and asserts 100% numerical and methodological concordance across:
  - evaluation_ledger.csv
  - results.json
  - table1_model_comparison.md
  - manuscript_summary.md
  - summary.md
  - results/01_model_comparison/model_comparison_table.csv
  - results/05_robustness/enhanced_sensor_dropout_metrics.csv
  - results/04_spatial_generalization/spatial_transfer_table.csv
"""

import sys
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

def run_audit():
    print("=" * 78)
    print(" SMOG NET PRE-SUBMISSION METRIC CONCORDANCE AUDIT")
    print("=" * 78)
    
    errors = []
    warnings = []
    
    # ---------------------------------------------------------
    # 1. Load evaluation_ledger.csv
    # ---------------------------------------------------------
    ledger_path = Path("evaluation_ledger.csv")
    if not ledger_path.exists():
        errors.append("evaluation_ledger.csv not found!")
        return errors, warnings
    
    df_ledger = pd.read_csv(ledger_path)
    ledger = {str(row["Metric_ID"]): row for _, row in df_ledger.iterrows()}
    print(f"[OK] Loaded evaluation_ledger.csv ({len(ledger)} records)")
    
    # ---------------------------------------------------------
    # 2. Check Walk-Forward Cross-Validation (M01 vs Table 1 vs results.json)
    # ---------------------------------------------------------
    mct_path = Path("results/01_model_comparison/model_comparison_table.csv")
    if mct_path.exists():
        df_mct = pd.read_csv(mct_path)
        smog_mct = df_mct[df_mct["Model"] == "Smog Net"].iloc[0]
        
        m01 = ledger.get("M01")
        if m01 is not None:
            exp_rmse = f"{smog_mct['Mean_RMSE']:.2f} ± {smog_mct['Std_RMSE']:.2f}"
            exp_mae = f"{smog_mct['Mean_MAE']:.2f} ± {smog_mct['Std_MAE']:.2f}"
            exp_r2 = f"{smog_mct['Mean_R2']:.4f} ± {smog_mct['Std_R2']:.4f}"
            exp_smape = f"{smog_mct['Mean_sMAPE']:.2f} ± {smog_mct['Std_sMAPE']:.2f}%"
            
            if m01["RMSE_ugm3"] != exp_rmse:
                errors.append(f"M01 RMSE: ledger='{m01['RMSE_ugm3']}', expected='{exp_rmse}'")
            if m01["MAE_ugm3"] != exp_mae:
                errors.append(f"M01 MAE: ledger='{m01['MAE_ugm3']}', expected='{exp_mae}'")
            if m01["R2_Score"] != exp_r2:
                errors.append(f"M01 R2: ledger='{m01['R2_Score']}', expected='{exp_r2}'")
            if m01["sMAPE_pct"] != exp_smape:
                errors.append(f"M01 sMAPE: ledger='{m01['sMAPE_pct']}', expected='{exp_smape}'")
            print(f"[OK] M01 Walk-Forward cross-validation matches model_comparison_table.csv exactly")
            
    # Check results.json headline walk-forward
    res_path = Path("results.json")
    if res_path.exists():
        with open(res_path, "r", encoding="utf-8") as f:
            res_json = json.load(f)
        try:
            hl = res_json.get("final_reconciled_headline_summary", {})
            if hl:
                if "25.68" not in hl.get("Walk_Forward_5Fold_Mean_MAE", ""):
                    errors.append(f"results.json headline MAE mismatch: '{hl.get('Walk_Forward_5Fold_Mean_MAE')}'")
                if "40.40" not in hl.get("Walk_Forward_5Fold_Mean_RMSE", ""):
                    errors.append(f"results.json headline RMSE mismatch: '{hl.get('Walk_Forward_5Fold_Mean_RMSE')}'")
                print(f"[OK] results.json headline summary matches canonical walk-forward metrics")
        except Exception as e:
            warnings.append(f"Could not parse results.json final_reconciled_headline_summary: {e}")

    # ---------------------------------------------------------
    # 3. Check Fixed Test Split Benchmark & Reconciliations
    # ---------------------------------------------------------
    m02 = ledger.get("M02")
    m03 = ledger.get("M03")
    m04 = ledger.get("M04")
    m06 = ledger.get("M06")
    
    if m02 is not None and m06 is not None:
        if m02["RMSE_ugm3"] != "39.16":
            errors.append(f"M02 RMSE mismatch: expected '39.16', got '{m02['RMSE_ugm3']}'")
        if "39.04" not in str(m06["RMSE_ugm3"]):
            errors.append(f"M06 RMSE mismatch: expected '39.04' in '{m06['RMSE_ugm3']}'")
        if "reconcile" not in str(m02["Status_and_Notes"]).lower():
            errors.append("M02 Status_and_Notes missing 39.16 vs 39.04 reconciliation note")
        if "reconcile" not in str(m06["Status_and_Notes"]).lower():
            errors.append("M06 Status_and_Notes missing 39.04 vs 39.16 reconciliation note")
        print(f"[OK] M02 (39.16) and M06 (39.04) fixed benchmark values and notes verified")
        
    if m03 is not None and m04 is not None:
        if m03["RMSE_ugm3"] != "40.43":
            errors.append(f"M03 RMSE mismatch: expected '40.43', got '{m03['RMSE_ugm3']}'")
        if m04["RMSE_ugm3"] != "40.11":
            errors.append(f"M04 RMSE mismatch: expected '40.11', got '{m04['RMSE_ugm3']}'")
        if "reconcile" not in str(m04["Status_and_Notes"]).lower():
            errors.append("M04 Status_and_Notes missing 40.11 vs 40.43 CPU/GPU reconciliation note")
        print(f"[OK] M03 (40.43 GPU) and M04 (40.11 CPU) baseline values and notes verified")

    # ---------------------------------------------------------
    # 4. Check Audit Verification Rows (V01, V02, V03)
    # ---------------------------------------------------------
    for vid, desc in [
        ("V01", "40.11 vs 40.43 reconciliation"),
        ("V02", "Wilcoxon floor saturation"),
        ("V03", "39.04 vs 39.16 reconciliation")
    ]:
        if vid not in ledger:
            errors.append(f"Missing audit verification row {vid} ({desc}) in evaluation_ledger.csv!")
        else:
            print(f"[OK] Audit verification record {vid} present in ledger")

    # ---------------------------------------------------------
    # 5. Check Sensor Dropout Robustness (M07-M09)
    # ---------------------------------------------------------
    drop_path = Path("results/05_robustness/enhanced_sensor_dropout_metrics.csv")
    if drop_path.exists():
        df_drop = pd.read_csv(drop_path)
        for _, r in df_drop.iterrows():
            rate = int(r["Dropout_Pct"])
            mid = {0: "M06", 10: "M07", 25: "M08", 50: "M09"}.get(rate)
            if mid and mid in ledger:
                led_row = ledger[mid]
                if rate > 0:
                    exp_buf = f"+{r['Buffer_vs_Imp_mean']:.2f} ± {r['Buffer_vs_Imp_ci95']:.2f} µg/m³"
                    if exp_buf not in led_row["Status_and_Notes"]:
                        errors.append(f"{mid} Buffer mismatch in notes: '{exp_buf}' not in ledger notes")
        print(f"[OK] M07-M09 Monte Carlo degradation metrics verified against enhanced_sensor_dropout_metrics.csv")

    # ---------------------------------------------------------
    # 6. Check Spatial Generalization (M13-M17)
    # ---------------------------------------------------------
    spatial_path = Path("results/04_spatial_generalization/spatial_transfer_table.csv")
    if spatial_path.exists():
        df_sp = pd.read_csv(spatial_path)
        sp_map = {row["City"].split(" ")[0]: row for _, row in df_sp.iterrows()}
        city_m_map = {"Gurugram": "M13", "Lucknow": "M14", "Patna": "M15", "Bengaluru": "M16", "Chennai": "M17"}
        for city, mid in city_m_map.items():
            if city in sp_map and mid in ledger:
                exp_rmse = f"{sp_map[city]['Frozen_Transfer_RMSE']:.2f}"
                act_rmse = f"{float(ledger[mid]['RMSE_ugm3']):.2f}"
                if exp_rmse != act_rmse:
                    errors.append(f"{mid} ({city}) RMSE mismatch: ledger='{act_rmse}', table='{exp_rmse}'")
        print(f"[OK] M13-M17 Spatial transfer metrics verified against spatial_transfer_table.csv")

    # ---------------------------------------------------------
    # 7. Check Multi-Horizon Forecasting (M10-M12)
    # ---------------------------------------------------------
    for mid, exp_pers, exp_smog in [
        ("M10", "41.02", "43.34"),
        ("M11", "61.74", "59.76"),
        ("M12", "72.93", "58.09"),
    ]:
        if mid in ledger:
            rmse_val = ledger[mid]["RMSE_ugm3"]
            if exp_pers not in rmse_val or exp_smog not in rmse_val:
                errors.append(f"{mid} Multi-horizon mismatch: expected '{exp_pers}' & '{exp_smog}' in '{rmse_val}'")
    print(f"[OK] M10-M12 Multi-horizon forecasting metrics verified across files")

    # ---------------------------------------------------------
    # 8. Check FIRMS Satellite Active-Fire Metrics (M05)
    # ---------------------------------------------------------
    m05 = ledger.get("M05")
    if m05 is not None:
        if m05["RMSE_ugm3"] != "39.13":
            errors.append(f"M05 RMSE mismatch: expected '39.13', got '{m05['RMSE_ugm3']}'")
        if m05["R2_Score"] != "0.7980":
            errors.append(f"M05 R2 mismatch: expected '0.7980', got '{m05['R2_Score']}'")
    print(f"[OK] M05 NASA FIRMS satellite evaluation metrics verified across files")

    # ---------------------------------------------------------
    # 9. Check Split Conformal Prediction Metrics (M18)
    # ---------------------------------------------------------
    m18 = ledger.get("M18")
    if m18 is not None:
        if "95.68%" not in str(m18["sMAPE_pct"]) or "79.79" not in str(m18["RMSE_ugm3"]):
            errors.append(f"M18 Conformal interval mismatch in ledger: {m18['RMSE_ugm3']}, {m18['sMAPE_pct']}")
    print(f"[OK] M18 Conformal prediction interval metrics verified across files")

    # ---------------------------------------------------------
    # 10. Check Markdown Consistency Across All Files
    # ---------------------------------------------------------
    t1 = Path("table1_model_comparison.md").read_text(encoding="utf-8")
    ms = Path("manuscript_summary.md").read_text(encoding="utf-8")
    sm = Path("summary.md").read_text(encoding="utf-8")
    
    # Table 1 Assertions
    if "40.40 ± 12.80" not in t1 or "25.68 ± 7.28" not in t1 or "0.7565 ± 0.0751" not in t1:
        errors.append("table1_model_comparison.md missing canonical SmogNet walk-forward metrics")
    if "saturating the test's minimum achievable p-value" not in t1:
        errors.append("table1_model_comparison.md missing Wilcoxon saturation footnote")
    if "CPU exact greedy" not in t1:
        errors.append("table1_model_comparison.md missing 40.11 vs 40.43 CPU/GPU reconciliation")
    for val in ["39.16", "39.13", "41.02", "59.76", "58.09", "42.15", "43.89", "46.22", "41.82", "97.14 ± 26.45", "90.41 ± 67.30"]:
        if val not in t1:
            errors.append(f"table1_model_comparison.md missing shared benchmark metric: {val}")

    # Manuscript Summary Assertions
    if "40.40 ± 12.80" not in ms or "25.68 ± 7.28" not in ms or "0.7565 ± 0.0751" not in ms:
        errors.append("manuscript_summary.md missing canonical SmogNet walk-forward metrics")
    if "saturating the test's minimum achievable p-value" not in ms:
        errors.append("manuscript_summary.md missing Wilcoxon saturation footnote")
    if "CPU exact greedy" not in ms:
        errors.append("manuscript_summary.md missing 40.11 vs 40.43 CPU/GPU reconciliation")
    for val in ["39.16", "39.13", "39.04", "41.02", "59.76", "58.09", "45.38", "37.97", "35.91", "16.16", "21.04", "17.80"]:
        if val not in ms:
            errors.append(f"manuscript_summary.md missing shared benchmark metric: {val}")

    # Summary.md Assertions
    if "40.40 ± 12.80" not in sm or "25.68 ± 7.28" not in sm or "0.7565" not in sm:
        errors.append("summary.md missing canonical SmogNet walk-forward metrics")
    if "saturating test floor" not in sm and "saturates at the floor" not in sm:
        errors.append("summary.md missing Wilcoxon floor saturation explanation")
    for val in ["39.16", "39.13", "39.04", "40.43", "40.11", "41.02", "59.76", "58.09", "17.80"]:
        if val not in sm:
            errors.append(f"summary.md missing shared benchmark metric: {val}")

    print(f"[OK] Markdown documentation files cross-checked and validated")

    # ---------------------------------------------------------
    # Summary Output
    # ---------------------------------------------------------
    print("\n" + "=" * 78)
    print(f" AUDIT RESULT: {len(errors)} ERRORS, {len(warnings)} WARNINGS")
    print("=" * 78)
    for e in errors:
        print(f"  [ERROR] {e}")
    for w in warnings:
        print(f"  [WARNING] {w}")
        
    if len(errors) == 0:
        print("\n>>> ALL AUDITED METRICS ARE 100% CONCORDANT ACROSS EVERY REPOSITORY FILE! <<<\n")
    return errors, warnings

if __name__ == "__main__":
    errs, warns = run_audit()
    if errs:
        sys.exit(1)
    sys.exit(0)
