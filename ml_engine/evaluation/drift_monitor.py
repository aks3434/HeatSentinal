"""
HeatSentinel — Model & Data Drift Monitor
Detects feature and prediction drift between historical baseline and incoming weather batches.
Uses two-sample Kolmogorov-Smirnov (KS) statistical tests.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PARQUET_PATH = DATA_DIR / "historical_weather_india.parquet"


def check_dataset_drift(significance_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Splits multi-year historical dataset into Reference (2019-2022) vs Recent (2023-2024)
    and checks for distribution drift on core climatic features.
    """
    if not PARQUET_PATH.exists():
        print(f"⚠️ Dataset not found at {PARQUET_PATH}")
        return {}

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Reference window (2019-2022) vs Inference window (2023-2024)
    ref_df = df[df["date"] < "2023-01-01"]
    curr_df = df[df["date"] >= "2023-01-01"]

    features_to_monitor = ["temp_max", "humidity", "wind_speed", "solar_rad", "heat_index", "wbgt"]
    drift_report = {}
    drift_detected = False

    print("🔍 Running HeatSentinel KS-Test Data Drift Monitor...")
    print(f"  • Reference Records: {len(ref_df):,}")
    print(f"  • Recent Records:    {len(curr_df):,}\n")

    for feat in features_to_monitor:
        if feat in ref_df.columns and feat in curr_df.columns:
            stat, p_value = stats.ks_2samp(ref_df[feat].dropna(), curr_df[feat].dropna())
            is_drift = p_value < significance_threshold

            if is_drift:
                drift_detected = True

            drift_report[feat] = {
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 6),
                "drift_detected": is_drift,
                "status": "DRIFT_ALERT" if is_drift else "STABLE"
            }
            print(f"  • {feat:<15} | KS-Stat: {stat:.4f} | p-value: {p_value:.5f} | [{drift_report[feat]['status']}]")

    print("\n--------------------------------------------------")
    if drift_detected:
        print("⚠️ ACTION RECOMMENDED: Climate distribution shift detected in recent years. Model retraining recommended.")
    else:
        print("✅ STABLE: All feature distributions are within statistical tolerance.")
    print("--------------------------------------------------")

    return {
        "drift_detected": drift_detected,
        "features": drift_report
    }


if __name__ == "__main__":
    check_dataset_drift()
