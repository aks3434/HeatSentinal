"""
HeatSentinel — Feature Engineering Formulas Module
Computes Rothfusz Heat Index, WBGT, 3d/7d/14d rolling stats, and IMD Severity labels.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from ml_engine.data_pipeline.census_hvi import calculate_hvi


def calculate_heat_index(temp_c: float, relative_humidity: float) -> float:
    """Computes NOAA/Rothfusz Heat Index (°C)."""
    T = temp_c * 9.0 / 5.0 + 32.0
    RH = relative_humidity

    hi = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))

    if hi >= 80.0:
        c1, c2, c3, c4 = -42.379, 2.04901523, 10.14333127, -0.22475541
        c5, c6, c7 = -6.83783e-3, -5.481717e-2, 1.22874e-3
        c8, c9 = 8.5282e-4, -1.99e-6

        hi = (
            c1 + c2*T + c3*RH + c4*T*RH + c5*(T**2)
            + c6*(RH**2) + c7*(T**2)*RH + c8*T*(RH**2) + c9*(T**2)*(RH**2)
        )

        if RH < 13.0 and 80.0 <= T <= 112.0:
            hi -= ((13.0 - RH) / 4.0) * np.sqrt((17.0 - abs(T - 95.0)) / 17.0)
        elif RH > 85.0 and 80.0 <= T <= 87.0:
            hi += ((RH - 85.0) / 10.0) * ((87.0 - T) / 5.0)

    return round(float((hi - 32.0) * 5.0 / 9.0), 2)


def calculate_wbgt(temp_c: float, relative_humidity: float) -> float:
    """Computes Wet Bulb Globe Temperature (WBGT) approximation (°C)."""
    e_hpa = (relative_humidity / 100.0) * 6.105 * np.exp((17.27 * temp_c) / (237.7 + temp_c))
    e_kpa = e_hpa * 0.1
    wbgt = 0.567 * temp_c + 0.393 * e_kpa + 3.94
    return round(float(wbgt), 2)


def compute_imd_severity(max_temp: float, baseline_normal: float, consecutive_hot_days: int = 1) -> Dict[str, Any]:
    """Classifies heatwave severity according to official IMD criteria."""
    anomaly = round(max_temp - baseline_normal, 2)
    is_heatwave = False
    severity = "Normal"

    if max_temp >= 48.0:
        is_heatwave = True
        severity = "Extreme"
    elif max_temp >= 47.0 or (max_temp >= 40.0 and anomaly >= 6.5):
        is_heatwave = True
        severity = "Severe"
    elif max_temp >= 45.0 or (max_temp >= 40.0 and anomaly >= 4.5 and consecutive_hot_days >= 2):
        is_heatwave = True
        severity = "Moderate"

    return {
        "is_heatwave": is_heatwave,
        "severity": severity,
        "temp_anomaly": anomaly,
        "consecutive_hot_days": consecutive_hot_days
    }


def engineer_dataframe_features(df: pd.DataFrame, district_code: str = "RJ01", baseline_normal: float = 40.5) -> pd.DataFrame:
    """Enriches raw weather DataFrame with all HeatSentinel engineered features."""
    df = df.sort_values("date").reset_index(drop=True).copy()

    df["heat_index"] = df.apply(lambda r: calculate_heat_index(r["temp_max"], r["humidity"]), axis=1)
    df["wbgt"] = df.apply(lambda r: calculate_wbgt(r["temp_max"], r["humidity"]), axis=1)
    df["temp_anomaly"] = (df["temp_max"] - baseline_normal).round(2)

    df["rolling_mean_3d"] = df["temp_max"].rolling(3, min_periods=1).mean().round(2)
    df["rolling_mean_7d"] = df["temp_max"].rolling(7, min_periods=1).mean().round(2)
    df["rolling_max_14d"] = df["temp_max"].rolling(14, min_periods=1).max().round(2)

    is_hot = (df["temp_max"] >= 40.0).astype(int)
    streak, cur = [], 0
    for val in is_hot:
        cur = cur + 1 if val == 1 else 0
        streak.append(cur)
    df["consecutive_hot_days"] = streak

    severities, is_hw, hvi_scores = [], [], []
    for _, r in df.iterrows():
        imd = compute_imd_severity(r["temp_max"], baseline_normal, r["consecutive_hot_days"])
        severities.append(imd["severity"])
        is_hw.append(imd["is_heatwave"])
        hvi_scores.append(calculate_hvi(district_code, r["heat_index"], imd["temp_anomaly"]))

    df["severity"] = severities
    df["is_heatwave"] = is_hw
    df["hvi_score"] = hvi_scores

    return df
