"""
HeatSentinel — Census & Heat Vulnerability Index (HVI) Module
Computes composite HVI combining meteorological risk with demographic census data.
Formula: HVI = 0.40 * MeteoRisk + 0.25 * ElderlyPopRatio + 0.20 * SlumDensity + 0.15 * OutdoorWorkersRatio
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

# District Census Demographics Registry
CENSUS_DEMOGRAPHICS = {
    "RJ01": {"elderly_ratio": 0.14, "slum_density": 0.22, "outdoor_ratio": 0.35, "total_pop": 3073350},
    "MH02": {"elderly_ratio": 0.16, "slum_density": 0.28, "outdoor_ratio": 0.38, "total_pop": 2405665},
    "TS03": {"elderly_ratio": 0.12, "slum_density": 0.30, "outdoor_ratio": 0.32, "total_pop": 3943323},
    "GJ04": {"elderly_ratio": 0.13, "slum_density": 0.25, "outdoor_ratio": 0.34, "total_pop": 5577940},
    "DL05": {"elderly_ratio": 0.15, "slum_density": 0.35, "outdoor_ratio": 0.40, "total_pop": 16787941},
    "BR06": {"elderly_ratio": 0.18, "slum_density": 0.32, "outdoor_ratio": 0.42, "total_pop": 5838465},
    "OD07": {"elderly_ratio": 0.17, "slum_density": 0.26, "outdoor_ratio": 0.36, "total_pop": 837737},
    "UP08": {"elderly_ratio": 0.16, "slum_density": 0.29, "outdoor_ratio": 0.39, "total_pop": 4589838},
    "TN09": {"elderly_ratio": 0.14, "slum_density": 0.27, "outdoor_ratio": 0.30, "total_pop": 4646732},
    "WB10": {"elderly_ratio": 0.19, "slum_density": 0.31, "outdoor_ratio": 0.33, "total_pop": 4496694},
}


def calculate_hvi(
    district_code: str,
    heat_index: float,
    temp_anomaly: float
) -> float:
    """
    Computes Heat Vulnerability Index (HVI) score normalized between 0.0 and 1.0.
    """
    demo = CENSUS_DEMOGRAPHICS.get(district_code, {
        "elderly_ratio": 0.15, "slum_density": 0.25, "outdoor_ratio": 0.35
    })

    # Normalized meteorological risk score
    meteo_risk = np.clip((heat_index - 35.0) / 20.0 + (temp_anomaly / 10.0), 0.0, 1.0)

    hvi = (
        0.40 * meteo_risk
        + 0.25 * demo["elderly_ratio"]
        + 0.20 * demo["slum_density"]
        + 0.15 * demo["outdoor_ratio"]
    )

    return round(float(np.clip(hvi, 0.0, 1.0)), 3)
