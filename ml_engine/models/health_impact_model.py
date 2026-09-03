"""
HeatSentinel — Health Impact & Epidemiological Model
Predicts estimated heat-stroke admissions and hospital stress per district.
"""

import numpy as np
from typing import Dict, Any
from ml_engine.data_pipeline.census_hvi import CENSUS_DEMOGRAPHICS


def predict_heat_stroke_cases(
    district_code: str,
    predicted_max_temp: float,
    hvi_score: float
) -> Dict[str, Any]:
    """
    Epidemiological regression model:
    ExpectedHeatStrokes = round(TotalPop * 10^-5 * (MaxTemp - 38)^1.4 * HVI)
    """
    demo = CENSUS_DEMOGRAPHICS.get(district_code, {"total_pop": 2500000})
    pop = demo["total_pop"]

    if predicted_max_temp < 38.0:
        expected_cases = 0
        risk_level = "Low"
    else:
        temp_excess = predicted_max_temp - 38.0
        expected_cases = int(round((pop * 1e-5) * (temp_excess ** 1.4) * hvi_score * 0.8))
        
        if expected_cases > 150:
            risk_level = "Critical / Hospital Surge Warning"
        elif expected_cases > 50:
            risk_level = "High Risk"
        else:
            risk_level = "Moderate"

    return {
        "district_code": district_code,
        "total_population": pop,
        "predicted_max_temp": predicted_max_temp,
        "hvi_score": hvi_score,
        "expected_heat_stroke_cases": expected_cases,
        "health_system_risk_level": risk_level
    }


if __name__ == "__main__":
    res = predict_heat_stroke_cases("RJ01", predicted_max_temp=44.5, hvi_score=0.72)
    print("\n🏥 Health Impact Prediction Test:")
    print(res)
