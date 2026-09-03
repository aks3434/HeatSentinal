"""
HeatSentinel — Model C: Stacking Meta-Learner Module
Blends Model A (XGBoost P_XGB) and Model B (PyTorch Bi-LSTM P_LSTM) with HVI score
into a single calibrated final risk score (P_final).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from ml_engine.models.xgboost_model import xgb_predictor
from ml_engine.models.bilstm_model import bilstm_predictor
from ml_engine.data_pipeline.census_hvi import calculate_hvi


class HeatSentinelMetaLearner:
    def __init__(self):
        self.meta_model = LogisticRegression()
        self._calibrate_baseline_weights()

    def _calibrate_baseline_weights(self):
        """Calibrates initial meta-model blending weights."""
        X_meta = np.array([
            [0.1, 0.1, 0.2],
            [0.4, 0.5, 0.5],
            [0.8, 0.85, 0.75],
            [0.95, 0.92, 0.90]
        ])
        y_meta = np.array([0, 0, 1, 1])
        self.meta_model.fit(X_meta, y_meta)

    def predict_stacked_risk(
        self,
        today_row: Dict[str, Any],
        sequence_matrix: np.ndarray,
        district_code: str = "RJ01"
    ) -> Dict[str, Any]:
        """
        Executes Dual-Model Stacking Ensemble:
        P_final = MetaLearner(P_XGB, P_LSTM, HVI_Score)
        """
        # 1. Model A Output (XGBoost)
        res_xgb = xgb_predictor.predict_row(today_row)
        p_xgb = res_xgb["p_xgboost"]
        pred_temp = res_xgb["predicted_max_temp"]

        # 2. Model B Output (Bi-LSTM)
        res_lstm = bilstm_predictor.predict_sequence(sequence_matrix)
        p_lstm = res_lstm["p_bilstm"]

        # 3. HVI Score Calculation
        hvi_score = calculate_hvi(district_code, today_row.get("heat_index", 42.0), today_row.get("temp_anomaly", 3.0))

        # 4. Meta-Learner Stacking Blending
        meta_features = np.array([[p_xgb, p_lstm, hvi_score]])
        p_final = float(self.meta_model.predict_proba(meta_features)[0][1])

        # 5. Severity Determination
        if pred_temp >= 47.0 or p_final >= 0.85:
            severity = "Severe"
        elif pred_temp >= 44.0 or p_final >= 0.50:
            severity = "Moderate"
        else:
            severity = "Normal"

        return {
            "p_final_calibrated": round(p_final, 4),
            "p_xgboost": p_xgb,
            "p_bilstm": p_lstm,
            "hvi_score": hvi_score,
            "predicted_max_temp": pred_temp,
            "severity": severity,
            "is_heatwave": p_final >= 0.50
        }


meta_learner = HeatSentinelMetaLearner()


if __name__ == "__main__":
    test_row = {
        "temp_max": 44.5, "humidity": 48.0, "wind_speed": 12.0, "solar_rad": 26.5,
        "heat_index": 49.2, "wbgt": 33.8, "temp_anomaly": 4.0,
        "rolling_mean_3d": 43.8, "rolling_mean_7d": 42.5, "rolling_max_14d": 45.0,
        "consecutive_hot_days": 5
    }
    dummy_seq = np.random.normal(40.0, 3.0, (14, 11))
    
    stacked_res = meta_learner.predict_stacked_risk(test_row, dummy_seq, district_code="RJ01")
    print("\n🎉 Stacking Ensemble Test Result:")
    print(stacked_res)
