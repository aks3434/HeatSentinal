"""
HeatSentinel — Model A: XGBoost Classifier & Regressor Module
Trains on 36,720 real historical records from data/historical_weather_india.parquet
and saves model weights to data/xgboost_clf.json and data/xgboost_reg.json.
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier
    from sklearn.ensemble import HistGradientBoostingRegressor as XGBRegressor
    XGB_AVAILABLE = False

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PARQUET_PATH = DATA_DIR / "historical_weather_india.parquet"
XGB_CLF_PATH = DATA_DIR / "xgboost_clf.json"
XGB_REG_PATH = DATA_DIR / "xgboost_reg.json"
SKLEARN_CLF_PATH = DATA_DIR / "sklearn_clf.joblib"
SKLEARN_REG_PATH = DATA_DIR / "sklearn_reg.joblib"


class HeatSentinelXGBoostPredictor:
    def __init__(self):
        self.features = [
            "temp_max", "humidity", "wind_speed", "solar_rad",
            "heat_index", "wbgt", "temp_anomaly",
            "rolling_mean_3d", "rolling_mean_7d", "rolling_max_14d",
            "consecutive_hot_days"
        ]
        
        if XGB_AVAILABLE:
            self.clf = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.04, random_state=42, eval_metric="logloss")
            self.reg = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.04, random_state=42)
        else:
            self.clf = XGBClassifier(max_iter=150, random_state=42)
            self.reg = XGBRegressor(max_iter=150, random_state=42)

        self.is_trained = False
        if XGB_AVAILABLE and XGB_CLF_PATH.exists() and XGB_REG_PATH.exists():
            self.clf.load_model(XGB_CLF_PATH)
            self.reg.load_model(XGB_REG_PATH)
            self.is_trained = True
            print(f"✅ Loaded pre-trained XGBoost models from {XGB_CLF_PATH}")
        else:
            self.train_on_historical_dataset()


    def train_on_historical_dataset(self):
        """Trains XGBoost on historical weather records and saves model weights to disk."""
        if not PARQUET_PATH.exists():
            print(f"⚠️ Parquet dataset not found at {PARQUET_PATH}.")
            return

        print(f"🤖 Training XGBoost Model A on real dataset ({PARQUET_PATH})...")
        df = pd.read_parquet(PARQUET_PATH)

        X = df[self.features]
        y_clf = df["is_heatwave"].astype(int)
        y_reg = df["temp_max"]

        self.clf.fit(X, y_clf)
        self.reg.fit(X, y_reg)
        self.is_trained = True

        # Save model weights to disk
        if XGB_AVAILABLE:
            self.clf.save_model(XGB_CLF_PATH)
            self.reg.save_model(XGB_REG_PATH)
            print(f"✅ Saved XGBoost Classifier model to {XGB_CLF_PATH}")
            print(f"✅ Saved XGBoost Regressor model to {XGB_REG_PATH}")
        else:
            joblib.dump(self.clf, SKLEARN_CLF_PATH)
            joblib.dump(self.reg, SKLEARN_REG_PATH)
            print(f"✅ Saved Fallback Classifier model to {SKLEARN_CLF_PATH}")

    def predict_row(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        X_input = pd.DataFrame([row_dict])[self.features]
        prob = float(self.clf.predict_proba(X_input)[0][1])
        pred_temp = float(self.reg.predict(X_input)[0])

        return {
            "p_xgboost": round(prob, 4),
            "predicted_max_temp": round(pred_temp, 2),
            "features": row_dict
        }


# Singleton Model Instance
xgb_predictor = HeatSentinelXGBoostPredictor()


if __name__ == "__main__":
    test_sample = {
        "temp_max": 44.5, "humidity": 48.0, "wind_speed": 12.0, "solar_rad": 26.5,
        "heat_index": 49.2, "wbgt": 33.8, "temp_anomaly": 4.0,
        "rolling_mean_3d": 43.8, "rolling_mean_7d": 42.5, "rolling_max_14d": 45.0,
        "consecutive_hot_days": 5
    }
    res = xgb_predictor.predict_row(test_sample)
    print("\n🧪 XGBoost Test Prediction Output:")
    print(res)
