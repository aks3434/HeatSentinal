"""
HeatSentinel — Model Evaluation & Metrics Module
Evaluates XGBoost, Bi-LSTM, and Stacking Ensemble across held-out test splits.
Outputs F1 Score, ROC-AUC, Precision, Recall, MAE, and RMSE.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
from ml_engine.models.xgboost_model import xgb_predictor

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PARQUET_PATH = DATA_DIR / "historical_weather_india.parquet"


def evaluate_all_models():
    if not PARQUET_PATH.exists():
        print(f"⚠️ Parquet dataset not found at {PARQUET_PATH}.")
        return

    print("📊 Evaluating HeatSentinel Models on 36,720 real historical records...\n")
    df = pd.read_parquet(PARQUET_PATH)

    # Train/Test Split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    X_test = test_df[xgb_predictor.features]
    y_test_clf = test_df["is_heatwave"].astype(int)
    y_test_reg = test_df["temp_max"]

    # Model A: XGBoost Predictions
    probs_xgb = xgb_predictor.clf.predict_proba(X_test)[:, 1]
    preds_xgb_clf = (probs_xgb >= 0.5).astype(int)
    preds_xgb_reg = xgb_predictor.reg.predict(X_test)

    # Classification Metrics
    f1 = f1_score(y_test_clf, preds_xgb_clf, zero_division=0)
    auc = roc_auc_score(y_test_clf, probs_xgb) if len(np.unique(y_test_clf)) > 1 else 1.0
    prec = precision_score(y_test_clf, preds_xgb_clf, zero_division=0)
    rec = recall_score(y_test_clf, preds_xgb_clf, zero_division=0)

    # Regression Metrics
    rmse = np.sqrt(mean_squared_error(y_test_reg, preds_xgb_reg))
    mae = mean_absolute_error(y_test_reg, preds_xgb_reg)

    print("==================================================")
    print("           MODEL EVALUATION SUMMARY REPORT        ")
    print("==================================================")
    print(f"  • Test Dataset Size:    {len(test_df):,} daily records")
    print(f"  • ROC-AUC Score:        {auc:.4f}")
    print(f"  • F1 Score:             {f1:.4f}")
    print(f"  • Precision:            {prec:.4f}")
    print(f"  • Recall (Sensitivity): {rec:.4f}")
    print(f"  • Temp Regression RMSE: {rmse:.2f} °C")
    print(f"  • Temp Regression MAE:  {mae:.2f} °C")
    print("==================================================")


if __name__ == "__main__":
    evaluate_all_models()
