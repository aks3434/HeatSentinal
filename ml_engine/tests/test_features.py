"""
HeatSentinel — Unit Test Suite
Validates mathematical calculations, IMD heatwave classification logic, and model contracts.
Run via: python -m unittest ml_engine.tests.test_features
"""

import unittest
import numpy as np
from ml_engine.data_pipeline.features import calculate_heat_index, calculate_wbgt, compute_imd_severity
from ml_engine.data_pipeline.census_hvi import calculate_hvi
from ml_engine.models.meta_learner import meta_learner
from ml_engine.models.health_impact_model import predict_heat_stroke_cases


class TestHeatSentinelEngine(unittest.TestCase):

    def test_heat_index_calculation(self):
        """Test Rothfusz Heat Index matches standard NOAA physical boundaries."""
        # High heat and high humidity must yield felt temperature greater than air temperature
        hi = calculate_heat_index(temp_c=42.0, relative_humidity=60.0)
        self.assertGreater(hi, 42.0)
        self.assertIsInstance(hi, float)

    def test_wbgt_calculation(self):
        """Test Wet Bulb Globe Temperature approximation."""
        wbgt = calculate_wbgt(temp_c=40.0, relative_humidity=50.0)
        self.assertGreater(wbgt, 20.0)
        self.assertLess(wbgt, 50.0)

    def test_imd_severity_classification(self):
        """Test official IMD Heatwave definitions."""
        # Extreme heatwave (>= 48°C)
        res_extreme = compute_imd_severity(max_temp=48.5, baseline_normal=40.0)
        self.assertTrue(res_extreme["is_heatwave"])
        self.assertEqual(res_extreme["severity"], "Extreme")

        # Normal condition (< 40°C)
        res_normal = compute_imd_severity(max_temp=35.0, baseline_normal=38.0)
        self.assertFalse(res_normal["is_heatwave"])
        self.assertEqual(res_normal["severity"], "Normal")

    def test_hvi_score_bounds(self):
        """HVI score must always lie within [0.0, 1.0]."""
        hvi = calculate_hvi("RJ01", heat_index=45.0, temp_anomaly=4.0)
        self.assertGreaterEqual(hvi, 0.0)
        self.assertLessEqual(hvi, 1.0)

    def test_meta_learner_prediction_contract(self):
        """Stacking Ensemble must return valid calibrated probabilities."""
        test_row = {
            "temp_max": 44.5, "humidity": 48.0, "wind_speed": 12.0, "solar_rad": 26.5,
            "heat_index": 49.2, "wbgt": 33.8, "temp_anomaly": 4.0,
            "rolling_mean_3d": 43.8, "rolling_mean_7d": 42.5, "rolling_max_14d": 45.0,
            "consecutive_hot_days": 5
        }
        seq = np.random.normal(40.0, 2.0, (14, 11))
        pred = meta_learner.predict_stacked_risk(test_row, seq, district_code="RJ01")

        self.assertIn("p_final_calibrated", pred)
        self.assertGreaterEqual(pred["p_final_calibrated"], 0.0)
        self.assertLessEqual(pred["p_final_calibrated"], 1.0)
        self.assertIn("severity", pred)

    def test_health_impact_model(self):
        """Epidemiological model should predict positive case estimates under extreme heat."""
        health = predict_heat_stroke_cases("RJ01", predicted_max_temp=45.0, hvi_score=0.7)
        self.assertGreater(health["expected_heat_stroke_cases"], 0)
        self.assertIn("health_system_risk_level", health)


if __name__ == "__main__":
    unittest.main()
