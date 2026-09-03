"""
HeatSentinel — Dynamic All-India Ingestion Pipeline
Loads district centroids dynamically from data/india_districts.json.
Scales seamlessly across all states & territories of India.
"""

import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
GEOJSON_PATH = DATA_DIR / "india_districts.json"


def load_all_indian_districts() -> List[Dict[str, Any]]:
    """Loads all district boundaries, centroids, and baseline parameters from GeoJSON."""
    if not GEOJSON_PATH.exists():
        return [
            {"code": "RJ01", "name": "Jaipur", "state": "Rajasthan", "lat": 26.9124, "lon": 75.7873, "baseline_normal": 40.5},
            {"code": "MH02", "name": "Nagpur", "state": "Maharashtra", "lat": 21.1458, "lon": 79.0882, "baseline_normal": 42.0},
        ]

    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    districts = []
    for feature in geojson.get("features", []):
        props = feature["properties"]
        geom = feature["geometry"]

        if geom["type"] == "Polygon":
            coords = np.array(geom["coordinates"][0])
            lon_c, lat_c = coords[:, 0].mean(), coords[:, 1].mean()
        elif geom["type"] == "MultiPolygon":
            coords = np.array(geom["coordinates"][0][0])
            lon_c, lat_c = coords[:, 0].mean(), coords[:, 1].mean()
        else:
            lat_c, lon_c = 20.5937, 78.9629

        districts.append({
            "code": props.get("district_code", f"DIST_{len(districts)+1}"),
            "name": props.get("district_name", "Unknown"),
            "state": props.get("state_name", "India"),
            "lat": round(float(lat_c), 4),
            "lon": round(float(lon_c), 4),
            "baseline_normal": props.get("baseline_normal", 39.5)
        })

    return districts


INDIAN_DISTRICTS = load_all_indian_districts()


def fetch_openmeteo_forecast(lat: float, lon: float, days: int = 14) -> pd.DataFrame:
    """Fetches 14-day weather sequence (historical + forecast) from OpenMeteo API."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,wind_speed_10m_max,shortwave_radiation_sum"
        f"&forecast_days=7&past_days=7&timezone=Asia%2FKolkata"
    )

    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            daily = resp.json().get("daily", {})
            return pd.DataFrame({
                "date": daily["time"],
                "temp_max": daily["temperature_2m_max"],
                "temp_min": daily["temperature_2m_min"],
                "humidity": daily["relative_humidity_2m_mean"],
                "wind_speed": daily["wind_speed_10m_max"],
                "solar_rad": daily["shortwave_radiation_sum"]
            })
    except Exception as e:
        print(f"[Ingestion Warning] API request failed ({e}). Using synthetic generator.")

    return _generate_fallback_weather(lat, lon, days=days)


def _generate_fallback_weather(lat: float, lon: float, days: int = 14, base_temp: float = 40.5) -> pd.DataFrame:
    np.random.seed(int(abs(lat * 100 + lon * 10) % 10000))
    start_date = datetime.now() - timedelta(days=7)
    dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

    temp_max = base_temp + np.sin(np.linspace(0, 3, days)) * 3.5 + np.random.normal(0, 0.8, days)
    temp_min = temp_max - np.random.uniform(12.0, 16.0, days)
    humidity = np.clip(60.0 - (temp_max - 35.0) * 2.2 + np.random.normal(0, 4, days), 20.0, 85.0)
    wind_speed = np.clip(12.0 + np.random.normal(0, 3, days), 4.0, 35.0)
    solar_rad = np.clip(22.0 + (temp_max - 35.0) * 0.5 + np.random.normal(0, 1.5, days), 15.0, 32.0)

    return pd.DataFrame({
        "date": dates,
        "temp_max": np.round(temp_max, 1),
        "temp_min": np.round(temp_min, 1),
        "humidity": np.round(humidity, 1),
        "wind_speed": np.round(wind_speed, 1),
        "solar_rad": np.round(solar_rad, 1)
    })

