"""
HeatSentinel — Multi-Year Historical Dataset Downloader & Ingestion
Includes automatic HTTP 429 rate-limit handling & retries for full-scale ingestion.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from ml_engine.data_pipeline.ingestion import INDIAN_DISTRICTS
from ml_engine.data_pipeline.census_hvi import CENSUS_DEMOGRAPHICS
from ml_engine.data_pipeline.features import engineer_dataframe_features

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_historical_district_data(district_info: dict, start_date: str = "2019-01-01", end_date: str = "2024-08-01", max_retries: int = 3) -> pd.DataFrame:
    """
    Queries real daily weather from OpenMeteo Archive API with HTTP 429 rate-limit retry logic.
    """
    code = district_info["code"]
    lat = district_info["lat"]
    lon = district_info["lon"]
    name = district_info["name"]
    state = district_info["state"]
    baseline = district_info["baseline_normal"]

    print(f"📥 Fetching weather history for {name}, {state} ({start_date} to {end_date})...")

    archive_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,wind_speed_10m_max,shortwave_radiation_sum"
        f"&timezone=Asia%2FKolkata"
    )

    for attempt in range(max_retries):
        try:
            res = requests.get(archive_url, timeout=30)
            
            if res.status_code == 200:
                data = res.json().get("daily", {})
                df = pd.DataFrame({
                    "date": data["time"],
                    "temp_max": data["temperature_2m_max"],
                    "temp_min": data["temperature_2m_min"],
                    "humidity": data["relative_humidity_2m_mean"],
                    "wind_speed": data["wind_speed_10m_max"],
                    "solar_rad": data["shortwave_radiation_sum"]
                })
                
                enriched_df = engineer_dataframe_features(df, district_code=code, baseline_normal=baseline)
                enriched_df["district_code"] = code
                enriched_df["district_name"] = name
                enriched_df["state_name"] = state
                
                # Polite pause to prevent API rate-limiting
                time.sleep(0.3)
                return enriched_df

            elif res.status_code == 429:
                wait_time = (attempt + 1) * 2
                print(f"⏳ HTTP 429 Rate Limited for {name}. Retrying in {wait_time}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)

            else:
                print(f"⚠️ OpenMeteo returned HTTP {res.status_code} for {name}.")
                break

        except Exception as e:
            print(f"❌ Connection error for {name}: {e}")
            time.sleep(2)

    return pd.DataFrame()


def save_census_demographics():
    records = []
    for code, demo in CENSUS_DEMOGRAPHICS.items():
        records.append({
            "district_code": code,
            "elderly_ratio": demo.get("elderly_ratio", 0.15),
            "slum_density": demo.get("slum_density", 0.25),
            "outdoor_ratio": demo.get("outdoor_ratio", 0.35),
            "total_pop": demo.get("total_pop", 2500000)
        })
    df_census = pd.DataFrame(records)
    census_path = DATA_DIR / "census_demographics.csv"
    df_census.to_csv(census_path, index=False)


def run_full_data_download():
    all_dfs = []
    save_census_demographics()

    districts = INDIAN_DISTRICTS
    print(f"🚀 Starting dataset download across {len(districts)} Indian districts...\n")

    for dist in districts:
        df_dist = download_historical_district_data(dist, start_date="2019-01-01", end_date="2024-08-01")
        if not df_dist.empty:
            all_dfs.append(df_dist)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        parquet_path = DATA_DIR / "historical_weather_india.parquet"
        full_df.to_parquet(parquet_path, index=False)
        print(f"\n🎉 Successfully saved Parquet dataset: {parquet_path} ({len(full_df):,} total daily records)")

        csv_path = DATA_DIR / "historical_weather_india.csv"
        full_df.to_csv(csv_path, index=False)
        print(f"🎉 Successfully saved CSV dataset: {csv_path}")


if __name__ == "__main__":
    run_full_data_download()
