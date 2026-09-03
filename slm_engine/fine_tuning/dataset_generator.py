"""
HeatSentinel — SLM Fine-Tuning Dataset Generator
Parses IMD Heat Action Plans & synthetic prediction outputs to create JSONL training data
for LoRA fine-tuning (Phi-3 Mini / Qwen2-1.5B).
"""

import json
import pandas as pd
from pathlib import Path
from ml_engine.data_pipeline.ingestion import INDIAN_DISTRICTS

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUTPUT_CORPUS_PATH = DATA_DIR / "slm_finetune_corpus.jsonl"


def generate_slm_finetuning_corpus():
    """Generates structured Alpaca/ShareGPT-style Q&A instructions for LoRA fine-tuning."""
    print("📝 Generating SLM Fine-Tuning Corpus from IMD Heat Action Plans...")
    corpus = []

    # 1. District Forecast & Advisory Q&A Pairs
    for dist in INDIAN_DISTRICTS:
        name = dist["name"]
        state = dist["state"]
        base_temp = dist["baseline_normal"]

        # Sample instruction 1: Risk Assessment
        corpus.append({
            "instruction": f"Summarize the heatwave risk and baseline temperature for {name}, {state}.",
            "input": f"District: {name}, State: {state}, Baseline Normal Temp: {base_temp}°C",
            "output": f"In {name}, {state}, the official IMD baseline normal maximum temperature is {base_temp}°C. During summer months, temperatures exceeding {base_temp + 4.5}°C for two consecutive days trigger a Moderate Heatwave alert, while temperatures above 47°C trigger a Severe Heatwave emergency warning."
        })

        # Sample instruction 2: Citizen Health Precautions
        corpus.append({
            "instruction": f"What precautions should citizens in {name} take when predicted max temp reaches {base_temp + 4.0}°C?",
            "input": f"District: {name}, Predicted Temp: {base_temp + 4.0}°C, Severity: Moderate",
            "output": f"Citizens in {name} should avoid direct sun exposure between 12:00 PM and 3:30 PM. Drink 3-4 liters of water daily along with ORS, buttermilk (chaach), or lemon water. Outdoor laborers and elderly residents should stay in shade and watch for symptoms of heat stroke (dizziness, nausea, rapid pulse)."
        })

    # 2. General IMD Climate FAQ pairs
    general_faqs = [
        {
            "instruction": "What is the difference between dry-bulb temperature and Heat Index?",
            "input": "",
            "output": "Dry-bulb temperature measures actual air temperature using a standard thermometer. Heat Index (Rothfusz equation) measures the 'apparent' felt temperature by combining air temperature with relative humidity. High relative humidity prevents sweat evaporation, making the felt temperature significantly hotter and more hazardous to human health."
        },
        {
            "instruction": "What defines an official IMD Severe Heatwave in India?",
            "input": "",
            "output": "According to the India Meteorological Department (IMD), a Severe Heatwave is declared when maximum temperature reaches >= 40°C in plains with an anomaly >= 6.5°C above normal, or when the absolute maximum temperature reaches >= 47°C regardless of normal baseline."
        }
    ]

    corpus.extend(general_faqs)

    # Save to JSONL
    with open(OUTPUT_CORPUS_PATH, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Successfully created SLM Fine-Tuning Corpus: {OUTPUT_CORPUS_PATH} ({len(corpus)} QA pairs)")


if __name__ == "__main__":
    generate_slm_finetuning_corpus()
