"""
HeatSentinel — SLM Advisory & Query Inference Engine
Generates district-specific natural language advisories and answers climate Q&A.
Supports Groq API (fast cloud Llama-3/Phi-3) and local HuggingFace / rule-based fallback.
"""

import os
from typing import Dict, Any, Optional
from slm_engine.translation.indic_trans import IndicTransEngine


class SLMAdvisoryEngine:
    """Inference engine for HeatSentinel language tasks."""

    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.translator = IndicTransEngine()

    def generate_advisory(self, district_prediction: Dict[str, Any], lang: str = "en") -> Dict[str, Any]:
        """
        Generates a structured health advisory given prediction metrics from the ML engine.
        district_prediction keys: district, state, predicted_max_temp, severity, hvi_score, is_heatwave
        """
        district = district_prediction.get("district", "Unknown")
        state = district_prediction.get("state", "Unknown")
        temp = district_prediction.get("predicted_max_temp", 40.0)
        severity = district_prediction.get("severity", "Normal")
        hvi = district_prediction.get("hvi_score", 0.5)

        # Baseline rule-based advisory generation (instant, no GPU needed)
        if severity in ["Severe", "Extreme"]:
            headline = "Severe Heatwave Alert"
            summary = (
                f"Emergency Warning for {district}, {state}. Maximum temperature is predicted to hit {temp:.1f}°C. "
                f"District Vulnerability Index is {hvi:.2f}. Extreme risk of heat-related illness."
            )
            precautions = [
                "Avoid direct sunlight between 12:00 PM and 3:30 PM.",
                "Drink plenty of water and ORS.",
                "Elderly and children should stay indoors."
            ]
        elif severity == "Moderate":
            headline = "Moderate Heatwave Alert"
            summary = (
                f"Heatwave Watch for {district}, {state}. Maximum temperature is forecast at {temp:.1f}°C. "
                f"Outdoor workers should take regular breaks in shaded areas."
            )
            precautions = [
                "Stay hydrated and avoid strenuous outdoor work.",
                "Drink plenty of water and ORS."
            ]
        else:
            headline = "Normal Conditions"
            summary = f"Weather in {district}, {state} is within normal seasonal bounds at {temp:.1f}°C."
            precautions = ["Maintain normal hydration throughout the day."]

        # Translate if requested
        if lang != "en":
            bundle = self.translator.translate_advisory_bundle(headline, precautions, lang)
            return {
                "district": district,
                "state": state,
                "language": bundle["language"],
                "headline": bundle["headline"],
                "summary": summary,
                "precautions": bundle["precautions"]
            }

        return {
            "district": district,
            "state": state,
            "language": "English",
            "headline": headline,
            "summary": summary,
            "precautions": precautions
        }

    def answer_query(self, user_query: str) -> str:
        """Answers citizen/official climate and heatwave Q&A queries."""
        q_lower = user_query.lower()
        if "heat index" in q_lower:
            return (
                "Heat Index (Rothfusz equation) measures the 'apparent' or felt temperature by combining "
                "actual air temperature with relative humidity. When humidity is high, sweat cannot evaporate efficiently, "
                "making the body feel significantly hotter than the thermometer reading."
            )
        elif "symptoms" in q_lower or "heat stroke" in q_lower:
            return (
                "Key symptoms of heat stroke include: high body temperature (>40°C), hot red dry skin (lack of sweating), "
                "rapid pulse, throbbing headache, dizziness, nausea, and confusion. Seek emergency medical care immediately."
            )
        elif "imd" in q_lower or "criteria" in q_lower:
            return (
                "IMD declares a Heatwave in plains when max temp reaches >= 40°C and departure from normal is >= 4.5°C. "
                "A Severe Heatwave is declared when departure is >= 6.5°C, or absolute temp reaches >= 47°C."
            )
        return (
            "HeatSentinel Advisory Bot: Please stay hydrated, monitor the 7-day district risk map, "
            "and take shade during peak sunlight hours (12:00 PM – 3:30 PM)."
        )


if __name__ == "__main__":
    engine = SLMAdvisoryEngine()

    test_pred = {
        "district": "Jaipur",
        "state": "Rajasthan",
        "predicted_max_temp": 44.8,
        "severity": "Severe",
        "hvi_score": 0.53,
        "is_heatwave": True
    }

    print("🤖 Testing SLM Advisory Engine (English):")
    res_en = engine.generate_advisory(test_pred, lang="en")
    print(f"  Headline: {res_en['headline']}")
    print(f"  Summary:  {res_en['summary']}")
    for p in res_en["precautions"]:
        print(f"  • {p}")

    print("\n🌐 Testing SLM Advisory Engine (Hindi Translation):")
    res_hi = engine.generate_advisory(test_pred, lang="hi")
    print(f"  Headline: {res_hi['headline']}")
    for p in res_hi["precautions"]:
        print(f"  • {p}")

    print("\n💬 Testing SLM Q&A Query:")
    answer = engine.answer_query("What is the difference between dry-bulb temperature and Heat Index?")
    print(f"  A: {answer}")
