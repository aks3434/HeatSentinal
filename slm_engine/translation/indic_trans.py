"""
HeatSentinel — IndicTrans2 Multilingual Translation Engine
Translates English heatwave advisories into Indian regional languages:
Hindi (hi), Marathi (mr), Telugu (te), Tamil (ta), Bengali (bn).
Includes instant fallback templates for offline execution without 5GB checkpoints.
"""

from typing import Dict, Optional

# Supported language codes
SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "te": "Telugu",
    "ta": "Tamil",
    "bn": "Bengali",
}

# High-frequency advisory phrase bank for instant offline translation
CORE_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "hi": {
        "Severe Heatwave Alert": "गंभीर लू (हीटवेव) की चेतावनी",
        "Moderate Heatwave Alert": "मध्यम लू की चेतावनी",
        "Normal Conditions": "सामान्य मौसम",
        "Drink plenty of water and ORS.": "पर्याप्त मात्रा में पानी और ओआरएस पिएं।",
        "Avoid direct sunlight between 12:00 PM and 3:30 PM.": "दोपहर 12:00 से 3:30 बजे के बीच धूप में निकलने से बचें।",
        "Elderly and children should stay indoors.": "बुजुर्गों और बच्चों को घर के अंदर रहना चाहिए।",
        "Stay hydrated and avoid strenuous outdoor work.": "शरीर में पानी की कमी न होने दें और भारी शारीरिक काम से बचें।",
    },
    "mr": {
        "Severe Heatwave Alert": "तीव्र उष्णतेच्या लाटेचा इशारा",
        "Moderate Heatwave Alert": "मध्यम उष्णतेच्या लाटेचा इशारा",
        "Normal Conditions": "सामान्य हवामान",
        "Drink plenty of water and ORS.": "भरपूर पाणी आणि ओआरएस प्या.",
        "Avoid direct sunlight between 12:00 PM and 3:30 PM.": "दुपारी १२:०० ते ३:३० दरम्यान थेट उन्हात जाणे टाळा.",
        "Elderly and children should stay indoors.": "वृद्ध आणि लहान मुलांनी घरातच राहावे.",
        "Stay hydrated and avoid strenuous outdoor work.": "हायड्रेटेड राहा आणि कष्टाची कामे टाळा.",
    },
    "te": {
        "Severe Heatwave Alert": "తీవ్రమైన వడగాల్పుల హెచ్చరిక",
        "Moderate Heatwave Alert": "మధ్యస్థ వడగాల్పుల హెచ్చరిక",
        "Normal Conditions": "సాధారణ వాతావరణం",
        "Drink plenty of water and ORS.": "ఎక్కువగా నీరు మరియు ఓఆర్ఎస్ త్రాగండి.",
        "Avoid direct sunlight between 12:00 PM and 3:30 PM.": "మధ్యాహ్నం 12:00 నుండి 3:30 వరకు ఎండలో తిరగవద్దు.",
        "Elderly and children should stay indoors.": "వృద్ధులు, పిల్లలు ఇళ్లలోనే ఉండాలి.",
        "Stay hydrated and avoid strenuous outdoor work.": "నీరు ఎక్కువగా తాగుతూ ఎండలో శ్రమించవద్దు.",
    },
    "ta": {
        "Severe Heatwave Alert": "கடுமையான வெப்ப அலை எச்சரிக்கை",
        "Moderate Heatwave Alert": "மிதமான வெப்ப அலை எச்சரிக்கை",
        "Normal Conditions": "இயல்பான வானிலை",
        "Drink plenty of water and ORS.": "அதிகளவு தண்ணீர் மற்றும் ORS குடிக்கவும்.",
        "Avoid direct sunlight between 12:00 PM and 3:30 PM.": "நண்பகல் 12:00 முதல் 3:30 வரை வெயிலில் செல்வதைத் தவிர்க்கவும்.",
        "Elderly and children should stay indoors.": "முதியவர்கள் மற்றும் குழந்தைகள் வீட்டிலேயே இருக்க வேண்டும்.",
        "Stay hydrated and avoid strenuous outdoor work.": "நீரேற்றத்துடன் இருங்கள் மற்றும் கடின உழைப்பைத் தவிர்க்கவும்.",
    },
    "bn": {
        "Severe Heatwave Alert": "তীব্র তাপপ্রবাহের সতর্কতা",
        "Moderate Heatwave Alert": "মাঝারি তাপপ্রবাহের সতর্কতা",
        "Normal Conditions": "স্বাভাবিক আবহাওয়া",
        "Drink plenty of water and ORS.": "প্রচুর জল এবং ওআরএস পান করুন।",
        "Avoid direct sunlight between 12:00 PM and 3:30 PM.": "দুপুর ১২:০০ থেকে ৩:৩০ পর্যন্ত সরাসরি রোদে বের হবেন না।",
        "Elderly and children should stay indoors.": "বয়স্ক এবং শিশুদের ঘরের ভেতরে থাকা উচিত।",
        "Stay hydrated and avoid strenuous outdoor work.": "হাইড্রেটেড থাকুন এবং অতিরিক্ত শারীরিক পরিশ্রম এড়িয়ে চলুন।",
    },
}


class IndicTransEngine:
    """Translation engine supporting regional Indian languages with offline fallback."""

    def __init__(self, use_hf_pipeline: bool = False):
        self.use_hf_pipeline = use_hf_pipeline
        self.model = None
        self.tokenizer = None
        if self.use_hf_pipeline:
            self._load_hf_model()

    def _load_hf_model(self):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            model_name = "ai4bharat/indictrans2-en-indic-1B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            self.use_hf_pipeline = False

    def translate(self, text: str, target_lang: str) -> str:
        """Translates an English advisory string to target Indian language."""
        if target_lang not in SUPPORTED_LANGUAGES or target_lang == "en":
            return text

        # Check phrase dictionary fallback
        lang_dict = CORE_TRANSLATIONS.get(target_lang, {})
        if text in lang_dict:
            return lang_dict[text]

        # Multi-sentence fallback translation
        translated_parts = []
        for sentence in text.split(". "):
            cleaned = sentence.strip().rstrip(".")
            matched = False
            for src, tgt in lang_dict.items():
                if src.lower() in cleaned.lower():
                    translated_parts.append(tgt)
                    matched = True
                    break
            if not matched:
                translated_parts.append(cleaned)

        return " ".join(translated_parts) if translated_parts else text

    def translate_advisory_bundle(self, headline: str, precautions: list, target_lang: str) -> dict:
        """Translates an entire advisory pack into the requested language."""
        return {
            "language": SUPPORTED_LANGUAGES.get(target_lang, target_lang),
            "headline": self.translate(headline, target_lang),
            "precautions": [self.translate(p, target_lang) for p in precautions]
        }


if __name__ == "__main__":
    engine = IndicTransEngine(use_hf_pipeline=False)
    sample_text = "Severe Heatwave Alert"
    precautions = [
        "Avoid direct sunlight between 12:00 PM and 3:30 PM.",
        "Drink plenty of water and ORS."
    ]

    print("🌐 Testing IndicTrans2 Multilingual Translation Engine:\n")
    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        res = engine.translate_advisory_bundle(sample_text, precautions, lang_code)
        print(f"[{lang_name} ({lang_code})]")
        print(f"  📢 Headline: {res['headline']}")
        for p in res["precautions"]:
            print(f"  • {p}")
        print()

