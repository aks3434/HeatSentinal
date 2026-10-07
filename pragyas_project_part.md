# 🔔 HeatSentinel — Alert & Notification System Guide (Pragya's Module)

> **Assigned Developer**: Pragya  
> **Module**: Real-time SMS, Push Notifications & Celery Async Dispatcher  
> **Status of Brain/ML/API (Team Lead)**: ✅ 100% Complete & Running on `http://127.0.0.1:8000`

---

## 🛑 1. Git & Conflict-Prevention Rules (CRITICAL)

To prevent merge conflicts with the completed ML Brain and Database modules:

1. **Your Workspace (Files you own and edit)**:
   - `backend/app/services/alert_dispatcher.py` (SMS & Push notification sending service)
   - `backend/app/tasks/celery_worker.py` (Celery background task queue + Redis)
   - `backend/app/api/v1/alerts.py` (FastAPI router for user subscription & manual trigger)

2. **Protected Files (DO NOT modify or overwrite)**:
   - ❌ `ml_engine/` — All ML models (XGBoost, Bi-LSTM, Stacking Ensemble) are finalized.
   - ❌ `slm_engine/` — Fine-tuning corpus, translation, and inference are finalized (you will *import* them, but not edit them).
   - ❌ `data/` — Contains trained weights and datasets.
   - ❌ `backend/app/api/v1/predict.py`, `explain.py`, `chat.py` — Lead's core APIs.
   - ❌ `backend/app/db/` — Owned by Krishan (coordinate with him for DB models).

---

## 🎯 2. What Pragya Needs to Build

Your module is responsible for the citizen-facing safety net:
1. **Multi-Channel Alert Dispatcher**:
   - **SMS**: Twilio or MSG91 (with an automatic Mock/Simulation mode for free local testing).
   - **Push Notifications**: Firebase Cloud Messaging (FCM).
2. **Multilingual SMS Translation**:
   - Seamlessly call Team Lead's `IndicTransEngine` so a citizen in Jaipur gets Hindi, Nagpur gets Marathi, and Hyderabad gets Telugu.
3. **Async Task Worker (Celery + Redis)**:
   - Run alert dispatches in the background so API requests don't freeze or time out.
4. **Anti-Spam Frequency Cap**:
   - Ensure no citizen receives more than **1 alert per 12 hours**.
5. **Subscription API Endpoints**:
   - Allow citizens to subscribe via phone number, pin code, and preferred language.

---

## 🏗️ 3. Technical Specification per File

### File A: `backend/app/services/alert_dispatcher.py`
This is your core dispatch service. It must support both **Live Mode** (Twilio/Firebase) and **Mock Mode** (prints formatted SMS alerts to terminal so you can test freely).

```python
"""
Key responsibilities of AlertDispatcher:
1. send_sms(phone_number, message)
2. send_push_notification(device_token, title, body)
3. dispatch_district_alert(district_name, severity, temp, affected_users)
"""
import os
from slm_engine.translation.indic_trans import IndicTransEngine

class AlertDispatcher:
    def __init__(self, mock_mode: bool = True):
        self.mock_mode = mock_mode
        self.translator = IndicTransEngine()
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")

    def format_and_translate_alert(self, district: str, temp: float, severity: str, lang: str = "en") -> str:
        headline = f"{severity} Heatwave Alert for {district}!"
        precautions = [
            "Avoid direct sunlight between 12:00 PM and 3:30 PM.",
            "Drink plenty of water and ORS."
        ]
        
        # Call Team Lead's IndicTrans engine for instant regional translation
        bundle = self.translator.translate_advisory_bundle(headline, precautions, lang)
        
        message = (
            f"⚠️ HeatSentinel Alert: {bundle['headline']}\n"
            f"Forecast: {temp:.1f}°C in {district}.\n"
            f"Precautions:\n"
            + "\n".join([f"• {p}" for p in bundle['precautions']])
        )
        return message

    def send_sms(self, to_phone: str, message: str) -> bool:
        if self.mock_mode:
            print(f"\n📱 [MOCK SMS DISPATCHED to {to_phone}]:\n{message}\n")
            return True
        # Production Twilio SDK call here
        return True



File B: backend/app/tasks/celery_worker.py
Sets up Celery with a Redis broker to execute dispatches asynchronously:

python
"""
Key responsibilities of Celery Worker:
1. Initialize Celery app connected to Redis (redis://localhost:6379/0).
2. Define async task: send_bulk_district_alerts.delay(district_code, severity, temp)
3. Prevent alert spam (max 1 alert every 12 hours per citizen).
"""
from celery import Celery
import os
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("heatsentinel_tasks", broker=REDIS_URL, backend=REDIS_URL)
@celery_app.task(name="tasks.send_bulk_alerts")
def send_bulk_alerts_task(district_code: str, temp: float, severity: str):
    # 1. Query Krishan's User table for users in this district
    # 2. Filter out users who received an alert in the last 12 hours
    # 3. Dispatch SMS/Push via AlertDispatcher
    print(f"🚀 Celery Worker: Processing bulk alerts for {district_code}")
    return {"status": "success", "district": district_code}
File C: backend/app/api/v1/alerts.py
The FastAPI router for user subscriptions and testing alerts:

python
"""
Endpoints to implement in backend/app/api/v1/alerts.py:
1. POST /api/v1/alerts/subscribe
   - Body: { phone_number, pincode, preferred_lang, lat, lon }
   - Stores subscriber in DB.
2. POST /api/v1/alerts/trigger-test
   - Body: { district_name, temp, severity, lang, test_phone }
   - Instantly triggers a test SMS/Mock alert to test the end-to-end flow!
3. GET /api/v1/alerts/history
   - Returns recent alert dispatch logs.
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
router = APIRouter(prefix="/alerts", tags=["Alert & Notification System"])
class AlertSubscriptionRequest(BaseModel):
    phone_number: str = Field(..., example="+919876543210")
    pincode: str = Field(..., example="302001")
    preferred_lang: str = Field(default="hi", example="hi")
class TestAlertTrigger(BaseModel):
    district_name: str = Field(default="Jaipur", example="Jaipur")
    temp: float = Field(default=44.5, example=44.5)
    severity: str = Field(default="Severe", example="Severe")
    preferred_lang: str = Field(default="hi", example="hi")
    phone_number: str = Field(default="+919876543210", example="+919876543210")
🧪 4. How Pragya Tests Locally (No Twilio Billing Needed)
Set mock_mode = True in AlertDispatcher.
When testing via Swagger (/docs) or terminal, all generated multilingual alerts will print cleanly to the console showing the exact message, language translation, and destination phone number.
If testing with Celery:
Run Redis locally or via Docker: docker run -d -p 6379:6379 redis
Start the Celery worker: celery -A backend.app.tasks.celery_worker.celery_app worker --loglevel=info
🔗 5. Integration with the Team Lead
Once your 3 files (alert_dispatcher.py, celery_worker.py, and alerts.py) are implemented:

Inform the Team Lead.
The Team Lead will mount your alerts_router into backend/app/main.py.
Your endpoints will appear immediately on the live Swagger documentation (http://127.0.0.1:8000/docs).