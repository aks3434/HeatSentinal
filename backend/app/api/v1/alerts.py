"""
HeatSentinel — Alert & Notification API Router (Pragya's Module)
Provides REST endpoints for citizen subscriptions, instant alert testing,
bulk dispatch triggering, and history inspection.
"""

import os
import sys
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, status
from pydantic import BaseModel, Field

# Ensure root path is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.services.alert_dispatcher import AlertDispatcher
from backend.app.tasks.celery_worker import (
    send_bulk_alerts_task,
    is_frequency_capped,
    record_alert_dispatched,
    celery_app
)

logger = logging.getLogger("heatsentinel.alerts_api")

router = APIRouter(prefix="/alerts", tags=["Alert & Notification System (Pragya)"])

# Shared singleton dispatcher instance
dispatcher = AlertDispatcher()

# In-memory storage for subscriptions (fallback when PostgreSQL is not connected)
_SUBSCRIBERS_STORE: Dict[str, Dict[str, Any]] = {}


# --- Request & Response Models ---

class AlertSubscriptionRequest(BaseModel):
    phone_number: str = Field(..., example="+919876543210", description="Indian E.164 phone number")
    pincode: str = Field(..., example="302001", description="6-digit Indian postal pincode")
    preferred_lang: str = Field(default="hi", example="hi", description="Regional language code: hi, mr, te, ta, bn, en")
    lat: Optional[float] = Field(default=None, example=26.9124, description="Citizen Latitude coordinate")
    lon: Optional[float] = Field(default=None, example=75.7873, description="Citizen Longitude coordinate")
    fcm_push_token: Optional[str] = Field(default=None, example="fcm_token_sample_abc123", description="Mobile Push token")


class AlertSubscriptionResponse(BaseModel):
    status: str
    message: str
    subscriber_id: str
    phone_number: str
    preferred_lang: str
    pincode: str


class TestAlertTrigger(BaseModel):
    district_name: str = Field(default="Jaipur", example="Jaipur")
    temp: float = Field(default=44.5, example=44.5, description="Forecast temperature in Celsius")
    severity: str = Field(default="Severe", example="Severe", description="Normal, Moderate, Severe, Extreme")
    preferred_lang: str = Field(default="hi", example="hi", description="Language code (hi, mr, te, ta, bn, en)")
    phone_number: str = Field(default="+919876543210", example="+919876543210")
    bypass_frequency_cap: bool = Field(default=True, description="Whether to bypass the 12-hour anti-spam cap for testing")


class TestAlertResponse(BaseModel):
    status: str
    recipient: str
    language: str
    message_text: str
    sms_dispatched: bool
    mock_mode: bool
    note: str


class BulkAlertRequest(BaseModel):
    district_code: str = Field(default="RJ01", example="RJ01")
    district_name: str = Field(default="Jaipur", example="Jaipur")
    temp: float = Field(default=45.0, example=45.0)
    severity: str = Field(default="Severe", example="Severe")
    use_celery_broker: bool = Field(default=False, description="Send via Redis Celery queue if running, or synchronous background task")


# --- Endpoints ---

@router.post(
    "/subscribe",
    response_model=AlertSubscriptionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Subscribe a citizen to localized heatwave alerts"
)
def subscribe_citizen(payload: AlertSubscriptionRequest):
    """
    Subscribes a citizen with their phone number, pincode, and preferred regional language.
    Persists to database or local storage for alert targeting.
    """
    # Clean phone number
    phone = payload.phone_number.strip()
    if not phone.startswith("+"):
        phone = f"+91{phone.lstrip('0')}" if len(phone) == 10 else f"+{phone}"

    sub_id = str(uuid.uuid4())
    user_record = {
        "id": sub_id,
        "phone_number": phone,
        "pincode": payload.pincode.strip(),
        "preferred_lang": payload.preferred_lang.lower().strip(),
        "lat": payload.lat,
        "lon": payload.lon,
        "fcm_push_token": payload.fcm_push_token,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    # 1. Attempt PostgreSQL persistence if Krishan's DB session is available
    db_saved = False
    try:
        from backend.app.db.session import SessionLocal
        from backend.app.db.models import User
        db = SessionLocal()
        try:
            existing = db.query(User).filter(User.phone_number == phone).first()
            if existing:
                existing.pincode = payload.pincode
                existing.preferred_lang = payload.preferred_lang
                existing.fcm_push_token = payload.fcm_push_token
            else:
                new_user = User(
                    phone_number=phone,
                    pincode=payload.pincode,
                    preferred_lang=payload.preferred_lang,
                    fcm_push_token=payload.fcm_push_token
                )
                db.add(new_user)
            db.commit()
            db_saved = True
        finally:
            db.close()
    except Exception:
        # DB not initialized yet - save in local memory cache
        pass

    # In-memory save fallback
    _SUBSCRIBERS_STORE[phone] = user_record

    return AlertSubscriptionResponse(
        status="success",
        message="Citizen successfully subscribed to HeatSentinel alerts.",
        subscriber_id=sub_id,
        phone_number=phone,
        preferred_lang=payload.preferred_lang,
        pincode=payload.pincode
    )


@router.post(
    "/trigger-test",
    response_model=TestAlertResponse,
    summary="Instantly trigger a test SMS/Mock alert (for verification)"
)
def trigger_test_alert(payload: TestAlertTrigger):
    """
    Instantly formats, translates, and dispatches an alert to verify the end-to-end pipeline.
    Shows the exact translated text and mock output in your terminal.
    """
    phone = payload.phone_number.strip()

    # Frequency cap check (unless bypassed for testing)
    if not payload.bypass_frequency_cap and is_frequency_capped(phone):
        raise HTTPException(
            status_code=429,
            detail=f"Frequency cap active: Alert already dispatched to {phone} within the last 12 hours."
        )

    # 1. Format and translate message using IndicTransEngine
    message = dispatcher.format_and_translate_alert(
        district=payload.district_name,
        temp=payload.temp,
        severity=payload.severity,
        lang=payload.preferred_lang
    )

    # 2. Dispatch via SMS / Mock
    sms_ok = dispatcher.send_sms(to_phone=phone, message=message)

    # 3. Update anti-spam record
    record_alert_dispatched(phone)

    return TestAlertResponse(
        status="dispatched" if sms_ok else "failed",
        recipient=phone,
        language=payload.preferred_lang,
        message_text=message,
        sms_dispatched=sms_ok,
        mock_mode=dispatcher.mock_mode,
        note="If mock_mode is True, verify the formatted alert in your terminal stdout."
    )


@router.post(
    "/dispatch-bulk",
    summary="Trigger bulk alerts for a district (Async via Celery or Background Task)"
)
def dispatch_bulk_alerts(payload: BulkAlertRequest, background_tasks: BackgroundTasks):
    """
    Triggers bulk alerts for all subscribers in a district.
    Can execute via Redis-backed Celery worker or FastAPI BackgroundTasks.
    """
    # Collect candidate subscribers from memory if DB is not present
    matched_users = [
        u for u in _SUBSCRIBERS_STORE.values()
    ]
    if not matched_users:
        # Provide default test targets if no users subscribed yet
        matched_users = [
            {"phone_number": "+919876543210", "preferred_lang": "hi"},
            {"phone_number": "+919823456789", "preferred_lang": "mr"},
            {"phone_number": "+919845678901", "preferred_lang": "te"},
        ]

    if payload.use_celery_broker:
        try:
            task = send_bulk_alerts_task.delay(
                district_code=payload.district_code,
                temp=payload.temp,
                severity=payload.severity,
                district_name=payload.district_name,
                candidate_users=matched_users
            )
            return {
                "status": "queued_in_celery",
                "task_id": str(task.id),
                "district": payload.district_name,
                "target_subscribers": len(matched_users)
            }
        except Exception as e:
            logger.warning(f"Celery broker unavailable ({e}), running via BackgroundTasks.")

    # Fallback / Direct Background execution
    background_tasks.add_task(
        send_bulk_alerts_task,
        district_code=payload.district_code,
        temp=payload.temp,
        severity=payload.severity,
        district_name=payload.district_name,
        candidate_users=matched_users
    )

    return {
        "status": "queued_in_background",
        "district": payload.district_name,
        "target_subscribers": len(matched_users),
        "note": "Alerts dispatched in the background. Check logs or /alerts/history."
    }


@router.get(
    "/history",
    summary="Get recent alert dispatch audit history"
)
def get_alert_history(limit: int = Query(default=20, ge=1, le=100)):
    """
    Returns audit logs of recently dispatched alerts, including recipient,
    channel, message preview, timestamp, and delivery status.
    """
    history = dispatcher.get_history(limit=limit)
    return {
        "total_returned": len(history),
        "history": list(reversed(history))
    }


@router.get(
    "/subscribers",
    summary="List currently registered alert subscribers (In-memory fallback view)"
)
def list_subscribers():
    """Lists currently registered citizen subscribers."""
    return {
        "total_subscribers": len(_SUBSCRIBERS_STORE),
        "subscribers": list(_SUBSCRIBERS_STORE.values())
    }
