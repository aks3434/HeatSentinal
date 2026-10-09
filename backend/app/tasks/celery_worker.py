"""
HeatSentinel — Celery Worker & Async Task Dispatcher (Pragya's Module)
Handles background bulk alert processing via Redis and enforces an anti-spam
frequency cap (maximum 1 alert per 12 hours per citizen).
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from celery import Celery
from backend.app.services.alert_dispatcher import AlertDispatcher

logger = logging.getLogger("heatsentinel.celery_worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Celery Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("heatsentinel_tasks", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes maximum per batch
)

# 12-Hour Anti-Spam Frequency Cap Duration in Seconds
FREQUENCY_CAP_SECONDS = 12 * 3600  # 43,200 seconds

# In-memory local cache for anti-spam tracking (used as fallback or testing without Redis)
_LOCAL_SPAM_CACHE: Dict[str, float] = {}


def is_frequency_capped(identifier: str) -> bool:
    """
    Checks if a citizen has received an alert in the last 12 hours.
    Returns True if capped (should be skipped), False if eligible for alert.
    """
    now = time.time()

    # Try Redis key lookup first if Redis client is available
    try:
        import redis
        r = redis.from_url(REDIS_URL, socket_connect_timeout=1)
        key = f"heatsentinel:spam_cap:{identifier}"
        last_sent = r.get(key)
        if last_sent is not None:
            # Key exists in Redis with active TTL
            return True
        return False
    except Exception:
        # Fallback to local in-memory cache
        last_timestamp = _LOCAL_SPAM_CACHE.get(identifier)
        if last_timestamp and (now - last_timestamp) < FREQUENCY_CAP_SECONDS:
            return True
        return False


def record_alert_dispatched(identifier: str):
    """Records the dispatch timestamp to enforce the 12-hour frequency cap."""
    now = time.time()

    # Record in Redis with 12-hour TTL
    try:
        import redis
        r = redis.from_url(REDIS_URL, socket_connect_timeout=1)
        key = f"heatsentinel:spam_cap:{identifier}"
        r.set(key, str(now), ex=FREQUENCY_CAP_SECONDS)
    except Exception:
        pass

    # Always keep in local memory cache as well
    _LOCAL_SPAM_CACHE[identifier] = now


@celery_app.task(name="tasks.send_bulk_alerts")
def send_bulk_alerts_task(
    district_code: str,
    temp: float,
    severity: str,
    district_name: Optional[str] = None,
    candidate_users: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Celery background worker task:
    1. Obtains target citizens (from candidate_users or DB query).
    2. Enforces 12-hour anti-spam frequency cap.
    3. Dispatches multilingual SMS & Push notifications via AlertDispatcher.
    """
    target_name = district_name or district_code
    dispatcher = AlertDispatcher()
    logger.info(f"🚀 Celery Worker: Processing bulk alerts for {target_name} ({district_code})")

    users_to_alert = []

    # 1. Gather users
    if candidate_users:
        users_to_alert = candidate_users
    else:
        # Attempt to query Krishan's Database models if available
        try:
            from backend.app.db.session import SessionLocal
            from backend.app.db.models import User
            db = SessionLocal()
            try:
                # Query users registered for this district
                db_users = db.query(User).filter(User.district_code == district_code).all()
                users_to_alert = [
                    {
                        "id": str(u.id),
                        "phone_number": u.phone_number,
                        "preferred_lang": getattr(u, "preferred_lang", "hi"),
                        "fcm_push_token": getattr(u, "fcm_push_token", None)
                    }
                    for u in db_users
                ]
            finally:
                db.close()
        except Exception:
            # Fallback mock users for standalone testing
            users_to_alert = [
                {"phone_number": "+919876543210", "preferred_lang": "hi", "district": target_name},
                {"phone_number": "+919823456789", "preferred_lang": "mr", "district": target_name},
                {"phone_number": "+919845678901", "preferred_lang": "te", "district": target_name},
            ]

    # 2. Filter out frequency-capped users (Max 1 alert per 12 hours)
    eligible_users = []
    skipped_count = 0

    for user in users_to_alert:
        phone = user.get("phone_number", "")
        if is_frequency_capped(phone):
            logger.info(f"⏭️ Skipping {phone} — Anti-spam cap active (alert sent <12h ago).")
            skipped_count += 1
        else:
            eligible_users.append(user)

    # 3. Dispatch to eligible citizens
    dispatch_results = dispatcher.dispatch_district_alert(
        district_name=target_name,
        severity=severity,
        temp=temp,
        affected_users=eligible_users
    )

    # 4. Mark dispatches to update the 12-hour frequency cap
    for user in eligible_users:
        phone = user.get("phone_number")
        if phone:
            record_alert_dispatched(phone)

    dispatch_results["skipped_frequency_cap"] = skipped_count
    logger.info(f"✅ Celery Worker: Bulk dispatch complete for {target_name}. Stats: {dispatch_results}")
    return dispatch_results
