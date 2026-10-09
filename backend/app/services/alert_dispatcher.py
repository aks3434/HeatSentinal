"""
HeatSentinel — Alert Dispatcher Service (Pragya's Module)
Provides multi-channel alert delivery (SMS & Push Notifications) with automatic
multilingual translation via IndicTransEngine and local simulation/mock mode.
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Ensure project root is in sys.path for relative engine imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from slm_engine.translation.indic_trans import IndicTransEngine, SUPPORTED_LANGUAGES
except ImportError:
    # Graceful fallback if engine path differs
    IndicTransEngine = None
    SUPPORTED_LANGUAGES = {"hi": "Hindi", "mr": "Marathi", "te": "Telugu", "ta": "Tamil", "bn": "Bengali", "en": "English"}

logger = logging.getLogger("heatsentinel.alert_dispatcher")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AlertDispatcher:
    """Multi-channel alert dispatcher supporting SMS, Push, and regional translation."""

    def __init__(self, mock_mode: Optional[bool] = None):
        # By default, use mock_mode if ALERT_MOCK_MODE is not explicitly set to 'false'
        if mock_mode is None:
            env_mock = os.getenv("ALERT_MOCK_MODE", "true").strip().lower()
            self.mock_mode = env_mock in ("true", "1", "yes")
        else:
            self.mock_mode = mock_mode

        # Initialize Translation Engine
        if IndicTransEngine:
            self.translator = IndicTransEngine(use_hf_pipeline=False)
        else:
            self.translator = None

        # Twilio credentials
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_phone = os.getenv("TWILIO_PHONE_NUMBER", "+15005550006")

        # In-memory history for local tracking and audit logs
        self._dispatch_history: List[Dict[str, Any]] = []

    def format_and_translate_alert(
        self,
        district: str,
        temp: float,
        severity: str = "Severe",
        lang: str = "en"
    ) -> str:
        """
        Generates advisory headline and life-saving precautions, then translates
        them into the citizen's preferred regional Indian language.
        """
        clean_severity = severity.capitalize() if severity else "Severe"
        headline = f"{clean_severity} Heatwave Alert for {district}!"

        # Curated clinical heatwave advisories matching IMD protocol
        if clean_severity in ("Severe", "Extreme"):
            precautions = [
                "Avoid direct sunlight between 12:00 PM and 3:30 PM.",
                "Drink plenty of water and ORS.",
                "Elderly and children should stay indoors."
            ]
        else:
            precautions = [
                "Drink plenty of water and ORS.",
                "Stay hydrated and avoid strenuous outdoor work."
            ]

        # Call Team Lead's IndicTrans engine for instant regional translation
        if self.translator and lang in SUPPORTED_LANGUAGES and lang != "en":
            try:
                bundle = self.translator.translate_advisory_bundle(headline, precautions, lang)
                translated_headline = bundle.get("headline", headline)
                translated_precautions = bundle.get("precautions", precautions)
            except Exception as e:
                logger.warning(f"Translation failed for lang={lang}: {e}. Falling back to English.")
                translated_headline = headline
                translated_precautions = precautions
        else:
            translated_headline = headline
            translated_precautions = precautions

        precaution_text = "\n".join([f"• {p}" for p in translated_precautions])

        message = (
            f"⚠️ HeatSentinel Alert: {translated_headline}\n"
            f"📍 District: {district} | Forecast: {temp:.1f}°C\n"
            f"💡 Precautions:\n"
            f"{precaution_text}\n"
            f"— HeatSentinel Emergency Cell"
        )
        return message

    def send_sms(self, to_phone: str, message: str) -> bool:
        """
        Dispatches an SMS to a citizen.
        In mock mode, prints a rich notification box to console and records the dispatch.
        """
        record = {
            "channel": "SMS",
            "recipient": to_phone,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "MOCK" if self.mock_mode else "LIVE",
            "status": "SENT"
        }

        if self.mock_mode:
            border = "=" * 60
            print(f"\n{border}")
            print(f"📱 [MOCK SMS DISPATCHED] -> {to_phone}")
            print(f"Time: {record['timestamp']}")
            print(f"{'-' * 60}")
            print(message)
            print(f"{border}\n")
            self._dispatch_history.append(record)
            return True

        # Production Twilio SDK Execution
        try:
            from twilio.rest import Client
            if not self.twilio_sid or not self.twilio_token:
                logger.error("Twilio credentials missing. Fallback to mock dispatch.")
                record["status"] = "FALLBACK_MOCK"
                self._dispatch_history.append(record)
                return False

            client = Client(self.twilio_sid, self.twilio_token)
            msg = client.messages.create(
                body=message,
                from_=self.twilio_phone,
                to=to_phone
            )
            record["twilio_sid"] = msg.sid
            record["status"] = "DELIVERED"
            self._dispatch_history.append(record)
            logger.info(f"Twilio SMS dispatched to {to_phone}: {msg.sid}")
            return True
        except Exception as e:
            logger.error(f"Failed to send SMS to {to_phone}: {e}")
            record["status"] = f"FAILED: {e}"
            self._dispatch_history.append(record)
            return False

    def send_push_notification(
        self,
        device_token: str,
        title: str,
        body: str,
        data: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Dispatches a Push Notification via Firebase Cloud Messaging (FCM).
        In mock mode, prints a notification box to console.
        """
        record = {
            "channel": "PUSH_FCM",
            "device_token": device_token,
            "title": title,
            "body": body,
            "data": data or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "MOCK" if self.mock_mode else "LIVE",
            "status": "SENT"
        }

        if self.mock_mode:
            border = "-" * 60
            print(f"\n{border}")
            print(f"🔔 [MOCK FCM PUSH DISPATCHED] -> Token: {device_token[:16]}...")
            print(f"Title: {title}")
            print(f"Body:  {body}")
            print(f"{border}\n")
            self._dispatch_history.append(record)
            return True

        # Production FCM Execution (firebase_admin)
        try:
            import firebase_admin
            from firebase_admin import messaging

            message = messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                data=data or {},
                token=device_token,
            )
            response = messaging.send(message)
            record["fcm_message_id"] = response
            record["status"] = "DELIVERED"
            self._dispatch_history.append(record)
            logger.info(f"FCM Push dispatched: {response}")
            return True
        except Exception as e:
            logger.error(f"Failed to dispatch FCM Push: {e}")
            record["status"] = f"FAILED: {e}"
            self._dispatch_history.append(record)
            return False

    def dispatch_district_alert(
        self,
        district_name: str,
        severity: str,
        temp: float,
        affected_users: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Dispatches localized alerts across multiple channels for all users in an affected district.
        Each user can have:
          - phone_number (str)
          - preferred_lang (str, e.g. 'hi', 'mr', 'te', 'en')
          - fcm_push_token (optional str)
        """
        results = {
            "district": district_name,
            "severity": severity,
            "temp": temp,
            "total_users": len(affected_users),
            "sms_sent": 0,
            "push_sent": 0,
            "failed": 0
        }

        for user in affected_users:
            phone = user.get("phone_number")
            lang = user.get("preferred_lang", "hi")
            push_token = user.get("fcm_push_token")

            # Format message translated to this user's preferred language
            translated_msg = self.format_and_translate_alert(
                district=district_name,
                temp=temp,
                severity=severity,
                lang=lang
            )

            # Send SMS
            if phone:
                success = self.send_sms(to_phone=phone, message=translated_msg)
                if success:
                    results["sms_sent"] += 1
                else:
                    results["failed"] += 1

            # Send Push if FCM token registered
            if push_token:
                title = f"⚠️ {severity} Heatwave Alert — {district_name}"
                body = f"Forecast {temp:.1f}°C. Drink water and avoid afternoon sun."
                push_ok = self.send_push_notification(
                    device_token=push_token,
                    title=title,
                    body=body,
                    data={"district": district_name, "severity": severity, "temp": str(temp)}
                )
                if push_ok:
                    results["push_sent"] += 1

        return results

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent alert dispatch logs."""
        return self._dispatch_history[-limit:]
