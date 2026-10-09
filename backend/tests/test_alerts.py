"""
Unit and Integration Tests for Pragya's Alert & Notification Module.
"""

import sys
import os
import unittest
from fastapi.testclient import TestClient

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.services.alert_dispatcher import AlertDispatcher
from backend.app.tasks.celery_worker import (
    is_frequency_capped,
    record_alert_dispatched,
    send_bulk_alerts_task,
    _LOCAL_SPAM_CACHE
)
from backend.app.main import app


class TestAlertDispatcher(unittest.TestCase):
    def setUp(self):
        self.dispatcher = AlertDispatcher(mock_mode=True)

    def test_hindi_translation(self):
        msg = self.dispatcher.format_and_translate_alert("Jaipur", 44.5, "Severe", lang="hi")
        self.assertIn("गंभीर लू", msg)
        self.assertIn("Jaipur", msg)
        self.assertIn("44.5°C", msg)

    def test_marathi_translation(self):
        msg = self.dispatcher.format_and_translate_alert("Nagpur", 43.8, "Severe", lang="mr")
        self.assertIn("उष्णतेच्या लाटेचा इशारा", msg)
        self.assertIn("Nagpur", msg)

    def test_telugu_translation(self):
        msg = self.dispatcher.format_and_translate_alert("Hyderabad", 42.0, "Severe", lang="te")
        self.assertIn("వడగాల్పుల హెచ్చరిక", msg)
        self.assertIn("Hyderabad", msg)

    def test_send_sms_mock(self):
        success = self.dispatcher.send_sms("+919876543210", "Test Emergency Alert")
        self.assertTrue(success)
        history = self.dispatcher.get_history()
        self.assertTrue(any(h["recipient"] == "+919876543210" for h in history))

    def test_send_push_notification_mock(self):
        success = self.dispatcher.send_push_notification(
            device_token="sample_token_12345",
            title="Heatwave Alert",
            body="High temp warning"
        )
        self.assertTrue(success)


class TestAntiSpamAndCelery(unittest.TestCase):
    def setUp(self):
        _LOCAL_SPAM_CACHE.clear()

    def test_frequency_cap_enforcement(self):
        phone = "+919999988888"
        self.assertFalse(is_frequency_capped(phone))
        record_alert_dispatched(phone)
        self.assertTrue(is_frequency_capped(phone))

    def test_bulk_dispatch_task_skips_capped_users(self):
        phone = "+919111122222"
        record_alert_dispatched(phone)

        users = [
            {"phone_number": phone, "preferred_lang": "hi"},
            {"phone_number": "+919333344444", "preferred_lang": "mr"}
        ]
        result = send_bulk_alerts_task(
            district_code="RJ01",
            temp=44.0,
            severity="Severe",
            district_name="Jaipur",
            candidate_users=users
        )
        self.assertEqual(result["skipped_frequency_cap"], 1)
        self.assertEqual(result["sms_sent"], 1)


class TestAlertsAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_subscribe_endpoint(self):
        res = self.client.post("/api/v1/alerts/subscribe", json={
            "phone_number": "+919876543210",
            "pincode": "302001",
            "preferred_lang": "hi"
        })
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["preferred_lang"], "hi")

    def test_trigger_test_endpoint(self):
        res = self.client.post("/api/v1/alerts/trigger-test", json={
            "district_name": "Jaipur",
            "temp": 45.5,
            "severity": "Severe",
            "preferred_lang": "hi",
            "phone_number": "+919876543210",
            "bypass_frequency_cap": True
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "dispatched")
        self.assertIn("गंभीर लू", data["message_text"])

    def test_history_endpoint(self):
        res = self.client.get("/api/v1/alerts/history?limit=5")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("history", data)


if __name__ == "__main__":
    unittest.main()
