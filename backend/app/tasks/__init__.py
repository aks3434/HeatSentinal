"""Celery background tasks package for HeatSentinel."""
from .celery_worker import celery_app, send_bulk_alerts_task

__all__ = ["celery_app", "send_bulk_alerts_task"]
