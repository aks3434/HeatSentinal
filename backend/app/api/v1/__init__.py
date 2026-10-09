"""API v1 Package for HeatSentinel Backend."""
from .alerts import router as alerts_router

__all__ = ["alerts_router"]
