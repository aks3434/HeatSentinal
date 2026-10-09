"""
HeatSentinel — Main FastAPI Application Entrypoint
Mounts routers for ML predictions, SLM chat, database features, and Pragya's Alert System.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure root directory is on Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.api.v1.alerts import router as alerts_router

app = FastAPI(
    title="HeatSentinel API",
    description="Early-warning system for extreme heatwaves across India with multi-channel multilingual notifications.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local React/Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Pragya's Alert and Notification router
app.include_router(alerts_router, prefix="/api/v1")


@app.get("/", tags=["Health"])
def health_check():
    return {
        "service": "HeatSentinel Core API",
        "status": "online",
        "docs": "/docs",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
