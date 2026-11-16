# app/main.py - FastAPI Application Entrypoint (Reload statsmodels)

import logging
import sys
import os

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.api.routes import ml, analytics, thermal, turbines, health, rul
from app.ml.model_state import load_ml_models

app = FastAPI(title="Wind Sentinel API", version="2.0.0")

# Allow all origins for seamless development/production frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to trigger model loading
@app.on_event("startup")
def startup_event():
    logger.info("Initializing Wind Sentinel Backend & Loading ML Models...")
    load_ml_models()

# Include routers with standardized prefixes
app.include_router(ml.router, prefix="/api/v1/ml", tags=["ML"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(thermal.router, prefix="/api/v1/thermal", tags=["Thermal"])
app.include_router(rul.router, prefix="/api/v1/rul", tags=["RUL"])
app.include_router(turbines.router, prefix="/api/v1/turbines", tags=["Turbines"])
app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])

@app.get("/")
async def root():
    return {"message": "Wind Sentinel API", "status": "running", "version": "2.0.0"}