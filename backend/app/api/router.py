# app/api/router.py - Unified API Router
from fastapi import APIRouter
from app.api.routes import health, turbines, thermal, analytics, ml, model_performance, rul

api_router = APIRouter()

# Include route modules
api_router.include_router(health.router)
api_router.include_router(turbines.router, tags=["turbines"])
api_router.include_router(thermal.router, tags=["thermal"])
api_router.include_router(analytics.router, tags=["analytics"])
api_router.include_router(ml.router, tags=["ml"])
api_router.include_router(rul.router, tags=["rul"])
if hasattr(model_performance, 'router'):
    api_router.include_router(model_performance.router, tags=["performance"])