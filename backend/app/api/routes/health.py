# app/api/routes/health.py
# Root endpoint moved VERBATIM from main.py. Path preserved.

from fastapi import APIRouter

from app.db.data_mode import check_data_mode
from app.ml.model_state import ML_MODELS

router = APIRouter()


@router.get("/")
def root():
    return {
        "status": "online", 
        "system": "Wind Analytics Engine v1.0",
        "ml_models_loaded": ML_MODELS["loaded"],
        "data_mode": check_data_mode(),
        "endpoints": {
            "turbines": "/api/v1/turbines",
            "analytics": "/api/v1/analytics",
            "ml": "/api/v1/ml"
        }
    }
