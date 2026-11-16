# app/core/lifespan.py
import logging
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
from app.ml.retrain_watcher import check_and_maybe_retrain
from app.ml.model_state import load_ml_models

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    # Startup
    logger.info("Startup: loading ML models")
    load_ml_models()

    scheduler = BackgroundScheduler()
    scheduler.add_job(check_and_maybe_retrain, "interval", minutes=30, id="retrain_watcher")
    scheduler.start()
    logger.info("Startup: retrain scheduler started")

    yield

    # Shutdown
    scheduler.shutdown(wait=False)
    logger.info("Shutdown: scheduler stopped")