# app/ml/retrain_watcher.py
#
# Decides WHEN retraining is recommended and makes sure only ONE training job
# runs at a time (whether it was started by the button or by anything else).
#
# This file never starts training by itself. It only:
#   1. counts the rows in the SCADA table (cached, so the database is not hit every few seconds)
#   2. compares with the row count at the last successful training
#   3. says "retrain recommended" when enough new rows have arrived
# A person still has to press the Retrain button.

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

from app.db.connection import get_db_connection

logger = logging.getLogger(__name__)

STATE_PATH = os.path.join(os.path.dirname(__file__), "retrain_state.json")

# Retraining is RECOMMENDED once this many NEW rows have arrived since the last training.
# Change this number to suit how fast your SCADA data arrives.
ROW_THRESHOLD = 50000

CHECK_TABLE = "site_dzh.scada_raw"   # change to your actual table name
COUNT_CACHE_SECONDS = 600            # re-count rows at most once every 10 minutes

# ── ONE shared "busy" sign for every way of starting training ──────────
_state_lock = threading.Lock()
_training_running = False

_count_lock = threading.Lock()
_count_cache = {"value": None, "at": 0.0}


def _load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {"last_trained_row_count": 0, "last_trained_at": None}
    with open(STATE_PATH, "r") as f:
        return json.load(f)


def _save_state(state: dict):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def get_current_row_count(max_age_seconds: int = COUNT_CACHE_SECONDS) -> int:
    """Number of rows in the SCADA table. Uses a recent count if one exists
    (pass max_age_seconds=0 to force a fresh count)."""
    with _count_lock:
        age = time.time() - _count_cache["at"]
        if _count_cache["value"] is not None and age < max_age_seconds:
            return _count_cache["value"]

        conn = get_db_connection()
        if conn is None:
            raise RuntimeError("Could not connect to database for row count check")
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {CHECK_TABLE}")
                value = int(cur.fetchone()[0])
        finally:
            conn.close()

        _count_cache["value"] = value
        _count_cache["at"] = time.time()
        return value


def mark_trained(row_count: int):
    _save_state({
        "last_trained_row_count": row_count,
        "last_trained_at": datetime.now(timezone.utc).isoformat(),
    })


def get_retrain_advice() -> dict:
    """Is a retrain recommended? Never raises: if the database cannot be reached,
    it simply says 'not recommended' with current_row_count = None."""
    try:
        state = _load_state() or {}
    except Exception as e:
        logger.warning(f"Could not read training state: {e}")
        state = {}
    last_count = state.get("last_trained_row_count") or 0

    try:
        current = get_current_row_count()
    except Exception as e:
        logger.warning(f"Could not count rows: {e}")
        return {"current_row_count": None, "new_rows": None,
                "row_threshold": ROW_THRESHOLD, "retrain_recommended": False}

    new_rows = max(0, current - last_count)
    return {
        "current_row_count": current,
        "new_rows": new_rows,
        "row_threshold": ROW_THRESHOLD,
        "retrain_recommended": new_rows >= ROW_THRESHOLD,
    }


def check_and_maybe_retrain():
    """Kept for old callers. It no longer trains anything: it only returns the
    advice. Training is started by a person through start_training()."""
    return get_retrain_advice()


def is_training() -> bool:
    return _training_running


def start_training(trigger: str = "manual") -> bool:
    """Start a training job in the background.
    Returns True if it started, False if one is already running."""
    global _training_running
    with _state_lock:
        if _training_running:
            return False
        _training_running = True

    def _worker():
        global _training_running
        try:
            try:
                row_count = get_current_row_count(0)   # fresh count at the moment of training
            except Exception as e:
                logger.warning(f"Could not get row count before training: {e}")
                row_count = None
            from app.ml.train_runner import run_training_job  # local import avoids circular import
            run_training_job(trigger=trigger, row_count_at_trigger=row_count)
        except Exception as e:
            logger.error(f"Training worker crashed: {e}")
        finally:
            with _state_lock:
                _training_running = False

    threading.Thread(target=_worker, daemon=True).start()
    return True