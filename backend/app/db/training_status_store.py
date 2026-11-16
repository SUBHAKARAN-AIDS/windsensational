# app/db/training_status_store.py
import json
import os

STATUS_PATH = os.path.join(os.path.dirname(__file__), "training_history.json")
MAX_RUNS_KEPT = 50


def record_training_run(**run: dict):
    history = []
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, "r") as f:
            history = json.load(f)
    history.insert(0, run)
    history = history[:MAX_RUNS_KEPT]
    with open(STATUS_PATH, "w") as f:
        json.dump(history, f, indent=2)


def get_training_history(limit: int = 10) -> list:
    if not os.path.exists(STATUS_PATH):
        return []
    with open(STATUS_PATH, "r") as f:
        history = json.load(f)
    return history[:limit]