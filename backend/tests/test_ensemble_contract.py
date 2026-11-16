"""Contract-regression tests for the ensemble inference logic.

These assert the *truthful, deterministic* model contract:
  - Voters are ONLY models with a trained absolute boundary (iso/ee/xgb).
  - LSTM is score-only (advisory) and never casts a vote.
  - VAE is excluded until its 3-feature mapping is recovered.
  - Count fields are internally consistent (ensemble_votes == vote_count ==
    len(voters_detected) <= total_voters).
  - Model load/inference failures are surfaced, never silently emptied.
  - `days` is anchored to the latest available data timestamp.

They require the same dependencies as the app (numpy/sklearn/xgboost, plus
torch/tensorflow for the import surface). They mock the database, so no live
PostgreSQL is needed.
"""
import asyncio
import numpy as np
import pytest

import app.api.routes.ml as mlmod
import app.db.data_mode as dm
from app.ml.model_state import load_ml_models, ML_MODELS
from app.core.config import (
    VOTING_MODELS, SCORE_ONLY_MODELS, MAIN_FEATURES, AE_FEATURES,
    SCADA_TABLE,
)

# Column order the ensemble SELECT returns (base 23 + appended q + oil_inlet).
_SELECT_WIDTH = 25


def _synth_rows(n, turbine="DZH1", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        r = [f"2026-02-{(i % 28) + 1:02d} 00:00:00", turbine,
             rng.normal(9, 2), rng.normal(3900, 1500), rng.normal(1500, 100),
             rng.normal(15, 2), rng.normal(500, 50),
             rng.normal(2, 1), rng.normal(2, 1), rng.normal(2, 1),
             rng.normal(58, 4), rng.normal(60, 4), rng.normal(60, 4), rng.normal(55, 4),
             rng.normal(90, 5), rng.normal(90, 5), rng.normal(90, 5),
             rng.normal(70, 5), rng.normal(70, 5),
             rng.normal(25, 5), rng.normal(30, 5),
             rng.normal(1, 0.3), rng.normal(1, 0.3),
             rng.normal(-8, 12), rng.normal(51, 3.7)]
        assert len(r) == _SELECT_WIDTH
        rows.append(tuple(r))
    return rows


class _Cur:
    def __init__(self, rows):
        self._rows = rows
    def execute(self, q, p=None):
        self._q = q
    def fetchall(self):
        return self._rows
    def fetchone(self):
        return (0,)          # check_data_mode count -> historical
    def close(self):
        pass


class _Conn:
    def __init__(self, rows):
        self._rows = rows
    def cursor(self):
        return _Cur(self._rows)
    def close(self):
        pass


@pytest.fixture(autouse=True)
def _loaded_models():
    load_ml_models()
    assert ML_MODELS["loaded"], "sklearn artifacts must load"
    yield


def _patch_db(monkeypatch, rows):
    monkeypatch.setattr(mlmod, "get_db_connection", lambda: _Conn(rows))
    monkeypatch.setattr(dm, "get_db_connection", lambda: _Conn(rows))


def _call(**kw):
    return asyncio.run(mlmod.get_ensemble_anomaly_detection(**kw))


# --------------------------------------------------------------------------
# rules 7, 8, 9 — dimension/contract validation
# --------------------------------------------------------------------------
def test_scaler_matches_config_26():
    v = ML_MODELS["validation"]["scaler"]
    assert v["valid"], v
    assert list(ML_MODELS["scaler"].feature_names_in_) == list(MAIN_FEATURES)


def test_scaler_ae_matches_lstm_contract_18():
    v = ML_MODELS["validation"]["scaler_ae"]
    assert v["valid"], v
    assert list(ML_MODELS["scaler_ae"].feature_names_in_) == list(AE_FEATURES)


def test_voter_models_are_26_features():
    for m in ("isolation_forest", "elliptic_envelope", "xgboost"):
        assert ML_MODELS["validation"][m]["valid"], m


# --------------------------------------------------------------------------
# rules 1-4, 11, 13 — voters only, internal consistency, participation
# --------------------------------------------------------------------------
def test_only_iso_ee_xgb_vote(monkeypatch):
    _patch_db(monkeypatch, _synth_rows(60, "DZH1", seed=1))
    res = _call(turbine_id="DZH1", days=1, limit=20, min_score=0.0)
    assert res["total_voters"] == 3
    assert set(res["voting_models"]) == {"Isolation Forest", "Elliptic Envelope", "XGBoost Classifier"}
    assert "LSTM Autoencoder" not in res["voting_models"]


def test_no_ambiguous_count_keys(monkeypatch):
    _patch_db(monkeypatch, _synth_rows(60, "DZH1", seed=2))
    res = _call(turbine_id="DZH1", days=1, limit=20, min_score=0.0)
    assert "total_models" not in res
    assert "models_used" not in res
    for a in res["anomalies"]:
        assert "total_models" not in a and "model_count" not in a


def test_per_anomaly_vote_consistency(monkeypatch):
    _patch_db(monkeypatch, _synth_rows(60, "DZH1", seed=3))
    res = _call(turbine_id="DZH1", days=1, limit=50, min_score=0.0)
    for a in res["anomalies"]:
        assert a["ensemble_votes"] == a["vote_count"] == len(a["voters_detected"])
        assert a["ensemble_votes"] <= a["total_voters"] == 3


def test_participation_metadata_present(monkeypatch):
    _patch_db(monkeypatch, _synth_rows(40, "DZH1", seed=4))
    res = _call(turbine_id="DZH1", days=7, limit=20, min_score=0.0)
    for k in ("voting_models", "score_only_models", "excluded_models",
              "failed_models", "total_voters", "ensemble_vote_threshold",
              "lookback_days"):
        assert k in res
    assert res["lookback_days"] == 7


# --------------------------------------------------------------------------
# rules 2, 3 — LSTM score-only never inflates the vote
# --------------------------------------------------------------------------
def test_lstm_is_score_only_not_voting(monkeypatch):
    class _FakeLSTM:
        def predict(self, X, verbose=0):
            recon = X.copy()
            recon[0] += 5.0          # make sequence 0 anomalous
            return recon
    monkeypatch.setitem(ML_MODELS, "lstm_autoencoder", _FakeLSTM())
    ML_MODELS["validation"]["lstm_autoencoder"] = {"valid": True, "detail": "injected"}
    ML_MODELS["validation"]["scaler_ae"] = {"valid": True, "detail": "ok"}
    _patch_db(monkeypatch, _synth_rows(60, "DZH1", seed=5))
    res = _call(turbine_id="DZH1", days=7, limit=50, min_score=0.0)
    assert "LSTM Autoencoder" in res["score_only_models"]
    assert "LSTM Autoencoder" not in res["voting_models"]
    for a in res["anomalies"]:
        assert a["ensemble_votes"] <= a["total_voters"] == 3


# --------------------------------------------------------------------------
# rule 14 — multiple turbines
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tid", ["DZH1", "DZH07", "DZH23"])
def test_multiple_turbines(monkeypatch, tid):
    _patch_db(monkeypatch, _synth_rows(40, tid, seed=hash(tid) % 100))
    res = _call(turbine_id=tid, days=1, limit=20, min_score=0.0)
    assert res["total_voters"] == 3
    for a in res["anomalies"]:
        assert a["turbine_id"] == tid


# --------------------------------------------------------------------------
# rule 10 — failures surfaced, not silently emptied
# --------------------------------------------------------------------------
def test_db_failure_surfaces_error(monkeypatch):
    monkeypatch.setattr(mlmod, "get_db_connection", lambda: None)
    monkeypatch.setattr(dm, "get_db_connection", lambda: None)
    res = _call(turbine_id="DZH1", days=1, limit=20, min_score=0.0)
    assert res.get("error")            # explicit error, not a clean empty
    assert res["anomalies"] == []


def test_models_not_loaded_surfaces_error(monkeypatch):
    _patch_db(monkeypatch, _synth_rows(30, "DZH1", seed=7))
    monkeypatch.setitem(ML_MODELS, "loaded", False)
    res = _call(turbine_id="DZH1", days=1, limit=20, min_score=0.0)
    assert res.get("error") == "Models not loaded"


# --------------------------------------------------------------------------
# rule 12 — days anchored to latest data timestamp
# --------------------------------------------------------------------------
def test_days_anchored_to_max_time():
    cond = dm.get_date_condition("historical", days=3)
    assert "MAX(\"time\")" in cond
    assert SCADA_TABLE in cond
    assert "NOW()" not in cond
