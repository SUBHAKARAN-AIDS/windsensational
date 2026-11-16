# Legacy / quarantined code

These modules are **dead-as-wired**: nothing in the live application imports or
routes to them. They are kept intact for the later **ML consolidation phase**.

- `ml_routes.py` — an `APIRouter` that duplicates the live ensemble/status
  endpoints implemented in `app/api/routes/`. It is **not** included by
  `app/api/router.py`. Import lines were repointed to new module locations so
  the file stays importable; its logic is unchanged.
- `ensemble_service.py` — `EnsembleAnomalyDetector`; used only by `ml_routes.py`.
- `anomaly_service.py` — `AnomalyDetectionService`; imported by nothing.
- `model_loader.py` — `ModelLoader`; used only by `ensemble_service.py`. Retains
  a CWD-relative `model_dir="app/saved_models"` default (harmless while dead;
  to be fixed if/when this cluster is revived).

Do **not** import from this package in live code. The live ML entry point is
`app/ml/model_state.py` (loaded once at startup via `app/core/lifespan.py`).
