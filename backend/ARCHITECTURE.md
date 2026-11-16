# Wind Sentinel — Project Architecture

## 1. Scope and system shape

Wind Sentinel is a full-stack wind-turbine monitoring and analytics application. It has two independently started applications:

- **Frontend:** React single-page application built with Vite. It provides fleet and turbine dashboards for telemetry, anomalies, thermal condition, dependencies, remaining useful life (RUL), and model views.
- **Backend:** Python API service built with FastAPI. It queries PostgreSQL SCADA and fault-event data, calculates analytics and health indicators, and runs inference using local model artifacts.
- **Persistence:** PostgreSQL is the source for SCADA and fault-event records. A small number of backend operational histories are stored as local JSON files.
- **ML artifacts:** serialized estimators, scalers, and neural-network models are stored under `backend/saved_models/`.

The repository is a small full-stack project, not a distributed microservice system. The UI and API are separate runtime processes; the API currently owns database access and inference.

## 2. Repository map

```text
wind-sensational-project/
├── backend/
│   ├── ARCHITECTURE.md
│   ├── requirements.txt
│   ├── saved_models/                 # serialized ML assets, thresholds, metadata, and feature lists
│   ├── tests/                        # Python tests for backend contracts and thermal logic
│   ├── venv_broken/                  # legacy/partial local virtualenv snapshot; not active runtime code
│   └── app/
│       ├── __init__.py
│       ├── main.py                   # FastAPI application bootstrap and router registration
│       ├── pipeline.py               # thermal orchestration entrypoint for SCADA processing
│       ├── api/
│       │   ├── __init__.py
│       │   ├── router.py             # aggregate router for active app modules
│       │   └── routes/
│       │       ├── __init__.py
│       │       ├── analytics.py
│       │       ├── health.py
│       │       ├── ml.py
│       │       ├── model_performance.py
│       │       ├── rul.py
│       │       ├── thermal.py
│       │       └── turbines.py
│       ├── analytics/
│       │   ├── __init__.py
│       │   ├── dependency_pipeline.py
│       │   ├── fallbacks.py
│       │   ├── model_backtest.py
│       │   ├── model_inference.py
│       │   ├── power_curve_gpr.py
│       │   └── rul_pipeline.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── lifespan.py
│       │   └── logging.py
│       ├── db/
│       │   ├── __init__.py
│       │   ├── anomaly_json_store.py
│       │   ├── connection.py
│       │   ├── data_mode.py
│       │   ├── database_pipeline.py
│       │   ├── db_telemetry.py
│       │   ├── event_queries.py
│       │   ├── physical_health_store.py
│       │   └── training_status_store.py
│       ├── legacy/
│       │   ├── __init__.py
│       │   ├── anomaly_service.py
│       │   ├── ensemble_service.py
│       │   ├── ml_routes.py
│       │   └── model_loader.py
│       └── ml/
│           ├── __init__.py
│           ├── lstm_numpy.py
│           ├── model_state.py
│           ├── retrain_watcher.py
│           └── train_runner.py
├── frontend/
│   ├── package.json                  # root package stub/metadata for the workspace
│   └── app/
│       ├── package.json              # front-end project manifest for Vite
│       ├── vite.config.js            # Vite config and proxy setup
│       ├── index.html                # browser entry document
│       ├── public/
│       └── src/
│           ├── App.jsx               # router, navigation and page wiring
│           ├── App.css
│           ├── index.css
│           ├── main.jsx              # React bootstrap entrypoint
│           ├── assets/
│           ├── components/
│           │   ├── common/
│           │   │   ├── Header.jsx
│           │   │   ├── KpiCard.jsx
│           │   │   └── Navigation.jsx
│           │   ├── models/
│           │   │   └── ModelCard.jsx
│           │   └── overview/
│           │       ├── FaultTable.jsx
│           │       └── TurbineMap.jsx
│           ├── hooks/
│           │   ├── useAnomalyData.js
│           │   └── useThermalData.js
│           ├── pages/
│           │   ├── AnomalyPage.jsx
│           │   ├── DependencyPage.jsx
│           │   ├── EnsemblePage.jsx
│           │   ├── ForecastPage.jsx
│           │   ├── ModelsPage.jsx
│           │   ├── OverviewPage.jsx
│           │   ├── RulPage.jsx
│           │   └── ThermalPage.jsx
│           └── services/
│               ├── analyticsApi.js
│               ├── api.js
│               ├── client.js
│               ├── endpoints.js
│               ├── mlApi.js
│               ├── thermalApi.js
│               └── turbineApi.js
└── ... (other non-source repository artifacts such as logs and model binaries)
```

The backend app tree is organized around a FastAPI API layer, database/query helpers, analytics pipelines, and ML state/training modules. The frontend is a Vite + React SPA whose UI pages, hooks, and services all live under `frontend/app/src/`. Generated dependencies, the checked-in Python environment directory, model binaries, logs, and caches are not application source modules.

## 3. Runtime entrypoints and initialization

### Backend

`backend/app/main.py` creates the FastAPI application (`Wind Sentinel API`, version `2.0.0`), configures CORS, invokes model loading from a startup event, and mounts the active route modules. The root route returns service status and version information. The API is intended to be launched from `backend/`, for example:

```text
uvicorn app.main:app --reload --port 8000
```

The active app mounts the ML, analytics, thermal, RUL, turbine, and health routers with `/api/v1/...` prefixes. The route details are summarized in section 5.

`backend/app/core/lifespan.py` separately defines an ASGI lifespan context that loads models and starts an APScheduler job to check for retraining every 30 minutes. **`main.py` does not pass this context as the FastAPI lifespan**, so that scheduler is not started by the current application entrypoint. Model loading is instead triggered by the startup event in `main.py` and may also be invoked lazily by ML endpoints.

### Frontend

`frontend/app/index.html` mounts `src/main.jsx` at `#root`. `main.jsx` renders `App` inside React `StrictMode`. `App.jsx` wraps its content in `BrowserRouter`, selects one page from a static route table, and combines the shared header/navigation with page-level loading state.

The Vite development server listens on port `5173` and proxies `/api` requests to `http://localhost:8000`. The frontend can be launched from `frontend/app/` with:

```text
npm run dev
```

The package also provides `npm run build`, `npm run lint`, and `npm run preview`.

## 4. Backend layers and responsibilities

### HTTP/API layer — `backend/app/api/`

The route modules translate HTTP requests into data-access and analytics calls. Most routes return JSON dictionaries/lists; request parameters are declared with FastAPI query/path parameters. There is no separate DTO/entity package or authentication router in the active app.

- `routes/turbines.py`: fleet IDs, latest readings, live telemetry, and per-turbine/fleet status. The status calls reuse the ML ensemble endpoint.
- `routes/analytics.py`: fleet overview, turbine analytics, power-curve data, fleet trends, dependency analysis, and Gaussian-process power curves.
- `routes/ml.py`: model status/reload, charts, ensemble anomaly detection, and manual training status/trigger.
- `routes/thermal.py`: loads per-turbine thermal SCADA records and invokes the thermal inference pipeline.
- `routes/rul.py`: fetches physical-health history and computes RUL, using a no-data fallback when history is insufficient.
- `routes/health.py`: service metadata and model/data-mode status.
- `routes/model_performance.py`: backtest endpoints are defined here but **are not mounted by `main.py`**.

`api/router.py` aggregates route modules, including model-performance routes, but `main.py` mounts the route modules directly rather than including this aggregate router. It therefore is not the active router composition.

### Core and configuration — `backend/app/core/`

`config.py` loads a `.env` file and centralizes database settings, SCADA table names, ML feature lists, model participation policy, and analytics thresholds. Database credentials are read from `WINDSENTINEL_DB_HOST`, `WINDSENTINEL_DB_PORT`, `WINDSENTINEL_DB_NAME`, `WINDSENTINEL_DB_USER`, and `WINDSENTINEL_DB_PASSWORD`; no credential value is part of this document.

The inference feature contracts are order-sensitive:

- `MAIN_FEATURES`: 26 fields for the primary anomaly models.
- `AE_FEATURES`: 18 fields for the LSTM autoencoder.
- The model loader validates scaler/model dimensions and records validation results in shared model state.

Other central settings include `SCADA_SCHEMA` / `SCADA_TABLE`, the optional `SAVED_MODELS_DIR`, the VAE enable flag, anomaly severity thresholds, default time windows, and RUL parameters. Not all queries consume these settings: several modules still reference `site_dzh.scada_raw` directly.

`logging.py` exposes a small `basicConfig` setup helper. The current `main.py` configures logging inline and does not call that helper.

### Data access and persistence — `backend/app/db/`

Database access is implemented with `psycopg2` and hand-written SQL rather than an application ORM model layer.

- `connection.py`: opens a PostgreSQL connection using the `WINDSENTINEL_DB_*` environment variables.
- `database_pipeline.py`: SCADA turbine IDs, latest readings, time-window history, power curves, fleet trends and RUL input queries.
- `db_telemetry.py`: latest fleet-wide telemetry projection.
- `data_mode.py`: tests for records in the last 24 hours and creates query windows relative to the latest available SCADA timestamp.
- `event_queries.py`: reads fault events from `event_raw` for backtesting and summaries.
- `physical_health_store.py`: transforms SCADA vibration and bearing-temperature history into daily physical-health values.
- `anomaly_json_store.py`: writes daily anomaly summaries to `anomaly_daily_scores.json` using a temporary file and atomic replacement.
- `training_status_store.py`: maintains a bounded local `training_history.json` log.

The primary observed data sources are:

1. `site_dzh.scada_raw`: turbine identity, timestamps, power, speed, pitch, temperature, and vibration measurements. Column names are aliased into application feature names by queries.
2. `event_raw`: event identifiers, turbine IDs, event times/types, severity, descriptions and codes used by model backtesting.

No SQLAlchemy model classes or explicit schema migration definitions were found in the application modules. Connections and SQL are generally scoped to individual helper calls; the code does not define a cross-request ORM session or a broad transactional unit of work. The exact deployed schema and indexes are external to this repository.

### Analytics and inference — `backend/app/analytics/` and `backend/app/pipeline.py`

- `model_inference.py`: shared model inference helpers.
- `dependency_pipeline.py`: fetches SCADA tag samples and calculates relationships including correlation, mutual information, SHAP, Granger tests, and a graphical-lasso network. Its analysis blocks use a thread pool.
- `power_curve_gpr.py`: fits a fleet power curve with clustering and Gaussian-process regression and produces uncertainty ranges.
- `thermal_pipeline.py`, invoked through `app/pipeline.py`: computes per-turbine thermal measures, warnings, anomalies, and alerts from SCADA records.
- `physical_health_store.py` and `rul_pipeline.py`: create a physical-health time series and fit an exponential degradation curve to estimate RUL and status.
- `model_backtest.py`: evaluates available model outputs against actual event records.
- `fallbacks.py`: supplies a fallback power curve and a no-data RUL response. The power-curve fallback is generated rather than read from the database.

The data-mode helper labels the database `live` when rows exist within the previous 24 hours and otherwise labels it `historical`. Query windows in the helper are anchored to the latest stored timestamp, supporting historical datasets. Individual route/data paths do not all use the same helper or identical fallback policy.

### ML state and training — `backend/app/ml/`

`model_state.py` owns the process-global `ML_MODELS` dictionary and `load_ml_models()`. The loader finds artifacts under `backend/saved_models/`, loads available estimators/scalers/neural networks, and records dimensions/feature-contract validation. Main ensemble voters are Isolation Forest, Elliptic Envelope, and XGBoost. The LSTM autoencoder is an advisory score-only model. The VAE checkpoint is loaded for validation but excluded from ensemble scoring by default because its recorded feature mapping is not available.

`train_runner.py` contains the data preparation and model training pipeline. The ML API's manual trigger starts a background daemon thread. `retrain_watcher.py` records row-count state and defines the 2,000-new-row threshold used for automatic retraining checks; the scheduled watcher is not currently connected to the active FastAPI startup, as described above. Local training status is persisted separately from the trained model artifacts.

`backend/saved_models/` contains the serialized classifiers, autoencoder models, scalers, feature metadata and thresholds. These files are runtime inputs, not Python modules. The loader accepts alternate file formats/names for some artifacts, so the exact set present on disk affects which models are available.

## 5. Active HTTP surface

The active route prefixes are mounted in `backend/app/main.py`. Key route groups are:

| Prefix | Main operations |
|---|---|
| `/api/v1/health` | service/data/model status |
| `/api/v1/turbines` | list, live telemetry, latest reading, single/fleet status |
| `/api/v1/analytics` | fleet overview, turbine analytics, dependency engine, trends and power curves |
| `/api/v1/ml` | model status/reload, anomaly chart data and ensemble results |
| `/api/v1/thermal` | thermal results by turbine ID |
| `/api/v1/rul` | RUL prediction by turbine ID |

Important endpoints exposed by the active routers include:

- `GET /api/v1/turbines/list`, `/live`, `/{turbine_id}/latest`, `/{turbine_id}/status`, and `/all-status`
- `GET /api/v1/analytics/power-curve`, `/fleet-overview`, `/turbine/{turbine_id}`, `/dependency-engine`, `/fleet-trend`, and `/power-curve-gpr`
- `GET /api/v1/ml/models/status`, `/isolation-forest-chart`, `/autoencoder-chart`, `/mahalanobis-chart`, `/tsne-chart`, `/cusum-chart`, and `/ensemble-anomaly-detection`; `POST /api/v1/ml/models/reload`
- `GET /api/v1/thermal/{turbine_id}` and `GET /api/v1/rul/{turbine_id}`

The ML router also declares training paths with an embedded `/api/v1/ml/...` prefix while it is mounted under `/api/v1/ml`. As written, this yields doubled paths for those operations. The frontend currently requests the undoubled training URLs, so the frontend/backend training path contract should be treated as inconsistent.

`model_performance.py` declares performance and summary routes, but because that router is not mounted from `main.py`, those endpoints are not part of the current application surface.

## 6. Frontend layers and functional areas

### Navigation and pages

`App.jsx` defines eight page routes:

| Browser path | Page | Primary responsibility |
|---|---|---|
| `/` | `OverviewPage` | fleet summary and operational trend views |
| `/thermal` | `ThermalPage` | thermal condition and alerts |
| `/anomaly` | `AnomalyPage` | anomaly charts and model-specific signals |
| `/dependency` | `DependencyPage` | dependencies among SCADA signals |
| `/models` | `ModelsPage` | model availability and training controls |
| `/rul` | `RulPage` | turbine remaining-useful-life estimates |
| `/forecast` | `ForecastPage` | telemetry-derived visual projections |
| `/ensemble` | `EnsemblePage` | ensemble voter metadata and anomaly results |

The forecast screen currently derives a one-step projection from latest telemetry in the browser; it is not backed by a dedicated forecast API. Shared components include header/navigation, KPI cards, model cards, a turbine map, and a fault table.

### Data access and state

The frontend has two data-access patterns:

- Shared Axios client plus endpoint constants/wrappers in `services/client.js`, `endpoints.js`, `analyticsApi.js`, `mlApi.js`, and `turbineApi.js`.
- Page-level `fetch()` calls and a dedicated Axios wrapper in `thermalApi.js`; `useThermalData.js` packages thermal loading, error, last-update and refetch state.

Most pages own local React state and effects rather than a global server-state store. `App.jsx` receives loading state from the active page and disables navigation while that page reports an in-flight load. The service client can attach an `authToken` from local storage and handles a `401` redirect, but no active backend authentication endpoints are mounted.

API base configuration is not fully unified: the main shared client uses `VITE_API_URL` (default `http://localhost:8000`), the Vite dev server proxies `/api`, the thermal client has a hard-coded local API URL, and a legacy API module uses Create React App's `REACT_APP_API_URL`. Page-level requests also construct API URLs directly. These represent distinct current integration paths.

## 7. End-to-end request and data flows

### Telemetry and dashboard analytics

1. A page or service calls an API URL, usually under `/api/v1`.
2. FastAPI dispatches to a route module registered by `main.py`.
3. The route calls a helper in `app/db/` to query SCADA/fault-event data.
4. Analytics modules transform records, calculate fleet/turbine metrics, or generate fallback responses where coded.
5. FastAPI serializes the result as JSON and the page renders charts, tables, cards or status indicators.

### ML anomaly results

1. The ML route obtains SCADA rows, maps/derives features, and aligns columns to the model feature contract.
2. Shared model objects and scalers are read from the in-process `ML_MODELS` registry.
3. Isolation Forest, Elliptic Envelope and XGBoost form the configured vote set; LSTM reconstruction error is score-only.
4. The route returns anomaly rows and model participation metadata. The daily score helper can aggregate results into a local JSON file.

### RUL

1. The RUL route requests physical measurements for a turbine.
2. Physical-health logic aggregates vibration and bearing temperature against a turbine-specific baseline into daily scores.
3. With enough history, the RUL pipeline fits an exponential degradation curve and returns an estimate/status; with insufficient history, it returns the no-data fallback.

## 8. Technology inventory

### Backend

- Python; the project requirements do not declare a Python version constraint.
- FastAPI `0.104.1`, Uvicorn `0.24.0`, Pydantic `2.5.0`.
- PostgreSQL driver: `psycopg2-binary 2.9.9`; SQLAlchemy `2.0.23` is declared, though active data access inspected here uses direct `psycopg2`.
- Data/scientific stack: pandas `2.1.3`, NumPy `1.23.5`, SciPy `1.11.4`, statsmodels `0.14.1`, scikit-learn `1.3.2`, SHAP `0.44.1`.
- ML stack: XGBoost `2.0.1`, TensorFlow `2.16.2` with Keras `3.15.0` (matches the saved LSTM artifact), PyTorch `2.1.0`, joblib `1.3.2`.
- Other declared packages: python-dotenv `1.0.0`, APScheduler `3.10.4`.

### Frontend

- React and React DOM `^19.2.8`, React Router `^7.18.3`, Vite `^8.2.0`, and `@vitejs/plugin-react ^6.0.4`.
- Axios `^1.20.0`; charting with Chart.js `^4.5.1`, react-chartjs-2 `^5.3.1`, Recharts `^3.10.1`, and D3 `^7.8.5`.
- ESLint `^10.8.0`, React hooks/refresh plugins, and React type packages.
- The package uses JavaScript/JSX; no TypeScript source or Node engine constraint was found in the application manifest.

Versions above are the declared manifest versions/ranges, not a report of all resolved transitive versions.

## 9. Tests and verification surface

The backend test suite includes:

- `tests/test_ensemble_contract.py`: ML voter participation, feature dimensions, result consistency, surfaced failures, and historical date-window anchoring; database access is mocked.
- `tests/test_thermal_pipeline.py`: thermal anomaly attribution and insufficient-history behavior.

The frontend package defines build and lint scripts but no test script in its manifest. This document update does not claim that runtime integration or database-backed behavior has been exercised.

## 10. Current integration caveats

These are observed implementation facts that affect how the architecture is operated or understood:

1. `main.py` mounts route modules directly. The aggregate `api/router.py`, the performance router, and the optional lifespan scheduler do not become active merely by existing in the source tree.
2. The ML training route prefix is repeated beneath the ML mount, while frontend calls use the shorter path.
3. Frontend API base configuration is split across Vite env, hard-coded local URLs, direct page requests, and a legacy CRA-style variable.
4. Data access is not uniformly routed through `SCADA_TABLE`; code contains direct references to `site_dzh.scada_raw` and `event_raw`.
5. Some analytics endpoints have synthetic/fallback behavior, while other code paths return empty/error responses. Consumers should use each endpoint's response status and source metadata rather than assuming every response represents live database data.
6. `backend/app/legacy/` and suffixed files such as `ml-SUBHAKARAN.py` and `model_state-SUBHAKARAN.py` are historical/compatibility material, not the primary runtime path. `legacy/LEGACY.md` documents the quarantined cluster.

## 11. Local development

Run the backend and frontend in separate terminals. Configure the backend database environment variables before starting it; the frontend development server proxies API requests to port `8000`.

```text
# Terminal 1
cd backend
uvicorn app.main:app --reload --port 8000

# Terminal 2
cd frontend/app
npm run dev
```

The backend model loader expects artifacts in `backend/saved_models/` unless its model-directory configuration is overridden. PostgreSQL schema/table availability and the `event_raw` source are external runtime prerequisites.
