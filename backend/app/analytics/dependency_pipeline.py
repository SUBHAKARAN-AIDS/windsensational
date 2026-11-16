"""
app/analytics/dependency_pipeline.py

Dependency Engine - variable relationship discovery over SCADA tags.

NOTHING about the sensors is typed in here any more. Everything is read from the
database:
  - which columns are sensors          -> information_schema + a look at the real data
  - which turbines exist                -> the turbine column
  - what the latest date is             -> MAX(time)
  - which tag the importance chart explains -> chosen from the data (or by the page)
  - which pairs get the "who moves first" test -> chosen from the data
  - the earlier period used for "what changed" -> same table, just earlier dates

The only fixed things are the table address and tuning numbers (row caps, cut-offs).
They live in SETTINGS / environment variables below, so changing them never needs
a code edit:
    SCADA_SCHEMA (default site_dzh)   SCADA_TABLE (default scada_raw)
    SCADA_TIME_COLUMN (default time)  SCADA_TURBINE_COLUMN (default name)

Public entry point:
    build_dependency_engine_payload(turbine_id=None, days=7, limit=5000,
                                    baseline_days=30, target=None)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

# ------------------------------------------------------------------
# IMPORTANT: the heavy libraries (numpy, pandas, scikit-learn, SHAP, statsmodels, psycopg2)
# are NOT imported when this file is loaded by the web server. They are imported by
# _import_heavy(), which only runs inside the separate analysis process (or when
# DEP_ISOLATE=0). This keeps them out of the web server, which already has TensorFlow
# and PyTorch loaded and crashed when these were added on top.
# ------------------------------------------------------------------
_HEAVY_LOADED = False
HAS_SHAP = False
HAS_STATS = False


def _import_heavy():
    global _HEAVY_LOADED, HAS_SHAP, HAS_STATS
    global np, pd, RealDictCursor, get_db_connection
    global GraphicalLassoCV, RandomForestRegressor, mutual_info_regression, StandardScaler
    global shap, adfuller, grangercausalitytests
    if _HEAVY_LOADED:
        return
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from psycopg2.extras import RealDictCursor  # type: ignore  # noqa: F811
    from app.db.connection import get_db_connection  # noqa: F811
    from sklearn.covariance import GraphicalLassoCV  # noqa: F811
    from sklearn.ensemble import RandomForestRegressor  # noqa: F811
    from sklearn.feature_selection import mutual_info_regression  # noqa: F811
    from sklearn.preprocessing import StandardScaler  # noqa: F811
    try:
        import shap  # noqa: F811
        HAS_SHAP = True
    except BaseException:
        shap = None
        HAS_SHAP = False
    try:
        from statsmodels.tsa.stattools import adfuller, grangercausalitytests  # noqa: F811
        HAS_STATS = True
    except BaseException:
        adfuller = None
        grangercausalitytests = None
        HAS_STATS = False
    _HEAVY_LOADED = True


logger = logging.getLogger(__name__)


# ============================================================
# SETTINGS (addresses + tuning numbers only - no sensor names)
# ============================================================
def _env(name, default, cast=str):
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except Exception:
        return default


SCHEMA = _env("SCADA_SCHEMA", "site_dzh")
TABLE = _env("SCADA_TABLE", "scada_raw")
TIME_COL = _env("SCADA_TIME_COLUMN", "time")
TURBINE_COL = _env("SCADA_TURBINE_COLUMN", "name")

# Columns that are codes / flags / counters are NOT physical sensors (a "status character index"
# says nothing about how the machine is behaving). Any column whose name contains one of
# these words is left out. Change the list with  DEP_EXCLUDE_WORDS=status,code,...  (empty = keep all)
_EXCLUDE_WORDS = {w.strip().lower() for w in _env(
    "DEP_EXCLUDE_WORDS",
    "status,state,code,index,flag,alarm,fault,mode,character,counter,serial,quality,row,id",
).split(",") if w.strip()}


def _is_code_column(name) -> bool:
    words = set(re.split(r"[^a-z0-9]+", str(name).lower()))
    return bool(words & _EXCLUDE_WORDS)


SETTINGS = {
    "max_tags": _env("DEP_MAX_TAGS", 20, int),              # most sensors shown at once
    "min_completeness": _env("DEP_MIN_COMPLETENESS", 0.7, float),  # sensor needs this share of real values
    "min_unique_values": _env("DEP_MIN_UNIQUE", 10, int),   # sensors that barely change are skipped
    "duplicate_corr": _env("DEP_DUPLICATE_CORR", 0.995, float),    # near-copies of another sensor are skipped
    "heavy_rows": _env("DEP_HEAVY_ROWS", 1500, int),        # cap for the slow calculations
    "edge_threshold": _env("DEP_EDGE_THRESHOLD", 0.05, float),     # network: smallest link shown
    "granger_max_lag": _env("DEP_GRANGER_MAX_LAG", 6, int),
    "granger_pairs": _env("DEP_GRANGER_PAIRS", 4, int),
    "granger_rows": _env("DEP_GRANGER_ROWS", 1500, int),
    "granger_alpha": _env("DEP_GRANGER_ALPHA", 0.05, float),
    "min_drift_rows": _env("DEP_MIN_DRIFT_ROWS", 100, int),
    "cache_seconds": _env("DEP_CACHE_SECONDS", 600, int),
    # 0 = run the five analyses one after another (safest, a few seconds slower)
    # 1 = run them at the same time
    "parallel": _env("DEP_PARALLEL", 0, int),
    # 1 = run the whole analysis in a separate Python process, so that if it ever
    #     crashes the web server stays up and the page shows what failed.
    # 0 = run inside the web server itself.
    "isolate": _env("DEP_ISOLATE", 1, int),
    "isolate_timeout": _env("DEP_ISOLATE_TIMEOUT", 240, int),
}

_NUMERIC_TYPES = {"smallint", "integer", "bigint", "numeric", "decimal", "real", "double precision"}
_TEXT_TYPES = {"text", "character varying", "character", "varchar", "char"}

_cache: dict = {}


def _qi(name: str) -> str:
    """Safely quote a column/table name that came from the database."""
    return '"' + str(name).replace('"', '""') + '"'


_TABLE_SQL = f"{_qi(SCHEMA)}.{_qi(TABLE)}"


def _cached(key, loader):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < SETTINGS["cache_seconds"]:
        return hit[1]
    value = loader()
    _cache[key] = (now, value)
    return value


# ============================================================
# 1. DISCOVERY + DATA FETCH (all from the database)
# ============================================================
def discover_candidate_columns(cursor) -> list:
    """Every column that could hold a sensor value (number or number-as-text)."""
    def load():
        cursor.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (SCHEMA, TABLE),
        )
        rows = cursor.fetchall()
        return [
            r["column_name"] for r in rows
            if r["column_name"] not in (TIME_COL, TURBINE_COL)
            and r["data_type"] in (_NUMERIC_TYPES | _TEXT_TYPES)
            and not _is_code_column(r["column_name"])
        ]
    return _cached("columns", load)


def _latest_time(cursor, turbine_id):
    if turbine_id:
        cursor.execute(
            f"SELECT MAX({_qi(TIME_COL)}) AS latest_time FROM {_TABLE_SQL} WHERE {_qi(TURBINE_COL)} = %s",
            (turbine_id,),
        )
    else:
        cursor.execute(f"SELECT MAX({_qi(TIME_COL)}) AS latest_time FROM {_TABLE_SQL}")
    row = cursor.fetchone()
    return row["latest_time"] if row else None


def list_turbines(cursor, latest) -> list:
    """Turbines that reported in the last day of data (cheap: only 1 day is scanned)."""
    def load():
        cursor.execute(
            f"SELECT DISTINCT {_qi(TURBINE_COL)} AS t FROM {_TABLE_SQL} "
            f"WHERE {_qi(TIME_COL)} >= %s AND {_qi(TURBINE_COL)} IS NOT NULL ORDER BY 1",
            (latest - timedelta(days=1),),
        )
        return [str(r["t"]) for r in cursor.fetchall()]
    return _cached(("turbines", str(latest)), load)


def _coerce_numeric(df: pd.DataFrame, columns) -> pd.DataFrame:
    for c in columns:
        if c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df


def _fetch_window(cursor, conn, columns, start, end, turbine_id, limit, end_inclusive=True) -> pd.DataFrame:
    """
    Rows between start and end, SPREAD EVENLY over the whole window (not just the
    newest rows). The share kept is worked out from the real row count.
    """
    end_op = "<=" if end_inclusive else "<"
    where = f"{_qi(TIME_COL)} >= %(start)s AND {_qi(TIME_COL)} {end_op} %(end)s"
    params = {"start": start, "end": end}
    if turbine_id:
        where += f" AND {_qi(TURBINE_COL)} = %(turbine)s"
        params["turbine"] = turbine_id

    cursor.execute(f"SELECT COUNT(*) AS n FROM {_TABLE_SQL} WHERE {where}", params)
    n = int(cursor.fetchone()["n"] or 0)
    if n == 0:
        return pd.DataFrame()

    stride = max(1, math.ceil(n / max(1, limit)))
    select = (
        f"{_qi(TIME_COL)} AS __time, {_qi(TURBINE_COL)} AS __turbine, "
        + ", ".join(_qi(c) for c in columns)
    )
    base = f"SELECT {select} FROM {_TABLE_SQL} WHERE {where}"
    tail = " ORDER BY __time DESC LIMIT %(limit)s"
    params["limit"] = int(limit)

    records = None
    if stride > 1:
        sample = f" AND MOD(hashtext({_qi(TURBINE_COL)}::text || {_qi(TIME_COL)}::text), %(stride)s) = 0"
        try:
            cursor.execute(base + sample + tail, {**params, "stride": stride})
            records = cursor.fetchall()
        except Exception as e:
            logger.warning(f"Even sampling not available ({e}); using newest rows instead")
            conn.rollback()
    if records is None:
        cursor.execute(base + tail, params)
        records = cursor.fetchall()

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df.sort_values("__time").reset_index(drop=True)
    return _coerce_numeric(df, columns)


def _fetch_series(cursor, columns, start, end, turbine, rows) -> pd.DataFrame:
    """The newest `rows` consecutive readings of ONE turbine (needed for 'who moves first')."""
    where = f"{_qi(TIME_COL)} >= %(start)s AND {_qi(TIME_COL)} <= %(end)s"
    params = {"start": start, "end": end, "n": int(rows)}
    if turbine:
        where += f" AND {_qi(TURBINE_COL)} = %(turbine)s"
        params["turbine"] = turbine
    cursor.execute(
        f"SELECT {_qi(TIME_COL)} AS __time, " + ", ".join(_qi(c) for c in columns)
        + f" FROM {_TABLE_SQL} WHERE {where} ORDER BY __time DESC LIMIT %(n)s",
        params,
    )
    recs = cursor.fetchall()
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs).sort_values("__time").reset_index(drop=True)
    return _coerce_numeric(df, columns)


# ============================================================
# 2. WORK OUT WHICH COLUMNS ARE REAL SENSORS (from the data itself)
# ============================================================
def _is_counter(df: pd.DataFrame, c: str) -> bool:
    """True when a column only ever goes up (or only down) - a counter or row number.
    Checked per turbine, because rows of different turbines are interleaved."""
    groups = df.groupby("__turbine")[c] if "__turbine" in df.columns and df["__turbine"].notna().any() else [(None, df[c])]
    checked = 0
    for _, g in groups:
        v = g.dropna()
        if len(v) < 10:
            continue
        checked += 1
        if not (v.is_monotonic_increasing or v.is_monotonic_decreasing):
            return False
    return checked > 0


def select_tag_columns(df: pd.DataFrame, candidates) -> list:
    n = len(df)
    s = SETTINGS
    stats = []
    for c in candidates:
        if c not in df.columns:
            continue
        col = df[c]
        cnt = int(col.notna().sum())
        if cnt < max(10, s["min_completeness"] * n):
            continue
        nun = int(col.nunique())
        sd = col.std()
        if nun < s["min_unique_values"] or not np.isfinite(sd) or sd == 0:
            continue
        if nun > 20 and _is_counter(df, c):
            continue  # counters / row numbers, not sensors
        stats.append((c, cnt, nun))
    if not stats:
        return []
    stats.sort(key=lambda t: (-t[1], -t[2]))
    names = [c for c, _, _ in stats]

    filled = df[names].fillna(df[names].mean())
    if len(filled) > 3000:
        filled = filled.sample(n=3000, random_state=42)
    corr = filled.corr().abs().fillna(0.0)
    kept = []
    for c in names:
        if any(corr.loc[c, k] > s["duplicate_corr"] for k in kept):
            continue
        kept.append(c)
        if len(kept) >= s["max_tags"]:
            break
    return kept


def _pretty(col: str) -> str:
    return re.sub(r"[_\s]+", " ", str(col)).strip().title()


def _unique_labels(columns) -> dict:
    labels, used = {}, set()
    for c in columns:
        lab = _pretty(c) or str(c)
        base, i = lab, 2
        while lab in used:
            lab = f"{base} ({i})"
            i += 1
        used.add(lab)
        labels[c] = lab
    return labels


def _to_label_frame(df: pd.DataFrame, col2label: dict) -> pd.DataFrame:
    cols = [c for c in col2label if c in df.columns]
    out = df[cols].copy()
    out.columns = [col2label[c] for c in cols]
    return out.reset_index(drop=True)


# ============================================================
# 3. ANALYSIS BLOCKS
# ============================================================
def _fill(sub: pd.DataFrame) -> pd.DataFrame:
    return sub.fillna(sub.mean(numeric_only=True)).fillna(0.0)


def _cap(frame: pd.DataFrame) -> pd.DataFrame:
    cap = SETTINGS["heavy_rows"]
    return frame.sample(n=cap, random_state=42) if len(frame) > cap else frame


def _empty_corr():
    return {"tags": [], "pearson": [], "spearman": [], "mutual_info": [], "controlled": [],
            "shared_load_share": None, "top_dependencies": [], "top_dependencies_controlled": []}


def compute_correlation_block(sub: pd.DataFrame) -> dict:
    if sub.shape[1] < 2 or sub.shape[0] < 10:
        return _empty_corr()
    filled = _cap(_fill(sub))
    tags = list(filled.columns)
    n = len(tags)

    pearson = filled.corr(method="pearson").fillna(0.0).values
    spearman = filled.corr(method="spearman").fillna(0.0).values

    # Relationships left over after removing the ONE pattern shared by everything
    # (how hard the turbine is working). Found from the data, not from any tag name.
    try:
        Z = StandardScaler().fit_transform(filled.values)
        eigvals, eigvecs = np.linalg.eigh(np.cov(Z, rowvar=False))
        comp = eigvecs[:, -1]                      # direction of the biggest shared pattern
        resid = Z - np.outer(Z @ comp, comp)
        controlled = np.nan_to_num(np.corrcoef(resid, rowvar=False))
        np.fill_diagonal(controlled, 1.0)
        shared = float(eigvals[-1] / eigvals.sum())
    except Exception as e:
        logger.warning(f"Controlled correlation failed: {e}")
        controlled, shared = np.zeros((n, n)), None

    mi_raw = np.zeros((n, n))
    values = filled.values
    for i in range(n):
        try:
            mi_raw[i, :] = mutual_info_regression(values, values[:, i], random_state=42, n_neighbors=3)
        except Exception as e:
            logger.warning(f"MI computation failed for {tags[i]}: {e}")
    mi_norm = mi_raw / (mi_raw.max() if mi_raw.max() > 0 else 1.0)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "a": tags[i], "b": tags[j],
                "pearson": round(float(pearson[i, j]), 3),
                "controlled": round(float(controlled[i, j]), 3),
                "mutual_info": round(float(mi_norm[i, j]), 3),
            })
    top = sorted(pairs, key=lambda d: abs(d["pearson"]), reverse=True)[:10]
    top_ctrl = sorted(pairs, key=lambda d: abs(d["controlled"]), reverse=True)[:10]

    rnd = lambda m: [[round(float(v), 3) for v in row] for row in m]
    return {
        "tags": tags, "pearson": rnd(pearson), "spearman": rnd(spearman),
        "mutual_info": rnd(mi_norm), "controlled": rnd(controlled),
        "shared_load_share": None if shared is None else round(shared, 3),
        "top_dependencies": top, "top_dependencies_controlled": top_ctrl,
    }


def default_target(sub: pd.DataFrame):
    """Tag that the rest of the sensors explain best (highest average link to the others)."""
    if sub.shape[1] < 2:
        return None
    c = _cap(_fill(sub)).corr().abs().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(c, 0.0)
    return list(sub.columns)[int(np.argmax(c.mean(axis=1)))]


def compute_shap_block(sub: pd.DataFrame, target: str) -> dict:
    empty = {"target": target, "features": [], "values": [], "direction": [], "r2": None}
    if target not in sub.columns or sub.shape[1] < 3:
        return empty
    sub = sub.dropna(subset=[target])
    if len(sub) < 30:
        return empty
    feature_cols = [c for c in sub.columns if c != target]
    X = sub[feature_cols].fillna(sub[feature_cols].mean()).fillna(0.0)
    y = sub[target]
    if len(X) > SETTINGS["heavy_rows"]:
        X = X.sample(n=SETTINGS["heavy_rows"], random_state=42)
        y = y.loc[X.index]
    try:
        model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42,
                                      n_jobs=1, oob_score=True)
        model.fit(X, y)
        r2 = float(getattr(model, "oob_score_", np.nan))
        direction = np.zeros(len(feature_cols))
        try:
            sv = shap.TreeExplainer(model).shap_values(X)
            mean_abs = np.abs(sv).mean(axis=0)
            for k in range(len(feature_cols)):
                if np.std(sv[:, k]) > 0 and np.std(X.iloc[:, k]) > 0:
                    direction[k] = np.sign(np.corrcoef(X.iloc[:, k], sv[:, k])[0, 1])
        except Exception as e:
            logger.warning(f"SHAP fallback to RF importances: {e}")
            mean_abs = model.feature_importances_
        order = np.argsort(mean_abs)[::-1]
        return {
            "target": target,
            "features": [feature_cols[i] for i in order],
            "values": [round(float(mean_abs[i]), 4) for i in order],
            "direction": [int(np.nan_to_num(direction[i])) for i in order],
            "r2": None if not np.isfinite(r2) else round(r2, 3),
        }
    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return empty


def compute_drift_block(cur: pd.DataFrame, base: pd.DataFrame, windows: dict) -> dict:
    out = {"status": "ok", "tags": [], "delta": [], "top_changes": [], "mean_change": None,
           "windows": windows}
    need = SETTINGS["min_drift_rows"]
    if len(base) < need or len(cur) < need:
        out["status"] = "not_enough_baseline"
        out["message"] = (f"Only {len(base)} earlier rows found (need {need}). "
                          f"Try a shorter 'earlier period' or pick a turbine with more history.")
        return out
    tags = [c for c in cur.columns if c in base.columns and base[c].notna().sum() >= 10]
    if len(tags) < 2:
        out["status"] = "not_enough_baseline"
        out["message"] = "Too few sensors with data in the earlier period."
        return out
    pc = _cap(_fill(cur[tags])).corr().fillna(0.0).values
    pb = _cap(_fill(base[tags])).corr().fillna(0.0).values
    delta = pc - pb
    n = len(tags)
    changes = []
    for i in range(n):
        for j in range(i + 1, n):
            changes.append({"a": tags[i], "b": tags[j],
                            "baseline": round(float(pb[i, j]), 3),
                            "current": round(float(pc[i, j]), 3),
                            "change": round(float(delta[i, j]), 3)})
    changes.sort(key=lambda d: abs(d["change"]), reverse=True)
    iu = np.triu_indices(n, 1)
    out.update({
        "tags": tags,
        "delta": [[round(float(v), 3) for v in row] for row in delta],
        "top_changes": changes[:10],
        "mean_change": round(float(np.abs(delta[iu]).mean()), 4),
    })
    return out


def compute_granger_block(series: pd.DataFrame, turbine_used) -> dict:
    s = SETTINGS
    out = {"status": "ok", "lags": [], "pairs": [], "turbine_used": turbine_used,
           "differenced": [], "rows_used": 0, "p_threshold": None}
    if not HAS_STATS:
        out["status"] = "unavailable"
        out["message"] = "statsmodels is not installed on the server."
        return out
    max_lag = s["granger_max_lag"]
    sub = series.dropna(axis=1, thresh=int(len(series) * 0.7)) if len(series) else series
    sub = sub.interpolate(limit=3, limit_direction="both").dropna()
    sub = sub.loc[:, sub.std() > 1e-8]
    if len(sub) < max_lag * 3 + 30 or sub.shape[1] < 2:
        out["status"] = "not_enough_data"
        out["message"] = "Not enough consecutive readings from one turbine."
        return out

    # Make each series steady first (trending data gives fake 'causes').
    stat = sub.copy()
    for col in sub.columns:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                not_steady = adfuller(sub[col].to_numpy())[1] > 0.05
            if not_steady:
                stat[col] = sub[col].diff()
                out["differenced"].append(col)
        except Exception:
            pass
    stat = stat.dropna()
    stat = stat.loc[:, stat.std() > 1e-8]
    if len(stat) < max_lag * 3 + 30 or stat.shape[1] < 2:
        out["status"] = "not_enough_data"
        out["message"] = "Not enough usable readings after making the series steady."
        return out
    out["rows_used"] = int(len(stat))

    cm = stat.corr().abs().fillna(0.0)
    cols = list(stat.columns)
    cand = sorted(((cm.loc[a, b], a, b) for i, a in enumerate(cols) for b in cols[i + 1:]), reverse=True)
    cand = cand[: s["granger_pairs"] * 2]
    n_tests = max(1, len(cand) * 2)
    out["p_threshold"] = s["granger_alpha"] / n_tests
    lags = list(range(1, max_lag + 1))

    def run(cause, effect):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(stat[[effect, cause]].values, maxlag=max_lag)
        f = [float(res[l][0]["ssr_ftest"][0]) for l in lags]
        p = [float(res[l][0]["ssr_ftest"][1]) for l in lags]
        return f, p

    results = []
    for _, a, b in cand:
        best = None
        for cause, effect in ((a, b), (b, a)):
            try:
                f, p = run(cause, effect)
            except Exception as e:
                logger.warning(f"Granger failed {cause}->{effect}: {e}")
                continue
            mp = min(p)
            if best is None or mp < best["min_p"]:
                best = {"cause": cause, "effect": effect,
                        "f_stats": [round(v, 3) for v in f],
                        "p_values": [float(f"{v:.3g}") for v in p],
                        "min_p": float(f"{mp:.3g}"), "best_lag": lags[int(np.argmin(p))],
                        "significant": bool(mp < out["p_threshold"])}
        if best:
            results.append(best)
    results.sort(key=lambda r: r["min_p"])
    out["lags"] = lags
    out["pairs"] = results[: s["granger_pairs"]]
    return out


def compute_network_block(sub: pd.DataFrame) -> dict:
    sub = sub.dropna(thresh=int(sub.shape[1] * 0.7))
    if sub.shape[0] < 30 or sub.shape[1] < 3:
        return {"nodes": [], "edges": []}
    filled = _cap(_fill(sub))
    tags = list(filled.columns)
    try:
        X = StandardScaler().fit_transform(filled.values)
        model = GraphicalLassoCV(cv=2, alphas=4, max_iter=100)
        model.fit(X)
        precision = model.precision_
        d = np.sqrt(np.diag(precision))
        partial = -precision / np.outer(d, d)
        np.fill_diagonal(partial, 0)
        edges = []
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                w = float(partial[i, j])
                if abs(w) > SETTINGS["edge_threshold"]:
                    edges.append({"source": tags[i], "target": tags[j], "weight": round(w, 3)})
        return {"nodes": [{"id": t} for t in tags], "edges": edges}
    except Exception as e:
        logger.error(f"Graphical lasso failed: {e}")
        return {"nodes": [{"id": t} for t in tags], "edges": []}


# ============================================================
# 4. ENTRY POINT
# ============================================================
def _empty_payload(status, message, turbine_id, turbines=None):
    return {
        "status": status, "message": message, "turbine_id": turbine_id,
        "turbines": turbines or [],
        "correlation": _empty_corr(),
        "drift": {"status": "not_enough_baseline", "tags": [], "delta": [], "top_changes": [],
                  "mean_change": None, "windows": {}},
        "shap": {"target": None, "features": [], "values": [], "direction": [], "r2": None,
                 "available_targets": []},
        "granger": {"lags": [], "pairs": [], "turbine_used": None, "differenced": [],
                    "rows_used": 0, "p_threshold": None, "status": "not_enough_data"},
        "network": {"nodes": [], "edges": []},
    }


def _clean(obj):
    """Make the payload safe for JSON: numpy numbers -> python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _safe_build(turbine_id=None, days=7, limit=5000, baseline_days=30, target=None) -> dict:
    try:
        _import_heavy()
        return _clean(_build(turbine_id, days, limit, baseline_days, target))
    except Exception as e:
        logger.exception("Dependency engine failed")
        return _empty_payload("error", str(e), turbine_id)


_RESULT_MARKER = "@@DEP_RESULT@@"


def _run_isolated(args: dict) -> dict:
    """Run the analysis in its own Python process. A crash there cannot take the server down."""
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ)
    env["PYTHONPATH"] = backend_dir + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-X", "faulthandler", "-m", "app.analytics.dependency_pipeline"]
    try:
        done = subprocess.run(cmd, input=json.dumps(args), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", cwd=backend_dir, env=env,
                              timeout=SETTINGS["isolate_timeout"])
    except subprocess.TimeoutExpired:
        msg = f"The analysis took longer than {SETTINGS['isolate_timeout']} seconds and was stopped. Try fewer days."
        logger.error(msg)
        return _empty_payload("error", msg, args.get("turbine_id"))

    for line in reversed((done.stdout or "").splitlines()):
        if line.startswith(_RESULT_MARKER):
            try:
                return json.loads(line[len(_RESULT_MARKER):])
            except Exception as e:
                logger.error(f"Could not read the analysis result: {e}")
                break

    tail = "\n".join((done.stderr or "").strip().splitlines()[-25:])
    stages = [l for l in (done.stderr or "").splitlines() if "Dependency engine:" in l]
    last = stages[-1].split("Dependency engine:", 1)[1].strip() if stages else "before it started"
    logger.error(f"Dependency analysis process stopped (exit code {done.returncode}); last step: {last}\n{tail}")
    return _empty_payload(
        "error",
        f"The analysis process crashed (exit code {done.returncode}). Last step reached: {last}. "
        f"The details are in the backend log. The rest of the app is still running.",
        args.get("turbine_id"),
    )


def build_dependency_engine_payload(turbine_id: str | None = None, days: int = 7, limit: int = 5000,
                                    baseline_days: int = 30, target: str | None = None) -> dict:
    args = {"turbine_id": turbine_id, "days": days, "limit": limit,
            "baseline_days": baseline_days, "target": target}
    if SETTINGS["isolate"]:
        return _run_isolated(args)
    return _safe_build(**args)


def _build(turbine_id, days, limit, baseline_days, target) -> dict:
    days = int(min(max(days, 1), 90))
    baseline_days = int(min(max(baseline_days, 1), 365))
    limit = int(min(max(limit, 500), 20000))
    turbine_id = turbine_id or None

    conn = get_db_connection()
    if not conn:
        return _empty_payload("error", "Database connection failed", turbine_id)

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            logger.info("Dependency engine: step 1/6 reading column list from the database")
            candidates = discover_candidate_columns(cur)
            if not candidates:
                return _empty_payload("error", f"No sensor columns found in {SCHEMA}.{TABLE}", turbine_id)

            logger.info(f"Dependency engine: found {len(candidates)} candidate columns; step 2/6 finding latest time")
            latest = _latest_time(cur, turbine_id)
            if latest is None:
                return _empty_payload("no_data", "No SCADA data available", turbine_id)
            turbines = list_turbines(cur, latest)

            logger.info(f"Dependency engine: latest={latest}; turbines={len(turbines)}; step 3/6 loading current window")
            cur_start = latest - timedelta(days=days)
            base_start = cur_start - timedelta(days=baseline_days)

            cur_raw = _fetch_window(cur, conn, candidates, cur_start, latest, turbine_id, limit)
            if cur_raw.empty:
                return _empty_payload("no_data", "No SCADA data in the selected window", turbine_id, turbines)

            logger.info(f"Dependency engine: loaded {len(cur_raw)} rows; step 4/6 choosing sensors")
            tag_cols = select_tag_columns(cur_raw, candidates)
            if len(tag_cols) < 2:
                return _empty_payload("no_data", "Fewer than 2 usable sensors found in this window",
                                      turbine_id, turbines)

            logger.info(f"Dependency engine: using {len(tag_cols)} sensors; step 5/6 loading earlier window")
            base_raw = _fetch_window(cur, conn, tag_cols, base_start, cur_start, turbine_id, limit,
                                     end_inclusive=False)

            # Granger needs consecutive readings from ONE turbine
            g_turbine = turbine_id
            if not g_turbine and "__turbine" in cur_raw.columns and cur_raw["__turbine"].notna().any():
                g_turbine = str(cur_raw["__turbine"].value_counts().idxmax())
            logger.info("Dependency engine: step 6/6 loading consecutive readings for 'who moves first'")
            g_raw = _fetch_series(cur, tag_cols, cur_start, latest, g_turbine, SETTINGS["granger_rows"])
    finally:
        conn.close()

    col2label = _unique_labels(tag_cols)
    cur_df = _to_label_frame(cur_raw, col2label)
    base_df = _to_label_frame(base_raw, col2label) if not base_raw.empty else pd.DataFrame(columns=cur_df.columns)
    g_df = _to_label_frame(g_raw, col2label) if not g_raw.empty else pd.DataFrame(columns=cur_df.columns)

    tags = list(cur_df.columns)
    if target not in tags:
        target = default_target(cur_df)

    windows = {
        "current": {"start": str(cur_start), "end": str(latest), "rows": int(len(cur_df))},
        "earlier": {"start": str(base_start), "end": str(cur_start), "rows": int(len(base_df))},
    }

    timings: dict = {}

    def timed(name, fn, *args):
        logger.info(f"Dependency engine: starting {name}")
        t0 = time.perf_counter()
        try:
            return fn(*args)
        finally:
            timings[name] = time.perf_counter() - t0
            logger.info(f"Dependency engine: finished {name} in {timings[name]:.1f}s")

    jobs = {
        "correlation": (compute_correlation_block, cur_df),
        "drift": (compute_drift_block, cur_df, base_df, windows),
        "network": (compute_network_block, cur_df),
        "shap": (compute_shap_block, cur_df, target),
        "granger": (compute_granger_block, g_df, g_turbine),
    }
    if SETTINGS["parallel"]:
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {k: ex.submit(timed, k, *v) for k, v in jobs.items()}
            results = {k: f.result() for k, f in futs.items()}
    else:
        results = {k: timed(k, *v) for k, v in jobs.items()}

    results["shap"]["available_targets"] = tags
    logger.info("Dependency engine timing - " + ", ".join(f"{k}: {v:.1f}s" for k, v in timings.items()))

    return {
        "status": "ok",
        "turbine_id": turbine_id,
        "turbines": turbines,
        "row_count": int(len(cur_df)),
        "days": days,
        "baseline_days": baseline_days,
        "tag_columns": {label: col for col, label in col2label.items()},
        **results,
    }


if __name__ == "__main__":
    # Child-process entry point (used by _run_isolated). Reads the options from stdin,
    # writes the finished payload as one marked line on stdout. Logs go to stderr.
    import faulthandler
    faulthandler.enable()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _args = json.loads(sys.stdin.read() or "{}")
    _result = _safe_build(**_args)
    sys.stdout.write("\n" + _RESULT_MARKER + json.dumps(_result) + "\n")
    sys.stdout.flush()