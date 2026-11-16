"""
Power Curve Analysis — Cluster vs Theoretical, via Gaussian Process Regression.

Fleet-wide GPR fit, with k-means clustering applied to the OPERATING POINTS
(wind_speed, active_power pairs) to separate distinct operating regimes
(curtailment, normal generation, near rated-power plateau) before fitting
a smooth curve with uncertainty bands.
"""
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

logger = logging.getLogger(__name__)

RATED_POWER_KW = 6500
CUT_IN_WS = 3
RATED_WS = 15


def theoretical_power(wind_speed: np.ndarray) -> np.ndarray:
    return np.where(
        wind_speed < CUT_IN_WS, 0.0,
        np.where(
            wind_speed >= RATED_WS, float(RATED_POWER_KW),
            RATED_POWER_KW * ((wind_speed - CUT_IN_WS) / (RATED_WS - CUT_IN_WS)) ** 3
        )
    )


def run_power_curve_gpr(records: list, n_clusters: int = 3, n_grid_points: int = 60):
    if not records:
        return None

    wind_speed = np.array([r['wind_speed'] for r in records], dtype=float)
    power = np.array([r['active_power'] for r in records], dtype=float)

    valid = (wind_speed >= 0) & (wind_speed <= 30) & (power > -50)
    wind_speed = wind_speed[valid]
    power = power[valid]

    if len(wind_speed) < 20:
        logger.warning("Insufficient points for GPR power curve fit")
        return None

    X_cluster = np.column_stack([wind_speed, power])
    X_norm = (X_cluster - X_cluster.mean(axis=0)) / X_cluster.std(axis=0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_norm)

    grid = np.linspace(0, 30, n_grid_points)
    clusters_out = []

    for c in range(n_clusters):
        mask = cluster_labels == c
        ws_c = wind_speed[mask]
        p_c = power[mask]

        if len(ws_c) < 10:
            continue

        max_fit_points = 800
        if len(ws_c) > max_fit_points:
            idx = np.random.choice(len(ws_c), max_fit_points, replace=False)
            ws_fit, p_fit = ws_c[idx], p_c[idx]
        else:
            ws_fit, p_fit = ws_c, p_c

        kernel = ConstantKernel(1.0) * RBF(length_scale=3.0) + WhiteKernel(noise_level=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=42)

        try:
            gpr.fit(ws_fit.reshape(-1, 1), p_fit)
            mean_pred, std_pred = gpr.predict(grid.reshape(-1, 1), return_std=True)
        except Exception as e:
            logger.error(f"GPR fit failed for cluster {c}: {e}")
            continue

        clusters_out.append({
            "cluster_id": int(c),
            "point_count": int(mask.sum()),
            "mean_wind_speed": float(ws_c.mean()),
            "mean_power": float(p_c.mean()),
            "scatter_sample": [
                {"wind_speed": float(w), "power": float(p)}
                for w, p in zip(ws_c[:300], p_c[:300])
            ],
            "gpr_mean": mean_pred.tolist(),
            "gpr_upper": (mean_pred + 1.96 * std_pred).tolist(),
            "gpr_lower": (mean_pred - 1.96 * std_pred).tolist(),
        })

    max_fit_points = 1500
    if len(wind_speed) > max_fit_points:
        idx = np.random.choice(len(wind_speed), max_fit_points, replace=False)
        ws_fit, p_fit = wind_speed[idx], power[idx]
    else:
        ws_fit, p_fit = wind_speed, power

    kernel = ConstantKernel(1.0) * RBF(length_scale=3.0) + WhiteKernel(noise_level=1.0)
    gpr_overall = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=42)
    gpr_overall.fit(ws_fit.reshape(-1, 1), p_fit)
    overall_mean, overall_std = gpr_overall.predict(grid.reshape(-1, 1), return_std=True)

    return {
        "grid_wind_speed": grid.tolist(),
        "theoretical_power": theoretical_power(grid).tolist(),
        "overall_gpr_mean": overall_mean.tolist(),
        "overall_gpr_upper": (overall_mean + 1.96 * overall_std).tolist(),
        "overall_gpr_lower": (overall_mean - 1.96 * overall_std).tolist(),
        "clusters": clusters_out,
        "total_points_used": int(len(wind_speed)),
    }