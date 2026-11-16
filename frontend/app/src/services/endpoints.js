// src/services/endpoints.js

const API_VERSION = '/api/v1';

export const ENDPOINTS = {
  // ─── Turbine Endpoints ─────────────────────────────────────
  turbines: {
    list: `${API_VERSION}/turbines/list`,
    live: `${API_VERSION}/turbines/live`,
    detail: (id) => `${API_VERSION}/turbines/${id}/latest`,           // ✅ Fixed
    status: (id) => `${API_VERSION}/turbines/${id}/status`,           // ✅ Fixed
    ensembleStatus: (id) => `${API_VERSION}/turbines/${id}/ensemble-status`, // ✅ Fixed
    allStatus: `${API_VERSION}/turbines/all-status`,
  },

  // ─── ML Endpoints ──────────────────────────────────────────
  ml: {
    modelsStatus: `${API_VERSION}/ml/models/status`,
    modelsReload: `${API_VERSION}/ml/models/reload`,
    ensembleAnomaly: `${API_VERSION}/ml/ensemble-anomaly-detection`,
    isolationForest: `${API_VERSION}/ml/isolation-forest-chart`,
    autoencoder: `${API_VERSION}/ml/autoencoder-chart`,
    mahalanobis: `${API_VERSION}/ml/mahalanobis-chart`,
    tsne: `${API_VERSION}/ml/tsne-chart`,
    cusum: `${API_VERSION}/ml/cusum-chart`,
  },

  // ─── Analytics Endpoints ──────────────────────────────────
  analytics: {
    powerCurve: `${API_VERSION}/analytics/power-curve`,
    dependencyEngine: `${API_VERSION}/analytics/dependency-engine`,
  },

  // ─── Thermal Endpoints ────────────────────────────────────
  thermal: {
    data: (turbineId) => `${API_VERSION}/thermal/${turbineId}`,
    summary: `${API_VERSION}/thermal/summary`,
  },

  // ─── Health ────────────────────────────────────────────────
  health: {
    check: '/health',
  },
};