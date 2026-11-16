// LEGACY — not used by any page (no component imports this file).
// It imports `axios` (absent from package.json) and uses CRA-style
// process.env, so it cannot run under Vite as-is. Its three endpoints
// (/fleet/status, /models/predict, /prognostics/rul/{id}) do NOT exist
// on the backend. Kept as-is; deferred to the API-contract fix phase.
// Live API access goes through src/services/api/*.

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fetchFleetStatus = async () => {
  const response = await api.get('/fleet/status');
  return response.data;
};

export const fetchModelPredictions = async (wtgId) => {
  const response = await api.post('/models/predict', { wtg_id: wtgId });
  return response.data;
};

export const fetchRulEstimates = async (wtgId) => {
  const response = await api.get(`/prognostics/rul/${wtgId}`);
  return response.data;
};

export default api;