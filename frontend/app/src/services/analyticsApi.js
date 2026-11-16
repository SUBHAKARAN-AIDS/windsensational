// services/analyticsApi.js
import apiClient from './client';
import { ENDPOINTS } from './endpoints';

export const getLiveTelemetry = async () => {
  const response = await apiClient.get(ENDPOINTS.turbines.live);
  return response.data;
};

export const getEnsembleAnomalyData = async (params = {}) => {
  const query = new URLSearchParams({
    days: String(params.days ?? 30),
    limit: String(params.limit ?? 500),
    min_score: String(params.min_score ?? 0),
  });
  if (params.turbine_id) query.set('turbine_id', params.turbine_id);
  const response = await apiClient.get(`${ENDPOINTS.ml.ensembleAnomaly}?${query.toString()}`);
  return response.data;
};

/**
 * Analytics API Service
 * All analytics API calls
 */
export const analyticsApi = {
  /**
   * Get power curve data
   */
  getPowerCurve: async (turbineId = null) => {
    try {
      let url = ENDPOINTS.analytics.powerCurve;
      if (turbineId) {
        url += `?turbine_id=${turbineId}`;
      }
      const response = await apiClient.get(url);
      return response.data;
    } catch (error) {
      console.error('Error fetching power curve:', error);
      throw error;
    }
  },

  /**
   * Get dependency engine data
   */
  getDependencyEngine: async (params = {}) => {
    const { turbine_id = null, days = 7, limit = 5000 } = params;
    
    let url = `${ENDPOINTS.analytics.dependencyEngine}?days=${days}&limit=${limit}`;
    if (turbine_id) {
      url += `&turbine_id=${turbine_id}`;
    }
    
    try {
      const response = await apiClient.get(url);
      return response.data;
    } catch (error) {
      console.error('Error fetching dependency engine:', error);
      throw error;
    }
  },
};

export const getPowerCurve = analyticsApi.getPowerCurve;
export const getDependencyEngine = analyticsApi.getDependencyEngine;

export default analyticsApi;