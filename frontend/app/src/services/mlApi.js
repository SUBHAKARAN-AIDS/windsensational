// services/mlApi.js
import apiClient from './client';
import { ENDPOINTS } from './endpoints';

/**
 * ML API Service
 * All machine learning model API calls
 */
export const mlApi = {
  /**
   * Get status of all loaded models
   */
  getModelsStatus: async () => {
    try {
      const response = await apiClient.get(ENDPOINTS.ml.modelsStatus);
      return response.data;
    } catch (error) {
      console.error('Error fetching models status:', error);
      throw error;
    }
  },

  /**
   * Reload ML models
   */
  reloadModels: async () => {
    try {
      const response = await apiClient.post(ENDPOINTS.ml.modelsReload);
      return response.data;
    } catch (error) {
      console.error('Error reloading models:', error);
      throw error;
    }
  },

  /**
   * Get ensemble anomaly detection results
   */
  getEnsembleAnomaly: async (params = {}) => {
    const { turbine_id = null, days = 7, limit = 500, min_score = 0.0 } = params;
    
    let url = `${ENDPOINTS.ml.ensembleAnomaly}?days=${days}&limit=${limit}&min_score=${min_score}`;
    if (turbine_id) {
      url += `&turbine_id=${turbine_id}`;
    }
    
    try {
      const response = await apiClient.get(url);
      return response.data;
    } catch (error) {
      console.error('Error fetching ensemble anomaly:', error);
      throw error;
    }
  },

  /**
   * Get Isolation Forest chart data
   */
  getIsolationForestChart: async (params = {}) => {
    const { turbine_id = null, days = 7, limit = 500 } = params;
    
    let url = `${ENDPOINTS.ml.isolationForest}?days=${days}&limit=${limit}`;
    if (turbine_id) {
      url += `&turbine_id=${turbine_id}`;
    }
    
    try {
      const response = await apiClient.get(url);
      return response.data;
    } catch (error) {
      console.error('Error fetching Isolation Forest chart:', error);
      throw error;
    }
  },

  /**
   * Get Autoencoder chart data
   */
  getAutoencoderChart: async (params = {}) => {
    const { turbine_id = null, days = 7, limit = 500 } = params;
    
    let url = `${ENDPOINTS.ml.autoencoder}?days=${days}&limit=${limit}`;
    if (turbine_id) {
      url += `&turbine_id=${turbine_id}`;
    }
    
    try {
      const response = await apiClient.get(url);
      return response.data;
    } catch (error) {
      console.error('Error fetching Autoencoder chart:', error);
      throw error;
    }
  },

  /**
   * Get Mahalanobis chart data
   */
  getMahalanobisChart: async (params = {}) => {
    const { turbine_id = null, days = 7, limit = 500 } = params;
    
    let url = `${ENDPOINTS.ml.mahalanobis}?days=${days}&limit=${limit}`;
    if (turbine_id) {
      url += `&turbine_id=${turbine_id}`;
    }
    
    try {
      const response = await apiClient.get(url);
      return response.data;
    } catch (error) {
      console.error('Error fetching Mahalanobis chart:', error);
      throw error;
    }
  },

  /**
   * Get t-SNE chart data
   */
  getTsneChart: async (params = {}) => {
    const { turbine_id = null, days = 7, limit = 300 } = params;
    
    let url = `${ENDPOINTS.ml.tsne}?days=${days}&limit=${limit}`;
    if (turbine_id) {
      url += `&turbine_id=${turbine_id}`;
    }
    
    try {
      const response = await apiClient.get(url);
      return response.data;
    } catch (error) {
      console.error('Error fetching t-SNE chart:', error);
      throw error;
    }
  },

  /**
   * Get CUSUM chart data
   */
  getCusumChart: async (params = {}) => {
    const { turbine_id = null, days = 7, limit = 500 } = params;
    
    let url = `${ENDPOINTS.ml.cusum}?days=${days}&limit=${limit}`;
    if (turbine_id) {
      url += `&turbine_id=${turbine_id}`;
    }
    
    try {
      const response = await apiClient.get(url);
      return response.data;
    } catch (error) {
      console.error('Error fetching CUSUM chart:', error);
      throw error;
    }
  },
};

// ─── Individual exports for convenience ─────────────────────
export const getModelsStatus = mlApi.getModelsStatus;
export const reloadModels = mlApi.reloadModels;
export const getEnsembleAnomaly = mlApi.getEnsembleAnomaly;
export const getIsolationForestChart = mlApi.getIsolationForestChart;
export const getAutoencoderChart = mlApi.getAutoencoderChart;
export const getMahalanobisChart = mlApi.getMahalanobisChart;
export const getTsneChart = mlApi.getTsneChart;
export const getCusumChart = mlApi.getCusumChart;

export default mlApi;