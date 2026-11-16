// services/turbineApi.js
import apiClient from './client';
import { ENDPOINTS } from './endpoints';

/**
 * Turbine API Service
 * All turbine-related API calls
 */
export const turbineApi = {
  /**
   * Get list of all turbines
   */
  getList: async () => {
    try {
      const response = await apiClient.get(ENDPOINTS.turbines.list);
      return response.data;
    } catch (error) {
      console.error('Error fetching turbine list:', error);
      throw error;
    }
  },

  /**
   * Get live turbine data
   */
  getLive: async () => {
    try {
      const response = await apiClient.get(ENDPOINTS.turbines.live);
      return response.data;
    } catch (error) {
      console.error('Error fetching live turbines:', error);
      throw error;
    }
  },

  /**
   * Get latest data for a specific turbine
   */
  getLatest: async (turbineId) => {
    try {
      const response = await apiClient.get(ENDPOINTS.turbines.detail(turbineId));
      return response.data;
    } catch (error) {
      console.error(`Error fetching latest for ${turbineId}:`, error);
      throw error;
    }
  },

  /**
   * Get status for a specific turbine
   */
  getStatus: async (turbineId, days = 7) => {
    try {
      const response = await apiClient.get(
        `${ENDPOINTS.turbines.status(turbineId)}?days=${days}`
      );
      return response.data;
    } catch (error) {
      console.error(`Error fetching status for ${turbineId}:`, error);
      throw error;
    }
  },

  /**
   * Get ensemble status for a specific turbine
   */
  getEnsembleStatus: async (turbineId, days = 7) => {
    try {
      const response = await apiClient.get(
        `${ENDPOINTS.turbines.ensembleStatus(turbineId)}?days=${days}`
      );
      return response.data;
    } catch (error) {
      console.error(`Error fetching ensemble status for ${turbineId}:`, error);
      throw error;
    }
  },

  /**
   * Get status for all turbines
   */
  getAllStatus: async () => {
    try {
      const response = await apiClient.get(ENDPOINTS.turbines.allStatus);
      return response.data;
    } catch (error) {
      console.error('Error fetching all statuses:', error);
      throw error;
    }
  },
};

// ─── Individual exports for convenience ─────────────────────
export const getList = turbineApi.getList;
export const getLiveTurbines = turbineApi.getLive;
export const getTurbineLatest = turbineApi.getLatest;
export const getTurbineStatus = turbineApi.getStatus;
export const getEnsembleStatus = turbineApi.getEnsembleStatus;
export const getAllStatus = turbineApi.getAllStatus;

export default turbineApi;