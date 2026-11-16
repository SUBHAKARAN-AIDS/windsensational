import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

/**
 * Get thermal analytics for a specific turbine
 * Hits: GET http://localhost:8000/api/v1/thermal/:turbineId?days=X&limit=Y
 */
export const getThermal = async (turbineId, days = 1, limit = 300) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/thermal/${turbineId}`, {
      params: { days, limit }
    });
    return response.data;
  } catch (error) {
    console.error(
      `Error fetching thermal data for ${turbineId}:`,
      error.response?.data || error.message
    );
    throw error;
  }
};