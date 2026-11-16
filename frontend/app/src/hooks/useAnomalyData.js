// hooks/useAnomalyData.js
import { useState, useEffect, useCallback } from 'react';
import { getEnsembleAnomaly } from '../services/mlApi';

export function useAnomalyData(turbineId, options = {}) {
  const { days = 7, limit = 500, min_score = 0.0, autoFetch = true } = options;
  
  const [data, setData] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const params = { days, limit, min_score };
      if (turbineId) {
        params.turbine_id = turbineId;
      }
      
      const result = await getEnsembleAnomaly(params);
      setData(result);
      setAnomalies(result.anomalies || []);
    } catch (err) {
      setError(err.message || 'Failed to fetch anomaly data');
    } finally {
      setLoading(false);
    }
  }, [turbineId, days, limit, min_score]);

  useEffect(() => {
    if (autoFetch) {
      fetchData();
    }
  }, [autoFetch, fetchData]);

  return { data, anomalies, loading, error, refetch: fetchData };
}