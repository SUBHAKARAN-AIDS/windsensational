// hooks/useThermalData.js
import { useState, useEffect, useCallback } from 'react';
import { getThermal } from '../services/thermalApi';

export function useThermalData(turbineId, options = {}) {
  const { days = 7, limit = 300, autoFetch = true } = options;
  
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  const fetchData = useCallback(async () => {
    if (!turbineId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await getThermal(turbineId, { days, limit });
      setData(result);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err.message || 'Failed to fetch thermal data');
    } finally {
      setLoading(false);
    }
  }, [turbineId, days, limit]);

  useEffect(() => {
    if (autoFetch && turbineId) {
      fetchData();
    }
  }, [autoFetch, fetchData, turbineId]);

  return { data, loading, error, lastUpdate, refetch: fetchData };
}