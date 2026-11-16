import React, { useEffect, useRef, useState, useCallback } from 'react';
import { getList, getAllStatus, getEnsembleStatus } from '../services/turbineApi';
import { getThermal } from '../services/thermalApi';
import { getEnsembleAnomaly } from '../services/mlApi';
import Chart from 'chart.js/auto';
import {
  FaThermometerHalf, FaExclamationTriangle,
  FaSpinner, FaClock,
  FaArrowUp, FaArrowDown, FaMinus, FaCheckCircle,
  FaTimesCircle, FaDatabase, FaWifi, FaBug, FaInfoCircle, FaQuestionCircle
} from 'react-icons/fa';
import '../App.css';

/**
 * ThermalPage
 *
 * Fixes applied vs. the original:
 * 1. Race-condition guard: switching turbines no longer shows stale/mismatched
 *    data. Each fetch is tagged with a request id; late responses from a
 *    turbine you've already navigated away from are dropped.
 * 2. Three distinct states instead of one blob: LOADING / NO DATA / ERROR /
 *    OK. A turbine with 0 sensors is no longer visually identical to a
 *    turbine that's still loading or one whose request failed.
 * 3. Polling hardening: in-flight guards prevent overlapping requests,
 *    failures back off instead of retrying every single interval at full
 *    fan-out, and polling pauses when the tab isn't visible.
 * 4. `onConnectionStatusChange` prop lets a parent/header component reflect
 *    the *real* connection state instead of a hardcoded "Live" badge.
 */
export default function ThermalPage({ onConnectionStatusChange, onLoadingChange }) {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  // ============================================================
  // STATE DECLARATIONS
  // ============================================================
  const [turbines, setTurbines] = useState([]);
  const [selectedTurbine, setSelectedTurbine] = useState('');

  // thermalData is null while loading OR on hard failure.
  // thermalStatus distinguishes those two + "ok" + "empty" (loaded but no sensors)
  const [thermalData, setThermalData] = useState(null);
  const [thermalStatus, setThermalStatus] = useState('idle'); // idle | loading | ok | empty | error
  const [thermalError, setThermalError] = useState(null);

  const [anomalyData, setAnomalyData] = useState([]);
  const [loadingAnomalies, setLoadingAnomalies] = useState(false);
  const [turbineAnomalyStatus, setTurbineAnomalyStatus] = useState({});
  const [turbineStatusMap, setTurbineStatusMap] = useState({});
  const [loadingStatuses, setLoadingStatuses] = useState(false);
  const [statusFilter, setStatusFilter] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [timeRange, setTimeRange] = useState('7d');

  const HIGH_CONFIDENCE_THRESHOLD = 3;
  const CARDS_PER_PAGE = 10;

  // Real mapping from the time-range buttons to actual query params.
  // Previously `timeRange` was tracked in state but never passed to
  // getThermal() — every range button silently did nothing. "All" pulls
  // everything available (anchored, same as the others, to this turbine's
  // own latest timestamp) with a larger row cap since a full history pull
  // needs more than 300 points to be meaningful.
  const RANGE_DAYS = { '24h': 1, '7d': 7, '30d': 30, 'all': 3650 };
  const RANGE_LIMIT = { '24h': 300, '7d': 300, '30d': 300, 'all': 5000 };
  const RANGE_LABEL = { '24h': 'the last 24 hours', '7d': 'the last 7 days', '30d': 'the last 30 days', 'all': 'the full available history' };

  const getHighConfidenceAnomalies = (anomalies) => {
    if (!anomalies || anomalies.length === 0) return [];
    return anomalies.filter(anom => {
      const votes = anom.ensemble_votes !== undefined ? anom.ensemble_votes :
                    (anom.model_count !== undefined ? anom.model_count : 0);
      return votes >= HIGH_CONFIDENCE_THRESHOLD;
    });
  };

  // Surface real connection status to a parent header/badge instead of a
  // hardcoded "Live" indicator that never reflects reality.
  useEffect(() => {
    if (typeof onConnectionStatusChange === 'function') {
      onConnectionStatusChange(connectionStatus);
    }
  }, [connectionStatus, onConnectionStatusChange]);

  // Tells App.jsx's navigation lock whether this page's initial data is
  // ready yet. `loading` covers the turbine-list fetch that gates the
  // whole page render below — once it's false (success OR error), it's
  // safe to let the user navigate away.
  useEffect(() => {
    if (typeof onLoadingChange === 'function') {
      onLoadingChange(loading);
    }
  }, [loading, onLoadingChange]);

  // ============================================================
  // TURBINE LIST
  // ============================================================
  useEffect(() => {
    let cancelled = false;
    const fetchTurbines = async () => {
      try {
        setLoading(true);
        setConnectionStatus('connecting');
        const data = await getList();
        if (cancelled) return;
        setTurbines(data.turbines || []);
        if (data.turbines && data.turbines.length > 0) {
          setSelectedTurbine(data.turbines[0]);
        }
        setConnectionStatus('connected');
      } catch (err) {
        if (cancelled) return;
        console.error('Error fetching turbines:', err);
        setConnectionStatus('error');
        setError('Cannot connect to backend server. Please ensure the server is running.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchTurbines();
    return () => { cancelled = true; };
  }, []);

  // ============================================================
  // TURBINE STATUS POLLING (grid cards)
  // ============================================================
  // in-flight guard so a slow request doesn't get re-fired on top of itself
  const statusFetchInFlight = useRef(false);
  // consecutive-failure counter drives simple backoff for the expensive
  // per-turbine fallback path, so one bad minute doesn't turn into N
  // requests stacking every 60s forever
  const statusFailureCount = useRef(0);

  const fetchIndividualStatuses = useCallback(async (turbineList) => {
    const results = await Promise.all(
      turbineList.map(async (id) => {
        try {
          const data = await getEnsembleStatus(id, 7);
          return {
            id,
            status: data.status || 'Unknown',
            health_score: data.health_score ?? null,
            sensorsCount: data.total_samples || 0,
            alertsCount: data.anomaly_count || 0,
            hasData: (data.total_samples || 0) > 0,
          };
        } catch (err) {
          console.error(`Error fetching status for ${id}:`, err);
          return { id, status: 'Error', health_score: null, sensorsCount: 0, alertsCount: 0, hasData: false };
        }
      })
    );

    const statusMap = {};
    results.forEach(r => {
      statusMap[r.id] = {
        status: r.status,
        health_score: r.health_score,
        sensorsCount: r.sensorsCount,
        alertsCount: r.alertsCount,
        hasData: r.hasData,
      };
    });
    setTurbineStatusMap(statusMap);
  }, []);

  const fetchAllTurbineStatuses = useCallback(async (turbineList) => {
    if (!turbineList || turbineList.length === 0) return;
    if (statusFetchInFlight.current) return; // don't overlap requests
    if (document.hidden) return; // don't poll a backgrounded tab

    statusFetchInFlight.current = true;
    setLoadingStatuses(true);

    try {
      const data = await getAllStatus();
      const statusMap = {};
      data.turbines.forEach(t => {
        const sensorsCount = t.total_samples || 0;
        statusMap[t.turbine_id] = {
          status: t.status || 'Unknown',
          health_score: t.health_score ?? null,
          sensorsCount,
          alertsCount: t.anomaly_count || 0,
          anomaly_ratio: t.anomaly_ratio || 0,
          hasData: sensorsCount > 0,
        };
      });
      setTurbineStatusMap(statusMap);

      const anomalyMap = {};
      data.turbines.forEach(t => {
        if (t.has_anomaly) anomalyMap[t.turbine_id] = true;
      });
      setTurbineAnomalyStatus(anomalyMap);
      statusFailureCount.current = 0;
    } catch (err) {
      console.error('Error fetching turbine statuses:', err);
      statusFailureCount.current += 1;

      // Only fall back to the expensive N-request path occasionally once
      // we're clearly in a failure streak, instead of every single 60s tick.
      const shouldFallback = statusFailureCount.current <= 1 || statusFailureCount.current % 3 === 0;
      if (shouldFallback) {
        await fetchIndividualStatuses(turbineList);
      }
    } finally {
      statusFetchInFlight.current = false;
      setLoadingStatuses(false);
    }
  }, [fetchIndividualStatuses]);

  const turbinesRef = useRef(turbines);
  useEffect(() => {
    turbinesRef.current = turbines;
  }, [turbines]);

  useEffect(() => {
    const fetchStatuses = () => fetchAllTurbineStatuses(turbinesRef.current);
    fetchStatuses();
    const interval = setInterval(fetchStatuses, 60000);
    return () => clearInterval(interval);
  }, [fetchAllTurbineStatuses]);

  // ============================================================
  // SELECTED TURBINE: THERMAL DATA (with race-condition guard)
  // ============================================================
  const thermalRequestId = useRef(0);

  useEffect(() => {
    if (!selectedTurbine) return;

    const requestId = ++thermalRequestId.current;
    let cancelled = false;

    // Clear stale data the instant the selection (turbine OR range) changes
    // so we never show turbine A's chart/health% under turbine B's name,
    // or a 7-day chart labeled as a 24-hour one.
    setThermalData(null);
    setThermalStatus('loading');
    setThermalError(null);

    const days = RANGE_DAYS[timeRange] ?? 7;
    const limit = RANGE_LIMIT[timeRange] ?? 300;

    const fetchThermalData = async () => {
      // don't flip back to "loading" on background refresh ticks once we
      // already have data for this turbine - avoids UI flicker
      if (thermalRequestId.current === requestId && thermalStatus !== 'ok' && thermalStatus !== 'empty') {
        setThermalStatus('loading');
      }

      try {
        const data = await getThermal(selectedTurbine, days, limit);
        if (cancelled || thermalRequestId.current !== requestId) return; // stale response

        const sensorCount = Object.keys(data?.thermal_metrics || {}).length;

        setThermalData(data);
        setThermalStatus(sensorCount > 0 ? 'ok' : 'empty');
        setThermalError(null);
        setLastUpdate(new Date());
        setConnectionStatus('connected');

        setTurbineStatusMap(prev => ({
          ...prev,
          [selectedTurbine]: {
            status: data.status || 'Unknown',
            health_score: data.health_score ?? null,
            sensorsCount: sensorCount,
            alertsCount: data.alerts?.length || 0,
            hasData: sensorCount > 0,
          }
        }));
      } catch (err) {
        if (cancelled || thermalRequestId.current !== requestId) return; // stale response
        console.error('Error fetching thermal data:', err);
        setThermalData(null);
        setThermalStatus('error');
        setThermalError(`Unable to load thermal data for ${selectedTurbine}. The backend may be unreachable or this turbine may not be reporting.`);
        setConnectionStatus('error');
      }
    };

    fetchThermalData();
    const interval = setInterval(() => {
      if (!document.hidden) fetchThermalData();
    }, 30000);

    return () => { cancelled = true; clearInterval(interval); };
  }, [selectedTurbine, timeRange]);

  // ============================================================
  // SELECTED TURBINE: ANOMALIES (also race-guarded)
  // ============================================================
  const anomalyRequestId = useRef(0);

  useEffect(() => {
    if (!selectedTurbine) return;

    const requestId = ++anomalyRequestId.current;
    let cancelled = false;
    setAnomalyData([]); // clear stale anomalies from the previous turbine

    const fetchAnomalies = async () => {
      if (document.hidden) return;
      setLoadingAnomalies(true);
      try {
        const data = await getEnsembleAnomaly(`?turbine_id=${selectedTurbine}&limit=200&min_score=0.80`);
        if (cancelled || anomalyRequestId.current !== requestId) return;
        setAnomalyData(data.anomalies || []);
      } catch (err) {
        if (cancelled || anomalyRequestId.current !== requestId) return;
        console.error('Error fetching anomalies:', err);
      } finally {
        if (!cancelled && anomalyRequestId.current === requestId) setLoadingAnomalies(false);
      }
    };

    fetchAnomalies();
    const interval = setInterval(fetchAnomalies, 60000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [selectedTurbine]);

  // ============================================================
  // CHART
  // ============================================================
  useEffect(() => {
    // Always tear down the previous chart instance first, whether or not
    // we're about to draw a new one - prevents a stale chart lingering
    // when a turbine has no sensors.
    if (chartInstance.current) {
      chartInstance.current.destroy();
      chartInstance.current = null;
    }

    if (thermalStatus !== 'ok' || !thermalData?.thermal_metrics || !chartRef.current) return;

    const ctx = chartRef.current.getContext('2d');
    const sensors = Object.keys(thermalData.thermal_metrics);
    const labels = sensors.map(s => thermalData.thermal_metrics[s].description || s);
    const currentValues = sensors.map(s => thermalData.thermal_metrics[s].current);
    const warningValues = sensors.map(s => thermalData.thermal_metrics[s].warning ?? null);
    const criticalValues = sensors.map(s => thermalData.thermal_metrics[s].critical ?? null);

    const colors = sensors.map(s => {
      const cat = thermalData.thermal_metrics[s].category || 'other';
      const value = thermalData.thermal_metrics[s].current;
      const warning = thermalData.thermal_metrics[s].warning;
      const critical = thermalData.thermal_metrics[s].critical;

      if (typeof critical === 'number' && value >= critical) return '#dc3545';
      if (typeof warning === 'number' && value >= warning) return '#ffc107';
      if (cat === 'gearbox') return '#0d6efd';
      if (cat === 'bearing') return '#198754';
      if (cat === 'generator') return '#6f42c1';
      return '#6c757d';
    });

    chartInstance.current = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Current Temperature (°C)',
            data: currentValues,
            backgroundColor: colors.map(c => c + '99'),
            borderColor: colors,
            borderWidth: 2,
            borderRadius: 4,
          },
          {
            label: 'Warning Threshold',
            data: warningValues,
            type: 'line',
            borderColor: '#ffc107',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
          },
          {
            label: 'Critical Threshold',
            data: criticalValues,
            type: 'line',
            borderColor: '#dc3545',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              font: { size: 11, weight: '500' },
              padding: 20,
              usePointStyle: true,
              pointStyle: 'line'
            }
          },
          tooltip: {
            backgroundColor: 'rgba(0,0,0,0.8)',
            titleFont: { size: 12, weight: '600' },
            bodyFont: { size: 11 },
            padding: 12,
            cornerRadius: 6,
            callbacks: {
              label: function(context) {
                const value = context.parsed.y;
                return `${context.dataset.label}: ${value.toFixed(1)}°C`;
              },
              afterLabel: function(context) {
                const sensor = sensors[context.dataIndex];
                const data = thermalData.thermal_metrics[sensor];
                if (data) {
                  const status = data.current >= data.critical ? 'CRITICAL' :
                                data.current >= data.warning ? 'WARNING' : 'NORMAL';
                  return [
                    `Status: ${status}`,
                    `Average: ${data.average.toFixed(1)}°C`,
                    `Maximum: ${data.max.toFixed(1)}°C`,
                    `Minimum: ${data.min.toFixed(1)}°C`,
                    data.is_anomaly ? 'ANOMALY DETECTED' : ''
                  ].filter(Boolean);
                }
                return null;
              }
            }
          }
        },
        scales: {
          x: {
            ticks: { maxRotation: 45, minRotation: 45, font: { size: 9 } },
            grid: { display: false }
          },
          y: {
            min: 0,
            max: 120,
            title: {
              display: true,
              text: 'Temperature (°C)',
              font: { size: 11, weight: '500' }
            },
            grid: { color: 'rgba(0,0,0,0.05)' }
          }
        }
      }
    });

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
        chartInstance.current = null;
      }
    };
  }, [thermalData, thermalStatus]);

  // ============================================================
  // DISPLAY HELPERS
  // ============================================================
  const getStatusColor = (status) => {
    if (status === 'Healthy') return '#198754';
    if (status === 'Warning') return '#ffc107';
    if (status === 'Monitor') return '#0d6efd';
    if (status === 'Error' || status === 'No Data') return '#6c757d';
    return '#dc3545';
  };

  const getStatusIcon = (status) => {
    if (status === 'Healthy') return FaCheckCircle;
    if (status === 'Warning') return FaExclamationTriangle;
    if (status === 'Monitor') return FaInfoCircle;
    if (status === 'Error' || status === 'No Data') return FaQuestionCircle;
    return FaTimesCircle;
  };

  // Prefer the backend's own rate_of_change (computed in pipeline.py from
  // the actual first-10 vs last-10 sample comparison over the full fetched
  // window) over re-deriving a trend from just current-vs-average here —
  // the backend value is the authoritative one and won't disagree with it.
  const getTrend = (sensor) => {
    const rateOfChange = sensor?.rate_of_change;
    if (typeof rateOfChange !== 'number') {
      return { icon: FaQuestionCircle, color: '#6c757d', label: 'Insufficient history' };
    }
    if (rateOfChange > 1.0) return { icon: FaArrowUp, color: '#dc3545', label: 'Rising' };
    if (rateOfChange < -1.0) return { icon: FaArrowDown, color: '#198754', label: 'Falling' };
    return { icon: FaMinus, color: '#6c757d', label: 'Stable' };
  };

  const getDropdownStatusBadgeStyle = (status) => {
    if (status === 'Critical') return { background: '#f8d7da', color: '#721c24' };
    if (status === 'Warning') return { background: '#fff3cd', color: '#856404' };
    if (status === 'Monitor') return { background: '#cfe2ff', color: '#084298' };
    if (status === 'Healthy') return { background: '#d1e7dd', color: '#0f5132' };
    if (status === 'Error') return { background: '#f8d7da', color: '#721c24' };
    return { background: '#e9ecef', color: '#495057' }; // No Data / Unknown
  };

  // A card is "No Data" (not "Unknown") when we've actually heard back from
  // the backend and it reported zero sensors - distinct from still-loading.
  const displayStatusFor = (id) => {
    const s = turbineStatusMap[id];
    if (!s) return 'Loading…';
    if (s.status === 'Error') return 'Error';
    if (!s.hasData) return 'No Data';
    return s.status;
  };

  const filterCounts = {
    all: turbines.length,
    healthy: turbines.filter(id => turbineStatusMap[id]?.status === 'Healthy').length,
    monitor: turbines.filter(id => turbineStatusMap[id]?.status === 'Monitor').length,
    warning: turbines.filter(id => turbineStatusMap[id]?.status === 'Warning').length,
    critical: turbines.filter(id => turbineStatusMap[id]?.status === 'Critical').length,
    anomaly: turbines.filter(id => turbineAnomalyStatus[id]).length,
  };

  const filterTabs = [
    { key: 'all', label: 'All', count: filterCounts.all },
    { key: 'healthy', label: 'Healthy', count: filterCounts.healthy },
    { key: 'monitor', label: 'Monitor', count: filterCounts.monitor },
    { key: 'warning', label: 'Warning', count: filterCounts.warning },
    { key: 'critical', label: 'Critical', count: filterCounts.critical },
    { key: 'anomaly', label: 'Anomaly', count: filterCounts.anomaly },
  ];

  const filteredTurbines = turbines.filter(id => {
    if (statusFilter === 'all') return true;
    if (statusFilter === 'anomaly') return !!turbineAnomalyStatus[id];
    const statusLabel = statusFilter.charAt(0).toUpperCase() + statusFilter.slice(1);
    return turbineStatusMap[id]?.status === statusLabel;
  });

  const totalPages = Math.max(1, Math.ceil(filteredTurbines.length / CARDS_PER_PAGE));
  const paginatedTurbines = filteredTurbines.slice(
    (currentPage - 1) * CARDS_PER_PAGE,
    currentPage * CARDS_PER_PAGE
  );

  // ============================================================
  // FULL-PAGE STATES
  // ============================================================
  if (connectionStatus === 'error' && !thermalData && turbines.length === 0) {
    return (
      <div className="error-container">
        <div className="error-box">
          <FaWifi size={60} color="#dc3545" style={{ opacity: 0.4 }} />
          <h2 className="error-title">Connection Error</h2>
          <p className="error-text">
            {error || 'Cannot connect to the backend server. Please ensure:'}
          </p>
          <ul style={{ textAlign: 'left', color: '#6c757d', marginTop: '15px' }}>
            <li>Backend server is running on port 8000</li>
            <li>FastAPI server is started with: <code>uvicorn main:app --reload</code></li>
            <li>No firewall blocking the connection</li>
          </ul>
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: '20px',
              padding: '10px 24px',
              background: '#0d6efd',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: 500
            }}
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', background: '#f8f9fa' }}>
        <div style={{ textAlign: 'center' }}>
          <FaSpinner size={40} className="spin" color="#0d6efd" />
          <p style={{ marginTop: '16px', color: '#6c757d', fontWeight: '500' }}>Loading turbine data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page" style={{ padding: '20px', backgroundColor: '#f4f6f8' }}>

      {/* Header */}
      <div className="header-container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div className="header-title-group" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="header-icon-box" style={{ background: '#e7f1ff', padding: '10px', borderRadius: '8px', color: '#0d6efd' }}>
            <FaThermometerHalf size={24} />
          </div>
          <div>
            <h1 className="header-title" style={{ fontSize: '20px', fontWeight: 700, margin: 0 }}>Thermal Monitoring</h1>
            <div className="header-meta" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: '#6c757d', marginTop: '4px' }}>
              <FaClock size={12} />
              <span>Last updated: {lastUpdate ? lastUpdate.toLocaleTimeString() : '—'}</span>
              <span>
                <FaDatabase size={12} color={connectionStatus === 'connected' ? '#198754' : '#dc3545'} />
                <span style={{ marginLeft: '4px', color: connectionStatus === 'connected' ? '#198754' : '#dc3545' }}>
                  {connectionStatus === 'connected' ? 'Live' : connectionStatus === 'connecting' ? 'Connecting…' : 'Disconnected'}
                </span>
              </span>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
          <div className="time-range-group">
            {['24h', '7d', '30d', 'all'].map(range => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`time-range-btn ${timeRange === range ? 'active' : 'inactive'}`}
              >
                {range === 'all' ? 'All' : range}
              </button>
            ))}
          </div>

          <div className="time-range-group">
            {filterTabs.map(tab => (
              <button
                key={tab.key}
                onClick={() => setStatusFilter(tab.key)}
                className={`time-range-btn ${statusFilter === tab.key ? 'active' : 'inactive'}`}
              >
                {tab.label} ({tab.count})
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Non-blocking error banner - a single failed request no longer hides the whole page */}
      {error && connectionStatus === 'error' && turbines.length > 0 && (
        <div style={{
          background: '#fff3cd', border: '1px solid #ffe69c', color: '#664d03',
          borderRadius: '8px', padding: '10px 14px', marginBottom: '16px',
          fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px'
        }}>
          <FaExclamationTriangle size={14} />
          <span>{error}</span>
        </div>
      )}

      {/* Turbine Cards Grid */}
      <div className="panel" style={{ background: '#fff', padding: '16px', borderRadius: '10px', marginBottom: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '8px' }}>
          <span style={{ fontWeight: 600, fontSize: '15px', color: '#1a1a1a' }}>
            Turbines — {filterTabs.find(t => t.key === statusFilter)?.label} ({filteredTurbines.length})
            {loadingStatuses && <FaSpinner className="spin" size={12} style={{ marginLeft: '8px', color: '#6c757d' }} />}
          </span>
          <span style={{ fontSize: '12px', color: '#6c757d' }}>
            Click a card to view full detail below
          </span>
        </div>

        {filteredTurbines.length === 0 ? (
          <div style={{ padding: '30px', textAlign: 'center', color: '#6c757d', background: '#f8f9fa', borderRadius: '8px' }}>
            No turbines match this filter.
          </div>
        ) : (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '12px' }}>
              {paginatedTurbines.map(id => {
                const turbineStatus = turbineStatusMap[id];
                const label = displayStatusFor(id);
                const badgeStyle = getDropdownStatusBadgeStyle(label === 'Loading…' ? 'Unknown' : label);
                const hasData = turbineStatus?.hasData;
                const healthScore = hasData ? (turbineStatus?.health_score ?? 0) : null;
                const barColor = healthScore == null ? '#e9ecef' : healthScore > 80 ? '#198754' : healthScore > 60 ? '#ffc107' : '#dc3545';

                return (
                  <div
                    key={id}
                    onClick={() => setSelectedTurbine(id)}
                    style={{
                      border: `1px solid ${selectedTurbine === id ? '#0d6efd' : '#e9ecef'}`,
                      borderRadius: '10px',
                      padding: '14px',
                      cursor: 'pointer',
                      background: selectedTurbine === id ? '#e7f1ff' : '#fff',
                      transition: 'all 0.15s'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: 700, fontSize: '14px', color: '#1a1a1a' }}>{id}</span>
                      <span style={{ padding: '2px 8px', borderRadius: '12px', fontSize: '10px', fontWeight: 600, background: badgeStyle.background, color: badgeStyle.color }}>
                        {label}
                      </span>
                    </div>

                    <div style={{ fontSize: '11px', color: '#6c757d', marginTop: '8px' }}>
                      Health Score: {healthScore == null ? 'N/A' : `${healthScore}%`}
                    </div>
                    <div className="health-bar-track" style={{ height: '6px', background: '#e9ecef', borderRadius: '3px', marginTop: '4px', overflow: 'hidden' }}>
                      <div className="health-bar-fill" style={{ height: '100%', width: `${healthScore ?? 0}%`, background: barColor }} />
                    </div>

                    <div style={{ display: 'flex', gap: '10px', marginTop: '10px', fontSize: '11px', color: '#6c757d', flexWrap: 'wrap' }}>
                      <span>{turbineStatus?.sensorsCount ?? 0} sensors</span>
                      <span>{turbineStatus?.alertsCount ?? 0} alerts</span>
                      {turbineAnomalyStatus[id] && (
                        <span className="alert-badge anomaly" style={{ display: 'flex', alignItems: 'center', gap: '3px', color: '#dc3545' }}>
                          <FaBug size={10} />
                          <span>Anomaly</span>
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {totalPages > 1 && (
              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '12px', marginTop: '18px' }}>
                <button
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  style={{ padding: '6px 14px', borderRadius: '6px', border: '1px solid #dee2e6', background: currentPage === 1 ? '#f8f9fa' : '#fff', color: currentPage === 1 ? '#adb5bd' : '#1a1a1a', cursor: currentPage === 1 ? 'not-allowed' : 'pointer', fontSize: '12px', fontWeight: 500 }}
                >
                  Prev
                </button>
                <span style={{ fontSize: '12px', color: '#6c757d' }}>
                  Page {currentPage} of {totalPages}
                </span>
                <button
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  style={{ padding: '6px 14px', borderRadius: '6px', border: '1px solid #dee2e6', background: currentPage === totalPages ? '#f8f9fa' : '#fff', color: currentPage === totalPages ? '#adb5bd' : '#1a1a1a', cursor: currentPage === totalPages ? 'not-allowed' : 'pointer', fontSize: '12px', fontWeight: 500 }}
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* Main Content - Thermal Data: now has a real state machine instead of "thermalData ? show : nothing" */}
      {thermalStatus === 'loading' && (
        <div style={{ background: '#fff', borderRadius: '10px', padding: '60px 20px', textAlign: 'center', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
          <FaSpinner size={28} className="spin" color="#0d6efd" />
          <p style={{ marginTop: '12px', color: '#6c757d', fontSize: '13px' }}>Loading thermal data for {selectedTurbine}…</p>
        </div>
      )}

      {thermalStatus === 'error' && (
        <div style={{ background: '#fff', borderRadius: '10px', padding: '40px 20px', textAlign: 'center', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', border: '1px solid #f8d7da' }}>
          <FaTimesCircle size={28} color="#dc3545" />
          <p style={{ marginTop: '12px', color: '#721c24', fontSize: '13px', fontWeight: 500 }}>{thermalError}</p>
        </div>
      )}

      {thermalStatus === 'empty' && (
        <div style={{ background: '#fff', borderRadius: '10px', padding: '40px 20px', textAlign: 'center', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', border: '1px solid #e9ecef' }}>
          <FaQuestionCircle size={28} color="#6c757d" />
          <p style={{ marginTop: '12px', color: '#6c757d', fontSize: '13px' }}>
            No thermal sensor data for <strong>{selectedTurbine}</strong> in {RANGE_LABEL[timeRange] || 'this window'}.
          </p>
          {timeRange !== 'all' && (
            <button
              onClick={() => setTimeRange('all')}
              style={{
                marginTop: '14px', padding: '8px 20px', borderRadius: '6px',
                border: '1px solid #0d6efd', background: '#fff', color: '#0d6efd',
                cursor: 'pointer', fontSize: '13px', fontWeight: 500,
              }}
            >
              Show all available history instead
            </button>
          )}
        </div>
      )}

      {thermalStatus === 'ok' && thermalData && (
        <>
          {/* Summary Cards Grid */}
          <div className="cards-grid" style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '16px',
            marginBottom: '24px'
          }}>
            {/* Card 1: System Status */}
            <div className="summary-card" style={{
              background: '#fff',
              padding: '16px',
              borderRadius: '8px',
              borderLeft: `4px solid ${getStatusColor(thermalData.status)}`,
              boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
            }}>
              <div className="card-label" style={{ fontSize: '12px', color: '#6c757d', fontWeight: 600 }}>System Status</div>
              <div className="card-value-group" style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '8px' }}>
                {React.createElement(getStatusIcon(thermalData.status), {
                  size: 20,
                  color: getStatusColor(thermalData.status)
                })}
                <span className="card-value" style={{ fontSize: '18px', fontWeight: 700, color: getStatusColor(thermalData.status) }}>
                  {thermalData.status || 'Unknown'}
                </span>
              </div>
            </div>

            {/* Card 2: Health Score */}
            <div className="summary-card" style={{
              background: '#fff',
              padding: '16px',
              borderRadius: '8px',
              borderLeft: '4px solid #0d6efd',
              boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
            }}>
              <div className="card-label" style={{ fontSize: '12px', color: '#6c757d', fontWeight: 600 }}>Health Score</div>
              <div className="card-value" style={{
                fontSize: '22px',
                fontWeight: 700,
                marginTop: '4px',
                color: thermalData.health_score > 80 ? '#198754' : thermalData.health_score > 60 ? '#ffc107' : '#dc3545'
              }}>
                {thermalData.health_score ?? 0}%
              </div>
              <div className="health-bar-track" style={{ height: '4px', background: '#e9ecef', borderRadius: '2px', marginTop: '8px', overflow: 'hidden' }}>
                <div className="health-bar-fill" style={{
                  height: '100%',
                  width: `${thermalData.health_score || 0}%`,
                  background: thermalData.health_score > 80 ? '#198754' : thermalData.health_score > 60 ? '#ffc107' : '#dc3545'
                }} />
              </div>
            </div>

            {/* Card 3: Active Sensors */}
            <div className="summary-card" style={{
              background: '#fff',
              padding: '16px',
              borderRadius: '8px',
              borderLeft: '4px solid #6f42c1',
              boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
            }}>
              <div className="card-label" style={{ fontSize: '12px', color: '#6c757d', fontWeight: 600 }}>Sensors</div>
              <div className="card-value" style={{ fontSize: '22px', fontWeight: 700, marginTop: '4px', color: '#1a1a1a' }}>
                {thermalData.summary?.sensors_monitored || Object.keys(thermalData.thermal_metrics || {}).length || 0}
              </div>
              <div style={{ fontSize: '11px', color: '#6c757d', marginTop: '2px' }}>
                Active sensors
              </div>
            </div>

            {/* Card 4: Active Alerts */}
            <div className="summary-card" style={{
              background: '#fff',
              padding: '16px',
              borderRadius: '8px',
              borderLeft: `4px solid ${(thermalData.summary?.active_alerts || thermalData.alerts?.length || 0) > 0 ? '#dc3545' : '#198754'}`,
              boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
            }}>
              <div className="card-label" style={{ fontSize: '12px', color: '#6c757d', fontWeight: 600 }}>Active Alerts</div>
              <div className="card-value" style={{
                fontSize: '22px',
                fontWeight: 700,
                marginTop: '4px',
                color: (thermalData.summary?.active_alerts || thermalData.alerts?.length || 0) > 0 ? '#dc3545' : '#198754'
              }}>
                {thermalData.summary?.active_alerts || thermalData.alerts?.length || 0}
              </div>
              <div style={{ fontSize: '11px', color: '#6c757d', marginTop: '2px' }}>
                Reported alerts
              </div>
            </div>
          </div>

          {/* Chart Panel */}
          <div className="panel" style={{ background: '#fff', padding: '16px', borderRadius: '10px', marginTop: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <span style={{ fontWeight: 600, fontSize: '16px', color: '#1a1a1a' }}>
                Thermal Sensor Readings — Turbine {selectedTurbine}
              </span>
              <span style={{ fontSize: '12px', color: '#6c757d' }}>
                Showing current temperatures against warning & critical limits
              </span>
            </div>
            <div style={{ height: '380px', position: 'relative' }}>
              <canvas ref={chartRef}></canvas>
            </div>
          </div>

          {/* Sensor Breakdown Section */}
          <div className="panel" style={{ background: '#fff', padding: '16px', borderRadius: '10px', marginTop: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <span style={{ fontWeight: 600, fontSize: '16px', color: '#1a1a1a' }}>
                Sensor Breakdown & Trend Analysis
              </span>
              <span style={{ fontSize: '12px', color: '#6c757d' }}>
                Detailed tracking of patterns, averages, and anomalies
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '14px' }}>
              {Object.keys(thermalData.thermal_metrics || {}).map(sensorKey => {
                const sensor = thermalData.thermal_metrics[sensorKey];
                const trend = getTrend(sensor);
                const TrendIcon = trend.icon;
                const isCrit = typeof sensor.critical === 'number' && sensor.current >= sensor.critical;
                const isWarn = typeof sensor.warning === 'number' && sensor.current >= sensor.warning;
                const isAnomaly = sensor.is_anomaly === true;
                const sensorStatus = isCrit ? 'CRITICAL' : isWarn ? 'WARNING' : isAnomaly ? 'ANOMALY' : 'NORMAL';

                return (
                  <div key={sensorKey} style={{
                    background: '#f8f9fa',
                    border: `1px solid ${isCrit || isAnomaly ? '#dc3545' : isWarn ? '#ffc107' : '#e9ecef'}`,
                    borderRadius: '8px',
                    padding: '14px'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: 600, fontSize: '13px', color: '#1a1a1a' }}>
                        {sensor.description || sensorKey}
                      </span>
                      <span style={{
                        fontSize: '10px',
                        padding: '2px 6px',
                        borderRadius: '4px',
                        background: isCrit || isAnomaly ? '#f8d7da' : isWarn ? '#fff3cd' : '#e9ecef',
                        color: isCrit || isAnomaly ? '#721c24' : isWarn ? '#856404' : '#495057',
                        fontWeight: 600
                      }}>
                        {sensorStatus}
                      </span>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginTop: '10px' }}>
                      <span style={{ fontSize: '20px', fontWeight: 700, color: '#1a1a1a' }}>
                        {sensor.current?.toFixed(1)}°C
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px', color: trend.color }}>
                        <TrendIcon size={12} />
                        <span>{trend.label}</span>
                      </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginTop: '12px', fontSize: '11px', color: '#6c757d', background: '#fff', padding: '8px', borderRadius: '6px' }}>
                      <div>Avg: <strong>{sensor.average?.toFixed(1)}°C</strong></div>
                      <div>Max: <strong>{sensor.max?.toFixed(1)}°C</strong></div>
                      <div>Min: <strong>{sensor.min?.toFixed(1)}°C</strong></div>
                      <div>Warning limit: <strong>{typeof sensor.warning === 'number' ? `${sensor.warning}°C` : 'Unavailable'}</strong></div>
                      <div>Critical limit: <strong>{typeof sensor.critical === 'number' ? `${sensor.critical}°C` : 'Unavailable'}</strong></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}