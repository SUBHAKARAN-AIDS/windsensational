import React, { useEffect, useRef, useState } from 'react';
import {
  FaShieldAlt, FaBrain, FaChartLine, FaDrawPolygon,
  FaWaveSquare, FaListUl
} from 'react-icons/fa';
import Chart from 'chart.js/auto';

const API_BASE = 'http://localhost:8000';

// ─── INSTRUMENT PANEL THEME ───────────────────────────────────────────
const THEME = {
  // Base Colors
  background: '#f4f5f7',            // Slightly warmer neutral gray
  panelBg: '#ffffff',
  hairline: '#d4d8dd',              // Thin, sharp borders
  headerBg: '#fafafa',

  // Typography
  textPrimary: '#1c1e21',           // Graphite ink
  textSecondary: '#4b5563',         // Dark gray
  textDim: '#9ca3af',               // Light gray

  // Signal Color (Deep Teal)
  accentPrimary: '#0F8B8D',         // The single signal color
  accentSecondary: '#0F8B8D',

  // Status Colors (Only for actual state)
  normal: '#10b981',                // Green
  normalBorder: '#059669',
  anomaly: '#dc2626',               // Red
  anomalyBorder: '#b91c1c',
  warn: '#d97706',                  // Amber
  
  // Chart Specifics
  score: '#0F8B8D',                 // Main plot line (Teal)
  threshold: 'rgba(220, 38, 38, 0.65)', // Sharp threshold line
  
  // Technical Radii
  radiusPanel: '6px',
  radiusChip: '2px',

  // Typography Families
  fontSans: '"IBM Plex Sans", sans-serif',
  fontMono: '"IBM Plex Mono", monospace',
};

export default function AnomalyPage({ onLoadingChange }) {
  // ─── REFS ──────────────────────────────────────────────────────────
  const isoChartRef = useRef(null);
  const autoencChartRef = useRef(null);
  const mahalChartRef = useRef(null);
  const tsneChartRef = useRef(null);
  const cusumChartRef = useRef(null);

  const isoChartInstance = useRef(null);
  const autoencChartInstance = useRef(null);
  const mahalChartInstance = useRef(null);
  const tsneChartInstance = useRef(null);
  const cusumChartInstance = useRef(null);

  const isInitialFetch = useRef(true);

  // ─── STATE ──────────────────────────────────────────────────────────
  const [chartData, setChartData] = useState({
    isolation_forest: null,
    autoencoder: null,
    mahalanobis: null,
    tsne: null,
    cusum: null,
  });
  const [anomalies, setAnomalies] = useState([]);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [totalPredictions, setTotalPredictions] = useState(0);

  const handleChartPointClick = (chartName, event, chart, getDetails) => {
    const [hit] = chart.getElementsAtEventForMode(
      event,
      'nearest',
      { intersect: true },
      true
    );
    if (!hit) {
      setSelectedPoint(null);
      return;
    }

    const dataset = chart.data.datasets[hit.datasetIndex];
    const point = dataset.data[hit.index];
    if (point === null || point === undefined) {
      setSelectedPoint(null);
      return;
    }

    const details = getDetails({ dataset, point, index: hit.index, chart });
    if (!details) {
      setSelectedPoint(null);
      return;
    }

    setSelectedPoint({
      chartName,
      series: dataset.label,
      details,
    });
  };

  const renderPointDetails = (chartName) => {
    const isSelected = selectedPoint?.chartName === chartName;
    return (
      <div aria-live="polite" style={{
        marginTop: '12px', padding: '10px 12px', minHeight: '58px',
        borderTop: `1px solid ${THEME.hairline}`, color: THEME.textSecondary,
        fontFamily: THEME.fontMono, fontSize: '11px'
      }}>
        {isSelected ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px 20px' }}>
            <span><strong style={{ color: THEME.textDim }}>SERIES</strong> {selectedPoint.series}</span>
            {Object.entries(selectedPoint.details).map(([label, value]) => (
              <span key={label}><strong style={{ color: THEME.textDim }}>{label.toUpperCase()}</strong> {value}</span>
            ))}
          </div>
        ) : (
          <span style={{ color: THEME.textDim }}>Select a point to view its details.</span>
        )}
      </div>
    );
  };

  const tooltipStyle = {
    backgroundColor: THEME.panelBg,
    titleColor: THEME.textPrimary,
    bodyColor: THEME.textSecondary,
    footerColor: THEME.textSecondary,
    borderColor: THEME.hairline,
    borderWidth: 1,
    cornerRadius: 0,
    padding: 8,
    mode: 'nearest',
    intersect: true,
  };

  // ─── FETCH ALL DATA ──────────────────────────────────────────────────
  useEffect(() => {
    const fetchAllData = async () => {
      try {
        if (isInitialFetch.current) setLoading(true);
        setError(null);

        console.log('Fetching data from backend:', API_BASE);

        // 1. Fetch Ensemble Anomaly Results
        const ensembleUrl = `${API_BASE}/api/v1/ml/ensemble-anomaly-detection?days=7&limit=500`;
        console.log('Fetching ensemble:', ensembleUrl);

        const ensembleRes = await fetch(ensembleUrl);
        if (!ensembleRes.ok) {
          throw new Error(`Ensemble API error: ${ensembleRes.status}`);
        }
        const ensembleData = await ensembleRes.json();
        console.log('Ensemble Data:', ensembleData);

        setAnomalies(ensembleData.anomalies || []);
        setTotalPredictions(ensembleData.total_predictions || 0);
        setLastUpdate(new Date());

        // 2. Fetch Isolation Forest Chart Data
        try {
          const isoRes = await fetch(`${API_BASE}/api/v1/ml/isolation-forest-chart?days=7&limit=500`);
          if (isoRes.ok) {
            const isoData = await isoRes.json();
            console.log('Isolation Forest Data:', isoData);
            setChartData(prev => ({ ...prev, isolation_forest: isoData }));
          } else {
            console.warn('Isolation Forest API returned:', isoRes.status);
          }
        } catch (err) {
          console.warn('Isolation Forest chart error:', err);
        }

        // 3. Fetch Autoencoder Chart Data
        try {
          const aeRes = await fetch(`${API_BASE}/api/v1/ml/autoencoder-chart?days=7&limit=500`);
          if (aeRes.ok) {
            const aeData = await aeRes.json();
            console.log('Autoencoder Data:', aeData);
            setChartData(prev => ({ ...prev, autoencoder: aeData }));
          }
        } catch (err) {
          console.warn('Autoencoder chart error:', err);
        }

        // 4. Fetch Mahalanobis Chart Data
        try {
          const mahalRes = await fetch(`${API_BASE}/api/v1/ml/mahalanobis-chart?days=7&limit=500`);
          if (mahalRes.ok) {
            const mahalData = await mahalRes.json();
            console.log('Mahalanobis Data:', mahalData);
            setChartData(prev => ({ ...prev, mahalanobis: mahalData }));
          }
        } catch (err) {
          console.warn('Mahalanobis chart error:', err);
        }

        // 5. Fetch t-SNE Chart Data
        try {
          const tsneRes = await fetch(`${API_BASE}/api/v1/ml/tsne-chart?days=7&limit=300`);
          if (tsneRes.ok) {
            const tsneData = await tsneRes.json();
            console.log('t-SNE Data:', tsneData);
            setChartData(prev => ({ ...prev, tsne: tsneData }));
          }
        } catch (err) {
          console.warn('t-SNE chart error:', err);
        }

        // 6. Fetch CUSUM Chart Data
        try {
          const cusumRes = await fetch(`${API_BASE}/api/v1/ml/cusum-chart?days=7&limit=500`);
          if (cusumRes.ok) {
            const cusumData = await cusumRes.json();
            console.log('CUSUM Data:', cusumData);
            setChartData(prev => ({ ...prev, cusum: cusumData }));
          }
        } catch (err) {
          console.warn('CUSUM chart error:', err);
        }

      } catch (err) {
        console.error('Error fetching data:', err);
        setError(err.message || 'Failed to load data');
      } finally {
        setLoading(false);
        isInitialFetch.current = false;
      }
    };

    fetchAllData();
    const interval = setInterval(fetchAllData, 60000);
    return () => clearInterval(interval);
  }, []);

  // Tells App.jsx's navigation lock whether this page's initial data is
  // ready yet. `loading` only flips true on the very first fetch (see
  // isInitialFetch above) and false in the finally block on both success
  // and error, so this can't get stuck.
  useEffect(() => {
    if (typeof onLoadingChange === 'function') {
      onLoadingChange(loading);
    }
  }, [loading, onLoadingChange]);

  // ─── CHART 1: ISOLATION FOREST ─────────────────────────────────────
  useEffect(() => {
    if (loading) return;
    const ctx = isoChartRef.current?.getContext('2d');
    if (!ctx) {
      console.warn('Isolation Forest canvas not ready');
      return;
    }

    if (isoChartInstance.current) {
      isoChartInstance.current.destroy();
      isoChartInstance.current = null;
    }

    const isoData = chartData.isolation_forest;

    if (!isoData || isoData.status === 'no_data' || isoData.status === 'error' || !isoData.normal_points) {
      console.warn('No Isolation Forest data available');
      isoChartInstance.current = new Chart(ctx, {
        type: 'scatter',
        data: { datasets: [{ label: 'No Data Available', data: [], backgroundColor: 'rgba(200,200,200,0.3)', pointRadius: 0 }] },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: {
            x: { title: { display: true, text: 'Wind Speed (m/s)', color: THEME.textSecondary, font: { family: THEME.fontSans, size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: THEME.textSecondary, font: { family: THEME.fontMono } } },
            y: { title: { display: true, text: 'Active Power (kW)', color: THEME.textSecondary, font: { family: THEME.fontSans, size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: THEME.textSecondary, font: { family: THEME.fontMono } } }
          }
        }
      });
      return;
    }

    const normalPoints = isoData.normal_points || [];
    const anomalyPoints = isoData.anomaly_points || [];
    const isolationScores = isoData.scores || [];

    console.log(`Isolation Forest: ${normalPoints.length} normal, ${anomalyPoints.length} anomaly points`);

    const datasets = [
      { label: 'Normal Points', data: normalPoints, backgroundColor: 'rgba(16, 185, 129, 0.7)', borderColor: THEME.normalBorder, borderWidth: 1, pointRadius: 4.5, pointHoverRadius: 7 },
      { label: 'Anomalies Detected', data: anomalyPoints, backgroundColor: 'rgba(220, 38, 38, 0.85)', borderColor: THEME.anomalyBorder, borderWidth: 1.5, pointRadius: 6, pointHoverRadius: 9, pointBorderColor: '#ffffff', pointBorderWidth: 1.5 }
    ];

    if (isolationScores.length > 0) {
      datasets.push({
        label: 'Isolation Scores',
        data: isolationScores,
        backgroundColor: 'rgba(15, 139, 141, 0.15)',
        borderColor: 'rgba(15, 139, 141, 0.4)',
        pointRadius: 3,
        pointBackgroundColor: isolationScores.map(s => {
          const intensity = Math.min(s.score || 0, 1);
          return `rgba(15, 139, 141, ${0.1 + intensity * 0.5})`;
        }),
        showLine: false,
      });
    }

    isoChartInstance.current = new Chart(ctx, {
      type: 'scatter', data: { datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        onClick: (event, _activeElements, chart) => handleChartPointClick(
          'Isolation Forest',
          event,
          chart,
          ({ dataset, point }) => {
            if (dataset.label === 'Isolation Scores') return null;
            return {
              turbine: point.turbine_id || 'Unknown',
              classification: dataset.label === 'Anomalies Detected' ? 'Anomaly' : 'Normal',
              wind: Number.isFinite(point.x) ? `${point.x.toFixed(2)} m/s` : 'N/A',
              power: Number.isFinite(point.y) ? `${point.y.toFixed(2)} kW` : 'N/A',
              ...(Number.isFinite(point.score) ? { score: point.score.toFixed(3) } : {}),
            };
          }
        ),
        plugins: {
          legend: { position: 'bottom', labels: { color: THEME.textSecondary, font: { family: THEME.fontSans, size: 10, weight: 500 }, usePointStyle: true, boxWidth: 6, padding: 12 } },
          tooltip: {
            ...tooltipStyle,
            callbacks: { label: function(context) { const raw = context.raw; if (!raw) return ''; return [`Turbine: ${raw.turbine_id || 'Unknown'}`, `Wind: ${raw.x?.toFixed(1) || 'N/A'} m/s`, `Power: ${raw.y?.toFixed(1) || 'N/A'} kW`]; } }
          }
        },
        scales: {
          x: { title: { display: true, text: 'Wind Speed (m/s)', color: THEME.textSecondary, font: { family: THEME.fontSans, size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: THEME.textSecondary, font: { family: THEME.fontMono, size: 10 } }, min: 0, max: 30 },
          y: { title: { display: true, text: 'Active Power (kW)', color: THEME.textSecondary, font: { family: THEME.fontSans, size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: THEME.textSecondary, font: { family: THEME.fontMono, size: 10 } }, min: 0, max: 7000 }
        }
      }
    });

    return () => { if (isoChartInstance.current) { isoChartInstance.current.destroy(); isoChartInstance.current = null; } };
  }, [chartData.isolation_forest, loading]);

  // ─── CHART 2: AUTOENCODER ──────────────────────────────────────────
  useEffect(() => {
    if (loading) return;
    const ctx = autoencChartRef.current?.getContext('2d');
    if (!ctx) return;
    if (autoencChartInstance.current) { autoencChartInstance.current.destroy(); autoencChartInstance.current = null; }

    const aeData = chartData.autoencoder;
    if (!aeData || aeData.status === 'no_data' || aeData.status === 'error' || !aeData.reconstruction_errors) {
      autoencChartInstance.current = new Chart(ctx, {
        type: 'line', data: { datasets: [{ label: 'No Data Available', data: [], borderColor: THEME.hairline, pointRadius: 0 }] },
        options: {
          responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
          scales: { x: { title: { display: true, text: 'Time', color: THEME.textSecondary, font: { family: THEME.fontSans } } }, y: { title: { display: true, text: 'Reconstruction Error', color: THEME.textSecondary, font: { family: THEME.fontSans } } } }
        }
      });
      return;
    }

    const reconstructionErrors = aeData.reconstruction_errors || [];
    const threshold = aeData.threshold || 0.1;
    const anomalies = aeData.anomaly_indices || [];
    const labels = aeData.timestamps || Array.from({ length: reconstructionErrors.length }, (_, i) => i);

    const datasets = [
      { label: 'Reconstruction Error', data: reconstructionErrors, borderColor: THEME.score, backgroundColor: 'rgba(15, 139, 141, 0.08)', fill: true, tension: 0.25, pointRadius: 3, pointBackgroundColor: reconstructionErrors.map((_, index) => anomalies.includes(index) ? THEME.anomaly : THEME.score), borderWidth: 2 },
      { label: 'Anomaly Threshold', data: reconstructionErrors.map(() => threshold), borderColor: THEME.threshold, borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0, fill: false }
    ];
    if (anomalies.length > 0) {
      datasets.push({ label: 'Anomalies Detected', data: reconstructionErrors.map((err, index) => anomalies.includes(index) ? err : null), backgroundColor: 'rgba(220, 38, 38, 0.9)', borderColor: '#ffffff', borderWidth: 1.5, pointRadius: 6, showLine: false });
    }

    autoencChartInstance.current = new Chart(ctx, {
      type: 'line', data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        onClick: (event, _activeElements, chart) => handleChartPointClick(
          'Autoencoder',
          event,
          chart,
          ({ point, index, chart: activeChart }) => ({
            sample: activeChart.data.labels[index] ?? index,
            classification: anomalies.includes(index) ? 'Anomaly' : 'Normal',
            reconstructionError: Number(point).toFixed(5),
            threshold: Number(threshold).toFixed(5),
          })
        ),
        plugins: {
          legend: { position: 'bottom', labels: { color: THEME.textSecondary, font: { family: THEME.fontSans, size: 10, weight: 500 }, usePointStyle: true, boxWidth: 6, padding: 12 } },
          tooltip: { ...tooltipStyle, callbacks: { label: function(context) { const value = context.parsed.y; if (value === null || value === undefined) return 'No data'; return `${context.dataset.label}: ${value.toFixed(4)}`; } } }
        },
        scales: {
          x: { title: { display: true, text: 'Time', color: THEME.textSecondary, font: { family: THEME.fontSans, size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: THEME.textSecondary, font: { family: THEME.fontMono, size: 10 } } },
          y: { title: { display: true, text: 'Reconstruction Error', color: THEME.textSecondary, font: { family: THEME.fontSans, size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: THEME.textSecondary, font: { family: THEME.fontMono, size: 10 } }, min: 0 }
        }
      }
    });

    return () => { if (autoencChartInstance.current) { autoencChartInstance.current.destroy(); autoencChartInstance.current = null; } };
  }, [chartData.autoencoder, loading]);

  // ─── CHART 3: MAHALANOBIS ──────────────────────────────────────────
  useEffect(() => {
    if (loading) return;
    const ctx = mahalChartRef.current?.getContext('2d');
    if (!ctx) return;
    if (mahalChartInstance.current) { mahalChartInstance.current.destroy(); mahalChartInstance.current = null; }

    const mahalData = chartData.mahalanobis;
    if (!mahalData || mahalData.status === 'no_data' || mahalData.status === 'error' || !mahalData.distances) {
      mahalChartInstance.current = new Chart(ctx, {
        type: 'line', data: { datasets: [{ label: 'No Data Available', data: [], borderColor: THEME.hairline, pointRadius: 0 }] },
        options: {
          responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
          scales: { x: { title: { display: true, text: 'Sample Index', color: THEME.textSecondary } }, y: { title: { display: true, text: 'Distance', color: THEME.textSecondary } } }
        }
      });
      return;
    }

    const distances = mahalData.distances || [];
    const threshold = mahalData.threshold || 7.5;
    const anomalyIndices = mahalData.anomaly_indices || [];
    const labels = mahalData.labels || Array.from({ length: distances.length }, (_, i) => i);

    mahalChartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Mahalanobis Distance', data: distances, borderColor: THEME.score, backgroundColor: 'rgba(15, 139, 141, 0.08)', fill: true, tension: 0.3, pointRadius: 2, pointBackgroundColor: distances.map((_, i) => anomalyIndices.includes(i) ? THEME.anomaly : THEME.score), borderWidth: 2 },
          { label: 'Chi-Squared Threshold', data: distances.map(() => threshold), borderColor: THEME.threshold, borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0, fill: false }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        onClick: (event, _activeElements, chart) => handleChartPointClick(
          'Mahalanobis',
          event,
          chart,
          ({ point, index, chart: activeChart }) => ({
            sample: activeChart.data.labels[index] ?? index,
            classification: anomalyIndices.includes(index) ? 'Anomaly' : 'Normal',
            distance: Number(point).toFixed(5),
            threshold: Number(threshold).toFixed(5),
          })
        ),
        plugins: {
          legend: { position: 'bottom', labels: { color: THEME.textSecondary, font: { family: THEME.fontSans, size: 10, weight: 500 }, usePointStyle: true } },
          tooltip: { ...tooltipStyle, callbacks: { label: function(context) { const value = context.parsed.y; if (value === null || value === undefined) return 'No data'; return `${context.dataset.label}: ${value.toFixed(4)}`; } } }
        },
        scales: {
          x: { title: { display: true, text: 'Sample Index', color: THEME.textSecondary, font: { family: THEME.fontSans, size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: THEME.textSecondary, font: { family: THEME.fontMono, size: 10 } } },
          y: { title: { display: true, text: 'Distance', color: THEME.textSecondary, font: { family: THEME.fontSans, size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: THEME.textSecondary, font: { family: THEME.fontMono, size: 10 } }, min: 0 }
        }
      }
    });

    return () => { if (mahalChartInstance.current) { mahalChartInstance.current.destroy(); mahalChartInstance.current = null; } };
  }, [chartData.mahalanobis, loading]);

  // ─── CHART 4: t-SNE ────────────────────────────────────────────────
  useEffect(() => {
    if (loading) return;
    const ctx = tsneChartRef.current?.getContext('2d');
    if (!ctx) return;
    if (tsneChartInstance.current) { tsneChartInstance.current.destroy(); tsneChartInstance.current = null; }

    const tsneData = chartData.tsne;
    if (!tsneData || tsneData.status === 'no_data' || tsneData.status === 'error' || !tsneData.normal_points) {
      tsneChartInstance.current = new Chart(ctx, {
        type: 'scatter', data: { datasets: [{ label: 'No Data Available', data: [], backgroundColor: 'rgba(200,200,200,0.3)', pointRadius: 0 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 't-SNE Dimension 1', color: THEME.textSecondary } }, y: { title: { display: true, text: 't-SNE Dimension 2', color: THEME.textSecondary } } } }
      });
      return;
    }

    const normalData = tsneData.normal_points || [];
    const anomalyData_tsne = tsneData.anomaly_points || [];

    tsneChartInstance.current = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          { label: 'Normal Operating States', data: normalData, backgroundColor: 'rgba(16, 185, 129, 0.7)', borderColor: THEME.normalBorder, borderWidth: 1, pointRadius: 4, pointHoverRadius: 7 },
          { label: 'Anomalous States', data: anomalyData_tsne, backgroundColor: 'rgba(220, 38, 38, 0.8)', borderColor: THEME.anomalyBorder, borderWidth: 1.5, pointRadius: 6, pointHoverRadius: 9, pointBorderColor: '#ffffff', pointBorderWidth: 1.5 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        onClick: (event, _activeElements, chart) => handleChartPointClick(
          't-SNE',
          event,
          chart,
          ({ dataset, point, index }) => ({
            sample: index + 1,
            classification: dataset.label === 'Anomalous States' ? 'Anomaly' : 'Normal',
            dimension1: Number(point.x).toFixed(5),
            dimension2: Number(point.y).toFixed(5),
          })
        ),
        plugins: {
          legend: { position: 'bottom', labels: { color: THEME.textSecondary, font: { family: THEME.fontSans, size: 10, weight: 500 }, usePointStyle: true } },
          tooltip: { ...tooltipStyle, callbacks: { label: function(context) { const raw = context.raw; if (!raw) return ''; return [`Dimension 1: ${raw.x?.toFixed(3) || 'N/A'}`, `Dimension 2: ${raw.y?.toFixed(3) || 'N/A'}`]; } } }
        },
        scales: {
          x: { title: { display: true, text: 't-SNE Dimension 1', color: THEME.textSecondary, font: { family: THEME.fontSans, size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: THEME.textSecondary, font: { family: THEME.fontMono, size: 10 } } },
          y: { title: { display: true, text: 't-SNE Dimension 2', color: THEME.textSecondary, font: { family: THEME.fontSans, size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: THEME.textSecondary, font: { family: THEME.fontMono, size: 10 } } }
        }
      }
    });

    return () => { if (tsneChartInstance.current) { tsneChartInstance.current.destroy(); tsneChartInstance.current = null; } };
  }, [chartData.tsne, loading]);

  // ─── CHART 5: CUSUM ────────────────────────────────────────────────
  useEffect(() => {
    if (loading) return;
    const ctx = cusumChartRef.current?.getContext('2d');
    if (!ctx) return;
    if (cusumChartInstance.current) { cusumChartInstance.current.destroy(); cusumChartInstance.current = null; }

    const cusumData = chartData.cusum;
    if (!cusumData || cusumData.status === 'no_data' || cusumData.status === 'error' || !cusumData.signal) {
      cusumChartInstance.current = new Chart(ctx, {
        type: 'line', data: { datasets: [{ label: 'No Data Available', data: [], borderColor: THEME.hairline, pointRadius: 0 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'Sample', color: THEME.textSecondary } }, y: { title: { display: true, text: 'Vibration Amplitude', color: THEME.textSecondary } } } }
      });
      return;
    }

    const cusumSignal = cusumData.signal || [];
    const anomalyIndices = cusumData.anomaly_indices || [];
    const labels = cusumData.labels || Array.from({ length: cusumSignal.length }, (_, i) => i);

    cusumChartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Vibration Signal (CUSUM)', data: cusumSignal, borderColor: THEME.score, backgroundColor: 'rgba(15, 139, 141, 0.08)', fill: true, tension: 0.2, pointRadius: 2, pointBackgroundColor: cusumSignal.map((_, i) => anomalyIndices.includes(i) ? THEME.anomaly : THEME.score), borderWidth: 2 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        onClick: (event, _activeElements, chart) => handleChartPointClick(
          'CUSUM',
          event,
          chart,
          ({ point, index, chart: activeChart }) => ({
            sample: activeChart.data.labels[index] ?? index,
            classification: anomalyIndices.includes(index) ? 'Change point' : 'Normal',
            vibration: Number(point).toFixed(5),
            threshold: Number(cusumData.threshold ?? 0).toFixed(5),
          })
        ),
        plugins: {
          legend: { position: 'bottom', labels: { color: THEME.textSecondary, font: { family: THEME.fontSans, size: 10, weight: 500 }, usePointStyle: true } },
          tooltip: { ...tooltipStyle, callbacks: { label: function(context) { const value = context.parsed.y; if (value === null || value === undefined) return 'No data'; return `Vibration: ${value.toFixed(4)}`; }, afterLabel: function(context) { if (anomalyIndices.includes(context.dataIndex)) { return 'Change point detected'; } return null; } } }
        },
        scales: {
          x: { title: { display: true, text: 'Sample', color: THEME.textSecondary, font: { family: THEME.fontSans, size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: THEME.textSecondary, font: { family: THEME.fontMono, size: 10 } } },
          y: { title: { display: true, text: 'Vibration Amplitude', color: THEME.textSecondary, font: { family: THEME.fontSans, size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: THEME.textSecondary, font: { family: THEME.fontMono, size: 10 } }, min: 0 }
        }
      }
    });

    return () => { if (cusumChartInstance.current) { cusumChartInstance.current.destroy(); cusumChartInstance.current = null; } };
  }, [chartData.cusum, loading]);

  // ─── LOADING STATE ──────────────────────────────────────────────────
  if (loading) {
    return (
      <div style={{
        display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px', gap: '14px', color: THEME.textDim, backgroundColor: THEME.background
      }}>
        <div style={{
          width: '20px', height: '20px',
          border: `2.5px solid ${THEME.hairline}`,
          borderTop: `2.5px solid ${THEME.accentPrimary}`,
          borderRadius: '50%', animation: 'spin 0.9s linear infinite'
        }} />
        <span style={{ fontSize: '13px', fontFamily: THEME.fontSans, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Loading models...</span>
        <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  // ─── MAIN RENDER (INSTRUMENT PANEL) ─────────────────────────────────
  return (
    <div style={{
      padding: '24px 32px', maxWidth: '1440px', margin: '0 auto',
      background: THEME.background, minHeight: '100vh',
      fontFamily: THEME.fontSans, color: THEME.textPrimary
    }}>
      
      {/* Header - Instrument Style */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: '20px', flexWrap: 'wrap', gap: '12px',
        padding: '0 0 16px 0', 
        borderBottom: `2px solid ${THEME.accentPrimary}`, // Solid Teal line to anchor the panel
      }}>
        <div>
          <h1 style={{ fontFamily: THEME.fontSans, fontSize: '20px', fontWeight: 600, color: THEME.textPrimary, margin: 0, letterSpacing: '-0.5px' }}>
            ANOMALY DETECTION
          </h1>
          <span style={{ fontSize: '10px', fontWeight: 500, color: THEME.textDim, marginTop: '4px', display: 'block', fontFamily: THEME.fontMono, letterSpacing: '1px', textTransform: 'uppercase' }}>
            Multi-Model Ensemble | 7 Day Window
          </span>
        </div>

        <div style={{ display: 'flex', gap: '24px', fontSize: '12px', color: THEME.textSecondary, alignItems: 'center', flexWrap: 'wrap', fontFamily: THEME.fontMono }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', background: error ? THEME.anomaly : THEME.normal, boxShadow: `0 0 4px ${error ? THEME.anomaly : THEME.normal}` }} />
            <span style={{ fontWeight: 500, fontSize: '10px', textTransform: 'uppercase', color: THEME.textSecondary }}>
              {error ? 'Offline' : 'Live'}
            </span>
          </div>
          <div style={{ display: 'flex', gap: '4px', alignItems: 'baseline' }}>
            <span style={{ color: THEME.textDim, fontSize: '10px' }}>PREDICTIONS</span>
            <strong style={{ fontSize: '14px', color: THEME.textPrimary }}>{totalPredictions}</strong>
          </div>
          <div style={{ display: 'flex', gap: '4px', alignItems: 'baseline' }}>
            <span style={{ color: THEME.textDim, fontSize: '10px' }}>ANOMALIES</span>
            <strong style={{ fontSize: '14px', color: anomalies.length > 0 ? THEME.anomaly : THEME.normal }}>
              {anomalies.length}
            </strong>
          </div>
          {lastUpdate && (
            <div style={{ fontSize: '10px', color: THEME.textDim }}>
              UPDATED {lastUpdate.toLocaleTimeString().toUpperCase()}
            </div>
          )}
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: '10px',
          background: '#fef2f2', border: `1px solid ${THEME.anomalyBorder}`,
          borderLeft: `3px solid ${THEME.anomaly}`, borderRadius: THEME.radiusPanel, // Tight radius
          padding: '10px 14px', marginBottom: '16px', color: THEME.textPrimary, fontSize: '12px'
        }}>
          <strong style={{ fontSize: '10px', textTransform: 'uppercase', color: THEME.anomaly }}>Error</strong>
          <span style={{ color: THEME.textSecondary }}>{error}</span>
          <button onClick={() => window.location.reload()} style={{
            marginLeft: 'auto', background: 'transparent', border: `1px solid ${THEME.anomalyBorder}`, color: THEME.anomaly, padding: '4px 12px', borderRadius: THEME.radiusChip, cursor: 'pointer', fontSize: '10px', textTransform: 'uppercase', fontFamily: THEME.fontSans
          }}>Retry</button>
        </div>
      )}

      {/* ROW 1: Isolation Forest + Autoencoder */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
        
        {/* Isolation Forest */}
        <div style={{ background: THEME.panelBg, border: `1px solid ${THEME.hairline}`, borderRadius: THEME.radiusPanel, padding: '16px', boxShadow: 'none' }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            marginBottom: '12px', paddingBottom: '10px', borderBottom: `1px solid ${THEME.hairline}`
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', fontWeight: 600, color: THEME.textPrimary, fontFamily: THEME.fontSans, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              <FaShieldAlt size={12} color={THEME.accentPrimary} /> Isolation Forest
            </div>
            <div style={{
              fontSize: '9px', color: chartData.isolation_forest ? THEME.textSecondary : THEME.warn,
              fontFamily: THEME.fontMono, background: chartData.isolation_forest ? 'transparent' : '#fef3c7',
              border: `1px solid ${chartData.isolation_forest ? THEME.hairline : '#f59e0b'}`,
              borderRadius: THEME.radiusChip, padding: '2px 8px', letterSpacing: '1px'
            }}>
              {chartData.isolation_forest ? 'OPERATIONAL' : 'NO DATA'}
            </div>
          </div>
          <div style={{ height: '220px', position: 'relative' }}>
            <canvas ref={isoChartRef} />
          </div>
          {renderPointDetails('Isolation Forest')}
        </div>

        {/* Autoencoder */}
        <div style={{ background: THEME.panelBg, border: `1px solid ${THEME.hairline}`, borderRadius: THEME.radiusPanel, padding: '16px', boxShadow: 'none' }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            marginBottom: '12px', paddingBottom: '10px', borderBottom: `1px solid ${THEME.hairline}`
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', fontWeight: 600, color: THEME.textPrimary, fontFamily: THEME.fontSans, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              <FaBrain size={12} color={THEME.accentPrimary} /> Autoencoder
            </div>
            <div style={{
              fontSize: '9px', color: chartData.autoencoder ? THEME.textSecondary : THEME.warn,
              fontFamily: THEME.fontMono, background: chartData.autoencoder ? 'transparent' : '#fef3c7',
              border: `1px solid ${chartData.autoencoder ? THEME.hairline : '#f59e0b'}`,
              borderRadius: THEME.radiusChip, padding: '2px 8px', letterSpacing: '1px'
            }}>
              {chartData.autoencoder ? 'OPERATIONAL' : 'NO DATA'}
            </div>
          </div>
          <div style={{ height: '220px', position: 'relative' }}>
            <canvas ref={autoencChartRef} />
          </div>
          {renderPointDetails('Autoencoder')}
        </div>
      </div>

      {/* ROW 2: Mahalanobis + t-SNE */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
        
        {/* Mahalanobis */}
        <div style={{ background: THEME.panelBg, border: `1px solid ${THEME.hairline}`, borderRadius: THEME.radiusPanel, padding: '16px', boxShadow: 'none' }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            marginBottom: '12px', paddingBottom: '10px', borderBottom: `1px solid ${THEME.hairline}`
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', fontWeight: 600, color: THEME.textPrimary, fontFamily: THEME.fontSans, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              <FaChartLine size={12} color={THEME.accentPrimary} /> Mahalanobis
            </div>
            <div style={{
              fontSize: '9px', color: chartData.mahalanobis ? THEME.textSecondary : THEME.warn,
              fontFamily: THEME.fontMono, background: chartData.mahalanobis ? 'transparent' : '#fef3c7',
              border: `1px solid ${chartData.mahalanobis ? THEME.hairline : '#f59e0b'}`,
              borderRadius: THEME.radiusChip, padding: '2px 8px', letterSpacing: '1px'
            }}>
              {chartData.mahalanobis ? 'OPERATIONAL' : 'NO DATA'}
            </div>
          </div>
          <div style={{ height: '220px', position: 'relative' }}>
            <canvas ref={mahalChartRef} />
          </div>
          {renderPointDetails('Mahalanobis')}
        </div>

        {/* t-SNE */}
        <div style={{ background: THEME.panelBg, border: `1px solid ${THEME.hairline}`, borderRadius: THEME.radiusPanel, padding: '16px', boxShadow: 'none' }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            marginBottom: '12px', paddingBottom: '10px', borderBottom: `1px solid ${THEME.hairline}`
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', fontWeight: 600, color: THEME.textPrimary, fontFamily: THEME.fontSans, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              <FaDrawPolygon size={12} color={THEME.accentPrimary} /> t-SNE Embedding
            </div>
            <div style={{
              fontSize: '9px', color: chartData.tsne ? THEME.textSecondary : THEME.warn,
              fontFamily: THEME.fontMono, background: chartData.tsne ? 'transparent' : '#fef3c7',
              border: `1px solid ${chartData.tsne ? THEME.hairline : '#f59e0b'}`,
              borderRadius: THEME.radiusChip, padding: '2px 8px', letterSpacing: '1px'
            }}>
              {chartData.tsne ? 'OPERATIONAL' : 'NO DATA'}
            </div>
          </div>
          <div style={{ height: '220px', position: 'relative' }}>
            <canvas ref={tsneChartRef} />
          </div>
          {renderPointDetails('t-SNE')}
        </div>
      </div>

      {/* ROW 3: CUSUM + Anomaly Log */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '16px' }}>
        
        {/* CUSUM */}
        <div style={{ background: THEME.panelBg, border: `1px solid ${THEME.hairline}`, borderRadius: THEME.radiusPanel, padding: '16px', boxShadow: 'none' }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            marginBottom: '12px', paddingBottom: '10px', borderBottom: `1px solid ${THEME.hairline}`
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', fontWeight: 600, color: THEME.textPrimary, fontFamily: THEME.fontSans, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              <FaWaveSquare size={12} color={THEME.accentPrimary} /> CUSUM Change Detection
            </div>
            <div style={{
              fontSize: '9px', color: chartData.cusum ? THEME.textSecondary : THEME.warn,
              fontFamily: THEME.fontMono, background: chartData.cusum ? 'transparent' : '#fef3c7',
              border: `1px solid ${chartData.cusum ? THEME.hairline : '#f59e0b'}`,
              borderRadius: THEME.radiusChip, padding: '2px 8px', letterSpacing: '1px'
            }}>
              {chartData.cusum ? 'OPERATIONAL' : 'NO DATA'}
            </div>
          </div>
          <div style={{ height: '160px', position: 'relative' }}>
            <canvas ref={cusumChartRef} />
          </div>
          {renderPointDetails('CUSUM')}
        </div>

        {/* Anomaly Log */}
        <div style={{ background: THEME.panelBg, border: `1px solid ${THEME.hairline}`, borderRadius: THEME.radiusPanel, padding: '16px', boxShadow: 'none' }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            marginBottom: '12px', paddingBottom: '10px', borderBottom: `1px solid ${THEME.hairline}`
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', fontWeight: 600, color: THEME.textPrimary, fontFamily: THEME.fontSans, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              <FaListUl size={12} color={THEME.accentPrimary} /> Anomaly Log
            </div>
            <div style={{
              fontSize: '9px', color: THEME.anomaly, fontFamily: THEME.fontMono,
              background: 'transparent', border: `1px solid ${THEME.anomalyBorder}`,
              borderRadius: THEME.radiusChip, padding: '2px 8px', letterSpacing: '1px'
            }}>LIVE</div>
          </div>
          <div style={{ padding: '8px', maxHeight: '160px', overflowY: 'auto' }}>
            {anomalies && anomalies.length > 0 ? (
              <table style={{
                width: '100%', fontSize: '11px', borderCollapse: 'collapse',
                color: THEME.textSecondary, fontFamily: THEME.fontMono, // Monospace for data!
              }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left', fontWeight: 500, padding: '4px', borderBottom: `1px solid ${THEME.hairline}`, color: THEME.textDim, fontSize: '9px', textTransform: 'uppercase' }}>Turbine</th>
                    <th style={{ textAlign: 'right', fontWeight: 500, padding: '4px', borderBottom: `1px solid ${THEME.hairline}`, color: THEME.textDim, fontSize: '9px', textTransform: 'uppercase' }}>Score</th>
                    <th style={{ textAlign: 'right', fontWeight: 500, padding: '4px', borderBottom: `1px solid ${THEME.hairline}`, color: THEME.textDim, fontSize: '9px', textTransform: 'uppercase' }}>Sev.</th>
                  </tr>
                </thead>
                <tbody>
                  {anomalies.slice(0, 10).map((anom, idx) => (
                    <tr key={idx} style={{ borderBottom: `1px solid ${THEME.hairline}` }}>
                      <td style={{ padding: '6px 4px', fontWeight: 500, color: THEME.textPrimary }}>{anom.turbine_id || 'Unknown'}</td>
                      <td style={{ padding: '6px 4px', textAlign: 'right', color: anom.score > 0.7 ? THEME.anomaly : THEME.warn }}>
                        {(anom.score || 0).toFixed(3)}
                      </td>
                      <td style={{ padding: '6px 4px', textAlign: 'right' }}>
                        <span style={{
                          padding: '1px 6px', borderRadius: THEME.radiusChip,
                          fontSize: '9px', fontWeight: 600, textTransform: 'uppercase', fontFamily: THEME.fontSans,
                          background: anom.severity === 'high' ? '#fef2f2' : '#fffbeb',
                          color: anom.severity === 'high' ? THEME.anomaly : THEME.warn
                        }}>
                          {anom.severity || 'med'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div style={{ textAlign: 'center', color: THEME.textDim, padding: '20px 0', fontSize: '11px', fontFamily: THEME.fontMono }}>
                NO ANOMALIES DETECTED
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: ${THEME.background}; }
        ::-webkit-scrollbar-thumb { background: ${THEME.hairline}; border-radius: 2px; }
        ::-webkit-scrollbar-thumb:hover { background: ${THEME.textDim}; }
      `}</style>
    </div>
  );
}