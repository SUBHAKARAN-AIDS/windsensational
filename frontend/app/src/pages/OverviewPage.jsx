import React, { useEffect, useRef, useState } from 'react';
import { getLiveTurbines } from '../services/turbineApi';
import { getPowerCurve } from '../services/analyticsApi';
import Chart from 'chart.js/auto';
import KpiCard from '../components/common/KpiCard';
import TurbineMap from '../components/overview/TurbineMap';
import FaultTable from '../components/overview/FaultTable';
import { LuMap, LuTriangleAlert, LuZap, LuTrendingUp, LuLoader, LuActivity } from 'react-icons/lu';
import '../App.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function OverviewPage({ onLoadingChange }) {
  const powerChartRef = useRef(null);
  const trendChartRef = useRef(null);
  const powerChartInstance = useRef(null);
  const trendChartInstance = useRef(null);

  const zigzagChartRef = useRef(null);
  const zigzagChartInstance = useRef(null);
  const fleetTrendChartRef = useRef(null);
  const fleetTrendChartInstance = useRef(null);
  const gprChartRef = useRef(null);
  const gprChartInstance = useRef(null);

  const [fleetData, setFleetData] = useState(null);
  const [powerCurveData, setPowerCurveData] = useState([]);
  const [fleetTrendData, setFleetTrendData] = useState([]);
  const [gprData, setGprData] = useState(null);
  const [gprLoading, setGprLoading] = useState(true);
  const [loading, setLoading] = useState(true);
  const [selectedTurbine, setSelectedTurbine] = useState(null);
  const [, setLastUpdate] = useState(null);

  const [voterConsensus, setVoterConsensus] = useState(null);
  const [consensusAnomalyCount, setConsensusAnomalyCount] = useState(0);

  useEffect(() => {
    const fetchOverviewData = async () => {
      try {
        const liveData = await getLiveTurbines();
        setFleetData(liveData);

        const pcData = await getPowerCurve();
        let resolvedCurve = [];
        if (Array.isArray(pcData) && pcData.length > 0) {
          resolvedCurve = pcData;
        } else if (pcData.power_curve && Array.isArray(pcData.power_curve)) {
          resolvedCurve = pcData.power_curve;
        } else if (pcData.data && Array.isArray(pcData.data)) {
          resolvedCurve = pcData.data;
        }
        if (resolvedCurve.length > 0) {
          setPowerCurveData(resolvedCurve);
          setLastUpdate(new Date());
        }

        // 90-day fleet trend — separate try/catch, non-fatal
        try {
          const res = await fetch(`${API_BASE}/api/v1/analytics/fleet-trend?days=90`);
          if (!res.ok) throw new Error(`Server responded with ${res.status}`);
          const data = await res.json();
          setFleetTrendData(data.trend || []);
        } catch (err) {
          console.error('Failed to load fleet trend:', err);
          setFleetTrendData([]);
        }

        // Voter consensus — unchanged from original
        try {
          const params = new URLSearchParams({ days: '7', limit: '500' });
          const res = await fetch(`${API_BASE}/api/v1/ml/ensemble-anomaly-detection?${params}`);
          if (!res.ok) throw new Error(`Server responded with ${res.status}`);
          const data = await res.json();
          const anomalies = data.anomalies || [];
          const withVoters = anomalies.filter(a => a.total_voters > 0);
          if (withVoters.length > 0) {
            const avgRatio = withVoters.reduce(
              (sum, a) => sum + (a.ensemble_votes / a.total_voters), 0
            ) / withVoters.length;
            setVoterConsensus((avgRatio * 100).toFixed(1));
            setConsensusAnomalyCount(withVoters.length);
          } else {
            setVoterConsensus(null);
            setConsensusAnomalyCount(0);
          }
        } catch (err) {
          console.error('Failed to load voter consensus:', err);
          setVoterConsensus(null);
          setConsensusAnomalyCount(0);
        }
      } catch (err) {
        console.error('Failed to load data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchOverviewData();
    const interval = setInterval(fetchOverviewData, 30000);
    return () => clearInterval(interval);
  }, []);

  // GPR fetch — separate effect, much slower endpoint, doesn't block main page load
  useEffect(() => {
    let cancelled = false;
    setGprLoading(true);
    fetch(`${API_BASE}/api/v1/analytics/power-curve-gpr?days=90&n_clusters=3`)
      .then(res => {
        if (!res.ok) throw new Error(`Server responded with ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (cancelled) return;
        setGprData(data.status === 'success' ? data : null);
        setGprLoading(false);
      })
      .catch(err => {
        console.error('Failed to load GPR power curve:', err);
        if (!cancelled) { setGprData(null); setGprLoading(false); }
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (typeof onLoadingChange === 'function') {
      onLoadingChange(loading);
    }
  }, [loading, onLoadingChange]);

  const rawTurbinesList = fleetData?.turbines || [];
  const turbinesList = rawTurbinesList.map((t) => {
    const activePower = Number(t.activePower ?? t.active_power ?? 0);
    return {
      ...t,
      id: t.id || t.turbine_id || t.name || 'WTG',
      name: t.name || t.turbine_id || 'WTG',
      status: t.status || (activePower > 0 ? 'Operational' : 'Unknown'),
      wind_speed: Number(t.windSpeed ?? t.wind_speed ?? 0),
      active_power: activePower,
    };
  });

  const totalCount = turbinesList.length;
  const activeCount = turbinesList.filter((t) => t.status.toLowerCase() === 'operational').length;
  const totalActivePower = turbinesList.reduce((acc, t) => acc + t.active_power, 0);
  const avgWindSpeed = totalCount > 0 ? (turbinesList.reduce((acc, t) => acc + t.wind_speed, 0) / totalCount).toFixed(1) : '0.0';
  const fleetHealthIndex = totalCount > 0 ? ((activeCount / totalCount) * 100).toFixed(1) : '0.0';
  const fleetAvgPower = totalCount > 0 ? totalActivePower / totalCount : 0;

  // ── Chart 1: fleet-wide power zig-zag ──
  useEffect(() => {
    if (loading || turbinesList.length === 0) return;
    const ctx = zigzagChartRef.current?.getContext('2d');
    if (!ctx) return;

    zigzagChartInstance.current?.destroy();

    const sorted = [...turbinesList].sort((a, b) => b.active_power - a.active_power);
    // Use t.id (real turbine ID — DZH1, DZH5, etc.) for the X-axis, not t.name,
    // since 'name' can silently fall back to a generic 'WTG' placeholder
    // if the live API response doesn't include a distinct name field.
    const labels = sorted.map(t => t.id);
    const values = sorted.map(t => t.active_power);

    zigzagChartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Active Power (kW)',
            data: values,
            borderColor: '#3a6ea5',
            backgroundColor: 'rgba(58,110,165,0.08)',
            fill: true,
            tension: 0.15,
            pointRadius: 2,
            pointBackgroundColor: values.map(v => v < fleetAvgPower * 0.6 ? '#c4432f' : v < fleetAvgPower * 0.85 ? '#c8860d' : '#3a6ea5'),
            borderWidth: 1.5,
          },
          {
            label: 'Fleet Average',
            data: Array(labels.length).fill(fleetAvgPower),
            borderColor: '#9a9584',
            borderDash: [6, 4],
            pointRadius: 0,
            borderWidth: 1.5,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: 'var(--text-muted)', font: { size: 10 }, boxWidth: 10 } },
          tooltip: {
            callbacks: {
              title: (items) => `Turbine ${items[0].label}`,
              label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(0)} kW`,
            },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: 'var(--text-muted)', font: { size: 8 }, maxRotation: 90, minRotation: 90 },
            title: { display: true, text: 'Turbine ID', color: 'var(--text-muted)', font: { size: 10 } },
          },
          y: {
            grid: { color: 'var(--border-color)' },
            ticks: { color: 'var(--text-muted)', font: { size: 10 } },
            title: { display: true, text: 'Active Power (kW)', color: 'var(--text-muted)', font: { size: 10 } },
          },
        },
      },
    });

    return () => zigzagChartInstance.current?.destroy();
  }, [turbinesList, loading, fleetAvgPower]);

  // ── Chart 2: 90-day fleet performance trend ──
  useEffect(() => {
    if (fleetTrendData.length === 0) return;
    const ctx = fleetTrendChartRef.current?.getContext('2d');
    if (!ctx) return;

    fleetTrendChartInstance.current?.destroy();

    const labels = fleetTrendData.map(d => new Date(d.day).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }));
    const totalPower = fleetTrendData.map(d => d.total_power / 1000); // kW -> MW
    const avgPower = fleetTrendData.map(d => d.avg_power);

    fleetTrendChartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Fleet Total Power (MW)',
            data: totalPower,
            borderColor: '#3f8a5b',
            backgroundColor: 'rgba(63,138,91,0.10)',
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2,
            yAxisID: 'y',
          },
          {
            label: 'Avg per Turbine (kW)',
            data: avgPower,
            borderColor: '#c8860d',
            borderDash: [4, 3],
            fill: false,
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 1.5,
            yAxisID: 'y1',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: 'var(--text-muted)', font: { size: 10 }, boxWidth: 10 } },
        },
        scales: {
          x: { grid: { color: 'var(--border-color)' }, ticks: { color: 'var(--text-muted)', font: { size: 9 }, maxTicksLimit: 12 } },
          y: {
            position: 'left',
            grid: { color: 'var(--border-color)' },
            ticks: { color: 'var(--text-muted)', font: { size: 10 } },
            title: { display: true, text: 'Total Power (MW)', color: 'var(--text-muted)', font: { size: 10 } },
          },
          y1: {
            position: 'right',
            grid: { display: false },
            ticks: { color: 'var(--text-muted)', font: { size: 10 } },
            title: { display: true, text: 'Avg / Turbine (kW)', color: 'var(--text-muted)', font: { size: 10 } },
          },
        },
      },
    });

    return () => fleetTrendChartInstance.current?.destroy();
  }, [fleetTrendData]);

  // ── Chart 3: GPR power curve — cluster vs theoretical ──
  useEffect(() => {
    if (!gprData) return;
    const ctx = gprChartRef.current?.getContext('2d');
    if (!ctx) return;

    gprChartInstance.current?.destroy();

    const clusterColors = ['#3a6ea5', '#c8860d', '#3f8a5b', '#6b5b95', '#c4432f'];

    const datasets = [
      {
        label: 'Theoretical Power Curve',
        data: gprData.grid_wind_speed.map((ws, i) => ({ x: ws, y: gprData.theoretical_power[i] })),
        borderColor: '#9a9584',
        borderDash: [6, 4],
        pointRadius: 0,
        borderWidth: 1.5,
        fill: false,
        type: 'line',
        order: 1,
      },
      {
        label: 'Fleet GPR Mean',
        data: gprData.grid_wind_speed.map((ws, i) => ({ x: ws, y: gprData.overall_gpr_mean[i] })),
        borderColor: '#1f2430',
        pointRadius: 0,
        borderWidth: 2.5,
        fill: false,
        type: 'line',
        order: 0,
      },
    ];

    gprData.clusters.forEach((cluster, i) => {
      const color = clusterColors[i % clusterColors.length];
      datasets.push({
        label: `Cluster ${cluster.cluster_id} (n=${cluster.point_count})`,
        data: cluster.scatter_sample.map(p => ({ x: p.wind_speed, y: p.power })),
        borderColor: color,
        backgroundColor: color + '55',
        pointRadius: 1.5,
        showLine: false,
        type: 'scatter',
        order: 3,
      });
    });

    gprChartInstance.current = new Chart(ctx, {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: 'var(--text-muted)', font: { size: 9 }, boxWidth: 8 } },
        },
        scales: {
          x: {
            type: 'linear',
            min: 0, max: 25,
            grid: { color: 'var(--border-color)' },
            ticks: { color: 'var(--text-muted)', font: { size: 10 } },
            title: { display: true, text: 'Wind Speed (m/s)', color: 'var(--text-muted)', font: { size: 10 } },
          },
          y: {
            grid: { color: 'var(--border-color)' },
            ticks: { color: 'var(--text-muted)', font: { size: 10 } },
            title: { display: true, text: 'Active Power (kW)', color: 'var(--text-muted)', font: { size: 10 } },
          },
        },
      },
    });

    return () => gprChartInstance.current?.destroy();
  }, [gprData]);

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh', gap: '12px', color: 'var(--text-primary)' }}>
        <LuLoader size={24} className="spin" />
        <span>Loading Live SCADA Data...</span>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px', background: 'var(--bg-main)', minHeight: 'calc(100vh - 100px)' }}>
      {/* KPI Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '12px', marginBottom: '24px' }}>
        <KpiCard label="FLEET HEALTH INDEX" value={`${fleetHealthIndex}%`} sub={`${activeCount} / ${totalCount} WTGs Online`} status="nominal" />
        <KpiCard label="TOTAL FLEET POWER" value={`${(totalActivePower / 1000).toFixed(2)} MW`} sub="Active generation" status="nominal" />
        <KpiCard label="AVG WIND SPEED" value={`${avgWindSpeed} m/s`} sub="Fleet live average" status="nominal" />
        <KpiCard label="TOTAL TURBINES" value={String(totalCount)} sub="All turbines" status="nominal" />
        <KpiCard label="ACTIVE TURBINES" value={String(activeCount)} sub="Producing power" status="nominal" />
        <KpiCard
          label="VOTER CONSENSUS"
          value={voterConsensus !== null ? `${voterConsensus}%` : 'N/A'}
          sub={voterConsensus !== null ? `${consensusAnomalyCount} anomalies, last 7d` : 'No anomalies in window'}
          status="nominal"
        />
      </div>

      {/* Fleet-wide power zig-zag */}
      <div className="panel" style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', padding: '16px', marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <LuZap size={16} color="var(--accent-ibm-blue)" />
            <span style={{ fontFamily: 'IBM Plex Sans, sans-serif', fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
              FLEET POWER OUTPUT — PER TURBINE (LIVE)
            </span>
          </div>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'IBM Plex Mono, monospace' }}>
            Fleet avg: {fleetAvgPower.toFixed(0)} kW
          </span>
        </div>
        <div style={{ height: '240px' }}>
          <canvas ref={zigzagChartRef} />
        </div>
      </div>

      {/* Turbine Map */}
      <div style={{ display: 'grid', gridTemplateColumns: '3fr 1fr', gap: '16px', marginBottom: '16px' }}>
        <div className="panel" style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', padding: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <LuMap size={16} color="var(--accent-ibm-blue)" />
              <span style={{ fontFamily: 'IBM Plex Sans, sans-serif', fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                WIND FARM FLEET MAP - REAL-TIME STATUS
              </span>
            </div>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'IBM Plex Mono, monospace' }}>
              {activeCount} / {totalCount} WTGs ACTIVE
            </span>
          </div>
          <TurbineMap onSelectTurbine={(wtg) => setSelectedTurbine(wtg)} turbines={turbinesList} />
        </div>

        <div className="panel" style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', padding: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <LuTriangleAlert size={16} color="var(--status-danger)" />
              <span style={{ fontFamily: 'IBM Plex Sans, sans-serif', fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                ACTIVE FAULTS
              </span>
            </div>
            <span style={{ fontSize: '10px', background: '#ffd7d9', color: 'var(--status-danger)', padding: '2px 6px', fontWeight: 600 }}>
              0 CRITICAL
            </span>
          </div>
          <FaultTable selectedTurbine={selectedTurbine} turbines={turbinesList} />
        </div>
      </div>

      {/* 90-day fleet performance trend */}
      <div className="panel" style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', padding: '16px', marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <LuTrendingUp size={16} color="var(--accent-ibm-blue)" />
            <span style={{ fontFamily: 'IBM Plex Sans, sans-serif', fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
              FLEET PERFORMANCE TREND — 90 DAYS
            </span>
          </div>
        </div>
        {fleetTrendData.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>
            No trend data available for this window.
          </div>
        ) : (
          <div style={{ height: '260px' }}>
            <canvas ref={fleetTrendChartRef} />
          </div>
        )}
      </div>

      {/* Power Curve Analysis — Cluster vs Theoretical, GPR */}
      <div className="panel" style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <LuActivity size={16} color="var(--accent-ibm-blue)" />
            <span style={{ fontFamily: 'IBM Plex Sans, sans-serif', fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
              POWER CURVE ANALYSIS — CLUSTER vs THEORETICAL (GAUSSIAN PROCESS REGRESSION)
            </span>
          </div>
          {gprData && (
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'IBM Plex Mono, monospace' }}>
              {gprData.total_points_used.toLocaleString()} points · {gprData.clusters.length} clusters
            </span>
          )}
        </div>
        {gprLoading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
            <LuLoader size={16} className="spin" />
            Fitting Gaussian Process models — this can take a few seconds…
          </div>
        ) : !gprData ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>
            GPR power curve data unavailable.
          </div>
        ) : (
          <div style={{ height: '360px' }}>
            <canvas ref={gprChartRef} />
          </div>
        )}
      </div>
    </div>
  );
}