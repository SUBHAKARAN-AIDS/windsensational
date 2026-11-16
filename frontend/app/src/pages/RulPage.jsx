import React, { useEffect, useRef, useState, useCallback } from 'react';
import Chart from 'chart.js/auto';

// ─── Design tokens ───
const T = {
  bgPage: '#f5f1e8',
  bgCard: '#ffffff',
  border: '#e2ddd0',
  borderStrong: '#d4cdb8',
  textPrimary: '#1f2430',
  textSecondary: '#5a6072',
  textDim: '#9a9584',
  accentWarn: '#c8860d',
  accentCritical: '#c4432f',
  accentStable: '#3f8a5b',
  accentInfo: '#3a6ea5',
  chipBg: '#f0ece0',
};

function statusColor(status) {
  if (status === 'critical') return T.accentCritical;
  if (status === 'monitor') return T.accentWarn;
  if (status === 'stable') return T.accentStable;
  return T.textDim;
}

function statusLabel(status) {
  if (status === 'critical') return 'CRITICAL';
  if (status === 'monitor') return 'MONITOR';
  if (status === 'stable') return 'STABLE';
  if (status === 'no_data') return 'NO DATA';
  return 'UNKNOWN';
}

// status sort priority for the fleet grid: worst first
function statusRank(status) {
  if (status === 'critical') return 0;
  if (status === 'monitor') return 1;
  if (status === 'stable') return 2;
  return 3; // no_data / unknown last
}

// ─── Fleet summary bar ───

function FleetSummary({ results }) {
  const counts = { critical: 0, monitor: 0, stable: 0, no_data: 0 };
  results.forEach(r => {
    const s = r.data?.status || 'no_data';
    counts[s] = (counts[s] || 0) + 1;
  });

  const items = [
    { key: 'critical', label: 'CRITICAL', color: T.accentCritical },
    { key: 'monitor', label: 'MONITOR', color: T.accentWarn },
    { key: 'stable', label: 'STABLE', color: T.accentStable },
    { key: 'no_data', label: 'NO DATA', color: T.textDim },
  ];

  return (
    <div style={{ display: 'flex', gap: '12px', marginBottom: '18px', flexWrap: 'wrap' }}>
      {items.map(item => (
        <div key={item.key} style={{
          background: T.bgCard,
          border: `1px solid ${T.border}`,
          borderRadius: '8px',
          padding: '12px 18px',
          flex: '1 1 140px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}>
          <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: item.color, flexShrink: 0 }} />
          <div>
            <div style={{ fontSize: '20px', fontWeight: 800, color: T.textPrimary, lineHeight: 1 }}>
              {counts[item.key] || 0}
            </div>
            <div style={{ fontSize: '10px', color: T.textDim, fontWeight: 700, letterSpacing: '0.05em' }}>
              {item.label}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Individual turbine card, with its own mini chart ───

function TurbineRulCard({ turbineId, data, loading }) {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    const history = data?.health_history || [];
    if (history.length === 0) return;
    const ctx = chartRef.current?.getContext('2d');
    if (!ctx) return;

    chartInstance.current?.destroy();

    const projected = data?.projected_health || [];
    const historyLen = history.length;
    const projectedLen = projected.length;
    const totalLen = historyLen + projectedLen;
    const threshold = data?.failure_threshold ?? 0.2;

    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array.from({ length: totalLen }, () => ''),
        datasets: [
          {
            data: [...history, ...Array(projectedLen).fill(null)],
            borderColor: statusColor(data.status),
            backgroundColor: 'transparent',
            fill: false,
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 1.75,
          },
          {
            data: [...Array(Math.max(historyLen - 1, 0)).fill(null), history[historyLen - 1], ...projected],
            borderColor: T.accentWarn,
            borderDash: [3, 3],
            backgroundColor: 'transparent',
            pointRadius: 0,
            tension: 0.3,
            borderWidth: 1.5,
          },
          {
            data: Array(totalLen).fill(threshold),
            borderColor: T.accentCritical,
            borderDash: [4, 3],
            pointRadius: 0,
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: {
          x: { display: false },
          y: { display: false, min: 0, max: 1.05 },
        },
        elements: { point: { radius: 0 } },
      },
    });

    return () => chartInstance.current?.destroy();
  }, [data]);

  const noData = !data || data.status === 'no_data';
  const rulDisplay = data?.rul_days != null ? data.rul_days.toFixed(1) : '—';
  const hasTrajectory = (data?.health_history?.length || 0) > 0;

  return (
    <div style={{
      background: T.bgCard,
      border: `1px solid ${T.border}`,
      borderLeft: `4px solid ${statusColor(data?.status)}`,
      borderRadius: '8px',
      padding: '14px 16px',
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: '14px', fontWeight: 700, color: T.textPrimary }}>{turbineId}</div>
          <div style={{ fontSize: '10px', fontWeight: 700, color: statusColor(data?.status), letterSpacing: '0.05em', marginTop: '2px' }}>
            {loading ? 'LOADING…' : statusLabel(data?.status)}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '24px', fontWeight: 800, color: noData ? T.textDim : statusColor(data?.status), lineHeight: 1 }}>
            {loading ? '…' : rulDisplay}
          </div>
          <div style={{ fontSize: '9px', color: T.textDim, fontWeight: 600 }}>
            {noData ? '' : 'DAYS'}
          </div>
        </div>
      </div>

      <div style={{ height: '48px' }}>
        {hasTrajectory ? (
          <canvas ref={chartRef} />
        ) : (
          <div style={{
            height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '10px', color: T.textDim,
          }}>
            {loading ? 'fetching…' : 'no trajectory data'}
          </div>
        )}
      </div>

      {!noData && data.confidence_pct != null && (
        <div style={{ fontSize: '10px', color: T.textSecondary, display: 'flex', justifyContent: 'space-between' }}>
          <span>Confidence: <strong style={{ color: T.textPrimary }}>{data.confidence_pct}%</strong></span>
          <span>Threshold: <strong style={{ color: T.textPrimary }}>{(data.failure_threshold ?? 0.2).toFixed(2)}</strong></span>
        </div>
      )}
    </div>
  );
}

// ─── Main page: loads all turbines automatically, no selection needed ───

export default function RulPage({ onLoadingChange }) {
  const [turbineList, setTurbineList] = useState([]);
  const [results, setResults] = useState({}); // { turbineId: rulResponseData }
  const [loadingIds, setLoadingIds] = useState(new Set());
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastFetched, setLastFetched] = useState(null);
  const [filter, setFilter] = useState('all'); // all | critical | monitor | stable | no_data

  // Fetch RUL for a single turbine
  const fetchOneRul = useCallback((id) => {
    return fetch(`/api/v1/rul/${id}`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => ({ id, data }))
      .catch(() => ({ id, data: { status: 'no_data', rul_days: null } }));
  }, []);

  // Fetch RUL for every turbine in parallel, update results as each resolves
  const fetchAll = useCallback((ids, isBackground) => {
    if (ids.length === 0) return;
    if (!isBackground) setLoadingIds(new Set(ids));

    ids.forEach(id => {
      fetchOneRul(id).then(({ id: doneId, data }) => {
        setResults(prev => ({ ...prev, [doneId]: data }));
        setLoadingIds(prev => {
          const next = new Set(prev);
          next.delete(doneId);
          return next;
        });
      });
    });

    setLastFetched(new Date());
  }, [fetchOneRul]);

  // 1. Load turbine list once, then immediately fetch RUL for all of them — no click required
  useEffect(() => {
    let cancelled = false;
    fetch('/api/v1/turbines/list')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (cancelled) return;
        const list = data.turbines || [];
        setTurbineList(list);
        setInitialLoading(false);
        fetchAll(list, false);
      })
      .catch(err => {
        if (cancelled) return;
        setError(err.message);
        setInitialLoading(false);
      });
    return () => { cancelled = true; };
  }, [fetchAll]);

  // 2. Background refresh every 60s for the whole fleet, no user action needed
  useEffect(() => {
    if (turbineList.length === 0) return;
    const interval = setInterval(() => fetchAll(turbineList, true), 60000);
    return () => clearInterval(interval);
  }, [turbineList, fetchAll]);

  // 3. Report loading state up to the app's navigation lock
  useEffect(() => {
    onLoadingChange && onLoadingChange(initialLoading);
  }, [initialLoading, onLoadingChange]);

  if (error) {
    return (
      <div style={{ padding: '20px', background: T.bgPage, minHeight: '100%' }}>
        <div style={{ background: T.bgCard, border: `1px solid ${T.border}`, borderRadius: '10px', padding: '18px', color: T.accentCritical, fontSize: '13px' }}>
          Failed to load turbine list: {error}
        </div>
      </div>
    );
  }

  if (initialLoading) {
    return (
      <div style={{ padding: '20px', background: T.bgPage, minHeight: '100%' }}>
        <div style={{ background: T.bgCard, border: `1px solid ${T.border}`, borderRadius: '10px', padding: '18px', color: T.textDim, fontSize: '13px' }}>
          Loading fleet turbine list…
        </div>
      </div>
    );
  }

  // Build the display list: attach each turbine's result, sort worst-first
  const resultList = turbineList
    .map(id => ({ id, data: results[id], loading: loadingIds.has(id) }))
    .sort((a, b) => {
      const ra = statusRank(a.data?.status);
      const rb = statusRank(b.data?.status);
      if (ra !== rb) return ra - rb;
      // within same status, sort by rul_days ascending (most urgent first)
      const da = a.data?.rul_days ?? Infinity;
      const db = b.data?.rul_days ?? Infinity;
      return da - db;
    });

  const filtered = filter === 'all' ? resultList : resultList.filter(r => (r.data?.status || 'no_data') === filter);

  const filterOptions = [
    { key: 'all', label: `All (${turbineList.length})` },
    { key: 'critical', label: 'Critical' },
    { key: 'monitor', label: 'Monitor' },
    { key: 'stable', label: 'Stable' },
    { key: 'no_data', label: 'No Data' },
  ];

  return (
    <div style={{ padding: '20px 24px', background: T.bgPage, minHeight: '100%' }}>

      {/* Header */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: '16px', flexWrap: 'wrap', gap: '10px',
      }}>
        <div style={{ fontSize: '11px', fontWeight: 700, color: T.textDim, letterSpacing: '0.12em' }}>
          ⏳ RUL PREDICTION — FLEET-WIDE REMAINING USEFUL LIFE PROGNOSTICS
        </div>
        {lastFetched && (
          <span style={{ fontSize: '11px', color: T.textDim }}>
            Last refreshed {lastFetched.toLocaleTimeString()} · {turbineList.length} turbines
          </span>
        )}
      </div>

      {/* Fleet summary counts */}
      <FleetSummary results={resultList} />

      {/* Filter tabs */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', flexWrap: 'wrap' }}>
        {filterOptions.map(opt => (
          <button
            key={opt.key}
            onClick={() => setFilter(opt.key)}
            style={{
              background: filter === opt.key ? T.textPrimary : T.bgCard,
              color: filter === opt.key ? T.bgCard : T.textSecondary,
              border: `1px solid ${filter === opt.key ? T.textPrimary : T.border}`,
              borderRadius: '20px',
              padding: '6px 14px',
              fontSize: '11px',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Fleet grid — every turbine, no selection required */}
      {filtered.length === 0 ? (
        <div style={{ background: T.bgCard, border: `1px solid ${T.border}`, borderRadius: '10px', padding: '30px', textAlign: 'center', color: T.textDim, fontSize: '13px' }}>
          No turbines match this filter.
        </div>
      ) : (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
          gap: '14px',
        }}>
          {filtered.map(({ id, data, loading }) => (
            <TurbineRulCard key={id} turbineId={id} data={data} loading={loading} />
          ))}
        </div>
      )}
    </div>
  );
}