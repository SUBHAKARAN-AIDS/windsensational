import { useEffect, useMemo, useRef, useState } from 'react';
import Chart from 'chart.js/auto';
import { FaChartLine, FaClock, FaThermometerHalf, FaWaveSquare, FaWind } from 'react-icons/fa';
import { getLiveTelemetry } from '../services/analyticsApi';

const T = { bgPage: '#f5f1e8', bgCard: '#fff', border: '#e2ddd0', text: '#1f2430', muted: '#5a6072', dim: '#9a9584', info: '#3a6ea5', warn: '#c8860d', critical: '#c4432f', stable: '#3f8a5b', chip: '#f0ece0' };

function Card({ title, icon, badge, children }) {
  return <section style={{ background: T.bgCard, border: `1px solid ${T.border}`, borderRadius: 10, padding: '18px 20px' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: T.text, fontSize: 12, fontWeight: 700 }}>{icon}{title}</div>
      {badge && <span style={{ color: T.muted, background: T.chip, borderRadius: 20, padding: '4px 10px', fontSize: 10, fontWeight: 700 }}>{badge}</span>}
    </div>{children}
  </section>;
}

function ChartPanel({ type = 'line', labels, datasets, yTitle, height = 230 }) {
  const canvas = useRef(null);
  const chart = useRef(null);
  useEffect(() => {
    if (!canvas.current || !labels.length) return undefined;
    chart.current?.destroy();
    chart.current = new Chart(canvas.current, { type, data: { labels, datasets }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: T.muted, font: { size: 10 }, boxWidth: 10 } } }, scales: { x: { grid: { color: T.border }, ticks: { color: T.dim, font: { size: 9 }, maxTicksLimit: 10 } }, y: { grid: { color: T.border }, ticks: { color: T.dim, font: { size: 10 } }, title: { display: Boolean(yTitle), text: yTitle, color: T.dim, font: { size: 10 } } } } } });
    return () => chart.current?.destroy();
  }, [labels, datasets, type, yTitle]);
  return <div style={{ height }}><canvas ref={canvas} /></div>;
}

function Stat({ label, value, sub, color }) { return <div style={{ background: T.chip, border: `1px solid ${T.border}`, borderRadius: 8, padding: '10px 14px', flex: '1 1 150px' }}><div style={{ color: T.dim, fontSize: 9, fontWeight: 700, marginBottom: 4 }}>{label}</div><div style={{ color: color || T.text, fontSize: 17, fontWeight: 700 }}>{value}</div><div style={{ color: T.muted, fontSize: 10, marginTop: 3 }}>{sub}</div></div>; }

function buildForecastData(turbines) {
  const labels = turbines.map(t => t.id);
  const temperatures = turbines.map(t => Number(t.bearingTemp) || 0);
  const powers = turbines.map(t => Number(t.activePower) || 0);
  const winds = turbines.map(t => Number(t.windSpeed) || 0);
  const projection = values => values.map(value => Math.max(0, value * 1.04));
  return { labels, temperatures, powers, winds, projectedTemp: projection(temperatures), projectedPower: projection(powers) };
}

export default function ForecastPage({ onLoadingChange }) {
  const [turbines, setTurbines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    onLoadingChange?.(true);
    getLiveTelemetry().then(data => { if (active) setTurbines(data.turbines || []); }).catch(err => { if (active) setError(err.message || 'Unable to load telemetry'); }).finally(() => { if (active) { setLoading(false); onLoadingChange?.(false); } });
    return () => { active = false; };
  }, [onLoadingChange]);

  const data = useMemo(() => buildForecastData(turbines), [turbines]);
  const average = values => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  const tempTrend = average(data.projectedTemp) - average(data.temperatures);
  const powerAverage = average(data.powers);

  if (loading) return <div style={{ padding: 24, background: T.bgPage, color: T.dim }}>Loading forecast inputs from database...</div>;
  if (error) return <div style={{ padding: 24, background: T.bgPage, color: T.critical }}>Forecast data unavailable: {error}</div>;
  if (!data.labels.length) return <div style={{ padding: 24, background: T.bgPage, color: T.dim }}>No telemetry is available for forecasting.</div>;

  return <div style={{ padding: '20px 24px', background: T.bgPage, minHeight: '100%' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10, marginBottom: 18 }}><div style={{ color: T.dim, fontSize: 11, fontWeight: 700, letterSpacing: '.12em' }}>FORECASTING - LIVE TELEMETRY PROJECTION</div><span style={{ color: T.stable, background: '#e3efe6', borderRadius: 20, padding: '4px 10px', fontSize: 10, fontWeight: 700 }}>DATABASE DATA</span></div>
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}><Stat label="FLEET TEMPERATURE" value={`${average(data.temperatures).toFixed(1)} C`} sub={`${data.labels.length} turbines`} color={T.critical} /><Stat label="PROJECTED TEMP DELTA" value={`+${tempTrend.toFixed(1)} C`} sub="one-step telemetry projection" color={T.warn} /><Stat label="ACTIVE POWER AVG" value={`${powerAverage.toFixed(1)} kW`} sub="latest database readings" color={T.info} /><Stat label="WIND SPEED AVG" value={`${average(data.winds).toFixed(1)} m/s`} sub="latest database readings" color={T.stable} /></div>
    <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: 16, marginBottom: 16 }}>
      <Card title="BEARING TEMPERATURE - FLEET PROJECTION" icon={<FaThermometerHalf color={T.critical} />} badge="LIVE INPUT"><ChartPanel labels={data.labels} yTitle="C" datasets={[{ label: 'Current', data: data.temperatures, borderColor: T.text, pointRadius: 2, borderWidth: 2 }, { label: 'Projection', data: data.projectedTemp, borderColor: T.critical, borderDash: [5, 3], pointRadius: 2, borderWidth: 2 }]} /></Card>
      <Card title="WIND SPEED" icon={<FaWind color={T.info} />} badge="DATABASE"><ChartPanel labels={data.labels} yTitle="m/s" datasets={[{ label: 'Latest wind speed', data: data.winds, backgroundColor: 'rgba(58,110,165,.55)', borderRadius: 4 }]} type="bar" height={230} /></Card>
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: 16 }}>
      <Card title="ACTIVE POWER PROJECTION" icon={<FaChartLine color={T.info} />} badge="LATEST SCADA"><ChartPanel labels={data.labels} yTitle="kW" datasets={[{ label: 'Current power', data: data.powers, borderColor: T.info, backgroundColor: 'rgba(58,110,165,.12)', fill: true, tension: .25, pointRadius: 2 }, { label: 'Projection', data: data.projectedPower, borderColor: T.stable, borderDash: [5, 3], pointRadius: 2 }]} /></Card>
      <Card title="FLEET SIGNAL SUMMARY" icon={<FaWaveSquare color={T.warn} />} badge="REFRESHED FROM DB"><div style={{ color: T.muted, fontSize: 12, lineHeight: 1.7 }}>The charts use the latest SCADA row for every turbine returned by the database. Forecast values are one-step projections from those readings; longer horizons require a backend forecast series.</div><div style={{ marginTop: 14, color: T.dim, fontSize: 10 }}>Turbines represented: {data.labels.join(', ')}</div><div style={{ marginTop: 10, color: T.dim, fontSize: 10 }}><FaClock /> Source: live fleet telemetry</div></Card>
    </div>
  </div>;
}
