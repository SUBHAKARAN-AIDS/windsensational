// DependencyPage.jsx - Sensor Dependency Center
//
// What this page answers, in plain words:
//   1. Which sensor looks wrong?              (health score for every sensor)
//   2. Which sensors move together?           (colour grid: now / before / what changed)
//   3. What does one sensor depend on?        (click anything to get a plain-words explanation)
//   4. Which sensor reacts first?             (timing check)
//   5. How are the sensors connected?         (draggable connection web, coloured by health)
//
// Everything comes from the backend (which reads it from the database). No sensor names,
// limits or turbine names are typed in here. Sensor names are turned into readable wording
// by prettyName() below.
//
// Fonts: add to index.html <head> (page falls back to system fonts without them):
//   <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  BarController,
  LineController,
  Filler,
  Legend,
  Tooltip,
  Title,
} from 'chart.js';
import { Chart } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, BarController, LineController, Filler, Legend, Tooltip, Title);

// ─── LOOK (IBM Carbon light theme) ──────────────────────────────────
const FONT_SANS = "'IBM Plex Sans', 'Segoe UI', system-ui, -apple-system, sans-serif";
const FONT_MONO = "'IBM Plex Mono', 'SF Mono', 'Consolas', monospace";

const C = {
  // IBM Carbon light (Gray 10 canvas, white cards)
  bg: '#f4f4f4',
  panel: '#ffffff',
  panelHead: '#f4f4f4',
  border: '#e0e0e0',
  hairline: '#e0e0e0',
  text: '#161616',
  textSecondary: '#525252',
  textDim: '#6f6f6f',
  // relationship colours (not good/bad): blue = move together, orange = move opposite
  pos: '#0f62fe',
  neg: '#eb6200',
  // health colours
  good: '#198038',
  warn: '#b28600',
  bad: '#da1e28',
  accent: '#0f62fe',
  purple: '#8a3ffc',
};
const SERIES_COLORS = [C.pos, C.purple, C.warn, C.good];
const DAY_OPTIONS = [3, 7, 14, 30];
const EARLIER_OPTIONS = [14, 30, 60, 90];

const signColor = (v) => (v >= 0 ? C.pos : C.neg);
const fmtSigned = (v) => `${v >= 0 ? '+' : ''}${Number(v).toFixed(2)}`;
const shortDate = (s) => (s ? String(s).slice(0, 10) : '');
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ─── Readable sensor names ──────────────────────────────────────────
// The database stores names like "Gearbox De Bearing Temperature" or "Wind Speed M S".
// An engineer would say "Gearbox drive-end bearing temperature" and "Wind speed (m/s)".
// This works on any name (there is no list of sensors typed in).
const WORDS = { de: 'drive-end', nde: 'non-drive-end', gen: 'generator', temp: 'temperature', vib: 'vibration', pos: 'position', avg: 'average', rot: 'rotor', hyd: 'hydraulic', pwr: 'power', spd: 'speed', pct: '', id: 'ID' };
const UNITS = [
  [['m', 's'], 'm/s'], [['mm', 's'], 'mm/s'], [['kw'], 'kW'], [['kwh'], 'kWh'], [['kvar'], 'kVAr'], [['mw'], 'MW'],
  [['rpm'], 'rpm'], [['hz'], 'Hz'], [['kv'], 'kV'], [['deg'], '°'], [['bar'], 'bar'], [['pct'], '%'], [['percent'], '%'],
];
const prettyName = (raw) => {
  let tokens = String(raw).replace(/_+/g, ' ').trim().split(/\s+/).filter(Boolean).map((t) => t.toLowerCase());
  let unit = '';
  for (const [u, label] of UNITS) {
    const k = u.length;
    if (tokens.length > k && u.every((w, i) => tokens[tokens.length - k + i] === w)) { unit = label; tokens = tokens.slice(0, tokens.length - k); break; }
  }
  if (!unit && tokens.length > 2 && tokens[tokens.length - 1] === 'c' && tokens.some((t) => t === 'temperature' || t === 'temp')) { unit = '°C'; tokens = tokens.slice(0, -1); }
  if (tokens.length > 2 && tokens[0] === 'temperature' && ['outside', 'inside'].includes(tokens[1])) tokens = [...tokens.slice(2), tokens[1], 'temperature'];
  else if (tokens.length > 1 && tokens[0] === 'vibration') tokens = [...tokens.slice(1), 'vibration'];
  const words = tokens.map((t) => (t in WORDS ? WORDS[t] : t)).filter(Boolean);
  let out = words.join(' ');
  out = out.charAt(0).toUpperCase() + out.slice(1);
  return unit ? `${out} (${unit})` : out;
};
// Replace every sensor name in the backend answer with its readable name. `rawOf` remembers
// the stored name so the "what affects this sensor" choice still sends the stored name back.
const prettify = (data) => {
  const names = new Set();
  const add = (x) => { if (typeof x === 'string') names.add(x); };
  (data.correlation?.tags || []).forEach(add);
  (data.drift?.tags || []).forEach(add);
  (data.shap?.features || []).forEach(add);
  add(data.shap?.target);
  (data.network?.nodes || []).forEach((n) => add(n.id));
  const label = {}; const used = {};
  [...names].forEach((raw) => {
    let nice = prettyName(raw);
    if (used[nice]) nice = `${nice} [${raw}]`;
    used[nice] = true; label[raw] = nice;
  });
  const L = (x) => (typeof x === 'string' && label[x] ? label[x] : x);
  const mapPairs = (arr, keys) => (arr || []).map((o) => { const c = { ...o }; keys.forEach((k) => { c[k] = L(c[k]); }); return c; });
  return {
    ...data,
    correlation: data.correlation && { ...data.correlation, tags: (data.correlation.tags || []).map(L), top_dependencies: mapPairs(data.correlation.top_dependencies, ['a', 'b']), top_dependencies_controlled: mapPairs(data.correlation.top_dependencies_controlled, ['a', 'b']) },
    drift: data.drift && { ...data.drift, tags: (data.drift.tags || []).map(L), top_changes: mapPairs(data.drift.top_changes, ['a', 'b']) },
    shap: data.shap && { ...data.shap, target: L(data.shap.target), features: (data.shap.features || []).map(L) },
    granger: data.granger && { ...data.granger, pairs: mapPairs(data.granger.pairs, ['cause', 'effect']), differenced: (data.granger.differenced || []).map(L) },
    network: data.network && { ...data.network, nodes: mapPairs(data.network.nodes, ['id']), edges: mapPairs(data.network.edges, ['source', 'target']) },
  };
};

// ─── Plain words ────────────────────────────────────────────────────
const strengthWord = (v) => {
  const a = Math.abs(v);
  if (a >= 0.7) return 'strongly';
  if (a >= 0.4) return 'fairly closely';
  if (a >= 0.2) return 'loosely';
  return 'hardly at all';
};
const togetherWord = (v) => (Math.abs(v) < 0.2 ? 'do not follow each other' : v > 0 ? 'rise and fall together' : 'move in opposite directions');
const changeSentence = (other, baseline, current) => {
  const was = Math.abs(baseline), now = Math.abs(current);
  const pre = `With ${other}: `;
  if (was >= 0.2 && now >= 0.2 && baseline * current < 0) return `${pre}they used to ${togetherWord(baseline)}, now they ${togetherWord(current)}.`;
  if (was >= 0.2 && now < 0.2) return `${pre}they used to ${togetherWord(baseline)} (${strengthWord(baseline)}), now they hardly follow each other any more.`;
  if (was < 0.2 && now >= 0.2) return `${pre}they used to have little to do with each other, now they ${togetherWord(current)} (${strengthWord(current)}).`;
  return `${pre}they used to follow each other ${strengthWord(baseline)}, now ${strengthWord(current)}.`;
};
const fitWord = (r2) => (r2 >= 0.6 ? 'good' : r2 >= 0.3 ? 'fair' : 'weak');

// Health of one sensor: 100 = still behaving like before. Every link it has is checked;
// if ONE sensor is faulty, ALL of its own links break, so the average change is what counts.
const verdictOf = (score) => (score >= 0.25
  ? { key: 'check', label: 'Check this one', color: C.bad }
  : score >= 0.12
    ? { key: 'watch', label: 'Keep an eye on it', color: C.warn }
    : { key: 'ok', label: 'Looks normal', color: C.good });
const healthOf = (score) => Math.round(100 * (1 - clamp(score / 0.4, 0, 1)));

// ─── Small pieces ───────────────────────────────────────────────────
const Ring = ({ value, color, size = 46, label }) => {
  const r = (size - 6) / 2, circ = 2 * Math.PI * r;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ flexShrink: 0 }} role="img" aria-label={`Health ${value} out of 100`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={C.border} strokeWidth="4" />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="4" strokeLinecap="round"
        strokeDasharray={`${(value / 100) * circ} ${circ}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} />
      <text x="50%" y="50%" dy="0.35em" textAnchor="middle" fontFamily={FONT_MONO} fontSize={size * 0.3} fontWeight="600" fill={C.text}>{label !== undefined ? label : value}</text>
    </svg>
  );
};

const cellColor = (v, diag) => {
  if (diag) return '#e0e0e0';
  const a = clamp(Math.abs(v), 0, 1);
  return v >= 0 ? `rgba(15,98,254,${0.05 + a * 0.95})` : `rgba(235,98,0,${0.05 + a * 0.95})`;
};

// Colour grid: plain HTML so it grows with the number of sensors
const Heatmap = ({ tags, get, selPair, selSensor, onHover, onPick, onPickSensor, compact }) => {
  const n = tags.length;
  const showNumbers = !compact && n <= 9;
  const cell = compact ? 14 : 24;
  return (
    <div style={{ overflowX: 'auto' }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `${compact ? 'minmax(70px, 110px)' : 'minmax(110px, 210px)'} repeat(${n}, minmax(${cell}px, 1fr))`,
          gap: '2px', alignItems: 'stretch',
          minWidth: `${(compact ? 110 : 210) + n * (cell + 4)}px`,
        }}
        onMouseLeave={() => onHover(null)}
      >
        <div />
        {tags.map((t, j) => (
          <div key={`c${j}`} title={t} onClick={() => onPickSensor(t)}
            style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', fontFamily: FONT_SANS, fontSize: compact ? '8.5px' : '10px', color: selSensor === t ? C.text : C.textSecondary, fontWeight: selSensor === t ? 600 : 400, height: compact ? '90px' : '128px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', justifySelf: 'center', cursor: 'pointer' }}>
            {t}
          </div>
        ))}
        {tags.map((rowTag, i) => (
          <React.Fragment key={`r${i}`}>
            <div title={rowTag} onClick={() => onPickSensor(rowTag)}
              style={{ fontFamily: FONT_SANS, fontSize: compact ? '8.5px' : '10px', color: selSensor === rowTag ? C.text : C.textSecondary, fontWeight: selSensor === rowTag ? 600 : 400, textAlign: 'right', paddingRight: '8px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', alignSelf: 'center', cursor: 'pointer' }}>
              {rowTag}
            </div>
            {tags.map((colTag, j) => {
              const diag = i === j;
              const v = diag ? 1 : get(rowTag, colTag);
              const picked = selPair && ((selPair.a === rowTag && selPair.b === colTag) || (selPair.a === colTag && selPair.b === rowTag));
              return (
                <div key={`c${i}-${j}`}
                  onMouseEnter={() => onHover({ a: rowTag, b: colTag, v, diag })}
                  onClick={() => !diag && onPick(rowTag, colTag)}
                  style={{ background: cellColor(v, diag), height: `${cell}px`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: FONT_MONO, fontSize: '9px', color: Math.abs(v) > 0.55 && !diag ? (v > 0 ? '#ffffff' : '#161616') : C.textSecondary, cursor: diag ? 'default' : 'pointer', outline: picked ? `2px solid ${C.text}` : 'none', outlineOffset: '-2px', borderRadius: '0px' }}>
                  {showNumbers && !diag ? v.toFixed(2) : ''}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

const Panel = ({ title, right, children, style, bodyStyle }) => (
  <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: '0px', overflow: 'hidden', ...style }}>
    <div style={{ padding: '11px 16px', borderBottom: `1px solid ${C.hairline}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: C.panelHead, gap: '10px', flexWrap: 'wrap' }}>
      <span style={{ fontFamily: FONT_SANS, fontSize: '12.5px', fontWeight: 600, color: C.text, letterSpacing: '0.2px' }}>{title}</span>
      {right}
    </div>
    <div style={{ padding: '14px 16px', ...bodyStyle }}>{children}</div>
  </div>
);

const Hint = ({ children }) => (
  <p style={{ fontFamily: FONT_SANS, fontSize: '11.5px', color: C.textSecondary, lineHeight: 1.55, margin: '0 0 10px 0' }}>{children}</p>
);

const Chip = ({ color, children }) => (
  <span style={{ fontSize: '10px', fontFamily: FONT_MONO, padding: '2px 8px', borderRadius: '10px', background: `${color}22`, color, border: `1px solid ${color}55`, whiteSpace: 'nowrap' }}>{children}</span>
);

const Empty = ({ children }) => (
  <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', minHeight: '80px', color: C.textDim, fontSize: '12.5px', fontFamily: FONT_SANS, textAlign: 'center', padding: '0 20px' }}>{children}</div>
);

// ═══════════════════════════════════════════════════════════════════
const DependencyPage = ({ onLoadingChange } = {}) => {
  const [payload, setPayload] = useState(null);
  const [turbines, setTurbines] = useState([]);
  const [targets, setTargets] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [view, setView] = useState('now'); // now | controlled | beforeNow | change
  const [order, setOrder] = useState('worst'); // worst | stored
  const [hover, setHover] = useState(null);
  const [selPair, setSelPair] = useState(null);
  const [selSensor, setSelSensor] = useState(null);
  const [copied, setCopied] = useState(false);
  const [netReady, setNetReady] = useState(0);
  const [draft, setDraft] = useState({ turbine: '', days: 7, earlier: 30, target: '' });
  const svgRef = useRef(null);
  const netRef = useRef(null);
  const reqId = useRef(0);
  const simRef = useRef(null);

  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const fetchDependencyData = useCallback(async (f) => {
    const myReq = ++reqId.current;
    setIsLoading(true);
    setError(null);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90000);
    try {
      const q = new URLSearchParams({ days: String(f.days), baseline_days: String(f.earlier), limit: '5000' });
      if (f.turbine) q.set('turbine_id', f.turbine);
      if (f.target) q.set('target', f.target);
      const response = await fetch(`${API_BASE}/api/v1/analytics/dependency-engine?${q.toString()}`, {
        headers: { Accept: 'application/json' },
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Server responded with status ${response.status}: ${response.statusText}`);
      const data = await response.json();
      if (myReq !== reqId.current) return;
      if (Array.isArray(data.turbines) && data.turbines.length) setTurbines(data.turbines);
      if (data.status === 'no_data') {
        setPayload(null);
        setError(data.message || 'No SCADA data available for the selected window.');
        return;
      }
      if (data.status === 'error') throw new Error(data.message || 'The backend could not work this out - check the backend terminal.');
      if (data.shap && Array.isArray(data.shap.available_targets) && data.shap.available_targets.length) {
        setTargets(data.shap.available_targets.map((t) => ({ raw: t, label: prettyName(t) })));
      }
      setPayload(prettify(data));
      setSelPair(null);
      setSelSensor(null);
      setLastUpdate(new Date().toLocaleTimeString());
    } catch (err) {
      if (myReq !== reqId.current) return;
      setError(err.name === 'AbortError' ? 'This took too long and was stopped. Try fewer days.' : (err.message || 'Failed to fetch data'));
    } finally {
      clearTimeout(timeoutId);
      if (myReq === reqId.current) setIsLoading(false);
    }
  }, [API_BASE]);

  useEffect(() => {
    fetchDependencyData(draft);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchDependencyData]);

  useEffect(() => {
    if (typeof onLoadingChange === 'function') onLoadingChange(isLoading);
  }, [isLoading, onLoadingChange]);

  // ─── Everything worked out from the backend answer ───────────────
  const corr = payload?.correlation || { tags: [], pearson: [], controlled: [], mutual_info: [], top_dependencies: [], top_dependencies_controlled: [] };
  const drift = payload?.drift || { status: 'not_enough_baseline', tags: [], delta: [], top_changes: [] };
  const shap = payload?.shap || { features: [], values: [], direction: [], available_targets: [] };
  const granger = payload?.granger || { lags: [], pairs: [], differenced: [] };
  const network = payload?.network || { nodes: [], edges: [] };
  const driftOk = drift.status === 'ok' && (drift.tags || []).length > 1;

  const model = useMemo(() => {
    const ci = {}; (corr.tags || []).forEach((t, i) => { ci[t] = i; });
    const di = {}; (drift.tags || []).forEach((t, i) => { di[t] = i; });
    const at = (m, idx, a, b) => (idx[a] !== undefined && idx[b] !== undefined && m && m[idx[a]] ? m[idx[a]][idx[b]] : undefined);
    const now = (a, b) => at(corr.pearson, ci, a, b);
    const ctrl = (a, b) => at(corr.controlled, ci, a, b);
    const mi = (a, b) => at(corr.mutual_info, ci, a, b);
    const delta = (a, b) => at(drift.delta, di, a, b);
    const before = (a, b) => { const n = now(a, b), d = delta(a, b); return n !== undefined && d !== undefined ? n - d : undefined; };

    // one score per sensor: average size of the change over all its links
    const leads = {}; const follows = {};
    (granger.pairs || []).forEach((p) => {
      if (!p.significant) return;
      (leads[p.cause] = leads[p.cause] || []).push({ other: p.effect, lag: p.best_lag });
      (follows[p.effect] = follows[p.effect] || []).push({ other: p.cause, lag: p.best_lag });
    });
    const shapRank = {}; (shap.features || []).forEach((f, i) => { shapRank[f] = i; });
    const adj = {}; (network.edges || []).forEach((e) => {
      (adj[e.source] = adj[e.source] || []).push({ other: e.target, w: e.weight });
      (adj[e.target] = adj[e.target] || []).push({ other: e.source, w: e.weight });
    });

    let list = [];
    if (driftOk) {
      const tags = drift.tags, n = tags.length;
      list = tags.map((t, i) => {
        let sum = 0, cnt = 0, broken = 0;
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          const d = Math.abs((drift.delta[i] && drift.delta[i][j]) || 0);
          sum += d; cnt += 1; if (d >= 0.3) broken += 1;
        }
        const score = cnt ? sum / cnt : 0;
        const links = (drift.top_changes || []).filter((c) => (c.a === t || c.b === t) && Math.abs(c.change) >= 0.15).slice(0, 3)
          .map((c) => ({ other: c.a === t ? c.b : c.a, baseline: c.baseline, current: c.current, change: c.change }));
        return { tag: t, score, health: healthOf(score), verdict: verdictOf(score), broken, total: cnt, links };
      }).sort((a, b) => b.score - a.score);
    }
    const bySensor = {}; list.forEach((s) => { bySensor[s.tag] = s; });
    return { now, ctrl, mi, delta, before, list, bySensor, leads, follows, shapRank, adj };
  }, [corr, drift, driftOk, granger, shap, network]);

  const suspects = model.list;
  const topChange = driftOk && (drift.top_changes || []).length ? drift.top_changes[0] : null;
  const needAttention = suspects.filter((s) => s.verdict.key !== 'ok').length;

  // sensors in the order shown on the grid (worst first, or as stored)
  const gridTags = useMemo(() => {
    const base = corr.tags || [];
    if (order === 'worst' && suspects.length) {
      const first = suspects.map((s) => s.tag).filter((t) => base.includes(t));
      return [...first, ...base.filter((t) => !first.includes(t))];
    }
    return base;
  }, [corr, suspects, order]);

  const getters = {
    now: model.now,
    controlled: model.ctrl,
    before: model.before,
    change: (a, b) => model.delta(a, b) ?? 0,
  };
  const safe = (fn) => (a, b) => { const v = fn(a, b); return v === undefined || Number.isNaN(v) ? 0 : v; };

  const topList = useMemo(() => {
    if (view === 'change' || view === 'beforeNow') {
      return (drift.top_changes || []).map((d) => ({ a: d.a, b: d.b, value: d.change, note: `was ${fmtSigned(d.baseline)}, now ${fmtSigned(d.current)}` }));
    }
    const list = view === 'controlled' ? corr.top_dependencies_controlled : corr.top_dependencies;
    return (list || []).map((d) => ({ a: d.a, b: d.b, value: view === 'controlled' ? d.controlled : d.pearson, note: view === 'controlled' ? `before taking out the load effect: ${fmtSigned(d.pearson)}` : `also tied in a curved way: ${Number(d.mutual_info).toFixed(2)}` }));
  }, [view, corr, drift]);

  const pickPair = useCallback((a, b) => { setSelPair({ a, b }); setSelSensor(null); }, []);
  const pickSensor = useCallback((t) => { setSelSensor(t); setSelPair(null); }, []);

  // ─── Plain-words explanation of one pair ─────────────────────────
  const pairStory = (a, b) => {
    const nowV = model.now(a, b), befV = model.before(a, b), ctl = model.ctrl(a, b), mut = model.mi(a, b);
    const lines = [];
    if (nowV !== undefined) lines.push({ k: 'Now', v: `${strengthWord(nowV)} - they ${togetherWord(nowV)} (${fmtSigned(nowV)})`, color: signColor(nowV) });
    if (befV !== undefined && driftOk) {
      const moved = Math.abs(nowV - befV) >= 0.3;
      lines.push({ k: 'Before', v: `${strengthWord(befV)} - they ${togetherWord(befV)} (${fmtSigned(befV)})${moved ? ' - this has changed a lot' : ' - about the same as now'}`, color: moved ? C.warn : C.textSecondary });
    }
    if (ctl !== undefined && nowV !== undefined) {
      const mostlyLoad = Math.abs(nowV) >= 0.4 && Math.abs(ctl) < Math.abs(nowV) * 0.5;
      const reallyTied = Math.abs(ctl) >= 0.3;
      let msg, col;
      if (mostlyLoad) { msg = `Mostly yes - once the effect of the turbine working harder or easier is taken out, the link drops to ${fmtSigned(ctl)}. Both just follow the load.`; col = C.textSecondary; }
      else if (reallyTied) { msg = `No - the link stays at ${fmtSigned(ctl)} even after taking the load effect out, so these two are really tied to each other.`; col = C.good; }
      else { msg = `There is little link between them either way (${fmtSigned(ctl)} after taking the load effect out).`; col = C.textDim; }
      lines.push({ k: 'Is it just the load?', v: msg, color: col });
    }
    if (mut !== undefined && nowV !== undefined && mut > 0.15 && Math.abs(nowV) < 0.3) lines.push({ k: 'Hidden tie', v: 'They do not rise and fall in a straight line, but one still depends on the other in a curved way.', color: C.purple });
    const g = (granger.pairs || []).find((p) => (p.cause === a && p.effect === b) || (p.cause === b && p.effect === a));
    if (g) lines.push({ k: 'Who goes first', v: g.significant ? `When ${g.cause} changes, ${g.effect} follows about ${g.best_lag} reading${g.best_lag === 1 ? '' : 's'} later.` : `Checked, but neither one clearly goes first.`, color: g.significant ? C.accent : C.textDim });
    const direct = (network.edges || []).some((e) => (e.source === a && e.target === b) || (e.source === b && e.target === a));
    lines.push({ k: 'Direct connection', v: direct ? 'Yes - they are tied to each other, not just through other sensors.' : 'No - any link is only through other sensors.', color: direct ? C.good : C.textDim });
    return lines;
  };

  // ─── Written report (copy / download) ────────────────────────────
  const reportText = () => {
    if (!payload) return '';
    const out = [];
    out.push(`Sensor dependency report - ${payload.turbine_id || 'all turbines'}`);
    out.push(`Period: ${shortDate(drift.windows?.current?.start)} to ${shortDate(drift.windows?.current?.end)} (${payload.row_count.toLocaleString()} readings), ${corr.tags.length} sensors checked.`);
    if (suspects.length) {
      out.push(needAttention ? `${needAttention} sensor(s) need attention:` : 'No sensor has gone out of step compared with the earlier period.');
      suspects.filter((s) => s.verdict.key !== 'ok').forEach((s, i) => {
        out.push(`${i + 1}. ${s.tag} - health ${s.health}/100 (${s.verdict.label}). Out of step with ${s.broken} of ${s.total} other sensors.`);
        s.links.slice(0, 2).forEach((l) => out.push(`   - ${changeSentence(l.other, l.baseline, l.current)}`));
      });
    } else out.push(drift.message || 'No comparison with the earlier period was possible.');
    if (corr.top_dependencies?.[0]) out.push(`Strongest tie right now: ${corr.top_dependencies[0].a} and ${corr.top_dependencies[0].b} (${fmtSigned(corr.top_dependencies[0].pearson)}).`);
    out.push('This shows where to look first. It does not say whether the sensor itself or the part it measures is at fault.');
    return out.join('\n');
  };
  const copyReport = async () => {
    try { await navigator.clipboard.writeText(reportText()); setCopied(true); setTimeout(() => setCopied(false), 2000); } catch (e) { setCopied(false); }
  };
  const downloadCsv = () => {
    try {
      const rows = [['Sensor', 'Health (0-100)', 'Verdict', 'Links out of step', 'Links checked']];
      suspects.forEach((s) => rows.push([s.tag, s.health, s.verdict.label, s.broken, s.total]));
      const csv = rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n');
      const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
      const a = document.createElement('a'); a.href = url; a.download = 'sensor_health.csv'; a.click(); URL.revokeObjectURL(url);
    } catch (e) { /* ignore */ }
  };

  // ─── Connection web (draggable) ─────────────────────────────────
  useEffect(() => {
    if (isLoading || !svgRef.current || !network.edges.length) return undefined;
    let cancelled = false;
    import('d3').then((d3) => {
      if (cancelled || !svgRef.current) return;
      const svg = d3.select(svgRef.current);
      svg.selectAll('*').remove();
      const rect = svgRef.current.getBoundingClientRect();
      const width = rect.width || 800;
      const height = 420;
      svg.attr('viewBox', `0 0 ${width} ${height}`);
      const degree = {};
      network.edges.forEach((e) => { degree[e.source] = (degree[e.source] || 0) + 1; degree[e.target] = (degree[e.target] || 0) + 1; });
      const nodes = network.nodes.map((n) => ({ id: n.id, deg: degree[n.id] || 0 }));
      const links = network.edges.map((e) => ({ source: e.source, target: e.target, weight: e.weight }));
      const radius = (d) => 7 + Math.min(11, d.deg * 1.8);
      const link = svg.append('g').selectAll('line').data(links).enter().append('line')
        .attr('stroke', (d) => signColor(d.weight)).attr('stroke-width', (d) => 1 + Math.abs(d.weight) * 5).attr('opacity', 0.55).style('cursor', 'pointer');
      const node = svg.append('g').selectAll('g').data(nodes).enter().append('g').style('cursor', 'pointer');
      node.append('circle').attr('r', radius).attr('stroke', '#ffffff').attr('stroke-width', 2);
      node.append('text').text((d) => d.id).attr('dx', (d) => radius(d) + 5).attr('dy', '0.35em')
        .attr('font-size', 10.5).attr('font-family', FONT_SANS).attr('fill', C.text)
        .attr('paint-order', 'stroke').attr('stroke', C.panel).attr('stroke-width', 3);
      node.append('title').text((d) => `${d.id} - tied directly to ${d.deg} other sensor${d.deg === 1 ? '' : 's'}`);
      const info = svg.append('text').attr('x', 12).attr('y', 18).attr('font-size', 11).attr('font-family', FONT_SANS).attr('fill', C.textSecondary);
      link.on('mouseenter', (_, d) => info.text(`${d.source.id} and ${d.target.id}: tied ${d.weight >= 0 ? 'together' : 'in opposite directions'}`)).on('mouseleave', () => info.text(''));
      link.on('click', (_, d) => pickPair(d.source.id, d.target.id));
      node.on('click', (_, d) => pickSensor(d.id));
      const sim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id((d) => d.id).distance((d) => 190 - Math.min(80, Math.abs(d.weight) * 90)))
        .force('charge', d3.forceManyBody().strength(-900))
        .force('x', d3.forceX(width / 2 - 40).strength(0.07))
        .force('y', d3.forceY(height / 2).strength(0.1))
        .force('center', d3.forceCenter(width / 2 - 40, height / 2))
        .force('collide', d3.forceCollide(58))
        .on('tick', () => {
          nodes.forEach((d) => { d.x = Math.max(24, Math.min(width - 190, d.x)); d.y = Math.max(24, Math.min(height - 24, d.y)); });
          link.attr('x1', (d) => d.source.x).attr('y1', (d) => d.source.y).attr('x2', (d) => d.target.x).attr('y2', (d) => d.target.y);
          node.attr('transform', (d) => `translate(${d.x},${d.y})`);
        });
      simRef.current = sim;
      node.call(d3.drag()
        .on('start', (event, d) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on('end', (event, d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));
      netRef.current = { node, link };
      setNetReady((k) => k + 1);
    });
    return () => { cancelled = true; if (simRef.current) simRef.current.stop(); netRef.current = null; };
  }, [network, isLoading, pickPair, pickSensor]);

  // colour / highlight the web (health colours; selected sensor or pair stands out)
  useEffect(() => {
    const nr = netRef.current;
    if (!nr) return;
    const colorFor = (id) => (model.bySensor[id] ? model.bySensor[id].verdict.color : C.accent);
    nr.node.select('circle').attr('fill', (d) => colorFor(d.id))
      .attr('stroke', (d) => (selSensor === d.id || (selPair && (selPair.a === d.id || selPair.b === d.id)) ? C.text : '#ffffff'))
      .attr('stroke-width', (d) => (selSensor === d.id || (selPair && (selPair.a === d.id || selPair.b === d.id)) ? 3 : 2));
    const near = (id) => {
      if (selSensor) return id === selSensor || (model.adj[selSensor] || []).some((x) => x.other === id);
      if (selPair) return id === selPair.a || id === selPair.b;
      return true;
    };
    nr.node.attr('opacity', (d) => (near(d.id) ? 1 : 0.22));
    nr.link.attr('opacity', (d) => {
      if (selSensor) return d.source.id === selSensor || d.target.id === selSensor ? 0.95 : 0.07;
      if (selPair) return (d.source.id === selPair.a && d.target.id === selPair.b) || (d.source.id === selPair.b && d.target.id === selPair.a) ? 1 : 0.1;
      return 0.55;
    });
  }, [netReady, selSensor, selPair, model]);

  // ─── Chart.js settings ──────────────────────────────────────────
  const axisTicks = { color: C.textDim, font: { family: FONT_MONO, size: 9 } };
  const gridLine = { color: 'rgba(0,0,0,0.08)' };
  const tooltipBase = { backgroundColor: '#161616', borderColor: '#393939', borderWidth: 1, titleColor: '#FFFFFF', bodyColor: '#c6c6c6', titleFont: { family: FONT_SANS, size: 10, weight: '600' }, bodyFont: { family: FONT_SANS, size: 10 }, padding: 10, cornerRadius: 0 };

  const shapTotal = (shap.values || []).reduce((a, b) => a + Math.abs(b), 0) || 1;
  const shapShare = (shap.values || []).map((v) => Math.round((Math.abs(v) / shapTotal) * 1000) / 10);
  const shapChartData = {
    labels: shap.features,
    datasets: [{ label: 'Share of the total say (%)', data: shapShare, backgroundColor: (shap.direction || []).map((d) => (d < 0 ? C.neg : d > 0 ? C.pos : C.textDim)), borderRadius: 0 }],
  };
  const shapOptions = {
    responsive: true, maintainAspectRatio: false, indexAxis: 'y',
    plugins: { legend: { display: false }, tooltip: { ...tooltipBase, callbacks: { afterLabel: (ctx) => { const d = (shap.direction || [])[ctx.dataIndex]; return d > 0 ? 'When this sensor is higher, it goes up' : d < 0 ? 'When this sensor is higher, it goes down' : 'No clear direction'; } } } },
    scales: {
      x: { grid: gridLine, ticks: axisTicks, title: { display: true, text: 'Share of the total say over this sensor (%)', color: C.textDim, font: { family: FONT_SANS, size: 9 } } },
      y: { grid: { display: false }, ticks: { color: C.textSecondary, font: { family: FONT_SANS, size: 9.5 } } },
    },
  };
  const grangerMax = Math.max(1e-9, ...(granger.pairs || []).flatMap((p) => (p.f_stats || []).map((v) => Math.abs(v))));
  const grangerChartData = {
    labels: (granger.lags || []).map((l) => `${l} later`),
    datasets: (granger.pairs || []).map((p, idx) => ({ label: `${p.cause} → ${p.effect}`, data: (p.f_stats || []).map((v) => Math.round((v / grangerMax) * 1000) / 10), borderColor: SERIES_COLORS[idx % SERIES_COLORS.length], backgroundColor: SERIES_COLORS[idx % SERIES_COLORS.length], borderWidth: 2, pointRadius: 3, tension: 0.3 })),
  };
  const grangerOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: C.textSecondary, boxWidth: 10, font: { family: FONT_SANS, size: 10 } } }, tooltip: tooltipBase },
    scales: {
      x: { grid: gridLine, ticks: axisTicks },
      y: { grid: gridLine, ticks: axisTicks, title: { display: true, text: 'Clearness (100 = clearest)', color: C.textDim, font: { family: FONT_SANS, size: 9 } } },
    },
  };

  // ─── Styles ─────────────────────────────────────────────────────
  const selectStyle = { fontFamily: FONT_SANS, fontSize: '12px', padding: '7px 9px', border: `1px solid ${C.border}`, borderRadius: '0px', background: C.panel, color: C.text, maxWidth: '230px' };
  const labelStyle = { display: 'flex', flexDirection: 'column', gap: '3px', fontSize: '10.5px', color: C.textDim, fontFamily: FONT_SANS };
  const segBtn = (active) => ({ fontFamily: FONT_SANS, fontSize: '11px', padding: '5px 11px', cursor: 'pointer', border: `1px solid ${active ? C.accent : C.border}`, background: active ? `${C.accent}22` : 'transparent', color: active ? C.text : C.textSecondary, borderRadius: '0px', fontWeight: active ? 600 : 400 });
  const ghostBtn = { fontFamily: FONT_SANS, fontSize: '11px', padding: '5px 12px', cursor: 'pointer', border: `1px solid ${C.border}`, background: 'transparent', color: C.textSecondary, borderRadius: '0px' };

  const statusColor = error ? C.bad : isLoading ? C.warn : C.good;
  const lowFit = shap.r2 !== null && shap.r2 !== undefined && shap.r2 < 0.3;
  const kpi = (label, value, sub, color) => (
    <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: '0px', padding: '12px 16px', minWidth: 0 }}>
      <div style={{ fontSize: '10.5px', color: C.textDim, fontFamily: FONT_SANS, marginBottom: '4px' }}>{label}</div>
      <div style={{ fontSize: '20px', fontWeight: 600, color: color || C.text, fontFamily: FONT_MONO, overflow: 'hidden', textOverflow: 'ellipsis' }}>{value}</div>
      <div style={{ fontSize: '10.5px', color: C.textSecondary, fontFamily: FONT_SANS, marginTop: '3px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={sub}>{sub}</div>
    </div>
  );

  // ─── Inspector (click anything) ─────────────────────────────────
  const renderInspector = () => {
    if (selPair) {
      const lines = pairStory(selPair.a, selPair.b);
      return (
        <>
          <div style={{ fontSize: '13px', fontWeight: 600, color: C.text, marginBottom: '2px' }}>{selPair.a}</div>
          <div style={{ fontSize: '11px', color: C.textDim, margin: '0 0 2px 0' }}>and</div>
          <div style={{ fontSize: '13px', fontWeight: 600, color: C.text, marginBottom: '10px' }}>{selPair.b}</div>
          {lines.map((l, i) => (
            <div key={i} style={{ padding: '7px 0', borderTop: `1px solid ${C.hairline}`, fontSize: '11.5px', lineHeight: 1.5 }}>
              <div style={{ color: C.textDim, fontSize: '10px', marginBottom: '1px' }}>{l.k}</div>
              <div style={{ color: l.color }}>{l.v}</div>
            </div>
          ))}
        </>
      );
    }
    if (selSensor) {
      const s = model.bySensor[selSensor];
      const nbrs = (model.adj[selSensor] || []).slice().sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
      const leads = model.leads[selSensor] || [], follows = model.follows[selSensor] || [];
      const rank = model.shapRank[selSensor];
      return (
        <>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '10px' }}>
            {s ? <Ring value={s.health} color={s.verdict.color} size={52} /> : <Ring value={0} color={C.textDim} size={52} label="–" />}
            <div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: C.text }}>{selSensor}</div>
              <div style={{ fontSize: '11px', color: s ? s.verdict.color : C.textDim }}>{s ? s.verdict.label : 'No comparison with the earlier period'}</div>
            </div>
          </div>
          {s && s.links.length > 0 && s.links.map((l, k) => <div key={k} style={{ fontSize: '11.5px', color: C.textSecondary, lineHeight: 1.5, padding: '4px 0' }}>• {changeSentence(l.other, l.baseline, l.current)}</div>)}
          <div style={{ padding: '7px 0', borderTop: `1px solid ${C.hairline}`, fontSize: '11.5px', lineHeight: 1.5, color: C.textSecondary }}>
            <div style={{ color: C.textDim, fontSize: '10px' }}>Tied directly to</div>
            {nbrs.length ? nbrs.map((x) => x.other).join(', ') : 'No other sensor directly.'}
          </div>
          {nbrs.length > 0 && (
            <div style={{ padding: '7px 0', borderTop: `1px solid ${C.hairline}`, fontSize: '11.5px', lineHeight: 1.5, color: C.textSecondary }}>
              <div style={{ color: C.textDim, fontSize: '10px' }}>If this sensor goes wrong</div>
              These are tied straight to it and may be affected or give a false alarm: {nbrs.slice(0, 4).map((x) => x.other).join(', ')}{nbrs.length > 4 ? ` and ${nbrs.length - 4} more` : ''}.
            </div>
          )}
          {leads.length > 0 && <div style={{ padding: '7px 0', borderTop: `1px solid ${C.hairline}`, fontSize: '11.5px', color: C.accent, lineHeight: 1.5 }}>It changes first, then {leads.map((x) => `${x.other} (${x.lag} reading${x.lag === 1 ? '' : 's'} later)`).join(', ')} follow{leads.length === 1 ? 's' : ''}.</div>}
          {follows.length > 0 && <div style={{ padding: '7px 0', borderTop: `1px solid ${C.hairline}`, fontSize: '11.5px', color: C.accent, lineHeight: 1.5 }}>It follows {follows.map((x) => x.other).join(', ')}.</div>}
          {rank !== undefined && shap.target && <div style={{ padding: '7px 0', borderTop: `1px solid ${C.hairline}`, fontSize: '11.5px', color: C.textSecondary, lineHeight: 1.5 }}>Has the number {rank + 1} say over “{shap.target}”.</div>}
        </>
      );
    }
    return <Empty>Click a square in the grid, a dot in the web, a sensor name or a pair in the list to see it explained here in plain words.</Empty>;
  };

  const viewHelp = {
    now: 'Each square compares two sensors. Strong blue = when one goes up, the other goes up too. Strong orange = when one goes up, the other goes down. Pale = they have little to do with each other.',
    controlled: `Most sensors rise and fall just because the turbine is working harder or easier (that one reason explains about ${corr.shared_load_share !== null && corr.shared_load_share !== undefined ? Math.round(corr.shared_load_share * 100) : '?'}% of all the movement). This view takes that out, so what is left are the links that are real and not just "everything follows the wind".`,
    beforeNow: 'Left: how the sensors behaved in the earlier period. Right: how they behave now. Look for a row that was strongly coloured and has gone pale.',
    change: 'Now compared with the earlier period. Strong colour = these two sensors now behave very differently from before. Pale = no change. This is the quickest way to see which sensor has gone out of step.',
  };

  // ═════════════════════════ PAGE ═════════════════════════════════
  return (
    <div style={{ padding: '20px 28px 40px', maxWidth: '1500px', margin: '0 auto', background: C.bg, fontFamily: FONT_SANS, color: C.text }}>
      {/* Header + controls */}
      <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: '0px', padding: '15px 20px', marginBottom: '14px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
            <h1 style={{ fontSize: '18px', fontWeight: 700, color: C.text, margin: 0, letterSpacing: '-0.2px' }}>
              WindSentinel AI
              <span style={{ fontSize: '12px', fontWeight: 400, color: C.textDim, marginLeft: '10px' }}>Sensor Dependency Center</span>
            </h1>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: C.textSecondary, fontFamily: FONT_MONO }}>
              <span style={{ width: '7px', height: '7px', borderRadius: '50%', background: statusColor, display: 'inline-block', boxShadow: 'none' }}></span>
              <span>{isLoading ? 'Loading' : error ? 'Error' : `Updated ${lastUpdate}`}</span>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end', flexWrap: 'wrap' }}>
            <label style={labelStyle}>Turbine
              <select style={selectStyle} value={draft.turbine} onChange={(e) => setDraft({ ...draft, turbine: e.target.value })}>
                <option value="">All turbines together</option>
                {turbines.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </label>
            <label style={labelStyle}>Look at the last
              <select style={selectStyle} value={draft.days} onChange={(e) => setDraft({ ...draft, days: Number(e.target.value) })}>
                {DAY_OPTIONS.map((d) => <option key={d} value={d}>{d} days</option>)}
              </select>
            </label>
            <label style={labelStyle}>Compare with the period before that
              <select style={selectStyle} value={draft.earlier} onChange={(e) => setDraft({ ...draft, earlier: Number(e.target.value) })}>
                {EARLIER_OPTIONS.map((d) => <option key={d} value={d}>{d} days</option>)}
              </select>
            </label>
            <label style={labelStyle}>Find what affects this sensor
              <select style={selectStyle} value={draft.target} onChange={(e) => setDraft({ ...draft, target: e.target.value })}>
                <option value="">Choose for me</option>
                {targets.map((t) => <option key={t.raw} value={t.raw} title={t.raw}>{t.label}</option>)}
              </select>
            </label>
            <button onClick={() => fetchDependencyData(draft)} disabled={isLoading}
              style={{ background: C.accent, border: 'none', color: '#ffffff', padding: '9px 20px', borderRadius: '0px', cursor: isLoading ? 'not-allowed' : 'pointer', fontFamily: FONT_SANS, fontSize: '12px', fontWeight: 700, opacity: isLoading ? 0.55 : 1 }}>
              {isLoading ? 'Loading…' : 'Run analysis'}
            </button>
          </div>
        </div>
        {payload && !isLoading && (
          <div style={{ marginTop: '10px', fontSize: '11px', color: C.textDim, fontFamily: FONT_MONO }}>
            {payload.row_count.toLocaleString()} readings · {payload.turbine_id || 'all turbines'} · {shortDate(drift.windows?.current?.start)} to {shortDate(drift.windows?.current?.end)} · {corr.tags.length} sensors checked
          </div>
        )}
      </div>

      {error && (
        <div style={{ background: `${C.bad}14`, border: `1px solid ${C.bad}40`, borderLeft: `3px solid ${C.bad}`, borderRadius: '0px', padding: '11px 16px', marginBottom: '14px', color: C.bad, fontSize: '12.5px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span>{error}</span>
          <button onClick={() => fetchDependencyData(draft)} style={{ marginLeft: 'auto', background: 'transparent', border: `1px solid ${C.bad}`, color: C.bad, padding: '4px 14px', borderRadius: '0px', cursor: 'pointer', fontFamily: FONT_SANS, fontSize: '11px', fontWeight: 600 }}>Retry</button>
        </div>
      )}

      {isLoading && (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '60px 0', color: C.textSecondary, fontSize: '13px' }}>
          <span style={{ display: 'inline-block', width: '18px', height: '18px', border: `2.5px solid ${C.border}`, borderTop: `2.5px solid ${C.accent}`, borderRadius: '50%', marginRight: '12px', animation: 'spin 0.9s linear infinite' }}></span>
          Studying how the sensors behave together (can take up to a minute)
        </div>
      )}

      {!isLoading && !error && payload && (
        <>
          {/* Summary strip */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '12px', marginBottom: '14px' }}>
            {kpi('Sensors checked', corr.tags.length, `${payload.turbine_id || 'all turbines'}, ${payload.days} days`)}
            {driftOk
              ? kpi('Sensors that need a look', needAttention, needAttention ? `Worst: ${suspects[0].tag}` : 'All behaving as before', needAttention ? (suspects[0].verdict.key === 'check' ? C.bad : C.warn) : C.good)
              : kpi('Sensors that need a look', '–', drift.message || 'No earlier period to compare with')}
            {corr.top_dependencies?.[0]
              ? kpi('Strongest tie right now', fmtSigned(corr.top_dependencies[0].pearson), `${corr.top_dependencies[0].a} and ${corr.top_dependencies[0].b}`, signColor(corr.top_dependencies[0].pearson))
              : kpi('Strongest tie right now', '–', 'Not enough data')}
            {topChange
              ? kpi('Biggest single change', `${fmtSigned(topChange.baseline)} → ${fmtSigned(topChange.current)}`, `${topChange.a} and ${topChange.b}`, C.warn)
              : kpi('Biggest single change', '–', 'Nothing to compare')}
          </div>

          {/* ROW A: which sensor looks wrong + inspector */}
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.55fr) minmax(0, 1fr)', gap: '14px', marginBottom: '14px' }}>
            <Panel title="Which sensor looks wrong?"
              right={<div style={{ display: 'flex', gap: '8px' }}><button style={ghostBtn} onClick={copyReport}>{copied ? 'Copied ✓' : 'Copy report'}</button><button style={ghostBtn} onClick={downloadCsv} disabled={!suspects.length}>Download table</button></div>}>
              {suspects.length === 0 ? (
                <div style={{ fontSize: '12.5px', color: C.textSecondary, lineHeight: 1.6 }}>
                  {drift.message || 'There is not enough older data to compare with, so I cannot say which sensor changed. Choose a shorter "compare with" period or a turbine that has more history.'}
                </div>
              ) : (
                <>
                  <Hint>
                    Sensors on a turbine normally move in step - when the wind picks up, power and temperatures rise too. Each sensor is compared with how it behaved in the earlier period.
                    When one sensor goes wrong, <b>all</b> of its own links break, not just one - so a sensor that has gone out of step with many others is the likely culprit.
                    The ring is its health: 100 = behaving like before. This tells you <b>where to look first</b>; it cannot say if the sensor or the part it measures is at fault.
                  </Hint>
                  {suspects.slice(0, 6).map((sx, i) => (
                    <div key={sx.tag} onClick={() => pickSensor(sx.tag)}
                      style={{ display: 'flex', gap: '12px', padding: '10px 8px', borderTop: i ? `1px solid ${C.hairline}` : 'none', cursor: 'pointer', background: selSensor === sx.tag ? `${C.accent}14` : 'transparent', borderRadius: '0px' }}>
                      <Ring value={sx.health} color={sx.verdict.color} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
                          <span style={{ fontSize: '13px', fontWeight: 600, color: C.text }}>{sx.tag}</span>
                          <Chip color={sx.verdict.color}>{sx.verdict.label}</Chip>
                        </div>
                        <div style={{ fontSize: '11.5px', color: C.textSecondary, lineHeight: 1.55, marginTop: '4px' }}>
                          {sx.broken > 0
                            ? <>Gone out of step with <b style={{ color: C.text }}>{sx.broken} of the {sx.total}</b> other sensors.</>
                            : sx.verdict.key === 'ok' ? <>Still moving in step with the other sensors, like before.</> : <>Slightly out of step with the others.</>}
                          {sx.links.slice(0, 2).map((l, k) => <div key={k} style={{ marginTop: '2px' }}>• {changeSentence(l.other, l.baseline, l.current)}</div>)}
                          {(model.leads[sx.tag] || []).length > 0 && <div style={{ marginTop: '2px', color: C.accent }}>• It tends to change <b>before</b> {(model.leads[sx.tag]).map((x) => x.other).join(', ')} - so it may be where the trouble starts.</div>}
                        </div>
                      </div>
                    </div>
                  ))}
                  {suspects[0].verdict.key === 'ok' && <div style={{ marginTop: '6px', fontSize: '12px', color: C.good }}>No sensor has gone out of step - everything is behaving as it did in the earlier period.</div>}
                </>
              )}
            </Panel>

            <Panel title="Explain it to me" right={(selPair || selSensor) ? <button style={ghostBtn} onClick={() => { setSelPair(null); setSelSensor(null); }}>Clear</button> : null} style={{ alignSelf: 'start' }}>
              {renderInspector()}
            </Panel>
          </div>

          {/* ROW B: grid + list */}
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 1.15fr)', gap: '14px', marginBottom: '14px' }}>
            <Panel title="Which sensors move together"
              right={
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                  <button style={segBtn(view === 'now')} onClick={() => setView('now')}>Now</button>
                  <button style={segBtn(view === 'controlled')} onClick={() => setView('controlled')}>Ignore how hard the turbine works</button>
                  <button style={segBtn(view === 'beforeNow')} onClick={() => setView('beforeNow')} disabled={!driftOk}>Before and now</button>
                  <button style={segBtn(view === 'change')} onClick={() => setView('change')} disabled={!driftOk}>What changed</button>
                </div>
              }>
              <Hint>{viewHelp[view]}</Hint>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '10px', fontSize: '10.5px', color: C.textDim }}>
                Order of sensors:
                <button style={segBtn(order === 'worst')} onClick={() => setOrder('worst')} disabled={!suspects.length}>Worst first</button>
                <button style={segBtn(order === 'stored')} onClick={() => setOrder('stored')}>As stored</button>
              </div>
              {gridTags.length > 1 ? (
                view === 'beforeNow' ? (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
                    <div><div style={{ fontSize: '11px', color: C.textSecondary, marginBottom: '4px' }}>Before ({shortDate(drift.windows?.earlier?.start)} to {shortDate(drift.windows?.earlier?.end)})</div>
                      <Heatmap compact tags={gridTags} get={safe(getters.before)} selPair={selPair} selSensor={selSensor} onHover={setHover} onPick={pickPair} onPickSensor={pickSensor} /></div>
                    <div><div style={{ fontSize: '11px', color: C.textSecondary, marginBottom: '4px' }}>Now ({shortDate(drift.windows?.current?.start)} to {shortDate(drift.windows?.current?.end)})</div>
                      <Heatmap compact tags={gridTags} get={safe(getters.now)} selPair={selPair} selSensor={selSensor} onHover={setHover} onPick={pickPair} onPickSensor={pickSensor} /></div>
                  </div>
                ) : (
                  <Heatmap tags={gridTags} get={safe(getters[view])} selPair={selPair} selSensor={selSensor} onHover={setHover} onPick={pickPair} onPickSensor={pickSensor} />
                )
              ) : <Empty>No data to show</Empty>}
              <div style={{ marginTop: '10px', minHeight: '18px', fontFamily: FONT_SANS, fontSize: '11.5px', color: C.textSecondary }}>
                {hover && !hover.diag
                  ? (view === 'change'
                    ? `${hover.a} and ${hover.b}: ${Math.abs(hover.v) < 0.1 ? 'behave about the same as before' : `their link has changed by ${Math.abs(hover.v).toFixed(2)}`}`
                    : `${hover.a} and ${hover.b}: ${strengthWord(hover.v)} - they ${togetherWord(hover.v)}`)
                  : 'Point at a square to read it in words. Click it to see the full story.'}
              </div>
              <div style={{ display: 'flex', gap: '16px', fontSize: '10.5px', color: C.textDim, marginTop: '6px' }}>
                <span><span style={{ display: 'inline-block', width: '10px', height: '10px', background: C.pos, marginRight: '5px', borderRadius: '0px' }}></span>rise and fall together</span>
                <span><span style={{ display: 'inline-block', width: '10px', height: '10px', background: C.neg, marginRight: '5px', borderRadius: '0px' }}></span>one up, other down</span>
              </div>
            </Panel>

            <Panel title={view === 'change' || view === 'beforeNow' ? 'Pairs that changed most' : 'Pairs most tied together'} style={{ alignSelf: 'start' }}>
              {topList.length > 0 ? topList.slice(0, 8).map((d, i) => (
                <div key={i} onClick={() => pickPair(d.a, d.b)} style={{ marginBottom: '9px', cursor: 'pointer', padding: '4px 4px', borderRadius: '0px', background: selPair && ((selPair.a === d.a && selPair.b === d.b) || (selPair.a === d.b && selPair.b === d.a)) ? `${C.accent}14` : 'transparent' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', fontSize: '11px', color: C.textSecondary }}>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }} title={`${d.a} and ${d.b}`}>{d.a} → {d.b}</span>
                    <span style={{ fontFamily: FONT_MONO, color: signColor(d.value), fontWeight: 500, flexShrink: 0 }}>{fmtSigned(d.value)}</span>
                  </div>
                  <div style={{ height: '5px', background: C.border, borderRadius: '0px', overflow: 'hidden', margin: '4px 0' }}>
                    <div style={{ width: `${Math.min(100, Math.abs(d.value) * 100)}%`, height: '100%', background: signColor(d.value) }}></div>
                  </div>
                  <div style={{ fontSize: '10px', color: C.textDim }}>{d.note}</div>
                </div>
              )) : <div style={{ color: C.textDim, fontSize: '12px' }}>Nothing to show here.</div>}
            </Panel>
          </div>

          {/* ROW C: what affects + who reacts first */}
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '14px', marginBottom: '14px' }}>
            <Panel title={`What affects “${shap.target || '…'}”`}
              right={shap.features.length > 0 ? <Chip color={lowFit ? C.warn : C.good}>{shap.r2 !== null && shap.r2 !== undefined ? `how sure we are: ${fitWord(shap.r2)}` : 'how sure we are: unknown'}</Chip> : null}>
              <Hint>
                Which other sensors have the most say over <b>{shap.target}</b>? Longer bar = more say. Blue = when that sensor reads higher, {shap.target} goes up. Orange = when that sensor reads higher, {shap.target} goes down.
              </Hint>
              {lowFit && (
                <div style={{ background: `${C.warn}14`, borderLeft: `3px solid ${C.warn}`, padding: '6px 10px', fontSize: '11px', color: C.textSecondary, marginBottom: '8px' }}>
                  The other sensors explain this one only weakly, so treat this ranking as a hint, not an answer.
                </div>
              )}
              <div style={{ height: '250px' }}>
                {shap.features.length > 0 ? <Chart type="bar" data={shapChartData} options={shapOptions} /> : <Empty>Nothing to show here</Empty>}
              </div>
            </Panel>

            <Panel title="Which sensor reacts first" right={granger.pairs.length > 0 ? <Chip color={C.accent}>{granger.turbine_used ? `turbine ${granger.turbine_used}` : 'timing check'}</Chip> : null}>
              <Hint>When one sensor changes, does another follow a few readings later? The pairs below are the ones most tied together. “Follows” is a clue about who goes first - it does not prove one causes the other.</Hint>
              {granger.pairs.length > 0 ? (
                <>
                  <div style={{ height: '175px' }}><Chart type="line" data={grangerChartData} options={grangerOptions} /></div>
                  <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {granger.pairs.map((p, i) => (
                      <div key={i} onClick={() => pickPair(p.cause, p.effect)} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: C.textSecondary, cursor: 'pointer' }}>
                        <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: SERIES_COLORS[i % SERIES_COLORS.length], flexShrink: 0 }}></span>
                        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }} title={`When ${p.cause} changes, ${p.effect} follows`}>{p.cause} → {p.effect}</span>
                        <span style={{ fontFamily: FONT_MONO, color: C.textDim }}>{p.best_lag} readings later</span>
                        <Chip color={p.significant ? C.good : C.warn}>{p.significant ? 'clear' : 'not clear'}</Chip>
                      </div>
                    ))}
                  </div>
                  {granger.differenced.length > 0 && (
                    <div style={{ marginTop: '8px', fontSize: '10.5px', color: C.textDim }}>
                      {granger.differenced.length} slowly-drifting sensor(s) were compared by how much they changed between readings, so a slow rise over time does not create a fake link.
                    </div>
                  )}
                </>
              ) : <div style={{ height: '200px' }}><Empty>{granger.message || 'Not enough data to check which sensor goes first'}</Empty></div>}
            </Panel>
          </div>

          {/* ROW D: web */}
          <Panel title="Direct connections" right={<Chip color={network.edges.length > 0 ? C.good : C.textDim}>{network.edges.length} direct connections</Chip>} bodyStyle={{ padding: 0 }}>
            <div style={{ padding: '14px 16px 0' }}>
              <Hint>
                Each dot is a sensor, coloured by its health (green = normal, amber = keep an eye, red = check). A line joins two sensors that are <b>directly</b> tied - not just because both follow the wind.
                A big dot with many lines is a key sensor: if it goes wrong, many others notice. Click a dot or a line to see it explained, drag a dot to move it.
              </Hint>
            </div>
            {network.edges.length > 0
              ? <svg ref={svgRef} style={{ width: '100%', height: '420px', display: 'block' }}></svg>
              : <div style={{ height: '200px' }}><Empty>No sensors are directly tied strongly enough to draw</Empty></div>}
          </Panel>
        </>
      )}

      <style>{`
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: ${C.panelHead}; }
        ::-webkit-scrollbar-thumb { background: #c6c6c6; border-radius: 0; }
        button:disabled { opacity: 0.4; cursor: not-allowed !important; }
        select:focus, button:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 1px; }
        @media (max-width: 1000px) { .dep-grid { grid-template-columns: 1fr !important; } }
      `}</style>
    </div>
  );
};

export default DependencyPage;