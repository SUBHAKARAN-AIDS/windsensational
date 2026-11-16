import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  LuGauge, LuTreePine, LuShieldCheck, LuGitBranch, LuBrain, LuWaves,
  LuSearch, LuRefreshCw, LuTriangleAlert, LuCircleCheck, LuCircleX, LuCircleDot,
} from 'react-icons/lu';

const FONT_SANS = "'IBM Plex Sans', 'Segoe UI', system-ui, -apple-system, sans-serif";
const FONT_MONO = "'IBM Plex Mono', 'SF Mono', 'Consolas', monospace";

const THEME = {
  background: '#F1F3F5',
  cardBg: '#FFFFFF',
  cardBorder: '#E1E4E9',
  headerBg: '#F8F9FB',
  textPrimary: '#171B21',
  textSecondary: '#4B5563',
  textDim: '#8B94A3',
  hairline: '#E1E4E9',
  signal: '#0F8B8D',
  signalDim: '#0F8B8D14',
  signalBorder: '#0F8B8D40',
  good: '#1F9D55',
  goodDim: '#1F9D5514',
  warn: '#B7791F',
  warnDim: '#B7791F14',
  bad: '#C53030',
  badDim: '#C5303014',
  mute: '#8B94A3',
  muteDim: '#8B94A314',
};

const useDebounce = (value, delay = 300) => {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const handler = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(handler);
  }, [value, delay]);
  return debounced;
};

const MODEL_META = {
  isolation_forest: {
    icon: LuTreePine,
    name: 'Isolation Forest',
    tagline: 'Unsupervised anomaly detection',
    detail: 'Isolates unusual readings by how few splits it takes to separate them from the rest.',
  },
  elliptic_envelope: {
    icon: LuShieldCheck,
    name: 'Elliptic Envelope',
    tagline: 'Robust covariance outlier detection',
    detail: 'Fits a boundary around normal operating conditions; anything outside it gets flagged.',
  },
  xgboost: {
    icon: LuGitBranch,
    name: 'XGBoost',
    tagline: 'Gradient boosted trees',
    detail: 'Learned classifier trained on labeled fault patterns from historical SCADA data.',
  },
  lstm_autoencoder: {
    icon: LuBrain,
    name: 'LSTM Autoencoder',
    tagline: 'Sequence reconstruction',
    detail: 'Learns normal time-series behavior; large reconstruction error signals a deviation.',
  },
  vae_autoencoder: {
    icon: LuWaves,
    name: 'Variational Autoencoder',
    tagline: 'Probabilistic reconstruction',
    detail: 'Available in the model registry but not currently part of the ensemble vote.',
  },
};

const ROLE_STYLE = {
  voter: { label: 'Voting', color: THEME.signal, dim: THEME.signalDim, border: THEME.signalBorder, Icon: LuCircleCheck },
  score_only: { label: 'Advisory', color: THEME.warn, dim: THEME.warnDim, border: THEME.warn + '40', Icon: LuCircleDot },
  excluded: { label: 'Excluded', color: THEME.bad, dim: THEME.badDim, border: THEME.bad + '40', Icon: LuCircleX },
  unknown: { label: 'Not loaded', color: THEME.mute, dim: THEME.muteDim, border: THEME.mute + '40', Icon: LuCircleX },
};

const SEVERITY_STYLE = {
  high: { color: THEME.bad, dim: THEME.badDim },
  medium: { color: THEME.warn, dim: THEME.warnDim },
  low: { color: THEME.good, dim: THEME.goodDim },
};

const MLModelsPage = ({ onLoadingChange } = {}) => {
  const [status, setStatus] = useState(null);
  const [statusError, setStatusError] = useState(null);

  const [turbines, setTurbines] = useState([]);
  const [selectedTurbine, setSelectedTurbine] = useState('');
  const [turbineQuery, setTurbineQuery] = useState('');
  const debouncedQuery = useDebounce(turbineQuery, 300);

  const [anomalyData, setAnomalyData] = useState(null);
  const [isLoadingAnomalies, setIsLoadingAnomalies] = useState(true);
  const [anomalyError, setAnomalyError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  const [trainingStatus, setTrainingStatus] = useState(null);
  const [confirmingRetrain, setConfirmingRetrain] = useState(false);
  const [isStartingRetrain, setIsStartingRetrain] = useState(false);
  const [retrainMessage, setRetrainMessage] = useState(null); // { type: 'ok' | 'error', text }
  const wasTrainingRef = useRef(false);

  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/ml/models/status`);
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();
      setStatus(data);
      setStatusError(null);
    } catch (err) {
      console.error('[MLModelsPage] Error fetching model status:', err);
      setStatusError(err.message || 'Failed to fetch model status');
    }
  }, [API_BASE]);

  const fetchTrainingStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/ml/training/status`);
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();
      setTrainingStatus(data);
    } catch (err) {
      console.error('[MLModelsPage] Error fetching training status:', err);
    }
  }, [API_BASE]);

  const triggerRetrain = useCallback(async () => {
    setIsStartingRetrain(true);
    setRetrainMessage(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/ml/training/trigger`, { method: 'POST' });
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();
      if (data.status === 'already_running') {
        setRetrainMessage({ type: 'error', text: 'Training is already running. Please wait for it to finish.' });
      } else {
        setRetrainMessage({ type: 'ok', text: 'Training started. This can take several minutes.' });
      }
      fetchTrainingStatus();
    } catch (err) {
      console.error('[MLModelsPage] Error triggering retrain:', err);
      setRetrainMessage({ type: 'error', text: `Could not start training: ${err.message}` });
    } finally {
      setIsStartingRetrain(false);
      setConfirmingRetrain(false);
    }
  }, [API_BASE, fetchTrainingStatus]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const isTraining = Boolean(trainingStatus?.is_training);

  // Check every 15 s normally, every 3 s while training is running.
  useEffect(() => {
    fetchTrainingStatus();
    const interval = setInterval(fetchTrainingStatus, isTraining ? 3000 : 15000);
    return () => clearInterval(interval);
  }, [fetchTrainingStatus, isTraining]);

  // When training finishes, refresh the model list and tell the user.
  useEffect(() => {
    if (wasTrainingRef.current && !isTraining) {
      fetchStatus();
      setRetrainMessage({ type: 'ok', text: 'Training finished. See the result in the list below.' });
    }
    wasTrainingRef.current = isTraining;
  }, [isTraining, fetchStatus]);

  useEffect(() => {
    const loadTurbines = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/turbines/live`);
        const data = await res.json();
        const list = data.turbines || data || [];
        setTurbines(list);
        if (list.length > 0) {
          const firstId = list[0].turbine_id || list[0].id || list[0];
          setSelectedTurbine(firstId);
        }
      } catch (err) {
        console.error('[MLModelsPage] Failed to load turbine list:', err);
      }
    };
    loadTurbines();
  }, [API_BASE]);

  const fetchAnomalies = useCallback(async () => {
    if (!selectedTurbine) {
      setAnomalyData(null);
      setLastUpdate(null);
      return;
    }
    setIsLoadingAnomalies(true);
    setAnomalyError(null);
    try {
      const params = new URLSearchParams({
        turbine_id: selectedTurbine,
        days: '7',
        limit: '20',
        min_score: '0.3',
      });
      const res = await fetch(`${API_BASE}/api/v1/ml/ensemble-anomaly-detection?${params}`);
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setAnomalyData(data);
      setLastUpdate(new Date().toLocaleTimeString());
    } catch (err) {
      console.error('[MLModelsPage] Error fetching ensemble results:', err);
      setAnomalyError(err.message || 'Failed to fetch ensemble results');
    } finally {
      setIsLoadingAnomalies(false);
      if (typeof onLoadingChange === 'function') {
        onLoadingChange(false);
      }
    }
  }, [API_BASE, selectedTurbine, onLoadingChange]);

  useEffect(() => {
    fetchAnomalies();
  }, [fetchAnomalies]);

  useEffect(() => {
    if (typeof onLoadingChange === 'function' && isLoadingAnomalies === false) {
      onLoadingChange(false);
    }
  }, [isLoadingAnomalies, onLoadingChange]);

  const refreshAll = () => {
    fetchStatus();
    fetchTrainingStatus();
    fetchAnomalies();
  };

  const votingModels = status?.participation_policy?.voting_models || [];
  const scoreOnlyModels = status?.participation_policy?.score_only_models || [];
  const excludedModels = status?.participation_policy?.excluded_models || {};

  const roleFor = (key) => {
    if (votingModels.includes(key)) return 'voter';
    if (scoreOnlyModels.includes(key)) return 'score_only';
    if (key in excludedModels) return 'excluded';
    return 'unknown';
  };

  const filteredTurbines = turbines.filter((t) => {
    const id = String(t.turbine_id || t.id || t);
    return id.toLowerCase().includes(debouncedQuery.toLowerCase());
  });

  const renderModelCard = (key) => {
    const meta = MODEL_META[key];
    if (!meta) return null;
    const Icon = meta.icon;
    const loaded = status?.models?.[key];
    const role = roleFor(key);
    const roleInfo = (!loaded && role === 'unknown') ? ROLE_STYLE.unknown : ROLE_STYLE[role];
    const RoleIcon = roleInfo.Icon;

    return (
      <div
        key={key}
        style={{
          display: 'flex',
          gap: '14px',
          padding: '14px',
          border: `1px solid ${THEME.hairline}`,
          borderRadius: '6px',
          background: loaded ? THEME.cardBg : THEME.headerBg,
          opacity: loaded ? 1 : 0.65,
        }}
      >
        <div style={{
          flexShrink: 0,
          width: '42px',
          height: '42px',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: roleInfo.dim,
          border: `1.5px solid ${roleInfo.border}`,
        }}>
          <Icon size={19} style={{ color: roleInfo.color }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
            <span style={{ fontFamily: FONT_SANS, fontSize: '13px', fontWeight: 600, color: THEME.textPrimary }}>
              {meta.name}
            </span>
            <span style={{
              display: 'flex', alignItems: 'center', gap: '4px',
              fontFamily: FONT_MONO, fontSize: '9.5px', fontWeight: 600,
              padding: '2px 8px', borderRadius: '2px',
              background: roleInfo.dim, color: roleInfo.color, border: `1px solid ${roleInfo.border}`,
              whiteSpace: 'nowrap',
            }}>
              <RoleIcon size={10} /> {roleInfo.label.toUpperCase()}
            </span>
          </div>
          <div style={{ fontFamily: FONT_SANS, fontSize: '11px', color: THEME.textSecondary, marginTop: '2px' }}>
            {meta.tagline}
          </div>
          <div style={{ fontFamily: FONT_SANS, fontSize: '10.5px', color: THEME.textDim, marginTop: '5px', lineHeight: 1.5 }}>
            {excludedModels[key] || meta.detail}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ padding: '20px 32px', maxWidth: '1440px', margin: '0 auto', background: THEME.background, minHeight: '100vh', fontFamily: FONT_SANS }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px',
        marginBottom: '18px', padding: '15px 20px', background: THEME.cardBg,
        borderRadius: '6px', border: `1px solid ${THEME.cardBorder}`,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <LuGauge size={18} style={{ color: THEME.signal }} />
          <span style={{ fontFamily: FONT_SANS, fontSize: '18px', fontWeight: 700, color: THEME.textPrimary }}>
            ML Model Ensemble
            <span style={{ fontSize: '12px', fontWeight: 400, color: THEME.textDim, marginLeft: '10px' }}>
              Live anomaly voting
            </span>
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontFamily: FONT_MONO, fontSize: '11px', color: THEME.textSecondary }}>
          {lastUpdate && !isLoadingAnomalies && !anomalyError && (
            <span style={{ color: THEME.textDim }}>Updated {lastUpdate}</span>
          )}
          <button
            onClick={refreshAll}
            disabled={isLoadingAnomalies}
            style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              background: THEME.signal, border: 'none', color: '#fff',
              padding: '7px 16px', borderRadius: '4px',
              cursor: isLoadingAnomalies ? 'not-allowed' : 'pointer',
              fontFamily: FONT_SANS, fontSize: '12px', fontWeight: 600,
              opacity: isLoadingAnomalies ? 0.55 : 1,
            }}
          >
            <LuRefreshCw size={12} style={{ animation: isLoadingAnomalies ? 'spin 1s linear infinite' : 'none' }} />
            {isLoadingAnomalies ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {(status?.scoring_notes || []).length > 0 && (
        <div style={{ background: THEME.warnDim, border: `1px solid ${THEME.warn}40`, borderLeft: `3px solid ${THEME.warn}`, borderRadius: '4px', padding: '10px 14px', marginBottom: '12px', fontSize: '12px', color: THEME.textSecondary, lineHeight: 1.5 }}>
          <b style={{ color: THEME.textPrimary }}>Scoring notes</b>
          <ul style={{ margin: '4px 0 0 16px', padding: 0 }}>
            {status.scoring_notes.map((n, i) => <li key={i}>{n}</li>)}
          </ul>
        </div>
      )}
      {statusError && (
        <div style={{ background: THEME.badDim, border: `1px solid ${THEME.bad}30`, borderLeft: `3px solid ${THEME.bad}`, borderRadius: '4px', padding: '10px 14px', marginBottom: '12px', color: THEME.bad, fontSize: '12.5px' }}>
          Model status unavailable — {statusError}
        </div>
      )}
      {anomalyError && (
        <div style={{ background: THEME.badDim, border: `1px solid ${THEME.bad}30`, borderLeft: `3px solid ${THEME.bad}`, borderRadius: '4px', padding: '10px 14px', marginBottom: '12px', color: THEME.bad, fontSize: '12.5px' }}>
          Ensemble detection unavailable — {anomalyError}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', alignItems: 'start' }}>

        <div style={{ background: THEME.cardBg, border: `1px solid ${THEME.cardBorder}`, borderRadius: '6px', overflow: 'hidden' }}>
          <div style={{ padding: '11px 16px', borderBottom: `1px solid ${THEME.hairline}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: THEME.headerBg }}>
            <span style={{ fontFamily: FONT_SANS, fontSize: '12.5px', fontWeight: 600, color: THEME.textPrimary }}>Model stack</span>
            <span style={{ fontFamily: FONT_MONO, fontSize: '9.5px', padding: '2px 8px', borderRadius: '2px', background: THEME.signalDim, color: THEME.signal, border: `1px solid ${THEME.signalBorder}` }}>
              {status?.total_voters ?? votingModels.length} voting · {scoreOnlyModels.length} advisory
            </span>
          </div>
          <div style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {Object.keys(MODEL_META).map(renderModelCard)}
          </div>
        </div>

        <div style={{ background: THEME.cardBg, border: `1px solid ${THEME.cardBorder}`, borderRadius: '6px', overflow: 'hidden' }}>
          <div style={{ padding: '11px 16px', borderBottom: `1px solid ${THEME.hairline}`, background: THEME.headerBg }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <span style={{ fontFamily: FONT_SANS, fontSize: '12.5px', fontWeight: 600, color: THEME.textPrimary }}>
                Recent anomalies — {selectedTurbine || '—'}
              </span>
              <span style={{ fontFamily: FONT_MONO, fontSize: '9.5px', padding: '2px 8px', borderRadius: '2px', background: THEME.signalDim, color: THEME.signal, border: `1px solid ${THEME.signalBorder}` }}>
                {anomalyData ? `${anomalyData.anomaly_count ?? 0} found` : '…'}
              </span>
            </div>

            <div style={{ position: 'relative', marginBottom: '8px' }}>
              <LuSearch size={12} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: THEME.textDim }} />
              <input
                value={turbineQuery}
                onChange={(e) => setTurbineQuery(e.target.value)}
                placeholder="Find a turbine…"
                style={{
                  width: '100%', padding: '6px 10px 6px 28px', borderRadius: '4px',
                  border: `1px solid ${THEME.hairline}`, background: THEME.cardBg,
                  fontFamily: FONT_SANS, fontSize: '11.5px', color: THEME.textPrimary,
                  boxSizing: 'border-box', outline: 'none',
                }}
              />
            </div>

            <div style={{ display: 'flex', gap: '6px', overflowX: 'auto', paddingBottom: '2px' }}>
              {filteredTurbines.length === 0 ? (
                <span style={{ fontFamily: FONT_SANS, fontSize: '11px', color: THEME.textDim }}>No turbines match "{debouncedQuery}"</span>
              ) : (
                filteredTurbines.map((t) => {
                  const id = t.turbine_id || t.id || t;
                  const active = id === selectedTurbine;
                  return (
                    <button
                      key={id}
                      onClick={() => setSelectedTurbine(id)}
                      style={{
                        flexShrink: 0,
                        fontFamily: FONT_MONO, fontSize: '11px', fontWeight: active ? 600 : 500,
                        padding: '5px 12px', borderRadius: '4px', cursor: 'pointer',
                        border: `1px solid ${active ? THEME.signal : THEME.hairline}`,
                        background: active ? THEME.signalDim : THEME.cardBg,
                        color: active ? THEME.signal : THEME.textSecondary,
                      }}
                      aria-selected={active}
                      role="button"
                    >
                      {id}
                    </button>
                  );
                })
              )}
            </div>
          </div>

          {trainingStatus && (
            <div style={{ padding: '12px 16px', borderBottom: `1px solid ${THEME.hairline}`, background: THEME.headerBg }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ fontFamily: FONT_SANS, fontSize: '12px', fontWeight: 600, color: THEME.textPrimary }}>
                  Model Training
                </span>
                <button
                  onClick={() => { setRetrainMessage(null); setConfirmingRetrain(true); }}
                  disabled={isTraining || confirmingRetrain}
                  style={{
                    fontFamily: FONT_SANS, fontSize: '11px', fontWeight: 600,
                    padding: '5px 12px', borderRadius: '4px',
                    cursor: (isTraining || confirmingRetrain) ? 'not-allowed' : 'pointer',
                    opacity: (isTraining || confirmingRetrain) ? 0.55 : 1,
                    border: `1px solid ${THEME.signalBorder}`,
                    background: THEME.signalDim, color: THEME.signal,
                  }}
                >
                  {isTraining ? 'Training…' : 'Retrain Now'}
                </button>
              </div>
              {trainingStatus.retrain_recommended && !isTraining && !confirmingRetrain && (
                <div style={{
                  background: THEME.warnDim, border: `1px solid ${THEME.warn}40`,
                  borderLeft: `3px solid ${THEME.warn}`, borderRadius: '4px',
                  padding: '8px 12px', marginBottom: '10px',
                  fontFamily: FONT_SANS, fontSize: '11px', color: THEME.textSecondary, lineHeight: 1.5,
                }}>
                  <b style={{ color: THEME.textPrimary }}>New data available.</b>{' '}
                  {trainingStatus.last_trained_at
                    ? `${Number(trainingStatus.new_rows).toLocaleString('en-IN')} new rows since the last training. Retraining is recommended.`
                    : `The models have not been retrained from this page yet (${Number(trainingStatus.new_rows).toLocaleString('en-IN')} rows since the starting point). Retraining is optional but recommended.`}
                </div>
              )}
              {confirmingRetrain && (
                <div style={{
                  background: THEME.warnDim, border: `1px solid ${THEME.warn}40`,
                  borderLeft: `3px solid ${THEME.warn}`, borderRadius: '4px',
                  padding: '10px 12px', marginBottom: '10px',
                }}>
                  <div style={{ fontFamily: FONT_SANS, fontSize: '12px', fontWeight: 600, color: THEME.textPrimary }}>
                    Are you sure you want to retrain the models?
                  </div>
                  <div style={{ fontFamily: FONT_SANS, fontSize: '11px', color: THEME.textSecondary, marginTop: '4px', lineHeight: 1.5 }}>
                    The models will be rebuilt from the last 90 days of SCADA data. This can take several minutes.
                    The current models are kept as a backup.
                  </div>
                  <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
                    <button
                      onClick={triggerRetrain}
                      disabled={isStartingRetrain}
                      style={{
                        fontFamily: FONT_SANS, fontSize: '11px', fontWeight: 600,
                        padding: '5px 12px', borderRadius: '4px', border: 'none',
                        background: THEME.signal, color: '#fff',
                        cursor: isStartingRetrain ? 'not-allowed' : 'pointer',
                        opacity: isStartingRetrain ? 0.55 : 1,
                      }}
                    >
                      {isStartingRetrain ? 'Starting…' : 'Yes, retrain'}
                    </button>
                    <button
                      onClick={() => setConfirmingRetrain(false)}
                      disabled={isStartingRetrain}
                      style={{
                        fontFamily: FONT_SANS, fontSize: '11px', fontWeight: 600,
                        padding: '5px 12px', borderRadius: '4px',
                        border: `1px solid ${THEME.hairline}`, background: THEME.cardBg,
                        color: THEME.textSecondary, cursor: 'pointer',
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
              {isTraining && (
                <div style={{ fontFamily: FONT_SANS, fontSize: '11px', color: THEME.warn, marginBottom: '8px' }}>
                  Training is in progress. This page checks every 3 seconds.
                </div>
              )}
              {retrainMessage && !isTraining && (
                <div style={{
                  fontFamily: FONT_SANS, fontSize: '11px', marginBottom: '8px',
                  color: retrainMessage.type === 'error' ? THEME.bad : THEME.good,
                }}>
                  {retrainMessage.text}
                </div>
              )}
              <div style={{ fontFamily: FONT_MONO, fontSize: '10.5px', color: THEME.textSecondary, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span>Last trained: {trainingStatus.last_trained_at ?? 'never'}</span>
                <span>Rows at last training: {trainingStatus.last_trained_row_count ?? '—'}</span>
                <span>Current row count: {trainingStatus.current_row_count ?? '—'}</span>
                <span>New rows since last training: {trainingStatus.new_rows ?? '—'} (recommend retraining at {trainingStatus.row_threshold ?? '—'})</span>
              </div>
              {Array.isArray(trainingStatus.history) && trainingStatus.history.length > 0 && (
                <ul style={{ margin: '8px 0 0 0', padding: '0 0 0 16px', fontFamily: FONT_MONO, fontSize: '10px', color: THEME.textDim }}>
                  {trainingStatus.history.map((run, i) => (
                    <li key={i} style={{ marginBottom: '2px' }}>
                      {run.started_at} — {run.status} ({run.trigger}){run.error ? ` — ${run.error}` : ''}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <div style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '420px', overflowY: 'auto' }}>
            {anomalyData?.anomalies?.length > 0 ? (
              anomalyData.anomalies.map((a, i) => {
                const sev = SEVERITY_STYLE[a.severity] || SEVERITY_STYLE.low;
                return (
                  <div key={i} style={{
                    padding: '10px 12px', borderRadius: '4px',
                    borderLeft: `3px solid ${sev.color}`, background: sev.dim,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <LuTriangleAlert size={12} style={{ color: sev.color }} />
                      <span style={{ fontFamily: FONT_MONO, fontSize: '11px', fontWeight: 700, color: sev.color }}>
                        {(a.severity || 'low').toUpperCase()} · {typeof a.score === 'number' ? a.score.toFixed(2) : '—'}
                      </span>
                    </div>
                    <div style={{ fontFamily: FONT_SANS, fontSize: '11px', color: THEME.textSecondary, marginTop: '4px' }}>
                      {new Date(a.timestamp).toLocaleString()} — {a.ensemble_votes}/{a.total_voters} voters agreed
                      {a.voters_detected?.length > 0 && ` (${a.voters_detected.join(', ')})`}
                      {a.score_only_signals?.length > 0 && ` · advisory: ${a.score_only_signals.join(', ')}`}
                    </div>
                  </div>
                );
              })
            ) : (
              <div style={{ fontFamily: FONT_SANS, fontSize: '12.5px', color: THEME.textDim, padding: '20px 0', textAlign: 'center' }}>
                {isLoadingAnomalies ? 'Loading…' : 'No anomalies above threshold in this window.'}
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        .spin-icon { animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: ${THEME.headerBg}; }
        ::-webkit-scrollbar-thumb { background: #C7CCD3; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #ADB4BE; }
      `}</style>
    </div>
  );
};

export default MLModelsPage;