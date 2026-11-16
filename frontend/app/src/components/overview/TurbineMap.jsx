// Cleaned & Validated Turbine Animated SVG Icon
const TurbineIcon = ({ status }) => {
  const color =
    status === 'danger'
      ? '#da1e28'
      : status === 'warning'
      ? '#b28600'
      : status === 'operational'
      ? '#0f62fe'
      : status === 'offline' || status === 'unknown'
      ? '#8d8d8d'
      : '#8d8d8d';

  return (
    <svg
      width="28"
      height="28"
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .blades { transform-origin: 12px 9px; animation: spin 3s linear infinite; }
      `}</style>
      
      {/* Tower & Base */}
      <path d="M12 9v12M9 21h6" />
      
      {/* Rotor & Blades */}
      <circle cx="12" cy="9" r="1.5" fill={color} />
      <g className={status === 'operational' ? 'blades' : ''}>
        <path d="M12 9 C11 5 12 1 12 1 C12 1 13 5 12 9" fill={color} />
        <path d="M12 9 C15 11 19 12 19 12 C19 12 15 13 12 9" fill={color} />
        <path d="M12 9 C9 13 5 16 5 16 C5 16 8 12 12 9" fill={color} />
      </g>
    </svg>
  );
};

export default function TurbineMap({ onSelectTurbine, turbines = [] }) {
  const displayTurbines = turbines.map((turbine) => {
    const power = Number(turbine.active_power ?? turbine.activePower ?? 0);
    const rawStatus = String(turbine.status || '').toLowerCase();
    const status = rawStatus === 'operational'
      ? 'operational'
      : rawStatus === 'idle' || rawStatus === 'offline'
      ? 'offline'
      : rawStatus === 'warning'
      ? 'warning'
      : rawStatus === 'danger' || rawStatus === 'critical'
      ? 'danger'
      : 'unknown';

    return { ...turbine, status, label: `${power.toFixed(0)} kW` };
  });

  const getStyle = (status) => {
    switch (status) {
      case 'warning':
        return { bg: '#fff8e6', border: '#f1c21b', text: '#b28600' };
      case 'danger':
        return { bg: '#fff0f1', border: '#ff7eb6', text: '#da1e28' };
      case 'offline':
        return { bg: '#f4f4f4', border: '#e0e0e0', text: '#8d8d8d' };
      case 'unknown':
        return { bg: '#f4f4f4', border: '#e0e0e0', text: '#8d8d8d' };
      default:
        return { bg: '#edf5ff', border: '#a6c8ff', text: '#0f62fe' };
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: '8px' }}>
      {displayTurbines.map((wtg, index) => {
        const style = getStyle(wtg.status);
        return (
          <div
            key={`${wtg.id || wtg.name}-${index}`}
            onClick={() => onSelectTurbine && onSelectTurbine(wtg)}
            style={{
              background: style.bg,
              border: `1px solid ${style.border}`,
              borderRadius: '4px',
              padding: '10px 4px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            <TurbineIcon status={wtg.status} />
            <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '11px', fontWeight: 700, color: '#161616', marginTop: '6px' }}>
              {wtg.id || wtg.name}
            </div>
            <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '10px', color: style.text, marginTop: '2px' }}>
              {wtg.label}
            </div>
          </div>
        );
      })}
    </div>
  );
}