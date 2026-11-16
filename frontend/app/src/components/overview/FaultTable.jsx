import React from 'react';

const FAULTS = [
  { wtg: 'WT-12', component: 'Main Bearing', severity: 'critical', eta: '14.2 Days', confidence: '94.2%' },
  { wtg: 'WT-07', component: 'Gearbox Oil', severity: 'high', eta: '28.5 Days', confidence: '88.7%' },
  { wtg: 'WT-21', component: 'Transformer', severity: 'medium', eta: '45.0 Days', confidence: '82.1%' }
];

export default function FaultTable() {
  const getBadgeStyle = (sev) => {
    switch (sev) {
      case 'critical': return { background: 'rgba(255,77,109,0.15)', color: '#ff4d6d', border: '1px solid rgba(255,77,109,0.3)' };
      case 'high': return { background: 'rgba(255,184,0,0.15)', color: '#ffb800', border: '1px solid rgba(255,184,0,0.3)' };
      default: return { background: 'rgba(168,85,247,0.15)', color: '#a855f7', border: '1px solid rgba(168,85,247,0.3)' };
    }
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      {/* No backend prognostics/RUL endpoint exists yet; this table is sample
          data. Labeled so it is not mistaken for live fault predictions. */}
      <div style={{
        display: 'inline-block',
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '9px',
        fontWeight: 700,
        letterSpacing: '0.04em',
        color: '#f59e0b',
        background: 'rgba(245,158,11,0.12)',
        border: '1px solid rgba(245,158,11,0.3)',
        borderRadius: '3px',
        padding: '2px 7px',
        marginBottom: '8px'
      }}>
        SAMPLE DATA — NO LIVE PROGNOSTICS ENDPOINT
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            <th style={{ fontFamily: 'JetBrains Mono', fontSize: '9px', color: 'var(--text-dim)', padding: '8px', textAlign: 'left' }}>WTG</th>
            <th style={{ fontFamily: 'JetBrains Mono', fontSize: '9px', color: 'var(--text-dim)', padding: '8px', textAlign: 'left' }}>COMPONENT</th>
            <th style={{ fontFamily: 'JetBrains Mono', fontSize: '9px', color: 'var(--text-dim)', padding: '8px', textAlign: 'left' }}>SEVERITY</th>
            <th style={{ fontFamily: 'JetBrains Mono', fontSize: '9px', color: 'var(--text-dim)', padding: '8px', textAlign: 'left' }}>ETA</th>
            <th style={{ fontFamily: 'JetBrains Mono', fontSize: '9px', color: 'var(--text-dim)', padding: '8px', textAlign: 'left' }}>CONFIDENCE</th>
          </tr>
        </thead>
        <tbody>
          {FAULTS.map((row, idx) => (
            <tr key={idx} style={{ borderBottom: '1px solid rgba(26,37,64,0.5)' }}>
              <td style={{ padding: '10px 8px', fontFamily: 'JetBrains Mono', color: 'var(--text-primary)' }}>{row.wtg}</td>
              <td style={{ padding: '10px 8px', color: 'var(--text-secondary)' }}>{row.component}</td>
              <td style={{ padding: '10px 8px' }}>
                <span style={{
                  padding: '2px 8px',
                  borderRadius: '4px',
                  fontFamily: 'JetBrains Mono',
                  fontSize: '9px',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  ...getBadgeStyle(row.severity)
                }}>
                  {row.severity}
                </span>
              </td>
              <td style={{ padding: '10px 8px', fontFamily: 'JetBrains Mono', color: 'var(--text-primary)' }}>{row.eta}</td>
              <td style={{ padding: '10px 8px', fontFamily: 'JetBrains Mono', color: 'var(--accent-secondary)' }}>{row.confidence}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}