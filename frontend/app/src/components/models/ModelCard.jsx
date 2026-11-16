import React from 'react';

export default function ModelCard({ number, title, description, metrics, titleColor }) {
  return (
    <div className="model-card" style={{
      background: 'var(--bg-card2)',
      border: '1px solid var(--border)',
      borderRadius: '8px',
      padding: '14px',
      marginBottom: '12px'
    }}>
      <div className="model-name" style={{
        fontFamily: 'Syne',
        fontWeight: 700,
        fontSize: '13px',
        marginBottom: '4px',
        color: titleColor
      }}>
        {number} {title}
      </div>
      <div className="model-desc" style={{
        fontSize: '11px',
        color: 'var(--text-secondary)',
        marginBottom: '10px',
        lineHeight: 1.5
      }}>
        {description}
      </div>
      <div className="model-metrics" style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {metrics.map((m, idx) => (
          <span key={idx} className="metric-pill" style={{
            fontFamily: 'JetBrains Mono',
            fontSize: '10px',
            padding: '3px 8px',
            borderRadius: '4px',
            background: 'rgba(0, 212, 255, 0.07)',
            color: m.color || 'var(--accent-primary)'
          }}>
            {m.label}: {m.value}
          </span>
        ))}
      </div>
    </div>
  );
}