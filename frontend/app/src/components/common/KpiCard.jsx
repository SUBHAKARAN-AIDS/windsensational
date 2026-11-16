import React from 'react';
import { LuTrendingUp, LuTrendingDown } from 'react-icons/lu';

export default function KpiCard({ label, value, sub, trend, status = 'nominal' }) {
  const getStatusColor = () => {
    switch (status) {
      case 'danger': return 'var(--status-danger)';
      case 'warning': return 'var(--status-warning)';
      default: return 'var(--accent-ibm-blue)';
    }
  };

  const isUp = trend?.includes('+') || trend?.includes('▲');

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border-color)',
      borderTop: `3px solid ${getStatusColor()}`,
      padding: '16px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between'
    }}>
      <div style={{
        fontSize: '11px',
        fontWeight: 600,
        color: 'var(--text-secondary)',
        textTransform: 'uppercase',
        letterSpacing: '0.5px'
      }}>
        {label}
      </div>

      <div style={{
        fontFamily: 'IBM Plex Mono, monospace',
        fontSize: '28px',
        fontWeight: 500,
        color: 'var(--text-primary)',
        margin: '12px 0 4px'
      }}>
        {value}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: 'var(--text-muted)' }}>
        <span>{sub}</span>
        {trend && (
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontFamily: 'IBM Plex Mono, monospace' }}>
            {isUp ? <LuTrendingUp size={14} color="var(--status-nominal)" /> : <LuTrendingDown size={14} color="var(--status-danger)" />}
            {trend}
          </span>
        )}
      </div>
    </div>
  );
}