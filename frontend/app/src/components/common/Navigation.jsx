import React from 'react';
import { 
  LuLayoutDashboard, 
  LuThermometer, 
  LuTriangleAlert,
  LuNetwork, 
  LuCpu, 
  LuHourglass, 
  LuTrendingUp, 
  LuLayers 
} from 'react-icons/lu';

const TABS = [
  { label: 'OVERVIEW', icon: LuLayoutDashboard, pageName: 'Overview' },
  { label: 'THERMAL ANALYSIS', icon: LuThermometer, pageName: 'Thermal Analysis' },
  { label: 'ANOMALY DETECTION', icon: LuTriangleAlert, pageName: 'Anomaly Detection' },
  { label: 'DEPENDENCY ENGINE', icon: LuNetwork, pageName: 'Dependency Engine' },
  { label: 'ML MODELS', icon: LuCpu, pageName: 'ML Models' },
  { label: 'RUL PREDICTION', icon: LuHourglass, pageName: 'RUL Prediction' },
  { label: 'FORECASTING', icon: LuTrendingUp, pageName: 'Forecasting' },
  { label: 'ENSEMBLE LAYER', icon: LuLayers, pageName: 'Ensemble Layer' }
];

// `disabled` is true while App.jsx reports the current page's data is
// still loading — tabs go inert and visibly dimmed so a user can't jump
// to another page mid-fetch and trigger overlapping requests.
export default function Navigation({ activeTab, onTabChange, disabled = false }) {
  return (
    <nav style={{
      display: 'flex',
      background: 'var(--bg-card)',
      borderBottom: '1px solid var(--border-color)',
      padding: '0 24px',
      overflowX: 'auto',
      boxSizing: 'border-box',
      position: 'sticky',
      top: '52px',
      zIndex: 99
    }}>
      {TABS.map((tab, idx) => {
        const Icon = tab.icon;
        const isActive = activeTab === idx;

        return (
          <button
            key={tab.label}
            onClick={() => { if (!disabled) onTabChange(idx, tab.pageName); }}
            disabled={disabled}
            title={disabled ? 'Please wait for the current page to finish loading' : undefined}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '12px 16px',
              fontFamily: 'IBM Plex Sans, sans-serif',
              fontSize: '12px',
              fontWeight: isActive ? 600 : 400,
              cursor: disabled ? 'not-allowed' : 'pointer',
              whiteSpace: 'nowrap',
              background: isActive ? 'var(--bg-main)' : 'transparent',
              border: 'none',
              borderBottom: isActive ? '2px solid var(--accent-ibm-blue)' : '2px solid transparent',
              color: isActive ? 'var(--accent-ibm-blue)' : 'var(--text-secondary)',
              outline: 'none',
              transition: 'all 0.15s ease',
              opacity: disabled && !isActive ? 0.4 : 1,
            }}
          >
            <Icon size={15} color={isActive ? 'var(--accent-ibm-blue)' : 'var(--text-muted)'} />
            <span>{tab.label}</span>
            {isActive && (
              <span style={{
                fontSize: '9px',
                background: 'var(--accent-ibm-blue)',
                color: '#fff',
                padding: '1px 8px',
                borderRadius: '12px',
                marginLeft: '4px'
              }}>
                {disabled ? 'Loading…' : 'Active'}
              </span>
            )}
          </button>
        );
      })}
    </nav>
  );
}