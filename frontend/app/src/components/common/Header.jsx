import React, { useEffect, useState } from 'react';
import { FaBolt, FaWifi, FaServer, FaClock, FaMicrochip } from 'react-icons/fa';

const T = {
  bg: '#EDE6D6',
  border: '#d4cdb8',
  textPrimary: '#2a2a2a',
  textDim: '#8a8574',
  accentStable: '#3f8a5b',
  accentWarn: '#c8860d',
  accentCritical: '#c4432f',
  accentInfo: '#3a6ea5',
  divider: '#c9c1ab',
};

function LiveClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', lineHeight: 1.15 }}>
      <span style={{
        fontFamily: '"JetBrains Mono", "Courier New", monospace',
        fontSize: '13px',
        fontWeight: 700,
        color: T.textPrimary,
        letterSpacing: '0.02em',
      }}>
        {now.toLocaleTimeString([], { hour12: true })}
      </span>
      <span style={{
        fontSize: '9px',
        color: T.textDim,
        fontWeight: 600,
        letterSpacing: '0.05em',
      }}>
        {now.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })}
      </span>
    </div>
  );
}

function TelemetryChip({ icon, label, value, accent }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
      padding: '4px 10px',
      background: 'rgba(255,255,255,0.5)',
      border: `1px solid ${T.border}`,
      borderRadius: '5px',
    }}>
      <span style={{ color: accent || T.textDim, fontSize: '11px', display: 'flex' }}>{icon}</span>
      <span style={{
        fontFamily: '"JetBrains Mono", "Courier New", monospace',
        fontSize: '10px',
        color: T.textDim,
        fontWeight: 600,
        letterSpacing: '0.03em',
      }}>
        {label}
      </span>
      <span style={{
        fontFamily: '"JetBrains Mono", "Courier New", monospace',
        fontSize: '11px',
        color: T.textPrimary,
        fontWeight: 700,
      }}>
        {value}
      </span>
    </div>
  );
}

function StatusPulse({ status = 'operational' }) {
  const config = {
    operational: { color: T.accentStable, label: 'OPERATIONAL' },
    degraded: { color: T.accentWarn, label: 'DEGRADED' },
    down: { color: T.accentCritical, label: 'OFFLINE' },
  }[status];

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
      <span style={{ position: 'relative', display: 'flex', width: '8px', height: '8px' }}>
        <span style={{
          position: 'absolute', inset: 0, borderRadius: '50%',
          background: config.color, opacity: 0.5,
          animation: 'header-pulse 1.8s ease-out infinite',
        }} />
        <span style={{
          position: 'relative', width: '8px', height: '8px', borderRadius: '50%',
          background: config.color,
        }} />
      </span>
      <span style={{
        fontFamily: '"JetBrains Mono", "Courier New", monospace',
        fontSize: '10px',
        fontWeight: 700,
        color: config.color,
        letterSpacing: '0.06em',
      }}>
        {config.label}
      </span>
    </div>
  );
}

export default function Header({
  currentPage = 'Overview',
  systemStatus = 'operational',   // 'operational' | 'degraded' | 'down' — wire to real backend health later
  turbineCount = 79,               // wire to /api/v1/turbines/list count
  connectionStatus = 'connected',  // 'connected' | 'connecting' | 'error'
}) {
  const connColor = connectionStatus === 'connected' ? T.accentStable
    : connectionStatus === 'connecting' ? T.accentWarn
    : T.accentCritical;

  return (
    <header style={{
      background: T.bg,
      borderBottom: `1px solid ${T.border}`,
      position: 'sticky',
      top: 0,
      zIndex: 100,
    }}>
      {/* Main bar */}
      <div style={{
        height: '54px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
      }}>
        {/* Left: brand + breadcrumb */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '30px', height: '30px', borderRadius: '6px',
            background: T.textPrimary, display: 'flex', alignItems: 'center',
            justifyContent: 'center', color: T.bg, fontSize: '15px', flexShrink: 0,
          }}>
            <FaBolt />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.1 }}>
            <span style={{
              fontWeight: 800, fontSize: '14px', color: T.textPrimary,
              letterSpacing: '0.01em',
            }}>
              KOKOS MAXIMO
            </span>
            <span style={{
              fontSize: '9px', color: T.textDim, fontWeight: 600,
              letterSpacing: '0.08em',
            }}>
              WIND FLEET INTELLIGENCE
            </span>
          </div>

          <span style={{ color: T.divider, fontSize: '18px', fontWeight: 200, margin: '0 2px' }}>/</span>

          <span style={{
            fontFamily: '"JetBrains Mono", "Courier New", monospace',
            fontSize: '13px',
            fontWeight: 600,
            color: T.textPrimary,
          }}>
            {currentPage}
          </span>
        </div>

        {/* Right: live clock */}
        <LiveClock />
      </div>

      {/* Telemetry strip */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '6px 24px',
        borderTop: `1px solid ${T.border}`,
        background: 'rgba(255,255,255,0.25)',
        flexWrap: 'wrap',
        gap: '8px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap' }}>
          <StatusPulse status={systemStatus} />
          <span style={{ color: T.divider }}>|</span>
          <TelemetryChip icon={<FaMicrochip />} label="FLEET" value={`${turbineCount} WTG`} accent={T.accentInfo} />
          <TelemetryChip icon={<FaServer />} label="API" value={connectionStatus.toUpperCase()} accent={connColor} />
          <TelemetryChip icon={<FaWifi />} label="LATENCY" value="24ms" accent={T.accentStable} />
        </div>
      </div>

      <style>{`
        @keyframes header-pulse {
          0% { transform: scale(1); opacity: 0.5; }
          70% { transform: scale(2.4); opacity: 0; }
          100% { transform: scale(2.4); opacity: 0; }
        }
      `}</style>
    </header>
  );
}