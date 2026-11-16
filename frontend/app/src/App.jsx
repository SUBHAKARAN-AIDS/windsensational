import React, { useCallback, useState } from 'react';
import { BrowserRouter, Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import './App.css';
import Header from './components/common/Header.jsx';
import Navigation from './components/common/Navigation';
import RulPage from './pages/RulPage';
import ForecastPage from './pages/ForecastPage';
import EnsemblePage from './pages/EnsemblePage';  

// Import Page Components
import OverviewPage from './pages/OverviewPage';
import ThermalPage from './pages/ThermalPage';
import AnomalyPage from './pages/AnomalyPage';
import DependencyPage from './pages/DependencyPage';
import ModelsPage from './pages/ModelsPage';
import './index.css';

// Stub pages have no data to fetch, so they report "ready" the instant
// they mount — they should never be able to hold the navigation lock.
function StubPage({ label, onLoadingChange }) {
  React.useEffect(() => {
    onLoadingChange && onLoadingChange(false);
  }, [onLoadingChange]);
  return <div style={{ padding: '20px', color: 'var(--text-muted)' }}>{label} — coming soon</div>;
}

const PAGE_ROUTES = [
  { path: '/', label: 'Overview', component: OverviewPage },
  { path: '/thermal', label: 'Thermal Analysis', component: ThermalPage },
  { path: '/anomaly', label: 'Anomaly Detection', component: AnomalyPage },
  { path: '/dependency', label: 'Dependency Engine', component: DependencyPage },
  { path: '/models', label: 'ML Models', component: ModelsPage },
  { path: '/rul', label: 'RUL Prediction', component: RulPage },
  { path: '/forecast', label: 'Forecasting', component: ForecastPage },
  { path: '/ensemble', label: 'Ensemble Layer', component: EnsemblePage },
];

function AppContent() {
  const location = useLocation();
  const navigate = useNavigate();

  // True while the currently-mounted page's own data fetch is still in
  // flight. Pages report this up through the onLoadingChange prop passed
  // to <currentRoute.component> below — see the wiring note at the
  // bottom of this file for the one-line pattern each page needs.
  const [isPageLoading, setIsPageLoading] = useState(false);

  const currentRoute = PAGE_ROUTES.find(r => r.path === location.pathname) || PAGE_ROUTES[0];
  const currentPage = currentRoute.label;
  const currentIndex = PAGE_ROUTES.findIndex(r => r.path === location.pathname);

  const handleTabChange = (index) => {
    if (isPageLoading) return; // locked — a page is still loading, ignore the click
    if (index === currentIndex) return; // already on this page
    navigate(PAGE_ROUTES[index].path);
  };

  // Call this with `true` while a page's data is loading and `false` once
  // it settles — success OR failure. Never leave it stuck on `true`, or
  // a broken backend call would lock navigation permanently.
  const handlePageLoadingChange = useCallback((loading) => {
    setIsPageLoading(loading);
  }, []);

  return (
    <div>
      <Header currentPage={currentPage} />
      <Navigation
        activeTab={currentIndex}
        onTabChange={handleTabChange}
        disabled={isPageLoading}
      />
      <main style={{ padding: '24px 32px', position: 'relative', minHeight: '60vh' }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '16px'
        }}>
          <h2 style={{
            fontSize: '20px',
            fontWeight: 600,
            margin: 0,
            color: 'var(--text-primary)'
          }}>
            {currentPage}
          </h2>
          <span style={{
            fontSize: '12px',
            color: 'var(--text-muted)',
            background: 'var(--bg-main)',
            padding: '4px 12px',
            borderRadius: '20px'
          }}>
            {currentIndex + 1} / {PAGE_ROUTES.length}
          </span>
        </div>

        <currentRoute.component onLoadingChange={handlePageLoadingChange} />

        {isPageLoading && (
          <div style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(255,255,255,0.72)',
            backdropFilter: 'blur(1px)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '14px',
            zIndex: 300,
          }}>
            <span style={{
              width: '28px',
              height: '28px',
              border: '3px solid #E1E4E9',
              borderTop: '3px solid #0F8B8D',
              borderRadius: '50%',
              animation: 'app-spin 0.9s linear infinite',
            }} />
            <p style={{
              fontFamily: 'IBM Plex Sans, sans-serif',
              fontSize: '13px',
              color: 'var(--text-secondary, #4B5563)',
              margin: 0,
            }}>
              Loading {currentPage}… please wait
            </p>
          </div>
        )}
      </main>

      <style>{`@keyframes app-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

// ─── Wiring a page into the loading lock ─────────────────────────────
// Any page component now receives an `onLoadingChange` prop. Each page
// should call it whenever its own "is my data ready yet" state changes:
//
//   useEffect(() => {
//     onLoadingChange && onLoadingChange(loading); // or isLoading — whatever
//   }, [loading, onLoadingChange]);                // that page already calls it
//
// Call it with `false` on BOTH success and error/catch — a page whose
// fetch failed must still release the lock, or the whole app freezes.
// ThermalPage.fixed.jsx, DependencyPage.redesigned.jsx, and
// MLModelsPage.redesigned.jsx already have this wired in (see the
// updated versions delivered alongside this file). OverviewPage.jsx and
// ModelsPage.jsx need the same one-line pattern added — paste those
// files if you want them wired too.