import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import TabsNav from './components/TabsNav';
import ExecutiveCockpitTab from './components/tabs/ExecutiveCockpitTab';
import NetworkGraphTab from './components/tabs/NetworkGraphTab';
import AllocationsTab from './components/tabs/AllocationsTab';
import TransactionFrictionTab from './components/tabs/TransactionFrictionTab';
import SimulatorTab from './components/tabs/SimulatorTab';
import Footer from './components/Footer';
import LandingPage from './components/LandingPage';
import { fetchAllData } from './api/finoraApi';

export default function App() {
  const [appData, setAppData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentMode, setCurrentMode] = useState('holdout'); // 'holdout' | 'insample'
  const [activeTab, setActiveTab] = useState('tab-cockpit');
  const [currentView, setCurrentView] = useState('landing'); // 'landing' | 'dashboard'

  // Load dataset
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchAllData();
      setAppData(data);
    } catch (err) {
      console.error('Failed to load Finora data:', err);
      setError('Could not connect to model data service.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Export CSV handler
  const handleExportCsv = useCallback(() => {
    if (!appData?.trajectories_test?.strategies) return;

    let csv = 'Strategy,Final_Value,Cumulative_Return,Annualized_Return,Sharpe_Ratio,Volatility,Max_Drawdown,Average_Turnover\n';
    const strats = appData.trajectories_test.strategies;

    Object.entries(strats).forEach(([k, s]) => {
      const m = s.metrics || {};
      csv += `"${k}",${m['Final Portfolio Value'] || 0},${m['Cumulative Return'] || 0},${m['Annualized Return'] || 0},${m['Sharpe Ratio'] || 0},${m['Volatility'] || 0},${m['Max Drawdown'] || 0},${m['Average Turnover'] || 0}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    downloadBlob(blob, 'finora_strategy_comparison.csv');
  }, [appData]);

  // Export JSON handler
  const handleExportJson = useCallback(() => {
    if (!appData) return;
    const blob = new Blob([JSON.stringify(appData, null, 2)], { type: 'application/json' });
    downloadBlob(blob, 'finora_model_evaluation.json');
  }, [appData]);

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  if (currentView === 'landing') {
    return (
      <LandingPage onEnterDashboard={() => setCurrentView('dashboard')} />
    );
  }

  return (
    <div className="app-container">
      <Header
        currentMode={currentMode}
        setCurrentMode={setCurrentMode}
        onExportCsv={handleExportCsv}
        onExportJson={handleExportJson}
        onNavigateHome={() => setCurrentView('landing')}
      />

      <TabsNav activeTab={activeTab} setActiveTab={setActiveTab} />

      {loading && (
        <div style={{ textAlign: 'center', padding: '6rem 0', color: '#fbbf24' }}>
          <div style={{ fontSize: '2rem', marginBottom: '1rem', animation: 'spin 1s linear infinite' }}>⟳</div>
          <div style={{ fontFamily: "'Outfit', sans-serif", fontSize: '1.2rem', fontWeight: 600 }}>
            Initializing Finora Intelligence Engine...
          </div>
          <div style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: '0.5rem' }}>
            Loading out-of-sample models, correlations, and trajectories
          </div>
        </div>
      )}

      {error && !loading && (
        <div style={{ textAlign: 'center', padding: '4rem 0', color: '#f43f5e' }}>
          <div style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '0.5rem' }}>{error}</div>
          <button
            type="button"
            className="btn-secondary"
            onClick={loadData}
            style={{ marginTop: '1rem' }}
          >
            Retry Connection
          </button>
        </div>
      )}

      {!loading && appData && (
        <main>
          {activeTab === 'tab-cockpit' && (
            <ExecutiveCockpitTab appData={appData} currentMode={currentMode} />
          )}
          {activeTab === 'tab-network' && (
            <NetworkGraphTab appData={appData} />
          )}
          {activeTab === 'tab-weights' && (
            <AllocationsTab appData={appData} />
          )}
          {activeTab === 'tab-friction' && (
            <TransactionFrictionTab appData={appData} />
          )}
          {activeTab === 'tab-simulator' && (
            <SimulatorTab appData={appData} />
          )}
        </main>
      )}

      <Footer />
    </div>
  );
}
