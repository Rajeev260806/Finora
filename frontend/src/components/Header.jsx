import React from 'react';
import { Calendar, Download, FileText, Database, Home } from 'lucide-react';

export default function Header({ currentMode, setCurrentMode, onExportCsv, onExportJson, onNavigateHome }) {
  return (
    <header className="top-nav">
      <div
        className="brand-section"
        onClick={onNavigateHome}
        style={{ cursor: onNavigateHome ? 'pointer' : 'default' }}
        title="Return to Landing Page"
      >
        <div className="brand-logo">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 3v18h18" />
            <path d="m19 9-5 5-4-4-3 3" />
            <circle cx="19" cy="9" r="2" fill="currentColor" />
          </svg>
        </div>
        <div className="brand-title">
          <h1>Finora Intelligence</h1>
          <span className="brand-badge">GNN + RL Portfolio Engine</span>
        </div>
      </div>

      <div className="nav-controls">
        {onNavigateHome && (
          <button
            type="button"
            className="btn-secondary"
            onClick={onNavigateHome}
            title="Return to Landing Page"
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <Home size={14} />
            <span>Home</span>
          </button>
        )}
        {/* Mode Switcher */}
        <div className="mode-pill">
          <button
            type="button"
            className={`mode-btn ${currentMode === 'holdout' ? 'active' : ''}`}
            onClick={() => setCurrentMode('holdout')}
            title="Strict Holdout Test Period"
          >
            <Calendar size={14} />
            Holdout Test (Out-of-Sample)
          </button>
          <button
            type="button"
            className={`mode-btn ${currentMode === 'insample' ? 'active' : ''}`}
            onClick={() => setCurrentMode('insample')}
            title="10-Year Historical Run"
          >
            <Database size={14} />
            10-Yr Historical Run
          </button>
        </div>

        {/* Period Badge */}
        <div className="badge-tag">
          <span className="badge-dot"></span>
          <span>
            {currentMode === 'holdout'
              ? 'Holdout Test (2022-12-29 to 2024-12-31)'
              : '10-Year In-Sample (2015-01-02 to 2024-12-31)'}
          </span>
        </div>

        {/* Export Buttons */}
        <button
          type="button"
          className="btn-secondary"
          onClick={onExportCsv}
          title="Export Strategy Metrics CSV"
        >
          <FileText size={14} />
          CSV
        </button>

        <button
          type="button"
          className="btn-secondary"
          onClick={onExportJson}
          title="Export Complete Evaluation JSON"
        >
          <Download size={14} />
          JSON
        </button>
      </div>
    </header>
  );
}
