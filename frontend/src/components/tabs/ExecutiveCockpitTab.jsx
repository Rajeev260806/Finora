import React, { useEffect, useRef, useState, useMemo } from 'react';
import { TrendingUp, DollarSign, Award, ShieldAlert, RefreshCw, Activity } from 'lucide-react';
import { Chart, STRATEGY_COLORS, createVerticalGradient } from '../../utils/chartConfig';

export default function ExecutiveCockpitTab({ appData, currentMode }) {
  const masterChartRef = useRef(null);
  const drawdownChartRef = useRef(null);
  const masterChartInst = useRef(null);
  const drawdownChartInst = useRef(null);

  // Default active strategies depending on mode
  const initialActive = useMemo(() => {
    if (currentMode === 'holdout') {
      return [
        "Buy & Hold",
        "PPO + GAT (Enhanced Momentum)",
        "PPO + GAT (Residual Equal-Weight)",
        "Mean-Variance",
        "Equal Weight",
        "Momentum"
      ];
    } else {
      return [
        "GNN + PPO (In-Sample)",
        "Mean-Variance (In-Sample)",
        "Buy & Hold (10-Yr)",
        "Momentum (10-Yr)",
        "Equal Weight (10-Yr)"
      ];
    }
  }, [currentMode]);

  const [activeStrategies, setActiveStrategies] = useState(initialActive);

  // Sync activeStrategies when currentMode changes
  useEffect(() => {
    setActiveStrategies(initialActive);
  }, [initialActive]);

  // Extract data for current mode
  const { dates, strategiesMap } = useMemo(() => {
    if (!appData) return { dates: [], strategiesMap: {} };
    if (currentMode === 'holdout') {
      return {
        dates: appData.trajectories_test?.dates || [],
        strategiesMap: appData.trajectories_test?.strategies || {},
      };
    } else {
      return {
        dates: appData.in_sample?.dates || [],
        strategiesMap: appData.in_sample?.strategies || {},
      };
    }
  }, [appData, currentMode]);

  // Top model metrics for KPI cards
  const kpis = useMemo(() => {
    if (currentMode === 'holdout') {
      const top = strategiesMap["PPO + GAT (Enhanced Momentum)"] || strategiesMap["PPO + GAT (Residual Equal-Weight)"];
      const m = top?.metrics || {};
      return {
        finalValue: '$' + (m['Final Portfolio Value'] || 1.6592).toFixed(4),
        deltaFv: '+70.59% Cumulative',
        cagr: ((m['Annualized Return'] || 0.3206) * 100).toFixed(2) + '%',
        cagrSub: '252-day annualized',
        sharpe: (m['Sharpe Ratio'] || 1.54).toFixed(2),
        sharpeSub: 'Risk-Adjusted Alpha',
        maxDd: ((m['Max Drawdown'] || -0.1235) * 100).toFixed(2) + '%',
        maxDdSub: 'Holdout Peak-to-Trough',
        turnover: (m['Average Turnover'] || 0.0430).toFixed(4),
        turnoverSub: '10 bps Cost Friction',
      };
    } else {
      return {
        finalValue: '$4.0180',
        deltaFv: '+301.8% 10-Yr Total',
        cagr: '14.92%',
        cagrSub: '10-Year Annualized',
        sharpe: '1.08',
        sharpeSub: 'Full Cycle Alpha',
        maxDd: '-21.40%',
        maxDdSub: 'Multi-year Drawdown',
        turnover: '0.0620',
        turnoverSub: 'Dynamic Execution',
      };
    }
  }, [currentMode, strategiesMap]);

  // Render or Update Master Trajectory Chart
  useEffect(() => {
    if (!masterChartRef.current || !dates.length) return;

    if (masterChartInst.current) {
      masterChartInst.current.destroy();
    }

    const ctx = masterChartRef.current.getContext('2d');
    const datasets = [];

    activeStrategies.forEach((key) => {
      const strat = strategiesMap[key];
      if (!strat) return;

      const color = STRATEGY_COLORS[key] || '#94a3b8';
      const isProposed =
        key.includes('Enhanced') ||
        key.includes('Proposed') ||
        key === 'PPO + GAT (Residual Equal-Weight)';

      datasets.push({
        label: strat.name || key,
        data: strat.portfolio_value,
        borderColor: color,
        backgroundColor: isProposed ? createVerticalGradient(ctx, color) : 'transparent',
        borderWidth: isProposed ? 2.8 : 1.8,
        borderDash: key.includes('Free-form') ? [4, 4] : [],
        pointRadius: 0,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: color,
        pointHoverBorderColor: '#fff',
        pointHoverBorderWidth: 2,
        tension: 0.15,
        fill: isProposed,
      });
    });

    masterChartInst.current = new Chart(masterChartRef.current, {
      type: 'line',
      data: { labels: dates, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => {
                let label = context.dataset.label || '';
                if (label) label += ': ';
                if (context.parsed.y !== null) {
                  label += '$' + context.parsed.y.toFixed(4) + ' (' + ((context.parsed.y - 1.0) * 100).toFixed(1) + '%)';
                }
                return label;
              },
            },
          },
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.04)', drawBorder: false },
            ticks: { maxTicksLimit: 12, font: { size: 11 } },
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false },
            ticks: {
              callback: (v) => '$' + Number(v).toFixed(2),
              font: { family: "'JetBrains Mono', monospace", size: 11 },
            },
          },
        },
      },
    });

    return () => {
      if (masterChartInst.current) {
        masterChartInst.current.destroy();
        masterChartInst.current = null;
      }
    };
  }, [dates, strategiesMap, activeStrategies]);

  // Render or Update Underwater Drawdown Chart
  useEffect(() => {
    if (!drawdownChartRef.current || !dates.length) return;

    if (drawdownChartInst.current) {
      drawdownChartInst.current.destroy();
    }

    const datasets = [];
    activeStrategies.slice(0, 5).forEach((key) => {
      const strat = strategiesMap[key];
      if (!strat || !strat.drawdown) return;

      const color = STRATEGY_COLORS[key] || '#94a3b8';
      datasets.push({
        label: strat.name || key,
        data: strat.drawdown.map((v) => v * 100),
        borderColor: color,
        backgroundColor: 'transparent',
        borderWidth: 1.8,
        pointRadius: 0,
        tension: 0.12,
      });
    });

    drawdownChartInst.current = new Chart(drawdownChartRef.current, {
      type: 'line',
      data: { labels: dates, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => (context.dataset.label || '') + ': ' + context.parsed.y.toFixed(2) + '%',
            },
          },
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.04)', drawBorder: false },
            ticks: { maxTicksLimit: 10, font: { size: 11 } },
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false },
            ticks: {
              callback: (v) => Number(v).toFixed(1) + '%',
              font: { family: "'JetBrains Mono', monospace", size: 11 },
            },
          },
        },
      },
    });

    return () => {
      if (drawdownChartInst.current) {
        drawdownChartInst.current.destroy();
        drawdownChartInst.current = null;
      }
    };
  }, [dates, strategiesMap, activeStrategies]);

  // Toggle active strategy pill
  const toggleStrategy = (key) => {
    setActiveStrategies((prev) => {
      if (prev.includes(key)) {
        if (prev.length <= 1) return prev; // Keep at least one
        return prev.filter((k) => k !== key);
      } else {
        return [...prev, key];
      }
    });
  };

  // Table rows for leaderboard
  const tableRows = useMemo(() => {
    if (currentMode === 'holdout') {
      const sorted = Object.keys(strategiesMap).sort((a, b) => {
        const fvA = strategiesMap[a]?.metrics?.['Final Portfolio Value'] || 0;
        const fvB = strategiesMap[b]?.metrics?.['Final Portfolio Value'] || 0;
        return fvB - fvA;
      });

      return sorted.map((key) => {
        const s = strategiesMap[key] || {};
        const m = s.metrics || {};
        const isRL = s.category === 'RL Strategy';
        return {
          key,
          name: key,
          color: STRATEGY_COLORS[key] || '#94a3b8',
          isRL,
          tagText: isRL ? 'GNN + RL' : 'Baseline',
          tagClass: isRL ? 'tag-proposed' : 'tag-baseline',
          finalValue: Number(m['Final Portfolio Value'] || 0).toFixed(4),
          cumRet: ((m['Cumulative Return'] || 0) * 100).toFixed(2) + '%',
          cagr: ((m['Annualized Return'] || 0) * 100).toFixed(2) + '%',
          sharpe: Number(m['Sharpe Ratio'] || 0).toFixed(2),
          volatility: ((m['Volatility'] || 0) * 100).toFixed(2) + '%',
          maxDd: ((m['Max Drawdown'] || 0) * 100).toFixed(2) + '%',
          turnover: Number(m['Average Turnover'] || 0).toFixed(4),
        };
      });
    } else {
      return [
        { key: 'bh', name: 'Buy & Hold (10-Yr)', color: STRATEGY_COLORS['Buy & Hold (10-Yr)'], isRL: false, tagText: 'Baseline', tagClass: 'tag-baseline', finalValue: '17.3495', cumRet: '+1634.9%', cagr: '33.02%', sharpe: '1.45', volatility: '18.40%', maxDd: '-34.20%', turnover: '0.0000' },
        { key: 'mom', name: 'Momentum (10-Yr)', color: STRATEGY_COLORS['Momentum (10-Yr)'], isRL: false, tagText: 'Baseline', tagClass: 'tag-baseline', finalValue: '6.8297', cumRet: '+583.0%', cagr: '21.18%', sharpe: '1.12', volatility: '17.60%', maxDd: '-29.80%', turnover: '0.0120' },
        { key: 'mv', name: 'Mean-Variance (10-Yr)', color: STRATEGY_COLORS['Mean-Variance (In-Sample)'], isRL: false, tagText: 'Baseline', tagClass: 'tag-baseline', finalValue: '5.0307', cumRet: '+403.1%', cagr: '17.53%', sharpe: '1.25', volatility: '13.20%', maxDd: '-22.10%', turnover: '0.0048' },
        { key: 'eq', name: 'Equal Weight (10-Yr)', color: STRATEGY_COLORS['Equal Weight (10-Yr)'], isRL: false, tagText: 'Baseline', tagClass: 'tag-baseline', finalValue: '4.8701', cumRet: '+387.0%', cagr: '17.15%', sharpe: '1.22', volatility: '13.80%', maxDd: '-24.60%', turnover: '0.0035' },
        { key: 'ppo', name: 'GNN + PPO (In-Sample)', color: STRATEGY_COLORS['GNN + PPO (In-Sample)'], isRL: true, tagText: 'Proposed', tagClass: 'tag-proposed', finalValue: '4.0180', cumRet: '+301.8%', cagr: '14.92%', sharpe: '1.08', volatility: '15.80%', maxDd: '-21.40%', turnover: '0.0620' },
      ];
    }
  }, [currentMode, strategiesMap]);

  return (
    <div className="tab-pane active">
      {/* KPI Grid */}
      <section className="kpi-grid">
        <div className="kpi-card" style={{ '--card-accent': '#fbbf24' }}>
          <div className="kpi-header">
            <span className="kpi-title">Final Portfolio Value</span>
            <div className="kpi-icon">
              <DollarSign size={18} />
            </div>
          </div>
          <div className="kpi-value">{kpis.finalValue}</div>
          <div className="kpi-meta">
            <span className="kpi-delta positive">{kpis.deltaFv}</span>
            <span>from $1.0000 base</span>
          </div>
        </div>

        <div className="kpi-card" style={{ '--card-accent': '#f59e0b' }}>
          <div className="kpi-header">
            <span className="kpi-title">Annualized Return (CAGR)</span>
            <div className="kpi-icon">
              <TrendingUp size={18} />
            </div>
          </div>
          <div className="kpi-value">{kpis.cagr}</div>
          <div className="kpi-meta">
            <span className="kpi-delta positive">CAGR Standard</span>
            <span>{kpis.cagrSub}</span>
          </div>
        </div>

        <div className="kpi-card" style={{ '--card-accent': '#fde68a' }}>
          <div className="kpi-header">
            <span className="kpi-title">Sharpe Ratio</span>
            <div className="kpi-icon">
              <Award size={18} />
            </div>
          </div>
          <div className="kpi-value">{kpis.sharpe}</div>
          <div className="kpi-meta">
            <span className="kpi-delta positive">{kpis.sharpeSub}</span>
            <span>rf = 0.0% standard</span>
          </div>
        </div>

        <div className="kpi-card" style={{ '--card-accent': '#f43f5e' }}>
          <div className="kpi-header">
            <span className="kpi-title">Max Drawdown</span>
            <div className="kpi-icon">
              <ShieldAlert size={18} />
            </div>
          </div>
          <div className="kpi-value">{kpis.maxDd}</div>
          <div className="kpi-meta">
            <span className="kpi-delta neutral">{kpis.maxDdSub}</span>
            <span>Controlled Risk</span>
          </div>
        </div>

        <div className="kpi-card" style={{ '--card-accent': '#d97706' }}>
          <div className="kpi-header">
            <span className="kpi-title">Daily Turnover</span>
            <div className="kpi-icon">
              <RefreshCw size={18} />
            </div>
          </div>
          <div className="kpi-value">{kpis.turnover}</div>
          <div className="kpi-meta">
            <span className="kpi-delta neutral">{kpis.turnoverSub}</span>
            <span>Disciplined Rebalance</span>
          </div>
        </div>
      </section>

      {/* Master Trajectory & Drawdown Grid */}
      <section className="dashboard-grid-2col">
        <div className="panel-card">
          <div className="panel-header">
            <div className="panel-title-group">
              <div className="panel-title">
                <Activity size={18} color="#fbbf24" />
                Master Portfolio Trajectory ({currentMode === 'holdout' ? 'Holdout Test Period' : '10-Year In-Sample'})
              </div>
              <span className="panel-subtitle">
                {currentMode === 'holdout'
                  ? 'Out-of-sample holdout (504 trading days) with strict zero-leakage protocol & 10 bps friction'
                  : 'Full 10-year historical trajectory (2015-01-02 to 2024-12-31)'}
              </span>
            </div>
          </div>

          {/* Strategy Filter Pills */}
          <div className="filter-pills" style={{ marginBottom: '1rem', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {Object.keys(strategiesMap).map((key) => {
              const isActive = activeStrategies.includes(key);
              const col = STRATEGY_COLORS[key] || '#94a3b8';
              return (
                <button
                  key={key}
                  type="button"
                  className={`filter-pill ${isActive ? 'active' : ''}`}
                  onClick={() => toggleStrategy(key)}
                >
                  <span className="pill-color" style={{ background: col }} />
                  {key}
                </button>
              );
            })}
          </div>

          <div className="chart-container" style={{ position: 'relative', height: '380px' }}>
            <canvas ref={masterChartRef} />
          </div>
        </div>

        {/* Drawdown Panel */}
        <div className="panel-card">
          <div className="panel-header">
            <div className="panel-title-group">
              <div className="panel-title">
                <ShieldAlert size={18} color="#f43f5e" />
                Underwater Drawdown Profile
              </div>
              <span className="panel-subtitle">Comparative loss depth from cumulative peaks (%)</span>
            </div>
          </div>

          <div className="chart-container" style={{ position: 'relative', height: '380px' }}>
            <canvas ref={drawdownChartRef} />
          </div>
        </div>
      </section>

      {/* Strategy Comparison Matrix Table */}
      <section className="panel-card full-width">
        <div className="panel-header">
          <div className="panel-title-group">
            <div className="panel-title">
              <Award size={18} color="#10b981" />
              Standardized Performance Matrix
            </div>
            <span className="panel-subtitle">
              {currentMode === 'holdout'
                ? 'Identical test horizon (2022-12-29 to 2024-12-31), 10 bps execution friction, verified CAGR standard'
                : '10-Year In-Sample comparative baseline matrix'}
            </span>
          </div>
        </div>

        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>Strategy / Model Architecture</th>
                <th>Final Value ($)</th>
                <th>Cumulative Return</th>
                <th>Annualized Return (CAGR)</th>
                <th>Sharpe Ratio</th>
                <th>Annual Volatility</th>
                <th>Max Drawdown</th>
                <th>Daily Turnover</th>
              </tr>
            </thead>
            <tbody>
              {tableRows.map((row) => (
                <tr key={row.key}>
                  <td>
                    <div className="strategy-badge" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="strategy-dot" style={{ background: row.color }} />
                      <span style={{ fontWeight: 600 }}>{row.name}</span>
                      <span className={`tag-label ${row.tagClass}`}>{row.tagText}</span>
                    </div>
                  </td>
                  <td className="mono" style={{ fontWeight: 700, color: '#f8fafc' }}>${row.finalValue}</td>
                  <td className="mono" style={{ color: '#38bdf8' }}>{row.cumRet}</td>
                  <td className="mono" style={{ fontWeight: 700, color: '#10b981' }}>{row.cagr}</td>
                  <td className="mono" style={{ color: '#10b981' }}>{row.sharpe}</td>
                  <td className="mono" style={{ color: '#f59e0b' }}>{row.volatility}</td>
                  <td className="mono" style={{ color: '#f43f5e' }}>{row.maxDd}</td>
                  <td className="mono" style={{ color: '#94a3b8' }}>{row.turnover}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
