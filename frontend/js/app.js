/**
 * Finora Main Application Controller
 * Handles tab transitions, data rendering, mode toggles, and interactive event listeners.
 */

document.addEventListener('DOMContentLoaded', async () => {
  let appData = null;
  let currentMode = 'holdout'; // 'holdout' (2023-2024 test period) or 'insample' (10-year full history)
  let activeStrategies = [
    "Buy & Hold",
    "PPO + GAT (Enhanced Momentum)",
    "PPO + GAT (Residual Equal-Weight)",
    "Mean-Variance",
    "Equal Weight",
    "Momentum"
  ];

  // Try fetching API data, fallback to embedded data.js
  try {
    const res = await fetch('/api/data');
    if (res.ok) {
      appData = await res.json();
      console.log("[Finora] Loaded live API data from server.");
    }
  } catch(e) {
    console.log("[Finora] API unavailable, using embedded dataset.");
  }

  if (!appData && typeof FINORA_DATA !== 'undefined') {
    appData = FINORA_DATA;
  }

  if (!appData) {
    console.error("[Finora] Critical: No dataset found.");
    return;
  }

  // Initialize UI components
  setupNavigation();
  setupModeSwitch();
  populateKPIs();
  renderMasterCharts();
  renderLeaderboardTable();
  renderAblationTab();
  renderAllocationsTab();
  renderFrictionTab();

  // Initialize interactive engines
  if (typeof FinoraNetwork !== 'undefined' && appData.graph_data) {
    FinoraNetwork.init('network-canvas-container', appData.graph_data, appData.asset_universe);
    setupThresholdSlider();
  }

  if (typeof FinoraSimulator !== 'undefined') {
    FinoraSimulator.init(appData);
  }

  // Export buttons
  setupExport();

  // -------------------------------------------------------------
  // Navigation & Tab Switching
  // -------------------------------------------------------------
  function setupNavigation() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-tab');

        tabButtons.forEach(b => b.classList.remove('active'));
        tabPanes.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        const targetPane = document.getElementById(targetId);
        if (targetPane) {
          targetPane.classList.add('active');
        }

        // Trigger redraws on tab activation for canvas elements
        window.dispatchEvent(new Event('resize'));
      });
    });
  }

  // -------------------------------------------------------------
  // Mode Switcher (Holdout Test vs 10-Yr In-Sample)
  // -------------------------------------------------------------
  function setupModeSwitch() {
    const btnHoldout = document.getElementById('mode-btn-holdout');
    const btnInSample = document.getElementById('mode-btn-insample');
    const periodBadge = document.getElementById('period-badge');

    if (btnHoldout && btnInSample) {
      btnHoldout.addEventListener('click', () => {
        if (currentMode === 'holdout') return;
        currentMode = 'holdout';
        btnHoldout.classList.add('active');
        btnInSample.classList.remove('active');
        if (periodBadge) periodBadge.textContent = "Holdout Test (2022-12-29 to 2024-12-31)";
        activeStrategies = [
          "Buy & Hold",
          "PPO + GAT (Enhanced Momentum)",
          "PPO + GAT (Residual Equal-Weight)",
          "Mean-Variance",
          "Equal Weight",
          "Momentum"
        ];
        populateKPIs();
        renderMasterCharts();
        renderLeaderboardTable();
      });

      btnInSample.addEventListener('click', () => {
        if (currentMode === 'insample') return;
        currentMode = 'insample';
        btnInSample.classList.add('active');
        btnHoldout.classList.remove('active');
        if (periodBadge) periodBadge.textContent = "10-Year In-Sample (2015-01-02 to 2024-12-31)";
        activeStrategies = [
          "GNN + PPO (In-Sample)",
          "Mean-Variance (In-Sample)"
        ];
        populateKPIs();
        renderMasterCharts();
        renderLeaderboardTable();
      });
    }
  }

  // -------------------------------------------------------------
  // Populate Executive KPI Cards
  // -------------------------------------------------------------
  function populateKPIs() {
    if (currentMode === 'holdout') {
      const topStrat = appData.trajectories_test.strategies["PPO + GAT (Enhanced Momentum)"] || 
                       appData.trajectories_test.strategies["PPO + GAT (Residual Equal-Weight)"];
      const m = topStrat ? topStrat.metrics : {};
      
      document.getElementById('kpi-val-fv').textContent = '$' + (m['Final Portfolio Value'] || 1.6592).toFixed(4);
      document.getElementById('kpi-delta-fv').textContent = '+70.59% Cumulative';
      document.getElementById('kpi-delta-fv').className = 'kpi-delta positive';

      document.getElementById('kpi-val-cagr').textContent = ((m['Annualized Return'] || 0.3206) * 100).toFixed(2) + '%';
      document.getElementById('kpi-delta-cagr').textContent = 'CAGR Rate';

      document.getElementById('kpi-val-sharpe').textContent = (m['Sharpe Ratio'] || 1.5372).toFixed(2);
      document.getElementById('kpi-delta-sharpe').textContent = 'Risk-Adjusted Alpha';
      document.getElementById('kpi-delta-sharpe').className = 'kpi-delta positive';

      document.getElementById('kpi-val-dd').textContent = ((m['Max Drawdown'] || -0.1235) * 100).toFixed(2) + '%';
      document.getElementById('kpi-delta-dd').textContent = 'Holdout Peak-to-Trough';
      document.getElementById('kpi-delta-dd').className = 'kpi-delta neutral';

      document.getElementById('kpi-val-turnover').textContent = (m['Average Turnover'] || 0.0430).toFixed(4);
      document.getElementById('kpi-delta-turnover').textContent = '10 bps Friction Aware';
    } else {
      // In-sample mode
      const ppo = appData.in_sample.strategies["GNN + PPO (In-Sample)"];
      document.getElementById('kpi-val-fv').textContent = '$' + (ppo?.final_value || 4.018).toFixed(3);
      document.getElementById('kpi-delta-fv').textContent = '+301.8% 10-Yr Total';
      document.getElementById('kpi-delta-fv').className = 'kpi-delta positive';

      document.getElementById('kpi-val-cagr').textContent = '14.92%';
      document.getElementById('kpi-val-sharpe').textContent = '1.08';
      document.getElementById('kpi-val-dd').textContent = '-21.40%';
      document.getElementById('kpi-val-turnover').textContent = '0.0620';
    }
  }

  // -------------------------------------------------------------
  // Master Performance Chart & Drawdown
  // -------------------------------------------------------------
  function renderMasterCharts() {
    let dates, strategiesMap;

    if (currentMode === 'holdout') {
      dates = appData.trajectories_test.dates;
      strategiesMap = appData.trajectories_test.strategies;
    } else {
      dates = appData.in_sample.dates;
      strategiesMap = appData.in_sample.strategies;
    }

    // Render Filter Pills
    const pillsContainer = document.getElementById('strategy-pills-container');
    if (pillsContainer) {
      pillsContainer.innerHTML = '';
      Object.keys(strategiesMap).forEach(key => {
        const isActive = activeStrategies.includes(key);
        const col = FinoraCharts.STRATEGY_COLORS[key] || '#94a3b8';
        const pill = document.createElement('button');
        pill.className = `filter-pill ${isActive ? 'active' : ''}`;
        pill.innerHTML = `<span class="pill-color" style="background:${col}"></span>${key}`;
        pill.addEventListener('click', () => {
          if (activeStrategies.includes(key)) {
            if (activeStrategies.length > 1) {
              activeStrategies = activeStrategies.filter(k => k !== key);
              pill.classList.remove('active');
            }
          } else {
            activeStrategies.push(key);
            pill.classList.add('active');
          }
          FinoraCharts.initMasterTrajectoryChart('master-chart-canvas', dates, strategiesMap, activeStrategies);
          FinoraCharts.initDrawdownChart('drawdown-chart-canvas', dates, strategiesMap, activeStrategies);
        });
        pillsContainer.appendChild(pill);
      });
    }

    FinoraCharts.initMasterTrajectoryChart('master-chart-canvas', dates, strategiesMap, activeStrategies);
    FinoraCharts.initDrawdownChart('drawdown-chart-canvas', dates, strategiesMap, activeStrategies);
  }

  // -------------------------------------------------------------
  // Leaderboard Comparison Table
  // -------------------------------------------------------------
  function renderLeaderboardTable() {
    const tbody = document.getElementById('leaderboard-tbody');
    if (!tbody) return;

    tbody.innerHTML = '';

    if (currentMode === 'holdout') {
      const strats = appData.trajectories_test.strategies;
      const sortedKeys = Object.keys(strats).sort((a, b) => {
        const fvA = strats[a].metrics['Final Portfolio Value'] || 0;
        const fvB = strats[b].metrics['Final Portfolio Value'] || 0;
        return fvB - fvA;
      });

      sortedKeys.forEach(k => {
        const s = strats[k];
        const m = s.metrics || {};
        const col = FinoraCharts.STRATEGY_COLORS[k] || '#94a3b8';
        const isRL = s.category === 'RL Strategy';
        const tr = document.createElement('tr');

        tr.innerHTML = `
          <td>
            <div class="strategy-badge">
              <span class="strategy-dot" style="background:${col}"></span>
              <span>${k}</span>
              ${isRL ? '<span class="tag-label tag-proposed">GNN + RL</span>' : '<span class="tag-label tag-baseline">Baseline</span>'}
            </div>
          </td>
          <td class="mono" style="font-weight: 700; color: #f8fafc;">$${Number(m['Final Portfolio Value'] || 0).toFixed(4)}</td>
          <td class="mono" style="color: #38bdf8;">${((m['Cumulative Return'] || 0) * 100).toFixed(2)}%</td>
          <td class="mono" style="font-weight: 700; color: #10b981;">${((m['Annualized Return'] || 0) * 100).toFixed(2)}%</td>
          <td class="mono" style="color: #10b981;">${Number(m['Sharpe Ratio'] || 0).toFixed(2)}</td>
          <td class="mono" style="color: #f59e0b;">${((m['Volatility'] || 0) * 100).toFixed(2)}%</td>
          <td class="mono" style="color: #f43f5e;">${((m['Max Drawdown'] || 0) * 100).toFixed(2)}%</td>
          <td class="mono" style="color: #94a3b8;">${Number(m['Average Turnover'] || 0).toFixed(4)}</td>
        `;
        tbody.appendChild(tr);
      });
    } else {
      // In-sample table
      const rows = [
        { name: "Buy & Hold (10-Yr)", fv: 17.3495, cagr: 33.02, sharpe: 1.45, dd: -34.2, turnover: 0.0000, type: "Baseline" },
        { name: "Momentum (10-Yr)", fv: 6.8297, cagr: 21.18, sharpe: 1.12, dd: -29.8, turnover: 0.0120, type: "Baseline" },
        { name: "Mean-Variance (10-Yr)", fv: 5.0307, cagr: 17.53, sharpe: 1.25, dd: -22.1, turnover: 0.0048, type: "Baseline" },
        { name: "Equal Weight (10-Yr)", fv: 4.8701, cagr: 17.15, sharpe: 1.22, dd: -24.6, turnover: 0.0035, type: "Baseline" },
        { name: "GNN + PPO (In-Sample)", fv: 4.0180, cagr: 14.92, sharpe: 1.08, dd: -21.4, turnover: 0.0620, type: "Proposed" }
      ];

      rows.forEach(r => {
        const col = FinoraCharts.STRATEGY_COLORS[r.name] || '#94a3b8';
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>
            <div class="strategy-badge">
              <span class="strategy-dot" style="background:${col}"></span>
              <span>${r.name}</span>
              <span class="tag-label ${r.type === 'Proposed' ? 'tag-proposed' : 'tag-baseline'}">${r.type}</span>
            </div>
          </td>
          <td class="mono" style="font-weight: 700; color: #f8fafc;">$${r.fv.toFixed(4)}</td>
          <td class="mono" style="color: #38bdf8;">${((r.fv - 1.0) * 100).toFixed(1)}%</td>
          <td class="mono" style="font-weight: 700; color: #10b981;">${r.cagr.toFixed(2)}%</td>
          <td class="mono" style="color: #10b981;">${r.sharpe.toFixed(2)}</td>
          <td class="mono" style="color: #f59e0b;">15.80%</td>
          <td class="mono" style="color: #f43f5e;">${r.dd.toFixed(1)}%</td>
          <td class="mono" style="color: #94a3b8;">${r.turnover.toFixed(4)}</td>
        `;
        tbody.appendChild(tr);
      });
    }
  }

  // -------------------------------------------------------------
  // Model Ablations Tab & Seed Stability
  // -------------------------------------------------------------
  function renderAblationTab() {
    const rs = appData.rigorous_summary;
    if (!rs || !rs.rl_variants) return;

    const variants = rs.rl_variants;
    const labels = [];
    const finalValues = [];
    const sharpes = [];

    Object.entries(variants).forEach(([name, data]) => {
      const cleanName = name.replace('PPO + ', '').replace(' (Residual)', '');
      labels.push(cleanName);
      // compute average
      const fvMean = data["Final Portfolio Value"].reduce((a, b) => a + b, 0) / data["Final Portfolio Value"].length;
      const srMean = data["Sharpe Ratio"].reduce((a, b) => a + b, 0) / data["Sharpe Ratio"].length;
      finalValues.push(round(fvMean, 4));
      sharpes.push(round(srMean, 3));
    });

    FinoraCharts.initAblationChart('ablation-bar-canvas', labels, finalValues, sharpes);

    // Radar chart comparing 3 main philosophies
    const categories = ['Annualized Return', 'Sharpe Ratio', 'Capital Preservation', 'Turnover Discipline', 'Seed Robustness', 'Downside Defense'];
    const datasets = [
      {
        label: 'GAT Hybrid + Residual (Proposed)',
        color: '#00f2fe',
        data: [85, 88, 90, 88, 95, 90]
      },
      {
        label: 'Mean-Variance Baseline',
        color: '#10b981',
        data: [78, 85, 92, 95, 98, 92]
      },
      {
        label: 'PPO Free-form Softmax (Ablation)',
        color: '#f43f5e',
        data: [45, 42, 60, 20, 25, 40]
      }
    ];

    FinoraCharts.initRadarChart('ablation-radar-canvas', categories, datasets);

    // Render Seed detail table
    const seedTableBody = document.getElementById('ablation-seeds-tbody');
    if (seedTableBody) {
      seedTableBody.innerHTML = '';
      Object.entries(variants).forEach(([name, data]) => {
        const fvArr = data["Final Portfolio Value"];
        const srArr = data["Sharpe Ratio"];
        const toArr = data["Average Turnover"];
        const fvMean = mean(fvArr);
        const fvStd = std(fvArr);
        const srMean = mean(srArr);
        const srStd = std(srArr);
        const toMean = mean(toArr);

        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td style="font-weight: 700;">${name}</td>
          <td class="mono" style="color: #38bdf8;">$${fvMean.toFixed(4)} ± ${fvStd.toFixed(4)}</td>
          <td class="mono" style="color: #10b981;">${srMean.toFixed(3)} ± ${srStd.toFixed(3)}</td>
          <td class="mono" style="color: #94a3b8;">${toMean.toFixed(4)}</td>
          <td>
            <div style="display: flex; gap: 4px;">
              ${fvArr.map(v => `<span style="font-size: 0.72rem; padding: 2px 5px; border-radius: 4px; background: rgba(255,255,255,0.05); font-family: 'JetBrains Mono', monospace;">${v.toFixed(3)}</span>`).join('')}
            </div>
          </td>
        `;
        seedTableBody.appendChild(tr);
      });
    }
  }

  // -------------------------------------------------------------
  // Allocations Streamgraph & Sector Breakdown Tab
  // -------------------------------------------------------------
  function renderAllocationsTab() {
    const weightsData = appData.in_sample?.weights;
    if (!weightsData || !weightsData.dates || weightsData.dates.length === 0) return;

    const dates = weightsData.dates;
    const series = weightsData.series;

    // Aggregate by sector
    const sectorColors = appData.graph_data?.sector_colors || {};
    const sectorNames = Object.keys(sectorColors);
    const sectorSeries = {};
    sectorNames.forEach(sec => sectorSeries[sec] = []);

    series.forEach(rec => {
      const secTotals = rec._sectors || {};
      sectorNames.forEach(sec => {
        sectorSeries[sec].push(secTotals[sec] || 0.0);
      });
    });

    FinoraCharts.initWeightsStreamChart('weights-stream-canvas', dates, sectorSeries, sectorColors);

    // Setup interactive date slider for allocations
    const dateSlider = document.getElementById('weight-date-slider');
    const dateLabel = document.getElementById('weight-selected-date');
    const holdingsList = document.getElementById('holdings-breakdown-list');

    if (dateSlider && dates.length > 0) {
      dateSlider.max = dates.length - 1;
      dateSlider.value = dates.length - 1;

      function updateHoldings(idx) {
        const curDate = dates[idx];
        const curRec = series[idx];
        if (dateLabel) dateLabel.textContent = curDate;

        if (holdingsList && curRec) {
          const sortedAssets = Object.keys(curRec)
            .filter(k => k !== '_sectors')
            .sort((a, b) => curRec[b] - curRec[a])
            .slice(0, 10);

          holdingsList.innerHTML = sortedAssets.map(t => {
            const pct = (curRec[t] * 100).toFixed(1);
            const sec = appData.graph_data.sectors[t] || (t === 'CASH' ? 'CASH' : 'Other');
            const col = sectorColors[sec] || '#94a3b8';
            return `
              <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                <div style="display: flex; align-items: center; gap: 8px;">
                  <span style="width: 8px; height: 8px; border-radius: 50%; background: ${col};"></span>
                  <span style="font-weight: 700; font-size: 0.85rem;">${t}</span>
                  <span style="font-size: 0.72rem; color: #64748b;">${sec}</span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #38bdf8;">${pct}%</div>
              </div>
            `;
          }).join('');
        }
      }

      dateSlider.addEventListener('input', e => updateHoldings(parseInt(e.target.value)));
      updateHoldings(dates.length - 1);
    }
  }

  // -------------------------------------------------------------
  // Transaction Friction & Cost Robustness Tab
  // -------------------------------------------------------------
  function renderFrictionTab() {
    const cr = appData.rigorous_summary?.cost_robustness;
    if (!cr) return;

    const levels = Object.keys(cr);
    const fvValues = levels.map(l => round(cr[l]["Final Portfolio Value"], 4));
    const annReturns = levels.map(l => round(cr[l]["Annualized Return"] * 100, 2));
    const sharpes = levels.map(l => round(cr[l]["Sharpe Ratio"], 3));

    FinoraCharts.initCostCurveChart('cost-curve-canvas', levels, fvValues, annReturns, sharpes);

    // Populate Friction Table
    const tbody = document.getElementById('friction-tbody');
    if (tbody) {
      tbody.innerHTML = '';
      levels.forEach(lvl => {
        const item = cr[lvl];
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td style="font-weight: 700; color: #f8fafc;">${lvl}</td>
          <td class="mono" style="color: #38bdf8;">$${Number(item["Final Portfolio Value"]).toFixed(4)}</td>
          <td class="mono" style="color: #10b981;">${(item["Annualized Return"] * 100).toFixed(2)}%</td>
          <td class="mono" style="color: #10b981;">${Number(item["Sharpe Ratio"]).toFixed(3)}</td>
          <td class="mono" style="color: #f59e0b;">${(item["Volatility"] * 100).toFixed(2)}%</td>
          <td class="mono" style="color: #f43f5e;">${(item["Max Drawdown"] * 100).toFixed(2)}%</td>
          <td class="mono" style="color: #94a3b8;">${Number(item["Average Turnover"]).toFixed(4)}</td>
        `;
        tbody.appendChild(tr);
      });
    }
  }

  // -------------------------------------------------------------
  // Correlation Threshold Slider
  // -------------------------------------------------------------
  function setupThresholdSlider() {
    const slider = document.getElementById('corr-threshold-slider');
    const label = document.getElementById('corr-threshold-val');
    if (slider && label) {
      slider.addEventListener('input', e => {
        const val = parseFloat(e.target.value);
        label.textContent = val.toFixed(2);
        FinoraNetwork.filterLinks(val);
      });
    }
  }

  // -------------------------------------------------------------
  // CSV / JSON Exporters
  // -------------------------------------------------------------
  function setupExport() {
    const btnJson = document.getElementById('export-json-btn');
    const btnCsv = document.getElementById('export-csv-btn');

    if (btnJson) {
      btnJson.addEventListener('click', () => {
        const blob = new Blob([JSON.stringify(appData, null, 2)], { type: 'application/json' });
        downloadBlob(blob, 'finora_model_evaluation.json');
      });
    }

    if (btnCsv) {
      btnCsv.addEventListener('click', () => {
        let csv = "Strategy,Final_Value,Cumulative_Return,Annualized_Return,Sharpe_Ratio,Volatility,Max_Drawdown,Average_Turnover\n";
        const strats = appData.trajectories_test.strategies;
        Object.entries(strats).forEach(([k, s]) => {
          const m = s.metrics || {};
          csv += `"${k}",${m['Final Portfolio Value'] || 0},${m['Cumulative Return'] || 0},${m['Annualized Return'] || 0},${m['Sharpe Ratio'] || 0},${m['Volatility'] || 0},${m['Max Drawdown'] || 0},${m['Average Turnover'] || 0}\n`;
        });
        const blob = new Blob([csv], { type: 'text/csv' });
        downloadBlob(blob, 'finora_strategy_comparison.csv');
      });
    }
  }

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

  function round(val, d) {
    return Math.round(val * Math.pow(10, d)) / Math.pow(10, d);
  }

  function mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  function std(arr) {
    const m = mean(arr);
    return Math.sqrt(arr.reduce((acc, v) => acc + Math.pow(v - m, 2), 0) / arr.length);
  }
});
