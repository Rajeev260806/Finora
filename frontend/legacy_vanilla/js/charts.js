/**
 * Finora Dashboard Charting Engine
 * Uses Chart.js with custom dark-mode styling and gradients.
 */

const FinoraCharts = (function() {
  let masterTrajectoryChart = null;
  let drawdownChart = null;
  let radarChart = null;
  let ablationChart = null;
  let costCurveChart = null;
  let weightsStreamChart = null;
  let simChart = null;

  // Modern Chart.js Global Theme Overrides
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
  Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 23, 42, 0.95)';
  Chart.defaults.plugins.tooltip.titleColor = '#f8fafc';
  Chart.defaults.plugins.tooltip.titleFont = { family: "'Outfit', sans-serif", size: 13, weight: 'bold' };
  Chart.defaults.plugins.tooltip.bodyColor = '#cbd5e1';
  Chart.defaults.plugins.tooltip.bodyFont = { family: "'JetBrains Mono', monospace", size: 12 };
  Chart.defaults.plugins.tooltip.borderColor = 'rgba(56, 189, 248, 0.3)';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 12;
  Chart.defaults.plugins.tooltip.boxPadding = 6;
  Chart.defaults.plugins.tooltip.usePointStyle = true;

  const STRATEGY_COLORS = {
    "Buy & Hold": "#e2e8f0",                         // Crisp White/Silver
    "PPO + GAT (Enhanced Momentum)": "#00f2fe",      // Neon Cyan (Primary proposed)
    "PPO + GAT (Residual Equal-Weight)": "#38bdf8",  // Sky Blue
    "Mean-Variance": "#10b981",                      // Emerald
    "Equal Weight": "#818cf8",                       // Indigo
    "Momentum": "#f59e0b",                           // Amber
    "PPO + GAT (Free-form Softmax)": "#f43f5e",      // Rose (High turnover ablation)
    "PPO + GCN (Residual)": "#c084fc",               // Light Purple
    "PPO + Random GNN (Residual)": "#a8a29e",        // Gray
    "PPO + No Embeddings (Residual)": "#34d399",     // Mint
    "GNN + PPO (In-Sample)": "#06b6d4",
    "Mean-Variance (In-Sample)": "#10b981"
  };

  /**
   * Initialize Master Equity Trajectory Chart
   */
  function initMasterTrajectoryChart(canvasId, dates, strategiesMap, activeKeys) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (masterTrajectoryChart) {
      masterTrajectoryChart.destroy();
    }

    const datasets = [];

    activeKeys.forEach(key => {
      const strat = strategiesMap[key];
      if (!strat) return;

      const color = STRATEGY_COLORS[key] || '#94a3b8';
      const isProposed = key.includes("Enhanced") || key.includes("Proposed") || key === "PPO + GAT (Residual Equal-Weight)";

      datasets.push({
        label: strat.name || key,
        data: strat.portfolio_value,
        borderColor: color,
        backgroundColor: isProposed ? createVerticalGradient(ctx, color) : 'transparent',
        borderWidth: isProposed ? 2.8 : 1.8,
        borderDash: key.includes("Free-form") ? [4, 4] : [],
        pointRadius: 0,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: color,
        pointHoverBorderColor: '#fff',
        pointHoverBorderWidth: 2,
        tension: 0.15,
        fill: isProposed
      });
    });

    masterTrajectoryChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: dates,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                let label = context.dataset.label || '';
                if (label) label += ': ';
                if (context.parsed.y !== null) {
                  label += '$' + context.parsed.y.toFixed(4) + ' (' + ((context.parsed.y - 1.0) * 100).toFixed(1) + '%)';
                }
                return label;
              }
            }
          }
        },
        scales: {
          x: {
            grid: {
              color: 'rgba(255, 255, 255, 0.04)',
              drawBorder: false
            },
            ticks: {
              maxTicksLimit: 12,
              font: { size: 11 }
            }
          },
          y: {
            grid: {
              color: 'rgba(255, 255, 255, 0.05)',
              drawBorder: false
            },
            ticks: {
              callback: function(value) {
                return '$' + value.toFixed(2);
              },
              font: { family: "'JetBrains Mono', monospace", size: 11 }
            }
          }
        }
      }
    });
  }

  /**
   * Initialize Underwater Drawdown Chart
   */
  function initDrawdownChart(canvasId, dates, strategiesMap, activeKeys) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (drawdownChart) {
      drawdownChart.destroy();
    }

    const datasets = [];

    activeKeys.slice(0, 4).forEach(key => {
      const strat = strategiesMap[key];
      if (!strat || !strat.drawdown) return;

      const color = STRATEGY_COLORS[key] || '#94a3b8';

      datasets.push({
        label: strat.name || key,
        data: strat.drawdown.map(v => v * 100),
        borderColor: color,
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.1,
        fill: false
      });
    });

    drawdownChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: dates,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: {
              boxWidth: 12,
              font: { size: 11 }
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '%';
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.03)' },
            ticks: { maxTicksLimit: 8 }
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: {
              callback: function(value) {
                return value.toFixed(1) + '%';
              },
              font: { family: "'JetBrains Mono', monospace" }
            }
          }
        }
      }
    });
  }

  /**
   * Strategy Radar Comparison Chart
   */
  function initRadarChart(canvasId, categories, datasets) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (radarChart) {
      radarChart.destroy();
    }

    radarChart = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: categories,
        datasets: datasets.map(d => ({
          label: d.label,
          data: d.data,
          borderColor: d.color,
          backgroundColor: hexToRgba(d.color, 0.2),
          borderWidth: 2,
          pointBackgroundColor: d.color,
          pointBorderColor: '#fff',
          pointRadius: 3
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { boxWidth: 12, font: { size: 11 } }
          }
        },
        scales: {
          r: {
            angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
            grid: { color: 'rgba(255, 255, 255, 0.06)' },
            pointLabels: {
              font: { size: 11, weight: '600' },
              color: '#94a3b8'
            },
            ticks: { display: false }
          }
        }
      }
    });
  }

  /**
   * Model Ablation & Seed Stability Bar Chart
   */
  function initAblationChart(canvasId, labels, finalValues, sharpes) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (ablationChart) {
      ablationChart.destroy();
    }

    ablationChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Final Portfolio Value ($)',
            data: finalValues,
            backgroundColor: '#0ea5e9',
            borderRadius: 6,
            yAxisID: 'y'
          },
          {
            label: 'Sharpe Ratio',
            data: sharpes,
            backgroundColor: '#10b981',
            borderRadius: 6,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { boxWidth: 12 }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { size: 11 } }
          },
          y: {
            type: 'linear',
            position: 'left',
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: {
              font: { family: "'JetBrains Mono', monospace" }
            }
          },
          y1: {
            type: 'linear',
            position: 'right',
            grid: { display: false },
            ticks: {
              font: { family: "'JetBrains Mono', monospace" }
            }
          }
        }
      }
    });
  }

  /**
   * Transaction Cost Sensitivity Chart
   */
  function initCostCurveChart(canvasId, levels, fvValues, annReturns, sharpes) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (costCurveChart) {
      costCurveChart.destroy();
    }

    costCurveChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: levels,
        datasets: [
          {
            label: 'Final Portfolio Value ($)',
            data: fvValues,
            borderColor: '#38bdf8',
            backgroundColor: 'rgba(56, 189, 248, 0.1)',
            fill: true,
            tension: 0.2,
            pointRadius: 6,
            pointHoverRadius: 8,
            yAxisID: 'y'
          },
          {
            label: 'Sharpe Ratio',
            data: sharpes,
            borderColor: '#10b981',
            tension: 0.2,
            pointRadius: 6,
            pointHoverRadius: 8,
            borderDash: [5, 5],
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top' }
        },
        scales: {
          y: {
            type: 'linear',
            position: 'left',
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          },
          y1: {
            type: 'linear',
            position: 'right',
            grid: { display: false }
          }
        }
      }
    });
  }

  /**
   * Dynamic Asset/Sector Allocations Streamgraph
   */
  function initWeightsStreamChart(canvasId, dates, sectorSeries, sectorColors) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (weightsStreamChart) {
      weightsStreamChart.destroy();
    }

    const sectors = Object.keys(sectorSeries);
    const datasets = sectors.map(sec => {
      const col = sectorColors[sec] || '#94a3b8';
      return {
        label: sec,
        data: sectorSeries[sec].map(v => v * 100),
        borderColor: col,
        backgroundColor: hexToRgba(col, 0.65),
        borderWidth: 1,
        pointRadius: 0,
        fill: true
      };
    });

    weightsStreamChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: dates,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            position: 'top',
            labels: { boxWidth: 12 }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.03)' },
            ticks: { maxTicksLimit: 10 }
          },
          y: {
            stacked: true,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: {
              callback: v => v + '%'
            }
          }
        }
      }
    });
  }

  /**
   * Interactive Simulator Result Chart
   */
  function updateSimulationChart(canvasId, dates, simPV, benchmarkPV, initialCapital) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (simChart) {
      simChart.destroy();
    }

    simChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: dates,
        datasets: [
          {
            label: 'Simulated Strategy ($)',
            data: simPV,
            borderColor: '#0ea5e9',
            backgroundColor: createVerticalGradient(ctx, '#0ea5e9'),
            borderWidth: 2.5,
            fill: true,
            pointRadius: 0,
            tension: 0.15
          },
          {
            label: 'Buy & Hold Baseline ($)',
            data: benchmarkPV.map(v => v * initialCapital),
            borderColor: '#94a3b8',
            borderWidth: 1.5,
            borderDash: [4, 4],
            fill: false,
            pointRadius: 0,
            tension: 0.15
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 12 } },
          tooltip: {
            callbacks: {
              label: context => context.dataset.label + ': $' + context.parsed.y.toLocaleString()
            }
          }
        },
        scales: {
          x: { grid: { color: 'rgba(255, 255, 255, 0.03)' }, ticks: { maxTicksLimit: 8 } },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: {
              callback: v => '$' + v.toLocaleString(),
              font: { family: "'JetBrains Mono', monospace" }
            }
          }
        }
      }
    });
  }

  // Helpers
  function hexToRgba(hex, alpha) {
    hex = hex.replace('#', '');
    if (hex.length === 3) {
      hex = hex.split('').map(c => c + c).join('');
    }
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function createVerticalGradient(ctx, color) {
    try {
      const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 360);
      gradient.addColorStop(0, hexToRgba(color, 0.25));
      gradient.addColorStop(1, hexToRgba(color, 0.0));
      return gradient;
    } catch(e) {
      return hexToRgba(color, 0.1);
    }
  }

  return {
    initMasterTrajectoryChart,
    initDrawdownChart,
    initRadarChart,
    initAblationChart,
    initCostCurveChart,
    initWeightsStreamChart,
    updateSimulationChart,
    STRATEGY_COLORS
  };
})();
