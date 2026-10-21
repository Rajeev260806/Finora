import { Chart, registerables } from 'chart.js';

// Register Chart.js components
Chart.register(...registerables);

// Configure Global Chart.js Defaults for Finora Dark Mode (Champagne Gold & Amber)
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(19, 23, 32, 0.95)';
Chart.defaults.plugins.tooltip.titleColor = '#f8fafc';
Chart.defaults.plugins.tooltip.titleFont = { family: "'Outfit', sans-serif", size: 13, weight: 'bold' };
Chart.defaults.plugins.tooltip.bodyColor = '#cbd5e1';
Chart.defaults.plugins.tooltip.bodyFont = { family: "'JetBrains Mono', monospace", size: 12 };
Chart.defaults.plugins.tooltip.borderColor = 'rgba(245, 158, 11, 0.4)';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.tooltip.boxPadding = 6;
Chart.defaults.plugins.tooltip.usePointStyle = true;

export const STRATEGY_COLORS = {
  "Buy & Hold": "#e2e8f0",                         // Crisp Platinum/Silver
  "PPO + GAT (Enhanced Momentum)": "#fbbf24",      // Champagne Gold (Primary Proposed Model)
  "PPO + GAT (Residual Equal-Weight)": "#f59e0b",  // Warm Amber
  "Mean-Variance": "#fde68a",                      // Light Champagne
  "Equal Weight": "#d97706",                       // Deep Warm Amber
  "Momentum": "#b45309",                           // Bronze Amber
  "PPO + GAT (Free-form Softmax)": "#f43f5e",      // Rose (High turnover ablation)
  "PPO + GCN (Residual)": "#eab308",               // Radiant Gold
  "PPO + Random GNN (Residual)": "#78716c",        // Warm Slate
  "PPO + No Embeddings (Residual)": "#fcd34d",     // Amber Gold
  "GNN + PPO (In-Sample)": "#fbbf24",
  "Mean-Variance (In-Sample)": "#fde68a",
  "Buy & Hold (10-Yr)": "#e2e8f0",
  "Momentum (10-Yr)": "#b45309",
  "Equal Weight (10-Yr)": "#d97706"
};

export function createVerticalGradient(ctx, color) {
  if (!ctx || !ctx.canvas) return 'transparent';
  const chartArea = ctx.chartArea || { top: 0, bottom: 300 };
  const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
  gradient.addColorStop(0, 'rgba(251, 191, 36, 0.35)');
  gradient.addColorStop(1, 'rgba(7, 8, 10, 0.0)');
  return gradient;
}

export { Chart };
