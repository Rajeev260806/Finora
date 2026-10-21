import React, { useEffect, useRef, useState, useMemo } from 'react';
import { Chart } from '../../utils/chartConfig';

export default function AllocationsTab({ appData }) {
  const chartRef = useRef(null);
  const chartInst = useRef(null);

  const weightsData = useMemo(() => appData?.in_sample?.weights || { dates: [], series: [] }, [appData]);
  const dates = weightsData.dates || [];
  const series = weightsData.series || [];

  const sectorColors = useMemo(() => appData?.graph_data?.sector_colors || {
    Technology: '#38bdf8',
    Financials: '#10b981',
    Energy: '#f59e0b',
    Utilities: '#8b5cf6',
    'Real Estate': '#ec4899',
    CASH: '#64748b'
  }, [appData]);

  const [scrubberIdx, setScrubberIdx] = useState(dates.length ? dates.length - 1 : 0);

  useEffect(() => {
    if (dates.length > 0) {
      setScrubberIdx(dates.length - 1);
    }
  }, [dates]);

  // Sector Series for Stacked Area Chart
  useEffect(() => {
    if (!chartRef.current || dates.length === 0 || series.length === 0) return;

    if (chartInst.current) {
      chartInst.current.destroy();
    }

    const sectorNames = Object.keys(sectorColors);
    const sectorDataMap = {};
    sectorNames.forEach((sec) => (sectorDataMap[sec] = []));

    series.forEach((rec) => {
      const secTotals = rec._sectors || {};
      sectorNames.forEach((sec) => {
        sectorDataMap[sec].push((secTotals[sec] || 0.0) * 100);
      });
    });

    const datasets = sectorNames.map((sec) => ({
      label: sec,
      data: sectorDataMap[sec],
      borderColor: sectorColors[sec],
      backgroundColor: sectorColors[sec].replace(')', ', 0.55)').replace('rgb', 'rgba').replace('#38bdf8', 'rgba(56, 189, 248, 0.55)').replace('#10b981', 'rgba(16, 185, 129, 0.55)').replace('#f59e0b', 'rgba(245, 158, 11, 0.55)').replace('#8b5cf6', 'rgba(139, 92, 246, 0.55)').replace('#ec4899', 'rgba(236, 72, 153, 0.55)').replace('#64748b', 'rgba(100, 116, 139, 0.55)'),
      fill: true,
      borderWidth: 1.2,
      pointRadius: 0,
      tension: 0.2,
    }));

    chartInst.current = new Chart(chartRef.current, {
      type: 'line',
      data: { labels: dates, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            position: 'top',
            labels: { font: { family: "'Outfit', sans-serif", size: 12 }, boxWidth: 14 }
          },
          tooltip: {
            callbacks: {
              label: (context) => (context.dataset.label || '') + ': ' + context.parsed.y.toFixed(1) + '%',
            },
          },
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.04)' },
            ticks: { maxTicksLimit: 12, font: { size: 11 } },
          },
          y: {
            stacked: true,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: {
              callback: (v) => Number(v).toFixed(0) + '%',
              font: { family: "'JetBrains Mono', monospace", size: 11 },
            },
          },
        },
      },
    });

    return () => {
      if (chartInst.current) {
        chartInst.current.destroy();
        chartInst.current = null;
      }
    };
  }, [dates, series, sectorColors]);

  // Holdings on currently scrubbed date
  const selectedDate = dates[scrubberIdx] || 'Latest';
  const selectedRec = series[scrubberIdx] || {};

  const topHoldings = useMemo(() => {
    if (!selectedRec) return [];
    return Object.keys(selectedRec)
      .filter((k) => k !== '_sectors')
      .sort((a, b) => selectedRec[b] - selectedRec[a])
      .slice(0, 10)
      .map((t) => {
        const sec = appData?.graph_data?.sectors?.[t] || (t === 'CASH' ? 'CASH' : 'Other');
        const col = sectorColors[sec] || '#94a3b8';
        return {
          ticker: t,
          sector: sec,
          color: col,
          weight: (selectedRec[t] * 100).toFixed(1),
        };
      });
  }, [selectedRec, appData, sectorColors]);

  return (
    <div className="tab-pane active">
      <div className="panel-card full-width">
        <div className="panel-header">
          <div className="panel-title-group">
            <div className="panel-title">Dynamic Sector Allocation Streamgraph Over Time</div>
            <span className="panel-subtitle">
              Proportional exposure across Technology, Financials, Energy, Utilities, Real Estate, and Cash
            </span>
          </div>
        </div>

        <div className="chart-container-lg" style={{ position: 'relative', height: '420px' }}>
          <canvas ref={chartRef} />
        </div>
      </div>

      <div className="dashboard-grid-2col" style={{ marginTop: '1.5rem' }}>
        <div className="panel-card">
          <div className="panel-header">
            <div className="panel-title-group">
              <div className="panel-title">Interactive Time Scrubber</div>
              <span className="panel-subtitle">
                Selected Date: <strong style={{ color: '#38bdf8', fontFamily: "'JetBrains Mono', monospace" }}>{selectedDate}</strong>
              </span>
            </div>
          </div>
          <div style={{ padding: '1rem 0' }}>
            <input
              type="range"
              min={0}
              max={dates.length > 0 ? dates.length - 1 : 0}
              value={scrubberIdx}
              onChange={(e) => setScrubberIdx(parseInt(e.target.value, 10))}
              className="sim-slider"
              style={{ width: '100%' }}
            />
          </div>
          <p style={{ fontSize: '0.82rem', color: '#94a3b8', lineHeight: 1.6 }}>
            Finora's policy dynamically rotates between high-beta momentum assets in market rallies and allocates up to 20% to cash or defensive utilities (DUK, SO, NEE) when market volatility spikes.
          </p>
        </div>

        <div className="panel-card">
          <div className="panel-header">
            <div className="panel-title-group">
              <div className="panel-title">Top Holdings on Selected Date</div>
              <span className="panel-subtitle">Asset weights breakdown (%)</span>
            </div>
          </div>
          <div>
            {topHoldings.map((h) => (
              <div
                key={h.ticker}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '6px 0',
                  borderBottom: '1px solid rgba(255,255,255,0.04)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: h.color }} />
                  <span style={{ fontWeight: 700, fontSize: '0.85rem' }}>{h.ticker}</span>
                  <span style={{ fontSize: '0.72rem', color: '#64748b' }}>{h.sector}</span>
                </div>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: '#38bdf8' }}>
                  {h.weight}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
