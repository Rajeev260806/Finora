import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';

// Centroids for each sector cluster spread out cleanly across the canvas
const SECTOR_CENTROIDS = {
  Technology: { x: 0.28, y: 0.32, label: 'TECHNOLOGY' },
  Financials: { x: 0.72, y: 0.30, label: 'FINANCIALS' },
  Energy: { x: 0.74, y: 0.72, label: 'ENERGY' },
  Utilities: { x: 0.26, y: 0.72, label: 'UTILITIES' },
  'Real Estate': { x: 0.50, y: 0.50, label: 'REAL ESTATE' },
};

function hexToRgba(hex, alpha = 1) {
  if (!hex) return `rgba(251, 191, 36, ${alpha})`;
  let c = hex.replace('#', '');
  if (c.length === 3) c = c.split('').map(x => x + x).join('');
  const r = parseInt(c.substring(0, 2), 16) || 0;
  const g = parseInt(c.substring(2, 4), 16) || 0;
  const b = parseInt(c.substring(4, 6), 16) || 0;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function getHeatmapColor(val) {
  if (val === 1.0) return 'rgba(251, 191, 36, 0.85)';
  if (val > 0) {
    const a = Math.min(0.85, val * 0.85);
    return `rgba(245, 158, 11, ${a.toFixed(2)})`;
  } else {
    const a = Math.min(0.85, Math.abs(val) * 0.85);
    return `rgba(244, 63, 94, ${a.toFixed(2)})`;
  }
}

export default function NetworkGraphTab({ appData }) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const animRef = useRef(null);

  const [threshold, setThreshold] = useState(0.40);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [tooltipState, setTooltipState] = useState({ visible: false, x: 0, y: 0, node: null });

  // Data references
  const graphData = useMemo(() => appData?.graph_data || { nodes: [], links: [], matrix: {}, sectors: {}, sector_colors: {} }, [appData]);
  const assetUniverse = useMemo(() => appData?.asset_universe || [], [appData]);

  // Lookup for asset stats
  const statsMap = useMemo(() => {
    const map = {};
    assetUniverse.forEach(a => {
      map[a.ticker] = a;
    });
    return map;
  }, [assetUniverse]);

  // Physics simulation state held in ref
  const simulationRef = useRef({
    nodes: [],
    links: [],
    isDragging: false,
    dragNode: null,
    width: 600,
    height: 520,
  });

  const hoverRef = useRef(null);
  const selectRef = useRef(null);

  useEffect(() => {
    hoverRef.current = hoveredNode;
  }, [hoveredNode]);

  useEffect(() => {
    selectRef.current = selectedNode;
  }, [selectedNode]);

  // Update link references based on current threshold
  const updateLinks = useCallback((nodesList, th) => {
    if (!graphData.links) return [];
    const nodeMap = {};
    nodesList.forEach(n => { nodeMap[n.id] = n; });

    return graphData.links
      .filter(l => Math.abs(l.weight) >= th)
      .map(l => ({
        source: nodeMap[l.source],
        target: nodeMap[l.target],
        weight: l.weight,
        positive: l.positive
      }))
      .filter(l => l.source && l.target);
  }, [graphData.links]);

  // Initialize and run stable force-directed physics
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !graphData.nodes || graphData.nodes.length === 0) return;

    const ctx = canvas.getContext('2d');
    const sim = simulationRef.current;

    // Measure dimensions safely
    const updateDimensions = () => {
      const rect = container.getBoundingClientRect();
      const w = Math.floor(rect.width) || 600;
      const h = Math.floor(rect.height) || 520;
      const dpr = window.devicePixelRatio || 1;

      sim.width = w;
      sim.height = h;

      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    };

    updateDimensions();

    const timeoutId = setTimeout(updateDimensions, 60);

    const handleWindowResize = () => {
      updateDimensions();
    };
    window.addEventListener('resize', handleWindowResize);

    // Group nodes by sector to calculate non-overlapping initial ring positions
    const sectorGroups = {};
    graphData.nodes.forEach(n => {
      if (!sectorGroups[n.sector]) sectorGroups[n.sector] = [];
      sectorGroups[n.sector].push(n);
    });

    const initialNodes = [];
    Object.entries(sectorGroups).forEach(([sector, secNodes]) => {
      const centroid = SECTOR_CENTROIDS[sector] || { x: 0.5, y: 0.5 };
      const cx = sim.width * centroid.x;
      const cy = sim.height * centroid.y;
      const count = secNodes.length;
      const ringRadius = 60; // Spreads nodes comfortably in a ring

      secNodes.forEach((n, idx) => {
        const angle = (idx / count) * 2 * Math.PI;
        const x = cx + Math.cos(angle) * ringRadius;
        const y = cy + Math.sin(angle) * ringRadius;

        initialNodes.push({
          id: n.id,
          sector: n.sector,
          color: n.color || '#38bdf8',
          x,
          y,
          vx: 0,
          vy: 0,
          radius: 17,
          targetCx: cx,
          targetCy: cy,
          stats: statsMap[n.id] || {}
        });
      });
    });

    sim.nodes = initialNodes;
    sim.links = updateLinks(sim.nodes, threshold);

    if (!selectedNode && sim.nodes.length > 0) {
      setSelectedNode(sim.nodes[0]);
    }

    // Main Stable Physics & Render Loop
    const loop = () => {
      const { nodes, links, width, height, isDragging, dragNode } = sim;
      const dpr = window.devicePixelRatio || 1;

      // 1. Hard Collision Non-Overlap Constraint (Prevents any overlap)
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const minDist = nodes[i].radius + nodes[j].radius + 14; // minimum 48px clearance

          if (dist < minDist) {
            const overlap = (minDist - dist);
            const nx = dx / dist;
            const ny = dy / dist;

            // Immediately separate physically
            if (nodes[i] !== dragNode) {
              nodes[i].x -= nx * overlap * 0.5;
              nodes[i].y -= ny * overlap * 0.5;
            }
            if (nodes[j] !== dragNode) {
              nodes[j].x += nx * overlap * 0.5;
              nodes[j].y += ny * overlap * 0.5;
            }

            // Dampen collision impulses
            nodes[i].vx *= 0.5;
            nodes[i].vy *= 0.5;
            nodes[j].vx *= 0.5;
            nodes[j].vy *= 0.5;
          } else if (dist < 130) {
            // Soft smooth repulsion
            const force = ((130 - dist) / 130) * 0.35;
            const nx = dx / dist;
            const ny = dy / dist;
            nodes[i].vx -= nx * force;
            nodes[i].vy -= ny * force;
            nodes[j].vx += nx * force;
            nodes[j].vy += ny * force;
          }
        }
      }

      // 2. Link Spring Tension (Normalized Direction Vectors)
      links.forEach(link => {
        const dx = link.target.x - link.source.x;
        const dy = link.target.y - link.source.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const desiredDist = 95 - Math.abs(link.weight) * 35; // 60 to 95px
        const delta = dist - desiredDist;
        const force = Math.min(1.2, Math.max(-1.2, delta * 0.003 * Math.abs(link.weight)));
        const nx = dx / dist;
        const ny = dy / dist;

        link.source.vx += nx * force;
        link.source.vy += ny * force;
        link.target.vx -= nx * force;
        link.target.vy -= ny * force;
      });

      // 3. Gentle Anchor pull towards sector community centroid
      nodes.forEach(n => {
        if (isDragging && dragNode === n) return;

        const centroid = SECTOR_CENTROIDS[n.sector] || { x: 0.5, y: 0.5 };
        const targetX = width * centroid.x;
        const targetY = height * centroid.y;

        const gx = targetX - n.x;
        const gy = targetY - n.y;
        n.vx += gx * 0.0018;
        n.vy += gy * 0.0018;

        // Mild pull toward canvas center
        n.vx += (width * 0.5 - n.x) * 0.0003;
        n.vy += (height * 0.5 - n.y) * 0.0003;

        // 4. Velocity Clamping & Heavy Damping (Kills vibration completely)
        const speed = Math.hypot(n.vx, n.vy);
        if (speed > 2.0) {
          n.vx = (n.vx / speed) * 2.0;
          n.vy = (n.vy / speed) * 2.0;
        }

        n.vx *= 0.76; // Strong damping brings equilibrium
        n.vy *= 0.76;

        // Freeze microscopic jitter
        if (Math.abs(n.vx) < 0.02) n.vx = 0;
        if (Math.abs(n.vy) < 0.02) n.vy = 0;

        n.x += n.vx;
        n.y += n.vy;

        // Bounds containment
        const pad = n.radius + 10;
        if (n.x < pad) { n.x = pad; n.vx = 0; }
        if (n.x > width - pad) { n.x = width - pad; n.vx = 0; }
        if (n.y < pad) { n.y = pad; n.vy = 0; }
        if (n.y > height - pad) { n.y = height - pad; n.vy = 0; }

        if (isNaN(n.x) || !isFinite(n.x)) n.x = width * 0.5;
        if (isNaN(n.y) || !isFinite(n.y)) n.y = height * 0.5;
      });

      // 5. Render Scene with Crisp Canvas Scaling
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      // Draw subtle sector boundary halos in the background
      Object.entries(SECTOR_CENTROIDS).forEach(([sec, info]) => {
        const cx = width * info.x;
        const cy = height * info.y;
        const secColor = graphData.sector_colors?.[sec] || '#38bdf8';

        // Subtle sector glow circle
        ctx.beginPath();
        ctx.arc(cx, cy, 85, 0, Math.PI * 2);
        ctx.fillStyle = hexToRgba(secColor, 0.035);
        ctx.fill();
        ctx.strokeStyle = hexToRgba(secColor, 0.12);
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Sector header label
        ctx.font = "bold 9px 'JetBrains Mono', monospace";
        ctx.fillStyle = hexToRgba(secColor, 0.45);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(info.label, cx, cy - 95);
      });

      const curHover = hoverRef.current;
      const curSelect = selectRef.current;

      // Draw Links
      links.forEach(link => {
        const isHoverConn = curHover && (link.source === curHover || link.target === curHover);
        const isSelectConn = curSelect && (link.source.id === curSelect.id || link.target.id === curSelect.id);

        ctx.beginPath();
        ctx.moveTo(link.source.x, link.source.y);
        ctx.lineTo(link.target.x, link.target.y);

        let alpha = 0.15 + Math.abs(link.weight) * 0.40;
        let lw = 1 + Math.abs(link.weight) * 2.0;

        if (curHover || curSelect) {
          if (isHoverConn || isSelectConn) {
            alpha = 0.95;
            lw += 1.8;
            ctx.strokeStyle = '#fbbf24';
          } else {
            alpha = 0.03;
            ctx.strokeStyle = '#334155';
          }
        } else {
          ctx.strokeStyle = link.positive
            ? `rgba(251, 191, 36, ${alpha.toFixed(2)})`
            : `rgba(244, 63, 94, ${alpha.toFixed(2)})`;
        }

        ctx.lineWidth = lw;
        ctx.stroke();
      });

      // Draw Nodes
      nodes.forEach(node => {
        const isHover = curHover && curHover.id === node.id;
        const isSelect = curSelect && curSelect.id === node.id;

        // Outer glow on focus
        if (isHover || isSelect) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius + 8, 0, Math.PI * 2);
          ctx.fillStyle = hexToRgba(node.color, 0.4);
          ctx.fill();
        }

        // Node Circle Body with solid background to pop out from background grid
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = '#0f121a';
        ctx.fill();

        ctx.lineWidth = isHover || isSelect ? 3 : 2;
        ctx.strokeStyle = node.color;
        ctx.stroke();

        // Ticker label
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.id, node.x, node.y);
      });

      ctx.restore();

      animRef.current = requestAnimationFrame(loop);
    };

    loop();

    return () => {
      clearTimeout(timeoutId);
      window.removeEventListener('resize', handleWindowResize);
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [graphData.nodes, statsMap, updateLinks]);

  // Dynamic link filtering on threshold change
  useEffect(() => {
    const sim = simulationRef.current;
    if (sim.nodes && sim.nodes.length > 0) {
      sim.links = updateLinks(sim.nodes, threshold);
    }
  }, [threshold, updateLinks]);

  // Pointer drag and inspect handlers
  const handlePointerDown = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const sim = simulationRef.current;
    const hit = sim.nodes.find(n => {
      const dx = n.x - x;
      const dy = n.y - y;
      return Math.sqrt(dx * dx + dy * dy) <= n.radius + 6;
    });

    if (hit) {
      sim.isDragging = true;
      sim.dragNode = hit;
      setSelectedNode(hit);
    }
  };

  const handlePointerMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const sim = simulationRef.current;

    if (sim.isDragging && sim.dragNode) {
      sim.dragNode.x = x;
      sim.dragNode.y = y;
      sim.dragNode.vx = 0;
      sim.dragNode.vy = 0;
      return;
    }

    const hit = sim.nodes.find(n => {
      const dx = n.x - x;
      const dy = n.y - y;
      return Math.sqrt(dx * dx + dy * dy) <= n.radius + 6;
    });

    if (hit !== hoveredNode) {
      setHoveredNode(hit || null);
    }

    if (hit) {
      setTooltipState({
        visible: true,
        x: e.clientX - rect.left + 15,
        y: e.clientY - rect.top + 15,
        node: hit
      });
    } else {
      setTooltipState(prev => ({ ...prev, visible: false }));
    }
  };

  const handlePointerUp = () => {
    const sim = simulationRef.current;
    sim.isDragging = false;
    sim.dragNode = null;
  };

  // Connected peers list for selected node
  const topPeers = useMemo(() => {
    if (!selectedNode || !graphData.matrix || !graphData.matrix.tickers) return [];
    const tickers = graphData.matrix.tickers;
    const values = graphData.matrix.values;
    const nodeIdx = tickers.indexOf(selectedNode.id);
    if (nodeIdx === -1) return [];

    const row = values[nodeIdx] || [];
    return tickers
      .map((t, idx) => ({ ticker: t, val: row[idx] }))
      .filter(item => item.ticker !== selectedNode.id)
      .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
      .slice(0, 5);
  }, [selectedNode, graphData.matrix]);

  // Heatmap Matrix Data
  const matrixTickers = graphData.matrix?.tickers || [];
  const matrixValues = graphData.matrix?.values || [];

  return (
    <div className="tab-pane active" style={{ minWidth: 0, width: '100%', overflow: 'hidden' }}>
      <div className="dashboard-grid-2col" style={{ minWidth: 0 }}>
        {/* Interactive Canvas Network */}
        <div className="panel-card" style={{ minWidth: 0 }}>
          <div className="panel-header">
            <div className="panel-title-group">
              <div className="panel-title">Interactive Stock Correlation Network & GNN Node Embeddings</div>
              <span className="panel-subtitle">
                20 assets clustered by sector. Drag nodes to explore cluster physics. Hover to view embeddings & metrics.
              </span>
            </div>
            <div className="panel-actions" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.78rem', color: '#94a3b8', fontWeight: 600 }}>
                Threshold: <span style={{ color: '#38bdf8', fontFamily: "'JetBrains Mono', monospace" }}>{threshold.toFixed(2)}</span>
              </span>
              <input
                type="range"
                min="0.30"
                max="0.85"
                step="0.05"
                value={threshold}
                onChange={e => setThreshold(parseFloat(e.target.value))}
                style={{ width: '110px' }}
                className="sim-slider"
              />
            </div>
          </div>

          <div
            ref={containerRef}
            className="network-container"
          >
            <canvas
              ref={canvasRef}
              className="network-canvas"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={() => {
                handlePointerUp();
                setHoveredNode(null);
                setTooltipState(prev => ({ ...prev, visible: false }));
              }}
            />

            {/* Hover Tooltip */}
            {tooltipState.visible && tooltipState.node && (
              <div
                className="network-tooltip"
                style={{
                  display: 'block',
                  position: 'absolute',
                  left: tooltipState.x,
                  top: tooltipState.y,
                  pointerEvents: 'none',
                  zIndex: 20
                }}
              >
                <div style={{ fontWeight: 800, fontSize: '0.95rem', color: tooltipState.node.color, display: 'flex', justifyContent: 'space-between', gap: '12px' }}>
                  <span>{tooltipState.node.id}</span>
                  <span style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 'normal' }}>{tooltipState.node.sector}</span>
                </div>
                <div style={{ marginTop: '6px', fontSize: '0.78rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontFamily: "'JetBrains Mono', monospace" }}>
                  <div>Ann. Return: <span style={{ color: '#38bdf8' }}>{tooltipState.node.stats.annualized_return || 0}%</span></div>
                  <div>Volatility: <span style={{ color: '#f59e0b' }}>{tooltipState.node.stats.volatility || 0}%</span></div>
                  <div>Sharpe: <span style={{ color: '#10b981' }}>{tooltipState.node.stats.sharpe || 0}</span></div>
                </div>
              </div>
            )}

            {/* Sector Cluster Legend */}
            <div className="network-legend">
              <div style={{ fontWeight: 700, marginBottom: '4px', color: '#f8fafc' }}>Sector Clusters</div>
              {Object.entries(graphData.sector_colors || {}).map(([sec, col]) => (
                <div key={sec} className="legend-item" style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem' }}>
                  <span className="legend-dot" style={{ width: '8px', height: '8px', borderRadius: '50%', background: col }} />
                  <span>{sec}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Node Inspector Card & Heatmap */}
        <div className="panel-card" style={{ minWidth: 0 }}>
          <div className="panel-header">
            <div className="panel-title-group">
              <div className="panel-title">Selected Node Details</div>
              <span className="panel-subtitle">Click any asset node on the left to inspect</span>
            </div>
          </div>

          <div id="node-inspect-card" style={{ minHeight: '120px' }}>
            {selectedNode ? (
              <div style={{ padding: '0.5rem 0' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                  <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: selectedNode.color, color: '#000', fontWeight: 800, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.05rem' }}>
                    {selectedNode.id}
                  </div>
                  <div>
                    <h4 style={{ color: '#f8fafc', fontFamily: "'Outfit', sans-serif", fontSize: '1.15rem', margin: 0 }}>
                      {selectedNode.id} <span style={{ fontSize: '0.85rem', color: '#94a3b8', fontWeight: 500 }}>({selectedNode.sector})</span>
                    </h4>
                    <span style={{ fontSize: '0.75rem', color: '#38bdf8' }}>GNN 32-dimensional Graph Node Active</span>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', marginBottom: '12px' }}>
                  <div style={{ background: 'rgba(11,17,26,0.6)', padding: '8px', borderRadius: '6px', textAlign: 'center', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>ANN. RETURN</div>
                    <div style={{ fontSize: '1.05rem', fontWeight: 700, color: '#38bdf8', fontFamily: "'JetBrains Mono', monospace" }}>
                      {selectedNode.stats.annualized_return || 0}%
                    </div>
                  </div>
                  <div style={{ background: 'rgba(11,17,26,0.6)', padding: '8px', borderRadius: '6px', textAlign: 'center', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>VOLATILITY</div>
                    <div style={{ fontSize: '1.05rem', fontWeight: 700, color: '#f59e0b', fontFamily: "'JetBrains Mono', monospace" }}>
                      {selectedNode.stats.volatility || 0}%
                    </div>
                  </div>
                  <div style={{ background: 'rgba(11,17,26,0.6)', padding: '8px', borderRadius: '6px', textAlign: 'center', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>SHARPE</div>
                    <div style={{ fontSize: '1.05rem', fontWeight: 700, color: '#10b981', fontFamily: "'JetBrains Mono', monospace" }}>
                      {selectedNode.stats.sharpe || 0}
                    </div>
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#94a3b8', marginBottom: '6px' }}>Top Correlated Peers (|r|):</div>
                  <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                    {topPeers.map(p => (
                      <span
                        key={p.ticker}
                        onClick={() => {
                          const n = simulationRef.current.nodes.find(item => item.id === p.ticker);
                          if (n) setSelectedNode(n);
                        }}
                        style={{
                          fontSize: '0.72rem',
                          padding: '3px 8px',
                          borderRadius: '4px',
                          background: p.val > 0 ? 'rgba(16,185,129,0.12)' : 'rgba(244,63,94,0.12)',
                          color: p.val > 0 ? '#10b981' : '#f43f5e',
                          fontFamily: "'JetBrains Mono', monospace",
                          fontWeight: 600,
                          cursor: 'pointer',
                          border: '1px solid rgba(255,255,255,0.06)'
                        }}
                      >
                        {p.ticker}: {p.val > 0 ? '+' : ''}{p.val.toFixed(2)}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <p style={{ color: '#64748b', fontSize: '0.85rem', padding: '1.5rem 0', textAlign: 'center' }}>
                Click any asset in the network graph to inspect its 32-dimensional GNN features, returns, and correlation peers.
              </p>
            )}
          </div>

          <div className="panel-header" style={{ marginTop: '1rem' }}>
            <div className="panel-title-group">
              <div className="panel-title" style={{ fontSize: '1rem' }}>20x20 Correlation Heatmap Matrix</div>
              <span className="panel-subtitle">Pearson correlation coefficients (|r| ≥ 0.40 generates GNN edges)</span>
            </div>
          </div>

          <div className="heatmap-container" style={{ overflowX: 'auto', maxHeight: '250px' }}>
            {matrixTickers.length > 0 && (
              <table className="heatmap-table">
                <thead>
                  <tr>
                    <th></th>
                    {matrixTickers.map(t => (
                      <th key={t}>{t}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {matrixTickers.map((rowT, i) => (
                    <tr key={rowT}>
                      <th>{rowT}</th>
                      {matrixTickers.map((colT, j) => {
                        const val = matrixValues[i]?.[j] ?? 0;
                        const col = getHeatmapColor(val);
                        const textColor = Math.abs(val) > 0.4 ? '#f8fafc' : '#94a3b8';
                        return (
                          <td
                            key={colT}
                            onClick={() => {
                              const n = simulationRef.current.nodes.find(item => item.id === rowT);
                              if (n) setSelectedNode(n);
                            }}
                            style={{
                              background: col,
                              color: textColor,
                              cursor: 'pointer'
                            }}
                            title={`${rowT} × ${colT}: ${val.toFixed(2)}`}
                          >
                            {val === 1.0 ? '1' : val.toFixed(2).replace('0.', '.')}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
