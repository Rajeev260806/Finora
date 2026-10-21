/**
 * Finora Stock Correlation Network & Heatmap Visualization Engine
 * Interactive Force-directed canvas simulation with physics, clustering, and tooltip inspection.
 */

const FinoraNetwork = (function() {
  let canvas, ctx;
  let nodes = [];
  let links = [];
  let rawLinks = [];
  let rawMatrix = null;
  let animId = null;
  let selectedNode = null;
  let hoveredNode = null;
  let isDragging = false;
  let dragNode = null;
  let currentThreshold = 0.40;

  // Sector community positions (approximate centroids to guide clustering)
  const SECTOR_CENTROIDS = {
    "Technology": { x: 0.30, y: 0.30 },
    "Financials": { x: 0.70, y: 0.30 },
    "Energy": { x: 0.75, y: 0.70 },
    "Utilities": { x: 0.25, y: 0.75 },
    "Real Estate": { x: 0.50, y: 0.50 }
  };

  function init(containerId, graphData, assetUniverse) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    canvas = document.createElement('canvas');
    canvas.className = 'network-canvas';
    container.appendChild(canvas);

    // Create Tooltip
    const tooltip = document.createElement('div');
    tooltip.className = 'network-tooltip';
    tooltip.id = 'net-tooltip';
    container.appendChild(tooltip);

    // Create Legend
    const legend = document.createElement('div');
    legend.className = 'network-legend';
    legend.innerHTML = `
      <div style="font-weight: 700; margin-bottom: 4px; color: #f8fafc;">Sector Clusters</div>
      ${Object.entries(graphData.sector_colors || {}).map(([sec, col]) => `
        <div class="legend-item">
          <span class="legend-dot" style="background: ${col}"></span>
          <span>${sec}</span>
        </div>
      `).join('')}
    `;
    container.appendChild(legend);

    // Resize canvas
    resize();
    window.addEventListener('resize', resize);

    // Prepare Node data with metrics lookup
    const statsMap = {};
    if (assetUniverse) {
      assetUniverse.forEach(a => statsMap[a.ticker] = a);
    }

    nodes = graphData.nodes.map((n, i) => {
      const targetCentroid = SECTOR_CENTROIDS[n.sector] || { x: 0.5, y: 0.5 };
      const jitterX = (Math.random() - 0.5) * 120;
      const jitterY = (Math.random() - 0.5) * 120;

      return {
        id: n.id,
        sector: n.sector,
        color: n.color || '#38bdf8',
        x: canvas.width * targetCentroid.x + jitterX,
        y: canvas.height * targetCentroid.y + jitterY,
        vx: 0,
        vy: 0,
        radius: 18,
        stats: statsMap[n.id] || {}
      };
    });

    rawLinks = graphData.links || [];
    rawMatrix = graphData.matrix || null;
    filterLinks(currentThreshold);

    // Event Listeners
    setupEvents(container, tooltip);

    // Start simulation loop
    startSimulation();

    // Render initial heatmap if container exists
    renderHeatmap('heatmap-container', graphData);
  }

  function resize() {
    if (!canvas) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx = canvas.getContext('2d');
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }

  function filterLinks(threshold) {
    currentThreshold = threshold;
    const nodeMap = {};
    nodes.forEach(n => nodeMap[n.id] = n);

    links = rawLinks
      .filter(l => Math.abs(l.weight) >= threshold)
      .map(l => ({
        source: nodeMap[l.source],
        target: nodeMap[l.target],
        weight: l.weight,
        positive: l.positive
      }))
      .filter(l => l.source && l.target);
  }

  function updatePhysics() {
    const width = canvas.width / window.devicePixelRatio;
    const height = canvas.height / window.devicePixelRatio;

    // Repulsion between all nodes
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[j].x - nodes[i].x;
        const dy = nodes[j].y - nodes[i].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        if (dist < 220) {
          const force = (220 - dist) / dist * 0.12;
          nodes[i].vx -= dx * force;
          nodes[i].vy -= dy * force;
          nodes[j].vx += dx * force;
          nodes[j].vy += dy * force;
        }
      }
    }

    // Attraction along links
    links.forEach(link => {
      const dx = link.target.x - link.source.x;
      const dy = link.target.y - link.source.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const desiredDist = 120 - (link.weight * 60);
      const force = (dist - desiredDist) * 0.008 * link.weight;

      link.source.vx += dx * force;
      link.source.vy += dy * force;
      link.target.vx -= dx * force;
      link.target.vy -= dy * force;
    });

    // Sector gravity towards centroid
    nodes.forEach(n => {
      if (n === dragNode) return;
      const target = SECTOR_CENTROIDS[n.sector] || { x: 0.5, y: 0.5 };
      const gx = width * target.x - n.x;
      const gy = height * target.y - n.y;
      n.vx += gx * 0.002;
      n.vy += gy * 0.002;

      // Center gravity
      n.vx += (width / 2 - n.x) * 0.0005;
      n.vy += (height / 2 - n.y) * 0.0005;

      // Friction & update position
      n.vx *= 0.88;
      n.vy *= 0.88;
      n.x += n.vx;
      n.y += n.vy;

      // Bounding box padding
      n.x = Math.max(30, Math.min(width - 30, n.x));
      n.y = Math.max(30, Math.min(height - 30, n.y));
    });
  }

  function draw() {
    const width = canvas.width / window.devicePixelRatio;
    const height = canvas.height / window.devicePixelRatio;

    ctx.clearRect(0, 0, width, height);

    // Draw Links
    links.forEach(link => {
      const isConnectedToHovered = hoveredNode && (link.source === hoveredNode || link.target === hoveredNode);
      const isConnectedToSelected = selectedNode && (link.source === selectedNode || link.target === selectedNode);

      ctx.beginPath();
      ctx.moveTo(link.source.x, link.source.y);
      ctx.lineTo(link.target.x, link.target.y);

      let alpha = 0.15 + (link.weight * 0.45);
      let lw = 1 + (link.weight * 2.5);

      if (hoveredNode || selectedNode) {
        if (isConnectedToHovered || isConnectedToSelected) {
          alpha = 0.85;
          lw += 1.5;
          ctx.strokeStyle = '#38bdf8';
        } else {
          alpha = 0.04;
          ctx.strokeStyle = '#475569';
        }
      } else {
        ctx.strokeStyle = link.positive ? 'rgba(56, 189, 248, ' + alpha + ')' : 'rgba(244, 63, 94, ' + alpha + ')';
      }

      ctx.lineWidth = lw;
      ctx.stroke();
    });

    // Draw Nodes
    nodes.forEach(node => {
      const isHovered = node === hoveredNode;
      const isSelected = node === selectedNode;

      // Outer glow if hovered or selected
      if (isHovered || isSelected) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius + 8, 0, Math.PI * 2);
        ctx.fillStyle = hexToRgba(node.color, 0.35);
        ctx.fill();
      }

      // Main Node circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
      ctx.fillStyle = '#0f172a';
      ctx.fill();

      ctx.lineWidth = isHovered || isSelected ? 3 : 2;
      ctx.strokeStyle = node.color;
      ctx.stroke();

      // Node label (Ticker symbol)
      ctx.fillStyle = '#f8fafc';
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.id, node.x, node.y);
    });
  }

  function startSimulation() {
    function loop() {
      updatePhysics();
      draw();
      animId = requestAnimationFrame(loop);
    }
    loop();
  }

  function setupEvents(container, tooltip) {
    function getNodeAt(x, y) {
      for (let i = nodes.length - 1; i >= 0; i--) {
        const n = nodes[i];
        const dx = x - n.x;
        const dy = y - n.y;
        if (Math.sqrt(dx * dx + dy * dy) <= n.radius + 4) {
          return n;
        }
      }
      return null;
    }

    canvas.addEventListener('mousemove', e => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (isDragging && dragNode) {
        dragNode.x = x;
        dragNode.y = y;
        return;
      }

      const found = getNodeAt(x, y);
      if (found !== hoveredNode) {
        hoveredNode = found;
        if (hoveredNode) {
          canvas.style.cursor = 'pointer';
          showTooltip(tooltip, hoveredNode, e.clientX - rect.left + 15, e.clientY - rect.top + 15);
        } else {
          canvas.style.cursor = 'default';
          hideTooltip(tooltip);
        }
      }
    });

    canvas.addEventListener('mousedown', e => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const found = getNodeAt(x, y);
      if (found) {
        isDragging = true;
        dragNode = found;
        selectedNode = found;
        triggerNodeSelect(found);
      } else {
        selectedNode = null;
        hideTooltip(tooltip);
      }
    });

    window.addEventListener('mouseup', () => {
      isDragging = false;
      dragNode = null;
    });

    canvas.addEventListener('mouseleave', () => {
      hoveredNode = null;
      hideTooltip(tooltip);
    });
  }

  function showTooltip(tooltip, node, x, y) {
    const stats = node.stats || {};
    // Find top correlated peers
    const peers = links
      .filter(l => l.source === node || l.target === node)
      .map(l => {
        const peer = l.source === node ? l.target : l.source;
        return `${peer.id} (${(l.weight).toFixed(2)})`;
      })
      .slice(0, 4)
      .join(', ');

    tooltip.style.display = 'block';
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
    tooltip.innerHTML = `
      <div style="font-weight: 800; font-size: 1rem; color: ${node.color}; display: flex; align-items: center; justify-content: space-between;">
        <span>${node.id}</span>
        <span style="font-size: 0.72rem; color: #94a3b8; font-weight: normal;">${node.sector}</span>
      </div>
      <div style="margin-top: 6px; font-size: 0.78rem; display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-family: 'JetBrains Mono', monospace;">
        <div>Ann. Return: <span style="color: #38bdf8;">${stats.annualized_return || 0}%</span></div>
        <div>Volatility: <span style="color: #f59e0b;">${stats.volatility || 0}%</span></div>
        <div>Sharpe Ratio: <span style="color: #10b981;">${stats.sharpe || 0}</span></div>
      </div>
      ${peers ? `<div style="margin-top: 6px; font-size: 0.72rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 4px;">Connected: ${peers}</div>` : ''}
    `;
  }

  function hideTooltip(tooltip) {
    if (tooltip) tooltip.style.display = 'none';
  }

  function triggerNodeSelect(node) {
    const inspectCard = document.getElementById('node-inspect-card');
    if (!inspectCard) return;

    const stats = node.stats || {};
    inspectCard.innerHTML = `
      <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <div style="width: 38px; height: 38px; border-radius: 10px; background: ${node.color}; color: #000; font-weight: 800; display: flex; align-items: center; justify-content: center; font-size: 1rem;">
          ${node.id}
        </div>
        <div>
          <h4 style="color: #f8fafc; font-family: 'Outfit', sans-serif; font-size: 1.1rem; margin: 0;">${node.id} (${node.sector})</h4>
          <span style="font-size: 0.75rem; color: #94a3b8;">GNN 32-dim Node Embedding Vector Active</span>
        </div>
      </div>
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px;">
        <div style="background: rgba(11,17,26,0.6); padding: 8px; border-radius: 6px; text-align: center;">
          <div style="font-size: 0.7rem; color: #64748b;">ANN. RETURN</div>
          <div style="font-size: 1.1rem; font-weight: 700; color: #38bdf8; font-family: 'JetBrains Mono', monospace;">${stats.annualized_return || 0}%</div>
        </div>
        <div style="background: rgba(11,17,26,0.6); padding: 8px; border-radius: 6px; text-align: center;">
          <div style="font-size: 0.7rem; color: #64748b;">VOLATILITY</div>
          <div style="font-size: 1.1rem; font-weight: 700; color: #f59e0b; font-family: 'JetBrains Mono', monospace;">${stats.volatility || 0}%</div>
        </div>
        <div style="background: rgba(11,17,26,0.6); padding: 8px; border-radius: 6px; text-align: center;">
          <div style="font-size: 0.7rem; color: #64748b;">SHARPE</div>
          <div style="font-size: 1.1rem; font-weight: 700; color: #10b981; font-family: 'JetBrains Mono', monospace;">${stats.sharpe || 0}</div>
        </div>
      </div>
    `;
  }

  function renderHeatmap(containerId, graphData) {
    const container = document.getElementById(containerId);
    if (!container || !graphData.matrix) return;

    const tickers = graphData.matrix.tickers;
    const values = graphData.matrix.values;

    let html = `<table class="heatmap-table"><thead><tr><th></th>`;
    tickers.forEach(t => { html += `<th>${t}</th>`; });
    html += `</tr></thead><tbody>`;

    for (let i = 0; i < tickers.length; i++) {
      html += `<tr><th>${tickers[i]}</th>`;
      for (let j = 0; j < tickers.length; j++) {
        const val = values[i][j];
        const col = getHeatmapColor(val);
        const textColor = Math.abs(val) > 0.4 ? '#f8fafc' : '#94a3b8';
        html += `<td style="background: ${col}; color: ${textColor};" title="${tickers[i]} - ${tickers[j]}: ${val.toFixed(2)}">${val === 1.0 ? '1' : val.toFixed(2).replace('0.', '.')}</td>`;
      }
      html += `</tr>`;
    }
    html += `</tbody></table>`;
    container.innerHTML = html;
  }

  function getHeatmapColor(val) {
    if (val === 1.0) return 'rgba(14, 165, 233, 0.7)';
    if (val > 0) {
      // cyan/blue scale
      const a = Math.min(0.8, val * 0.85);
      return `rgba(14, 165, 233, ${a.toFixed(2)})`;
    } else {
      // rose/red scale
      const a = Math.min(0.8, Math.abs(val) * 0.85);
      return `rgba(244, 63, 94, ${a.toFixed(2)})`;
    }
  }

  function hexToRgba(hex, alpha) {
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  return {
    init,
    filterLinks
  };
})();
