/**
 * AI Compiler & Kernel Playground - Dynamic Interactive Graph Visualizer Engine
 * Single Source of Truth: Renders dynamic computation graphs directly from ActiveProgramContext
 * Features: Before/After Optimization Toggle, Zoom, Pan, Fit, Node Tooltips & Compiler Metrics
 */

let currentGraphMode = 'after'; // 'before' or 'after'
let graphScale = 1.0;
let graphTranslate = { x: 0, y: 0 };
let isPanning = false;
let startPan = { x: 0, y: 0 };

document.addEventListener('DOMContentLoaded', () => {
  initGraphVisualizer();
});

function initGraphVisualizer() {
  const container = document.getElementById('graph-canvas-container');
  if (!container) return;

  // Render graph from active context
  renderGraph();

  // Mode Buttons (Before / After Optimization)
  const btnBefore = document.getElementById('btn-graph-before');
  const btnAfter = document.getElementById('btn-graph-after');
  
  if (btnBefore) {
    btnBefore.addEventListener('click', () => {
      currentGraphMode = 'before';
      updateActiveToggleButtons();
      renderGraph();
    });
  }

  if (btnAfter) {
    btnAfter.addEventListener('click', () => {
      currentGraphMode = 'after';
      updateActiveToggleButtons();
      renderGraph();
    });
  }

  // Zoom / Pan Controls
  const btnZoomIn = document.getElementById('btn-graph-zoom-in');
  const btnZoomOut = document.getElementById('btn-graph-zoom-out');
  const btnFit = document.getElementById('btn-graph-fit');

  if (btnZoomIn) {
    btnZoomIn.addEventListener('click', () => {
      graphScale = Math.min(graphScale + 0.15, 2.5);
      applyGraphTransform();
    });
  }

  if (btnZoomOut) {
    btnZoomOut.addEventListener('click', () => {
      graphScale = Math.max(graphScale - 0.15, 0.5);
      applyGraphTransform();
    });
  }

  if (btnFit) {
    btnFit.addEventListener('click', () => {
      resetGraphTransform();
    });
  }

  // Mouse Pan Handling
  container.addEventListener('mousedown', (e) => {
    isPanning = true;
    startPan = { x: e.clientX - graphTranslate.x, y: e.clientY - graphTranslate.y };
  });

  window.addEventListener('mousemove', (e) => {
    if (!isPanning) return;
    graphTranslate.x = e.clientX - startPan.x;
    graphTranslate.y = e.clientY - startPan.y;
    applyGraphTransform();
  });

  window.addEventListener('mouseup', () => {
    isPanning = false;
  });

  // Wheel zoom
  container.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.1 : -0.1;
    graphScale = Math.max(0.5, Math.min(2.5, graphScale + delta));
    applyGraphTransform();
  });
}

function updateActiveToggleButtons() {
  const btnBefore = document.getElementById('btn-graph-before');
  const btnAfter = document.getElementById('btn-graph-after');
  if (btnBefore) btnBefore.classList.toggle('active', currentGraphMode === 'before');
  if (btnAfter) btnAfter.classList.toggle('active', currentGraphMode === 'after');
}

function resetGraphTransform() {
  graphScale = 1.0;
  graphTranslate = { x: 0, y: 0 };
  applyGraphTransform();
}

function applyGraphTransform() {
  const g = document.getElementById('graph-svg-root-group');
  if (g) {
    g.setAttribute('transform', `translate(${graphTranslate.x}, ${graphTranslate.y}) scale(${graphScale})`);
  }
}

/**
 * Dynamically Renders the Computation Graph from ActiveProgramContext
 */
function renderGraph() {
  const svg = document.getElementById('graph-svg-element');
  const emptyState = document.getElementById('graph-empty-state');
  const mainContent = document.getElementById('graph-main-content');
  const titleElem = document.getElementById('graph-workload-title');

  const ctx = window.ActiveProgramContext;
  const isAnalyzed = !!(ctx && ctx.analyzed && ctx.analysis && ctx.analysis.graph);

  if (!isAnalyzed) {
    if (emptyState) emptyState.style.display = 'block';
    if (mainContent) mainContent.style.display = 'none';
    return;
  }

  if (emptyState) emptyState.style.display = 'none';
  if (mainContent) mainContent.style.display = 'block';

  const graphData = ctx.analysis.graph;
  const graph = graphData[currentGraphMode] || graphData.after || graphData.before;

  if (!graph || !graph.nodes) {
    if (emptyState) emptyState.style.display = 'block';
    if (mainContent) mainContent.style.display = 'none';
    return;
  }

  if (titleElem) {
    titleElem.textContent = `${ctx.workload || 'Active Workload'} (${currentGraphMode === 'after' ? 'Optimized' : 'Baseline'})`;
  }

  // Update Graph stats UI
  updateGraphStatsUI(graph.stats);

  if (!svg) return;

  let html = `<g id="graph-svg-root-group" transform="translate(${graphTranslate.x}, ${graphTranslate.y}) scale(${graphScale})">`;

  // Draw Edges with smooth Bézier curves
  (graph.edges || []).forEach(edge => {
    const fromNode = graph.nodes.find(n => n.id === edge.from);
    const toNode = graph.nodes.find(n => n.id === edge.to);
    if (fromNode && toNode) {
      const fromWidth = fromNode.type === 'fused' ? 220 : 180;
      const toWidth = toNode.type === 'fused' ? 220 : 180;
      const x1 = fromNode.x + (fromWidth / 2);
      const y1 = fromNode.y + 44;
      const x2 = toNode.x + (toWidth / 2);
      const y2 = toNode.y;

      const highlightClass = edge.highlight ? 'highlight' : '';
      const strokeColor = edge.highlight ? '#3B82F6' : '#2D3D52';

      const midY = (y1 + y2) / 2;
      const pathD = `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`;

      html += `
        <path d="${pathD}" class="graph-edge-path ${highlightClass}" stroke="${strokeColor}" stroke-width="${edge.highlight ? '2.5' : '1.5'}" fill="none" />
        <circle cx="${x2}" cy="${y2}" r="3.5" fill="${strokeColor}" />
      `;
    }
  });

  // Draw Nodes
  (graph.nodes || []).forEach(node => {
    const isFused = node.type === 'fused';
    const isOutput = node.type === 'output';
    const isInput = node.type === 'input';
    const isMemory = node.type === 'memory';

    let strokeColor = '#2D3D52';
    let fillColor = '#151D28';
    let titleColor = '#F3F4F6';

    if (isFused) {
      strokeColor = '#10B981';
      fillColor = '#0B291E';
      titleColor = '#34D399';
    } else if (isOutput) {
      strokeColor = '#06B6D4';
      fillColor = '#0E242B';
      titleColor = '#67E8F9';
    } else if (isInput) {
      strokeColor = '#374151';
      fillColor = '#111827';
      titleColor = '#E5E7EB';
    } else if (isMemory) {
      strokeColor = '#F59E0B';
      fillColor = '#261C0E';
      titleColor = '#FCD34D';
    }

    const width = isFused ? 220 : 180;
    const height = 48;
    const posX = node.x;
    const posY = node.y;

    html += `
      <g class="graph-node-group" data-tooltip="${node.tooltip || ''}" style="cursor: pointer;" transform="translate(${posX}, ${posY})">
        <rect width="${width}" height="${height}" rx="8" fill="${fillColor}" stroke="${strokeColor}" stroke-width="1.5" class="graph-node-rect ${isFused ? 'fused' : ''}"/>
        <text x="${width/2}" y="20" text-anchor="middle" class="graph-node-title" fill="${titleColor}" font-weight="600" font-size="12">${node.label}</text>
        <text x="${width/2}" y="36" text-anchor="middle" class="graph-node-sub" fill="#9CA3AF" font-size="10.5">${node.sub || ''}</text>
      </g>
    `;
  });

  html += `</g>`;
  svg.innerHTML = html;

  // Attach Tooltip events
  attachNodeTooltips();
}

function updateGraphStatsUI(stats) {
  if (!stats) return;
  const container = document.getElementById('graph-stats-container');
  if (!container) return;

  let html = '';
  for (const [key, val] of Object.entries(stats)) {
    const formattedKey = key.replace(/_/g, ' ').toUpperCase();
    html += `
      <div class="kpi-card" style="padding: 10px 14px;">
        <span class="kpi-label">${formattedKey}</span>
        <strong style="font-size: 13px; color: var(--accent-cyan);">${val}</strong>
      </div>
    `;
  }
  container.innerHTML = html;
}

function attachNodeTooltips() {
  const tooltip = document.getElementById('graph-tooltip-box');
  if (!tooltip) return;

  document.querySelectorAll('.graph-node-group').forEach(nodeGroup => {
    nodeGroup.addEventListener('mouseenter', (e) => {
      const text = nodeGroup.dataset.tooltip;
      if (text) {
        tooltip.textContent = text;
        tooltip.style.display = 'block';
        tooltip.style.left = (e.pageX + 12) + 'px';
        tooltip.style.top = (e.pageY + 12) + 'px';
      }
    });

    nodeGroup.addEventListener('mousemove', (e) => {
      tooltip.style.left = (e.pageX + 12) + 'px';
      tooltip.style.top = (e.pageY + 12) + 'px';
    });

    nodeGroup.addEventListener('mouseleave', () => {
      tooltip.style.display = 'none';
    });
  });
}

// Global API
window.renderGraph = renderGraph;

