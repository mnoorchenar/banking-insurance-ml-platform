/* ═══════════════════════════════════════════════════════════════
   Banking & Insurance ML Platform — Shared JS
   ═══════════════════════════════════════════════════════════════ */

// ── API helpers ─────────────────────────────────────────────────

async function apiFetch(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`API error: ${r.status}`);
  return r.json();
}

async function apiPost(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`API error: ${r.status}`);
  return r.json();
}

// ── DOM helpers ─────────────────────────────────────────────────
function show(id) { const el = document.getElementById(id); if (el) el.style.display = ''; }
function hide(id) { const el = document.getElementById(id); if (el) el.style.display = 'none'; }

// ── Plotly layout factory ────────────────────────────────────────

function layout(title, xLabel, yLabel, extra = {}) {
  return {
    title:   { text: title, font: { size: 14, color: '#1e293b' }, x: 0 },
    xaxis:   { title: xLabel, color: '#64748b', gridcolor: '#f1f5f9' },
    yaxis:   { title: yLabel, color: '#64748b', gridcolor: '#f1f5f9' },
    paper_bgcolor: 'transparent',
    plot_bgcolor:  'rgba(0,0,0,0)',
    font:    { family: 'Inter, system-ui, sans-serif', size: 12, color: '#374151' },
    legend:  { orientation: 'h', y: -0.15, font: { size: 11 } },
    margin:  { t: 50, l: 60, r: 20, b: 70 },
    hoverlabel: { bgcolor: '#1e293b', font: { color: '#fff', size: 12 } },
    ...extra,
  };
}

// ── Metric cards renderer ────────────────────────────────────────

function renderMetrics(containerId, metrics, trainSize, testSize) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const items = [
    { label: 'Accuracy',    value: (metrics.accuracy  * 100).toFixed(2) + '%', color: '#4f46e5' },
    { label: 'AUC-ROC',     value: (metrics.auc       * 100).toFixed(2) + '%', color: '#7c3aed' },
    { label: 'F1 Score',    value: (metrics.f1         * 100).toFixed(2) + '%', color: '#16a34a' },
    { label: 'Precision',   value: (metrics.precision  * 100).toFixed(2) + '%', color: '#f97316' },
    { label: 'Recall',      value: (metrics.recall     * 100).toFixed(2) + '%', color: '#0d9488' },
    { label: 'Train / Test',value: `${trainSize} / ${testSize}`,                color: '#6b7280' },
  ];

  container.innerHTML = items.map(({ label, value, color }) => `
    <div class="col-6 col-sm-4 col-lg-2">
      <div class="metric-item">
        <div class="label">${label}</div>
        <div class="value" style="color:${color}">${value}</div>
      </div>
    </div>
  `).join('');
}

// ── KPI card helper ──────────────────────────────────────────────

function kpiCard(icon, bg, label, value, sub) {
  return `
    <div class="col-6 col-xl-3">
      <div class="kpi-card">
        <div class="kpi-icon ${bg}"><i class="bi ${icon}"></i></div>
        <div class="kpi-body">
          <div class="kpi-label">${label}</div>
          <div class="kpi-value">${value}</div>
          <div class="kpi-sub">${sub}</div>
        </div>
      </div>
    </div>`;
}

// ── Confusion matrix renderer ────────────────────────────────────

function plotCM(divId, cm, dataset) {
  const isBank   = dataset === 'banking';
  const labels   = isBank ? ['No Default', 'Default'] : ['Low Claim', 'High Claim'];
  const z        = cm;
  const total    = z.flat().reduce((a, b) => a + b, 0);
  const zText    = z.map(row => row.map(v => `${v}<br>${(v / total * 100).toFixed(1)}%`));

  Plotly.newPlot(divId, [{
    type: 'heatmap',
    z: z, x: labels.map(l => 'Pred: ' + l), y: labels.map(l => 'Actual: ' + l),
    colorscale: 'Blues', showscale: false,
    text: zText, texttemplate: '%{text}',
    hovertemplate: 'Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
  }], {
    title: { text: 'Confusion Matrix', font: { size: 14 }, x: 0 },
    paper_bgcolor: 'transparent', plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { t: 50, l: 90, r: 20, b: 70 },
    annotations: [
      { x: 0, y: 0, text: 'TN', showarrow: false, font: { size: 10, color: '#6b7280' } },
      { x: 1, y: 0, text: 'FP', showarrow: false, font: { size: 10, color: '#6b7280' } },
      { x: 0, y: 1, text: 'FN', showarrow: false, font: { size: 10, color: '#6b7280' } },
      { x: 1, y: 1, text: 'TP', showarrow: false, font: { size: 10, color: '#6b7280' } },
    ],
  }, { responsive: true, displayModeBar: false });
}

// ── Heatmap (correlation) ────────────────────────────────────────

function plotHeatmap(divId, matrix, labels, title) {
  Plotly.newPlot(divId, [{
    type: 'heatmap', z: matrix, x: labels, y: labels,
    colorscale: [
      [0, '#b91c1c'], [0.25, '#fca5a5'], [0.5, '#f9fafb'],
      [0.75, '#93c5fd'], [1, '#1d4ed8'],
    ],
    zmin: -1, zmax: 1, showscale: true,
    text: matrix.map(row => row.map(v => v.toFixed(2))),
    texttemplate: '%{text}',
    hovertemplate: '%{y} × %{x}<br>r = %{z:.3f}<extra></extra>',
  }], {
    title: { text: title, font: { size: 14 }, x: 0 },
    paper_bgcolor: 'transparent', plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { t: 50, l: 120, r: 60, b: 120 },
    xaxis: { tickangle: -35, color: '#374151' },
    yaxis: { color: '#374151' },
  }, { responsive: true, displayModeBar: false });
}
