// ─────────────────────────────────────────────
// PLOTLY DEFAULTS
// ─────────────────────────────────────────────
const COLORS = ['#3b82f6','#22c55e','#f59e0b','#a78bfa','#22d3ee','#fb7185','#94a3b8'];

const LAYOUT_BASE = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(0,0,0,0)',
  font:   { color: '#94a3b8', size: 11 },
  margin: { t: 20, b: 40, l: 50, r: 20 },
  legend: { bgcolor: 'rgba(0,0,0,0)', bordercolor: '#334155', borderwidth: 1 },
  xaxis:  { gridcolor: '#1e293b', zerolinecolor: '#334155' },
  yaxis:  { gridcolor: '#1e293b', zerolinecolor: '#334155' },
};

function layout(extra = {}) {
  return Object.assign({}, LAYOUT_BASE, extra);
}

function plt(id, data, l) {
  Plotly.newPlot(id, data, l, { responsive: true, displayModeBar: false });
}

// ─────────────────────────────────────────────
// NAV
// ─────────────────────────────────────────────
function show(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  event.currentTarget.classList.add('active');
  renderPage(page);
}

// ─────────────────────────────────────────────
// OVERVIEW
// ─────────────────────────────────────────────
function safeAuc(dataset, model) {
  try { const v = dataset[model] && dataset[model].auc; return (v && isFinite(v)) ? +v : 0.5; }
  catch(e) { return 0.5; }
}

function renderOverview() {
  const models  = ['Decision Tree','Bagging (RF)','AdaBoost','Gradient Boosting','Logistic (GLM)'];
  const cr_aucs = models.map(m => safeAuc(CR,  m));
  const ch_aucs = models.map(m => safeAuc(CHN, m));

  plt('ov-radar', [
    { type:'scatterpolar', r:cr_aucs, theta:models, fill:'toself',
      name:'Credit Risk', line:{color:'#3b82f6'} },
    { type:'scatterpolar', r:ch_aucs, theta:models, fill:'toself',
      name:'Churn',       line:{color:'#22c55e'} },
  ], layout({
    polar: {
      bgcolor: 'rgba(0,0,0,0)',
      radialaxis: { visible:true, range:[0.5,1], gridcolor:'#334155', color:'#64748b' },
      angularaxis: { gridcolor:'#334155', color:'#64748b' },
    }
  }));

  // Dataset sizes — read safely from _meta, fallback to known sizes
  const crN   = (CR._meta  && CR._meta.n_train  && CR._meta.n_test)
                ? CR._meta.n_train  + CR._meta.n_test  : 2500;
  const insN  = (INS._meta && INS._meta.n_train && INS._meta.n_test)
                ? INS._meta.n_train + INS._meta.n_test : 2500;
  const chnN  = (CHN._meta && CHN._meta.n_train && CHN._meta.n_test)
                ? CHN._meta.n_train + CHN._meta.n_test : 2500;
  const frdN  = (FRD._meta && FRD._meta.n_test)
                ? FRD._meta.n_test * 4 : 3000;

  plt('ov-bar', [{
    type:'bar',
    x: ['Credit Risk','Insurance','Churn','Fraud'],
    y: [crN, insN, chnN, frdN],
    marker: { color: COLORS },
    text: [crN, insN, chnN, frdN].map(String),
    textposition: 'outside',
  }], layout({ showlegend:false, yaxis:{ title:'Records', range:[0, Math.max(crN,insN,chnN,frdN)*1.2] } }));
}

// ─────────────────────────────────────────────
// CREDIT RISK
// ─────────────────────────────────────────────
function renderCredit() {
  const ORDER = ['Decision Tree','Bagging (RF)','AdaBoost','Gradient Boosting',
                 'Logistic (GLM)','GLM (statsmodels)'];
  const BADGE = ['badge-dt','badge-rf','badge-ab','badge-gb','badge-lr','badge-glm'];

  const kpis = document.getElementById('cr-kpis');
  kpis.innerHTML = '';
  ORDER.forEach((m, i) => {
    const d = CR[m]; if (!d) return;
    kpis.innerHTML += `<div class="col-6 col-lg-2">
      <div class="kpi-card text-center">
        <span class="model-badge ${BADGE[i]}">${m.split(' ')[0]}</span>
        <div class="kpi-val mt-1">${d.auc}</div>
        <div class="kpi-label">AUC</div>
        <div style="font-size:.75rem;color:#64748b;">Acc ${(d.accuracy*100).toFixed(1)}%</div>
      </div></div>`;
  });

  // ROC
  const rocTraces = ORDER.map((m, i) => {
    const d = CR[m]; if (!d) return null;
    return { x:d.fpr, y:d.tpr, mode:'lines', name:`${m} (${d.auc})`,
             line:{ color:COLORS[i], width:2 } };
  }).filter(Boolean);
  rocTraces.push({ x:[0,1], y:[0,1], mode:'lines', name:'Random',
                   line:{ color:'#475569', dash:'dash', width:1 } });
  plt('cr-roc', rocTraces, layout({
    xaxis:{ title:'False Positive Rate' }, yaxis:{ title:'True Positive Rate' }
  }));

  // Importance (GBM)
  const imp = CR['Gradient Boosting'].feat_imp;
  plt('cr-imp', [{
    type:'bar', orientation:'h',
    x: imp.map(d => d.importance),
    y: imp.map(d => d.feature),
    marker:{ color:'#f59e0b' }
  }], layout({ margin:{l:130,r:20,t:10,b:30}, showlegend:false,
               yaxis:{ autorange:'reversed' } }));

  // Confusion matrix
  const cm = CR['Gradient Boosting'].cm;
  plt('cr-cm', [{
    type:'heatmap', z:cm,
    x:['Pred 0','Pred 1'], y:['Act 0','Act 1'],
    colorscale:[[0,'#1e293b'],[1,'#2563eb']],
    text: cm.map(r => r.map(v => v.toString())),
    texttemplate:'%{text}', showscale:false,
  }], layout({ margin:{t:10,b:60,l:70,r:20} }));

  // GLM Coefficients
  const glm = CR['GLM (statsmodels)'];
  if (glm && glm.coef) {
    const entries = Object.entries(glm.coef).filter(([k]) => k !== 'const');
    const sorted  = entries.sort((a, b) => b[1] - a[1]);
    plt('cr-coef', [{
      type:'bar', orientation:'h',
      x: sorted.map(d => d[1]),
      y: sorted.map(d => d[0]),
      marker:{ color: sorted.map(d => d[1] > 0 ? '#ef4444' : '#22c55e') }
    }], layout({ margin:{l:130,r:20,t:10,b:30}, showlegend:false,
                 yaxis:{ autorange:'reversed' } }));
  }

  document.getElementById('cr-rules').textContent =
    CR['Decision Tree'].tree_rules || 'No rules available';
}

// ─────────────────────────────────────────────
// INSURANCE
// ─────────────────────────────────────────────
let insModels = [];

function renderInsurance() {
  const ORDER = ['Decision Tree','Random Forest','Gradient Boosting',
                 'Tweedie GLM','GLM Gamma (statsmodels)'];
  const COLS  = ['text-primary','text-success','text-warning','text-info','text-danger'];

  const kpis = document.getElementById('ins-kpis');
  kpis.innerHTML = '';
  insModels = [];
  ORDER.forEach((m, i) => {
    const d = INS[m]; if (!d) return;
    insModels.push(m);
    kpis.innerHTML += `<div class="col-6 col-lg-2">
      <div class="kpi-card text-center">
        <span class="${COLS[i]} fw-bold" style="font-size:.78rem;">${m}</span>
        <div class="kpi-val mt-1" style="font-size:1.4rem;">${d.r2.toFixed(3)}</div>
        <div class="kpi-label">R²</div>
        <div style="font-size:.75rem;color:#64748b;">RMSE ${d.rmse}</div>
      </div></div>`;
  });

  const sel = document.getElementById('ins-model-sel');
  sel.innerHTML = insModels.map(m => `<option value="${m}">${m}</option>`).join('');
  sel.onchange = () => renderInsScatter(sel.value);
  renderInsScatter(insModels[2] || insModels[0]);

  // Feature importance
  const imp = INS['Gradient Boosting'].feat_imp;
  plt('ins-imp', [{
    type:'bar', orientation:'h',
    x: imp.map(d => d.importance),
    y: imp.map(d => d.feature),
    marker:{ color:'#f59e0b' }
  }], layout({ margin:{l:140,r:20,t:10,b:30}, showlegend:false,
               yaxis:{ autorange:'reversed' } }));

  // Metrics comparison
  const maxRMSE = Math.max(...insModels.map(n => INS[n].rmse));
  const maxMAE  = Math.max(...insModels.map(n => INS[n].mae));
  plt('ins-compare', insModels.map((m, i) => ({
    type:'bar', name:m,
    x: ['R²','RMSE (norm)','MAE (norm)'],
    y: [INS[m].r2, INS[m].rmse / maxRMSE, INS[m].mae / maxMAE],
    marker:{ color: COLORS[i] }
  })), layout({ barmode:'group', yaxis:{ title:'Score (normalised for RMSE/MAE)' } }));
}

function renderInsScatter(model) {
  const d = INS[model];
  const allVals = [...d.actual_sample, ...d.pred_sample];
  const mn = Math.min(...allVals), mx = Math.max(...allVals);
  plt('ins-scatter', [
    { type:'scatter', mode:'markers', x:d.actual_sample, y:d.pred_sample,
      marker:{ color:'#3b82f6', opacity:.5, size:5 }, name:'Data' },
    { type:'scatter', mode:'lines', x:[mn,mx], y:[mn,mx],
      line:{ color:'#ef4444', dash:'dash' }, name:'Perfect fit' },
  ], layout({ xaxis:{ title:'Actual Premium' }, yaxis:{ title:'Predicted Premium' } }));
}

// ─────────────────────────────────────────────
// CHURN
// ─────────────────────────────────────────────
function renderChurn() {
  const ORDER = ['Decision Tree','Bagging (RF)','AdaBoost','Gradient Boosting',
                 'Logistic (GLM)','GLM (statsmodels)'];
  const BADGE = ['badge-dt','badge-rf','badge-ab','badge-gb','badge-lr','badge-glm'];

  const kpis = document.getElementById('chn-kpis');
  kpis.innerHTML = '';
  ORDER.forEach((m, i) => {
    const d = CHN[m]; if (!d) return;
    kpis.innerHTML += `<div class="col-6 col-lg-2">
      <div class="kpi-card text-center">
        <span class="model-badge ${BADGE[i]}">${m.split(' ')[0]}</span>
        <div class="kpi-val mt-1">${d.auc}</div>
        <div class="kpi-label">AUC</div>
      </div></div>`;
  });

  // ROC
  const rocTraces = ORDER.map((m, i) => {
    const d = CHN[m]; if (!d) return null;
    return { x:d.fpr, y:d.tpr, mode:'lines', name:`${m} (${d.auc})`,
             line:{ color:COLORS[i], width:2 } };
  }).filter(Boolean);
  rocTraces.push({ x:[0,1], y:[0,1], mode:'lines', name:'Random',
                   line:{ color:'#475569', dash:'dash', width:1 } });
  plt('chn-roc', rocTraces, layout({
    xaxis:{ title:'FPR' }, yaxis:{ title:'TPR' }
  }));

  // Importance
  const imp = CHN['Gradient Boosting'].feat_imp;
  plt('chn-imp', [{
    type:'bar', orientation:'h',
    x: imp.map(d => d.importance),
    y: imp.map(d => d.feature),
    marker:{ color:'#a78bfa' }
  }], layout({ margin:{l:140,r:20,t:10,b:30}, showlegend:false,
               yaxis:{ autorange:'reversed' } }));

  // Satisfaction bins
  plt('chn-sat', [{
    type:'bar', x:CHN_SAT_X, y:CHN_SAT_Y,
    marker:{ color: CHN_SAT_Y.map(v => v>0.3?'#ef4444':v>0.2?'#f59e0b':'#22c55e') }
  }], layout({ xaxis:{ title:'Satisfaction Score Bin' }, yaxis:{ title:'Churn Rate' }, showlegend:false }));

  // Tenure bins
  plt('chn-tenure', [{
    type:'bar', x:CHN_TEN_X, y:CHN_TEN_Y,
    marker:{ color:'#6366f1' }
  }], layout({ xaxis:{ title:'Tenure Band (months)' }, yaxis:{ title:'Churn Rate' }, showlegend:false }));
}

// ─────────────────────────────────────────────
// FRAUD
// ─────────────────────────────────────────────
function renderFraud() {
  const s = FRD.scatter;
  const fi = s.label.map((v, i) => v===1 ? i : -1).filter(i => i >= 0);
  const li = s.label.map((v, i) => v===0 ? i : -1).filter(i => i >= 0);

  plt('frd-scatter', [
    { type:'scatter', mode:'markers', name:'Legitimate',
      x: li.map(i => s.amount[i]),   y: li.map(i => s.velocity[i]),
      marker:{ color:'#3b82f6', opacity:.4, size:5 } },
    { type:'scatter', mode:'markers', name:'Fraud',
      x: fi.map(i => s.amount[i]),   y: fi.map(i => s.velocity[i]),
      marker:{ color:'#ef4444', opacity:.8, size:7, symbol:'x' } },
  ], layout({ xaxis:{ title:'Amount ($)', type:'log' }, yaxis:{ title:'Velocity (tx/hr)' } }));

  plt('frd-roc', [
    { x:FRD.decision_tree.fpr,    y:FRD.decision_tree.tpr,    mode:'lines',
      name:`DT (${FRD.decision_tree.auc})`,     line:{ color:'#60a5fa' } },
    { x:FRD.gradient_boosting.fpr, y:FRD.gradient_boosting.tpr, mode:'lines',
      name:`GBM (${FRD.gradient_boosting.auc})`, line:{ color:'#22c55e', width:2 } },
    { x:[0,1], y:[0,1], mode:'lines', name:'Random', line:{ color:'#475569', dash:'dash', width:1 } },
  ], layout({ xaxis:{ title:'FPR' }, yaxis:{ title:'TPR' } }));

  const imp = FRD.gradient_boosting.feat_imp;
  plt('frd-imp', [{
    type:'bar', orientation:'h',
    x: imp.map(d => d.importance),
    y: imp.map(d => d.feature),
    marker:{ color:'#f59e0b' }
  }], layout({ margin:{l:120,r:20,t:10,b:30}, showlegend:false,
               yaxis:{ autorange:'reversed' } }));

  document.getElementById('frd-rules').textContent = FRD.decision_tree.tree_rules;
}

// ─────────────────────────────────────────────
// MODEL COMPARISON
// ─────────────────────────────────────────────
function renderCompare() {
  const models = ['Decision Tree','Bagging (RF)','AdaBoost','Gradient Boosting','Logistic (GLM)'];
  const tasks  = ['Credit Risk','Churn'];
  const data   = [CR, CHN];

  const z = models.map(m => data.map(d => d[m] ? d[m].auc : null));
  plt('cmp-auc', [{
    type:'heatmap', z, x:tasks, y:models,
    colorscale: [[0,'#1e293b'],[0.5,'#2563eb'],[1,'#22c55e']],
    text: z.map(r => r.map(v => v ? v.toString() : '')),
    texttemplate:'%{text}', zmin:0.5, zmax:1.0,
  }], layout({ margin:{l:160,r:20,t:20,b:60} }));

  plt('cmp-acc', models.map((m, i) => ({
    type:'bar', name:m, x:tasks,
    y: data.map(d => d[m] ? d[m].accuracy : 0),
    marker:{ color: COLORS[i] }
  })), layout({ barmode:'group', yaxis:{ title:'Accuracy' } }));

  plt('cmp-f1', models.map((m, i) => ({
    type:'bar', name:m, x:tasks,
    y: data.map(d => d[m] ? d[m].f1 : 0),
    marker:{ color: COLORS[i] }
  })), layout({ barmode:'group', yaxis:{ title:'F1 Score' } }));
}

// ─────────────────────────────────────────────
// DATA EXPLORER
// ─────────────────────────────────────────────
let currentKey = 'credit';

function renderData() {
  currentKey = document.getElementById('data-sel').value;
  const rows = FULL[currentKey];
  if (!rows || !rows.length) return;
  const cols = Object.keys(rows[0]);
  const tbl  = document.getElementById('data-table');
  tbl.innerHTML = `<thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead>
    <tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${
      typeof r[c]==='number' ? (Number.isInteger(r[c]) ? r[c] : r[c].toFixed(3)) : r[c]
    }</td>`).join('')}</tr>`).join('')}</tbody>`;

  const sel = document.getElementById('dist-feat-sel');
  sel.innerHTML = cols.map(c => `<option value="${c}">${c}</option>`).join('');
  renderDist();
}

function renderDist() {
  const feat = document.getElementById('dist-feat-sel').value;
  const dist = DIST[currentKey];
  if (!dist || !dist[feat]) return;

  plt('data-dist', [{
    type:'histogram', x:dist[feat],
    marker:{ color:'#3b82f6', opacity:.8 }, nbinsx:30
  }], layout({ xaxis:{ title:feat }, yaxis:{ title:'Count' }, showlegend:false }));

  const targets = { credit:'default', insurance:'premium', churn:'churned', fraud:'is_fraud' };
  const tgt = targets[currentKey];
  const vals = dist[feat];
  if (dist[tgt] && feat !== tgt) {
    const bins = 5, mn = Math.min(...vals), mx = Math.max(...vals), step = (mx-mn)/bins;
    const bx=[], by=[];
    for (let i=0; i<bins; i++) {
      const lo=mn+i*step, hi=mn+(i+1)*step;
      const idx = vals.map((v,j)=>v>=lo&&v<hi?j:-1).filter(j=>j>=0);
      if (!idx.length) continue;
      const rate = idx.reduce((s,j)=>s+(dist[tgt][j]||0),0)/idx.length;
      bx.push(`${lo.toFixed(1)}-${hi.toFixed(1)}`);
      by.push(+rate.toFixed(4));
    }
    plt('data-target', [{
      type:'bar', x:bx, y:by,
      marker:{ color: by.map(v=>v>0.3?'#ef4444':v>0.15?'#f59e0b':'#22c55e') }
    }], layout({ xaxis:{title:feat}, yaxis:{title:`${tgt} Rate`}, showlegend:false }));
  }
}

// ─────────────────────────────────────────────
// PAGE ROUTER
// ─────────────────────────────────────────────
const rendered = {};

function renderPage(p) {
  if (rendered[p]) return;
  rendered[p] = true;
  if (p === 'overview')  renderOverview();
  if (p === 'credit')    renderCredit();
  if (p === 'insurance') renderInsurance();
  if (p === 'churn')     renderChurn();
  if (p === 'fraud')     renderFraud();
  if (p === 'compare')   renderCompare();
  if (p === 'data')      renderData();
}