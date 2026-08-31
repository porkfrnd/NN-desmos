// Main controller: equation input (Desmos-like) + presets, charts, training.
// Upgraded: SR Omega, embeddings (Fourier/Chebyshev), weight decay, domain bounds, tuning presets.

function $(sel, root) { return (root || document).querySelector(sel); }
function $all(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }

const App = {
  loopPromise: null,

  init() {
    if (typeof tf === 'undefined') {
      this.showToast('TensorFlow.js failed to load — check internet and refresh.', 'error');
      console.error('tf undefined — CDN failed');
    }
    if (typeof Chart === 'undefined') {
      this.showToast('Chart.js failed to load — charts will not render.', 'error');
      console.error('Chart undefined — CDN failed');
    }
    if (typeof Equation === 'undefined') {
      this.showToast('Equation parser not loaded.', 'error');
    }
    // migrate legacy fourierFeatures -> embedding
    try {
      const m = Store.get('model');
      if (m.fourierFeatures && (!m.embedding || m.embedding === 'none')) {
        Store.set({ model: { ...m, embedding: 'fourier' } });
      }
    } catch (_) {}
    try { this.setupCharts(); } catch (e) { console.error('Charts init failed', e); }
    this.setupPresets();
    this.setupTuningPresets();
    this.setupEquation();
    this.setupArchitecture();
    this.setupHyperparams();
    this.setupDomain();
    this.setupControls();
    this.setupStatus();
    this.bindDataSubscriptions();
    // sync UI from store after all setups
    this.syncAllUIFromStore();
    const def = PRESET_DEFS['sine'];
    const initEq = def && def.equation ? def.equation : 'sin(2*pi*x)';
    try { this.applyEquation(initEq, 'sine'); } catch (e) { console.error(e); this.showToast(e.message, 'error'); }
    let resizeTimer;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => { try { Charts.resize(); } catch (_) {} }, 80);
    }, { passive: true });
  },

  syncAllUIFromStore() {
    const m = Store.get('model');
    const t = Store.get('training');
    const d = Store.get('domain');
    // activation
    const actSel = $('#activation'); if (actSel) actSel.value = m.activation;
    this.updateOmegaVisibility();
    const omega = $('#omega0'); if (omega) { omega.value = m.omega0 ?? 30; const v = $('#omega0Val'); if (v) v.textContent = String(m.omega0 ?? 30); }
    // embedding
    const emb = $('#embedding'); if (emb) emb.value = m.embedding || 'none';
    this.updateEmbeddingVisibility();
    const fn = $('#fourierN'); if (fn) { fn.value = m.fourierN ?? 3; const v=$('#fourierNVal'); if(v) v.textContent = String(m.fourierN ?? 3); }
    const fs = $('#fourierSigma'); if (fs) { fs.value = m.fourierSigma ?? 1; const v=$('#fourierSigmaVal'); if(v) v.textContent = Number(m.fourierSigma ?? 1).toFixed(1); }
    const cd = $('#chebyshevDegree'); if (cd) { cd.value = m.chebyshevDegree ?? 6; const v=$('#chebyshevDegreeVal'); if(v) v.textContent = String(m.chebyshevDegree ?? 6); }
    // training
    const lr = $('#learningRate'); if (lr) {
      const exp = Math.log10(t.learningRate);
      lr.value = String(exp);
      const v=$('#learningRateVal'); if(v) v.textContent = Number(t.learningRate).toExponential(1);
    }
    const wd = $('#weightDecay'); if (wd) {
      const val = t.weightDecay ?? 0;
      let sliderVal = 0;
      if (val > 0) sliderVal = Math.round(1 + (Math.log10(val) + 4) / 2 * 99);
      sliderVal = Math.max(0, Math.min(100, sliderVal));
      wd.value = String(sliderVal);
      const v=$('#weightDecayVal'); if(v) v.textContent = val === 0 ? '0' : Number(val).toExponential(1);
    }
    const hl = $('#hiddenLayers'); if (hl) { hl.value = String(m.hiddenLayers); const v=$('#hiddenLayersVal'); if(v) v.textContent = String(m.hiddenLayers); }
    const nl = $('#neuronsPerLayer'); if (nl) { nl.value = String(m.neuronsPerLayer); const v=$('#neuronsPerLayerVal'); if(v) v.textContent = String(m.neuronsPerLayer); }
    // domain
    const trMin = $('#trainMin'), trMax = $('#trainMax'), evMin = $('#evalMin'), evMax = $('#evalMax');
    if (trMin) trMin.value = String(d.trainMin);
    if (trMax) trMax.value = String(d.trainMax);
    if (evMin) evMin.value = String(d.evalMin);
    if (evMax) evMax.value = String(d.evalMax);
    const me = $('#maxEpochs'); if (me) { me.value = String(t.maxEpochs); const v=$('#maxEpochsVal'); if(v) v.textContent = String(t.maxEpochs); }
    const opt = $('#optimizer'); if (opt) opt.value = t.optimizer;
  },

  setupCharts() {
    Charts.init(document.getElementById('predChart'), document.getElementById('lossChart'));
    window.addEventListener('resize', () => Charts.resize());
  },

  setupPresets() {
    const grid = $('#presetGrid');
    if (!grid) return;
    grid.innerHTML = '';
    const presetIds = ['sine', 'square', 'damped', 'composite'];
    presetIds.forEach((id) => {
      const btn = document.createElement('button');
      btn.className = 'preset-btn';
      btn.dataset.preset = id;
      const def = PRESET_DEFS[id];
      btn.innerHTML = '<span class="preset-name">' + def.name + '</span>' +
                      '<span class="preset-formula">' + def.formula + '</span>';
      btn.addEventListener('click', () => this.loadPreset(id));
      grid.appendChild(btn);
    });
  },

  setupTuningPresets() {
    $all('[data-tuning]').forEach(btn => {
      btn.addEventListener('click', () => {
        const key = btn.dataset.tuning;
        const preset = TUNING_PRESETS[key];
        if (!preset) return;
        const m = Store.get('model');
        const t = Store.get('training');
        Store.set({
          model: { ...m, ...preset.config.model },
          training: { ...t, ...preset.config.training },
        });
        this.syncAllUIFromStore();
        // rebuild with new config, keep data
        this.resetWeights();
        this.showToast(preset.name + ' preset applied', 'success');
      });
    });
  },

  loadPreset(id) {
    const def = PRESET_DEFS[id];
    if (!def) return;
    const eq = def.equation || '';
    const input = $('#equationInput');
    if (input) input.value = eq;
    this.applyEquation(eq, id);
  },

  setupEquation() {
    const input = $('#equationInput');
    const btn = $('#equationApply');
    const errBox = $('#equationError');
    if (!input) return;
    const apply = () => {
      const raw = input.value.trim();
      if (!raw) { this.showError('Please type an equation, e.g.  x^2 + 6*x'); return; }
      try { this.applyEquation(raw, null); this.clearError(); } catch (e) { this.showError(e.message); }
    };
    btn && btn.addEventListener('click', apply);
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); apply(); } });
    let t;
    input.addEventListener('input', () => {
      clearTimeout(t);
      input.classList.remove('is-error');
      if (errBox) { errBox.hidden = true; errBox.textContent = ''; }
      t = setTimeout(() => {
        const v = input.value.trim();
        if (!v) return;
        try {
          const dom = Store.get('domain');
          const parsed = Equation.sampleString(v, 100, dom.trainMin, dom.trainMax);
          Store.set({ data: { source: 'equation', presetId: null, equation: parsed.compiled.src, xs: parsed.xs, ys: parsed.ys } });
          Store.set({ lossHistory: [], predictions: { xs: [], ys: [] } });
          Training.buildModel(); Training.setDataTensors(); Training.resetEpochCounter();
          this.setStatus('idle');
          this.renderAll();
          $all('#presetGrid .preset-btn').forEach(b => b.classList.remove('active'));
          this.clearError();
        } catch (_) {}
      }, 600);
    });
  },

  applyEquation(raw, presetId) {
    const input = $('#equationInput');
    const eqStr = raw && String(raw).trim() ? String(raw).trim() : (input ? input.value.trim() : '');
    if (!eqStr) throw new Error('Empty equation. Try  x^2 + 6*x');
    const dom = Store.get('domain');
    let parsed;
    try { parsed = Equation.sampleString(eqStr, 100, dom.trainMin, dom.trainMax); } catch (e) { this.showError(e.message); throw e; }
    this.clearError();
    if (input && document.activeElement !== input) input.value = parsed.compiled.src;
    Training.setStopRequested(true);
    Store.set({ data: { source: presetId ? 'preset' : 'equation', presetId: presetId || null, equation: parsed.compiled.src, xs: parsed.xs, ys: parsed.ys } });
    Store.set({ lossHistory: [], predictions: { xs: [], ys: [] } });
    $all('#presetGrid .preset-btn').forEach(b => b.classList.toggle('active', b.dataset.preset === presetId));
    Training.buildModel();
    Training.setDataTensors();
    Training.resetEpochCounter();
    this.setStatus('idle');
    this.renderAll();
  },

  showError(msg) {
    const box = $('#equationError');
    const input = $('#equationInput');
    if (box) { box.textContent = msg; box.hidden = false; }
    if (input) input.classList.add('is-error');
  },
  clearError() {
    const box = $('#equationError');
    const input = $('#equationInput');
    if (box) { box.hidden = true; box.textContent = ''; }
    if (input) input.classList.remove('is-error');
  },

  updateOmegaVisibility() {
    const row = $('#omegaRow');
    const act = Store.get('model').activation;
    if (!row) return;
    row.hidden = act !== 'sine';
  },

  updateEmbeddingVisibility() {
    const emb = Store.get('model').embedding || 'none';
    const fr = $('#fourierNRow'), fs = $('#fourierSigmaRow'), cr = $('#chebyshevRow');
    if (fr) fr.hidden = emb !== 'fourier';
    if (fs) fs.hidden = emb !== 'fourier';
    if (cr) cr.hidden = emb !== 'chebyshev';
  },

  setupArchitecture() {
    const layersRange = $('#hiddenLayers');
    const neuronsRange = $('#neuronsPerLayer');
    const actSelect = $('#activation');
    const omega = $('#omega0');
    const embedding = $('#embedding');
    const fourierN = $('#fourierN');
    const fourierSigma = $('#fourierSigma');
    const cheb = $('#chebyshevDegree');
    if (!layersRange) return;
    const updHL = () => { const v=$('#hiddenLayersVal'); if(v) v.textContent = layersRange.value; };
    const updNL = () => { const v=$('#neuronsPerLayerVal'); if(v) v.textContent = neuronsRange.value; };
    const updOmega = () => { const v=$('#omega0Val'); if(v) v.textContent = String(omega.value); };
    const updFN = () => { const v=$('#fourierNVal'); if(v) v.textContent = String(fourierN.value); };
    const updFS = () => { const v=$('#fourierSigmaVal'); if(v) v.textContent = Number(fourierSigma.value).toFixed(1); };
    const updCh = () => { const v=$('#chebyshevDegreeVal'); if(v) v.textContent = String(cheb.value); };
    layersRange.addEventListener('input', updHL);
    neuronsRange.addEventListener('input', updNL);
    if (omega) omega.addEventListener('input', updOmega);
    if (fourierN) fourierN.addEventListener('input', updFN);
    if (fourierSigma) fourierSigma.addEventListener('input', updFS);
    if (cheb) cheb.addEventListener('input', updCh);

    const apply = () => {
      const m = Store.get('model');
      Store.set({
        model: {
          ...m,
          hiddenLayers: parseInt(layersRange.value, 10),
          neuronsPerLayer: parseInt(neuronsRange.value, 10),
          activation: actSelect ? actSelect.value : m.activation,
          embedding: embedding ? embedding.value : (m.embedding || 'none'),
          fourierN: fourierN ? parseInt(fourierN.value, 10) : (m.fourierN ?? 3),
          fourierSigma: fourierSigma ? parseFloat(fourierSigma.value) : (m.fourierSigma ?? 1),
          chebyshevDegree: cheb ? parseInt(cheb.value, 10) : (m.chebyshevDegree ?? 6),
          omega0: omega ? parseFloat(omega.value) : (m.omega0 ?? 30),
          fourierFeatures: embedding ? embedding.value === 'fourier' : !!m.fourierFeatures,
        },
      });
      this.updateOmegaVisibility();
      this.updateEmbeddingVisibility();
      this.resetWeights();
    };
    layersRange.addEventListener('change', apply);
    neuronsRange.addEventListener('change', apply);
    if (actSelect) actSelect.addEventListener('change', () => { this.updateOmegaVisibility(); apply(); });
    if (omega) omega.addEventListener('change', apply);
    if (embedding) embedding.addEventListener('change', () => { this.updateEmbeddingVisibility(); apply(); });
    if (fourierN) fourierN.addEventListener('change', apply);
    if (fourierSigma) fourierSigma.addEventListener('change', apply);
    if (cheb) cheb.addEventListener('change', apply);
    updHL(); updNL(); if(omega) updOmega(); if(fourierN) updFN(); if(fourierSigma) updFS(); if(cheb) updCh();
    this.updateOmegaVisibility();
    this.updateEmbeddingVisibility();
  },

  resetWeights() {
    Training.setStopRequested(true);
    Training.buildModel();
    Training.setDataTensors();
    Training.resetEpochCounter();
    Store.set({ lossHistory: [], predictions: { xs: [], ys: [] } });
    this.renderAll();
    this.setStatus('idle');
  },

  setupHyperparams() {
    const lr = $('#learningRate');
    const wd = $('#weightDecay');
    const opt = $('#optimizer');
    const epochsRange = $('#maxEpochs');
    if (!lr) return;
    const updLr = () => {
      const v = Math.pow(10, parseFloat(lr.value));
      const el=$('#learningRateVal'); if(el) el.textContent = v.toExponential(1);
    };
    const updWd = () => {
      const raw = parseInt(wd ? wd.value : '0', 10);
      let val = 0;
      if (raw > 0) val = Math.pow(10, -4 + (raw - 1) / 99 * 2);
      const el=$('#weightDecayVal'); if(el) el.textContent = val === 0 ? '0' : Number(val).toExponential(1);
    };
    const updEp = () => { const el=$('#maxEpochsVal'); if(el) el.textContent = epochsRange.value; };
    lr.addEventListener('input', updLr);
    if (wd) wd.addEventListener('input', updWd);
    if (epochsRange) epochsRange.addEventListener('input', updEp);

    lr.addEventListener('change', () => {
      Store.set({ training: { ...Store.get('training'), learningRate: Math.pow(10, parseFloat(lr.value)) } });
      if (Training.modelExists && Training.epochCounter > 0) {
        if (typeof Training.rebuildOptimizer === 'function') Training.rebuildOptimizer();
        else { Training.buildModel(); Training.setDataTensors(); }
        this.showToast('Learning rate will apply on next Start', 'success');
      } else { Training.buildModel(); Training.setDataTensors(); }
    });
    if (wd) wd.addEventListener('change', () => {
      let raw = parseInt(wd.value, 10);
      let val = 0;
      if (raw > 0) val = Math.pow(10, -4 + (raw - 1) / 99 * 2);
      Store.set({ training: { ...Store.get('training'), weightDecay: val } });
      this.showToast('Weight decay ' + (val===0?'off':val.toExponential(1)), 'success');
    });
    if (opt) opt.addEventListener('change', () => {
      Store.set({ training: { ...Store.get('training'), optimizer: opt.value } });
      if (Training.modelExists && Training.epochCounter > 0) {
        if (typeof Training.rebuildOptimizer === 'function') Training.rebuildOptimizer();
        else { Training.buildModel(); Training.setDataTensors(); }
        this.showToast('Optimizer will apply on next Start', 'success');
      } else { Training.buildModel(); Training.setDataTensors(); }
    });
    if (epochsRange) epochsRange.addEventListener('change', () => {
      Store.set({ training: { ...Store.get('training'), maxEpochs: parseInt(epochsRange.value, 10) } });
    });
    updLr(); if(wd) updWd(); if(epochsRange) updEp();
  },

  setupDomain() {
    const trMin = $('#trainMin'), trMax = $('#trainMax'), evMin = $('#evalMin'), evMax = $('#evalMax');
    if (!trMin) return;
    const apply = () => {
      let tMin = parseFloat(trMin.value), tMax = parseFloat(trMax.value);
      let eMin = parseFloat(evMin.value), eMax = parseFloat(evMax.value);
      if (!isFinite(tMin) || !isFinite(tMax) || tMin >= tMax) { this.showToast('Training range: min must be < max', 'warning'); return; }
      if (!isFinite(eMin) || !isFinite(eMax) || eMin >= eMax) { this.showToast('Eval range: min must be < max', 'warning'); return; }
      Store.set({ domain: { trainMin: tMin, trainMax: tMax, evalMin: eMin, evalMax: eMax } });
      // resample current equation/preset over new train range and update chart view to new eval range
      const data = Store.get('data');
      let newData = null;
      if (data.presetId && PRESET_DEFS[data.presetId]) {
        newData = samplePreset(data.presetId, 100, tMin, tMax);
        if (newData) newData = { xs: newData.xs, ys: clipYs(newData.ys) };
      } else if (data.equation) {
        try { const p = Equation.sampleString(data.equation, 100, tMin, tMax); newData = { xs: p.xs, ys: p.ys }; } catch (e) { this.showToast(e.message,'error'); return; }
      }
      if (newData) {
        Store.set({ data: { ...data, xs: newData.xs, ys: newData.ys } });
        Store.set({ lossHistory: [], predictions: { xs: [], ys: [] } });
        Training.buildModel(); Training.setDataTensors(); Training.resetEpochCounter();
        this.setStatus('idle');
        try { Charts.setDomainAndReset(tMin, tMax, eMin, eMax); } catch (_) {}
        this.renderAll();
      } else {
        try { Charts.setDomainAndReset(tMin, tMax, eMin, eMax); } catch (_) {}
        this.renderAll();
      }
    };
    [trMin, trMax, evMin, evMax].forEach(el => el && el.addEventListener('change', apply));
  },

  setupControls() {
    const startBtn = $('#btnStart');
    const pauseBtn = $('#btnPause');
    const stepBtn = $('#btnStep');
    const resetBtn = $('#btnResetWeights');
    const exportBtn = $('#btnExport');
    if (startBtn) startBtn.addEventListener('click', () => this.startTraining());
    if (pauseBtn) pauseBtn.addEventListener('click', () => this.togglePause());
    if (stepBtn) stepBtn.addEventListener('click', () => this.runStep());
    if (resetBtn) resetBtn.addEventListener('click', () => this.resetWeights());
    if (exportBtn) exportBtn.addEventListener('click', () => this.exportWeights());
  },

  async startTraining() {
    const data = Store.get('data');
    if (!data.xs || data.xs.length === 0) { this.showToast('Type an equation and press Plot first', 'warning'); return; }
    if (Store.get('run').status === 'training') return;
    if (!Training.modelExists) { Training.buildModel(); Training.setDataTensors(); }
    Training.setPaused(false);
    Training.setStopRequested(false);
    this.setStatus('training');
    this.runLoop();
  },

  async runLoop() {
    const cfg = Store.get('training');
    const target = cfg.maxEpochs;
    while (!Training.stopRequested && !Training.isPaused) {
      if (Training.epochCounter >= target) {
        this.setStatus('idle');
        this.showToast('Reached ' + target + ' epochs', 'success');
        return;
      }
      const remaining = target - Training.epochCounter;
      const chunk = Math.min(remaining, 50);
      const status = await Training.runEpochs(chunk);
      if (status === 'nan' || status === 'diverged') { this.handleRunEnd(status); return; }
      if (status === 'stopped') { this.setStatus('idle'); return; }
      if (Training.isPaused) { this.setStatus('paused'); return; }
      if (status === 'done' && Training.epochCounter >= target) { this.setStatus('idle'); return; }
      await tf.nextFrame();
    }
    if (Training.stopRequested) this.setStatus('idle');
  },

  async togglePause() {
    const run = Store.get('run');
    if (run.status === 'training') { Training.setPaused(true); this.setStatus('paused'); }
    else if (run.status === 'paused' || run.status === 'error') { Training.setPaused(false); this.setStatus('training'); this.runLoop(); }
  },

  async runStep() {
    const data = Store.get('data');
    if (!data.xs || data.xs.length === 0) { this.showToast('Plot an equation first', 'warning'); return; }
    Training.setStopRequested(true);
    await new Promise(r => setTimeout(r, 60));
    Training.setStopRequested(false);
    if (!Training.modelExists) { Training.buildModel(); Training.setDataTensors(); }
    const wasPaused = Training.isPaused;
    const wasTraining = Store.get('run').status === 'training';
    Training.setPaused(false); this.setStatus('training');
    const status = await Training.runEpochs(10);
    if (status === 'nan' || status === 'diverged') { this.handleRunEnd(status); return; }
    Training.setPaused(wasPaused || !wasTraining);
    this.setStatus(wasPaused || !wasTraining ? 'paused' : 'idle');
  },

  exportWeights() {
    if (!Training.modelExists) { this.showToast('Train a model first — then export.', 'warning'); return; }
    const payload = Training.exportWeights();
    if (!payload) { this.showToast('No weights to export.', 'warning'); return; }
    const json = JSON.stringify(payload, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const safeEq = (Store.get('data').equation || 'model').replace(/[^a-z0-9_-]+/gi, '_').slice(0, 40);
    a.href = url; a.download = `nn-desmos-${safeEq}-${Date.now()}.json`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    this.showToast('Weights downloaded ✓', 'success');
  },

  handleRunEnd(status) {
    if (status === 'nan' || status === 'diverged') {
      this.showToast(status === 'nan' ? 'NaN loss — auto-paused' : 'Loss diverged — auto-paused', 'error');
      this.setStatus('error');
    } else { this.setStatus(Training.isPaused ? 'paused' : 'idle'); }
  },

  setupStatus() {
    this.statusEl = $('#statusDot');
    this.epochEl = $('#epochVal');
    this.lossEl = $('#lossVal');
  },

  setStatus(status) {
    const el = this.statusEl; if (!el) return;
    el.classList.remove('training','paused','error');
    const label = $('#statusLabel');
    if (status === 'training') { el.classList.add('training'); if(label) label.textContent='Training'; }
    else if (status === 'paused') { el.classList.add('paused'); if(label) label.textContent='Paused'; }
    else if (status === 'error') { el.classList.add('error'); if(label) label.textContent='Error'; }
    else { if(label) label.textContent='Idle'; }
  },

  bindDataSubscriptions() {
    Store.subscribe('data', () => this.renderAll());
    Store.subscribe('predictions', () => this.updatePredictionOnly());
    Store.subscribe('lossHistory', () => Charts.setLoss(Store.get('lossHistory')));
    Store.subscribe('run', (run) => {
      if (this.epochEl) this.epochEl.textContent = run.epoch ?? 0;
      if (this.lossEl) this.lossEl.textContent = run.loss != null ? Number(run.loss).toExponential(2) : '—';
    });
    // keep chart view in sync when domain changes elsewhere
    Store.subscribe('domain', (dom) => {
      try { Charts.setDomainAndReset(dom.trainMin, dom.trainMax, dom.evalMin, dom.evalMax); } catch (_) {}
    });
  },

  renderAll() {
    const data = Store.get('data');
    const pred = Store.get('predictions');
    if (!data.xs || !data.xs.length) return;
    const gtXs = data.xs, gtYs = data.ys;
    let predXs = pred.xs || [], predYs = pred.ys || [];
    Charts.setLoss(Store.get('lossHistory'));
    if (!predYs.length && Training.modelExists) {
      const dom = Store.get('domain');
      const evalMin = dom ? dom.evalMin : -2, evalMax = dom ? dom.evalMax : 2;
      const xs = [];
      for (let i=0;i<140;i++) xs.push(evalMin + (evalMax - evalMin) * i / 139);
      Training.predictXs(xs).then((ys) => {
        if (ys) Store.set({ predictions: { xs, ys } });
        else {
          // need gt over eval range too
          const g = this.sampleTruthOverEval();
          Charts.setPrediction(g.xs, g.ys, [], []);
        }
      });
      const g = this.sampleTruthOverEval();
      Charts.setPrediction(g.xs, g.ys, [], []);
      return;
    }
    // gt may be train range only; for display we want eval range truth
    const g = this.sampleTruthOverEval();
    Charts.setPrediction(g.xs, g.ys, predXs, predYs);
  },

  sampleTruthOverEval() {
    const data = Store.get('data');
    const dom = Store.get('domain');
    const evalMin = dom.evalMin, evalMax = dom.evalMax;
    if (data.presetId && PRESET_DEFS[data.presetId]) {
      const fn = PRESET_DEFS[data.presetId].fn;
      const xs=[], ys=[];
      for(let i=0;i<140;i++){ const x=evalMin+(evalMax-evalMin)*i/139; xs.push(x); ys.push(fn(x)); }
      return { xs, ys: clipYs(ys) };
    }
    if (data.equation) {
      try {
        const cmp = Equation.compile(data.equation);
        const s = Equation.sample(cmp, 140, evalMin, evalMax);
        return s;
      } catch (_) { return { xs: data.xs, ys: data.ys }; }
    }
    return { xs: data.xs, ys: data.ys };
  },

  updatePredictionOnly() {
    const data = Store.get('data');
    const pred = Store.get('predictions');
    if (!data.xs || !pred.xs) return;
    const g = this.sampleTruthOverEval();
    Charts.setPrediction(g.xs, g.ys, pred.xs, pred.ys);
    Charts.setLoss(Store.get('lossHistory'));
  },

  showToast(msg, type='warning') {
    const cont = $('#toastContainer'); if (!cont) return;
    const t = document.createElement('div');
    t.className = 'toast toast-' + (type || 'warning');
    t.textContent = msg; cont.appendChild(t);
    setTimeout(() => t.remove(), 4000);
  },
};

function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }
document.addEventListener('DOMContentLoaded', () => App.init());
