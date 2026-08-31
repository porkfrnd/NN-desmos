// Main controller: equation input (Desmos-like) + presets, charts, training.

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
    try { this.setupCharts(); } catch (e) { console.error('Charts init failed', e); }
    this.setupPresets();
    this.setupEquation();
    this.setupArchitecture();
    this.setupHyperparams();
    this.setupControls();
    this.setupStatus();
    this.bindDataSubscriptions();
    // initial equation: use Sine preset equation
    const def = PRESET_DEFS['sine'];
    const initEq = def && def.equation ? def.equation : 'sin(2*pi*x)';
    try { this.applyEquation(initEq, 'sine'); } catch (e) { console.error(e); this.showToast(e.message, 'error'); }
    let resizeTimer;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => { try { Charts.resize(); } catch (_) {} }, 80);
    }, { passive: true });
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

  loadPreset(id) {
    const def = PRESET_DEFS[id];
    if (!def) return;
    const eq = def.equation || '';
    // update input without triggering duplicate apply
    const input = $('#equationInput');
    if (input) input.value = eq;
    this.applyEquation(eq, id);
  },

  // ---- Equation ----
  setupEquation() {
    const input = $('#equationInput');
    const btn = $('#equationApply');
    const errBox = $('#equationError');
    if (!input) return;

    const apply = () => {
      const raw = input.value.trim();
      if (!raw) {
        this.showError('Please type an equation, e.g.  x^2 + 6*x');
        return;
      }
      try {
        this.applyEquation(raw, null);
        this.clearError();
      } catch (e) {
        this.showError(e.message);
      }
    };

    btn && btn.addEventListener('click', apply);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); apply(); }
    });
    // live preview debounced, but only if already valid (don't spam errors while typing)
    let t;
    input.addEventListener('input', () => {
      clearTimeout(t);
      input.classList.remove('is-error');
      if (errBox) { errBox.hidden = true; errBox.textContent = ''; }
      t = setTimeout(() => {
        const v = input.value.trim();
        if (!v) return;
        try {
          // try to parse; if ok, preview without resetting training
          const parsed = Equation.sampleString(v, 100);
          // show preview ground truth only (don't rebuild model until user hits Plot / preset)
          // we plot immediately so they see the curve
          Store.set({ data: { source: 'equation', presetId: null, equation: parsed.compiled.src, xs: parsed.xs, ys: parsed.ys } });
          Store.set({ lossHistory: [], predictions: { xs: [], ys: [] } });
          Training.buildModel(); Training.setDataTensors(); Training.resetEpochCounter();
          this.setStatus('idle');
          this.renderAll();
          // highlight no preset as active
          $all('#presetGrid .preset-btn').forEach(b => b.classList.remove('active'));
          this.clearError();
        } catch (_) { /* silent while typing */ }
      }, 600);
    });
  },

  applyEquation(raw, presetId) {
    const input = $('#equationInput');
    const eqStr = raw && String(raw).trim() ? String(raw).trim() : (input ? input.value.trim() : '');
    if (!eqStr) throw new Error('Empty equation. Try  x^2 + 6*x');
    let parsed;
    try {
      parsed = Equation.sampleString(eqStr, 100);
    } catch (e) {
      this.showError(e.message);
      throw e;
    }
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

  // ---- Architecture ----
  setupArchitecture() {
    const layersRange = $('#hiddenLayers');
    const neuronsRange = $('#neuronsPerLayer');
    const actSelect = $('#activation');
    const fourierToggle = $('#fourierFeatures');
    if (!layersRange) return;
    const updateLayersLabel = () => { $('#hiddenLayersVal').textContent = layersRange.value; };
    const updateNeuronsLabel = () => { $('#neuronsPerLayerVal').textContent = neuronsRange.value; };
    layersRange.addEventListener('input', updateLayersLabel);
    neuronsRange.addEventListener('input', updateNeuronsLabel);
    layersRange.addEventListener('change', () => this.applyArchFromDom(layersRange, neuronsRange, actSelect, fourierToggle));
    neuronsRange.addEventListener('change', () => this.applyArchFromDom(layersRange, neuronsRange, actSelect, fourierToggle));
    actSelect.addEventListener('change', () => this.applyArchFromDom(layersRange, neuronsRange, actSelect, fourierToggle));
    fourierToggle.addEventListener('click', () => {
      const on = !fourierToggle.classList.contains('active');
      fourierToggle.classList.toggle('active', on);
      fourierToggle.setAttribute('aria-checked', on ? 'true' : 'false');
      this.applyArchFromDom(layersRange, neuronsRange, actSelect, fourierToggle);
    });
    updateLayersLabel(); updateNeuronsLabel();
  },

  applyArchFromDom(layersRange, neuronsRange, actSelect, fourierToggle) {
    Store.set({
      model: {
        hiddenLayers: parseInt(layersRange.value, 10),
        neuronsPerLayer: parseInt(neuronsRange.value, 10),
        activation: actSelect.value,
        fourierFeatures: fourierToggle.classList.contains('active'),
      },
    });
    this.resetWeights();
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

  // ---- Hyperparams ----
  setupHyperparams() {
    const lr = $('#learningRate');
    const opt = $('#optimizer');
    const epochsRange = $('#maxEpochs');
    if (!lr) return;
    const updateLrLabel = () => {
      const v = Math.pow(10, parseFloat(lr.value));
      $('#learningRateVal').textContent = v.toExponential(1);
    };
    const updateEpochs = () => { $('#maxEpochsVal').textContent = epochsRange.value; };
    lr.addEventListener('input', updateLrLabel);
    epochsRange.addEventListener('input', updateEpochs);
    lr.addEventListener('change', () => {
      Store.set({ training: { ...Store.get('training'), learningRate: Math.pow(10, parseFloat(lr.value)) } });
      if (Training.modelExists && Training.epochCounter > 0) {
        if (typeof Training.rebuildOptimizer === 'function') Training.rebuildOptimizer();
        else { Training.buildModel(); Training.setDataTensors(); }
        this.showToast('Learning rate will apply on next Start', 'success');
      } else { Training.buildModel(); Training.setDataTensors(); }
    });
    opt.addEventListener('change', () => {
      Store.set({ training: { ...Store.get('training'), optimizer: opt.value } });
      if (Training.modelExists && Training.epochCounter > 0) {
        if (typeof Training.rebuildOptimizer === 'function') Training.rebuildOptimizer();
        else { Training.buildModel(); Training.setDataTensors(); }
        this.showToast('Optimizer will apply on next Start', 'success');
      } else { Training.buildModel(); Training.setDataTensors(); }
    });
    epochsRange.addEventListener('change', () => {
      Store.set({ training: { ...Store.get('training'), maxEpochs: parseInt(epochsRange.value, 10) } });
    });
    updateLrLabel(); updateEpochs();
  },

  // ---- Controls ----
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
    if (!data.xs || data.xs.length === 0) {
      this.showToast('Type an equation and press Plot first', 'warning');
      return;
    }
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
    if (run.status === 'training') {
      Training.setPaused(true); this.setStatus('paused');
    } else if (run.status === 'paused' || run.status === 'error') {
      Training.setPaused(false); this.setStatus('training'); this.runLoop();
    }
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
    a.href = url;
    a.download = `nn-desmos-${safeEq}-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
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
  },

  renderAll() {
    const data = Store.get('data');
    const pred = Store.get('predictions');
    if (!data.xs || !data.xs.length) return;
    const gtXs = data.xs, gtYs = data.ys;
    let predXs = pred.xs || [], predYs = pred.ys || [];
    Charts.setLoss(Store.get('lossHistory'));
    if (!predYs.length && Training.modelExists) {
      const xs = predXs.length ? predXs : gtXs;
      Training.predictXs(xs).then((ys) => {
        if (ys) Store.set({ predictions: { xs, ys } });
        else Charts.setPrediction(gtXs, gtYs, [], []);
      });
      Charts.setPrediction(gtXs, gtYs, [], []);
      return;
    }
    Charts.setPrediction(gtXs, gtYs, predXs, predYs);
  },

  updatePredictionOnly() {
    const data = Store.get('data');
    const pred = Store.get('predictions');
    if (!data.xs || !pred.xs) return;
    Charts.setPrediction(data.xs, data.ys, pred.xs, pred.ys);
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
