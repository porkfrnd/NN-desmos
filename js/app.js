// Main app controller: wires up DOM, canvas, controls, charts, and training.

// ---- Small DOM helper ----
function $(sel, root) {
  return (root || document).querySelector(sel);
}
function $all(sel, root) {
  return Array.from((root || document).querySelectorAll(sel));
}

const App = {
  // Reference to a long-running training promise (used to coordinate stop).
  loopPromise: null,

  init() {
    // dependency guard — show friendly message if CDNs fail (offline)
    if (typeof tf === 'undefined') {
      this.showToast('TensorFlow.js failed to load — check your internet connection and refresh.', 'error');
      console.error('tf is undefined — CDN load failed');
    }
    if (typeof Chart === 'undefined') {
      this.showToast('Chart.js failed to load — charts will not render.', 'error');
      console.error('Chart is undefined — CDN load failed');
    }
    try { this.setupCharts(); } catch (e) { console.error('Charts init failed', e); }
    this.setupPresets();
    try { this.setupCanvas(); } catch (e) { console.error('Canvas init failed', e); }
    this.setupArchitecture();
    this.setupHyperparams();
    this.setupControls();
    this.setupStatus();
    this.bindDataSubscriptions();
    try { this.loadPreset('sine'); } catch (e) { console.error('loadPreset failed', e); this.showToast('Failed to load preset — ' + e.message, 'error'); }
    // register service-worker-like resize observer for smoother phone rotation
    let resizeTimer;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => { try { Charts.resize(); } catch (_) {} }, 80);
    }, { passive: true });
  },

  // ---- Charts ----
  setupCharts() {
    Charts.init(document.getElementById('predChart'), document.getElementById('lossChart'));
    window.addEventListener('resize', () => Charts.resize());
  },

  // ---- Dataset / presets ----
  setupPresets() {
    const grid = $('#presetGrid');
    const presetIds = ['sine', 'square', 'damped', 'composite'];
    presetIds.forEach((id) => {
      const btn = document.createElement('button');
      btn.className = 'preset-btn';
      btn.dataset.preset = id;
      const def = PRESET_DEFS[id];
      btn.innerHTML =
        '<span class="preset-name">' + def.name + '</span>' +
        '<span class="preset-formula">' + def.formula + '</span>';
      btn.addEventListener('click', () => this.loadPreset(id));
      grid.appendChild(btn);
    });
  },

  loadPreset(id) {
    Training.setStopRequested(true);
    const sampled = samplePreset(id, 100);
    if (!sampled) return;
    sampled.ys = clipYs(sampled.ys);
    Store.set({ data: { source: 'preset', presetId: id, xs: sampled.xs, ys: sampled.ys } });
    Store.set({ lossHistory: [], predictions: { xs: [], ys: [] } });

    // Highlight active preset btn
    $all('#presetGrid .preset-btn').forEach((b) => {
      b.classList.toggle('active', b.dataset.preset === id);
    });

    // Rebuild model with the new target (resets weights).
    Training.buildModel();
    Training.setDataTensors();
    Training.resetEpochCounter();
    this.setStatus('idle');
    this.renderAll();
  },

  // ---- Canvas ----
  setupCanvas() {
    const cv = $('#drawCanvas');
    CanvasDraw.init(cv, {
      onData: (d) => {
        d.ys = clipYs(d.ys);
        Store.set({ data: { source: 'custom', presetId: null, xs: d.xs, ys: d.ys } });
        Training.setStopRequested(true);
        Training.buildModel();
        Training.setDataTensors();
        Training.resetEpochCounter();
        Store.set({ lossHistory: [], predictions: { xs: [], ys: [] } });
        this.setStatus('idle');
        this.renderAll();
      },
    });
    $('#clearCanvas').addEventListener('click', async () => {
      CanvasDraw.clear();
      // After clearing, restore the currently selected preset.
      const data = Store.get('data');
      if (data && data.presetId) this.loadPreset(data.presetId);
    });
  },

  // ---- Architecture ----
  setupArchitecture() {
    const layersRange = $('#hiddenLayers');
    const neuronsRange = $('#neuronsPerLayer');
    const actSelect = $('#activation');
    const fourierToggle = $('#fourierFeatures');

    const updateLayersLabel = () => {
      $('#hiddenLayersVal').textContent = layersRange.value;
    };
    const updateNeuronsLabel = () => {
      $('#neuronsPerLayerVal').textContent = neuronsRange.value;
    };

    layersRange.addEventListener('input', updateLayersLabel);
    neuronsRange.addEventListener('input', updateNeuronsLabel);

    layersRange.addEventListener('change', () => { this.applyArchFromDom(layersRange, neuronsRange, actSelect, fourierToggle); });
    neuronsRange.addEventListener('change', () => { this.applyArchFromDom(layersRange, neuronsRange, actSelect, fourierToggle); });
    actSelect.addEventListener('change', () => { this.applyArchFromDom(layersRange, neuronsRange, actSelect, fourierToggle); });
    fourierToggle.addEventListener('click', () => {
      const on = !fourierToggle.classList.contains('active');
      fourierToggle.classList.toggle('active', on);
      fourierToggle.setAttribute('aria-checked', on ? 'true' : 'false');
      this.applyArchFromDom(layersRange, neuronsRange, actSelect, fourierToggle);
    });

    // Initialize labels
    updateLayersLabel();
    updateNeuronsLabel();
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

  // Rebuild the model from current architecture + dataset (resets weights).
  resetWeights() {
    Training.setStopRequested(true);
    // Recreate model with the current architecture/dataset.
    Training.buildModel();
    Training.setDataTensors();
    Training.resetEpochCounter();
    Store.set({ lossHistory: [], predictions: { xs: [], ys: [] } });
    this.renderAll();
    this.setStatus('idle');
  },

  // ---- Hyperparameters ----
  setupHyperparams() {
    const lr = $('#learningRate');
    const opt = $('#optimizer');
    const epochsRange = $('#maxEpochs');

    const updateLrLabel = () => {
      const v = Math.pow(10, parseFloat(lr.value));
      $('#learningRateVal').textContent = v.toExponential(1);
    };
    const updateEpochs = () => {
      $('#maxEpochsVal').textContent = epochsRange.value;
    };

    lr.addEventListener('input', updateLrLabel);
    epochsRange.addEventListener('input', updateEpochs);

    lr.addEventListener('change', () => {
      Store.set({ training: { ...Store.get('training'), learningRate: Math.pow(10, parseFloat(lr.value)) } });
      // keep current weights if already training — just swap optimizer
      if (Training.modelExists && Training.epochCounter > 0) {
        if (typeof Training.rebuildOptimizer === 'function') Training.rebuildOptimizer();
        else { Training.buildModel(); Training.setDataTensors(); }
        this.showToast('Learning rate will apply to next Start', 'success');
      } else {
        Training.buildModel(); Training.setDataTensors();
      }
    });
    opt.addEventListener('change', () => {
      Store.set({ training: { ...Store.get('training'), optimizer: opt.value } });
      if (Training.modelExists && Training.epochCounter > 0) {
        if (typeof Training.rebuildOptimizer === 'function') Training.rebuildOptimizer();
        else { Training.buildModel(); Training.setDataTensors(); }
        this.showToast('Optimizer will apply to next Start', 'success');
      } else {
        Training.buildModel(); Training.setDataTensors();
      }
    });
    epochsRange.addEventListener('change', () => {
      Store.set({ training: { ...Store.get('training'), maxEpochs: parseInt(epochsRange.value, 10) } });
    });

    updateLrLabel();
    updateEpochs();
  },

  // ---- Execution controls ----
  setupControls() {
    const startBtn = $('#btnStart');
    const pauseBtn = $('#btnPause');
    const stepBtn = $('#btnStep');
    const resetBtn = $('#btnResetWeights');

    startBtn.addEventListener('click', () => this.startTraining());
    pauseBtn.addEventListener('click', () => this.togglePause());
    stepBtn.addEventListener('click', () => this.runStep());
    resetBtn.addEventListener('click', () => { this.resetWeights(); });
  },

  async startTraining() {
    const data = Store.get('data');
    if (!data.xs || data.xs.length === 0) {
      this.showToast('Draw or select a function first', 'warning');
      return;
    }
    if (Store.get('run').status === 'training') return;
    if (!Training.modelExists) {
      Training.buildModel();
      Training.setDataTensors();
    }
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
      if (status === 'nan' || status === 'diverged') {
        this.handleRunEnd(status);
        return;
      }
      if (status === 'stopped') { this.setStatus('idle'); return; }
      if (Training.isPaused) { this.setStatus('paused'); return; }
      if (status === 'done' && Training.epochCounter >= target) {
        this.setStatus('idle');
        return;
      }
      await tf.nextFrame();
    }
    if (Training.stopRequested) this.setStatus('idle');
  },

  async togglePause() {
    const run = Store.get('run');
    if (run.status === 'training') {
      Training.setPaused(true);
      this.setStatus('paused');
    } else if (run.status === 'paused' || run.status === 'error') {
      Training.setPaused(false);
      this.setStatus('training');
      this.runLoop();
    }
  },

  // Step exactly 10 epochs regardless of pause state, updates once at end.
  async runStep() {
    const data = Store.get('data');
    if (!data.xs || data.xs.length === 0) {
      this.showToast('Draw or select a function first', 'warning');
      return;
    }
    // stop any continuous run, wait a tick for it to exit, then do 10
    Training.setStopRequested(true);
    await new Promise(r => setTimeout(r, 60));
    Training.setStopRequested(false);
    if (!Training.modelExists) {
      Training.buildModel();
      Training.setDataTensors();
    }
    const wasPaused = Training.isPaused;
    const wasTraining = Store.get('run').status === 'training';
    Training.setPaused(false);
    this.setStatus('training');
    const status = await Training.runEpochs(10);
    if (status === 'nan' || status === 'diverged') {
      this.handleRunEnd(status);
      return;
    }
    // Step always ends paused/idle, never continues the loop
    Training.setPaused(wasPaused || !wasTraining);
    this.setStatus(wasPaused || !wasTraining ? 'paused' : 'idle');
  },

  handleRunEnd(status) {
    if (status === 'nan' || status === 'diverged') {
      this.showToast(status === 'nan' ? 'NaN loss — auto-paused' : 'Loss diverged — auto-paused', 'error');
      this.setStatus('error');
    } else {
      this.setStatus(Training.isPaused ? 'paused' : 'idle');
    }
  },

  // ---- Status UI ----
  setupStatus() {
    this.statusEl = $('#statusDot');
    this.epochEl = $('#epochVal');
    this.lossEl = $('#lossVal');
    // run subscription is handled in bindDataSubscriptions
  },

  setStatus(status) {
    const el = this.statusEl;
    if (!el) return;
    el.classList.remove('training', 'paused', 'error');
    const label = $('#statusLabel');
    if (status === 'training') {
      el.classList.add('training');
      if (label) label.textContent = 'Training';
    } else if (status === 'paused') {
      el.classList.add('paused');
      if (label) label.textContent = 'Paused';
    } else if (status === 'error') {
      el.classList.add('error');
      if (label) label.textContent = 'Error';
    } else {
      if (label) label.textContent = 'Idle';
    }
  },

  // ---- Data rendering (charts) ----
  bindDataSubscriptions() {
    Store.subscribe('data', () => this.renderAll());
    Store.subscribe('predictions', () => this.updatePredictionOnly());
    Store.subscribe('lossHistory', () => {
      Charts.setLoss(Store.get('lossHistory'));
    });
    // also update when model changes theme
    Store.subscribe('run', (run) => {
      if (this.epochEl) this.epochEl.textContent = run.epoch ?? 0;
      if (this.lossEl) this.lossEl.textContent = run.loss != null ? Number(run.loss).toExponential(2) : '—';
    });
  },

  renderAll() {
    const data = Store.get('data');
    const pred = Store.get('predictions');
    if (!data.xs || !data.xs.length) return;
    const gtXs = data.xs;
    const gtYs = data.ys;
    let predXs = pred.xs || [];
    let predYs = pred.ys || [];
    // keep loss chart in sync
    Charts.setLoss(Store.get('lossHistory'));
    if (!predYs.length && Training.modelExists) {
      const xs = predXs.length ? predXs : gtXs;
      Training.predictXs(xs).then((ys) => {
        if (ys) {
          // store it so future renders are consistent
          Store.set({ predictions: { xs, ys } });
        } else {
          Charts.setPrediction(gtXs, gtYs, [], []);
        }
      });
      // show ground truth alone until prediction arrives
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

  // ---- Toasts ----
  showToast(msg, type = 'warning') {
    const cont = $('#toastContainer');
    if (!cont) return;
    const t = document.createElement('div');
    t.className = 'toast toast-' + (type || 'warning');
    t.textContent = msg;
    cont.appendChild(t);
    setTimeout(() => t.remove(), 4000);
  },
};

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

document.addEventListener('DOMContentLoaded', () => App.init());
