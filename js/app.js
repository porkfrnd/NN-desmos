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
    this.setupCharts();
    this.setupPresets();
    this.setupCanvas();
    this.setupArchitecture();
    this.setupHyperparams();
    this.setupControls();
    this.setupStatus();
    this.bindDataSubscriptions();
    this.loadPreset('sine'); // initial dataset
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
      fourierToggle.classList.toggle('active');
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
      // LR change rebuilds the optimizer (keeps model weights via buildModel
      // reconstruction on the same architecture).
      Training.buildModel();
      Training.setDataTensors();
    });
    opt.addEventListener('change', () => {
      Store.set({ training: { ...Store.get('training'), optimizer: opt.value } });
      Training.buildModel();
      Training.setDataTensors();
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
    while (!Training.stopRequested && !Training.isPaused) {
      const status = await Training.runEpochs(Math.min(cfg.maxEpochs, 100));
      if (status === 'nan' || status === 'diverged') {
        this.handleRunEnd(status);
        return;
      }
      if (status === 'stopped') {
        this.setStatus('idle');
        return;
      }
      if (Training.isPaused) {
        this.setStatus('paused');
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

  // Step exactly 10 epochs regardless of pause state, update once at end.
  async runStep() {
    const data = Store.get('data');
    if (!data.xs || data.xs.length === 0) {
      this.showToast('Draw or select a function first', 'warning');
      return;
    }
    Training.setStopRequested(true); // stop any running loop
    if (!Training.modelExists) {
      Training.buildModel();
      Training.setDataTensors();
    }
    const wasPaused = Training.isPaused;
    Training.setPaused(false);
    this.setStatus('training');
    const status = await Training.runEpochs(10);
    // Restore pause/status as appropriate.
    if (status === 'nan' || status === 'diverged' || status === 'stopped') {
      this.setStatus(wasPaused ? 'paused' : 'idle');
      return;
    }
    this.setStatus(wasPaused ? 'paused' : Training.isPaused ? 'paused' : 'idle');
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
    Store.subscribe('run', (run) => {
      if (run.epoch !== undefined && this.epochEl) this.epochEl.textContent = run.epoch;
      if (run.loss != null && this.lossEl) {
        this.lossEl.textContent = run.loss.toExponential(3);
      }
    });
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
  },

  renderAll() {
    const data = Store.get('data');
    const pred = Store.get('predictions');
    // Ground truth from store x; if custom, xs from data.
    if (data.xs && data.xs.length) {
      const gtXs = data.xs;
      const gtYs = data.ys;
      let predXs = pred.xs || [];
      let predYs = pred.ys || [];
      if (!predYs.length && Training.modelExists) {
        // Compute an initial prediction immediately.
        Training.predictXs(predXs.length ? predXs : gtXs).then((ys) => {
          if (ys) Charts.setPrediction(gtXs, gtYs, predXs, ys);
        });
        return;
      }
      Charts.setPrediction(gtXs, gtYs, predXs, predYs);
    }
  },

  updatePredictionOnly() {
    const data = Store.get('data');
    const pred = Store.get('predictions');
    if (data.xs && pred.xs) {
      Charts.setPrediction(data.xs, data.ys, pred.xs, pred.ys);
    }
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
