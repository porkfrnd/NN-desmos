// Model construction + manual training loop.
//
// The loop uses a `for` loop over epochs with an `isPaused` ref checked each
// iteration, so Pause/Resume halts cleanly BETWEEN epochs rather than only
// after a monolithic fit() completes. We do NOT use model.fit() for the main
// loop; instead each step calls optimizer.minimize(fn, true) directly and,
// every UPDATE_EVERY epochs, yields with await tf.nextFrame() and refreshes
// chart state.
//
// Memory discipline:
//   - Every per-step scratch tensor is wrapped in tf.tidy().
//   - The model's own weight tensors are owned by the layer and NOT tidy'd.
//   - Prediction/loss scratch tensors are read via .dataSync() only on the
//     update cadence (not every epoch), then disposed.

const TRAIN_UPDATE_EVERY = 5; // refresh UI every N epochs
const PRED_SAMPLES = 120;
const NAN_THRESHOLD = 1e6; // treat loss above this as divergent (guard NaN/inf sgd+tanh)

const Training = (() => {
  const ctx = {
    model: null,
    optimizer: null,
    xTrain: null, // tf.Tensor [N, features]
    yTrain: null, // tf.Tensor [N, 1]
    isPaused: false,
    stopRequested: false,
    epochCounter: 0,
  };

  // ---- Feature transform ------------------------------------------------
  // Fourier features: x -> [sin(2^k π x), cos(2^k π x)] for k=0..3 => 8 feats.
  // When disabled, the raw normalized x ([-1,1]) is the single feature.
  function buildFeatureFn(fourier) {
    if (!fourier) return (x) => x;
    return (x) => {
      const parts = [];
      for (let k = 0; k < 4; k++) {
        const f = (1 << k) * Math.PI;
        parts.push(Math.sin(f * x), Math.cos(f * x));
      }
      return parts;
    };
  }

  // ---- Model construction ----------------------------------------------
  function buildModel() {
    if (typeof tf === 'undefined') {
      console.error('tf not loaded');
      try { if (typeof App !== 'undefined' && App.showToast) App.showToast('TensorFlow.js not loaded — cannot build model', 'error'); } catch (_) {}
      return null;
    }
    disposeContext();

    const m = Store.get('model');
    const fourier = !!m.fourierFeatures;
    ctx.featureFn = buildFeatureFn(fourier);
    const inputDim = fourier ? 8 : 1;

    const inputs = tf.input({ shape: [inputDim] });
    let x = inputs;

    // Hidden layers. Activation 'sine' uses the custom SIREN layer.
    for (let i = 0; i < m.hiddenLayers; i++) {
      if (m.activation === 'sine') {
        // First hidden layer gets the ±1/fan_in init; subsequent SIREN
        // layers get ±√(6/fan_in)/ω₀.
        const layer = sirenDense(m.neuronsPerLayer, i === 0);
        x = layer.apply(x);
      } else {
        x = tf.layers
          .dense({ units: m.neuronsPerLayer, activation: m.activation, useBias: true })
          .apply(x);
      }
    }

    // Output layer: 1 unit, linear (default), always.
    const output = tf.layers.dense({ units: 1, activation: 'linear' }).apply(x);

    ctx.model = tf.model({ inputs, outputs: output });

    // Optimizer
    const optCfg = Store.get('training');
    if (optCfg.optimizer === 'adam') {
      ctx.optimizer = tf.train.adam(optCfg.learningRate);
    } else {
      ctx.optimizer = tf.train.sgd(optCfg.learningRate);
    }

    ctx.lastLoss = null;
    return ctx.model;
  }

  // ---- Dataset / tensors -----------------------------------------------
  function setDataTensors() {
    // Convert stored JS arrays to tf tensors using the current feature fn.
    const d = Store.get('data');
    const rows = d.xs.map((x) => ctx.featureFn(x));
    const n = rows.length;
    if (n === 0) return;
    ctx.xTrain = tf.tensor2d(rows, [n, rows[0].length]);
    ctx.yTrain = tf.tensor2d(d.ys.map((y) => [y]), [n, 1]);
  }

  function disposeContext() {
    if (ctx.model) { try { ctx.model.dispose(); } catch (e) {} ctx.model = null; }
    if (ctx.xTrain) { try { ctx.xTrain.dispose(); } catch (e) {} ctx.xTrain = null; }
    if (ctx.yTrain) { try { ctx.yTrain.dispose(); } catch (e) {} ctx.yTrain = null; }
    if (ctx.optimizer) { ctx.optimizer.dispose(); ctx.optimizer = null; }
    Store.set({ run: { ...Store.get('run'), status: 'idle', loss: null } });
  }

  // Rebuild dataset tensors after a feature/arch change (keeps weights if
  // arch unchanged, but the model is rebuilt so weights reset anyway).
  function refreshDataTensors() {
    if (ctx.xTrain) { ctx.xTrain.dispose(); ctx.xTrain = null; }
    if (ctx.yTrain) { ctx.yTrain.dispose(); ctx.yTrain = null; }
    setDataTensors();
  }

  // ---- Prediction calls ------------------------------------------------
  // Predict on densely-spaced x for the chart (does not touch train tensors).
  async function predictXs(xs) {
    if (!ctx.model || !xs || xs.length === 0) return null;
    const rows = xs.map((x) => {
      const f = ctx.featureFn(x);
      return Array.isArray(f) ? f : [f];
    });
    const xT = tf.tensor2d(rows, [rows.length, rows[0].length]);
    const out = ctx.model.predict(xT);
    let vals;
    try {
      vals = await out.array();
    } finally {
      xT.dispose();
      if (out && out.dispose) out.dispose();
    }
    return vals.map((r) => r[0]);
  }

  // ---- Training loop ----------------------------------------------------
  // Runs 0..count-1 epochs, checking isPaused/stopRequested each iteration.
  // The per-step loss is computed inside tf.tidy() and kept as a float value
  // only (no sync tensor read every epoch). UI refresh (which reads tensors
  // via .dataSync/.array) happens only on the TRAIN_UPDATE_EVERY cadence.
  async function runEpochs(count) {
    if (!ctx.model || !ctx.xTrain) return 'no-model';

    for (let i = 0; i < count; i++) {
      if (ctx.stopRequested) return 'stopped';
      if (ctx.isPaused) return 'paused';

      // One gradient step. minimize(fn, true) returns the loss tensor, which
      // is inside the tidy scope and disposed after we read its scalar value.
      let lossValue = null;
      tf.tidy(() => {
        const loss = ctx.optimizer.minimize(() => meanSquaredError(), true);
        if (loss) lossValue = loss.dataSync()[0];
      });
      // Note: dataSync per epoch is O(1) on a scalar and cheap; the expensive
      // sync reads are the prediction arrays which we gate on the cadence.

      if (typeof lossValue !== 'number' || isNaN(lossValue)) {
        Store.set({
          run: { ...Store.get('run'), status: 'error', message: 'NaN loss detected — auto-paused. Lower learning rate or switch away from high-LR SGD + Sine.' },
        });
        ctx.isPaused = true;
        return 'nan';
      }
      if (lossValue > NAN_THRESHOLD) {
        Store.set({
          run: { ...Store.get('run'), status: 'error', message: 'Loss diverged (very large). Auto-paused. Lower learning rate or increase samples.' },
        });
        ctx.isPaused = true;
        return 'diverged';
      }
      ctx.lastLoss = lossValue;
      ctx.epochCounter++;

      if (i % TRAIN_UPDATE_EVERY === TRAIN_UPDATE_EVERY - 1 || i === count - 1) {
        await refreshUi(lossValue);
        await tf.nextFrame();
      }
    }
    return 'done';
  }

  function meanSquaredError() {
    // Compute MSE between model output and target; returns a scalar tensor.
    const pred = ctx.model.predict(ctx.xTrain);
    const loss = tf.losses.meanSquaredError(ctx.yTrain, pred);
    pred.dispose();
    return loss;
  }

  // Refresh prediction curve + loss history from store-bound tensors.
  async function refreshUi(epochLoss) {
    const run = Store.get('run');
    // Append to loss history
    const hist = [...Store.get('lossHistory')];
    if (typeof epochLoss === 'number') {
      hist.push({ epoch: ctx.epochCounter, loss: epochLoss });
    }
    Store.set({ lossHistory: hist, run: { ...run, epoch: ctx.epochCounter, loss: epochLoss } });

    // Recompute prediction curve over the domain
    const xs = [];
    for (let i = 0; i < PRED_SAMPLES; i++) {
      xs.push(-1 + (2 * i) / (PRED_SAMPLES - 1));
    }
    const preds = await predictXs(xs);
    if (preds) Store.set({ predictions: { xs, ys: preds } });
  }

  function rebuildOptimizer() {
    if (ctx.optimizer) { try { ctx.optimizer.dispose(); } catch (_) {} ctx.optimizer = null; }
    const optCfg = Store.get('training');
    ctx.optimizer = optCfg.optimizer === 'adam' ? tf.train.adam(optCfg.learningRate) : tf.train.sgd(optCfg.learningRate);
  }

  // ---- Public API --------------------------------------------------------
  return {
    buildModel,
    setDataTensors,
    refreshDataTensors,
    runEpochs,
    disposeContext,
    predictXs,
    rebuildOptimizer,
    get isPaused() { return ctx.isPaused; },
    get stopRequested() { return ctx.stopRequested; },
    get modelExists() { return !!ctx.model; },
    get epochCounter() { return ctx.epochCounter; },
    setPaused(v) {
      ctx.isPaused = !!v;
      const run = Store.get('run');
      if (v) Store.set({ run: { ...run, status: 'paused' } });
      else Store.set({ run: { ...run, status: 'training' } });
    },
    resetEpochCounter() { ctx.epochCounter = 0; Store.set({ run: { ...Store.get('run'), epoch: 0, loss: null } }); },
    setStopRequested(v) { ctx.stopRequested = !!v; },
  };
})();

// Re-export for clarity; used by app.js.
window.Training = Training;
