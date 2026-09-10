/**
 * Model & training — the ML heart.
 *
 * Why manual loop, not model.fit(): `fit()` can't pause mid-epoch.
 * We use `optimizer.minimize()` + `isPaused` ref checked every epoch,
 * and yield to the browser so the UI stays responsive during training.
 *
 * Why time-budgeted yielding: the old loop waited for a frame every 2
 * epochs, which capped training at ~2 epochs per 16.7ms no matter how fast
 * tf.js actually was (and read the loss tensor every epoch, stalling the
 * WebGL pipeline). Now we train for ~12ms, then give the browser one frame —
 * on a WebGL backend that's ~5-10× more epochs per second with the same
 * responsiveness. The graph/loss redraw is capped at ~11×/s; the chart, not
 * the math, is the bottleneck at high epoch rates.
 *
 * Races: rebuilding the model while a loop is mid-epoch used to train the
 * wrong tensors or explode on disposed ones. Every run now takes a
 * *generation* token; anything that replaces the model bumps the generation
 * and stale loops retire quietly at their next check.
 *
 * Memory: every scratch tensor is in `tf.tidy()`. Only the model's own
 * weights survive. `predictXs` creates its own tensors and disposes them
 * explicitly — never inside an async tidy.
 *
 * Embeddings live in features.js (pure) — one source of truth for the
 * feature map AND the input dim, scaled to the train range.
 * Regularization: L2 is added to MSE as `wd * sum(W^2)` only when wd>0.
 */
// Model construction + manual training loop.
//
// Time-budgeted yielding, generation tokens for safe rebuilds, and L2 weight decay.
// Activations: relu, tanh, sigmoid, softplus, silu, gelu, sine (SIREN).

const YIELD_BUDGET_MS = 12;   // train this long, then give the browser a frame
const UI_REFRESH_MS = 90;     // redraw graph + loss at most ~11x/s
const LOSS_EVERY = 10;        // read the loss every N epochs (each readback stalls the GPU)
const NAN_THRESHOLD = 1e6;
const PRED_SAMPLES = 140;

const nowMs = () => (typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now());

const Training = (() => {
  const ctx = {
    model: null,
    optimizer: null,
    xTrain: null,
    yTrain: null,
    isPaused: false,
    stopRequested: false,
    epochCounter: 0,
    lastLoss: null,
    featureFn: null,
    generation: 0,   // bumped whenever the run is replaced/cancelled
    pendingLoss: [], // loss points waiting for the next UI flush
  };

  // ---- Feature transforms (see features.js) ----
  function buildFeatureFn() {
    return Features.buildFeatureFn(Store.get('model'), Store.get('domain'));
  }
  function inputDimForModel() {
    return Features.featureDim(Store.get('model'));
  }

  // self-explaining: SiLU and GELU aren't built into tfjs dense, so we
  // implement them as tiny custom layers. Defined once, reused per layer.
  function applyCustomActivation(t, name) {
    switch (name) {
      case 'silu': return tf.tidy(() => tf.mul(t, tf.sigmoid(t)));
      case 'gelu': return tf.tidy(() => {
        const c = Math.sqrt(2 / Math.PI);
        const inner = tf.add(t, tf.mul(0.044715, tf.pow(t, tf.scalar(3))));
        const tanhInner = tf.tanh(tf.mul(c, inner));
        return tf.mul(tf.mul(0.5, t), tf.add(tf.scalar(1), tanhInner));
      });
      default: return t;
    }
  }
  // Single reusable layer class for SiLU/GELU — avoids defining a new class per layer (was leaking)
  const CustomActLayer = (() => {
    class _CustomAct extends tf.layers.Layer {
      constructor(cfg) { super(cfg || {}); this.actName = cfg.actName; }
      call(inp) { const t = Array.isArray(inp) ? inp[0] : inp; return applyCustomActivation(t, this.actName); }
      computeOutputShape(s) { return s; }
      getClassName() { return 'CustomAct_' + this.actName; }
    }
    return _CustomAct;
  })();

  // ---- Model construction ----
  function buildModel() {
    if (typeof tf === 'undefined') {
      console.error('tf not loaded');
      try { if (typeof App !== 'undefined' && App.showToast) App.showToast('TensorFlow.js not loaded', 'error'); } catch (_) {}
      return null;
    }
    disposeContext();
    const m = Store.get('model');
    ctx.featureFn = buildFeatureFn();
    const inputDim = inputDimForModel();
    const inputs = tf.input({ shape: [inputDim] });
    let x = inputs;
    const act = m.activation || 'tanh';
    const isCustomAct = act === 'silu' || act === 'gelu';
    const isSiren = act === 'sine';

    for (let i = 0; i < m.hiddenLayers; i++) {
      if (isSiren) {
        const w0 = m.omega0 ?? (m.embedding && m.embedding !== 'none' ? 1.0 : 30.0);
        const layer = sirenDense(m.neuronsPerLayer, i === 0, w0);
        x = layer.apply(x);
      } else if (isCustomAct) {
        const dense = tf.layers.dense({ units: m.neuronsPerLayer, activation: 'linear', useBias: true });
        x = dense.apply(x);
        const ca = new CustomActLayer({ actName: act });
        x = ca.apply(x);
      } else {
        // built-in: relu, tanh, sigmoid, softplus, etc.
        const tfAct = act === 'sine' ? 'linear' : act;
        x = tf.layers.dense({ units: m.neuronsPerLayer, activation: tfAct, useBias: true }).apply(x);
      }
    }
    const output = tf.layers.dense({ units: 1, activation: 'linear' }).apply(x);
    ctx.model = tf.model({ inputs, outputs: output });

    buildOptimizer();
    ctx.lastLoss = null;
    return ctx.model;
  }

  function buildOptimizer() {
    if (ctx.optimizer) { try { ctx.optimizer.dispose(); } catch (e) {} ctx.optimizer = null; }
    const optCfg = Store.get('training');
    ctx.optimizer = optCfg.optimizer === 'adam' ? tf.train.adam(optCfg.learningRate) : tf.train.sgd(optCfg.learningRate);
  }

  // ---- Dataset / tensors ----
  function setDataTensors() {
    const d = Store.get('data');
    if (!d.xs || !d.xs.length) return;
    if (!ctx.featureFn) ctx.featureFn = buildFeatureFn();
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
    if (ctx.optimizer) { try { ctx.optimizer.dispose(); } catch (e) {} ctx.optimizer = null; }
    ctx.generation++; // any in-flight loop must stop touching the tensors we just freed
    ctx.pendingLoss = [];
    Store.set({ run: { ...Store.get('run'), status: 'idle', loss: null } });
  }

  function refreshDataTensors() {
    if (ctx.xTrain) { try { ctx.xTrain.dispose(); } catch (e) {} ctx.xTrain = null; }
    if (ctx.yTrain) { try { ctx.yTrain.dispose(); } catch (e) {} ctx.yTrain = null; }
    setDataTensors();
  }

  // ---- Prediction ----
  async function predictXs(xs) {
    if (!ctx.model || !xs || xs.length === 0) return null;
    if (!ctx.featureFn) ctx.featureFn = buildFeatureFn();
    const rows = xs.map((x) => {
      const f = ctx.featureFn(x);
      return Array.isArray(f) ? f : [f];
    });
    const xT = tf.tensor2d(rows, [rows.length, rows[0].length]);
    const out = ctx.model.predict(xT);
    let vals;
    try { vals = await out.array(); } finally { xT.dispose(); if (out && out.dispose) out.dispose(); }
    return vals.map((r) => r[0]);
  }

  // ---- Training loop ----
  async function runEpochs(count, gen) {
    if (!ctx.model || !ctx.xTrain) return 'no-model';
    const myGen = (gen != null) ? gen : ctx.generation;
    let lastYield = nowMs(), lastUi = 0;
    for (let i = 0; i < count; i++) {
      if (ctx.stopRequested || myGen !== ctx.generation) { flushLoss(); return 'stopped'; }
      if (ctx.isPaused) { flushLoss(); return 'paused'; }

      // read the loss (and check divergence) only every LOSS_EVERY epochs —
      // each dataSync is a GPU pipeline stall, and at high epoch rates that
      // stall was the single biggest per-epoch cost
      const computeLoss = ((ctx.epochCounter + 1) % LOSS_EVERY === 0) || i === count - 1;
      let lossValue = null;
      tf.tidy(() => {
        const loss = ctx.optimizer.minimize(() => meanSquaredError(), computeLoss);
        if (computeLoss && loss) lossValue = loss.dataSync()[0];
      });
      ctx.epochCounter++;
      if (computeLoss) {
        if (typeof lossValue !== 'number' || isNaN(lossValue)) {
          ctx.pendingLoss = []; // NaN history is noise, don't flush it
          Store.set({ run: { ...Store.get('run'), status: 'error', message: 'NaN loss — auto-paused. Try lower LR or change activation.' } });
          ctx.isPaused = true;
          return 'nan';
        }
        if (lossValue > NAN_THRESHOLD) {
          flushLoss();
          Store.set({ run: { ...Store.get('run'), status: 'error', message: 'Loss diverged — auto-paused. Lower LR.' } });
          ctx.isPaused = true;
          return 'diverged';
        }
        ctx.lastLoss = lossValue;
        ctx.pendingLoss.push({ epoch: ctx.epochCounter, loss: lossValue });
      }

      // time-budgeted yield: keep the math running for a slice of a frame,
      // then let the browser breathe — instead of a fixed 2-epochs-per-frame
      if (nowMs() - lastYield >= YIELD_BUDGET_MS || i === count - 1) {
        if (nowMs() - lastUi >= UI_REFRESH_MS || i === count - 1) {
          await refreshUi();
          lastUi = nowMs();
        }
        await tf.nextFrame();
        lastYield = nowMs();
      }
    }
    flushLoss();
    return 'done';
  }

  // L2 is the classic lag culprit: summing all weights each epoch is heavy.
  // We make it fast by (1) early exit when wd===0 (default), and (2) when
  // wd>0, doing it in a single tidy with explicit dispose so no intermediate
  // tensors leak and the GPU stays at 60fps.
  function meanSquaredError() {
    const wd = Store.get('training').weightDecay ?? 0;
    if (wd === 0) {
      const pred = ctx.model.predict(ctx.xTrain);
      const loss = tf.losses.meanSquaredError(ctx.yTrain, pred);
      pred.dispose();
      return loss;
    }
    return tf.tidy(() => {
      const pred = ctx.model.predict(ctx.xTrain);
      const mse = tf.losses.meanSquaredError(ctx.yTrain, pred);
      let l2 = tf.scalar(0);
      for (const w of ctx.model.getWeights()) {
        const cur = tf.sum(tf.square(w));
        const nxt = tf.add(l2, cur);
        l2.dispose(); cur.dispose();
        l2 = nxt;
      }
      const reg = tf.mul(l2, wd);
      const out = tf.add(mse, reg);
      // mse, l2, reg are intermediates — tidy keeps only `out`
      return out;
    });
  }

  // hand buffered loss points to the store (cheap: one array concat per flush)
  function flushLoss() {
    if (!ctx.pendingLoss.length) return;
    Store.set({
      lossHistory: [...Store.get('lossHistory'), ...ctx.pendingLoss],
      run: { ...Store.get('run'), epoch: ctx.epochCounter, loss: ctx.lastLoss },
    });
    ctx.pendingLoss = [];
  }

  async function refreshUi() {
    flushLoss();
    const dom = Store.get('domain');
    const xs = [];
    for (let i = 0; i < PRED_SAMPLES; i++) xs.push(dom.evalMin + (dom.evalMax - dom.evalMin) * i / (PRED_SAMPLES - 1));
    try {
      const preds = await predictXs(xs);
      if (preds) Store.set({ predictions: { xs, ys: preds } });
    } catch (_) {
      // the model was swapped mid-predict (equation/settings changed) —
      // the run that replaced us will refresh on its own
    }
  }

  function rebuildOptimizer() {
    buildOptimizer();
  }

  function exportWeights() {
    if (!ctx.model) return null;
    const m = Store.get('model');
    const d = Store.get('data');
    const dom = Store.get('domain');
    const tr = Store.get('training');
    const layers = [];
    for (let i = 1; i < ctx.model.layers.length; i++) {
      const layer = ctx.model.layers[i];
      const ws = layer.getWeights();
      if (!ws.length) continue;
      const isSiren = layer.getClassName && layer.getClassName() === 'SirenDense';
      const kind = isSiren ? 'siren' : (layer.getClassName && layer.getClassName().startsWith('CustomAct') ? 'custom-act' : 'dense');
      if (kind === 'custom-act') continue;
      const wT = ws[0], bT = ws[1] || null;
      layers.push({
        index: i,
        kind,
        units: layer.units || (bT ? bT.shape[0] : wT.shape[1]),
        activation: isSiren ? 'sine' : (m.activation || 'linear'),
        omega0: isSiren ? (m.omega0 ?? 30) : undefined,
        kernel: { shape: wT.shape.slice(), data: Array.from(wT.dataSync()) },
        bias: bT ? { shape: bT.shape.slice(), data: Array.from(bT.dataSync()) } : null,
      });
    }
    return {
      meta: {
        exportedAt: new Date().toISOString(),
        tfjsVersion: (typeof tf !== 'undefined' && tf.version && tf.version.tfjs) || null,
        architecture: { ...m, inputDim: inputDimForModel() },
        training: { ...tr },
        domain: dom ? { ...dom } : { xMin: -1, xMax: 1, yClip: [-1.5, 1.5] },
        equation: d.equation || d.presetId || null,
        epochsTrained: ctx.epochCounter,
        lastLoss: ctx.lastLoss ?? null,
      },
      layers,
    };
  }

  return {
    buildModel,
    setDataTensors,
    refreshDataTensors,
    runEpochs,
    disposeContext,
    predictXs,
    rebuildOptimizer,
    exportWeights,
    get isPaused() { return ctx.isPaused; },
    get stopRequested() { return ctx.stopRequested; },
    get modelExists() { return !!ctx.model; },
    get epochCounter() { return ctx.epochCounter; },
    // A new run claims the generation; cancelled loops see the mismatch at
    // their next epoch check and exit without touching the fresh model.
    beginRun() { ctx.stopRequested = false; ctx.isPaused = false; return ++ctx.generation; },
    cancelRun() {
      ctx.stopRequested = true;
      ctx.generation++;
      const run = Store.get('run');
      if (run.status === 'training') Store.set({ run: { ...run, status: 'idle' } });
    },
    currentGeneration() { return ctx.generation; },
    setPaused(v) {
      ctx.isPaused = !!v;
      const run = Store.get('run');
      if (v) Store.set({ run: { ...run, status: 'paused' } });
      else Store.set({ run: { ...run, status: 'training' } });
    },
    resetEpochCounter() {
      ctx.epochCounter = 0;
      ctx.pendingLoss = [];
      ctx.lastLoss = null;
      Store.set({ run: { ...Store.get('run'), epoch: 0, loss: null } });
    },
    setStopRequested(v) { ctx.stopRequested = !!v; },
  };
})();

window.Training = Training;
