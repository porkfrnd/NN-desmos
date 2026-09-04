/**
 * Model & training — the ML heart.
 *
 * Why manual loop, not model.fit(): `fit()` can't pause mid-epoch.
 * We use `optimizer.minimize()` + `isPaused` ref checked each epoch,
 * and `await tf.nextFrame()` to yield to the browser so the UI stays
 * at 60fps even while training.
 *
 * Memory: every scratch tensor is in `tf.tidy()`. Only the model's
 * own weights survive. `predictXs` creates its own tensors and
 * disposes them explicitly — never inside an async tidy.
 *
 * Embeddings: Fourier (trig) for periodic, Chebyshev for polynomials.
 * Regularization: L2 is added to MSE as `wd * sum(W^2)` only when wd>0.
 */
// Model construction + manual training loop.
//
// Manual loop with isPaused ref, tf.tidy discipline, and L2 weight decay.
// Supports embeddings: none / fourier (variable N, sigma) / chebyshev,
// and activations: relu, tanh, sigmoid, softplus, silu, gelu, sine (SIREN).

const TRAIN_UPDATE_EVERY = 10; // 10 = smoother, less chart thrash at 60fps
const PRED_SAMPLES = 140;
const NAN_THRESHOLD = 1e6;

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
  };

  // ---- Feature transforms ----
  function buildFeatureFn() {
    const m = Store.get('model');
    const emb = m.embedding || (m.fourierFeatures ? 'fourier' : 'none');
    if (emb === 'fourier') {
      const N = Math.max(0, Math.min(6, m.fourierN ?? 3));
      const sigma = m.fourierSigma ?? 1.0;
      return (x) => {
        const out = [];
        for (let k = 0; k <= N; k++) {
          const f = (1 << k) * Math.PI * sigma;
          out.push(Math.sin(f * x), Math.cos(f * x));
        }
        return out;
      };
    }
    if (emb === 'chebyshev') {
      const deg = Math.max(1, Math.min(16, m.chebyshevDegree ?? 6));
      return (x) => {
        // Chebyshev T_n(x), clamped to [-1,1] domain (extrapolation will diverge naturally)
        const out = [];
        let t0 = 1, t1 = x;
        out.push(t0);
        if (deg >= 1) out.push(t1);
        for (let n = 2; n <= deg; n++) {
          const tn = 2 * x * t1 - t0;
          out.push(tn);
          t0 = t1; t1 = tn;
        }
        return out;
      };
    }
    return (x) => [x];
  }

  function inputDimForModel() {
    const m = Store.get('model');
    const emb = m.embedding || (m.fourierFeatures ? 'fourier' : 'none');
    if (emb === 'fourier') {
      const N = Math.max(0, Math.min(6, m.fourierN ?? 3));
      return 2 * (N + 1);
    }
    if (emb === 'chebyshev') {
      const deg = Math.max(1, Math.min(16, m.chebyshevDegree ?? 6));
      return deg + 1;
    }
    return 1;
  }

  // self-explaining: SiLU and GELU aren't built into tfjs dense, so we
  // implement them as tiny custom layers. Defined once, reused for every hidden layer.
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

    const optCfg = Store.get('training');
    if (optCfg.optimizer === 'adam') ctx.optimizer = tf.train.adam(optCfg.learningRate);
    else ctx.optimizer = tf.train.sgd(optCfg.learningRate);

    ctx.lastLoss = null;
    return ctx.model;
  }

  // ---- Dataset / tensors ----
  function setDataTensors() {
    const d = Store.get('data');
    if (!d.xs || !d.xs.length) return;
    if (!ctx.featureFn) ctx.featureFn = buildFeatureFn();
    // filter out null/NaN gaps (e.g. log(x) for x<0) — training needs finite values
    const pts = d.xs.map((x, i) => ({ x, y: d.ys[i] })).filter(p => p.y !== null && Number.isFinite(p.y));
    if (pts.length === 0) return;
    const rows = pts.map(p => ctx.featureFn(p.x));
    const ys = pts.map(p => p.y);
    ctx.xTrain = tf.tensor2d(rows, [rows.length, rows[0].length]);
    ctx.yTrain = tf.tensor2d(ys.map(y => [y]), [ys.length, 1]);
  }

  function disposeContext() {
    if (ctx.model) { try { ctx.model.dispose(); } catch (e) {} ctx.model = null; }
    if (ctx.xTrain) { try { ctx.xTrain.dispose(); } catch (e) {} ctx.xTrain = null; }
    if (ctx.yTrain) { try { ctx.yTrain.dispose(); } catch (e) {} ctx.yTrain = null; }
    if (ctx.optimizer) { try { ctx.optimizer.dispose(); } catch (e) {} ctx.optimizer = null; }
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
  async function runEpochs(count) {
    if (!ctx.model || !ctx.xTrain) return 'no-model';
    for (let i = 0; i < count; i++) {
      if (ctx.stopRequested) return 'stopped';
      if (ctx.isPaused) return 'paused';
      let lossValue = null;
      tf.tidy(() => {
        const loss = ctx.optimizer.minimize(() => meanSquaredError(), true);
        if (loss) lossValue = loss.dataSync()[0];
      });
      if (typeof lossValue !== 'number' || isNaN(lossValue)) {
        Store.set({ run: { ...Store.get('run'), status: 'error', message: 'NaN loss — auto-paused. Try lower LR or change activation.' } });
        ctx.isPaused = true;
        return 'nan';
      }
      if (lossValue > NAN_THRESHOLD) {
        Store.set({ run: { ...Store.get('run'), status: 'error', message: 'Loss diverged — auto-paused. Lower LR.' } });
        ctx.isPaused = true;
        return 'diverged';
      }
      ctx.lastLoss = lossValue;
      ctx.epochCounter++;
      // Desmos-like smoothness: update charts every 10, but yield to UI every 2 epochs
      // so sliders and graph stay at 60fps even during heavy matmuls.
      const shouldRefresh = (i % TRAIN_UPDATE_EVERY === TRAIN_UPDATE_EVERY - 1) || i === count - 1;
      if (shouldRefresh) await refreshUi(lossValue);
      if (i % 2 === 1) await tf.nextFrame();
      else if (shouldRefresh) await tf.nextFrame();
    }
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

  async function refreshUi(epochLoss) {
    const run = Store.get('run');
    const hist = [...Store.get('lossHistory')];
    if (typeof epochLoss === 'number') hist.push({ epoch: ctx.epochCounter, loss: epochLoss });
    Store.set({ lossHistory: hist, run: { ...run, epoch: ctx.epochCounter, loss: epochLoss } });
    const dom = Store.get('domain');
    const evalMin = dom ? dom.evalMin : -1, evalMax = dom ? dom.evalMax : 1;
    const xs = [];
    for (let i = 0; i < PRED_SAMPLES; i++) xs.push(evalMin + (evalMax - evalMin) * i / (PRED_SAMPLES - 1));
    const preds = await predictXs(xs);
    if (preds) Store.set({ predictions: { xs, ys: preds } });
  }

  function rebuildOptimizer() {
    if (ctx.optimizer) { try { ctx.optimizer.dispose(); } catch (_) {} ctx.optimizer = null; }
    const optCfg = Store.get('training');
    ctx.optimizer = optCfg.optimizer === 'adam' ? tf.train.adam(optCfg.learningRate) : tf.train.sgd(optCfg.learningRate);
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

window.Training = Training;
