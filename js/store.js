/**
 * Tiny pub/sub store — the single source of truth.
 *
 * Why this exists: the app has no framework, so panels share state
 * (dataset, model config, training status) without prop-drilling.
 * Each section (data, model, training, domain, run) is plain JSON.
 * Views subscribe to a key and re-render when it changes.
 *
 * Performance note: we avoid deep equality on large arrays (data, predictions)
 * to keep training at 60fps. Reference change is enough.
 */
// Tiny pub/sub store so the panels share state (model config, dataset,
// training state) without prop-drilling. Each palette is plain data; views
// subscribe to key changes.

const Store = (() => {
  let debug = false;

  const state = {
    // Dataset / target
    data: {
      source: null, // 'preset' | 'equation'
      presetId: null,
      equation: null,
      xs: [],
      ys: [],
    },
    predictions: {
      xs: [],
      ys: [],
    },
    lossHistory: [],

    // Model architecture
    model: {
      hiddenLayers: 3,
      neuronsPerLayer: 16,
      activation: 'tanh', // 'relu'|'tanh'|'sigmoid'|'sine'|'silu'|'gelu'|'softplus'
      embedding: 'none', // 'none'|'fourier'|'chebyshev'
      fourierN: 3, // k=0..N => 2*(N+1) features
      fourierSigma: 1.0, // frequency scale
      chebyshevDegree: 6, // T0..TN => N+1 features
      omega0: 30, // SIREN frequency multiplier
      fourierFeatures: false, // legacy, kept for compat
    },

    // Training / optimizer
    training: {
      optimizer: 'adam',
      learningRate: 0.001, // 1e-4 to 1e-1 log scale
      weightDecay: 0.0, // L2 0 to 1e-2
      noise: 0.0, // Gaussian noise std 0..0.3 added to y
      maxEpochs: 500,
    },

    // Domain & evaluation bounds
    domain: {
      trainMin: -1,
      trainMax: 1,
      evalMin: -2,
      evalMax: 2,
    },

    // Runtime state
    run: {
      status: 'idle', // 'idle'|'training'|'paused'|'error'
      epoch: 0,
      loss: null,
      message: null,
    },
  };

  const listeners = {};

  function emit(key) {
    (listeners[key] || []).forEach((fn) => {
      try { fn(state[key], state); } catch (e) { if (debug) console.error(e); }
    });
    (listeners['*'] || []).forEach((fn) => {
      try { fn(state[key], state); } catch (e) { if (debug) console.error(e); }
    });
  }

  function subscribe(key, fn) {
    (listeners[key] = listeners[key] || []).push(fn);
    return () => {
      const i = (listeners[key] || []).indexOf(fn);
      if (i > -1) listeners[key].splice(i, 1);
    };
  }

  // Fast, self-explaining: we intentionally skip deep equality for large arrays
  // (lossHistory, predictions, data.xs/ys). Reference change is enough — the UI
  // subscribes to the *section* and will re-render. This avoids JSON.stringify
  // on 100+ point arrays every 5 epochs, which was the main lag source.
  function set(patch) {
    const changed = [];
    for (const section in patch) {
      if (!Object.prototype.hasOwnProperty.call(state, section)) continue;
      // For tiny config objects (model, training, domain, run) a cheap
      // JSON check is fine and avoids redundant renders. For large data
      // (data, predictions, lossHistory) we use reference equality.
      const prev = state[section];
      const next = patch[section];
      const isLarge = section === 'data' || section === 'predictions' || section === 'lossHistory';
      const equal = isLarge ? prev === next : JSON.stringify(prev) === JSON.stringify(next);
      if (!equal) {
        state[section] = next;
        changed.push(section);
      }
    }
    changed.forEach((key) => emit(key));
  }

  function get(key) { return key ? state[key] : state; }
  function setDebug(v) { debug = !!v; }

  return { get, set, subscribe, setDebug };
})();
