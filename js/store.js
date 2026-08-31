// Tiny pub/sub store so the panels share state (model config, dataset,
// training state) without prop-drilling. Each palette is plain data; views
// subscribe to key changes.

const Store = (() => {
  let debug = false;

  const state = {
    // Dataset / target
    data: {
      source: null, // 'preset' | 'custom'
      presetId: null,
      xs: [],      // Float32-compatible arrays, already on [-1,1]
      ys: [],
    },
    predictions: {
      xs: [],      // for charting (100-150 points)
      ys: [],
    },
    lossHistory: [],  // array of { epoch, loss }

    // Model architecture
    model: {
      hiddenLayers: 3,
      neuronsPerLayer: 16,
      activation: 'tanh',   // 'relu' | 'tanh' | 'sigmoid' | 'sine'
      fourierFeatures: false,
    },

    // Training / optimizer
    training: {
      optimizer: 'adam',    // 'adam' | 'sgd'
      learningRate: 0.001,
      maxEpochs: 500,
    },

    // Runtime state
    run: {
      status: 'idle',       // 'idle' | 'training' | 'paused' | 'error'
      epoch: 0,
      loss: null,
      message: null,
    },
  };

  // Each key is an array of subscriber callbacks. '*' fires on any change.
  const listeners = {};

  function emit(key) {
    (listeners[key] || []).forEach((fn) => {
      try {
        fn(state[key], state);
      } catch (e) {
        if (debug) console.error(e);
      }
    });
    (listeners['*'] || []).forEach((fn) => {
      try {
        fn(state[key], state);
      } catch (e) {
        if (debug) console.error(e);
      }
    });
  }

  function subscribe(key, fn) {
    (listeners[key] = listeners[key] || []).push(fn);
    return () => {
      const i = (listeners[key] || []).indexOf(fn);
      if (i > -1) listeners[key].splice(i, 1);
    };
  }

  function set(patch) {
    const changed = new Map();
    for (const section in patch) {
      if (!Object.prototype.hasOwnProperty.call(state, section)) continue;
      const prev = state[section];
      const next = patch[section];
      if (JSON.stringify(prev) !== JSON.stringify(next)) {
        state[section] = next;
        changed.set(section, next);
      }
    }
    changed.forEach((_, key) => emit(key));
  }

  function get(key) {
    return key ? state[key] : state;
  }

  function setDebug(v) {
    debug = !!v;
  }

  return { get, set, subscribe, setDebug };
})();
