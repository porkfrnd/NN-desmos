// Preset target functions and tuning presets.

const Presets = {
  SIN: 'sine',
  SQUARE: 'square',
  DAMPED: 'damped',
  COMPOSITE: 'composite',
  CUSTOM: 'custom',
};

const PRESET_DEFS = {
  [Presets.SIN]: {
    name: 'Sine',
    formula: 'sin(2\u03C0x)',
    equation: 'sin(2*pi*x)',
    fn: (x) => Math.sin(2 * Math.PI * x),
  },
  [Presets.SQUARE]: {
    name: 'Square',
    formula: 'sign(sin(2\u03C0x))',
    equation: 'sign(sin(2*pi*x))',
    fn: (x) => Math.sign(Math.sin(2 * Math.PI * x)),
  },
  [Presets.DAMPED]: {
    name: 'Damped',
    formula: 'e\u207B\u02E3cos(4\u03C0x)',
    equation: 'exp(-x)*cos(4*pi*x)',
    fn: (x) => Math.exp(-x) * Math.cos(4 * Math.PI * x),
  },
  [Presets.COMPOSITE]: {
    name: 'Composite',
    formula: 'sin(2\u03C0x)+0.5sin(10\u03C0x)',
    equation: 'sin(2*pi*x) + 0.5*sin(10*pi*x)',
    fn: (x) => Math.sin(2 * Math.PI * x) + 0.5 * Math.sin(10 * Math.PI * x),
  },
  [Presets.CUSTOM]: {
    name: 'Custom',
    formula: 'equation',
    equation: '',
    fn: null,
  },
};

// One-click tuning presets for architecture/hyperparams
const TUNING_PRESETS = {
  smooth: {
    name: 'Smooth / Polynomial',
    desc: 'GELU, no Fourier, low LR',
    config: {
      model: { hiddenLayers: 3, neuronsPerLayer: 16, activation: 'gelu', embedding: 'none', fourierN: 3, fourierSigma: 1, chebyshevDegree: 6, omega0: 30 },
      training: { learningRate: 0.0005, weightDecay: 0.0001, optimizer: 'adam' },
    },
  },
  periodic: {
    name: 'Periodic / High Freq',
    desc: 'Fourier + SIREN',
    config: {
      model: { hiddenLayers: 3, neuronsPerLayer: 24, activation: 'sine', embedding: 'fourier', fourierN: 4, fourierSigma: 1.2, omega0: 30 },
      training: { learningRate: 0.001, weightDecay: 0, optimizer: 'adam' },
    },
  },
  step: {
    name: 'Discontinuous / Step',
    desc: 'Deep ReLU, no Fourier',
    config: {
      model: { hiddenLayers: 5, neuronsPerLayer: 24, activation: 'relu', embedding: 'none', omega0: 30 },
      training: { learningRate: 0.001, weightDecay: 0.0005, optimizer: 'adam' },
    },
  },
};

const DOMAIN_MIN = -1;
const DOMAIN_MAX = 1;

function samplePreset(id, count = 100, trainMin = DOMAIN_MIN, trainMax = DOMAIN_MAX) {
  const def = PRESET_DEFS[id];
  if (!def || !def.fn) return null;
  const xs = [], ys = [];
  for (let i = 0; i < count; i++) {
    const t = count === 1 ? 0 : i / (count - 1);
    const x = trainMin + (trainMax - trainMin) * t;
    xs.push(x);
    ys.push(def.fn(x));
  }
  return { xs, ys };
}

function clipYs(ys, limit = 1.5) {
  return ys.map((y) => Math.max(-limit, Math.min(limit, y)));
}
