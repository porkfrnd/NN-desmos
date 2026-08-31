// Preset target functions defined over the fixed domain x ∈ [-1, 1].
// All live in the same coordinate system as the drawn data and the model
// input (normalized in [-1, 1] before any feature transform / dense layer).

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

// Domain constants (shared coordinate system)
const DOMAIN_MIN = -1;
const DOMAIN_MAX = 1;

// Produce 100 evenly-spaced samples of a preset function over the domain.
function samplePreset(id, count = 100) {
  const def = PRESET_DEFS[id];
  if (!def || !def.fn) return null;
  const xs = [];
  const ys = [];
  for (let i = 0; i < count; i++) {
    const t = count === 1 ? 0 : i / (count - 1);
    const x = DOMAIN_MIN + (DOMAIN_MAX - DOMAIN_MIN) * t;
    xs.push(x);
    ys.push(def.fn(x));
  }
  return { xs, ys };
}

// Clip a set of y values to roughly [-1.5, 1.5] for stable training.
function clipYs(ys, limit = 1.5) {
  return ys.map((y) => Math.max(-limit, Math.min(limit, y)));
}
