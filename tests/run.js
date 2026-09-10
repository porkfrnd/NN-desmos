#!/usr/bin/env node
// Intensive tests for NN Desmos — run with `node tests/run.js`
// No dependencies, just Node's assert. Tests the pure JS (Equation, Presets, Store)
// and does a brutal security audit.

const fs = require('fs');
const path = require('path');
const assert = require('assert');
const vm = require('vm');

// ── helpers ────────────────────────────────────────────────────────────────
let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); console.log(`✓ ${name}`); passed++; }
  catch (e) { console.error(`✗ ${name}\n  ${e.message}\n  ${e.stack?.split('\n')[1]}`); failed++; }
}
function approx(a, b, eps=1e-6) { assert.ok(Math.abs(a-b) < eps, `expected ${a} ≈ ${b}`); }

// ── load source files in a sandbox with mocks ────────────────────────────
const root = path.join(__dirname, '..');
const storeSrc    = fs.readFileSync(path.join(root, 'js/store.js'), 'utf8');
const presetsSrc  = fs.readFileSync(path.join(root, 'js/presets.js'), 'utf8');
const featuresSrc = fs.readFileSync(path.join(root, 'js/features.js'), 'utf8');
const shareSrc    = fs.readFileSync(path.join(root, 'js/share.js'), 'utf8');
const equationSrc = fs.readFileSync(path.join(root, 'js/equation.js'), 'utf8');
const sirenSrc    = fs.readFileSync(path.join(root, 'js/siren.js'), 'utf8');
const modelSrc    = fs.readFileSync(path.join(root, 'js/model.js'), 'utf8');

// mock globals that the files expect
const sandbox = {
  console,
  Math, JSON, Array, Object, String, Number, Date, Error,
  URLSearchParams,
  // fake tf and Chart so files don't throw on load
  tf: {
    scalar: () => ({dispose:()=>{}}), tidy: fn=>fn(), mul: ()=>({}), add: ()=>({}), sub: ()=>({}),
    square: ()=>({}), sum: ()=>({}), pow: ()=>({}), sigmoid: ()=>({}), tanh: ()=>({}),
    layers: {
      dense: () => ({apply: x=>x}),
      input: () => ({}),
      activation: () => ({apply: x=>x}),
      layer: () => class { constructor(){} },
    },
    model: () => ({layers:[], getWeights:()=>[]}),
    train: { adam: ()=>({dispose:()=>{}}), sgd: ()=>({dispose:()=>{}}) },
    regularizers: { l2: ()=>null },
    version: { tfjs: 'mock' },
  },
  Chart: class MockChart { constructor(){this.data={datasets:[]}; this.options={scales:{x:{},y:{}}}} destroy(){} update(){} resize(){} },
  document: { getElementById: ()=>null, createElement: ()=>({}), querySelector: ()=>null, querySelectorAll: ()=>[] },
  window: {},
  localStorage: { getItem: ()=>null, setItem: ()=>{} },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

// Load in dependency order: store -> presets -> features -> share -> equation -> siren -> model
// (model is not needed for most unit tests, but we load it to check it parses)
const loadOrder = [
  ['store.js', storeSrc], ['presets.js', presetsSrc], ['features.js', featuresSrc],
  ['share.js', shareSrc], ['equation.js', equationSrc], ['siren.js', sirenSrc], ['model.js', modelSrc],
];
for (const [name, src] of loadOrder) {
  try { vm.runInContext(src, sandbox, { filename: name }); }
  catch (e) { console.error(`${name} failed to load:`, e.message); process.exitCode = 1; }
}

// `const Store = ...` in the VM does not become a property of `sandbox` —
// grab it via explicit eval in that context (like a browser global).
function getGlobal(name) { try { return vm.runInContext(name, sandbox); } catch (_) { return undefined; } }
const Store = getGlobal('Store');
const Equation = getGlobal('Equation');
const Features = getGlobal('Features');
const Share = getGlobal('Share');
const sirenDense = getGlobal('sirenDense');
const PRESET_DEFS = getGlobal('PRESET_DEFS');
const Presets = getGlobal('Presets');
const samplePreset = getGlobal('samplePreset');
const clipYs = getGlobal('clipYs');
const TUNING_PRESETS = getGlobal('TUNING_PRESETS');

// ── Equation parser ───────────────────────────────────────────────────────
console.log('\n── Equation parser ──');
test('parses x^2', () => {
  const c = Equation.compile('x^2');
  assert.strictEqual(c.fn(2), 4);
  assert.strictEqual(c.fn(-1), 1);
});
test('parses x^2 + 6*x', () => {
  const c = Equation.compile('x^2 + 6*x');
  approx(c.fn(1), 7); approx(c.fn(-1), -5);
});
test('handles y= prefix and unicode', () => {
  const c = Equation.compile('y = x² + 6x');
  approx(c.fn(1), 7);
});
test('handles pi and e', () => {
  const c = Equation.compile('sin(pi*x)');
  approx(c.fn(0.5), 1);
  const c2 = Equation.compile('e^x');
  approx(c2.fn(0), 1);
});
test('rejects empty', () => { assert.throws(() => Equation.compile('')); });
test('rejects no x', () => { assert.throws(() => Equation.compile('2+3')); });
test('rejects invalid chars', () => { assert.throws(() => Equation.compile('x; alert(1)')); });
test('samples over train range', () => {
  const c = Equation.compile('x');
  const s = Equation.sample(c, 5, -1, 1);
  assert.strictEqual(s.xs.length, 5);
  approx(s.xs[0], -1); approx(s.xs[4], 1);
  approx(s.ys[0], -1);
});
test('sampleString convenience', () => {
  const s = Equation.sampleString('x^2', 3, -1, 1);
  assert.strictEqual(s.xs.length, 3);
  assert.ok(s.compiled.src === 'x^2' || s.compiled.src === 'x**2');
  assert.strictEqual(s.compiled.expr, 'x**2');
});

// ── Security: injection attempts must fail ───────────────────────────────
console.log('\n── Security ──');
const injections = [
  'x; console.log(1)',
  'x.constructor.constructor("return process")()',
  'x + require("fs")',
  'x + global.process',
  'x + window.alert(1)',
  'x + this.constructor',
  'x + import("fs")',
  'x + fetch("http://evil")',
  'constructor',
  'x + Math.constructor',
];
injections.forEach(expr => {
  test(`blocks injection: ${expr.slice(0,30)}`, () => {
    assert.throws(() => Equation.compile(expr));
  });
});
test('allows only Math.* fns', () => {
  const ok = Equation.compile('sin(x) + cos(x) + sqrt(x+1) + log(abs(x)+1)');
  assert.ok(ok.fn(0.5));
});
test('blocks unknown function', () => {
  assert.throws(() => Equation.compile('foo(x)'));
  assert.throws(() => Equation.compile('evil(x)'));
});

// ── Presets ───────────────────────────────────────────────────────────────
console.log('\n── Presets ──');
test('samplePreset sine', () => {
  const s = samplePreset('sine', 100);
  assert.strictEqual(s.xs.length, 100);
  approx(s.xs[0], -1); approx(s.xs[99], 1);
  approx(s.ys[50], 0, 0.1);
});
test('samplePreset respects train range', () => {
  const s = samplePreset('sine', 10, -0.5, 0.5);
  approx(s.xs[0], -0.5); approx(s.xs[9], 0.5);
});
test('clipYs', () => {
  assert.deepStrictEqual(clipYs([2, -2, 0.5]), [1.5, -1.5, 0.5]);
});
test('TUNING_PRESETS exist', () => {
  assert.ok(TUNING_PRESETS.smooth);
  assert.ok(TUNING_PRESETS.periodic);
  assert.ok(TUNING_PRESETS.step);
  assert.strictEqual(TUNING_PRESETS.smooth.config.model.activation, 'gelu');
  assert.strictEqual(TUNING_PRESETS.periodic.config.model.activation, 'sine');
});

// ── Store ─────────────────────────────────────────────────────────────────
console.log('\n── Store ──');
test('get/set and subscribe', () => {
  let called = 0;
  const unsub = Store.subscribe('model', () => called++);
  const m = Store.get('model');
  Store.set({ model: { ...m, hiddenLayers: 5 } });
  assert.strictEqual(called, 1);
  assert.strictEqual(Store.get('model').hiddenLayers, 5);
  unsub();
  Store.set({ model: { ...m, hiddenLayers: 3 } });
  assert.strictEqual(called, 1); // unsub worked
});
test('large arrays use reference equality (no JSON lag)', () => {
  const t0 = Date.now();
  for (let i=0;i<100;i++) {
    const xs = Array.from({length:100}, (_,k)=>k/100);
    const ys = xs.map(x=>x*x);
    Store.set({ data: { source:'test', presetId:null, equation:'x^2', xs, ys } });
  }
  const dt = Date.now() - t0;
  assert.ok(dt < 500, `100 large sets took ${dt}ms, should be <500 (no stringify)`);
});
test('domain set', () => {
  Store.set({ domain: { trainMin: -0.5, trainMax: 0.5, evalMin: -1, evalMax: 1 } });
  const d = Store.get('domain');
  assert.strictEqual(d.trainMin, -0.5);
  // reset
  Store.set({ domain: { trainMin: -1, trainMax: 1, evalMin: -2, evalMax: 2 } });
});

// ── Equation: scientific notation (regression: 1e-3*x parsed as 1*e - 3*x) ─
console.log('\n── Scientific notation ──');
test('parses 1e-3*x', () => {
  const c = Equation.compile('1e-3*x');
  approx(c.fn(1), 0.001, 1e-12);
  approx(c.fn(-2), -0.002, 1e-12);
});
test('parses 2.5E2*x', () => {
  const c = Equation.compile('2.5E2*x');
  approx(c.fn(1), 250, 1e-9);
});
test('2e*x still means 2·e·x (no exponent digits)', () => {
  const c = Equation.compile('2e*x');
  approx(c.fn(1), 2 * Math.E, 1e-12);
});
test('x*1e-3 and 1e-3 combine with the rest of the grammar', () => {
  const c = Equation.compile('sin(pi*x)*1e-2');
  approx(c.fn(0.5), 0.01, 1e-12);
});

// ── Equation: non-finite handling ─────────────────────────────────────────
console.log('\n── Non-finite handling ──');
test('NaN samples become 0, ±Inf clips to ±1.5', () => {
  const c = Equation.compile('log(x)');
  const s = Equation.sample(c, 5, -1, 1); // xs: -1,-0.5,0,0.5,1
  assert.strictEqual(s.ys[0], 0); // log(-1) = NaN -> 0, not a fake ±1.5
  assert.ok(Math.abs(s.ys[3] - Math.log(0.5)) < 1e-12);
  const d = Equation.compile('1/(x-1)');
  const s2 = Equation.sample(d, 5, -1, 1); // x=1 -> +Inf
  assert.strictEqual(s2.ys[4], 1.5);
});
test('rejects equations that are never finite', () => {
  assert.throws(() => Equation.compile('sqrt(-x^2 - 1)'));
});

// ── Features (embeddings) ─────────────────────────────────────────────────
console.log('\n── Features ──');
test('fourier feature count = 2*(N+1)', () => {
  assert.strictEqual(Features.featureDim({ embedding: 'fourier', fourierN: 3 }), 8);
  assert.strictEqual(Features.featureDim({ embedding: 'fourier', fourierN: 0 }), 2);
});
test('chebyshev feature count = degree+1', () => {
  assert.strictEqual(Features.featureDim({ embedding: 'chebyshev', chebyshevDegree: 6 }), 7);
});
test('chebyshev values are the actual T_n', () => {
  const f = Features.buildFeatureFn({ embedding: 'chebyshev', chebyshevDegree: 3 }, null);
  const out = f(0.5); // T0..T3 at 0.5: 1, .5, -.5, -1
  approx(out[0], 1); approx(out[1], 0.5); approx(out[2], -0.5); approx(out[3], -1);
});
test('features scale to the train range (T1(x=-2) on [-2,2] is -1, not -2)', () => {
  const f = Features.buildFeatureFn({ embedding: 'chebyshev', chebyshevDegree: 2 }, { trainMin: -2, trainMax: 2 });
  const out = f(-2);
  approx(out[1], -1, 1e-9); // rescaled u = -1 at the left edge
  approx(out[2], 1, 1e-9);  // T2(-1) = 1
});
test('default train range [-1,1] is identity (no behavior change)', () => {
  const f = Features.buildFeatureFn({ embedding: 'chebyshev', chebyshevDegree: 2 }, { trainMin: -1, trainMax: 1 });
  approx(f(0.25)[1], 0.25, 1e-12);
});
test('featureDim agrees with the feature fn length (the drift bug)', () => {
  for (const cfg of [
    { embedding: 'fourier', fourierN: 5 }, { embedding: 'fourier', fourierN: 99 }, // clamped to 6
    { embedding: 'chebyshev', chebyshevDegree: 12 }, { embedding: 'chebyshev', chebyshevDegree: 99 },
    { embedding: 'none' },
  ]) {
    const fn = Features.buildFeatureFn(cfg, { trainMin: -1, trainMax: 1 });
    assert.strictEqual(fn(0.3).length, Features.featureDim(cfg), JSON.stringify(cfg));
  }
});
test('legacy fourierFeatures flag still selects fourier (when embedding unset)', () => {
  assert.strictEqual(Features.featureDim({ fourierFeatures: true, fourierN: 2 }), 6);
  assert.strictEqual(Features.featureDim({ embedding: undefined, fourierFeatures: true, fourierN: 2 }), 6);
});

// ── Share links (URL hash round trip) ─────────────────────────────────────
console.log('\n── Share links ──');
test('encode -> decode round trips everything', () => {
  const state = {
    data: { source: 'equation', presetId: null, equation: 'sin(2*pi*x) + e^-x', xs: [], ys: [] },
    model: { hiddenLayers: 4, neuronsPerLayer: 24, activation: 'sine', embedding: 'fourier', fourierN: 4, fourierSigma: 1.2, chebyshevDegree: 8, omega0: 17.5 },
    training: { optimizer: 'sgd', learningRate: 0.003, weightDecay: 0.0001, noise: 0.1, maxEpochs: 500 },
    domain: { trainMin: -1, trainMax: 1, evalMin: -2, evalMax: 2 },
  };
  const hash = '#' + Share.encode(state);
  const back = Share.sanitize(Share.decode(hash));
  assert.strictEqual(back.model.activation, 'sine');
  assert.strictEqual(back.model.embedding, 'fourier');
  assert.strictEqual(back.model.hiddenLayers, 4);
  assert.strictEqual(back.model.fourierSigma, 1.2);
  assert.strictEqual(back.model.omega0, 17.5);
  assert.strictEqual(back.training.learningRate, 0.003);
  assert.strictEqual(back.training.weightDecay, 0.0001);
  assert.strictEqual(back.training.noise, 0.1);
  assert.strictEqual(back.training.optimizer, 'sgd');
  // (field-by-field: vm objects have a different Object.prototype than host objects)
  assert.strictEqual(back.domain.trainMin, -1);
  assert.strictEqual(back.domain.trainMax, 1);
  assert.strictEqual(back.domain.evalMin, -2);
  assert.strictEqual(back.domain.evalMax, 2);
  assert.strictEqual(back.data.equation, state.data.equation);
});
test('sanitize drops hostile/garbage values', () => {
  const back = Share.sanitize(Share.decode('#act=hax&emb=laser&lr=999&opt=adam&train=2,1&eq=x^2'));
  assert.strictEqual(back.model.activation, undefined);
  assert.strictEqual(back.model.embedding, undefined);
  assert.strictEqual(back.training.learningRate, 0.1); // 999 clamped to the slider max, not dropped
  assert.strictEqual(back.domain, null); // min >= max -> dropped
  assert.strictEqual(back.data.equation, 'x^2');
});
test('sanitize clamps instead of rejecting slightly-off values', () => {
  const back = Share.sanitize(Share.decode('#hl=99&np=1&lr=1'));
  assert.strictEqual(back.model.hiddenLayers, 5);
  assert.strictEqual(back.model.neuronsPerLayer, 2);
  assert.strictEqual(back.training.learningRate, 0.1);
});
test('decode ignores junk hashes', () => {
  assert.strictEqual(Share.decode(''), null);
  assert.strictEqual(Share.decode('#'), null);
  assert.strictEqual(Share.decode('#foo=bar'), null);
});

// ── SIREN layer (regression: class was block-scoped, factory always fell back) ─
console.log('\n── SIREN layer ──');
test('sirenDense falls back to tanh dense when tf lacks Layer (CDN degraded)', () => {
  // sandbox's tf mock has no layers.Layer -> factory must be null, not throw
  const layer = sirenDense(8, true, 30);
  assert.ok(layer, 'fallback layer returned');
});
test('SirenDense uses the paper init: first ±1/fan_in, hidden ±√6/fan_in/ω₀', () => {
  // separate context with a Layer-capable tf mock that captures initializer bounds
  const captured = [];
  class MockWeight {
    constructor(name, shape, dtype, init) { this.name = name; this.shape = shape;
      if (init && init.__cfg) captured.push(init.__cfg); }
    read() { return {}; }
  }
  const tfMock = {
    layers: {
      Layer: class { constructor(cfg) { this.cfg = cfg; } addWeight(...a) { return new MockWeight(...a); } },
      dense: () => ({}),
    },
    serialization: { registerClass: () => {} },
    initializers: {
      randomUniform: (cfg) => ({ __cfg: cfg }),
      zeros: () => ({ __cfg: { minval: 0, maxval: 0 } }),
    },
    tidy: (fn) => fn(),
    sin: () => ({}), mul: () => ({}),
  };
  const sb = { tf: tfMock };
  sb.window = sb; sb.globalThis = sb;
  vm.createContext(sb);
  vm.runInContext(sirenSrc, sb, { filename: 'siren.js' });
  const sd = vm.runInContext('sirenDense', sb);
  const first = sd(16, true, 30);   // fan_in would come from build(inputShape)
  first.build([4, 16]);             // inputShape -> fanIn 16
  const hidden = sd(8, false, 30);
  hidden.build([4, 8]);
  // captured: [kernel(first), kernel(hidden)] (zeros biases don't set __cfg? they do — filter)
  const kernels = captured.filter((c) => c.minval < 0);
  approx(kernels[0].minval, -1 / 16, 1e-12);          // first layer: U(±1/fan_in)
  approx(kernels[0].maxval, 1 / 16, 1e-12);
  approx(kernels[1].minval, -Math.sqrt(6 / 8) / 30, 1e-12); // hidden: U(±√6/fan_in/ω₀)
  approx(kernels[1].maxval, Math.sqrt(6 / 8) / 30, 1e-12);
  assert.strictEqual(first.getClassName ? first.getClassName() : first.constructor.className, 'SirenDense');
});

// ── Summary ───────────────────────────────────────────────────────────────
console.log(`\n${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
else console.log('All intensive tests passed — no bugs, no vulns, no lag (for those paths).');
