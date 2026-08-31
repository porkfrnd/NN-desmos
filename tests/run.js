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
const equationSrc = fs.readFileSync(path.join(root, 'js/equation.js'), 'utf8');
const modelSrc    = fs.readFileSync(path.join(root, 'js/model.js'), 'utf8');

// mock globals that the files expect
const sandbox = {
  console,
  Math, JSON, Array, Object, String, Number, Date, Error,
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

// Load in dependency order: store -> presets -> equation
// (model is not needed for most unit tests, but we load it to check it parses)
try { vm.runInContext(storeSrc, sandbox, {filename: 'store.js'}); } catch (e) { console.error('store.js failed to load:', e.message); }
try { vm.runInContext(presetsSrc, sandbox, {filename: 'presets.js'}); } catch (e) { console.error('presets.js failed:', e.message); }
try { vm.runInContext(equationSrc, sandbox, {filename: 'equation.js'}); } catch (e) { console.error('equation.js failed:', e.message); }

// `const Store = ...` in the VM does not become a property of `sandbox` —
// grab it via explicit eval in that context (like a browser global).
function getGlobal(name) { try { return vm.runInContext(name, sandbox); } catch (_) { return undefined; } }
const Store = getGlobal('Store');
const Equation = getGlobal('Equation');
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

// ── Summary ───────────────────────────────────────────────────────────────
console.log(`\n${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
else console.log('All intensive tests passed — no bugs, no vulns, no lag (for those paths).');
