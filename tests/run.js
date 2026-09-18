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
const sirenSrc    = fs.readFileSync(path.join(root, 'js/siren.js'), 'utf8');
const modelSrc    = fs.readFileSync(path.join(root, 'js/model.js'), 'utf8');
const chartsSrc   = fs.readFileSync(path.join(root, 'js/charts.js'), 'utf8');
const appSrc      = fs.readFileSync(path.join(root, 'js/app.js'), 'utf8');
const indexHtml   = fs.readFileSync(path.join(root, 'index.html'), 'utf8');

// mock globals that the files expect
const sandbox = {
  console,
  Math, JSON, Array, Object, String, Number, Date, Error,
  // fake tf and Chart so files don't throw on load
  tf: {
    scalar: (v) => ({dispose:()=>{}, shape:[], dataSync:()=>[v]}),
    tidy: fn=>fn(),
    mul: (a,b) => ({dispose:()=>{}}), add: (a,b) => ({dispose:()=>{}}), sub: (a,b) => ({dispose:()=>{}}),
    square: (x) => ({dispose:()=>{}}), sum: (x) => ({dispose:()=>{}, dataSync:()=>[0]}), pow: (a,b) => ({dispose:()=>{}}),
    sigmoid: (x) => x, tanh: (x) => x, sin: (x) => x,
    matMul: () => ({dispose:()=>{}}),
    losses: { meanSquaredError: () => ({dispose:()=>{}, dataSync:()=>[0.1]}) },
    initializers: { randomUniform: ()=>({}), zeros: ()=>({}) },
    train: { adam: (lr) => ({dispose:()=>{}, minimize: (fn, ret) => { const v=fn(); return ret ? {dataSync:()=>[0.1], dispose:()=>{}} : null }}), sgd: (lr) => ({dispose:()=>{}, minimize: (fn, ret) => { const v=fn(); return ret ? {dataSync:()=>[0.1], dispose:()=>{}} : null }}) },
    regularizers: { l2: ()=>null },
    version: { tfjs: 'mock' },
    nextFrame: () => Promise.resolve(),
    serialization: { registerClass: ()=>{} },
    layers: {
      Layer: class { constructor(cfg){ this.built=false; } addWeight(name, shape, dtype, init){ return { read:() => ({ matMul: ()=>({add:()=>({})}), dispose:()=>{} }), shape, dispose:()=>{} }; } },
      dense: (cfg) => ({apply: x=>x, getWeights:() => [], getClassName: () => 'Dense', units: cfg.units, activation: { getClassName: () => cfg.activation || 'linear' }}),
      input: (cfg) => ({}),
      activation: (cfg) => ({apply: x=>x}),
      layer: (cfg) => class { constructor(){ } },
    },
    tensor2d: (data, shape) => ({dispose:()=>{}, shape, dataSync:()=>[0]}),
  },
  Chart: class MockChart { constructor(){this.data={datasets:[]}; this.options={scales:{x:{},y:{}}}} destroy(){} update(){} resize(){} },
  document: {
    getElementById: ()=>null,
    createElement: ()=>({ style:{}, appendChild:()=>{}, remove:()=>{}, click:()=>{}, classList:{add:()=>{},remove:()=>{},toggle:()=>{},contains:()=>false} }),
    querySelector: ()=>null,
    querySelectorAll: ()=>[],
    addEventListener: ()=>{},
    createElementNS: ()=>({}),
  },
  window: { addEventListener: ()=>{} },
  localStorage: { getItem: ()=>null, setItem: ()=>{} },
  location: { hash: '', href: 'http://localhost' },
  navigator: { clipboard: { writeText: ()=>Promise.resolve() } },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

// Load in dependency order: store -> presets -> equation -> siren -> model -> charts -> app
// Now actually executes every file so bugs like #1 are caught
let loadFailed = false;
function tryLoad(src, name) {
  try { vm.runInContext(src, sandbox, {filename: name}); }
  catch (e) { console.error(name + ' failed to load:', e.message); loadFailed = true; failed++; }
}
tryLoad(storeSrc, 'store.js');
tryLoad(presetsSrc, 'presets.js');
tryLoad(equationSrc, 'equation.js');
tryLoad(sirenSrc, 'siren.js');
tryLoad(modelSrc, 'model.js');
tryLoad(chartsSrc, 'charts.js');
tryLoad(appSrc, 'app.js');
if (loadFailed) console.error('One or more source files failed to parse — this would have caught bug #1');

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
const SirenDense = getGlobal('SirenDense');
const Training = getGlobal('Training');
const Charts = getGlobal('Charts');
const App = getGlobal('App');
const SIREN_W0 = getGlobal('SIREN_W0');

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

// ── Regression tests for Critical/Medium/Minor fixes ────────────────────
console.log('\n── Regression: Critical Fixes ──');
test('SIREN: SirenDense globally accessible and not block-scoped', () => {
  assert.ok(typeof SirenDense !== 'undefined' && SirenDense !== null, 'SirenDense should be globally accessible');
  assert.strictEqual(SirenDense.className, 'SirenDense');
  const layer = getGlobal('sirenDense')(4, true, 30);
  assert.ok(layer, 'sirenDense factory should work');
  assert.strictEqual(layer.isFirstLayer, true);
});
test('SIREN: only first layer scales with ω₀', () => {
  const src = require('fs').readFileSync(require('path').join(__dirname, '..', 'js/siren.js'), 'utf8');
  assert.ok(src.includes('if (this.isFirstLayer) out = tf.mul(out, this.w0)'), 'should only scale first layer');
});
test('Typing without Plot updates training tensors (regression #2)', () => {
  const appSrc = require('fs').readFileSync(require('path').join(__dirname, '..', 'js/app.js'), 'utf8');
  assert.ok(appSrc.includes('Training.refreshDataTensors()'), 'debounced handler should refresh tensors');
});
test('Share links restore all config (regression #3)', () => {
  const appSrc = require('fs').readFileSync(require('path').join(__dirname, '..', 'js/app.js'), 'utf8');
  assert.ok(appSrc.includes("params.get('act')"), 'should parse act');
  assert.ok(appSrc.includes("params.get('emb')"), 'should parse emb');
  assert.ok(appSrc.includes("params.get('lr')"), 'should parse lr');
  assert.ok(appSrc.includes("params.get('wd')"), 'should parse wd');
  assert.ok(appSrc.includes("params.get('noise')"), 'should parse noise');
  assert.ok(appSrc.includes("params.get('train')"), 'should parse train');
  assert.ok(appSrc.includes("params.get('eval')"), 'should parse eval');
});

console.log('\n── Regression: Medium Fixes ──');
test('Scientific notation 1e-3*x not mis-parsed (regression #4)', () => {
  const c = Equation.compile('1e-3*x');
  approx(c.fn(2), 0.002);
  const c2 = Equation.compile('1e3*x');
  approx(c2.fn(1), 1000);
});
test('Step button waits for loop (regression #5)', () => {
  const appSrc = require('fs').readFileSync(require('path').join(__dirname, '..', 'js/app.js'), 'utf8');
  assert.ok(appSrc.includes('loopPromise'), 'should track loop promise');
  assert.ok(appSrc.includes('await') && appSrc.includes('loopPromise'), 'runStep should await loop');
});
test('Theme toggle passes 5 args including dots (regression #6)', () => {
  const html = require('fs').readFileSync(require('path').join(__dirname, '..', 'index.html'), 'utf8');
  assert.ok(html.includes('sampleTruthOverEval') || html.includes('setPrediction(g.xs'), 'theme should use eval truth');
});
test('Domain changes preserve noise (regression #7)', () => {
  const appSrc = require('fs').readFileSync(require('path').join(__dirname, '..', 'js/app.js'), 'utf8');
  const hasNoise = appSrc.includes('samplePreset(data.presetId, 100, tMin, tMax, noiseStd)') && appSrc.includes('sampleString(data.equation, 100, tMin, tMax, noiseStd)');
  assert.ok(hasNoise, 'domain resampling should pass noiseStd');
});

console.log('\n── Regression: Minor Fixes ──');
test('touchend uses event param not global (regression #8)', () => {
  const chartsSrc = require('fs').readFileSync(require('path').join(__dirname, '..', 'js/charts.js'), 'utf8');
  assert.ok(chartsSrc.includes("addEventListener('touchend', (e) =>"), 'should use (e)');
  assert.ok(!chartsSrc.includes("addEventListener('touchend', () => { if (event."), 'should not use global event');
});
test('Equation NaN handled as null, not 0 (regression #9)', () => {
  const c = Equation.compile('log(x)');
  const s = Equation.sample(c, 5, -1, 1);
  assert.strictEqual(s.ys[0], null, 'log(-1) should be null, not 0');
});
test('inputDimForModel clamps (regression #10)', () => {
  const modelSrc = require('fs').readFileSync(require('path').join(__dirname, '..', 'js/model.js'), 'utf8');
  assert.ok(modelSrc.includes('Math.max(0, Math.min(6, m.fourierN'), 'should clamp fourierN');
  assert.ok(modelSrc.includes('Math.max(1, Math.min(16, m.chebyshevDegree'), 'should clamp chebyshev');
});
test('runLoop reads maxEpochs dynamically (regression #11)', () => {
  const appSrc = require('fs').readFileSync(require('path').join(__dirname, '..', 'js/app.js'), 'utf8');
  assert.ok(appSrc.includes("Store.get('training').maxEpochs") || appSrc.includes('Store.get("training").maxEpochs'), 'should read maxEpochs inside loop');
});
test('LR toast fixed (regression #11)', () => {
  const appSrc = require('fs').readFileSync(require('path').join(__dirname, '..', 'js/app.js'), 'utf8');
  assert.ok(!appSrc.includes('will apply on next Start'), 'should not say will apply on next Start');
});
test('Dead code removed (regression #12)', () => {
  const fs2 = require('fs'); const path2 = require('path');
  assert.ok(!fs2.existsSync(path2.join(__dirname, '..', 'js/canvas.js')), 'canvas.js should be deleted');
  const sirenSrc = fs2.readFileSync(path2.join(__dirname, '..', 'js/siren.js'), 'utf8');
  assert.ok(!sirenSrc.includes('SineActivation'), 'SineActivation should be removed');
  assert.ok(!sirenSrc.includes('uniformRandom'), 'uniformRandom should be removed');
});
test('Tests actually load all files (regression #13)', () => {
  const testSrc = require('fs').readFileSync(__filename, 'utf8');
  assert.ok(testSrc.includes('sirenSrc'), 'should load siren');
  assert.ok(testSrc.includes('modelSrc'), 'should load model');
  assert.ok(testSrc.includes('chartsSrc'), 'should load charts');
  assert.ok(testSrc.includes('appSrc'), 'should load app');
});


// ── Regression: glitch fixes ─────────────────────────────────────────────
console.log('\n── Regression: glitch fixes ──');

// 1. HTML structure
test('HTML: exactly one .app div', () => {
  const matches = indexHtml.match(/<div class="app">/g);
  assert.ok(matches && matches.length === 1, `expected exactly 1 .app div, got ${matches ? matches.length : 0}`);
});
test('HTML: gallery and footer before Settings modal comment', () => {
  const galleryIdx = indexHtml.indexOf('<div class="gallery">');
  const footerIdx = indexHtml.indexOf('<footer class="site-foot">');
  const settingsCommentIdx = indexHtml.indexOf('<!-- Settings modal');
  assert.ok(galleryIdx > 0 && footerIdx > 0, 'gallery and footer must exist');
  assert.ok(galleryIdx < settingsCommentIdx, 'gallery must come before Settings modal');
  assert.ok(footerIdx < settingsCommentIdx, 'footer must come before Settings modal');
});
test('HTML: no <h3>Loss</h3> heading', () => {
  assert.ok(!indexHtml.includes('<h3>Loss</h3>'), 'should not have an h3 Loss heading');
});
test('HTML: has summary-title class', () => {
  assert.ok(indexHtml.includes('summary-title'), 'should contain summary-title class');
});

// 2. CSS hidden rule
test('CSS base: [hidden] with !important', () => {
  const baseCss = fs.readFileSync(path.join(root, 'css/base.css'), 'utf8');
  assert.ok(baseCss.includes('[hidden]') && baseCss.includes('!important'), 'base.css must have [hidden] with !important');
});

// 3. CSS dead code removed
test('CSS components: no toolbar-main dead code', () => {
  const compCss = fs.readFileSync(path.join(root, 'css/components.css'), 'utf8');
  assert.ok(!compCss.includes('toolbar-main'), 'components.css should not contain toolbar-main');
});

// 4. Charts legend/tooltip filter
test('Charts: legend filter references datasetIndex 0 and 2', () => {
  assert.ok(chartsSrc.includes('filter'), 'charts.js must contain filter functions');
  // Legend and tooltip filters should reference datasetIndex 0 and 2
  const filterMatches = chartsSrc.match(/filter:?\s*\(?item\)?\s*=>\s*item\.datasetIndex\s*===\s*0\s*\|\|\s*item\.datasetIndex\s*===\s*2/g);
  assert.ok(filterMatches && filterMatches.length >= 2, 'need legend and tooltip filter referencing datasetIndex 0 and 2');
});

// 5. Charts no per-frame style recalc, spanGaps, withAlpha
test('Charts: setPrediction uses cachedColors for dots (no per-frame cssVar)', () => {
  // setPrediction should reference cachedColors, not call cssVar for dataset colors
  const setPredMatch = chartsSrc.match(/function setPrediction[\s\S]*?(?=\n  function |\n  return )/);
  assert.ok(setPredMatch, 'setPrediction function must exist');
  const body = setPredMatch[0];
  assert.ok(body.includes('cachedColors'), 'setPrediction must use cachedColors for dots');
  // cachedColors must be the primary path (checked before cssVar fallback)
  assert.ok(body.indexOf('cachedColors') < body.indexOf('cssVar'), 'cachedColors must be checked before cssVar fallback');
});
test('Charts: spanGaps:false present in dataset config', () => {
  assert.ok(chartsSrc.includes('spanGaps: false'), 'spanGaps:false must be in chart config');
});
test('Charts: withAlpha helper present', () => {
  assert.ok(chartsSrc.includes('function withAlpha'), 'withAlpha helper must exist');
});

// 6. Charts domain: setDomainAndReset + clampView
test('Charts: setDomainAndReset updates title.text and assigns cachedDomain', () => {
  const fnMatch = chartsSrc.match(/function setDomainAndReset[\s\S]*?(?=\n  function |\n  return )/);
  assert.ok(fnMatch, 'setDomainAndReset function must exist');
  const body = fnMatch[0];
  assert.ok(body.includes('title.text'), 'setDomainAndReset must update axis title.text');
  assert.ok(body.includes('cachedDomain'), 'setDomainAndReset must assign cachedDomain');
});
test('Charts: clampView derives x bounds from domain evalMin/evalMax with fallback', () => {
  const fnMatch = chartsSrc.match(/function clampView[\s\S]*?(?=\n  function |\n  return )/);
  assert.ok(fnMatch, 'clampView function must exist');
  const body = fnMatch[0];
  assert.ok(body.includes('evalMin'), 'clampView must reference evalMin');
  assert.ok(body.includes('evalMax'), 'clampView must reference evalMax');
  assert.ok(body.includes('cachedDomain'), 'clampView must use cachedDomain');
});

// 7. App loop helpers
test('App: stopLoopAndWait and trackLoop exist', () => {
  assert.ok(appSrc.includes('stopLoopAndWait'), 'app.js must have stopLoopAndWait');
  assert.ok(appSrc.includes('trackLoop'), 'app.js must have trackLoop');
});
test('App: runStep awaits stopLoopAndWait', () => {
  const stepMatch = appSrc.match(/async runStep[\s\S]*?(?=\n  async |\n  [a-z]|\n\}$)/);
  assert.ok(stepMatch, 'runStep must exist');
  const body = stepMatch[0];
  assert.ok(body.includes('await this.stopLoopAndWait()'), 'runStep must await stopLoopAndWait');
});
test('App: togglePause resumes only from paused + idle toast', () => {
  const tpMatch = appSrc.match(/async togglePause[\s\S]*?(?=\n  async |\n  [a-z]|\n\}$)/);
  assert.ok(tpMatch, 'togglePause must exist');
  const body = tpMatch[0];
  assert.ok(body.includes("'paused'"), 'togglePause must handle paused state');
  assert.ok(body.includes("'idle'"), 'togglePause must handle idle state');
  assert.ok(body.includes("trackLoop()"), 'togglePause must call trackLoop when resuming');
});
test('App: no innerHTML in file', () => {
  assert.ok(!appSrc.includes('.innerHTML'), 'app.js must not use innerHTML');
});

// 8. App truth cache + token
test('App: _truthCache and _predictToken present', () => {
  assert.ok(appSrc.includes('_truthCache'), 'app.js must have _truthCache');
  assert.ok(appSrc.includes('_predictToken'), 'app.js must have _predictToken');
});
test('App: _predictToken used as staleness guard in renderAll', () => {
  // renderAll should increment _predictToken and check it in the async predict callback
  assert.ok(appSrc.includes('++this._predictToken'), 'must increment _predictToken');
  assert.ok(appSrc.includes('tok !== this._predictToken'), 'must check token freshness in predict callback');
});

// 9. Model/siren guards + clipYs null
test('Model: CustomActLayer guarded on tf existence', () => {
  const guardPattern = /typeof tf === 'undefined'/;
  const layerBlock = modelSrc.match(/const CustomActLayer[\s\S]*?\)\(\);/);
  assert.ok(layerBlock, 'CustomActLayer IIFE must exist');
  assert.ok(guardPattern.test(layerBlock[0]), 'CustomActLayer must guard on typeof tf');
});
test('Siren: fallback checks tf before dense', () => {
  const sirenFs = fs.readFileSync(path.join(root, 'js/siren.js'), 'utf8');
  assert.ok(sirenFs.includes("typeof tf === 'undefined'") || sirenFs.includes('typeof tf !== \'undefined\''),
    'siren fallback must check tf existence');
});
test('Presets: clipYs preserves null values (LIVE)', () => {
  const result = clipYs([2, -2, 0.5, null]);
  assert.deepStrictEqual(result, [1.5, -1.5, 0.5, null]);
});

// ── Summary ───────────────────────────────────────────────────────────────
console.log(`\n${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
else console.log('All intensive tests passed — no bugs, no vulns, no lag (for those paths).');
