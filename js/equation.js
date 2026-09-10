/**
 * Equation parser — Desmos-like, safe, self-explaining.
 *
 * Why not eval: user input is transpiled to a safe `Math.*` expression
 * and built with `new Function('x', ...)` in a sandbox with only `x` and `Math`.
 * We block `;`, `constructor`, backticks, etc., via an allow-list and throw
 * on any unknown char. Supports `y=`, unicode `²π`, implicit `2x`, `^` → `**`, `pi`/`e`.
 *
 * Sampling respects the *training* range (not fixed [-1,1]) and optional
 * Gaussian noise for robustness tests.
 */
// Equation parser — Desmos-like: y = x^2 + 6*x, sin(2*pi*x), etc.
// Turns user string into a JS function f(x) over x ∈ [-1,1].
// No eval on raw input — we transpile to a safe Math.* expression and
// construct a Function with only `x` and `Math` in scope.

const Equation = (() => {
  // map of user-facing names -> Math.* (case-insensitive)
  const FN_MAP = {
    sin: 'Math.sin', cos: 'Math.cos', tan: 'Math.tan',
    asin: 'Math.asin', acos: 'Math.acos', atan: 'Math.atan',
    sinh: 'Math.sinh', cosh: 'Math.cosh', tanh: 'Math.tanh',
    exp: 'Math.exp', log: 'Math.log', ln: 'Math.log',
    sqrt: 'Math.sqrt', abs: 'Math.abs', sign: 'Math.sign',
    ceil: 'Math.ceil', floor: 'Math.floor', round: 'Math.round',
    pow: 'Math.pow', max: 'Math.max', min: 'Math.min',
  };

  function normalize(src) {
    let s = String(src || '').trim();
    // strip leading "y=" / "y =" / "f(x)=" etc.
    s = s.replace(/^\s*(y|f\s*\(\s*x\s*\))\s*=\s*/i, '');
    // unicode niceties
    s = s.replace(/²/g, '^2').replace(/³/g, '^3')
         .replace(/π/g, 'pi').replace(/Π/g, 'pi')
         .replace(/÷/g, '/').replace(/×/g, '*')
         .replace(/·/g, '*').replace(/—/g, '-').replace(/−/g, '-');
    return s.trim();
  }

  function transpile(src) {
    let s = normalize(src);
    if (!s) throw new Error('Empty equation. Try e.g.  x^2 + 6*x  or  sin(2*pi*x)');

    // ^ -> **  (do before fn replacement so we don't mangle)
    s = s.replace(/\^/g, '**');

    // scientific notation FIRST — `1e-3*x` must become (1*10**(-3))*x before
    // the implicit-mult pass below splits it into `1*e - 3*x` (a real curve,
    // silently wrong — the worst kind of bug). `2e` (no exponent digits) is
    // left alone so it still means 2·e.
    s = s.replace(/((?:\d*\.)?\d+)[eE]([+-]?\d+)/g, '($1*10**($2))');

    // implicit multiplication: 2x -> 2*x, 2pi -> 2*pi, )x -> )*x, )( -> )*(
    // also 3sin -> 3*sin  — but only when not part of a longer name
    s = s.replace(/(\d)\s*(?=[a-zA-Z\(])/g, '$1*');
    s = s.replace(/(\))\s*(?=[a-zA-Z0-9\(])/g, '$1*');
    // avoid double-replacing if user already typed Math.PI
    s = s.replace(/(?<!Math\.)\bpi\b/gi, 'Math.PI');
    // e as constant: standalone e not part of exp/sqrt etc.
    s = s.replace(/(?<![a-zA-Z0-9_])e(?![a-zA-Z0-9_])/g, 'Math.E');

    // function names -> Math.*
    // sort by length descending so `asin` before `sin`
    const fns = Object.keys(FN_MAP).sort((a,b)=>b.length-a.length);
    for (const k of fns) {
      const re = new RegExp('\\b' + k + '\\b', 'gi');
      s = s.replace(re, FN_MAP[k]);
    }

    // allow only safe chars after transpilation:
    // strip the Math.* tokens we emit, then everything left must be
    // digits / x / operators / parens — anything else is rejected
    const stripped = s.replace(/Math\.(PI|E|sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|exp|log|sqrt|abs|sign|ceil|floor|round|pow|max|min)/g, '');
    if (!/^[0-9x+\-*/%()., \t]+$/.test(stripped)) {
      // collect offending chars for a nicer error
      const bad = stripped.match(/[^0-9x+\-*/%()., \t]/g);
      if (bad) throw new Error('Invalid characters: ' + [...new Set(bad)].join(' '));
    }

    // must contain x
    if (!/\bx\b/.test(s)) throw new Error('Equation must contain x — e.g.  x^2 + 6*x');

    return s;
  }

  function compile(src) {
    const expr = transpile(src);
    let fn;
    try {
      fn = new Function('x', '"use strict"; return (' + expr + ');');
    } catch (e) {
      throw new Error('Could not parse equation: ' + e.message);
    }
    // smoke test — a few non-finite probes are fine (log(0), 1/(x-1)…),
    // but if NO probe is finite the equation can't be plotted at all
    let finite = 0;
    for (const x of [-1, 0, 0.5, 1]) {
      let y;
      try { y = fn(x); } catch (e) { throw new Error('Error at x=' + x + ': ' + e.message); }
      if (typeof y !== 'number') throw new Error('Equation must return a number');
      if (isFinite(y)) finite++;
    }
    if (finite === 0) throw new Error('Never returns a finite value on [-1,1] — e.g. log(x) needs x > 0');
    return { fn, expr, src: normalize(src) };
  }

  // self-explaining: Gaussian noise via Box-Muller, added only if training.noise > 0
  function gaussianNoise(std) {
    if (std <= 0) return 0;
    let u=0, v=0;
    while(u===0) u=Math.random();
    while(v===0) v=Math.random();
    return Math.sqrt(-2*Math.log(u)) * Math.cos(2*Math.PI*v) * std;
  }
  function sample(compiled, count = 100, trainMin = -1, trainMax = 1, noiseStd = 0) {
    const fn = compiled.fn;
    const xs = [], ys = [];
    for (let i = 0; i < count; i++) {
      const x = trainMin + (trainMax - trainMin) * i / (count - 1);
      let y;
      try { y = fn(x); } catch (_) { y = NaN; }
      // NaN -> 0 (a "gap" that reads as the axis), ±Inf -> clip to the plot box
      if (!isFinite(y)) y = Number.isNaN(y) ? 0 : Math.sign(y) * 1.5;
      else if (noiseStd > 0) y += gaussianNoise(noiseStd);
      xs.push(x);
      ys.push(y);
    }
    return { xs, ys: clipYs(ys) };
  }

  function sampleString(src, count = 100, trainMin = -1, trainMax = 1, noiseStd = 0) {
    const c = compile(src);
    return { compiled: c, ...sample(c, count, trainMin, trainMax, noiseStd) };
  }

  return { normalize, transpile, compile, sample, sampleString };
})();
