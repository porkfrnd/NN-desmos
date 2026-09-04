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

    // implicit multiplication: 2x -> 2*x, 2pi -> 2*pi, )x -> )*x, )( -> )*(
    // also 3sin -> 3*sin — but only when not part of a longer name.
    // Must NOT break scientific notation like 1e-3 (so 1e-3 stays 1e-3, not 1*e-3).
    s = s.replace(/(\d)\s*(?=(?![eE][-+]?\d)[a-zA-Z\(])/g, '$1*');
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

    // allow only safe chars after transpilation — strip Math.* then check remainder
    // include e/E for scientific notation like 1e-3
    const stripped = s.replace(/Math\.(PI|E|sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|exp|log|sqrt|abs|sign|ceil|floor|round|pow|max|min)/g, '');
    if (!/^[0-9x+\-*/%()., \t*eE]+$/.test(stripped)) {
      const bad = stripped.match(/[^0-9x+\-*/%()., \t*eE]/g);
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
    // smoke test — just ensure it doesn't throw for a few sample points;
    // non-finite values are allowed (e.g. 1/(x-1) at x=1) and handled at sample time
    for (const x of [-1, 0, 0.5, 1]) {
      try { fn(x); } catch (e) { throw new Error('Error at x=' + x + ': ' + e.message); }
    }
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
      if (!isFinite(y)) {
        // explicit NaN handling: leave a gap in the chart (null) instead of a fake 0
        if (Number.isNaN(y)) y = null;
        else y = Math.sign(y) * 1.5;
      } else if (noiseStd > 0) {
        y += gaussianNoise(noiseStd);
      }
      xs.push(x);
      ys.push(y);
    }
    // clip, but preserve null gaps
    return { xs, ys: ys.map(v => v === null ? null : Math.max(-1.5, Math.min(1.5, v))) };
  }

  function sampleString(src, count = 100, trainMin = -1, trainMax = 1, noiseStd = 0) {
    const c = compile(src);
    return { compiled: c, ...sample(c, count, trainMin, trainMax, noiseStd) };
  }

  return { normalize, transpile, compile, sample, sampleString };
})();
