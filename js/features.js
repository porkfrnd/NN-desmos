// Input embeddings — the "make small nets learn fast" trick.
//
// Pure functions, no tf: given the model config we return a feature mapper
// x -> [f0..fn] and the input dim it implies. Training tensors, predictions
// and the model's input shape ALL go through here, so they can never drift
// apart (they used to: one side clamped fourierN, the other didn't).
//
// Why domain-aware: Fourier/Chebyshev features assume inputs in [-1,1].
// Train on [-2,2] and raw x makes T_6 hit ~1351 — the net spends forever
// taming huge numbers instead of learning. So x is rescaled to [-1,1] over
// the *train* range first. Identity when the range already is [-1,1]; eval
// points outside the range extrapolate naturally (|u| > 1).

const Features = (() => {
  function clampInt(v, lo, hi, dflt) {
    const n = parseInt(v, 10);
    if (!isFinite(n)) return dflt;
    return Math.max(lo, Math.min(hi, n));
  }

  // one normalization for everyone (slider ranges live in index.html; these
  // bounds are the safety net for programmatic/URL values)
  function normalizedConfig(m) {
    m = m || {};
    let sigma = m.fourierSigma;
    if (typeof sigma !== 'number' || !isFinite(sigma) || sigma <= 0) sigma = 1;
    return {
      embedding: m.embedding || (m.fourierFeatures ? 'fourier' : 'none'),
      fourierN: clampInt(m.fourierN ?? 3, 0, 6, 3),
      fourierSigma: sigma,
      chebyshevDegree: clampInt(m.chebyshevDegree ?? 6, 1, 16, 6),
    };
  }

  function featureDim(modelCfg) {
    const c = normalizedConfig(modelCfg);
    if (c.embedding === 'fourier') return 2 * (c.fourierN + 1);
    if (c.embedding === 'chebyshev') return c.chebyshevDegree + 1;
    return 1;
  }

  function buildFeatureFn(modelCfg, domain) {
    const c = normalizedConfig(modelCfg);
    if (c.embedding === 'none') return (x) => [x];

    // rescale x from [trainMin, trainMax] to [-1, 1]
    const d = domain || { trainMin: -1, trainMax: 1 };
    const mid = (d.trainMin + d.trainMax) / 2;
    const half = Math.max(1e-9, (d.trainMax - d.trainMin) / 2);
    const u = (x) => (x - mid) / half;

    if (c.embedding === 'fourier') {
      const N = c.fourierN, sigma = c.fourierSigma;
      return (x) => {
        const t = u(x), out = [];
        for (let k = 0; k <= N; k++) {
          const f = (1 << k) * Math.PI * sigma;
          out.push(Math.sin(f * t), Math.cos(f * t));
        }
        return out;
      };
    }

    // Chebyshev T_0..T_deg over the rescaled input
    const deg = c.chebyshevDegree;
    return (x) => {
      const t = u(x);
      const out = [1];
      let t0 = 1, t1 = t;
      if (deg >= 1) out.push(t1);
      for (let n = 2; n <= deg; n++) {
        const tn = 2 * t * t1 - t0;
        out.push(tn);
        t0 = t1; t1 = tn;
      }
      return out;
    };
  }

  return { buildFeatureFn, featureDim };
})();
