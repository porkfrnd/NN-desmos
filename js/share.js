// Share links — the URL hash is the whole "save file".
//
// Pure encode/decode/sanitize (no DOM) so the round trip is testable in node.
// app.js just wires these to Store. Everything that defines a run lives in
// the hash: equation, architecture, optimizer knobs, domain — a link must
// open exactly what the sender saw. (It used to write 8 params and read back
// 2, so "look at SIREN nail this" links opened as a plain tanh net.)

const Share = (() => {
  function encode(state) {
    const p = new URLSearchParams();
    const d = state.data || {}, m = state.model || {}, t = state.training || {}, dom = state.domain || {};
    if (d.equation) p.set('eq', d.equation);
    if (d.presetId) p.set('preset', d.presetId);
    if (m.activation) p.set('act', m.activation);
    if (m.embedding) p.set('emb', m.embedding);
    if (m.hiddenLayers != null) p.set('hl', String(m.hiddenLayers));
    if (m.neuronsPerLayer != null) p.set('np', String(m.neuronsPerLayer));
    if (m.fourierN != null) p.set('fn', String(m.fourierN));
    if (m.fourierSigma != null) p.set('fs', String(m.fourierSigma));
    if (m.chebyshevDegree != null) p.set('cd', String(m.chebyshevDegree));
    if (m.omega0 != null) p.set('om', String(m.omega0));
    if (t.learningRate != null) p.set('lr', String(t.learningRate));
    if (t.weightDecay != null) p.set('wd', String(t.weightDecay));
    if (t.noise != null) p.set('noise', String(t.noise));
    if (t.optimizer) p.set('opt', t.optimizer);
    if (dom.trainMin != null) p.set('train', dom.trainMin + ',' + dom.trainMax);
    if (dom.evalMin != null) p.set('eval', dom.evalMin + ',' + dom.evalMax);
    return p.toString();
  }

  function num(v, dflt) {
    const n = parseFloat(v);
    return isFinite(n) ? n : dflt;
  }

  function decode(hash) {
    if (!hash) return null;
    const s = hash.charAt(0) === '#' ? hash.slice(1) : hash;
    if (!s) return null;
    const p = new URLSearchParams(s);
    const out = { data: {}, model: {}, training: {} };
    const eq = p.get('eq'); if (eq) out.data.equation = eq;
    const preset = p.get('preset'); if (preset) out.data.presetId = preset;
    const act = p.get('act'); if (act) out.model.activation = act;
    const emb = p.get('emb'); if (emb) out.model.embedding = emb;
    if (p.get('hl') != null) out.model.hiddenLayers = num(p.get('hl'), 3);
    if (p.get('np') != null) out.model.neuronsPerLayer = num(p.get('np'), 16);
    if (p.get('fn') != null) out.model.fourierN = num(p.get('fn'), 3);
    if (p.get('fs') != null) out.model.fourierSigma = num(p.get('fs'), 1);
    if (p.get('cd') != null) out.model.chebyshevDegree = num(p.get('cd'), 6);
    if (p.get('om') != null) out.model.omega0 = num(p.get('om'), 30);
    if (p.get('lr') != null) out.training.learningRate = num(p.get('lr'), 0.001);
    if (p.get('wd') != null) out.training.weightDecay = num(p.get('wd'), 0);
    if (p.get('noise') != null) out.training.noise = num(p.get('noise'), 0);
    const opt = p.get('opt'); if (opt) out.training.optimizer = opt;
    const tr = p.get('train'), ev = p.get('eval');
    if (tr) {
      const a = tr.split(',');
      if (a.length === 2) out.domain = { trainMin: num(a[0], -1), trainMax: num(a[1], 1), evalMin: -2, evalMax: 2 };
    }
    if (ev) {
      const a = ev.split(',');
      if (a.length === 2) {
        out.domain = out.domain || { trainMin: -1, trainMax: 1 };
        out.domain.evalMin = num(a[0], -2);
        out.domain.evalMax = num(a[1], 2);
      }
    }
    const known = ['eq', 'preset', 'act', 'emb', 'hl', 'np', 'fn', 'fs', 'cd', 'om', 'lr', 'wd', 'noise', 'opt', 'train', 'eval'];
    if (!known.some((k) => p.get(k) != null)) return null;
    return out;
  }

  // links can come from anywhere (old versions, hand-edits) — drop anything
  // the UI couldn't have produced, clamp the rest to slider ranges
  function sanitize(state) {
    if (!state) return null;
    const ACTS = ['tanh', 'relu', 'sigmoid', 'softplus', 'silu', 'gelu', 'sine'];
    const EMBS = ['none', 'fourier', 'chebyshev'];
    const OPTS = ['adam', 'sgd'];
    const clamp = (v, lo, hi) => (typeof v === 'number' && isFinite(v)) ? Math.max(lo, Math.min(hi, v)) : undefined;

    const m = {};
    const rawM = state.model || {};
    if (rawM.activation && ACTS.includes(rawM.activation)) m.activation = rawM.activation;
    if (rawM.embedding && EMBS.includes(rawM.embedding)) m.embedding = rawM.embedding;
    const hl = clamp(rawM.hiddenLayers, 1, 5); if (hl !== undefined) m.hiddenLayers = Math.round(hl);
    const np = clamp(rawM.neuronsPerLayer, 2, 64); if (np !== undefined) m.neuronsPerLayer = Math.round(np);
    const fn = clamp(rawM.fourierN, 0, 5); if (fn !== undefined) m.fourierN = Math.round(fn);
    const fs = clamp(rawM.fourierSigma, 0.5, 5); if (fs !== undefined) m.fourierSigma = fs;
    const cd = clamp(rawM.chebyshevDegree, 3, 12); if (cd !== undefined) m.chebyshevDegree = Math.round(cd);
    const om = clamp(rawM.omega0, 1, 30); if (om !== undefined) m.omega0 = om;

    const t = {};
    const rawT = state.training || {};
    const lr = clamp(rawT.learningRate, 1e-4, 1e-1); if (lr !== undefined) t.learningRate = lr;
    const wd = clamp(rawT.weightDecay, 0, 1e-2); if (wd !== undefined) t.weightDecay = wd;
    const noise = clamp(rawT.noise, 0, 0.3); if (noise !== undefined) t.noise = noise;
    if (rawT.optimizer && OPTS.includes(rawT.optimizer)) t.optimizer = rawT.optimizer;

    let domain = null;
    const d = state.domain;
    if (d && isFinite(d.trainMin) && isFinite(d.trainMax) && isFinite(d.evalMin) && isFinite(d.evalMax)
        && d.trainMin < d.trainMax && d.evalMin < d.evalMax) {
      domain = { trainMin: d.trainMin, trainMax: d.trainMax, evalMin: d.evalMin, evalMax: d.evalMax };
    }

    return { data: state.data || {}, model: m, training: t, domain };
  }

  return { encode, decode, sanitize };
})();
