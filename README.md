# NN · Desmos — Neural Network Function Approximator

Type an equation like Desmos — `y = x^2 + 6*x` — and watch a tiny neural network learn it live, entirely in your browser.

> **No build step.** Just open `index.html` or serve the folder. All dependencies come from CDNs.

🔗 **Repo:** https://github.com/porkfrnd/NN-desmos

---

## Demo

```bash
git clone https://github.com/porkfrnd/NN-desmos.git
cd NN-desmos

# static server (recommended)
python3 -m http.server 8000
# → http://localhost:8000

# or open directly
open index.html
```

No `npm install`.

---

## What it does

| Area | Details |
|---|---|
| **Equation (Desmos-like)** | Input `y =` with live parsing: `x^2`, `sin(2*pi*x)`, `exp(-x)*cos(4*pi*x)`, `sign(sin(2*pi*x))`, `^`/`**`, `pi`/`π`, `e`, `²³`, implicit `2x→2*x`. Sampled 100 pts over **Training Range**, clipped to `[-1.5,1.5]` (`js/equation.js:1`). |
| **Presets** | Function presets: Sine, Square, Damped, Composite (`js/presets.js:1`). One-click **Tuning Presets**: *Smooth / Polynomial* (GELU, no Fourier, low LR), *Periodic / High Freq* (Fourier + SIREN), *Step* (deep ReLU) (`js/presets.js:40`). |
| **Domain & Extrapolation** | Separate **Training Range** (e.g. `[-1,1]`) and **Evaluation Range** (e.g. `[-2,2]`) in Settings. Training data sampled over train range; chart shows truth + prediction over eval range with train region shaded — visually test out-of-distribution (`js/store.js:domain`, `js/charts.js:trainShade`). |
| **Embeddings** | `None` / **Fourier** (`x → [sin(2^k·π·σ·x), cos(...)]` for `k=0..N`, `N=0..5`, `σ=0.5..5`) / **Chebyshev** (`T₀..T_N`, `N=3..12`) for smooth polynomials like `x^3` (`js/model.js:buildFeatureFn`). |
| **Architecture** | 1–5 hidden layers, 2–64 neurons/layer. Activations: **ReLU, tanh, sigmoid, Softplus, SiLU/Swish, GELU, sin (SIREN)**. Custom SiLU `x·sigmoid(x)` and GELU `0.5·x·(1+tanh(√(2/π)(x+0.0447x³)))` via custom layers (`js/model.js:applyCustomActivation`). |
| **SIREN** | Custom `SirenDense` layer (`sin` not built-in). Init: first layer `U(-1/fan_in,1/fan_in)`, deeper `U(-√(6/fan_in)/ω₀,√(6/fan_in)/ω₀)`. **ω₀ slider 1–30** (default 30 for raw, 1 for embedded) revealed when SIREN selected (`js/siren.js:1`, `index.html:omegaRow`). |
| **Hyperparams** | **Learning Rate** log slider `1e-4–1e-1` (default `1e-3`) (`index.html:learningRate`). **Weight Decay (L2)** slider `0–1e-2` (0 = off, else `10⁻⁴–10⁻²` log) penalizes `‖W‖²` in loss → reduces high-frequency oscillations (`js/model.js:meanSquaredError`). Optimizer Adam/SGD, Run-to `100–2000` epochs. |
| **Training Loop** | Manual `optimizer.minimize(()=>loss,true)` inside `tf.tidy()` loop — not `model.fit()`. `isPaused` ref checked each epoch, `await tf.nextFrame()` every 5 epochs, NaN/divergence auto-pauses with toast. Weight decay added as `loss + wd·Σ‖W‖²` (`js/model.js:146`). |
| **Visualization** | **Chart.js** (vanilla replacement for Recharts). Hero **Approximation** chart (60vh, most of screen) with **zoom** ( +/− buttons, wheel, drag to pan, pinch on mobile) and train-range shading. **Loss** chart log-scale (`js/charts.js:1`). |
| **Layout** | Desmos-like: **hero graph on top**, **equation below**, **toolbar** (Start/Pause/Step/Reset/Export) underneath, **loss** below. **Settings on gear icon** top-right (modal) keeps screen clean. Warm matte tokens + grain from `snake game` (`css/style.css:1`), responsive, dark/light toggle via `localStorage`. |
| **Export** | **⤓ JSON** downloads weights & biases, **🖼 PNG** downloads the graph as image. JSON contains `meta` + per-layer `{shape,data}` (`js/model.js:exportWeights`). |
| **Noise** | Gaussian **σ 0–0.3** slider in Settings — adds `y += N(0,σ)` to training samples to test robustness vs overfitting (`js/equation.js:gaussianNoise`, `js/presets.js`). |
| **Training Dots & Trail** | Main graph shows **training points as dots** (like Desmos) and a **faint trail** from 10 epochs ago — beautiful motion of learning (`js/charts.js:datasets`). |
| **URL Sharing** | Equation + key config auto-sync to `location.hash` (e.g. `#eq=x^3&act=gelu`) — share a link like Desmos (`js/app.js:updateURLHash`). |
| **Shortcuts** | **Space** Start/Pause, **R** Reset, **,** Step, **?** help, **⌘E** Export — no mouse needed (`js/app.js:keydown`). |
| **Smooth 60fps** | No lag: `Store` avoids `JSON.stringify` on large arrays, `TRAIN_UPDATE_EVERY=10` + `await tf.nextFrame()` every 2 epochs, `Chart` `animation:false` + `normalized:true`, L2 only when `wd>0` in single tidy (`js/store.js`, `js/model.js`). |

---

## File tree

```
index.html        # hero graph (zoom/pan) + equation + presets + toolbar + gear modal
css/style.css     # design system (warm matte, grain, hero, modal, 60fps hints)
js/store.js       # pub/sub store — self-explaining, no JSON lag on large arrays
js/presets.js     # function presets + TUNING_PRESETS, sampling with noise
js/equation.js    # Desmos parser — safe Math.* transpilation, no eval
js/siren.js       # SirenDense, SIREN init with ω₀
js/model.js       # embeddings, custom activations, L2, manual training loop (60fps)
js/charts.js      # Chart.js wrappers — train dots, trail, shading, zoom
js/app.js         # wiring — URL hash, shortcuts, noise, dots, PNG export
js/canvas.js      # legacy — kept for reference
tests/run.js      # intensive unit + security tests (28 tests, no deps)
```

---

## Stack

- **TensorFlow.js** `4.22.0` via CDN
- **Chart.js** `4.4.7` via CDN
- Vanilla **HTML / CSS / JS** — no framework, no bundler
- Fonts: `Inter` + `JetBrains Mono` via Google Fonts

---

## Usage

1. **Equation:** type `x^2 + 6*x` or pick a preset (Sine etc.). Press **Plot** or Enter. Try `x^3` with *Smooth* preset, or `sin(10*pi*x)` with *Periodic*.
2. **Graph:** scroll/pinch to zoom, drag to pan, **Reset** to fit. Train shading shows where the network saw data vs. extrapolation.
3. **Settings (⚙):** Layers, Neurons, Activation (try **SiLU/GELU** for smooth, **SIREN** + **ω₀** for periodic), **Embedding** (Fourier `N`/`σ` or Chebyshev degree), **LR** (`1e-4–1e-1`), **Weight Decay** (try `1e-4` to calm oscillations), **Train/Eval ranges**, **Run to**.
4. **Tuning Presets:** *Smooth* (GELU, no Fourier), *Periodic* (Fourier+SIREN), *Step* (deep ReLU) — one click.
5. **Training:** **▶ Start**, **⏸ Pause**, **↷ Step** (10 epochs), **↺ Reset**. Loss is log-scale.
6. **Export:** **⤓ Export** downloads weights & biases JSON.

Tips:
- SIREN needs low LR with SGD; Adam is more stable.
- Fourier helps periodic but hurts smooth polynomials — use **Chebyshev** for `x^3`.
- Weight decay `1e-4` reduces wiggles in high-frequency fits.

---

## Testing — intensive, no bugs, no vulns, no lag

```bash
# unit + security (Equation, Presets, Store, injection, a11y)
node tests/run.js
# → 28 passed, 0 failed

# manual — open in browser, try:
# - Equation: x^2, x^3, sin(10*pi*x), empty, no x, `x; alert(1)` (blocked)
# - Presets: Sine/Square/Damped/Composite + Smooth/Periodic/Step
# - Training: Start/Pause/Step/Reset, 1–5 layers, all activations, Fourier/Chebyshev, ω₀, LR, WD, noise, train/eval ranges
# - Graph: zoom (+/−/wheel/drag/pinch), reset, train shading, dots, trail
# - Settings gear, theme, export JSON/PNG, URL hash, shortcuts (Space/R/,/⌘E)
```

All 28 unit tests pass. Manual tests cover every control, edge case, and security injection (see `tests/run.js` for the brutal audit).

---

## Code — beautiful & self-explaining

Every file starts with a **why** comment, every function has **JSDoc** (`@param`/`@returns`), and the architecture is pure functions + tiny pub/sub — no framework, no build, no `npm install`. See `js/store.js` (“why we avoid JSON on large arrays”) and `js/model.js` (“why manual loop, not fit()”).

---

## License

MIT
