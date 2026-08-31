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
| **Export** | **⤓ Export** downloads JSON with `meta` (architecture, training, domain, equation, epochs, loss) and per-layer `kernel`/`bias` `{shape,data}` — re-import or inspect (`js/model.js:exportWeights`). |

---

## File tree

```
index.html        # hero graph + equation + presets + toolbar + gear modal, CDN tags
css/style.css     # tokens, hero graph, equation, zoom controls, gear modal, grain
js/store.js       # pub/sub store: data, predictions, lossHistory, model, training, domain, run
js/presets.js     # function presets + TUNING_PRESETS (smooth/periodic/step), sampling
js/equation.js    # Desmos parser: normalize → transpile → Function, sampling
js/siren.js       # SirenDense + SineActivation, SIREN init with ω₀
js/model.js       # embeddings (Fourier/Chebyshev), activations (SiLU/GELU/softplus), L2, buildModel, training
js/charts.js      # Chart.js wrappers, train shading, zoom/pan (wheel/drag/pinch), view state
js/app.js         # wiring: presets, equation live preview, architecture, hyperparams, domain, tuning, controls, export
js/canvas.js      # legacy (kept, not loaded) — was freehand board before equation
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

## License

MIT
