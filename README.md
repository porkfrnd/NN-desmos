# NN-Desmos — Neural Network Function Approximation Visualizer

Interactive single-page app that trains a small neural network **entirely in the browser** (TensorFlow.js) to approximate arbitrary 1D functions. Draw a curve or pick a preset, tweak the architecture, and watch the network learn in real time.

> **No build step.** Just open `index.html` or serve the folder statically. All dependencies come from CDNs.

🔗 **Repo:** https://github.com/porkfrnd/NN-desmos

---

## Demo

```
python3 -m http.server 8000
# open http://localhost:8000
```

Or open `index.html` directly — no `npm install`, no bundler.

---

## What it does

| Area | Details |
|---|---|
| **Domain** | Fixed `x ∈ [-1, 1]`, shared by presets, canvas, and model input. `y` clipped to `[-1.5, 1.5]` for stable training. |
| **Presets** | 100 evenly-spaced samples: `sin(2πx)`, `sign(sin(2πx))`, `e^{-x}cos(4πx)`, `sin(2πx)+0.5sin(10πx)` (`js/presets.js:1`). |
| **Custom draw** | HTML5 canvas with Pointer Events (mouse + touch). Strokes resampled to 120 evenly-spaced-in-x points via linear interpolation — raw pointer points are uneven in x and would break training (`js/canvas.js:1`). |
| **Architecture** | 1–5 hidden layers, 2–64 neurons/layer, activations ReLU / Tanh / Sigmoid / **Sine (SIREN)**. Fourier features toggle: `x → [sin(2^k·πx), cos(2^k·πx)]` for `k=0..3` (8 features). SIREN and Fourier features can be combined (`js/model.js:31`, `js/siren.js:1`). |
| **SIREN** | Custom `SirenDense` layer — `sin` is not a built-in tf.js dense activation. SIREN-specific init: first layer `U(-1/fan_in, 1/fan_in)`, deeper layers `U(-√(6/fan_in)/ω₀, √(6/fan_in)/ω₀)` with `ω₀=30`. Without this the sine network visibly fails to train. |
| **Training loop** | Manual loop with `optimizer.minimize(() => loss, true)` — **not** `model.fit()`. An `isPaused` ref is checked each epoch so Pause halts between epochs. Every 5 epochs: `await tf.nextFrame()`, update prediction curve + loss chart. Every tensor op wrapped in `tf.tidy()`; model weights tracked separately from scratch tensors. NaN/divergence auto-pauses with a toast (`js/model.js:133`). |
| **Hyperparams** | Learning rate `10⁻³–10⁻¹` (log-scale slider), optimizer Adam/SGD, run-to `100–2000` epochs/chunk. Changing architecture or activation auto-resets weights. |
| **Visualization** | `recharts` was the spec; this build uses **Chart.js** (Recharts requires React, this app is vanilla JS). Prediction chart: dashed ground-truth + solid prediction. Loss chart: MSE vs. epoch, log-scale y. Recharts-style cadence-driven animation, no CSS transition fights. |
| **Layout** | Dark dashboard, left scrollable control panel / right sticky plots, stacks vertically below `md` breakpoint (`css/style.css:1`). |

---

## File tree

```
index.html          # layout, CDN script tags (tfjs, Chart.js), module order
css/style.css       # dark theme, dashboard grid, controls, charts, toasts
js/store.js         # tiny pub/sub store (dataset, model config, training state)
js/presets.js       # preset definitions, sampling, clipping
js/siren.js         # SirenDense + SineActivation, SIREN weight init
js/model.js         # model construction, feature transform, training loop, predictions
js/canvas.js        # pointer-event drawing + resampling
js/charts.js        # Chart.js prediction + loss chart wrappers
js/app.js           # wiring, preset/canvas/arch/hyperparam UI, Start/Pause/Step/Reset
```

---

## Stack

- **TensorFlow.js** `4.22.0` via CDN (`@tensorflow/tfjs`)
- **Chart.js** `4.4.7` via CDN (vanilla-JS replacement for `recharts`)
- Vanilla **HTML / CSS / JS** — no framework, no bundler, no `package.json`
- Fonts: `Inter` + `JetBrains Mono` via Google Fonts

---

## Setup

```bash
git clone https://github.com/porkfrnd/NN-desmos.git
cd NN-desmos

# Option A — static server (recommended, avoids file:// CORS quirks)
python3 -m http.server 8000
# → http://localhost:8000

# Option B — open directly
open index.html          # macOS
xdg-open index.html      # Linux
```

No `npm install`. No `npm run dev`.

---

## Usage

1. Pick a preset (**Sine**, **Square**, **Damped**, **Composite**) or draw on the canvas.
2. Adjust **Hidden Layers**, **Neurons / Layer**, **Activation**, **Fourier Features**.
3. Set **Learning Rate** (log scale), **Optimizer**, **Run to** epochs/chunk.
4. **Start** — watch the prediction converge. **Pause** to freeze between epochs, **Step** to advance 10 epochs, **Reset** to re-initialize weights.

Tips:
- SIREN (`Sine`) shines on high-frequency targets (Square, Composite) but needs a low LR with SGD.
- Fourier features help ReLU/Tanh networks on periodic functions.
- If loss turns `NaN` or explodes, the run auto-pauses — lower the learning rate or switch to Adam.

---

## License

MIT
