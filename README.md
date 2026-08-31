# NN Desmos

I was messing with Desmos and neural nets at 2am and thought — what if the curve could *learn*?

Type `y = x^2 + 6*x` and watch a tiny network try to copy it. Live. In your browser. No backend.

<p>
  <a href="https://github.com/porkfrnd/NN-desmos"><b>Try it →</b></a> — just open <code>index.html</code>
</p>

---

### What is this

Desmos lets you plot `y = sin(x)`. This lets you *fit* `y = sin(x)` with a neural net and see where it fails.

- **Type like Desmos.** `x^2`, `sin(2*pi*x)`, `exp(-x)*cos(4*pi*x)`, `²³π` all work. `y =` is optional.
- **The graph is the app.** Most of your screen is the plot — ground truth (dashed), prediction (solid), the faint trail from 10 epochs ago, and the actual training dots. Zoom, pan, pinch. Train region is shaded.
- **Tweak like a lab.** Layers, neurons, `tanh`/`ReLU`/`GELU`/`SIREN`, Fourier vs Chebyshev embeddings, `ω₀`, learning rate, weight decay, noise. All in the gear menu. One-click presets for *Smooth*, *Periodic*, *Step*.
- **See extrapolation.** Set train on `[-1,1]`, test on `[-2,2]` — watch it hallucinate outside.
- **All local.** TensorFlow.js, Chart.js, vanilla JS. `python3 -m http.server` and open. Share via URL hash.

I built this because most NN visualizers hide the math. This one doesn't.

---

### Quick start

```bash
git clone https://github.com/porkfrnd/NN-desmos
cd NN-desmos
python3 -m http.server 8000
# http://localhost:8000
```

Or just double-click `index.html`. No `npm install`.

---

### Try these

| Type this | Pick preset | Then |
|---|---|---|
| `x^3` | Smooth | Watch Chebyshev nail it while Fourier wiggles |
| `sin(10*pi*x)` | Periodic | Needs Fourier + SIREN, otherwise it just gives up |
| `sign(sin(2*pi*x))` | Step | See a shallow net struggle with a discontinuity |
| `x^2` + add noise `σ=0.1` | — | See it overfit, then turn on weight decay `1e-4` |

Shortcuts: `Space` start/pause, `R` reset, `,` step, `?` help.

---

### How it works (no magic)

```
Equation → 100 points over train range → clip to [-1.5,1.5]
         → Fourier / Chebyshev embedding (or not)
         → 1-5 dense layers (tanh/ReLU/SiLU/GELU/SIREN with ω₀)
         → manual training loop: optimizer.minimize(mse + wd·‖W‖²)
         → every 10 epochs: predict over eval range → update graph + loss (log scale)
```

Why manual loop, not `model.fit()`? `fit()` can't pause mid-epoch. We check `isPaused` each epoch and `await tf.nextFrame()` every 2 so the UI stays 60fps.

Why `Store` without `JSON.stringify` on large arrays? That was the lag. Reference check is enough.

See `js/model.js` and `js/store.js` — each file starts with a *why* comment.

---

### Stack

TensorFlow.js 4.22 + Chart.js 4.4 + vanilla HTML/CSS/JS. No bundler.
Fonts: Inter + JetBrains Mono. Colors + grain borrowed from my [snake game](https://github.com/porkfrnd/snake-game).

```
index.html        # hero graph + equation + toolbar + gear modal
css/              # tokens, base, layout, components, graph, equation, modal, toast
js/store.js       # pub/sub, no JSON lag
js/equation.js    # safe Math.* parser (allow-list, no eval)
js/model.js       # training loop, embeddings, L2
js/charts.js      # dots, trail, train shading, zoom
js/app.js         # wiring, URL hash, shortcuts
tests/run.js      # 28 tests (node tests/run.js)
```

---

### Testing

```bash
node tests/run.js
# 28 passed
# Try: empty, no x, `x; alert(1)` (blocked), every preset, every activation,
# Fourier/Chebyshev, zoom, theme, export, URL, shortcuts
```

---

### Gallery

Click these in the app:

- `sin(3*pi*x) * exp(-0.5*x)` — wave packet
- `x^3 - 2*x` — cubic
- `abs(x)` — cusp
- `tanh(5*x)` — smooth step

---

### Roadmap

I want to add: multi-model duel (ReLU vs SIREN side-by-side), PyTorch code export that actually runs, and a gallery of community functions. PRs welcome — keep it vanilla, keep it 60fps.

---

Built by Binayak — a student who likes Desmos more than it likes him.

MIT. If Google made a neural net playground, it would probably look like this. Maybe.
