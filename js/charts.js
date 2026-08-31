// Chart.js charts — prediction + log loss. Fixed data mapping and log-scale guards.

const Charts = (() => {
  let predChart = null;
  let lossChart = null;

  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function init(canvasPred, canvasLoss) {
    if (!canvasPred || !canvasLoss) return;
    // destroy existing if re-init (e.g. theme toggle)
    if (predChart) { try { predChart.destroy(); } catch (_) {} predChart = null; }
    if (lossChart) { try { lossChart.destroy(); } catch (_) {} lossChart = null; }

    const isDark = document.documentElement.dataset.theme === 'dark';
    const accent = cssVar('--accent', '#2563eb');
    const text = cssVar('--text', isDark ? '#e8e6e1' : '#1c1c1a');
    const muted = cssVar('--muted', '#77776f');
    const border = cssVar('--border', '#d8d8d2');
    const gridCol = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)';

    predChart = new Chart(canvasPred.getContext('2d'), {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Ground Truth',
            data: [],
            borderColor: muted,
            borderDash: [6, 4],
            borderWidth: 1.8,
            pointRadius: 0,
            fill: false,
            tension: 0.15,
            parsing: false,
          },
          {
            label: 'Prediction',
            data: [],
            borderColor: accent,
            borderWidth: 2.2,
            pointRadius: 0,
            fill: false,
            tension: 0.15,
            parsing: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 120 },
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: muted, boxWidth: 14, padding: 14, font: { size: 11 } } },
          tooltip: {
            backgroundColor: cssVar('--panel', '#fff'),
            borderColor: border,
            borderWidth: 1,
            titleColor: text,
            bodyColor: text,
            callbacks: {
              title: (items) => items.length ? `x = ${Number(items[0].parsed.x).toFixed(3)}` : '',
            },
          },
        },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'x  ∈  [-1, 1]', color: muted, font: { size: 10 } },
            grid: { color: gridCol },
            ticks: { color: muted, maxTicksLimit: 9 },
            min: -1, max: 1,
          },
          y: {
            title: { display: true, text: 'f(x)', color: muted, font: { size: 10 } },
            grid: { color: gridCol },
            ticks: { color: muted },
            min: -1.6, max: 1.6,
          },
        },
      },
    });

    lossChart = new Chart(canvasLoss.getContext('2d'), {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'MSE',
            data: [],
            borderColor: '#f59e0b',
            borderWidth: 1.8,
            pointRadius: 0,
            fill: false,
            tension: 0.2,
            parsing: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 120 },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: cssVar('--panel', '#fff'),
            borderColor: border,
            borderWidth: 1,
            titleColor: text,
            bodyColor: text,
          },
        },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Epoch', color: muted, font: { size: 10 } },
            grid: { color: gridCol },
            ticks: { color: muted, maxTicksLimit: 8 },
          },
          y: {
            type: 'logarithmic',
            title: { display: true, text: 'MSE (log)', color: muted, font: { size: 10 } },
            grid: { color: gridCol },
            ticks: {
              color: muted,
              callback: (v) => Number(v).toExponential(0),
            },
          },
        },
      },
    });
  }

  function setPrediction(gtXs, gtYs, predXs, predYs) {
    if (!predChart) return;
    // use {x,y} so pred and gt can have different sampling
    const gt = (gtXs || []).map((x, i) => ({ x, y: gtYs[i] }));
    const pr = (predXs || []).map((x, i) => ({ x, y: predYs[i] })).filter(p => Number.isFinite(p.y));
    predChart.data.datasets[0].data = gt;
    predChart.data.datasets[1].data = pr;
    // Chart.js with parsing:false expects update without re-parsing; just update
    predChart.update('none');
  }

  function setLoss(hist) {
    if (!lossChart) return;
    const pts = (hist || [])
      .filter(h => Number.isFinite(h.loss) && h.loss > 0)
      .map(h => ({ x: h.epoch, y: Math.max(h.loss, 1e-6) }));
    // if all losses are zero, show flat line at 1e-6 instead of log(0) crash
    if (!pts.length && hist && hist.length) {
      pts.push({ x: hist[hist.length-1].epoch, y: 1e-6 });
    }
    lossChart.data.datasets[0].data = pts;
    lossChart.update('none');
  }

  function resize() {
    if (predChart) try { predChart.resize(); } catch (_) {}
    if (lossChart) try { lossChart.resize(); } catch (_) {}
  }

  function destroy() {
    if (predChart) try { predChart.destroy(); } catch (_) {}
    if (lossChart) try { lossChart.destroy(); } catch (_) {}
    predChart = null; lossChart = null;
  }

  return { init, setPrediction, setLoss, resize, destroy };
})();
