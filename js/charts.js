// Chart.js charts — prediction + log loss. Desmos-like zoom/pan on main graph.

const Charts = (() => {
  let predChart = null;
  let lossChart = null;
  // view for pred chart (Desmos-like)
  let view = { xMin: -1, xMax: 1, yMin: -1.6, yMax: 1.6 };
  const initialView = { xMin: -1, xMax: 1, yMin: -1.6, yMax: 1.6 };
  let interactBound = false;

  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function init(canvasPred, canvasLoss) {
    if (!canvasPred || !canvasLoss) return;
    if (predChart) { try { predChart.destroy(); } catch (_) {} predChart = null; }
    if (lossChart) { try { lossChart.destroy(); } catch (_) {} lossChart = null; }
    // keep view across theme toggles if already zoomed; otherwise reset
    // (if view is still initial, keep it; else keep current)
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
          { label: 'Ground Truth', data: [], borderColor: muted, borderDash: [6, 4], borderWidth: 1.8, pointRadius: 0, fill: false, tension: 0.15, parsing: false },
          { label: 'Prediction', data: [], borderColor: accent, borderWidth: 2.2, pointRadius: 0, fill: false, tension: 0.15, parsing: false },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 120 },
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: muted, boxWidth: 14, padding: 14, font: { size: 11 } } },
          tooltip: {
            backgroundColor: cssVar('--panel', '#fff'), borderColor: border, borderWidth: 1,
            titleColor: text, bodyColor: text,
            callbacks: { title: (items) => items.length ? `x = ${Number(items[0].parsed.x).toFixed(3)}` : '' },
          },
        },
        scales: {
          x: { type: 'linear', title: { display: true, text: 'x  ∈  [-1, 1]', color: muted, font: { size: 10 } }, grid: { color: gridCol }, ticks: { color: muted, maxTicksLimit: 9 }, min: view.xMin, max: view.xMax },
          y: { title: { display: true, text: 'f(x)', color: muted, font: { size: 10 } }, grid: { color: gridCol }, ticks: { color: muted }, min: view.yMin, max: view.yMax },
        },
      },
    });

    lossChart = new Chart(canvasLoss.getContext('2d'), {
      type: 'line',
      data: { datasets: [{ label: 'MSE', data: [], borderColor: '#f59e0b', borderWidth: 1.8, pointRadius: 0, fill: false, tension: 0.2, parsing: false }] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 120 },
        plugins: { legend: { display: false }, tooltip: { backgroundColor: cssVar('--panel', '#fff'), borderColor: border, borderWidth: 1, titleColor: text, bodyColor: text } },
        scales: {
          x: { type: 'linear', title: { display: true, text: 'Epoch', color: muted, font: { size: 10 } }, grid: { color: gridCol }, ticks: { color: muted, maxTicksLimit: 8 } },
          y: { type: 'logarithmic', title: { display: true, text: 'MSE (log)', color: muted, font: { size: 10 } }, grid: { color: gridCol }, ticks: { color: muted, callback: (v) => Number(v).toExponential(0) } },
        },
      },
    });

    bindInteractions(canvasPred);
    bindZoomButtons();
  }

  function setPrediction(gtXs, gtYs, predXs, predYs) {
    if (!predChart) return;
    const gt = (gtXs || []).map((x, i) => ({ x, y: gtYs[i] }));
    const pr = (predXs || []).map((x, i) => ({ x, y: predYs[i] })).filter(p => Number.isFinite(p.y));
    predChart.data.datasets[0].data = gt;
    predChart.data.datasets[1].data = pr;
    predChart.update('none');
  }

  function setLoss(hist) {
    if (!lossChart) return;
    const pts = (hist || []).filter(h => Number.isFinite(h.loss) && h.loss > 0).map(h => ({ x: h.epoch, y: Math.max(h.loss, 1e-6) }));
    if (!pts.length && hist && hist.length) pts.push({ x: hist[hist.length-1].epoch, y: 1e-6 });
    lossChart.data.datasets[0].data = pts;
    lossChart.update('none');
  }

  function resize() {
    if (predChart) try { predChart.resize(); } catch (_) {}
    if (lossChart) try { lossChart.resize(); } catch (_) {}
  }

  function destroy() {
    // keep view across destroys (theme toggle preserves zoom)
    if (predChart) try { predChart.destroy(); } catch (_) {}
    if (lossChart) try { lossChart.destroy(); } catch (_) {}
    predChart = null; lossChart = null;
    interactBound = false;
  }

  // ---- zoom / pan ----
  function applyView() {
    if (!predChart) return;
    predChart.options.scales.x.min = view.xMin;
    predChart.options.scales.x.max = view.xMax;
    predChart.options.scales.y.min = view.yMin;
    predChart.options.scales.y.max = view.yMax;
    predChart.update('none');
  }

  function zoomIn() {
    const cx = (view.xMin + view.xMax) / 2, cy = (view.yMin + view.yMax) / 2;
    const rx = (view.xMax - view.xMin) * 0.42, ry = (view.yMax - view.yMin) * 0.42;
    view.xMin = cx - rx; view.xMax = cx + rx;
    view.yMin = cy - ry; view.yMax = cy + ry;
    clampView(); applyView();
  }

  function zoomOut() {
    const cx = (view.xMin + view.xMax) / 2, cy = (view.yMin + view.yMax) / 2;
    const rx = (view.xMax - view.xMin) * 0.62, ry = (view.yMax - view.yMin) * 0.62;
    view.xMin = cx - rx; view.xMax = cx + rx;
    view.yMin = cy - ry; view.yMax = cy + ry;
    clampView(); applyView();
  }

  function resetZoom() {
    view = { ...initialView };
    applyView();
  }

  function clampView() {
    // keep view within reasonable bounds, avoid infinite zoom
    const xSpan = view.xMax - view.xMin, ySpan = view.yMax - view.yMin;
    const minSpan = 0.08, maxSpanX = 6, maxSpanY = 8;
    let cx = (view.xMin + view.xMax)/2, cy = (view.yMin + view.yMax)/2;
    let sx = xSpan, sy = ySpan;
    if (sx < minSpan) sx = minSpan;
    if (sy < minSpan) sy = minSpan;
    if (sx > maxSpanX) sx = maxSpanX;
    if (sy > maxSpanY) sy = maxSpanY;
    view.xMin = cx - sx/2; view.xMax = cx + sx/2;
    view.yMin = cy - sy/2; view.yMax = cy + sy/2;
    // keep at least a bit of the original domain visible (soft clamp)
    if (view.xMin < -3) { const d = -3 - view.xMin; view.xMin += d; view.xMax += d; }
    if (view.xMax > 3) { const d = view.xMax - 3; view.xMin -= d; view.xMax -= d; }
    if (view.yMin < -4) { const d = -4 - view.yMin; view.yMin += d; view.yMax += d; }
    if (view.yMax > 4) { const d = view.yMax - 4; view.yMin -= d; view.yMax -= d; }
  }

  function bindZoomButtons() {
    const zi = document.getElementById('zoomIn'), zo = document.getElementById('zoomOut'), zr = document.getElementById('zoomReset');
    if (zi && !zi._bound) { zi.addEventListener('click', zoomIn); zi._bound = true; }
    if (zo && !zo._bound) { zo.addEventListener('click', zoomOut); zo._bound = true; }
    if (zr && !zr._bound) { zr.addEventListener('click', resetZoom); zr._bound = true; }
  }

  function bindInteractions(canvas) {
    if (!canvas || interactBound) return;
    interactBound = true;
    const box = canvas.parentElement;
    if (box) box.style.touchAction = 'none';

    // wheel zoom centered at cursor
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) / rect.width;
      const my = 1 - (e.clientY - rect.top) / rect.height;
      const factor = e.deltaY < 0 ? 0.88 : 1.12;
      const x = view.xMin + mx * (view.xMax - view.xMin);
      const y = view.yMin + my * (view.yMax - view.yMin);
      const nxSpan = (view.xMax - view.xMin) * factor;
      const nySpan = (view.yMax - view.yMin) * factor;
      view.xMin = x - mx * nxSpan; view.xMax = view.xMin + nxSpan;
      view.yMin = y - my * nySpan; view.yMax = view.yMin + nySpan;
      clampView(); applyView();
    }, { passive: false });

    // drag to pan (mouse / single touch)
    let dragging = false, startX = 0, startY = 0, startView = null;
    let lastPinchDist = null;

    canvas.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'touch' && e.isPrimary === false) return;
      dragging = true; startX = e.clientX; startY = e.clientY; startView = { ...view };
      try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!dragging || !startView) return;
      const rect = canvas.getBoundingClientRect();
      const dx = (e.clientX - startX) / rect.width;
      const dy = (e.clientY - startY) / rect.height;
      const xSpan = startView.xMax - startView.xMin;
      const ySpan = startView.yMax - startView.yMin;
      view.xMin = startView.xMin - dx * xSpan;
      view.xMax = startView.xMax - dx * xSpan;
      view.yMin = startView.yMin + dy * ySpan;
      view.yMax = startView.yMax + dy * ySpan;
      clampView(); applyView();
    });
    const endDrag = () => { dragging = false; startView = null; lastPinchDist = null; };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);
    canvas.addEventListener('pointerleave', () => { dragging = false; });

    // pinch zoom (touch)
    canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 2) {
        e.preventDefault();
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        lastPinchDist = Math.hypot(dx, dy);
      }
    }, { passive: false });
    canvas.addEventListener('touchmove', (e) => {
      if (e.touches.length === 2 && lastPinchDist) {
        e.preventDefault();
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.hypot(dx, dy);
        const factor = lastPinchDist / dist;
        const mx = (e.touches[0].clientX + e.touches[1].clientX)/2;
        const my = (e.touches[0].clientY + e.touches[1].clientY)/2;
        const rect = canvas.getBoundingClientRect();
        const rx = (mx - rect.left)/rect.width, ry = 1 - (my - rect.top)/rect.height;
        const x = view.xMin + rx * (view.xMax - view.xMin);
        const y = view.yMin + ry * (view.yMax - view.yMin);
        const nxSpan = (view.xMax - view.xMin) * factor;
        const nySpan = (view.yMax - view.yMin) * factor;
        view.xMin = x - rx * nxSpan; view.xMax = view.xMin + nxSpan;
        view.yMin = y - ry * nySpan; view.yMax = view.yMin + nySpan;
        clampView(); applyView();
        lastPinchDist = dist;
      }
    }, { passive: false });
    canvas.addEventListener('touchend', () => { if (event.touches && event.touches.length < 2) lastPinchDist = null; });
  }

  return { init, setPrediction, setLoss, resize, destroy, zoomIn, zoomOut, resetZoom };
})();
