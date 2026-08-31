// HTML5 freehand-drawing canvas for a custom target function.
//
// Maps canvas pixel space -> [-1,1] x [-1,1] domain. Pointer events (mouse +
// touch via PointerEvents). The raw stroke is resampled into exactly N
// evenly-spaced-in-x points via linear interpolation, because raw stroke
// points are unevenly spaced in x and would break training.

const CanvasDraw = (() => {
  const DEFAULTS = {
    padding: 12, // px of gutter inside the canvas
    samples: 120,
  };

  let canvas = null;
  let ctx = null;
  let drawing = false;
  let points = []; // raw pointer points in domain coords (x: -1..1, y: 1..-1)
  let lastPoint = null;
  let onData = null; // callback(data: {xs, ys})
  let size = 0; // current canvas drawing size in px (square)

  function init(cv, opts = {}) {
    canvas = cv;
    onData = opts.onData || null;
    resizeCanvas();
    cv.addEventListener('pointerdown', onPointerDown);
    cv.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerUp);
    window.addEventListener('resize', resizeCanvas);
    // prevent page scroll while drawing on touch
    cv.style.touchAction = 'none';
  }

  function resizeCanvas() {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    // keep canvas square to the smallest side of its container
    size = Math.floor(Math.min(rect.width, rect.height) || rect.width || 300);
    if (size < 40) size = Math.floor(rect.width || 300);
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    const c = canvas.getContext('2d');
    if (c) {
      c.setTransform(1, 0, 0, 1, 0, 0);
      c.scale(dpr, dpr);
      ctx = c;
    }
    canvas.style.width = size + 'px';
    canvas.style.height = size + 'px';
    redraw();
  }

  function onPointerDown(e) {
    e.preventDefault();
    drawing = true;
    points = [];
    lastPoint = null;
    try { canvas.setPointerCapture && canvas.setPointerCapture(e.pointerId); } catch (_) {}
    addPoint(e);
    redraw();
  }

  function onPointerMove(e) {
    if (!drawing) return;
    e.preventDefault();
    addPoint(e);
    redraw();
  }

  function onPointerUp() {
    if (!drawing) return;
    drawing = false;
    const sampled = resample(points, DEFAULTS.samples);
    if (onData && sampled.xs.length >= 2) {
      onData(sampled);
    }
  }

  // Convert pointer event to a domain point (y flipped).
  function addPoint(e) {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const x = -1 + (2 * px) / size;
    const y = 1 - (2 * py) / size;
    points.push(clamp({ x, y }));
    lastPoint = { x, y, px, py };
  }

  function clamp(p) {
    return { x: Math.max(-1, Math.min(1, p.x)), y: Math.max(-1.5, Math.min(1.5, p.y)) };
  }

  // Resample a raw point list into evenly-spaced-in-x samples of length n.
  function resample(raw, n) {
    if (raw.length < 2) return { xs: [], ys: [] };
    const minX = raw[0].x, maxX = raw[raw.length - 1].x;
    // Sort by x to guarantee monotonic (defensive).
    const sorted = raw.slice().sort((a, b) => a.x - b.x);
    const xs = [];
    const ys = [];
    for (let i = 0; i < n; i++) {
      const t = n === 1 ? 0 : i / (n - 1);
      const x = minX + (maxX - minX) * t;
      xs.push(x);
      ys.push(interpY(sorted, x));
    }
    return { xs, ys };
  }

  // Linear interpolation of y at a given x over the sorted raw points.
  function interpY(sorted, x) {
    if (x <= sorted[0].x) return sorted[0].y;
    if (x >= sorted[sorted.length - 1].x) return sorted[sorted.length - 1].y;
    for (let i = 0; i < sorted.length - 1; i++) {
      const a = sorted[i], b = sorted[i + 1];
      if (x >= a.x && x <= b.x) {
        const t = (x - a.x) / (b.x - a.x || 1);
        return a.y + t * (b.y - a.y);
      }
    }
    return sorted[sorted.length - 1].y;
  }

  function pxOfX(x) {
    return ((x + 1) / 2) * size;
  }

  function pyOfY(y) {
    return size * (0.5 - y / 3);
  }

  function redraw() {
    if (!ctx || !size) return;
    ctx.clearRect(0, 0, size, size);
    drawBackground();
    drawAxes();
    if (points.length > 1) {
      ctx.beginPath();
      ctx.moveTo(pxOfX(points[0].x), pyOfY(points[0].y));
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(pxOfX(points[i].x), pyOfY(points[i].y));
      }
      const isDark = document.documentElement.dataset.theme !== 'light' && getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() !== '';
      ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent') || '#2563eb';
      ctx.lineWidth = 2.2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.stroke();
    }
  }

  function drawBackground() {
    const bg = getComputedStyle(document.documentElement).getPropertyValue('--board-bg').trim() || '#ecece7';
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, size, size);
  }

  function drawAxes() {
    const grid = getComputedStyle(document.documentElement).getPropertyValue('--grid-line').trim() || 'rgba(0,0,0,0.08)';
    const border = getComputedStyle(document.documentElement).getPropertyValue('--border').trim() || '#d8d8d2';
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pxOfX(0), 0); ctx.lineTo(pxOfX(0), size);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, pyOfY(0)); ctx.lineTo(size, pyOfY(0));
    ctx.stroke();
    ctx.strokeStyle = border;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pxOfX(-1), 0); ctx.lineTo(pxOfX(-1), size);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pxOfX(1), 0); ctx.lineTo(pxOfX(1), size);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function getSamples() {
    return resample(points, DEFAULTS.samples);
  }

  function clear() {
    points = [];
    drawing = false;
    lastPoint = null;
    redraw();
  }

  return { init, clear, getSamples };
})();
