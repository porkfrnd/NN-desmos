// Chart.js-based charts. Prediction plot = dashed ground truth + solid
// glowing prediction line. Loss plot = MSE vs epoch with log-scale y-axis.
//
// We keep a single source-of-truth merge of current data per epoch cadence,
// and update it in place so the charts visibly step rather than violently
// jump between frames. Chart.js native animation is enabled (it gives smooth
// stepping) but with short durations so frequent re-renders don't fight.

const Charts = (() => {
  let predChart = null;
  let lossChart = null;

  function init(canvasPred, canvasLoss) {
    // ---- Prediction chart ----
    const predCtx = canvasPred.getContext('2d');
    predChart = new Chart(predCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Ground Truth',
            data: [],
            borderColor: '#6b6f82',
            borderDash: [6, 4],
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
          },
          {
            label: 'Prediction',
            data: [],
            borderColor: '#6c8cff',
            borderWidth: 2.5,
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 150 },
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            labels: { color: '#9498ab', boxWidth: 18, padding: 16 },
          },
          tooltip: {
            backgroundColor: '#1a1d27',
            borderColor: '#363b4e',
            borderWidth: 1,
            titleColor: '#e4e6ef',
            bodyColor: '#e4e6ef',
          },
        },
        scales: {
          x: {
            title: { display: true, text: 'x', color: '#6b6f82' },
            grid: { color: 'rgba(255,255,255,0.06)' },
            ticks: { color: '#6b6f82', maxTicksLimit: 9 },
            min: -1,
            max: 1,
          },
          y: {
            title: { display: true, text: 'f(x)', color: '#6b6f82' },
            grid: { color: 'rgba(255,255,255,0.06)' },
            ticks: { color: '#6b6f82' },
            min: -1.6,
            max: 1.6,
          },
        },
      },
    });

    // ---- Loss chart ----
    const lossCtx = canvasLoss.getContext('2d');
    lossChart = new Chart(lossCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'MSE Loss',
            data: [],
            borderColor: '#ff922b',
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 150 },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1a1d27',
            borderColor: '#363b4e',
            borderWidth: 1,
            titleColor: '#e4e6ef',
            bodyColor: '#e4e6ef',
          },
        },
        scales: {
          x: {
            title: { display: true, text: 'Epoch', color: '#6b6f82' },
            grid: { color: 'rgba(255,255,255,0.06)' },
            ticks: { color: '#6b6f82', maxTicksLimit: 8 },
          },
          y: {
            type: 'logarithmic',
            title: { display: true, text: 'MSE (log)', color: '#6b6f82' },
            grid: { color: 'rgba(255,255,255,0.06)' },
            ticks: { color: '#6b6f82' },
          },
        },
      },
    });
  }

  // Update the prediction chart with ground truth and prediction curves.
  function setPrediction(gtXs, gtYs, predXs, predYs) {
    if (!predChart) return;
    predChart.data.labels = gtXs;
    predChart.data.datasets[0].data = gtYs;
    predChart.data.datasets[1].data = predXs.map((x, i) => predYs[i]);
    predChart.update();
  }

  // Update the loss chart with history [{epoch, loss}].
  function setLoss(hist) {
    if (!lossChart) return;
    lossChart.data.labels = hist.map((h) => h.epoch);
    lossChart.data.datasets[0].data = hist.map((h) => h.loss);
    lossChart.update();
  }

  function resize() {
    if (predChart) predChart.resize();
    if (lossChart) lossChart.resize();
  }

  return { init, setPrediction, setLoss, resize };
})();
