const API = "https://cryptousd.onrender.com/history";

async function loadChart() {
  const res = await fetch(API);
  const data = await res.json();
  if (data.length === 0) return;

  const t = data.map(d => d.time);
  const o = data.map(d => d.open);
  const h = data.map(d => d.high);
  const l = data.map(d => d.low);
  const c = data.map(d => d.close);
  const p = data.map(d => d.prediction);

  const buys = data.filter(d => d.signal === "BUY");
  const sells = data.filter(d => d.signal === "SELL");

  Plotly.newPlot("chart", [
    {
      type: "candlestick",
      x: t, open: o, high: h, low: l, close: c,
      name: "BTC Price"
    },
    {
      type: "scatter",
      x: t, y: p,
      mode: "lines",
      name: "Prediction",
      line: { color: "orange" }
    },
    {
      type: "scatter",
      mode: "markers",
      name: "BUY",
      x: buys.map(d => d.time),
      y: buys.map(d => d.close),
      marker: { color: "green", symbol: "triangle-up", size: 12 }
    },
    {
      type: "scatter",
      mode: "markers",
      name: "SELL",
      x: sells.map(d => d.time),
      y: sells.map(d => d.close),
      marker: { color: "red", symbol: "triangle-down", size: 12 }
    }
  ], {
    xaxis: { rangeslider: { visible: false } },
    yaxis: { title: "Price" },
    title: "BTCUSD — Live ML Prediction Dashboard"
  });
}

setInterval(loadChart, 60000);
loadChart();
