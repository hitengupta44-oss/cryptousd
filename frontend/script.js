const API = "https://cryptousd.onrender.com/history";

async function loadChart() {
  const res = await fetch(API);
  const data = await res.json();

  const times = data.map(d => d.time);
  const prices = data.map(d => d.price);
  const preds = data.map(d => d.prediction);

  Plotly.newPlot("chart", [
    { x: times, y: prices, mode: "lines", name: "Price" },
    { x: times, y: preds, mode: "lines", name: "Prediction" }
  ]);
}

setInterval(loadChart, 60000);
loadChart();
