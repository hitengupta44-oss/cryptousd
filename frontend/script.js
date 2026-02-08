const API = "https://cryptousd.onrender.com";

let prices = [];
let preds = [];
let times = [];

async function loadData() {
  const res = await fetch(API);
  const data = await res.json();

  if (!data.price) return;

  prices.push(data.price);
  preds.push(data.prediction);
  times.push(data.time);

  Plotly.newPlot("chart", [
    { x: times, y: prices, mode: "lines", name: "Price" },
    { x: times, y: preds, mode: "lines", name: "Prediction" }
  ]);
}

setInterval(loadData, 60000);
loadData();
