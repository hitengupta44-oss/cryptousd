const API = "https://cryptousd.onrender.com/data";

async function loadChart() {
  const res = await fetch(API);
  const data = await res.json();
  if (!data || data.length === 0) return;

  const real = data.filter(d => d.type === "real");
  const pred = data.filter(d => d.type === "prediction");

  Plotly.newPlot("chart", [
    {
      type: "candlestick",
      name: "Live Market",
      x: real.map(d => d.time),
      open: real.map(d => d.open),
      high: real.map(d => d.high),
      low: real.map(d => d.low),
      close: real.map(d => d.close),
      increasing: { line: { color: "lime" } },
      decreasing: { line: { color: "red" } }
    },
    {
      type: "candlestick",
      name: "AI Forecast (Next 10 min)",
      x: pred.map(d => d.time),
      open: pred.map(d => d.open),
      high: pred.map(d => d.high),
      low: pred.map(d => d.low),
      close: pred.map(d => d.close),
      increasing: { line: { color: "orange" } },
      decreasing: { line: { color: "olive" } }
    }
  ], {
    paper_bgcolor: "black",
    plot_bgcolor: "black",
    xaxis: { rangeslider: { visible: false }, color: "white" },
    yaxis: { color: "white" },
    title: "BTCUSD — Live + Rolling 10-Minute AI Forecast"
  });
}

loadChart();
setInterval(loadChart, 60000);
