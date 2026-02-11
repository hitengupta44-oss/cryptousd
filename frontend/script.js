async function loadChart() {
    const res = await fetch("/data");
    const data = await res.json();

    const real = data.filter(d => d.type === "real");
    const pred = data.filter(d => d.type === "prediction");
    const buys = data.filter(d => d.signal === "BUY");
    const sells = data.filter(d => d.signal === "SELL");

    const realTrace = {
        x: real.map(d => d.time),
        open: real.map(d => d.open),
        high: real.map(d => d.high),
        low: real.map(d => d.low),
        close: real.map(d => d.close),
        type: "candlestick",
        name: "Real",
        increasing: { line: { color: "green" } },
        decreasing: { line: { color: "red" } }
    };

    const predTrace = {
        x: pred.map(d => d.time),
        open: pred.map(d => d.open),
        high: pred.map(d => d.high),
        low: pred.map(d => d.low),
        close: pred.map(d => d.close),
        type: "candlestick",
        name: "Prediction",
        opacity: 0.8,
        increasing: { line: { color: "orange" } },
        decreasing: { line: { color: "yellow" } }
    };

    const emaTrace = {
        x: data.map(d => d.time),
        y: data.map(d => d.ema20),
        type: "scatter",
        mode: "lines",
        name: "EMA20",
        line: { color: "blue" }
    };

    const smaTrace = {
        x: data.map(d => d.time),
        y: data.map(d => d.sma50),
        type: "scatter",
        mode: "lines",
        name: "SMA50",
        line: { color: "orange" }
    };

    const buyTrace = {
        x: buys.map(d => d.time),
        y: buys.map(d => d.close),
        mode: "markers",
        name: "BUY",
        marker: { symbol: "triangle-up", size: 12, color: "lime" }
    };

    const sellTrace = {
        x: sells.map(d => d.time),
        y: sells.map(d => d.close),
        mode: "markers",
        name: "SELL",
        marker: { symbol: "triangle-down", size: 12, color: "red" }
    };

    Plotly.newPlot("chart", [
        realTrace,
        predTrace,
        emaTrace,
        smaTrace,
        buyTrace,
        sellTrace
    ], {
        title: "BTCUSDT Live + Institutional Prediction",
        xaxis: { title: "Time" },
        yaxis: { title: "Price" }
    });
}

setInterval(loadChart, 5000);
loadChart();
