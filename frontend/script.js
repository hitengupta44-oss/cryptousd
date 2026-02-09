const API = "https://cryptousd.onrender.com/data";

async function loadChart() {
    const res = await fetch(API);
    let data = await res.json();
    if (!data || data.length === 0) return;

    // ===== Last 60 real + latest 10 predictions =====
    const real = data.filter(d => d.type === "real").slice(-60);
    const future = data.filter(d => d.type === "prediction").slice(-10);
    const combined = [...real, ...future];

    const times = combined.map(d => d.time);

    const candleTrace = {
        type: "candlestick",
        x: times,
        open: combined.map(d => d.open),
        high: combined.map(d => d.high),
        low: combined.map(d => d.low),
        close: combined.map(d => d.close),
        increasing: { line: { color: "green" } },
        decreasing: { line: { color: "red" } },
        name: "BTCUSD"
    };

    // ===== Indicators (real only) =====
    const emaTrace = {
        x: real.map(d => d.time),
        y: real.map(d => d.ema20),
        type: "scatter",
        mode: "lines",
        line: { color: "#00e5ff", width: 2 },
        name: "EMA20"
    };

    const smaTrace = {
        x: real.map(d => d.time),
        y: real.map(d => d.sma50),
        type: "scatter",
        mode: "lines",
        line: { color: "#ffd700", width: 2 },
        name: "SMA50"
    };

    const vwapTrace = {
        x: real.map(d => d.time),
        y: real.map(d => d.vwap),
        type: "scatter",
        mode: "lines",
        line: { color: "#b388ff", width: 2 },
        name: "VWAP"
    };

    // ===== Buy/Sell markers =====
    const buy = real.filter(d => d.signal === "BUY");
    const sell = real.filter(d => d.signal === "SELL");

    const buyMarkers = {
        x: buy.map(d => d.time),
        y: buy.map(d => d.low),
        mode: "markers",
        type: "scatter",
        marker: { color: "lime", size: 9, symbol: "triangle-up" },
        name: "BUY"
    };

    const sellMarkers = {
        x: sell.map(d => d.time),
        y: sell.map(d => d.high),
        mode: "markers",
        type: "scatter",
        marker: { color: "red", size: 9, symbol: "triangle-down" },
        name: "SELL"
    };

    // ===== RSI Panel =====
    const rsiTrace = {
        x: real.map(d => d.time),
        y: real.map(d => d.rsi),
        type: "scatter",
        mode: "lines",
        line: { color: "#ff9800", width: 2 },
        name: "RSI",
        xaxis: "x2",
        yaxis: "y2"
    };

    const layout = {
        paper_bgcolor: "#000",
        plot_bgcolor: "#000",
        font: { color: "#ccc" },

        margin: { t: 30, b: 20, l: 50, r: 20 },

        xaxis: {
            rangeslider: { visible: false },
            showgrid: false
        },

        yaxis: {
            domain: [0.32, 1],
            title: "Price"
        },

        xaxis2: {
            anchor: "y2",
            showgrid: false
        },

        yaxis2: {
            domain: [0, 0.25],
            title: "RSI",
            range: [0, 100]
        },

        legend: {
            orientation: "h",
            y: 1.02
        },

        title: {
            text: "BTCUSD — AI Professional Dashboard",
            font: { size: 18 }
        }
    };

    Plotly.newPlot(
        "chart",
        [
            candleTrace,
            emaTrace,
            smaTrace,
            vwapTrace,
            buyMarkers,
            sellMarkers,
            rsiTrace
        ],
        layout,
        { responsive: true }
    );
}

setInterval(loadChart, 60000);
loadChart();
