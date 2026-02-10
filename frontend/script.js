const API = "https://cryptousdlive-1.onrender.com/data";

async function loadChart() {
    const res = await fetch(API);
    let data = await res.json();
    if (!data || data.length === 0) return;

    // ===== Split data =====
    const real = data.filter(d => d.type === "real").slice(-60);
    const future = data.filter(d => d.type === "prediction").slice(-10);

    // ===== PRICE CANDLES =====
    const realTrace = {
        type: "candlestick",
        x: real.map(d => d.time),
        open: real.map(d => d.open),
        high: real.map(d => d.high),
        low: real.map(d => d.low),
        close: real.map(d => d.close),
        increasing: { line: { color: "#26a69a" } },
        decreasing: { line: { color: "#ef5350" } },
        name: "Real"
    };

    const predTrace = {
        type: "candlestick",
        x: future.map(d => d.time),
        open: future.map(d => d.open),
        high: future.map(d => d.high),
        low: future.map(d => d.low),
        close: future.map(d => d.close),
        increasing: { line: { color: "orange" } },
        decreasing: { line: { color: "olive" } },
        opacity: 0.9,
        name: "Prediction"
    };

    // ===== INDICATORS =====
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

    // ===== BUY / SELL MARKERS (only when backend sends signal) =====
    const buys = real.filter(d => d.signal === "BUY");
    const sells = real.filter(d => d.signal === "SELL");

    const buyMarkers = {
        x: buys.map(d => d.time),
        y: buys.map(d => d.low),
        mode: "markers",
        type: "scatter",
        marker: {
            symbol: "triangle-up",
            size: 10,
            color: "lime"
        },
        name: "BUY"
    };

    const sellMarkers = {
        x: sells.map(d => d.time),
        y: sells.map(d => d.high),
        mode: "markers",
        type: "scatter",
        marker: {
            symbol: "triangle-down",
            size: 10,
            color: "red"
        },
        name: "SELL"
    };

    // ===== RSI PANEL =====
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

    // ===== Future Zone Highlight =====
    const shapes = future.length ? [{
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: future[0].time,
        x1: future[future.length - 1].time,
        y0: 0,
        y1: 1,
        fillcolor: "rgba(255,165,0,0.05)",
        line: { width: 0 }
    }] : [];

    // ===== Layout =====
    const layout = {
        paper_bgcolor: "#0a0a0a",
        plot_bgcolor: "#0a0a0a",
        font: { color: "#ccc" },

        margin: { t: 20, b: 30, l: 50, r: 20 },

        // Price chart
        yaxis: {
            domain: [0.30, 1],
            gridcolor: "#111",
            title: "Price"
        },

        // RSI chart
        yaxis2: {
            domain: [0, 0.22],
            range: [0, 100],
            title: "RSI",
            gridcolor: "#111"
        },

        xaxis: {
            rangeslider: { visible: false }
        },

        xaxis2: {
            anchor: "y2",
            matches: "x"
        },

        legend: {
            orientation: "h",
            y: 1.02
        },

        shapes: shapes
    };

    Plotly.newPlot(
        "chart",
        [
            realTrace,
            predTrace,
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

// Refresh every 10 sec
setInterval(loadChart, 10000);
loadChart();
