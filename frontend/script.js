const API = "https://cryptousdlive-1.onrender.com/data";

async function loadChart() {
    const res = await fetch(API);
    let data = await res.json();
    if (!data || data.length === 0) return;

    // Separate real and prediction
    const real = data.filter(d => d.type === "real").slice(-60);
    const pred = data.filter(d => d.type === "prediction").slice(-10);

    // ================= Candles =================
    const realCandle = {
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

    const predCandle = {
        type: "candlestick",
        x: pred.map(d => d.time),
        open: pred.map(d => d.open),
        high: pred.map(d => d.high),
        low: pred.map(d => d.low),
        close: pred.map(d => d.close),
        increasing: { line: { color: "orange" } },
        decreasing: { line: { color: "olive" } },
        opacity: 0.8,
        name: "Prediction"
    };

    // ================= Indicators =================
    const ema = {
        x: real.map(d => d.time),
        y: real.map(d => d.ema20),
        type: "scatter",
        mode: "lines",
        line: { color: "#00e5ff", width: 2 },
        name: "EMA20"
    };

    const sma = {
        x: real.map(d => d.time),
        y: real.map(d => d.sma50),
        type: "scatter",
        mode: "lines",
        line: { color: "#ffd700", width: 2 },
        name: "SMA50"
    };

    const vwap = {
        x: real.map(d => d.time),
        y: real.map(d => d.vwap),
        type: "scatter",
        mode: "lines",
        line: { color: "#b388ff", width: 2 },
        name: "VWAP"
    };

    // ================= RSI Panel =================
    const rsi = {
        x: real.map(d => d.time),
        y: real.map(d => d.rsi),
        type: "scatter",
        mode: "lines",
        line: { color: "#ff9800", width: 2 },
        name: "RSI",
        xaxis: "x2",
        yaxis: "y2"
    };

    // ================= Layout =================
    const layout = {
        paper_bgcolor: "#000",
        plot_bgcolor: "#000",
        font: { color: "#ccc" },

        margin: { t: 20, b: 30, l: 50, r: 20 },

        xaxis: { rangeslider: { visible: false } },
        yaxis: { domain: [0.3, 1], title: "Price" },

        xaxis2: { anchor: "y2" },
        yaxis2: {
            domain: [0, 0.25],
            range: [0, 100],
            title: "RSI"
        },

        showlegend: true,
        legend: { orientation: "h", y: 1.02 }
    };

    Plotly.newPlot(
        "chart",
        [realCandle, predCandle, ema, sma, vwap, rsi],
        layout,
        { responsive: true }
    );
}

// Refresh every minute
setInterval(loadChart, 60000);
loadChart();
