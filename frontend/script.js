const API = "https://cryptoliveusdt.onrender.com/data";

async function loadChart() {
    try {
        const res = await fetch(API);
        let data = await res.json();
        if (!data || data.length === 0) return;

        // Convert time
        data.forEach(d => d.time = new Date(d.time));

        // Separate data
        const real = data.filter(d => d.type === "real").slice(-60);
        const pred = data.filter(d => d.type === "prediction").slice(-10);

        if (real.length === 0) return;

        const combined = [...real, ...pred];

        const safe = (arr, key) => arr.map(d => d[key] ?? null);

        // ======================
        // REAL CANDLES
        // ======================
        const realTrace = {
            type: "candlestick",
            x: real.map(d => d.time),
            open: safe(real, "open"),
            high: safe(real, "high"),
            low: safe(real, "low"),
            close: safe(real, "close"),
            increasing: { line: { color: "#26a69a" } },
            decreasing: { line: { color: "#ef5350" } },
            name: "Real"
        };

        // ======================
        // PREDICTION CANDLES
        // ======================
        const predTrace = {
            type: "candlestick",
            x: pred.map(d => d.time),
            open: safe(pred, "open"),
            high: safe(pred, "high"),
            low: safe(pred, "low"),
            close: safe(pred, "close"),
            increasing: { line: { color: "green" } },
            decreasing: { line: { color: "olive" } },
            name: "Prediction"
        };

        // ======================
        // INDICATORS (single lines)
        // ======================
        const emaTrace = {
            x: combined.map(d => d.time),
            y: safe(combined, "ema20"),
            type: "scatter",
            mode: "lines",
            line: { color: "#00e5ff", width: 2 },
            name: "EMA20"
        };

        const smaTrace = {
            x: combined.map(d => d.time),
            y: safe(combined, "sma50"),
            type: "scatter",
            mode: "lines",
            line: { color: "#ffd700", width: 2 },
            name: "SMA50"
        };

        const vwapTrace = {
            x: combined.map(d => d.time),
            y: safe(combined, "vwap"),
            type: "scatter",
            mode: "lines",
            line: { color: "#b388ff", width: 2 },
            name: "VWAP"
        };

        // ======================
        // RSI PANEL (single)
        // ======================
        const rsiTrace = {
            x: combined.map(d => d.time),
            y: safe(combined, "rsi"),
            type: "scatter",
            mode: "lines",
            line: { color: "#ff9800", width: 2 },
            name: "RSI",
            xaxis: "x2",
            yaxis: "y2"
        };

        // ======================
        // BUY / SELL SIGNALS
        // ======================
        const buys = combined.filter(d => d.signal === "BUY");
        const sells = combined.filter(d => d.signal === "SELL");

        const buyTrace = {
            x: buys.map(d => d.time),
            y: buys.map(d => d.close),
            mode: "markers",
            marker: {
                symbol: "triangle-up",
                color: "lime",
                size: 10
            },
            name: "BUY"
        };

        const sellTrace = {
            x: sells.map(d => d.time),
            y: sells.map(d => d.close),
            mode: "markers",
            marker: {
                symbol: "triangle-down",
                color: "red",
                size: 10
            },
            name: "SELL"
        };

        // ======================
        // LAYOUT
        // ======================
        const layout = {
            paper_bgcolor: "#000",
            plot_bgcolor: "#000",
            font: { color: "#ccc" },
            margin: { t: 20, b: 20, l: 50, r: 20 },

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
                range: [0, 100],
                title: "RSI"
            },

            legend: {
                orientation: "h",
                y: 1.02
            }
        };

        Plotly.react(
            "chart",
            [
                realTrace,
                predTrace,
                emaTrace,
                smaTrace,
                vwapTrace,
                buyTrace,
                sellTrace,
                rsiTrace
            ],
            layout,
            { responsive: true }
        );

    } catch (err) {
        console.error("Chart error:", err);
    }
}

setInterval(loadChart, 10000);
loadChart();
