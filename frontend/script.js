const API = "https://cryptoliveusdt.onrender.com/data";

async function loadChart() {
    try {
        const res = await fetch(API);
        let data = await res.json();
        if (!data || data.length === 0) return;

        // Convert time to Date objects
        data.forEach(d => {
            d.time = new Date(d.time);
        });

        // Separate real & prediction
        const real = data.filter(d => d.type === "real");
        const pred = data.filter(d => d.type === "prediction");

        const real60 = real.slice(-60);
        const pred10 = pred.slice(-10);

        const combined = [...real60, ...pred10];

        if (combined.length === 0) return;

        // Helper to avoid null values
        const safe = (arr, key) => arr.map(d => d[key] ?? null);

        // ====================
        // REAL Candles
        // ====================
        const candleTrace = {
            type: "candlestick",
            x: combined.map(d => d.time),
            open: safe(combined, "open"),
            high: safe(combined, "high"),
            low: safe(combined, "low"),
            close: safe(combined, "close"),
            increasing: { line: { color: "#26a69a" } },
            decreasing: { line: { color: "#ef5350" } },
            name: "Price"
        };

        // ====================
        // Prediction Overlay
        // ====================
        const predTrace = {
            type: "candlestick",
            x: pred10.map(d => d.time),
            open: safe(pred10, "open"),
            high: safe(pred10, "high"),
            low: safe(pred10, "low"),
            close: safe(pred10, "close"),
            increasing: { line: { color: "orange" } },
            decreasing: { line: { color: "olive" } },
            name: "Prediction",
            opacity: 0.9
        };

        // ====================
        // Indicators (Real + Pred)
        // ====================
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

        // ====================
        // RSI Panel
        // ====================
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

        // ====================
        // Layout
        // ====================
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
                candleTrace,
                predTrace,
                emaTrace,
                smaTrace,
                vwapTrace,
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
