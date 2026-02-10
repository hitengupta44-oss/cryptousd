const API = "https://cryptousdlive-1.onrender.com/data";

async function loadChart() {
    try {
        const res = await fetch(API);
        let data = await res.json();

        if (!data || data.length === 0) return;

        // ===== Separate Real & Prediction =====
        const real = data.filter(d => d.type === "real").slice(-60);
        const future = data.filter(d => d.type === "prediction").slice(-10);

        if (real.length === 0) return;

        // ================= REAL CANDLES =================
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

        // ================= PREDICTION CANDLES =================
        const predTrace = {
            type: "candlestick",
            x: future.map(d => d.time),
            open: future.map(d => d.open),
            high: future.map(d => d.high),
            low: future.map(d => d.low),
            close: future.map(d => d.close),
            increasing: { line: { color: "#ffa500" } }, // orange
            decreasing: { line: { color: "#808000" } }, // olive
            name: "Prediction",
            opacity: 0.8
        };

        // ================= INDICATORS (REAL ONLY) =================
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

        // ================= BUY / SELL MARKERS =================
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

        // ================= RSI PANEL =================
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

        // ================= LAYOUT =================
        const layout = {
            paper_bgcolor: "#000",
            plot_bgcolor: "#000",
            font: { color: "#ccc" },

            margin: { t: 20, b: 30, l: 50, r: 20 },

            // Main chart
            xaxis: {
                rangeslider: { visible: false },
                showgrid: false
            },

            yaxis: {
                domain: [0.3, 1],
                title: "Price"
            },

            // RSI panel
            xaxis2: {
                anchor: "y2",
                showgrid: false
            },

            yaxis2: {
                domain: [0, 0.22],
                range: [0, 100],
                title: "RSI"
            },

            legend: {
                orientation: "h",
                y: 1.05
            }
        };

        // ================= PLOT =================
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

    } catch (err) {
        console.log("Chart error:", err);
    }
}

// ===== Refresh every minute =====
setInterval(loadChart, 60000);
loadChart();
