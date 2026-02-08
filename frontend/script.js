const API = "https://cryptousd.onrender.com/data";

async function loadChart() {
    const res = await fetch(API);
    const data = await res.json();
    if (!data || data.length === 0) return;

    const t = data.map(d => d.time);
    const o = data.map(d => d.open);
    const h = data.map(d => d.high);
    const l = data.map(d => d.low);
    const c = data.map(d => d.close);
    const types = data.map(d => d.type);
    const signals = data.map(d => d.signal);

    // Colors: Real = green/red, Prediction = orange/olive
    const colors = data.map((d, i) => {
        if (types[i] === "real") return signals[i] === "BUY" ? "green" : "red";
        else return signals[i] === "BUY" ? "orange" : "olive";
    });

    const trace = {
        type: "candlestick",
        x: t,
        open: o,
        high: h,
        low: l,
        close: c,
        increasing: { line: { color: "green" } },
        decreasing: { line: { color: "red" } },
        name: "BTCUSD",
        showlegend: false
    };

    const layout = {
        plot_bgcolor: "#0a0a0a",
        paper_bgcolor: "#0a0a0a",
        xaxis: { rangeslider: { visible: false }, color: "#fff" },
        yaxis: { color: "#fff", title: "Price" },
        font: { color: "#fff" },
        title: { text: "BTCUSD Live + 10-min Prediction", font: { color: "#fff", size: 20 } }
    };

    Plotly.newPlot("chart", [trace]);
}

setInterval(loadChart, 60000);
loadChart();
