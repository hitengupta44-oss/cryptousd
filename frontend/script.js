const API = "https://cryptousdlive-1.onrender.com/data";

async function loadChart() {
    const res = await fetch(API);
    let data = await res.json();
    if (!data.length) return;

    const real = data.filter(d => d.type === "real").slice(-60);
    const future = data.filter(d => d.type === "prediction").slice(-10);

    const realTrace = {
        type: "candlestick",
        x: real.map(d => d.time),
        open: real.map(d => d.open),
        high: real.map(d => d.high),
        low: real.map(d => d.low),
        close: real.map(d => d.close),
        increasing: {line:{color:"green"}},
        decreasing: {line:{color:"red"}},
        name: "Real"
    };

    const futureTrace = {
        type: "candlestick",
        x: future.map(d => d.time),
        open: future.map(d => d.open),
        high: future.map(d => d.high),
        low: future.map(d => d.low),
        close: future.map(d => d.close),
        increasing: {line:{color:"orange"}},
        decreasing: {line:{color:"yellow"}},
        name: "Prediction"
    };

    const ema = {
        x: real.map(d => d.time),
        y: real.map(d => d.ema20),
        type: "scatter",
        mode: "lines",
        name: "EMA20"
    };

    const sma = {
        x: real.map(d => d.time),
        y: real.map(d => d.sma50),
        type: "scatter",
        mode: "lines",
        name: "SMA50"
    };

    const vwap = {
        x: real.map(d => d.time),
        y: real.map(d => d.vwap),
        type: "scatter",
        mode: "lines",
        name: "VWAP"
    };

    Plotly.newPlot("chart",
        [realTrace, futureTrace, ema, sma, vwap],
        {
            paper_bgcolor:"#000",
            plot_bgcolor:"#000",
            font:{color:"#ccc"},
            xaxis:{rangeslider:{visible:false}}
        },
        {responsive:true}
    );
}

setInterval(loadChart,10000);
loadChart();
