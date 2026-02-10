const API = "https://cryptousdlive-1.onrender.com/data";

async function loadChart() {
    const res = await fetch(API);
    const data = await res.json();
    if (!data.length) return;

    const real = data.filter(d => d.type === "real").slice(-60);
    const future = data.filter(d => d.type === "prediction").slice(-10);

    const realTrace = {
        type: "candlestick",
        x: real.map(d=>d.time),
        open: real.map(d=>d.open),
        high: real.map(d=>d.high),
        low: real.map(d=>d.low),
        close: real.map(d=>d.close),
        increasing:{line:{color:"#26a69a"}},
        decreasing:{line:{color:"#ef5350"}},
        name:"Real"
    };

    const predTrace = {
        type: "candlestick",
        x: future.map(d=>d.time),
        open: future.map(d=>d.open),
        high: future.map(d=>d.high),
        low: future.map(d=>d.low),
        close: future.map(d=>d.close),
        increasing:{line:{color:"orange"}},
        decreasing:{line:{color:"olive"}},
        name:"Prediction"
    };

    const ema = {
        x: real.map(d=>d.time),
        y: real.map(d=>d.ema20),
        type:"scatter",
        mode:"lines",
        line:{color:"#00e5ff"},
        name:"EMA20"
    };

    const sma = {
        x: real.map(d=>d.time),
        y: real.map(d=>d.sma50),
        type:"scatter",
        mode:"lines",
        line:{color:"#ffd700"},
        name:"SMA50"
    };

    const vwap = {
        x: real.map(d=>d.time),
        y: real.map(d=>d.vwap),
        type:"scatter",
        mode:"lines",
        line:{color:"#b388ff"},
        name:"VWAP"
    };

    const rsi = {
        x: real.map(d=>d.time),
        y: real.map(d=>d.rsi),
        type:"scatter",
        mode:"lines",
        xaxis:"x2",
        yaxis:"y2",
        line:{color:"#ff9800"},
        name:"RSI"
    };

    const layout = {
        paper_bgcolor:"#0a0a0a",
        plot_bgcolor:"#0a0a0a",
        font:{color:"#ccc"},
        yaxis:{domain:[0.3,1]},
        yaxis2:{domain:[0,0.2],range:[0,100]},
        xaxis2:{matches:"x"},
        xaxis:{rangeslider:{visible:false}}
    };

    Plotly.newPlot("chart",
        [realTrace,predTrace,ema,sma,vwap,rsi],
        layout,
        {responsive:true}
    );
}

setInterval(loadChart,10000);
loadChart();
