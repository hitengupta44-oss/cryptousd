const API = "https://cryptousdlive-1.onrender.com/data";

async function loadChart() {
    const res = await fetch(API);
    const data = await res.json();
    if (!data.length) return;

    const real = data.filter(d => d.type==="real").slice(-60);
    const future = data.filter(d => d.type==="prediction").slice(-10);

    const all = [...real, ...future];

    const candle = {
        type: "candlestick",
        x: all.map(d=>d.time),
        open: all.map(d=>d.open),
        high: all.map(d=>d.high),
        low: all.map(d=>d.low),
        close: all.map(d=>d.close),
        increasing:{line:{color:"#26a69a"}},
        decreasing:{line:{color:"#ef5350"}}
    };

    const ema = {
        x: all.map(d=>d.time),
        y: all.map(d=>d.ema20),
        type:"scatter",
        line:{color:"#00e5ff"},
        name:"EMA"
    };

    const sma = {
        x: all.map(d=>d.time),
        y: all.map(d=>d.sma50),
        type:"scatter",
        line:{color:"#ffd700"},
        name:"SMA"
    };

    const vwap = {
        x: all.map(d=>d.time),
        y: all.map(d=>d.vwap),
        type:"scatter",
        line:{color:"#b388ff"},
        name:"VWAP"
    };

    const rsi = {
        x: real.map(d=>d.time),
        y: real.map(d=>d.rsi),
        xaxis:"x2",
        yaxis:"y2",
        type:"scatter",
        line:{color:"orange"},
        name:"RSI"
    };

    const layout = {
        paper_bgcolor:"#000",
        plot_bgcolor:"#000",
        font:{color:"#ccc"},
        xaxis:{rangeslider:{visible:false}},
        yaxis:{domain:[0.3,1]},
        xaxis2:{anchor:"y2"},
        yaxis2:{domain:[0,0.25],range:[0,100]}
    };

    Plotly.newPlot("chart",[candle,ema,sma,vwap,rsi],layout);
}

setInterval(loadChart,60000);
loadChart();
