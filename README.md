CryptoUSD is a real-time cryptocurrency monitoring and prediction system that displays live BTC/USD price data along with short-term future price forecasts using a deep learning model.

Project is to simulate a live trading terminal where users can see the latest market candles and a machine learning-based prediction for the next few minutes, combines real-time data handling, time-series forecasting, backend APIs, and an interactive frontend.

Fetches live BTC/USD market data

Shows the latest 60 candles on a chart

Predicts the next 10 minutes of price movement

Displays prediction candles alongside real data

Updates automatically to simulate a live market

Provides a clean and minimal trading-style interface

Learning time-series forecasting

Understanding real-time data pipelines

Tech Stack

Backend

Python

FastAPI

NumPy / Pandas

TensorFlow / Keras (LSTM Model)

Frontend

HTML

JavaScript

Charting Library (candlestick visualization)

Deployment

Backend: Render

Frontend: Vercel

Project Architecture

The system is divided into three main parts:

1. Data Layer (Producer)

Fetches live BTC/USD price data from an exchange API

Converts price data into 1-minute candles

Continuously sends data to the backend

2. Backend (API + ML)

Stores recent candle data

Keeps a rolling window of the last 60 minutes

Sends this data to the LSTM model

Generates predictions for the next 10 minutes

Returns both:

Historical candles

Predicted future candles

3. Frontend (Dashboard)

Calls backend API periodically

Displays:

Last 60 real candles

Next 10 predicted candles

Automatically refreshes to show new data

Machine Learning Model

The prediction engine is based on an LSTM (Long Short-Term Memory) network designed for time-series forecasting.

Model Architecture

LSTM (64 units)

Dropout layer

LSTM (32 units)

Dense (16)

Dense (1 output)

Purpose

Learn patterns from the last 60 minutes of price data

Predict short-term market movement for the next 10 minutes

Learning trading academy section for beginners to understand the concepts of trading like indicatiors and candlestick pattrns.

Chatbot features for help and getting live market insights with understanding market concepts with generalized chat.

<img width="1918" height="860" alt="image" src="https://github.com/user-attachments/assets/763fa90f-594b-4846-b477-47f5bbbb3ef8" />
<img width="1918" height="855" alt="image" src="https://github.com/user-attachments/assets/7f3db7f2-2aaf-4f25-aac6-bd282a053c29" />
<img width="1918" height="875" alt="image" src="https://github.com/user-attachments/assets/f0298363-0c5c-4d3b-986a-f1b3d87eb45b" />
<img width="1918" height="870" alt="image" src="https://github.com/user-attachments/assets/d9e01c26-0aba-4800-9ee9-6771cc7178c4" />
<img width="1918" height="867" alt="image" src="https://github.com/user-attachments/assets/f706432a-13a2-4038-833c-cdc868201383" />
<img width="1918" height="872" alt="image" src="https://github.com/user-attachments/assets/0bc2bb1b-33d4-41a7-a99a-23125f962865" />
<img width="1918" height="872" alt="image" src="https://github.com/user-attachments/assets/0c5f5d16-c26a-470a-81a7-31bd67640564" />




Live Dashboard Link
https://cryptoliveai.vercel.app/


