import streamlit as st
import requests
import pandas as pd
import numpy as np
import pickle
import json
import pandas_ta_classic as ta
import tensorflow as tf
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(
    page_title="BTC/USDT ML Signal",
    page_icon="₿",
    layout="wide"
)

MODEL_PATH = "saved_model/btcusdt_lstm.keras"
SCALER_PATH = "saved_model/scaler.pkl"
FEATURE_PATH = "saved_model/feature_columns.pkl"
CONFIG_PATH = "saved_model/config.json"

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(FEATURE_PATH, "rb") as f:
    feature_cols = pickle.load(f)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)

SYMBOL = config["symbol"]
INTERVAL = config["interval"]
WINDOW_SIZE = config["window_size"]
FUTURE_WINDOW = config["future_window"]

@st.cache_data(ttl=60)
def get_binance_data(symbol="BTCUSDT", interval="1h", limit=150):
    url = "https://data-api.binance.vision/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore"
    ]

    df = pd.DataFrame(data, columns=columns)

    df = df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms"
    )

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = df[column].astype(float)

    df = df.drop_duplicates("open_time")
    df = df.sort_values("open_time")
    df = df.reset_index(drop=True)

    return df

def create_features(df):
    df = df.copy()

    df["SMA_10"] = ta.sma(
        df["close"],
        length=10
    )

    df["SMA_50"] = ta.sma(
        df["close"],
        length=50
    )

    df["EMA_10"] = ta.ema(
        df["close"],
        length=10
    )

    df["EMA_50"] = ta.ema(
        df["close"],
        length=50
    )

    df["RSI"] = ta.rsi(
        df["close"],
        length=14
    )

    macd = ta.macd(df["close"])
    df = pd.concat([df, macd], axis=1)

    bbands = ta.bbands(
        df["close"],
        length=20
    )

    df = pd.concat([df, bbands], axis=1)

    df["ATR"] = ta.atr(
        df["high"],
        df["low"],
        df["close"],
        length=14
    )

    df["return_1h"] = df["close"].pct_change(1)
    df["return_5h"] = df["close"].pct_change(5)

    df["volatility_10"] = (
        df["return_1h"]
        .rolling(10)
        .std()
    )

    df["candle_range"] = (
        (df["high"] - df["low"])
        / df["close"]
    )

    df["candle_body"] = (
        (df["close"] - df["open"])
        / df["open"]
    )

    df = df.dropna().reset_index(drop=True)

    return df

def get_prediction(df):
    feature_data = df[feature_cols].copy()

    scaled = scaler.transform(feature_data)

    sequence = scaled[-WINDOW_SIZE:]

    sequence = np.expand_dims(
        sequence,
        axis=0
    )

    probability = float(
        model.predict(
            sequence,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:
        signal = "UP"
        confidence = probability
    else:
        signal = "DOWN"
        confidence = 1 - probability

    return signal, probability, confidence

st.title("BTC/USDT ML Signal Research")

st.caption(
    "LSTM-based directional prediction using historical BTC/USDT market data"
)

st.info(
    "Educational ML research project. This system predicts directional movement "
    "and is not a financial trading or investment recommendation."
)

try:
    df = get_binance_data(
        symbol=SYMBOL,
        interval=INTERVAL,
        limit=150
    )

    df = create_features(df)

    signal, probability, confidence = get_prediction(df)

    latest_price = df["close"].iloc[-1]
    latest_time = df["open_time"].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "BTC Price",
            f"${latest_price:,.2f}"
        )

    with col2:
        st.metric(
            "Model Signal",
            signal
        )

    with col3:
        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

    with col4:
        st.metric(
            "Prediction Horizon",
            f"{FUTURE_WINDOW} hours"
        )

    st.divider()

    st.subheader("Price")

    chart_df = df.tail(72)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_df["open_time"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="BTC/USDT"
        )
    )

    fig.update_layout(
        height=500,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Model Output")

    probability_df = pd.DataFrame(
        {
            "Direction": ["DOWN", "UP"],
            "Probability": [
                1 - probability,
                probability
            ]
        }
    )

    st.bar_chart(
        probability_df.set_index("Direction")
    )

    st.subheader("Latest Market Data")

    display_df = df[
        [
            "open_time",
            "close",
            "RSI",
            "return_1h",
            "return_5h",
            "volatility_10",
            "candle_range",
            "candle_body"
        ]
    ].tail(10)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Project Overview")

    st.markdown(
        """
This project uses an LSTM neural network to predict the directional
movement of BTC/USDT over a future time horizon.

The model is trained on historical market data and technical features
including momentum, volatility, trend and price-action indicators.

Model evaluation focuses on out-of-sample performance and risk-aware
backtesting rather than optimizing raw classification accuracy.
"""
    )

    st.subheader("Research Notes")

    st.markdown(
        """
> Built and backtested an LSTM-based directional trading signal for BTC/USDT,
> evaluating performance through risk-adjusted metrics such as Sharpe ratio,
> maximum drawdown and win rate rather than raw accuracy.

> Identified and fixed a label-leakage bug in an earlier iteration; redesigned
> the labeling and evaluation pipeline to avoid test-set contamination.

> Conducted segmented out-of-sample backtesting that revealed regime-dependent
> signal degradation, demonstrating rigorous ML evaluation practice over naive
> metric optimization.
"""
    )

    st.caption(
        f"Last processed candle: {latest_time} UTC"
    )

except Exception as e:
    st.error(
        f"Unable to generate prediction: {str(e)}"
    )