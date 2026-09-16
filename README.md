# BTC Directional Signal — LSTM + Backtested Evaluation

An end-to-end machine learning pipeline that predicts short-term Bitcoin price direction using an LSTM trained on technical market indicators, then evaluates the resulting signal through out-of-sample backtesting rather than relying on classification accuracy alone.

## Live Demo

[Live Demo](https://crypto-price-predictor-burhan.streamlit.app/)

> Educational machine learning research project. This application does not provide financial or investment advice and is not intended for real-money trading.

## Problem

Predict whether BTC/USDT will move up or down over the next 10 hourly candles using only information available from historical price and volume data.

The goal is not to build a guaranteed-profit trading bot, but to investigate whether an LSTM can extract useful short-term directional patterns from technical market features and whether those patterns remain effective across different market regimes.

## Approach

* **Data:** 100,000 hourly BTC/USDT candles fetched from Binance's public API
* **Timeframe:** 1-hour candles
* **Prediction Horizon:** 10 hourly candles ahead
* **Features:** 15 technical and price-action features
* **Feature Engineering:** SMA, EMA, RSI, MACD, Bollinger Bands, ATR, returns, volatility and candle structure
* **Labeling:** Binary UP/DOWN classification based on the sign of the 10-candle-ahead return
* **Scaling:** StandardScaler fitted only on the training period
* **Sequence Length:** 60 hourly candles
* **Model:** LSTM (32 units) → Dropout → Dense → Dropout → Sigmoid
* **Training:** Chronological train/validation/test split with early stopping
* **Evaluation:** Out-of-sample backtesting using transaction costs, confidence-based trade filtering, Sharpe ratio, maximum drawdown and win rate

## Machine Learning Pipeline

```text
Binance Historical Data
        ↓
Technical Feature Engineering
        ↓
Chronological Train / Validation / Test Split
        ↓
StandardScaler
        ↓
60-Candle Sequences
        ↓
LSTM Neural Network
        ↓
UP / DOWN Probability
        ↓
Confidence-Based Signal
        ↓
Out-of-Sample Backtesting
        ↓
Risk-Adjusted Evaluation
```

## Features

The model uses price-derived features including:

* 1-hour return
* 5-hour return
* SMA 10
* SMA 50
* EMA 10
* EMA 50
* RSI 14
* MACD
* MACD histogram
* MACD signal
* Bollinger Band position
* ATR
* 10-period volatility
* Candle range
* Candle body

All model features are calculated using information available up to the prediction point.

## Labeling

The current model uses a binary directional target:

```text
Future Return > 0  → UP
Future Return ≤ 0  → DOWN
```

The future 10-candle return is used only to construct the training target and is never provided to the model as an input feature.

An earlier 3-class labeling approach was removed after identifying a label-generation issue in the previous experimental pipeline where threshold information was derived using the full dataset. The labeling and evaluation pipeline was subsequently redesigned to maintain chronological separation between training and future evaluation data.

## Model

The final model is intentionally compact:

```text
Input
  ↓
LSTM — 32 units
  ↓
Dropout — 0.4
  ↓
Dense — 16 units
  ↓
Dropout — 0.3
  ↓
Sigmoid Output
```

The model uses Adam optimization with a reduced learning rate and early stopping based on validation loss.

## Evaluation

The project does not treat classification accuracy as the primary success metric.

Predictions are converted into trading directions only when model confidence passes a defined threshold.

The backtesting framework evaluates:

* Sharpe ratio
* Maximum drawdown
* Win rate
* Total return
* Number of trades
* Transaction costs
* Performance against buy-and-hold

Transaction costs of 0.05% per executed trade are included in the simulation.

## Key Findings

The model achieved approximately:

* **53.6% directional win rate**
* **0.30 Sharpe ratio**
* Positive risk-adjusted performance during selected stable market periods
* Negative performance during a later volatile regime shift

Segmented out-of-sample backtesting revealed that the signal is strongly regime-dependent.

During some stable periods, the strategy produced strong risk-adjusted performance with Sharpe ratios above 3.5. However, performance degraded sharply during a later volatile regime, demonstrating that historical success in one market regime does not guarantee generalization to another.

Attempts to address the degradation using volatility-targeted position sizing and volatility-based circuit breakers did not eliminate the problem. This suggests that the primary limitation is the predictive signal itself rather than simply the position-sizing strategy.

The results highlight an important limitation of relying primarily on price-derived technical indicators: patterns learned from historical price behavior may not generalize reliably across changing market regimes.

## What This Demonstrates

Rather than optimizing for an impressive-looking accuracy number, this project focuses on rigorous machine learning evaluation.

It demonstrates:

* Chronological time-series splitting
* Prevention of test-set contamination
* Train-only feature scaling
* Sequence-based LSTM modeling
* Out-of-sample evaluation
* Transaction-cost-aware backtesting
* Risk-adjusted performance analysis
* Buy-and-hold comparison
* Segmented performance analysis
* Identification of regime-dependent model degradation

The project therefore treats financial machine learning as an evaluation problem rather than simply a model-training problem.

## Important Limitation

This system should not be interpreted as a profitable trading strategy or financial recommendation.

Cryptocurrency markets are highly noisy and non-stationary. A model can perform well during one market regime and fail during another.

The purpose of this project is to demonstrate an end-to-end machine learning workflow for financial time-series research and to investigate the limitations of technical-indicator-based prediction.

## Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Pandas TA Classic
* Plotly
* Streamlit
* Binance Public API

## Project Structure

```text
Crypto Signal System/
│
├── app.py
├── requirements.txt
├── README.md
│
└── saved_model/
    ├── btcusdt_lstm.keras
    ├── scaler.pkl
    ├── feature_columns.pkl
    ├── config.json
    └── backtest_results.json
```

## Running Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "Crypto Signal System"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application fetches recent BTC/USDT market data from Binance's public API, reproduces the same feature-engineering pipeline used during training, applies the saved scaler and generates the latest model prediction.

## Future Improvements

Possible research extensions include:

* Regime-detection layer to condition predictions on market state
* Walk-forward validation and retraining
* Additional market features beyond price history
* Order-flow data
* Funding rates
* On-chain metrics
* Multi-timeframe features
* Alternative sequence architectures
* Transformer-based time-series models
* More robust transaction-cost and slippage modeling
* Cross-market validation across different crypto assets

## Disclaimer

This project is for educational and research purposes only.

Nothing in this project constitutes financial, investment, trading or other professional advice. The predictions generated by the model should not be used as a basis for real-money trading decisions.

## Develop by
Burhan Arshad 
Computer Science Student
