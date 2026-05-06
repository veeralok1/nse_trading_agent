# 📈 NSE Intraday Trading Agent

A production-ready, modular, multi-agent stock analysis system for the Indian stock market (NSE), powered by Yahoo Finance data — **100% free, no paid API required**.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+ 
- pip

### 2. Installation

```bash
# Clone / unzip the project
cd nse_trading_agent

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Web UI (Streamlit)

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### 4. Run the CLI

```bash
python orchestrator.py
```

---

## 🗂️ Project Structure

```
nse_trading_agent/
│
├── app.py                       # Streamlit web UI (5 tabs)
├── orchestrator.py              # Main pipeline coordinator + CLI
├── config.py                    # Symbols, parameters, settings
├── requirements.txt
│
├── agents/
│   ├── data_agent.py            # Yahoo Finance fetcher + cache
│   ├── indicator_agent.py       # RSI, MACD, VWAP, BB, MAs, ATR...
│   ├── signal_agent.py          # 6 trading strategies → signals
│   ├── risk_agent.py            # Stop loss, targets, position size
│   ├── ranking_agent.py         # Screen & rank top intraday stocks
│   └── conversational_agent.py  # NLP query parser & router
│
├── backtesting/
│   └── backtester.py            # Event-driven backtester
│
└── .cache/                      # Auto-created disk cache
```

---

## 🤖 Agent Architecture

| Agent | Responsibility |
|---|---|
| **DataAgent** | Fetches OHLCV from Yahoo Finance, caches to disk, handles retries |
| **IndicatorAgent** | RSI, MACD, VWAP, Bollinger Bands, MA 20/50/200, ATR, OBV, Stochastic, ADX |
| **SignalAgent** | 6 strategies: ORB, VWAP Bounce, Momentum Breakout, RSI Reversal, MA Cross, BB Squeeze |
| **RiskAgent** | ATR-based stop loss, 3-tier targets, position sizing, R:R ratio |
| **RankingAgent** | Composite score (volatility + liquidity + momentum + signal) → top-N |
| **ConversationalAgent** | Intent detection, entity extraction, routes to correct agents |

---

## 📊 UI Tabs

| Tab | What it shows |
|---|---|
| **Analysis** | Candlestick chart + indicators + signals + risk profile for any stock |
| **Top Stocks** | Ranked intraday candidates with score breakdown |
| **Chat** | Natural language interface — ask anything |
| **Backtest** | Run 5 strategies on any stock over 30/60/90 days |
| **Screener** | Live signal scanner across the Nifty 50 universe |

---

## 💬 Example Queries (Chat Tab)

```
Analyze RELIANCE
Is TCS good for intraday?
Top 5 stocks for today
What is the trend of HDFC Bank?
Price of INFY
Best stocks to buy today
Give me top 10 intraday picks
```

---

## ⚙️ Configuration

All parameters are in `config.py`:

```python
# Change the symbol universe
NIFTY_50_SYMBOLS = [...]

# Adjust indicator periods
INDICATOR_CFG.rsi_period = 14
INDICATOR_CFG.macd_fast  = 12

# Risk settings
RISK_CFG.capital              = 100_000   # INR
RISK_CFG.risk_per_trade_pct   = 0.01      # 1% per trade
RISK_CFG.atr_stop_multiplier  = 1.5
```

---

## 📡 Supported Strategies

| Strategy | Description |
|---|---|
| **ORB Breakout** | First 15-min range breakout with volume confirmation |
| **VWAP Bounce** | Price bounces from VWAP with RSI recovery |
| **Momentum Breakout** | New N-bar high/low with MACD + volume spike |
| **RSI Reversal** | Oversold/overbought RSI crossover |
| **MA Crossover** | Golden/Death cross (MA20 vs MA50) |
| **BB Squeeze** | Bollinger Band compression + momentum breakout |

---

## 📉 Backtesting

The backtester runs on **daily data** (avoids intraday data limits).  
Each strategy uses a fixed 1.5% stop-loss and 3% take-profit (1:2 R:R).

Metrics reported:
- Total trades, Win rate
- Total return, Average win/loss
- Profit factor, Max drawdown, Sharpe ratio
- Full equity curve + trade log

---

## 🗺️ Extending the System

### Add a New Strategy

```python
# In agents/signal_agent.py — add a new function:
def _my_custom_signal(df, symbol):
    ...
    return Signal(symbol=symbol, action="BUY", strategy="MY_STRATEGY", ...)

# Register it:
class SignalAgent:
    STRATEGIES = [..., _my_custom_signal]
```

### Add a New NSE Symbol

```python
# In config.py:
NIFTY_50_SYMBOLS.append("NEWSTOCK.NS")
SYMBOL_ALIASES["NEWSTOCK"] = "NEWSTOCK.NS"
```

### Broker Integration (Zerodha / Upstox)

Replace `DataAgent._fetch_with_retry()` with the broker's WebSocket feed.  
The rest of the pipeline is broker-agnostic.

---

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**.  
It does NOT constitute financial or investment advice.  
Always do your own research. Past performance ≠ future results.

---

## 📦 Tech Stack

| Layer | Library |
|---|---|
| Data | `yfinance` |
| Indicators | Pure NumPy/Pandas (no TA-Lib C dependency) |
| UI | `streamlit` + `plotly` |
| Cache | `pyarrow` / parquet |
| Backtest | Custom event-driven engine |
