# NSE Trading Agent — Master Build Prompt

> **Purpose:** Drop this file into a new Claude conversation (Cowork, Claude Code, or API) to rebuild the entire NSE Trading Agent project from scratch. Everything Claude needs to know — architecture, file structure, every agent's logic, UI behaviour, deployment rules, and known bug fixes — is captured here.

---

## 1. Project Overview

Build a **multi-agent NSE (India) stock trading assistant** deployable on **Streamlit Cloud** (free tier). The app helps retail investors and traders with:

- Live intraday analysis (5-min chart signals, VWAP, ORB)
- Swing / positional investment picks (daily charts, 1–12 month horizon)
- Single-stock deep-dive (indicators + signals + risk + price targets)
- Natural-language chat interface (rule-based + optional LLM commentary)
- Stock screener / ranker for Nifty 50 universe
- Historical backtester
- Telegram + email alerts (optional, via env vars)

**Tech stack:** Python 3.12, Streamlit, yfinance, Plotly, pure NumPy/Pandas indicators, OpenRouter LLM (optional). No broker API required for data.

---

## 2. File Structure

```
nse_trading_agent/
├── app.py                          # Streamlit UI — main entry point
├── config.py                       # All config, env resolution, symbol universe
├── requirements.txt
├── runtime.txt                     # python-3.12  (Streamlit Cloud)
├── .python-version                 # 3.12         (local / pyenv)
├── .env.example                    # Template for local secrets
├── README.md
└── agents/
    ├── __init__.py
    ├── data_agent.py               # yfinance fetch, cache, bulk download
    ├── indicator_agent.py          # RSI, MACD, VWAP, BB, ATR, OBV, ADX
    ├── signal_agent.py             # Strategy signals (ORB, VWAP bounce, momentum)
    ├── risk_agent.py               # Position sizing, stop-loss, targets, R:R
    ├── ranking_agent.py            # Rank stocks for intraday OR swing
    ├── conversational_agent.py     # NLP intent routing, response formatting
    ├── llm_agent.py                # OpenRouter / OpenAI LLM wrapper
    └── orchestrator.py             # TradingOrchestrator — wires all agents
```

---

## 3. config.py — Full Specification

### Secret resolution (priority order)
1. `st.secrets` (Streamlit Cloud Secrets dashboard)
2. `.env` file via python-dotenv
3. OS environment variables

```python
def _env(key, default=""):
    # LAZY resolution — called on every access, not at import time
    # This avoids a race where module-level constants evaluate before
    # Streamlit finishes loading secrets.
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val is not None:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, default)
```

**CRITICAL:** Never push secrets to `os.environ` in a one-shot block at import time — this causes a race condition where `ENABLE_LLM_CHAT` evaluates to `"false"` before Streamlit loads secrets, so the chat shows "AI OFF" even when keys are configured.

### Keys to expose via `_env()`
```
OPENROUTER_API_KEY       # primary LLM key
OPENROUTER_MODEL         # default: anthropic/claude-haiku-4-5
ANTHROPIC_API_KEY        # optional direct
OPENAI_API_KEY           # optional direct
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
ALERT_EMAIL_SENDER / ALERT_EMAIL_PASSWORD / ALERT_EMAIL_RECEIVER
ALERT_SMTP_HOST          # default: smtp.gmail.com
ALERT_SMTP_PORT          # default: 587
ZERODHA_API_KEY / ZERODHA_API_SECRET / ZERODHA_ACCESS_TOKEN
UPSTOX_API_KEY / UPSTOX_API_SECRET
ENABLE_LLM_CHAT          # "true" / "false"
LLM_PROVIDER             # "openrouter" | "anthropic" | "openai"
TRADING_CAPITAL          # default: 100000
RISK_PER_TRADE           # default: 0.01
```

### Symbol universe
- `NIFTY_50_SYMBOLS` — 50 symbols ending in `.NS`. **Remove HDFC.NS** (delisted Jul 2023, merged into HDFCBANK). Add `SHRIRAMFIN.NS` as replacement.
- `NIFTY_BANK_SYMBOLS` — 11 symbols
- `MIDCAP_SYMBOLS` — 35 symbols
- `DEFAULT_WATCHLIST` = `NIFTY_50_SYMBOLS[:20]`
- `SYMBOL_ALIASES` dict — maps "RELIANCE" → "RELIANCE.NS", "AIRTEL" → "BHARTIARTL.NS", "SBI" → "SBIN.NS", "HDFC" → "HDFCBANK.NS", etc.

### `DELISTED_SYMBOLS` set
```python
DELISTED_SYMBOLS: set = {
    "HDFC.NS",       # merged into HDFCBANK Jul 2023
    "INFRATEL.NS",   # merged into BHARTIARTL
    "ZEEL.NS",       # suspended at various points
}
```

### Dataclass configs
- `IndicatorConfig` — RSI(14), MACD(12,26,9), BB(20,2), MA(20,50,200), VWAP anchored
- `StrategyConfig` — ORB 15min, VWAP band ±0.2%, momentum lookback 5 bars
- `RiskConfig` — R:R 2.0, max stop 2%, ATR multiplier 1.5, capital from env
- `RankingConfig` — min avg daily vol 500k (intraday) / 200k (swing), min price ₹50

---

## 4. agents/data_agent.py — Full Specification

### Key design decisions
- **Suppress yfinance logger** at module level AND in `__init__()`:
  ```python
  logging.getLogger("yfinance").setLevel(logging.CRITICAL)
  ```
  This stops noisy `ERROR:yfinance: HTTP Error 404` spam for period-limited symbols.

- **Period cap map** — some valid Nifty 50 stocks return 404 on Yahoo for longer periods (e.g. `period=90d`) but work fine at `60d`. Keep this as a class-level dict:
  ```python
  _PERIOD_CAP: Dict[str, str] = {
      "TATAMOTORS.NS": "60d",
      "LTIM.NS":       "60d",
  }
  ```
  In `bulk_fetch()`, pull capped symbols out of the batch and fetch them individually with their safe period before running `yf.download()` on the rest.

- **Period fallback ladder** in `_fetch_with_retry()`:
  ```python
  _PERIOD_FALLBACKS = {
      "90d":  ["60d", "30d"],
      "180d": ["90d", "60d", "30d"],
      "365d": ["180d", "90d", "60d"],
      "30d":  ["20d"],
      "1d":   [],
  }
  ```
  When a symbol returns 404 / empty / YFPricesMissingError, try the next shorter period instead of retrying the same failing period.

- **Batch download** constants: `BATCH_SIZE = 8`, `BATCH_DELAY = 2.0s`, `threads=False` in `yf.download()` — gentler on shared Streamlit Cloud IPs.

- **Exponential backoff** for rate limits: `wait = (4 ** attempt) + jitter` seconds.

- **Cache** — disk parquet with TTL per interval:
  ```python
  CACHE_TTL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1d": 3600}
  ```

### Methods
| Method | Purpose |
|---|---|
| `fetch(symbol, interval, period)` | Single stock — cache → `_fetch_with_retry` |
| `_fetch_with_retry(symbol, interval, period)` | Retry + period fallback ladder |
| `get_live_quote(symbol)` | Dict with last_price, change, change_pct, volume |
| `fetch_historical(symbol, days)` | Daily OHLCV for N days |
| `fetch_multi_tf(symbol, intervals)` | Dict of DataFrames per interval |
| `bulk_fetch(symbols, interval, period)` | Batched download, capped symbols handled separately |
| `get_info(symbol)` | Fundamental data (P/E, market cap, sector, 52w range) |
| `get_universe(index)` | "nifty50" / "niftybank" / "midcap" / "all" |

---

## 5. agents/indicator_agent.py — Full Specification

All indicators computed in **pure NumPy/Pandas** — no TA library dependency.

### Indicators to compute
| Indicator | Parameters | Notes |
|---|---|---|
| RSI | 14 periods | Wilder's smoothing |
| MACD | 12, 26, 9 | Line + signal + histogram |
| Bollinger Bands | 20, ±2σ | Upper, lower, %B, bandwidth |
| VWAP | Session-anchored | Cumulative (price × vol) / cumulative vol |
| MA | 20, 50, 200 | Simple moving average |
| ATR | 14 | True range, rolling mean |
| OBV | — | On-balance volume |
| ADX | 14 | Average directional index + DI+/DI- |
| Momentum | 5 bars | (Close - Close[5]) / Close[5] × 100 |
| Volume Ratio | 20-bar avg | Current vol / rolling mean vol |

### Methods
| Method | Returns |
|---|---|
| `compute_all(df)` | df with all indicator columns appended |
| `get_summary(df)` | Dict of latest indicator values (rounded) |
| `trend_direction(df)` | "BULLISH" / "BEARISH" / "SIDEWAYS" |

### Trend logic
- **BULLISH** if: close > MA20 AND close > MA50 AND (close > MA200 OR MA20 > MA50)
- **BEARISH** if: close < MA20 AND close < MA50 AND (close < MA200 OR MA20 < MA50)
- Otherwise **SIDEWAYS**

---

## 6. agents/signal_agent.py — Full Specification

### Signal dataclass
```python
@dataclass
class Signal:
    symbol:     str
    strategy:   str        # "ORB_BREAKOUT" | "VWAP_BOUNCE" | "MOMENTUM_BREAKOUT"
    action:     str        # "BUY" | "SELL" | "HOLD"
    confidence: float      # 0.0–1.0
    entry:      float
    stop_loss:  float
    target:     float
    reasons:    List[str]
    timestamp:  datetime
```

### Strategies
**1. Opening Range Breakout (ORB)**
- Define range = High/Low of first 15 candles
- BUY signal when price breaks above range high with volume > 1.5× average
- SELL signal when price breaks below range low with volume confirmation
- Confidence scales with volume ratio (1.0 = 2× average volume)

**2. VWAP Bounce**
- BUY when: price is within ±0.2% of VWAP AND RSI < 55 AND price was below VWAP in previous bar (bounce)
- SELL when: price crosses below VWAP with RSI > 50
- Confidence: 0.6 base + boost if ADX > 25

**3. Momentum Breakout**
- BUY when: 5-bar momentum > 0, volume ratio > 2.0, RSI between 50–70, price > MA20
- SELL when: momentum < 0, volume spike, RSI > 70 or < 30
- Confidence: scaled by volume ratio and RSI proximity to extremes

### `aggregate_action(df, symbol)`
Returns `(action, confidence)` — weighted majority vote across all signals.

---

## 7. agents/risk_agent.py — Full Specification

### RiskProfile dataclass
```python
@dataclass
class RiskProfile:
    entry_price:     float
    stop_loss:       float
    target_1:        float
    target_2:        float
    target_3:        float
    risk_pct:        float      # stop distance as % of entry
    risk_reward:     float      # R:R ratio
    position_size:   int        # shares
    risk_amount_inr: float      # ₹ at risk
    notes:           str
```

### Stop-loss calculation
- Primary: `entry - (ATR × 1.5)`
- Capped at: `entry × (1 - 0.02)` (2% max stop)
- Whichever is tighter

### Targets
- T1 = entry + 1× risk (1:1)
- T2 = entry + 2× risk (1:2) — base case
- T3 = entry + 3× risk (1:3) — aggressive

### Position sizing
```python
risk_per_trade = capital × risk_per_trade_pct   # e.g. ₹1,000 on ₹100k capital
position_size  = floor(risk_per_trade / (entry - stop_loss))
```

---

## 8. agents/ranking_agent.py — Full Specification

### StockRank dataclass
```python
@dataclass
class StockRank:
    symbol, rank, total_score,
    volatility_score, liquidity_score, momentum_score, signal_score,
    trend, action, confidence, price, atr_pct, volume_ratio,
    signals, indicators
```

### `rank()` — Intraday mode (5-min bars)
Scoring weights:
```
volatility_score × 0.30   # ATR% — higher is better for day trading
liquidity_score  × 0.25   # volume vs 20-bar avg
signal_score     × 0.30   # from SignalAgent (BUY=1.0, HOLD=0.5, SELL=0.0)
momentum_score   × 0.15   # 5-bar price momentum
```
Volume filter: avg daily volume > 500,000. Price filter: ₹50–₹50,000.

### `rank_swing()` — Swing/investment mode (daily bars)
Scoring weights:
```
trend_score × 0.40    # price vs MA50/MA200 alignment
mom_score   × 0.25    # RSI-based daily momentum
sig_score   × 0.25    # signal strength
liq_score   × 0.10    # volume ratio
```
Volume filter: avg daily volume > 200,000 (relaxed vs intraday).
Trend score: close > MA200 (+40), close > MA50 (+30), MA50 > MA200 (+20), else 0.
RSI momentum: score = (RSI - 30) / 40, clamped 0–1.

### `to_dataframe(ranked)` → pd.DataFrame for display in Streamlit.

---

## 9. agents/conversational_agent.py — Full Specification

### Intent detection (order matters — first match wins)

```python
INTENT_PATTERNS = {
    "SWING_BEST": (
        r"\b(invest|investment|portfolio|positional|swing|long.?term|wealth)\b"
        r"|"
        r"\b(\d+)\s*(month|week|year|yr)s?\b"
        r"|"
        r"\b(3\s*month|6\s*month|1\s*year|quarterly)\b"
    ),
    "INTRADAY_BEST": (
        r"\b(best|top|good|strong|find)\b.{0,40}\b(intraday|day.?trad|scalp)\b"
        r"|"
        r"\b(intraday|day.?trad|scalp)\b.{0,40}\b(stock|pick|opportunit)\b"
    ),
    "ANALYZE":  r"\b(analyz|analyse|analysis|check|look at|tell me about)\b",
    "TREND":    r"\b(trend|direction|bullish|bearish|moving)\b",
    "SIGNAL":   r"\b(buy|sell|signal|action|should i|good for|worth|entry)\b",
    "PRICE":    r"\b(price|ltp|current|live|quote|last)\b",
    "COMPARE":  r"\b(compare|versus|vs\.?|better)\b",
    "HELP":     r"\b(help|how|what can|commands?|features?)\b",
    "BACKTEST": r"\b(backtest|back.?test|historical performance|past returns)\b",
    "SWING_BEST_GENERIC": r"\b(top|best|find|give|suggest|recommend)\b.{0,40}\b(stocks?|shares?|scrips?)\b",
}
```

### CRITICAL routing rule in `query()`

```python
def query(self, user_input):
    intent = detect_intent(user_input)
    symbol = extract_symbol(user_input)

    if intent == "SWING_BEST":
        if symbol is not None:
            # "HAL analysis for 3 months" → single-stock swing report, NOT screener
            horizon = extract_horizon(user_input)
            return self._swing_stock_analysis(symbol, horizon)
        return self._best_swing(extract_number(user_input), extract_horizon(user_input))

    if intent == "INTRADAY_BEST":
        if symbol is not None:
            # "Should I buy HAL intraday?" → single-stock signal analysis
            return self._signal_analysis(symbol)
        return self._best_intraday(extract_number(user_input))
```

Without this check, a query like *"give me HAL analysis for 3 months"* incorrectly routes to the top-N screener because "3 months" triggers `SWING_BEST` before any symbol extraction.

### `extract_symbol()` — comprehensive stop-word list

The function must scan all uppercase tokens (2–15 chars) and skip a full set of common English words that are not tickers:

```python
SKIP = {
    "THE", "AND", "FOR", "BUY", "SELL", "ARE", "TOP", "GOOD", "BEST",
    "WHAT", "GIVE", "TODAY", "INTRADAY", "TREND", "ANALYZE", "MARKET",
    "FIND", "SWING", "LONG", "TERM", "INVEST", "MONTH", "YEAR",
    "WEEKS", "STOCK", "STOCKS", "PICK", "PICKS", "SUGGEST", "TELL",
    "ANALYSIS", "ANALYSE", "CHECK", "LOOK", "ABOUT", "WITH", "THIS",
    "SHOULD", "VISION", "PERSPECTIVE", "TARGET", "RANGE", "PRICE",
    "QUARTER", "QUARTERLY", "WEEK", "DAY", "DAYS", "MONTHS", "YEARS",
    "PLEASE", "KNOW", "INVEST", "INVESTMENT", "PORTFOLIO", "RETURNS",
    "YES", "NOT", "LET", "CAN", "WILL", "WANT", "GIVE", "SHOW",
    "ME", "MY", "WE", "IS", "IT", "IN", "AT", "ON", "OF", "TO",
    "NIFTY", "SENSEX", "NSE", "BSE", "SEBI",
}
```

### `extract_horizon()` — map query text to period strings
- "1 year" / "12 month" / "annual" → "365d"
- "6 month" / "half year" → "180d"
- "3 month" / "quarter" / "90 day" → "90d"
- "1 month" / "30 day" → "30d"
- "week" / "7 day" → "5d"
- Default → "90d"

### `_swing_stock_analysis(symbol, period)` — single-stock investment report
Uses **daily bars** (not 5-min). Must produce:
1. Live quote + change %
2. Investment verdict: YES / AVOID / NEUTRAL based on (action + trend + RSI)
3. Key daily indicators: MA50 vs MA200 alignment, RSI with overbought/oversold labels, Bollinger position
4. Price target table (markdown):
   - Stop Loss (swing) = max(ATR × 1.5, price × 3%) below price
   - T1 = entry + 2R (conservative)
   - T2 = entry + 3R (base case) ← display % move
   - T3 = entry + 5R (aggressive)
5. Risk profile (position size, R:R, notes)
6. Active signals from SignalAgent
7. Disclaimer footer

### `_best_swing()` — top-N screener (no symbol in query)
Calls `rk.rank_swing(NIFTY_50_SYMBOLS[:50], period, top_n)` and formats table.
Labels: *"3-month Investment Perspective"*, *"Based on daily charts"*.

### `_best_intraday()` — top-N intraday screener
Calls `rk.rank(NIFTY_50_SYMBOLS[:25], interval="5m", period="5d", top_n)`.
Labels: *"Day Trading"*, *"5-min chart signals"*.

---

## 10. agents/llm_agent.py — Full Specification

### `LLMAgent`
```python
class LLMAgent:
    SYSTEM_PROMPT = """You are an expert NSE stock market analyst...
    - For intraday: momentum, VWAP, volume spikes, 5-min signals
    - For swing/investment: trend (MA50/MA200), RSI on daily chart,
      positional setups, 1–6 month price targets
    Always cite the indicator values provided. Be concise (3–5 sentences).
    End with a disclaimer that this is for educational purposes only."""

    def __init__(self):
        # Check ENABLE_LLM_CHAT via _env() — lazy, not import-time
        self.enabled = _env("ENABLE_LLM_CHAT", "false").lower() == "true"
        self.provider = _env("LLM_PROVIDER", "openrouter")
        self.client = None
        if self.enabled:
            self._init_client()

    def _init_client(self):
        # Use OpenAI SDK for both OpenRouter and direct OpenAI
        # OpenRouter is OpenAI-compatible: just change base_url
        from openai import OpenAI
        if self.provider == "openrouter":
            self.client = OpenAI(
                api_key=_env("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
            self.model = _env("OPENROUTER_MODEL", "anthropic/claude-haiku-4-5")
        elif self.provider == "anthropic":
            # Use anthropic SDK directly
            ...
        else:  # openai
            self.client = OpenAI(api_key=_env("OPENAI_API_KEY"))
            self.model = "gpt-4o-mini"

    def chat(self, user_message: str) -> str:
        """Return LLM commentary or empty string if disabled."""
        if not self.enabled or self.client is None:
            return ""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=400,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM error: %s", e)
            return ""

    @property
    def status(self) -> str:
        """Human-readable status for debug display."""
        if not self.enabled:
            return "⛔ AI OFF — Rule-based only"
        if self.client is None:
            return "⚠️ AI key missing"
        return f"✅ AI ON — {self.provider} / {self.model}"
```

---

## 11. agents/orchestrator.py — Full Specification

```python
class TradingOrchestrator:
    """Wires all agents together. Stateless — call methods directly."""

    def __init__(self):
        self.da  = DataAgent()
        self.ia  = IndicatorAgent()
        self.sa  = SignalAgent()
        self.ra  = RiskAgent()
        self.rk  = RankingAgent(data_agent=self.da, indicator_agent=self.ia, signal_agent=self.sa)
        self.ca  = ConversationalAgent(self.da, self.ia, self.sa, self.ra, self.rk)
        self.llm = LLMAgent()

    def analyse(self, symbol, interval="5m", period="2d"):
        """Full pipeline: fetch → indicators → signals → risk."""
        ...

    def screen(self, symbols=None, mode="intraday", top_n=5):
        """Run screener in intraday or swing mode."""
        ...

    def chat(self, user_input):
        """Route user query through ConversationalAgent + optional LLM."""
        resp = self.ca.query(user_input)
        if self.llm.enabled and resp["intent"] in ("SWING_BEST","INTRADAY_BEST","SWING_STOCK"):
            # Generate LLM commentary on top of rule-based output
            stocks = resp["data"].get("stocks", [])
            if stocks:
                top_list = ", ".join(
                    f"{r.symbol.replace('.NS','')} ({r.action} {r.confidence:.0%})"
                    for r in stocks[:5]
                )
                ai_reply = self.llm.chat(
                    f"The user asked: \"{user_input}\"\n\n"
                    f"Rule-based screener returned: {top_list}\n\n"
                    "Provide a concise market context and investment perspective."
                )
                if ai_reply:
                    resp["text"] += f"\n\n---\n🤖 **AI Perspective:**\n{ai_reply}"
        return resp
```

---

## 12. app.py — Streamlit UI Specification

### Agent initialisation
Use `@st.cache_resource` for the orchestrator so agents are not re-created on every Streamlit rerun:
```python
@st.cache_resource
def get_orchestrator():
    return TradingOrchestrator()
```

### Layout — 5 tabs
```
📊 Dashboard | 📈 Analysis | 🏆 Top Stocks | 💬 Chat | 🔄 Backtester
```

### Tab 1: Dashboard
- Sidebar with watchlist (default = `NIFTY_50_SYMBOLS[:10]`)
- Add/remove symbols via text input
- Show live quote cards: symbol, price, change %, volume bar
- Market status indicator (IST 9:15–15:30)
- Nifty 50 index price

### Tab 2: Analysis
- Symbol selector (dropdown + free text)
- Timeframe selector: 1m / 5m / 15m / 30m / 1h / 1d
- Period selector: 1d / 5d / 30d / 90d
- Plotly candlestick chart with:
  - MA20, MA50 overlays
  - VWAP line
  - Bollinger Bands (upper/lower shaded)
  - Volume bar subplot
  - RSI subplot (30/70 reference lines)
  - MACD subplot (line + signal + histogram)
- Signal annotations on chart (▲ BUY / ▽ SELL markers)
- Metrics row: RSI, ATR, Volume Ratio, Trend, Signal
- Risk profile table (entry / stop / T1 / T2 / T3 / position size)

### Tab 3: Top Stocks
- Mode radio: "📈 Swing / Investment" (default) | "⚡ Intraday"
- Horizon dropdown (only visible in Swing mode): 1 month / 3 months / 6 months / 1 year
- "🔍 Screen Now" button → triggers `rk.rank()` or `rk.rank_swing()`
- Results table: Rank, Symbol, Action (coloured badge), Price, Score, Trend
- Expandable score breakdown (only render when results are not empty)
- **BUG TO AVOID:** Do not render score breakdown chart using the raw `ranked` list when the display table is filtered — gate both on `not display_df.empty`

### Tab 4: Chat
- `st.chat_input` / `st.chat_message` interface
- Full conversation history in `st.session_state.messages`
- Debug expander "🔧 AI Config Debug" showing live values of:
  - `ENABLE_LLM_CHAT`
  - API key presence (True/False, never the actual key)
  - Model name
  - `llm.status`
- For SWING_BEST / INTRADAY_BEST responses: show ranked stocks as a formatted list
- For SWING_STOCK / ANALYZE: show full analysis + price target table
- LLM commentary (if enabled) appended after rule-based text with "🤖 AI Perspective:" label

### Tab 5: Backtester
- Symbol + strategy selector
- Date range picker
- "Run Backtest" button
- Results: equity curve (Plotly line), metrics table (total return %, CAGR, Sharpe, max drawdown, win rate, total trades)

### Plotly chart helpers
- Use `go.Figure` with `make_subplots(rows=4, shared_xaxes=True)` for multi-panel charts
- Template: `"plotly_dark"` or `"plotly"` matching Streamlit theme
- Responsive width: `use_container_width=True`

---

## 13. requirements.txt

```
yfinance>=0.2.36,<0.3.0
pandas>=2.0.0,<3.0.0
numpy>=1.24.0,<2.0.0       # numpy 2.x breaks several packages
pyarrow>=14.0.0             # parquet cache support
plotly>=5.18.0
streamlit>=1.32.0
streamlit-extras>=0.4.0
requests>=2.31.0
python-dateutil>=2.8.0
python-dotenv>=1.0.0
openai>=1.14.0              # used for OpenRouter AND direct OpenAI
```

**Do NOT include `pandas-ta`** — it is incompatible with Python 3.14 and is not used. All indicators are computed in pure NumPy/Pandas inside `indicator_agent.py`.

---

## 14. runtime.txt and .python-version

```
# runtime.txt (Streamlit Cloud)
python-3.12

# .python-version (pyenv / local)
3.12
```

Python 3.14 breaks several packages. Pin to 3.12.

---

## 15. Streamlit Cloud Deployment

### API keys
Do NOT commit `.env` to git. On Streamlit Cloud:
1. Go to your app → **Settings → Secrets**
2. Add secrets in TOML format:
   ```toml
   OPENROUTER_API_KEY = "sk-or-..."
   OPENROUTER_MODEL = "anthropic/claude-haiku-4-5"
   ENABLE_LLM_CHAT = "true"
   LLM_PROVIDER = "openrouter"
   TRADING_CAPITAL = "100000"
   RISK_PER_TRADE = "0.01"
   ```
3. The `_env()` function in `config.py` reads from `st.secrets` automatically.

### `.env.example` to commit
```
OPENROUTER_API_KEY=
OPENROUTER_MODEL=anthropic/claude-haiku-4-5
ENABLE_LLM_CHAT=false
LLM_PROVIDER=openrouter
TRADING_CAPITAL=100000
RISK_PER_TRADE=0.01
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

---

## 16. Known Bugs and Their Fixes

### Bug 1 — "AI OFF" even when secrets are configured
**Cause:** Old `config.py` pushed `st.secrets` to `os.environ` in a one-shot block at import time. Module-level constants (like `ENABLE_LLM_CHAT`) evaluated before that block ran = always `"false"`.
**Fix:** Make `_env()` call `st.secrets.get(key)` lazily on every invocation. Never push to `os.environ`.

### Bug 2 — Chat routes "HAL analysis for 3 months" to top-N screener
**Cause:** `SWING_BEST` intent matched "3 months" before symbol was checked. `query()` immediately called `_best_swing()`.
**Fix:** In `query()`, after detecting `SWING_BEST`, check `if symbol is not None` → call `_swing_stock_analysis(symbol, horizon)` instead.

### Bug 3 — Top Stocks tab shows score breakdown with empty table
**Cause:** Score breakdown used raw `ranked` list; display table used filtered `display_df`. Breakdown rendered even when table was empty after signal/action filter.
**Fix:** Gate both the table and score breakdown on `if not display_df.empty`. Only include symbols in `display_df` when rendering the chart.

### Bug 4 — Streamlit Cloud build fails with Python 3.14
**Cause:** `pandas-ta` in requirements.txt; incompatible with Python 3.14.
**Fix:** Remove `pandas-ta` entirely. Add `runtime.txt` with `python-3.12`.

### Bug 5 — HDFC.NS causes YFRateLimitError / missing data errors
**Cause:** HDFC Ltd merged into HDFCBANK in July 2023 — delisted.
**Fix:** Remove from `NIFTY_50_SYMBOLS`, add to `DELISTED_SYMBOLS`. Route alias "HDFC" → "HDFCBANK.NS".

### Bug 6 — TATAMOTORS.NS and LTIM.NS cause YFPricesMissingError(period=90d)
**Cause:** Yahoo Finance returns 404 for these symbols with `period=90d` (data gap issue on their side).
**Fix:**
- Add both to `_PERIOD_CAP = {"TATAMOTORS.NS": "60d", "LTIM.NS": "60d"}` in `DataAgent`
- In `bulk_fetch()`, pull capped symbols out before `yf.download()` and fetch individually
- Add period fallback ladder in `_fetch_with_retry()` — try 60d then 30d if 90d fails
- Suppress yfinance logger: `logging.getLogger("yfinance").setLevel(logging.CRITICAL)`

### Bug 7 — `extract_symbol()` misses short tickers like HAL, BEL
**Cause:** Old regex required 3+ chars or checked only the first token.
**Fix:** Scan all `\b[A-Z][A-Z0-9&\-]{1,14}\b` tokens, skip a comprehensive stop-word set. Take the first plausible match.

---

## 17. Build Instructions for Claude

When building this project from this prompt:

1. **Read this entire prompt first** before writing any code.
2. Create `config.py` first — all other agents import from it.
3. Create agents in this order: `data_agent → indicator_agent → signal_agent → risk_agent → ranking_agent → conversational_agent → llm_agent → orchestrator`.
4. Create `app.py` last.
5. Apply **all** bug fixes from Section 16 proactively — do not wait for errors.
6. **Never** add `pandas-ta` to `requirements.txt`.
7. **Always** use `runtime.txt` = `python-3.12`.
8. **Always** use the lazy `_env()` pattern — never push to `os.environ` at import time.
9. After writing all files, verify imports with a quick Python syntax check.
10. Save all files under `nse_trading_agent/` directory.

---

*Generated from a working Streamlit Cloud deployment. Last updated: May 2026.*
