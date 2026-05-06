# NSE Trading Agent — Claude Project Context

> This file is read automatically by Claude (Cowork / Claude Code) at the start of every session.
> It contains the complete specification, architecture, known bugs, and build instructions for this project.
> Claude should treat this as the authoritative source of truth for all decisions in this codebase.

---

## What This Project Is

A **multi-agent NSE (India) stock trading assistant** deployable on **Streamlit Cloud** (free tier).

**What it does:**
- Live intraday analysis — 5-min chart signals, VWAP, Opening Range Breakout
- Swing / positional investment picks — daily charts, 1–12 month horizons
- Single-stock deep-dive — indicators + signals + risk profile + price targets (T1/T2/T3)
- Natural-language chat — rule-based intent routing + optional LLM (OpenRouter / OpenAI) commentary
- Stock screener — ranks Nifty 50 universe for intraday OR swing mode
- Historical backtester — equity curve, Sharpe, drawdown, win rate
- Optional alerts — Telegram + email via env vars

**Tech stack:** Python 3.12 · Streamlit · yfinance · Plotly · NumPy/Pandas (all indicators pure, no TA lib) · OpenRouter LLM (optional, off by default)

**Owner / email:** veeralok@gmail.com

---

## File Structure

```
nse_trading_agent/
├── CLAUDE.md                       ← this file
├── PROJECT_PROMPT.md               ← long-form rebuild prompt (full details)
├── app.py                          ← Streamlit UI — main entry point
├── config.py                       ← env resolution, symbol universe, dataclass configs
├── requirements.txt
├── runtime.txt                     ← python-3.12  (Streamlit Cloud pin)
├── .python-version                 ← 3.12          (local / pyenv)
├── .env.example                    ← template (never commit real secrets)
├── README.md
└── agents/
    ├── __init__.py
    ├── data_agent.py               ← yfinance fetch, disk cache, bulk batched download
    ├── indicator_agent.py          ← RSI, MACD, VWAP, BB, ATR, OBV, ADX (pure NumPy)
    ├── signal_agent.py             ← ORB, VWAP Bounce, Momentum Breakout strategies
    ├── risk_agent.py               ← position sizing, ATR stop, T1/T2/T3 targets
    ├── ranking_agent.py            ← rank() intraday + rank_swing() positional
    ├── conversational_agent.py     ← NLP intent detection, response formatting
    ├── llm_agent.py                ← OpenRouter/OpenAI wrapper (optional)
    └── orchestrator.py             ← TradingOrchestrator — wires all agents
```

---

## Critical Rules Claude Must Always Follow in This Project

### 1. Secret resolution — always lazy, never import-time
`config.py` uses a `_env(key)` function that calls `st.secrets.get(key)` on every invocation. **Never** push `st.secrets` to `os.environ` in a one-shot block at module level — this causes a race where `ENABLE_LLM_CHAT` evaluates to `"false"` before Streamlit finishes loading secrets (the "AI OFF bug").

```python
def _env(key, default=""):
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val is not None:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, default)
```

### 2. Python version — always 3.12
`runtime.txt` = `python-3.12`, `.python-version` = `3.12`. Never upgrade to 3.14+ — it breaks packages.

### 3. Never add pandas-ta to requirements.txt
All indicators are computed in pure NumPy/Pandas inside `indicator_agent.py`. `pandas-ta` is not imported anywhere and breaks Python 3.14 builds.

### 4. Chat routing — symbol takes priority over intent
When a user query contains a specific stock symbol (e.g. "HAL", "RELIANCE"), route to a **single-stock analysis** even if the intent matched `SWING_BEST` or `INTRADAY_BEST`. The screeners (`_best_swing`, `_best_intraday`) only run when **no symbol is present** in the query.

```python
if intent == "SWING_BEST":
    if symbol is not None:
        return self._swing_stock_analysis(symbol, horizon)   # single stock
    return self._best_swing(n, horizon)                      # screener
```

### 5. yfinance error suppression
Set `logging.getLogger("yfinance").setLevel(logging.CRITICAL)` at both module level and in `DataAgent.__init__()`. This suppresses 404 spam from period-limited symbols (TATAMOTORS.NS, LTIM.NS).

### 6. Period-capped symbols
TATAMOTORS.NS and LTIM.NS return 404 from Yahoo Finance for `period=90d`. They are kept in `DataAgent._PERIOD_CAP = {"TATAMOTORS.NS": "60d", "LTIM.NS": "60d"}` and fetched individually with their safe period in `bulk_fetch()`, never passed to `yf.download()` with a longer period.

### 7. HDFC.NS is delisted
HDFC Ltd merged into HDFCBANK in July 2023. It is in `DELISTED_SYMBOLS` and removed from `NIFTY_50_SYMBOLS`. The alias `"HDFC"` maps to `"HDFCBANK.NS"`.

---

## Agent Architecture Summary

| Agent | Responsibility |
|---|---|
| `DataAgent` | Fetch/cache OHLCV from yfinance. Batched bulk download. Period fallback ladder. |
| `IndicatorAgent` | Compute RSI, MACD, VWAP, Bollinger, MA20/50/200, ATR, OBV, ADX, Momentum from raw OHLCV. |
| `SignalAgent` | Generate BUY/SELL/HOLD signals using ORB, VWAP Bounce, Momentum Breakout strategies. |
| `RiskAgent` | ATR-based stop loss, T1/T2/T3 targets, position sizing from capital + risk%. |
| `RankingAgent` | `rank()` for intraday (5m, ATR/vol/signal/momentum), `rank_swing()` for positional (daily, trend/RSI/signal/liquidity). |
| `ConversationalAgent` | Regex intent detection + symbol extraction → route to correct handler. |
| `LLMAgent` | Optional OpenRouter/OpenAI commentary layer. Reads secrets lazily at chat time. |
| `TradingOrchestrator` | Singleton (cached with `@st.cache_resource`). Wires all agents. Entry point for `app.py`. |

---

## Scoring Formulas

### Intraday (`rank()`) — 5-min bars
```
total = volatility_score×0.30 + liquidity_score×0.25 + signal_score×0.30 + momentum_score×0.15
```
Volume filter: avg daily vol > 500,000.

### Swing (`rank_swing()`) — daily bars
```
total = trend_score×0.40 + mom_score×0.25 + sig_score×0.25 + liq_score×0.10
```
Trend score: price>MA200(+40) + price>MA50(+30) + MA50>MA200(+20).
RSI momentum: `(RSI - 30) / 40`, clamped 0–1.
Volume filter: avg daily vol > 200,000.

---

## Intent Patterns (conversational_agent.py)

```
SWING_BEST        → invest | investment | swing | \d+ month/year  (checked first)
INTRADAY_BEST     → intraday | day trading | scalp (with stock/best qualifier)
ANALYZE           → analyz | analyse | check | look at | tell me about
TREND             → trend | direction | bullish | bearish
SIGNAL            → buy | sell | signal | should i | entry
PRICE             → price | ltp | current | live | quote
BACKTEST          → backtest | historical performance
SWING_BEST_GENERIC→ top/best/suggest + stocks (fallback screener)
```

`extract_symbol()` scans all uppercase 2–15 char tokens and skips a comprehensive stop-word set (THE, AND, FOR, INVEST, STOCKS, NIFTY, etc.). First non-skip token is treated as the ticker.

---

## Streamlit UI — Tab Layout

```
📊 Dashboard | 📈 Analysis | 🏆 Top Stocks | 💬 Chat | 🔄 Backtester
```

- **Dashboard** — live quote cards for watchlist, market status (IST hours)
- **Analysis** — candlestick + MA/VWAP/BB/Volume/RSI/MACD subplots, signal markers, risk table
- **Top Stocks** — Swing/Intraday mode toggle, horizon picker, screener results table + score breakdown (gated on non-empty results)
- **Chat** — `st.chat_input` interface, AI Config debug expander, LLM commentary appended if enabled
- **Backtester** — equity curve, metrics (CAGR, Sharpe, drawdown, win rate)

---

## Deployment (Streamlit Cloud)

1. Push code to GitHub (do NOT commit `.env`)
2. In Streamlit Cloud → your app → **Settings → Secrets**, add TOML:
   ```toml
   OPENROUTER_API_KEY = "sk-or-..."
   OPENROUTER_MODEL   = "anthropic/claude-haiku-4-5"
   ENABLE_LLM_CHAT    = "true"
   LLM_PROVIDER       = "openrouter"
   TRADING_CAPITAL    = "100000"
   RISK_PER_TRADE     = "0.01"
   ```
3. `_env()` in `config.py` reads from `st.secrets` automatically — no code change needed.

---

## Known Bug Log

| # | Bug | Root Cause | Fix Applied |
|---|---|---|---|
| 1 | "AI OFF" despite secrets set | Import-time `os.environ` push race | Lazy `_env()` reads `st.secrets` on every call |
| 2 | "HAL 3 months" → screener instead of HAL analysis | `SWING_BEST` matched before symbol check | Symbol-first routing in `query()` |
| 3 | Score breakdown with empty table | Used raw `ranked` not filtered `display_df` | Gate both on `not display_df.empty` |
| 4 | Streamlit Cloud Python 3.14 build fail | `pandas-ta` in requirements | Removed; `runtime.txt` pins Python 3.12 |
| 5 | HDFC.NS 404 errors | Delisted Jul 2023 | Removed from symbols; alias → HDFCBANK.NS |
| 6 | TATAMOTORS / LTIM 404 (period=90d) | Yahoo data gap | `_PERIOD_CAP` dict; fetch individually at 60d |
| 7 | Short tickers (HAL, BEL) not extracted | Regex too narrow | Scan all tokens with comprehensive skip-list |

---

## How to Rebuild From Scratch

If asked to rebuild this project, read `PROJECT_PROMPT.md` (full detail) and follow this order:

1. `config.py` first (all agents import from it)
2. `agents/data_agent.py`
3. `agents/indicator_agent.py`
4. `agents/signal_agent.py`
5. `agents/risk_agent.py`
6. `agents/ranking_agent.py`
7. `agents/conversational_agent.py`
8. `agents/llm_agent.py`
9. `agents/orchestrator.py`
10. `app.py` last
11. `requirements.txt`, `runtime.txt`, `.python-version`, `.env.example`
12. Apply all 7 bug fixes from the bug log proactively — do not wait for errors.

---

*Last updated: May 2026 — veeralok@gmail.com*
