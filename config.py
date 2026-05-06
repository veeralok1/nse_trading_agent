"""
config.py — Central configuration for NSE Trading Agent System

Secret resolution priority (first match wins):
  1. Streamlit Cloud Secrets  (st.secrets)  — used in production
  2. .env file on disk        (python-dotenv) — used in local dev
  3. OS environment variables — fallback
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict

# ─────────────────────────────────────────────
# STEP 1 — Try Streamlit secrets (production)
# ─────────────────────────────────────────────
# When deployed on Streamlit Cloud, secrets added via the dashboard
# are available as st.secrets.  We push them into os.environ so the
# rest of config.py works identically in both environments.

try:
    import streamlit as st
    if hasattr(st, "secrets") and len(st.secrets) > 0:
        for _k, _v in st.secrets.items():
            # Only set if not already present (don't override local env)
            if _k not in os.environ:
                os.environ[_k] = str(_v)
except Exception:
    pass   # Not running inside Streamlit — skip silently


# ─────────────────────────────────────────────
# STEP 2 — Load .env for local development
# ─────────────────────────────────────────────
# Values already set from st.secrets above are NOT overridden
# because override=False is the default in load_dotenv.

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass   # python-dotenv not installed; rely on os.environ


def _env(key: str, default: str = "") -> str:
    """Read a config value — works in both local and Streamlit Cloud."""
    return os.environ.get(key, default)


# ─────────────────────────────────────────────
# API KEYS  (loaded from .env)
# ─────────────────────────────────────────────

# OpenRouter — single key for 200+ models (recommended)
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = _env("OPENROUTER_MODEL", "anthropic/claude-haiku-4-5")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"   # OpenAI-compatible endpoint

# Direct provider keys (optional, only needed if not using OpenRouter)
ANTHROPIC_API_KEY  = _env("ANTHROPIC_API_KEY")
OPENAI_API_KEY     = _env("OPENAI_API_KEY")

# Telegram alerts
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _env("TELEGRAM_CHAT_ID")

# Email alerts
ALERT_EMAIL_SENDER   = _env("ALERT_EMAIL_SENDER")
ALERT_EMAIL_PASSWORD = _env("ALERT_EMAIL_PASSWORD")
ALERT_EMAIL_RECEIVER = _env("ALERT_EMAIL_RECEIVER")
ALERT_SMTP_HOST      = _env("ALERT_SMTP_HOST", "smtp.gmail.com")
ALERT_SMTP_PORT      = int(_env("ALERT_SMTP_PORT", "587"))

# Broker — Zerodha
ZERODHA_API_KEY      = _env("ZERODHA_API_KEY")
ZERODHA_API_SECRET   = _env("ZERODHA_API_SECRET")
ZERODHA_ACCESS_TOKEN = _env("ZERODHA_ACCESS_TOKEN")

# Broker — Upstox
UPSTOX_API_KEY    = _env("UPSTOX_API_KEY")
UPSTOX_API_SECRET = _env("UPSTOX_API_SECRET")

# Feature flags
ENABLE_LLM_CHAT = _env("ENABLE_LLM_CHAT", "false").lower() == "true"
LLM_PROVIDER    = _env("LLM_PROVIDER", "openrouter")  # "openrouter" | "anthropic" | "openai"

# ─────────────────────────────────────────────
# NSE SYMBOL UNIVERSE
# ─────────────────────────────────────────────

NIFTY_50_SYMBOLS = [
    # HDFC.NS removed — HDFC Ltd merged into HDFCBANK in Jul 2023 (delisted)
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS",
    "BHARTIARTL.NS", "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS",
    "KOTAKBANK.NS", "LT.NS", "LTIM.NS", "MARUTI.NS",
    "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS",
    "M&M.NS", "SHRIRAMFIN.NS",   # SHRIRAMFIN replaced HDFC.NS in Nifty50
]

# Symbols known to be delisted, suspended, or renamed on Yahoo Finance.
# bulk_fetch will skip these automatically — update as needed.
DELISTED_SYMBOLS: set = {
    "HDFC.NS",       # merged into HDFCBANK Jul 2023
    "INFRATEL.NS",   # merged into BHARTIARTL
    "ZEEL.NS",       # suspended at various points
}

NIFTY_BANK_SYMBOLS = [
    "AUBANK.NS", "AXISBANK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "IDFCFIRSTB.NS", "INDUSINDBK.NS",
    "KOTAKBANK.NS", "PNB.NS", "SBIN.NS",
]

MIDCAP_SYMBOLS = [
    "ABFRL.NS", "AARTIIND.NS", "ACC.NS", "APLAPOLLO.NS",
    "BALKRISIND.NS", "BHARATFORG.NS", "CANBK.NS", "CHOLAFIN.NS",
    "CONCOR.NS", "COROMANDEL.NS", "CUMMINSIND.NS", "DEEPAKNTR.NS",
    "GLENMARK.NS", "GODREJPROP.NS", "HAVELLS.NS", "IPCALAB.NS",
    "LICHSGFIN.NS", "LUPIN.NS", "MANAPPURAM.NS", "MARICO.NS",
    "MPHASIS.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS", "NAVINFLUOR.NS",
    "OFSS.NS", "PERSISTENT.NS", "PETRONET.NS", "PIIND.NS",
    "POLYCAB.NS", "SAIL.NS", "TATACOMM.NS", "TATACONSUM.NS",
    "TORNTPHARM.NS", "TRENT.NS", "VOLTAS.NS",
]

# All watchlist symbols
DEFAULT_WATCHLIST = NIFTY_50_SYMBOLS[:20]  # default to top-20 Nifty50

# Symbol aliases for user-friendly queries
SYMBOL_ALIASES: Dict[str, str] = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "INFOSYS": "INFY.NS",
    "HDFC": "HDFCBANK.NS",        # HDFC Ltd merged → route to HDFCBANK
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "WIPRO": "WIPRO.NS",
    "TECHM": "TECHM.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "MARUTI": "MARUTI.NS",
    "AXISBANK": "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "DRREDDY": "DRREDDY.NS",
    "CIPLA": "CIPLA.NS",
    "TITAN": "TITAN.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "LTIM": "LTIM.NS",
    "LT": "LT.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "AIRTEL": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "NTPC": "NTPC.NS",
    "POWERGRID": "POWERGRID.NS",
    "ONGC": "ONGC.NS",
    "BPCL": "BPCL.NS",
    "COALINDIA": "COALINDIA.NS",
}

# ─────────────────────────────────────────────
# TIMEFRAMES
# ─────────────────────────────────────────────

INTRADAY_INTERVALS = ["1m", "5m", "15m", "30m", "1h"]
HISTORICAL_PERIOD = "30d"   # for daily data
INTRADAY_PERIOD   = "1d"    # for intraday data
SWING_PERIOD      = "90d"   # for swing / positional

# ─────────────────────────────────────────────
# INDICATOR PARAMETERS
# ─────────────────────────────────────────────

@dataclass
class IndicatorConfig:
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    bb_period: int = 20
    bb_std: float = 2.0

    ma_short: int = 20
    ma_medium: int = 50
    ma_long: int = 200

    vwap_anchored: bool = True          # anchor VWAP to session open

INDICATOR_CFG = IndicatorConfig()

# ─────────────────────────────────────────────
# STRATEGY PARAMETERS
# ─────────────────────────────────────────────

@dataclass
class StrategyConfig:
    # Opening Range Breakout
    orb_minutes: int = 15          # first N minutes define the range
    orb_volume_multiplier: float = 1.5   # volume spike threshold

    # VWAP bounce
    vwap_band_pct: float = 0.002   # ±0.2% around VWAP counts as "bounce zone"

    # Momentum
    momentum_lookback: int = 5     # bars for momentum confirmation
    volume_spike_multiplier: float = 2.0

STRATEGY_CFG = StrategyConfig()

# ─────────────────────────────────────────────
# RISK PARAMETERS
# ─────────────────────────────────────────────

@dataclass
class RiskConfig:
    default_risk_reward: float = 2.0      # min R:R to consider a trade
    max_stop_loss_pct: float = 0.02       # 2% max stop from entry
    atr_stop_multiplier: float = 1.5      # ATR-based stop = 1.5 × ATR
    capital: float = float(_env("TRADING_CAPITAL", "100000"))   # from .env
    risk_per_trade_pct: float = float(_env("RISK_PER_TRADE", "0.01"))  # from .env

RISK_CFG = RiskConfig()

# ─────────────────────────────────────────────
# RANKING PARAMETERS
# ─────────────────────────────────────────────

@dataclass
class RankingConfig:
    min_avg_daily_volume: int = 500_000   # minimum avg DAILY volume (not per-bar)
    min_price: float = 50.0              # filter out penny stocks
    max_price: float = 50_000.0
    top_n: int = 10                       # how many stocks to return

RANKING_CFG = RankingConfig()

# ─────────────────────────────────────────────
# CACHE / MISC
# ─────────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_TTL_SECONDS = {
    "1m":  60,
    "5m":  300,
    "15m": 900,
    "1d":  3600,
}

# Market hours (IST = UTC+5:30)
MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MIN    = 15
MARKET_CLOSE_HOUR  = 15
MARKET_CLOSE_MIN   = 30
