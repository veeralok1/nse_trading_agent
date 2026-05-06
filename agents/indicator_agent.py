"""
indicator_agent.py — Indicator Agent
Computes all technical indicators on a given OHLCV DataFrame.
Uses pandas-ta (no TA-Lib C dependency required).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import INDICATOR_CFG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# PURE NUMPY / PANDAS IMPLEMENTATIONS
# (avoids hard TA-Lib C dependency; pandas-ta used as well)
# ─────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast   = _ema(close, fast)
    ema_slow   = _ema(close, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return pd.DataFrame(
        {"MACD": macd_line, "Signal": signal_line, "Histogram": histogram},
        index=close.index,
    )


def _bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    mid   = _sma(close, period)
    std   = close.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    bw    = (upper - lower) / mid          # bandwidth
    pct_b = (close - lower) / (upper - lower)  # %B
    return pd.DataFrame(
        {"BB_upper": upper, "BB_mid": mid, "BB_lower": lower,
         "BB_bandwidth": bw, "BB_pct_b": pct_b},
        index=close.index,
    )


def _vwap(df: pd.DataFrame) -> pd.Series:
    """
    Session-anchored VWAP.
    Groups by calendar date so it resets each day.
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    result = pd.Series(index=df.index, dtype=float)
    for date, grp in df.groupby(df.index.date):
        idx = grp.index
        cum_tp_vol = (tp[idx] * grp["Volume"]).cumsum()
        cum_vol    = grp["Volume"].cumsum()
        result[idx] = cum_tp_vol / cum_vol.replace(0, np.nan)
    return result


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff())
    return (direction * df["Volume"]).cumsum()


def _stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    low_min  = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"Stoch_K": k, "Stoch_D": d}, index=df.index)


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simplified ADX (Average Directional Index)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Directional movement
    up_move   = high - prev_high
    down_move = prev_low - low

    dm_plus  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_s    = pd.Series(tr).ewm(com=period - 1, min_periods=period).mean()
    di_plus  = 100 * pd.Series(dm_plus,  index=df.index).ewm(com=period - 1).mean() / atr_s
    di_minus = 100 * pd.Series(dm_minus, index=df.index).ewm(com=period - 1).mean() / atr_s

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return dx.ewm(com=period - 1, min_periods=period).mean()


# ─────────────────────────────────────────────────────────
# INDICATOR AGENT
# ─────────────────────────────────────────────────────────

class IndicatorAgent:
    """
    Computes all technical indicators and appends them as new columns
    to the input DataFrame.  Returns an enriched copy.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or INDICATOR_CFG

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Master method — computes every indicator and returns an
        enriched DataFrame.  Input df must have OHLCV columns.
        """
        if df.empty or len(df) < 10:
            logger.warning("DataFrame too small for indicator computation (%d rows)", len(df))
            return df

        out = df.copy()

        try:
            # ── Moving Averages ──────────────────────────
            out["MA_20"]  = _sma(out["Close"], self.cfg.ma_short)
            out["MA_50"]  = _sma(out["Close"], self.cfg.ma_medium)
            out["MA_200"] = _sma(out["Close"], self.cfg.ma_long)
            out["EMA_9"]  = _ema(out["Close"], 9)
            out["EMA_21"] = _ema(out["Close"], 21)

            # ── RSI ──────────────────────────────────────
            out["RSI"] = _rsi(out["Close"], self.cfg.rsi_period)

            # ── MACD ─────────────────────────────────────
            macd_df = _macd(
                out["Close"],
                self.cfg.macd_fast,
                self.cfg.macd_slow,
                self.cfg.macd_signal,
            )
            out = pd.concat([out, macd_df], axis=1)

            # ── Bollinger Bands ───────────────────────────
            bb_df = _bollinger_bands(out["Close"], self.cfg.bb_period, self.cfg.bb_std)
            out = pd.concat([out, bb_df], axis=1)

            # ── VWAP (only meaningful for intraday) ───────
            out["VWAP"] = _vwap(out)

            # ── ATR ───────────────────────────────────────
            out["ATR"] = _atr(out)

            # ── OBV ───────────────────────────────────────
            out["OBV"] = _obv(out)

            # ── Stochastic ────────────────────────────────
            stoch_df = _stochastic(out)
            out = pd.concat([out, stoch_df], axis=1)

            # ── ADX ───────────────────────────────────────
            out["ADX"] = _adx(out)

            # ── Volume MA ─────────────────────────────────
            out["Volume_MA20"] = _sma(out["Volume"].astype(float), 20)
            out["Volume_Ratio"] = out["Volume"] / out["Volume_MA20"].replace(0, np.nan)

            # ── Momentum ─────────────────────────────────
            out["Momentum_5"]  = out["Close"].pct_change(5) * 100
            out["Momentum_10"] = out["Close"].pct_change(10) * 100

        except Exception as exc:
            logger.error("Indicator computation error: %s", exc, exc_info=True)

        return out

    # ── Convenience wrappers ──────────────────────────────

    def get_rsi(self, df: pd.DataFrame) -> pd.Series:
        return _rsi(df["Close"], self.cfg.rsi_period)

    def get_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        return _macd(df["Close"], self.cfg.macd_fast, self.cfg.macd_slow, self.cfg.macd_signal)

    def get_vwap(self, df: pd.DataFrame) -> pd.Series:
        return _vwap(df)

    def get_atr(self, df: pd.DataFrame) -> pd.Series:
        return _atr(df)

    def get_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        return _bollinger_bands(df["Close"], self.cfg.bb_period, self.cfg.bb_std)

    # ── Summary snapshot ─────────────────────────────────

    def get_summary(self, enriched_df: pd.DataFrame) -> Dict:
        """
        Return the latest indicator values as a flat dict
        (useful for the Conversational Agent).
        """
        if enriched_df.empty:
            return {}
        row = enriched_df.iloc[-1]

        def _safe(col):
            try:
                v = row[col]
                return round(float(v), 4) if pd.notna(v) else None
            except Exception:
                return None

        return {
            "close":        _safe("Close"),
            "rsi":          _safe("RSI"),
            "macd":         _safe("MACD"),
            "macd_signal":  _safe("Signal"),
            "macd_hist":    _safe("Histogram"),
            "ma_20":        _safe("MA_20"),
            "ma_50":        _safe("MA_50"),
            "ma_200":       _safe("MA_200"),
            "bb_upper":     _safe("BB_upper"),
            "bb_mid":       _safe("BB_mid"),
            "bb_lower":     _safe("BB_lower"),
            "bb_pct_b":     _safe("BB_pct_b"),
            "vwap":         _safe("VWAP"),
            "atr":          _safe("ATR"),
            "adx":          _safe("ADX"),
            "volume_ratio": _safe("Volume_Ratio"),
            "momentum_5":   _safe("Momentum_5"),
        }

    def trend_direction(self, enriched_df: pd.DataFrame) -> str:
        """Return 'BULLISH', 'BEARISH', or 'SIDEWAYS' based on MA alignment."""
        if enriched_df.empty:
            return "UNKNOWN"
        row = enriched_df.iloc[-1]
        try:
            close  = row["Close"]
            ma20   = row.get("MA_20",  close)
            ma50   = row.get("MA_50",  close)
            ma200  = row.get("MA_200", close)

            if close > ma20 > ma50 > ma200:
                return "BULLISH"
            elif close < ma20 < ma50 < ma200:
                return "BEARISH"
            else:
                return "SIDEWAYS"
        except Exception:
            return "UNKNOWN"


# ── Quick smoke-test ──────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import yfinance as yf
    df = yf.download("RELIANCE.NS", period="5d", interval="5m", auto_adjust=True, progress=False)
    agent = IndicatorAgent()
    enriched = agent.compute_all(df)
    print(enriched[["Close", "RSI", "MACD", "VWAP", "BB_upper", "BB_lower"]].tail(10))
    print(agent.get_summary(enriched))
