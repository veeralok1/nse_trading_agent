"""
signal_agent.py — Signal Agent
Generates BUY / SELL / HOLD signals from enriched indicator DataFrames.
Implements: ORB, VWAP Bounce, Momentum Breakout, RSI Reversal, MA Cross.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import INDICATOR_CFG, STRATEGY_CFG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────

@dataclass
class Signal:
    symbol:     str
    action:     str          # "BUY" | "SELL" | "HOLD"
    strategy:   str          # e.g. "ORB_BREAKOUT"
    confidence: float        # 0.0 – 1.0
    price:      float
    reasons:    List[str] = field(default_factory=list)
    metadata:   Dict       = field(default_factory=dict)

    def __str__(self):
        reasons_str = " | ".join(self.reasons)
        return (
            f"[{self.action}] {self.symbol} via {self.strategy} "
            f"@ ₹{self.price:.2f}  conf={self.confidence:.0%}  — {reasons_str}"
        )


# ─────────────────────────────────────────────────────────
# INDIVIDUAL STRATEGY FUNCTIONS
# ─────────────────────────────────────────────────────────

def _orb_signal(df: pd.DataFrame, symbol: str, orb_minutes: int = 15) -> Optional[Signal]:
    """
    Opening Range Breakout:
    First *orb_minutes* bars define the high/low range.
    A close above the range with volume spike = BUY.
    A close below the range = SELL.
    """
    if len(df) < orb_minutes + 2:
        return None

    # Identify the first trading session's first N bars
    first_date = df.index[0].date()
    session_df = df[df.index.date == first_date]

    if len(session_df) < orb_minutes:
        return None

    orb_bars = session_df.iloc[:orb_minutes]
    orb_high = orb_bars["High"].max()
    orb_low  = orb_bars["Low"].min()

    latest   = df.iloc[-1]
    close    = latest["Close"]
    volume_ratio = latest.get("Volume_Ratio", 1.0)

    vol_spike = volume_ratio >= STRATEGY_CFG.orb_volume_multiplier

    if close > orb_high and vol_spike:
        confidence = min(0.9, 0.6 + 0.15 * (volume_ratio - 1.0))
        return Signal(
            symbol=symbol,
            action="BUY",
            strategy="ORB_BREAKOUT",
            confidence=round(confidence, 2),
            price=close,
            reasons=[
                f"Close ₹{close:.2f} > ORB High ₹{orb_high:.2f}",
                f"Volume ratio {volume_ratio:.1f}x (spike confirmed)",
            ],
            metadata={"orb_high": orb_high, "orb_low": orb_low},
        )
    elif close < orb_low and vol_spike:
        confidence = min(0.85, 0.55 + 0.15 * (volume_ratio - 1.0))
        return Signal(
            symbol=symbol,
            action="SELL",
            strategy="ORB_BREAKDOWN",
            confidence=round(confidence, 2),
            price=close,
            reasons=[
                f"Close ₹{close:.2f} < ORB Low ₹{orb_low:.2f}",
                f"Volume ratio {volume_ratio:.1f}x (breakdown confirmed)",
            ],
            metadata={"orb_high": orb_high, "orb_low": orb_low},
        )
    return None


def _vwap_bounce_signal(df: pd.DataFrame, symbol: str) -> Optional[Signal]:
    """
    VWAP Bounce:
    Price dips to VWAP band and bounces with RSI recovering from oversold.
    """
    if "VWAP" not in df.columns or "RSI" not in df.columns:
        return None
    if len(df) < 5:
        return None

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    close   = latest["Close"]
    vwap    = latest["VWAP"]
    rsi     = latest["RSI"]
    prev_rsi = prev["RSI"]

    band = STRATEGY_CFG.vwap_band_pct * vwap

    price_near_vwap  = abs(close - vwap) <= band
    bounce_above     = close > vwap and prev["Close"] <= prev["VWAP"]
    rsi_rising       = pd.notna(rsi) and pd.notna(prev_rsi) and rsi > prev_rsi
    rsi_was_oversold = pd.notna(prev_rsi) and prev_rsi < INDICATOR_CFG.rsi_oversold + 10

    if (price_near_vwap or bounce_above) and rsi_rising and rsi_was_oversold:
        confidence = 0.65 + (0.1 if bounce_above else 0)
        return Signal(
            symbol=symbol,
            action="BUY",
            strategy="VWAP_BOUNCE",
            confidence=round(confidence, 2),
            price=close,
            reasons=[
                f"Price ₹{close:.2f} bouncing off VWAP ₹{vwap:.2f}",
                f"RSI {prev_rsi:.1f} → {rsi:.1f} (recovering)",
            ],
            metadata={"vwap": vwap, "rsi": rsi},
        )

    # VWAP rejection (price fails to hold above VWAP)
    if prev["Close"] > prev["VWAP"] and close < vwap and pd.notna(rsi) and rsi < 50:
        return Signal(
            symbol=symbol,
            action="SELL",
            strategy="VWAP_REJECTION",
            confidence=0.60,
            price=close,
            reasons=[
                f"Price ₹{close:.2f} failed VWAP ₹{vwap:.2f}",
                f"RSI {rsi:.1f} < 50",
            ],
            metadata={"vwap": vwap, "rsi": rsi},
        )
    return None


def _momentum_breakout_signal(df: pd.DataFrame, symbol: str) -> Optional[Signal]:
    """
    Momentum Breakout:
    Price makes new N-bar high with strong volume and positive MACD.
    """
    if len(df) < STRATEGY_CFG.momentum_lookback + 5:
        return None
    if "MACD" not in df.columns:
        return None

    lb     = STRATEGY_CFG.momentum_lookback
    latest = df.iloc[-1]
    window = df.iloc[-(lb + 1):-1]

    close    = latest["Close"]
    macd     = latest.get("MACD", 0)
    signal_v = latest.get("Signal", 0)
    vol_r    = latest.get("Volume_Ratio", 1.0)

    new_high   = close > window["High"].max()
    macd_bull  = pd.notna(macd) and pd.notna(signal_v) and macd > signal_v
    vol_spike  = vol_r >= STRATEGY_CFG.volume_spike_multiplier

    if new_high and macd_bull and vol_spike:
        confidence = 0.70 + min(0.15, 0.05 * (vol_r - 2.0))
        return Signal(
            symbol=symbol,
            action="BUY",
            strategy="MOMENTUM_BREAKOUT",
            confidence=round(confidence, 2),
            price=close,
            reasons=[
                f"New {lb}-bar high at ₹{close:.2f}",
                f"MACD {macd:.3f} > Signal {signal_v:.3f}",
                f"Volume {vol_r:.1f}x average",
            ],
            metadata={"new_high": close, "volume_ratio": vol_r},
        )

    new_low    = close < window["Low"].min()
    macd_bear  = pd.notna(macd) and pd.notna(signal_v) and macd < signal_v
    if new_low and macd_bear and vol_spike:
        confidence = 0.65 + min(0.15, 0.05 * (vol_r - 2.0))
        return Signal(
            symbol=symbol,
            action="SELL",
            strategy="MOMENTUM_BREAKDOWN",
            confidence=round(confidence, 2),
            price=close,
            reasons=[
                f"New {lb}-bar low at ₹{close:.2f}",
                f"MACD {macd:.3f} < Signal {signal_v:.3f}",
                f"Volume {vol_r:.1f}x average",
            ],
            metadata={"new_low": close, "volume_ratio": vol_r},
        )
    return None


def _rsi_reversal_signal(df: pd.DataFrame, symbol: str) -> Optional[Signal]:
    """RSI overbought/oversold reversal signal."""
    if "RSI" not in df.columns or len(df) < 3:
        return None

    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    rsi    = latest.get("RSI")
    prev_rsi = prev.get("RSI")

    if pd.isna(rsi) or pd.isna(prev_rsi):
        return None

    close = latest["Close"]

    # Oversold reversal: RSI was below 30, now crossing back up
    if prev_rsi < INDICATOR_CFG.rsi_oversold and rsi > INDICATOR_CFG.rsi_oversold:
        return Signal(
            symbol=symbol,
            action="BUY",
            strategy="RSI_OVERSOLD_REVERSAL",
            confidence=0.62,
            price=close,
            reasons=[
                f"RSI crossed above {INDICATOR_CFG.rsi_oversold} ({prev_rsi:.1f} → {rsi:.1f})",
                "Potential oversold bounce",
            ],
            metadata={"rsi": rsi},
        )

    # Overbought reversal: RSI was above 70, now crossing back down
    if prev_rsi > INDICATOR_CFG.rsi_overbought and rsi < INDICATOR_CFG.rsi_overbought:
        return Signal(
            symbol=symbol,
            action="SELL",
            strategy="RSI_OVERBOUGHT_REVERSAL",
            confidence=0.60,
            price=close,
            reasons=[
                f"RSI crossed below {INDICATOR_CFG.rsi_overbought} ({prev_rsi:.1f} → {rsi:.1f})",
                "Potential overbought pullback",
            ],
            metadata={"rsi": rsi},
        )
    return None


def _ma_crossover_signal(df: pd.DataFrame, symbol: str) -> Optional[Signal]:
    """Golden / Death cross on 20-MA vs 50-MA."""
    if "MA_20" not in df.columns or "MA_50" not in df.columns:
        return None
    if len(df) < 3:
        return None

    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    close  = latest["Close"]

    ma20, ma50       = latest.get("MA_20"), latest.get("MA_50")
    prev_ma20, prev_ma50 = prev.get("MA_20"), prev.get("MA_50")

    if any(pd.isna(v) for v in [ma20, ma50, prev_ma20, prev_ma50]):
        return None

    # Golden cross: MA20 crosses above MA50
    if prev_ma20 < prev_ma50 and ma20 > ma50:
        return Signal(
            symbol=symbol,
            action="BUY",
            strategy="GOLDEN_CROSS",
            confidence=0.72,
            price=close,
            reasons=[
                f"MA20 ₹{ma20:.2f} crossed above MA50 ₹{ma50:.2f}",
                "Bullish golden cross",
            ],
            metadata={"ma20": ma20, "ma50": ma50},
        )

    # Death cross: MA20 crosses below MA50
    if prev_ma20 > prev_ma50 and ma20 < ma50:
        return Signal(
            symbol=symbol,
            action="SELL",
            strategy="DEATH_CROSS",
            confidence=0.70,
            price=close,
            reasons=[
                f"MA20 ₹{ma20:.2f} crossed below MA50 ₹{ma50:.2f}",
                "Bearish death cross",
            ],
            metadata={"ma20": ma20, "ma50": ma50},
        )
    return None


def _bollinger_squeeze_signal(df: pd.DataFrame, symbol: str) -> Optional[Signal]:
    """Bollinger Band squeeze breakout."""
    if "BB_bandwidth" not in df.columns or len(df) < 20:
        return None

    latest = df.iloc[-1]
    close  = latest["Close"]
    bb_bw  = latest.get("BB_bandwidth")
    hist_bw = df["BB_bandwidth"].iloc[-20:]

    if pd.isna(bb_bw):
        return None

    squeeze = bb_bw < hist_bw.quantile(0.20)  # bandwidth in bottom 20%

    if squeeze:
        # Detect direction of breakout
        mom = latest.get("Momentum_5", 0) or 0
        if mom > 0.5:
            return Signal(
                symbol=symbol,
                action="BUY",
                strategy="BB_SQUEEZE_BREAKOUT",
                confidence=0.63,
                price=close,
                reasons=[
                    f"Bollinger Bands squeezing (BW={bb_bw:.4f})",
                    f"Upward momentum {mom:.2f}%",
                ],
                metadata={"bb_bandwidth": bb_bw},
            )
        elif mom < -0.5:
            return Signal(
                symbol=symbol,
                action="SELL",
                strategy="BB_SQUEEZE_BREAKDOWN",
                confidence=0.60,
                price=close,
                reasons=[
                    f"Bollinger Bands squeezing (BW={bb_bw:.4f})",
                    f"Downward momentum {mom:.2f}%",
                ],
                metadata={"bb_bandwidth": bb_bw},
            )
    return None


# ─────────────────────────────────────────────────────────
# SIGNAL AGENT
# ─────────────────────────────────────────────────────────

class SignalAgent:
    """
    Runs all strategy functions on an enriched DataFrame and
    returns the strongest (or all) signals.
    """

    STRATEGIES = [
        _orb_signal,
        _vwap_bounce_signal,
        _momentum_breakout_signal,
        _rsi_reversal_signal,
        _ma_crossover_signal,
        _bollinger_squeeze_signal,
    ]

    def generate(
        self,
        df: pd.DataFrame,
        symbol: str,
        return_all: bool = False,
    ) -> List[Signal]:
        """
        Run every strategy. Return list of Signal objects sorted by confidence.
        If return_all=False, only returns signals with confidence >= 0.60.
        """
        signals: List[Signal] = []
        for fn in self.STRATEGIES:
            try:
                sig = fn(df, symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                logger.debug("Strategy %s raised: %s", fn.__name__, exc)

        signals.sort(key=lambda s: s.confidence, reverse=True)

        if not return_all:
            signals = [s for s in signals if s.confidence >= 0.60]

        return signals

    def best_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Return highest-confidence signal or None."""
        sigs = self.generate(df, symbol)
        return sigs[0] if sigs else None

    def aggregate_action(self, df: pd.DataFrame, symbol: str) -> Tuple[str, float]:
        """
        Aggregate all strategy votes into a single action + confidence.
        Majority vote weighted by individual confidence.
        """
        sigs = self.generate(df, symbol, return_all=True)
        if not sigs:
            return "HOLD", 0.5

        buy_score  = sum(s.confidence for s in sigs if s.action == "BUY")
        sell_score = sum(s.confidence for s in sigs if s.action == "SELL")
        total      = buy_score + sell_score

        if total == 0:
            return "HOLD", 0.5

        if buy_score > sell_score:
            return "BUY", round(buy_score / total, 2)
        else:
            return "SELL", round(sell_score / total, 2)


# ── Quick smoke-test ──────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from agents.data_agent import DataAgent
    from agents.indicator_agent import IndicatorAgent

    da = DataAgent()
    ia = IndicatorAgent()
    sa = SignalAgent()

    df = da.fetch("RELIANCE", "5m", "2d")
    enriched = ia.compute_all(df)
    signals = sa.generate(enriched, "RELIANCE.NS")
    for s in signals:
        print(s)
    print("Aggregate:", sa.aggregate_action(enriched, "RELIANCE.NS"))
