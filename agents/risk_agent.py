"""
risk_agent.py — Risk Management Agent
Computes stop loss, target price, position size, and risk-reward ratio
for a given signal and enriched DataFrame.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RISK_CFG
from agents.signal_agent import Signal

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────

@dataclass
class RiskProfile:
    symbol:          str
    action:          str
    entry_price:     float
    stop_loss:       float
    target_1:        float   # 1:1 R:R
    target_2:        float   # 1.5:1 R:R
    target_3:        float   # 2:1 R:R (primary target)
    risk_reward:     float
    risk_pct:        float   # stop distance as % of entry
    position_size:   int     # shares to buy given capital & risk
    risk_amount_inr: float   # ₹ at risk
    atr:             float
    notes:           str = ""

    def summary(self) -> str:
        direction = "↑" if self.action == "BUY" else "↓"
        return (
            f"{direction} {self.symbol} | Entry: ₹{self.entry_price:.2f} | "
            f"SL: ₹{self.stop_loss:.2f} ({self.risk_pct:.1f}%) | "
            f"T1: ₹{self.target_1:.2f}  T2: ₹{self.target_2:.2f}  T3: ₹{self.target_3:.2f} | "
            f"R:R = 1:{self.risk_reward:.1f} | "
            f"Qty: {self.position_size} shares | "
            f"Risk: ₹{self.risk_amount_inr:,.0f}"
        )


# ─────────────────────────────────────────────────────────
# RISK AGENT
# ─────────────────────────────────────────────────────────

class RiskAgent:
    """
    Computes stop-loss, targets, and position sizing for a given signal.

    Stop-loss methods (in priority order):
    1. ATR-based:   entry ± (ATR_multiplier × ATR)
    2. Swing-based: nearest recent swing high/low
    3. Fixed %:     entry ± max_stop_loss_pct
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or RISK_CFG

    # ── Main entry point ──────────────────────────────────

    def evaluate(
        self,
        signal: Signal,
        df: pd.DataFrame,
        capital: Optional[float] = None,
    ) -> Optional[RiskProfile]:
        """
        Given a Signal and enriched DataFrame, return a full RiskProfile.
        """
        if signal.action not in ("BUY", "SELL"):
            return None

        capital = capital or self.cfg.capital
        entry   = signal.price
        atr     = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and pd.notna(df["ATR"].iloc[-1]) else entry * 0.01

        # ── Compute Stop Loss ────────────────────────────
        sl = self._compute_stop_loss(signal, df, entry, atr)

        # Clamp stop to max allowed percentage
        max_sl_dist = entry * self.cfg.max_stop_loss_pct
        if signal.action == "BUY":
            sl = max(sl, entry - max_sl_dist)
        else:
            sl = min(sl, entry + max_sl_dist)

        risk_per_share = abs(entry - sl)
        if risk_per_share < 0.01:
            logger.warning("Risk per share too small for %s, skipping", signal.symbol)
            return None

        risk_pct = risk_per_share / entry * 100

        # ── Compute Targets ──────────────────────────────
        t1 = self._target(entry, risk_per_share, signal.action, multiplier=1.0)
        t2 = self._target(entry, risk_per_share, signal.action, multiplier=1.5)
        t3 = self._target(entry, risk_per_share, signal.action, multiplier=2.0)
        rr = 2.0   # primary target is always 1:2

        # ── Position Sizing ───────────────────────────────
        risk_amount = capital * self.cfg.risk_per_trade_pct
        position_size = max(1, int(risk_amount / risk_per_share))

        notes = self._generate_notes(signal, df, rr)

        return RiskProfile(
            symbol        = signal.symbol,
            action        = signal.action,
            entry_price   = round(entry, 2),
            stop_loss     = round(sl, 2),
            target_1      = round(t1, 2),
            target_2      = round(t2, 2),
            target_3      = round(t3, 2),
            risk_reward   = rr,
            risk_pct      = round(risk_pct, 2),
            position_size = position_size,
            risk_amount_inr = round(risk_amount, 2),
            atr           = round(atr, 2),
            notes         = notes,
        )

    # ── Stop Loss Computation ─────────────────────────────

    def _compute_stop_loss(
        self,
        signal: Signal,
        df: pd.DataFrame,
        entry: float,
        atr: float,
    ) -> float:
        """Try ATR-based, then swing-based, fall back to fixed %."""

        # 1. ATR-based stop
        atr_stop_dist = self.cfg.atr_stop_multiplier * atr
        if signal.action == "BUY":
            sl = entry - atr_stop_dist
        else:
            sl = entry + atr_stop_dist

        # 2. Override with swing-based if more meaningful
        swing_sl = self._swing_stop(signal, df, lookback=10)
        if swing_sl is not None:
            # Take the tighter stop (better risk management)
            if signal.action == "BUY":
                sl = max(sl, swing_sl)   # highest of the two lows = tighter
            else:
                sl = min(sl, swing_sl)   # lowest of the two highs = tighter

        return sl

    def _swing_stop(
        self, signal: Signal, df: pd.DataFrame, lookback: int = 10
    ) -> Optional[float]:
        """Nearest swing low (for BUY) or swing high (for SELL)."""
        if len(df) < lookback:
            return None
        window = df.iloc[-lookback:]
        if signal.action == "BUY":
            return float(window["Low"].min())
        else:
            return float(window["High"].max())

    # ── Target Calculation ────────────────────────────────

    @staticmethod
    def _target(entry: float, risk: float, action: str, multiplier: float) -> float:
        if action == "BUY":
            return entry + risk * multiplier
        else:
            return entry - risk * multiplier

    # ── Notes / Warnings ─────────────────────────────────

    def _generate_notes(self, signal: Signal, df: pd.DataFrame, rr: float) -> str:
        notes = []
        if rr < self.cfg.default_risk_reward:
            notes.append(f"⚠️ R:R {rr:.1f} below minimum {self.cfg.default_risk_reward:.1f}")
        if "ADX" in df.columns:
            adx = df["ADX"].iloc[-1]
            if pd.notna(adx):
                if adx < 20:
                    notes.append("⚠️ ADX < 20 — weak trend, trade cautiously")
                elif adx > 40:
                    notes.append("✅ ADX > 40 — strong trend")
        if "RSI" in df.columns:
            rsi = df["RSI"].iloc[-1]
            if pd.notna(rsi):
                if signal.action == "BUY" and rsi > 70:
                    notes.append("⚠️ RSI overbought at entry — consider waiting for pullback")
                elif signal.action == "SELL" and rsi < 30:
                    notes.append("⚠️ RSI oversold at entry — consider waiting for bounce")
        return "  ".join(notes) if notes else "✅ Trade parameters look acceptable"

    # ── Portfolio-level helpers ───────────────────────────

    def max_positions(self, capital: float) -> int:
        """How many concurrent positions at current risk level."""
        return max(1, int(1.0 / self.cfg.risk_per_trade_pct))

    def daily_loss_limit(self, capital: float) -> float:
        """Conservative 2% daily loss limit."""
        return capital * 0.02

    def to_dict(self, profile: RiskProfile) -> Dict:
        return {
            "symbol":          profile.symbol,
            "action":          profile.action,
            "entry_price":     profile.entry_price,
            "stop_loss":       profile.stop_loss,
            "target_1":        profile.target_1,
            "target_2":        profile.target_2,
            "target_3":        profile.target_3,
            "risk_reward":     profile.risk_reward,
            "risk_pct":        profile.risk_pct,
            "position_size":   profile.position_size,
            "risk_amount_inr": profile.risk_amount_inr,
            "atr":             profile.atr,
            "notes":           profile.notes,
        }


# ── Quick smoke-test ──────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from agents.data_agent import DataAgent
    from agents.indicator_agent import IndicatorAgent
    from agents.signal_agent import SignalAgent

    da = DataAgent()
    ia = IndicatorAgent()
    sa = SignalAgent()
    ra = RiskAgent()

    df = da.fetch("TCS", "5m", "2d")
    enriched = ia.compute_all(df)
    signal = sa.best_signal(enriched, "TCS.NS")

    if signal:
        profile = ra.evaluate(signal, enriched)
        if profile:
            print(profile.summary())
    else:
        print("No signal for TCS")
