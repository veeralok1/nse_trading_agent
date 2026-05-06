"""
ranking_agent.py — Ranking Agent
Scores and ranks NSE stocks for intraday suitability using:
  • Volatility score (ATR / price %)
  • Liquidity score (volume vs average)
  • Signal strength (from SignalAgent)
  • Momentum score
  • Trend alignment
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
from config import RANKING_CFG, NIFTY_50_SYMBOLS
from agents.data_agent import DataAgent
from agents.indicator_agent import IndicatorAgent
from agents.signal_agent import SignalAgent, Signal

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────

@dataclass
class StockRank:
    symbol:            str
    rank:              int
    total_score:       float     # 0–100
    volatility_score:  float
    liquidity_score:   float
    momentum_score:    float
    signal_score:      float
    trend:             str       # "BULLISH" | "BEARISH" | "SIDEWAYS"
    action:            str       # best signal action
    confidence:        float
    price:             float
    atr_pct:           float
    volume_ratio:      float
    signals:           List[Signal] = field(default_factory=list)
    indicators:        Dict = field(default_factory=dict)

    def __str__(self):
        return (
            f"#{self.rank:2d}  {self.symbol:<20s}  Score={self.total_score:.1f}  "
            f"{self.action:<4s} ({self.confidence:.0%})  "
            f"Trend={self.trend}  ATR%={self.atr_pct:.2f}  "
            f"Vol={self.volume_ratio:.1f}x  ₹{self.price:.2f}"
        )


# ─────────────────────────────────────────────────────────
# RANKING AGENT
# ─────────────────────────────────────────────────────────

class RankingAgent:
    """
    Screens a universe of NSE stocks, computes a composite intraday
    suitability score, and returns the top-N ranked candidates.
    """

    def __init__(
        self,
        data_agent:      Optional[DataAgent]      = None,
        indicator_agent: Optional[IndicatorAgent] = None,
        signal_agent:    Optional[SignalAgent]    = None,
        cfg=None,
    ):
        self.da  = data_agent      or DataAgent()
        self.ia  = indicator_agent or IndicatorAgent()
        self.sa  = signal_agent    or SignalAgent()
        self.cfg = cfg or RANKING_CFG

    # ── Main screening routine ────────────────────────────

    # Maps interval → approximate number of bars in a trading day (NSE: 6.25 hrs)
    BARS_PER_DAY = {
        "1m": 375, "5m": 75, "15m": 25, "30m": 13, "1h": 7, "1d": 1,
    }

    def rank(
        self,
        symbols:  Optional[List[str]] = None,
        interval: str = "5m",
        period:   str = "5d",           # fetch 5 days so early-morning runs still get enough bars
        top_n:    Optional[int] = None,
    ) -> List[StockRank]:
        """
        Rank *symbols* for intraday trading.
        Returns a sorted list of StockRank objects (best first).
        """
        symbols = symbols or NIFTY_50_SYMBOLS[:25]
        top_n   = top_n or self.cfg.top_n

        logger.info("Ranking %d symbols [%s / %s]", len(symbols), interval, period)

        # Bulk download for speed
        data_map = self.da.bulk_fetch(symbols, interval=interval, period=period)

        bars_per_day = self.BARS_PER_DAY.get(interval, 75)
        passed = failed_data = failed_price = failed_vol = 0

        ranks: List[StockRank] = []
        for sym, df in data_map.items():
            try:
                if df is None or df.empty:
                    failed_data += 1
                    continue
                rank_entry = self._score_stock(sym, df, bars_per_day)
                if rank_entry is not None:
                    ranks.append(rank_entry)
                    passed += 1
            except Exception as exc:
                logger.warning("Error scoring %s: %s", sym, exc, exc_info=True)

        logger.info(
            "Screener result: %d passed | %d no-data | total=%d",
            passed, failed_data, len(data_map),
        )

        # Sort by total_score descending, assign rank numbers
        ranks.sort(key=lambda r: r.total_score, reverse=True)
        for i, r in enumerate(ranks[:top_n], start=1):
            r.rank = i

        logger.info("Ranking complete. Top stock: %s", ranks[0].symbol if ranks else "—")
        return ranks[:top_n]

    def _score_stock(
        self, symbol: str, df: pd.DataFrame, bars_per_day: int = 75
    ) -> Optional[StockRank]:
        # Need at least 20 bars for indicator computation
        if df is None or df.empty or len(df) < 20:
            logger.debug("Skipping %s: only %d bars", symbol, len(df) if df is not None else 0)
            return None

        # Price filter
        price = float(df["Close"].iloc[-1])
        if price < self.cfg.min_price or price > self.cfg.max_price:
            logger.debug("Skipping %s: price ₹%.2f out of range", symbol, price)
            return None

        # Volume filter — convert per-bar average to implied daily volume
        # e.g. for 5m bars: avg_bar_vol × 75 bars/day ≈ daily volume
        avg_bar_vol    = float(df["Volume"].mean())
        avg_daily_vol  = avg_bar_vol * bars_per_day
        if avg_daily_vol < self.cfg.min_avg_daily_volume:
            logger.debug(
                "Skipping %s: est. daily vol %,.0f < min %,.0f",
                symbol, avg_daily_vol, self.cfg.min_avg_daily_volume,
            )
            return None

        # Enrich
        enriched = self.ia.compute_all(df)

        # ── Volatility Score (ATR % of price) ────────────
        atr = float(enriched["ATR"].iloc[-1]) if "ATR" in enriched.columns else 0.0
        atr_pct = (atr / price * 100) if price > 0 else 0.0
        # Ideal range 0.5%–3% for intraday
        if atr_pct < 0.3:
            vol_score = 20.0
        elif atr_pct > 5.0:
            vol_score = 40.0
        else:
            vol_score = min(100.0, 20 + (atr_pct / 3.0) * 80)

        # ── Liquidity Score (volume ratio) ───────────────
        vol_ratio = float(enriched["Volume_Ratio"].iloc[-1]) if "Volume_Ratio" in enriched.columns else 1.0
        liq_score = min(100.0, 50 + (vol_ratio - 1.0) * 25)

        # ── Momentum Score ───────────────────────────────
        mom5 = float(enriched["Momentum_5"].iloc[-1]) if "Momentum_5" in enriched.columns else 0.0
        mom_score = min(100.0, 50 + abs(mom5) * 10)

        # ── Signal Score ─────────────────────────────────
        signals = self.sa.generate(enriched, symbol)
        if signals:
            best_sig   = signals[0]
            sig_score  = best_sig.confidence * 100
            action     = best_sig.action
            confidence = best_sig.confidence
        else:
            sig_score  = 0.0
            action     = "HOLD"
            confidence = 0.0

        # ── Trend ────────────────────────────────────────
        trend = self.ia.trend_direction(enriched)
        trend_bonus = 10.0 if trend in ("BULLISH", "BEARISH") else 0.0

        # ── Composite Score (weighted) ───────────────────
        total_score = (
            vol_score  * 0.25 +
            liq_score  * 0.20 +
            mom_score  * 0.20 +
            sig_score  * 0.25 +
            trend_bonus * 0.10
        ) + trend_bonus

        total_score = min(100.0, total_score)

        return StockRank(
            symbol           = symbol,
            rank             = 0,         # assigned later
            total_score      = round(total_score, 2),
            volatility_score = round(vol_score, 2),
            liquidity_score  = round(liq_score, 2),
            momentum_score   = round(mom_score, 2),
            signal_score     = round(sig_score, 2),
            trend            = trend,
            action           = action,
            confidence       = confidence,
            price            = round(price, 2),
            atr_pct          = round(atr_pct, 2),
            volume_ratio     = round(vol_ratio, 2),
            signals          = signals,
            indicators       = self.ia.get_summary(enriched),
        )

    # ── Swing / Positional ranking ────────────────────────

    def rank_swing(
        self,
        symbols:  Optional[List[str]] = None,
        period:   str = "90d",
        top_n:    Optional[int] = None,
    ) -> List["StockRank"]:
        """
        Rank stocks for swing / positional trading using DAILY bars.
        Longer lookback gives reliable MA-200, trend, and momentum signals.
        Different scoring weights vs intraday:
          - Trend alignment is weighted heavily (40%)
          - Momentum over daily bars (25%)
          - Signal strength (25%)
          - Liquidity / volume (10%)
        """
        symbols  = symbols or NIFTY_50_SYMBOLS[:50]
        top_n    = top_n or self.cfg.top_n
        interval = "1d"

        logger.info("Swing ranking %d symbols [%s / %s]", len(symbols), interval, period)

        data_map     = self.da.bulk_fetch(symbols, interval=interval, period=period)
        bars_per_day = 1   # daily bars

        ranks: List[StockRank] = []
        for sym, df in data_map.items():
            try:
                if df is None or df.empty or len(df) < 20:
                    continue

                price = float(df["Close"].iloc[-1])
                if price < self.cfg.min_price or price > self.cfg.max_price:
                    continue

                # For daily bars, volume filter is the raw daily volume
                avg_vol = float(df["Volume"].mean())
                # Daily minimum: 200k shares (relaxed for positional)
                if avg_vol < 200_000:
                    logger.debug("Swing skip %s: avg daily vol %.0f", sym, avg_vol)
                    continue

                enriched = self.ia.compute_all(df)
                ind      = self.ia.get_summary(enriched)
                trend    = self.ia.trend_direction(enriched)

                # ── Trend Score ──────────────────────────────
                # Strong uptrend: price > MA50 > MA200
                ma50  = ind.get("ma_50")  or price
                ma200 = ind.get("ma_200") or price
                if price > ma50 > ma200:
                    trend_score = 100.0
                elif price > ma50:
                    trend_score = 70.0
                elif price > ma200:
                    trend_score = 50.0
                elif price < ma50 < ma200:
                    trend_score = 10.0   # strong downtrend
                else:
                    trend_score = 30.0

                # ── Momentum Score (daily) ───────────────────
                mom5 = ind.get("momentum_5") or 0.0
                # Use RSI as momentum proxy for daily bars
                rsi  = ind.get("rsi") or 50.0
                # Ideal RSI for buying: 45–60 (not overbought, not oversold)
                if 45 <= rsi <= 60:
                    rsi_score = 100.0
                elif 30 <= rsi < 45 or 60 < rsi <= 70:
                    rsi_score = 70.0
                elif rsi < 30:
                    rsi_score = 40.0   # oversold — risky but potential
                else:
                    rsi_score = 20.0   # overbought
                mom_score = (rsi_score * 0.6 + min(100.0, 50 + abs(mom5) * 5) * 0.4)

                # ── Signal Score ─────────────────────────────
                signals = self.sa.generate(enriched, sym)
                if signals:
                    best_sig   = signals[0]
                    sig_score  = best_sig.confidence * 100
                    action     = best_sig.action
                    confidence = best_sig.confidence
                else:
                    sig_score  = 0.0
                    action     = "HOLD"
                    confidence = 0.0

                # ── Liquidity Score ───────────────────────────
                vol_ratio = ind.get("volume_ratio") or 1.0
                liq_score = min(100.0, 50 + (vol_ratio - 1.0) * 20)

                # ── ATR for display only ──────────────────────
                atr = 0.0
                if "ATR" in enriched.columns:
                    atr = float(enriched["ATR"].iloc[-1])
                atr_pct = (atr / price * 100) if price > 0 else 0.0

                # ── Composite (swing weights) ─────────────────
                total_score = (
                    trend_score * 0.40 +
                    mom_score   * 0.25 +
                    sig_score   * 0.25 +
                    liq_score   * 0.10
                )
                total_score = min(100.0, total_score)

                ranks.append(StockRank(
                    symbol           = sym,
                    rank             = 0,
                    total_score      = round(total_score, 2),
                    volatility_score = round(trend_score, 2),   # repurposed for display
                    liquidity_score  = round(liq_score, 2),
                    momentum_score   = round(mom_score, 2),
                    signal_score     = round(sig_score, 2),
                    trend            = trend,
                    action           = action,
                    confidence       = confidence,
                    price            = round(price, 2),
                    atr_pct          = round(atr_pct, 2),
                    volume_ratio     = round(float(vol_ratio), 2),
                    signals          = signals,
                    indicators       = ind,
                ))

            except Exception as exc:
                logger.warning("Swing error scoring %s: %s", sym, exc)

        ranks.sort(key=lambda r: r.total_score, reverse=True)
        for i, r in enumerate(ranks[:top_n], start=1):
            r.rank = i

        logger.info("Swing ranking done. Top: %s", ranks[0].symbol if ranks else "—")
        return ranks[:top_n]

    # ── Convenience wrappers ──────────────────────────────

    def top_buy_candidates(
        self, symbols: Optional[List[str]] = None, top_n: int = 5
    ) -> List[StockRank]:
        """Return top BUY-rated stocks."""
        ranked = self.rank(symbols, top_n=50)
        buys   = [r for r in ranked if r.action == "BUY"]
        return buys[:top_n]

    def top_sell_candidates(
        self, symbols: Optional[List[str]] = None, top_n: int = 5
    ) -> List[StockRank]:
        """Return top SELL-rated stocks (short-selling candidates)."""
        ranked = self.rank(symbols, top_n=50)
        sells  = [r for r in ranked if r.action == "SELL"]
        return sells[:top_n]

    def to_dataframe(self, ranks: List[StockRank]) -> pd.DataFrame:
        """Convert rank list to a display-ready DataFrame."""
        rows = []
        for r in ranks:
            rows.append({
                "Rank":       r.rank,
                "Symbol":     r.symbol.replace(".NS", ""),
                "Price (₹)":  r.price,
                "Signal":     r.action,
                "Conf.":      f"{r.confidence:.0%}",
                "Score":      r.total_score,
                "Trend":      r.trend,
                "ATR %":      r.atr_pct,
                "Vol Ratio":  r.volume_ratio,
                "RSI":        round(r.indicators.get("rsi", 0) or 0, 1),
                "Strategy":   r.signals[0].strategy if r.signals else "—",
            })
        return pd.DataFrame(rows)


# ── Quick smoke-test ──────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from config import NIFTY_50_SYMBOLS

    agent = RankingAgent()
    top   = agent.rank(NIFTY_50_SYMBOLS[:10], top_n=5)
    for r in top:
        print(r)
    print(agent.to_dataframe(top).to_string(index=False))
