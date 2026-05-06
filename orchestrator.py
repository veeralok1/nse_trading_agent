"""
orchestrator.py — Main Pipeline Coordinator
Wires all agents together and exposes a clean API for the Streamlit UI.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from config import NIFTY_50_SYMBOLS
from agents.data_agent import DataAgent, normalise_symbol
from agents.indicator_agent import IndicatorAgent
from agents.signal_agent import SignalAgent
from agents.risk_agent import RiskAgent, RiskProfile
from agents.ranking_agent import RankingAgent, StockRank
from agents.conversational_agent import ConversationalAgent

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """
    Singleton-like orchestrator that:
    - Creates and owns all agent instances (shared, for cache efficiency)
    - Provides high-level methods called by the Streamlit UI
    - Ensures clean error handling at the boundary
    """

    def __init__(self, use_cache: bool = True):
        logger.info("Initialising TradingOrchestrator …")
        self.da  = DataAgent(use_cache=use_cache)
        self.ia  = IndicatorAgent()
        self.sa  = SignalAgent()
        self.ra  = RiskAgent()
        self.rk  = RankingAgent(
            data_agent=self.da,
            indicator_agent=self.ia,
            signal_agent=self.sa,
        )
        self.ca  = ConversationalAgent(
            data_agent=self.da,
            indicator_agent=self.ia,
            signal_agent=self.sa,
            risk_agent=self.ra,
            ranking_agent=self.rk,
        )
        logger.info("All agents ready.")

    # ─────────────────────────────────────────────────────
    # PRIMARY API
    # ─────────────────────────────────────────────────────

    def analyze(
        self,
        symbol: str,
        interval: str = "5m",
        period:   str = "2d",
    ) -> Dict:
        """
        Full analysis pipeline for a single stock.
        Returns dict ready for Streamlit rendering.
        """
        symbol = normalise_symbol(symbol)
        df     = self.da.fetch(symbol, interval, period)

        if df is None or df.empty:
            return {"error": f"No data for {symbol}. Market may be closed or symbol invalid."}

        enriched = self.ia.compute_all(df)
        signals  = self.sa.generate(enriched, symbol)
        ind_sum  = self.ia.get_summary(enriched)
        trend    = self.ia.trend_direction(enriched)
        quote    = self.da.get_live_quote(symbol)

        risk_profile = None
        if signals:
            risk_profile = self.ra.evaluate(signals[0], enriched)

        action, conf = self.sa.aggregate_action(enriched, symbol)

        return {
            "symbol":       symbol,
            "df":           enriched,
            "quote":        quote,
            "indicators":   ind_sum,
            "trend":        trend,
            "signals":      signals,
            "risk_profile": risk_profile,
            "action":       action,
            "confidence":   conf,
            "interval":     interval,
            "period":       period,
        }

    def chat(self, user_input: str) -> Dict:
        """Route a natural-language query through the Conversational Agent."""
        return self.ca.query(user_input)

    def get_top_intraday(
        self,
        universe: str = "nifty50",
        top_n:    int = 10,
        interval: str = "5m",
    ) -> List[StockRank]:
        """Return ranked intraday candidates."""
        symbols = self.da.get_universe(universe)[:30]
        return self.rk.rank(symbols, interval=interval, top_n=top_n)

    def get_live_quote(self, symbol: str) -> Dict:
        return self.da.get_live_quote(symbol)

    def get_historical(self, symbol: str, days: int = 30) -> pd.DataFrame:
        df = self.da.fetch_historical(normalise_symbol(symbol), days)
        return self.ia.compute_all(df)

    def multi_timeframe_analysis(self, symbol: str) -> Dict[str, pd.DataFrame]:
        symbol = normalise_symbol(symbol)
        mtf    = self.da.fetch_multi_tf(symbol)
        return {iv: self.ia.compute_all(df) for iv, df in mtf.items()}

    def get_rank_dataframe(self, universe: str = "nifty50", top_n: int = 10) -> pd.DataFrame:
        ranked = self.get_top_intraday(universe, top_n)
        return self.rk.to_dataframe(ranked)

    def screen_signals(
        self,
        universe: str = "nifty50",
        action_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Return a list of stocks with active signals.
        action_filter: "BUY" | "SELL" | None (all)
        """
        symbols  = self.da.get_universe(universe)[:30]
        data_map = self.da.bulk_fetch(symbols, interval="5m", period="1d")
        results  = []

        for sym, df in data_map.items():
            if df is None or df.empty or len(df) < 20:
                continue
            enriched = self.ia.compute_all(df)
            signals  = self.sa.generate(enriched, sym)
            if not signals:
                continue
            best = signals[0]
            if action_filter and best.action != action_filter:
                continue
            results.append({
                "symbol":     sym,
                "action":     best.action,
                "strategy":   best.strategy,
                "confidence": best.confidence,
                "price":      float(df["Close"].iloc[-1]),
                "reasons":    best.reasons,
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results


# ── CLI entry point ────────────────────────────────────────

def main():
    """Interactive CLI for the trading system."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("  NSE Intraday Trading Agent  (type 'quit' to exit)")
    print("=" * 60)
    print("Examples:")
    print("  > Analyze RELIANCE")
    print("  > Top 5 intraday stocks today")
    print("  > Is TCS good for intraday?")
    print("  > Price of HDFC Bank")
    print()

    orch = TradingOrchestrator()

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not q:
            continue

        result = orch.chat(q)
        print("\nAgent:\n")
        print(result["text"])
        print()


if __name__ == "__main__":
    main()
