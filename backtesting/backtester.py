"""
backtester.py — Event-driven backtesting engine
Supports: RSI Reversal, MACD Crossover, ORB Breakout, MA Crossover, BB Squeeze
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agents.data_agent import DataAgent, normalise_symbol
from agents.indicator_agent import IndicatorAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# TRADE RECORD
# ─────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    entry_price: float
    exit_price:  float
    action:      str          # "BUY" | "SELL"
    strategy:    str
    pnl_pct:     float
    pnl_abs:     float
    result:      str          # "WIN" | "LOSS"


# ─────────────────────────────────────────────────────────
# STRATEGY SIGNAL GENERATORS (return pd.Series of 1/0/-1)
# ─────────────────────────────────────────────────────────

def _rsi_signals(df: pd.DataFrame) -> pd.Series:
    """Buy when RSI crosses above 30, Sell when crosses below 70."""
    rsi   = df["RSI"]
    entry = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if rsi.iloc[i - 1] < 30 <= rsi.iloc[i]:
            entry.iloc[i] = 1    # BUY
        elif rsi.iloc[i - 1] > 70 >= rsi.iloc[i]:
            entry.iloc[i] = -1   # SELL / short
    return entry


def _macd_signals(df: pd.DataFrame) -> pd.Series:
    """MACD line crosses Signal line."""
    macd  = df.get("MACD", pd.Series(dtype=float))
    sig   = df.get("Signal", pd.Series(dtype=float))
    entry = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if pd.isna(macd.iloc[i]) or pd.isna(sig.iloc[i]):
            continue
        if macd.iloc[i - 1] < sig.iloc[i - 1] and macd.iloc[i] >= sig.iloc[i]:
            entry.iloc[i] = 1
        elif macd.iloc[i - 1] > sig.iloc[i - 1] and macd.iloc[i] <= sig.iloc[i]:
            entry.iloc[i] = -1
    return entry


def _orb_signals(df: pd.DataFrame, orb_bars: int = 15) -> pd.Series:
    """ORB: breakout above/below first-N-bar range."""
    entry = pd.Series(0, index=df.index)
    if len(df) < orb_bars + 5:
        return entry

    # Group by day
    for date, grp in df.groupby(df.index.date):
        if len(grp) < orb_bars + 2:
            continue
        orb = grp.iloc[:orb_bars]
        orb_high = orb["High"].max()
        orb_low  = orb["Low"].min()
        post_orb  = grp.iloc[orb_bars:]
        for idx, row in post_orb.iterrows():
            if row["Close"] > orb_high:
                entry.at[idx] = 1
            elif row["Close"] < orb_low:
                entry.at[idx] = -1
    return entry


def _ma_cross_signals(df: pd.DataFrame) -> pd.Series:
    """MA20 / MA50 crossover."""
    ma20  = df.get("MA_20", pd.Series(dtype=float))
    ma50  = df.get("MA_50", pd.Series(dtype=float))
    entry = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if pd.isna(ma20.iloc[i]) or pd.isna(ma50.iloc[i]):
            continue
        if ma20.iloc[i - 1] < ma50.iloc[i - 1] and ma20.iloc[i] >= ma50.iloc[i]:
            entry.iloc[i] = 1
        elif ma20.iloc[i - 1] > ma50.iloc[i - 1] and ma20.iloc[i] <= ma50.iloc[i]:
            entry.iloc[i] = -1
    return entry


def _bb_squeeze_signals(df: pd.DataFrame) -> pd.Series:
    """BB Squeeze + momentum."""
    bw    = df.get("BB_bandwidth", pd.Series(dtype=float))
    mom   = df.get("Momentum_5", pd.Series(dtype=float))
    entry = pd.Series(0, index=df.index)
    if bw.empty:
        return entry
    squeeze_thresh = bw.rolling(20).quantile(0.20)
    for i in range(1, len(df)):
        if pd.isna(bw.iloc[i]) or pd.isna(squeeze_thresh.iloc[i]):
            continue
        if bw.iloc[i] < squeeze_thresh.iloc[i]:
            if pd.notna(mom.iloc[i]) and mom.iloc[i] > 0.5:
                entry.iloc[i] = 1
            elif pd.notna(mom.iloc[i]) and mom.iloc[i] < -0.5:
                entry.iloc[i] = -1
    return entry


STRATEGY_MAP = {
    "rsi":       _rsi_signals,
    "macd":      _macd_signals,
    "orb":       _orb_signals,
    "ma_cross":  _ma_cross_signals,
    "bb_squeeze": _bb_squeeze_signals,
}


# ─────────────────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────────────────

class Backtester:

    def __init__(
        self,
        data_agent:      Optional[DataAgent]      = None,
        indicator_agent: Optional[IndicatorAgent] = None,
        stop_loss_pct:   float = 0.015,    # 1.5% stop
        take_profit_pct: float = 0.030,    # 3% target  (1:2 R:R)
        commission_pct:  float = 0.0003,   # 0.03% per leg
        initial_capital: float = 100_000.0,
    ):
        self.da  = data_agent      or DataAgent()
        self.ia  = indicator_agent or IndicatorAgent()
        self.sl  = stop_loss_pct
        self.tp  = take_profit_pct
        self.com = commission_pct
        self.cap = initial_capital

    # ── Main entry point ──────────────────────────────────

    def run(
        self,
        symbol:   str,
        strategy: str = "rsi",
        period:   str = "60d",
        interval: str = "1d",
    ) -> Dict:
        """
        Run backtest and return performance report.
        """
        symbol = normalise_symbol(symbol)
        df     = self.da.fetch(symbol, interval=interval, period=period)
        if df is None or df.empty:
            return {"error": f"No data for {symbol}"}

        enriched = self.ia.compute_all(df)
        enriched.dropna(subset=["Close"], inplace=True)

        if strategy not in STRATEGY_MAP:
            return {"error": f"Unknown strategy '{strategy}'. Choose from {list(STRATEGY_MAP)}"}

        signal_fn = STRATEGY_MAP[strategy]
        entries   = signal_fn(enriched)

        trades = self._simulate_trades(enriched, entries, symbol, strategy)
        return self._compute_report(trades, enriched)

    # ── Trade simulation ──────────────────────────────────

    def _simulate_trades(
        self,
        df:       pd.DataFrame,
        entries:  pd.Series,
        symbol:   str,
        strategy: str,
    ) -> List[Trade]:
        trades: List[Trade] = []
        in_trade  = False
        trade_action = None
        entry_price  = 0.0
        entry_time   = None

        for i in range(len(df)):
            price = float(df["Close"].iloc[i])
            ts    = df.index[i]
            sig   = int(entries.iloc[i])

            if not in_trade:
                if sig in (1, -1):
                    in_trade     = True
                    trade_action = "BUY" if sig == 1 else "SELL"
                    entry_price  = price * (1 + self.com)
                    entry_time   = ts
            else:
                # Check exit conditions
                if trade_action == "BUY":
                    ret    = (price - entry_price) / entry_price
                    exit_  = ret <= -self.sl or ret >= self.tp or sig == -1
                elif trade_action == "SELL":
                    ret    = (entry_price - price) / entry_price
                    exit_  = ret <= -self.sl or ret >= self.tp or sig == 1
                else:
                    exit_  = False

                if exit_ or i == len(df) - 1:
                    exit_price = price * (1 - self.com)
                    if trade_action == "BUY":
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price

                    pnl_abs = pnl_pct * self.cap
                    trades.append(Trade(
                        entry_time  = entry_time,
                        exit_time   = ts,
                        entry_price = round(entry_price, 2),
                        exit_price  = round(exit_price, 2),
                        action      = trade_action,
                        strategy    = strategy,
                        pnl_pct     = round(pnl_pct, 6),
                        pnl_abs     = round(pnl_abs, 2),
                        result      = "WIN" if pnl_pct > 0 else "LOSS",
                    ))
                    in_trade = False

        return trades

    # ── Report compilation ────────────────────────────────

    def _compute_report(self, trades: List[Trade], df: pd.DataFrame) -> Dict:
        if not trades:
            return {
                "total_trades": 0, "win_rate": 0, "total_return": 0,
                "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
                "max_drawdown": 0, "sharpe": 0,
                "trades": pd.DataFrame(), "equity_curve": pd.DataFrame(),
            }

        n          = len(trades)
        wins       = [t for t in trades if t.result == "WIN"]
        losses     = [t for t in trades if t.result == "LOSS"]
        win_rate   = len(wins) / n
        total_ret  = sum(t.pnl_pct for t in trades)
        avg_win    = np.mean([t.pnl_pct for t in wins]) if wins   else 0.0
        avg_loss   = np.mean([t.pnl_pct for t in losses]) if losses else 0.0
        gross_wins = sum(t.pnl_abs for t in wins)
        gross_loss = abs(sum(t.pnl_abs for t in losses))
        pf         = (gross_wins / gross_loss) if gross_loss > 0 else float("inf")

        # Equity curve
        equity = self.cap
        eq_rows = []
        for t in trades:
            equity += t.pnl_abs
            eq_rows.append({"date": t.exit_time, "equity": equity})
        eq_df = pd.DataFrame(eq_rows).set_index("date")

        # Max drawdown
        roll_max   = eq_df["equity"].cummax()
        drawdown   = (eq_df["equity"] - roll_max) / roll_max
        max_dd     = float(drawdown.min())

        # Sharpe (simplified, annualised)
        rets   = [t.pnl_pct for t in trades]
        sharpe = (np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0.0

        # Trade log DataFrame
        trade_df = pd.DataFrame([{
            "Entry Time":   t.entry_time.strftime("%Y-%m-%d %H:%M"),
            "Exit Time":    t.exit_time.strftime("%Y-%m-%d %H:%M"),
            "Action":       t.action,
            "Entry":        f"₹{t.entry_price:,.2f}",
            "Exit":         f"₹{t.exit_price:,.2f}",
            "P&L %":        f"{t.pnl_pct:.2%}",
            "P&L ₹":        f"₹{t.pnl_abs:,.2f}",
            "Result":       t.result,
        } for t in trades])

        return {
            "total_trades":  n,
            "win_rate":      win_rate,
            "total_return":  total_ret,
            "avg_win":       avg_win,
            "avg_loss":      avg_loss,
            "profit_factor": round(pf, 2),
            "max_drawdown":  max_dd,
            "sharpe":        round(sharpe, 2),
            "equity_curve":  eq_df,
            "trades":        trade_df,
        }


# ── Quick smoke-test ──────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bt = Backtester()
    report = bt.run("RELIANCE", strategy="rsi", period="90d", interval="1d")
    print(f"Trades: {report['total_trades']}")
    print(f"Win Rate: {report['win_rate']:.1%}")
    print(f"Total Return: {report['total_return']:.2%}")
    print(f"Sharpe: {report['sharpe']:.2f}")
    print(report["trades"])
