"""
conversational_agent.py — Conversational Agent
Parses natural-language user queries and routes them to the
appropriate agents, returning structured + human-readable responses.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SYMBOL_ALIASES, NIFTY_50_SYMBOLS
from agents.data_agent import DataAgent, normalise_symbol
from agents.indicator_agent import IndicatorAgent
from agents.signal_agent import SignalAgent
from agents.risk_agent import RiskAgent
from agents.ranking_agent import RankingAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────────────────

# Order matters: more specific patterns first
INTENT_PATTERNS = {
    # Swing / positional / long-term investment queries
    "SWING_BEST": (
        r"\b(invest|investment|portfolio|positional|swing|long.?term|wealth)\b"
        r"|"
        r"\b(\d+)\s*(month|week|year|yr)s?\b"
        r"|"
        r"\b(3\s*month|6\s*month|1\s*year|quarterly)\b"
    ),
    # Intraday-specific
    "INTRADAY_BEST": (
        r"\b(best|top|good|strong|find)\b.{0,40}"
        r"\b(intraday|day.?trad|scalp)\b"
        r"|"
        r"\b(intraday|day.?trad|scalp)\b.{0,40}\b(stock|pick|opportunit)\b"
    ),
    "ANALYZE":       r"\b(analyz|analyse|analysis|check|look at|tell me about)\b",
    "TREND":         r"\b(trend|direction|bullish|bearish|moving)\b",
    "SIGNAL":        r"\b(buy|sell|signal|action|should i|good for|worth|entry)\b",
    "PRICE":         r"\b(price|ltp|current|live|quote|last)\b",
    "COMPARE":       r"\b(compare|versus|vs\.?|better)\b",
    "HELP":          r"\b(help|how|what can|commands?|features?)\b",
    "BACKTEST":      r"\b(backtest|back.?test|historical performance|past returns)\b",
    # Generic "top N stocks" without intraday/swing clue → default swing
    "SWING_BEST_GENERIC": r"\b(top|best|find|give|suggest|recommend)\b.{0,40}\b(stocks?|shares?|scrips?)\b",
}


def detect_intent(query: str) -> str:
    q = query.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, q, re.IGNORECASE):
            # Normalise the generic variant
            if intent == "SWING_BEST_GENERIC":
                return "SWING_BEST"
            return intent
    return "ANALYZE"   # default fallback


def extract_symbol(query: str) -> Optional[str]:
    """
    Extract the first NSE symbol mentioned in the query.

    Strategy:
      1. Check SYMBOL_ALIASES (longest match first to handle "TATA MOTORS" etc.)
      2. Scan all uppercase tokens (2-15 chars) and return the first that is NOT
         a common English stop-word or intent keyword.
    """
    q = query.upper()

    # Check aliases (multi-word first)
    for alias in sorted(SYMBOL_ALIASES.keys(), key=len, reverse=True):
        if re.search(rf'\b{re.escape(alias)}\b', q):
            return SYMBOL_ALIASES[alias]

    # Words that look like tickers but are definitely not stock symbols
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

    # Scan all word-boundary tokens — take the first plausible ticker
    for match in re.finditer(r'\b([A-Z][A-Z0-9&\-]{1,14}(?:\.NS)?)\b', q):
        token = match.group(1)
        base  = token.replace(".NS", "").replace("-", "")
        if base not in SKIP and len(base) >= 2:
            return normalise_symbol(token)

    return None


def extract_number(query: str, keyword: str = "top") -> int:
    """Extract digit after keyword (e.g. 'top 5' → 5)."""
    # First try after keyword
    m = re.search(rf'\b{keyword}\s+(\d+)\b', query.lower())
    if m:
        return int(m.group(1))
    # Then any digit in range 1-20
    m = re.search(r'\b([1-9]|1\d|20)\b', query)
    return int(m.group(1)) if m else 5


def extract_horizon(query: str) -> str:
    """Detect investment horizon from query (returns '1d', '5d', '30d', '90d', '180d', '365d')."""
    q = query.lower()
    if re.search(r'\b(1\s*year|12\s*month|annual|yearly)\b', q):
        return "365d"
    if re.search(r'\b(6\s*month|half.?year)\b', q):
        return "180d"
    if re.search(r'\b(3\s*month|quarter|90\s*day)\b', q):
        return "90d"
    if re.search(r'\b(1\s*month|30\s*day)\b', q):
        return "30d"
    if re.search(r'\b(week|7\s*day)\b', q):
        return "5d"
    return "90d"   # default for swing queries


# ─────────────────────────────────────────────────────────
# RESPONSE HELPERS
# ─────────────────────────────────────────────────────────

def _emoji_action(action: str) -> str:
    return {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}.get(action, action)


def _trend_emoji(trend: str) -> str:
    return {"BULLISH": "📈 BULLISH", "BEARISH": "📉 BEARISH",
            "SIDEWAYS": "➡️ SIDEWAYS"}.get(trend, trend)


def _horizon_label(period: str) -> str:
    return {
        "5d": "1 week", "30d": "1 month", "90d": "3 months",
        "180d": "6 months", "365d": "1 year",
    }.get(period, period)


# ─────────────────────────────────────────────────────────
# CONVERSATIONAL AGENT
# ─────────────────────────────────────────────────────────

class ConversationalAgent:
    """
    Single entry-point for all user queries.
    Returns a dict with:
      - "text"    : human-readable answer (markdown)
      - "data"    : structured payload (for the UI)
      - "intent"  : detected intent
      - "symbol"  : extracted symbol (if any)
    """

    def __init__(
        self,
        data_agent:      Optional[DataAgent]      = None,
        indicator_agent: Optional[IndicatorAgent] = None,
        signal_agent:    Optional[SignalAgent]    = None,
        risk_agent:      Optional[RiskAgent]      = None,
        ranking_agent:   Optional[RankingAgent]   = None,
    ):
        self.da = data_agent      or DataAgent()
        self.ia = indicator_agent or IndicatorAgent()
        self.sa = signal_agent    or SignalAgent()
        self.ra = risk_agent      or RiskAgent()
        self.rk = ranking_agent   or RankingAgent(
            data_agent=self.da, indicator_agent=self.ia, signal_agent=self.sa
        )

    # ── Main entry point ──────────────────────────────────

    def query(self, user_input: str) -> Dict:
        intent = detect_intent(user_input)
        symbol = extract_symbol(user_input)
        logger.info("Query: '%s' | Intent=%s | Symbol=%s", user_input, intent, symbol)

        if intent == "HELP":
            return self._help_response()

        # ── Key routing fix ───────────────────────────────────────────────────
        # If a specific stock symbol is present, the user wants analysis of THAT
        # stock — even if "invest", "3 months", "target" etc. appear in the query.
        # SWING_BEST / INTRADAY_BEST screeners only run when NO symbol is given.
        # ─────────────────────────────────────────────────────────────────────
        if intent == "SWING_BEST":
            if symbol is not None:
                # User said "analyse HAL for 3 months" — give per-stock swing analysis
                horizon = extract_horizon(user_input)
                return self._swing_stock_analysis(symbol, horizon)
            n       = extract_number(user_input)
            horizon = extract_horizon(user_input)
            return self._best_swing(n, horizon)

        if intent == "INTRADAY_BEST":
            if symbol is not None:
                # User said "should I buy HAL intraday?" — single-stock analysis
                return self._signal_analysis(symbol)
            n = extract_number(user_input)
            return self._best_intraday(n)

        if symbol is None and intent in ("ANALYZE", "TREND", "SIGNAL", "PRICE"):
            return {
                "text":   "❓ I couldn't identify a stock symbol in your query.\n\n"
                          "Please mention a stock name like **RELIANCE**, **TCS**, or **INFY**.\n\n"
                          "Or try: _\"Top 5 stocks for 3 months\"_ or _\"Best intraday stocks today\"_",
                "data":   {},
                "intent": intent,
                "symbol": None,
            }

        if intent == "PRICE":
            return self._live_price(symbol)

        if intent == "TREND":
            return self._trend_analysis(symbol)

        if intent == "SIGNAL":
            return self._signal_analysis(symbol)

        if intent == "BACKTEST":
            return {
                "text":   "📊 Please use the **Backtester** tab to run historical strategy analysis.",
                "data":   {},
                "intent": intent,
                "symbol": symbol,
            }

        # Default: full analysis
        return self._full_analysis(symbol)

    # ── Intent handlers ───────────────────────────────────

    def _full_analysis(self, symbol: str) -> Dict:
        df       = self.da.fetch(symbol, "5m", "2d")
        enriched = self.ia.compute_all(df)
        signals  = self.sa.generate(enriched, symbol)
        ind_sum  = self.ia.get_summary(enriched)
        trend    = self.ia.trend_direction(enriched)
        quote    = self.da.get_live_quote(symbol)

        risk_profile = None
        if signals:
            risk_profile = self.ra.evaluate(signals[0], enriched)

        name = symbol.replace(".NS", "")
        lines = [
            f"## 📊 Analysis: {name}",
            f"**Price:** ₹{quote.get('last_price', '—')}  "
            f"({quote.get('change_pct', 0):+.2f}%)",
            f"**Trend:** {_trend_emoji(trend)}",
            "",
            "### 📐 Indicators",
            f"- RSI: `{ind_sum.get('rsi') or '—'}`"
              + (" ⚠️ Overbought" if (ind_sum.get('rsi') or 0) > 70
                 else " ⚠️ Oversold" if (ind_sum.get('rsi') or 0) < 30 else ""),
            f"- MACD: `{ind_sum.get('macd') or '—'}` | Signal: `{ind_sum.get('macd_signal') or '—'}`",
            f"- VWAP: ₹`{ind_sum.get('vwap') or '—'}`",
            f"- MA20/50/200: ₹`{ind_sum.get('ma_20') or '—'}` / "
              f"₹`{ind_sum.get('ma_50') or '—'}` / ₹`{ind_sum.get('ma_200') or '—'}`",
            f"- Bollinger %B: `{ind_sum.get('bb_pct_b') or '—'}`",
            f"- ATR: `{ind_sum.get('atr') or '—'}`",
            f"- Volume Ratio: `{ind_sum.get('volume_ratio') or '—'}x`",
            "",
        ]

        if signals:
            lines += ["### 🔔 Signals"]
            for s in signals[:3]:
                lines.append(f"- **{_emoji_action(s.action)}** via `{s.strategy}` "
                             f"(conf {s.confidence:.0%}): {'; '.join(s.reasons)}")

        if risk_profile:
            lines += [
                "",
                "### 🛡️ Risk Profile",
                f"- **Entry:** ₹{risk_profile.entry_price}",
                f"- **Stop Loss:** ₹{risk_profile.stop_loss} ({risk_profile.risk_pct:.1f}%)",
                f"- **Target 1:** ₹{risk_profile.target_1}  **T2:** ₹{risk_profile.target_2}  "
                  f"**T3:** ₹{risk_profile.target_3}",
                f"- **R:R Ratio:** 1:{risk_profile.risk_reward:.1f}",
                f"- **Position Size:** {risk_profile.position_size} shares "
                  f"(risk ₹{risk_profile.risk_amount_inr:,.0f})",
                f"- {risk_profile.notes}",
            ]
        else:
            lines.append("\n_No high-confidence signal — no trade recommended._")

        return {
            "text":   "\n".join(lines),
            "data":   {
                "quote":        quote,
                "indicators":   ind_sum,
                "trend":        trend,
                "signals":      [{"action": s.action, "strategy": s.strategy,
                                  "confidence": s.confidence, "reasons": s.reasons}
                                 for s in signals],
                "risk_profile": self.ra.to_dict(risk_profile) if risk_profile else None,
                "df":           enriched,
            },
            "intent": "ANALYZE",
            "symbol": symbol,
        }

    def _live_price(self, symbol: str) -> Dict:
        quote = self.da.get_live_quote(symbol)
        name  = symbol.replace(".NS", "")
        arrow = "▲" if quote.get("change", 0) >= 0 else "▼"
        text  = (
            f"**{name}** — ₹{quote.get('last_price', '—')} "
            f"{arrow} {quote.get('change', 0):+.2f} ({quote.get('change_pct', 0):+.2f}%)\n"
            f"Volume: {quote.get('volume', 0):,}  |  Prev Close: ₹{quote.get('prev_close', '—')}"
        )
        return {"text": text, "data": quote, "intent": "PRICE", "symbol": symbol}

    def _trend_analysis(self, symbol: str) -> Dict:
        df       = self.da.fetch(symbol, "5m", "5d")
        enriched = self.ia.compute_all(df)
        trend    = self.ia.trend_direction(enriched)
        ind_sum  = self.ia.get_summary(enriched)
        name     = symbol.replace(".NS", "")

        vol_str = ""
        if ind_sum.get("volume_ratio"):
            vol_str = f"\n- Volume Ratio: `{ind_sum['volume_ratio']:.2f}x`"

        text = (
            f"## Trend Analysis: {name}\n"
            f"**Overall Trend:** {_trend_emoji(trend)}\n\n"
            f"- MA20: ₹`{ind_sum.get('ma_20') or '—'}`\n"
            f"- MA50: ₹`{ind_sum.get('ma_50') or '—'}`\n"
            f"- MA200: ₹`{ind_sum.get('ma_200') or '—'}`\n"
            f"- Momentum (5-bar): `{ind_sum.get('momentum_5') or '—'}%`{vol_str}\n"
        )
        return {"text": text, "data": {"trend": trend, "indicators": ind_sum},
                "intent": "TREND", "symbol": symbol}

    def _signal_analysis(self, symbol: str) -> Dict:
        df       = self.da.fetch(symbol, "5m", "2d")
        enriched = self.ia.compute_all(df)
        signals  = self.sa.generate(enriched, symbol)
        action, conf = self.sa.aggregate_action(enriched, symbol)
        name     = symbol.replace(".NS", "")

        if signals:
            lines = [f"## Signal Analysis: {name}",
                     f"**Aggregate: {_emoji_action(action)} ({conf:.0%})**", ""]
            for s in signals:
                lines.append(f"- **{s.strategy}** ({s.confidence:.0%}): {'; '.join(s.reasons)}")
        else:
            lines = [f"## Signal Analysis: {name}",
                     "No strong signals detected. **HOLD / Monitor**.",
                     "_Wait for a clearer setup before entering._"]

        return {
            "text": "\n".join(lines),
            "data": {
                "action": action, "confidence": conf,
                "signals": [{"action": s.action, "strategy": s.strategy,
                             "confidence": s.confidence} for s in signals],
                "df": enriched,
            },
            "intent": "SIGNAL",
            "symbol": symbol,
        }

    def _swing_stock_analysis(self, symbol: str, period: str = "90d") -> Dict:
        """
        Full swing / investment analysis for a SINGLE stock over the given horizon.
        Uses daily bars, longer-period MAs, and gives price targets.
        """
        name    = symbol.replace(".NS", "")
        horizon = _horizon_label(period)

        # ── Fetch daily data for swing horizon ───────────────────────────────
        df       = self.da.fetch(symbol, interval="1d", period=period)
        if df is None or df.empty:
            # Fallback to 5m data if daily unavailable
            df = self.da.fetch(symbol, "5m", "5d")
        enriched = self.ia.compute_all(df)

        # ── Also grab a quick live quote ─────────────────────────────────────
        quote    = self.da.get_live_quote(symbol)
        signals  = self.sa.generate(enriched, symbol)
        ind_sum  = self.ia.get_summary(enriched)
        trend    = self.ia.trend_direction(enriched)

        price    = quote.get("last_price") or (float(df["Close"].iloc[-1]) if not df.empty else 0)
        change_p = quote.get("change_pct", 0)

        # ── Risk / target levels ─────────────────────────────────────────────
        risk_profile = None
        if signals:
            risk_profile = self.ra.evaluate(signals[0], enriched)

        # ── Derive swing-specific price levels ───────────────────────────────
        # Use ATR for stop, and Fibonacci-style targets
        atr_val   = ind_sum.get("atr") or 0
        ma50      = ind_sum.get("ma_50") or 0
        ma200     = ind_sum.get("ma_200") or 0
        rsi_val   = ind_sum.get("rsi") or 0
        bb_upper  = ind_sum.get("bb_upper") or 0
        bb_lower  = ind_sum.get("bb_lower") or 0

        # Swing stop = 1.5× ATR below price (or 3% — whichever is wider)
        atr_stop_gap  = max(atr_val * 1.5, price * 0.03) if atr_val else price * 0.03
        swing_stop    = round(price - atr_stop_gap, 2)
        # Conservative / mid / aggressive targets (2R / 3R / 5R)
        risk_pts      = price - swing_stop
        tgt1          = round(price + risk_pts * 2, 2)
        tgt2          = round(price + risk_pts * 3, 2)
        tgt3          = round(price + risk_pts * 5, 2)
        potential_pct = round((tgt2 - price) / price * 100, 1) if price else 0

        # ── Trend description ────────────────────────────────────────────────
        ma_comment = ""
        if ma50 and ma200:
            if price > ma200 > 0:
                ma_comment = "Price is **above MA200** — long-term uptrend intact ✅"
            elif price < ma200:
                ma_comment = "Price is **below MA200** — long-term downtrend ⚠️"
        if ma50 and price > ma50:
            ma_comment += "  |  Above MA50 (medium trend bullish)"
        elif ma50:
            ma_comment += "  |  Below MA50 (medium trend weak)"

        # ── RSI comment ──────────────────────────────────────────────────────
        rsi_comment = ""
        if rsi_val > 70:
            rsi_comment = f"RSI `{rsi_val:.1f}` — **Overbought**, wait for a pullback before entering 🔴"
        elif rsi_val < 30:
            rsi_comment = f"RSI `{rsi_val:.1f}` — **Oversold**, potential reversal zone 🟢"
        elif 40 <= rsi_val <= 60:
            rsi_comment = f"RSI `{rsi_val:.1f}` — **Neutral zone**, healthy for accumulation"
        else:
            rsi_comment = f"RSI `{rsi_val:.1f}`"

        # ── Investment verdict ───────────────────────────────────────────────
        action, conf = self.sa.aggregate_action(enriched, symbol)
        if action == "BUY" and trend in ("BULLISH",) and (rsi_val < 65 or rsi_val == 0):
            verdict = f"✅ **YES — Consider investing in {name}** for the {horizon} horizon."
            verdict_detail = (
                f"The stock shows a {_trend_emoji(trend)} trend with a {_emoji_action(action)} signal "
                f"at {conf:.0%} confidence. Risk-adjusted targets are attractive."
            )
        elif action == "SELL" or trend == "BEARISH":
            verdict = f"⛔ **AVOID or WAIT** — {name} is showing weakness right now."
            verdict_detail = (
                f"Current signal is {_emoji_action(action)} with trend {_trend_emoji(trend)}. "
                f"Wait for the trend to stabilise before entering for {horizon}."
            )
        else:
            verdict = f"🟡 **NEUTRAL / MONITOR** — {name} has mixed signals for {horizon}."
            verdict_detail = (
                f"Signal is {_emoji_action(action)} ({conf:.0%}), trend {_trend_emoji(trend)}. "
                f"Watch for breakout above MA50 before committing capital."
            )

        # ── Build response text ──────────────────────────────────────────────
        lines = [
            f"## 📊 {name} — {horizon} Investment Analysis",
            f"**Current Price:** ₹{price:,.2f}  ({change_p:+.2f}%)",
            f"**Horizon:** {horizon}  |  **Trend:** {_trend_emoji(trend)}",
            "",
            f"### 🏦 Investment Verdict",
            verdict,
            verdict_detail,
            "",
            "### 📐 Key Indicators (Daily Chart)",
            ma_comment,
            rsi_comment,
        ]

        if bb_upper and bb_lower:
            lines.append(
                f"Bollinger Bands: ₹`{bb_lower:,.2f}` — ₹`{bb_upper:,.2f}` "
                f"(Price {'near upper band ⚠️' if price > bb_upper * 0.97 else 'within bands ✅' if price > bb_lower * 1.03 else 'near lower band 🟢'})"
            )

        lines += [
            "",
            "### 🎯 Price Targets",
            f"| Level | Price | Move |",
            f"|---|---|---|",
            f"| 🛑 Stop Loss (swing) | ₹{swing_stop:,.2f} | {(swing_stop-price)/price*100:+.1f}% |",
            f"| 🎯 Target 1 (Conservative) | ₹{tgt1:,.2f} | {(tgt1-price)/price*100:+.1f}% |",
            f"| 🎯 Target 2 (Base case) | ₹{tgt2:,.2f} | {(tgt2-price)/price*100:+.1f}% |",
            f"| 🚀 Target 3 (Aggressive) | ₹{tgt3:,.2f} | {(tgt3-price)/price*100:+.1f}% |",
            "",
        ]

        if risk_profile:
            lines += [
                "### 🛡️ Risk Profile",
                f"- **Position Size:** {risk_profile.position_size} shares "
                  f"(risk ₹{risk_profile.risk_amount_inr:,.0f})",
                f"- **R:R Ratio:** 1:{risk_profile.risk_reward:.1f}",
                f"- {risk_profile.notes}",
                "",
            ]

        if signals:
            lines += ["### 🔔 Active Signals"]
            for s in signals[:2]:
                lines.append(
                    f"- **{_emoji_action(s.action)}** via `{s.strategy}` "
                    f"({s.confidence:.0%}): {'; '.join(s.reasons)}"
                )
            lines.append("")

        lines.append(
            f"> ⚠️ _This is a technical analysis for educational purposes only. "
            f"For {horizon} investment, also review fundamentals (P/E, earnings, sector trends) "
            f"and consult a SEBI-registered advisor before investing._"
        )

        return {
            "text": "\n".join(lines),
            "data": {
                "quote":        quote,
                "indicators":   ind_sum,
                "trend":        trend,
                "action":       action,
                "confidence":   conf,
                "signals":      [{"action": s.action, "strategy": s.strategy,
                                  "confidence": s.confidence, "reasons": s.reasons}
                                 for s in signals],
                "risk_profile": self.ra.to_dict(risk_profile) if risk_profile else None,
                "targets": {
                    "stop_loss": swing_stop,
                    "target_1":  tgt1,
                    "target_2":  tgt2,
                    "target_3":  tgt3,
                },
                "df": enriched,
            },
            "intent":  "SWING_STOCK",
            "symbol":  symbol,
            "horizon": period,
        }

    def _best_intraday(self, top_n: int = 5) -> Dict:
        """Return top intraday stocks (short timeframe, 5-min bars)."""
        ranked  = self.rk.rank(NIFTY_50_SYMBOLS[:25], interval="5m", period="5d", top_n=top_n)
        rank_df = self.rk.to_dataframe(ranked)

        lines = [f"## 🏆 Top {top_n} **Intraday** Stocks (Day Trading)\n",
                 "> ⚡ These are short-term (same-day) opportunities based on 5-min chart signals.\n"]
        for r in ranked:
            lines.append(
                f"**#{r.rank} {r.symbol.replace('.NS','')}** — "
                f"{_emoji_action(r.action)} ({r.confidence:.0%})  "
                f"₹{r.price:.2f}  Score: {r.total_score:.1f}  {_trend_emoji(r.trend)}"
            )
            if r.signals:
                lines.append(f"  _Strategy: {r.signals[0].strategy}_")
            lines.append("")

        lines.append("\n> ⚠️ _For educational purposes only. Always use stop-losses._")

        return {
            "text":   "\n".join(lines),
            "data":   {"ranked": rank_df, "stocks": ranked},
            "intent": "INTRADAY_BEST",
            "symbol": None,
        }

    def _best_swing(self, top_n: int = 5, period: str = "90d") -> Dict:
        """Return top positional/swing/investment stocks (daily bars, longer period)."""
        ranked  = self.rk.rank_swing(NIFTY_50_SYMBOLS[:50], period=period, top_n=top_n)
        rank_df = self.rk.to_dataframe(ranked)
        horizon = _horizon_label(period)

        lines = [
            f"## 📈 Top {top_n} Stocks — **{horizon} Investment Perspective**\n",
            f"> 📅 Analysis based on **daily charts over {horizon}**. "
            f"Suitable for positional/swing trading, not intraday.\n",
        ]
        for r in ranked:
            sym = r.symbol.replace(".NS", "")
            lines.append(
                f"**#{r.rank} {sym}** — {_emoji_action(r.action)} ({r.confidence:.0%})  "
                f"₹{r.price:.2f}  Score: {r.total_score:.1f}  {_trend_emoji(r.trend)}"
            )
            ind = r.indicators
            ma50  = ind.get("ma_50")
            ma200 = ind.get("ma_200")
            rsi   = ind.get("rsi")
            details = []
            if ma50 and ma200:
                above = "above" if r.price > ma50 else "below"
                details.append(f"Price {above} MA50")
            if rsi:
                details.append(f"RSI {rsi:.1f}")
            if details:
                lines.append(f"  _{' | '.join(details)}_")
            if r.signals:
                lines.append(f"  _Signal: {r.signals[0].strategy}_")
            lines.append("")

        lines.append(
            f"\n> ⚠️ _These are based on technical analysis only. "
            f"For long-term investing, also consider fundamentals (P/E, earnings, sector outlook). "
            f"This is for educational purposes only._"
        )

        return {
            "text":   "\n".join(lines),
            "data":   {"ranked": rank_df, "stocks": ranked, "horizon": period},
            "intent": "SWING_BEST",
            "symbol": None,
        }

    def _help_response(self) -> Dict:
        text = """## 🤖 NSE Trading Agent — Help

**Example queries you can ask:**

| Query | What it does |
|---|---|
| `Analyze RELIANCE` | Full technical analysis with signals & risk |
| `Is TCS good for intraday?` | Intraday signal analysis |
| `Top 5 intraday stocks today` | Best day-trading opportunities |
| `Top 5 stocks for 3 months` | Best swing/positional picks |
| `Best stocks for investment` | Long-term positional analysis |
| `What is the trend of HDFC Bank?` | Trend analysis |
| `Price of INFY` | Live quote |
| `Best stocks to buy today` | Ranked BUY list (swing) |

**Time horizon detection:**
- "intraday", "today", "scalp" → Intraday (5-min charts)
- "3 months", "invest", "swing", "positional" → Swing (daily charts)

**Supported stocks:** All Nifty 50, Nifty Bank & Midcap symbols.
"""
        return {"text": text, "data": {}, "intent": "HELP", "symbol": None}


# ── Quick smoke-test ──────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = ConversationalAgent()

    queries = [
        "Top 5 stocks for 3 months investment",
        "Best intraday stocks today",
        "Analyze RELIANCE",
        "Is TCS good for intraday today?",
        "Best stocks for 6 months",
        "What is the trend of HDFC Bank?",
        "Price of INFY",
    ]
    for q in queries:
        print(f"\n{'='*60}")
        intent = detect_intent(q)
        print(f"Q: {q}  →  Intent: {intent}")
