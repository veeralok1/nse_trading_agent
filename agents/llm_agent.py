"""
llm_agent.py — LLM-powered analysis agent
Uses OpenRouter (recommended) / Anthropic / OpenAI to generate
natural-language explanations of stock analysis results.

Enable by setting in .env:
    ENABLE_LLM_CHAT=true
    OPENROUTER_API_KEY=sk-or-v1-...
    OPENROUTER_MODEL=anthropic/claude-haiku-4-5
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    ENABLE_LLM_CHAT, LLM_PROVIDER,
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL,
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# SYSTEM PROMPT — sets the LLM persona
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert NSE (National Stock Exchange India) stock market analyst
covering both intraday trading and positional/swing investing.

You have deep knowledge of:
- Technical analysis (RSI, MACD, Moving Averages, VWAP, Bollinger Bands, ATR)
- Indian market microstructure (Nifty50, sectoral trends, F&O dynamics)
- Short-term intraday strategies AND medium-term swing/positional strategies
- Risk management for both day traders and investors

When responding:
- Adapt your tone to the question: intraday → fast, tactical; investment → measured, strategic
- For intraday: focus on momentum, VWAP, volume spikes, and 5-min signals
- For swing/investment (weeks to months): focus on trend (MA50/MA200 alignment), RSI on
  daily chart, sector momentum, and support/resistance levels
- Always state the time horizon explicitly (intraday / 1-month / 3-month etc.)
- Explain signals in plain English — avoid excessive jargon
- Keep responses concise (under 180 words) unless asked for more detail
- Format numbers in Indian style (₹ for rupees, use Lakhs/Crores where appropriate)
- NEVER give personalised financial advice — always clarify "this is for educational purposes only"
"""


# ─────────────────────────────────────────────────────────
# LLM AGENT
# ─────────────────────────────────────────────────────────

class LLMAgent:
    """
    Wraps OpenRouter / Anthropic / OpenAI to generate natural-language
    stock analysis. Falls back gracefully if no API key is configured.
    """

    def __init__(self):
        self.enabled  = ENABLE_LLM_CHAT
        self.provider = LLM_PROVIDER
        self._client  = None   # lazy-init

        if self.enabled:
            self._init_client()
        else:
            logger.info("LLMAgent disabled (ENABLE_LLM_CHAT=false in .env)")

    # ── Client initialisation ─────────────────────────────

    def _init_client(self):
        """Lazily create the API client based on LLM_PROVIDER."""
        try:
            if self.provider == "openrouter":
                self._init_openrouter()
            elif self.provider == "anthropic":
                self._init_anthropic()
            elif self.provider == "openai":
                self._init_openai()
            else:
                logger.warning("Unknown LLM_PROVIDER '%s' — LLM disabled", self.provider)
                self.enabled = False
        except ImportError as e:
            logger.error(
                "Missing library for %s: %s. Run: pip install openai",
                self.provider, e
            )
            self.enabled = False

    def _init_openrouter(self):
        """
        OpenRouter uses the OpenAI-compatible API format.
        Just point the base_url to openrouter.ai and use your OR key.
        """
        if not OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY is empty — LLM disabled")
            self.enabled = False
            return
        from openai import OpenAI
        self._client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self._model = OPENROUTER_MODEL
        logger.info("LLMAgent ready via OpenRouter (model=%s)", self._model)

    def _init_anthropic(self):
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY is empty — LLM disabled")
            self.enabled = False
            return
        import anthropic
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._model  = "claude-haiku-4-5"
        logger.info("LLMAgent ready via Anthropic direct (model=%s)", self._model)

    def _init_openai(self):
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is empty — LLM disabled")
            self.enabled = False
            return
        from openai import OpenAI
        self._client = OpenAI(api_key=OPENAI_API_KEY)
        self._model  = "gpt-4o-mini"
        logger.info("LLMAgent ready via OpenAI direct (model=%s)", self._model)

    # ── Core chat method ─────────────────────────────────

    def chat(self, user_message: str, context: Optional[Dict] = None) -> str:
        """
        Send a message to the LLM with optional analysis context.
        Returns the response string, or a fallback message if LLM is disabled.
        """
        if not self.enabled or self._client is None:
            return self._fallback(user_message)

        try:
            prompt = self._build_prompt(user_message, context)

            if self.provider == "anthropic":
                return self._call_anthropic(prompt)
            else:
                # OpenRouter and OpenAI both use the same OpenAI SDK format
                return self._call_openai_compatible(prompt)

        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return f"⚠️ AI analysis unavailable right now ({exc}). Using rule-based analysis above."

    def _call_openai_compatible(self, prompt: str) -> str:
        """Works for both OpenRouter and OpenAI (same SDK)."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=300,
            temperature=0.4,
            extra_headers={
                # OpenRouter-specific: identifies your app (optional but good practice)
                "HTTP-Referer": "https://github.com/nse-trading-agent",
                "X-Title":      "NSE Trading Agent",
            } if self.provider == "openrouter" else {},
        )
        return response.choices[0].message.content.strip()

    def _call_anthropic(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    # ── Prompt builders ───────────────────────────────────

    def _build_prompt(self, user_message: str, context: Optional[Dict]) -> str:
        """Attach analysis data to the user's question."""
        if not context:
            return user_message

        symbol   = context.get("symbol", "")
        quote    = context.get("quote", {})
        ind      = context.get("indicators", {})
        trend    = context.get("trend", "")
        signals  = context.get("signals", [])
        risk     = context.get("risk_profile")

        parts = [f"User question: {user_message}", ""]

        if symbol:
            parts.append(f"Stock: {symbol}")
        if quote:
            parts.append(
                f"Price: ₹{quote.get('last_price','?')}  "
                f"Change: {quote.get('change_pct', 0):+.2f}%"
            )
        if ind:
            parts.append(
                f"Indicators — RSI: {ind.get('rsi','?')}  "
                f"MACD: {ind.get('macd','?')}  "
                f"VWAP: {ind.get('vwap','?')}  "
                f"ATR: {ind.get('atr','?')}  "
                f"Vol Ratio: {ind.get('volume_ratio','?')}x"
            )
        if trend:
            parts.append(f"Trend: {trend}")
        if signals:
            sig_strs = [
                f"{s['action']} via {s['strategy']} ({s['confidence']:.0%})"
                for s in signals[:3]
            ]
            parts.append(f"Signals: {', '.join(sig_strs)}")
        if risk:
            parts.append(
                f"Risk — Entry: ₹{risk.get('entry_price','?')}  "
                f"SL: ₹{risk.get('stop_loss','?')}  "
                f"Target: ₹{risk.get('target_3','?')}  "
                f"R:R 1:{risk.get('risk_reward','?')}"
            )

        return "\n".join(parts)

    def analyze_stock(self, symbol: str, context: Dict) -> str:
        """Convenience wrapper for a full stock analysis prompt."""
        return self.chat(
            f"Give me a brief intraday trading analysis for {symbol}.",
            context=context,
        )

    def explain_signal(self, signal_name: str, symbol: str) -> str:
        """Explain what a strategy signal means in plain English."""
        return self.chat(
            f"Explain what a '{signal_name}' signal means for {symbol} "
            f"in 2-3 sentences for a beginner trader.",
        )

    # ── Fallback (no LLM) ─────────────────────────────────

    def _fallback(self, user_message: str) -> str:
        return (
            "🔒 **AI analysis is disabled.**\n\n"
            "To enable it, set these in your `.env` file:\n"
            "```\n"
            "OPENROUTER_API_KEY=sk-or-v1-your-key-here\n"
            "ENABLE_LLM_CHAT=true\n"
            "LLM_PROVIDER=openrouter\n"
            "```\n"
            "Get a free key at [openrouter.ai](https://openrouter.ai/keys)"
        )

    @property
    def is_available(self) -> bool:
        return self.enabled and self._client is not None

    @property
    def model_name(self) -> str:
        return getattr(self, "_model", "none")


# ── Quick smoke-test ──────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = LLMAgent()
    print(f"Provider : {agent.provider}")
    print(f"Available: {agent.is_available}")
    print(f"Model    : {agent.model_name}")
    if agent.is_available:
        reply = agent.chat("Is RELIANCE good for intraday today?")
        print(f"\nResponse:\n{reply}")
    else:
        print(agent._fallback("test"))
