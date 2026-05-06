"""
data_agent.py — Data Agent
Fetches, validates, and caches NSE stock data via yfinance.
Handles retries, symbol normalisation, and multi-timeframe requests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

# Resolve config relative to project root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    CACHE_DIR, CACHE_TTL_SECONDS, SYMBOL_ALIASES,
    NIFTY_50_SYMBOLS, NIFTY_BANK_SYMBOLS, MIDCAP_SYMBOLS,
    HISTORICAL_PERIOD, INTRADAY_PERIOD, DELISTED_SYMBOLS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def normalise_symbol(raw: str) -> str:
    """Convert user-friendly names to Yahoo Finance NSE symbols."""
    raw = raw.strip().upper()
    if raw in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[raw]
    if not raw.endswith(".NS") and not raw.endswith(".BO"):
        return raw + ".NS"
    return raw


def _cache_key(symbol: str, interval: str, period: str) -> str:
    h = hashlib.md5(f"{symbol}_{interval}_{period}".encode()).hexdigest()[:10]
    return os.path.join(CACHE_DIR, f"{h}.parquet")


def _is_cache_valid(path: str, ttl: int) -> bool:
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < ttl


# ─────────────────────────────────────────────────────────
# DATA AGENT
# ─────────────────────────────────────────────────────────

class DataAgent:
    """
    Fetches OHLCV data from Yahoo Finance with:
    - Automatic NSE symbol mapping
    - Disk-based cache with TTL
    - Exponential-backoff retries with jitter
    - Batched bulk downloading (avoids rate limits)
    - Graceful skip of delisted / missing symbols
    """

    MAX_RETRIES  = 3
    RETRY_DELAY  = 3.0    # seconds base (doubles + jitter each retry)
    BATCH_SIZE   = 8      # symbols per yf.download() call
    BATCH_DELAY  = 2.0    # seconds between batches (rate-limit guard)

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        logger.info("DataAgent initialised (cache=%s)", use_cache)

    # ── Single stock ──────────────────────────────────────

    def fetch(
        self,
        symbol: str,
        interval: str = "5m",
        period: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for *symbol* at *interval* over *period*.

        Parameters
        ----------
        symbol   : NSE ticker, e.g. "RELIANCE.NS" or alias "RELIANCE"
        interval : yfinance interval string, e.g. "1m","5m","15m","1d"
        period   : yfinance period string, e.g. "1d","5d","30d","90d"

        Returns
        -------
        pd.DataFrame with columns [Open, High, Low, Close, Volume]
        """
        symbol = normalise_symbol(symbol)
        cache_path = _cache_key(symbol, interval, period)
        ttl = CACHE_TTL_SECONDS.get(interval, 300)

        if self.use_cache and _is_cache_valid(cache_path, ttl):
            logger.debug("Cache hit: %s %s %s", symbol, interval, period)
            return pd.read_parquet(cache_path)

        df = self._fetch_with_retry(symbol, interval, period)

        if df is not None and not df.empty and self.use_cache:
            df.to_parquet(cache_path)

        return df if df is not None else pd.DataFrame()

    def _fetch_with_retry(
        self, symbol: str, interval: str, period: str
    ) -> Optional[pd.DataFrame]:
        import random

        # Skip known-delisted symbols immediately
        if symbol in DELISTED_SYMBOLS:
            logger.info("Skipping delisted symbol: %s", symbol)
            return None

        delay = self.RETRY_DELAY
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(interval=interval, period=period, auto_adjust=True)
                if df.empty:
                    logger.warning("Empty data for %s (attempt %d) — may be delisted", symbol, attempt)
                    if attempt == self.MAX_RETRIES:
                        return None          # give up — don't keep retrying missing symbols
                    time.sleep(delay + random.uniform(0, 1))
                    delay *= 2
                    continue
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                df.dropna(inplace=True)
                logger.info("Fetched %d rows for %s [%s/%s]", len(df), symbol, interval, period)
                return df

            except Exception as exc:
                exc_str = str(exc)
                # Delisted / no data — no point retrying
                if "delisted" in exc_str.lower() or "no price data" in exc_str.lower() \
                        or "YFPricesMissingError" in type(exc).__name__:
                    logger.warning("Symbol %s skipped — %s", symbol, exc_str[:80])
                    return None
                # Rate limit — wait longer
                if "rate" in exc_str.lower() or "429" in exc_str or "Too Many" in exc_str:
                    wait = delay * 2 + random.uniform(2, 5)
                    logger.warning("Rate limited on %s — waiting %.1fs", symbol, wait)
                    time.sleep(wait)
                    delay *= 2
                    continue
                logger.error("Attempt %d failed for %s: %s", attempt, symbol, exc_str[:120])
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2
        return None

    # ── Live quote ────────────────────────────────────────

    def get_live_quote(self, symbol: str) -> Dict:
        """Return the latest price, volume, and change info."""
        symbol = normalise_symbol(symbol)
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            df = self.fetch(symbol, interval="1m", period="1d")
            last_price = float(df["Close"].iloc[-1]) if not df.empty else float(info.last_price or 0)
            prev_close = float(info.previous_close or 0)
            change     = last_price - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0.0
            volume     = int(df["Volume"].iloc[-1]) if not df.empty else 0
            return {
                "symbol":      symbol,
                "last_price":  round(last_price, 2),
                "prev_close":  round(prev_close, 2),
                "change":      round(change, 2),
                "change_pct":  round(change_pct, 2),
                "volume":      volume,
                "timestamp":   datetime.now().isoformat(),
            }
        except Exception as exc:
            logger.error("Live quote failed for %s: %s", symbol, exc)
            return {"symbol": symbol, "error": str(exc)}

    # ── Historical daily data ─────────────────────────────

    def fetch_historical(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Return daily OHLCV for the past *days* days."""
        period = f"{days}d"
        return self.fetch(symbol, interval="1d", period=period)

    # ── Multi-timeframe ───────────────────────────────────

    def fetch_multi_tf(
        self, symbol: str, intervals: List[str] = ("5m", "15m", "1d")
    ) -> Dict[str, pd.DataFrame]:
        """Return a dict of DataFrames keyed by interval."""
        result = {}
        for iv in intervals:
            period = INTRADAY_PERIOD if iv in ("1m", "5m", "15m", "30m", "1h") else HISTORICAL_PERIOD
            result[iv] = self.fetch(symbol, interval=iv, period=period)
        return result

    # ── Bulk download ─────────────────────────────────────

    def bulk_fetch(
        self,
        symbols: List[str],
        interval: str = "5m",
        period: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Download multiple symbols in small batches to avoid Yahoo Finance
        rate limits on shared IPs (e.g. Streamlit Cloud).

        Downloads BATCH_SIZE symbols at a time with BATCH_DELAY seconds
        between batches. Already-cached symbols are served from disk.
        Delisted / missing symbols are skipped silently.

        Returns dict {normalised_symbol: DataFrame}.
        """
        import random

        # Normalise and filter out known-delisted symbols up front
        norm_symbols = [
            normalise_symbol(s) for s in symbols
            if normalise_symbol(s) not in DELISTED_SYMBOLS
        ]
        result: Dict[str, pd.DataFrame] = {}
        to_download: List[str] = []

        # Serve from cache where possible
        ttl = CACHE_TTL_SECONDS.get(interval, 300)
        for s in norm_symbols:
            cache_path = _cache_key(s, interval, period)
            if self.use_cache and _is_cache_valid(cache_path, ttl):
                result[s] = pd.read_parquet(cache_path)
            else:
                to_download.append(s)

        if not to_download:
            return result

        logger.info(
            "Bulk downloading %d symbols in batches of %d [%s/%s]",
            len(to_download), self.BATCH_SIZE, interval, period,
        )

        # Split into batches
        for batch_start in range(0, len(to_download), self.BATCH_SIZE):
            batch = to_download[batch_start: batch_start + self.BATCH_SIZE]
            batch_ok = False

            for attempt in range(1, self.MAX_RETRIES + 1):
                try:
                    raw = yf.download(
                        tickers=" ".join(batch),
                        interval=interval,
                        period=period,
                        group_by="ticker",
                        auto_adjust=True,
                        threads=False,     # serial within batch — gentler on rate limits
                        progress=False,
                    )

                    for s in batch:
                        try:
                            if len(batch) == 1:
                                df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                            else:
                                if s not in raw.columns.get_level_values(0):
                                    logger.warning("No data returned for %s — skipping", s)
                                    result[s] = pd.DataFrame()
                                    continue
                                df = raw[s][["Open", "High", "Low", "Close", "Volume"]].copy()

                            df.index = pd.to_datetime(df.index)
                            df.dropna(inplace=True)

                            if not df.empty and self.use_cache:
                                df.to_parquet(_cache_key(s, interval, period))
                            result[s] = df

                        except Exception as parse_exc:
                            exc_str = str(parse_exc)
                            if "delisted" in exc_str.lower() or "no price" in exc_str.lower():
                                logger.warning("Skipping %s (delisted/no data)", s)
                            else:
                                logger.warning("Parse error for %s: %s", s, exc_str[:80])
                            result[s] = pd.DataFrame()

                    batch_ok = True
                    break   # batch succeeded

                except Exception as exc:
                    exc_str = str(exc)
                    if "rate" in exc_str.lower() or "429" in exc_str or "Too Many" in exc_str:
                        wait = (4 ** attempt) + random.uniform(1, 3)
                        logger.warning(
                            "Rate limited (batch %d, attempt %d) — waiting %.1fs",
                            batch_start // self.BATCH_SIZE + 1, attempt, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.error("Batch download error (attempt %d): %s", attempt, exc_str[:120])
                        time.sleep(self.RETRY_DELAY * attempt)

            if not batch_ok:
                # Final fallback: fetch individually
                logger.warning("Batch failed — falling back to individual fetches for %s", batch)
                for s in batch:
                    result[s] = self.fetch(s, interval, period)

            # Polite delay between batches (skip after last batch)
            if batch_start + self.BATCH_SIZE < len(to_download):
                time.sleep(self.BATCH_DELAY + random.uniform(0, 1))

        logger.info(
            "Bulk fetch complete: %d/%d symbols returned data",
            sum(1 for df in result.values() if df is not None and not df.empty),
            len(norm_symbols),
        )
        return result

    # ── Stock info ────────────────────────────────────────

    def get_info(self, symbol: str) -> Dict:
        """Return fundamental / descriptive info for a symbol."""
        symbol = normalise_symbol(symbol)
        try:
            info = yf.Ticker(symbol).info
            keys = [
                "longName", "sector", "industry", "marketCap",
                "trailingPE", "forwardPE", "dividendYield",
                "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
                "averageVolume", "beta",
            ]
            return {k: info.get(k) for k in keys}
        except Exception as exc:
            logger.error("Info fetch failed for %s: %s", symbol, exc)
            return {}

    # ── Screening helper ─────────────────────────────────

    def get_universe(self, index: str = "nifty50") -> List[str]:
        """Return the pre-configured symbol universe."""
        mapping = {
            "nifty50":   NIFTY_50_SYMBOLS,
            "niftybank": NIFTY_BANK_SYMBOLS,
            "midcap":    MIDCAP_SYMBOLS,
            "all":       NIFTY_50_SYMBOLS + NIFTY_BANK_SYMBOLS + MIDCAP_SYMBOLS,
        }
        return mapping.get(index.lower(), NIFTY_50_SYMBOLS)


# ── Quick smoke-test ──────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = DataAgent()
    df = agent.fetch("RELIANCE", interval="5m", period="1d")
    print(df.tail())
    print(agent.get_live_quote("TCS"))
