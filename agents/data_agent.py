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

# Suppress yfinance's internal noisy ERROR/WARNING logs for 404s and missing data.
# Our own DataAgent already handles these gracefully; the yfinance messages add no value.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

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

    # Symbols where Yahoo Finance returns 404 for longer periods.
    # We cap the period for these symbols in bulk_fetch to avoid noisy errors.
    # Format: { "SYMBOL.NS": "max_safe_period" }
    _PERIOD_CAP: Dict[str, str] = {
        "TATAMOTORS.NS": "60d",
        "LTIM.NS":       "60d",
    }

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        # Re-apply yfinance log suppression here in case Streamlit reconfigures
        # the root logger after module import, which would un-suppress it.
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
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

    # Fallback period ladder: if the requested period fails, try shorter ones.
    # This handles symbols like LTIM.NS and TATAMOTORS.NS that return 404 for 90d
    # but have valid data for 60d or 30d windows.
    _PERIOD_FALLBACKS: Dict[str, List[str]] = {
        "90d":  ["60d", "30d"],
        "180d": ["90d", "60d", "30d"],
        "365d": ["180d", "90d", "60d"],
        "30d":  ["20d"],
        "1d":   [],     # intraday — no fallback
    }

    def _fetch_with_retry(
        self, symbol: str, interval: str, period: str
    ) -> Optional[pd.DataFrame]:
        import random

        # Skip known-delisted symbols immediately
        if symbol in DELISTED_SYMBOLS:
            logger.info("Skipping delisted symbol: %s", symbol)
            return None

        # Build the list of periods to attempt: requested period + fallbacks
        periods_to_try = [period] + self._PERIOD_FALLBACKS.get(period, [])

        for try_period in periods_to_try:
            delay = self.RETRY_DELAY
            for attempt in range(1, self.MAX_RETRIES + 1):
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(interval=interval, period=try_period, auto_adjust=True)
                    if df.empty:
                        if attempt == self.MAX_RETRIES:
                            break       # try next period in fallback ladder
                        time.sleep(delay + random.uniform(0, 1))
                        delay *= 2
                        continue
                    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.index = pd.to_datetime(df.index)
                    df.dropna(inplace=True)
                    if try_period != period:
                        logger.info(
                            "Fetched %d rows for %s [%s/%s] (fallback from %s)",
                            len(df), symbol, interval, try_period, period,
                        )
                    else:
                        logger.info("Fetched %d rows for %s [%s/%s]", len(df), symbol, interval, period)
                    return df

                except Exception as exc:
                    exc_str = str(exc)
                    # 404 / delisted / no price — try shorter period immediately
                    if (
                        "delisted" in exc_str.lower()
                        or "no price data" in exc_str.lower()
                        or "not found" in exc_str.lower()
                        or "404" in exc_str
                        or "YFPricesMissingError" in type(exc).__name__
                    ):
                        logger.debug(
                            "Symbol %s period=%s returned no data — trying shorter period",
                            symbol, try_period,
                        )
                        break   # break inner retry loop → try next period
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
            # Inner loop exhausted without return → try next fallback period

        logger.warning("All periods exhausted for %s [%s] — skipping", symbol, interval)
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
            full_batch = to_download[batch_start: batch_start + self.BATCH_SIZE]

            # Separate symbols that need a capped period from the rest.
            # Sending them together in the same batch with different periods is not
            # possible with yf.download, so we pull them out and fetch individually.
            capped = [s for s in full_batch if s in self._PERIOD_CAP]
            batch  = [s for s in full_batch if s not in self._PERIOD_CAP]

            # Fetch capped symbols individually right now (before the batch call)
            for s in capped:
                safe_period = self._PERIOD_CAP[s]
                df = self._fetch_with_retry(s, interval, safe_period) or pd.DataFrame()
                result[s] = df
                if not df.empty and self.use_cache:
                    df.to_parquet(_cache_key(s, interval, period))

            if not batch:
                # All symbols in this chunk were capped — skip the batch download
                if batch_start + self.BATCH_SIZE < len(to_download):
                    time.sleep(self.BATCH_DELAY + random.uniform(0, 1))
                continue

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
                                    # Not in batch result — retry individually with period fallback
                                    logger.debug("No data in batch for %s — retrying individually", s)
                                    df = self._fetch_with_retry(s, interval, period) or pd.DataFrame()
                                    result[s] = df
                                    if not df.empty and self.use_cache:
                                        df.to_parquet(_cache_key(s, interval, period))
                                    continue
                                df = raw[s][["Open", "High", "Low", "Close", "Volume"]].copy()

                            df.index = pd.to_datetime(df.index)
                            df.dropna(inplace=True)

                            # If batch returned empty for this symbol, retry individually
                            if df.empty:
                                logger.debug("Empty batch result for %s — retrying individually", s)
                                df = self._fetch_with_retry(s, interval, period) or pd.DataFrame()

                            if not df.empty and self.use_cache:
                                df.to_parquet(_cache_key(s, interval, period))
                            result[s] = df

                        except Exception as parse_exc:
                            exc_str = str(parse_exc)
                            if "delisted" in exc_str.lower() or "no price" in exc_str.lower():
                                logger.debug("Skipping %s (delisted/no data)", s)
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
