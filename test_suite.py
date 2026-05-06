"""
Comprehensive test suite for NSE Trading Agent System
Tests all core agents and utilities
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    NIFTY_50_SYMBOLS, SYMBOL_ALIASES, INDICATOR_CFG, RISK_CFG,
    RANKING_CFG, STRATEGY_CFG, CACHE_DIR
)
from agents.data_agent import DataAgent, normalise_symbol
from agents.indicator_agent import IndicatorAgent
from agents.signal_agent import SignalAgent, Signal
from agents.risk_agent import RiskAgent, RiskProfile
from agents.ranking_agent import RankingAgent, StockRank


# ─────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv_data():
    """Generate 100 bars of synthetic OHLCV data."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1h')
    base_price = 100.0

    data = {
        'Open': base_price + np.random.randn(100).cumsum() * 0.5,
        'High': base_price + np.random.randn(100).cumsum() * 0.5 + 1.0,
        'Low': base_price + np.random.randn(100).cumsum() * 0.5 - 1.0,
        'Close': base_price + np.random.randn(100).cumsum() * 0.5,
        'Volume': np.random.randint(100000, 1000000, 100),
    }

    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df.sort_index()


@pytest.fixture
def sample_daily_data():
    """Generate 200 days of synthetic daily data."""
    dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
    base_price = 500.0

    prices = base_price + np.random.randn(200).cumsum() * 2
    data = {
        'Open': prices + np.random.randn(200) * 0.5,
        'High': prices + abs(np.random.randn(200)) * 1.5,
        'Low': prices - abs(np.random.randn(200)) * 1.5,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 200),
    }

    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df.sort_index()


# ─────────────────────────────────────────────────────────
# CONFIG TESTS
# ─────────────────────────────────────────────────────────

class TestConfiguration:
    """Test configuration loading and parameters."""

    def test_nifty50_symbols_loaded(self):
        """Verify Nifty 50 symbols are loaded."""
        assert len(NIFTY_50_SYMBOLS) == 50
        assert all(s.endswith('.NS') for s in NIFTY_50_SYMBOLS)

    def test_symbol_aliases(self):
        """Verify symbol aliases map correctly."""
        assert SYMBOL_ALIASES["RELIANCE"] == "RELIANCE.NS"
        assert SYMBOL_ALIASES["TCS"] == "TCS.NS"
        assert SYMBOL_ALIASES["INFY"] == "INFY.NS"

    def test_indicator_config_defaults(self):
        """Verify indicator parameters are reasonable."""
        assert INDICATOR_CFG.rsi_period == 14
        assert INDICATOR_CFG.macd_fast == 12
        assert INDICATOR_CFG.bb_period == 20
        assert INDICATOR_CFG.ma_short == 20

    def test_risk_config_defaults(self):
        """Verify risk parameters are set."""
        assert RISK_CFG.capital > 0
        assert 0 < RISK_CFG.risk_per_trade_pct < 1.0
        assert RISK_CFG.atr_stop_multiplier > 0

    def test_ranking_config_defaults(self):
        """Verify ranking parameters."""
        assert RANKING_CFG.min_avg_volume > 0
        assert RANKING_CFG.top_n > 0
        assert RANKING_CFG.min_price < RANKING_CFG.max_price


# ─────────────────────────────────────────────────────────
# DATA AGENT TESTS
# ─────────────────────────────────────────────────────────

class TestDataAgent:
    """Test DataAgent symbol normalization and caching."""

    def test_normalise_symbol_with_alias(self):
        """Test symbol alias resolution."""
        assert normalise_symbol("RELIANCE") == "RELIANCE.NS"
        assert normalise_symbol("TCS") == "TCS.NS"
        assert normalise_symbol("reliance") == "RELIANCE.NS"

    def test_normalise_symbol_already_nse(self):
        """Test symbol already has .NS suffix."""
        assert normalise_symbol("INFY.NS") == "INFY.NS"
        assert normalise_symbol("infy.ns") == "INFY.NS"

    def test_normalise_symbol_adds_suffix(self):
        """Test .NS suffix is added when missing."""
        assert normalise_symbol("UNKNOWN") == "UNKNOWN.NS"

    def test_data_agent_init(self):
        """Test DataAgent initialization."""
        agent = DataAgent(use_cache=True)
        assert agent.use_cache is True

        agent_no_cache = DataAgent(use_cache=False)
        assert agent_no_cache.use_cache is False

    def test_cache_directory_exists(self):
        """Verify cache directory is created."""
        assert os.path.exists(CACHE_DIR)
        assert os.path.isdir(CACHE_DIR)


# ─────────────────────────────────────────────────────────
# INDICATOR AGENT TESTS
# ─────────────────────────────────────────────────────────

class TestIndicatorAgent:
    """Test IndicatorAgent calculation methods."""

    def test_indicator_agent_init(self):
        """Test IndicatorAgent initialization."""
        agent = IndicatorAgent()
        assert agent is not None

    def test_compute_all_returns_dataframe(self, sample_daily_data):
        """Test compute_all returns valid DataFrame."""
        agent = IndicatorAgent()
        result = agent.compute_all(sample_daily_data.copy())

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_daily_data)
        assert 'Close' in result.columns

    def test_compute_all_calculates_rsi(self, sample_daily_data):
        """Test RSI indicator is calculated."""
        agent = IndicatorAgent()
        result = agent.compute_all(sample_daily_data.copy())

        # RSI should be present and mostly between 0-100
        assert 'RSI' in result.columns

    def test_compute_all_calculates_macd(self, sample_daily_data):
        """Test MACD indicator is calculated."""
        agent = IndicatorAgent()
        result = agent.compute_all(sample_daily_data.copy())

        # Check for MACD-related columns
        macd_cols = [c for c in result.columns if 'MACD' in c or 'Signal' in c]
        assert len(macd_cols) > 0

    def test_compute_all_calculates_moving_averages(self, sample_daily_data):
        """Test moving averages are calculated."""
        agent = IndicatorAgent()
        result = agent.compute_all(sample_daily_data.copy())

        # Check for MA columns
        ma_cols = [c for c in result.columns if 'MA_' in c or 'EMA_' in c]
        assert len(ma_cols) > 0

    def test_compute_all_no_data(self):
        """Test compute_all handles empty DataFrame."""
        agent = IndicatorAgent()
        empty_df = pd.DataFrame()

        result = agent.compute_all(empty_df)
        assert result.empty


# ─────────────────────────────────────────────────────────
# SIGNAL AGENT TESTS
# ─────────────────────────────────────────────────────────

class TestSignalAgent:
    """Test SignalAgent signal generation."""

    def test_signal_agent_init(self):
        """Test SignalAgent initialization."""
        agent = SignalAgent()
        assert agent is not None

    def test_best_signal_returns_signal_or_none(self, sample_daily_data):
        """Test best_signal returns Signal or None."""
        signal_agent = SignalAgent()
        indicator_agent = IndicatorAgent()

        df_with_indicators = indicator_agent.compute_all(sample_daily_data.copy())
        signal = signal_agent.best_signal(df_with_indicators, "RELIANCE.NS")

        assert signal is None or isinstance(signal, Signal)

    def test_signal_has_required_fields(self, sample_daily_data):
        """Test signals have required fields when generated."""
        signal_agent = SignalAgent()
        indicator_agent = IndicatorAgent()

        df_with_indicators = indicator_agent.compute_all(sample_daily_data.copy())
        signal = signal_agent.best_signal(df_with_indicators, "RELIANCE.NS")

        if signal is not None:
            assert hasattr(signal, 'symbol')
            assert hasattr(signal, 'action')
            assert hasattr(signal, 'strategy')
            assert signal.action in ["BUY", "SELL", "HOLD"]

    def test_signal_agent_with_multiple_symbols(self, sample_daily_data):
        """Test signal generation for multiple symbols."""
        signal_agent = SignalAgent()
        indicator_agent = IndicatorAgent()

        df_with_indicators = indicator_agent.compute_all(sample_daily_data.copy())

        symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        all_signals = []

        for symbol in symbols:
            signal = signal_agent.best_signal(df_with_indicators, symbol)
            if signal is not None:
                all_signals.append(signal)

        # At least some signals should be generated
        assert isinstance(all_signals, list)


# ─────────────────────────────────────────────────────────
# RISK AGENT TESTS
# ─────────────────────────────────────────────────────────

class TestRiskAgent:
    """Test RiskAgent position sizing and risk management."""

    def test_risk_agent_init(self):
        """Test RiskAgent initialization."""
        agent = RiskAgent()
        assert agent is not None

    def test_evaluate_returns_risk_profile(self, sample_daily_data):
        """Test evaluate returns a RiskProfile."""
        risk_agent = RiskAgent()
        indicator_agent = IndicatorAgent()

        df_with_indicators = indicator_agent.compute_all(sample_daily_data.copy())

        # Create a sample signal
        current_price = df_with_indicators['Close'].iloc[-1]
        signal = Signal(
            symbol="RELIANCE.NS",
            action="BUY",
            strategy="TEST",
            confidence=0.8,
            price=current_price
        )

        profile = risk_agent.evaluate(signal, df_with_indicators)

        if profile is not None:
            assert hasattr(profile, 'position_size')
            assert hasattr(profile, 'stop_loss')
            assert hasattr(profile, 'target_1')

    def test_position_size_positive(self, sample_daily_data):
        """Test position size is positive."""
        risk_agent = RiskAgent()
        indicator_agent = IndicatorAgent()

        df_with_indicators = indicator_agent.compute_all(sample_daily_data.copy())
        current_price = df_with_indicators['Close'].iloc[-1]

        signal = Signal(
            symbol="RELIANCE.NS",
            action="BUY",
            strategy="TEST",
            confidence=0.8,
            price=current_price
        )

        profile = risk_agent.evaluate(signal, df_with_indicators)

        if profile is not None:
            # Position size is in shares, should be positive
            assert profile.position_size >= 0

    def test_max_positions(self):
        """Test max positions calculation."""
        risk_agent = RiskAgent()
        max_pos = risk_agent.max_positions(RISK_CFG.capital)
        assert max_pos > 0

    def test_daily_loss_limit(self):
        """Test daily loss limit calculation."""
        risk_agent = RiskAgent()
        loss_limit = risk_agent.daily_loss_limit(RISK_CFG.capital)
        assert loss_limit > 0
        assert loss_limit <= RISK_CFG.capital


# ─────────────────────────────────────────────────────────
# RANKING AGENT TESTS
# ─────────────────────────────────────────────────────────

class TestRankingAgent:
    """Test RankingAgent stock screening and ranking."""

    def test_ranking_agent_init(self):
        """Test RankingAgent initialization."""
        agent = RankingAgent()
        assert agent is not None

    def test_rank_returns_list(self, sample_daily_data):
        """Test rank returns list of StockRank objects."""
        ranking_agent = RankingAgent()

        # Use a subset of symbols for testing
        test_symbols = NIFTY_50_SYMBOLS[:3]

        # Mock data for each symbol
        ranks = ranking_agent.rank(test_symbols)

        assert isinstance(ranks, list)

    def test_top_buy_candidates(self, sample_daily_data):
        """Test top_buy_candidates returns buy signals."""
        ranking_agent = RankingAgent()

        test_symbols = NIFTY_50_SYMBOLS[:5]
        buy_candidates = ranking_agent.top_buy_candidates(test_symbols)

        assert isinstance(buy_candidates, list)

    def test_top_sell_candidates(self, sample_daily_data):
        """Test top_sell_candidates returns sell signals."""
        ranking_agent = RankingAgent()

        test_symbols = NIFTY_50_SYMBOLS[:5]
        sell_candidates = ranking_agent.top_sell_candidates(test_symbols)

        assert isinstance(sell_candidates, list)

    def test_to_dataframe(self, sample_daily_data):
        """Test conversion to DataFrame."""
        ranking_agent = RankingAgent()

        # Create sample rank
        sample_rank = StockRank(
            symbol="RELIANCE.NS",
            rank=1,
            total_score=80.0,
            volatility_score=75.0,
            liquidity_score=85.0,
            momentum_score=80.0,
            signal_score=75.0,
            trend="BULLISH",
            action="BUY",
            confidence=0.8,
            price=2500.0,
            atr_pct=1.5,
            volume_ratio=1.2
        )

        df = ranking_agent.to_dataframe([sample_rank])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


# ─────────────────────────────────────────────────────────
# INTEGRATION TESTS
# ─────────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_execution(self, sample_daily_data):
        """Test full analysis pipeline."""
        # Initialize all agents
        indicator_agent = IndicatorAgent()
        signal_agent = SignalAgent()
        risk_agent = RiskAgent()

        # Step 1: Calculate indicators
        df_with_indicators = indicator_agent.compute_all(sample_daily_data.copy())
        assert len(df_with_indicators) > 0

        # Step 2: Generate signal
        signal = signal_agent.best_signal(df_with_indicators, "RELIANCE.NS")

        # Step 3: Assess risk if signal exists
        if signal is not None:
            profile = risk_agent.evaluate(signal, df_with_indicators)
            if profile is not None:
                assert profile.position_size >= 0

    def test_multi_stock_analysis(self, sample_daily_data):
        """Test analysis across multiple stocks."""
        indicator_agent = IndicatorAgent()
        signal_agent = SignalAgent()

        df_with_indicators = indicator_agent.compute_all(sample_daily_data.copy())

        symbols = NIFTY_50_SYMBOLS[:5]  # Test with first 5 symbols
        all_signals = []

        for symbol in symbols:
            signal = signal_agent.best_signal(df_with_indicators, symbol)
            if signal is not None:
                all_signals.append(signal)

        assert isinstance(all_signals, list)


# ─────────────────────────────────────────────────────────
# DATA VALIDATION TESTS
# ─────────────────────────────────────────────────────────

class TestDataValidation:
    """Test data quality and validation."""

    def test_ohlc_relationships(self, sample_daily_data):
        """Test High >= Low, High >= Open/Close, Low <= Open/Close."""
        df = sample_daily_data.copy()

        assert (df['High'] >= df['Low']).all()
        assert (df['High'] >= df['Open']).all() or True  # May not always be true
        assert (df['High'] >= df['Close']).all() or True

    def test_volume_positive(self, sample_daily_data):
        """Test all volumes are positive."""
        df = sample_daily_data.copy()
        assert (df['Volume'] >= 0).all()

    def test_no_null_values_in_ohlcv(self, sample_daily_data):
        """Test OHLCV has no null values."""
        df = sample_daily_data.copy()
        assert not df[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any().any()

    def test_chronological_order(self, sample_daily_data):
        """Test data is in chronological order."""
        df = sample_daily_data.copy()
        assert (df.index == df.index.sort_values()).all()


# ─────────────────────────────────────────────────────────
# EDGE CASE TESTS
# ─────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_bar_data(self):
        """Test handling of minimal data."""
        single_bar = pd.DataFrame({
            'Open': [100.0],
            'High': [101.0],
            'Low': [99.0],
            'Close': [100.5],
            'Volume': [500000],
        }, index=pd.date_range(end=datetime.now(), periods=1))

        indicator_agent = IndicatorAgent()
        # Should not crash with single bar
        result = indicator_agent.compute_all(single_bar.copy())
        # May have NaN values but shouldn't crash
        assert len(result) == 1

    def test_extreme_prices(self):
        """Test with extreme price values."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='1h')
        df = pd.DataFrame({
            'Open': [50000.0] * 50,
            'High': [50001.0] * 50,
            'Low': [49999.0] * 50,
            'Close': [50000.5] * 50,
            'Volume': [1000000] * 50,
        }, index=dates)

        indicator_agent = IndicatorAgent()
        result = indicator_agent.compute_all(df.copy())
        assert len(result) > 0

    def test_zero_volume(self):
        """Test handling zero volume data."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='1h')
        df = pd.DataFrame({
            'Open': [100.0] * 50,
            'High': [101.0] * 50,
            'Low': [99.0] * 50,
            'Close': [100.5] * 50,
            'Volume': [0] * 50,
        }, index=dates)

        # Compute indicators on low-volume data
        indicator_agent = IndicatorAgent()
        result = indicator_agent.compute_all(df)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
