"""
src/tests/test_technical_indicators.py - Unit tests for technical indicators
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from src.trading.technical_indicators import (
    TechnicalIndicators,
    RSIResult,
    MACDResult,
    BollingerResult
)

@pytest.fixture
def indicators():
    return TechnicalIndicators()

@pytest.fixture
def sample_prices():
    # Create sample price data with known characteristics
    return [
        100.0, 102.0, 104.0, 103.0, 106.0,  # Uptrend
        105.0, 103.0, 102.0, 101.0, 99.0,   # Downtrend
        100.0, 101.0, 100.0, 101.0, 100.0,  # Sideways
        102.0, 104.0, 106.0, 108.0, 110.0,  # Strong uptrend
        108.0, 106.0, 104.0, 102.0, 100.0   # Strong downtrend
    ]

class TestRSI:
    def test_rsi_calculation(self, indicators, sample_prices):
        result = indicators.calculate_rsi(sample_prices)
        assert isinstance(result, RSIResult)
        assert 0 <= result.value <= 100
        
    def test_rsi_overbought(self, indicators):
        # Create prices that should trigger overbought
        prices = [100.0] + [101.0 + i for i in range(15)]  # Strong uptrend
        result = indicators.calculate_rsi(prices)
        assert result.overbought
        assert result.signal == "sell"
        
    def test_rsi_oversold(self, indicators):
        # Create prices that should trigger oversold
        prices = [100.0] + [99.0 - i for i in range(15)]  # Strong downtrend
        result = indicators.calculate_rsi(prices)
        assert result.oversold
        assert result.signal == "buy"

    def test_rsi_insufficient_data(self, indicators):
        result = indicators.calculate_rsi([100.0, 101.0])  # Too few prices
        assert result is None

class TestMACD:
    def test_macd_calculation(self, indicators, sample_prices):
        result = indicators.calculate_macd(sample_prices)
        assert isinstance(result, MACDResult)
        
    def test_macd_buy_signal(self, indicators):
        # Create prices that should generate buy signal
        prices = [100.0] * 20 + [101.0 + i for i in range(10)]
        result = indicators.calculate_macd(prices)
        assert result.signal == "buy"
        assert result.histogram > 0
        
    def test_macd_sell_signal(self, indicators):
        # Create prices that should generate sell signal
        prices = [100.0] * 20 + [99.0 - i for i in range(10)]
        result = indicators.calculate_macd(prices)
        assert result.signal == "sell"
        assert result.histogram < 0

    def test_macd_insufficient_data(self, indicators):
        result = indicators.calculate_macd([100.0] * 10)  # Too few prices
        assert result is None

class TestBollingerBands:
    def test_bollinger_calculation(self, indicators, sample_prices):
        result = indicators.calculate_bollinger_bands(sample_prices)
        assert isinstance(result, BollingerResult)
        assert result.upper > result.middle > result.lower
        
    def test_bollinger_buy_signal(self, indicators):
        # Create prices that should generate buy signal (price below lower band)
        base_price = 100.0
        prices = [base_price] * 19 + [base_price * 0.95]
        result = indicators.calculate_bollinger_bands(prices)
        assert result.signal == "buy"
        assert result.value < result.lower
        
    def test_bollinger_sell_signal(self, indicators):
        # Create prices that should generate sell signal (price above upper band)
        base_price = 100.0
        prices = [base_price] * 19 + [base_price * 1.05]
        result = indicators.calculate_bollinger_bands(prices)
        assert result.signal == "sell"
        assert result.value > result.upper

    def test_bollinger_insufficient_data(self, indicators):
        result = indicators.calculate_bollinger_bands([100.0] * 10)  # Too few prices
        assert result is None

class TestPriceAction:
    def test_price_action_analysis(self, indicators, sample_prices):
        result = indicators.analyze_price_action(sample_prices)
        assert isinstance(result, dict)
        assert 'indicators' in result
        assert 'signal_strength' in result
        assert 'combined_signal' in result
        
    def test_strong_buy_signal(self, indicators):
        # Create prices that should generate strong buy signal
        prices = [100.0] * 20 + [98.0] * 5 + [99.0, 100.0, 101.0, 102.0, 103.0]
        result = indicators.analyze_price_action(prices)
        assert result['combined_signal'] == "buy"
        assert result['signal_strength'] > 0.5
        
    def test_strong_sell_signal(self, indicators):
        # Create prices that should generate strong sell signal
        prices = [100.0] * 20 + [102.0] * 5 + [101.0, 100.0, 99.0, 98.0, 97.0]
        result = indicators.analyze_price_action(prices)
        assert result['combined_signal'] == "sell"
        assert result['signal_strength'] < -0.5

    def test_neutral_signal(self, indicators):
        # Create prices that should generate neutral signal
        prices = [100.0] * 30  # Flat price action
        result = indicators.analyze_price_action(prices)
        assert result['combined_signal'] == "neutral"
        assert -0.5 <= result['signal_strength'] <= 0.5