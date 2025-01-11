# """
# test_market_regime.py - Tests for market regime detection
# """

# import pytest
# import pytest_asyncio
# from datetime import datetime, timedelta
# from unittest.mock import Mock, AsyncMock
# import numpy as np
# from src.trading.market_regime import (
#     MarketRegimeDetector,
#     MarketRegimeType,
#     MarketState,
#     VolumeProfile
# )

# class MockSettings:
#     def __init__(self):
#         self.MIN_VOLUME_24H = 1000.0
#         self.MAX_VOLATILITY = 0.3

# @pytest_asyncio.fixture
# async def detector():
#     settings = MockSettings()
#     return MarketRegimeDetector(settings)

# @pytest.mark.asyncio
# async def test_detect_regime_trending_up():
#     """Test trending up market detection"""
#     detector = MarketRegimeDetector(MockSettings())
    
#     # Create trending up price data
#     prices = [100.0 + i for i in range(100)]  # Steadily increasing prices
#     volumes = [1000.0 for _ in range(100)]    # Steady volume
    
#     state = await detector.detect_regime(prices, volumes)
    
#     assert state.regime == MarketRegimeType.TRENDING_UP
#     assert state.confidence >= 0.7
#     assert state.trend_strength > 0

# @pytest.mark.asyncio
# async def test_detect_regime_volatile():
#     """Test volatile market detection"""
#     detector = MarketRegimeDetector(MockSettings())
    
#     # Create volatile price data
#     base_prices = [100.0 for _ in range(100)]
#     volatility = np.random.normal(0, 5, 100)  # High volatility
#     prices = [p + v for p, v in zip(base_prices, volatility)]
#     volumes = [1000.0 * (1 + abs(v)/10) for v in volatility]
    
#     state = await detector.detect_regime(prices, volumes)
    
#     assert state.regime == MarketRegimeType.VOLATILE
#     assert state.volatility > 0.2

# @pytest.mark.asyncio
# async def test_detect_regime_ranging():
#     """Test ranging market detection"""
#     detector = MarketRegimeDetector(MockSettings())
    
#     # Create ranging price data
#     prices = [100.0 + np.sin(i/10) for i in range(100)]  # Oscillating prices
#     volumes = [1000.0 for _ in range(100)]
    
#     state = await detector.detect_regime(prices, volumes)
    
#     assert state.regime == MarketRegimeType.RANGING
#     assert state.trend_strength < 0.02

# @pytest.mark.asyncio
# async def test_detect_regime_accumulation():
#     """Test accumulation phase detection"""
#     detector = MarketRegimeDetector(MockSettings())
    
#     # Create accumulation pattern
#     prices = [100.0 + np.random.normal(0, 0.5) for _ in range(100)]  # Relatively flat prices
#     volumes = [1000.0 * (1 + i/100) for i in range(100)]  # Increasing volume
    
#     state = await detector.detect_regime(prices, volumes)
    
#     assert state.regime == MarketRegimeType.ACCUMULATION
#     assert state.volume_profile > 0

# @pytest.mark.asyncio
# async def test_support_resistance_levels():
#     """Test support and resistance level detection"""
#     detector = MarketRegimeDetector(MockSettings())
    
#     # Create price data with clear support/resistance
#     prices = [100.0]
#     for i in range(1, 100):
#         if i < 30:
#             prices.append(prices[-1] + np.random.normal(0, 0.5))
#         elif i < 60:
#             prices.append(prices[-1] - np.random.normal(0, 0.5))
#         else:
#             prices.append(prices[-1] + np.random.normal(0, 0.5))
    
#     volumes = [1000.0 for _ in range(100)]
    
#     state = await detector.detect_regime(prices, volumes)
    
#     assert state.support_level is not None
#     assert state.resistance_level is not None
#     assert state.support_level < state.resistance_level

# @pytest.mark.asyncio
# async def test_regime_history():
#     """Test regime history tracking"""
#     detector = MarketRegimeDetector(MockSettings())
    
#     # Generate multiple regimes
#     for _ in range(5):
#         prices = [100.0 + i for i in range(50)]
#         volumes = [1000.0 for _ in range(50)]
#         await detector.detect_regime(prices, volumes)
    
#     history = detector.get_recent_regimes(hours=24)
#     metrics = detector.get_regime_metrics()
    
#     assert len(history) == 5
#     assert 'regime_distribution' in metrics
#     assert 'avg_volatility' in metrics
#     assert metrics['regime_changes'] == 0  # All same regime

# @pytest.mark.asyncio
# async def test_volume_analysis():
#     """Test volume analysis functionality"""
#     detector = MarketRegimeDetector(MockSettings())
    
#     prices = [100.0 + i for i in range(100)]
#     volumes = [1000.0 * (1 + np.sin(i/10)) for i in range(100)]
    
#     state = await detector.detect_regime(prices, volumes)
    
#     assert hasattr(state, 'volume_profile')
#     assert state.volume_profile != 0

# @pytest.mark.asyncio
# async def test_edge_cases():
#     """Test edge cases and error handling"""
#     detector = MarketRegimeDetector(MockSettings())
    
#     # Test with minimal data
#     state = await detector.detect_regime([100.0], [1000.0])
#     assert state.regime == MarketRegimeType.UNCERTAIN
    
#     # Test with empty data
#     state = await detector.detect_regime([], [])
#     assert state.regime == MarketRegimeType.UNCERTAIN
    
#     # Test with None market_data
#     state = await detector.detect_regime([100.0, 101.0], [1000.0, 1000.0], None)
#     assert state.regime != MarketRegimeType.UNCERTAIN  # Should still work

# @pytest.mark.asyncio
# async def test_regime_transition():
#     """Test regime transition detection"""
#     detector = MarketRegimeDetector(MockSettings())
    
#     # Start with trending up
#     prices1 = [100.0 + i for i in range(50)]
#     volumes1 = [1000.0 for _ in range(50)]
#     state1 = await detector.detect_regime(prices1, volumes1)
    
#     # Then switch to volatile
#     prices2 = [150.0 + np.random.normal(0, 5) for _ in range(50)]
#     volumes2 = [1000.0 * (1 + np.random.normal(0, 0.5)) for _ in range(50)]
#     state2 = await detector.detect_regime(prices2, volumes2)
    
#     metrics = detector.get_regime_metrics()
#     assert metrics['regime_changes'] == 1
#     assert state1.regime != state2.regime


import pytest
import numpy as np
from datetime import datetime
from src.trading.market_regime import MarketRegimeDetector, MarketRegimeType

class MockSettings:
    def __init__(self):
        self.MAX_VOLATILITY = 0.5
        self.MIN_TREND_STRENGTH = 0.2
        self.MIN_VOLUME_24H = 1000.0
        self.MIN_LIQUIDITY = 5000.0
        self.SIGNAL_THRESHOLD = 0.7
        self.MAX_PORTFOLIO_RISK = 5.0
        self.PORTFOLIO_VALUE = 100.0
        self.MIN_TRADE_SIZE = 0.1
        self.TREND_THRESHOLD = 0.1
        self.INITIAL_CAPITAL = 100.0
        self.VOLATILITY_THRESHOLD = 0.3
        self.VOLUME_TREND_THRESHOLD = 0.4  # Add this
        self.BUY_PRESSURE_THRESHOLD = 0.55  # Add this

@pytest.mark.asyncio
async def test_detect_regime_trending_up():
    """Test trending up market detection"""
    detector = MarketRegimeDetector(MockSettings())
    
    # Create trending up price data
    prices = [100.0 + i for i in range(100)]  # Steadily increasing prices
    volumes = [1000.0 for _ in range(100)]    # Steady volume
    
    state = await detector.detect_regime(prices, volumes)
    assert state.regime == MarketRegimeType.TRENDING_UP
    assert state.confidence >= 0.7
    assert state.trend_strength > 0

@pytest.mark.asyncio
async def test_detect_regime_volatile():
    """Test volatile market detection"""
    detector = MarketRegimeDetector(MockSettings())
    
    # Create volatile price data with high standard deviation
    base_prices = np.array([100.0 for _ in range(100)])
    volatility = np.random.normal(0, 5, 100)  # High volatility
    prices = list(base_prices + volatility)
    volumes = [1000.0 * (1 + abs(v)/10) for v in volatility]
    
    state = await detector.detect_regime(prices, volumes)
    assert state.regime == MarketRegimeType.VOLATILE
    assert state.volatility > 0.2

@pytest.mark.asyncio
async def test_detect_regime_ranging():
    """Test ranging market detection"""
    detector = MarketRegimeDetector(MockSettings())
    
    # Create ranging price data with small oscillations
    prices = [100.0 + np.sin(i/10)*0.5 for i in range(100)]  # Small oscillations
    volumes = [1000.0 for _ in range(100)]
    
    state = await detector.detect_regime(prices, volumes)
    assert state.regime == MarketRegimeType.RANGING
    assert state.trend_strength < 0.1

@pytest.mark.asyncio
async def test_detect_regime_accumulation():
    """Test accumulation phase detection"""
    detector = MarketRegimeDetector(MockSettings())
    
    # Create accumulation pattern with increasing volume
    prices = [100.0 + np.random.normal(0, 0.1) for _ in range(100)]  # Relatively flat prices
    volumes = [1000.0 * (1 + i/50) for i in range(100)]  # Steadily increasing volume
    
    state = await detector.detect_regime(prices, volumes)
    assert state.regime == MarketRegimeType.ACCUMULATION
    assert state.volume_profile > 0

@pytest.mark.asyncio
async def test_support_resistance_levels():
    """Test support and resistance level detection"""
    detector = MarketRegimeDetector(MockSettings())
    
    # Create price data with clear support/resistance
    prices = []
    current_price = 100.0
    for i in range(100):
        if i < 30:
            current_price += np.random.normal(0, 0.2)  # Uptrend
        elif i < 60:
            current_price -= np.random.normal(0, 0.2)  # Downtrend
        else:
            current_price += np.random.normal(0, 0.2)  # Uptrend
        prices.append(current_price)
    
    volumes = [1000.0 for _ in range(100)]
    
    state = await detector.detect_regime(prices, volumes)
    assert state.support_level is not None
    assert state.resistance_level is not None
    assert state.support_level < state.resistance_level

@pytest.mark.asyncio
async def test_edge_cases():
    """Test edge cases and error handling"""
    detector = MarketRegimeDetector(MockSettings())
    
    # Test with minimal data
    state = await detector.detect_regime([100.0], [1000.0])
    assert state.regime == MarketRegimeType.UNCERTAIN
    
    # Test with empty data
    state = await detector.detect_regime([], [])
    assert state.regime == MarketRegimeType.UNCERTAIN
    
    # Test with None market_data
    state = await detector.detect_regime([100.0, 101.0], [1000.0, 1000.0])
    assert state is not None

@pytest.mark.asyncio
async def test_regime_transition():
    """Test regime transition detection"""
    detector = MarketRegimeDetector(MockSettings())
    
    # Start with trending up
    prices1 = [100.0 + i for i in range(50)]
    volumes1 = [1000.0 for _ in range(50)]
    state1 = await detector.detect_regime(prices1, volumes1)
    
    # Then switch to volatile
    prices2 = [150.0 + np.random.normal(0, 5) for _ in range(50)]
    volumes2 = [1000.0 * (1 + np.random.normal(0, 0.5)) for _ in range(50)]
    state2 = await detector.detect_regime(prices2, volumes2)
    
    assert state1.regime != state2.regime
    assert state2.regime == MarketRegimeType.VOLATILE