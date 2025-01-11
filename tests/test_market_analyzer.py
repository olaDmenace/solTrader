import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock
import numpy as np
from src.trading.market_analyzer import MarketAnalyzer, MarketIndicators

class MockSettings:
    def __init__(self):
        self.MIN_VOLUME_24H = 1000.0
        self.MIN_LIQUIDITY = 5000.0

@pytest.fixture
def mock_jupiter_client() -> AsyncMock:
    client = AsyncMock()
    price_history = [{'price': float(i)} for i in range(1, 31)]
    client.get_price_history.return_value = price_history
    
    client.get_market_depth.return_value = {
        'bids': [{'size': 1000.0, 'price': 1.0}],
        'asks': [{'size': 1000.0, 'price': 1.0}],
        'totalLiquidity': 1000000.0,
        'recent_volumes': [1000.0] * 30
    }
    
    client.get_price.return_value = {'price': 1.0, 'outAmount': '1000000'}
    return client

@pytest.fixture
def market_analyzer(mock_jupiter_client):
    settings = MockSettings()
    return MarketAnalyzer(mock_jupiter_client, settings)

@pytest.mark.asyncio
async def test_analyze_market(market_analyzer, mock_jupiter_client):
    result = await market_analyzer.analyze_market("test_token")
    assert isinstance(result, MarketIndicators)
    assert 0 <= result.rsi <= 100
    assert result.volume_profile > 0
    assert -1 <= result.price_momentum <= 1
    assert 0 <= result.liquidity_score <= 1
    assert result.volatility >= 0
    assert 0 <= result.trend_strength <= 1
    assert result.macd is not None
    assert result.bollinger is not None
    assert -1 <= result.signal_strength <= 1

@pytest.mark.asyncio
async def test_technical_indicators_integration(market_analyzer):
    test_prices = [float(i) for i in range(1, 31)]
    test_volumes = [1000.0] * 30
    
    result = await market_analyzer.analyze_market(
        "test_token",
        price_data=test_prices,
        volume_data=test_volumes
    )
    
    assert result.macd is not None
    assert hasattr(result.macd, 'macd_line')
    assert hasattr(result.macd, 'signal_line')
    assert hasattr(result.macd, 'histogram')
    
    assert result.bollinger is not None
    assert hasattr(result.bollinger, 'upper')
    assert hasattr(result.bollinger, 'middle')
    assert hasattr(result.bollinger, 'lower')
    assert result.bollinger.upper > result.bollinger.middle > result.bollinger.lower

@pytest.mark.asyncio
async def test_insufficient_data(market_analyzer):
    result = await market_analyzer.analyze_market(
        "test_token",
        price_data=[1.0],
        volume_data=[1000.0]
    )
    assert result is None

@pytest.mark.asyncio
async def test_get_price_data(market_analyzer, mock_jupiter_client):
    prices = await market_analyzer._get_price_data("test_token")
    assert isinstance(prices, list)
    assert len(prices) == 30
    assert all(isinstance(p, float) for p in prices)

@pytest.mark.asyncio
async def test_get_volume_data(market_analyzer, mock_jupiter_client):
    volumes = await market_analyzer._get_volume_data("test_token")
    assert isinstance(volumes, list)
    assert len(volumes) == 30
    assert all(isinstance(v, float) for v in volumes)

def test_calculate_rsi(market_analyzer):
    prices = [100.0, 101.0, 102.0, 101.5, 103.0]
    rsi = market_analyzer._calculate_rsi(prices)
    assert 0 <= rsi <= 100

def test_analyze_volume_profile(market_analyzer):
    volumes = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
    profile = market_analyzer._analyze_volume_profile(volumes)
    assert profile > 0

def test_calculate_momentum(market_analyzer):
    prices = [100.0, 101.0, 102.0, 103.0, 104.0]
    momentum = market_analyzer._calculate_momentum(prices)
    assert -1 <= momentum <= 1

def test_calculate_liquidity_score(market_analyzer):
    volumes = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
    score = market_analyzer._calculate_liquidity_score(volumes)
    assert 0 <= score <= 1

def test_calculate_volatility(market_analyzer):
    prices = [100.0, 101.0, 102.0, 101.5, 103.0]
    volatility = market_analyzer._calculate_volatility(prices)
    assert volatility >= 0

def test_calculate_trend_strength(market_analyzer):
    prices = list(range(100, 151))
    strength = market_analyzer._calculate_trend_strength(prices)
    assert 0 <= strength <= 1

@pytest.mark.asyncio
async def test_error_handling(market_analyzer, mock_jupiter_client):
    mock_jupiter_client.get_price_history = AsyncMock(return_value=None)
    result = await market_analyzer.analyze_market("test_token")
    assert result is None

    mock_jupiter_client.get_market_depth = AsyncMock(return_value=None)
    result = await market_analyzer.analyze_market("test_token")
    assert result is None

@pytest.mark.asyncio
async def test_cache_functionality(market_analyzer, mock_jupiter_client):
    await market_analyzer._get_price_data("test_token")
    assert mock_jupiter_client.get_price_history.call_count == 1

    await market_analyzer._get_price_data("test_token")
    assert mock_jupiter_client.get_price_history.call_count == 1

    await market_analyzer._get_volume_data("test_token")
    assert mock_jupiter_client.get_market_depth.call_count == 1

    await market_analyzer._get_volume_data("test_token")
    assert mock_jupiter_client.get_market_depth.call_count == 1