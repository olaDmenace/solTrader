import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock
from src.trading.enhanced_risk import EnhancedRiskManager

class MockSettings:
    def __init__(self):
        self.MAX_POSITION_SIZE = 5.0
        self.MAX_PORTFOLIO_RISK = 10.0
        self.MAX_DRAWDOWN = 20.0
        self.MAX_POSITIONS = 3
        self.MIN_LIQUIDITY = 1000.0
        self.MIN_VOLUME_24H = 100.0
        self.SLIPPAGE_TOLERANCE = 0.01
        self.MAX_TRADES_PER_DAY = 10
        self.MAX_DAILY_LOSS = 1.0

@pytest_asyncio.fixture
async def risk_manager():
    settings = MockSettings()
    return EnhancedRiskManager(settings)

@pytest.mark.asyncio
async def test_evaluate_trade_risk(risk_manager):
    # Test basic risk evaluation
    is_acceptable, metrics = await risk_manager.evaluate_trade_risk(
        token_address="test_token",
        position_size=1.0,
        current_price=100.0
    )
    
    assert isinstance(is_acceptable, bool)
    assert isinstance(metrics, dict)
    assert 'position_risk' in metrics
    assert 'total_risk' in metrics

@pytest.mark.asyncio
async def test_portfolio_impact(risk_manager):
    # Add a position and test impact
    risk_manager.update_position("token1", 1.0, 100.0)
    impact = await risk_manager._calculate_portfolio_impact("token2", 0.5)
    assert isinstance(impact, float)
    assert impact >= 0.0

@pytest.mark.asyncio
async def test_combine_risk_factors(risk_manager):
    total_risk = risk_manager._combine_risk_factors(
        position_risk=10.0,
        portfolio_impact=5.0,
        correlation_risk=0.5,
        volatility_risk=15.0
    )
    assert isinstance(total_risk, float)
    assert 0 <= total_risk <= 100.0

@pytest.mark.asyncio
async def test_risk_thresholds(risk_manager):
    # Test risk evaluation thresholds
    risk_manager.update_position("token1", 1.0, 100.0)
    profile = risk_manager._get_risk_profile()
    
    assert hasattr(profile, 'max_position_size')
    assert hasattr(profile, 'max_portfolio_risk')
    
    is_acceptable = risk_manager._evaluate_risk_thresholds(
        total_risk=5.0,
        profile=profile
    )
    assert isinstance(is_acceptable, bool)

@pytest.mark.asyncio
async def test_position_updates(risk_manager):
    # Test position tracking
    risk_manager.update_position("test_token", 1.0, 100.0)
    assert "test_token" in risk_manager.positions
    
    risk_manager.remove_position("test_token")
    assert "test_token" not in risk_manager.positions

@pytest.mark.asyncio
async def test_drawdown_calculation(risk_manager):
    # Test drawdown tracking
    risk_manager.update_metrics(1000.0)  # Initial value
    risk_manager.update_metrics(900.0)   # 10% drawdown
    
    current_dd = risk_manager._get_current_drawdown()
    assert isinstance(current_dd, float)
    assert current_dd >= 0.0

@pytest.mark.asyncio
async def test_risk_metrics(risk_manager):
    # Test risk metrics calculation
    metrics = risk_manager.get_risk_metrics()
    
    assert isinstance(metrics, dict)
    assert 'current_risk' in metrics
    assert 'max_drawdown' in metrics
    assert isinstance(metrics['current_risk'], float)