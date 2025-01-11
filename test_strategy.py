import pytest
from typing import Any, AsyncIterator
from unittest.mock import Mock, AsyncMock
from datetime import datetime
import asyncio

from src.trading.strategy import TradingStrategy, EntrySignal
from src.trading.position import Position, TradeEntry

class MockSettings:
    def __init__(self):
        self.MIN_BALANCE = 1.0
        self.MAX_POSITIONS = 3
        self.SCAN_INTERVAL = 1
        self.STOP_LOSS_PERCENTAGE = 0.05
        self.TAKE_PROFIT_PERCENTAGE = 0.1
        self.MAX_TRADE_SIZE = 0.1
        self.MAX_POSITION_SIZE = 1.0
        self.SLIPPAGE_TOLERANCE = 0.01
        self.CLOSE_POSITIONS_ON_STOP = True
        self.MAX_PRICE_IMPACT = 1.0
        self.MIN_LIQUIDITY = 1000.0
        self.MAX_SLIPPAGE = 0.01
        self.VOLUME_THRESHOLD = 100.0
        self.MAX_DAILY_LOSS = 1.0
        self.MAX_DRAWDOWN = 5.0
        self.MAX_TRADES_PER_DAY = 10
        self.ERROR_THRESHOLD = 5
        self.MIN_VOLUME_24H = 100.0
        self.SIGNAL_THRESHOLD = 0.7
        self.MAX_PORTFOLIO_RISK = 5.0
        self.PORTFOLIO_VALUE = 100.0
        self.MIN_TRADE_SIZE = 0.1
        self.INITIAL_CAPITAL = 100.0

@pytest.fixture
def mock_settings() -> MockSettings:
    return MockSettings()

@pytest.fixture
def mock_wallet() -> AsyncMock:
    wallet = AsyncMock()
    wallet.get_balance = AsyncMock(return_value=10.0)
    return wallet

@pytest.fixture
def mock_jupiter() -> AsyncMock:
    jupiter = AsyncMock()
    jupiter.get_price = AsyncMock(return_value={"price": 1.0})
    return jupiter

@pytest.fixture
def mock_scanner() -> AsyncMock:
    scanner = AsyncMock()
    scanner.scan_new_listings = AsyncMock(return_value=[])
    return scanner

@pytest.fixture
def mock_position() -> Position:
    return Position(
        token_address="test_token",
        entry_price=1.0,
        current_price=1.0,
        size=0.1,
        stop_loss=0.95,
        take_profit=1.1,
        unrealized_pnl=0.0,
        status='open',
        trade_entry=TradeEntry(
            token_address="test_token",
            entry_price=1.0,
            entry_time=datetime.now(),
            size=0.1
        )
    )

@pytest.fixture
async def strategy(mock_jupiter: AsyncMock, mock_wallet: AsyncMock, mock_settings: MockSettings, mock_scanner: AsyncMock) -> AsyncIterator[TradingStrategy]:
    strategy = TradingStrategy(
        jupiter_client=mock_jupiter,
        wallet=mock_wallet,
        settings=mock_settings,
        scanner=mock_scanner
    )
    
    # Setup mocks with proper return values
    strategy.swap_executor = AsyncMock()
    strategy.swap_executor.execute_swap.return_value = True
    
    strategy.position_manager = AsyncMock()
    strategy.position_manager.positions = {}
    strategy.position_manager.get_open_positions = AsyncMock(return_value={})
    strategy.position_manager.close_position = AsyncMock(return_value=True)
    
    strategy.risk_manager = Mock()
    strategy.performance_monitor = AsyncMock()
    strategy.performance_monitor.get_performance_summary = AsyncMock(return_value={'total_trades': 0, 'pnl': 0})
    
    strategy.alert_system = AsyncMock()
    strategy.signal_generator = AsyncMock()
    
    # Initialize daily stats
    strategy.daily_stats = {
        'trade_count': 0,
        'error_count': 0,
        'total_pnl': 0.0,
        'last_reset': datetime.now()
    }
    
    yield strategy

@pytest.mark.asyncio
async def test_start_trading_with_insufficient_balance(strategy: TradingStrategy, mock_wallet: AsyncMock) -> None:
    mock_wallet.get_balance = AsyncMock(return_value=0.5)  # Below MIN_BALANCE
    strategy.wallet = mock_wallet
    
    await strategy.start_trading()
    
    assert not strategy.is_trading
    assert strategy._monitor_task is None
    assert "Insufficient balance" in str(strategy.last_error)

@pytest.mark.asyncio
async def test_start_trading_success(strategy: TradingStrategy, mock_wallet: AsyncMock) -> None:
    mock_wallet.get_balance = AsyncMock(return_value=10.0)  # Above MIN_BALANCE
    strategy.wallet = mock_wallet
    strategy._trading_loop = AsyncMock()
    
    success = await strategy.start_trading()
    
    assert success
    assert strategy.is_trading
    assert strategy._monitor_task is not None
    assert strategy.last_error is None

@pytest.mark.asyncio
async def test_stop_trading(strategy: TradingStrategy) -> None:
    strategy.is_trading = True
    strategy._monitor_task = asyncio.create_task(asyncio.sleep(1))
    
    success = await strategy.stop_trading()
    
    assert success
    assert not strategy.is_trading
    assert strategy._monitor_task.cancelled()

@pytest.mark.asyncio
async def test_execute_entry(strategy: TradingStrategy, mock_position: Position) -> None:
    signal = EntrySignal(
        token_address="test_token",
        price=1.0,
        confidence=0.8,
        entry_type="standard",
        size=0.1,
        stop_loss=0.95,
        take_profit=1.1
    )
    
    strategy._get_current_price = AsyncMock(return_value=1.0)
    strategy.swap_executor.execute_swap.return_value = True
    strategy.position_manager.open_position.return_value = mock_position
    
    success = await strategy.execute_entry(signal)
    assert success

@pytest.mark.asyncio
async def test_get_metrics(strategy: TradingStrategy) -> None:
    metrics = await strategy._get_metrics()
    
    assert isinstance(metrics, dict)
    assert 'performance' in metrics
    assert 'risk' in metrics
    assert 'trading_status' in metrics
    assert isinstance(metrics['trading_status']['is_trading'], bool)

@pytest.mark.asyncio
async def test_max_positions_limit(strategy: TradingStrategy, mock_position: Position) -> None:
    # Setup multiple positions
    strategy.position_manager.positions = {
        'token1': mock_position,
        'token2': mock_position,
        'token3': mock_position
    }
    
    signal = EntrySignal(
        token_address="test_token",
        price=1.0,
        confidence=0.8,
        entry_type="standard",
        size=0.1,
        stop_loss=0.95,
        take_profit=1.1
    )
    
    # Should fail as we've hit max positions (3)
    success = await strategy.execute_entry(signal)
    assert not success

@pytest.mark.asyncio
async def test_daily_loss_limit(strategy: TradingStrategy) -> None:
    # Set daily loss
    strategy.daily_stats['total_pnl'] = -strategy.settings.MAX_DAILY_LOSS
    
    signal = EntrySignal(
        token_address="test_token",
        price=1.0,
        confidence=0.8,
        entry_type="standard",
        size=0.1,
        stop_loss=0.95,
        take_profit=1.1
    )
    
    # Should fail as we've hit daily loss limit
    success = await strategy.execute_entry(signal)
    assert not success

@pytest.mark.asyncio
async def test_stop_loss_trigger(strategy: TradingStrategy, mock_position: Position) -> None:
    position_addr = "test_token"
    strategy.position_manager.positions = {position_addr: mock_position}
    strategy.position_manager.close_position = AsyncMock(return_value=True)
    strategy._get_current_price = AsyncMock(return_value=mock_position.stop_loss * 0.99)

    # Monitor positions should trigger stop loss
    await strategy._monitor_positions()

    assert strategy.position_manager.close_position.called
    assert strategy.position_manager.close_position.call_args[0][0] == position_addr
    assert strategy.position_manager.close_position.call_args[0][1] == "stop_loss"

@pytest.mark.asyncio
async def test_take_profit_execution(strategy: TradingStrategy, mock_position: Position) -> None:
    position_addr = "test_token"
    strategy.position_manager.positions = {position_addr: mock_position}
    strategy.position_manager.close_position = AsyncMock(return_value=True)
    strategy._get_current_price = AsyncMock(return_value=mock_position.take_profit * 1.01)

    # Monitor positions should trigger take profit
    await strategy._monitor_positions()

    assert strategy.position_manager.close_position.called
    assert strategy.position_manager.close_position.call_args[0][0] == position_addr
    assert strategy.position_manager.close_position.call_args[0][1] == "take_profit"