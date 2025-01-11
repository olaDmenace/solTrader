import pytest
import pytest_asyncio
from typing import AsyncGenerator, cast
from unittest.mock import AsyncMock, Mock, create_autospec
from src.trading.order_manager import OrderManager, OrderStatus, OrderType

class MockSettings:
    def __init__(self):
        self.SLIPPAGE_TOLERANCE = 0.01
        self.MAX_PRICE_IMPACT = 1.0
        self.MIN_LIQUIDITY = 1000.0
        self.MIN_VOLUME_24H = 100.0
        self.MAX_POSITION_SIZE = 5.0
        self.MAX_TRADES_PER_DAY = 10
        self.MAX_DAILY_LOSS = 1.0

@pytest.fixture(name="mock_wallet")
async def fixture_mock_wallet() -> AsyncGenerator[AsyncMock, None]:
    wallet = AsyncMock()
    wallet.get_balance = AsyncMock(return_value=10.0)
    yield wallet

@pytest.fixture(name="mock_swap_executor")
async def fixture_mock_swap_executor() -> AsyncGenerator[AsyncMock, None]:
    executor = AsyncMock()
    executor.execute_swap = AsyncMock(return_value=True)
    executor.get_market_depth = AsyncMock(return_value={
        'bids': [{'size': 10000.0, 'price': 1.0}],
        'asks': [{'size': 10000.0, 'price': 1.0}],
        'totalLiquidity': 20000.0
    })
    executor.get_token_info = AsyncMock(return_value={'mint': 'test_token'})
    yield executor

@pytest_asyncio.fixture(name="order_manager")
async def fixture_order_manager(
    mock_wallet: AsyncMock,
    mock_swap_executor: AsyncMock
) -> AsyncGenerator[OrderManager, None]:
    settings = MockSettings()
    manager = OrderManager(mock_swap_executor, mock_wallet, settings)
    yield manager

@pytest.mark.asyncio
async def test_submit_market_order_success(order_manager: OrderManager) -> None:
    order_id = await order_manager.submit_order(
        token_address='test_token',
        side='buy',
        size=1.0,
        price=1.0,
        order_type=OrderType.MARKET
    )
    
    assert order_id is not None
    assert order_id in order_manager.active_orders
    assert order_manager.active_orders[order_id].status == OrderStatus.CONFIRMED

@pytest.mark.asyncio
async def test_submit_order_insufficient_balance(
    order_manager: OrderManager,
    mock_wallet: AsyncMock
) -> None:
    mock_wallet.get_balance = AsyncMock(return_value=0.1)
    
    order_id = await order_manager.submit_order(
        token_address='test_token',
        side='buy',
        size=1.0,
        price=1.0
    )
    
    assert order_id is None

@pytest.mark.asyncio
async def test_cancel_order(order_manager: OrderManager) -> None:
    order_id = await order_manager.submit_order(
        token_address='test_token',
        side='buy',
        size=1.0,
        price=1.0
    )
    
    assert order_id is not None, "submit_order returned None, order could not be created"
    
    success = await order_manager.cancel_order(order_id)
    assert success
    assert order_id not in order_manager.active_orders
    assert len(order_manager.order_history) == 1
    assert order_manager.order_history[0].status == OrderStatus.CANCELLED

@pytest.mark.asyncio
async def test_get_order_status(order_manager: OrderManager) -> None:
    order_id = await order_manager.submit_order(
        token_address='test_token',
        side='buy',
        size=1.0,
        price=1.0
    )
    
    assert order_id is not None, "submit_order returned None, order could not be created"
    status = await order_manager.get_order_status(order_id)
    assert status == OrderStatus.CONFIRMED

@pytest.mark.asyncio
async def test_market_depth_validation(
    order_manager: OrderManager,
    mock_swap_executor: AsyncMock
) -> None:
    mock_swap_executor.get_market_depth = AsyncMock(return_value={
        'bids': [{'size': 100.0, 'price': 1.0}],
        'totalLiquidity': 100.0
    })
    
    order_id = await order_manager.submit_order(
        token_address='test_token',
        side='buy',
        size=10.0,
        price=1.0
    )
    
    assert order_id is None  # Order should fail due to insufficient liquidity