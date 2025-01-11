import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock, create_autospec
from telegram import Update
from telegram.ext import Application, CallbackContext

from src.telegram_bot import TradingBot
from src.trading.strategy import TradingStrategy
from src.config.settings import Settings

class MockTradingStrategy:
    async def start_trading(self): pass
    async def stop_trading(self): pass
    async def get_metrics(self): pass
    is_trading: bool = False

@pytest.fixture
def mock_settings():
    settings = Mock(spec=Settings)
    settings.TELEGRAM_BOT_TOKEN = "test_token"
    settings.TELEGRAM_CHAT_ID = "123456"
    settings.MAX_POSITIONS = 3
    settings.MAX_TRADE_SIZE = 1.0
    settings.STOP_LOSS_PERCENTAGE = 0.05
    settings.MAX_POSITION_SIZE = 5.0
    return settings

@pytest.fixture
async def mock_strategy(mock_settings):
    strategy = create_autospec(MockTradingStrategy, instance=True)
    strategy.is_trading = False
    strategy.settings = mock_settings
    strategy.position_manager = Mock()
    strategy.position_manager.positions = {}

    # Setup metrics
    metrics = {
        'trading_status': {
            'is_trading': False,
            'open_positions': 0,
            'pending_orders': 0
        },
        'performance': {
            'total_trades': 0,
            'pnl': 0.0
        }
    }

    # Setup async methods
    strategy.start_trading = AsyncMock(return_value=True)
    strategy.stop_trading = AsyncMock(return_value=True)
    strategy.get_metrics = AsyncMock(return_value=metrics)

    return strategy

@pytest_asyncio.fixture
async def mock_application():
    app = AsyncMock(spec=Application)
    app.bot = AsyncMock()
    app.bot.send_message = AsyncMock()
    return app

@pytest_asyncio.fixture
async def bot(mock_application, mock_strategy, mock_settings):
    return TradingBot(mock_application, mock_strategy, mock_settings)

@pytest.mark.asyncio
async def test_start_command(bot):
    update = Mock(spec=Update)
    update.message = AsyncMock()
    update.message.reply_text = AsyncMock()
    context = Mock(spec=CallbackContext)
    
    await bot.start_command(update, context)
    
    assert update.message.reply_text.called
    response = update.message.reply_text.call_args[0][0]
    assert "Trading Bot Commands" in response
    assert "/start_trading" in response
    assert "/stop_trading" in response
    assert "/status" in response

@pytest.mark.asyncio
async def test_start_trading_command(bot):
    update = Mock(spec=Update)
    update.message = AsyncMock()
    update.message.reply_text = AsyncMock()
    context = Mock(spec=CallbackContext)
    
    bot.strategy.is_trading = False
    await bot.start_trading_command(update, context)
    
    assert bot.strategy.start_trading.called
    assert "🚀 Trading started!" in update.message.reply_text.call_args[0][0]
    
    # Test when trading is already active
    bot.strategy.is_trading = True
    await bot.start_trading_command(update, context)
    assert "already active" in update.message.reply_text.call_args[0][0].lower()

@pytest.mark.asyncio
async def test_stop_trading_command(bot):
    update = Mock(spec=Update)
    update.message = AsyncMock()
    update.message.reply_text = AsyncMock()
    context = Mock(spec=CallbackContext)
    
    bot.strategy.is_trading = True
    await bot.stop_trading_command(update, context)
    
    assert bot.strategy.stop_trading.called
    assert "Trading stopped" in update.message.reply_text.call_args[0][0]
    
    bot.strategy.is_trading = False
    await bot.stop_trading_command(update, context)
    assert "already stopped" in update.message.reply_text.call_args[0][0].lower()

@pytest.mark.asyncio
async def test_status_command(bot):
    update = Mock(spec=Update)
    update.message = AsyncMock()
    update.message.reply_text = AsyncMock()
    context = Mock(spec=CallbackContext)
    
    # Set up metrics that will be returned
    metrics = {
        'trading_status': {
            'is_trading': False,
            'open_positions': 0,
            'pending_orders': 0
        }
    }
    
    # Mock the get_metrics method
    async def mock_get_metrics():
        return metrics
    
    bot.strategy.get_metrics = mock_get_metrics
    
    await bot.status_command(update, context)
    
    # Verify the response
    assert update.message.reply_text.called
    response = update.message.reply_text.call_args[0][0]
    assert "🤖 Trading Status" in response

@pytest.mark.asyncio
async def test_positions_command_no_positions(bot):
    update = Mock(spec=Update)
    update.message = AsyncMock()
    update.message.reply_text = AsyncMock()
    context = Mock(spec=CallbackContext)
    
    bot.strategy.position_manager.positions = {}
    await bot.positions_command(update, context)
    
    assert "No active positions" in update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_positions_command_with_positions(bot):
    update = Mock(spec=Update)
    update.message = AsyncMock()
    update.message.reply_text = AsyncMock()
    context = Mock(spec=CallbackContext)
    
    # Mock active position
    position = Mock()
    position.entry_price = 1.0
    position.current_price = 1.1
    position.unrealized_pnl = 0.1
    position.size = 1.0
    position.status = 'open'
    position.token_address = 'test_token'  # Add token_address attribute
    
    bot.strategy.position_manager.positions = {position.token_address: position}
    await bot.positions_command(update, context)
    
    response = update.message.reply_text.call_args[0][0]
    # Just check for parts of address to account for truncation
    assert "test" in response
    assert "$1.0000" in response  # Entry price
    assert "$1.1000" in response  # Current price
    assert "10.00%" in response   # PnL percentage