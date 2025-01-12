"""
test_backtesting.py - Comprehensive tests for the backtesting engine
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
import numpy as np
from src.trading.backtesting import BacktestEngine, BacktestResult
from src.trading.market_analyzer import MarketAnalyzer
from src.trading.signals import SignalGenerator

# @pytest.fixture
# def mock_settings():
#     settings = Mock()
#     settings.INITIAL_CAPITAL = 1000.0
#     settings.MAX_POSITION_SIZE = 0.1
#     settings.STOP_LOSS_PERCENTAGE = 0.05
#     settings.TAKE_PROFIT_PERCENTAGE = 0.1
#     settings.MAX_POSITIONS = 3
#     settings.SIGNAL_THRESHOLD = 0.7
#     settings.MIN_LIQUIDITY = 1000.0
#     return settings

# @pytest.fixture
# def mock_jupiter_client():
#     client = AsyncMock()
    
#     # Mock price history data
#     price_history = []
#     base_price = 100.0
#     timestamp = datetime.now() - timedelta(days=7)
    
#     for i in range(168):  # 7 days * 24 hours
#         price_history.append({
#             'price': base_price * (1 + np.sin(i/24)/10),  # Add sine wave variation
#             'timestamp': int(timestamp.timestamp()),
#             'volume': 1000.0 * (1 + np.random.random()/5)
#         })
#         timestamp += timedelta(hours=1)
        
#     client.get_price_history.return_value = price_history
    
#     # Mock market depth
#     client.get_market_depth.return_value = {
#         'bids': [{'price': 100.0, 'size': 1000.0}],
#         'asks': [{'price': 101.0, 'size': 1000.0}],
#         'liquidity': 10000.0
#     }
    
#     # Mock price quotes
#     client.get_price.return_value = {'price': 100.0}
    
#     return client

# @pytest_asyncio.fixture
# async def backtest_engine(mock_settings, mock_jupiter_client):
#     market_analyzer = MarketAnalyzer(mock_jupiter_client, mock_settings)
#     signal_generator = SignalGenerator(mock_settings)
#     return BacktestEngine(mock_settings, market_analyzer, signal_generator, mock_jupiter_client)

@pytest.fixture
def mock_jupiter_client():
    client = AsyncMock()

    # Mock price history data
    price_history = []
    base_price = 100.0
    timestamp = datetime.now() - timedelta(days=7)

    for i in range(168):  # 7 days * 24 hours
        price_history.append({
            'price': base_price * (1 + np.sin(i/24)/10),  # Add sine wave variation
            'timestamp': int(timestamp.timestamp()),
            'volume': 1000.0 * (1 + np.random.random()/5)
        })
        timestamp += timedelta(hours=1)

    # Set up mock returns
    client.get_price_history.return_value = price_history
    client.get_tokens_list.return_value = [{
        'address': 'So11111111111111111111111111111111111111112',
        'symbol': 'SOL',
        'decimals': 9
    }]
    client.get_market_depth.return_value = {
        'bids': [{'price': 100.0, 'size': 1000.0}],
        'asks': [{'price': 101.0, 'size': 1000.0}],
        'liquidity': 10000.0
    }

    return client

@pytest.fixture
def mock_settings():
    settings = Mock()
    settings.INITIAL_CAPITAL = 1000.0
    settings.MAX_POSITION_SIZE = 0.1
    settings.STOP_LOSS_PERCENTAGE = 0.05
    settings.TAKE_PROFIT_PERCENTAGE = 0.1
    settings.MAX_POSITIONS = 3
    settings.SIGNAL_THRESHOLD = 0.7
    settings.MIN_LIQUIDITY = 1000.0
    return settings

@pytest.fixture
async def backtest_engine(mock_settings, mock_jupiter_client):
    market_analyzer = Mock()
    signal_generator = Mock()

    # Configure mock market analyzer
    market_analyzer.analyze_market.return_value = Mock(
        signal_strength=0.8,
        price_momentum=0.5,
        trend_strength=0.7,
        volatility=0.2
    )

    # Configure mock signal generator
    signal_generator.analyze_token.return_value = {
        'token_address': 'So11111111111111111111111111111111111111112',
        'price': 100.0,
        'timestamp': datetime.now(),
        'strength': 0.8,
        'type': 'long'
    }

    return BacktestEngine(mock_settings, market_analyzer, signal_generator, mock_jupiter_client)

@pytest.mark.asyncio
async def test_basic_backtest(backtest_engine):
    """Test basic backtest execution"""
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()
    
    parameters = {
        'initial_balance': 1000.0,
        'max_positions': 3
    }
    
    result = await backtest_engine.run_backtest(start_date, end_date, parameters)
    
    assert isinstance(result, BacktestResult)
    assert result.total_trades >= 0
    assert 0 <= result.win_rate <= 1
    assert result.profit_factor >= 0
    assert 0 <= result.max_drawdown <= 1
    assert isinstance(result.trades, list)
    assert isinstance(result.equity_curve, list)
    assert len(result.equity_curve) > 0

@pytest.mark.asyncio
async def test_parameter_optimization(backtest_engine):
    """Test parameter optimization"""
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()
    
    result = await backtest_engine.optimize_parameters(start_date, end_date)
    
    assert isinstance(result, dict)
    assert 'parameters' in result
    assert 'sharpe_ratio' in result
    assert 'total_return' in result
    assert 'max_drawdown' in result
    assert 'STOP_LOSS_PERCENTAGE' in result['parameters']
    assert 'TAKE_PROFIT_PERCENTAGE' in result['parameters']
    assert result['parameters'].get('STOP_LOSS_PERCENTAGE', 0) > 0
    assert result['parameters'].get('TAKE_PROFIT_PERCENTAGE', 0) > 0

@pytest.mark.asyncio
async def test_signal_generation(backtest_engine):
    """Test signal generation during backtest"""
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()
    timestamp = start_date
    
    price_data = await backtest_engine._get_historical_data(start_date, end_date)
    signals = await backtest_engine._generate_signals(timestamp, price_data)
    
    assert isinstance(signals, list)
    if signals:
        signal = signals[0]
        assert 'token_address' in signal
        assert 'price' in signal
        assert 'timestamp' in signal
        assert 'strength' in signal
        assert 'type' in signal

@pytest.mark.asyncio
async def test_position_management(backtest_engine):
    """Test position management in backtest"""
    start_date = datetime.now() - timedelta(days=1)
    timestamp = start_date
    token_address = "So11111111111111111111111111111111111111112"
    
    # Create test signal
    signal = {
        'token_address': token_address,
        'price': 100.0,
        'timestamp': timestamp,
        'strength': 0.8,
        'type': 'long'
    }
    
    # Execute entry
    current_balance = 1000.0
    await backtest_engine._execute_entry(signal, timestamp, current_balance)
    
    assert token_address in backtest_engine.positions
    position = backtest_engine.positions[token_address]
    assert position.entry_price == 100.0
    assert position.size > 0
    assert position.stop_loss < position.entry_price
    assert position.take_profit > position.entry_price

@pytest.mark.asyncio
async def test_metrics_calculation(backtest_engine):
    """Test trading metrics calculation"""
    # Add some test trades
    trades = [
        {'type': 'exit', 'pnl': 10.0},
        {'type': 'exit', 'pnl': -5.0},
        {'type': 'exit', 'pnl': 8.0},
        {'type': 'exit', 'pnl': 12.0},
        {'type': 'exit', 'pnl': -3.0}
    ]

    equity = [1000.0, 1010.0, 1005.0, 1013.0, 1025.0, 1022.0]
    daily_returns = np.diff(equity) / equity[:-1]

    metrics = backtest_engine._calculate_metrics(equity, daily_returns.tolist(), trades)

    assert isinstance(metrics, dict)
    assert 'win_rate' in metrics
    assert 'profit_factor' in metrics
    assert 'max_drawdown' in metrics
    assert 'sharpe_ratio' in metrics
    assert metrics['win_rate'] == pytest.approx(60.0)  # 3 winning trades out of 5

@pytest.mark.asyncio
async def test_market_condition_handling(backtest_engine):
    """Test handling of different market conditions"""
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()

    # Configure mock market analyzer with specific market conditions
    backtest_engine.market_analyzer.analyze_market.return_value = Mock(
        signal_strength=0.8,
        price_momentum=0.5,
        trend_strength=0.7,
        volatility=0.2,
        market_condition='uptrend'
    )

    # Configure mock signal generator with strong signals
    backtest_engine.signal_generator.analyze_token.return_value = {
        'token_address': 'So11111111111111111111111111111111111111112',
        'price': 100.0,
        'timestamp': datetime.now(),
        'strength': 0.8,
        'type': 'long',
        'market_condition': 'uptrend'
    }

    # Run backtest with the configured market conditions
    result = await backtest_engine.run_backtest(start_date, end_date, {
        'initial_balance': 1000.0,
        'max_positions': 3,
        'stop_loss': 0.05,
        'take_profit': 0.1
    })

    # Assert the backtest results reflect the market conditions
    assert isinstance(result, BacktestResult)
    assert result.total_trades >= 0
    assert result.win_rate >= 0
    assert result.profit_factor >= 0
    assert result.max_drawdown >= 0
    assert isinstance(result.trades, list)
    assert isinstance(result.equity_curve, list)
    assert len(result.equity_curve) > 0

@pytest.mark.asyncio
async def test_error_handling(backtest_engine, mock_jupiter_client):
    """Test error handling during backtest"""
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()
    
    # Test API failure
    mock_jupiter_client.get_price_history.side_effect = Exception("API Error")
    
    with pytest.raises(Exception):
        await backtest_engine.run_backtest(start_date, end_date, {
            'initial_balance': 1000.0,
            'max_positions': 3
        })

@pytest.mark.asyncio
async def test_historical_data_processing(backtest_engine):
    """Test historical data processing"""
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()

    # Create mock price history data
    price_history = []
    base_price = 100.0
    timestamp = start_date

    while timestamp <= end_date:
        price_history.append({
            'price': base_price * (1 + np.sin((timestamp - start_date).total_seconds()/86400)/10),
            'timestamp': int(timestamp.timestamp()),
            'volume': 1000.0
        })
        timestamp += timedelta(hours=1)

    # Configure mock Jupiter client
    backtest_engine.jupiter.get_price_history.return_value = price_history
    backtest_engine.jupiter.get_tokens_list.return_value = [{
        'address': 'So11111111111111111111111111111111111111112',
        'symbol': 'SOL',
        'decimals': 9
    }]

    # Get historical data
    historical_data = await backtest_engine._get_historical_data(start_date, end_date)

    # Verify the data structure
    assert isinstance(historical_data, dict)
    assert 'So11111111111111111111111111111111111111112' in historical_data

    token_data = historical_data['So11111111111111111111111111111111111111112']
    assert 'price_history' in token_data
    assert 'token_info' in token_data
    assert 'market_metrics' in token_data

    # Verify price history data
    assert len(token_data['price_history']) > 0
    assert all(isinstance(p['price'], (int, float)) for p in token_data['price_history'])
    assert all(isinstance(p['timestamp'], int) for p in token_data['price_history'])

@pytest.mark.asyncio
async def test_validation_checks(backtest_engine):
    """Test various validation checks"""
    token_address = "So11111111111111111111111111111111111111112"
    timestamp = datetime.now()
    
    # Test signal validation
    signal = {
        'token_address': token_address,
        'price': 100.0,
        'timestamp': timestamp,
        'strength': 0.8,
        'type': 'long'
    }
    
    is_valid = await backtest_engine._validate_entry(signal, 1000.0)
    assert isinstance(is_valid, bool)
    
    # Test position size validation
    result = await backtest_engine._validate_entry(signal, 0.0)  # Zero balance
    assert not result  # Should reject entry with zero balance

@pytest.mark.asyncio
async def test_performance_metrics(backtest_engine):
    """Test calculation of performance metrics"""
    # Simulate a complete backtest
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    result = await backtest_engine.run_backtest(start_date, end_date, {
        'initial_balance': 1000.0,
        'max_positions': 3
    })
    
    # Verify all required metrics are present
    required_metrics = [
        'total_trades',
        'win_rate',
        'profit_factor',
        'max_drawdown',
        'sharpe_ratio',
        'sortino_ratio',
        'total_return'
    ]
    
    for metric in required_metrics:
        assert hasattr(result, metric)
        assert getattr(result, metric) is not None