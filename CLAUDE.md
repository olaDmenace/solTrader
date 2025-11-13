# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
SolTrader is a sophisticated multi-strategy Solana trading bot that implements momentum trading, mean reversion, and other automated trading strategies. The bot is designed with a phased architecture that allows for incremental strategy deployment and testing.

## Development Commands

### Environment Setup
- **Create virtual environment**: `python -m venv venv`
- **Activate virtual environment**: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix)
- **Install dependencies**: `pip install -r requirements_updated.txt`
- **Verify setup**: `python verify_setup.py`

### Running the Bot
- **Start bot (interactive)**: `start_bot.bat` (Windows) - Provides menu for paper/live trading
- **Start bot (direct)**: `python main.py`
- **Run tests**: `python test_ape_strategy.py`
- **Run web dashboard**: `run_web_dashboard.bat`

### Testing
- **Run pytest suite**: `pytest` (uses pytest.ini config)
- **Test individual components**: `python test_enhanced_bot.py`
- **Integration tests**: `python test_live_trading_integration.py`

### Type Checking
- **Pyright**: Uses basic type checking mode (see pyrightconfig.json)
- Type checking is configured to be lenient due to dynamic Solana API usage

## Architecture

### Core Components

#### Main Entry Point (`main.py`)
- **TradingBot class**: Central coordinator for all components
- **Phased strategy loading**: Momentum (60%) + Mean Reversion (40%) allocation
- **Component lifecycle management**: Startup, monitoring, graceful shutdown
- **Health monitoring integration**: Automated system health checks

#### Trading Strategies (`src/trading/`)
- **TradingStrategy**: Primary momentum-based trading logic
- **MeanReversionStrategy**: Counter-trend trading using RSI and technical indicators
- **Position**: Token position tracking with proper token balance management
- **SwapExecutor**: Jupiter DEX integration with comprehensive error handling

#### API Integrations (`src/api/`)
- **JupiterClient**: Jupiter DEX integration for swaps
- **AlchemyClient**: Solana RPC provider (primary)
- **SolanaTrackerClient**: Token analytics and momentum data
- **Multi-RPC fallback**: Helius, QuickNode, and other RPC providers

#### Enhanced Systems (`src/`)
- **EnhancedTokenScanner**: Advanced token discovery with age filtering
- **PerformanceAnalytics**: Trade tracking and portfolio analytics
- **EmailNotificationSystem**: Critical alerts and daily reports
- **UnifiedWebDashboard**: Real-time monitoring interface
- **PortfolioManager**: Dynamic capital allocation across strategies

### Configuration System

#### Settings (`src/config/settings.py`)
All configuration is centralized in the Settings dataclass with these key categories:
- **Paper Trading**: Safe testing environment with simulated balance
- **Position Management**: Risk controls and position sizing
- **Token Filtering**: Liquidity, age, and momentum thresholds
- **Strategy Parameters**: RSI levels, momentum thresholds, exit conditions

#### Environment Variables (`.env`)
Required configuration:
```bash
ALCHEMY_RPC_URL=https://solana-mainnet.g.alchemy.com/v2/YOUR_KEY
WALLET_ADDRESS=your_solana_wallet_address
PAPER_TRADING=true  # Start with paper trading
```

### Key Architecture Patterns

#### Strategy Coordination
The bot implements a proven dual-strategy approach:
- **Momentum Strategy (60% allocation)**: Captures trending tokens with high momentum
- **Mean Reversion Strategy (40% allocation)**: Profits from oversold/overbought conditions
- **Portfolio-level risk management**: Cross-strategy position correlation and limits

#### Error Handling & Recovery
- **Jupiter API quota management**: Rate limiting and fallback mechanisms
- **RPC provider fallback**: Multiple Solana RPC endpoints with health monitoring
- **Emergency controls**: Automatic trading suspension on critical errors
- **Position tracking**: Proper token balance management to prevent exit calculation errors

#### Critical Bug Fixes (September 2025)
The bot has undergone major architectural improvements:
- **Fixed token balance tracking**: Positions now store actual token amounts received
- **Enhanced swap execution**: Proper output amount tracking and error handling
- **Emergency control validation**: 99.99% loss triggers work as designed
- **Multi-strategy coordination**: No conflicts between trading strategies

## Development Workflow

### Phase-Based Development
The bot follows a disciplined phase-based approach:

1. **Phase 1 (Complete)**: Foundation - Single momentum strategy
2. **Phase 2 (Complete)**: Multi-strategy - Added mean reversion
3. **Phase 3 (Ready)**: Advanced systems - Grid trading, enhanced coordination
4. **Phase 4 (Planned)**: Production optimization

### Testing Strategy
- **Always start with paper trading**: `PAPER_TRADING=true` in .env
- **Validate each component**: Use dedicated test files for isolated testing
- **Integration testing**: Run full system tests before deployment
- **Performance monitoring**: Use dashboard and logs to validate improvements

### Adding New Features

#### Before Adding New Strategies:
1. **Baseline measurement**: Document current performance metrics
2. **Isolation testing**: Test new strategy independently
3. **Integration testing**: Ensure no conflicts with existing strategies
4. **Validation criteria**: Must maintain or improve overall profitability

#### Code Style Guidelines:
- **Follow existing patterns**: Mirror the architecture of existing strategies
- **Comprehensive logging**: All major operations should log status and errors
- **Error handling**: Always handle API failures gracefully with fallbacks
- **Type hints**: Use type annotations for better IDE support

### Security Considerations
- **Never commit credentials**: Use .env files for sensitive data
- **Paper trading first**: Always test with simulated trades
- **Position size limits**: Enforce maximum position sizes to limit risk
- **Emergency controls**: Implement circuit breakers for abnormal conditions

## Common Development Tasks

### Adding a New Trading Strategy
1. Create strategy class in `src/trading/` following existing patterns
2. Integrate with `PortfolioManager` in `main.py`
3. Add configuration parameters to `settings.py`
4. Create isolated tests for the strategy
5. Update allocation percentages in strategy coordinator

### Debugging Trading Issues
1. Check logs in `logs/trading.log`
2. Verify API connections with test scripts
3. Use paper trading mode to isolate issues
4. Monitor dashboard at `http://localhost:5000` for real-time data
5. Validate wallet balance and transaction history

### Performance Optimization
1. Monitor Jupiter API quota usage
2. Optimize token scanning frequency
3. Review position monitoring intervals
4. Analyze trade execution times in dashboard
5. Use performance analytics for strategy tuning

## Repository Structure
- **`main.py`**: Bot entry point and coordination
- **`src/trading/`**: Core trading logic and strategies
- **`src/api/`**: External API integrations
- **`src/config/`**: Configuration management
- **`src/dashboard/`**: Web monitoring interface
- **`src/portfolio/`**: Portfolio and risk management
- **`logs/`**: Runtime logs and trade history
- **`data/`**: Cached data and position tracking
- **Documentation**: Comprehensive guides in markdown files

The bot is production-ready with critical bugs fixed and proven performance in both paper and live trading scenarios.
- never add unicode characters to logs/log outputs in projects written in Python