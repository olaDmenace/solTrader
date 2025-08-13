# 📚 SolTrader Bot Complete .env Configuration Documentation

## 🎯 Overview

This comprehensive guide documents every setting in your SolTrader bot's `.env` configuration file. Understanding these settings is crucial for optimizing your bot's performance, transitioning from paper to live trading, and achieving your target trading strategy.

## 📋 Table of Contents

1. [🔑 API Keys and Authentication](#-api-keys-and-authentication)
2. [🎮 Trading Mode and General Settings](#-trading-mode-and-general-settings)
3. [📊 Paper Trading Settings](#-paper-trading-settings)
4. [⚡ Risk Management Settings](#-risk-management-settings)
5. [🔍 Scanner and Market Analysis Settings](#-scanner-and-market-analysis-settings)
6. [⚖️ Arbitrage Settings](#-arbitrage-settings)
7. [🤖 Machine Learning Settings](#-machine-learning-settings)
8. [📈 API Rate Limits](#-api-rate-limits)
9. [📧 Notification Settings](#-notification-settings)
10. [🔴 Live Trading Configuration](#-live-trading-configuration)
11. [⛽ Gas and Priority Fee Settings](#-gas-and-priority-fee-settings)
12. [🛡️ Live Trading Safety Settings](#-live-trading-safety-settings)
13. [📋 Transaction Management](#-transaction-management)
14. [🚨 Emergency Controls](#-emergency-controls)
15. [⚙️ Enhanced Filter Parameters](#-enhanced-filter-parameters)

---

## 🔑 API Keys and Authentication

### `ALCHEMY_RPC_URL` **(REQUIRED)**
- **Purpose**: Solana blockchain RPC endpoint for network communication
- **Format**: `https://solana-mainnet.g.alchemy.com/v2/YOUR_API_KEY`
- **Impact**: Core blockchain connectivity - without this, the bot cannot function
- **Performance**: Higher tier accounts get better rate limits and faster responses
- **Security**: Keep secure, moderate sensitivity

### `ALCHEMY_API_KEY` **(REQUIRED)**
- **Purpose**: API key extracted from the RPC URL for authentication
- **Format**: Alphanumeric string (32+ characters)
- **Impact**: Must match the key in `ALCHEMY_RPC_URL`
- **Security**: Keep secure

### `JUPITER_API_KEY`
- **Purpose**: Enhanced rate limits for Jupiter DEX API (price/quote data)
- **Default**: Public endpoint if not provided
- **Format**: URL or API key string
- **Impact**: Better performance with dedicated key
- **Current**: `https://api.jup.ag/price/v2`

### `WALLET_ADDRESS` **(REQUIRED)**
- **Purpose**: Your Solana wallet public address for monitoring balances
- **Format**: Base58 Solana address (32-44 characters)
- **Impact**: Used for balance checking and transaction monitoring
- **Security**: Public address, safe to share

### `PRIVATE_KEY` ⚠️ **HIGHLY SENSITIVE**
- **Purpose**: Private key for live trading execution
- **Usage**: Only required for live trading mode
- **Security**: EXTREMELY SENSITIVE - compromised key = stolen funds
- **Storage**: Encrypt in production, use hardware wallet when possible
- **Format**: Base58 encoded private key

---

## 🎮 Trading Mode and General Settings

### `TRADING_MODE`
- **Options**: `paper`, `live`, `backtest`
- **Default**: `paper`
- **Impact**: 
  - `paper`: Simulated trading with fake balance
  - `live`: Real trading with actual funds ⚠️
  - `backtest`: Historical data testing
- **Recommendation**: Always start with `paper`

### `DEBUG_MODE`
- **Options**: `true`, `false`
- **Default**: `false`
- **Impact**: Enables verbose logging for troubleshooting
- **Performance**: Reduces performance slightly due to extra logging
- **Usage**: Enable when debugging issues

### `LOG_LEVEL`
- **Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- **Default**: `INFO`
- **Impact**: Controls verbosity of log output
- **Performance**: `DEBUG` creates large log files
- **Production**: Use `INFO` or `WARNING` for live trading

### Path Settings
- `LOG_PATH`: Directory for log files (default: `./logs`)
- `DATA_PATH`: Directory for data storage (default: `./data`)
- `MODEL_PATH`: Directory for ML models (default: `./models`)
- `RESULTS_PATH`: Directory for results (default: `./results`)

---

## 📊 Paper Trading Settings

### `PAPER_TRADING` 
- **Options**: `true`, `false`
- **Default**: `true`
- **Impact**: Enables/disables paper trading simulation
- **Safety**: Keep `true` until confident in strategy

### `INITIAL_PAPER_BALANCE`
- **Purpose**: Starting balance for paper trading (in SOL)
- **Default**: `100.0`
- **Range**: Any positive number
- **Impact**: Larger balance allows more simultaneous positions
- **Recommendation**: Start with 100-500 SOL to test strategy thoroughly

### Paper Trading Specific Parameters

#### `PAPER_MIN_MOMENTUM_THRESHOLD`
- **Purpose**: Minimum momentum % required for paper trades
- **Default**: `3.0` (3%)
- **Range**: 0.0-50.0%
- **Impact**: Lower = more opportunities, higher = higher quality signals
- **Current Setting**: `3.0%` (very permissive for maximum opportunities)

#### `PAPER_MIN_LIQUIDITY`
- **Purpose**: Minimum liquidity in SOL for paper trading
- **Default**: `50.0`
- **Range**: 1.0-10,000.0 SOL
- **Impact**: Lower = more tokens qualify, higher = more liquid markets
- **Current Setting**: `50.0 SOL` (ultra-permissive)

#### `PAPER_TRADING_SLIPPAGE`
- **Purpose**: Maximum acceptable slippage for paper trades
- **Default**: `0.50` (50%)
- **Range**: 0.01-1.0 (1%-100%)
- **Impact**: Higher slippage = more trades execute in volatile markets
- **Current Setting**: `50%` (very high for meme tokens)

#### `PAPER_BASE_POSITION_SIZE`
- **Purpose**: Base position size as fraction of balance
- **Default**: `0.1` (10%)
- **Range**: 0.01-0.5 (1%-50%)
- **Impact**: Larger positions = higher risk/reward
- **Current Setting**: `10%` (moderate risk)

#### `PAPER_MAX_POSITION_SIZE`
- **Purpose**: Maximum position size as fraction of balance
- **Default**: `0.5` (50%)
- **Range**: 0.05-1.0 (5%-100%)
- **Impact**: Caps position size for risk management
- **Current Setting**: `50%` (aggressive)

#### `PAPER_SIGNAL_THRESHOLD`
- **Purpose**: Minimum signal strength for paper trade execution
- **Default**: `0.3`
- **Range**: 0.1-1.0
- **Impact**: Lower = more trades, higher = higher quality trades
- **Current Setting**: `0.3` (permissive)

---

## ⚡ Risk Management Settings

### Position Sizing
#### `MAX_POSITION_SIZE`
- **Purpose**: Maximum position size as fraction of balance
- **Default**: `0.2` (20%)
- **Range**: 0.01-1.0 (1%-100%)
- **Impact**: Controls maximum risk per trade
- **Live Trading**: Should be much lower (2-5%)

#### `TRADING_MAX_SIZE`
- **Purpose**: Maximum SOL per trade as percentage of balance
- **Default**: `0.02` (2%)
- **Range**: 0.005-0.1 (0.5%-10%)
- **Impact**: Risk per trade - critical for capital preservation
- **Live Trading**: 1-3% recommended

### Stop Loss and Take Profit
#### `STOP_LOSS_PERCENTAGE`
- **Purpose**: Automatic exit when loss reaches this percentage
- **Default**: `0.20` (20%)
- **Range**: 0.05-0.5 (5%-50%)
- **Impact**: Larger = more patience, smaller = quicker exits
- **Current**: `15%` in settings.py (reasonable for volatile tokens)

#### `TAKE_PROFIT_PERCENTAGE`
- **Purpose**: Automatic exit when profit reaches this percentage
- **Default**: `0.30` (30%)
- **Range**: 0.1-2.0 (10%-200%)
- **Impact**: Larger = let winners run, smaller = quick profits
- **Current**: `50%` in settings.py (let winners run strategy)

### Portfolio Risk
#### `MAX_DRAWDOWN`
- **Purpose**: Maximum portfolio decline before position reduction
- **Default**: `5.0` (5%)
- **Range**: 1.0-20.0%
- **Impact**: Portfolio-level circuit breaker
- **Current**: `10%` in settings.py (higher risk tolerance)

#### `MAX_PORTFOLIO_RISK`
- **Purpose**: Maximum portfolio risk percentage
- **Default**: `5.0` (5%)
- **Range**: 1.0-25.0%
- **Impact**: Overall portfolio risk limit
- **Current**: `10%` in settings.py (aggressive)

#### `MAX_VOLATILITY`
- **Purpose**: Maximum annualized volatility tolerance
- **Default**: `0.3` (30%)
- **Range**: 0.1-1.0 (10%-100%)
- **Impact**: Higher volatility = more opportunities but more risk
- **Current**: `80%` in settings.py (very high for meme tokens)

### Position Limits
#### `MAX_POSITIONS`
- **Purpose**: Maximum number of open positions
- **Default**: `3`
- **Range**: 1-20
- **Impact**: More positions = more diversification but complex management
- **Current**: `3` (focused approach)

#### `MAX_SIMULTANEOUS_POSITIONS`
- **Purpose**: Maximum concurrent positions (should match MAX_POSITIONS)
- **Default**: `5`
- **Impact**: Must align with MAX_POSITIONS
- **Current**: Inconsistent (3 vs 5) - should be aligned

#### `MAX_TRADES_PER_DAY`
- **Purpose**: Daily trade limit for risk control
- **Default**: `10`
- **Range**: 1-100
- **Impact**: Higher = more opportunities, lower = more selective
- **Current**: `20` in settings.py (active trading)

---

## 🔍 Scanner and Market Analysis Settings

### Scan Timing
#### `SCANNER_INTERVAL`
- **Purpose**: Seconds between token discovery scans
- **Default**: `900` (15 minutes)
- **Range**: 60-3600 (1 minute - 1 hour)
- **Impact**: Faster = more opportunities but more API usage
- **API Impact**: Critical for quota management
- **Current**: `900s` (quota-conscious setting)

### Token Quality Filters
#### `MIN_LIQUIDITY`
- **Purpose**: Minimum liquidity in SOL for token selection
- **Default**: `1000.0`
- **Range**: 10.0-100,000.0 SOL
- **Impact**: Higher = safer tokens, lower = more opportunities
- **Current**: `100.0` SOL (very permissive for new tokens)
- **Trading Impact**: Low liquidity = higher slippage

#### `MIN_VOLUME_24H`
- **Purpose**: Minimum 24h volume in SOL for selection
- **Default**: `100.0`
- **Range**: 1.0-10,000.0 SOL
- **Impact**: Higher = more active markets, lower = earlier entry
- **Current**: `50.0` SOL (early entry focus)

### Token Filtering Parameters
#### `MAX_TOKEN_AGE_HOURS`
- **Purpose**: Maximum age of tokens to consider (hours)
- **Default**: Not in .env
- **Range**: 1-168 hours (1 hour - 1 week)
- **Impact**: Lower = newer tokens only, higher = more mature tokens
- **Current**: `24 hours` in scanner (focus on new launches)

#### `NEW_TOKEN_MAX_AGE_MINUTES`
- **Purpose**: Consider tokens as "new" if under this age
- **Default**: `2880` (48 hours)
- **Range**: 60-10080 minutes (1 hour - 1 week)
- **Impact**: Defines "new token" category for special handling

### Price and Market Cap Filters
#### `MAX_TOKEN_PRICE_SOL`
- **Purpose**: Maximum token price in SOL
- **Default**: `0.00000000001`
- **Impact**: Focuses on micro-cap tokens
- **Strategy**: Targets very early-stage tokens

#### `MIN_TOKEN_PRICE_SOL`
- **Purpose**: Minimum token price in SOL
- **Default**: `0.000001`
- **Impact**: Avoids dust/worthless tokens

#### `MAX_MARKET_CAP_SOL`
- **Purpose**: Maximum market cap in SOL
- **Default**: `10000000.0`
- **Impact**: Targets small to medium cap tokens

#### `MIN_MARKET_CAP_SOL`
- **Purpose**: Minimum market cap in SOL
- **Default**: `1.0`
- **Impact**: Very permissive for micro-cap tokens

### Signal Generation
#### `MIN_SIGNAL_INTERVAL`
- **Purpose**: Minimum seconds between signals for same token
- **Default**: `300` (5 minutes)
- **Range**: 60-3600 seconds
- **Impact**: Prevents spam signals on same token

#### `MAX_SIGNALS_PER_HOUR`
- **Purpose**: Maximum signals generated per hour
- **Default**: `5`
- **Range**: 1-50
- **Impact**: Rate limiting for signal generation

### Analysis Weights
#### `VOLUME_WEIGHT`
- **Purpose**: Weight of volume in overall token score
- **Default**: `0.3` (30%)
- **Range**: 0.0-1.0
- **Impact**: Higher weight = volume more important

#### `LIQUIDITY_WEIGHT`
- **Purpose**: Weight of liquidity in token score
- **Default**: `0.3` (30%)
- **Range**: 0.0-1.0
- **Impact**: Higher weight = liquidity more important

#### `MOMENTUM_WEIGHT`
- **Purpose**: Weight of price momentum in score
- **Default**: `0.2` (20%)
- **Range**: 0.0-1.0
- **Impact**: Higher weight = momentum more important

#### `MARKET_IMPACT_WEIGHT`
- **Purpose**: Weight of market impact in score
- **Default**: `0.2` (20%)
- **Range**: 0.0-1.0
- **Impact**: Higher weight = market impact more important

---

## ⚖️ Arbitrage Settings

### Basic Arbitrage
#### `ARBITRAGE_MIN_PROFIT`
- **Purpose**: Minimum profit percentage for arbitrage execution
- **Default**: `0.5` (0.5%)
- **Range**: 0.1-5.0%
- **Impact**: Lower = more opportunities, higher = better profits

#### `ARBITRAGE_MAX_SIZE`
- **Purpose**: Maximum trade size for arbitrage (USD)
- **Default**: `1000.0`
- **Range**: 100-10000 USD
- **Impact**: Larger size = higher absolute profits but more risk

#### `ARBITRAGE_AUTO_EXECUTE`
- **Purpose**: Automatically execute arbitrage opportunities
- **Default**: `false`
- **Safety**: Keep false until thoroughly tested
- **Impact**: Manual vs automatic execution

#### `ARBITRAGE_SCAN_INTERVAL`
- **Purpose**: Scan interval for arbitrage opportunities (seconds)
- **Default**: `1.0`
- **Range**: 0.1-10.0 seconds
- **Impact**: Faster scanning = more opportunities but more resources

### Triangular Arbitrage
#### `TRIANGULAR_MIN_PROFIT`
- **Purpose**: Minimum profit for triangular arbitrage
- **Default**: `0.3` (0.3%)
- **Range**: 0.1-2.0%
- **Impact**: More complex arbitrage strategy

#### `TRIANGULAR_AUTO_EXECUTE`
- **Purpose**: Auto-execute triangular arbitrage
- **Default**: `false`
- **Safety**: Keep disabled for safety

---

## 🤖 Machine Learning Settings

### ML Core Settings
#### `ENABLE_ML`
- **Purpose**: Enable machine learning features
- **Default**: `true`
- **Impact**: Adds ML-based signal analysis
- **Performance**: Increases computational requirements

#### `ML_CONFIDENCE_THRESHOLD`
- **Purpose**: Minimum confidence for ML signals
- **Default**: `0.7` (70%)
- **Range**: 0.1-0.99
- **Impact**: Higher = fewer but higher quality ML signals

#### `ML_RETRAINING_INTERVAL`
- **Purpose**: Hours between model retraining
- **Default**: `24`
- **Range**: 1-168 hours
- **Impact**: More frequent = adaptive but resource intensive

### Feature Engineering
#### `ML_FEATURE_LOOKBACK`
- **Purpose**: Periods to look back for features
- **Default**: `20`
- **Range**: 5-100
- **Impact**: More periods = better context but slower processing

#### `ML_USE_SENTIMENT`
- **Purpose**: Include sentiment analysis
- **Default**: `true`
- **Impact**: Adds sentiment-based features

#### `ML_SENTIMENT_WEIGHT`
- **Purpose**: Weight of sentiment in predictions
- **Default**: `0.3` (30%)
- **Range**: 0.0-1.0
- **Impact**: Higher = sentiment more influential

#### `ML_PREDICT_HORIZON`
- **Purpose**: Prediction time horizon
- **Default**: `1h`
- **Options**: `15m`, `30m`, `1h`, `4h`, `1d`
- **Impact**: Longer horizon = different trading strategy

---

## 📈 API Rate Limits

Critical for managing API quota and avoiding rate limiting:

#### `JUPITER_RATE_LIMIT`
- **Purpose**: Requests per minute to Jupiter API
- **Default**: `100`
- **Range**: 10-1000
- **Impact**: Higher = faster execution but may hit limits

#### `RAYDIUM_RATE_LIMIT`
- **Purpose**: Requests per minute to Raydium
- **Default**: `60`
- **Impact**: DEX-specific rate limiting

#### `OPENBOOK_RATE_LIMIT`
- **Purpose**: Requests per minute to OpenBook
- **Default**: `60`
- **Impact**: DEX-specific rate limiting

#### `ALCHEMY_RATE_LIMIT`
- **Purpose**: Requests per minute to Alchemy RPC
- **Default**: `100`
- **Impact**: Blockchain RPC rate limiting

---

## 📧 Notification Settings

### Telegram Integration
#### `TELEGRAM_BOT_TOKEN`
- **Purpose**: Telegram bot token for notifications
- **Format**: `XXXXXXXXX:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`
- **Security**: Keep secure, moderate sensitivity
- **Usage**: Real-time trade alerts

#### `TELEGRAM_CHAT_ID`
- **Purpose**: Telegram chat/channel ID for messages
- **Format**: Numeric ID (positive or negative)
- **Usage**: Target chat for notifications

### Discord Integration
#### `DISCORD_WEBHOOK_URL`
- **Purpose**: Discord webhook for notifications
- **Format**: Full webhook URL
- **Security**: Keep secure
- **Usage**: Alternative to Telegram

### Email System
#### `EMAIL_ENABLED`
- **Purpose**: Enable email notifications
- **Default**: `true`
- **Impact**: Daily reports and alerts via email

#### Email Configuration
- `EMAIL_SMTP_SERVER`: SMTP server (default: `smtp.gmail.com`)
- `EMAIL_PORT`: SMTP port (default: `587`)
- `EMAIL_USER`: Sender email address
- `EMAIL_PASSWORD`: Email password or app password
- `EMAIL_TO`: Recipient email address
- `DAILY_REPORT_TIME`: Time for daily reports (default: `20:00`)

---

## 🔴 Live Trading Configuration ⚠️ **DANGER ZONE**

### Primary Safety Switch
#### `LIVE_TRADING_ENABLED` ⚠️ **CRITICAL**
- **Purpose**: Master switch for live trading
- **Default**: `false`
- **DANGER**: Setting to `true` enables real money trading
- **Security**: Multiple verification layers required

### Network Configuration
#### `SOLANA_NETWORK`
- **Purpose**: Solana network selection
- **Options**: `mainnet-beta`, `devnet`, `testnet`
- **Default**: `devnet` (safe for testing)
- **Live Trading**: Must be `mainnet-beta`

#### `SOLANA_RPC_URL`
- **Purpose**: Custom RPC endpoint
- **Usage**: Alternative to Alchemy for live trading
- **Performance**: Can improve execution speed

---

## ⛽ Gas and Priority Fee Settings

Critical for transaction execution on Solana:

#### `MAX_PRIORITY_FEE`
- **Purpose**: Maximum priority fee per transaction (SOL)
- **Default**: `0.01`
- **Range**: 0.001-0.1 SOL
- **Impact**: Higher fee = faster execution, more cost

#### `BASE_PRIORITY_FEE`
- **Purpose**: Base priority fee for transactions
- **Default**: `0.005`
- **Impact**: Minimum fee paid per transaction

#### `GAS_OPTIMIZATION`
- **Purpose**: Enable gas optimization algorithms
- **Default**: `true`
- **Impact**: Reduces transaction costs

#### `MAX_GAS_PRICE`
- **Purpose**: Maximum gas price in microlamports
- **Default**: `200`
- **Impact**: Transaction speed vs cost tradeoff

#### `TRANSACTION_FEE_BUFFER`
- **Purpose**: Buffer for transaction fees (SOL)
- **Default**: `0.01`
- **Impact**: Prevents insufficient fee failures

---

## 🛡️ Live Trading Safety Settings ⚠️

### Confirmation Requirements
#### `REQUIRE_CONFIRMATION`
- **Purpose**: Require manual confirmation for trades
- **Default**: `true`
- **Safety**: Keep `true` until fully automated
- **Impact**: Manual vs automatic execution

### Position Limits
#### `MAX_LIVE_POSITION_SIZE`
- **Purpose**: Maximum position size for live trading (SOL)
- **Default**: `0.5`
- **Range**: 0.1-10.0 SOL
- **Safety**: Start very small (0.1-0.5 SOL)

#### `MAX_LIVE_POSITIONS`
- **Purpose**: Maximum live positions allowed
- **Default**: `3`
- **Range**: 1-10
- **Safety**: Start with 1-2 positions

### Emergency Controls
#### `EMERGENCY_STOP_LOSS`
- **Purpose**: Emergency stop loss percentage
- **Default**: `0.2` (20%)
- **Range**: 0.1-0.5
- **Safety**: Immediate exit trigger

#### `MAX_DAILY_LOSS`
- **Purpose**: Maximum daily loss before stopping (SOL)
- **Default**: `1.0`
- **Range**: 0.1-10.0 SOL
- **Safety**: Daily circuit breaker

#### `MAX_TRANSACTION_AMOUNT`
- **Purpose**: Maximum transaction size (SOL)
- **Default**: `10.0`
- **Safety**: Prevents accidental large trades

---

## 📋 Transaction Management

### Retry Logic
#### `TRANSACTION_MAX_RETRIES`
- **Purpose**: Maximum retries for failed transactions
- **Default**: `3`
- **Range**: 1-10
- **Impact**: Reliability vs latency tradeoff

#### `TRANSACTION_TIMEOUT`
- **Purpose**: Transaction confirmation timeout (seconds)
- **Default**: `60`
- **Range**: 10-300 seconds
- **Impact**: How long to wait for confirmation

#### `TRANSACTION_MONITORING`
- **Purpose**: Enable transaction monitoring
- **Default**: `true`
- **Impact**: Track transaction status

---

## 🚨 Emergency Controls

Critical safety mechanisms for live trading:

### Loss Limits
#### `EMERGENCY_MAX_DAILY_LOSS`
- **Purpose**: Emergency daily loss limit (USD)
- **Default**: `100.0`
- **Impact**: Hard stop on daily losses

#### `EMERGENCY_MAX_DRAWDOWN`
- **Purpose**: Emergency drawdown percentage
- **Default**: `10.0`
- **Impact**: Portfolio decline limit

### Position Controls
#### `EMERGENCY_MAX_POSITION_SIZE`
- **Purpose**: Emergency position size limit (USD)
- **Default**: `500.0`
- **Impact**: Individual position limit

#### `EMERGENCY_POSITION_LIMIT`
- **Purpose**: Emergency total position limit
- **Default**: `5`
- **Impact**: Maximum open positions

### Trading Controls
#### `EMERGENCY_MAX_TRADES_HOUR`
- **Purpose**: Emergency hourly trade limit
- **Default**: `20`
- **Impact**: Prevents runaway trading

#### `EMERGENCY_MAX_ERROR_RATE`
- **Purpose**: Emergency error rate threshold
- **Default**: `0.3` (30%)
- **Impact**: Stops trading if too many errors

#### `EMERGENCY_MIN_BALANCE`
- **Purpose**: Emergency minimum balance (SOL)
- **Default**: `0.1`
- **Impact**: Preserves minimum funds

#### `EMERGENCY_MAX_VOLATILITY`
- **Purpose**: Emergency volatility threshold
- **Default**: `50.0` (50%)
- **Impact**: Stops trading in extreme volatility

---

## ⚙️ Enhanced Filter Parameters

New parameters for optimized token discovery:

### API Strategy
#### `API_STRATEGY`
- **Purpose**: API provider selection strategy
- **Options**: `dual`, `geckoterminal`, `solana_tracker`
- **Default**: `dual`
- **Impact**: Data source for token discovery
- **Current**: Smart dual-API for maximum coverage

#### `API_PROVIDER`
- **Purpose**: Primary API provider
- **Default**: `geckoterminal`
- **Impact**: Primary data source

#### `BIRDEYE_API_KEY`
- **Purpose**: Birdeye API key for enhanced data
- **Impact**: Access to trending token data

#### `SOLANA_TRACKER_KEY`
- **Purpose**: Solana Tracker API key
- **Impact**: Additional token discovery source

### Trending Filters
#### `ENABLE_TRENDING_FILTER`
- **Purpose**: Enable trending token filtering
- **Default**: `true`
- **Impact**: Prioritizes trending tokens

#### `TRENDING_FALLBACK_MODE`
- **Purpose**: Fallback behavior for trending API
- **Default**: `permissive`
- **Options**: `permissive`, `strict`
- **Impact**: How to handle trending API failures

#### `MAX_TRENDING_RANK`
- **Purpose**: Maximum trending rank to consider
- **Default**: `50`
- **Range**: 1-100
- **Impact**: Top trending tokens only

#### `MIN_PRICE_CHANGE_24H`
- **Purpose**: Minimum 24h price change percentage
- **Default**: `20.0` (20%)
- **Range**: 5.0-100.0%
- **Impact**: Momentum filter

#### `MIN_TRENDING_SCORE`
- **Purpose**: Minimum trending score for selection
- **Default**: `60.0`
- **Range**: 1.0-100.0
- **Impact**: Quality filter for trending tokens

### Enhanced Filters
#### `MIN_CONTRACT_SCORE`
- **Purpose**: Minimum contract security score
- **Default**: `50`
- **Range**: 1-100
- **Impact**: Security screening

---

## 🔧 Settings Interaction Matrix

### Critical Relationships

1. **Position Sizing Chain**:
   - `INITIAL_PAPER_BALANCE` → `MAX_POSITION_SIZE` → `PAPER_MAX_POSITION_SIZE`
   - Impact: Larger balance allows larger absolute positions

2. **Risk Management Hierarchy**:
   - `STOP_LOSS_PERCENTAGE` < `EMERGENCY_STOP_LOSS` < `MAX_DAILY_LOSS`
   - Impact: Multiple safety layers

3. **Token Filtering Pipeline**:
   - `MIN_LIQUIDITY` → `MIN_VOLUME_24H` → `MIN_MOMENTUM_PERCENTAGE`
   - Impact: Sequential filtering reduces opportunities

4. **API Rate Limiting**:
   - `SCANNER_INTERVAL` × `*_RATE_LIMIT` = Total API usage
   - Impact: Balance discovery speed vs quota usage

5. **Paper vs Live Settings**:
   - Paper settings more permissive than live equivalents
   - Impact: Different approval rates between modes

---

## ⚠️ Common Configuration Issues

### 1. Inconsistent Position Limits
- **Problem**: `MAX_POSITIONS` ≠ `MAX_SIMULTANEOUS_POSITIONS`
- **Solution**: Align both settings
- **Impact**: Trading logic confusion

### 2. Over-restrictive Filters
- **Problem**: `MIN_LIQUIDITY` + `MIN_VOLUME_24H` + `MIN_MOMENTUM_PERCENTAGE` too high
- **Solution**: Lower one or more filters
- **Impact**: Zero trading opportunities

### 3. API Quota Exhaustion
- **Problem**: `SCANNER_INTERVAL` too low + high rate limits
- **Solution**: Increase scan interval or reduce rate limits
- **Impact**: API blocking

### 4. Risk Parameter Conflicts
- **Problem**: `MAX_POSITION_SIZE` > `MAX_DAILY_LOSS`
- **Solution**: Ensure daily loss > single position risk
- **Impact**: Risk management failure

---

## 🎯 Quick Configuration Recommendations

### Conservative Paper Trading
```env
PAPER_MIN_LIQUIDITY=1000.0
PAPER_SIGNAL_THRESHOLD=0.7
MAX_POSITION_SIZE=0.1
STOP_LOSS_PERCENTAGE=0.10
TAKE_PROFIT_PERCENTAGE=0.20
```

### Balanced Paper Trading (Current)
```env
PAPER_MIN_LIQUIDITY=50.0
PAPER_SIGNAL_THRESHOLD=0.3
MAX_POSITION_SIZE=0.2
STOP_LOSS_PERCENTAGE=0.15
TAKE_PROFIT_PERCENTAGE=0.5
```

### Aggressive Paper Trading
```env
PAPER_MIN_LIQUIDITY=10.0
PAPER_SIGNAL_THRESHOLD=0.1
MAX_POSITION_SIZE=0.35
STOP_LOSS_PERCENTAGE=0.20
TAKE_PROFIT_PERCENTAGE=1.0
```

### Live Trading Transition
```env
LIVE_TRADING_ENABLED=true
MAX_LIVE_POSITION_SIZE=0.1
MAX_LIVE_POSITIONS=1
REQUIRE_CONFIRMATION=true
EMERGENCY_MAX_DAILY_LOSS=10.0
```

---

## 🔄 Configuration Update Process

1. **Backup Current Config**: Always backup working `.env`
2. **Test in Paper Mode**: Verify changes work as expected
3. **Monitor Performance**: Check approval rates and execution
4. **Gradual Adjustments**: Make small incremental changes
5. **Document Changes**: Track what worked and what didn't
6. **Validate Settings**: Ensure no conflicts or invalid ranges

---

## 📊 Performance Impact Guide

### High Impact Settings
- `SCANNER_INTERVAL`: Directly affects opportunity discovery
- `MIN_LIQUIDITY`: Major filter for token qualification
- `PAPER_SIGNAL_THRESHOLD`: Controls trade frequency
- `MAX_POSITION_SIZE`: Risk per trade

### Medium Impact Settings  
- `STOP_LOSS_PERCENTAGE`: Exit timing
- `TAKE_PROFIT_PERCENTAGE`: Profit capture
- `MAX_POSITIONS`: Diversification level

### Low Impact Settings
- `LOG_LEVEL`: Mainly affects debugging
- `EMAIL_ENABLED`: Notification only
- Path settings: Storage locations

This comprehensive documentation should guide you in optimizing your SolTrader bot for maximum performance while maintaining appropriate risk management.