# 📊 Paper vs Live Trading Settings Guide

## Overview
This document outlines the key differences between paper trading (development/testing) and live trading (production) settings for the SolTrader bot.

## Core Philosophy Differences

### 📝 Paper Trading Approach
- **Maximize learning opportunities**
- **Test strategy effectiveness** 
- **Gather performance data**
- **More permissive thresholds**
- **Higher risk tolerance for testing**

### 💰 Live Trading Approach
- **Minimize risk and preserve capital**
- **Conservative position sizing**
- **Higher quality signals only**
- **Stricter validation criteria**
- **Robust error handling**

## Detailed Settings Comparison

### 🎯 Signal Generation

| Setting | Paper Trading | Live Trading | Reasoning |
|---------|---------------|--------------|-----------|
| `SIGNAL_THRESHOLD` | 0.3 | 0.7 | Lower threshold in paper mode captures more signals for testing |
| `MIN_MOMENTUM_THRESHOLD` | 3.0 | 5.0 | More conservative momentum requirement for real money |
| `MIN_LIQUIDITY` | 50 SOL | 1000 SOL | Ensure sufficient liquidity for actual execution |

### 💼 Position Management

| Setting | Paper Trading | Live Trading | Reasoning |
|---------|---------------|--------------|-----------|
| `MAX_POSITION_SIZE` | 0.5 (50%) | 0.05 (5%) | Start with small positions in live trading |
| `PAPER_MAX_POSITION_SIZE` | 0.5 | N/A | Separate paper trading limit |
| `PAPER_BASE_POSITION_SIZE` | 0.1 | N/A | Base position for paper trades |

### 🎢 Risk & Slippage

| Setting | Paper Trading | Live Trading | Reasoning |
|---------|---------------|--------------|-----------|
| `SLIPPAGE_TOLERANCE` | 0.50 (50%) | 0.25 (25%) | High slippage acceptable for testing |
| `PAPER_TRADING_SLIPPAGE` | 0.50 | N/A | Separate slippage for paper mode |
| `MAX_SLIPPAGE` | 0.30 | 0.25 | Conservative slippage for real execution |

### 📈 Trading Frequency

| Setting | Paper Trading | Live Trading | Reasoning |
|---------|---------------|--------------|-----------|
| `MAX_TRADES_PER_DAY` | 20 | 10 | Fewer trades with real money to reduce risk |
| `MAX_DAILY_LOSS` | 50% | 10% | Conservative daily loss limit for live trading |

### 🔍 Token Validation

| Setting | Paper Trading | Live Trading | Reasoning |
|---------|---------------|--------------|-----------|
| `MIN_VOLUME_24H` | 0 SOL | 500 SOL | Ensure adequate trading volume |
| `MIN_HOLDERS` | 0 | 100 | Require established token community |
| `MIN_MARKET_CAP` | 0 SOL | 10,000 SOL | Filter out very small cap tokens |

## Configuration Files

### 📝 Paper Trading (.env)
```bash
# Core Mode
PAPER_TRADING=true
LIVE_TRADING_ENABLED=false

# Paper Trading Specific
PAPER_TRADING_SLIPPAGE=0.50
PAPER_MAX_POSITION_SIZE=0.5
PAPER_BASE_POSITION_SIZE=0.1
PAPER_MIN_MOMENTUM_THRESHOLD=3.0
PAPER_MIN_LIQUIDITY=50.0
PAPER_SIGNAL_THRESHOLD=0.3

# General Trading
MAX_TRADES_PER_DAY=20
MAX_POSITION_SIZE=0.35
SLIPPAGE_TOLERANCE=0.30
MAX_SLIPPAGE=0.30

# Risk Management (Relaxed)
MAX_DAILY_LOSS=50.0
MIN_BALANCE=0.1
STOP_LOSS_PERCENTAGE=0.15
TAKE_PROFIT_PERCENTAGE=0.30
```

### 💰 Live Trading (.env)
```bash
# Core Mode (CRITICAL CHANGE)
PAPER_TRADING=false
LIVE_TRADING_ENABLED=true

# Live Trading Settings
PRIVATE_KEY=your_actual_private_key_here
SLIPPAGE_TOLERANCE=0.25
MAX_POSITION_SIZE=0.05  # Start with 5%
MIN_MOMENTUM_THRESHOLD=5.0
MIN_LIQUIDITY=1000.0
SIGNAL_THRESHOLD=0.7

# Conservative Trading
MAX_TRADES_PER_DAY=10
MIN_VOLUME_24H=500
MIN_HOLDERS=100
MIN_MARKET_CAP=10000

# Strict Risk Management
MAX_DAILY_LOSS=10.0     # 10% maximum daily loss
MIN_BALANCE=2.0         # Keep 2 SOL minimum
STOP_LOSS_PERCENTAGE=0.10   # Tighter stop loss
TAKE_PROFIT_PERCENTAGE=0.20 # Conservative take profit

# Position Limits
MAX_SIMULTANEOUS_POSITIONS=2
MIN_TRADE_SIZE=0.05
```

## Behavioral Differences

### 🧪 Paper Trading Behavior
- **Simulated execution**: No real blockchain transactions
- **Perfect fills**: Assumes orders always fill at expected price
- **No gas fees**: Simulated gas costs only
- **High risk tolerance**: More aggressive position sizing
- **Learning focused**: Captures more trading opportunities

### 🎯 Live Trading Behavior
- **Real execution**: Actual blockchain transactions
- **Slippage impact**: Real market conditions affect fills
- **Actual gas fees**: Real SOL spent on transactions
- **Risk averse**: Smaller, more conservative positions
- **Profit focused**: Only high-confidence opportunities

## Migration Strategy

### 🔄 Transitioning from Paper to Live

#### Phase 1: Validation (Paper Trading)
1. Run paper trading for minimum 48 hours
2. Achieve >60% win rate consistently
3. Verify 5-15% approval rate
4. Ensure no critical errors
5. Validate dashboard accuracy

#### Phase 2: Conservative Live Start
1. Set `MAX_POSITION_SIZE=0.02` (2%)
2. Set `MAX_TRADES_PER_DAY=5`
3. Use highest quality signals only
4. Monitor every trade manually

#### Phase 3: Gradual Scaling
1. Increase position size to 3-5% if profitable
2. Allow more daily trades (up to 10)
3. Relax some conservative thresholds
4. Maintain strict risk controls

## Performance Expectations

### 📊 Paper Trading Targets
- **Win Rate**: 60-80% (more opportunities tested)
- **Approval Rate**: 10-25% (more permissive filters)
- **Daily Trades**: 5-20 (high activity for testing)
- **Position Size**: 10-50% (testing different sizes)

### 🎯 Live Trading Targets
- **Win Rate**: 65-75% (higher quality signals)
- **Approval Rate**: 5-15% (stricter filtering)
- **Daily Trades**: 2-10 (quality over quantity)
- **Position Size**: 2-10% (conservative sizing)

## Monitoring Differences

### 🔍 Paper Trading Monitoring
- Focus on strategy effectiveness
- Track approval rates and signal quality
- Monitor for false positives
- Optimize thresholds for better performance

### 📈 Live Trading Monitoring
- Focus on capital preservation
- Track actual vs expected execution
- Monitor gas fee efficiency
- Watch for unexpected losses

## Risk Scenarios

### ⚠️ Paper Trading Risks
- **Overconfidence**: Paper results may not match live results
- **Poor risk habits**: Relaxed settings may create bad practices
- **False positives**: Strategy may not work with real slippage

### 🚨 Live Trading Risks
- **Capital loss**: Real money at risk
- **Execution failure**: Slippage and failed transactions
- **Market impact**: Large positions may move prices

## Best Practices

### ✅ Do's
- Always test thoroughly in paper mode first
- Start with conservative live settings
- Scale position sizes gradually
- Keep detailed performance logs
- Have emergency stop procedures ready

### ❌ Don'ts
- Never jump directly to live trading without testing
- Don't use paper trading position sizes in live mode
- Don't ignore slippage and gas fee impacts
- Don't risk more than you can afford to lose
- Don't disable risk controls in live mode

---

**💡 Remember**: The goal of paper trading is to validate your strategy. The goal of live trading is to make profit while preserving capital. The settings should reflect these different objectives.