# 🚀 Production Deployment Checklist

## Pre-Deployment Requirements

### 📋 Configuration Validation
- [ ] **Environment Variables**
  - [ ] `PRIVATE_KEY` set (CRITICAL - Required for live trading)
  - [ ] `WALLET_ADDRESS` matches private key
  - [ ] `ALCHEMY_RPC_URL` configured
  - [ ] `SOLANA_TRACKER_KEY` valid and active
  - [ ] Email settings configured (optional but recommended)

- [ ] **Trading Mode Settings**
  - [ ] `PAPER_TRADING=false` (Switch to live trading)
  - [ ] `LIVE_TRADING_ENABLED=true`
  - [ ] Risk management parameters reviewed
  - [ ] Position size limits appropriate for capital

### 🔐 Security Checklist
- [ ] **Private Key Security**
  - [ ] Private key stored securely (not in code/Git)
  - [ ] Wallet has minimal required funds
  - [ ] Test wallet used first before main wallet
  - [ ] Private key never logged or displayed

- [ ] **Risk Controls**
  - [ ] `MAX_POSITION_SIZE` set conservatively (start with 2-5%)
  - [ ] `MAX_DAILY_LOSS` configured (e.g., 10% of capital)
  - [ ] `MAX_TRADES_PER_DAY` set to prevent overtrading
  - [ ] Emergency stop procedures tested

### 💰 Financial Preparation
- [ ] **Wallet Setup**
  - [ ] Sufficient SOL balance for trades + gas fees
  - [ ] Reserve 1-2 SOL for transaction fees
  - [ ] Backup funds available if needed
  - [ ] Portfolio value matches `INITIAL_CAPITAL` setting

### 🧪 Testing Validation
- [ ] **Paper Trading Results**
  - [ ] Paper trading run for at least 24 hours
  - [ ] Positive win rate (>60% recommended)
  - [ ] Reasonable approval rate (5-15%)
  - [ ] No critical errors in logs
  - [ ] Dashboard showing accurate data

## Deployment Process

### Step 1: Final Configuration
```bash
# Update .env file
PAPER_TRADING=false
LIVE_TRADING_ENABLED=true
PRIVATE_KEY=your_private_key_here

# Verify settings
python check_setup.py
```

### Step 2: Health Monitoring Setup
- [ ] Health monitor configured and running
- [ ] Email notifications tested
- [ ] Dashboard accessible
- [ ] Log monitoring in place

### Step 3: Gradual Deployment
- [ ] **Start Small**: Begin with minimum position sizes
- [ ] **Monitor Closely**: Watch first few trades manually
- [ ] **Verify Execution**: Check actual vs expected trade outcomes
- [ ] **Scale Gradually**: Increase position sizes over time

## Post-Deployment Monitoring

### 📊 Key Metrics to Watch
- **First Hour**
  - [ ] Trades executing successfully
  - [ ] Position sizes as expected
  - [ ] No execution errors
  - [ ] Gas fees reasonable

- **First 24 Hours**
  - [ ] Win rate maintaining (should be similar to paper trading)
  - [ ] Portfolio value trending correctly
  - [ ] No unexpected losses
  - [ ] API limits not exceeded

- **First Week**
  - [ ] Consistent performance
  - [ ] Risk controls working
  - [ ] No emergency stops triggered
  - [ ] Profitability trending positive

### 🚨 Warning Signs (Stop Immediately If)
- Consecutive losses (>5 losing trades)
- Win rate drops below 40%
- Daily loss exceeds configured limit
- Unusual gas fee spikes
- API rate limit errors
- Wallet balance draining unexpectedly

## Emergency Procedures

### 🛑 Manual Emergency Stop
```bash
# Method 1: Environment variable
export TRADING_PAUSED=true

# Method 2: Kill process
pkill -f "python.*main.py"

# Method 3: Close all positions manually through dashboard
```

### 💊 Recovery Actions
- [ ] Review logs for errors
- [ ] Check wallet balance and transactions
- [ ] Verify API key status
- [ ] Restart with paper trading if needed
- [ ] Contact support if issues persist

## Settings Differences: Paper vs Live

### Paper Trading Settings (Development)
```python
PAPER_TRADING = True
PAPER_MIN_MOMENTUM_THRESHOLD = 3.0  # More permissive
PAPER_MIN_LIQUIDITY = 50.0          # Lower requirements
PAPER_TRADING_SLIPPAGE = 0.50       # High slippage tolerance
PAPER_MAX_POSITION_SIZE = 0.5       # 50% max position
PAPER_SIGNAL_THRESHOLD = 0.3        # Lower signal threshold
```

### Live Trading Settings (Production)
```python
PAPER_TRADING = False
MIN_MOMENTUM_THRESHOLD = 5.0        # More conservative
MIN_LIQUIDITY = 1000.0              # Higher liquidity required
SLIPPAGE_TOLERANCE = 0.25           # Conservative slippage
MAX_POSITION_SIZE = 0.05            # 5% max position (start small)
SIGNAL_THRESHOLD = 0.7              # Higher signal confidence required
```

## Common Edge Cases & Fixes

### 🔧 Token Discovery Issues
- **Low approval rates**: Increase `MIN_MOMENTUM_THRESHOLD`
- **No tokens found**: Check API keys and network connectivity
- **High false positives**: Raise `SIGNAL_THRESHOLD`

### 💸 Execution Problems
- **Slippage too high**: Reduce `MAX_SLIPPAGE` or increase liquidity requirements
- **Failed transactions**: Increase gas fees or reduce position sizes
- **Insufficient balance**: Monitor wallet balance and set appropriate limits

### 📈 Performance Issues
- **Poor win rate**: Review signal generation and validation criteria
- **High gas costs**: Optimize trade frequency and position sizing
- **Missed opportunities**: Check API response times and processing speed

## Success Metrics

### 🎯 Target Performance (After Optimization)
- **Win Rate**: >60%
- **Approval Rate**: 5-15% (depends on market conditions)
- **Daily ROI**: 1-5% (varies with market volatility)
- **Max Drawdown**: <10%
- **API Usage**: <90% of daily limit

### 📝 Documentation Requirements
- [ ] Trading log maintained
- [ ] Performance metrics tracked daily
- [ ] Configuration changes documented
- [ ] Issues and resolutions logged

## Final Go/No-Go Decision

### ✅ Ready for Production When:
- [ ] All checklist items completed
- [ ] Paper trading profitable for 48+ hours
- [ ] All tests passing
- [ ] Risk controls verified
- [ ] Monitoring systems operational
- [ ] Emergency procedures tested

### ❌ Do NOT Deploy If:
- [ ] Paper trading showing consistent losses
- [ ] Critical errors in logs
- [ ] API keys invalid or rate-limited
- [ ] Insufficient testing performed
- [ ] Risk controls not properly configured
- [ ] Monitoring systems not working

---

**⚠️ IMPORTANT**: Live trading involves real money. Start with small amounts and scale gradually. Always monitor closely during initial deployment.

**🆘 Support**: If issues arise, immediately stop trading and revert to paper trading mode while investigating.