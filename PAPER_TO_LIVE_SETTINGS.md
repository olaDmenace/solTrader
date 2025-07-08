# 📊 Paper Trading → Live Trading Settings Guide

## 🔄 **Settings Changes Required**

### **1. Core Trading Mode**
```python
# In src/config/settings.py or .env file
PAPER_TRADING: bool = False  # Change from True to False
```

### **2. Wallet Configuration** 
```bash
# .env file updates
WALLET_ADDRESS=your_actual_solana_wallet_address
PRIVATE_KEY=your_actual_private_key_encrypted
```

### **3. Capital & Position Sizing**

#### **Conservative ($10 start):**
```python
INITIAL_CAPITAL: float = 0.067  # ~$10 worth of SOL
MAX_POSITION_SIZE: float = 0.013  # ~$2 per trade
MAX_TRADE_SIZE: float = 0.007  # ~$1 per trade
MIN_TRADE_SIZE: float = 0.001  # ~$0.15 per trade
```

#### **Moderate ($100 start):**
```python
INITIAL_CAPITAL: float = 0.67  # ~$100 worth of SOL
MAX_POSITION_SIZE: float = 0.13  # ~$20 per trade
MAX_TRADE_SIZE: float = 0.067  # ~$10 per trade
MIN_TRADE_SIZE: float = 0.007  # ~$1 per trade
```

### **4. Risk Management Updates**
```python
# Tighter controls for live trading
MAX_DAILY_LOSS: float = 0.013  # ~$2 daily loss limit
MAX_POSITIONS: int = 2  # Reduce to 2 simultaneous positions
MAX_DAILY_TRADES: int = 10  # Reduce daily trade frequency
```

## ⚙️ **Environment Variables (.env)**

### **Complete .env Template for Live Trading:**
```bash
# ============ CORE SETTINGS ============
ALCHEMY_RPC_URL=https://solana-mainnet.g.alchemy.com/v2/YOUR_API_KEY
WALLET_ADDRESS=your_actual_solana_wallet_address
PRIVATE_KEY=your_encrypted_private_key

# ============ TRADING MODE ============
PAPER_TRADING=false
TRADING_PAUSED=false

# ============ CAPITAL MANAGEMENT ============
INITIAL_CAPITAL=0.067                  # $10 worth of SOL
MAX_POSITION_SIZE=0.013                # $2 per position
MAX_TRADE_SIZE=0.007                   # $1 per trade
MIN_TRADE_SIZE=0.001                   # $0.15 minimum

# ============ RISK MANAGEMENT ============
MAX_DAILY_LOSS=0.013                   # $2 daily loss limit
MAX_POSITIONS=2                        # 2 simultaneous positions
MAX_DAILY_TRADES=10                    # 10 trades per day max
STOP_LOSS_PERCENTAGE=0.15              # 15% stop loss
TAKE_PROFIT_PERCENTAGE=0.5             # 50% take profit

# ============ TOKEN FILTERING ============
MAX_TOKEN_PRICE_SOL=0.01               # Max $1.50 per token
MIN_TOKEN_PRICE_SOL=0.000001           # Min $0.00015 per token
MAX_MARKET_CAP_SOL=10000.0             # Max $1.5M market cap
MIN_MARKET_CAP_SOL=10.0                # Min $1.5K market cap
NEW_TOKEN_MAX_AGE_MINUTES=2880         # 48 hours max age

# ============ EXECUTION SETTINGS ============
SLIPPAGE_TOLERANCE=0.25                # 25% slippage tolerance
PRIORITY_FEE_MULTIPLIER=2.0            # 2x priority fees
SCAN_INTERVAL=5                        # 5-second scanning
MIN_LIQUIDITY=500.0                    # Min 500 SOL liquidity

# ============ NOTIFICATIONS (Optional) ============
DISCORD_WEBHOOK_URL=your_discord_webhook_url
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## 🔄 **Step-by-Step Transition**

### **Step 1: Backup Current Configuration**
```bash
cp .env .env.paper_backup
cp src/config/settings.py src/config/settings.py.backup
```

### **Step 2: Update Settings File**
```python
# In src/config/settings.py, change:
PAPER_TRADING: bool = False  # Live trading mode
INITIAL_CAPITAL: float = 0.067  # Start with $10 worth of SOL
```

### **Step 3: Update Environment Variables**
```bash
# Edit .env file
nano .env

# Add your actual wallet details
WALLET_ADDRESS=your_real_wallet
PRIVATE_KEY=your_encrypted_key
PAPER_TRADING=false
```

### **Step 4: Test Configuration**
```bash
# Validate settings
python3 -c "from src.config.settings import load_settings; s=load_settings(); print(f'Paper: {s.PAPER_TRADING}, Capital: {s.INITIAL_CAPITAL}')"
```

### **Step 5: Start Live Trading**
```bash
# Start the bot in live mode
python3 main.py
```

## ⚠️ **Critical Safety Checks**

### **Before Going Live:**
1. **✅ Wallet has small test amount** (~$15-20 SOL)
2. **✅ Private key is securely stored**
3. **✅ Risk limits are set appropriately**
4. **✅ Paper trading worked correctly**
5. **✅ You understand the risks**

### **During First Live Session:**
1. **Monitor every trade manually**
2. **Check dashboard updates**
3. **Verify positions are accurate**
4. **Watch for any errors**
5. **Be ready to use emergency stop**

### **Emergency Controls:**
```bash
# Stop trading immediately
python3 enable_trading.py disable

# Check current positions
python3 -c "from src.trading.strategy import TradingStrategy; print('Positions:', strategy.position_manager.get_positions())"
```

## 📊 **Key Differences: Paper vs Live**

| Setting | Paper Trading | Live Trading |
|---------|---------------|--------------|
| **Capital** | 100 SOL (simulated) | 0.067 SOL (~$10) |
| **Risk** | No real money | Real money at risk |
| **Position Size** | 5 SOL | 0.013 SOL (~$2) |
| **Trades/Day** | 20 | 10 |
| **Monitoring** | Casual | Active supervision |
| **Slippage** | Simulated | Real market impact |

## 🎯 **Success Metrics for Live Trading**

### **Week 1 Goals:**
- Zero critical errors
- 2-5 successful token discoveries per day
- 60%+ win rate on trades
- Stay within daily loss limits

### **Week 2-4 Goals:**
- Consistent positive returns
- Smooth execution without manual intervention
- Refined position sizing based on results
- Gradual capital increase if successful

## 💡 **Pro Tips for Live Trading**

### **Start Conservative:**
- Use minimum position sizes initially
- Trade only during high liquidity hours
- Monitor every trade for first week
- Don't chase losses with bigger positions

### **Scale Gradually:**
- Increase position size only after proven success
- Add capital monthly, not daily
- Keep detailed records of all trades
- Review and optimize weekly

### **Risk Management:**
- Never risk more than you can afford to lose
- Use stop-losses religiously
- Take profits on 50%+ gains
- Exit quickly on 15%+ losses

---

## ✅ **Ready for Live Trading**

Once you've updated these settings and followed the safety checks, your bot will be ready for live trading with real money. The enhanced 48-hour token detection window and realistic simulation should provide plenty of trading opportunities while maintaining risk control.

**Remember: Start small, monitor closely, and scale gradually!**