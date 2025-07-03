# 🦍 SolTrader APE STRATEGY - Complete User Guide

## 🎯 What Is The Ape Strategy?

The **APE Strategy** is an aggressive automated trading approach that:
- **Scans** the Solana blockchain for brand new token launches (< 30 minutes old)
- **Apes in** quickly with small positions when security checks pass
- **Rides momentum** as long as price keeps going up
- **Exits fast** when momentum turns negative or shows weakness

**Key Philosophy**: Small losses, big wins. Get in early, let winners run, cut losers fast.

## 🌐 Monitoring Your APE Bot

### Web Dashboard (Recommended)
- **Real-time browser interface** at `http://localhost:5000`
- **Performance metrics**: P&L, win rate, trade count, balance
- **Live trade history** with entry/exit details
- **Event feed**: Token discoveries, trades, errors
- **Auto-refreshes** every 5 seconds

**Quick Start**:
1. Run: `python main.py` (start bot)
2. Run: `python create_monitoring_dashboard.py` (start dashboard) 
3. Open: `http://localhost:5000` in browser

### Console Monitoring (Alternative)
- **Terminal-based** real-time monitoring
- **Color-coded** activity display
- **Performance stats** and recent events

**Quick Start**: `python monitor_bot.py`

---

## 🚀 How The Bot Works (Step-by-Step Flow)

### 1. **Token Discovery Phase** (Every 5 seconds)
```
┌─ Scanner detects new token
├─ Check: Is token < 30 minutes old?
├─ Check: Contract security score > 70/100
├─ Check: Minimum liquidity ($500+)
└─ Check: Minimum holders (5-15 people)
```

### 2. **Entry Decision** (Lightning Fast)
```
If ALL checks pass:
├─ Calculate position size (max 1 SOL per token)
├─ Execute buy with high gas priority
├─ Start momentum monitoring (every 3 seconds)
└─ Log entry for success rate tracking
```

### 3. **Position Monitoring** (High Frequency - Every 3 seconds)
```
For each open position:
├─ Update current price & volume
├─ Calculate momentum indicators:
│   ├─ 5-period price momentum
│   ├─ RSI (overbought detection)
│   └─ Volume trend analysis
└─ Check exit conditions
```

### 4. **Exit Logic** (Dynamic & Momentum-Based)
```
Exit triggers (in priority order):
1. 🔴 Momentum Reversal: -3% momentum + declining volume
2. 🟡 Overbought Divergence: RSI > 80 + momentum < 1%
3. 🟣 Time Limit: Hold for max 3 hours (prevent bag holding)
4. 🟢 Profit Protection: 20%+ profit → 5% trailing stop
5. ⚡ Fast Loss Cut: -5% momentum + 10% loss
```

---

## ⚙️ Configuration Settings

### 🎛️ Core Ape Settings
```python
# Position Management
MAX_POSITION_PER_TOKEN = 1.0      # Max 1 SOL per token
MAX_SIMULTANEOUS_POSITIONS = 5    # Max 5 positions at once
MAX_HOLD_TIME_MINUTES = 180       # 3 hours max hold

# Entry Criteria  
MIN_CONTRACT_SCORE = 70            # Security threshold
NEW_TOKEN_MAX_AGE_MINUTES = 30     # Only trade very new tokens
MIN_LIQUIDITY = 500               # $500 minimum liquidity

# Exit Settings
MOMENTUM_THRESHOLD = -0.03         # -3% momentum triggers exit
RSI_OVERBOUGHT = 80               # RSI overbought level
PROFIT_PROTECTION_THRESHOLD = 0.2  # Start trailing at 20% profit
```

### 🏃‍♂️ Speed Optimization
```python
# Monitoring Frequency
SCAN_INTERVAL = 5                  # Scan every 5 seconds
POSITION_MONITOR_INTERVAL = 3      # Monitor positions every 3 seconds

# Gas Settings (for speed)
PRIORITY_FEE_MULTIPLIER = 2.0      # 2x gas for fast execution
MAX_GAS_PRICE = 200               # Higher gas limit
MAX_SLIPPAGE = 0.03               # 3% slippage tolerance
```

### 🛡️ Risk Management
```python
# Daily Limits
MAX_DAILY_TRADES = 15             # Max 15 trades per day
MAX_DAILY_LOSS = 2.0              # Stop at -2 SOL daily loss
ERROR_THRESHOLD = 10              # Allow more errors for speed

# Position Sizing
MAX_TRADE_SIZE = 2.0              # Max 2 SOL per trade
STOP_LOSS_PERCENTAGE = 0.15       # 15% traditional stop loss
```

---

## 📊 Understanding Success Metrics

### 💡 What Good Performance Looks Like
- **Win Rate**: 40-60% (quality over quantity)
- **Risk/Reward**: Small losses (-10%), big wins (+50-200%)
- **Daily Trades**: 5-15 trades per day
- **Hold Time**: Average 30-90 minutes

### 📈 Example Successful Day
```
Trade 1: MEME1 → +120% (2.2 SOL profit)
Trade 2: DOGE2 → -8% (0.08 SOL loss)
Trade 3: PEPE3 → +45% (0.45 SOL profit)
Trade 4: SHIB4 → -12% (0.12 SOL loss)
Trade 5: MOON5 → +80% (0.8 SOL profit)

Net: +3.25 SOL profit (65% win rate)
```

### ⚠️ Warning Signs (Bad Day)
```
Multiple -80% losses = Rug pulls detected
High gas fees, failed transactions = Network congestion
No trades for hours = No new tokens launching
All exits at time limit = Not catching momentum
```

---

## 🚨 Risk Warnings & Reality Check

### ❌ This Strategy Will NOT Always Make Money
- **Rug pulls** can cause -90% losses instantly
- **Bot competition** may frontrun your entries  
- **Network congestion** can cause failed transactions
- **Market downturns** reduce new token activity

### 🛡️ Essential Safety Rules
1. **Never risk more than you can lose completely**
2. **Start with small amounts** (10-50 SOL max)
3. **Monitor daily losses** - stop if you hit -20%
4. **Check bot performance** daily and adjust settings
5. **Have exit strategy** - know when to turn off the bot

### 💡 Realistic Expectations
- **Good days**: +20% to +50% returns
- **Bad days**: -30% to -70% losses  
- **Weekly**: Highly volatile, not guaranteed profits
- **Success requires**: Patience, risk management, continuous monitoring

---

## 🔧 Setup Instructions

### 1. Environment Configuration
Create `.env` file:
```bash
# Required
ALCHEMY_RPC_URL=your_alchemy_url_here
WALLET_ADDRESS=your_solana_wallet_address

# Optional (for notifications)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Trading Settings
PAPER_TRADING=true                 # Start with paper trading!
INITIAL_PAPER_BALANCE=100.0
MOMENTUM_EXIT_ENABLED=true
MIN_CONTRACT_SCORE=70
MAX_POSITION_PER_TOKEN=1.0
```

### 2. Start Paper Trading (RECOMMENDED)
```bash
# Install dependencies
pip install -r requirements.txt

# Run paper trading first
python main.py
```

### 3. Monitor Performance

#### Web Dashboard (Recommended)
```bash
# Start the web dashboard (in new terminal)
python create_monitoring_dashboard.py

# Or use the convenient script
run_web_dashboard.bat

# Open browser to: http://localhost:5000
```

#### Console Monitoring (Alternative)
```bash
# Start console monitor (in new terminal)
python monitor_bot.py
```

#### Manual Log Checking
```bash
# Check logs manually
tail -f logs/trading.log
```

### 4. Go Live (Only After Paper Trading Success)
```bash
# In .env file
PAPER_TRADING=false

# Start with small balance
python main.py
```

---

## 🌐 Web Dashboard Features

| Section | Description |
|---------|-------------|
| **Performance Metrics** | Total P&L, Win Rate, Trade Count, Balance |
| **Recent Trades** | Last 10 trades with entry/exit prices and reasons |
| **Live Events** | Real-time feed of discoveries, trades, errors |
| **Bot Status** | Running/Stopped indicator with uptime |
| **Auto-refresh** | Updates every 5 seconds automatically |

**Dashboard URL**: `http://localhost:5000`

### Hosting Options
- **Local**: `http://localhost:5000` (default)
- **Network**: `http://YOUR_IP:5000` (accessible from other devices)
- **Cloud**: Deploy to Heroku/DigitalOcean for remote access

---

## 🔍 Troubleshooting Common Issues

### 🐛 Bot Not Finding New Tokens
- Check Alchemy RPC connection
- Verify scan interval settings
- Check Jupiter API status
- Lower MIN_LIQUIDITY threshold

### 💸 Too Many Failed Transactions
- Increase gas settings
- Reduce position sizes
- Check network congestion
- Increase slippage tolerance

### 📉 Poor Performance
- Check contract analysis accuracy
- Review exit timing
- Analyze win/loss ratios
- Consider market conditions

### 🔧 Bot Crashes/Errors
- Check log files for details
- Verify all dependencies installed
- Check RPC rate limits
- Restart with paper trading

---

## 🎓 Advanced Tips

### 🧠 Learning From Performance
- Track which contract scores perform best
- Note optimal hold times for profits
- Identify best market conditions
- Learn from failed entries

### ⚡ Optimization Strategies
- **Morning rush**: More new tokens 9-11 AM EST
- **Low gas times**: Early morning or late night
- **Market sentiment**: Bull markets = more opportunities
- **Network congestion**: Avoid peak usage times

### 🔄 Continuous Improvement
- Adjust settings based on success rate
- Monitor and update contract analysis
- Refine entry criteria over time
- Stay updated on new token patterns

---

## 📞 Support & Community

- **Discord**: [Join trading community]
- **Telegram**: [Official support channel]  
- **GitHub**: [Report bugs and contribute]
- **Documentation**: [Full technical docs]

---

## ⚖️ Legal Disclaimer

This software is for educational purposes. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk and never invest more than you can afford to lose completely.

---

**Happy Aping! 🦍🚀**

*Remember: The best apes are patient, disciplined, and always manage their risk.*