# 🚀 SolTrader APE Bot - Complete Setup Guide

This guide will walk you through setting up and running your enhanced SolTrader APE strategy bot from scratch.

---

## 📋 Prerequisites

- **Windows 10/11** (this guide is for Windows)
- **Python 3.9 or higher** installed
- **Git** (optional, for version control)
- **Basic command line knowledge**
- **Solana wallet** with some SOL for trading (or testing)

---

## 🛠️ Step 1: Install Python (If Not Already Installed)

### Check if Python is installed:
```cmd
python --version
```
or
```cmd
python3 --version
```

### If Python is not installed:
1. Go to [python.org/downloads](https://python.org/downloads)
2. Download Python 3.9+ for Windows
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Verify installation:
   ```cmd
   python --version
   ```

---

## 🏗️ Step 2: Set Up Virtual Environment

### Navigate to your project folder:
```cmd
cd C:\Users\ADMIN\Desktop\projects\solTrader
```

### Create virtual environment:
```cmd
python -m venv venv
```

### Activate virtual environment:
```cmd
# Windows Command Prompt
venv\Scripts\activate

# Windows PowerShell  
venv\Scripts\Activate.ps1

# Git Bash
source venv/Scripts/activate
```

**You should see `(venv)` at the beginning of your command prompt when activated.**

### To deactivate later (when you're done):
```cmd
deactivate
```

---

## 📦 Step 3: Install Dependencies

### With virtual environment activated, install required packages:
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

### If you get errors, install packages individually:
```cmd
pip install aiohttp==3.8.5
pip install anchorpy==0.18.0
pip install base58==2.1.1
pip install python-dotenv==1.0.0
pip install solana==0.30.2
pip install "solders>=0.18.0,<0.19.0"
pip install pytest==7.4.3
pip install pytest-asyncio==0.21.1
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scipy==1.11.3
pip install "async-timeout>=4.0.0"
pip install "python-telegram-bot>=20.0"
pip install "backoff>=2.2.1"
```

### Verify installation:
```cmd
pip list
```

---

## ⚙️ Step 4: Configuration Setup

### Create environment file:
Create a file called `.env` in the root directory (`C:\Users\ADMIN\Desktop\projects\solTrader\.env`):

```bash
# ===================
# REQUIRED SETTINGS
# ===================

# Alchemy RPC URL (get from https://alchemy.com)
ALCHEMY_RPC_URL=https://solana-mainnet.g.alchemy.com/v2/YOUR_API_KEY_HERE

# Your Solana wallet address
WALLET_ADDRESS=YOUR_SOLANA_WALLET_ADDRESS_HERE

# ===================
# TRADING SETTINGS
# ===================

# Start with paper trading (HIGHLY RECOMMENDED)
PAPER_TRADING=true
INITIAL_PAPER_BALANCE=100.0

# APE Strategy Settings
MOMENTUM_EXIT_ENABLED=true
MIN_CONTRACT_SCORE=70
MAX_POSITION_PER_TOKEN=1.0
MAX_SIMULTANEOUS_POSITIONS=5
MAX_HOLD_TIME_MINUTES=180

# Risk Management
MAX_DAILY_TRADES=15
MAX_DAILY_LOSS=2.0
STOP_LOSS_PERCENTAGE=0.15
TAKE_PROFIT_PERCENTAGE=0.5

# Speed Settings
SCAN_INTERVAL=5
POSITION_MONITOR_INTERVAL=3
PRIORITY_FEE_MULTIPLIER=2.0

# ===================
# OPTIONAL SETTINGS
# ===================

# Telegram notifications (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Debug mode
DEBUG=false
```

### Get Required API Keys:

#### 1. Alchemy RPC URL:
1. Go to [alchemy.com](https://alchemy.com)
2. Sign up for free account
3. Create a new app for "Solana" → "Mainnet"
4. Copy the HTTPS URL and paste it in your `.env` file

#### 2. Solana Wallet Address:
- Use your existing Solana wallet address (Phantom, Solflare, etc.)
- **For testing**: You can use any valid Solana address initially

---

## 🧪 Step 5: Test the Setup

### Run basic environment check:
```cmd
python check_env.py
```

### Run the comprehensive test suite:
```cmd
python test_ape_strategy.py
```

**Expected output:**
```
🧪 Starting APE Strategy Test Suite...
🔧 Setting up test environment...
✅ Test environment ready
📊 Testing Position Management...
✅ Basic Position Creation: PASS
✅ Position Price Update: PASS
📈 Testing Momentum Calculations...
✅ Momentum Calculation: PASS
🚪 Testing Exit Logic...
✅ Momentum Reversal Exit: PASS
✅ Profit Protection Exit: PASS
✅ Time Limit Exit: PASS
✅ Continue Holding: PASS
🎉 All tests passed - APE strategy is ready!
```

### If tests fail:
1. Check that all dependencies are installed
2. Verify your `.env` file is correctly configured
3. Make sure virtual environment is activated

---

## 🎮 Step 6: Start Paper Trading (RECOMMENDED)

### Important: Always start with paper trading!

```cmd
# Make sure virtual environment is activated
venv\Scripts\activate

# Start the bot in paper trading mode
python main.py
```

**Expected output:**
```
2024-01-01 10:00:00 - INFO - Starting trading bot...
2024-01-01 10:00:00 - INFO - Paper trading settings:
2024-01-01 10:00:00 - INFO - Initial balance: 100.0
2024-01-01 10:00:00 - INFO - Max position size: 1.0
2024-01-01 10:00:00 - INFO - Bot initialized in Paper trading mode
2024-01-01 10:00:00 - INFO - 🔍 Scanning for new tokens...
```

### Monitor the bot:
- Watch the console output for activity
- Look for new token discoveries
- Monitor paper trading performance
- Check for any errors

### Stop the bot:
```cmd
Ctrl + C
```

---

## 📊 Step 7: Monitor Performance

### View logs:
```cmd
# View recent activity
type logs\trading.log

# Follow live logs (if using Git Bash)
tail -f logs/trading.log
```

### Check performance metrics:
The bot will display:
- Tokens scanned
- Positions opened/closed
- P&L results
- Exit reasons
- Success rates

### Sample successful paper trading output:
```
✅ Successfully bought 1.0 SOL worth of TOKEN123ABC
📈 Position monitoring: TOKEN123ABC at $1.25 (+25%)
🚪 Position closed due to momentum_reversal: +0.25 SOL profit
📊 Current paper balance: 100.25 SOL
```

---

## 🎯 Step 8: Fine-Tune Settings (Optional)

### Based on paper trading results, you can adjust settings in `.env`:

#### More Conservative:
```bash
MIN_CONTRACT_SCORE=80
MAX_POSITION_PER_TOKEN=0.5
MAX_DAILY_TRADES=10
```

#### More Aggressive:
```bash
MIN_CONTRACT_SCORE=60
MAX_POSITION_PER_TOKEN=2.0
MAX_DAILY_TRADES=20
SCAN_INTERVAL=3
```

#### Restart after changes:
```cmd
# Stop bot with Ctrl+C
# Start again
python main.py
```

---

## 🚨 Step 9: Go Live (Only After Successful Paper Trading)

### ⚠️ WARNING: Only proceed if:
- ✅ Paper trading was profitable for 1+ weeks
- ✅ You understand the risks
- ✅ You can afford to lose the trading capital
- ✅ Bot has proven performance

### Switch to live trading:
1. **Edit `.env` file:**
   ```bash
   PAPER_TRADING=false
   ```

2. **Start with small balance:**
   - Transfer only 10-50 SOL to start
   - Don't use your entire wallet

3. **Run the bot:**
   ```cmd
   python main.py
   ```

4. **Monitor closely:**
   - Watch first few trades carefully
   - Be ready to stop if something goes wrong
   - Check wallet balance regularly

---

## 📱 Step 10: Set Up Telegram Notifications (Optional)

### Create Telegram bot:
1. Message `@BotFather` on Telegram
2. Send `/newbot`
3. Follow instructions to create bot
4. Copy the bot token

### Get your chat ID:
1. Message your bot
2. Visit: `https://api.telegram.org/bot<YourBOTToken>/getUpdates`
3. Find your chat ID in the response

### Add to `.env`:
```bash
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyZ
TELEGRAM_CHAT_ID=987654321
```

### Test notifications:
```cmd
python main.py
```

You should receive Telegram messages about bot status and trades.

---

## 🛠️ Troubleshooting Common Issues

### Issue: "python: command not found"
**Solution:**
```cmd
# Try with python3
python3 main.py

# Or reinstall Python with PATH option checked
```

### Issue: "No module named 'solana.transaction'"
**Solution:**
```cmd
# Activate virtual environment first
venv\Scripts\activate

# Then install dependencies
pip install -r requirements.txt
```

### Issue: Virtual environment activation fails
**Solution:**
```cmd
# For PowerShell, enable scripts first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate:
venv\Scripts\Activate.ps1
```

### Issue: "Settings validation failed"
**Solution:**
- Check your `.env` file exists
- Verify ALCHEMY_RPC_URL is valid
- Ensure WALLET_ADDRESS is a valid Solana address

### Issue: Bot finds no tokens
**Solution:**
- Lower `MIN_LIQUIDITY` in `.env`
- Check Alchemy API key is working
- Verify internet connection

### Issue: High gas fees or failed transactions
**Solution:**
- Increase `MAX_GAS_PRICE` in `.env`
- Reduce `MAX_TRADES_PER_DAY`
- Try trading during less busy hours

---

## 📋 Daily Operation Checklist

### Before starting the bot:
- [ ] Virtual environment activated
- [ ] Recent logs checked
- [ ] Wallet balance verified
- [ ] Market conditions assessed

### While running:
- [ ] Monitor console output
- [ ] Check for errors
- [ ] Review trading performance
- [ ] Watch for circuit breaker triggers

### After stopping:
- [ ] Review daily P&L
- [ ] Check trade success rate
- [ ] Analyze exit reasons
- [ ] Plan any setting adjustments

---

## 🚨 Emergency Procedures

### Stop bot immediately:
```cmd
Ctrl + C
```

### Emergency stop via Telegram:
```
/emergency
```

### Reset paper trading balance:
```bash
# In .env file
INITIAL_PAPER_BALANCE=100.0
```

### Backup important data:
- Copy `.env` file
- Save `logs/` folder
- Export any trading history

---

## 📞 Support Resources

### Log Files:
- `logs/trading.log` - Main trading log
- `logs/error.log` - Error log (if it exists)

### Configuration Files:
- `.env` - Your settings
- `pyrightconfig.json` - Type checking config
- `requirements.txt` - Dependencies

### Test Files:
- `test_ape_strategy.py` - Run anytime to test
- `check_env.py` - Verify environment setup

### Documentation:
- `docs/APE_STRATEGY_GUIDE.md` - Strategy details
- `ENHANCEMENTS_SUMMARY.md` - Technical changes

---

## 🎉 You're Ready to APE!

Your SolTrader APE bot is now configured and ready to:
- 🔍 Scan for new Solana tokens every 5 seconds
- ⚡ Enter positions within seconds of discovery  
- 📈 Hold positions as long as momentum continues
- 🚪 Exit quickly when momentum reverses
- 🛡️ Protect your capital with smart risk management

**Remember**: Start with paper trading, monitor performance closely, and never risk more than you can afford to lose!

**Happy APEing! 🦍🚀**

---

*Need help? Check the troubleshooting section above or review the logs for specific error messages.*