# 🚀 SolTrader - Complete Setup & Remote Access Guide

## Current Status: NOT RUNNING
**Last Activity**: August 21, 2025 (2+ months ago)
**Issue**: Missing .env configuration prevented bot from starting
**6 Positions**: Stuck "open" since August 20-21 (monitoring never ran)

---

## Part 1: Getting the Bot Running

### Step 1: Configure .env File

I've created `/home/user/solTrader/.env` with template. You need to fill in:

```bash
# Edit the .env file
nano /home/user/solTrader/.env

# OR use your preferred editor
vi /home/user/solTrader/.env
```

**Required values to replace:**
1. `ALCHEMY_RPC_URL` - Get from https://dashboard.alchemy.com/
   - Create Solana Mainnet app
   - Copy the HTTPS URL

2. `WALLET_ADDRESS` - Your Phantom/Solflare public address

3. `SOLANA_TRACKER_KEY` - Get from https://solanatracker.io/
   - Free tier available
   - Sign up and copy API key

4. `EMAIL_USER` & `EMAIL_PASSWORD` (Optional but recommended)
   - For Gmail, use App Password: https://support.google.com/accounts/answer/185833

### Step 2: Install Dependencies

```bash
cd /home/user/solTrader

# Create virtual environment if not exists
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements_updated.txt

# Verify installation
python verify_setup.py
```

### Step 3: Clear Old State

```bash
# The bot_data.json has stale August data
# Rename it to backup
mv bot_data.json bot_data_august_backup.json

# Bot will create fresh state when it starts
```

### Step 4: Start the Bot (Paper Trading)

```bash
# Make sure .env has PAPER_TRADING=true
python main.py

# You should see:
# [INIT] Initializing SolTrader APE Bot...
# [OK] Alchemy connection successful
# [READY] Bot startup complete - starting trading strategy...
```

---

## Part 2: Remote Log Monitoring Options

### Option A: Simple HTTP Log Server (Recommended for Quick Setup)

Create a simple Flask endpoint to stream logs:

```bash
# Create log server script
cat > /home/user/solTrader/log_server.py << 'EOF'
#!/usr/bin/env python3
"""
Simple log streaming server for remote monitoring
Access via: http://YOUR_IP:8080/logs
"""
from flask import Flask, Response, jsonify
import os
import time
from pathlib import Path

app = Flask(__name__)
LOG_DIR = Path("/home/user/solTrader/logs")

@app.route('/logs')
def get_logs():
    """Return last 1000 lines of trading.log"""
    log_file = LOG_DIR / "trading.log"
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return Response('\n'.join(lines[-1000:]), mimetype='text/plain')
    return "No logs found", 404

@app.route('/logs/stream')
def stream_logs():
    """Stream logs in real-time"""
    def generate():
        log_file = LOG_DIR / "trading.log"
        with open(log_file, 'r') as f:
            # Start from end
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                yield f"data: {line}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/status')
def get_status():
    """Get bot status from bot_data.json"""
    status_file = Path("/home/user/solTrader/bot_data.json")
    if status_file.exists():
        with open(status_file, 'r') as f:
            import json
            return jsonify(json.load(f))
    return jsonify({"error": "No status file"}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "timestamp": time.time(),
        "logs_exist": (LOG_DIR / "trading.log").exists()
    })

if __name__ == '__main__':
    # Run on all interfaces, port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
EOF

# Make executable
chmod +x /home/user/solTrader/log_server.py

# Install Flask
pip install flask

# Run log server in background
nohup python log_server.py > log_server.log 2>&1 &

# Now access from anywhere:
# http://YOUR_SERVER_IP:8080/logs - Last 1000 lines
# http://YOUR_SERVER_IP:8080/status - Bot status
# http://YOUR_SERVER_IP:8080/logs/stream - Real-time stream
```

### Option B: Cloud Logging (Papertrail)

```bash
# Install Papertrail handler
pip install logging-papertrail

# Add to main.py setup_logging() function:
from logging.handlers import SysLogHandler

papertrail_host = os.getenv('PAPERTRAIL_HOST')
papertrail_port = int(os.getenv('PAPERTRAIL_PORT', 0))

if papertrail_host and papertrail_port:
    syslog = SysLogHandler(address=(papertrail_host, papertrail_port))
    syslog.setLevel(logging.INFO)
    logger.addHandler(syslog)

# Then in .env add:
PAPERTRAIL_HOST=logs.papertrailapp.com
PAPERTRAIL_PORT=YOUR_PORT

# View logs at: https://papertrailapp.com/
```

### Option C: SSH + tmux (Most Control)

```bash
# Start bot in tmux session
tmux new -s soltrader

# Inside tmux:
cd /home/user/solTrader
source venv/bin/activate
python main.py

# Detach: Ctrl+B then D

# Later, reattach:
tmux attach -t soltrader

# Or view logs remotely:
ssh user@yourserver "tail -f /home/user/solTrader/logs/trading.log"
```

### Option D: Simple rsync + Local Viewing

```bash
# On your local machine, sync logs periodically:
while true; do
    rsync -avz user@yourserver:/home/user/solTrader/logs/ ./soltrader_logs/
    sleep 30
done

# Then tail locally:
tail -f ./soltrader_logs/trading.log
```

---

## Part 3: Setting Up Telegram Bot Monitoring for Claude

If you want me (Claude) to monitor your bot automatically:

### Option 1: Webhook Integration

```python
# Add to main.py after successful trades/errors:
import requests

CLAUDE_WEBHOOK = os.getenv('CLAUDE_WEBHOOK_URL')

def notify_claude(event_type, data):
    """Send event to Claude monitoring endpoint"""
    if CLAUDE_WEBHOOK:
        requests.post(CLAUDE_WEBHOOK, json={
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        })

# Usage:
notify_claude("position_opened", {"token": token_address, "size": size})
notify_claude("position_closed", {"token": token_address, "pnl": pnl})
notify_claude("error", {"message": str(error)})
```

### Option 2: Shared Google Sheet/Airtable

Export bot status to a shared spreadsheet I can access.

### Option 3: Discord/Telegram Bot

Set up a bot that posts updates to a channel I can monitor.

---

## Part 4: Debugging Checklist

### When Bot Won't Start:

```bash
# Check .env exists and has values
cat .env | grep -v "^#" | grep "="

# Check Python version
python3 --version  # Should be 3.8+

# Check dependencies
pip list | grep -E "aiohttp|solana|requests"

# Check logs directory
ls -la logs/

# Try running with verbose output
python -v main.py
```

### When Positions Won't Close:

```bash
# Check if monitoring loop is running
# Look for these log patterns:
grep "\[MONITOR\]" logs/trading.log
grep "\[HOLD\]" logs/trading.log
grep "\[EXIT\]" logs/trading.log

# Check if position updates are happening
grep "position_update" logs/trading.log | tail -10

# Check current positions
python -c "
import json
with open('bot_data.json', 'r') as f:
    data = json.load(f)
    print(f'Open positions: {len([t for t in data[\"trades\"] if t[\"status\"] == \"open\"])}')
"
```

### When No Trades Execute:

```bash
# Check scanner is finding tokens
grep "\[SCAN\]" logs/trading.log

# Check signal generation
grep "signal_generated" logs/trading.log

# Check API connections
grep "connection" logs/trading.log | grep -E "success|fail"

# Check wallet balance (paper trading)
grep "balance" logs/trading.log | tail -5
```

---

## Part 5: Production Deployment

### Using systemd (Auto-restart on crash)

```bash
# Create service file
sudo nano /etc/systemd/system/soltrader.service

# Add:
[Unit]
Description=SolTrader Trading Bot
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/user/solTrader
Environment="PATH=/home/user/solTrader/venv/bin"
ExecStart=/home/user/solTrader/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable soltrader
sudo systemctl start soltrader

# Check status
sudo systemctl status soltrader

# View logs
sudo journalctl -u soltrader -f
```

### Using Docker (Portable)

```bash
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app
COPY requirements_updated.txt .
RUN pip install --no-cache-dir -r requirements_updated.txt

COPY . .

CMD ["python", "main.py"]
EOF

# Build and run
docker build -t soltrader .
docker run -d --name soltrader --restart unless-stopped \
    -v $(pwd)/.env:/app/.env \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/data:/app/data \
    soltrader

# View logs
docker logs -f soltrader
```

---

## Part 6: Monitoring Dashboard URLs

Once log server is running, bookmark these:

```
http://YOUR_IP:8080/health        # Quick health check
http://YOUR_IP:8080/status        # Full bot status
http://YOUR_IP:8080/logs          # Last 1000 log lines
http://YOUR_IP:8080/logs/stream   # Real-time log stream
```

You can share these URLs with me (Claude) for monitoring!

---

## Part 7: Quick Reference Commands

```bash
# Start bot (foreground)
python main.py

# Start bot (background)
nohup python main.py > bot.log 2>&1 &

# View logs live
tail -f logs/trading.log

# Check if bot is running
ps aux | grep main.py

# Stop bot (graceful)
pkill -SIGTERM -f main.py

# Force stop
pkill -9 -f main.py

# Check last 50 trades
tail -50 logs/trading.log | grep -E "\\[BUY\\]|\\[SELL\\]|\\[EXIT\\]"

# Check current balance
grep "Balance" logs/trading.log | tail -1

# Check errors
grep "ERROR" logs/trading.log | tail -20
```

---

## Next Steps:

1. ✅ **.env created** - Fill in your API keys
2. ⏳ **Install dependencies** - Run pip install
3. ⏳ **Clear old state** - Backup old bot_data.json
4. ⏳ **Start bot** - python main.py
5. ⏳ **Set up log access** - Choose Option A-D above
6. ⏳ **Monitor for 24 hours** - Verify trades execute and close
7. ⏳ **Share log URL with Claude** - For collaborative debugging

**Questions to decide:**
- What's your server setup? (VPS provider, home server, etc.)
- Do you have a public IP/domain?
- Preferred log monitoring method?
- Want me to create the log server code ready to run?

---

## Contact Points for Claude Access

To give me access to monitor your bot, you can:

1. **Share log server URL** - I can periodically check status
2. **Discord webhook** - Bot posts updates, I monitor channel
3. **Telegram bot** - Same as Discord
4. **Email forwards** - Forward bot alert emails to a shared inbox
5. **API endpoint** - I call your endpoint to get status

**Most practical**: Option 1 (log server) + Discord/Telegram webhooks

Let me know which approach you prefer!
