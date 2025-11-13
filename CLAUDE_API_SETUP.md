# 🤖 Claude API Server - Setup Instructions

This API server allows Claude to remotely monitor, troubleshoot, and manage your SolTrader bot securely.

---

## 📋 Prerequisites

- Server with SolTrader bot installed
- Python 3.8+ installed
- Port 8080 available (or choose different port)
- API key for authentication

---

## 🔐 Step 1: Generate Your Claude API Key

Generate a strong random API key:

```bash
# Option 1: Using Python
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Option 2: Using OpenSSL
openssl rand -base64 32

# Option 3: Using /dev/urandom
cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1
```

**Save this key securely!** You'll need to:
1. Add it to your server's `.env` file
2. Share it with Claude for authentication

---

## 📦 Step 2: Install Dependencies

```bash
cd /home/user/solTrader

# Activate virtual environment if using one
source venv/bin/activate

# Install API server dependencies
pip install -r claude_api_requirements.txt
```

---

## ⚙️ Step 3: Configure Environment

Add the API key to your `.env` file:

```bash
# Open .env file
nano .env

# Add this line (replace with your generated key):
CLAUDE_API_KEY=your-generated-key-here
```

**Important:** Make sure `.env` is in `.gitignore` (already done).

---

## 🚀 Step 4: Test the Server Locally

```bash
cd /home/user/solTrader
python3 claude_api_server.py
```

You should see:
```
============================================================
Claude API Server Starting...
============================================================
Bot Directory: /home/user/solTrader
Log File: /home/user/solTrader/logs/trading.log
API Key: ****************************abcd
Listening on: 0.0.0.0:8080
============================================================
```

Test health endpoint (no authentication needed):
```bash
curl http://localhost:8080/health
```

Test authenticated endpoint:
```bash
curl -H "X-API-Key: your-api-key-here" http://localhost:8080/status
```

If working correctly, press `Ctrl+C` to stop and proceed to production setup.

---

## 🔥 Step 5: Production Deployment (Choose One)

### Option A: systemd Service (Recommended)

Create a systemd service file:

```bash
sudo nano /etc/systemd/system/claude-api.service
```

Add this content (adjust paths if needed):

```ini
[Unit]
Description=Claude API Server for SolTrader Monitoring
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/user/solTrader
Environment="PATH=/home/user/solTrader/venv/bin"
EnvironmentFile=/home/user/solTrader/.env
ExecStart=/home/user/solTrader/venv/bin/python3 claude_api_server.py
Restart=always
RestartSec=10
StandardOutput=append:/home/user/solTrader/logs/claude-api-server.log
StandardError=append:/home/user/solTrader/logs/claude-api-server-error.log

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable claude-api

# Start the service
sudo systemctl start claude-api

# Check status
sudo systemctl status claude-api

# View logs
sudo journalctl -u claude-api -f
```

### Option B: Screen/tmux (Quick & Simple)

```bash
# Using screen
screen -S claude-api
cd /home/user/solTrader
python3 claude_api_server.py
# Press Ctrl+A then D to detach

# Reattach later
screen -r claude-api

# Or using tmux
tmux new -s claude-api
cd /home/user/solTrader
python3 claude_api_server.py
# Press Ctrl+B then D to detach

# Reattach later
tmux attach -t claude-api
```

### Option C: Nohup (Background Process)

```bash
cd /home/user/solTrader
nohup python3 claude_api_server.py > logs/claude-api.log 2>&1 &

# Check if running
ps aux | grep claude_api_server

# View logs
tail -f logs/claude-api.log

# Stop when needed
pkill -f claude_api_server
```

---

## 🌐 Step 6: Configure Firewall

Allow port 8080 through your firewall:

### For UFW (Ubuntu/Debian):
```bash
sudo ufw allow 8080/tcp
sudo ufw status
```

### For firewalld (CentOS/RHEL):
```bash
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload
```

### For AWS/DigitalOcean/Cloud VPS:
- Go to your cloud provider's security groups/firewall settings
- Add inbound rule: Port 8080, TCP, Source: Anywhere (or specific IP)

---

## 🔍 Step 7: Verify External Access

Test from your local machine (replace YOUR_SERVER_IP):

```bash
# Health check (no auth)
curl http://YOUR_SERVER_IP:8080/health

# Authenticated request
curl -H "X-API-Key: your-api-key-here" \
     http://YOUR_SERVER_IP:8080/status
```

---

## 🎯 Step 8: Share Credentials with Claude

Provide Claude with:

1. **Server URL:** `http://YOUR_SERVER_IP:8080`
2. **API Key:** Your generated key from Step 1

**How to share securely:**
- In your conversation with Claude, simply say:
  ```
  Claude API Server is ready:
  URL: http://123.456.789.0:8080
  API Key: [paste your key here]
  ```

Claude can now:
- ✅ Monitor logs in real-time
- ✅ Check bot status
- ✅ Execute troubleshooting commands
- ✅ Start/stop the bot
- ✅ Read configuration files
- ✅ Diagnose issues

---

## 📊 Available Endpoints

Once running, Claude can access:

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/health` | GET | Health check | No |
| `/execute` | POST | Execute command | Yes |
| `/logs` | GET | Get recent logs | Yes |
| `/logs/stream` | GET | Stream logs live | Yes |
| `/status` | GET | Bot status JSON | Yes |
| `/files` | GET | List files | Yes |
| `/file` | GET | Read specific file | Yes |
| `/bot/start` | POST | Start trading bot | Yes |
| `/bot/stop` | POST | Stop trading bot | Yes |
| `/bot/status` | GET | Check if bot running | Yes |

---

## 🔒 Security Features

The API server includes:

✅ **API Key Authentication** - All endpoints except /health require valid key
✅ **Rate Limiting** - Prevents abuse (100 requests/hour, adjustable)
✅ **Command Whitelist** - Only safe commands can be executed
✅ **Execution Timeout** - Commands limited to 60 seconds max
✅ **Directory Restrictions** - File access limited to bot directory
✅ **Audit Logging** - All access logged to `logs/claude_api.log`
✅ **No Write Access** - Claude can read and execute, not modify files directly

---

## 🛠️ Troubleshooting

### Server won't start:
```bash
# Check if API key is set
grep CLAUDE_API_KEY .env

# Check if port 8080 is already in use
sudo lsof -i :8080

# Check Python version
python3 --version  # Must be 3.8+

# Check dependencies
pip list | grep -E "Flask|Limiter"
```

### Can't access from outside:
```bash
# Check if server is listening
netstat -tulpn | grep 8080

# Check firewall
sudo ufw status  # or: sudo firewall-cmd --list-all

# Test locally first
curl http://localhost:8080/health
```

### Authentication fails:
```bash
# Verify API key in .env matches what you're using
cat .env | grep CLAUDE_API_KEY

# Check API logs for errors
tail -f logs/claude_api.log
```

### Commands not executing:
```bash
# Check allowed commands in claude_api_server.py
# The whitelist is in the ALLOWED_COMMANDS list

# Check API logs for blocked commands
grep "BLOCKED" logs/claude_api.log
```

---

## 🔧 Management Commands

```bash
# Check if API server is running
ps aux | grep claude_api_server

# View API access logs
tail -f logs/claude_api.log

# View server output logs
tail -f logs/claude-api-server.log

# Restart systemd service
sudo systemctl restart claude-api

# Stop systemd service
sudo systemctl stop claude-api

# Check service status
sudo systemctl status claude-api
```

---

## 📈 Usage Examples (For Claude)

Once Claude has your credentials, Claude can:

### Monitor Logs
```
GET http://YOUR_IP:8080/logs?lines=100
Headers: X-API-Key: your-key
```

### Check Bot Status
```
GET http://YOUR_IP:8080/status
Headers: X-API-Key: your-key
```

### Execute Command
```
POST http://YOUR_IP:8080/execute
Headers: X-API-Key: your-key
Body: {"command": "tail -50 logs/trading.log"}
```

### Check if Bot is Running
```
GET http://YOUR_IP:8080/bot/status
Headers: X-API-Key: your-key
```

---

## 🔄 Updating the API Server

When new versions are released:

```bash
# Pull latest code
git pull origin main

# Restart service
sudo systemctl restart claude-api

# Or if using screen/tmux
screen -r claude-api
# Ctrl+C to stop
python3 claude_api_server.py
# Ctrl+A then D to detach
```

---

## 🎉 You're All Set!

Once you've completed these steps:

1. ✅ API server running on your VPS
2. ✅ Port 8080 accessible from internet
3. ✅ API key generated and configured
4. ✅ Credentials shared with Claude

Claude can now provide real-time support, monitoring, and troubleshooting for your trading bot!

---

## 📞 Next Steps

Tell Claude:
```
Claude API Server is ready!
URL: http://YOUR_SERVER_IP:8080
API Key: your-generated-api-key

Please verify access and start monitoring.
```

Claude will then:
- Test connectivity
- Check bot status
- Review recent logs
- Identify any issues
- Provide recommendations

Happy trading! 🚀
