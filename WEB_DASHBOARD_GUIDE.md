# 🌐 SolTrader Web Dashboard Guide

## Overview
The SolTrader Web Dashboard provides a professional browser-based interface to monitor your APE bot in real-time. This replaces Telegram notifications with a rich web UI that updates automatically.

## Features
- 🦍 **Real-time Bot Status** - See if your APE bot is running or stopped
- 📊 **Performance Metrics** - Total P&L, Win Rate, Trade Count, Balance
- 📈 **Trade History** - Complete log of all your trades with entry/exit prices
- 📱 **Live Activity Feed** - Real-time events, errors, and token discoveries
- 🔄 **Auto-refresh** - Updates every 5 seconds automatically
- 🎨 **Professional UI** - Dark theme with color-coded information

## Quick Start

### 1. Install Dependencies
First, make sure Flask is installed:
```bash
pip install flask>=2.0.0
```

Or run the complete installation:
```bash
install_all_deps.bat
```

### 2. Start the Trading Bot
Open terminal #1 and start the bot:
```bash
python main.py
```

### 3. Start the Web Dashboard
Open terminal #2 and start the dashboard:
```bash
python create_monitoring_dashboard.py
```

Or use the convenient batch file:
```bash
run_web_dashboard.bat
```

### 4. Access the Dashboard
Open your browser and go to:
```
http://localhost:5000
```

## Dashboard Sections

### 📊 Performance Metrics
- **Total P&L**: Your cumulative profit/loss
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of trades executed
- **Balance**: Current account balance

### 📈 Recent Trades
Table showing your last 10 trades with:
- Token address (abbreviated)
- Entry and exit prices
- Profit/loss (color coded: green=profit, red=loss)
- Exit reason (momentum_exit, take_profit, etc.)
- Timestamp

### 📱 Recent Events
Live feed of the last 20 events including:
- 🟢 **Trade Entries** - When positions are opened
- 🔵 **Trade Exits** - When positions are closed
- 🟡 **Token Discoveries** - New tokens found for scanning
- 🔴 **Errors** - Any system errors
- 🟣 **Warnings** - Warning messages

## Hosting Options

### Local Development (Default)
- **URL**: `http://localhost:5000`
- **Access**: Only from your computer
- **Use Case**: Development and testing

### Network Access
To access from other devices on your network:

1. Find your computer's IP address:
```bash
ipconfig
```

2. Edit `create_monitoring_dashboard.py` line 382:
```python
app.run(host='0.0.0.0', port=5000, debug=False)
```

3. Access from other devices: `http://YOUR_IP:5000`

### Cloud Hosting
For remote access, you can deploy to:

#### Option 1: Heroku (Free Tier)
1. Install Heroku CLI
2. Create `Procfile`:
```
web: python create_monitoring_dashboard.py
```
3. Deploy:
```bash
git add .
git commit -m "Deploy dashboard"
heroku create your-app-name
git push heroku main
```

#### Option 2: DigitalOcean/AWS/GCP
1. Create a VPS instance
2. Install Python and dependencies
3. Run the dashboard with public IP
4. Configure firewall for port 5000

## Security Considerations

### For Production Use:
1. **Enable HTTPS** - Use SSL certificates
2. **Add Authentication** - Implement login system
3. **Firewall Rules** - Restrict access to specific IPs
4. **Environment Variables** - Store sensitive data securely

### Basic Password Protection:
Add to your dashboard before `@app.route('/')`:
```python
from functools import wraps

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username != 'admin' or auth.password != 'your_password':
            return Response('Login Required', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})
        return f(*args, **kwargs)
    return decorated

@app.route('/')
@require_auth
def dashboard():
    # ... existing code
```

## Troubleshooting

### Dashboard Won't Start
```bash
# Check if Flask is installed
python -c "import flask; print('Flask OK')"

# If not installed:
pip install flask

# Check if port 5000 is in use:
netstat -an | findstr :5000
```

### Can't Access Dashboard
1. Check firewall settings
2. Verify the bot is running and creating logs
3. Check `logs/trading.log` exists
4. Try accessing `http://127.0.0.1:5000` instead

### No Data Showing
1. Make sure `main.py` is running first
2. Check that logs are being created in `logs/trading.log`
3. Restart the dashboard to re-read logs

## Customization

### Change Refresh Rate
Edit line 333 in the HTML template:
```javascript
autoRefreshInterval = setInterval(refreshData, 3000); // 3 seconds
```

### Change Port
Edit line 382 in `create_monitoring_dashboard.py`:
```python
app.run(host='0.0.0.0', port=8080, debug=False)  # Use port 8080
```

### Add Custom Metrics
1. Edit the `parse_log_line` method to detect your events
2. Add new cards to the HTML template
3. Update the API endpoint to return new data

## Cost Considerations

### Minimum Investment
- **$50-$100**: Reasonable starting amount for paper trading transition
- **$10-$20**: Absolute minimum for testing (very small positions)
- **$500+**: Recommended for serious APE trading with good diversification

### Risk Management
- Start with **paper trading mode** first
- Never risk more than 1-2% per trade
- Set daily loss limits
- Monitor performance for at least a week before increasing size

## Advanced Features

### Performance Analytics
The dashboard tracks:
- Win/loss ratios
- Average trade duration
- Profit/loss distribution
- Token success rates

### Integration with External Tools
- Export data to CSV for analysis
- Send alerts to Discord/Slack
- Integration with portfolio trackers

## Support

If you encounter issues:
1. Check the logs in `logs/trading.log`
2. Verify all dependencies are installed
3. Ensure the bot is running in paper mode first
4. Test with small amounts before scaling up

Remember: **Never invest more than you can afford to lose!**