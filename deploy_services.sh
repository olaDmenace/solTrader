#!/bin/bash
# Complete SolTrader Systemd Service Deployment
# Run these commands step by step

echo "🚀 Setting up SolTrader for 24/7 operation..."

# Step 1: Create logs directory
echo "📁 Creating logs directory..."
mkdir -p /home/trader/solTrader/logs

# Step 2: Stop any running background processes
echo "🛑 Stopping background processes..."
pkill -f "monitoring_dashboard" 2>/dev/null || echo "No dashboard process to stop"
pkill -f "main.py" 2>/dev/null || echo "No bot process to stop"

# Step 3: Create Bot Service File
echo "🤖 Creating bot service file..."
sudo tee /etc/systemd/system/soltrader-bot.service > /dev/null <<'EOF'
[Unit]
Description=SolTrader APE Bot
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/solTrader
Environment=PATH=/home/trader/solTrader/venv/bin
ExecStart=/home/trader/solTrader/venv/bin/python main.py
Restart=always
RestartSec=10

# Logging
StandardOutput=append:/home/trader/solTrader/logs/systemd-bot.log
StandardError=append:/home/trader/solTrader/logs/systemd-bot-error.log

[Install]
WantedBy=multi-user.target
EOF

# Step 4: Create Dashboard Service File  
echo "📊 Creating dashboard service file..."
sudo tee /etc/systemd/system/soltrader-dashboard.service > /dev/null <<'EOF'
[Unit]
Description=SolTrader Web Dashboard
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/solTrader
Environment=PATH=/home/trader/solTrader/venv/bin
ExecStart=/home/trader/solTrader/venv/bin/python create_monitoring_dashboard.py
Restart=always
RestartSec=10

# Logging
StandardOutput=append:/home/trader/solTrader/logs/systemd-dashboard.log
StandardError=append:/home/trader/solTrader/logs/systemd-dashboard-error.log

[Install]
WantedBy=multi-user.target
EOF

# Step 5: Reload systemd and enable services
echo "🔄 Reloading systemd and enabling services..."
sudo systemctl daemon-reload
sudo systemctl enable soltrader-bot
sudo systemctl enable soltrader-dashboard

# Step 6: Start services
echo "▶️ Starting services..."
sudo systemctl start soltrader-bot
sudo systemctl start soltrader-dashboard

# Step 7: Check service status
echo "✅ Checking service status..."
echo ""
echo "=== Bot Service Status ==="
sudo systemctl status soltrader-bot --no-pager -l
echo ""
echo "=== Dashboard Service Status ==="
sudo systemctl status soltrader-dashboard --no-pager -l

# Step 8: Test dashboard access
echo ""
echo "🌐 Testing dashboard access..."
sleep 3
curl -I https://bot.technicity.digital

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📊 Dashboard URL: https://bot.technicity.digital"
echo "📝 View logs with:"
echo "   tail -f /home/trader/solTrader/logs/trading.log"
echo "   sudo journalctl -f -u soltrader-bot"
echo "   sudo journalctl -f -u soltrader-dashboard"
echo ""
echo "🔧 Manage services with:"
echo "   sudo systemctl restart soltrader-bot"
echo "   sudo systemctl restart soltrader-dashboard"
echo "   sudo systemctl status soltrader-bot"
echo "   sudo systemctl status soltrader-dashboard"
