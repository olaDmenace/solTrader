#!/bin/bash
echo "🧪 Testing SolTrader Bot Manually..."
cd /home/trader/solTrader
source venv/bin/activate
export PYTHONPATH=/home/trader/solTrader
export PYTHONUNBUFFERED=1

echo "Starting bot in test mode..."
python main.py
