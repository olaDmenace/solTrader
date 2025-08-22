#!/usr/bin/env python3
"""
Simple Web Dashboard for SolTrader APE Bot Monitoring
Creates a local web interface to monitor bot performance without Telegram
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import threading
import time
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.utils.simple_token_namer import SimpleTokenNamer

try:
    from flask import Flask, render_template_string, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

class BotMonitor:
    """Simple bot monitoring system"""
    
    def __init__(self, log_file="logs/trading.log", data_file="dashboard_data.json"):
        self.log_file = log_file
        self.data_file = data_file
        self.app = None
        self.token_namer = SimpleTokenNamer()
        
        # Initialize data structure
        self.data = {
            "status": "stopped",
            "start_time": None,
            "trades": [],
            "positions": [],
            "performance": {
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "balance": 100.0
            },
            "recent_events": [],
            "last_update": datetime.now().isoformat()
        }
        
        self.load_data()
    
    def load_data(self):
        """Load existing data"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self.data.update(json.load(f))
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def save_data(self):
        """Save current data"""
        try:
            self.data["last_update"] = datetime.now().isoformat()
            with open(self.data_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def update_from_logs(self):
        """Parse log file for updates"""
        try:
            if not os.path.exists(self.log_file):
                # Create logs directory and file if it doesn't exist
                os.makedirs("logs", exist_ok=True)
                with open(self.log_file, 'w') as f:
                    f.write(f"{datetime.now().isoformat()} - Monitor - Log file created\n")
                return
                
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Get recent log entries (last 100 lines)
            recent_lines = lines[-100:] if len(lines) > 100 else lines
            
            for line in recent_lines:
                self.parse_log_line(line.strip())
                
        except Exception as e:
            print(f"Error reading logs: {e}")
    
    def parse_log_line(self, line: str):
        """Parse individual log line"""
        try:
            timestamp = datetime.now().isoformat()
            
            if "Paper trading started" in line or "Live trading started" in line:
                self.data["status"] = "running"
                self.data["start_time"] = timestamp
                
            elif "Position opened" in line or "bought" in line.lower():
                event = {
                    "type": "trade_entry",
                    "message": line,
                    "timestamp": timestamp
                }
                self.data["recent_events"].append(event)
                # Don't increment here - use JSON data instead
                
            elif "Position closed" in line or "sold" in line.lower():
                event = {
                    "type": "trade_exit", 
                    "message": line,
                    "timestamp": timestamp
                }
                self.data["recent_events"].append(event)
                
                # Try to extract P&L
                words = line.split()
                for i, word in enumerate(words):
                    if word.startswith('$') or word.startswith('+') or word.startswith('-'):
                        try:
                            pnl = float(word.replace('$', '').replace('+', ''))
                            self.data["performance"]["total_pnl"] += pnl
                            
                            # Update balance
                            self.data["performance"]["balance"] += pnl
                            
                            # Update win rate
                            wins = sum(1 for t in self.data["trades"] if t.get("pnl", 0) > 0)
                            total = len(self.data["trades"])
                            if total > 0:
                                self.data["performance"]["win_rate"] = (wins / total) * 100
                            break
                        except:
                            pass
                    
            elif any(indicator in line for indicator in [
                "New token", "token found", "Processing dict token", "Processing object token",
                "Strong signal found", "Creating entry signal", "Successfully processed opportunity",
                "Opportunity alert emitted", "Starting opportunity scan", "Scanner returned"
            ]):
                event = {
                    "type": "token_discovery",
                    "message": line,
                    "timestamp": timestamp
                }
                self.data["recent_events"].append(event)
                
            elif "ERROR" in line:
                event = {
                    "type": "error",
                    "message": line,
                    "timestamp": timestamp
                }
                self.data["recent_events"].append(event)
                
            elif "WARNING" in line:
                event = {
                    "type": "warning",
                    "message": line,
                    "timestamp": timestamp
                }
                self.data["recent_events"].append(event)
                
            # Keep only recent events (last 50)
            if len(self.data["recent_events"]) > 50:
                self.data["recent_events"] = self.data["recent_events"][-50:]
                
        except Exception as e:
            print(f"Error parsing log line: {e}")
    
    def add_trade(self, token_address: str, entry_price: float, exit_price: float, 
                  pnl: float, reason: str):
        """Add a completed trade"""
        trade = {
            "token": token_address,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        self.data["trades"].append(trade)
        self.data["performance"]["total_trades"] += 1
        self.data["performance"]["total_pnl"] += pnl
        
        # Calculate win rate
        wins = sum(1 for t in self.data["trades"] if t["pnl"] > 0)
        self.data["performance"]["win_rate"] = (wins / len(self.data["trades"])) * 100
        
        self.save_data()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status with token names"""
        # Load fresh data from the actual dashboard JSON file
        self.load_data()
        self.update_from_logs()  # Still get recent events from logs
        
        # Extract unique token addresses for name generation
        unique_tokens = set()
        trades = self.data.get('trades', [])
        
        for trade in trades:
            token_addr = trade.get('token_address', '')
            if token_addr and not token_addr.startswith('TEST'):
                unique_tokens.add(token_addr)
        
        # Generate token names quickly (no external API calls)
        token_info_batch = self.token_namer.get_batch_token_info(list(unique_tokens))
        
        # Enrich trades with token names
        enriched_trades = []
        for trade in trades:
            token_addr = trade.get('token_address', '')
            token_info = token_info_batch.get(token_addr, {})
            
            enriched_trade = trade.copy()
            enriched_trade.update({
                'token_name': token_info.get('name', f'Token {token_addr[:8]}...'),
                'token_symbol': token_info.get('symbol', token_addr[:6]),
                'token_source': token_info.get('source', 'unknown')
            })
            enriched_trades.append(enriched_trade)
        
        self.data['trades'] = enriched_trades
        
        # Process active positions for display from trading data
        active_positions = []
        closed_tokens = set()
        
        # Find latest position updates and closed positions
        for activity in self.data.get('activity', []):
            if activity.get('type') == 'position_update':
                token = activity['data']['token']
                token_info = token_info_batch.get(token, {})
                
                # Convert to display format
                position = {
                    'token': token,
                    'token_name': token_info.get('name', f'Token {token[:8]}...'),
                    'token_symbol': token_info.get('symbol', token[:6]),
                    'entry_price': activity['data'].get('entry_price', 0),
                    'current_price': activity['data'].get('current_price', 0),
                    'age_minutes': activity['data'].get('age_minutes', 0),
                    'pnl_percentage': activity['data'].get('pnl_percentage', 0),
                    'unrealized_pnl': activity['data'].get('unrealized_pnl', 0)
                }
                active_positions.append(position)
            elif activity.get('type') == 'position_closed':
                token = activity['data']['token']
                closed_tokens.add(token)
        
        # Remove closed positions from active positions
        active_positions = [pos for pos in active_positions if pos['token'] not in closed_tokens]
        
        # Add processed active positions to data
        self.data['positions'] = active_positions
        
        # Ensure performance data is properly formatted for dashboard
        if 'performance' not in self.data:
            self.data['performance'] = {}
        
        # Map the actual performance data from dashboard_data.json
        perf_data = self.data.get('performance', {})
        self.data['performance'].update({
            'total_pnl': perf_data.get('total_pnl', 0),
            'win_rate': perf_data.get('win_rate', 0),
            'total_trades': perf_data.get('total_trades', 0),
            'balance': perf_data.get('balance', 100.0),
            'unrealized_pnl': perf_data.get('unrealized_pnl', 0),
            'open_positions': perf_data.get('open_positions', len(active_positions))
        })
        
        return self.data

# HTML Template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🦍 SolTrader APE Bot Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="3">
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            background: #0a0e27; 
            color: #00ff41; 
            margin: 0; 
            padding: 20px; 
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            color: #ff6b35; 
        }
        .status { 
            display: inline-block; 
            padding: 5px 15px; 
            border-radius: 20px; 
            font-weight: bold; 
        }
        .running { background: #28a745; color: white; }
        .stopped { background: #dc3545; color: white; }
        .card { 
            background: #1a1a2e; 
            border: 1px solid #16213e; 
            border-radius: 10px; 
            padding: 20px; 
            margin: 20px 0; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.3); 
        }
        .metrics { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; 
        }
        .metric { 
            text-align: center; 
            padding: 15px; 
            background: #16213e; 
            border-radius: 8px; 
        }
        .metric-value { 
            font-size: 2em; 
            font-weight: bold; 
            color: #00ff41; 
        }
        .metric-label { 
            color: #8892b0; 
            margin-top: 5px; 
        }
        .trades-table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 10px; 
        }
        .trades-table th, .trades-table td { 
            border: 1px solid #16213e; 
            padding: 8px; 
            text-align: left; 
        }
        .trades-table th { 
            background: #16213e; 
            color: #ff6b35; 
        }
        .profit { color: #28a745; }
        .loss { color: #dc3545; }
        .refresh-btn { 
            background: #ff6b35; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 5px; 
            cursor: pointer; 
            margin: 10px; 
        }
        .auto-refresh { 
            position: fixed; 
            top: 20px; 
            right: 20px; 
            background: #16213e; 
            padding: 10px; 
            border-radius: 5px; 
        }
        .events { 
            max-height: 200px; 
            overflow-y: auto; 
            background: #0f0f0f; 
            padding: 10px; 
            border-radius: 5px; 
        }
    </style>
</head>
<body>
    <div class="auto-refresh">
        <label>
            <input type="checkbox" id="autoRefresh" checked> Auto-refresh (3s)
        </label>
    </div>
    
    <div class="header">
        <h1>🦍 SolTrader APE Bot Dashboard</h1>
        <span class="status" id="status">{{ data.get('status', 'Unknown') }}</span>
        <button class="refresh-btn" onclick="refreshData()">🔄 Refresh</button>
    </div>
    
    <div class="card">
        <h2>📊 Performance Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">${{ "%.6f"|format(data.get('performance', {}).get('total_pnl', 0)) }}</div>
                <div class="metric-label">Total P&L</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f"|format(data.get('performance', {}).get('win_rate', 0)) }}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ data.get('performance', {}).get('total_trades', 0) }}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value">${{ "%.2f"|format(data.get('performance', {}).get('balance', 0)) }}</div>
                <div class="metric-label">Balance</div>
            </div>
            <div class="metric">
                <div class="metric-value">${{ "%.6f"|format(data.get('performance', {}).get('unrealized_pnl', 0)) }}</div>
                <div class="metric-label">Unrealized P&L</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ data.get('performance', {}).get('active_positions', 0) }}</div>
                <div class="metric-label">Open Positions</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>📊 Active Positions</h2>
        <table class="trades-table">
            <thead>
                <tr>
                    <th>Token</th>
                    <th>Entry Price</th>
                    <th>Current Price</th>
                    <th>Age</th>
                    <th>P&L</th>
                    <th>Unrealized P&L</th>
                </tr>
            </thead>
            <tbody>
                {% for position in data.get('positions', []) %}
                <tr>
                    <td title="{{ position.get('token', 'N/A') }} - {{ position.get('token_name', 'Unknown') }}">
                        <strong>{{ position.get('token_symbol', position.get('token', 'N/A')[:8]) }}</strong><br>
                        <small style="color: #8892b0;">{{ position.get('token_name', 'Unknown')[:20] }}{% if position.get('token_name', '')|length > 20 %}...{% endif %}</small>
                    </td>
                    <td>${{ "%.6f"|format(position.get('entry_price', 0)) }}</td>
                    <td>${{ "%.6f"|format(position.get('current_price', 0)) }}</td>
                    <td>{{ "%.1f"|format(position.get('age_minutes', 0)) }}m</td>
                    <td class="{% if position.get('pnl_percentage', 0) > 0 %}profit{% else %}loss{% endif %}">
                        {{ "%.2f"|format(position.get('pnl_percentage', 0)) }}%
                    </td>
                    <td class="{% if position.get('unrealized_pnl', 0) > 0 %}profit{% else %}loss{% endif %}">
                        ${{ "%.6f"|format(position.get('unrealized_pnl', 0)) }}
                    </td>
                </tr>
                {% else %}
                <tr>
                    <td colspan="6" style="text-align: center; color: #8892b0;">No active positions</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    <div class="card">
        <h2>📈 Recent Trades</h2>
        <table class="trades-table">
            <thead>
                <tr>
                    <th>Token</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>P&L</th>
                    <th>Reason</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
                {% for trade in data.get('trades', [])[-10:] %}
                <tr>
                    <td title="{{ trade.get('token_address', 'N/A') }} - {{ trade.get('token_name', 'Unknown') }}">
                        <strong>{{ trade.get('token_symbol', trade.get('token_address', 'N/A')[:8]) }}</strong><br>
                        <small style="color: #8892b0;">{{ trade.get('token_name', 'Unknown')[:20] }}{% if trade.get('token_name', '')|length > 20 %}...{% endif %}</small>
                    </td>
                    <td>${{ "%.4f"|format(trade.get('price', 0)) }}</td>
                    <td>${{ "%.4f"|format(trade.get('exit_price', 0)) }}</td>
                    <td class="{% if trade.get('pnl', 0) > 0 %}profit{% else %}loss{% endif %}">
                        ${{ "%.6f"|format(trade.get('pnl', 0)) }}
                    </td>
                    <td>{{ trade.get('reason', 'N/A') }}</td>
                    <td>{{ trade.get('timestamp', 'N/A')[:19] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    <div class="card">
        <h2>📱 Recent Events</h2>
        <div class="events">
            {% for event in data.get('recent_events', [])[-20:] %}
            <div>
                <strong>{{ event.get('timestamp', 'N/A')[:19] }}</strong> - {{ event.get('message', 'N/A') }}
            </div>
            {% endfor %}
            
            {% for activity in data.get('activity', [])[-10:] %}
                {% if activity.get('type') == 'position_update' %}
                <div style="color: #00ff41;">
                    <strong>{{ activity.get('timestamp', 'N/A')[:19] }}</strong> - 📊 Position Update: {{ activity.get('data', {}).get('token', 'N/A')[:8] }}... P&L: {{ "%.2f"|format(activity.get('data', {}).get('pnl_percentage', 0)) }}%
                </div>
                {% elif activity.get('type') == 'position_closed' %}
                <div style="color: #ff6b35;">
                    <strong>{{ activity.get('timestamp', 'N/A')[:19] }}</strong> - 🔴 Position Closed: {{ activity.get('data', {}).get('token', 'N/A')[:8] }}... Reason: {{ activity.get('data', {}).get('exit_reason', 'N/A') }}
                </div>
                {% elif activity.get('type') in ['scan_started', 'scan_completed', 'signal_generated'] %}
                <div style="color: #8892b0;">
                    <strong>{{ activity.get('timestamp', 'N/A')[:19] }}</strong> - 
                    {% if activity.get('type') == 'scan_started' %}🔍 Scan Started{% endif %}
                    {% if activity.get('type') == 'scan_completed' %}✅ Scan Completed ({{ activity.get('data', {}).get('tokens_found', 0) }} tokens){% endif %}
                    {% if activity.get('type') == 'signal_generated' %}🚀 Signal: {{ activity.get('data', {}).get('token', 'N/A')[:8] }}... @ ${{ "%.6f"|format(activity.get('data', {}).get('price', 0)) }}{% endif %}
                </div>
                {% endif %}
            {% endfor %}
        </div>
    </div>
    
    <script>
        let autoRefreshInterval;
        
        function refreshData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    location.reload(); // Simple refresh for now
                });
        }
        
        function toggleAutoRefresh() {
            const checkbox = document.getElementById('autoRefresh');
            if (checkbox.checked) {
                autoRefreshInterval = setInterval(refreshData, 3000);
            } else {
                clearInterval(autoRefreshInterval);
            }
        }
        
        // Initialize auto-refresh
        document.getElementById('autoRefresh').addEventListener('change', toggleAutoRefresh);
        toggleAutoRefresh();
        
        // Update status styling
        const statusElement = document.getElementById('status');
        statusElement.className = 'status ' + statusElement.textContent.toLowerCase();
    </script>
</body>
</html>
"""

def create_flask_app(monitor: BotMonitor):
    """Create Flask app for web dashboard"""
    if not FLASK_AVAILABLE:
        return None
        
    app = Flask(__name__)
    
    @app.route('/')
    def dashboard():
        data = monitor.get_status()
        return render_template_string(DASHBOARD_HTML, data=data)
    
    @app.route('/api/status')
    def api_status():
        return jsonify(monitor.get_status())
    
    return app

def run_dashboard(port=5000):
    """Run the web dashboard"""
    monitor = BotMonitor()
    
    if not FLASK_AVAILABLE:
        print("Flask not available. Install with: pip install flask")
        return
    
    app = create_flask_app(monitor)
    print(f"Starting SolTrader Dashboard at http://localhost:{port}")
    print("Open your browser and navigate to the URL above")
    print("Dashboard auto-refreshes every 3 seconds")
    
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    run_dashboard()