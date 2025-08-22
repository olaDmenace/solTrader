#!/usr/bin/env python3
"""
Mobile-Responsive Dashboard for SolTrader
Enhanced version with mobile-first design and responsive layouts
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from flask import Flask, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

# Mobile-responsive HTML template
MOBILE_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
    <title>📱 SolTrader Mobile Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <style>
        /* CSS Variables for consistent theming */
        :root {
            --primary-bg: #0a0e27;
            --secondary-bg: #1a1a2e;
            --card-bg: #16213e;
            --accent-color: #ff6b35;
            --success-color: #28a745;
            --error-color: #dc3545;
            --text-primary: #00ff41;
            --text-secondary: #8892b0;
            --border-color: #16213e;
        }
        
        /* Reset and base styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: var(--primary-bg);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            padding: 10px;
        }
        
        /* Mobile-first approach */
        .container {
            max-width: 100%;
            margin: 0 auto;
        }
        
        /* Header */
        .header {
            text-align: center;
            margin-bottom: 20px;
            padding: 15px;
            background: var(--secondary-bg);
            border-radius: 12px;
            position: relative;
        }
        
        .header h1 {
            font-size: 1.5rem;
            color: var(--accent-color);
            margin-bottom: 10px;
        }
        
        .status-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9rem;
            text-transform: uppercase;
        }
        
        .status-running { 
            background: var(--success-color); 
            color: white; 
        }
        .status-stopped { 
            background: var(--error-color); 
            color: white; 
        }
        
        /* Auto-refresh indicator */
        .refresh-indicator {
            position: absolute;
            top: 10px;
            right: 10px;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        
        /* Card layout */
        .card {
            background: var(--secondary-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .card h2 {
            color: var(--accent-color);
            margin-bottom: 15px;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Metrics grid - mobile responsive */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
        }
        
        .metric-card {
            background: var(--card-bg);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.4rem;
            font-weight: bold;
            color: var(--text-primary);
            margin-bottom: 4px;
        }
        
        .metric-label {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        
        /* Tables - responsive design */
        .table-container {
            overflow-x: auto;
            margin-top: 10px;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            min-width: 600px; /* Minimum width for horizontal scroll */
        }
        
        .data-table th,
        .data-table td {
            border: 1px solid var(--border-color);
            padding: 8px 6px;
            text-align: left;
            font-size: 0.85rem;
        }
        
        .data-table th {
            background: var(--card-bg);
            color: var(--accent-color);
            font-weight: bold;
            position: sticky;
            top: 0;
        }
        
        .data-table td {
            word-break: break-word;
        }
        
        /* Color coding */
        .profit { color: var(--success-color); }
        .loss { color: var(--error-color); }
        .neutral { color: var(--text-secondary); }
        
        /* Events feed */
        .events-feed {
            max-height: 250px;
            overflow-y: auto;
            background: var(--primary-bg);
            padding: 12px;
            border-radius: 8px;
            font-size: 0.85rem;
        }
        
        .event-item {
            padding: 8px 0;
            border-bottom: 1px solid var(--border-color);
        }
        
        .event-item:last-child {
            border-bottom: none;
        }
        
        .event-time {
            color: var(--text-secondary);
            font-size: 0.75rem;
        }
        
        /* Quick stats bar */
        .quick-stats {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .quick-stat {
            flex: 1;
            min-width: 120px;
            background: var(--card-bg);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        
        .quick-stat-value {
            font-size: 1.1rem;
            font-weight: bold;
        }
        
        .quick-stat-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        
        /* Responsive breakpoints */
        @media (min-width: 768px) {
            .container {
                max-width: 1200px;
                padding: 0 20px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .metrics-grid {
                grid-template-columns: repeat(3, 1fr);
            }
            
            .card {
                padding: 20px;
            }
            
            .data-table th,
            .data-table td {
                padding: 12px;
                font-size: 0.9rem;
            }
        }
        
        @media (min-width: 1024px) {
            .metrics-grid {
                grid-template-columns: repeat(6, 1fr);
            }
            
            .quick-stats {
                justify-content: space-around;
            }
            
            .quick-stat {
                flex: 0 1 auto;
                min-width: 150px;
            }
        }
        
        /* Utility classes */
        .text-center { text-align: center; }
        .text-small { font-size: 0.8rem; }
        .mb-1 { margin-bottom: 10px; }
        .mt-1 { margin-top: 10px; }
        
        /* Loading animation */
        .loading {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 2px solid var(--text-secondary);
            border-radius: 50%;
            border-top-color: var(--accent-color);
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Touch-friendly improvements */
        @media (max-width: 767px) {
            .card {
                margin-bottom: 15px;
                padding: 12px;
            }
            
            .metric-card {
                padding: 10px 8px;
            }
            
            .metric-value {
                font-size: 1.2rem;
            }
            
            .data-table {
                min-width: 500px;
            }
            
            .data-table th,
            .data-table td {
                padding: 6px 4px;
                font-size: 0.8rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="refresh-indicator">
            <span class="loading"></span> Auto-refresh: 5s
        </div>
        
        <header class="header">
            <h1>📱 SolTrader Dashboard</h1>
            <span class="status-badge status-{{ 'running' if data.get('status') == 'running' else 'stopped' }}">
                {{ data.get('status', 'Unknown').upper() }}
            </span>
            <div class="text-small mt-1">
                Last Update: {{ data.get('last_update', 'Never')[:19] if data.get('last_update') else 'Never' }}
            </div>
        </header>
        
        <!-- Quick Stats Bar -->
        <div class="quick-stats">
            <div class="quick-stat">
                <div class="quick-stat-value {{ 'profit' if (data.get('performance', {}).get('total_pnl', 0) > 0) else 'loss' if (data.get('performance', {}).get('total_pnl', 0) < 0) else 'neutral' }}">
                    ${{ "%.3f"|format(data.get('performance', {}).get('total_pnl', 0)) }}
                </div>
                <div class="quick-stat-label">Total P&L</div>
            </div>
            <div class="quick-stat">
                <div class="quick-stat-value">{{ data.get('performance', {}).get('total_trades', 0) }}</div>
                <div class="quick-stat-label">Trades</div>
            </div>
            <div class="quick-stat">
                <div class="quick-stat-value">{{ "%.1f"|format(data.get('performance', {}).get('win_rate', 0)) }}%</div>
                <div class="quick-stat-label">Win Rate</div>
            </div>
            <div class="quick-stat">
                <div class="quick-stat-value">{{ data.get('performance', {}).get('open_positions', 0) }}</div>
                <div class="quick-stat-label">Open</div>
            </div>
        </div>
        
        <!-- Performance Metrics -->
        <div class="card">
            <h2>📊 Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value {{ 'profit' if (data.get('performance', {}).get('total_pnl', 0) > 0) else 'loss' if (data.get('performance', {}).get('total_pnl', 0) < 0) else 'neutral' }}">
                        ${{ "%.6f"|format(data.get('performance', {}).get('total_pnl', 0)) }}
                    </div>
                    <div class="metric-label">Total P&L</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.1f"|format(data.get('performance', {}).get('win_rate', 0)) }}%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ data.get('performance', {}).get('total_trades', 0) }}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.3f"|format(data.get('performance', {}).get('balance', 0)) }} SOL</div>
                    <div class="metric-label">Balance</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {{ 'profit' if (data.get('performance', {}).get('unrealized_pnl', 0) > 0) else 'loss' if (data.get('performance', {}).get('unrealized_pnl', 0) < 0) else 'neutral' }}">
                        ${{ "%.6f"|format(data.get('performance', {}).get('unrealized_pnl', 0)) }}
                    </div>
                    <div class="metric-label">Unrealized P&L</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ data.get('performance', {}).get('open_positions', 0) }}</div>
                    <div class="metric-label">Open Positions</div>
                </div>
            </div>
        </div>
        
        <!-- Recent Trades -->
        <div class="card">
            <h2>📈 Recent Trades (Last 10)</h2>
            <div class="table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Token</th>
                            <th>Type</th>
                            <th>Price</th>
                            <th>P&L</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for trade in data.get('trades', [])[-10:] %}
                        <tr>
                            <td class="text-small">{{ trade.get('timestamp', '')[:16] }}</td>
                            <td class="text-small">{{ trade.get('symbol', trade.get('token_address', 'Unknown'))[:8] }}...</td>
                            <td>{{ trade.get('type', 'Unknown') }}</td>
                            <td>${{ "%.4f"|format(trade.get('price', 0)) }}</td>
                            <td class="{{ 'profit' if (trade.get('pnl', 0) > 0) else 'loss' if (trade.get('pnl', 0) < 0) else 'neutral' }}">
                                ${{ "%.4f"|format(trade.get('pnl', 0)) }}
                            </td>
                            <td>{{ trade.get('status', 'Unknown') }}</td>
                        </tr>
                        {% else %}
                        <tr>
                            <td colspan="6" class="text-center text-small">No trades recorded yet</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Open Positions -->
        {% if data.get('positions') %}
        <div class="card">
            <h2>🎯 Open Positions</h2>
            <div class="table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Token</th>
                            <th>Entry</th>
                            <th>Current</th>
                            <th>P&L</th>
                            <th>Age</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for position in data.get('positions', []) %}
                        <tr>
                            <td class="text-small">{{ position.get('symbol', position.get('token_address', 'Unknown'))[:10] }}...</td>
                            <td>${{ "%.4f"|format(position.get('entry_price', 0)) }}</td>
                            <td>${{ "%.4f"|format(position.get('current_price', 0)) }}</td>
                            <td class="{{ 'profit' if (position.get('unrealized_pnl', 0) > 0) else 'loss' if (position.get('unrealized_pnl', 0) < 0) else 'neutral' }}">
                                ${{ "%.4f"|format(position.get('unrealized_pnl', 0)) }}
                            </td>
                            <td class="text-small">{{ position.get('age', 'Unknown') }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endif %}
        
        <!-- Recent Activity -->
        <div class="card">
            <h2>📱 Recent Activity</h2>
            <div class="events-feed">
                {% for event in data.get('activity', [])[-15:] %}
                <div class="event-item">
                    <div class="event-time">{{ event.get('timestamp', '')[:19] }}</div>
                    <div>{{ event.get('action', 'Unknown action') }}</div>
                    {% if event.get('details') %}
                    <div class="text-small text-secondary">{{ event.get('details', '') }}</div>
                    {% endif %}
                </div>
                {% else %}
                <div class="text-center text-small">No recent activity</div>
                {% endfor %}
            </div>
        </div>
    </div>
    
    <script>
        // Simple touch-friendly interactions
        document.addEventListener('DOMContentLoaded', function() {
            // Auto-scroll to newest content on mobile
            if (window.innerWidth <= 768) {
                window.scrollTo(0, 0);
            }
            
            // Add touch feedback for interactive elements
            const interactiveElements = document.querySelectorAll('.card, .metric-card, .quick-stat');
            interactiveElements.forEach(element => {
                element.addEventListener('touchstart', function() {
                    this.style.opacity = '0.8';
                });
                element.addEventListener('touchend', function() {
                    this.style.opacity = '1';
                });
            });
        });
    </script>
</body>
</html>
"""

class MobileDashboard:
    """Mobile-responsive dashboard for SolTrader"""
    
    def __init__(self, data_file="dashboard_data.json"):
        self.data_file = data_file
        self.app = None
        
    def load_dashboard_data(self):
        """Load dashboard data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            else:
                return self._get_default_data()
        except Exception as e:
            print(f"Error loading dashboard data: {e}")
            return self._get_default_data()
    
    def _get_default_data(self):
        """Return default dashboard data structure"""
        return {
            "status": "stopped",
            "start_time": None,
            "trades": [],
            "positions": [],
            "performance": {
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "balance": 100.0,
                "unrealized_pnl": 0.0,
                "open_positions": 0
            },
            "activity": [],
            "last_update": datetime.now().isoformat()
        }
    
    def create_flask_app(self):
        """Create Flask app for mobile dashboard"""
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for the mobile dashboard")
        
        app = Flask(__name__)
        
        @app.route('/')
        def dashboard():
            """Main dashboard route"""
            data = self.load_dashboard_data()
            return render_template_string(MOBILE_DASHBOARD_TEMPLATE, data=data)
        
        @app.route('/api/data')
        def api_data():
            """API endpoint for dashboard data"""
            data = self.load_dashboard_data()
            return jsonify(data)
        
        @app.route('/api/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            })
        
        self.app = app
        return app
    
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the mobile dashboard server"""
        if not self.app:
            self.create_flask_app()
        
        print(f"Starting SolTrader Mobile Dashboard...")
        print(f"Mobile-optimized interface available at:")
        print(f"   Local: http://localhost:{port}")
        print(f"   Network: http://{host}:{port}")
        print(f"API endpoint: http://localhost:{port}/api/data")
        print(f"Health check: http://localhost:{port}/api/health")
        print("\nFeatures:")
        print("   - Mobile-first responsive design")
        print("   - Touch-friendly interface")
        print("   - Auto-refresh every 5 seconds")
        print("   - Optimized for phones and tablets")
        print("   - Horizontal table scrolling")
        print("   - Quick stats overview")
        
        try:
            self.app.run(host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            print("\nMobile dashboard stopped")
        except Exception as e:
            print(f"Error running mobile dashboard: {e}")

def main():
    """Main function to run the mobile dashboard"""
    dashboard = MobileDashboard()
    
    # Check if Flask is available
    if not FLASK_AVAILABLE:
        print("Flask is not installed. Please install it with:")
        print("   pip install flask")
        return
    
    try:
        dashboard.run(port=5001, debug=False)
    except Exception as e:
        print(f"Failed to start mobile dashboard: {e}")

if __name__ == "__main__":
    main()