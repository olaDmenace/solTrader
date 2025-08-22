#!/usr/bin/env python3
"""
Performance Charts Generator
Creates interactive charts for trading performance visualization
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class PerformanceChartsGenerator:
    """Generate interactive performance charts for trading analytics"""
    
    def __init__(self):
        self.charts_dir = Path("charts")
        self.charts_dir.mkdir(exist_ok=True)
        
        self.analytics_dir = Path("analytics")
        
    def load_trade_data(self) -> List[Dict[str, Any]]:
        """Load trade data from analytics"""
        try:
            trades_file = self.analytics_dir / "trades_history.json"
            if trades_file.exists():
                with open(trades_file, 'r') as f:
                    data = json.load(f)
                    return data.get('trades', [])
            return []
        except Exception as e:
            logger.error(f"Failed to load trade data: {e}")
            return []
    
    def load_dashboard_data(self) -> Dict[str, Any]:
        """Load dashboard data for additional trade information"""
        try:
            dashboard_file = Path("dashboard_data.json")
            if dashboard_file.exists():
                with open(dashboard_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load dashboard data: {e}")
            return {}
    
    def generate_pnl_chart(self, days: int = 30) -> str:
        """Generate cumulative P&L chart"""
        trades = self.load_trade_data()
        if not trades:
            return self._create_no_data_chart("No trade data available")
        
        # Filter trades by date range
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [
            trade for trade in trades 
            if datetime.fromisoformat(trade['timestamp']) > cutoff_date
        ]
        
        if not recent_trades:
            return self._create_no_data_chart(f"No trades in last {days} days")
        
        # Sort by timestamp
        recent_trades.sort(key=lambda x: x['timestamp'])
        
        # Calculate cumulative P&L
        cumulative_pnl = 0
        chart_data = []
        
        for trade in recent_trades:
            cumulative_pnl += trade.get('profit_loss_usd', 0)
            chart_data.append({
                'timestamp': trade['timestamp'][:19],  # Remove timezone
                'cumulative_pnl': round(cumulative_pnl, 4),
                'trade_pnl': round(trade.get('profit_loss_usd', 0), 4),
                'symbol': trade.get('token_symbol', 'Unknown')
            })
        
        return self._create_line_chart(
            title=f"Cumulative P&L - Last {days} Days",
            data=chart_data,
            x_field='timestamp',
            y_field='cumulative_pnl',
            chart_id='pnl_chart'
        )
    
    def generate_daily_pnl_chart(self, days: int = 14) -> str:
        """Generate daily P&L bar chart"""
        trades = self.load_trade_data()
        if not trades:
            return self._create_no_data_chart("No trade data available")
        
        # Group trades by day
        daily_pnl = defaultdict(float)
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for trade in trades:
            trade_date = datetime.fromisoformat(trade['timestamp'])
            if trade_date > cutoff_date:
                day_key = trade_date.strftime('%Y-%m-%d')
                daily_pnl[day_key] += trade.get('profit_loss_usd', 0)
        
        if not daily_pnl:
            return self._create_no_data_chart(f"No trades in last {days} days")
        
        # Convert to chart data
        chart_data = []
        for date_str, pnl in sorted(daily_pnl.items()):
            chart_data.append({
                'date': date_str,
                'pnl': round(pnl, 4),
                'color': '#28a745' if pnl >= 0 else '#dc3545'
            })
        
        return self._create_bar_chart(
            title=f"Daily P&L - Last {days} Days",
            data=chart_data,
            x_field='date',
            y_field='pnl',
            chart_id='daily_pnl_chart'
        )
    
    def generate_trade_distribution_chart(self) -> str:
        """Generate trade P&L distribution histogram"""
        trades = self.load_trade_data()
        if not trades:
            return self._create_no_data_chart("No trade data available")
        
        # Get P&L values
        pnl_values = [trade.get('profit_percentage', 0) for trade in trades]
        
        if not pnl_values:
            return self._create_no_data_chart("No P&L data available")
        
        # Create histogram bins
        bins = [-50, -20, -10, -5, -2, 0, 2, 5, 10, 20, 50]
        bin_labels = ['< -50%', '-50% to -20%', '-20% to -10%', '-10% to -5%', 
                     '-5% to -2%', '-2% to 0%', '0% to 2%', '2% to 5%', 
                     '5% to 10%', '10% to 20%', '> 20%']
        
        # Count trades in each bin
        histogram_data = []
        for i in range(len(bins) - 1):
            count = sum(1 for pnl in pnl_values if bins[i] <= pnl < bins[i + 1])
            histogram_data.append({
                'range': bin_labels[i],
                'count': count,
                'color': '#28a745' if bins[i] >= 0 else '#dc3545'
            })
        
        # Handle values above the highest bin
        count_above = sum(1 for pnl in pnl_values if pnl >= bins[-1])
        if count_above > 0:
            histogram_data.append({
                'range': bin_labels[-1],
                'count': count_above,
                'color': '#28a745'
            })
        
        return self._create_bar_chart(
            title="Trade P&L Distribution",
            data=histogram_data,
            x_field='range',
            y_field='count',
            chart_id='distribution_chart'
        )
    
    def generate_win_rate_chart(self, days: int = 30) -> str:
        """Generate win rate over time chart"""
        trades = self.load_trade_data()
        if not trades:
            return self._create_no_data_chart("No trade data available")
        
        # Filter and sort trades
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [
            trade for trade in trades 
            if datetime.fromisoformat(trade['timestamp']) > cutoff_date
        ]
        recent_trades.sort(key=lambda x: x['timestamp'])
        
        if not recent_trades:
            return self._create_no_data_chart(f"No trades in last {days} days")
        
        # Calculate rolling win rate (10-trade window)
        window_size = min(10, len(recent_trades))
        chart_data = []
        
        for i in range(window_size - 1, len(recent_trades)):
            window_trades = recent_trades[i - window_size + 1:i + 1]
            wins = sum(1 for trade in window_trades if trade.get('profit_loss_usd', 0) > 0)
            win_rate = (wins / len(window_trades)) * 100
            
            chart_data.append({
                'timestamp': recent_trades[i]['timestamp'][:19],
                'win_rate': round(win_rate, 1),
                'trade_number': i + 1
            })
        
        return self._create_line_chart(
            title=f"Rolling Win Rate ({window_size}-trade window)",
            data=chart_data,
            x_field='timestamp',
            y_field='win_rate',
            chart_id='win_rate_chart'
        )
    
    def generate_token_performance_chart(self, top_n: int = 10) -> str:
        """Generate top/worst performing tokens chart"""
        trades = self.load_trade_data()
        if not trades:
            return self._create_no_data_chart("No trade data available")
        
        # Group trades by token
        token_performance = defaultdict(lambda: {'pnl': 0, 'trades': 0})
        
        for trade in trades:
            symbol = trade.get('token_symbol', 'Unknown')
            pnl = trade.get('profit_loss_usd', 0)
            
            token_performance[symbol]['pnl'] += pnl
            token_performance[symbol]['trades'] += 1
        
        if not token_performance:
            return self._create_no_data_chart("No token performance data")
        
        # Sort by P&L and take top/bottom performers
        sorted_tokens = sorted(
            token_performance.items(),
            key=lambda x: x[1]['pnl'],
            reverse=True
        )
        
        # Take top N and bottom N
        top_performers = sorted_tokens[:top_n]
        worst_performers = sorted_tokens[-top_n:] if len(sorted_tokens) > top_n else []
        
        chart_data = []
        
        # Add top performers
        for symbol, data in top_performers:
            chart_data.append({
                'symbol': symbol[:8] + "..." if len(symbol) > 8 else symbol,
                'pnl': round(data['pnl'], 4),
                'trades': data['trades'],
                'color': '#28a745' if data['pnl'] >= 0 else '#dc3545'
            })
        
        return self._create_bar_chart(
            title=f"Top {top_n} Token Performance",
            data=chart_data,
            x_field='symbol',
            y_field='pnl',
            chart_id='token_performance_chart'
        )
    
    def _create_line_chart(self, title: str, data: List[Dict], x_field: str, 
                          y_field: str, chart_id: str) -> str:
        """Create line chart HTML"""
        return f"""
        <div class="chart-container">
            <h3>{title}</h3>
            <canvas id="{chart_id}" width="800" height="400"></canvas>
            <script>
                const {chart_id}_data = {json.dumps(data)};
                const {chart_id}_ctx = document.getElementById('{chart_id}').getContext('2d');
                
                new Chart({chart_id}_ctx, {{
                    type: 'line',
                    data: {{
                        labels: {chart_id}_data.map(d => d.{x_field}),
                        datasets: [{{
                            label: '{y_field.replace("_", " ").title()}',
                            data: {chart_id}_data.map(d => d.{y_field}),
                            borderColor: '#ff6b35',
                            backgroundColor: 'rgba(255, 107, 53, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.3
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{
                                labels: {{
                                    color: '#00ff41'
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                ticks: {{
                                    color: '#8892b0',
                                    maxTicksLimit: 10
                                }},
                                grid: {{
                                    color: '#16213e'
                                }}
                            }},
                            y: {{
                                ticks: {{
                                    color: '#8892b0'
                                }},
                                grid: {{
                                    color: '#16213e'
                                }}
                            }}
                        }}
                    }}
                }});
            </script>
        </div>
        """
    
    def _create_bar_chart(self, title: str, data: List[Dict], x_field: str, 
                         y_field: str, chart_id: str) -> str:
        """Create bar chart HTML"""
        return f"""
        <div class="chart-container">
            <h3>{title}</h3>
            <canvas id="{chart_id}" width="800" height="400"></canvas>
            <script>
                const {chart_id}_data = {json.dumps(data)};
                const {chart_id}_ctx = document.getElementById('{chart_id}').getContext('2d');
                
                new Chart({chart_id}_ctx, {{
                    type: 'bar',
                    data: {{
                        labels: {chart_id}_data.map(d => d.{x_field}),
                        datasets: [{{
                            label: '{y_field.replace("_", " ").title()}',
                            data: {chart_id}_data.map(d => d.{y_field}),
                            backgroundColor: {chart_id}_data.map(d => d.color || '#ff6b35'),
                            borderColor: {chart_id}_data.map(d => d.color || '#ff6b35'),
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{
                                display: false
                            }}
                        }},
                        scales: {{
                            x: {{
                                ticks: {{
                                    color: '#8892b0',
                                    maxRotation: 45
                                }},
                                grid: {{
                                    color: '#16213e'
                                }}
                            }},
                            y: {{
                                ticks: {{
                                    color: '#8892b0'
                                }},
                                grid: {{
                                    color: '#16213e'
                                }}
                            }}
                        }}
                    }}
                }});
            </script>
        </div>
        """
    
    def _create_no_data_chart(self, message: str) -> str:
        """Create placeholder for charts with no data"""
        return f"""
        <div class="chart-container">
            <div class="no-data-message">
                <h3>📊 Chart Unavailable</h3>
                <p>{message}</p>
                <p>Charts will appear once trading data is available.</p>
            </div>
        </div>
        """
    
    def generate_dashboard_with_charts(self) -> str:
        """Generate complete dashboard with all charts"""
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>📈 SolTrader Performance Charts</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #0a0e27;
                    color: #00ff41;
                    margin: 0;
                    padding: 20px;
                    line-height: 1.6;
                }}
                
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                    padding: 20px;
                    background: #1a1a2e;
                    border-radius: 12px;
                }}
                
                .header h1 {{
                    color: #ff6b35;
                    margin-bottom: 10px;
                }}
                
                .chart-container {{
                    background: #1a1a2e;
                    border: 1px solid #16213e;
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                }}
                
                .chart-container h3 {{
                    color: #ff6b35;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                
                .charts-grid {{
                    display: grid;
                    gap: 30px;
                }}
                
                .no-data-message {{
                    text-align: center;
                    padding: 40px;
                    color: #8892b0;
                }}
                
                .no-data-message h3 {{
                    color: #ff6b35;
                    margin-bottom: 15px;
                }}
                
                .refresh-info {{
                    text-align: center;
                    color: #8892b0;
                    margin-bottom: 20px;
                    font-size: 0.9em;
                }}
                
                @media (min-width: 1200px) {{
                    .charts-grid {{
                        grid-template-columns: repeat(2, 1fr);
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📈 SolTrader Performance Charts</h1>
                <div class="refresh-info">
                    Last Updated: {timestamp}<br>
                    Auto-refresh: Open in browser and refresh page for latest data
                </div>
            </div>
            
            <div class="charts-grid">
                {pnl_chart}
                {daily_pnl_chart}
                {win_rate_chart}
                {distribution_chart}
                {token_performance_chart}
            </div>
        </body>
        </html>
        """
        
        # Generate all charts
        charts = {
            'pnl_chart': self.generate_pnl_chart(30),
            'daily_pnl_chart': self.generate_daily_pnl_chart(14),
            'win_rate_chart': self.generate_win_rate_chart(30),
            'distribution_chart': self.generate_trade_distribution_chart(),
            'token_performance_chart': self.generate_token_performance_chart(10),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return html_template.format(**charts)
    
    def save_charts_dashboard(self) -> str:
        """Save charts dashboard to file"""
        dashboard_html = self.generate_dashboard_with_charts()
        
        charts_file = self.charts_dir / "performance_charts.html"
        with open(charts_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        return str(charts_file)

# Global charts generator instance
charts_generator = PerformanceChartsGenerator()

def generate_performance_charts() -> str:
    """Generate and save performance charts dashboard"""
    return charts_generator.save_charts_dashboard()

def get_pnl_chart_data(days: int = 30) -> Dict[str, Any]:
    """Get P&L chart data for API usage"""
    trades = charts_generator.load_trade_data()
    cutoff_date = datetime.now() - timedelta(days=days)
    
    recent_trades = [
        trade for trade in trades 
        if datetime.fromisoformat(trade['timestamp']) > cutoff_date
    ]
    
    if not recent_trades:
        return {'error': 'No trades found'}
    
    recent_trades.sort(key=lambda x: x['timestamp'])
    
    cumulative_pnl = 0
    chart_data = []
    
    for trade in recent_trades:
        cumulative_pnl += trade.get('profit_loss_usd', 0)
        chart_data.append({
            'timestamp': trade['timestamp'],
            'cumulative_pnl': cumulative_pnl,
            'trade_pnl': trade.get('profit_loss_usd', 0)
        })
    
    return {
        'data': chart_data,
        'total_trades': len(recent_trades),
        'total_pnl': cumulative_pnl,
        'date_range': f"Last {days} days"
    }