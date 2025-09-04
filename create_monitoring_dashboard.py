#!/usr/bin/env python3
"""
Enhanced Web Dashboard for SolTrader Professional Trading System
Creates a comprehensive web interface with:
- Token metadata caching and display
- Risk management monitoring
- Multi-RPC status tracking  
- Real-time portfolio analysis
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
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import enhanced components
from src.cache import get_token_cache, get_token_display_name, TokenMetadata
from src.risk import get_risk_manager, RiskLevel
from src.utils.simple_token_namer import SimpleTokenNamer

logger = logging.getLogger(__name__)

try:
    from flask import Flask, render_template_string, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

class BotMonitor:
    """Enhanced bot monitoring system with professional features"""
    
    def __init__(self, log_file="logs/trading.log", data_file="dashboard_data.json"):
        self.log_file = log_file
        self.data_file = data_file
        self.app = None
        self.token_namer = SimpleTokenNamer()
        
        # Enhanced components
        self.token_cache = get_token_cache()
        self.risk_manager = get_risk_manager()
        
        # Cache refresh tracking
        self.last_cache_refresh = time.time()
        self.cache_refresh_interval = 300  # 5 minutes
        
        # Initialize enhanced data structure
        self.data = {
            "status": "stopped",
            "start_time": None,
            "trades": [],
            "positions": [],
            "enhanced_positions": {},
            "performance": {
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "balance": 100.0
            },
            "risk_analysis": {
                "summary": {},
                "active_alerts": [],
                "emergency_status": {},
                "position_analysis": {},
                "performance_metrics": {}
            },
            "token_metadata_stats": {
                "cache_performance": {},
                "metadata_sources": {}
            },
            "rpc_status": {
                "providers": {},
                "current_provider": "unknown",
                "performance_stats": {}
            },
            "recent_events": [],
            "last_update": datetime.now().isoformat()
        }
        
        self.load_data()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background tasks for data updates"""
        def background_update():
            while True:
                try:
                    # Update enhanced data every 10 seconds
                    asyncio.run(self._update_enhanced_data())
                    time.sleep(10)
                except Exception as e:
                    logger.error(f"Background update error: {e}")
                    time.sleep(30)
        
        update_thread = threading.Thread(target=background_update, daemon=True)
        update_thread.start()
    
    async def _update_enhanced_data(self):
        """Update enhanced dashboard data"""
        try:
            # Update risk analysis
            await self._update_risk_analysis()
            
            # Update token metadata stats
            await self._update_token_metadata_stats()
            
            # Update enhanced positions
            await self._update_enhanced_positions()
            
            # Periodic cache refresh
            if time.time() - self.last_cache_refresh > self.cache_refresh_interval:
                asyncio.create_task(self._refresh_token_cache_background())
                self.last_cache_refresh = time.time()
            
            # Update timestamp
            self.data["last_update"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Enhanced data update failed: {e}")
    
    async def _update_risk_analysis(self):
        """Update risk analysis data"""
        try:
            risk_summary = self.risk_manager.get_risk_summary()
            active_alerts = self.risk_manager.get_active_alerts()
            
            self.data["risk_analysis"] = {
                "summary": risk_summary,
                "active_alerts": active_alerts,
                "emergency_status": {
                    "emergency_stop_active": risk_summary.get('emergency_stop_active', False),
                    "alert_count": len(active_alerts),
                    "critical_alerts": len([a for a in active_alerts if a.get('level') == 'critical'])
                },
                "position_analysis": {
                    "total_positions": risk_summary.get('active_positions', 0),
                    "position_limits": {
                        "max_positions": risk_summary.get('limits', {}).get('max_positions', 0),
                        "max_position_size": risk_summary.get('limits', {}).get('max_position_size', 0)
                    }
                },
                "performance_metrics": {
                    "daily_pnl": risk_summary.get('daily_pnl', 0),
                    "daily_trade_count": risk_summary.get('daily_trade_count', 0),
                    "current_drawdown": risk_summary.get('portfolio_metrics', {}).get('current_drawdown', 0),
                    "max_drawdown": risk_summary.get('portfolio_metrics', {}).get('max_drawdown', 0)
                }
            }
        except Exception as e:
            logger.error(f"Risk analysis update failed: {e}")
    
    async def _update_token_metadata_stats(self):
        """Update token metadata cache statistics"""
        try:
            cache_stats = self.token_cache.get_cache_stats()
            self.data["token_metadata_stats"] = {
                "cache_performance": {
                    "memory_entries": cache_stats.get('memory_cache_size', 0),
                    "disk_entries": cache_stats.get('disk_cache_files', 0),
                    "last_refresh": cache_stats.get('last_refresh'),
                    "cache_directory": cache_stats.get('cache_directory')
                },
                "metadata_sources": {
                    "jupiter_primary": True,
                    "birdeye_enhancement": bool(getattr(self.token_cache, 'birdeye_api_key', None)),
                    "solana_rpc_fallback": True
                }
            }
        except Exception as e:
            logger.error(f"Token metadata stats update failed: {e}")
    
    async def _update_enhanced_positions(self):
        """Update enhanced positions with token metadata"""
        try:
            enhanced_positions = {}
            
            # Get positions from existing data
            positions = self.data.get("positions", [])
            
            if positions:
                # Extract token addresses
                token_addresses = []
                for pos in positions:
                    token_addr = pos.get('token_address') or pos.get('token')
                    if token_addr:
                        token_addresses.append(token_addr)
                
                # Get batch metadata
                if token_addresses:
                    metadata_batch = await self.token_cache.get_batch_metadata(token_addresses)
                    
                    for pos in positions:
                        token_addr = pos.get('token_address') or pos.get('token')
                        if token_addr:
                            metadata = metadata_batch.get(token_addr)
                            
                            enhanced_positions[token_addr] = {
                                **pos,
                                'token_info': {
                                    'display_name': metadata.display_name if metadata else f"{token_addr[:8]}...",
                                    'full_display_name': metadata.full_display_name if metadata else f"Unknown ({token_addr[:8]}...)",
                                    'symbol': metadata.symbol if metadata else 'UNKNOWN',
                                    'name': metadata.name if metadata else 'Unknown Token',
                                    'verified': metadata.verified if metadata else False,
                                    'price_usd': metadata.price_usd if metadata else None,
                                    'logo_uri': metadata.logo_uri if metadata else None,
                                    'warnings': metadata.warnings if metadata else [],
                                }
                            }
            
            self.data["enhanced_positions"] = enhanced_positions
            
        except Exception as e:
            logger.error(f"Enhanced positions update failed: {e}")
    
    async def _refresh_token_cache_background(self):
        """Background refresh of token metadata cache"""
        try:
            await self.token_cache.refresh_popular_tokens()
            await self.token_cache.clear_expired_cache()
            logger.info("Background token cache refresh completed")
        except Exception as e:
            logger.error(f"Background token cache refresh failed: {e}")
    
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
    <title>🚀 SolTrader Professional Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
    <meta http-equiv="refresh" content="5">
    <style>
        /* Modern CSS Variables for Consistent Theming */
        :root {
            --primary-bg: #0a0e27;
            --secondary-bg: #1a1a2e;
            --card-bg: #16213e;
            --accent-bg: #0f0f23;
            --primary-color: #00ff41;
            --accent-color: #ff6b35;
            --text-muted: #8892b0;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --info-color: #17a2b8;
            --border-color: #2d3748;
            --shadow: 0 4px 12px rgba(0,0,0,0.15);
            --shadow-hover: 0 8px 25px rgba(0,0,0,0.25);
            --border-radius: 12px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Global Styles */
        * {
            box-sizing: border-box;
        }

        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, var(--primary-bg) 0%, #0f1419 100%);
            color: var(--primary-color);
            margin: 0; 
            padding: 0;
            overflow-x: hidden;
            line-height: 1.6;
            min-height: 100vh;
        }

        /* Container */
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: clamp(15px, 3vw, 30px);
        }

        /* Modern Header */
        .header { 
            text-align: center; 
            margin-bottom: clamp(20px, 4vw, 40px);
            background: linear-gradient(135deg, var(--card-bg), var(--secondary-bg));
            padding: clamp(20px, 4vw, 40px);
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
        }
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-color), var(--primary-color), var(--info-color));
            border-radius: var(--border-radius) var(--border-radius) 0 0;
        }
        .header h1 {
            font-size: clamp(1.8rem, 5vw, 3rem);
            margin: 0;
            background: linear-gradient(135deg, var(--accent-color), var(--primary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .header .subtitle {
            margin-top: 8px;
            font-size: clamp(0.9rem, 2vw, 1.1rem);
            color: var(--text-muted);
            font-weight: 400;
        }
        /* Status Indicators */
        .status { 
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px; 
            border-radius: 25px; 
            font-weight: 600;
            font-size: clamp(0.85rem, 2.2vw, 0.95rem);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            transition: var(--transition);
            position: relative;
            overflow: hidden;
        }
        .status::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            transition: left 0.6s;
        }
        .status:hover::before {
            left: 100%;
        }
        .running { 
            background: linear-gradient(135deg, var(--success-color), #34ce57); 
            color: white; 
            box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);
        }
        .stopped { 
            background: linear-gradient(135deg, var(--danger-color), #e55370); 
            color: white;
            box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3);
        }

        /* Modern Card Design */
        .card { 
            background: linear-gradient(135deg, var(--secondary-bg), var(--card-bg));
            border: 1px solid var(--border-color); 
            border-radius: var(--border-radius); 
            padding: clamp(16px, 4vw, 24px);
            margin: clamp(15px, 3vw, 25px) 0;
            box-shadow: var(--shadow);
            overflow: hidden;
            position: relative;
            transition: var(--transition);
            backdrop-filter: blur(10px);
        }
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--accent-color), var(--primary-color));
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }
        .card:hover {
            box-shadow: var(--shadow-hover);
            transform: translateY(-2px);
            border-color: var(--accent-color);
        }
        .card:hover::before {
            transform: scaleX(1);
        }
        .card h2, .card h3 {
            color: var(--accent-color);
            margin-top: 0;
            margin-bottom: clamp(12px, 3vw, 20px);
            font-size: clamp(1.1rem, 2.8vw, 1.4rem);
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Modern Grid Layout */
        .metrics { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); 
            gap: clamp(12px, 2.5vw, 20px);
            margin-bottom: 20px;
        }
        .metric { 
            text-align: center; 
            padding: clamp(16px, 3vw, 20px);
            background: linear-gradient(135deg, var(--accent-bg), var(--card-bg));
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            transition: var(--transition);
            position: relative;
            overflow: hidden;
        }
        .metric::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, var(--accent-color), var(--primary-color));
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .metric:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow);
        }
        .metric:hover::before {
            opacity: 0.05;
        }
        .metric-value { 
            font-size: clamp(1.4rem, 4.5vw, 2.2rem);
            font-weight: 800; 
            color: var(--primary-color);
            margin-bottom: 4px;
            position: relative;
            z-index: 1;
        }
        .metric-label { 
            color: var(--text-muted); 
            font-size: clamp(0.8rem, 2vw, 0.95rem);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            position: relative;
            z-index: 1;
        }
        /* Enhanced Responsive Table Container */
        .table-container {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            margin-top: 10px;
            position: relative;
            border-radius: var(--border-radius);
            background: var(--accent-bg);
            box-shadow: inset 0 0 0 1px var(--border-color);
        }
        
        /* Scroll indicator shadows */
        .table-container::before,
        .table-container::after {
            content: '';
            position: absolute;
            top: 0;
            bottom: 0;
            width: 20px;
            pointer-events: none;
            z-index: 2;
            transition: opacity 0.3s ease;
        }
        
        .table-container::before {
            left: 0;
            background: linear-gradient(90deg, var(--accent-bg), transparent);
            opacity: 0;
        }
        
        .table-container::after {
            right: 0;
            background: linear-gradient(-90deg, var(--accent-bg), transparent);
            opacity: 1;
        }
        
        .table-container.scrolled-left::before {
            opacity: 1;
        }
        
        .table-container.scrolled-right::after {
            opacity: 0;
        }
        
        .trades-table, .positions-table { 
            width: 100%; 
            border-collapse: collapse; 
            font-size: clamp(0.7rem, 2vw, 0.9rem);
            min-width: 600px;
            background: var(--secondary-bg);
        }
        
        .trades-table th, .trades-table td,
        .positions-table th, .positions-table td { 
            border: 1px solid var(--border-color); 
            padding: clamp(6px, 1.8vw, 12px);
            text-align: left;
            word-break: break-word;
            transition: background-color 0.2s ease;
        }
        
        .trades-table th, .positions-table th { 
            background: var(--card-bg); 
            color: var(--accent-color);
            position: sticky;
            top: 0;
            z-index: 1;
            font-weight: 600;
            text-transform: uppercase;
            font-size: clamp(0.65rem, 1.8vw, 0.8rem);
            letter-spacing: 0.5px;
        }
        
        .trades-table tbody tr:hover td,
        .positions-table tbody tr:hover td {
            background: var(--accent-bg);
        }
        .profit { color: #28a745; }
        .loss { color: #dc3545; }
        .refresh-btn { 
            background: #ff6b35; 
            color: white; 
            border: none; 
            padding: clamp(8px, 2vw, 10px) clamp(15px, 3vw, 20px);
            border-radius: 5px; 
            cursor: pointer; 
            margin: 10px;
            font-size: clamp(0.8rem, 2vw, 1rem);
            touch-action: manipulation;
        }
        .auto-refresh { 
            position: fixed; 
            top: clamp(10px, 2vw, 20px);
            right: clamp(10px, 2vw, 20px);
            background: #16213e; 
            padding: clamp(5px, 1.5vw, 10px);
            border-radius: 5px;
            font-size: clamp(0.7rem, 1.8vw, 0.9rem);
        }
        .events { 
            max-height: clamp(150px, 30vw, 200px);
            overflow-y: auto; 
            background: #0f0f0f; 
            padding: clamp(8px, 2vw, 10px);
            border-radius: 5px;
            font-size: clamp(0.75rem, 2vw, 0.9rem);
            -webkit-overflow-scrolling: touch;
        }
        
        /* Enhanced Mobile Optimizations */
        @media (max-width: 768px) {
            .metrics { 
                grid-template-columns: repeat(2, 1fr); 
                gap: 10px;
            }
            
            .trades-table, .positions-table {
                min-width: 450px;
                font-size: 0.75rem;
            }
            
            .trades-table th, .trades-table td,
            .positions-table th, .positions-table td { 
                padding: clamp(4px, 1.5vw, 6px);
            }
            
            .auto-refresh {
                position: relative;
                top: auto;
                right: auto;
                margin: 10px 0;
                display: block;
                text-align: center;
            }
            
            /* Hide less critical columns on medium screens */
            .trades-table th:nth-child(n+5), .trades-table td:nth-child(n+5) {
                display: none;
            }
        }
        
        @media (max-width: 480px) {
            .metrics { 
                grid-template-columns: 1fr; 
            }
            
            .trades-table, .positions-table {
                min-width: 320px;
                font-size: 0.7rem;
            }
            
            .trades-table th, .trades-table td,
            .positions-table th, .positions-table td { 
                padding: 4px 2px;
                font-size: 0.65rem;
            }
            
            /* Show only essential columns on small screens */
            .trades-table th:nth-child(n+4), .trades-table td:nth-child(n+4) {
                display: none;
            }
            
            .positions-table th:nth-child(n+6), .positions-table td:nth-child(n+6) {
                display: none;
            }
        }
        
        /* Touch-friendly interactions */
        @media (hover: none) and (pointer: coarse) {
            .refresh-btn {
                padding: 12px 20px;
                min-height: 44px;
            }
            
            .card {
                transition: transform 0.1s ease;
            }
            
            .card:active {
                transform: scale(0.98);
            }
        }
        
        /* Scrollbar styling */
        .events::-webkit-scrollbar,
        .table-container::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        
        .events::-webkit-scrollbar-track,
        .table-container::-webkit-scrollbar-track {
            background: #1a1a2e;
        }
        
        .events::-webkit-scrollbar-thumb,
        .table-container::-webkit-scrollbar-thumb {
            background: #16213e;
            border-radius: 3px;
        }
        
        /* Risk Management Styles */
        .risk-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
        }
        .risk-item {
            display: flex;
            justify-content: space-between;
            background: #0f0f0f;
            padding: 8px 12px;
            border-radius: 5px;
            border-left: 3px solid #ff6b35;
        }
        .risk-label {
            font-weight: bold;
            color: #8892b0;
        }
        
        /* Alert Styles */
        .alerts {
            margin-top: 15px;
        }
        .alert {
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert-low { border-color: #28a745; background: rgba(40, 167, 69, 0.1); }
        .alert-moderate { border-color: #ffc107; background: rgba(255, 193, 7, 0.1); }
        .alert-high { border-color: #fd7e14; background: rgba(253, 126, 20, 0.1); }
        .alert-critical { border-color: #dc3545; background: rgba(220, 53, 69, 0.1); }
        .alert-emergency { border-color: #e83e8c; background: rgba(232, 62, 140, 0.1); }
        
        /* Token Metadata Cache Styles */
        .cache-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        .cache-item {
            display: flex;
            justify-content: space-between;
            background: #0f0f0f;
            padding: 8px 12px;
            border-radius: 5px;
            border-left: 3px solid #00ff41;
        }
        .cache-label {
            font-weight: bold;
            color: #8892b0;
        }
        
        /* Enhanced Position Styles */
        .verified {
            color: #28a745;
            font-weight: bold;
        }
        .unverified {
            color: #dc3545;
            font-weight: bold;
        }
        .no-price {
            color: #8892b0;
            font-style: italic;
        }
        .warning {
            background: rgba(255, 193, 7, 0.2);
            color: #ffc107;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.8em;
            margin: 0 2px;
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
        <div class="table-container">
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
    </div>
    
    <div class="card">
        <h2>📈 Recent Trades</h2>
        <div class="table-container">
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
    </div>
    
    <div class="card">
        <h2>🛡️ Risk Management</h2>
        <div class="risk-grid">
            <div class="risk-item">
                <span class="risk-label">Emergency Stop:</span>
                <span class="status {% if data.get('risk_analysis', {}).get('emergency_status', {}).get('emergency_stop_active') %}stopped{% else %}running{% endif %}">
                    {% if data.get('risk_analysis', {}).get('emergency_status', {}).get('emergency_stop_active') %}ACTIVE{% else %}INACTIVE{% endif %}
                </span>
            </div>
            <div class="risk-item">
                <span class="risk-label">Daily P&L:</span>
                <span class="{% if data.get('risk_analysis', {}).get('performance_metrics', {}).get('daily_pnl', 0) >= 0 %}profit{% else %}loss{% endif %}">
                    ${{ "%.2f"|format(data.get('risk_analysis', {}).get('performance_metrics', {}).get('daily_pnl', 0)) }}
                </span>
            </div>
            <div class="risk-item">
                <span class="risk-label">Daily Trades:</span>
                <span>{{ data.get('risk_analysis', {}).get('performance_metrics', {}).get('daily_trade_count', 0) }}</span>
            </div>
            <div class="risk-item">
                <span class="risk-label">Active Positions:</span>
                <span>{{ data.get('risk_analysis', {}).get('position_analysis', {}).get('total_positions', 0) }} / {{ data.get('risk_analysis', {}).get('position_analysis', {}).get('position_limits', {}).get('max_positions', 0) }}</span>
            </div>
            <div class="risk-item">
                <span class="risk-label">Max Position Size:</span>
                <span>{{ data.get('risk_analysis', {}).get('position_analysis', {}).get('position_limits', {}).get('max_position_size', 0) }} SOL</span>
            </div>
            <div class="risk-item">
                <span class="risk-label">Current Drawdown:</span>
                <span class="{% if data.get('risk_analysis', {}).get('performance_metrics', {}).get('current_drawdown', 0) > 0.05 %}loss{% else %}profit{% endif %}">
                    {{ "%.1f"|format(data.get('risk_analysis', {}).get('performance_metrics', {}).get('current_drawdown', 0) * 100) }}%
                </span>
            </div>
        </div>
        
        {% if data.get('risk_analysis', {}).get('active_alerts', []) %}
        <div class="alerts">
            <h3>⚠️ Active Alerts</h3>
            {% for alert in data.get('risk_analysis', {}).get('active_alerts', [])[:5] %}
            <div class="alert alert-{{ alert.get('level', 'info') }}">
                <strong>{{ alert.get('level', 'INFO').upper() }}:</strong> {{ alert.get('message', 'No message') }}
                <small>{{ alert.get('timestamp', 'N/A')[:19] }}</small>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>
    
    <div class="card">
        <h2>💎 Enhanced Positions</h2>
        {% if data.get('enhanced_positions', {}) %}
        <div class="table-container">
            <table class="positions-table">
            <thead>
                <tr>
                    <th>Token</th>
                    <th>Symbol</th>
                    <th>Verified</th>
                    <th>Price</th>
                    <th>Amount</th>
                    <th>Value</th>
                    <th>Warnings</th>
                </tr>
            </thead>
            <tbody>
                {% for addr, pos in data.get('enhanced_positions', {}).items() %}
                <tr>
                    <td title="{{ pos.get('token_info', {}).get('name', 'Unknown') }}">
                        <strong>{{ pos.get('token_info', {}).get('display_name', addr[:8] + '...') }}</strong>
                        {% if pos.get('token_info', {}).get('logo_uri') %}
                        <img src="{{ pos.get('token_info', {}).get('logo_uri') }}" alt="Logo" style="width: 16px; height: 16px; vertical-align: middle; margin-left: 5px;">
                        {% endif %}
                    </td>
                    <td>{{ pos.get('token_info', {}).get('symbol', 'UNKNOWN') }}</td>
                    <td>
                        {% if pos.get('token_info', {}).get('verified') %}
                        <span class="verified">✅ Yes</span>
                        {% else %}
                        <span class="unverified">❌ No</span>
                        {% endif %}
                    </td>
                    <td>
                        {% if pos.get('token_info', {}).get('price_usd') %}
                        ${{ "%.6f"|format(pos.get('token_info', {}).get('price_usd')) }}
                        {% else %}
                        <span class="no-price">N/A</span>
                        {% endif %}
                    </td>
                    <td>{{ "%.4f"|format(pos.get('amount', 0)) }}</td>
                    <td>{{ "%.2f"|format(pos.get('value', 0)) }} SOL</td>
                    <td>
                        {% for warning in pos.get('token_info', {}).get('warnings', []) %}
                        <span class="warning">{{ warning }}</span>
                        {% endfor %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
            </table>
        </div>
        {% else %}
        <p style="text-align: center; color: #8892b0;">No enhanced position data available</p>
        {% endif %}
    </div>
    
    <div class="card">
        <h2>🔍 Token Metadata Cache</h2>
        <div class="cache-stats">
            <div class="cache-item">
                <span class="cache-label">Memory Cache:</span>
                <span>{{ data.get('token_metadata_stats', {}).get('cache_performance', {}).get('memory_entries', 0) }} entries</span>
            </div>
            <div class="cache-item">
                <span class="cache-label">Disk Cache:</span>
                <span>{{ data.get('token_metadata_stats', {}).get('cache_performance', {}).get('disk_entries', 0) }} files</span>
            </div>
            <div class="cache-item">
                <span class="cache-label">Jupiter API:</span>
                <span class="status running">{{ "✅ Active" if data.get('token_metadata_stats', {}).get('metadata_sources', {}).get('jupiter_primary') else "❌ Inactive" }}</span>
            </div>
            <div class="cache-item">
                <span class="cache-label">Birdeye Enhancement:</span>
                <span class="status {% if data.get('token_metadata_stats', {}).get('metadata_sources', {}).get('birdeye_enhancement') %}running{% else %}stopped{% endif %}">
                    {{ "✅ Enabled" if data.get('token_metadata_stats', {}).get('metadata_sources', {}).get('birdeye_enhancement') else "❌ Disabled" }}
                </span>
            </div>
        </div>
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
        let lastUpdateTime = new Date();
        let isRefreshing = false;
        
        // Enhanced refresh with loading states and animations
        async function refreshData() {
            if (isRefreshing) return;
            isRefreshing = true;
            
            // Add loading state to refresh button
            const refreshBtn = document.querySelector('.refresh-btn');
            const originalText = refreshBtn?.textContent;
            if (refreshBtn) {
                refreshBtn.textContent = '🔄 Updating...';
                refreshBtn.style.opacity = '0.7';
            }
            
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // Add smooth refresh animation
                document.body.style.transition = 'opacity 0.3s ease';
                document.body.style.opacity = '0.8';
                
                setTimeout(() => {
                    location.reload();
                }, 200);
                
            } catch (error) {
                console.error('Failed to refresh data:', error);
                // Show error state
                if (refreshBtn) {
                    refreshBtn.textContent = '❌ Error';
                    setTimeout(() => {
                        refreshBtn.textContent = originalText;
                        refreshBtn.style.opacity = '1';
                    }, 2000);
                }
            }
            
            isRefreshing = false;
        }
        
        function toggleAutoRefresh() {
            const checkbox = document.getElementById('autoRefresh');
            if (checkbox.checked) {
                autoRefreshInterval = setInterval(refreshData, 3000);
                addRealTimeIndicator();
            } else {
                clearInterval(autoRefreshInterval);
                removeRealTimeIndicator();
            }
        }
        
        // Add real-time update indicator
        function addRealTimeIndicator() {
            const header = document.querySelector('.header h1');
            if (header && !header.querySelector('.real-time-indicator')) {
                const indicator = document.createElement('span');
                indicator.className = 'real-time-indicator';
                indicator.title = 'Live updates active';
                header.prepend(indicator);
            }
        }
        
        function removeRealTimeIndicator() {
            const indicator = document.querySelector('.real-time-indicator');
            if (indicator) {
                indicator.remove();
            }
        }
        
        // Enhanced scroll indicators for tables
        function setupTableScrollIndicators() {
            const tableContainers = document.querySelectorAll('.table-container');
            
            tableContainers.forEach(container => {
                function updateScrollIndicators() {
                    const scrollLeft = container.scrollLeft;
                    const scrollWidth = container.scrollWidth;
                    const clientWidth = container.clientWidth;
                    
                    // Update scroll indicator classes
                    container.classList.toggle('scrolled-left', scrollLeft > 0);
                    container.classList.toggle('scrolled-right', scrollLeft < scrollWidth - clientWidth - 1);
                }
                
                // Initial check
                updateScrollIndicators();
                
                // Update on scroll with throttling
                let scrollTimeout;
                container.addEventListener('scroll', () => {
                    clearTimeout(scrollTimeout);
                    scrollTimeout = setTimeout(updateScrollIndicators, 10);
                });
                
                // Update on resize
                window.addEventListener('resize', updateScrollIndicators);
            });
        }
        
        // Smooth animations for new data
        function animateNewData() {
            const cards = document.querySelectorAll('.card');
            cards.forEach((card, index) => {
                card.style.animation = `fadeInUp 0.5s ease ${index * 0.1}s both`;
            });
        }
        
        // Connection status monitoring
        function monitorConnection() {
            window.addEventListener('online', () => {
                showConnectionStatus('Connected', 'success');
            });
            
            window.addEventListener('offline', () => {
                showConnectionStatus('Disconnected', 'error');
                clearInterval(autoRefreshInterval);
            });
        }
        
        function showConnectionStatus(message, type) {
            // Remove existing status
            const existing = document.querySelector('.connection-status');
            if (existing) existing.remove();
            
            const status = document.createElement('div');
            status.className = `connection-status ${type}`;
            status.textContent = message;
            status.style.cssText = `
                position: fixed;
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: ${type === 'success' ? 'var(--success-color)' : 'var(--danger-color)'};
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: 600;
                z-index: 1000;
                animation: slideDown 0.3s ease;
            `;
            
            document.body.appendChild(status);
            
            setTimeout(() => {
                status.style.animation = 'slideUp 0.3s ease';
                setTimeout(() => status.remove(), 300);
            }, 3000);
        }
        
        // Performance monitoring
        function trackPerformance() {
            if ('navigation' in performance) {
                const loadTime = performance.navigation.loadEventEnd - performance.navigation.loadEventStart;
                if (loadTime > 1000) {
                    console.warn(`Dashboard load time: ${loadTime}ms`);
                }
            }
        }
        
        // Initialize everything when DOM is ready
        document.addEventListener('DOMContentLoaded', () => {
            // Initialize auto-refresh
            const autoRefreshCheckbox = document.getElementById('autoRefresh');
            if (autoRefreshCheckbox) {
                autoRefreshCheckbox.addEventListener('change', toggleAutoRefresh);
                toggleAutoRefresh();
            }
            
            // Update status styling with animation
            const statusElement = document.getElementById('status');
            if (statusElement) {
                statusElement.className = 'status ' + statusElement.textContent.toLowerCase();
                statusElement.style.animation = 'fadeIn 0.5s ease';
            }
            
            // Setup table scroll indicators
            setupTableScrollIndicators();
            
            // Add entrance animations
            animateNewData();
            
            // Monitor connection status
            monitorConnection();
            
            // Track performance
            trackPerformance();
            
            // Add smooth refresh button interaction
            const refreshBtn = document.querySelector('.refresh-btn');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', refreshData);
            }
            
            // Last update time display
            lastUpdateTime = new Date();
            const updateTime = document.createElement('small');
            updateTime.style.cssText = 'color: var(--text-muted); margin-left: 10px;';
            updateTime.textContent = `Last updated: ${lastUpdateTime.toLocaleTimeString()}`;
            
            const header = document.querySelector('.header');
            if (header) {
                header.appendChild(updateTime);
            }
        });
        
        // Add CSS animations via JavaScript
        const style = document.createElement('style');
        style.textContent = `
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            @keyframes fadeInUp {
                from { 
                    opacity: 0; 
                    transform: translateY(20px); 
                }
                to { 
                    opacity: 1; 
                    transform: translateY(0); 
                }
            }
            
            @keyframes slideDown {
                from { 
                    opacity: 0; 
                    transform: translateX(-50%) translateY(-20px); 
                }
                to { 
                    opacity: 1; 
                    transform: translateX(-50%) translateY(0); 
                }
            }
            
            @keyframes slideUp {
                from { 
                    opacity: 1; 
                    transform: translateX(-50%) translateY(0); 
                }
                to { 
                    opacity: 0; 
                    transform: translateX(-50%) translateY(-20px); 
                }
            }
            
            .real-time-indicator {
                display: inline-block;
                width: 8px;
                height: 8px;
                background: var(--success-color);
                border-radius: 50%;
                margin-right: 0.5rem;
                animation: pulse 1s ease-in-out infinite alternate;
            }
            
            @keyframes pulse {
                0% { 
                    box-shadow: 0 0 5px rgba(0, 255, 65, 0.5); 
                }
                100% { 
                    box-shadow: 0 0 20px rgba(0, 255, 65, 0.8), 0 0 30px rgba(0, 255, 65, 0.4); 
                }
            }
        `;
        document.head.appendChild(style);
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