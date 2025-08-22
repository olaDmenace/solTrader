#!/usr/bin/env python3
"""
Performance Analytics System for SolTrader
Tracks daily/weekly P&L summaries, trading patterns, and performance metrics
"""
import json
import logging
import time
import asyncio
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

@dataclass
class TradeMetrics:
    """Individual trade metrics"""
    timestamp: str
    token_address: str
    token_symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    profit_loss_usd: float
    profit_loss_sol: float
    profit_percentage: float
    duration_minutes: float
    trade_type: str  # "buy", "sell"
    source: str  # "trending", "volume", etc.

@dataclass
class DailyPerformance:
    """Daily performance summary"""
    date: str
    total_trades: int
    profitable_trades: int
    losing_trades: int
    win_rate: float
    total_profit_loss_usd: float
    total_profit_loss_sol: float
    best_trade_profit: float
    worst_trade_loss: float
    average_trade_duration: float
    total_volume_usd: float
    unique_tokens_traded: int
    trading_hours_active: float
    
    # Advanced metrics
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    average_win: float
    average_loss: float

@dataclass
class WeeklyPerformance:
    """Weekly performance summary"""
    week_start: str
    week_end: str
    days_active: int
    total_trades: int
    win_rate: float
    total_profit_loss_usd: float
    total_profit_loss_sol: float
    daily_performances: List[DailyPerformance]
    
    # Weekly specific metrics
    consistency_score: float  # How consistent daily returns are
    best_day_profit: float
    worst_day_loss: float
    weekly_sharpe: float

class PerformanceAnalytics:
    """Main performance analytics engine"""
    
    def __init__(self, data_directory: str = "analytics"):
        self.data_dir = Path(data_directory)
        self.data_dir.mkdir(exist_ok=True)
        
        # Data storage files
        self.trades_file = self.data_dir / "trades_history.json"
        self.daily_file = self.data_dir / "daily_performance.json"
        self.weekly_file = self.data_dir / "weekly_performance.json"
        
        # In-memory data
        self.trades_cache = []
        self.daily_cache = {}
        self.weekly_cache = {}
        
        # Load existing data
        self._load_data()
        
        # Performance tracking
        self.last_analysis_time = time.time()
        self.analysis_interval = 300  # Analyze every 5 minutes
        
    def _load_data(self):
        """Load existing performance data"""
        try:
            # Load trades
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    trades_data = json.load(f)
                    self.trades_cache = [TradeMetrics(**trade) for trade in trades_data]
            
            # Load daily performance
            if self.daily_file.exists():
                with open(self.daily_file, 'r') as f:
                    daily_data = json.load(f)
                    self.daily_cache = {
                        date_str: DailyPerformance(**perf) 
                        for date_str, perf in daily_data.items()
                    }
            
            # Load weekly performance
            if self.weekly_file.exists():
                with open(self.weekly_file, 'r') as f:
                    weekly_data = json.load(f)
                    self.weekly_cache = {
                        week_key: WeeklyPerformance(**perf)
                        for week_key, perf in weekly_data.items()
                    }
                    
            logger.info(f"Loaded {len(self.trades_cache)} trades, {len(self.daily_cache)} daily summaries")
            
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            # Initialize empty data if loading fails
            self.trades_cache = []
            self.daily_cache = {}
            self.weekly_cache = {}
    
    def _save_data(self):
        """Save performance data to files"""
        try:
            # Save trades
            with open(self.trades_file, 'w') as f:
                json.dump([asdict(trade) for trade in self.trades_cache], f, indent=2)
            
            # Save daily performance
            with open(self.daily_file, 'w') as f:
                json.dump({k: asdict(v) for k, v in self.daily_cache.items()}, f, indent=2)
            
            # Save weekly performance
            with open(self.weekly_file, 'w') as f:
                # Convert weekly performance (contains daily performance objects)
                weekly_serializable = {}
                for k, v in self.weekly_cache.items():
                    weekly_data = asdict(v)
                    # Convert daily_performances list
                    weekly_data['daily_performances'] = [asdict(dp) for dp in v.daily_performances]
                    weekly_serializable[k] = weekly_data
                json.dump(weekly_serializable, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def add_trade(self, trade_data: Dict[str, Any]):
        """Add a new trade to the analytics system"""
        try:
            # Convert trade data to TradeMetrics
            trade = TradeMetrics(
                timestamp=trade_data.get('timestamp', datetime.now().isoformat()),
                token_address=trade_data.get('token_address', ''),
                token_symbol=trade_data.get('token_symbol', 'UNKNOWN'),
                entry_price=float(trade_data.get('entry_price', 0)),
                exit_price=float(trade_data.get('exit_price', 0)),
                quantity=float(trade_data.get('quantity', 0)),
                profit_loss_usd=float(trade_data.get('profit_loss_usd', 0)),
                profit_loss_sol=float(trade_data.get('profit_loss_sol', 0)),
                profit_percentage=float(trade_data.get('profit_percentage', 0)),
                duration_minutes=float(trade_data.get('duration_minutes', 0)),
                trade_type=trade_data.get('trade_type', 'buy'),
                source=trade_data.get('source', 'unknown')
            )
            
            self.trades_cache.append(trade)
            
            # Trigger analysis update
            self._update_analytics()
            
            logger.info(f"Added trade: {trade.token_symbol} P&L: ${trade.profit_loss_usd:.2f}")
            
        except Exception as e:
            logger.error(f"Error adding trade: {e}")
    
    def import_from_dashboard_data(self, dashboard_data_file: str = "dashboard_data.json"):
        """Import existing trades from dashboard data"""
        try:
            dashboard_path = Path(dashboard_data_file)
            if not dashboard_path.exists():
                logger.warning(f"Dashboard data file not found: {dashboard_data_file}")
                return 0
            
            with open(dashboard_path, 'r') as f:
                data = json.load(f)
            
            trades = data.get('trades', [])
            imported_count = 0
            
            for trade in trades:
                try:
                    # Check if trade already exists (avoid duplicates)
                    existing = any(
                        t.timestamp == trade.get('timestamp') and 
                        t.token_address == trade.get('token_address')
                        for t in self.trades_cache
                    )
                    
                    if not existing:
                        # Convert dashboard trade format to analytics format
                        trade_metrics = {
                            'timestamp': trade.get('timestamp', datetime.now().isoformat()),
                            'token_address': trade.get('token_address', ''),
                            'token_symbol': trade.get('token_symbol', 'UNKNOWN'),
                            'entry_price': trade.get('entry_price', 0),
                            'exit_price': trade.get('exit_price', 0),
                            'quantity': trade.get('quantity', 0),
                            'profit_loss_usd': trade.get('profit_usd', 0),
                            'profit_loss_sol': trade.get('profit_sol', 0),
                            'profit_percentage': trade.get('profit_percentage', 0),
                            'duration_minutes': trade.get('duration_minutes', 0),
                            'trade_type': 'buy',  # Default
                            'source': trade.get('source', 'imported')
                        }
                        
                        self.add_trade(trade_metrics)
                        imported_count += 1
                        
                except Exception as e:
                    logger.error(f"Error importing trade: {e}")
                    continue
            
            logger.info(f"Imported {imported_count} trades from dashboard data")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing dashboard data: {e}")
            return 0
    
    def _update_analytics(self):
        """Update daily and weekly analytics"""
        if not self.trades_cache:
            return
        
        # Group trades by date
        trades_by_date = defaultdict(list)
        for trade in self.trades_cache:
            trade_date = datetime.fromisoformat(trade.timestamp).date().isoformat()
            trades_by_date[trade_date].append(trade)
        
        # Calculate daily performance for each date
        for date_str, date_trades in trades_by_date.items():
            self.daily_cache[date_str] = self._calculate_daily_performance(date_str, date_trades)
        
        # Calculate weekly performance
        self._calculate_weekly_performance()
        
        # Save updated data
        self._save_data()
    
    def _calculate_daily_performance(self, date_str: str, trades: List[TradeMetrics]) -> DailyPerformance:
        """Calculate performance metrics for a single day"""
        if not trades:
            return DailyPerformance(
                date=date_str, total_trades=0, profitable_trades=0, losing_trades=0,
                win_rate=0, total_profit_loss_usd=0, total_profit_loss_sol=0,
                best_trade_profit=0, worst_trade_loss=0, average_trade_duration=0,
                total_volume_usd=0, unique_tokens_traded=0, trading_hours_active=0,
                sharpe_ratio=0, max_drawdown=0, profit_factor=0, average_win=0, average_loss=0
            )
        
        # Basic metrics
        total_trades = len(trades)
        profitable_trades = len([t for t in trades if t.profit_loss_usd > 0])
        losing_trades = len([t for t in trades if t.profit_loss_usd < 0])
        win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_profit_loss_usd = sum(t.profit_loss_usd for t in trades)
        total_profit_loss_sol = sum(t.profit_loss_sol for t in trades)
        
        profits = [t.profit_loss_usd for t in trades if t.profit_loss_usd > 0]
        losses = [abs(t.profit_loss_usd) for t in trades if t.profit_loss_usd < 0]
        
        best_trade_profit = max(profits) if profits else 0
        worst_trade_loss = max(losses) if losses else 0
        average_win = statistics.mean(profits) if profits else 0
        average_loss = statistics.mean(losses) if losses else 0
        
        # Duration and volume metrics
        average_trade_duration = statistics.mean([t.duration_minutes for t in trades])
        total_volume_usd = sum(abs(t.entry_price * t.quantity) for t in trades)
        unique_tokens_traded = len(set(t.token_address for t in trades))
        
        # Trading hours (estimate based on trade spread)
        timestamps = [datetime.fromisoformat(t.timestamp) for t in trades]
        if len(timestamps) > 1:
            trading_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600
            trading_hours_active = min(trading_span, 24)  # Cap at 24 hours
        else:
            trading_hours_active = 1  # Assume 1 hour if only one trade
        
        # Advanced metrics
        returns = [t.profit_percentage for t in trades]
        
        # Sharpe ratio (simplified)
        if len(returns) > 1:
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Max drawdown (simplified)
        cumulative_returns = []
        cumulative = 0
        for ret in returns:
            cumulative += ret
            cumulative_returns.append(cumulative)
        
        if cumulative_returns:
            peak = cumulative_returns[0]
            max_drawdown = 0
            for value in cumulative_returns:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100 if peak != 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            max_drawdown = 0
        
        # Profit factor
        total_gross_profit = sum(profits) if profits else 0
        total_gross_loss = sum(losses) if losses else 0
        profit_factor = (total_gross_profit / total_gross_loss) if total_gross_loss > 0 else 0
        
        return DailyPerformance(
            date=date_str,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit_loss_usd=total_profit_loss_usd,
            total_profit_loss_sol=total_profit_loss_sol,
            best_trade_profit=best_trade_profit,
            worst_trade_loss=worst_trade_loss,
            average_trade_duration=average_trade_duration,
            total_volume_usd=total_volume_usd,
            unique_tokens_traded=unique_tokens_traded,
            trading_hours_active=trading_hours_active,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss
        )
    
    def _calculate_weekly_performance(self):
        """Calculate weekly performance summaries"""
        if not self.daily_cache:
            return
        
        # Group daily performances by week
        weeks = defaultdict(list)
        for date_str, daily_perf in self.daily_cache.items():
            date_obj = datetime.fromisoformat(date_str).date()
            # Get Monday of the week (ISO week)
            monday = date_obj - timedelta(days=date_obj.weekday())
            week_key = monday.isoformat()
            weeks[week_key].append(daily_perf)
        
        # Calculate weekly summaries
        for week_start, daily_performances in weeks.items():
            if not daily_performances:
                continue
            
            # Sort by date
            daily_performances.sort(key=lambda x: x.date)
            
            week_end = (datetime.fromisoformat(week_start).date() + timedelta(days=6)).isoformat()
            
            # Aggregate weekly metrics
            total_trades = sum(dp.total_trades for dp in daily_performances)
            total_profit_loss_usd = sum(dp.total_profit_loss_usd for dp in daily_performances)
            total_profit_loss_sol = sum(dp.total_profit_loss_sol for dp in daily_performances)
            
            profitable_days = len([dp for dp in daily_performances if dp.total_profit_loss_usd > 0])
            win_rate = (profitable_days / len(daily_performances)) * 100 if daily_performances else 0
            
            # Daily P&L for consistency calculation
            daily_pnl = [dp.total_profit_loss_usd for dp in daily_performances]
            consistency_score = (1 / statistics.stdev(daily_pnl)) * 100 if len(daily_pnl) > 1 and statistics.stdev(daily_pnl) > 0 else 0
            
            best_day_profit = max(daily_pnl) if daily_pnl else 0
            worst_day_loss = min(daily_pnl) if daily_pnl else 0
            
            # Weekly Sharpe ratio
            if len(daily_pnl) > 1:
                avg_daily_return = statistics.mean(daily_pnl)
                std_daily_return = statistics.stdev(daily_pnl)
                weekly_sharpe = (avg_daily_return / std_daily_return) if std_daily_return > 0 else 0
            else:
                weekly_sharpe = 0
            
            weekly_perf = WeeklyPerformance(
                week_start=week_start,
                week_end=week_end,
                days_active=len(daily_performances),
                total_trades=total_trades,
                win_rate=win_rate,
                total_profit_loss_usd=total_profit_loss_usd,
                total_profit_loss_sol=total_profit_loss_sol,
                daily_performances=daily_performances,
                consistency_score=consistency_score,
                best_day_profit=best_day_profit,
                worst_day_loss=worst_day_loss,
                weekly_sharpe=weekly_sharpe
            )
            
            self.weekly_cache[week_start] = weekly_perf
    
    def get_daily_summary(self, date_str: Optional[str] = None) -> Optional[DailyPerformance]:
        """Get daily performance summary"""
        if date_str is None:
            date_str = date.today().isoformat()
        
        return self.daily_cache.get(date_str)
    
    def get_weekly_summary(self, week_start: Optional[str] = None) -> Optional[WeeklyPerformance]:
        """Get weekly performance summary"""
        if week_start is None:
            # Get current week's Monday
            today = date.today()
            monday = today - timedelta(days=today.weekday())
            week_start = monday.isoformat()
        
        return self.weekly_cache.get(week_start)
    
    def get_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for recent days"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        recent_daily = {}
        total_trades = 0
        total_profit_usd = 0
        total_profit_sol = 0
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.isoformat()
            daily_perf = self.daily_cache.get(date_str)
            
            if daily_perf:
                recent_daily[date_str] = daily_perf
                total_trades += daily_perf.total_trades
                total_profit_usd += daily_perf.total_profit_loss_usd
                total_profit_sol += daily_perf.total_profit_loss_sol
            
            current_date += timedelta(days=1)
        
        return {
            'period_days': days,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'daily_summaries': recent_daily,
            'total_trades': total_trades,
            'total_profit_usd': total_profit_usd,
            'total_profit_sol': total_profit_sol,
            'active_days': len(recent_daily),
            'average_daily_profit': total_profit_usd / len(recent_daily) if recent_daily else 0
        }
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        return {
            'total_trades_tracked': len(self.trades_cache),
            'daily_summaries_count': len(self.daily_cache),
            'weekly_summaries_count': len(self.weekly_cache),
            'date_range': {
                'first_trade': min([t.timestamp for t in self.trades_cache]) if self.trades_cache else None,
                'last_trade': max([t.timestamp for t in self.trades_cache]) if self.trades_cache else None
            },
            'data_files': {
                'trades': str(self.trades_file),
                'daily': str(self.daily_file),
                'weekly': str(self.weekly_file)
            },
            'last_analysis': self.last_analysis_time
        }

# Global analytics instance
performance_analytics = PerformanceAnalytics()

def add_trade_to_analytics(trade_data: Dict[str, Any]):
    """Convenience function to add trade to analytics"""
    performance_analytics.add_trade(trade_data)

def get_daily_performance(date_str: str = None) -> Optional[DailyPerformance]:
    """Convenience function to get daily performance"""
    return performance_analytics.get_daily_summary(date_str)

def get_weekly_performance(week_start: str = None) -> Optional[WeeklyPerformance]:
    """Convenience function to get weekly performance"""
    return performance_analytics.get_weekly_summary(week_start)

def import_existing_trades():
    """Import trades from existing dashboard data"""
    return performance_analytics.import_from_dashboard_data()

def get_analytics_stats():
    """Get analytics system statistics"""
    return performance_analytics.get_analytics_summary()