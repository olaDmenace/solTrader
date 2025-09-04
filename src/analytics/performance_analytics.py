import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import math

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    timestamp: datetime
    token_address: str
    token_symbol: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: float
    pnl_percentage: float
    hold_time_minutes: float
    gas_fees: float
    discovery_source: str
    exit_reason: str
    is_open: bool = False

@dataclass
class DailyStats:
    date: datetime
    tokens_scanned: int
    tokens_approved: int
    approval_rate: float
    trades_executed: int
    trades_won: int
    trades_lost: int
    win_rate: float
    total_pnl: float
    best_trade: float
    worst_trade: float
    avg_hold_time: float
    gas_fees: float
    api_requests_used: int
    high_momentum_bypasses: int
    portfolio_value: float
    active_positions: int

@dataclass
class WeeklyStats:
    week_start: datetime
    trading_days: int
    total_trades: int
    cumulative_pnl: float
    return_percentage: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trades_per_day: float
    longest_win_streak: int
    longest_loss_streak: int
    best_category: str
    best_hour: str
    source_effectiveness: Dict[str, float]

class PerformanceAnalytics:
    def __init__(self, settings):
        self.settings = settings
        
        # Balance tracking for paper trading
        paper_balance = getattr(settings, 'INITIAL_PAPER_BALANCE', 200.0)
        self.initial_balance = paper_balance
        self.current_balance = paper_balance
        
        # Trade tracking
        self.trades: List[TradeRecord] = []
        self.open_positions: Dict[str, TradeRecord] = {}
        
        # Daily statistics
        self.daily_stats: Dict[str, DailyStats] = {}
        self.current_day_stats = self._init_daily_stats()
        
        # Real-time metrics
        self.real_time_metrics = {
            'portfolio_value': paper_balance,  # Use paper trading balance
            'available_balance': paper_balance,
            'active_positions': 0,
            'todays_pnl': 0.0,
            'total_pnl': 0.0,
            'api_requests_remaining': 333,
            'system_uptime': time.time(),
            'last_trade': None,
            'win_rate_24h': 0.0,
            'approval_rate_24h': 0.0
        }
        
        # Token discovery analytics
        self.discovery_stats = {
            'sources': {
                'trending': {'discovered': 0, 'approved': 0, 'traded': 0, 'profitable': 0},
                'volume': {'discovered': 0, 'approved': 0, 'traded': 0, 'profitable': 0},
                'memescope': {'discovered': 0, 'approved': 0, 'traded': 0, 'profitable': 0}
            },
            'avg_discovery_age': 0.0,
            'quality_score_distribution': defaultdict(int),
            'liquidity_analysis': {'min': 0, 'max': 0, 'avg': 0},
            'hourly_performance': defaultdict(lambda: {'trades': 0, 'pnl': 0.0})
        }
        
        # Performance tracking queues for real-time updates
        self.recent_trades = deque(maxlen=100)  # Last 100 trades
        self.hourly_pnl = deque(maxlen=168)  # 7 days of hourly data
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Risk metrics
        self.risk_metrics = {
            'current_drawdown': 0.0,
            'max_drawdown': 0.0,
            'risk_score': 0.0,
            'position_concentration': 0.0,
            'volatility': 0.0
        }
        
        logger.info("Performance Analytics system initialized")

    def _init_daily_stats(self) -> DailyStats:
        """Initialize daily statistics"""
        return DailyStats(
            date=datetime.now().date(),
            tokens_scanned=0,
            tokens_approved=0,
            approval_rate=0.0,
            trades_executed=0,
            trades_won=0,
            trades_lost=0,
            win_rate=0.0,
            total_pnl=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            avg_hold_time=0.0,
            gas_fees=0.0,
            api_requests_used=0,
            high_momentum_bypasses=0,
            portfolio_value=100.0,
            active_positions=0
        )

    def record_trade_entry(self, token_address: str, token_symbol: str, entry_price: float,
                          quantity: float, gas_fees: float, discovery_source: str) -> str:
        """Record a new trade entry"""
        trade_id = f"{token_address}_{int(time.time())}"
        
        trade = TradeRecord(
            timestamp=datetime.now(),
            token_address=token_address,
            token_symbol=token_symbol,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            pnl=0.0,
            pnl_percentage=0.0,
            hold_time_minutes=0.0,
            gas_fees=gas_fees,
            discovery_source=discovery_source,
            exit_reason="",
            is_open=True
        )
        
        self.open_positions[trade_id] = trade
        self.current_day_stats.trades_executed += 1
        self.current_day_stats.gas_fees += gas_fees
        self.current_day_stats.active_positions = len(self.open_positions)
        
        # Update real-time metrics
        self.real_time_metrics['active_positions'] = len(self.open_positions)
        self.real_time_metrics['last_trade'] = datetime.now()
        
        # Update discovery stats
        source_stats = self.discovery_stats['sources'].get(discovery_source, 
                                                          {'discovered': 0, 'approved': 0, 'traded': 0, 'profitable': 0})
        source_stats['traded'] += 1
        self.discovery_stats['sources'][discovery_source] = source_stats
        
        logger.info(f"Trade entry recorded: {token_symbol} @ ${entry_price:.6f}")
        return trade_id

    def record_trade_exit(self, trade_id: str, exit_price: float, exit_reason: str, gas_fees: float = 0.0):
        """Record a trade exit"""
        if trade_id not in self.open_positions:
            logger.warning(f"Trade ID {trade_id} not found in open positions")
            return
        
        trade = self.open_positions[trade_id]
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.gas_fees += gas_fees
        trade.is_open = False
        
        # Calculate PnL
        trade.pnl = (exit_price - trade.entry_price) * trade.quantity - trade.gas_fees
        trade.pnl_percentage = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        
        # Calculate hold time
        hold_time = datetime.now() - trade.timestamp
        trade.hold_time_minutes = hold_time.total_seconds() / 60
        
        # Move to completed trades
        self.trades.append(trade)
        self.recent_trades.append(trade)
        del self.open_positions[trade_id]
        
        # Update daily stats
        self.current_day_stats.total_pnl += trade.pnl
        self.current_day_stats.gas_fees += gas_fees
        self.current_day_stats.active_positions = len(self.open_positions)
        
        if trade.pnl > 0:
            self.current_day_stats.trades_won += 1
        else:
            self.current_day_stats.trades_lost += 1
        
        # Update win rate
        total_completed = self.current_day_stats.trades_won + self.current_day_stats.trades_lost
        if total_completed > 0:
            self.current_day_stats.win_rate = (self.current_day_stats.trades_won / total_completed) * 100
        
        # Update best/worst trades
        if trade.pnl_percentage > self.current_day_stats.best_trade:
            self.current_day_stats.best_trade = trade.pnl_percentage
        if trade.pnl_percentage < self.current_day_stats.worst_trade:
            self.current_day_stats.worst_trade = trade.pnl_percentage
        
        # Update real-time metrics
        self.real_time_metrics['active_positions'] = len(self.open_positions)
        self.real_time_metrics['todays_pnl'] = self.current_day_stats.total_pnl
        self.real_time_metrics['total_pnl'] += trade.pnl
        
        # Update discovery source profitability
        if trade.pnl > 0:
            source_stats = self.discovery_stats['sources'].get(trade.discovery_source)
            if source_stats:
                source_stats['profitable'] += 1
        
        # Update hourly performance
        hour = trade.timestamp.hour
        self.discovery_stats['hourly_performance'][hour]['trades'] += 1
        self.discovery_stats['hourly_performance'][hour]['pnl'] += trade.pnl
        
        logger.info(f"Trade exit recorded: {trade.token_symbol} PnL: ${trade.pnl:.4f} ({trade.pnl_percentage:.1f}%)")

    def update_scanner_stats(self, tokens_scanned: int, tokens_approved: int, 
                           high_momentum_bypasses: int, api_requests_used: int):
        """Update scanner-related statistics"""
        self.current_day_stats.tokens_scanned += tokens_scanned
        self.current_day_stats.tokens_approved += tokens_approved
        self.current_day_stats.high_momentum_bypasses += high_momentum_bypasses
        self.current_day_stats.api_requests_used = api_requests_used
        
        # Calculate approval rate
        if self.current_day_stats.tokens_scanned > 0:
            self.current_day_stats.approval_rate = (
                self.current_day_stats.tokens_approved / self.current_day_stats.tokens_scanned
            ) * 100
        
        # Update real-time metrics
        self.real_time_metrics['approval_rate_24h'] = self.current_day_stats.approval_rate
        self.real_time_metrics['api_requests_remaining'] = 333 - api_requests_used

    def update_discovery_analytics(self, source: str, discovered: int, approved: int, 
                                 avg_age: float, liquidity_stats: Dict):
        """Update token discovery analytics"""
        if source in self.discovery_stats['sources']:
            stats = self.discovery_stats['sources'][source]
            stats['discovered'] += discovered
            stats['approved'] += approved
        
        # Update average discovery age
        self.discovery_stats['avg_discovery_age'] = avg_age
        
        # Update liquidity analysis
        if liquidity_stats:
            self.discovery_stats['liquidity_analysis'].update(liquidity_stats)

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time performance metrics"""
        # Calculate system uptime
        uptime_seconds = time.time() - self.real_time_metrics['system_uptime']
        uptime_hours = uptime_seconds / 3600
        
        # Calculate 24h win rate from recent trades
        if self.recent_trades:
            recent_wins = sum(1 for trade in self.recent_trades if trade.pnl > 0)
            self.real_time_metrics['win_rate_24h'] = (recent_wins / len(self.recent_trades)) * 100
        
        return {
            'current_portfolio_value': self.real_time_metrics['portfolio_value'],
            'available_balance': self.real_time_metrics['available_balance'],
            'current_balance': self.current_balance,  # Add this for dashboard compatibility
            'active_positions': self.real_time_metrics['active_positions'],
            'todays_pnl': self.real_time_metrics['todays_pnl'],
            'total_pnl': self.real_time_metrics['total_pnl'],
            'api_requests_remaining': self.real_time_metrics['api_requests_remaining'],
            'system_uptime_hours': uptime_hours,
            'last_trade': self.real_time_metrics['last_trade'].isoformat() if self.real_time_metrics['last_trade'] else None,
            'win_rate_24h': self.real_time_metrics['win_rate_24h'],
            'approval_rate_24h': self.real_time_metrics['approval_rate_24h'],
            'current_drawdown': self.risk_metrics['current_drawdown'],
            'risk_score': self.risk_metrics['risk_score'],
            'win_rate': self.real_time_metrics['win_rate_24h']  # Add win_rate alias for dashboard
        }

    def get_daily_breakdown(self) -> Dict[str, Any]:
        """Get comprehensive daily statistics"""
        stats = asdict(self.current_day_stats)
        
        # Add trading mode information
        stats['paper_trading_mode'] = getattr(self.settings, 'PAPER_TRADING', True)
        
        # Calculate average hold time
        if self.recent_trades:
            hold_times = [trade.hold_time_minutes for trade in self.recent_trades if not trade.is_open]
            if hold_times:
                stats['avg_hold_time'] = statistics.mean(hold_times)
        
        # Add source breakdown
        stats['source_breakdown'] = {}
        for source, data in self.discovery_stats['sources'].items():
            effectiveness = 0
            if data['discovered'] > 0:
                effectiveness = (data['approved'] / data['discovered']) * 100
            
            profitability = 0
            if data['traded'] > 0:
                profitability = (data['profitable'] / data['traded']) * 100
            
            stats['source_breakdown'][source] = {
                'discovered': data['discovered'],
                'approved': data['approved'],
                'traded': data['traded'],
                'profitable': data['profitable'],
                'effectiveness': f"{effectiveness:.1f}%",
                'profitability': f"{profitability:.1f}%"
            }
        
        return stats

    def get_weekly_report(self) -> WeeklyStats:
        """Generate comprehensive weekly performance report"""
        week_start = datetime.now() - timedelta(days=7)
        
        # Filter trades from last week
        week_trades = [
            trade for trade in self.trades 
            if trade.timestamp >= week_start and not trade.is_open
        ]
        
        if not week_trades:
            return WeeklyStats(
                week_start=week_start,
                trading_days=0,
                total_trades=0,
                cumulative_pnl=0.0,
                return_percentage=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                avg_trades_per_day=0.0,
                longest_win_streak=0,
                longest_loss_streak=0,
                best_category="N/A",
                best_hour="N/A",
                source_effectiveness={}
            )
        
        # Calculate basic metrics
        total_trades = len(week_trades)
        cumulative_pnl = sum(trade.pnl for trade in week_trades)
        
        # Calculate return percentage
        initial_value = 100.0  # Starting portfolio value
        return_percentage = (cumulative_pnl / initial_value) * 100
        
        # Calculate Sharpe ratio
        daily_returns = self._calculate_daily_returns(week_trades)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        
        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown(week_trades)
        
        # Calculate streaks
        win_streak, loss_streak = self._calculate_streaks(week_trades)
        
        # Find best performing category and hour
        best_category = self._find_best_category(week_trades)
        best_hour = self._find_best_hour(week_trades)
        
        # Calculate source effectiveness
        source_effectiveness = self._calculate_source_effectiveness(week_trades)
        
        # Count trading days
        trading_days = len(set(trade.timestamp.date() for trade in week_trades))
        avg_trades_per_day = total_trades / max(trading_days, 1)
        
        return WeeklyStats(
            week_start=week_start,
            trading_days=trading_days,
            total_trades=total_trades,
            cumulative_pnl=cumulative_pnl,
            return_percentage=return_percentage,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_trades_per_day=avg_trades_per_day,
            longest_win_streak=win_streak,
            longest_loss_streak=loss_streak,
            best_category=best_category,
            best_hour=best_hour,
            source_effectiveness=source_effectiveness
        )

    def get_token_discovery_intelligence(self) -> Dict[str, Any]:
        """Get detailed token discovery analytics"""
        total_discovered = sum(stats['discovered'] for stats in self.discovery_stats['sources'].values())
        total_approved = sum(stats['approved'] for stats in self.discovery_stats['sources'].values())
        
        intelligence = {
            'total_discovered': total_discovered,
            'total_approved': total_approved,
            'overall_approval_rate': (total_approved / total_discovered * 100) if total_discovered > 0 else 0,
            'avg_discovery_age_minutes': self.discovery_stats['avg_discovery_age'],
            'source_effectiveness': {},
            'hourly_patterns': self._analyze_hourly_patterns(),
            'liquidity_distribution': self.discovery_stats['liquidity_analysis'],
            'quality_trends': dict(self.discovery_stats['quality_score_distribution']),
            'discovery_timing_analysis': self._analyze_discovery_timing()
        }
        
        # Calculate source effectiveness
        for source, stats in self.discovery_stats['sources'].items():
            if stats['discovered'] > 0:
                effectiveness = (stats['approved'] / stats['discovered']) * 100
                profitability = (stats['profitable'] / stats['traded']) * 100 if stats['traded'] > 0 else 0
                
                intelligence['source_effectiveness'][source] = {
                    'discovery_rate': stats['discovered'],
                    'approval_rate': f"{effectiveness:.1f}%",
                    'trade_rate': stats['traded'],
                    'profitability_rate': f"{profitability:.1f}%",
                    'avg_pnl': self._calculate_avg_pnl_by_source(source)
                }
        
        return intelligence

    def _calculate_daily_returns(self, trades: List[TradeRecord]) -> List[float]:
        """Calculate daily returns from trades"""
        daily_pnl = defaultdict(float)
        for trade in trades:
            date = trade.timestamp.date()
            daily_pnl[date] += trade.pnl
        
        return list(daily_pnl.values())

    def _calculate_sharpe_ratio(self, daily_returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(daily_returns) < 2:
            return 0.0
        
        mean_return = statistics.mean(daily_returns)
        std_dev = statistics.stdev(daily_returns)
        
        if std_dev == 0:
            return 0.0
        
        # Assuming risk-free rate of 0 for simplicity
        return mean_return / std_dev

    def _calculate_max_drawdown(self, trades: List[TradeRecord]) -> float:
        """Calculate maximum drawdown"""
        if not trades:
            return 0.0
        
        cumulative_pnl = 0.0
        peak = 0.0
        max_drawdown = 0.0
        
        for trade in sorted(trades, key=lambda x: x.timestamp):
            cumulative_pnl += trade.pnl
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            drawdown = (peak - cumulative_pnl) / max(peak, 1) * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown

    def _calculate_streaks(self, trades: List[TradeRecord]) -> Tuple[int, int]:
        """Calculate longest winning and losing streaks"""
        if not trades:
            return 0, 0
        
        sorted_trades = sorted(trades, key=lambda x: x.timestamp)
        
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for trade in sorted_trades:
            if trade.pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        return max_win_streak, max_loss_streak

    def _find_best_category(self, trades: List[TradeRecord]) -> str:
        """Find best performing token discovery source"""
        source_pnl = defaultdict(float)
        for trade in trades:
            source_pnl[trade.discovery_source] += trade.pnl
        
        if not source_pnl:
            return "N/A"
        
        return max(source_pnl.items(), key=lambda x: x[1])[0]

    def _find_best_hour(self, trades: List[TradeRecord]) -> str:
        """Find best performing hour of day"""
        hourly_pnl = defaultdict(float)
        for trade in trades:
            hour = trade.timestamp.hour
            hourly_pnl[hour] += trade.pnl
        
        if not hourly_pnl:
            return "N/A"
        
        best_hour = max(hourly_pnl.items(), key=lambda x: x[1])[0]
        return f"{best_hour:02d}:00"

    def _calculate_source_effectiveness(self, trades: List[TradeRecord]) -> Dict[str, float]:
        """Calculate effectiveness percentage by source"""
        source_stats = defaultdict(lambda: {'total': 0, 'profitable': 0})
        
        for trade in trades:
            source_stats[trade.discovery_source]['total'] += 1
            if trade.pnl > 0:
                source_stats[trade.discovery_source]['profitable'] += 1
        
        effectiveness = {}
        for source, stats in source_stats.items():
            if stats['total'] > 0:
                effectiveness[source] = (stats['profitable'] / stats['total']) * 100
        
        return effectiveness

    def _analyze_hourly_patterns(self) -> Dict[str, Any]:
        """Analyze trading patterns by hour"""
        hourly_data = self.discovery_stats['hourly_performance']
        
        best_hours = sorted(
            hourly_data.items(),
            key=lambda x: x[1]['pnl'] / max(x[1]['trades'], 1),
            reverse=True
        )[:3]
        
        return {
            'best_hours': [f"{hour:02d}:00" for hour, _ in best_hours],
            'total_hourly_trades': {f"{hour:02d}:00": data['trades'] for hour, data in hourly_data.items()},
            'hourly_pnl': {f"{hour:02d}:00": data['pnl'] for hour, data in hourly_data.items()}
        }

    def _analyze_discovery_timing(self) -> Dict[str, Any]:
        """Analyze token discovery timing patterns"""
        if not self.trades:
            return {'avg_discovery_to_trade': 0, 'timing_efficiency': 'N/A'}
        
        # This would require additional data about discovery timestamps
        # For now, return placeholder analysis
        return {
            'avg_discovery_to_trade': self.discovery_stats['avg_discovery_age'],
            'timing_efficiency': 'Good' if self.discovery_stats['avg_discovery_age'] < 30 else 'Needs Improvement'
        }

    def _calculate_avg_pnl_by_source(self, source: str) -> float:
        """Calculate average PnL for a specific discovery source"""
        source_trades = [trade for trade in self.trades if trade.discovery_source == source]
        if not source_trades:
            return 0.0
        
        return sum(trade.pnl for trade in source_trades) / len(source_trades)

    def reset_daily_stats(self):
        """Reset daily statistics (called at midnight)"""
        # Save current day stats
        date_key = self.current_day_stats.date.isoformat()
        self.daily_stats[date_key] = self.current_day_stats
        
        # Initialize new day
        self.current_day_stats = self._init_daily_stats()
        self.current_day_stats.portfolio_value = self.real_time_metrics['portfolio_value']
        
        logger.info(f"Daily statistics reset for new day: {self.current_day_stats.date}")

    def export_data(self, format_type: str = "json") -> str:
        """Export analytics data in specified format"""
        data = {
            'trades': [asdict(trade) for trade in self.trades],
            'daily_stats': {k: asdict(v) for k, v in self.daily_stats.items()},
            'current_day': asdict(self.current_day_stats),
            'real_time_metrics': self.get_real_time_metrics(),
            'discovery_analytics': self.get_token_discovery_intelligence(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        if format_type.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            # Could add CSV export here
            return json.dumps(data, indent=2, default=str)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'real_time_metrics': self.get_real_time_metrics(),
            'daily_breakdown': self.get_daily_breakdown(),
            'weekly_report': asdict(self.get_weekly_report()),
            'discovery_intelligence': self.get_token_discovery_intelligence(),
            'risk_metrics': self.risk_metrics.copy(),
            'system_health': {
                'total_trades': len(self.trades),
                'open_positions': len(self.open_positions),
                'data_points': len(self.recent_trades),
                'last_update': datetime.now().isoformat()
            }
        }
    
    def get_historical_analysis(self) -> Dict[str, Any]:
        """Get historical analysis data for dashboard"""
        return {
            'total_trades': len(self.trades),
            'win_rate': self.get_real_time_metrics().get('win_rate', 0.0),
            'total_pnl': self.current_balance - self.initial_balance,
            'avg_trade_pnl': sum(t.pnl for t in self.trades) / max(len(self.trades), 1),
            'best_trade': max([t.pnl for t in self.trades], default=0.0),
            'worst_trade': min([t.pnl for t in self.trades], default=0.0),
            'trading_volume': sum(abs(t.quantity * t.price) for t in self.trades),
            'daily_returns': self.get_daily_breakdown(),
            'sharpe_ratio': self.risk_metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': self.risk_metrics.get('max_drawdown', 0.0)
        }
    
    def get_current_positions(self) -> Dict[str, Any]:
        """Get current positions data for dashboard"""
        positions_data = {}
        
        for token, position in self.open_positions.items():
            positions_data[token] = {
                'quantity': position.get('quantity', 0.0),
                'entry_price': position.get('entry_price', 0.0),
                'current_value': position.get('current_value', 0.0),
                'pnl': position.get('unrealized_pnl', 0.0),
                'pnl_percentage': position.get('pnl_percentage', 0.0),
                'entry_time': position.get('entry_time', datetime.now()).isoformat(),
                'strategy': position.get('strategy', 'unknown')
            }
        
        return {
            'positions': positions_data,
            'total_positions': len(self.open_positions),
            'total_value': sum(pos.get('current_value', 0.0) for pos in self.open_positions.values()),
            'total_unrealized_pnl': sum(pos.get('unrealized_pnl', 0.0) for pos in self.open_positions.values())
        }