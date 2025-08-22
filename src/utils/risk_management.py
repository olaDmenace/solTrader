#!/usr/bin/env python3
"""
Risk Management Metrics for SolTrader
Advanced risk analysis including drawdown tracking, win streaks, and position sizing
"""
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import statistics

logger = logging.getLogger(__name__)

@dataclass
class DrawdownEvent:
    """Represents a drawdown period"""
    start_date: str
    end_date: str
    start_balance: float
    peak_balance: float
    trough_balance: float
    drawdown_amount: float
    drawdown_percentage: float
    recovery_date: Optional[str]
    duration_days: int
    trades_during_drawdown: int

@dataclass
class WinStreakEvent:
    """Represents a winning or losing streak"""
    streak_type: str  # 'win' or 'loss'
    start_date: str
    end_date: str
    streak_length: int
    total_profit_loss: float
    average_trade_size: float
    largest_win: float
    largest_loss: float
    trades: List[str]  # Trade IDs or timestamps

@dataclass
class RiskMetrics:
    """Current risk metrics snapshot"""
    current_balance: float
    peak_balance: float
    current_drawdown: float
    max_drawdown_ever: float
    max_drawdown_30d: float
    
    # Streak information
    current_streak_type: str
    current_streak_length: int
    longest_win_streak: int
    longest_loss_streak: int
    
    # Position sizing metrics
    average_position_size: float
    largest_position_size: float
    position_size_consistency: float  # Standard deviation
    
    # Risk ratios
    sharpe_ratio_30d: float
    sortino_ratio_30d: float  # Downside deviation only
    calmar_ratio_30d: float   # Return / Max Drawdown
    
    # Volatility metrics
    daily_volatility: float
    maximum_daily_loss: float
    value_at_risk_95: float   # 95% VaR
    
    # Trade distribution
    win_rate_30d: float
    average_win_30d: float
    average_loss_30d: float
    profit_factor_30d: float
    
    # Risk alerts
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    active_warnings: List[str]

class RiskManager:
    """Advanced risk management and analysis system"""
    
    def __init__(self, data_directory: str = "analytics"):
        self.data_dir = Path(data_directory)
        self.data_dir.mkdir(exist_ok=True)
        
        # Risk-specific data files
        self.drawdowns_file = self.data_dir / "drawdown_events.json"
        self.streaks_file = self.data_dir / "streak_events.json"
        self.risk_metrics_file = self.data_dir / "risk_metrics.json"
        
        # Risk tracking data
        self.drawdown_events: List[DrawdownEvent] = []
        self.streak_events: List[WinStreakEvent] = []
        self.daily_balances: deque = deque(maxlen=365)  # Keep 1 year of daily balances
        self.trade_sequence: deque = deque(maxlen=1000)  # Keep last 1000 trades
        
        # Current state
        self.current_balance = 0.0
        self.peak_balance = 0.0
        self.current_streak_type = None
        self.current_streak_count = 0
        self.current_streak_start = None
        
        # Risk thresholds (configurable)
        self.risk_thresholds = {
            'max_drawdown_warning': 10.0,     # Warn at 10% drawdown
            'max_drawdown_critical': 20.0,    # Critical at 20% drawdown
            'max_daily_loss_warning': 5.0,    # Warn at 5% daily loss
            'max_daily_loss_critical': 10.0,  # Critical at 10% daily loss
            'min_sharpe_ratio': 1.0,          # Warn below 1.0 Sharpe
            'max_position_size_pct': 25.0,    # Warn if position > 25% of balance
        }
        
        # Load existing data
        self._load_risk_data()
    
    def _load_risk_data(self):
        """Load existing risk management data"""
        try:
            # Load drawdown events
            if self.drawdowns_file.exists():
                with open(self.drawdowns_file, 'r') as f:
                    drawdown_data = json.load(f)
                    self.drawdown_events = [DrawdownEvent(**dd) for dd in drawdown_data]
            
            # Load streak events
            if self.streaks_file.exists():
                with open(self.streaks_file, 'r') as f:
                    streak_data = json.load(f)
                    self.streak_events = [WinStreakEvent(**se) for se in streak_data]
            
            # Load risk metrics history
            if self.risk_metrics_file.exists():
                with open(self.risk_metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    # Load daily balances and trade sequence
                    self.daily_balances = deque(metrics_data.get('daily_balances', []), maxlen=365)
                    self.trade_sequence = deque(metrics_data.get('trade_sequence', []), maxlen=1000)
                    self.current_balance = metrics_data.get('current_balance', 0.0)
                    self.peak_balance = metrics_data.get('peak_balance', 0.0)
            
            logger.info(f"Loaded {len(self.drawdown_events)} drawdown events, {len(self.streak_events)} streak events")
            
        except Exception as e:
            logger.error(f"Error loading risk data: {e}")
    
    def _save_risk_data(self):
        """Save risk management data"""
        try:
            # Save drawdown events
            with open(self.drawdowns_file, 'w') as f:
                json.dump([asdict(dd) for dd in self.drawdown_events], f, indent=2)
            
            # Save streak events
            with open(self.streaks_file, 'w') as f:
                json.dump([asdict(se) for se in self.streak_events], f, indent=2)
            
            # Save risk metrics and state
            with open(self.risk_metrics_file, 'w') as f:
                json.dump({
                    'daily_balances': list(self.daily_balances),
                    'trade_sequence': list(self.trade_sequence),
                    'current_balance': self.current_balance,
                    'peak_balance': self.peak_balance,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving risk data: {e}")
    
    def update_trade(self, trade_data: Dict[str, Any]):
        """Update risk metrics with new trade data"""
        try:
            trade_profit = float(trade_data.get('profit_loss_usd', 0))
            trade_timestamp = trade_data.get('timestamp', datetime.now().isoformat())
            trade_size = abs(float(trade_data.get('entry_price', 0)) * float(trade_data.get('quantity', 0)))
            
            # Update balance
            self.current_balance += trade_profit
            
            # Update peak balance
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
            
            # Add to trade sequence
            trade_record = {
                'timestamp': trade_timestamp,
                'profit_loss': trade_profit,
                'balance': self.current_balance,
                'trade_size': trade_size,
                'token_symbol': trade_data.get('token_symbol', 'UNKNOWN')
            }
            self.trade_sequence.append(trade_record)
            
            # Update daily balance (once per day)
            today = datetime.now().date().isoformat()
            if not self.daily_balances or self.daily_balances[-1]['date'] != today:
                self.daily_balances.append({
                    'date': today,
                    'balance': self.current_balance,
                    'trades_count': 1
                })
            else:
                # Update today's balance
                self.daily_balances[-1]['balance'] = self.current_balance
                self.daily_balances[-1]['trades_count'] += 1
            
            # Check for drawdowns
            self._check_drawdown()
            
            # Check for streaks
            self._update_streak(trade_profit > 0, trade_timestamp)
            
            # Save updated data
            self._save_risk_data()
            
        except Exception as e:
            logger.error(f"Error updating trade in risk manager: {e}")
    
    def _check_drawdown(self):
        """Check if we're in a drawdown and track it"""
        current_drawdown_pct = ((self.peak_balance - self.current_balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0
        
        if current_drawdown_pct > 1.0:  # More than 1% drawdown
            # Check if this is a new drawdown or continuation
            if not self.drawdown_events or self.drawdown_events[-1].recovery_date is not None:
                # New drawdown
                drawdown = DrawdownEvent(
                    start_date=datetime.now().date().isoformat(),
                    end_date=datetime.now().date().isoformat(),
                    start_balance=self.peak_balance,
                    peak_balance=self.peak_balance,
                    trough_balance=self.current_balance,
                    drawdown_amount=self.peak_balance - self.current_balance,
                    drawdown_percentage=current_drawdown_pct,
                    recovery_date=None,
                    duration_days=1,
                    trades_during_drawdown=1
                )
                self.drawdown_events.append(drawdown)
            else:
                # Update ongoing drawdown
                ongoing = self.drawdown_events[-1]
                ongoing.end_date = datetime.now().date().isoformat()
                ongoing.trough_balance = min(ongoing.trough_balance, self.current_balance)
                ongoing.drawdown_amount = max(ongoing.drawdown_amount, self.peak_balance - self.current_balance)
                ongoing.drawdown_percentage = max(ongoing.drawdown_percentage, current_drawdown_pct)
                ongoing.trades_during_drawdown += 1
                
                # Calculate duration
                start = datetime.fromisoformat(ongoing.start_date)
                end = datetime.fromisoformat(ongoing.end_date)
                ongoing.duration_days = (end - start).days + 1
        
        elif current_drawdown_pct <= 0.1 and self.drawdown_events and self.drawdown_events[-1].recovery_date is None:
            # Recovery from drawdown (back to within 0.1% of peak)
            self.drawdown_events[-1].recovery_date = datetime.now().date().isoformat()
    
    def _update_streak(self, is_winning_trade: bool, timestamp: str):
        """Update win/loss streak tracking"""
        current_streak_type = 'win' if is_winning_trade else 'loss'
        
        if self.current_streak_type != current_streak_type:
            # Streak broken - save previous streak if it existed
            if self.current_streak_type is not None and self.current_streak_count > 0:
                streak_event = WinStreakEvent(
                    streak_type=self.current_streak_type,
                    start_date=self.current_streak_start,
                    end_date=datetime.fromisoformat(timestamp).date().isoformat(),
                    streak_length=self.current_streak_count,
                    total_profit_loss=sum([t['profit_loss'] for t in list(self.trade_sequence)[-self.current_streak_count:]]),
                    average_trade_size=statistics.mean([abs(t['profit_loss']) for t in list(self.trade_sequence)[-self.current_streak_count:]]),
                    largest_win=max([t['profit_loss'] for t in list(self.trade_sequence)[-self.current_streak_count:] if t['profit_loss'] > 0], default=0),
                    largest_loss=min([t['profit_loss'] for t in list(self.trade_sequence)[-self.current_streak_count:] if t['profit_loss'] < 0], default=0),
                    trades=[t['timestamp'] for t in list(self.trade_sequence)[-self.current_streak_count:]]
                )
                self.streak_events.append(streak_event)
            
            # Start new streak
            self.current_streak_type = current_streak_type
            self.current_streak_count = 1
            self.current_streak_start = datetime.fromisoformat(timestamp).date().isoformat()
        else:
            # Continue current streak
            self.current_streak_count += 1
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Basic balance metrics
            current_drawdown = ((self.peak_balance - self.current_balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0
            
            # Historical drawdown analysis
            max_drawdown_ever = max([dd.drawdown_percentage for dd in self.drawdown_events], default=0)
            
            # 30-day metrics
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_trades = [t for t in self.trade_sequence if datetime.fromisoformat(t['timestamp']) >= thirty_days_ago]
            
            if recent_trades:
                recent_profits = [t['profit_loss'] for t in recent_trades]
                recent_wins = [p for p in recent_profits if p > 0]
                recent_losses = [p for p in recent_profits if p < 0]
                
                win_rate_30d = (len(recent_wins) / len(recent_trades)) * 100
                average_win_30d = statistics.mean(recent_wins) if recent_wins else 0
                average_loss_30d = abs(statistics.mean(recent_losses)) if recent_losses else 0
                profit_factor_30d = sum(recent_wins) / abs(sum(recent_losses)) if recent_losses else float('inf')
                
                # Sharpe ratio (simplified)
                if len(recent_profits) > 1:
                    avg_return = statistics.mean(recent_profits)
                    std_return = statistics.stdev(recent_profits)
                    sharpe_ratio_30d = (avg_return / std_return) if std_return > 0 else 0
                else:
                    sharpe_ratio_30d = 0
                
                # Sortino ratio (downside deviation only)
                if recent_losses:
                    downside_deviation = statistics.stdev(recent_losses)
                    sortino_ratio_30d = (statistics.mean(recent_profits) / downside_deviation) if downside_deviation > 0 else 0
                else:
                    sortino_ratio_30d = float('inf')
                
                # Daily volatility
                daily_returns = []
                if len(self.daily_balances) > 1:
                    for i in range(1, len(self.daily_balances)):
                        prev_balance = self.daily_balances[i-1]['balance']
                        curr_balance = self.daily_balances[i]['balance']
                        if prev_balance > 0:
                            daily_return = (curr_balance - prev_balance) / prev_balance
                            daily_returns.append(daily_return)
                
                daily_volatility = statistics.stdev(daily_returns) * 100 if len(daily_returns) > 1 else 0
                maximum_daily_loss = min(daily_returns) * 100 if daily_returns else 0
                
                # Value at Risk (95th percentile)
                sorted_returns = sorted(daily_returns) if daily_returns else [0]
                var_index = max(0, int(len(sorted_returns) * 0.05) - 1)
                value_at_risk_95 = abs(sorted_returns[var_index] * 100) if sorted_returns else 0
                
            else:
                # Default values when no recent trades
                win_rate_30d = 0
                average_win_30d = 0
                average_loss_30d = 0
                profit_factor_30d = 0
                sharpe_ratio_30d = 0
                sortino_ratio_30d = 0
                daily_volatility = 0
                maximum_daily_loss = 0
                value_at_risk_95 = 0
            
            # Position sizing metrics
            if recent_trades:
                trade_sizes = [t['trade_size'] for t in recent_trades if t['trade_size'] > 0]
                average_position_size = statistics.mean(trade_sizes) if trade_sizes else 0
                largest_position_size = max(trade_sizes) if trade_sizes else 0
                position_size_consistency = statistics.stdev(trade_sizes) if len(trade_sizes) > 1 else 0
            else:
                average_position_size = 0
                largest_position_size = 0
                position_size_consistency = 0
            
            # Streak metrics
            win_streaks = [s.streak_length for s in self.streak_events if s.streak_type == 'win']
            loss_streaks = [s.streak_length for s in self.streak_events if s.streak_type == 'loss']
            
            longest_win_streak = max(win_streaks, default=0)
            longest_loss_streak = max(loss_streaks, default=0)
            
            # 30-day max drawdown
            thirty_day_drawdowns = [dd.drawdown_percentage for dd in self.drawdown_events 
                                   if datetime.fromisoformat(dd.start_date) >= thirty_days_ago]
            max_drawdown_30d = max(thirty_day_drawdowns, default=current_drawdown)
            
            # Calmar ratio (annual return / max drawdown)
            annual_return = (self.current_balance / max(self.peak_balance, 1)) * 100 - 100
            calmar_ratio_30d = annual_return / max(max_drawdown_30d, 1) if max_drawdown_30d > 0 else 0
            
            # Risk assessment and warnings
            risk_level, active_warnings = self._assess_risk_level(current_drawdown, sharpe_ratio_30d, daily_volatility)
            
            return RiskMetrics(
                current_balance=self.current_balance,
                peak_balance=self.peak_balance,
                current_drawdown=current_drawdown,
                max_drawdown_ever=max_drawdown_ever,
                max_drawdown_30d=max_drawdown_30d,
                current_streak_type=self.current_streak_type or 'none',
                current_streak_length=self.current_streak_count,
                longest_win_streak=longest_win_streak,
                longest_loss_streak=longest_loss_streak,
                average_position_size=average_position_size,
                largest_position_size=largest_position_size,
                position_size_consistency=position_size_consistency,
                sharpe_ratio_30d=sharpe_ratio_30d,
                sortino_ratio_30d=sortino_ratio_30d,
                calmar_ratio_30d=calmar_ratio_30d,
                daily_volatility=daily_volatility,
                maximum_daily_loss=maximum_daily_loss,
                value_at_risk_95=value_at_risk_95,
                win_rate_30d=win_rate_30d,
                average_win_30d=average_win_30d,
                average_loss_30d=average_loss_30d,
                profit_factor_30d=profit_factor_30d,
                risk_level=risk_level,
                active_warnings=active_warnings
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            # Return default metrics on error
            return RiskMetrics(
                current_balance=self.current_balance,
                peak_balance=self.peak_balance,
                current_drawdown=0,
                max_drawdown_ever=0,
                max_drawdown_30d=0,
                current_streak_type='none',
                current_streak_length=0,
                longest_win_streak=0,
                longest_loss_streak=0,
                average_position_size=0,
                largest_position_size=0,
                position_size_consistency=0,
                sharpe_ratio_30d=0,
                sortino_ratio_30d=0,
                calmar_ratio_30d=0,
                daily_volatility=0,
                maximum_daily_loss=0,
                value_at_risk_95=0,
                win_rate_30d=0,
                average_win_30d=0,
                average_loss_30d=0,
                profit_factor_30d=0,
                risk_level='UNKNOWN',
                active_warnings=['Error calculating metrics']
            )
    
    def _assess_risk_level(self, current_drawdown: float, sharpe_ratio: float, daily_volatility: float) -> Tuple[str, List[str]]:
        """Assess current risk level and generate warnings"""
        warnings = []
        
        # Drawdown warnings
        if current_drawdown >= self.risk_thresholds['max_drawdown_critical']:
            warnings.append(f"CRITICAL DRAWDOWN: {current_drawdown:.1f}%")
        elif current_drawdown >= self.risk_thresholds['max_drawdown_warning']:
            warnings.append(f"High drawdown: {current_drawdown:.1f}%")
        
        # Sharpe ratio warnings
        if sharpe_ratio < self.risk_thresholds['min_sharpe_ratio']:
            warnings.append(f"Low Sharpe ratio: {sharpe_ratio:.2f}")
        
        # Daily volatility warnings
        if daily_volatility > 15.0:
            warnings.append(f"High volatility: {daily_volatility:.1f}%")
        
        # Determine overall risk level
        if current_drawdown >= self.risk_thresholds['max_drawdown_critical'] or daily_volatility > 20.0:
            risk_level = 'CRITICAL'
        elif current_drawdown >= self.risk_thresholds['max_drawdown_warning'] or daily_volatility > 15.0 or sharpe_ratio < 0.5:
            risk_level = 'HIGH'
        elif current_drawdown >= 5.0 or daily_volatility > 10.0 or sharpe_ratio < 1.0:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return risk_level, warnings
    
    def get_drawdown_history(self) -> List[DrawdownEvent]:
        """Get complete drawdown history"""
        return self.drawdown_events.copy()
    
    def get_streak_history(self) -> List[WinStreakEvent]:
        """Get complete streak history"""
        return self.streak_events.copy()
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        metrics = self.calculate_risk_metrics()
        
        return {
            'risk_level': metrics.risk_level,
            'active_warnings': metrics.active_warnings,
            'current_drawdown': f"{metrics.current_drawdown:.2f}%",
            'max_drawdown_ever': f"{metrics.max_drawdown_ever:.2f}%",
            'current_streak': f"{metrics.current_streak_length} {metrics.current_streak_type}s",
            'sharpe_ratio': f"{metrics.sharpe_ratio_30d:.2f}",
            'win_rate_30d': f"{metrics.win_rate_30d:.1f}%",
            'daily_volatility': f"{metrics.daily_volatility:.2f}%",
            'total_drawdown_events': len(self.drawdown_events),
            'total_streak_events': len(self.streak_events),
            'balance_info': {
                'current': f"${metrics.current_balance:.2f}",
                'peak': f"${metrics.peak_balance:.2f}",
                'from_peak': f"${metrics.peak_balance - metrics.current_balance:.2f}"
            }
        }

# Global risk manager instance
risk_manager = RiskManager()

def update_risk_metrics(trade_data: Dict[str, Any]):
    """Convenience function to update risk metrics"""
    risk_manager.update_trade(trade_data)

def get_current_risk_metrics() -> RiskMetrics:
    """Get current risk metrics"""
    return risk_manager.calculate_risk_metrics()

def get_risk_summary() -> Dict[str, Any]:
    """Get risk summary"""
    return risk_manager.get_risk_summary()

def check_risk_alerts() -> List[str]:
    """Check for active risk alerts"""
    metrics = risk_manager.calculate_risk_metrics()
    return metrics.active_warnings