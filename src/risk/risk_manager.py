"""
Risk Management System

Provides comprehensive risk management for trading operations:
- Position sizing and limits
- Portfolio risk monitoring
- Drawdown protection
- Volatility controls
- Emergency stop mechanisms
- Risk alerting system
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications with comparable values"""
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

class RiskEvent(Enum):
    """Risk event types"""
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    DRAWDOWN_LIMIT_EXCEEDED = "drawdown_limit_exceeded"
    VOLATILITY_SPIKE = "volatility_spike"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    PORTFOLIO_CONCENTRATION = "portfolio_concentration"
    EMERGENCY_STOP = "emergency_stop"
    BALANCE_TOO_LOW = "balance_too_low"
    TRADE_FREQUENCY_HIGH = "trade_frequency_high"

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    event_type: RiskEvent
    level: RiskLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

@dataclass
class PositionRisk:
    """Position-specific risk metrics"""
    token_address: str
    position_size: Decimal
    position_value_usd: Optional[Decimal] = None
    portfolio_percentage: Optional[Decimal] = None
    volatility_score: Optional[Decimal] = None
    concentration_risk: RiskLevel = RiskLevel.LOW

@dataclass
class PortfolioRisk:
    """Portfolio-wide risk metrics"""
    total_value_usd: Decimal
    daily_pnl: Decimal
    max_drawdown: Decimal
    current_drawdown: Decimal
    volatility: Decimal
    position_count: int
    concentration_score: Decimal
    overall_risk_level: RiskLevel = RiskLevel.LOW

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self):
        # Load configuration from environment
        self._load_config()
        
        # Risk state tracking
        self.active_alerts: List[RiskAlert] = []
        self.alert_history: List[RiskAlert] = []
        self.emergency_stop_active = False
        self.last_emergency_check = time.time()
        
        # Performance tracking
        self.daily_pnl = Decimal('0')
        self.daily_trade_count = 0
        self.current_positions: Dict[str, PositionRisk] = {}
        self.balance_history: List[Tuple[datetime, Decimal]] = []
        self.portfolio_metrics: Optional[PortfolioRisk] = None
        
        # Rate limiting
        self.trade_timestamps: List[datetime] = []
        self.daily_reset_time = None
        
        logger.info("[RISK] Risk management system initialized")
    
    def _load_config(self):
        """Load risk management configuration from environment"""
        # Position limits
        self.max_position_size = Decimal(os.getenv('MAX_LIVE_POSITION_SIZE', '0.5'))
        self.max_positions = int(os.getenv('MAX_LIVE_POSITIONS', '3'))
        self.max_portfolio_risk = Decimal(os.getenv('MAX_PORTFOLIO_RISK', '5.0')) / 100
        
        # Loss limits
        self.max_daily_loss = Decimal(os.getenv('MAX_DAILY_LOSS', '3.0'))
        self.max_drawdown = Decimal(os.getenv('MAX_DRAWDOWN', '5.0')) / 100
        self.emergency_stop_loss = Decimal(os.getenv('EMERGENCY_STOP_LOSS', '0.2'))
        
        # Balance limits
        self.min_balance = Decimal(os.getenv('EMERGENCY_MIN_BALANCE', '0.1'))
        self.trading_min_balance = Decimal(os.getenv('TRADING_MIN_BALANCE', '0.005'))
        
        # Trading frequency
        self.max_trades_per_day = int(os.getenv('MAX_TRADES_PER_DAY', '5'))
        self.max_trades_per_hour = int(os.getenv('EMERGENCY_MAX_TRADES_HOUR', '20'))
        
        # Volatility limits
        self.max_volatility = Decimal(os.getenv('MAX_VOLATILITY', '0.3'))
        self.emergency_max_volatility = Decimal(os.getenv('EMERGENCY_MAX_VOLATILITY', '50.0')) / 100
        
        # Emergency controls
        self.emergency_max_daily_loss = Decimal(os.getenv('EMERGENCY_MAX_DAILY_LOSS', '4.0'))
        self.emergency_max_drawdown = Decimal(os.getenv('EMERGENCY_MAX_DRAWDOWN', '10.0')) / 100
        self.emergency_max_error_rate = Decimal(os.getenv('EMERGENCY_MAX_ERROR_RATE', '0.3'))
        
        logger.info(f"[RISK] Configuration loaded - Max Position: {self.max_position_size} SOL, Max Daily Loss: ${self.max_daily_loss}")
    
    def validate_trade(self, token_address: str, trade_size: Decimal, current_balance: Decimal, 
                      trade_type: str = "buy") -> Tuple[bool, str, RiskLevel]:
        """
        Comprehensive trade validation
        
        Args:
            token_address: Token being traded
            trade_size: Size of the trade in SOL
            current_balance: Current SOL balance
            trade_type: "buy" or "sell"
            
        Returns:
            (is_allowed, reason, risk_level)
        """
        try:
            # Emergency stop check
            if self.emergency_stop_active:
                return False, "Emergency stop is active", RiskLevel.EMERGENCY
            
            # Balance checks
            if current_balance < self.min_balance:
                self._trigger_emergency_stop("Balance below emergency minimum")
                return False, f"Balance {current_balance} below emergency minimum {self.min_balance}", RiskLevel.EMERGENCY
            
            if current_balance - trade_size < self.trading_min_balance:
                return False, f"Trade would leave balance below minimum trading threshold", RiskLevel.HIGH
            
            # Position size checks
            if trade_size > self.max_position_size:
                return False, f"Trade size {trade_size} exceeds maximum {self.max_position_size}", RiskLevel.HIGH
            
            # Position count check
            if trade_type == "buy" and len(self.current_positions) >= self.max_positions:
                return False, f"Maximum positions ({self.max_positions}) already reached", RiskLevel.MODERATE
            
            # Daily loss check
            if self.daily_pnl < -self.max_daily_loss:
                return False, f"Daily loss limit reached: ${abs(self.daily_pnl)}", RiskLevel.HIGH
            
            # Trade frequency checks
            if not self._check_trade_frequency():
                return False, "Trade frequency limits exceeded", RiskLevel.MODERATE
            
            # Portfolio concentration check
            if trade_type == "buy":
                concentration_risk = self._check_concentration_risk(token_address, trade_size, current_balance)
                if concentration_risk == RiskLevel.HIGH:
                    return False, "Position would create excessive portfolio concentration", RiskLevel.HIGH
            
            # Volatility check
            volatility_risk = self._check_volatility_risk()
            if volatility_risk >= RiskLevel.HIGH:
                return False, "Portfolio volatility too high for new positions", volatility_risk
            
            # All checks passed
            risk_level = self._calculate_trade_risk_level(trade_size, current_balance)
            return True, "Trade approved", risk_level
            
        except Exception as e:
            logger.error(f"[RISK] Error in trade validation: {e}")
            return False, f"Risk validation error: {str(e)}", RiskLevel.CRITICAL
    
    def update_position(self, token_address: str, position_size: Decimal, 
                       position_value_usd: Optional[Decimal] = None):
        """Update position tracking"""
        try:
            if position_size <= 0:
                # Remove position
                if token_address in self.current_positions:
                    del self.current_positions[token_address]
                    logger.debug(f"[RISK] Removed position: {token_address[:8]}...")
            else:
                # Update or create position
                portfolio_pct = None
                if position_value_usd and self.portfolio_metrics:
                    portfolio_pct = position_value_usd / self.portfolio_metrics.total_value_usd * 100
                
                self.current_positions[token_address] = PositionRisk(
                    token_address=token_address,
                    position_size=position_size,
                    position_value_usd=position_value_usd,
                    portfolio_percentage=portfolio_pct
                )
                logger.debug(f"[RISK] Updated position: {token_address[:8]}... = {position_size} SOL")
            
            # Update portfolio risk metrics
            self._update_portfolio_risk()
            
        except Exception as e:
            logger.error(f"[RISK] Error updating position: {e}")
    
    def update_daily_pnl(self, pnl_change: Decimal):
        """Update daily P&L tracking"""
        try:
            self.daily_pnl += pnl_change
            logger.debug(f"[RISK] Daily P&L updated: ${self.daily_pnl}")
            
            # Check emergency loss limits
            if self.daily_pnl < -self.emergency_max_daily_loss:
                self._trigger_emergency_stop(f"Emergency daily loss limit exceeded: ${abs(self.daily_pnl)}")
            elif self.daily_pnl < -self.max_daily_loss:
                self._create_alert(
                    RiskEvent.DAILY_LOSS_LIMIT,
                    RiskLevel.HIGH,
                    f"Daily loss limit exceeded: ${abs(self.daily_pnl)}",
                    {"daily_pnl": float(self.daily_pnl), "limit": float(self.max_daily_loss)}
                )
            
        except Exception as e:
            logger.error(f"[RISK] Error updating daily P&L: {e}")
    
    def update_balance(self, new_balance: Decimal):
        """Update balance tracking and drawdown calculations"""
        try:
            current_time = datetime.now()
            self.balance_history.append((current_time, new_balance))
            
            # Keep only last 24 hours of balance history
            cutoff_time = current_time - timedelta(hours=24)
            self.balance_history = [
                (ts, balance) for ts, balance in self.balance_history 
                if ts > cutoff_time
            ]
            
            # Calculate drawdown
            if self.balance_history:
                max_balance = max(balance for _, balance in self.balance_history)
                current_drawdown = (max_balance - new_balance) / max_balance if max_balance > 0 else Decimal('0')
                
                # Check drawdown limits
                if current_drawdown > self.emergency_max_drawdown:
                    self._trigger_emergency_stop(f"Emergency drawdown limit exceeded: {current_drawdown:.2%}")
                elif current_drawdown > self.max_drawdown:
                    self._create_alert(
                        RiskEvent.DRAWDOWN_LIMIT_EXCEEDED,
                        RiskLevel.HIGH,
                        f"Drawdown limit exceeded: {current_drawdown:.2%}",
                        {"current_drawdown": float(current_drawdown), "limit": float(self.max_drawdown)}
                    )
            
        except Exception as e:
            logger.error(f"[RISK] Error updating balance: {e}")
    
    def record_trade(self, success: bool):
        """Record trade execution for frequency and error rate tracking"""
        try:
            current_time = datetime.now()
            self.trade_timestamps.append(current_time)
            self.daily_trade_count += 1
            
            # Reset daily counter at midnight
            if self.daily_reset_time is None or current_time.date() > self.daily_reset_time:
                self.daily_trade_count = 1
                self.daily_reset_time = current_time.date()
            
            # Clean old timestamps (keep last hour)
            cutoff_time = current_time - timedelta(hours=1)
            self.trade_timestamps = [ts for ts in self.trade_timestamps if ts > cutoff_time]
            
            logger.debug(f"[RISK] Trade recorded: {'success' if success else 'failure'}, daily count: {self.daily_trade_count}")
            
        except Exception as e:
            logger.error(f"[RISK] Error recording trade: {e}")
    
    def _check_trade_frequency(self) -> bool:
        """Check if trade frequency is within limits"""
        try:
            # Daily limit check
            if self.daily_trade_count >= self.max_trades_per_day:
                self._create_alert(
                    RiskEvent.TRADE_FREQUENCY_HIGH,
                    RiskLevel.MODERATE,
                    f"Daily trade limit reached: {self.daily_trade_count}/{self.max_trades_per_day}",
                    {"daily_trades": self.daily_trade_count, "daily_limit": self.max_trades_per_day}
                )
                return False
            
            # Hourly limit check
            hourly_trades = len(self.trade_timestamps)
            if hourly_trades >= self.max_trades_per_hour:
                self._create_alert(
                    RiskEvent.TRADE_FREQUENCY_HIGH,
                    RiskLevel.HIGH,
                    f"Hourly trade limit reached: {hourly_trades}/{self.max_trades_per_hour}",
                    {"hourly_trades": hourly_trades, "hourly_limit": self.max_trades_per_hour}
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"[RISK] Error checking trade frequency: {e}")
            return False
    
    def _check_concentration_risk(self, token_address: str, trade_size: Decimal, 
                                 current_balance: Decimal) -> RiskLevel:
        """Check portfolio concentration risk"""
        try:
            # Calculate what portfolio percentage this trade would represent
            total_portfolio_value = current_balance  # Simplified - in production, include all positions
            trade_percentage = trade_size / total_portfolio_value
            
            if trade_percentage > 0.5:  # 50% concentration
                return RiskLevel.HIGH
            elif trade_percentage > 0.3:  # 30% concentration
                return RiskLevel.MODERATE
            elif trade_percentage > 0.1:  # 10% concentration
                return RiskLevel.LOW
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"[RISK] Error checking concentration risk: {e}")
            return RiskLevel.MODERATE
    
    def _check_volatility_risk(self) -> RiskLevel:
        """Check portfolio volatility risk"""
        try:
            # Simplified volatility check based on balance history
            if len(self.balance_history) < 2:
                return RiskLevel.LOW
            
            # Calculate volatility from recent balance changes
            recent_balances = [balance for _, balance in self.balance_history[-10:]]
            if len(recent_balances) < 2:
                return RiskLevel.LOW
            
            # Simple volatility calculation (standard deviation of returns)
            returns = []
            for i in range(1, len(recent_balances)):
                if recent_balances[i-1] > 0:
                    returns.append(float(recent_balances[i] / recent_balances[i-1] - 1))
            
            if not returns:
                return RiskLevel.LOW
            
            import statistics
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0
            
            if volatility > float(self.emergency_max_volatility):
                return RiskLevel.EMERGENCY
            elif volatility > float(self.max_volatility):
                return RiskLevel.HIGH
            elif volatility > float(self.max_volatility) / 2:
                return RiskLevel.MODERATE
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"[RISK] Error checking volatility risk: {e}")
            return RiskLevel.MODERATE
    
    def _calculate_trade_risk_level(self, trade_size: Decimal, current_balance: Decimal) -> RiskLevel:
        """Calculate overall risk level for a trade"""
        try:
            # Base risk on trade size relative to balance
            trade_percentage = trade_size / current_balance if current_balance > 0 else 1
            
            if trade_percentage > 0.1:  # >10% of balance
                return RiskLevel.HIGH
            elif trade_percentage > 0.05:  # >5% of balance
                return RiskLevel.MODERATE
            else:
                return RiskLevel.LOW
                
        except Exception:
            return RiskLevel.MODERATE
    
    def _update_portfolio_risk(self):
        """Update overall portfolio risk metrics"""
        try:
            if not self.current_positions:
                self.portfolio_metrics = None
                return
            
            total_positions = len(self.current_positions)
            total_value = sum(
                pos.position_value_usd for pos in self.current_positions.values() 
                if pos.position_value_usd
            ) or Decimal('0')
            
            # Calculate concentration score
            concentration_scores = [
                pos.portfolio_percentage for pos in self.current_positions.values()
                if pos.portfolio_percentage
            ]
            max_concentration = max(concentration_scores) if concentration_scores else Decimal('0')
            
            # Determine overall risk level
            overall_risk = RiskLevel.LOW
            if total_positions > self.max_positions * 0.8:
                overall_risk = RiskLevel.MODERATE
            if max_concentration > 50:  # 50% concentration in one position
                overall_risk = RiskLevel.HIGH
            
            self.portfolio_metrics = PortfolioRisk(
                total_value_usd=total_value,
                daily_pnl=self.daily_pnl,
                max_drawdown=self._calculate_max_drawdown(),
                current_drawdown=self._calculate_current_drawdown(),
                volatility=Decimal('0.0'),  # Simplified volatility for portfolio metrics
                position_count=total_positions,
                concentration_score=max_concentration,
                overall_risk_level=overall_risk
            )
            
        except Exception as e:
            logger.error(f"[RISK] Error updating portfolio risk: {e}")
    
    def _calculate_max_drawdown(self) -> Decimal:
        """Calculate maximum drawdown from balance history"""
        try:
            if len(self.balance_history) < 2:
                return Decimal('0')
            
            balances = [balance for _, balance in self.balance_history]
            max_drawdown = Decimal('0')
            peak = balances[0]
            
            for balance in balances:
                if balance > peak:
                    peak = balance
                drawdown = (peak - balance) / peak if peak > 0 else Decimal('0')
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return max_drawdown
            
        except Exception:
            return Decimal('0')
    
    def _calculate_current_drawdown(self) -> Decimal:
        """Calculate current drawdown from recent peak"""
        try:
            if len(self.balance_history) < 2:
                return Decimal('0')
            
            recent_balances = [balance for _, balance in self.balance_history[-24:]]  # Last 24 data points
            if not recent_balances:
                return Decimal('0')
            
            peak = max(recent_balances)
            current = recent_balances[-1]
            
            return (peak - current) / peak if peak > 0 else Decimal('0')
            
        except Exception:
            return Decimal('0')
    
    def _create_alert(self, event_type: RiskEvent, level: RiskLevel, message: str, data: Dict[str, Any]):
        """Create and store risk alert"""
        try:
            alert = RiskAlert(
                event_type=event_type,
                level=level,
                message=message,
                data=data
            )
            
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            
            # Keep alert history manageable
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-500:]
            
            logger.warning(f"[RISK ALERT] {level.value.upper()}: {message}")
            
        except Exception as e:
            logger.error(f"[RISK] Error creating alert: {e}")
    
    def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop mechanism"""
        try:
            if not self.emergency_stop_active:
                self.emergency_stop_active = True
                self.last_emergency_check = time.time()
                
                self._create_alert(
                    RiskEvent.EMERGENCY_STOP,
                    RiskLevel.EMERGENCY,
                    f"EMERGENCY STOP TRIGGERED: {reason}",
                    {"reason": reason, "timestamp": datetime.now().isoformat()}
                )
                
                logger.critical(f"[RISK] 🚨 EMERGENCY STOP TRIGGERED: {reason}")
            
        except Exception as e:
            logger.error(f"[RISK] Error triggering emergency stop: {e}")
    
    def clear_emergency_stop(self, manual_override: bool = False) -> bool:
        """Clear emergency stop (manual override required)"""
        try:
            if not manual_override:
                return False
            
            self.emergency_stop_active = False
            self.last_emergency_check = time.time()
            
            # Resolve emergency alerts
            for alert in self.active_alerts:
                if alert.event_type == RiskEvent.EMERGENCY_STOP:
                    alert.resolved = True
            
            logger.warning("[RISK] Emergency stop cleared via manual override")
            return True
            
        except Exception as e:
            logger.error(f"[RISK] Error clearing emergency stop: {e}")
            return False
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            active_alerts_by_level = {}
            for level in RiskLevel:
                active_alerts_by_level[level.value] = len([
                    alert for alert in self.active_alerts 
                    if alert.level == level and not alert.resolved
                ])
            
            return {
                "emergency_stop_active": self.emergency_stop_active,
                "daily_pnl": float(self.daily_pnl),
                "daily_trade_count": self.daily_trade_count,
                "active_positions": len(self.current_positions),
                "active_alerts": len([a for a in self.active_alerts if not a.resolved]),
                "alerts_by_level": active_alerts_by_level,
                "portfolio_metrics": {
                    "total_positions": len(self.current_positions),
                    "max_drawdown": float(self._calculate_max_drawdown()),
                    "current_drawdown": float(self._calculate_current_drawdown()),
                    "overall_risk_level": self.portfolio_metrics.overall_risk_level.value if self.portfolio_metrics else "unknown"
                },
                "limits": {
                    "max_position_size": float(self.max_position_size),
                    "max_positions": self.max_positions,
                    "max_daily_loss": float(self.max_daily_loss),
                    "max_drawdown": float(self.max_drawdown),
                    "min_balance": float(self.min_balance)
                }
            }
            
        except Exception as e:
            logger.error(f"[RISK] Error getting risk summary: {e}")
            return {"error": str(e)}
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active risk alerts"""
        try:
            active = [alert for alert in self.active_alerts if not alert.resolved]
            return [
                {
                    "event_type": alert.event_type.value,
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "data": alert.data
                }
                for alert in active
            ]
            
        except Exception as e:
            logger.error(f"[RISK] Error getting active alerts: {e}")
            return []

# Global risk manager instance
_global_risk_manager: Optional[RiskManager] = None

def get_risk_manager() -> RiskManager:
    """Get global risk manager instance"""
    global _global_risk_manager
    if _global_risk_manager is None:
        _global_risk_manager = RiskManager()
    return _global_risk_manager