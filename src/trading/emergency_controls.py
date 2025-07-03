"""
Emergency Controls Module for Trading Bot
Advanced circuit breaker and emergency control system
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Callable, Awaitable, Optional
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class EmergencyType(Enum):
    """Types of emergency events"""
    EXCESSIVE_LOSS = "excessive_loss"
    BALANCE_CRITICAL = "balance_critical"
    HIGH_ERROR_RATE = "high_error_rate"
    POSITION_LIMIT = "position_limit"
    RAPID_DRAWDOWN = "rapid_drawdown"
    NETWORK_ISSUES = "network_issues"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    MANUAL_STOP = "manual_stop"

class EmergencyLevel(Enum):
    """Emergency severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class EmergencyEvent:
    """Emergency event data structure"""
    event_type: EmergencyType
    level: EmergencyLevel
    message: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    max_daily_loss: float = 100.0
    max_drawdown_percent: float = 15.0
    max_position_size: float = 500.0
    max_trades_per_hour: int = 20
    max_error_rate: float = 0.3
    min_balance_threshold: float = 10.0
    position_limit: int = 5
    max_volatility_threshold: float = 50.0


class EmergencyControls:
    """Emergency control system for trading operations"""
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize emergency controls
        
        Args:
            config: Circuit breaker configuration        
        """
        self.config = config or CircuitBreakerConfig()
        self.events: List[EmergencyEvent] = []
        self.emergency_stop_active = False
        self.paused_operations: set = set()
        self.callbacks: List[Callable[[EmergencyEvent], Awaitable[None]]] = []
        
        # Load config from environment
        self._load_env_config()
        
        # Monitoring state
        self.daily_stats = {
            'trades': 0,
            'errors': 0,
            'pnl': 0.0,
            'last_reset': datetime.now().date()
        }
        
        self.hourly_stats = {
            'trades': 0,
            'last_reset': datetime.now().replace(minute=0, second=0, microsecond=0)
        }
    
    def _load_env_config(self) -> None:
        """Load configuration from environment variables"""
        try:
            self.config.max_daily_loss = float(os.getenv('EMERGENCY_MAX_DAILY_LOSS', '100.0'))
            self.config.max_drawdown_percent = float(os.getenv('EMERGENCY_MAX_DRAWDOWN', '10.0'))
            self.config.max_position_size = float(os.getenv('EMERGENCY_MAX_POSITION_SIZE', '500.0'))
            self.config.max_trades_per_hour = int(os.getenv('EMERGENCY_MAX_TRADES_HOUR', '20'))
            self.config.max_error_rate = float(os.getenv('EMERGENCY_MAX_ERROR_RATE', '0.3'))
            self.config.min_balance_threshold = float(os.getenv('EMERGENCY_MIN_BALANCE', '10.0'))
            self.config.position_limit = int(os.getenv('EMERGENCY_POSITION_LIMIT', '5'))
            
            logger.info("Emergency controls configuration loaded from environment")
            
        except Exception as e:
            logger.warning(f"Error loading emergency config from environment: {e}")
    
    def add_emergency_callback(
        self, 
        callback: Callable[[EmergencyEvent], Awaitable[None]]
    ) -> None:
        """Add callback for emergency events"""
        self.callbacks.append(callback)
    
    async def check_circuit_breakers(
        self, 
        current_balance: float,
        current_positions: Dict[str, Any],
        daily_pnl: float,
        error_count: int,
        trade_count: int
    ) -> bool:
        """Check all circuit breakers and trigger if necessary"""
        try:
            # Reset daily stats if new day
            self._reset_daily_stats_if_needed()
            self._reset_hourly_stats_if_needed()
            
            # Update current stats
            self.daily_stats['pnl'] = daily_pnl
            self.daily_stats['errors'] = error_count
            self.daily_stats['trades'] = trade_count
            
            triggered_breakers = []
            
            # Check daily loss limit
            if daily_pnl <= -self.config.max_daily_loss:
                triggered_breakers.append(self._create_emergency_event(
                    EmergencyType.EXCESSIVE_LOSS,
                    EmergencyLevel.HIGH,
                    f"Daily loss limit exceeded: {daily_pnl:.2f} USD",
                    {'daily_pnl': daily_pnl, 'limit': -self.config.max_daily_loss}
                ))
            
            # Check balance threshold
            if current_balance < self.config.min_balance_threshold:
                triggered_breakers.append(self._create_emergency_event(
                    EmergencyType.BALANCE_CRITICAL,
                    EmergencyLevel.CRITICAL,
                    f"Balance below critical threshold: {current_balance:.4f} SOL",
                    {'balance': current_balance, 'threshold': self.config.min_balance_threshold}
                ))
            
            # Check position count
            position_count = len(current_positions)
            if position_count >= self.config.position_limit:
                triggered_breakers.append(self._create_emergency_event(
                    EmergencyType.POSITION_LIMIT,
                    EmergencyLevel.MEDIUM,
                    f"Position limit reached: {position_count}/{self.config.position_limit}",
                    {'position_count': position_count, 'limit': self.config.position_limit}
                ))
            
            # Check error rate
            if trade_count > 0:
                error_rate = error_count / trade_count
                if error_rate > self.config.max_error_rate:
                    triggered_breakers.append(self._create_emergency_event(
                        EmergencyType.HIGH_ERROR_RATE,
                        EmergencyLevel.HIGH,
                        f"Error rate too high: {error_rate:.1%}",
                        {'error_rate': error_rate, 'limit': self.config.max_error_rate}
                    ))
            
            # Check hourly trade limit
            if self.hourly_stats['trades'] >= self.config.max_trades_per_hour:
                triggered_breakers.append(self._create_emergency_event(
                    EmergencyType.RAPID_DRAWDOWN,
                    EmergencyLevel.MEDIUM,
                    f"Hourly trade limit exceeded: {self.hourly_stats['trades']}",
                    {'hourly_trades': self.hourly_stats['trades'], 'limit': self.config.max_trades_per_hour}
                ))
            
            # Process triggered breakers
            if triggered_breakers:
                for event in triggered_breakers:
                    await self._handle_emergency_event(event)
                return False  # Circuit breakers triggered
                
            return True  # All checks passed
            
        except Exception as e:
            logger.error(f"Error checking circuit breakers: {e}")
            return False  # Fail safe
    
    async def emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop"""
        try:
            self.emergency_stop_active = True
            
            event = self._create_emergency_event(
                EmergencyType.NETWORK_ISSUES,
                EmergencyLevel.CRITICAL,
                f"Emergency stop activated: {reason}",
                {'reason': reason}
            )
            
            await self._handle_emergency_event(event)
            
            logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
            
        except Exception as e:
            logger.error(f"Error in emergency stop: {e}")
    
    def _create_emergency_event(
        self,
        event_type: EmergencyType,
        level: EmergencyLevel,
        message: str,
        data: Dict[str, Any]
    ) -> EmergencyEvent:
        """Create emergency event"""
        return EmergencyEvent(
            event_type=event_type,
            level=level,
            message=message,
            data=data
        )
    
    async def _handle_emergency_event(self, event: EmergencyEvent) -> None:
        """Handle emergency event"""
        try:
            # Add to events list
            self.events.append(event)
            
            # Keep only recent events (last 100)
            if len(self.events) > 100:
                self.events = self.events[-100:]
                
            # Log event
            log_level = {
                EmergencyLevel.LOW: logger.info,
                EmergencyLevel.MEDIUM: logger.warning,
                EmergencyLevel.HIGH: logger.error,
                EmergencyLevel.CRITICAL: logger.critical
            }.get(event.level, logger.warning)
            
            log_level(f"Emergency event: {event.message}")
            
            # Call registered callbacks
            for callback in self.callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Error in emergency callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling emergency event: {e}")
    
    def _reset_daily_stats_if_needed(self) -> None:
        """Reset daily stats if new day"""
        today = datetime.now().date()
        if today > self.daily_stats['last_reset']:
            self.daily_stats = {
                'trades': 0,
                'errors': 0,
                'pnl': 0.0,
                'last_reset': today
            }
            logger.info("Daily stats reset for new day")
    
    def _reset_hourly_stats_if_needed(self) -> None:
        """Reset hourly stats if new hour"""
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        if current_hour > self.hourly_stats['last_reset']:
            self.hourly_stats = {
                'trades': 0,
                'last_reset': current_hour
            }
    
    def record_trade(self) -> None:
        """Record a trade for rate limiting"""
        self.hourly_stats['trades'] += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get emergency controls status"""
        return {
            'emergency_stop_active': self.emergency_stop_active,
            'paused_operations': list(self.paused_operations),
            'recent_events': [
                {
                    'type': event.event_type.value,
                    'level': event.level.value,
                    'message': event.message,
                    'timestamp': event.timestamp.isoformat(),
                    'resolved': event.resolved
                }
                for event in self.events[-10:]  # Last 10 events
            ],
            'config': {
                'max_daily_loss': self.config.max_daily_loss,
                'max_drawdown_percent': self.config.max_drawdown_percent,
                'max_position_size': self.config.max_position_size,
                'position_limit': self.config.position_limit
            }
        }
    
    def clear_emergency_stop(self) -> None:
        """Clear emergency stop (manual intervention required)"""
        self.emergency_stop_active = False
        self.paused_operations.clear()
        logger.warning("Emergency stop cleared manually")