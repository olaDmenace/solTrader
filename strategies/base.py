"""
Base Strategy Interface - NEW for Day 3 strategy foundation

MIGRATION NOTE: New unified interface for all 4 trading strategies
CRITICAL: Foundation for massive strategy extraction in Days 4-5
Enhanced with Sentry error tracking and Prometheus metrics for professional monitoring

This interface defines the contract that all strategies must implement:
- Momentum Strategy (to be extracted from 3,326-line strategy.py)
- Mean Reversion Strategy 
- Grid Trading Strategy
- Arbitrage Strategy
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# Import our model dependencies
from models.position import Position
from models.trade import Trade, TradeDirection, TradeType
from models.signal import Signal
from models.portfolio import Portfolio

# Sentry integration for professional error tracking
from utils.sentry_config import capture_api_error

# Prometheus metrics for professional monitoring
try:
    from utils.prometheus_metrics import get_metrics
except ImportError:
    # Fallback during development
    def get_metrics():
        return None

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Strategy type enumeration"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    GRID_TRADING = "grid_trading"
    ARBITRAGE = "arbitrage"
    TREND_FOLLOWING = "trend_following"
    SCALPING = "scalping"

class StrategyStatus(Enum):
    """Strategy status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class StrategyConfig:
    """Base strategy configuration"""
    strategy_name: str
    strategy_type: StrategyType
    max_positions: int = 10
    max_position_size: float = 0.1  # 10% of allocated capital
    position_timeout_minutes: int = 180  # 3 hours
    
    # Risk management
    stop_loss_percentage: float = 0.15  # 15% stop loss
    take_profit_percentage: float = 0.5  # 50% take profit
    max_drawdown: float = 0.1  # 10% max strategy drawdown
    
    # Signal filtering
    min_signal_strength: float = 0.6
    min_liquidity_sol: float = 1000.0
    min_volume_24h_sol: float = 500.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type.value,
            "max_positions": self.max_positions,
            "max_position_size": self.max_position_size,
            "position_timeout_minutes": self.position_timeout_minutes,
            "stop_loss_percentage": self.stop_loss_percentage,
            "take_profit_percentage": self.take_profit_percentage,
            "max_drawdown": self.max_drawdown,
            "min_signal_strength": self.min_signal_strength,
            "min_liquidity_sol": self.min_liquidity_sol,
            "min_volume_24h_sol": self.min_volume_24h_sol
        }

@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    # Position metrics
    total_positions: int = 0
    open_positions: int = 0
    winning_positions: int = 0
    losing_positions: int = 0
    
    # Performance metrics
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    win_rate: float = 0.0
    average_position_duration: float = 0.0
    
    # Risk metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Capital allocation
    allocated_capital: float = 0.0
    deployed_capital: float = 0.0
    capital_efficiency: float = 0.0
    
    # Signal metrics
    signals_generated: int = 0
    signals_acted_upon: int = 0
    signal_accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_positions": self.total_positions,
            "open_positions": self.open_positions,
            "winning_positions": self.winning_positions,
            "losing_positions": self.losing_positions,
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "win_rate": self.win_rate,
            "average_position_duration": self.average_position_duration,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "allocated_capital": self.allocated_capital,
            "deployed_capital": self.deployed_capital,
            "capital_efficiency": self.capital_efficiency,
            "signals_generated": self.signals_generated,
            "signals_acted_upon": self.signals_acted_upon,
            "signal_accuracy": self.signal_accuracy
        }

class BaseStrategy(ABC):
    """
    Base class for all trading strategies
    
    This interface defines the contract that all strategies must implement.
    Each strategy should:
    1. Analyze signals and decide whether to trade
    2. Manage positions according to its logic
    3. Report performance metrics
    4. Handle risk management
    """
    
    def __init__(self, config: StrategyConfig, portfolio: Portfolio, settings: Any):
        self.config = config
        self.portfolio = portfolio
        self.settings = settings
        self.status = StrategyStatus.ACTIVE
        self.metrics = StrategyMetrics()
        
        # Strategy state
        self.positions: Dict[str, Position] = {}
        self.signals_cache: Dict[str, Signal] = {}
        self.last_update: datetime = datetime.now()
        
        # Performance tracking
        self.daily_pnl_history: List[Tuple[datetime, float]] = []
        self.position_history: List[Position] = []
        
        logger.info(f"[{self.config.strategy_name.upper()}] Strategy initialized: {self.config.strategy_type.value}")

    @abstractmethod
    async def analyze_signal(self, signal: Signal) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Analyze a signal and decide whether to trade
        
        Args:
            signal: Trading signal to analyze
            
        Returns:
            Tuple of (should_trade, position_size, metadata)
        """
        pass

    @abstractmethod
    async def manage_positions(self) -> List[Tuple[str, str]]:
        """
        Manage existing positions
        
        Returns:
            List of (token_address, action) tuples where action is 'close', 'reduce', 'hold'
        """
        pass

    @abstractmethod
    def calculate_position_size(self, signal: Signal, available_capital: float) -> float:
        """
        Calculate optimal position size for a signal
        
        Args:
            signal: Trading signal
            available_capital: Available capital for this strategy
            
        Returns:
            Position size in SOL
        """
        pass

    @abstractmethod
    def should_exit_position(self, position: Position, current_price: float) -> Tuple[bool, str]:
        """
        Check if position should be exited
        
        Args:
            position: Position to check
            current_price: Current token price
            
        Returns:
            Tuple of (should_exit, reason)
        """
        pass

    # Concrete methods that all strategies can use
    
    async def process_signal(self, signal: Signal) -> Optional[Trade]:
        """Process a trading signal"""
        try:
            # Check strategy status
            if self.status != StrategyStatus.ACTIVE:
                logger.debug(f"[{self.config.strategy_name}] Strategy not active: {self.status.value}")
                return None
            
            # Update signal cache
            self.signals_cache[signal.token_address] = signal
            self.metrics.signals_generated += 1
            
            # Check basic signal validation
            if not self._validate_signal(signal):
                return None
            
            # Check portfolio constraints
            available_capital = self.portfolio.calculate_available_capital(self.config.strategy_name)
            if available_capital <= 0:
                logger.debug(f"[{self.config.strategy_name}] No capital available")
                return None
            
            # Analyze signal (strategy-specific)
            should_trade, position_size, metadata = await self.analyze_signal(signal)
            
            if not should_trade:
                return None
            
            # Validate position size
            position_size = min(position_size, available_capital)
            position_size = min(position_size, available_capital * self.config.max_position_size)
            
            if position_size < self.settings.MIN_POSITION_SIZE_SOL:
                logger.debug(f"[{self.config.strategy_name}] Position size too small: {position_size}")
                return None
            
            # Check portfolio approval
            can_open, reason = self.portfolio.can_open_position(self.config.strategy_name, position_size)
            if not can_open:
                logger.debug(f"[{self.config.strategy_name}] Position rejected: {reason}")
                return None
            
            # Create trade
            trade = Trade(
                trade_id=f"{self.config.strategy_name}_{signal.token_address[:8]}_{int(time.time())}",
                token_address=signal.token_address,
                direction=TradeDirection.BUY,
                trade_type=TradeType.MARKET,
                size=position_size,
                strategy_name=self.config.strategy_name,
                signal_strength=signal.strength,
                signal_type=signal.signal_type
            )
            
            self.metrics.signals_acted_upon += 1
            
            logger.info(f"[{self.config.strategy_name}] Generated trade: {trade.trade_id} "
                       f"({position_size:.2f} SOL @ {signal.price:.8f})")

            # Record signal processing metrics with Prometheus
            try:
                metrics = get_metrics()
                if metrics:
                    metrics.record_strategy_signal_processed(self.config.strategy_name, signal.strength, should_trade)
            except Exception as e:
                logger.debug(f"[METRICS] Strategy signal recording failed (non-critical): {e}")

            return trade

        except Exception as e:
            # Capture error with Sentry for professional error tracking
            capture_api_error(
                error=e,
                api_name="base_strategy",
                endpoint="process_signal",
                context={
                    "strategy_name": self.config.strategy_name,
                    "token_address": signal.token_address if signal else "unknown"
                }
            )
            logger.error(f"[{self.config.strategy_name}] Error processing signal: {e}")
            self.status = StrategyStatus.ERROR
            return None

    def _validate_signal(self, signal: Signal) -> bool:
        """Validate signal meets basic requirements"""
        try:
            # Check signal strength
            if signal.strength < self.config.min_signal_strength:
                logger.debug(f"[{self.config.strategy_name}] Signal too weak: {signal.strength:.3f} < {self.config.min_signal_strength}")
                return False
            
            # Check market data requirements
            market_data = signal.market_data
            liquidity = market_data.get('liquidity', 0)
            volume_24h = market_data.get('volume24h', 0)
            
            if liquidity < self.config.min_liquidity_sol:
                logger.debug(f"[{self.config.strategy_name}] Insufficient liquidity: {liquidity:.1f} < {self.config.min_liquidity_sol}")
                return False
            
            if volume_24h < self.config.min_volume_24h_sol:
                logger.debug(f"[{self.config.strategy_name}] Insufficient volume: {volume_24h:.1f} < {self.config.min_volume_24h_sol}")
                return False
            
            # Check if we already have a position in this token
            if signal.token_address in self.positions:
                logger.debug(f"[{self.config.strategy_name}] Already have position in {signal.token_address[:8]}...")
                return False
            
            # Check max positions limit
            if len(self.positions) >= self.config.max_positions:
                logger.debug(f"[{self.config.strategy_name}] Max positions reached: {len(self.positions)}/{self.config.max_positions}")
                return False
            
            return True

        except Exception as e:
            logger.error(f"[{self.config.strategy_name}] Error validating signal: {e}")
            return False

    def update_metrics(self) -> None:
        """Update strategy performance metrics"""
        try:
            # Count positions
            self.metrics.total_positions = len(self.position_history)
            self.metrics.open_positions = len(self.positions)
            
            # Calculate PnL
            self.metrics.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            # Realized PnL calculation would require trade completion tracking
            self.metrics.total_pnl = self.metrics.realized_pnl + self.metrics.unrealized_pnl
            
            # Calculate win/loss metrics
            winning_positions = [pos for pos in self.positions.values() if pos.unrealized_pnl > 0]
            self.metrics.winning_positions = len(winning_positions)
            self.metrics.losing_positions = self.metrics.open_positions - self.metrics.winning_positions
            
            if self.metrics.total_positions > 0:
                self.metrics.win_rate = self.metrics.winning_positions / self.metrics.total_positions
            
            # Calculate signal accuracy
            if self.metrics.signals_generated > 0:
                self.metrics.signal_accuracy = self.metrics.signals_acted_upon / self.metrics.signals_generated
            
            # Update capital metrics
            strategy_allocation = self.portfolio.strategy_allocations.get(self.config.strategy_name)
            if strategy_allocation:
                self.metrics.allocated_capital = strategy_allocation.total_capital
                self.metrics.deployed_capital = sum(pos.size for pos in self.positions.values())
                
                if self.metrics.allocated_capital > 0:
                    self.metrics.capital_efficiency = self.metrics.deployed_capital / self.metrics.allocated_capital
            
            self.last_update = datetime.now()

            # Record strategy metrics with Prometheus
            try:
                metrics = get_metrics()
                if metrics:
                    metrics.update_strategy_metrics(
                        self.config.strategy_name,
                        self.metrics.total_pnl,
                        self.metrics.open_positions,
                        self.metrics.win_rate
                    )
            except Exception as e:
                logger.debug(f"[METRICS] Strategy metrics recording failed (non-critical): {e}")

        except Exception as e:
            # Capture error with Sentry for professional error tracking
            capture_api_error(
                error=e,
                api_name="base_strategy",
                endpoint="update_metrics",
                context={"strategy_name": self.config.strategy_name}
            )
            logger.error(f"[{self.config.strategy_name}] Error updating metrics: {e}")

    def get_status_summary(self) -> Dict[str, Any]:
        """Get strategy status summary"""
        return {
            "strategy_name": self.config.strategy_name,
            "strategy_type": self.config.strategy_type.value,
            "status": self.status.value,
            "last_update": self.last_update.isoformat(),
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "open_positions": len(self.positions),
            "cached_signals": len(self.signals_cache)
        }

    def pause(self, reason: str = "manual") -> None:
        """Pause strategy execution"""
        self.status = StrategyStatus.PAUSED
        logger.warning(f"[{self.config.strategy_name}] Strategy paused: {reason}")

    def resume(self) -> None:
        """Resume strategy execution"""
        if self.status == StrategyStatus.PAUSED:
            self.status = StrategyStatus.ACTIVE
            logger.info(f"[{self.config.strategy_name}] Strategy resumed")

    def stop(self, reason: str = "manual") -> None:
        """Stop strategy execution"""
        self.status = StrategyStatus.STOPPED
        logger.warning(f"[{self.config.strategy_name}] Strategy stopped: {reason}")

    def add_position(self, position: Position) -> None:
        """Add position to strategy tracking"""
        try:
            position.strategy_name = self.config.strategy_name  # Tag position with strategy
            self.positions[position.token_address] = position
            self.position_history.append(position)
            logger.info(f"[{self.config.strategy_name}] Added position: {position.token_address[:8]}... ({position.size:.2f} SOL)")

        except Exception as e:
            logger.error(f"[{self.config.strategy_name}] Error adding position: {e}")

    def remove_position(self, token_address: str, reason: str = "closed") -> Optional[Position]:
        """Remove position from strategy tracking"""
        try:
            position = self.positions.pop(token_address, None)
            if position:
                logger.info(f"[{self.config.strategy_name}] Removed position: {token_address[:8]}... ({reason})")
            return position

        except Exception as e:
            logger.error(f"[{self.config.strategy_name}] Error removing position: {e}")
            return None