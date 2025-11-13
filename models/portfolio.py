"""
Portfolio Models - NEW for Day 3 data model consolidation

MIGRATION NOTE: New comprehensive portfolio management models for multi-strategy coordination
CRITICAL: Foundation for strategy extraction and capital allocation
Enhanced with Sentry error tracking and Prometheus metrics for professional monitoring
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# Import our model dependencies
from models.position import Position, PositionManager
from models.trade import Trade, TradeBook
from models.signal import Signal

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

class PortfolioStatus(Enum):
    """Portfolio status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    LIQUIDATING = "liquidating"

class AllocationStrategy(Enum):
    """Capital allocation strategy enumeration"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MOMENTUM_BASED = "momentum_based"
    PERFORMANCE_BASED = "performance_based"
    DYNAMIC = "dynamic"

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percentage: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Position metrics
    total_positions: int = 0
    winning_positions: int = 0
    losing_positions: int = 0
    win_rate: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    successful_trades: int = 0
    trade_success_rate: float = 0.0
    average_trade_duration: float = 0.0
    
    # Strategy metrics
    strategy_allocations: Dict[str, float] = field(default_factory=dict)
    strategy_performance: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_value": self.total_value,
            "total_pnl": self.total_pnl,
            "total_pnl_percentage": self.total_pnl_percentage,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "total_positions": self.total_positions,
            "winning_positions": self.winning_positions,
            "losing_positions": self.losing_positions,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "successful_trades": self.successful_trades,
            "trade_success_rate": self.trade_success_rate,
            "average_trade_duration": self.average_trade_duration,
            "strategy_allocations": self.strategy_allocations,
            "strategy_performance": self.strategy_performance
        }

@dataclass
class StrategyAllocation:
    """Strategy allocation configuration"""
    strategy_name: str
    target_allocation: float  # Percentage of total capital (0.0 to 1.0)
    current_allocation: float = 0.0
    max_allocation: float = 0.5  # Maximum allowed allocation
    min_allocation: float = 0.0  # Minimum required allocation
    
    # Performance tracking
    total_capital: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Risk limits
    max_positions: int = 10
    max_position_size: float = 0.1  # Max 10% per position
    
    @property
    def current_pnl_percentage(self) -> float:
        """Get current PnL percentage"""
        if self.total_capital <= 0:
            return 0.0
        return (self.unrealized_pnl + self.realized_pnl) / self.total_capital

    @property
    def allocation_deviation(self) -> float:
        """Get deviation from target allocation"""
        return abs(self.current_allocation - self.target_allocation)

    def to_dict(self) -> Dict[str, Any]:
        """Convert allocation to dictionary"""
        return {
            "strategy_name": self.strategy_name,
            "target_allocation": self.target_allocation,
            "current_allocation": self.current_allocation,
            "max_allocation": self.max_allocation,
            "min_allocation": self.min_allocation,
            "total_capital": self.total_capital,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "current_pnl_percentage": self.current_pnl_percentage,
            "allocation_deviation": self.allocation_deviation,
            "max_positions": self.max_positions,
            "max_position_size": self.max_position_size
        }

@dataclass
class Portfolio:
    """Comprehensive portfolio management"""
    portfolio_id: str
    initial_capital: float
    
    # Core components
    position_manager: PositionManager
    trade_book: TradeBook = field(default_factory=TradeBook)
    
    # Portfolio configuration
    status: PortfolioStatus = PortfolioStatus.ACTIVE
    allocation_strategy: AllocationStrategy = AllocationStrategy.DYNAMIC
    
    # Strategy allocations
    strategy_allocations: Dict[str, StrategyAllocation] = field(default_factory=dict)
    
    # Risk management
    max_portfolio_drawdown: float = 0.2  # 20% max drawdown
    emergency_stop_loss: float = 0.3  # 30% emergency stop
    max_positions: int = 50
    max_position_concentration: float = 0.1  # 10% max per position
    
    # Performance tracking
    metrics: PortfolioMetrics = field(default_factory=PortfolioMetrics)
    daily_values: List[Tuple[datetime, float]] = field(default_factory=list)
    high_water_mark: float = field(init=False)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize portfolio after creation"""
        self.high_water_mark = self.initial_capital
        self.metrics.total_value = self.initial_capital

    def add_strategy_allocation(self, allocation: StrategyAllocation) -> None:
        """Add strategy allocation"""
        try:
            self.strategy_allocations[allocation.strategy_name] = allocation
            logger.info(f"[PORTFOLIO] Added strategy allocation: {allocation.strategy_name} ({allocation.target_allocation:.1%})")

            # Record strategy allocation metrics with Prometheus
            try:
                metrics = get_metrics()
                if metrics:
                    metrics.record_strategy_allocation(allocation.strategy_name, allocation.target_allocation)
            except Exception as e:
                logger.debug(f"[METRICS] Strategy allocation recording failed (non-critical): {e}")

        except Exception as e:
            # Capture error with Sentry for professional error tracking
            capture_api_error(
                error=e,
                api_name="portfolio",
                endpoint="add_strategy_allocation",
                context={"strategy_name": allocation.strategy_name}
            )
            logger.error(f"[PORTFOLIO] Error adding strategy allocation: {e}")

    def update_strategy_allocation(self, strategy_name: str, new_target: float) -> bool:
        """Update strategy target allocation"""
        try:
            if strategy_name not in self.strategy_allocations:
                logger.error(f"[PORTFOLIO] Strategy {strategy_name} not found")
                return False
            
            old_target = self.strategy_allocations[strategy_name].target_allocation
            self.strategy_allocations[strategy_name].target_allocation = new_target
            
            logger.info(f"[PORTFOLIO] Updated {strategy_name} allocation: {old_target:.1%} -> {new_target:.1%}")
            return True

        except Exception as e:
            # Capture error with Sentry for professional error tracking
            capture_api_error(
                error=e,
                api_name="portfolio",
                endpoint="update_strategy_allocation",
                context={"strategy_name": strategy_name, "new_target": new_target}
            )
            logger.error(f"[PORTFOLIO] Error updating strategy allocation: {e}")
            return False

    def calculate_available_capital(self, strategy_name: str) -> float:
        """Calculate available capital for a strategy"""
        try:
            if strategy_name not in self.strategy_allocations:
                return 0.0
            
            allocation = self.strategy_allocations[strategy_name]
            total_portfolio_value = self.get_total_value()
            
            # Calculate target capital for this strategy
            target_capital = total_portfolio_value * allocation.target_allocation
            
            # Calculate currently allocated capital
            strategy_positions = [pos for pos in self.position_manager.get_open_positions().values() 
                                if hasattr(pos, 'strategy_name') and pos.strategy_name == strategy_name]
            
            current_capital = sum(pos.size for pos in strategy_positions)
            
            # Available capital is target minus current
            available = max(0.0, target_capital - current_capital)
            
            logger.debug(f"[PORTFOLIO] Available capital for {strategy_name}: {available:.2f} SOL")
            return available

        except Exception as e:
            # Capture error with Sentry for professional error tracking
            capture_api_error(
                error=e,
                api_name="portfolio",
                endpoint="calculate_available_capital",
                context={"strategy_name": strategy_name}
            )
            logger.error(f"[PORTFOLIO] Error calculating available capital for {strategy_name}: {e}")
            return 0.0

    def can_open_position(self, strategy_name: str, position_size: float) -> Tuple[bool, str]:
        """Check if position can be opened"""
        try:
            # Check portfolio status
            if self.status != PortfolioStatus.ACTIVE:
                return False, f"Portfolio not active: {self.status.value}"
            
            # Check strategy exists
            if strategy_name not in self.strategy_allocations:
                return False, f"Strategy {strategy_name} not configured"
            
            allocation = self.strategy_allocations[strategy_name]
            
            # Check available capital
            available_capital = self.calculate_available_capital(strategy_name)
            if position_size > available_capital:
                return False, f"Insufficient capital: need {position_size}, have {available_capital}"
            
            # Check position count limits
            current_positions = len([pos for pos in self.position_manager.get_open_positions().values() 
                                   if hasattr(pos, 'strategy_name') and pos.strategy_name == strategy_name])
            
            if current_positions >= allocation.max_positions:
                return False, f"Max positions reached: {current_positions}/{allocation.max_positions}"
            
            # Check position concentration
            portfolio_value = self.get_total_value()
            position_concentration = position_size / portfolio_value if portfolio_value > 0 else 1.0
            
            if position_concentration > self.max_position_concentration:
                return False, f"Position too large: {position_concentration:.1%} > {self.max_position_concentration:.1%}"
            
            return True, "Position approved"

        except Exception as e:
            # Capture error with Sentry for professional error tracking
            capture_api_error(
                error=e,
                api_name="portfolio",
                endpoint="can_open_position",
                context={"strategy_name": strategy_name, "position_size": position_size}
            )
            logger.error(f"[PORTFOLIO] Error checking position approval: {e}")
            return False, "Error checking position"

    def update_metrics(self) -> None:
        """Update portfolio metrics"""
        try:
            # Get current positions and trades
            open_positions = self.position_manager.get_open_positions()
            completed_trades = self.trade_book.get_completed_trades()
            
            # Calculate basic metrics
            self.metrics.total_positions = len(open_positions)
            self.metrics.total_trades = len(self.trade_book.trades)
            
            # Calculate PnL
            self.metrics.unrealized_pnl = sum(pos.unrealized_pnl for pos in open_positions.values())
            self.metrics.realized_pnl = sum(trade.realized_pnl for trade in completed_trades)
            self.metrics.total_pnl = self.metrics.realized_pnl + self.metrics.unrealized_pnl
            
            # Calculate total value
            self.metrics.total_value = self.initial_capital + self.metrics.total_pnl
            
            if self.initial_capital > 0:
                self.metrics.total_pnl_percentage = self.metrics.total_pnl / self.initial_capital
            
            # Update high water mark and drawdown
            if self.metrics.total_value > self.high_water_mark:
                self.high_water_mark = self.metrics.total_value
            
            if self.high_water_mark > 0:
                self.metrics.current_drawdown = (self.high_water_mark - self.metrics.total_value) / self.high_water_mark
            
            # Calculate win/loss metrics
            winning_positions = [pos for pos in open_positions.values() if pos.unrealized_pnl > 0]
            self.metrics.winning_positions = len(winning_positions)
            self.metrics.losing_positions = self.metrics.total_positions - self.metrics.winning_positions
            
            if self.metrics.total_positions > 0:
                self.metrics.win_rate = self.metrics.winning_positions / self.metrics.total_positions
            
            # Update strategy allocations
            total_value = self.metrics.total_value
            for strategy_name, allocation in self.strategy_allocations.items():
                strategy_positions = [pos for pos in open_positions.values() 
                                    if hasattr(pos, 'strategy_name') and pos.strategy_name == strategy_name]
                
                strategy_capital = sum(pos.size for pos in strategy_positions)
                allocation.current_allocation = strategy_capital / total_value if total_value > 0 else 0.0
                allocation.total_capital = strategy_capital
                allocation.unrealized_pnl = sum(pos.unrealized_pnl for pos in strategy_positions)
                
                self.metrics.strategy_allocations[strategy_name] = allocation.current_allocation
                self.metrics.strategy_performance[strategy_name] = allocation.current_pnl_percentage
            
            # Check for emergency conditions
            self._check_emergency_conditions()
            
            # Record daily value
            now = datetime.now()
            if not self.daily_values or (now - self.daily_values[-1][0]).total_seconds() > 3600:  # Hourly tracking
                self.daily_values.append((now, self.metrics.total_value))
                
                # Keep only last 30 days
                cutoff = now - timedelta(days=30)
                self.daily_values = [entry for entry in self.daily_values if entry[0] >= cutoff]
            
            self.last_updated = now

            # Record portfolio metrics with Prometheus
            try:
                metrics = get_metrics()
                if metrics:
                    metrics.update_portfolio_metrics(
                        self.metrics.total_value,
                        self.metrics.total_pnl,
                        self.metrics.current_drawdown,
                        len(open_positions)
                    )
            except Exception as e:
                logger.debug(f"[METRICS] Portfolio metrics recording failed (non-critical): {e}")

        except Exception as e:
            # Capture error with Sentry for professional error tracking
            capture_api_error(
                error=e,
                api_name="portfolio",
                endpoint="update_metrics",
                context={"portfolio_id": self.portfolio_id}
            )
            logger.error(f"[PORTFOLIO] Error updating metrics: {e}")

    def _check_emergency_conditions(self) -> None:
        """Check for emergency stop conditions"""
        try:
            # Check drawdown limits
            if self.metrics.current_drawdown > self.emergency_stop_loss:
                logger.error(f"[PORTFOLIO] EMERGENCY STOP: Drawdown {self.metrics.current_drawdown:.1%} > {self.emergency_stop_loss:.1%}")
                self.status = PortfolioStatus.EMERGENCY_STOP
                
            elif self.metrics.current_drawdown > self.max_portfolio_drawdown:
                logger.warning(f"[PORTFOLIO] Max drawdown exceeded: {self.metrics.current_drawdown:.1%} > {self.max_portfolio_drawdown:.1%}")
                if self.status == PortfolioStatus.ACTIVE:
                    self.status = PortfolioStatus.PAUSED

        except Exception as e:
            logger.error(f"[PORTFOLIO] Error checking emergency conditions: {e}")

    def get_total_value(self) -> float:
        """Get current total portfolio value"""
        return self.metrics.total_value

    def rebalance_strategies(self) -> Dict[str, float]:
        """Calculate rebalancing requirements"""
        try:
            rebalance_actions = {}
            total_value = self.get_total_value()
            
            for strategy_name, allocation in self.strategy_allocations.items():
                target_capital = total_value * allocation.target_allocation
                current_capital = allocation.total_capital
                
                difference = target_capital - current_capital
                if abs(difference) > (total_value * 0.05):  # 5% threshold for rebalancing
                    rebalance_actions[strategy_name] = difference
                    logger.info(f"[REBALANCE] {strategy_name}: {difference:+.2f} SOL needed")
            
            return rebalance_actions

        except Exception as e:
            # Capture error with Sentry for professional error tracking
            capture_api_error(
                error=e,
                api_name="portfolio",
                endpoint="rebalance_strategies",
                context={"portfolio_id": self.portfolio_id}
            )
            logger.error(f"[PORTFOLIO] Error calculating rebalancing: {e}")
            return {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio to dictionary"""
        return {
            "portfolio_id": self.portfolio_id,
            "initial_capital": self.initial_capital,
            "status": self.status.value,
            "allocation_strategy": self.allocation_strategy.value,
            "metrics": self.metrics.to_dict(),
            "strategy_allocations": {name: alloc.to_dict() for name, alloc in self.strategy_allocations.items()},
            "max_portfolio_drawdown": self.max_portfolio_drawdown,
            "emergency_stop_loss": self.emergency_stop_loss,
            "max_positions": self.max_positions,
            "max_position_concentration": self.max_position_concentration,
            "high_water_mark": self.high_water_mark,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }