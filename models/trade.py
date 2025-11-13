"""
Trade Models - NEW for Day 3 data model consolidation

MIGRATION NOTE: New comprehensive trade tracking models for portfolio management
CRITICAL: Foundation for strategy extraction and portfolio coordination
Enhanced with Sentry error tracking and Prometheus metrics for professional monitoring
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

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

class TradeStatus(Enum):
    """Trade status enumeration"""
    PENDING = "pending"
    EXECUTED = "executed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TradeDirection(Enum):
    """Trade direction enumeration"""
    BUY = "buy"
    SELL = "sell"

class TradeType(Enum):
    """Trade type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"

class ExecutionVenue(Enum):
    """Execution venue enumeration"""
    JUPITER = "jupiter"
    RAYDIUM = "raydium"
    ORCA = "orca"
    SERUM = "serum"
    METEORA = "meteora"

@dataclass
class TradeExecution:
    """Individual trade execution details"""
    execution_id: str
    timestamp: datetime
    price: float
    size: float
    venue: ExecutionVenue
    transaction_hash: str
    fees: float = 0.0
    slippage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution to dictionary"""
        return {
            "execution_id": self.execution_id,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "size": self.size,
            "venue": self.venue.value,
            "transaction_hash": self.transaction_hash,
            "fees": self.fees,
            "slippage": self.slippage
        }

@dataclass
class Trade:
    """Comprehensive trade model for portfolio tracking"""
    trade_id: str
    token_address: str
    direction: TradeDirection
    trade_type: TradeType
    size: float  # SOL amount for buy orders, token amount for sell orders
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Execution tracking
    status: TradeStatus = TradeStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
    # Execution details
    executions: List[TradeExecution] = field(default_factory=list)
    total_executed_size: float = 0.0
    average_execution_price: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    
    # Strategy metadata
    strategy_name: str = "unknown"
    signal_strength: float = 0.0
    signal_type: str = "unknown"
    
    # Risk management
    max_slippage: float = 0.05
    max_execution_time: int = 300  # 5 minutes
    
    def __post_init__(self):
        """Initialize trade after creation"""
        if not self.trade_id:
            self.trade_id = f"{self.direction.value}_{self.token_address[:8]}_{int(time.time())}"

    def add_execution(self, execution: TradeExecution) -> None:
        """Add execution to trade"""
        try:
            self.executions.append(execution)
            self.total_executed_size += execution.size
            self.total_fees += execution.fees
            self.total_slippage += execution.slippage
            
            # Recalculate average execution price
            total_value = sum(exec.price * exec.size for exec in self.executions)
            self.average_execution_price = total_value / self.total_executed_size if self.total_executed_size > 0 else 0.0
            
            # Update status
            if self.total_executed_size >= self.size:
                self.status = TradeStatus.EXECUTED
                self.executed_at = datetime.now()
            elif self.total_executed_size > 0:
                self.status = TradeStatus.PARTIAL
                
            logger.info(f"[TRADE] Execution added to {self.trade_id}: {execution.size} @ {execution.price}")

            # Record trade execution metrics with Prometheus
            try:
                metrics = get_metrics()
                if metrics:
                    metrics.record_trade_execution(
                        self.token_address, 
                        self.direction.value, 
                        execution.size, 
                        execution.price
                    )
            except Exception as e:
                logger.debug(f"[METRICS] Trade execution recording failed (non-critical): {e}")

        except Exception as e:
            # Capture error with Sentry for professional error tracking
            capture_api_error(
                error=e,
                api_name="trade_model",
                endpoint="add_execution",
                context={"trade_id": self.trade_id, "execution_id": execution.execution_id}
            )
            logger.error(f"[TRADE] Error adding execution to trade {self.trade_id}: {e}")

    def cancel(self, reason: str = "manual") -> None:
        """Cancel the trade"""
        try:
            self.status = TradeStatus.CANCELLED
            self.cancelled_at = datetime.now()
            logger.info(f"[TRADE] Trade {self.trade_id} cancelled: {reason}")

            # Record trade cancellation metrics with Prometheus
            try:
                metrics = get_metrics()
                if metrics:
                    metrics.record_trade_cancellation(self.token_address, reason)
            except Exception as e:
                logger.debug(f"[METRICS] Trade cancellation recording failed (non-critical): {e}")

        except Exception as e:
            # Capture error with Sentry for professional error tracking
            capture_api_error(
                error=e,
                api_name="trade_model",
                endpoint="cancel",
                context={"trade_id": self.trade_id, "reason": reason}
            )
            logger.error(f"[TRADE] Error cancelling trade {self.trade_id}: {e}")

    def mark_failed(self, reason: str = "execution_failed") -> None:
        """Mark trade as failed"""
        try:
            self.status = TradeStatus.FAILED
            logger.error(f"[TRADE] Trade {self.trade_id} marked as failed: {reason}")

            # Record trade failure metrics with Prometheus
            try:
                metrics = get_metrics()
                if metrics:
                    metrics.record_trade_failure(self.token_address, reason)
            except Exception as e:
                logger.debug(f"[METRICS] Trade failure recording failed (non-critical): {e}")

        except Exception as e:
            # Capture error with Sentry for professional error tracking
            capture_api_error(
                error=e,
                api_name="trade_model",
                endpoint="mark_failed",
                context={"trade_id": self.trade_id, "reason": reason}
            )
            logger.error(f"[TRADE] Error marking trade {self.trade_id} as failed: {e}")

    @property
    def is_complete(self) -> bool:
        """Check if trade is complete"""
        return self.status in [TradeStatus.EXECUTED, TradeStatus.FAILED, TradeStatus.CANCELLED]

    @property
    def is_expired(self) -> bool:
        """Check if trade has expired"""
        if self.is_complete:
            return False
        age_seconds = (datetime.now() - self.created_at).total_seconds()
        return age_seconds > self.max_execution_time

    @property
    def fill_percentage(self) -> float:
        """Get fill percentage (0.0 to 1.0)"""
        if self.size <= 0:
            return 0.0
        return min(self.total_executed_size / self.size, 1.0)

    @property
    def realized_pnl(self) -> float:
        """Calculate realized PnL for completed trades"""
        if self.status != TradeStatus.EXECUTED:
            return 0.0
        
        # For sell orders, PnL is the difference from entry price
        # This requires position tracking integration
        # Placeholder calculation for now
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        return {
            "trade_id": self.trade_id,
            "token_address": self.token_address,
            "direction": self.direction.value,
            "trade_type": self.trade_type.value,
            "size": self.size,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "total_executed_size": self.total_executed_size,
            "average_execution_price": self.average_execution_price,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "strategy_name": self.strategy_name,
            "signal_strength": self.signal_strength,
            "signal_type": self.signal_type,
            "fill_percentage": self.fill_percentage,
            "is_complete": self.is_complete,
            "is_expired": self.is_expired,
            "executions": [exec.to_dict() for exec in self.executions]
        }

@dataclass
class TradeBook:
    """Trade book for tracking all trades"""
    trades: Dict[str, Trade] = field(default_factory=dict)
    
    def add_trade(self, trade: Trade) -> None:
        """Add trade to book"""
        try:
            self.trades[trade.trade_id] = trade
            logger.info(f"[TRADE_BOOK] Added trade {trade.trade_id}: {trade.direction.value} {trade.size} {trade.token_address[:8]}...")

            # Record trade addition metrics with Prometheus
            try:
                metrics = get_metrics()
                if metrics:
                    metrics.record_trade_created(trade.token_address, trade.direction.value, trade.size)
            except Exception as e:
                logger.debug(f"[METRICS] Trade creation recording failed (non-critical): {e}")

        except Exception as e:
            # Capture error with Sentry for professional error tracking
            capture_api_error(
                error=e,
                api_name="trade_book",
                endpoint="add_trade",
                context={"trade_id": trade.trade_id}
            )
            logger.error(f"[TRADE_BOOK] Error adding trade {trade.trade_id}: {e}")

    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get trade by ID"""
        return self.trades.get(trade_id)

    def get_trades_by_token(self, token_address: str) -> List[Trade]:
        """Get all trades for a specific token"""
        return [trade for trade in self.trades.values() if trade.token_address == token_address]

    def get_trades_by_strategy(self, strategy_name: str) -> List[Trade]:
        """Get all trades for a specific strategy"""
        return [trade for trade in self.trades.values() if trade.strategy_name == strategy_name]

    def get_pending_trades(self) -> List[Trade]:
        """Get all pending trades"""
        return [trade for trade in self.trades.values() if trade.status == TradeStatus.PENDING]

    def get_active_trades(self) -> List[Trade]:
        """Get all active (pending or partial) trades"""
        return [trade for trade in self.trades.values() if trade.status in [TradeStatus.PENDING, TradeStatus.PARTIAL]]

    def get_completed_trades(self, since: Optional[datetime] = None) -> List[Trade]:
        """Get completed trades, optionally since a specific time"""
        completed = [trade for trade in self.trades.values() if trade.is_complete]
        if since:
            completed = [trade for trade in completed if trade.executed_at and trade.executed_at >= since]
        return completed

    def cleanup_old_trades(self, max_age_days: int = 30) -> int:
        """Clean up old completed trades"""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            old_trades = []
            
            for trade_id, trade in self.trades.items():
                if trade.is_complete and trade.executed_at and trade.executed_at < cutoff_date:
                    old_trades.append(trade_id)
            
            for trade_id in old_trades:
                del self.trades[trade_id]
            
            logger.info(f"[TRADE_BOOK] Cleaned up {len(old_trades)} old trades (older than {max_age_days} days)")
            return len(old_trades)

        except Exception as e:
            # Capture error with Sentry for professional error tracking
            capture_api_error(
                error=e,
                api_name="trade_book",
                endpoint="cleanup_old_trades",
                context={"max_age_days": max_age_days}
            )
            logger.error(f"[TRADE_BOOK] Error cleaning up old trades: {e}")
            return 0

    def get_trade_summary(self) -> Dict[str, Any]:
        """Get trade book summary"""
        try:
            total_trades = len(self.trades)
            pending_trades = len(self.get_pending_trades())
            active_trades = len(self.get_active_trades())
            completed_trades = len(self.get_completed_trades())
            
            # Calculate success rate
            executed_trades = [t for t in self.trades.values() if t.status == TradeStatus.EXECUTED]
            failed_trades = [t for t in self.trades.values() if t.status == TradeStatus.FAILED]
            success_rate = len(executed_trades) / (len(executed_trades) + len(failed_trades)) if (executed_trades or failed_trades) else 0.0
            
            # Calculate total volume
            total_volume = sum(trade.total_executed_size * trade.average_execution_price 
                             for trade in executed_trades if trade.average_execution_price > 0)
            
            return {
                "total_trades": total_trades,
                "pending_trades": pending_trades,
                "active_trades": active_trades,
                "completed_trades": completed_trades,
                "executed_trades": len(executed_trades),
                "failed_trades": len(failed_trades),
                "success_rate": success_rate,
                "total_volume": total_volume
            }

        except Exception as e:
            # Capture error with Sentry for professional error tracking
            capture_api_error(
                error=e,
                api_name="trade_book",
                endpoint="get_trade_summary",
                context={}
            )
            logger.error(f"[TRADE_BOOK] Error generating trade summary: {e}")
            return {
                "total_trades": 0,
                "pending_trades": 0,
                "active_trades": 0,
                "completed_trades": 0,
                "executed_trades": 0,
                "failed_trades": 0,
                "success_rate": 0.0,
                "total_volume": 0.0
            }