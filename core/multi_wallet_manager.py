#!/usr/bin/env python3
"""
MULTI-WALLET MANAGER - Day 10 Implementation
Advanced Multi-Wallet Support with Strategy Isolation

This module provides sophisticated multi-wallet management with complete strategy isolation,
risk compartmentalization, and intelligent capital flow management.

Key Features:
- Strategy-specific wallet isolation (Momentum, Mean Reversion, Grid Trading, Arbitrage)
- Risk isolation preventing cross-contamination between strategies
- Dynamic capital allocation and rebalancing
- Emergency controls and circuit breakers per wallet
- Performance-based resource allocation
- Comprehensive monitoring and alerting
"""

import asyncio
import logging
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from collections import defaultdict, deque
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)

class WalletType(Enum):
    """Wallet type classification"""
    MOMENTUM = "momentum"           # High-risk, high-reward momentum trading
    MEAN_REVERSION = "mean_reversion"  # Conservative mean reversion
    GRID_TRADING = "grid_trading"   # Range-bound trading
    ARBITRAGE = "arbitrage"         # Low-risk arbitrage opportunities
    EMERGENCY = "emergency"         # Emergency reserve wallet
    REBALANCE = "rebalance"        # Temporary rebalancing wallet

class WalletStatus(Enum):
    """Wallet operational status"""
    ACTIVE = "active"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    MAINTENANCE = "maintenance"
    LIQUIDATING = "liquidating"
    REBALANCING = "rebalancing"
    DISABLED = "disabled"

class CapitalFlowDirection(Enum):
    """Capital flow direction"""
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    REBALANCE_IN = "rebalance_in"
    REBALANCE_OUT = "rebalance_out"
    EMERGENCY_WITHDRAW = "emergency_withdraw"

@dataclass
class WalletConfig:
    """Comprehensive wallet configuration"""
    wallet_id: str
    wallet_type: WalletType
    strategy_name: str
    
    # Capital allocation
    target_allocation_pct: float  # Percentage of total portfolio
    min_allocation_pct: float = 0.05  # Minimum 5% allocation
    max_allocation_pct: float = 0.60  # Maximum 60% allocation
    
    # Risk parameters
    max_position_size_pct: float = 0.15  # 15% of wallet max per position
    max_daily_loss_pct: float = 0.05     # 5% daily loss limit
    max_drawdown_pct: float = 0.20       # 20% maximum drawdown
    emergency_stop_loss_pct: float = 0.10  # 10% emergency stop
    
    # Trading limits
    max_trades_per_day: int = 20
    max_concurrent_positions: int = 10
    min_trade_interval_seconds: int = 30
    
    # Rebalancing parameters
    rebalance_threshold_pct: float = 0.05  # 5% drift triggers rebalance
    max_rebalance_frequency: timedelta = timedelta(hours=4)
    
    # Reserve management
    emergency_reserve_pct: float = 0.10   # 10% emergency reserve
    operational_reserve_pct: float = 0.05  # 5% operational buffer
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert timedelta to seconds for JSON serialization
        result['max_rebalance_frequency'] = self.max_rebalance_frequency.total_seconds()
        result['wallet_type'] = self.wallet_type.value
        return result

@dataclass
class WalletMetrics:
    """Comprehensive wallet performance metrics"""
    wallet_id: str
    timestamp: datetime
    
    # Balance information
    allocated_capital: Decimal
    current_balance: Decimal
    available_balance: Decimal  # Balance minus reserved amounts
    reserved_balance: Decimal   # Emergency + operational reserves
    utilized_capital: Decimal   # Currently in positions
    
    # Performance metrics
    total_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    daily_pnl: Decimal = Decimal('0')
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    active_positions: int = 0
    avg_trade_size: Decimal = Decimal('0')
    
    # Risk metrics
    risk_score: float = 0.0  # 0-100 risk score
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_consecutive_losses: int = 0
    
    # Operational metrics
    trades_today: int = 0
    last_trade_time: Optional[datetime] = None
    last_rebalance_time: Optional[datetime] = None
    error_count_24h: int = 0
    health_score: float = 1.0  # 0-1 health score
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate percentage"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    def calculate_roi(self) -> float:
        """Calculate return on investment percentage"""
        if self.allocated_capital == 0:
            return 0.0
        return float((self.total_pnl / self.allocated_capital) * 100)
    
    def calculate_utilization(self) -> float:
        """Calculate capital utilization percentage"""
        if self.available_balance == 0:
            return 0.0
        return float((self.utilized_capital / self.available_balance) * 100)
    
    def is_healthy(self) -> bool:
        """Check if wallet is in healthy operational state"""
        return (
            self.health_score > 0.7 and
            self.error_count_24h < 5 and
            self.current_drawdown < 0.15 and  # Less than 15% drawdown
            float(self.daily_pnl) > -50.0     # Less than $50 daily loss
        )

@dataclass
class CapitalFlowEvent:
    """Capital flow event tracking"""
    timestamp: datetime
    from_wallet_id: str
    to_wallet_id: str
    amount: Decimal
    flow_direction: CapitalFlowDirection
    reason: str
    
    # Execution details
    success: bool = False
    execution_time_ms: float = 0.0
    transaction_hash: Optional[str] = None
    gas_cost: Decimal = Decimal('0')
    
    # Context
    trigger_event: str = ""
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    risk_metrics_before: Dict[str, Any] = field(default_factory=dict)
    risk_metrics_after: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'from_wallet_id': self.from_wallet_id,
            'to_wallet_id': self.to_wallet_id,
            'amount': float(self.amount),
            'flow_direction': self.flow_direction.value,
            'reason': self.reason,
            'success': self.success,
            'execution_time_ms': self.execution_time_ms,
            'transaction_hash': self.transaction_hash,
            'gas_cost': float(self.gas_cost),
            'trigger_event': self.trigger_event,
            'market_conditions': self.market_conditions,
            'risk_metrics_before': self.risk_metrics_before,
            'risk_metrics_after': self.risk_metrics_after
        }

class MultiWalletManager:
    """
    MULTI-WALLET MANAGER - Advanced multi-wallet support with strategy isolation
    
    This system provides comprehensive multi-wallet management with complete strategy
    isolation, preventing cross-contamination of risks and enabling optimized
    capital allocation across different trading approaches.
    """
    
    def __init__(self, settings=None, risk_manager=None):
        self.settings = settings
        self.risk_manager = risk_manager
        
        # Load configuration
        self._load_config()
        
        # Wallet management
        self.wallets: Dict[str, WalletConfig] = {}
        self.wallet_metrics: Dict[str, WalletMetrics] = {}
        self.wallet_balances: Dict[str, Decimal] = {}
        
        # Capital flow tracking
        self.capital_flow_history: deque = deque(maxlen=1000)
        self.pending_flows: List[CapitalFlowEvent] = []
        self.daily_flows: Dict[str, Decimal] = defaultdict(lambda: Decimal('0'))
        
        # Performance tracking
        self.total_portfolio_value = Decimal('0')
        self.strategy_performance_scores: Dict[str, float] = {}
        self.rebalancing_frequency = timedelta(hours=6)
        self.last_global_rebalance = datetime.now()
        
        # Emergency controls
        self.emergency_mode = False
        self.liquidation_mode = False
        self.capital_controls_active = False
        
        # Monitoring
        self.health_check_interval = timedelta(minutes=5)
        self.last_health_check = datetime.now()
        self.alerts: List[Dict[str, Any]] = []
        
        logger.info("[MULTI_WALLET] Multi-Wallet Manager initialized with strategy isolation")
    
    def _load_config(self):
        """Load multi-wallet configuration"""
        # Total portfolio capital
        self.total_capital = Decimal(os.getenv('TOTAL_PORTFOLIO_CAPITAL', '1000.0'))
        
        # Emergency thresholds
        self.global_emergency_loss_pct = float(os.getenv('GLOBAL_EMERGENCY_LOSS_PCT', '15.0'))
        self.max_daily_capital_flow_pct = float(os.getenv('MAX_DAILY_CAPITAL_FLOW_PCT', '20.0'))
        
        # Rebalancing parameters
        self.auto_rebalancing_enabled = os.getenv('AUTO_REBALANCING_ENABLED', 'true').lower() == 'true'
        self.rebalance_sensitivity = float(os.getenv('REBALANCE_SENSITIVITY', '0.05'))  # 5% drift
        
        # Performance thresholds
        self.underperformance_threshold = float(os.getenv('UNDERPERFORMANCE_THRESHOLD', '-10.0'))  # -10% return
        self.reallocation_trigger_threshold = float(os.getenv('REALLOCATION_TRIGGER_THRESHOLD', '0.15'))  # 15% drift
        
        logger.info(f"[MULTI_WALLET] Configuration loaded - Total Capital: ${self.total_capital}, "
                   f"Auto-rebalancing: {self.auto_rebalancing_enabled}")
    
    async def initialize_wallets(self) -> bool:
        """Initialize all strategy wallets with optimal configurations"""
        try:
            # Define strategy-specific wallet configurations
            wallet_configs = [
                # Momentum wallet - High risk, high reward
                WalletConfig(
                    wallet_id="momentum_wallet",
                    wallet_type=WalletType.MOMENTUM,
                    strategy_name="momentum",
                    target_allocation_pct=0.35,  # 35% allocation
                    max_allocation_pct=0.50,     # Can go up to 50%
                    min_allocation_pct=0.20,     # Minimum 20%
                    max_position_size_pct=0.20,  # Larger positions allowed
                    max_daily_loss_pct=0.08,     # Higher risk tolerance
                    max_drawdown_pct=0.25,       # Higher drawdown tolerance
                    max_trades_per_day=25,       # More active trading
                    max_concurrent_positions=12,
                    emergency_reserve_pct=0.05   # Lower reserves for more aggressive trading
                ),
                
                # Mean reversion wallet - Conservative approach
                WalletConfig(
                    wallet_id="mean_reversion_wallet",
                    wallet_type=WalletType.MEAN_REVERSION,
                    strategy_name="mean_reversion",
                    target_allocation_pct=0.30,  # 30% allocation
                    max_allocation_pct=0.40,     # Conservative max
                    min_allocation_pct=0.15,     # Stable minimum
                    max_position_size_pct=0.12,  # Smaller, safer positions
                    max_daily_loss_pct=0.03,     # Conservative loss limit
                    max_drawdown_pct=0.15,       # Lower drawdown tolerance
                    max_trades_per_day=15,       # Moderate trading frequency
                    max_concurrent_positions=8,
                    emergency_reserve_pct=0.15   # Higher reserves for stability
                ),
                
                # Grid trading wallet - Range-bound markets
                WalletConfig(
                    wallet_id="grid_trading_wallet",
                    wallet_type=WalletType.GRID_TRADING,
                    strategy_name="grid_trading",
                    target_allocation_pct=0.20,  # 20% allocation
                    max_allocation_pct=0.30,     # Moderate max
                    min_allocation_pct=0.10,     # Flexible minimum
                    max_position_size_pct=0.10,  # Many small positions
                    max_daily_loss_pct=0.04,     # Moderate risk
                    max_drawdown_pct=0.18,       # Moderate drawdown tolerance
                    max_trades_per_day=30,       # High frequency for grid
                    max_concurrent_positions=15, # Many small positions
                    emergency_reserve_pct=0.10   # Standard reserves
                ),
                
                # Arbitrage wallet - Low risk, high frequency
                WalletConfig(
                    wallet_id="arbitrage_wallet",
                    wallet_type=WalletType.ARBITRAGE,
                    strategy_name="arbitrage",
                    target_allocation_pct=0.15,  # 15% allocation
                    max_allocation_pct=0.25,     # Conservative max
                    min_allocation_pct=0.05,     # Can scale down significantly
                    max_position_size_pct=0.08,  # Small, quick positions
                    max_daily_loss_pct=0.02,     # Very low risk tolerance
                    max_drawdown_pct=0.10,       # Minimal drawdown tolerance
                    max_trades_per_day=50,       # High frequency
                    max_concurrent_positions=20, # Many quick trades
                    emergency_reserve_pct=0.05,  # Minimal reserves needed
                    min_trade_interval_seconds=15  # Fast execution
                )
            ]
            
            # Initialize each wallet
            initialization_results = []
            for config in wallet_configs:
                try:
                    success = await self._initialize_single_wallet(config)
                    initialization_results.append(success)
                    
                    if success:
                        logger.info(f"[MULTI_WALLET] Wallet {config.wallet_id} initialized successfully "
                                   f"({config.target_allocation_pct:.1%} allocation)")
                    else:
                        logger.error(f"[MULTI_WALLET] Failed to initialize wallet {config.wallet_id}")
                        
                except Exception as e:
                    logger.error(f"[MULTI_WALLET] Error initializing wallet {config.wallet_id}: {e}")
                    initialization_results.append(False)
            
            # Validate total allocations
            total_allocation = sum(config.target_allocation_pct for config in wallet_configs)
            if abs(total_allocation - 1.0) > 0.01:  # Allow 1% tolerance
                logger.warning(f"[MULTI_WALLET] Total allocation {total_allocation:.2%} != 100%")
            
            success_count = sum(initialization_results)
            total_count = len(initialization_results)
            
            logger.info(f"[MULTI_WALLET] Wallet initialization complete: {success_count}/{total_count} successful")
            
            return success_count == total_count
            
        except Exception as e:
            logger.error(f"[MULTI_WALLET] Wallet initialization failed: {e}")
            return False
    
    async def _initialize_single_wallet(self, config: WalletConfig) -> bool:
        """Initialize a single strategy wallet"""
        try:
            # Calculate initial capital allocation
            allocated_capital = self.total_capital * Decimal(str(config.target_allocation_pct))
            
            # Calculate reserves
            emergency_reserve = allocated_capital * Decimal(str(config.emergency_reserve_pct))
            operational_reserve = allocated_capital * Decimal(str(config.operational_reserve_pct))
            total_reserves = emergency_reserve + operational_reserve
            
            # Available balance for trading
            available_balance = allocated_capital - total_reserves
            
            # Create wallet metrics
            metrics = WalletMetrics(
                wallet_id=config.wallet_id,
                timestamp=datetime.now(),
                allocated_capital=allocated_capital,
                current_balance=allocated_capital,
                available_balance=available_balance,
                reserved_balance=total_reserves,
                utilized_capital=Decimal('0')
            )
            
            # Store configuration and metrics
            self.wallets[config.wallet_id] = config
            self.wallet_metrics[config.wallet_id] = metrics
            self.wallet_balances[config.wallet_id] = allocated_capital
            
            # Initialize performance tracking
            self.strategy_performance_scores[config.strategy_name] = 0.5  # Neutral start
            
            logger.debug(f"[MULTI_WALLET] Wallet {config.wallet_id}: ${allocated_capital} allocated, "
                        f"${available_balance} available, ${total_reserves} reserved")
            
            return True
            
        except Exception as e:
            logger.error(f"[MULTI_WALLET] Error initializing wallet {config.wallet_id}: {e}")
            return False
    
    async def execute_capital_flow(self, 
                                 from_wallet_id: str, 
                                 to_wallet_id: str, 
                                 amount: Decimal, 
                                 reason: str,
                                 flow_direction: CapitalFlowDirection = CapitalFlowDirection.REBALANCE_IN) -> bool:
        """Execute capital flow between wallets with comprehensive validation"""
        try:
            start_time = time.time()
            
            # Validate wallets exist
            if from_wallet_id not in self.wallets or to_wallet_id not in self.wallets:
                logger.error(f"[MULTI_WALLET] Invalid wallet IDs for capital flow")
                return False
            
            from_metrics = self.wallet_metrics[from_wallet_id]
            to_metrics = self.wallet_metrics[to_wallet_id]
            
            # Validate flow constraints
            if not await self._validate_capital_flow(from_wallet_id, to_wallet_id, amount, reason):
                return False
            
            # Execute the capital transfer
            transfer_success = await self._execute_capital_transfer(
                from_metrics, to_metrics, amount
            )
            
            if not transfer_success:
                logger.error(f"[MULTI_WALLET] Capital transfer execution failed")
                return False
            
            # Create flow event record
            flow_event = CapitalFlowEvent(
                timestamp=datetime.now(),
                from_wallet_id=from_wallet_id,
                to_wallet_id=to_wallet_id,
                amount=amount,
                flow_direction=flow_direction,
                reason=reason,
                success=True,
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
            # Record the flow
            self.capital_flow_history.append(flow_event)
            
            # Update daily flow tracking
            today = datetime.now().date().isoformat()
            self.daily_flows[today] += amount
            
            # Update total portfolio tracking
            await self._update_portfolio_metrics()
            
            logger.info(f"[MULTI_WALLET] Capital flow executed: ${amount} from {from_wallet_id} to {to_wallet_id} ({reason})")
            
            return True
            
        except Exception as e:
            logger.error(f"[MULTI_WALLET] Capital flow execution failed: {e}")
            return False
    
    async def _validate_capital_flow(self, from_wallet_id: str, to_wallet_id: str, 
                                   amount: Decimal, reason: str) -> bool:
        """Validate capital flow constraints"""
        try:
            from_metrics = self.wallet_metrics[from_wallet_id]
            to_config = self.wallets[to_wallet_id]
            
            # Check source wallet has sufficient available balance
            if amount > from_metrics.available_balance:
                logger.warning(f"[MULTI_WALLET] Insufficient balance in {from_wallet_id}: "
                             f"${amount} requested, ${from_metrics.available_balance} available")
                return False
            
            # Check daily flow limits
            today = datetime.now().date().isoformat()
            today_flows = self.daily_flows[today]
            max_daily_flow = self.total_capital * Decimal(str(self.max_daily_capital_flow_pct / 100))
            
            if today_flows + amount > max_daily_flow:
                logger.warning(f"[MULTI_WALLET] Daily flow limit exceeded: ${today_flows + amount} > ${max_daily_flow}")
                return False
            
            # Check target wallet allocation limits
            to_metrics = self.wallet_metrics[to_wallet_id]
            new_balance = to_metrics.current_balance + amount
            new_allocation_pct = float(new_balance / self.total_capital)
            
            if new_allocation_pct > to_config.max_allocation_pct:
                logger.warning(f"[MULTI_WALLET] Target wallet allocation limit exceeded: "
                             f"{new_allocation_pct:.2%} > {to_config.max_allocation_pct:.2%}")
                return False
            
            # Emergency mode restrictions
            if self.emergency_mode and reason != "emergency":
                logger.warning(f"[MULTI_WALLET] Capital flows restricted due to emergency mode")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"[MULTI_WALLET] Capital flow validation error: {e}")
            return False
    
    async def _execute_capital_transfer(self, 
                                      from_metrics: WalletMetrics, 
                                      to_metrics: WalletMetrics, 
                                      amount: Decimal) -> bool:
        """Execute the actual capital transfer between wallets"""
        try:
            # Atomic update of balances
            from_metrics.current_balance -= amount
            from_metrics.available_balance -= amount
            from_metrics.timestamp = datetime.now()
            
            to_metrics.current_balance += amount
            to_metrics.available_balance += amount
            to_metrics.timestamp = datetime.now()
            
            # Update tracking balances
            self.wallet_balances[from_metrics.wallet_id] = from_metrics.current_balance
            self.wallet_balances[to_metrics.wallet_id] = to_metrics.current_balance
            
            return True
            
        except Exception as e:
            logger.error(f"[MULTI_WALLET] Capital transfer execution error: {e}")
            return False
    
    async def rebalance_portfolio(self, reason: str = "scheduled") -> Dict[str, Any]:
        """Execute comprehensive portfolio rebalancing across all wallets"""
        try:
            start_time = time.time()
            
            if not self.auto_rebalancing_enabled and reason == "scheduled":
                return {"success": False, "reason": "Auto-rebalancing disabled"}
            
            # Calculate current allocations
            current_allocations = await self._calculate_current_allocations()
            
            # Calculate target allocations based on performance
            target_allocations = await self._calculate_optimal_allocations()
            
            # Determine required capital movements
            capital_movements = await self._calculate_rebalancing_movements(
                current_allocations, target_allocations
            )
            
            # Execute rebalancing movements
            rebalancing_results = []
            for movement in capital_movements:
                if abs(movement['amount']) < Decimal('10'):  # Skip small movements
                    continue
                
                success = await self.execute_capital_flow(
                    from_wallet_id=movement['from_wallet'],
                    to_wallet_id=movement['to_wallet'],
                    amount=movement['amount'],
                    reason=f"rebalancing_{reason}",
                    flow_direction=CapitalFlowDirection.REBALANCE_IN
                )
                
                rebalancing_results.append({
                    'from': movement['from_wallet'],
                    'to': movement['to_wallet'],
                    'amount': float(movement['amount']),
                    'success': success
                })
            
            # Update last rebalance time
            self.last_global_rebalance = datetime.now()
            
            # Calculate results summary
            total_moved = sum(abs(movement['amount']) for movement in capital_movements)
            successful_moves = sum(1 for result in rebalancing_results if result['success'])
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            logger.info(f"[MULTI_WALLET] Portfolio rebalancing complete: ${total_moved} moved, "
                       f"{successful_moves}/{len(rebalancing_results)} movements successful")
            
            return {
                "success": True,
                "reason": reason,
                "total_amount_moved": float(total_moved),
                "movements_executed": len(rebalancing_results),
                "successful_movements": successful_moves,
                "current_allocations": {k: float(v) for k, v in current_allocations.items()},
                "target_allocations": {k: float(v) for k, v in target_allocations.items()},
                "execution_time_ms": execution_time_ms,
                "movements": rebalancing_results
            }
            
        except Exception as e:
            logger.error(f"[MULTI_WALLET] Portfolio rebalancing failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_comprehensive_wallet_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all wallets and capital flows"""
        try:
            wallet_status = {}
            total_allocated = Decimal('0')
            total_pnl = Decimal('0')
            
            for wallet_id, metrics in self.wallet_metrics.items():
                config = self.wallets[wallet_id]
                
                wallet_status[wallet_id] = {
                    'config': {
                        'strategy_name': config.strategy_name,
                        'wallet_type': config.wallet_type.value,
                        'target_allocation_pct': config.target_allocation_pct,
                        'max_allocation_pct': config.max_allocation_pct,
                        'min_allocation_pct': config.min_allocation_pct
                    },
                    'balances': {
                        'allocated_capital': float(metrics.allocated_capital),
                        'current_balance': float(metrics.current_balance),
                        'available_balance': float(metrics.available_balance),
                        'reserved_balance': float(metrics.reserved_balance),
                        'utilized_capital': float(metrics.utilized_capital)
                    },
                    'performance': {
                        'total_pnl': float(metrics.total_pnl),
                        'roi_pct': metrics.calculate_roi(),
                        'win_rate_pct': metrics.calculate_win_rate(),
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'max_drawdown_pct': metrics.max_drawdown * 100,
                        'current_drawdown_pct': metrics.current_drawdown * 100
                    },
                    'activity': {
                        'total_trades': metrics.total_trades,
                        'trades_today': metrics.trades_today,
                        'active_positions': metrics.active_positions,
                        'last_trade': metrics.last_trade_time.isoformat() if metrics.last_trade_time else None
                    },
                    'health': {
                        'health_score': metrics.health_score,
                        'is_healthy': metrics.is_healthy(),
                        'error_count_24h': metrics.error_count_24h,
                        'risk_score': metrics.risk_score
                    }
                }
                
                total_allocated += metrics.allocated_capital
                total_pnl += metrics.total_pnl
            
            # Recent capital flows
            recent_flows = [
                flow.to_dict() for flow in list(self.capital_flow_history)[-10:]
            ]
            
            # Daily flow summary
            today = datetime.now().date().isoformat()
            daily_flow_total = float(self.daily_flows[today])
            
            return {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_wallets": len(self.wallets),
                    "total_allocated_capital": float(total_allocated),
                    "total_portfolio_pnl": float(total_pnl),
                    "portfolio_roi_pct": float((total_pnl / total_allocated * 100) if total_allocated > 0 else 0),
                    "emergency_mode": self.emergency_mode,
                    "auto_rebalancing_enabled": self.auto_rebalancing_enabled,
                    "last_rebalance": self.last_global_rebalance.isoformat()
                },
                "wallets": wallet_status,
                "capital_flows": {
                    "daily_total": daily_flow_total,
                    "daily_limit": float(self.total_capital * Decimal(str(self.max_daily_capital_flow_pct / 100))),
                    "recent_flows": recent_flows,
                    "total_flow_events": len(self.capital_flow_history)
                },
                "risk_status": {
                    "global_emergency_active": self.emergency_mode,
                    "liquidation_mode": self.liquidation_mode,
                    "capital_controls_active": self.capital_controls_active,
                    "healthy_wallets": sum(1 for metrics in self.wallet_metrics.values() if metrics.is_healthy())
                }
            }
            
        except Exception as e:
            logger.error(f"[MULTI_WALLET] Error getting comprehensive status: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    # Helper methods for internal operations
    async def _calculate_current_allocations(self) -> Dict[str, Decimal]:
        """Calculate current capital allocations by wallet"""
        allocations = {}
        for wallet_id, metrics in self.wallet_metrics.items():
            allocations[wallet_id] = metrics.current_balance / self.total_capital
        return allocations
    
    async def _calculate_optimal_allocations(self) -> Dict[str, Decimal]:
        """Calculate optimal allocations based on performance and market conditions"""
        # For now, return target allocations
        # In production, this would use sophisticated algorithms
        allocations = {}
        for wallet_id, config in self.wallets.items():
            allocations[wallet_id] = Decimal(str(config.target_allocation_pct))
        return allocations
    
    async def _calculate_rebalancing_movements(self, 
                                            current: Dict[str, Decimal], 
                                            target: Dict[str, Decimal]) -> List[Dict[str, Any]]:
        """Calculate required capital movements for rebalancing"""
        movements = []
        
        for wallet_id in current.keys():
            current_amount = current[wallet_id] * self.total_capital
            target_amount = target[wallet_id] * self.total_capital
            difference = target_amount - current_amount
            
            if abs(difference) > self.total_capital * Decimal(str(self.rebalance_sensitivity)):
                if difference > 0:
                    # Need to add capital to this wallet
                    movements.append({
                        'from_wallet': self._find_excess_wallet(current, target),
                        'to_wallet': wallet_id,
                        'amount': difference,
                        'type': 'rebalance_in'
                    })
                else:
                    # Need to remove capital from this wallet
                    movements.append({
                        'from_wallet': wallet_id,
                        'to_wallet': self._find_deficit_wallet(current, target),
                        'amount': abs(difference),
                        'type': 'rebalance_out'
                    })
        
        return movements
    
    def _find_excess_wallet(self, current: Dict[str, Decimal], target: Dict[str, Decimal]) -> str:
        """Find wallet with most excess capital"""
        max_excess = Decimal('0')
        excess_wallet = list(current.keys())[0]  # Default
        
        for wallet_id in current.keys():
            excess = (current[wallet_id] - target[wallet_id]) * self.total_capital
            if excess > max_excess:
                max_excess = excess
                excess_wallet = wallet_id
        
        return excess_wallet
    
    def _find_deficit_wallet(self, current: Dict[str, Decimal], target: Dict[str, Decimal]) -> str:
        """Find wallet with most capital deficit"""
        max_deficit = Decimal('0')
        deficit_wallet = list(current.keys())[0]  # Default
        
        for wallet_id in current.keys():
            deficit = (target[wallet_id] - current[wallet_id]) * self.total_capital
            if deficit > max_deficit:
                max_deficit = deficit
                deficit_wallet = wallet_id
        
        return deficit_wallet
    
    async def _update_portfolio_metrics(self):
        """Update overall portfolio metrics"""
        try:
            self.total_portfolio_value = sum(
                metrics.current_balance for metrics in self.wallet_metrics.values()
            )
        except Exception as e:
            logger.error(f"[MULTI_WALLET] Error updating portfolio metrics: {e}")
    
    async def shutdown(self):
        """Shutdown multi-wallet manager and save state"""
        try:
            logger.info("[MULTI_WALLET] Shutting down Multi-Wallet Manager...")
            
            # Save final state
            await self._save_wallet_state()
            
            logger.info("[MULTI_WALLET] Multi-Wallet Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"[MULTI_WALLET] Error during shutdown: {e}")
    
    async def _save_wallet_state(self):
        """Save wallet state to file"""
        try:
            os.makedirs("data/wallet_state", exist_ok=True)
            
            state_data = {
                'wallets': {wid: config.to_dict() for wid, config in self.wallets.items()},
                'wallet_metrics': {wid: asdict(metrics) for wid, metrics in self.wallet_metrics.items()},
                'capital_flows': [flow.to_dict() for flow in list(self.capital_flow_history)],
                'performance_scores': self.strategy_performance_scores,
                'last_updated': datetime.now().isoformat()
            }
            
            # Handle datetime serialization
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, Decimal):
                    return float(obj)
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            with open("data/wallet_state/multi_wallet_state.json", 'w') as f:
                json.dump(state_data, f, indent=2, default=serialize_datetime)
            
            logger.info("[MULTI_WALLET] Wallet state saved successfully")
            
        except Exception as e:
            logger.error(f"[MULTI_WALLET] Error saving wallet state: {e}")

# Global multi-wallet manager instance
_global_multi_wallet_manager: Optional[MultiWalletManager] = None

def get_multi_wallet_manager(settings=None, risk_manager=None) -> MultiWalletManager:
    """Get global multi-wallet manager instance"""
    global _global_multi_wallet_manager
    if _global_multi_wallet_manager is None:
        _global_multi_wallet_manager = MultiWalletManager(settings, risk_manager)
    return _global_multi_wallet_manager