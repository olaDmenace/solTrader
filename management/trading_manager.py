#!/usr/bin/env python3
"""
UNIFIED TRADING MANAGER - Day 10 Implementation
Master Trading Manager with Multi-Wallet Support Architecture

This unified trading manager serves as the central coordinator for all trading operations,
implementing a sophisticated multi-wallet architecture that isolates risk across strategies.

Key Features:
- Central trading coordination for all 4 strategies (Momentum, Mean Reversion, Grid Trading, Arbitrage)
- Multi-wallet support with risk isolation per strategy
- Intelligent strategy coordination and conflict resolution
- Dynamic capital flow management between wallets
- Integration with Day 8 Unified Risk Manager and Day 9 Unified Portfolio Manager
- Emergency controls and circuit breakers
- Performance-based resource allocation
- Real-time monitoring and alerting
"""

import asyncio
import logging
import time
import json
import sqlite3
import aiosqlite
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from collections import defaultdict, deque
import statistics
from pathlib import Path

# Import our core components
from management.risk_manager import UnifiedRiskManager, RiskLevel, RiskEvent
from management.portfolio_manager import UnifiedPortfolioManager, PortfolioStatus, AllocationStrategy

# Import strategy interfaces
from strategies.base import BaseStrategy, StrategyType, StrategyStatus, StrategyConfig
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.grid_trading import GridTradingStrategy
from strategies.arbitrage import ArbitrageStrategy

# Import master coordinator for advanced coordination
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from strategies.coordinator import MasterStrategyCoordinator

# Import model dependencies
from models.position import Position
from models.trade import Trade, TradeDirection, TradeType
from models.signal import Signal

logger = logging.getLogger(__name__)

class WalletStatus(Enum):
    """Wallet status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    MAINTENANCE = "maintenance"
    LIQUIDATING = "liquidating"

class CoordinationMode(Enum):
    """Strategy coordination modes"""
    INDEPENDENT = "independent"      # No coordination
    COOPERATIVE = "cooperative"      # Share opportunities
    COMPETITIVE = "competitive"      # Compete for resources
    HIERARCHICAL = "hierarchical"    # Priority-based
    BALANCED = "balanced"           # Balanced approach

class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    FIRST_COME_FIRST_SERVE = "fcfs"
    HIGHEST_CONFIDENCE = "highest_confidence"
    BEST_RISK_REWARD = "best_risk_reward"
    STRATEGY_PRIORITY = "strategy_priority"
    RANDOM = "random"

@dataclass
class StrategyWallet:
    """Multi-wallet configuration for strategy isolation"""
    wallet_id: str
    strategy_name: str
    strategy_type: StrategyType
    
    # Wallet configuration
    allocated_capital: Decimal
    current_balance: Decimal
    reserved_balance: Decimal = Decimal('0')  # Emergency reserves
    
    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: Decimal = Decimal('0')
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Risk controls
    max_position_size: Decimal = Decimal('0.1')  # 10% of wallet
    max_daily_loss: Decimal = Decimal('50')      # $50 daily loss limit
    risk_budget: float = 0.15                    # 15% risk budget
    
    # Status and health
    status: WalletStatus = WalletStatus.ACTIVE
    last_trade: Optional[datetime] = None
    last_rebalance: Optional[datetime] = None
    error_count: int = 0
    health_score: float = 1.0
    
    # Strategy-specific parameters
    strategy_config: Optional[Dict[str, Any]] = None
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate for this wallet"""
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
    
    def calculate_roi(self) -> float:
        """Calculate return on investment for this wallet"""
        if self.allocated_capital > 0:
            return float(self.total_pnl / self.allocated_capital)
        return 0.0
    
    def needs_rebalancing(self, threshold: float = 0.05) -> bool:
        """Check if wallet needs capital rebalancing"""
        if self.allocated_capital == 0:
            return False
        balance_ratio = float(self.current_balance / self.allocated_capital)
        return abs(balance_ratio - 1.0) > threshold
    
    def is_healthy(self) -> bool:
        """Check if wallet is in healthy state"""
        return (self.status == WalletStatus.ACTIVE and 
                self.health_score > 0.7 and 
                self.error_count < 5 and
                float(self.total_pnl) > -float(self.max_daily_loss))

@dataclass
class TradeOpportunity:
    """Unified trade opportunity representation"""
    token_address: str
    signal: Signal
    strategy_name: str
    confidence: float
    expected_return: float
    risk_score: float
    position_size: Decimal
    entry_price: float
    
    # Priority scoring
    priority_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Execution details
    wallet_id: Optional[str] = None
    assigned_strategy: Optional[str] = None
    execution_status: str = "pending"
    
    def calculate_risk_reward(self) -> float:
        """Calculate risk-reward ratio"""
        if self.risk_score > 0:
            return self.expected_return / self.risk_score
        return 0.0
    
    def calculate_priority_score(self, strategy_performance: Dict[str, float]) -> float:
        """Calculate priority score for opportunity allocation"""
        # Base score from confidence and expected return
        base_score = (self.confidence * 0.4) + (self.expected_return * 0.3)
        
        # Strategy performance bonus
        strategy_performance_score = strategy_performance.get(self.strategy_name, 0.5)
        performance_bonus = strategy_performance_score * 0.2
        
        # Risk adjustment
        risk_adjustment = max(0, 1.0 - self.risk_score) * 0.1
        
        self.priority_score = base_score + performance_bonus + risk_adjustment
        return self.priority_score

@dataclass
class CoordinationEvent:
    """Strategy coordination event logging"""
    timestamp: datetime
    event_type: str  # "conflict", "allocation", "rebalance", etc.
    involved_strategies: List[str]
    token_address: Optional[str] = None
    resolution: Optional[str] = None
    outcome: str = "pending"
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'involved_strategies': self.involved_strategies,
            'token_address': self.token_address,
            'resolution': self.resolution,
            'outcome': self.outcome,
            'details': self.details
        }

class UnifiedTradingManager:
    """
    UNIFIED TRADING MANAGER - Central coordinator for all trading operations
    
    This system provides comprehensive trading coordination with multi-wallet support,
    strategy coordination, and intelligent resource allocation.
    """
    
    def __init__(self, 
                 settings=None, 
                 risk_manager: UnifiedRiskManager = None,
                 portfolio_manager: UnifiedPortfolioManager = None,
                 db_path: str = "logs/unified_trading.db",
                 master_coordinator=None):
        
        self.settings = settings
        self.risk_manager = risk_manager
        self.portfolio_manager = portfolio_manager
        self.db_path = db_path
        self.master_coordinator = master_coordinator
        
        # Load configuration
        self._load_config()
        
        # Multi-wallet system
        self.strategy_wallets: Dict[str, StrategyWallet] = {}
        self.wallet_balances: Dict[str, Decimal] = {}
        
        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_performance: Dict[str, float] = {}
        self.strategy_configs: Dict[str, StrategyConfig] = {}
        
        # Coordination system
        self.coordination_mode = CoordinationMode.BALANCED
        self.conflict_resolution = ConflictResolution.HIGHEST_CONFIDENCE
        self.pending_opportunities: List[TradeOpportunity] = []
        self.coordination_events: deque = deque(maxlen=1000)
        
        # Resource allocation
        self.allocation_strategy = AllocationStrategy.PERFORMANCE_BASED
        self.rebalance_threshold = 0.05  # 5% drift threshold
        self.last_rebalance = datetime.now()
        self.rebalance_interval = timedelta(hours=6)  # Rebalance every 6 hours
        
        # Emergency controls
        self.emergency_stop = False
        self.liquidation_mode = False
        self.max_concurrent_trades = 10
        self.trade_frequency_limit = timedelta(minutes=1)  # Min 1 minute between trades
        
        # Performance tracking
        self.total_trades_today = 0
        self.successful_coordinations = 0
        self.failed_coordinations = 0
        self.coordination_success_rate = 0.0
        
        # Monitoring and alerting
        self.last_health_check = datetime.now()
        self.health_check_interval = timedelta(minutes=5)
        self.alerts: List[Dict[str, Any]] = []
        
        logger.info("[UNIFIED_TRADING] Unified Trading Manager initialized with multi-wallet support")
    
    def _load_config(self):
        """Load trading manager configuration"""
        # Multi-wallet configuration
        self.initial_capital_per_wallet = {
            'momentum': Decimal(os.getenv('MOMENTUM_WALLET_CAPITAL', '600.0')),    # 60% allocation
            'mean_reversion': Decimal(os.getenv('MEAN_REVERSION_WALLET_CAPITAL', '400.0')),  # 40% allocation
            'grid_trading': Decimal(os.getenv('GRID_TRADING_WALLET_CAPITAL', '300.0')),      # 30% allocation
            'arbitrage': Decimal(os.getenv('ARBITRAGE_WALLET_CAPITAL', '200.0'))             # 20% allocation
        }
        
        # Trading limits
        self.max_trades_per_day = int(os.getenv('MAX_TRADES_PER_DAY', '50'))
        self.max_trades_per_strategy = int(os.getenv('MAX_TRADES_PER_STRATEGY', '15'))
        self.min_trade_interval_seconds = int(os.getenv('MIN_TRADE_INTERVAL_SECONDS', '30'))
        
        # Risk controls
        self.max_total_exposure = Decimal(os.getenv('MAX_TOTAL_EXPOSURE', '0.8'))  # 80% max exposure
        self.emergency_liquidation_threshold = Decimal(os.getenv('EMERGENCY_LIQUIDATION_THRESHOLD', '0.15'))  # 15% loss
        
        # Strategy priorities (higher number = higher priority)
        self.strategy_priorities = {
            'momentum': int(os.getenv('MOMENTUM_PRIORITY', '4')),
            'mean_reversion': int(os.getenv('MEAN_REVERSION_PRIORITY', '3')),
            'arbitrage': int(os.getenv('ARBITRAGE_PRIORITY', '5')),  # Highest priority
            'grid_trading': int(os.getenv('GRID_TRADING_PRIORITY', '2'))
        }
        
        logger.info(f"[UNIFIED_TRADING] Configuration loaded - Max trades/day: {self.max_trades_per_day}, "
                   f"Strategy priorities: {self.strategy_priorities}")
    
    async def initialize(self):
        """Initialize trading manager and all components"""
        try:
            # Create database tables
            await self._create_trading_tables()
            
            # Initialize multi-wallet system
            await self._initialize_wallets()
            
            # Initialize strategies
            await self._initialize_strategies()
            
            # Load historical data
            await self._load_historical_data()
            
            # Start monitoring tasks
            asyncio.create_task(self._health_monitor_loop())
            asyncio.create_task(self._rebalancing_loop())
            asyncio.create_task(self._coordination_loop())
            
            logger.info("[UNIFIED_TRADING] Trading manager initialization complete")
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Initialization failed: {e}")
            raise
    
    async def _create_trading_tables(self):
        """Create comprehensive trading management database tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Strategy wallets table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_wallets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        wallet_id TEXT UNIQUE NOT NULL,
                        strategy_name TEXT NOT NULL,
                        strategy_type TEXT NOT NULL,
                        allocated_capital REAL NOT NULL,
                        current_balance REAL NOT NULL,
                        total_pnl REAL DEFAULT 0,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'active',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                
                # Trade coordination events
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS coordination_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        involved_strategies TEXT NOT NULL,
                        token_address TEXT,
                        resolution TEXT,
                        outcome TEXT DEFAULT 'pending',
                        details TEXT
                    )
                """)
                
                # Strategy performance tracking
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        performance_score REAL NOT NULL,
                        sharpe_ratio REAL DEFAULT 0,
                        win_rate REAL DEFAULT 0,
                        total_return REAL DEFAULT 0,
                        max_drawdown REAL DEFAULT 0
                    )
                """)
                
                # Capital rebalancing events
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS rebalancing_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        old_allocations TEXT NOT NULL,
                        new_allocations TEXT NOT NULL,
                        capital_moved REAL NOT NULL,
                        success BOOLEAN DEFAULT FALSE,
                        execution_time_ms REAL DEFAULT 0
                    )
                """)
                
                await db.commit()
                logger.info("[UNIFIED_TRADING] Trading management database tables created")
                
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Database initialization failed: {e}")
            raise
    
    async def _initialize_wallets(self):
        """Initialize multi-wallet system with strategy isolation"""
        try:
            for strategy_name, capital in self.initial_capital_per_wallet.items():
                wallet_id = f"wallet_{strategy_name}"
                
                # Determine strategy type
                strategy_type_map = {
                    'momentum': StrategyType.MOMENTUM,
                    'mean_reversion': StrategyType.MEAN_REVERSION,
                    'grid_trading': StrategyType.GRID_TRADING,
                    'arbitrage': StrategyType.ARBITRAGE
                }
                
                strategy_type = strategy_type_map.get(strategy_name, StrategyType.MOMENTUM)
                
                # Create strategy wallet
                wallet = StrategyWallet(
                    wallet_id=wallet_id,
                    strategy_name=strategy_name,
                    strategy_type=strategy_type,
                    allocated_capital=capital,
                    current_balance=capital,
                    reserved_balance=capital * Decimal('0.1'),  # 10% reserve
                    max_position_size=capital * Decimal('0.15'),  # 15% max position
                    max_daily_loss=capital * Decimal('0.05'),     # 5% daily loss limit
                    strategy_config=self._get_strategy_config(strategy_name)
                )
                
                self.strategy_wallets[strategy_name] = wallet
                self.wallet_balances[wallet_id] = capital
                
                # Store in database
                await self._store_wallet_config(wallet)
                
                logger.info(f"[UNIFIED_TRADING] Initialized wallet for {strategy_name}: ${capital}")
            
            logger.info(f"[UNIFIED_TRADING] Multi-wallet system initialized with {len(self.strategy_wallets)} wallets")
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Wallet initialization failed: {e}")
            raise
    
    async def _initialize_strategies(self):
        """Initialize all trading strategies with their dedicated wallets"""
        try:
            # Try to import strategy classes, but don't fail if they don't exist
            strategy_classes = {}
            try:
                from strategies.momentum import MomentumStrategy
                strategy_classes['momentum'] = MomentumStrategy
            except ImportError:
                logger.warning("[UNIFIED_TRADING] MomentumStrategy not found, skipping")
            
            try:
                from strategies.mean_reversion import MeanReversionStrategy
                strategy_classes['mean_reversion'] = MeanReversionStrategy
            except ImportError:
                logger.warning("[UNIFIED_TRADING] MeanReversionStrategy not found, skipping")
            
            try:
                from strategies.grid_trading import GridTradingStrategy
                strategy_classes['grid_trading'] = GridTradingStrategy
            except ImportError:
                logger.warning("[UNIFIED_TRADING] GridTradingStrategy not found, skipping")
            
            try:
                from strategies.arbitrage import ArbitrageStrategy
                strategy_classes['arbitrage'] = ArbitrageStrategy
            except ImportError:
                logger.warning("[UNIFIED_TRADING] ArbitrageStrategy not found, skipping")
            
            if not strategy_classes:
                logger.info("[UNIFIED_TRADING] No strategy classes found, strategies will be set up separately")
            
            for strategy_name, strategy_class in strategy_classes.items():
                if strategy_name in self.strategy_wallets:
                    wallet = self.strategy_wallets[strategy_name]
                    
                    # Create strategy configuration
                    config = StrategyConfig(
                        strategy_name=strategy_name,
                        strategy_type=wallet.strategy_type,
                        max_positions=10,
                        max_position_size=float(wallet.max_position_size)
                        # Note: Strategy-specific config handled separately to avoid unexpected keyword args
                    )
                    
                    # Initialize strategy - use minimal parameters to avoid issues
                    try:
                        strategy = strategy_class(config=config)
                    except Exception as e:
                        logger.warning(f"Strategy class {strategy_class.__name__} failed with config parameter, trying without: {e}")
                        # Fallback: try to initialize with just the class
                        strategy = strategy_class()
                    
                    await strategy.initialize()
                    
                    self.strategies[strategy_name] = strategy
                    self.strategy_configs[strategy_name] = config
                    self.strategy_performance[strategy_name] = 0.5  # Neutral starting performance
                    
                    logger.info(f"[UNIFIED_TRADING] Strategy {strategy_name} initialized with wallet {wallet.wallet_id}")
            
            logger.info(f"[UNIFIED_TRADING] All {len(self.strategies)} strategies initialized successfully")
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Strategy initialization failed: {e}")
            raise
    
    async def process_trading_signals(self, signals: List[Signal]) -> Dict[str, Any]:
        """
        ENHANCED CENTRAL SIGNAL PROCESSING - Master coordinator integration
        
        This is the heart of the trading coordination system with master coordinator support
        """
        try:
            if self.emergency_stop or len(signals) == 0:
                return {"processed": 0, "executed": 0, "conflicts": 0}
            
            start_time = time.time()
            logger.info(f"[UNIFIED_TRADING] Processing {len(signals)} signals with enhanced coordination")
            
            # PHASE 1: Signal Pre-Processing with Master Coordinator
            if self.master_coordinator:
                try:
                    # Let master coordinator filter and enhance signals
                    enhanced_signals = await self.master_coordinator.process_incoming_signals(signals)
                    logger.info(f"[UNIFIED_TRADING] Master coordinator enhanced {len(signals)} -> {len(enhanced_signals)} signals")
                    signals = enhanced_signals
                except Exception as e:
                    logger.warning(f"[UNIFIED_TRADING] Master coordinator signal processing failed: {e}, using original signals")
            
            # PHASE 2: Convert signals to trade opportunities
            opportunities = []
            for signal in signals:
                opportunity = await self._convert_signal_to_opportunity(signal)
                if opportunity:
                    opportunities.append(opportunity)
            
            logger.info(f"[UNIFIED_TRADING] Created {len(opportunities)} opportunities from {len(signals)} signals")
            
            # PHASE 3: Master Coordinator Advanced Coordination
            if self.master_coordinator and opportunities:
                try:
                    # Let master coordinator handle advanced coordination
                    coordinated_opportunities = await self.master_coordinator.coordinate_strategy_opportunities(
                        opportunities, self.strategy_performance, self.strategy_wallets
                    )
                    logger.info(f"[UNIFIED_TRADING] Master coordinator coordinated {len(opportunities)} -> {len(coordinated_opportunities)} opportunities")
                except Exception as e:
                    logger.warning(f"[UNIFIED_TRADING] Master coordinator coordination failed: {e}, using fallback")
                    # Fallback to local coordination
                    coordinated_opportunities = await self._coordinate_opportunities(opportunities)
            else:
                # Local coordination fallback
                coordinated_opportunities = await self._coordinate_opportunities(opportunities)
            
            # PHASE 4: Execute coordinated trades
            execution_results = await self._execute_coordinated_trades(coordinated_opportunities)
            
            # PHASE 5: Post-execution coordination feedback
            if self.master_coordinator:
                try:
                    await self.master_coordinator.process_execution_results(
                        coordinated_opportunities, execution_results
                    )
                except Exception as e:
                    logger.warning(f"[UNIFIED_TRADING] Master coordinator feedback failed: {e}")
            
            # PHASE 6: Log comprehensive coordination event
            coordination_event = CoordinationEvent(
                timestamp=datetime.now(),
                event_type="enhanced_signal_processing",
                involved_strategies=list(set(op.strategy_name for op in opportunities)),
                resolution=f"processed_{len(opportunities)}_opportunities_with_master_coordinator",
                outcome="success",
                details={
                    "input_signals": len(signals),
                    "opportunities_created": len(opportunities),
                    "opportunities_coordinated": len(coordinated_opportunities),
                    "trades_executed": len(execution_results.get("successful", [])),
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "master_coordinator_active": self.master_coordinator is not None,
                    "coordination_quality": "high" if len(coordinated_opportunities) > 0 else "none"
                }
            )
            
            self.coordination_events.append(coordination_event)
            await self._log_coordination_event(coordination_event)
            
            # Update performance metrics
            self._update_coordination_metrics(len(opportunities), len(execution_results.get("successful", [])))
            
            result = {
                "processed": len(opportunities),
                "executed": len(execution_results.get("successful", [])),
                "conflicts": len(opportunities) - len(coordinated_opportunities),
                "execution_time_ms": (time.time() - start_time) * 1000,
                "coordination_success_rate": self.coordination_success_rate,
                "master_coordinator_enabled": self.master_coordinator is not None
            }
            
            logger.info(f"[UNIFIED_TRADING] Signal processing complete: {result['executed']}/{result['processed']} executed ({result['coordination_success_rate']:.1%} success)")
            return result
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Enhanced signal processing failed: {e}")
            return {"processed": 0, "executed": 0, "conflicts": 0, "error": str(e)}
    
    async def _convert_signal_to_opportunity(self, signal: Signal) -> Optional[TradeOpportunity]:
        """Convert a trading signal to a coordinated trade opportunity"""
        try:
            # Determine which strategy should handle this signal
            best_strategy = await self._determine_best_strategy_for_signal(signal)
            if not best_strategy:
                return None
            
            # Get strategy wallet
            wallet = self.strategy_wallets.get(best_strategy)
            if not wallet or not wallet.is_healthy():
                return None
            
            # Calculate position size based on wallet capacity and signal strength
            max_position_value = wallet.current_balance * wallet.max_position_size
            position_size = min(
                max_position_value,
                Decimal(str(signal.strength)) * max_position_value
            )
            
            # Create trade opportunity
            opportunity = TradeOpportunity(
                token_address=signal.token_address,
                signal=signal,
                strategy_name=best_strategy,
                confidence=signal.confidence,
                expected_return=signal.expected_return,
                risk_score=signal.risk_score,
                position_size=position_size,
                entry_price=signal.entry_price,
                wallet_id=wallet.wallet_id
            )
            
            # Calculate priority score
            opportunity.calculate_priority_score(self.strategy_performance)
            
            return opportunity
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error converting signal to opportunity: {e}")
            return None
    
    async def _determine_best_strategy_for_signal(self, signal: Signal) -> Optional[str]:
        """Determine the best strategy to handle a given signal"""
        try:
            # Strategy-specific signal matching
            strategy_scores = {}
            
            for strategy_name, strategy in self.strategies.items():
                # Skip inactive strategies
                if not self.strategy_wallets[strategy_name].is_healthy():
                    continue
                
                # Calculate strategy suitability score
                suitability = await self._calculate_strategy_suitability(strategy, signal)
                if suitability > 0.3:  # Minimum suitability threshold
                    strategy_scores[strategy_name] = suitability
            
            if not strategy_scores:
                return None
            
            # Return strategy with highest suitability score
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            
            logger.debug(f"[UNIFIED_TRADING] Signal for {signal.token_address[:8]}... assigned to {best_strategy}")
            return best_strategy
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error determining best strategy: {e}")
            return None
    
    async def _calculate_strategy_suitability(self, strategy: BaseStrategy, signal: Signal) -> float:
        """ENHANCED STRATEGY SUITABILITY - Advanced signal-strategy matching"""
        try:
            # Get strategy name safely
            strategy_name = getattr(strategy.config, 'strategy_name', None) if hasattr(strategy, 'config') else None
            if not strategy_name:
                # Fallback: derive from class name
                strategy_name = strategy.__class__.__name__.lower().replace('strategy', '')
            
            base_score = 0.5  # Base suitability
            
            # ENHANCED STRATEGY-SPECIFIC SUITABILITY LOGIC
            if hasattr(strategy, 'calculate_signal_suitability'):
                try:
                    strategy_score = await strategy.calculate_signal_suitability(signal)
                    base_score = strategy_score
                except Exception as e:
                    logger.debug(f"[UNIFIED_TRADING] Strategy suitability method failed: {e}")
                    # Use built-in logic as fallback
                    base_score = await self._builtin_strategy_suitability(strategy_name, signal)
            else:
                # Built-in suitability logic
                base_score = await self._builtin_strategy_suitability(strategy_name, signal)
            
            # PERFORMANCE ADJUSTMENT (Enhanced)
            performance = self.strategy_performance.get(strategy_name, 0.5)
            
            # More nuanced performance multiplier
            if performance > 0.7:
                performance_multiplier = 1.2  # High performing strategies get bonus
            elif performance > 0.5:
                performance_multiplier = 1.0  # Neutral strategies
            elif performance > 0.3:
                performance_multiplier = 0.8  # Underperforming strategies get penalty
            else:
                performance_multiplier = 0.6  # Poor performing strategies heavily penalized
            
            # CAPACITY ADJUSTMENT (Enhanced)
            wallet = self.strategy_wallets.get(strategy_name)
            if not wallet:
                return 0.0  # No wallet = no suitability
            
            capacity_ratio = float(wallet.current_balance / wallet.allocated_capital) if wallet.allocated_capital > 0 else 0
            
            # More sophisticated capacity scoring
            if capacity_ratio > 0.8:
                capacity_multiplier = 1.0  # Plenty of capital
            elif capacity_ratio > 0.5:
                capacity_multiplier = 0.9  # Moderate capital
            elif capacity_ratio > 0.2:
                capacity_multiplier = 0.7  # Low capital
            else:
                capacity_multiplier = 0.3  # Very low capital
            
            # HEALTH ADJUSTMENT
            health_multiplier = 1.0
            if not wallet.is_healthy():
                health_multiplier = 0.5  # Unhealthy wallets get reduced suitability
            
            # WORKLOAD ADJUSTMENT
            # If strategy is overloaded, reduce suitability
            current_positions = len(getattr(strategy, 'positions', []))
            max_positions = getattr(strategy.config, 'max_positions', 10) if hasattr(strategy, 'config') else 10
            workload_ratio = current_positions / max_positions
            
            if workload_ratio > 0.8:
                workload_multiplier = 0.7  # Overloaded
            elif workload_ratio > 0.6:
                workload_multiplier = 0.9  # Busy
            else:
                workload_multiplier = 1.0  # Available
            
            # COMBINE ALL FACTORS
            final_score = (base_score * 
                          performance_multiplier * 
                          capacity_multiplier * 
                          health_multiplier * 
                          workload_multiplier)
            
            # Bound the result
            final_score = max(0.0, min(1.0, final_score))
            
            logger.debug(f"[UNIFIED_TRADING] Strategy {strategy_name} suitability: {final_score:.3f} "
                        f"(base: {base_score:.3f}, perf: {performance_multiplier:.3f}, "
                        f"cap: {capacity_multiplier:.3f}, health: {health_multiplier:.3f})")
            
            return final_score
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error calculating strategy suitability: {e}")
            return 0.0
    
    async def _builtin_strategy_suitability(self, strategy_name: str, signal: Signal) -> float:
        """Built-in strategy suitability calculation when strategy doesn't provide its own"""
        try:
            # SIGNAL CHARACTERISTIC ANALYSIS
            signal_strength = getattr(signal, 'strength', 0.5)
            signal_confidence = getattr(signal, 'confidence', 0.5)
            signal_timeframe = getattr(signal, 'timeframe', 'medium')
            signal_type = getattr(signal, 'signal_type', 'unknown')
            
            # Base suitability score
            suitability = 0.4
            
            # STRATEGY-SPECIFIC MATCHING LOGIC
            if 'momentum' in strategy_name.lower():
                # Momentum strategies prefer strong, confident signals with short-medium timeframes
                if signal_strength > 0.6:
                    suitability += 0.2
                if signal_confidence > 0.7:
                    suitability += 0.2
                if signal_type in ['breakout', 'trend', 'momentum']:
                    suitability += 0.15
                if signal_timeframe in ['short', 'medium']:
                    suitability += 0.1
                    
            elif 'mean_reversion' in strategy_name.lower():
                # Mean reversion strategies prefer oversold/overbought signals
                if signal_type in ['oversold', 'overbought', 'reversal']:
                    suitability += 0.25
                if signal_timeframe in ['medium', 'long']:
                    suitability += 0.15
                if hasattr(signal, 'rsi_value') and (signal.rsi_value < 30 or signal.rsi_value > 70):
                    suitability += 0.2
                    
            elif 'grid' in strategy_name.lower():
                # Grid trading prefers sideways/range-bound markets
                if signal_type in ['range', 'sideways', 'consolidation']:
                    suitability += 0.25
                if 0.4 <= signal_strength <= 0.7:  # Moderate signals are better for grid
                    suitability += 0.15
                    
            elif 'arbitrage' in strategy_name.lower():
                # Arbitrage strategies prefer immediate execution opportunities
                if signal_type in ['arbitrage', 'price_difference', 'immediate']:
                    suitability += 0.3
                if signal_timeframe == 'immediate':
                    suitability += 0.2
                if signal_confidence > 0.8:  # High confidence required
                    suitability += 0.15
            
            # SIGNAL QUALITY ADJUSTMENTS
            if signal_confidence > 0.8:
                suitability += 0.1  # High confidence bonus
            elif signal_confidence < 0.4:
                suitability -= 0.15  # Low confidence penalty
            
            # Expected return adjustment
            expected_return = getattr(signal, 'expected_return', 0.0)
            if expected_return > 0.1:  # >10% expected return
                suitability += 0.1
            elif expected_return < 0.02:  # <2% expected return
                suitability -= 0.1
            
            # Risk adjustment
            risk_score = getattr(signal, 'risk_score', 0.5)
            if risk_score < 0.3:  # Low risk
                suitability += 0.05
            elif risk_score > 0.7:  # High risk
                suitability -= 0.1
            
            return max(0.0, min(1.0, suitability))
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error in built-in suitability calculation: {e}")
            return 0.4  # Conservative default
    
    async def _coordinate_opportunities(self, opportunities: List[TradeOpportunity]) -> List[TradeOpportunity]:
        """Apply coordination logic to resolve conflicts and optimize allocation"""
        try:
            if len(opportunities) <= 1:
                return opportunities
            
            # Group opportunities by token to detect conflicts
            token_groups = defaultdict(list)
            for opp in opportunities:
                token_groups[opp.token_address].append(opp)
            
            coordinated_opportunities = []
            conflicts_resolved = 0
            
            # Process each token group
            for token_address, token_opportunities in token_groups.items():
                if len(token_opportunities) == 1:
                    # No conflict
                    coordinated_opportunities.extend(token_opportunities)
                else:
                    # Conflict resolution needed
                    resolved = await self._resolve_opportunity_conflict(token_address, token_opportunities)
                    coordinated_opportunities.extend(resolved)
                    conflicts_resolved += len(token_opportunities) - len(resolved)
            
            # Apply global resource constraints
            final_opportunities = await self._apply_resource_constraints(coordinated_opportunities)
            
            logger.info(f"[UNIFIED_TRADING] Coordination complete: {len(opportunities)} -> {len(final_opportunities)} "
                       f"(resolved {conflicts_resolved} conflicts)")
            
            return final_opportunities
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Opportunity coordination failed: {e}")
            return opportunities  # Return original list on error
    
    async def _resolve_opportunity_conflict(self, token_address: str, 
                                          conflicting_opportunities: List[TradeOpportunity]) -> List[TradeOpportunity]:
        """ENHANCED CONFLICT RESOLUTION - Advanced multi-algorithm approach"""
        try:
            if len(conflicting_opportunities) <= 1:
                return conflicting_opportunities
            
            logger.info(f"[UNIFIED_TRADING] Resolving conflict for token {token_address[:8]}... with {len(conflicting_opportunities)} strategies")
            
            resolved_opportunities = []
            
            # ENHANCED RESOLUTION ALGORITHMS
            if self.conflict_resolution == ConflictResolution.HIGHEST_CONFIDENCE:
                # Choose opportunity with highest confidence, break ties with risk-reward
                def confidence_key(opp):
                    return (opp.confidence, opp.calculate_risk_reward(), self.strategy_performance.get(opp.strategy_name, 0.5))
                
                best_opportunity = max(conflicting_opportunities, key=confidence_key)
                resolved_opportunities = [best_opportunity]
                
            elif self.conflict_resolution == ConflictResolution.BEST_RISK_REWARD:
                # Choose opportunity with best risk-reward ratio
                def risk_reward_key(opp):
                    base_ratio = opp.calculate_risk_reward()
                    # Adjust for strategy performance
                    performance_multiplier = 0.5 + (self.strategy_performance.get(opp.strategy_name, 0.5))
                    return base_ratio * performance_multiplier
                
                best_opportunity = max(conflicting_opportunities, key=risk_reward_key)
                resolved_opportunities = [best_opportunity]
                
            elif self.conflict_resolution == ConflictResolution.STRATEGY_PRIORITY:
                # Enhanced priority-based resolution with performance weighting
                priority_scores = []
                for opp in conflicting_opportunities:
                    base_priority = self.strategy_priorities.get(opp.strategy_name, 1)
                    performance_bonus = self.strategy_performance.get(opp.strategy_name, 0.5) * 0.5
                    confidence_bonus = opp.confidence * 0.3
                    
                    total_score = base_priority + performance_bonus + confidence_bonus
                    priority_scores.append((total_score, opp))
                
                priority_scores.sort(key=lambda x: x[0], reverse=True)
                resolved_opportunities = [priority_scores[0][1]]
                
            elif self.conflict_resolution == ConflictResolution.FIRST_COME_FIRST_SERVE:
                # Choose earliest opportunity with minimum quality threshold
                quality_filtered = [opp for opp in conflicting_opportunities if opp.confidence > 0.4]
                if quality_filtered:
                    earliest = min(quality_filtered, key=lambda x: x.timestamp)
                    resolved_opportunities = [earliest]
                else:
                    # Fall back to highest confidence if no quality opportunities
                    best_opportunity = max(conflicting_opportunities, key=lambda x: x.confidence)
                    resolved_opportunities = [best_opportunity]
            
            else:  # Default: HYBRID APPROACH
                # Comprehensive scoring combining all factors
                def hybrid_score(opp):
                    confidence_score = opp.confidence * 0.25
                    risk_reward_score = min(opp.calculate_risk_reward() / 5.0, 0.25)  # Cap at 0.25
                    performance_score = self.strategy_performance.get(opp.strategy_name, 0.5) * 0.20
                    priority_score = (self.strategy_priorities.get(opp.strategy_name, 1) / 5.0) * 0.15  # Normalize to 0-0.15
                    recency_score = max(0, 1.0 - (datetime.now() - opp.timestamp).total_seconds() / 300) * 0.15  # 5-min decay
                    
                    return confidence_score + risk_reward_score + performance_score + priority_score + recency_score
                
                best_opportunity = max(conflicting_opportunities, key=hybrid_score)
                resolved_opportunities = [best_opportunity]
            
            # PORTFOLIO DIVERSIFICATION CHECK
            if len(resolved_opportunities) > 0:
                winner = resolved_opportunities[0]
                
                # Check if this strategy is over-allocated
                strategy_exposure = self._calculate_strategy_exposure(winner.strategy_name)
                max_strategy_exposure = 0.4  # 40% max exposure per strategy
                
                if strategy_exposure > max_strategy_exposure:
                    # Find next best opportunity from different strategy
                    other_strategies = [opp for opp in conflicting_opportunities 
                                     if opp.strategy_name != winner.strategy_name]
                    
                    if other_strategies:
                        if self.conflict_resolution == ConflictResolution.HIGHEST_CONFIDENCE:
                            fallback = max(other_strategies, key=lambda x: x.confidence)
                        else:
                            fallback = max(other_strategies, key=hybrid_score)
                        
                        resolved_opportunities = [fallback]
                        logger.info(f"[UNIFIED_TRADING] Switched to {fallback.strategy_name} due to diversification limits")
            
            # Log detailed conflict resolution
            winner = resolved_opportunities[0] if resolved_opportunities else None
            losing_strategies = [opp.strategy_name for opp in conflicting_opportunities 
                               if winner and opp.strategy_name != winner.strategy_name]
            
            # Calculate resolution metrics for logging
            if winner:
                winner_metrics = {
                    "confidence": winner.confidence,
                    "risk_reward_ratio": winner.calculate_risk_reward(),
                    "priority": self.strategy_priorities.get(winner.strategy_name, 1),
                    "performance_score": self.strategy_performance.get(winner.strategy_name, 0.5)
                }
            else:
                winner_metrics = {}
            
            coordination_event = CoordinationEvent(
                timestamp=datetime.now(),
                event_type="advanced_conflict_resolution",
                involved_strategies=[opp.strategy_name for opp in conflicting_opportunities],
                token_address=token_address,
                resolution=self.conflict_resolution.value,
                outcome=f"winner_{winner.strategy_name if winner else 'none'}",
                details={
                    "conflicting_strategies": len(conflicting_opportunities),
                    "winner": winner.strategy_name if winner else None,
                    "winner_metrics": winner_metrics,
                    "losing_strategies": losing_strategies,
                    "resolution_method": self.conflict_resolution.value,
                    "resolution_quality": "high" if winner and winner.confidence > 0.7 else "medium"
                }
            )
            
            self.coordination_events.append(coordination_event)
            await self._log_coordination_event(coordination_event)
            
            logger.info(f"[UNIFIED_TRADING] Conflict resolved: {winner.strategy_name if winner else 'none'} selected for {token_address[:8]}...")
            
            return resolved_opportunities
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error resolving opportunity conflict: {e}")
            # Intelligent fallback: choose highest performing strategy
            try:
                fallback = max(conflicting_opportunities, 
                              key=lambda x: self.strategy_performance.get(x.strategy_name, 0.5))
                return [fallback]
            except:
                return conflicting_opportunities[:1]  # Return first opportunity as last resort
    
    async def _apply_resource_constraints(self, opportunities: List[TradeOpportunity]) -> List[TradeOpportunity]:
        """ENHANCED RESOURCE CONSTRAINTS - Advanced allocation with quality prioritization"""
        try:
            if not opportunities:
                return []
            
            logger.info(f"[UNIFIED_TRADING] Applying resource constraints to {len(opportunities)} opportunities")
            
            # MULTI-TIER SORTING: Quality, then Priority, then Performance
            def comprehensive_sort_key(opp):
                # Tier 1: Signal quality (confidence + expected return)
                quality_score = (opp.confidence * 0.6) + (opp.expected_return * 0.4)
                
                # Tier 2: Priority score
                priority_score = opp.priority_score
                
                # Tier 3: Strategy performance
                strategy_performance = self.strategy_performance.get(opp.strategy_name, 0.5)
                
                # Tier 4: Risk-adjusted return
                risk_adjusted_return = opp.expected_return / max(opp.risk_score, 0.1)
                
                return (quality_score, priority_score, strategy_performance, risk_adjusted_return)
            
            sorted_opportunities = sorted(opportunities, key=comprehensive_sort_key, reverse=True)
            
            final_opportunities = []
            total_capital_required = Decimal('0')
            strategy_trade_counts = defaultdict(int)
            strategy_capital_usage = defaultdict(lambda: Decimal('0'))
            
            # PROGRESSIVE FILTERING WITH DETAILED LOGGING
            filtered_stats = {
                'total_input': len(sorted_opportunities),
                'passed_quality': 0,
                'passed_limits': 0,
                'passed_capital': 0,
                'passed_exposure': 0,
                'passed_risk': 0,
                'final_approved': 0
            }
            
            for i, opportunity in enumerate(sorted_opportunities):
                
                # QUALITY FILTER - Minimum thresholds
                if opportunity.confidence < 0.3 or opportunity.expected_return < 0.01:  # Below minimum quality
                    logger.debug(f"[UNIFIED_TRADING] Opportunity {i+1} filtered: Low quality (conf: {opportunity.confidence:.3f}, ret: {opportunity.expected_return:.3f})")
                    continue
                filtered_stats['passed_quality'] += 1
                
                # GLOBAL TRADE LIMITS
                if len(final_opportunities) >= self.max_concurrent_trades:
                    logger.info(f"[UNIFIED_TRADING] Global trade limit reached: {self.max_concurrent_trades}")
                    break
                
                # PER-STRATEGY LIMITS
                if strategy_trade_counts[opportunity.strategy_name] >= self.max_trades_per_strategy:
                    logger.debug(f"[UNIFIED_TRADING] Strategy {opportunity.strategy_name} limit reached: {self.max_trades_per_strategy}")
                    continue
                filtered_stats['passed_limits'] += 1
                
                # CAPITAL AVAILABILITY CHECK
                wallet = self.strategy_wallets.get(opportunity.strategy_name)
                if not wallet:
                    logger.warning(f"[UNIFIED_TRADING] No wallet found for strategy {opportunity.strategy_name}")
                    continue
                
                if not wallet.is_healthy():
                    logger.debug(f"[UNIFIED_TRADING] Wallet {opportunity.strategy_name} is not healthy")
                    continue
                
                available_capital = wallet.current_balance - strategy_capital_usage[opportunity.strategy_name]
                if opportunity.position_size > available_capital:
                    logger.debug(f"[UNIFIED_TRADING] Insufficient capital for {opportunity.strategy_name}: need {opportunity.position_size}, have {available_capital}")
                    continue
                filtered_stats['passed_capital'] += 1
                
                # TOTAL EXPOSURE CHECK (Enhanced)
                total_capital_available = sum(float(w.current_balance) for w in self.strategy_wallets.values())
                exposure_ratio = float(total_capital_required + opportunity.position_size) / total_capital_available
                
                if exposure_ratio > float(self.max_total_exposure):
                    logger.debug(f"[UNIFIED_TRADING] Total exposure limit exceeded: {exposure_ratio:.2%} > {float(self.max_total_exposure):.2%}")
                    continue
                filtered_stats['passed_exposure'] += 1
                
                # STRATEGY DIVERSIFICATION CHECK
                strategy_exposure = strategy_capital_usage[opportunity.strategy_name] + opportunity.position_size
                max_strategy_capital = wallet.allocated_capital * Decimal('0.6')  # Max 60% of allocated capital per batch
                
                if strategy_exposure > max_strategy_capital:
                    logger.debug(f"[UNIFIED_TRADING] Strategy {opportunity.strategy_name} would exceed batch limit")
                    continue
                
                # ENHANCED RISK VALIDATION
                risk_approved = True
                if self.risk_manager:
                    try:
                        risk_check = await self.risk_manager.validate_trade(
                            opportunity.token_address,
                            opportunity.strategy_name,
                            float(opportunity.position_size),
                            float(wallet.current_balance),
                            opportunity.entry_price
                        )
                        
                        if isinstance(risk_check, tuple):
                            risk_approved = risk_check[0]
                            if not risk_approved:
                                logger.debug(f"[UNIFIED_TRADING] Risk manager rejected trade: {risk_check[1] if len(risk_check) > 1 else 'No reason provided'}")
                        else:
                            risk_approved = bool(risk_check)
                    except Exception as e:
                        logger.warning(f"[UNIFIED_TRADING] Risk manager validation failed: {e}")
                        risk_approved = False  # Fail safe
                
                if not risk_approved:
                    continue
                filtered_stats['passed_risk'] += 1
                
                # PORTFOLIO CORRELATION CHECK (Advanced)
                correlation_approved = await self._check_portfolio_correlation(opportunity, final_opportunities)
                if not correlation_approved:
                    logger.debug(f"[UNIFIED_TRADING] Portfolio correlation limit exceeded for {opportunity.token_address[:8]}...")
                    continue
                
                # ALL CHECKS PASSED - APPROVE OPPORTUNITY
                final_opportunities.append(opportunity)
                total_capital_required += opportunity.position_size
                strategy_trade_counts[opportunity.strategy_name] += 1
                strategy_capital_usage[opportunity.strategy_name] += opportunity.position_size
                filtered_stats['final_approved'] += 1
                
                logger.debug(f"[UNIFIED_TRADING] Approved opportunity {i+1}/{len(sorted_opportunities)}: {opportunity.strategy_name} - {opportunity.token_address[:8]}...")
            
            # LOG COMPREHENSIVE FILTERING RESULTS
            success_rate = (filtered_stats['final_approved'] / filtered_stats['total_input']) * 100 if filtered_stats['total_input'] > 0 else 0
            
            logger.info(f"[UNIFIED_TRADING] Resource filtering complete: {filtered_stats['total_input']} -> {filtered_stats['final_approved']} ({success_rate:.1f}% approval rate)")
            logger.info(f"[UNIFIED_TRADING] Filter breakdown: Quality({filtered_stats['passed_quality']}) -> Limits({filtered_stats['passed_limits']}) -> Capital({filtered_stats['passed_capital']}) -> Exposure({filtered_stats['passed_exposure']}) -> Risk({filtered_stats['passed_risk']}) -> Final({filtered_stats['final_approved']})")
            
            if final_opportunities:
                avg_confidence = statistics.mean([opp.confidence for opp in final_opportunities])
                avg_expected_return = statistics.mean([opp.expected_return for opp in final_opportunities])
                strategy_distribution = {name: count for name, count in strategy_trade_counts.items() if count > 0}
                
                logger.info(f"[UNIFIED_TRADING] Approved batch quality: Avg confidence {avg_confidence:.3f}, Avg return {avg_expected_return:.3f}, Strategy distribution: {strategy_distribution}")
            
            return final_opportunities
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error applying resource constraints: {e}")
            # Return top 3 opportunities as emergency fallback
            try:
                emergency_fallback = sorted(opportunities, key=lambda x: x.confidence, reverse=True)[:3]
                logger.warning(f"[UNIFIED_TRADING] Using emergency fallback: {len(emergency_fallback)} opportunities")
                return emergency_fallback
            except:
                return []
    
    async def _check_portfolio_correlation(self, new_opportunity: TradeOpportunity, existing_opportunities: List[TradeOpportunity]) -> bool:
        """Check if adding this opportunity would create excessive portfolio correlation"""
        try:
            if not existing_opportunities:
                return True  # No correlation issues with empty portfolio
            
            # Check for same token (100% correlation)
            same_token_count = sum(1 for opp in existing_opportunities if opp.token_address == new_opportunity.token_address)
            if same_token_count > 0:
                return False  # Don't allow duplicate tokens in same batch
            
            # Check strategy concentration
            same_strategy_count = sum(1 for opp in existing_opportunities if opp.strategy_name == new_opportunity.strategy_name)
            max_per_strategy = max(2, len(existing_opportunities) // 2)  # At most half from same strategy, minimum 2
            
            if same_strategy_count >= max_per_strategy:
                return False
            
            # All correlation checks passed
            return True
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error checking portfolio correlation: {e}")
            return True  # Default to allowing if check fails
    
    async def _execute_coordinated_trades(self, opportunities: List[TradeOpportunity]) -> Dict[str, Any]:
        """Execute coordinated trades across multiple strategies"""
        try:
            if not opportunities:
                return {"successful": [], "failed": [], "total": 0}
            
            execution_results = {
                "successful": [],
                "failed": [],
                "total": len(opportunities)
            }
            
            # Execute trades concurrently by strategy
            strategy_groups = defaultdict(list)
            for opp in opportunities:
                strategy_groups[opp.strategy_name].append(opp)
            
            # Launch concurrent executions
            execution_tasks = []
            for strategy_name, strategy_opportunities in strategy_groups.items():
                task = asyncio.create_task(
                    self._execute_strategy_trades(strategy_name, strategy_opportunities)
                )
                execution_tasks.append(task)
            
            # Wait for all executions to complete
            strategy_results = await asyncio.gather(*execution_tasks, return_exceptions=True)
            
            # Aggregate results
            for result in strategy_results:
                if isinstance(result, Exception):
                    logger.error(f"[UNIFIED_TRADING] Strategy execution failed: {result}")
                    continue
                
                execution_results["successful"].extend(result.get("successful", []))
                execution_results["failed"].extend(result.get("failed", []))
            
            # Update wallet balances and performance
            await self._update_post_execution_metrics(execution_results)
            
            logger.info(f"[UNIFIED_TRADING] Trade execution complete: {len(execution_results['successful'])}/{len(opportunities)} successful")
            
            return execution_results
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Coordinated trade execution failed: {e}")
            return {"successful": [], "failed": [], "total": len(opportunities), "error": str(e)}
    
    async def _execute_strategy_trades(self, strategy_name: str, 
                                     opportunities: List[TradeOpportunity]) -> Dict[str, Any]:
        """Execute trades for a specific strategy"""
        try:
            strategy = self.strategies.get(strategy_name)
            wallet = self.strategy_wallets.get(strategy_name)
            
            if not strategy or not wallet:
                return {"successful": [], "failed": opportunities}
            
            results = {"successful": [], "failed": []}
            
            for opportunity in opportunities:
                try:
                    # Validate wallet balance
                    if opportunity.position_size > wallet.current_balance:
                        results["failed"].append(opportunity)
                        continue
                    
                    # Execute trade through strategy
                    trade_result = await strategy.execute_trade(
                        token_address=opportunity.token_address,
                        signal=opportunity.signal,
                        position_size=float(opportunity.position_size)
                    )
                    
                    if trade_result and trade_result.get("success", False):
                        # Update wallet balance
                        wallet.current_balance -= opportunity.position_size
                        wallet.total_trades += 1
                        
                        opportunity.execution_status = "executed"
                        results["successful"].append(opportunity)
                        
                        # Record trade in risk manager
                        if self.risk_manager:
                            self.risk_manager.record_trade(
                                opportunity.token_address,
                                strategy_name,
                                True,
                                None,  # P&L calculated later
                                opportunity.position_size
                            )
                        
                    else:
                        opportunity.execution_status = "failed"
                        results["failed"].append(opportunity)
                
                except Exception as e:
                    logger.error(f"[UNIFIED_TRADING] Trade execution error for {strategy_name}: {e}")
                    opportunity.execution_status = "error"
                    results["failed"].append(opportunity)
            
            return results
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Strategy trade execution failed for {strategy_name}: {e}")
            return {"successful": [], "failed": opportunities}
    
    async def rebalance_capital(self, reason: str = "manual") -> Dict[str, Any]:
        """Rebalance capital across strategy wallets based on performance"""
        try:
            start_time = time.time()
            
            # Calculate new allocations based on performance
            new_allocations = await self._calculate_optimal_allocations()
            
            # Get current allocations
            current_allocations = {
                name: float(wallet.current_balance) 
                for name, wallet in self.strategy_wallets.items()
            }
            
            # Calculate capital movements needed
            capital_movements = self._calculate_capital_movements(current_allocations, new_allocations)
            
            # Execute capital rebalancing
            rebalance_results = await self._execute_capital_rebalancing(capital_movements)
            
            # Log rebalancing event
            rebalance_event = {
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'old_allocations': current_allocations,
                'new_allocations': new_allocations,
                'capital_moved': sum(abs(movement) for movement in capital_movements.values()),
                'execution_time_ms': (time.time() - start_time) * 1000,
                'success': rebalance_results.get("success", False)
            }
            
            await self._log_rebalancing_event(rebalance_event)
            
            self.last_rebalance = datetime.now()
            
            logger.info(f"[UNIFIED_TRADING] Capital rebalancing complete: {reason}")
            
            return {
                "success": rebalance_results.get("success", False),
                "old_allocations": current_allocations,
                "new_allocations": new_allocations,
                "capital_moved": rebalance_event['capital_moved'],
                "execution_time_ms": rebalance_event['execution_time_ms']
            }
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Capital rebalancing failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Health monitoring and maintenance methods
    async def _health_monitor_loop(self):
        """Continuous health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.now()
                if current_time - self.last_health_check >= self.health_check_interval:
                    await self._perform_health_check()
                    self.last_health_check = current_time
                
            except Exception as e:
                logger.error(f"[UNIFIED_TRADING] Health monitor error: {e}")
    
    async def _rebalancing_loop(self):
        """Automatic rebalancing loop"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                current_time = datetime.now()
                if current_time - self.last_rebalance >= self.rebalance_interval:
                    # Check if rebalancing is needed
                    if await self._needs_rebalancing():
                        await self.rebalance_capital("scheduled")
                
            except Exception as e:
                logger.error(f"[UNIFIED_TRADING] Rebalancing loop error: {e}")
    
    async def _coordination_loop(self):
        """Process pending opportunities coordination"""
        while True:
            try:
                await asyncio.sleep(5)  # Process every 5 seconds
                
                if self.pending_opportunities:
                    # Process pending opportunities
                    opportunities_to_process = self.pending_opportunities[:10]  # Process in batches
                    self.pending_opportunities = self.pending_opportunities[10:]
                    
                    coordinated = await self._coordinate_opportunities(opportunities_to_process)
                    if coordinated:
                        await self._execute_coordinated_trades(coordinated)
                
            except Exception as e:
                logger.error(f"[UNIFIED_TRADING] Coordination loop error: {e}")
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive trading manager status"""
        try:
            # Wallet status
            wallet_status = {}
            for name, wallet in self.strategy_wallets.items():
                wallet_status[name] = {
                    'wallet_id': wallet.wallet_id,
                    'allocated_capital': float(wallet.allocated_capital),
                    'current_balance': float(wallet.current_balance),
                    'total_pnl': float(wallet.total_pnl),
                    'roi': wallet.calculate_roi(),
                    'win_rate': wallet.calculate_win_rate(),
                    'total_trades': wallet.total_trades,
                    'status': wallet.status.value,
                    'health_score': wallet.health_score,
                    'last_trade': wallet.last_trade.isoformat() if wallet.last_trade else None
                }
            
            # Strategy performance
            strategy_status = {}
            for name, strategy in self.strategies.items():
                strategy_status[name] = {
                    'status': strategy.status.value if hasattr(strategy, 'status') else 'active',
                    'performance_score': self.strategy_performance.get(name, 0.5),
                    'priority': self.strategy_priorities.get(name, 1),
                    'active_positions': len(getattr(strategy, 'positions', [])),
                    'max_positions': strategy.config.max_positions
                }
            
            # Coordination metrics
            recent_events = [event.to_dict() for event in list(self.coordination_events)[-10:]]
            
            return {
                "timestamp": datetime.now().isoformat(),
                "status": {
                    "emergency_stop": self.emergency_stop,
                    "liquidation_mode": self.liquidation_mode,
                    "coordination_mode": self.coordination_mode.value,
                    "conflict_resolution": self.conflict_resolution.value
                },
                "performance": {
                    "total_trades_today": self.total_trades_today,
                    "coordination_success_rate": self.coordination_success_rate,
                    "successful_coordinations": self.successful_coordinations,
                    "failed_coordinations": self.failed_coordinations
                },
                "wallets": wallet_status,
                "strategies": strategy_status,
                "recent_coordination_events": recent_events,
                "limits": {
                    "max_trades_per_day": self.max_trades_per_day,
                    "max_concurrent_trades": self.max_concurrent_trades,
                    "max_total_exposure": float(self.max_total_exposure)
                }
            }
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error getting comprehensive status: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    # Helper methods for internal operations
    def _get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get strategy-specific configuration"""
        base_config = {
            'min_signal_strength': 0.6,
            'position_timeout_minutes': 180,
            'stop_loss_percentage': 0.15,
            'take_profit_percentage': 0.5
        }
        
        # Strategy-specific configurations
        strategy_configs = {
            'momentum': {
                'momentum_threshold': 0.15,
                'volume_threshold_sol': 1000.0,
                'min_age_hours': 24
            },
            'mean_reversion': {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'bollinger_std_dev': 2.0
            },
            'grid_trading': {
                'grid_levels': 10,
                'grid_spacing': 0.02,
                'rebalance_threshold': 0.05
            },
            'arbitrage': {
                'min_profit_bps': 50,
                'max_execution_time_ms': 5000,
                'slippage_tolerance': 0.005
            }
        }
        
        base_config.update(strategy_configs.get(strategy_name, {}))
        return base_config
    
    async def _store_wallet_config(self, wallet: StrategyWallet):
        """Store wallet configuration in database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO strategy_wallets
                    (wallet_id, strategy_name, strategy_type, allocated_capital, current_balance,
                     total_pnl, total_trades, winning_trades, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    wallet.wallet_id,
                    wallet.strategy_name,
                    wallet.strategy_type.value,
                    float(wallet.allocated_capital),
                    float(wallet.current_balance),
                    float(wallet.total_pnl),
                    wallet.total_trades,
                    wallet.winning_trades,
                    wallet.status.value,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                await db.commit()
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error storing wallet config: {e}")
    
    # Additional helper methods would be implemented here...
    # (Including _calculate_optimal_allocations, _execute_capital_rebalancing, etc.)
    
    async def shutdown(self):
        """Shutdown trading manager and all components"""
        try:
            logger.info("[UNIFIED_TRADING] Shutting down Unified Trading Manager...")
            
            # Stop all strategies
            for strategy in self.strategies.values():
                if hasattr(strategy, 'shutdown'):
                    await strategy.shutdown()
            
            # Save final state
            await self._save_final_state()
            
            logger.info("[UNIFIED_TRADING] Unified Trading Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error during shutdown: {e}")
    
    # CRITICAL IMPLEMENTATION METHODS - Day 13 Fixes
    
    async def _load_historical_data(self):
        """Load historical trading and performance data"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Load strategy performance history
                cursor = await db.execute(
                    "SELECT strategy_name, performance_score FROM strategy_performance ORDER BY timestamp DESC LIMIT 100"
                )
                rows = await cursor.fetchall()
                
                performance_data = defaultdict(list)
                for row in rows:
                    performance_data[row[0]].append(row[1])
                
                # Calculate moving averages for performance
                for strategy_name, scores in performance_data.items():
                    if scores:
                        self.strategy_performance[strategy_name] = statistics.mean(scores[-10:])  # Last 10 data points
                    else:
                        self.strategy_performance[strategy_name] = 0.5  # Neutral starting point
                
                logger.info(f"[UNIFIED_TRADING] Historical data loaded for {len(self.strategy_performance)} strategies")
                
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error loading historical data: {e}")
            # Set default performance scores
            for strategy_name in self.strategy_wallets.keys():
                self.strategy_performance[strategy_name] = 0.5
    
    async def _perform_health_check(self):
        """Perform comprehensive system health check"""
        try:
            health_issues = []
            
            # Check strategy wallet health
            for name, wallet in self.strategy_wallets.items():
                if not wallet.is_healthy():
                    health_issues.append(f"Wallet {name} is unhealthy: {wallet.status.value}")
                    
                    # Auto-recovery for minor issues
                    if wallet.error_count >= 5:
                        wallet.status = WalletStatus.MAINTENANCE
                        logger.warning(f"[UNIFIED_TRADING] Wallet {name} moved to maintenance mode")
            
            # Check overall coordination performance
            if self.coordination_success_rate < 0.7:  # Less than 70% success
                health_issues.append(f"Low coordination success rate: {self.coordination_success_rate:.2%}")
            
            # Check emergency conditions
            total_daily_loss = sum(float(wallet.total_pnl) for wallet in self.strategy_wallets.values() if wallet.total_pnl < 0)
            if total_daily_loss < -float(self.emergency_liquidation_threshold) * 1000:  # Assuming $1000 base capital
                self.emergency_stop = True
                health_issues.append(f"Emergency stop triggered: Daily loss ${abs(total_daily_loss):.2f}")
            
            # Generate health alerts
            if health_issues:
                self.alerts.extend([
                    {
                        "timestamp": datetime.now().isoformat(),
                        "level": "high" if "emergency" in issue.lower() else "medium",
                        "message": issue
                    } for issue in health_issues
                ])
                
                logger.warning(f"[UNIFIED_TRADING] Health check found {len(health_issues)} issues")
            else:
                logger.info("[UNIFIED_TRADING] System health check: All systems operational")
                
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Health check error: {e}")
    
    async def _needs_rebalancing(self) -> bool:
        """Check if capital rebalancing is needed"""
        try:
            for wallet in self.strategy_wallets.values():
                if wallet.needs_rebalancing(self.rebalance_threshold):
                    return True
            
            # Check performance-based rebalancing need
            performance_variance = statistics.pvariance(self.strategy_performance.values()) if len(self.strategy_performance) > 1 else 0
            if performance_variance > 0.1:  # High variance in performance
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error checking rebalancing need: {e}")
            return False
    
    async def _calculate_optimal_allocations(self) -> Dict[str, float]:
        """Calculate optimal capital allocations based on performance"""
        try:
            total_capital = sum(float(wallet.allocated_capital) for wallet in self.strategy_wallets.values())
            
            # Performance-based allocation
            performance_scores = {}
            total_performance = 0
            
            for strategy_name in self.strategy_wallets.keys():
                # Combine historical performance with recent success
                historical_performance = self.strategy_performance.get(strategy_name, 0.5)
                wallet = self.strategy_wallets[strategy_name]
                recent_performance = wallet.calculate_win_rate() if wallet.total_trades > 0 else 0.5
                
                # Weighted combination (70% historical, 30% recent)
                combined_score = (historical_performance * 0.7) + (recent_performance * 0.3)
                performance_scores[strategy_name] = max(0.1, combined_score)  # Minimum 10% allocation
                total_performance += performance_scores[strategy_name]
            
            # Normalize to allocate total capital
            optimal_allocations = {}
            for strategy_name, score in performance_scores.items():
                allocation_percentage = score / total_performance
                optimal_allocations[strategy_name] = total_capital * allocation_percentage
            
            logger.info(f"[UNIFIED_TRADING] Calculated optimal allocations based on performance")
            return optimal_allocations
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error calculating optimal allocations: {e}")
            # Return current allocations on error
            return {name: float(wallet.allocated_capital) for name, wallet in self.strategy_wallets.items()}
    
    def _calculate_strategy_exposure(self, strategy_name: str) -> float:
        """Calculate current exposure percentage for a strategy"""
        try:
            total_capital = sum(float(wallet.allocated_capital) for wallet in self.strategy_wallets.values())
            if total_capital == 0:
                return 0.0
            
            strategy_capital = float(self.strategy_wallets[strategy_name].allocated_capital)
            return strategy_capital / total_capital
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error calculating strategy exposure: {e}")
            return 0.0
    
    def _calculate_capital_movements(self, current: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
        """Calculate required capital movements for rebalancing"""
        movements = {}
        for strategy_name in current.keys():
            current_amount = current.get(strategy_name, 0)
            target_amount = new.get(strategy_name, 0)
            movement = target_amount - current_amount
            
            # Only move significant amounts (>$10)
            if abs(movement) > 10:
                movements[strategy_name] = movement
        
        return movements
    
    async def _execute_capital_rebalancing(self, movements: Dict[str, float]) -> Dict[str, Any]:
        """Execute capital rebalancing between wallets"""
        try:
            successful_moves = 0
            total_moves = len(movements)
            
            for strategy_name, movement_amount in movements.items():
                wallet = self.strategy_wallets.get(strategy_name)
                if not wallet:
                    continue
                
                if movement_amount > 0:  # Adding capital
                    wallet.current_balance += Decimal(str(movement_amount))
                    wallet.allocated_capital += Decimal(str(movement_amount))
                elif movement_amount < 0:  # Removing capital
                    removal_amount = Decimal(str(abs(movement_amount)))
                    if wallet.current_balance >= removal_amount:
                        wallet.current_balance -= removal_amount
                        wallet.allocated_capital -= removal_amount
                    else:
                        # Can't remove more than available
                        continue
                
                successful_moves += 1
                
                # Update wallet in database
                await self._store_wallet_config(wallet)
            
            success_rate = successful_moves / total_moves if total_moves > 0 else 1.0
            
            logger.info(f"[UNIFIED_TRADING] Capital rebalancing: {successful_moves}/{total_moves} moves successful")
            
            return {
                "success": success_rate > 0.8,  # 80% success rate required
                "successful_moves": successful_moves,
                "total_moves": total_moves,
                "success_rate": success_rate
            }
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Capital rebalancing execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _update_post_execution_metrics(self, results: Dict[str, Any]):
        """Update performance metrics after trade execution"""
        try:
            successful_trades = results.get("successful", [])
            failed_trades = results.get("failed", [])
            
            # Update strategy performance scores
            strategy_successes = defaultdict(int)
            strategy_totals = defaultdict(int)
            
            for trade in successful_trades + failed_trades:
                strategy_name = trade.strategy_name
                strategy_totals[strategy_name] += 1
                if trade in successful_trades:
                    strategy_successes[strategy_name] += 1
            
            # Update performance scores (exponential moving average)
            alpha = 0.1  # Learning rate
            for strategy_name, total in strategy_totals.items():
                success_rate = strategy_successes[strategy_name] / total
                current_performance = self.strategy_performance.get(strategy_name, 0.5)
                
                # EMA update
                new_performance = (alpha * success_rate) + ((1 - alpha) * current_performance)
                self.strategy_performance[strategy_name] = max(0.1, min(0.9, new_performance))  # Bound between 10%-90%
            
            # Update wallet metrics
            for trade in successful_trades:
                wallet = self.strategy_wallets.get(trade.strategy_name)
                if wallet:
                    wallet.winning_trades += 1
                    # P&L will be updated when positions are closed
            
            # Store updated performance in database
            await self._store_performance_metrics()
            
            logger.info(f"[UNIFIED_TRADING] Post-execution metrics updated for {len(strategy_totals)} strategies")
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error updating post-execution metrics: {e}")
    
    def _update_coordination_metrics(self, processed: int, executed: int):
        """Update coordination performance metrics"""
        try:
            self.total_trades_today += executed
            
            if processed > 0:
                coordination_success = executed / processed
                
                if coordination_success > 0.5:
                    self.successful_coordinations += 1
                else:
                    self.failed_coordinations += 1
                
                total_coordinations = self.successful_coordinations + self.failed_coordinations
                if total_coordinations > 0:
                    self.coordination_success_rate = self.successful_coordinations / total_coordinations
            
            logger.debug(f"[UNIFIED_TRADING] Coordination metrics updated: {self.coordination_success_rate:.2%} success rate")
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error updating coordination metrics: {e}")
    
    async def _log_coordination_event(self, event: CoordinationEvent):
        """Log coordination event to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO coordination_events
                    (timestamp, event_type, involved_strategies, token_address, resolution, outcome, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.timestamp.isoformat(),
                    event.event_type,
                    ','.join(event.involved_strategies),
                    event.token_address,
                    event.resolution,
                    event.outcome,
                    json.dumps(event.details)
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error logging coordination event: {e}")
    
    async def _log_rebalancing_event(self, event: Dict[str, Any]):
        """Log rebalancing event to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO rebalancing_events
                    (timestamp, reason, old_allocations, new_allocations, capital_moved, success, execution_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    event['timestamp'],
                    event['reason'],
                    json.dumps(event['old_allocations']),
                    json.dumps(event['new_allocations']),
                    event['capital_moved'],
                    event['success'],
                    event.get('execution_time_ms', 0)
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error logging rebalancing event: {e}")
    
    async def _store_performance_metrics(self):
        """Store current performance metrics in database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                for strategy_name, performance in self.strategy_performance.items():
                    wallet = self.strategy_wallets.get(strategy_name)
                    if wallet:
                        await db.execute("""
                            INSERT INTO strategy_performance
                            (timestamp, strategy_name, performance_score, sharpe_ratio, win_rate, total_return, max_drawdown)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            datetime.now().isoformat(),
                            strategy_name,
                            performance,
                            wallet.sharpe_ratio,
                            wallet.calculate_win_rate(),
                            wallet.calculate_roi(),
                            wallet.max_drawdown
                        ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error storing performance metrics: {e}")
    
    async def _save_final_state(self):
        """Save final state before shutdown"""
        try:
            # Save all wallet configurations
            for wallet in self.strategy_wallets.values():
                await self._store_wallet_config(wallet)
            
            # Save final performance metrics
            await self._store_performance_metrics()
            
            # Save coordination summary
            final_summary = {
                "shutdown_timestamp": datetime.now().isoformat(),
                "total_trades_today": self.total_trades_today,
                "coordination_success_rate": self.coordination_success_rate,
                "successful_coordinations": self.successful_coordinations,
                "failed_coordinations": self.failed_coordinations,
                "strategy_performance": dict(self.strategy_performance),
                "wallet_balances": {name: float(wallet.current_balance) for name, wallet in self.strategy_wallets.items()}
            }
            
            with open("logs/final_trading_state.json", "w") as f:
                json.dump(final_summary, f, indent=2)
            
            logger.info("[UNIFIED_TRADING] Final state saved successfully")
            
        except Exception as e:
            logger.error(f"[UNIFIED_TRADING] Error saving final state: {e}")

# Global unified trading manager instance
_global_unified_trading_manager: Optional[UnifiedTradingManager] = None

def get_unified_trading_manager(settings=None, risk_manager=None, portfolio_manager=None, master_coordinator=None) -> UnifiedTradingManager:
    """Get global unified trading manager instance with master coordinator support"""
    global _global_unified_trading_manager
    if _global_unified_trading_manager is None:
        _global_unified_trading_manager = UnifiedTradingManager(settings, risk_manager, portfolio_manager, master_coordinator=master_coordinator)
    return _global_unified_trading_manager

# Compatibility aliases for existing code
def get_trading_manager(settings=None, risk_manager=None, portfolio_manager=None) -> UnifiedTradingManager:
    """Compatibility alias for existing code"""
    return get_unified_trading_manager(settings, risk_manager, portfolio_manager)

class TradingManager(UnifiedTradingManager):
    """Compatibility alias for existing code"""
    pass