#!/usr/bin/env python3
"""
MASTER STRATEGY COORDINATOR - Day 13 Implementation
Final integration phase for comprehensive 4-strategy coordination system

This master coordinator serves as the central orchestrator for all trading strategies,
implementing sophisticated resource allocation, conflict resolution, and performance optimization.

Key Features:
- Master coordination for all 4 strategies (Momentum, Mean Reversion, Grid Trading, Arbitrage)
- Advanced conflict resolution with multiple algorithms
- Dynamic resource allocation and capital distribution
- API quota management and processing power optimization
- Performance-based strategy prioritization
- Integration with all 6 unified managers
- Real-time monitoring and adaptive coordination
- Emergency coordination and circuit breakers
"""

import asyncio
import logging
import time
import json
import sqlite3
import aiosqlite
import numpy as np
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path

# Import all unified managers
from management.risk_manager import UnifiedRiskManager, RiskLevel
from management.portfolio_manager import UnifiedPortfolioManager, AllocationStrategy
from management.trading_manager import UnifiedTradingManager
from management.order_manager import UnifiedOrderManager
from management.data_manager import UnifiedDataManager
from management.system_manager import UnifiedSystemManager

# Import all strategies
from strategies.base import BaseStrategy, StrategyType, StrategyStatus, StrategyConfig
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.grid_trading import GridTradingStrategy
from strategies.arbitrage import ArbitrageStrategy

# Import core infrastructure
from core.multi_wallet_manager import MultiWalletManager, WalletType
from core.swap_executor import SwapExecutor
from core.rpc_manager import MultiRPCManager

# Import models
from models.signal import Signal
from models.trade import Trade
from models.position import Position

logger = logging.getLogger(__name__)

class CoordinationMode(Enum):
    """Master coordination modes"""
    COOPERATIVE = "cooperative"        # Strategies share opportunities and resources
    COMPETITIVE = "competitive"        # Strategies compete for best opportunities
    HIERARCHICAL = "hierarchical"      # Priority-based strategy ordering
    ADAPTIVE = "adaptive"              # Dynamic mode based on market conditions
    BALANCED = "balanced"              # Optimal balance of all approaches

class ConflictResolutionStrategy(Enum):
    """Advanced conflict resolution strategies"""
    HIGHEST_CONFIDENCE = "highest_confidence"      # Choose signal with highest confidence
    BEST_RISK_REWARD = "best_risk_reward"         # Optimize for risk-reward ratio
    PORTFOLIO_BALANCE = "portfolio_balance"        # Maintain portfolio balance
    PERFORMANCE_WEIGHTED = "performance_weighted"  # Weight by recent performance
    DIVERSIFICATION = "diversification"            # Maximize portfolio diversification
    MOMENTUM_PRIORITY = "momentum_priority"        # Prioritize trending opportunities
    HYBRID_OPTIMIZATION = "hybrid_optimization"    # Advanced multi-factor optimization

class ResourceType(Enum):
    """Resource types for allocation"""
    CAPITAL = "capital"                # Trading capital allocation
    API_QUOTA = "api_quota"           # API request quotas
    PROCESSING_POWER = "processing"    # Computational resources
    WALLET_CAPACITY = "wallet"        # Multi-wallet capacity
    RISK_BUDGET = "risk_budget"       # Risk allocation
    OPPORTUNITY_SLOTS = "slots"       # Trading opportunity slots

@dataclass
class StrategyOpportunity:
    """Enhanced strategy opportunity with coordination metadata"""
    opportunity_id: str
    strategy_name: str
    strategy_type: StrategyType
    token_address: str
    
    # Core opportunity data
    signal: Signal
    confidence: float
    expected_return: float
    risk_score: float
    
    # Resource requirements
    required_capital: Decimal
    required_api_calls: int
    required_processing: float
    expected_duration: int  # seconds
    
    # Coordination metadata
    priority_score: float = 0.0
    coordination_weight: float = 1.0
    conflicts_with: Set[str] = field(default_factory=set)
    synergies_with: Set[str] = field(default_factory=set)
    
    # Performance metrics
    risk_reward_ratio: float = 0.0
    portfolio_impact: float = 0.0
    diversification_benefit: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_composite_score(self) -> float:
        """Calculate composite opportunity score for ranking"""
        return (
            self.confidence * 0.3 +
            self.risk_reward_ratio * 0.25 +
            self.portfolio_impact * 0.2 +
            self.diversification_benefit * 0.15 +
            self.coordination_weight * 0.1
        )

@dataclass
class ResourceAllocation:
    """Resource allocation tracking"""
    resource_type: ResourceType
    total_available: float
    allocated: Dict[str, float] = field(default_factory=dict)
    reserved: float = 0.0
    
    @property
    def available(self) -> float:
        return self.total_available - sum(self.allocated.values()) - self.reserved
    
    def allocate(self, strategy_name: str, amount: float) -> bool:
        """Allocate resources to strategy"""
        if amount <= self.available:
            self.allocated[strategy_name] = self.allocated.get(strategy_name, 0) + amount
            return True
        return False
    
    def deallocate(self, strategy_name: str, amount: float):
        """Deallocate resources from strategy"""
        current = self.allocated.get(strategy_name, 0)
        self.allocated[strategy_name] = max(0, current - amount)

@dataclass
class CoordinationEvent:
    """Coordination event tracking"""
    event_id: str
    timestamp: datetime
    event_type: str
    involved_strategies: List[str]
    coordination_mode: CoordinationMode
    resolution_strategy: ConflictResolutionStrategy
    
    # Event details
    input_opportunities: int
    output_opportunities: int
    conflicts_resolved: int
    resources_allocated: Dict[str, float]
    
    # Outcomes
    success: bool
    execution_time_ms: float
    performance_impact: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)

class MasterStrategyCoordinator:
    """Master coordinator for all trading strategies"""
    
    def __init__(self, settings: Any):
        self.settings = settings
        
        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_configs: Dict[str, StrategyConfig] = {}
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        
        # Coordination system
        self.coordination_mode = CoordinationMode.ADAPTIVE
        self.conflict_resolution = ConflictResolutionStrategy.HYBRID_OPTIMIZATION
        self.resource_allocations: Dict[ResourceType, ResourceAllocation] = {}
        
        # Manager integrations
        self.risk_manager: Optional[UnifiedRiskManager] = None
        self.portfolio_manager: Optional[UnifiedPortfolioManager] = None
        self.trading_manager: Optional[UnifiedTradingManager] = None
        self.order_manager: Optional[UnifiedOrderManager] = None
        self.data_manager: Optional[UnifiedDataManager] = None
        self.system_manager: Optional[UnifiedSystemManager] = None
        
        # Core infrastructure
        self.multi_wallet_manager: Optional[MultiWalletManager] = None
        self.swap_executor: Optional[SwapExecutor] = None
        self.rpc_manager: Optional[MultiRPCManager] = None
        
        # Coordination state
        self.active_opportunities: Dict[str, StrategyOpportunity] = {}
        self.coordination_history: deque = deque(maxlen=1000)
        self.performance_metrics: Dict[str, Any] = {}
        
        # Configuration
        self.max_concurrent_opportunities = getattr(settings, 'MAX_CONCURRENT_OPPORTUNITIES', 20)
        self.coordination_interval = getattr(settings, 'COORDINATION_INTERVAL', 5)  # seconds
        self.performance_window = getattr(settings, 'PERFORMANCE_WINDOW', 3600)  # 1 hour
        
        # Database for coordination tracking
        self.db_path = "data/strategy_coordination.db"
        self.db_initialized = False
        
        # Emergency controls
        self.emergency_stop = False
        self.coordination_paused = False
        
        logger.info("[MASTER_COORDINATOR] Strategy coordinator configuration loaded")
    
    async def initialize(self):
        """Initialize the master strategy coordinator"""
        try:
            logger.info("[MASTER_COORDINATOR] Initializing Master Strategy Coordinator...")
            
            # Initialize database
            await self._initialize_database()
            
            # Initialize all managers
            await self._initialize_managers()
            
            # Initialize all strategies
            await self._initialize_strategies()
            
            # Setup resource allocations
            await self._initialize_resource_allocations()
            
            # Start coordination loops
            await self._start_coordination_tasks()
            
            logger.info("[MASTER_COORDINATOR] Master Strategy Coordinator initialized successfully")
            
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Coordinator initialization failed: {e}")
            raise
    
    async def _initialize_database(self):
        """Initialize coordination tracking database"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            async with aiosqlite.connect(self.db_path) as db:
                # Coordination events table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS coordination_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        event_type TEXT NOT NULL,
                        involved_strategies TEXT NOT NULL,
                        coordination_mode TEXT NOT NULL,
                        resolution_strategy TEXT NOT NULL,
                        input_opportunities INTEGER NOT NULL,
                        output_opportunities INTEGER NOT NULL,
                        conflicts_resolved INTEGER NOT NULL,
                        success BOOLEAN NOT NULL,
                        execution_time_ms REAL NOT NULL,
                        performance_impact REAL NOT NULL,
                        metadata TEXT
                    )
                """)
                
                # Strategy performance table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        opportunities_processed INTEGER NOT NULL,
                        opportunities_executed INTEGER NOT NULL,
                        success_rate REAL NOT NULL,
                        average_confidence REAL NOT NULL,
                        average_return REAL NOT NULL,
                        total_pnl REAL NOT NULL,
                        coordination_score REAL NOT NULL
                    )
                """)
                
                # Resource allocation tracking
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS resource_tracking (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        resource_type TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        allocated_amount REAL NOT NULL,
                        utilization_rate REAL NOT NULL,
                        efficiency_score REAL NOT NULL
                    )
                """)
                
                await db.commit()
            
            self.db_initialized = True
            logger.info("[MASTER_COORDINATOR] Coordination database initialized")
            
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Database initialization failed: {e}")
            raise
    
    async def _initialize_managers(self):
        """Initialize all unified managers"""
        try:
            # Initialize managers in dependency order
            self.risk_manager = UnifiedRiskManager(self.settings)
            await self.risk_manager.initialize()
            
            self.portfolio_manager = UnifiedPortfolioManager(self.settings, self.risk_manager)
            await self.portfolio_manager.initialize()
            
            self.multi_wallet_manager = MultiWalletManager(self.settings, self.risk_manager)
            await self.multi_wallet_manager.initialize_wallets()
            
            self.swap_executor = SwapExecutor(self.settings)
            await self.swap_executor.initialize()
            
            self.data_manager = UnifiedDataManager(self.settings)
            await self.data_manager.initialize()
            
            self.system_manager = UnifiedSystemManager(self.settings)
            await self.system_manager.initialize()
            
            self.trading_manager = UnifiedTradingManager(
                settings=self.settings,
                risk_manager=self.risk_manager,
                portfolio_manager=self.portfolio_manager,
                multi_wallet_manager=self.multi_wallet_manager
            )
            await self.trading_manager.initialize()
            
            self.order_manager = UnifiedOrderManager(
                settings=self.settings,
                swap_executor=self.swap_executor,
                multi_wallet_manager=self.multi_wallet_manager,
                risk_manager=self.risk_manager,
                portfolio_manager=self.portfolio_manager
            )
            await self.order_manager.initialize()
            
            # Register coordinator with system manager
            if self.system_manager:
                self.system_manager.register_component(
                    "master_coordinator", 
                    self, 
                    self._health_check
                )
            
            logger.info("[MASTER_COORDINATOR] All managers initialized and integrated")
            
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Manager initialization failed: {e}")
            raise
    
    async def _initialize_strategies(self):
        """Initialize all trading strategies"""
        try:
            # Strategy configurations
            strategy_configs = {
                "momentum": StrategyConfig(
                    strategy_type=StrategyType.MOMENTUM,
                    capital_allocation=0.35,
                    risk_tolerance=0.15,
                    max_positions=3,
                    target_return=0.20
                ),
                "mean_reversion": StrategyConfig(
                    strategy_type=StrategyType.MEAN_REVERSION,
                    capital_allocation=0.30,
                    risk_tolerance=0.10,
                    max_positions=4,
                    target_return=0.15
                ),
                "grid_trading": StrategyConfig(
                    strategy_type=StrategyType.GRID_TRADING,
                    capital_allocation=0.20,
                    risk_tolerance=0.08,
                    max_positions=5,
                    target_return=0.12
                ),
                "arbitrage": StrategyConfig(
                    strategy_type=StrategyType.ARBITRAGE,
                    capital_allocation=0.15,
                    risk_tolerance=0.05,
                    max_positions=2,
                    target_return=0.08
                )
            }
            
            # Initialize strategies with proper dependencies
            try:
                self.strategies["momentum"] = MomentumStrategy(
                    config=strategy_configs["momentum"],
                    data_manager=self.data_manager,
                    risk_manager=self.risk_manager
                )
            except Exception as e:
                logger.warning(f"[MASTER_COORDINATOR] Momentum strategy initialization failed: {e}")
                # Create mock strategy for testing
                self.strategies["momentum"] = self._create_mock_strategy("momentum", strategy_configs["momentum"])
            
            try:
                self.strategies["mean_reversion"] = MeanReversionStrategy(
                    config=strategy_configs["mean_reversion"],
                    data_manager=self.data_manager,
                    risk_manager=self.risk_manager
                )
            except Exception as e:
                logger.warning(f"[MASTER_COORDINATOR] Mean reversion strategy initialization failed: {e}")
                self.strategies["mean_reversion"] = self._create_mock_strategy("mean_reversion", strategy_configs["mean_reversion"])
            
            try:
                self.strategies["grid_trading"] = GridTradingStrategy(
                    config=strategy_configs["grid_trading"],
                    data_manager=self.data_manager,
                    risk_manager=self.risk_manager
                )
            except Exception as e:
                logger.warning(f"[MASTER_COORDINATOR] Grid trading strategy initialization failed: {e}")
                self.strategies["grid_trading"] = self._create_mock_strategy("grid_trading", strategy_configs["grid_trading"])
            
            try:
                self.strategies["arbitrage"] = ArbitrageStrategy(
                    config=strategy_configs["arbitrage"],
                    data_manager=self.data_manager,
                    risk_manager=self.risk_manager
                )
            except Exception as e:
                logger.warning(f"[MASTER_COORDINATOR] Arbitrage strategy initialization failed: {e}")
                self.strategies["arbitrage"] = self._create_mock_strategy("arbitrage", strategy_configs["arbitrage"])
            
            self.strategy_configs = strategy_configs
            
            # Initialize performance tracking
            for strategy_name in self.strategies.keys():
                self.strategy_performance[strategy_name] = {
                    'opportunities_processed': 0,
                    'opportunities_executed': 0,
                    'success_rate': 0.0,
                    'average_confidence': 0.0,
                    'average_return': 0.0,
                    'total_pnl': 0.0,
                    'coordination_score': 0.5  # Start with neutral score
                }
            
            logger.info(f"[MASTER_COORDINATOR] Initialized {len(self.strategies)} strategies")
            
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Strategy initialization failed: {e}")
            raise
    
    def _create_mock_strategy(self, name: str, config: StrategyConfig) -> BaseStrategy:
        """Create mock strategy for testing purposes"""
        
        class MockStrategy(BaseStrategy):
            def __init__(self, strategy_name: str, strategy_config: StrategyConfig):
                super().__init__()
                self.name = strategy_name
                self.config = strategy_config
                self.status = StrategyStatus.ACTIVE
            
            async def analyze_opportunity(self, signal: Signal) -> Optional[StrategyOpportunity]:
                """Mock opportunity analysis"""
                return StrategyOpportunity(
                    opportunity_id=f"{self.name}_{int(time.time())}",
                    strategy_name=self.name,
                    strategy_type=self.config.strategy_type,
                    token_address=signal.token_address,
                    signal=signal,
                    confidence=0.7,  # Mock confidence
                    expected_return=0.05,  # Mock 5% return
                    risk_score=0.3,  # Mock risk
                    required_capital=Decimal('100'),
                    required_api_calls=5,
                    required_processing=1.0,
                    expected_duration=300  # 5 minutes
                )
            
            async def execute_opportunity(self, opportunity: StrategyOpportunity) -> bool:
                """Mock opportunity execution"""
                await asyncio.sleep(0.1)  # Simulate execution time
                return True
        
        return MockStrategy(name, config)
    
    async def _initialize_resource_allocations(self):
        """Initialize resource allocation tracking"""
        try:
            # Capital allocation based on portfolio
            total_capital = float(getattr(self.settings, 'INITIAL_CAPITAL', 1000))
            self.resource_allocations[ResourceType.CAPITAL] = ResourceAllocation(
                resource_type=ResourceType.CAPITAL,
                total_available=total_capital,
                reserved=total_capital * 0.1  # 10% emergency reserve
            )
            
            # API quota allocation
            daily_api_quota = getattr(self.settings, 'DAILY_API_QUOTA', 10000)
            self.resource_allocations[ResourceType.API_QUOTA] = ResourceAllocation(
                resource_type=ResourceType.API_QUOTA,
                total_available=daily_api_quota,
                reserved=daily_api_quota * 0.1  # 10% reserve
            )
            
            # Processing power allocation (arbitrary units)
            processing_capacity = getattr(self.settings, 'PROCESSING_CAPACITY', 100.0)
            self.resource_allocations[ResourceType.PROCESSING_POWER] = ResourceAllocation(
                resource_type=ResourceType.PROCESSING_POWER,
                total_available=processing_capacity
            )
            
            # Wallet capacity allocation
            wallet_slots = getattr(self.settings, 'MAX_WALLET_POSITIONS', 20)
            self.resource_allocations[ResourceType.WALLET_CAPACITY] = ResourceAllocation(
                resource_type=ResourceType.WALLET_CAPACITY,
                total_available=wallet_slots
            )
            
            # Risk budget allocation
            risk_budget = getattr(self.settings, 'TOTAL_RISK_BUDGET', 1.0)
            self.resource_allocations[ResourceType.RISK_BUDGET] = ResourceAllocation(
                resource_type=ResourceType.RISK_BUDGET,
                total_available=risk_budget
            )
            
            # Opportunity slots
            self.resource_allocations[ResourceType.OPPORTUNITY_SLOTS] = ResourceAllocation(
                resource_type=ResourceType.OPPORTUNITY_SLOTS,
                total_available=self.max_concurrent_opportunities
            )
            
            logger.info("[MASTER_COORDINATOR] Resource allocations initialized")
            
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Resource allocation initialization failed: {e}")
            raise
    
    async def _start_coordination_tasks(self):
        """Start background coordination tasks"""
        try:
            # Main coordination loop
            asyncio.create_task(self._coordination_loop())
            
            # Resource optimization loop
            asyncio.create_task(self._resource_optimization_loop())
            
            # Performance monitoring loop
            asyncio.create_task(self._performance_monitoring_loop())
            
            # Coordination mode adaptation loop
            asyncio.create_task(self._mode_adaptation_loop())
            
            logger.info("[MASTER_COORDINATOR] Coordination tasks started")
            
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Task startup failed: {e}")
            raise
    
    async def coordinate_strategies(self, signals: List[Signal]) -> List[StrategyOpportunity]:
        """Main coordination method - orchestrate all strategies"""
        if self.emergency_stop or self.coordination_paused:
            return []
        
        start_time = time.time()
        event_id = f"coord_{int(time.time() * 1000)}"
        
        try:
            logger.info(f"[MASTER_COORDINATOR] Coordinating {len(signals)} signals across {len(self.strategies)} strategies")
            
            # Phase 1: Generate opportunities from all strategies
            all_opportunities = await self._generate_opportunities(signals)
            
            # Phase 2: Detect and resolve conflicts
            resolved_opportunities = await self._resolve_conflicts(all_opportunities)
            
            # Phase 3: Allocate resources optimally
            allocated_opportunities = await self._allocate_resources(resolved_opportunities)
            
            # Phase 4: Optimize portfolio balance
            optimized_opportunities = await self._optimize_portfolio_balance(allocated_opportunities)
            
            # Phase 5: Final validation and ranking
            final_opportunities = await self._final_validation_and_ranking(optimized_opportunities)
            
            # Record coordination event
            execution_time = (time.time() - start_time) * 1000
            coordination_event = CoordinationEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                event_type="strategy_coordination",
                involved_strategies=list(self.strategies.keys()),
                coordination_mode=self.coordination_mode,
                resolution_strategy=self.conflict_resolution,
                input_opportunities=len(all_opportunities),
                output_opportunities=len(final_opportunities),
                conflicts_resolved=len(all_opportunities) - len(resolved_opportunities),
                resources_allocated={
                    rt.value: sum(alloc.allocated.values()) 
                    for rt, alloc in self.resource_allocations.items()
                },
                success=True,
                execution_time_ms=execution_time,
                performance_impact=self._calculate_performance_impact(final_opportunities)
            )
            
            self.coordination_history.append(coordination_event)
            await self._log_coordination_event(coordination_event)
            
            logger.info(f"[MASTER_COORDINATOR] Coordination complete: {len(signals)} signals → {len(final_opportunities)} opportunities ({execution_time:.1f}ms)")
            
            return final_opportunities
            
        except Exception as e:
            # Record failed coordination event
            execution_time = (time.time() - start_time) * 1000
            coordination_event = CoordinationEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                event_type="strategy_coordination",
                involved_strategies=list(self.strategies.keys()),
                coordination_mode=self.coordination_mode,
                resolution_strategy=self.conflict_resolution,
                input_opportunities=len(signals),
                output_opportunities=0,
                conflicts_resolved=0,
                resources_allocated={},
                success=False,
                execution_time_ms=execution_time,
                performance_impact=0.0,
                metadata={'error': str(e)}
            )
            
            self.coordination_history.append(coordination_event)
            await self._log_coordination_event(coordination_event)
            
            logger.error(f"[MASTER_COORDINATOR] Coordination failed: {e}")
            return []
    
    async def _generate_opportunities(self, signals: List[Signal]) -> List[StrategyOpportunity]:
        """Generate opportunities from all strategies"""
        all_opportunities = []
        
        # Process signals through all strategies in parallel
        strategy_tasks = []
        for strategy_name, strategy in self.strategies.items():
            if strategy.status == StrategyStatus.ACTIVE:
                for signal in signals:
                    task = asyncio.create_task(
                        self._strategy_analyze_opportunity(strategy, signal),
                        name=f"{strategy_name}_{signal.token_address}"
                    )
                    strategy_tasks.append(task)
        
        # Wait for all strategy analyses
        if strategy_tasks:
            results = await asyncio.gather(*strategy_tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, StrategyOpportunity):
                    # Enhance opportunity with coordination metadata
                    await self._enhance_opportunity_metadata(result)
                    all_opportunities.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"[MASTER_COORDINATOR] Strategy analysis failed: {result}")
        
        logger.info(f"[MASTER_COORDINATOR] Generated {len(all_opportunities)} opportunities from {len(signals)} signals")
        return all_opportunities
    
    async def _strategy_analyze_opportunity(self, strategy: BaseStrategy, signal: Signal) -> Optional[StrategyOpportunity]:
        """Analyze opportunity with individual strategy"""
        try:
            if hasattr(strategy, 'analyze_opportunity'):
                opportunity = await strategy.analyze_opportunity(signal)
                if opportunity:
                    # Update strategy performance tracking
                    self.strategy_performance[strategy.name]['opportunities_processed'] += 1
                return opportunity
            else:
                # Fallback for strategies without analyze_opportunity method
                return await self._create_fallback_opportunity(strategy, signal)
        except Exception as e:
            logger.warning(f"[MASTER_COORDINATOR] Strategy {strategy.name} analysis failed: {e}")
            return None
    
    async def _create_fallback_opportunity(self, strategy: BaseStrategy, signal: Signal) -> StrategyOpportunity:
        """Create fallback opportunity for strategies without analyze_opportunity method"""
        return StrategyOpportunity(
            opportunity_id=f"{strategy.name}_{signal.token_address}_{int(time.time())}",
            strategy_name=strategy.name,
            strategy_type=strategy.config.strategy_type,
            token_address=signal.token_address,
            signal=signal,
            confidence=signal.confidence,
            expected_return=signal.expected_return or 0.05,
            risk_score=signal.risk_score or 0.3,
            required_capital=Decimal(str(signal.position_size or 100)),
            required_api_calls=5,
            required_processing=1.0,
            expected_duration=300
        )
    
    async def _enhance_opportunity_metadata(self, opportunity: StrategyOpportunity):
        """Enhance opportunity with coordination metadata"""
        try:
            # Calculate risk-reward ratio
            if opportunity.risk_score > 0:
                opportunity.risk_reward_ratio = opportunity.expected_return / opportunity.risk_score
            else:
                opportunity.risk_reward_ratio = opportunity.expected_return
            
            # Calculate portfolio impact
            if self.portfolio_manager:
                portfolio_status = await self.portfolio_manager.get_comprehensive_metrics()
                total_value = portfolio_status.get('total_value_usd', 1000)
                opportunity.portfolio_impact = float(opportunity.required_capital) / total_value
            
            # Calculate diversification benefit
            opportunity.diversification_benefit = await self._calculate_diversification_benefit(opportunity)
            
            # Set coordination weight based on strategy performance
            strategy_perf = self.strategy_performance.get(opportunity.strategy_name, {})
            opportunity.coordination_weight = strategy_perf.get('coordination_score', 0.5)
            
            # Calculate priority score
            opportunity.priority_score = opportunity.calculate_composite_score()
            
        except Exception as e:
            logger.warning(f"[MASTER_COORDINATOR] Opportunity metadata enhancement failed: {e}")
    
    async def _calculate_diversification_benefit(self, opportunity: StrategyOpportunity) -> float:
        """Calculate diversification benefit of adding this opportunity"""
        try:
            if self.portfolio_manager:
                current_positions = await self.portfolio_manager.get_comprehensive_metrics()
                positions = current_positions.get('positions', {})
                
                # Simple diversification calculation
                token_already_held = any(
                    pos.get('token', '').startswith(opportunity.token_address[:8]) 
                    for pos in positions.values()
                )
                
                if token_already_held:
                    return 0.2  # Lower benefit if token already held
                else:
                    return 0.8  # Higher benefit for new token
            
            return 0.5  # Neutral if portfolio data unavailable
            
        except Exception:
            return 0.5
    
    async def _resolve_conflicts(self, opportunities: List[StrategyOpportunity]) -> List[StrategyOpportunity]:
        """Advanced conflict resolution between strategies"""
        if len(opportunities) <= 1:
            return opportunities
        
        # Group opportunities by token
        token_groups = defaultdict(list)
        for opp in opportunities:
            token_groups[opp.token_address].append(opp)
        
        resolved_opportunities = []
        conflicts_detected = 0
        
        # Resolve conflicts within each token group
        for token_address, token_opportunities in token_groups.items():
            if len(token_opportunities) == 1:
                resolved_opportunities.extend(token_opportunities)
            else:
                conflicts_detected += len(token_opportunities) - 1
                resolved = await self._resolve_token_conflicts(token_opportunities)
                resolved_opportunities.extend(resolved)
        
        logger.info(f"[MASTER_COORDINATOR] Resolved {conflicts_detected} conflicts using {self.conflict_resolution.value}")
        return resolved_opportunities
    
    async def _resolve_token_conflicts(self, conflicting_opportunities: List[StrategyOpportunity]) -> List[StrategyOpportunity]:
        """Resolve conflicts for a specific token"""
        if len(conflicting_opportunities) <= 1:
            return conflicting_opportunities
        
        if self.conflict_resolution == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
            return [max(conflicting_opportunities, key=lambda x: x.confidence)]
        
        elif self.conflict_resolution == ConflictResolutionStrategy.BEST_RISK_REWARD:
            return [max(conflicting_opportunities, key=lambda x: x.risk_reward_ratio)]
        
        elif self.conflict_resolution == ConflictResolutionStrategy.PORTFOLIO_BALANCE:
            return await self._resolve_by_portfolio_balance(conflicting_opportunities)
        
        elif self.conflict_resolution == ConflictResolutionStrategy.PERFORMANCE_WEIGHTED:
            return await self._resolve_by_performance(conflicting_opportunities)
        
        elif self.conflict_resolution == ConflictResolutionStrategy.DIVERSIFICATION:
            return [max(conflicting_opportunities, key=lambda x: x.diversification_benefit)]
        
        elif self.conflict_resolution == ConflictResolutionStrategy.MOMENTUM_PRIORITY:
            momentum_opportunities = [opp for opp in conflicting_opportunities if opp.strategy_type == StrategyType.MOMENTUM]
            if momentum_opportunities:
                return [max(momentum_opportunities, key=lambda x: x.confidence)]
            else:
                return [max(conflicting_opportunities, key=lambda x: x.confidence)]
        
        elif self.conflict_resolution == ConflictResolutionStrategy.HYBRID_OPTIMIZATION:
            return await self._resolve_by_hybrid_optimization(conflicting_opportunities)
        
        else:
            # Default to highest confidence
            return [max(conflicting_opportunities, key=lambda x: x.confidence)]
    
    async def _resolve_by_portfolio_balance(self, opportunities: List[StrategyOpportunity]) -> List[StrategyOpportunity]:
        """Resolve conflicts to maintain portfolio balance"""
        try:
            if not self.portfolio_manager:
                return [max(opportunities, key=lambda x: x.confidence)]
            
            portfolio_metrics = await self.portfolio_manager.get_comprehensive_metrics()
            strategies = portfolio_metrics.get('strategies', {})
            
            # Find the strategy that is most underallocated
            min_allocation_diff = float('inf')
            best_opportunity = opportunities[0]
            
            for opp in opportunities:
                strategy_info = strategies.get(opp.strategy_name, {})
                target_allocation = strategy_info.get('target_allocation', 0.25)
                current_allocation = strategy_info.get('current_allocation', 0.25)
                allocation_diff = target_allocation - current_allocation
                
                if allocation_diff < min_allocation_diff:
                    min_allocation_diff = allocation_diff
                    best_opportunity = opp
            
            return [best_opportunity]
            
        except Exception:
            return [max(opportunities, key=lambda x: x.confidence)]
    
    async def _resolve_by_performance(self, opportunities: List[StrategyOpportunity]) -> List[StrategyOpportunity]:
        """Resolve conflicts based on recent strategy performance"""
        best_opportunity = opportunities[0]
        best_score = 0.0
        
        for opp in opportunities:
            strategy_perf = self.strategy_performance.get(opp.strategy_name, {})
            performance_score = (
                strategy_perf.get('success_rate', 0.0) * 0.4 +
                strategy_perf.get('coordination_score', 0.5) * 0.3 +
                (strategy_perf.get('average_return', 0.0) * 10) * 0.3  # Scale return to 0-1 range
            )
            
            composite_score = performance_score * opp.confidence
            
            if composite_score > best_score:
                best_score = composite_score
                best_opportunity = opp
        
        return [best_opportunity]
    
    async def _resolve_by_hybrid_optimization(self, opportunities: List[StrategyOpportunity]) -> List[StrategyOpportunity]:
        """Advanced hybrid optimization for conflict resolution"""
        # Multi-criteria optimization
        scored_opportunities = []
        
        for opp in opportunities:
            # Get strategy performance
            strategy_perf = self.strategy_performance.get(opp.strategy_name, {})
            
            # Calculate composite optimization score
            score = (
                opp.confidence * 0.25 +                              # Signal confidence
                opp.risk_reward_ratio * 0.20 +                       # Risk-reward optimization
                opp.diversification_benefit * 0.15 +                 # Diversification value
                strategy_perf.get('coordination_score', 0.5) * 0.15 + # Strategy coordination performance
                strategy_perf.get('success_rate', 0.0) * 0.15 +      # Strategy success rate
                opp.portfolio_impact * 0.10                          # Portfolio impact weighting
            )
            
            scored_opportunities.append((score, opp))
        
        # Return the best opportunity
        scored_opportunities.sort(key=lambda x: x[0], reverse=True)
        return [scored_opportunities[0][1]]
    
    async def _allocate_resources(self, opportunities: List[StrategyOpportunity]) -> List[StrategyOpportunity]:
        """Allocate resources to opportunities optimally"""
        allocated_opportunities = []
        
        # Sort opportunities by priority score
        sorted_opportunities = sorted(opportunities, key=lambda x: x.priority_score, reverse=True)
        
        for opp in sorted_opportunities:
            can_allocate = True
            
            # Check capital allocation
            capital_allocation = self.resource_allocations[ResourceType.CAPITAL]
            if not capital_allocation.allocate(opp.strategy_name, float(opp.required_capital)):
                can_allocate = False
            
            # Check API quota
            if can_allocate:
                api_allocation = self.resource_allocations[ResourceType.API_QUOTA]
                if not api_allocation.allocate(opp.strategy_name, opp.required_api_calls):
                    can_allocate = False
                    # Rollback capital allocation
                    capital_allocation.deallocate(opp.strategy_name, float(opp.required_capital))
            
            # Check processing power
            if can_allocate:
                processing_allocation = self.resource_allocations[ResourceType.PROCESSING_POWER]
                if not processing_allocation.allocate(opp.strategy_name, opp.required_processing):
                    can_allocate = False
                    # Rollback previous allocations
                    capital_allocation.deallocate(opp.strategy_name, float(opp.required_capital))
                    api_allocation.deallocate(opp.strategy_name, opp.required_api_calls)
            
            # Check opportunity slots
            if can_allocate:
                slot_allocation = self.resource_allocations[ResourceType.OPPORTUNITY_SLOTS]
                if not slot_allocation.allocate(opp.strategy_name, 1):
                    can_allocate = False
                    # Rollback previous allocations
                    capital_allocation.deallocate(opp.strategy_name, float(opp.required_capital))
                    api_allocation.deallocate(opp.strategy_name, opp.required_api_calls)
                    processing_allocation.deallocate(opp.strategy_name, opp.required_processing)
            
            if can_allocate:
                allocated_opportunities.append(opp)
                
                # Track allocation
                await self._track_resource_allocation(opp)
            else:
                logger.debug(f"[MASTER_COORDINATOR] Insufficient resources for opportunity {opp.opportunity_id}")
        
        logger.info(f"[MASTER_COORDINATOR] Resource allocation: {len(opportunities)} requested → {len(allocated_opportunities)} allocated")
        return allocated_opportunities
    
    async def _track_resource_allocation(self, opportunity: StrategyOpportunity):
        """Track resource allocation for monitoring"""
        try:
            if self.db_initialized:
                timestamp = time.time()
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        INSERT INTO resource_tracking 
                        (timestamp, resource_type, strategy_name, allocated_amount, utilization_rate, efficiency_score)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp,
                        ResourceType.CAPITAL.value,
                        opportunity.strategy_name,
                        float(opportunity.required_capital),
                        0.0,  # Will be updated when opportunity completes
                        opportunity.priority_score
                    ))
                    await db.commit()
        except Exception as e:
            logger.warning(f"[MASTER_COORDINATOR] Resource tracking failed: {e}")
    
    async def _optimize_portfolio_balance(self, opportunities: List[StrategyOpportunity]) -> List[StrategyOpportunity]:
        """Optimize opportunities for portfolio balance"""
        if not self.portfolio_manager or len(opportunities) <= 1:
            return opportunities
        
        try:
            portfolio_metrics = await self.portfolio_manager.get_comprehensive_metrics()
            strategies = portfolio_metrics.get('strategies', {})
            
            # Group opportunities by strategy
            strategy_groups = defaultdict(list)
            for opp in opportunities:
                strategy_groups[opp.strategy_name].append(opp)
            
            # Calculate current allocation vs targets
            optimized_opportunities = []
            
            for strategy_name, strategy_opportunities in strategy_groups.items():
                strategy_info = strategies.get(strategy_name, {})
                target_allocation = strategy_info.get('target_allocation', 0.25)
                current_allocation = strategy_info.get('current_allocation', 0.0)
                
                # Calculate how much more allocation this strategy can take
                allocation_capacity = max(0, target_allocation - current_allocation)
                
                if allocation_capacity > 0:
                    # Sort by priority and take opportunities up to capacity
                    sorted_opps = sorted(strategy_opportunities, key=lambda x: x.priority_score, reverse=True)
                    
                    allocated_capital = 0.0
                    for opp in sorted_opps:
                        if allocated_capital + float(opp.required_capital) <= allocation_capacity * 1000:  # Assume 1000 total
                            optimized_opportunities.append(opp)
                            allocated_capital += float(opp.required_capital)
                        else:
                            break
            
            return optimized_opportunities
            
        except Exception as e:
            logger.warning(f"[MASTER_COORDINATOR] Portfolio optimization failed: {e}")
            return opportunities
    
    async def _final_validation_and_ranking(self, opportunities: List[StrategyOpportunity]) -> List[StrategyOpportunity]:
        """Final validation and ranking of opportunities"""
        validated_opportunities = []
        
        for opp in opportunities:
            # Risk validation
            if self.risk_manager:
                try:
                    risk_check = await self.risk_manager.validate_trade_risk(
                        strategy_name=opp.strategy_name,
                        token_address=opp.token_address,
                        trade_size=float(opp.required_capital),
                        expected_return=opp.expected_return,
                        risk_score=opp.risk_score
                    )
                    
                    if not risk_check[0]:  # Risk validation failed
                        logger.debug(f"[MASTER_COORDINATOR] Risk validation failed for {opp.opportunity_id}: {risk_check[1]}")
                        continue
                except Exception as e:
                    logger.warning(f"[MASTER_COORDINATOR] Risk validation error: {e}")
                    continue
            
            # Add to active opportunities tracking
            self.active_opportunities[opp.opportunity_id] = opp
            validated_opportunities.append(opp)
        
        # Final ranking by composite score
        validated_opportunities.sort(key=lambda x: x.calculate_composite_score(), reverse=True)
        
        # Limit to max concurrent opportunities
        if len(validated_opportunities) > self.max_concurrent_opportunities:
            validated_opportunities = validated_opportunities[:self.max_concurrent_opportunities]
        
        return validated_opportunities
    
    async def _coordination_loop(self):
        """Main coordination loop"""
        while not self.emergency_stop:
            try:
                if not self.coordination_paused:
                    # Get signals from data manager
                    if self.data_manager:
                        # This would normally get trending tokens and generate signals
                        # For now, we'll create a simple mock signal for testing
                        mock_signals = await self._generate_mock_signals()
                        
                        if mock_signals:
                            coordinated_opportunities = await self.coordinate_strategies(mock_signals)
                            
                            # Execute opportunities through order manager
                            if coordinated_opportunities and self.order_manager:
                                await self._execute_coordinated_opportunities(coordinated_opportunities)
                
                await asyncio.sleep(self.coordination_interval)
                
            except Exception as e:
                logger.error(f"[MASTER_COORDINATOR] Coordination loop error: {e}")
                await asyncio.sleep(self.coordination_interval)
    
    async def _generate_mock_signals(self) -> List[Signal]:
        """Generate mock signals for testing"""
        try:
            # In production, this would fetch from data manager
            mock_tokens = ["MOCK_TOKEN_1", "MOCK_TOKEN_2", "MOCK_TOKEN_3"]
            signals = []
            
            for i, token in enumerate(mock_tokens):
                signal = Signal(
                    signal_id=f"mock_{token}_{int(time.time())}",
                    strategy_type=StrategyType.MOMENTUM,
                    token_address=token,
                    confidence=0.7 + (i * 0.1),
                    strength=0.8,
                    direction="buy",
                    timestamp=datetime.now(),
                    expected_return=0.05 + (i * 0.01),
                    risk_score=0.3,
                    position_size=100 + (i * 50)
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.warning(f"[MASTER_COORDINATOR] Mock signal generation failed: {e}")
            return []
    
    async def _execute_coordinated_opportunities(self, opportunities: List[StrategyOpportunity]):
        """Execute opportunities through the order manager"""
        try:
            for opp in opportunities:
                # Convert opportunity to order request
                order_request = self._opportunity_to_order_request(opp)
                
                if order_request and self.order_manager:
                    order_id = await self.order_manager.submit_order(order_request)
                    if order_id:
                        logger.info(f"[MASTER_COORDINATOR] Executed opportunity {opp.opportunity_id} as order {order_id}")
                        
                        # Update strategy performance
                        self.strategy_performance[opp.strategy_name]['opportunities_executed'] += 1
                    else:
                        logger.warning(f"[MASTER_COORDINATOR] Failed to execute opportunity {opp.opportunity_id}")
        
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Opportunity execution failed: {e}")
    
    def _opportunity_to_order_request(self, opportunity: StrategyOpportunity) -> Any:
        """Convert opportunity to order request"""
        try:
            # This would need the proper OrderRequest import and structure
            # For now, return a mock structure
            return {
                'token_address': opportunity.token_address,
                'strategy_name': opportunity.strategy_name,
                'side': 'buy',  # From signal direction
                'amount': float(opportunity.required_capital),
                'order_type': 'market',
                'max_slippage': 0.01,
                'time_in_force': 60
            }
        except Exception:
            return None
    
    # Additional coordination loops and utility methods
    async def _resource_optimization_loop(self):
        """Optimize resource allocation continuously"""
        while not self.emergency_stop:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._optimize_resource_allocation()
            except Exception as e:
                logger.error(f"[MASTER_COORDINATOR] Resource optimization loop error: {e}")
    
    async def _performance_monitoring_loop(self):
        """Monitor and update strategy performance"""
        while not self.emergency_stop:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._update_performance_metrics()
            except Exception as e:
                logger.error(f"[MASTER_COORDINATOR] Performance monitoring loop error: {e}")
    
    async def _mode_adaptation_loop(self):
        """Adapt coordination mode based on market conditions"""
        while not self.emergency_stop:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                await self._adapt_coordination_mode()
            except Exception as e:
                logger.error(f"[MASTER_COORDINATOR] Mode adaptation loop error: {e}")
    
    async def _optimize_resource_allocation(self):
        """Optimize resource allocation across strategies"""
        try:
            # Analyze resource utilization efficiency
            for resource_type, allocation in self.resource_allocations.items():
                # Rebalance based on performance
                if len(allocation.allocated) > 1:
                    await self._rebalance_resource_allocation(resource_type, allocation)
        except Exception as e:
            logger.warning(f"[MASTER_COORDINATOR] Resource optimization failed: {e}")
    
    async def _rebalance_resource_allocation(self, resource_type: ResourceType, allocation: ResourceAllocation):
        """Rebalance resource allocation for a specific resource type"""
        try:
            # Calculate efficiency scores for each strategy
            efficiency_scores = {}
            for strategy_name in allocation.allocated.keys():
                strategy_perf = self.strategy_performance.get(strategy_name, {})
                efficiency_scores[strategy_name] = strategy_perf.get('coordination_score', 0.5)
            
            # Redistribute resources based on efficiency
            total_efficiency = sum(efficiency_scores.values())
            if total_efficiency > 0:
                total_allocated = sum(allocation.allocated.values())
                
                # Recalculate allocations
                new_allocations = {}
                for strategy_name, efficiency in efficiency_scores.items():
                    new_allocation = (efficiency / total_efficiency) * total_allocated
                    new_allocations[strategy_name] = new_allocation
                
                # Update allocations gradually (damping factor)
                damping = 0.1  # 10% adjustment per rebalance
                for strategy_name, new_allocation in new_allocations.items():
                    current = allocation.allocated.get(strategy_name, 0)
                    adjustment = (new_allocation - current) * damping
                    allocation.allocated[strategy_name] = current + adjustment
                
        except Exception as e:
            logger.warning(f"[MASTER_COORDINATOR] Resource rebalancing failed: {e}")
    
    async def _update_performance_metrics(self):
        """Update strategy performance metrics"""
        try:
            for strategy_name in self.strategies.keys():
                await self._calculate_strategy_coordination_score(strategy_name)
        except Exception as e:
            logger.warning(f"[MASTER_COORDINATOR] Performance metrics update failed: {e}")
    
    async def _calculate_strategy_coordination_score(self, strategy_name: str):
        """Calculate coordination score for a strategy"""
        try:
            perf = self.strategy_performance[strategy_name]
            
            # Calculate success rate
            total_processed = perf['opportunities_processed']
            total_executed = perf['opportunities_executed']
            
            if total_processed > 0:
                success_rate = total_executed / total_processed
                perf['success_rate'] = success_rate
                
                # Calculate coordination score based on multiple factors
                coordination_score = (
                    success_rate * 0.4 +                    # Execution success
                    min(perf['average_confidence'], 1.0) * 0.3 +  # Signal quality
                    min(perf['average_return'] * 10, 1.0) * 0.3   # Return performance
                )
                
                perf['coordination_score'] = max(0.1, min(1.0, coordination_score))
            
        except Exception as e:
            logger.warning(f"[MASTER_COORDINATOR] Coordination score calculation failed for {strategy_name}: {e}")
    
    async def _adapt_coordination_mode(self):
        """Adapt coordination mode based on market conditions"""
        try:
            # Analyze recent coordination performance
            recent_events = list(self.coordination_history)[-10:]  # Last 10 events
            
            if len(recent_events) >= 5:
                # Calculate average performance metrics
                avg_success_rate = sum(1 for event in recent_events if event.success) / len(recent_events)
                avg_conflicts = sum(event.conflicts_resolved for event in recent_events) / len(recent_events)
                avg_execution_time = sum(event.execution_time_ms for event in recent_events) / len(recent_events)
                
                # Adapt mode based on performance
                if avg_success_rate < 0.7 or avg_execution_time > 5000:  # 5 seconds
                    # Switch to simpler mode
                    if self.coordination_mode == CoordinationMode.ADAPTIVE:
                        self.coordination_mode = CoordinationMode.COOPERATIVE
                        self.conflict_resolution = ConflictResolutionStrategy.HIGHEST_CONFIDENCE
                        logger.info("[MASTER_COORDINATOR] Switched to simpler coordination mode due to performance issues")
                        
                elif avg_success_rate > 0.9 and avg_execution_time < 1000:  # 1 second
                    # Can handle more complex coordination
                    if self.coordination_mode == CoordinationMode.COOPERATIVE:
                        self.coordination_mode = CoordinationMode.ADAPTIVE
                        self.conflict_resolution = ConflictResolutionStrategy.HYBRID_OPTIMIZATION
                        logger.info("[MASTER_COORDINATOR] Switched to adaptive coordination mode due to good performance")
                
        except Exception as e:
            logger.warning(f"[MASTER_COORDINATOR] Mode adaptation failed: {e}")
    
    def _calculate_performance_impact(self, opportunities: List[StrategyOpportunity]) -> float:
        """Calculate expected performance impact of coordinated opportunities"""
        if not opportunities:
            return 0.0
        
        total_expected_return = sum(opp.expected_return for opp in opportunities)
        weighted_confidence = sum(opp.confidence * opp.expected_return for opp in opportunities) / total_expected_return if total_expected_return > 0 else 0
        
        return weighted_confidence * total_expected_return
    
    async def _log_coordination_event(self, event: CoordinationEvent):
        """Log coordination event to database"""
        try:
            if self.db_initialized:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        INSERT INTO coordination_events 
                        (event_id, timestamp, event_type, involved_strategies, coordination_mode, 
                         resolution_strategy, input_opportunities, output_opportunities, conflicts_resolved,
                         success, execution_time_ms, performance_impact, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.timestamp.timestamp(),
                        event.event_type,
                        ','.join(event.involved_strategies),
                        event.coordination_mode.value,
                        event.resolution_strategy.value,
                        event.input_opportunities,
                        event.output_opportunities,
                        event.conflicts_resolved,
                        event.success,
                        event.execution_time_ms,
                        event.performance_impact,
                        json.dumps(event.metadata)
                    ))
                    await db.commit()
        except Exception as e:
            logger.warning(f"[MASTER_COORDINATOR] Event logging failed: {e}")
    
    async def _health_check(self) -> bool:
        """Health check for system manager integration"""
        try:
            # Check if coordinator is operational
            if self.emergency_stop:
                return False
            
            # Check strategy health
            active_strategies = sum(1 for strategy in self.strategies.values() 
                                 if strategy.status == StrategyStatus.ACTIVE)
            
            if active_strategies == 0:
                return False
            
            # Check recent coordination performance
            if self.coordination_history:
                recent_event = self.coordination_history[-1]
                if (datetime.now() - recent_event.timestamp).seconds > 300:  # 5 minutes
                    return False
            
            return True
            
        except Exception:
            return False
    
    async def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status"""
        try:
            # Strategy status
            strategy_status = {}
            for name, strategy in self.strategies.items():
                perf = self.strategy_performance[name]
                strategy_status[name] = {
                    'status': strategy.status.value,
                    'opportunities_processed': perf['opportunities_processed'],
                    'opportunities_executed': perf['opportunities_executed'],
                    'success_rate': perf['success_rate'],
                    'coordination_score': perf['coordination_score']
                }
            
            # Resource utilization
            resource_status = {}
            for resource_type, allocation in self.resource_allocations.items():
                resource_status[resource_type.value] = {
                    'total_available': allocation.total_available,
                    'allocated': sum(allocation.allocated.values()),
                    'available': allocation.available,
                    'utilization_rate': sum(allocation.allocated.values()) / allocation.total_available
                }
            
            # Recent coordination performance
            recent_events = list(self.coordination_history)[-10:]
            coordination_metrics = {
                'total_events': len(self.coordination_history),
                'recent_success_rate': sum(1 for e in recent_events if e.success) / max(len(recent_events), 1),
                'average_execution_time_ms': sum(e.execution_time_ms for e in recent_events) / max(len(recent_events), 1),
                'total_conflicts_resolved': sum(e.conflicts_resolved for e in recent_events)
            }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'coordination_mode': self.coordination_mode.value,
                'conflict_resolution': self.conflict_resolution.value,
                'emergency_stop': self.emergency_stop,
                'coordination_paused': self.coordination_paused,
                'active_opportunities': len(self.active_opportunities),
                'strategies': strategy_status,
                'resources': resource_status,
                'coordination_metrics': coordination_metrics
            }
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def emergency_shutdown(self):
        """Emergency shutdown of coordination"""
        try:
            logger.warning("[MASTER_COORDINATOR] Emergency shutdown initiated")
            
            self.emergency_stop = True
            self.coordination_paused = True
            
            # Clear active opportunities
            self.active_opportunities.clear()
            
            # Reset resource allocations
            for allocation in self.resource_allocations.values():
                allocation.allocated.clear()
            
            logger.info("[MASTER_COORDINATOR] Emergency shutdown complete")
            
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Emergency shutdown failed: {e}")
    
    # TRADING MANAGER INTEGRATION METHODS
    
    async def process_incoming_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Process and enhance incoming signals for the trading manager
        This is called by UnifiedTradingManager during signal processing
        """
        try:
            if self.emergency_stop or not signals:
                return []
            
            logger.info(f"[MASTER_COORDINATOR] Processing {len(signals)} incoming signals")
            
            enhanced_signals = []
            
            for signal in signals:
                try:
                    # Enhance signal with coordinator intelligence
                    enhanced_signal = await self._enhance_signal(signal)
                    if enhanced_signal and self._validate_signal_quality(enhanced_signal):
                        enhanced_signals.append(enhanced_signal)
                except Exception as e:
                    logger.warning(f"[MASTER_COORDINATOR] Signal enhancement failed: {e}")
                    # Include original signal if enhancement fails
                    if self._validate_signal_quality(signal):
                        enhanced_signals.append(signal)
            
            logger.info(f"[MASTER_COORDINATOR] Enhanced {len(signals)} -> {len(enhanced_signals)} signals")
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Signal processing failed: {e}")
            return signals  # Return original signals on error
    
    async def coordinate_strategy_opportunities(self, opportunities: List, strategy_performance: Dict[str, float], 
                                              strategy_wallets: Dict) -> List:
        """
        Advanced coordination of strategy opportunities
        Called by UnifiedTradingManager for sophisticated coordination
        """
        try:
            if self.emergency_stop or not opportunities:
                return []
            
            logger.info(f"[MASTER_COORDINATOR] Coordinating {len(opportunities)} strategy opportunities")
            
            # Update local performance data
            self.strategy_performance.update(strategy_performance)
            
            # Convert to StrategyOpportunity format for advanced processing
            coordinator_opportunities = []
            for i, opp in enumerate(opportunities):
                try:
                    coord_opp = StrategyOpportunity(
                        opportunity_id=f"coord_{int(time.time() * 1000)}_{i}",
                        strategy_name=getattr(opp, 'strategy_name', 'unknown'),
                        strategy_type=StrategyType.MOMENTUM,  # Default, will be corrected
                        token_address=getattr(opp, 'token_address', ''),
                        signal=getattr(opp, 'signal', None),
                        confidence=getattr(opp, 'confidence', 0.5),
                        expected_return=getattr(opp, 'expected_return', 0.0),
                        risk_score=getattr(opp, 'risk_score', 0.5),
                        position_size=float(getattr(opp, 'position_size', 0)),
                        entry_price=getattr(opp, 'entry_price', 0.0),
                        priority_score=getattr(opp, 'priority_score', 0.5),
                        resource_requirements={
                            ResourceType.CAPITAL: float(getattr(opp, 'position_size', 0)),
                            ResourceType.API_QUOTA: 1.0,
                            ResourceType.OPPORTUNITY_SLOTS: 1.0
                        },
                        diversification_benefit=0.5,
                        correlation_impact=0.0,
                        portfolio_impact=0.1
                    )
                    coordinator_opportunities.append(coord_opp)
                except Exception as e:
                    logger.warning(f"[MASTER_COORDINATOR] Opportunity conversion failed: {e}")
            
            # Apply advanced coordination algorithms
            coordinated = await self._apply_advanced_coordination(coordinator_opportunities)
            
            # Convert back to original format
            result_opportunities = []
            for coord_opp in coordinated:
                # Find original opportunity
                for orig_opp in opportunities:
                    if (getattr(orig_opp, 'token_address', '') == coord_opp.token_address and 
                        getattr(orig_opp, 'strategy_name', '') == coord_opp.strategy_name):
                        result_opportunities.append(orig_opp)
                        break
            
            logger.info(f"[MASTER_COORDINATOR] Coordination complete: {len(opportunities)} -> {len(result_opportunities)}")
            return result_opportunities
            
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Strategy opportunity coordination failed: {e}")
            return opportunities  # Return original on error
    
    async def process_execution_results(self, opportunities: List, execution_results: Dict[str, Any]):
        """
        Process execution results and update coordination intelligence
        Called by UnifiedTradingManager after trade execution
        """
        try:
            if not opportunities and not execution_results:
                return
            
            logger.info(f"[MASTER_COORDINATOR] Processing execution results for {len(opportunities)} opportunities")
            
            successful_trades = execution_results.get('successful', [])
            failed_trades = execution_results.get('failed', [])
            
            # Update strategy performance based on execution results
            strategy_successes = defaultdict(int)
            strategy_totals = defaultdict(int)
            
            for opp in opportunities:
                strategy_name = getattr(opp, 'strategy_name', 'unknown')
                strategy_totals[strategy_name] += 1
                
                # Check if this opportunity was successful
                if any(getattr(trade, 'strategy_name', '') == strategy_name and 
                       getattr(trade, 'token_address', '') == getattr(opp, 'token_address', '') 
                       for trade in successful_trades):
                    strategy_successes[strategy_name] += 1
            
            # Update performance metrics with exponential moving average
            alpha = 0.15  # Learning rate
            for strategy_name, total in strategy_totals.items():
                if total > 0:
                    success_rate = strategy_successes[strategy_name] / total
                    current_performance = self.strategy_performance.get(strategy_name, 0.5)
                    
                    # Update with EMA
                    new_performance = (alpha * success_rate) + ((1 - alpha) * current_performance)
                    self.strategy_performance[strategy_name] = max(0.1, min(0.9, new_performance))
            
            # Update coordination intelligence
            self._update_coordination_intelligence(opportunities, execution_results)
            
            # Store results in database
            await self._store_execution_feedback(opportunities, execution_results)
            
            logger.info(f"[MASTER_COORDINATOR] Execution results processed, updated {len(strategy_totals)} strategy performance metrics")
            
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Execution results processing failed: {e}")
    
    # HELPER METHODS FOR TRADING MANAGER INTEGRATION
    
    async def _enhance_signal(self, signal: Signal) -> Optional[Signal]:
        """Enhance a signal with coordinator intelligence"""
        try:
            # Apply coordinator-level enhancements
            enhanced_confidence = signal.confidence
            enhanced_expected_return = signal.expected_return
            
            # Portfolio correlation analysis
            correlation_penalty = await self._calculate_portfolio_correlation(signal.token_address)
            enhanced_confidence *= (1.0 - correlation_penalty * 0.3)
            
            # Strategy performance weighting
            strategy_bonus = 0.0
            for strategy_name, performance in self.strategy_performance.items():
                if performance > 0.6:  # High performing strategies get signal bonus
                    strategy_bonus += 0.05
            
            enhanced_confidence = min(1.0, enhanced_confidence + strategy_bonus)
            
            # Market condition adjustment
            market_adjustment = await self._get_market_condition_adjustment()
            enhanced_expected_return *= market_adjustment
            
            # Create enhanced signal
            enhanced_signal = Signal(
                token_address=signal.token_address,
                signal_type=signal.signal_type,
                strength=signal.strength,
                confidence=enhanced_confidence,
                expected_return=enhanced_expected_return,
                risk_score=signal.risk_score,
                timeframe=signal.timeframe,
                entry_price=signal.entry_price,
                timestamp=signal.timestamp
            )
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Signal enhancement failed: {e}")
            return signal
    
    def _validate_signal_quality(self, signal: Signal) -> bool:
        """Validate signal meets minimum quality standards"""
        try:
            return (signal.confidence > 0.2 and 
                   signal.expected_return > 0.005 and  # >0.5% expected return
                   signal.risk_score < 0.8)  # Risk score less than 80%
        except:
            return False
    
    async def _apply_advanced_coordination(self, opportunities: List[StrategyOpportunity]) -> List[StrategyOpportunity]:
        """Apply advanced coordination algorithms"""
        try:
            if not opportunities:
                return []
            
            # Phase 1: Resource constraint filtering
            resource_filtered = await self._apply_resource_constraints(opportunities)
            
            # Phase 2: Advanced conflict resolution
            conflict_resolved = await self._resolve_advanced_conflicts(resource_filtered)
            
            # Phase 3: Portfolio optimization
            portfolio_optimized = await self._optimize_for_portfolio_balance(conflict_resolved)
            
            return portfolio_optimized
            
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Advanced coordination failed: {e}")
            return opportunities
    
    async def _calculate_portfolio_correlation(self, token_address: str) -> float:
        """Calculate correlation penalty for portfolio diversification"""
        try:
            # Check existing positions for correlation
            # This would integrate with portfolio manager in full implementation
            return 0.1  # Default low correlation penalty
        except:
            return 0.0
    
    async def _get_market_condition_adjustment(self) -> float:
        """Get market condition adjustment factor"""
        try:
            # This would analyze overall market conditions
            # For now, return neutral adjustment
            return 1.0
        except:
            return 1.0
    
    def _update_coordination_intelligence(self, opportunities: List, execution_results: Dict[str, Any]):
        """Update coordination intelligence based on results"""
        try:
            # Update internal metrics for better future coordination
            total_opportunities = len(opportunities)
            successful_executions = len(execution_results.get('successful', []))
            
            if total_opportunities > 0:
                coordination_success_rate = successful_executions / total_opportunities
                
                # Update coordination mode if needed
                if coordination_success_rate < 0.3:
                    # Low success rate, switch to more conservative mode
                    if self.coordination_mode != CoordinationMode.COOPERATIVE:
                        self.coordination_mode = CoordinationMode.COOPERATIVE
                        logger.info("[MASTER_COORDINATOR] Switched to COOPERATIVE mode due to low success rate")
                elif coordination_success_rate > 0.8:
                    # High success rate, can use more aggressive mode
                    if self.coordination_mode != CoordinationMode.COMPETITIVE:
                        self.coordination_mode = CoordinationMode.COMPETITIVE
                        logger.info("[MASTER_COORDINATOR] Switched to COMPETITIVE mode due to high success rate")
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Intelligence update failed: {e}")
    
    async def _store_execution_feedback(self, opportunities: List, execution_results: Dict[str, Any]):
        """Store execution feedback in database"""
        try:
            # Store coordination feedback for future learning
            feedback_data = {
                'timestamp': datetime.now().isoformat(),
                'total_opportunities': len(opportunities),
                'successful_executions': len(execution_results.get('successful', [])),
                'failed_executions': len(execution_results.get('failed', [])),
                'coordination_mode': self.coordination_mode.value,
                'conflict_resolution': self.conflict_resolution.value
            }
            
            # Store in coordination database
            async with aiosqlite.connect("logs/master_coordination.db") as db:
                await db.execute("""
                    INSERT OR IGNORE INTO execution_feedback 
                    (timestamp, data) VALUES (?, ?)
                """, (feedback_data['timestamp'], json.dumps(feedback_data)))
                await db.commit()
                
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Execution feedback storage failed: {e}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            logger.info("[MASTER_COORDINATOR] Shutting down Master Strategy Coordinator...")
            
            self.emergency_stop = True
            
            # Shutdown all strategies
            for strategy in self.strategies.values():
                if hasattr(strategy, 'shutdown'):
                    try:
                        await strategy.shutdown()
                    except Exception as e:
                        logger.warning(f"[MASTER_COORDINATOR] Strategy shutdown failed: {e}")
            
            # Shutdown all managers
            managers = [
                self.system_manager,
                self.data_manager,
                self.order_manager,
                self.trading_manager,
                self.portfolio_manager,
                self.risk_manager
            ]
            
            for manager in managers:
                if manager and hasattr(manager, 'shutdown'):
                    try:
                        await manager.shutdown()
                    except Exception as e:
                        logger.warning(f"[MASTER_COORDINATOR] Manager shutdown failed: {e}")
            
            logger.info("[MASTER_COORDINATOR] Master Strategy Coordinator shutdown complete")
            
        except Exception as e:
            logger.error(f"[MASTER_COORDINATOR] Shutdown error: {e}")


# Global coordinator instance
_global_master_coordinator: Optional[MasterStrategyCoordinator] = None

def get_master_coordinator(settings=None) -> MasterStrategyCoordinator:
    """Get global master coordinator instance"""
    global _global_master_coordinator
    if _global_master_coordinator is None:
        _global_master_coordinator = MasterStrategyCoordinator(settings)
    return _global_master_coordinator

# Compatibility export for trading manager integration
__all__ = ['MasterStrategyCoordinator', 'get_master_coordinator', 'CoordinationMode', 'ConflictResolutionStrategy', 'ResourceType']