#!/usr/bin/env python3
"""
Dynamic Capital Allocator
Intelligent capital allocation system that dynamically distributes capital across
multiple trading strategies based on performance metrics, risk profiles, and market conditions.

Key Features:
1. Real-time performance monitoring and allocation adjustments
2. Risk-adjusted allocation based on Sharpe ratio, drawdown, and volatility
3. Strategic rebalancing with configurable thresholds and cooldown periods  
4. Market regime detection for tactical allocation shifts
5. Multi-strategy support with individual performance tracking
6. Portfolio-level risk constraints and position sizing
7. Historical allocation tracking and analytics
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import deque, defaultdict
import statistics
import sqlite3
import aiosqlite
from threading import Lock

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN" 
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"

class AllocationMode(Enum):
    CONSERVATIVE = "CONSERVATIVE"  # Risk-first allocation
    BALANCED = "BALANCED"         # Balance risk and return
    AGGRESSIVE = "AGGRESSIVE"     # Return-first allocation
    ADAPTIVE = "ADAPTIVE"         # Market regime based

@dataclass
class StrategyMetrics:
    """Real-time performance metrics for a strategy"""
    strategy_name: str
    
    # Performance metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics  
    max_drawdown: float = 0.0
    volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk
    
    # Recent performance (last 24h, 7d, 30d)
    return_24h: float = 0.0
    return_7d: float = 0.0
    return_30d: float = 0.0
    
    # Trading activity
    trades_count: int = 0
    avg_trade_duration: float = 0.0  # Hours
    last_trade_time: Optional[datetime] = None
    
    # Capital utilization
    allocated_capital: float = 0.0
    utilized_capital: float = 0.0
    max_position_size: float = 0.0
    
    # Strategy health
    is_active: bool = True
    error_rate: float = 0.0  # Error rate in last 24h
    last_update: datetime = field(default_factory=datetime.now)
    
    def risk_adjusted_score(self) -> float:
        """Calculate risk-adjusted performance score"""
        try:
            # Base score from Sharpe ratio
            base_score = max(0, self.sharpe_ratio)
            
            # Adjust for win rate
            win_rate_bonus = (self.win_rate - 0.5) * 0.5 if self.win_rate > 0.5 else 0
            
            # Penalize high drawdown
            drawdown_penalty = min(0.5, self.max_drawdown) * 2
            
            # Penalize high volatility
            vol_penalty = min(0.3, max(0, self.volatility - 0.2)) * 2
            
            # Activity bonus for active strategies
            activity_bonus = 0.1 if self.is_active and self.trades_count > 10 else 0
            
            score = base_score + win_rate_bonus - drawdown_penalty - vol_penalty + activity_bonus
            return max(0, score)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"[ALLOCATOR] Error calculating risk score for {self.strategy_name}: {e}")
            return 0.0

@dataclass 
class AllocationTarget:
    """Target allocation for a strategy"""
    strategy_name: str
    target_percentage: float  # 0.0 to 1.0
    min_allocation: float = 0.0
    max_allocation: float = 1.0
    current_allocation: float = 0.0
    rebalance_threshold: float = 0.05  # 5% drift threshold
    last_rebalance: Optional[datetime] = None
    
    def needs_rebalancing(self) -> bool:
        """Check if allocation needs rebalancing"""
        drift = abs(self.current_allocation - self.target_percentage)
        return drift >= self.rebalance_threshold

@dataclass
class AllocationEvent:
    """Record of allocation changes"""
    timestamp: datetime
    strategy_name: str
    old_allocation: float
    new_allocation: float
    total_capital: float
    reason: str
    market_regime: MarketRegime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'strategy_name': self.strategy_name,
            'old_allocation': self.old_allocation,
            'new_allocation': self.new_allocation,
            'total_capital': self.total_capital,
            'reason': self.reason,
            'market_regime': self.market_regime.value
        }

class AllocationDatabase:
    """SQLite database for allocation history and metrics"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = Lock()
    
    async def initialize(self):
        """Initialize database tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Allocation events table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS allocation_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        old_allocation REAL NOT NULL,
                        new_allocation REAL NOT NULL,
                        total_capital REAL NOT NULL,
                        reason TEXT NOT NULL,
                        market_regime TEXT NOT NULL
                    )
                """)
                
                # Strategy metrics table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        allocated_capital REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        win_rate REAL NOT NULL,
                        total_return REAL NOT NULL,
                        volatility REAL NOT NULL,
                        is_active BOOLEAN NOT NULL
                    )
                """)
                
                # Portfolio snapshots table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_capital REAL NOT NULL,
                        allocated_capital REAL NOT NULL,
                        cash_percentage REAL NOT NULL,
                        num_active_strategies INTEGER NOT NULL,
                        portfolio_sharpe REAL NOT NULL,
                        portfolio_drawdown REAL NOT NULL,
                        market_regime TEXT NOT NULL
                    )
                """)
                
                await db.commit()
                
            logger.info("[ALLOCATOR_DB] Database initialized successfully")
            
        except Exception as e:
            logger.error(f"[ALLOCATOR_DB] Database initialization failed: {e}")
            raise
    
    async def log_allocation_event(self, event: AllocationEvent):
        """Log an allocation change event"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO allocation_events 
                    (timestamp, strategy_name, old_allocation, new_allocation, 
                     total_capital, reason, market_regime)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.timestamp.isoformat(),
                    event.strategy_name,
                    event.old_allocation,
                    event.new_allocation,
                    event.total_capital,
                    event.reason,
                    event.market_regime.value
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"[ALLOCATOR_DB] Failed to log allocation event: {e}")
    
    async def log_strategy_metrics(self, metrics: StrategyMetrics):
        """Log strategy performance metrics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO strategy_metrics
                    (timestamp, strategy_name, allocated_capital, sharpe_ratio,
                     max_drawdown, win_rate, total_return, volatility, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    metrics.strategy_name,
                    metrics.allocated_capital,
                    metrics.sharpe_ratio,
                    metrics.max_drawdown,
                    metrics.win_rate,
                    metrics.total_return,
                    metrics.volatility,
                    metrics.is_active
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"[ALLOCATOR_DB] Failed to log strategy metrics: {e}")
    
    async def get_allocation_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent allocation history"""
        try:
            since = datetime.now() - timedelta(hours=hours)
            
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT * FROM allocation_events 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (since.isoformat(),)) as cursor:
                    
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    return [dict(zip(columns, row)) for row in rows]
                    
        except Exception as e:
            logger.error(f"[ALLOCATOR_DB] Failed to get allocation history: {e}")
            return []

class MarketRegimeDetector:
    """Detects current market regime for allocation adjustments"""
    
    def __init__(self, lookback_hours: int = 24):
        self.lookback_hours = lookback_hours
        self.price_history: deque = deque(maxlen=100)  # Last 100 price points
        self.volatility_history: deque = deque(maxlen=50)  # Last 50 volatility measures
    
    def update_market_data(self, price: float, volume: float):
        """Update market data for regime detection"""
        try:
            timestamp = datetime.now()
            self.price_history.append({'price': price, 'volume': volume, 'timestamp': timestamp})
            
            # Calculate rolling volatility if we have enough data
            if len(self.price_history) >= 10:
                recent_prices = [p['price'] for p in list(self.price_history)[-10:]]
                returns = np.diff(recent_prices) / recent_prices[:-1]
                volatility = np.std(returns)
                self.volatility_history.append(volatility)
                
        except Exception as e:
            logger.error(f"[REGIME_DETECTOR] Error updating market data: {e}")
    
    def detect_regime(self) -> MarketRegime:
        """Detect current market regime"""
        try:
            if len(self.price_history) < 20:
                return MarketRegime.RANGING  # Default
            
            # Get recent price data
            prices = [p['price'] for p in self.price_history]
            recent_prices = prices[-20:]  # Last 20 data points
            
            # Calculate trend
            x = np.arange(len(recent_prices))
            trend_slope = np.polyfit(x, recent_prices, 1)[0]
            price_change_pct = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # Calculate volatility
            returns = np.diff(recent_prices) / recent_prices[:-1]
            current_volatility = np.std(returns)
            avg_volatility = np.mean(list(self.volatility_history)) if self.volatility_history else current_volatility
            
            # Regime detection logic
            if current_volatility > avg_volatility * 1.5:
                return MarketRegime.HIGH_VOLATILITY
            elif current_volatility < avg_volatility * 0.5:
                return MarketRegime.LOW_VOLATILITY
            elif trend_slope > 0 and price_change_pct > 0.02:  # 2% upward trend
                return MarketRegime.TRENDING_UP
            elif trend_slope < 0 and price_change_pct < -0.02:  # 2% downward trend
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            logger.error(f"[REGIME_DETECTOR] Error detecting regime: {e}")
            return MarketRegime.RANGING

class DynamicCapitalAllocator:
    """
    Intelligent dynamic capital allocation system for multi-strategy trading
    """
    
    def __init__(self, settings: Any):
        self.settings = settings
        self.total_capital = float(settings.INITIAL_PAPER_BALANCE) if settings.PAPER_TRADING else 10000.0
        
        # Core components
        self.strategies: Dict[str, StrategyMetrics] = {}
        self.allocation_targets: Dict[str, AllocationTarget] = {}
        # Use logs directory relative to project root
        import os
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.db = AllocationDatabase(f"{log_dir}/allocations.db")
        self.regime_detector = MarketRegimeDetector()
        
        # Allocation settings
        self.allocation_mode = AllocationMode.BALANCED
        self.rebalance_cooldown = timedelta(hours=1)  # Minimum time between rebalances
        self.min_allocation = 0.05  # 5% minimum allocation
        self.max_allocation = 0.40  # 40% maximum allocation per strategy
        self.cash_reserve = 0.20    # 20% cash reserve
        
        # Enhanced Exposure Constraints
        self.max_strategy_allocation = 0.50  # Hard cap: 50% max per strategy
        self.sector_exposure_limits = {
            'momentum': 0.60,      # Max 60% in momentum strategies
            'mean_reversion': 0.60, # Max 60% in mean reversion strategies
            'arbitrage': 0.50,     # Max 50% in arbitrage strategies
            'grid_trading': 0.40   # Max 40% in grid trading strategies
        }
        self.token_exposure_limits = {
            'SOL': 0.70,           # Max 70% exposure to SOL-based strategies
            'BTC': 0.30,           # Max 30% exposure to BTC-based strategies
            'ETH': 0.30            # Max 30% exposure to ETH-based strategies
        }
        
        # Market regime allocation profiles
        self.regime_profiles = {
            MarketRegime.TRENDING_UP: {'momentum_weight': 0.6, 'mean_reversion_weight': 0.2, 'arbitrage_weight': 0.2},
            MarketRegime.TRENDING_DOWN: {'momentum_weight': 0.3, 'mean_reversion_weight': 0.4, 'arbitrage_weight': 0.3},
            MarketRegime.RANGING: {'momentum_weight': 0.2, 'mean_reversion_weight': 0.5, 'arbitrage_weight': 0.3},
            MarketRegime.HIGH_VOLATILITY: {'momentum_weight': 0.3, 'mean_reversion_weight': 0.3, 'arbitrage_weight': 0.4},
            MarketRegime.LOW_VOLATILITY: {'momentum_weight': 0.5, 'mean_reversion_weight': 0.3, 'arbitrage_weight': 0.2}
        }
        
        # History tracking
        self.allocation_history: List[AllocationEvent] = []
        self.last_rebalance = datetime.now()
        self.metrics_update_interval = timedelta(minutes=5)
        self.last_metrics_update = datetime.now()
        
        logger.info(f"[ALLOCATOR] Dynamic Capital Allocator initialized with {self.total_capital} total capital")
    
    async def start(self):
        """Start the dynamic capital allocator"""
        try:
            await self.db.initialize()
            
            # Initialize default strategies if none exist
            if not self.strategies:
                await self._initialize_default_strategies()
            
            logger.info("[ALLOCATOR] Dynamic Capital Allocator started successfully")
            
        except Exception as e:
            logger.error(f"[ALLOCATOR] Failed to start allocator: {e}")
            raise
    
    async def stop(self):
        """Stop the dynamic capital allocator"""
        try:
            # Save final state
            await self._save_portfolio_snapshot()
            logger.info("[ALLOCATOR] Dynamic Capital Allocator stopped")
            
        except Exception as e:
            logger.error(f"[ALLOCATOR] Error during allocator shutdown: {e}")
    
    async def register_strategy(
        self,
        strategy_name: str,
        initial_allocation: float = 0.1,
        min_allocation: float = 0.05,
        max_allocation: float = 0.40
    ) -> bool:
        """Register a new strategy with the allocator"""
        try:
            if strategy_name in self.strategies:
                logger.warning(f"[ALLOCATOR] Strategy {strategy_name} already registered")
                return False
            
            # Create strategy metrics
            self.strategies[strategy_name] = StrategyMetrics(
                strategy_name=strategy_name,
                allocated_capital=self.total_capital * initial_allocation,
                is_active=True
            )
            
            # Create allocation target
            self.allocation_targets[strategy_name] = AllocationTarget(
                strategy_name=strategy_name,
                target_percentage=initial_allocation,
                min_allocation=min_allocation,
                max_allocation=max_allocation,
                current_allocation=initial_allocation
            )
            
            logger.info(f"[ALLOCATOR] Registered strategy '{strategy_name}' with {initial_allocation:.1%} allocation")
            return True
            
        except Exception as e:
            logger.error(f"[ALLOCATOR] Failed to register strategy {strategy_name}: {e}")
            return False
    
    async def update_strategy_metrics(self, strategy_name: str, **metrics_kwargs) -> bool:
        """Update performance metrics for a strategy"""
        try:
            if strategy_name not in self.strategies:
                logger.warning(f"[ALLOCATOR] Unknown strategy: {strategy_name}")
                return False
            
            strategy = self.strategies[strategy_name]
            
            # Update provided metrics
            for key, value in metrics_kwargs.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
            
            strategy.last_update = datetime.now()
            
            # Log to database periodically
            if datetime.now() - self.last_metrics_update >= self.metrics_update_interval:
                await self.db.log_strategy_metrics(strategy)
                self.last_metrics_update = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"[ALLOCATOR] Failed to update metrics for {strategy_name}: {e}")
            return False
    
    async def rebalance_portfolio(self, force: bool = False) -> Dict[str, Any]:
        """Rebalance portfolio allocations based on performance"""
        try:
            # Check cooldown period
            if not force and datetime.now() - self.last_rebalance < self.rebalance_cooldown:
                return {'status': 'skipped', 'reason': 'cooldown_active'}
            
            # Get current market regime
            current_regime = self.regime_detector.detect_regime()
            
            # Calculate new target allocations
            new_targets = await self._calculate_optimal_allocations(current_regime)
            
            # Identify strategies needing rebalancing
            rebalance_actions = []
            total_rebalanced = 0.0
            
            for strategy_name, target in self.allocation_targets.items():
                new_target = new_targets.get(strategy_name, target.target_percentage)
                
                if abs(new_target - target.current_allocation) >= target.rebalance_threshold or force:
                    old_allocation = target.current_allocation
                    target.target_percentage = new_target
                    target.current_allocation = new_target
                    target.last_rebalance = datetime.now()
                    
                    # Update strategy capital
                    if strategy_name in self.strategies:
                        old_capital = self.strategies[strategy_name].allocated_capital
                        new_capital = self.total_capital * new_target
                        self.strategies[strategy_name].allocated_capital = new_capital
                        
                        rebalance_actions.append({
                            'strategy': strategy_name,
                            'old_allocation': old_allocation,
                            'new_allocation': new_target,
                            'old_capital': old_capital,
                            'new_capital': new_capital,
                            'change': new_target - old_allocation
                        })
                        
                        total_rebalanced += abs(new_target - old_allocation)
                        
                        # Log allocation event
                        event = AllocationEvent(
                            timestamp=datetime.now(),
                            strategy_name=strategy_name,
                            old_allocation=old_allocation,
                            new_allocation=new_target,
                            total_capital=self.total_capital,
                            reason='performance_rebalance',
                            market_regime=current_regime
                        )
                        
                        await self.db.log_allocation_event(event)
                        self.allocation_history.append(event)
            
            self.last_rebalance = datetime.now()
            
            # Save portfolio snapshot
            await self._save_portfolio_snapshot()
            
            result = {
                'status': 'completed',
                'rebalanced_strategies': len(rebalance_actions),
                'total_rebalanced': total_rebalanced,
                'actions': rebalance_actions,
                'market_regime': current_regime.value,
                'timestamp': self.last_rebalance.isoformat()
            }
            
            logger.info(f"[ALLOCATOR] Portfolio rebalanced: {len(rebalance_actions)} strategies, "
                       f"{total_rebalanced:.2%} total rebalanced, regime: {current_regime.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"[ALLOCATOR] Portfolio rebalancing failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _calculate_optimal_allocations(self, market_regime: MarketRegime) -> Dict[str, float]:
        """Calculate optimal allocations based on performance and market regime"""
        try:
            if not self.strategies:
                return {}
            
            # Get regime-specific weights if available
            regime_weights = self.regime_profiles.get(market_regime, {})
            
            # Calculate risk-adjusted scores for each strategy
            strategy_scores = {}
            total_score = 0.0
            
            for name, strategy in self.strategies.items():
                if not strategy.is_active:
                    strategy_scores[name] = 0.0
                    continue
                
                base_score = strategy.risk_adjusted_score()
                
                # Apply regime-specific adjustments
                regime_multiplier = 1.0
                if name in regime_weights:
                    regime_multiplier = regime_weights[name]
                elif 'momentum' in name.lower():
                    regime_multiplier = regime_weights.get('momentum_weight', 1.0)
                elif 'mean_reversion' in name.lower() or 'grid' in name.lower():
                    regime_multiplier = regime_weights.get('mean_reversion_weight', 1.0) 
                elif 'arbitrage' in name.lower():
                    regime_multiplier = regime_weights.get('arbitrage_weight', 1.0)
                
                adjusted_score = base_score * regime_multiplier
                strategy_scores[name] = max(0.1, adjusted_score)  # Minimum score to prevent zero allocation
                total_score += adjusted_score
            
            # Calculate allocations based on scores
            new_allocations = {}
            available_capital = 1.0 - self.cash_reserve  # Reserve cash
            
            for name, score in strategy_scores.items():
                if total_score > 0:
                    raw_allocation = (score / total_score) * available_capital
                else:
                    raw_allocation = available_capital / len(strategy_scores)
                
                # Apply min/max constraints with enhanced exposure limits
                target = self.allocation_targets.get(name)
                min_alloc = target.min_allocation if target else self.min_allocation
                max_alloc = min(
                    target.max_allocation if target else self.max_allocation,
                    self.max_strategy_allocation  # Enforce hard cap
                )
                
                allocation = np.clip(raw_allocation, min_alloc, max_alloc)
                new_allocations[name] = allocation
            
            # Apply sector and token exposure constraints
            new_allocations = self._enforce_exposure_constraints(new_allocations)
            
            # Normalize to ensure total doesn't exceed available capital
            total_allocated = sum(new_allocations.values())
            if total_allocated > available_capital:
                normalization_factor = available_capital / total_allocated
                new_allocations = {name: alloc * normalization_factor for name, alloc in new_allocations.items()}
            
            logger.debug(f"[ALLOCATOR] Calculated allocations for {market_regime.value}: {new_allocations}")
            return new_allocations
            
        except Exception as e:
            logger.error(f"[ALLOCATOR] Error calculating optimal allocations: {e}")
            return {}
    
    def _enforce_exposure_constraints(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Enforce sector and token exposure constraints"""
        try:
            constrained_allocations = allocations.copy()
            
            # Enforce sector exposure limits
            sector_exposures = self._calculate_sector_exposures(constrained_allocations)
            
            for sector, exposure in sector_exposures.items():
                if exposure > self.sector_exposure_limits.get(sector, 1.0):
                    # Reduce allocations for strategies in this sector
                    sector_strategies = self._get_strategies_by_sector(sector)
                    reduction_factor = self.sector_exposure_limits[sector] / exposure
                    
                    for strategy_name in sector_strategies:
                        if strategy_name in constrained_allocations:
                            constrained_allocations[strategy_name] *= reduction_factor
                    
                    logger.info(f"[ALLOCATOR] Reduced {sector} sector exposure from {exposure:.1%} to {self.sector_exposure_limits[sector]:.1%}")
            
            # Enforce token exposure limits
            token_exposures = self._calculate_token_exposures(constrained_allocations)
            
            for token, exposure in token_exposures.items():
                if exposure > self.token_exposure_limits.get(token, 1.0):
                    # Reduce allocations for strategies exposed to this token
                    token_strategies = self._get_strategies_by_token(token)
                    reduction_factor = self.token_exposure_limits[token] / exposure
                    
                    for strategy_name in token_strategies:
                        if strategy_name in constrained_allocations:
                            constrained_allocations[strategy_name] *= reduction_factor
                    
                    logger.info(f"[ALLOCATOR] Reduced {token} token exposure from {exposure:.1%} to {self.token_exposure_limits[token]:.1%}")
            
            return constrained_allocations
            
        except Exception as e:
            logger.error(f"[ALLOCATOR] Error enforcing exposure constraints: {e}")
            return allocations
    
    def _calculate_sector_exposures(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Calculate current sector exposures"""
        sector_exposures = defaultdict(float)
        
        for strategy_name, allocation in allocations.items():
            # Classify strategy by sector based on name
            if 'momentum' in strategy_name.lower():
                sector_exposures['momentum'] += allocation
            elif 'mean_reversion' in strategy_name.lower() or 'grid' in strategy_name.lower():
                sector_exposures['mean_reversion'] += allocation
            elif 'arbitrage' in strategy_name.lower():
                sector_exposures['arbitrage'] += allocation
            elif 'grid_trading' in strategy_name.lower():
                sector_exposures['grid_trading'] += allocation
        
        return dict(sector_exposures)
    
    def _calculate_token_exposures(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Calculate current token exposures"""
        token_exposures = defaultdict(float)
        
        for strategy_name, allocation in allocations.items():
            # For Solana-based trading, most strategies will have SOL exposure
            # In production, this would be based on actual token holdings/exposure
            if 'sol' in strategy_name.lower() or allocation > 0:
                token_exposures['SOL'] += allocation  # Assume SOL exposure for all active strategies
        
        return dict(token_exposures)
    
    def _get_strategies_by_sector(self, sector: str) -> List[str]:
        """Get list of strategies in a given sector"""
        strategies = []
        
        for strategy_name in self.strategies.keys():
            if sector == 'momentum' and 'momentum' in strategy_name.lower():
                strategies.append(strategy_name)
            elif sector == 'mean_reversion' and ('mean_reversion' in strategy_name.lower() or 'grid' in strategy_name.lower()):
                strategies.append(strategy_name)
            elif sector == 'arbitrage' and 'arbitrage' in strategy_name.lower():
                strategies.append(strategy_name)
            elif sector == 'grid_trading' and 'grid_trading' in strategy_name.lower():
                strategies.append(strategy_name)
        
        return strategies
    
    def _get_strategies_by_token(self, token: str) -> List[str]:
        """Get list of strategies exposed to a given token"""
        strategies = []
        
        for strategy_name in self.strategies.keys():
            # For Solana trading, assume all active strategies have SOL exposure
            if token == 'SOL':
                strategies.append(strategy_name)
            # Add logic for other tokens as needed
        
        return strategies
    
    async def get_current_allocations(self) -> Dict[str, Any]:
        """Get current portfolio allocations and metrics"""
        try:
            current_regime = self.regime_detector.detect_regime()
            
            allocations = {}
            total_allocated = 0.0
            active_strategies = 0
            
            for name, strategy in self.strategies.items():
                allocation_pct = strategy.allocated_capital / self.total_capital if self.total_capital > 0 else 0
                
                allocations[name] = {
                    'allocated_capital': strategy.allocated_capital,
                    'allocation_percentage': allocation_pct,
                    'sharpe_ratio': strategy.sharpe_ratio,
                    'win_rate': strategy.win_rate,
                    'max_drawdown': strategy.max_drawdown,
                    'is_active': strategy.is_active,
                    'risk_score': strategy.risk_adjusted_score(),
                    'last_update': strategy.last_update.isoformat()
                }
                
                if strategy.is_active:
                    total_allocated += allocation_pct
                    active_strategies += 1
            
            cash_percentage = 1.0 - total_allocated
            
            portfolio_metrics = await self._calculate_portfolio_metrics()
            
            return {
                'total_capital': self.total_capital,
                'total_allocated_percentage': total_allocated,
                'cash_percentage': cash_percentage,
                'active_strategies': active_strategies,
                'market_regime': current_regime.value,
                'allocations': allocations,
                'portfolio_metrics': portfolio_metrics,
                'last_rebalance': self.last_rebalance.isoformat(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"[ALLOCATOR] Error getting current allocations: {e}")
            return {}
    
    async def _calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio-level performance metrics"""
        try:
            if not self.strategies:
                return {}
            
            active_strategies = [s for s in self.strategies.values() if s.is_active]
            if not active_strategies:
                return {}
            
            # Weight metrics by allocation
            total_weight = sum(s.allocated_capital for s in active_strategies)
            if total_weight == 0:
                return {}
            
            weighted_sharpe = sum(s.sharpe_ratio * s.allocated_capital / total_weight for s in active_strategies)
            weighted_return = sum(s.total_return * s.allocated_capital / total_weight for s in active_strategies)
            weighted_drawdown = sum(s.max_drawdown * s.allocated_capital / total_weight for s in active_strategies)
            weighted_win_rate = sum(s.win_rate * s.allocated_capital / total_weight for s in active_strategies)
            
            return {
                'portfolio_sharpe': weighted_sharpe,
                'portfolio_return': weighted_return,
                'portfolio_drawdown': weighted_drawdown,
                'portfolio_win_rate': weighted_win_rate,
                'diversification_ratio': len(active_strategies),
                'capital_utilization': total_weight / self.total_capital
            }
            
        except Exception as e:
            logger.error(f"[ALLOCATOR] Error calculating portfolio metrics: {e}")
            return {}
    
    async def _save_portfolio_snapshot(self):
        """Save current portfolio state to database"""
        try:
            current_regime = self.regime_detector.detect_regime()
            portfolio_metrics = await self._calculate_portfolio_metrics()
            
            async with aiosqlite.connect(self.db.db_path) as db:
                await db.execute("""
                    INSERT INTO portfolio_snapshots
                    (timestamp, total_capital, allocated_capital, cash_percentage,
                     num_active_strategies, portfolio_sharpe, portfolio_drawdown, market_regime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    self.total_capital,
                    sum(s.allocated_capital for s in self.strategies.values()),
                    1.0 - sum(s.allocated_capital for s in self.strategies.values() if s.is_active) / self.total_capital,
                    len([s for s in self.strategies.values() if s.is_active]),
                    portfolio_metrics.get('portfolio_sharpe', 0.0),
                    portfolio_metrics.get('portfolio_drawdown', 0.0),
                    current_regime.value
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"[ALLOCATOR] Failed to save portfolio snapshot: {e}")
    
    async def _initialize_default_strategies(self):
        """Initialize with default strategy allocations"""
        try:
            default_strategies = [
                {'name': 'momentum_strategy', 'allocation': 0.25},
                {'name': 'grid_trading_strategy', 'allocation': 0.25},
                {'name': 'arbitrage_strategy', 'allocation': 0.20},
            ]
            
            for strategy in default_strategies:
                await self.register_strategy(
                    strategy_name=strategy['name'],
                    initial_allocation=strategy['allocation']
                )
            
            logger.info(f"[ALLOCATOR] Initialized {len(default_strategies)} default strategies")
            
        except Exception as e:
            logger.error(f"[ALLOCATOR] Failed to initialize default strategies: {e}")
    
    async def update_market_data(self, price: float, volume: float):
        """Update market data for regime detection"""
        self.regime_detector.update_market_data(price, volume)
    
    async def get_allocation_recommendations(self) -> List[Dict[str, Any]]:
        """Get allocation recommendations based on current performance"""
        try:
            recommendations = []
            current_regime = self.regime_detector.detect_regime()
            
            for name, strategy in self.strategies.items():
                target = self.allocation_targets.get(name)
                if not target:
                    continue
                
                risk_score = strategy.risk_adjusted_score()
                
                # Generate recommendations
                if not strategy.is_active and risk_score > 0.5:
                    recommendations.append({
                        'strategy': name,
                        'action': 'ACTIVATE',
                        'reason': f'High risk score ({risk_score:.2f}) suggests reactivation',
                        'priority': 'medium'
                    })
                elif strategy.is_active and risk_score < 0.2:
                    recommendations.append({
                        'strategy': name,
                        'action': 'REDUCE_ALLOCATION',
                        'reason': f'Low risk score ({risk_score:.2f}) suggests reducing allocation',
                        'priority': 'high'
                    })
                elif strategy.max_drawdown > 0.15:
                    recommendations.append({
                        'strategy': name,
                        'action': 'RISK_REVIEW',
                        'reason': f'High drawdown ({strategy.max_drawdown:.1%}) needs review',
                        'priority': 'high'
                    })
                elif strategy.sharpe_ratio > 2.0 and target.current_allocation < target.max_allocation:
                    recommendations.append({
                        'strategy': name,
                        'action': 'INCREASE_ALLOCATION',
                        'reason': f'Excellent Sharpe ratio ({strategy.sharpe_ratio:.2f}) suggests more allocation',
                        'priority': 'medium'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"[ALLOCATOR] Error generating recommendations: {e}")
            return []

    def get_summary(self) -> Dict[str, Any]:
        """Get allocator summary for dashboard/monitoring"""
        try:
            active_count = len([s for s in self.strategies.values() if s.is_active])
            total_allocated = sum(s.allocated_capital for s in self.strategies.values())
            
            return {
                'total_capital': self.total_capital,
                'strategies_count': len(self.strategies),
                'active_strategies': active_count,
                'total_allocated': total_allocated,
                'cash_available': self.total_capital - total_allocated,
                'last_rebalance': self.last_rebalance.isoformat(),
                'allocation_mode': self.allocation_mode.value,
                'market_regime': self.regime_detector.detect_regime().value
            }
            
        except Exception as e:
            logger.error(f"[ALLOCATOR] Error generating summary: {e}")
            return {}