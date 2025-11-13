#!/usr/bin/env python3
"""
Cross-Strategy Coordination System 🎯

This module provides sophisticated coordination between multiple trading strategies:
- Position conflict prevention (no opposing positions)
- Dynamic capital allocation based on market conditions
- Real-time strategy performance comparison
- Strategy priority management
- Risk-based strategy selection
- Portfolio optimization across strategies

Integrates: Momentum, Mean Reversion, and Grid Trading strategies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Available trading strategies"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"  
    GRID_TRADING = "grid_trading"

class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

@dataclass
class StrategyAllocation:
    """Capital allocation for a strategy"""
    strategy_type: StrategyType
    allocated_capital: float
    allocation_percentage: float
    current_exposure: float
    available_capital: float
    performance_score: float
    active_positions: int
    
@dataclass
class PositionConflict:
    """Detected position conflict between strategies"""
    token_address: str
    existing_strategy: StrategyType
    proposed_strategy: StrategyType
    conflict_type: str  # 'opposing_direction', 'over_allocation', 'resource_conflict'
    severity: int  # 1-5, 5 = critical
    resolution_action: str

@dataclass
class StrategyPerformance:
    """Performance tracking for each strategy"""
    strategy_type: StrategyType
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_pnl_per_trade: float
    sharpe_ratio: float
    max_drawdown: float
    current_exposure: float
    last_updated: datetime
    performance_trend: str  # 'improving', 'stable', 'declining'
    performance_score: float = 1.0  # Overall performance score (0.1-2.0)

class StrategyCoordinator:
    """Advanced cross-strategy coordination system"""
    
    def __init__(self, settings, analytics=None):
        self.settings = settings
        self.analytics = analytics
        
        # Strategy allocations (configurable percentages)
        self.strategy_allocations = {
            StrategyType.MOMENTUM: StrategyAllocation(
                strategy_type=StrategyType.MOMENTUM,
                allocated_capital=self.settings.PORTFOLIO_VALUE * 0.6,  # 60% momentum
                allocation_percentage=0.6,
                current_exposure=0.0,
                available_capital=0.0,
                performance_score=1.0,
                active_positions=0
            ),
            StrategyType.MEAN_REVERSION: StrategyAllocation(
                strategy_type=StrategyType.MEAN_REVERSION,
                allocated_capital=self.settings.PORTFOLIO_VALUE * 0.3,  # 30% mean reversion
                allocation_percentage=0.3,
                current_exposure=0.0,
                available_capital=0.0,
                performance_score=1.0,
                active_positions=0
            ),
            StrategyType.GRID_TRADING: StrategyAllocation(
                strategy_type=StrategyType.GRID_TRADING,
                allocated_capital=self.settings.PORTFOLIO_VALUE * 0.1,  # 10% grid trading
                allocation_percentage=0.1,
                current_exposure=0.0,
                available_capital=0.0,
                performance_score=1.0,
                active_positions=0
            )
        }
        
        # Active positions by strategy
        self.active_positions = {
            strategy: {} for strategy in StrategyType
        }
        
        # Performance tracking
        self.strategy_performance = {
            strategy: StrategyPerformance(
                strategy_type=strategy,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                average_pnl_per_trade=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                current_exposure=0.0,
                last_updated=datetime.now(),
                performance_trend='stable'
            ) for strategy in StrategyType
        }
        
        # Market regime detection
        self.current_market_regime = MarketRegime.UNKNOWN
        self.market_regime_confidence = 0.5
        
        # Coordination parameters
        self.MAX_POSITION_OVERLAP = 0.02  # 2% max overlap between strategies
        self.PERFORMANCE_WINDOW_DAYS = 7  # Look back 7 days for performance
        self.REBALANCE_THRESHOLD = 0.05  # 5% allocation drift triggers rebalance
        
        logger.info("Strategy Coordinator initialized with cross-strategy management")
    
    def detect_market_regime(self, price_history: List[float], volume_history: List[float]) -> Tuple[MarketRegime, float]:
        """Detect current market regime"""
        
        if len(price_history) < 20:
            return MarketRegime.UNKNOWN, 0.5
        
        prices = np.array(price_history)
        volumes = np.array(volume_history) if volume_history else np.ones(len(prices))
        
        # Calculate trend indicators
        short_ma = np.mean(prices[-5:])  # 5-period moving average
        medium_ma = np.mean(prices[-10:])  # 10-period moving average
        long_ma = np.mean(prices[-20:])  # 20-period moving average
        
        # Calculate volatility
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Calculate range characteristics
        recent_high = np.max(prices[-10:])
        recent_low = np.min(prices[-10:])
        range_ratio = (recent_high - recent_low) / recent_low
        
        # Trend detection
        trend_strength = abs((short_ma - long_ma) / long_ma)
        is_uptrend = short_ma > medium_ma > long_ma
        is_downtrend = short_ma < medium_ma < long_ma
        
        # Regime classification with confidence
        if volatility > 0.15:  # Very high volatility
            regime = MarketRegime.VOLATILE
            confidence = min(volatility / 0.1, 1.0)
        elif trend_strength > 0.05:  # Strong trend
            if is_uptrend:
                regime = MarketRegime.TRENDING_UP
            elif is_downtrend:
                regime = MarketRegime.TRENDING_DOWN
            else:
                regime = MarketRegime.VOLATILE
            confidence = min(trend_strength / 0.03, 1.0)
        elif range_ratio < 0.1:  # Low range, sideways market
            regime = MarketRegime.RANGING
            confidence = 1.0 - range_ratio / 0.1
        else:
            regime = MarketRegime.UNKNOWN
            confidence = 0.5
        
        return regime, min(confidence, 1.0)
    
    def get_optimal_strategy_for_regime(self, market_regime: MarketRegime) -> List[Tuple[StrategyType, float]]:
        """Get optimal strategy allocation for current market regime"""
        
        if market_regime == MarketRegime.TRENDING_UP:
            return [
                (StrategyType.MOMENTUM, 0.8),      # 80% momentum in uptrend
                (StrategyType.GRID_TRADING, 0.15), # 15% grid for sideways periods
                (StrategyType.MEAN_REVERSION, 0.05) # 5% mean reversion
            ]
        elif market_regime == MarketRegime.TRENDING_DOWN:
            return [
                (StrategyType.MEAN_REVERSION, 0.6), # 60% mean reversion in downtrend
                (StrategyType.MOMENTUM, 0.3),       # 30% momentum for bounces
                (StrategyType.GRID_TRADING, 0.1)    # 10% grid trading
            ]
        elif market_regime == MarketRegime.RANGING:
            return [
                (StrategyType.GRID_TRADING, 0.6),   # 60% grid in ranging market
                (StrategyType.MEAN_REVERSION, 0.3), # 30% mean reversion
                (StrategyType.MOMENTUM, 0.1)        # 10% momentum for breakouts
            ]
        elif market_regime == MarketRegime.VOLATILE:
            return [
                (StrategyType.MEAN_REVERSION, 0.5), # 50% mean reversion in volatility
                (StrategyType.MOMENTUM, 0.3),       # 30% momentum for quick trades
                (StrategyType.GRID_TRADING, 0.2)    # 20% grid for opportunities
            ]
        else:  # UNKNOWN
            return [
                (StrategyType.MOMENTUM, 0.5),       # 50% momentum (default)
                (StrategyType.MEAN_REVERSION, 0.3), # 30% mean reversion
                (StrategyType.GRID_TRADING, 0.2)    # 20% grid trading
            ]
    
    def check_position_conflicts(self, token_address: str, proposed_strategy: StrategyType, 
                                proposed_position_size: float) -> List[PositionConflict]:
        """Check for conflicts with existing positions"""
        
        conflicts = []
        
        # Check if token already has positions in other strategies
        for strategy, positions in self.active_positions.items():
            if strategy == proposed_strategy:
                continue
            
            if token_address in positions:
                existing_position = positions[token_address]
                
                # Check for opposing directions (simplified logic)
                conflict_type = "opposing_direction"
                severity = 3
                
                # Check for over-allocation
                total_exposure = existing_position.get('size', 0) + proposed_position_size
                max_single_token_exposure = self.settings.MAX_POSITION_SIZE
                
                if total_exposure > max_single_token_exposure:
                    conflict_type = "over_allocation"
                    severity = 4
                
                # Determine resolution action
                if severity >= 4:
                    resolution_action = "reject_new_position"
                elif existing_position.get('performance', 0) > 0:
                    resolution_action = "reduce_new_position_size"
                else:
                    resolution_action = "close_existing_position"
                
                conflicts.append(PositionConflict(
                    token_address=token_address,
                    existing_strategy=strategy,
                    proposed_strategy=proposed_strategy,
                    conflict_type=conflict_type,
                    severity=severity,
                    resolution_action=resolution_action
                ))
        
        return conflicts
    
    def resolve_position_conflicts(self, conflicts: List[PositionConflict]) -> Dict[str, Any]:
        """Resolve position conflicts between strategies"""
        
        if not conflicts:
            return {'action': 'proceed', 'modifications': []}
        
        resolutions = []
        
        for conflict in conflicts:
            if conflict.severity >= 4:  # Critical conflict
                resolutions.append({
                    'action': 'reject',
                    'reason': f"Critical conflict: {conflict.conflict_type}",
                    'token': conflict.token_address
                })
            elif conflict.resolution_action == "reduce_new_position_size":
                resolutions.append({
                    'action': 'modify',
                    'modification': 'reduce_position_size',
                    'new_size_multiplier': 0.5,  # Reduce to 50%
                    'token': conflict.token_address
                })
            elif conflict.resolution_action == "close_existing_position":
                resolutions.append({
                    'action': 'close_existing',
                    'strategy': conflict.existing_strategy,
                    'token': conflict.token_address
                })
        
        return {'action': 'resolve', 'resolutions': resolutions}
    
    def allocate_capital_to_strategies(self, market_regime: MarketRegime, 
                                     performance_scores: Dict[StrategyType, float]) -> Dict[StrategyType, float]:
        """Dynamically allocate capital based on market regime and performance"""
        
        # Get optimal allocation for current regime
        optimal_allocations = self.get_optimal_strategy_for_regime(market_regime)
        
        # Adjust based on recent performance
        adjusted_allocations = {}
        total_performance = sum(performance_scores.values())
        
        for strategy, base_allocation in optimal_allocations:
            performance_weight = performance_scores.get(strategy, 1.0)
            
            # Blend regime-based allocation with performance
            regime_weight = 0.7  # 70% regime, 30% performance
            performance_adjustment = (performance_weight / total_performance) * 3  # Normalize
            
            final_allocation = (
                base_allocation * regime_weight +
                (base_allocation * performance_adjustment * (1 - regime_weight))
            )
            
            adjusted_allocations[strategy] = min(final_allocation, 0.8)  # Cap at 80%
        
        # Normalize to 100%
        total_allocation = sum(adjusted_allocations.values())
        if total_allocation > 0:
            adjusted_allocations = {
                strategy: allocation / total_allocation
                for strategy, allocation in adjusted_allocations.items()
            }
        
        return adjusted_allocations
    
    def update_strategy_performance(self, strategy: StrategyType, trade_result: Dict):
        """Update performance tracking for a strategy"""
        
        perf = self.strategy_performance[strategy]
        
        # Update trade counts
        perf.total_trades += 1
        pnl = trade_result.get('pnl', 0)
        
        if pnl > 0:
            perf.winning_trades += 1
        else:
            perf.losing_trades += 1
        
        # Update metrics
        perf.win_rate = perf.winning_trades / perf.total_trades if perf.total_trades > 0 else 0
        perf.total_pnl += pnl
        perf.average_pnl_per_trade = perf.total_pnl / perf.total_trades if perf.total_trades > 0 else 0
        perf.last_updated = datetime.now()
        
        # Update performance score (simplified)
        base_score = perf.win_rate * 2 + (perf.average_pnl_per_trade * 100)
        perf.performance_score = max(0.1, min(base_score, 2.0))  # Scale 0.1-2.0
        
        # Update trend (simplified)
        if perf.total_trades >= 10:
            recent_performance = perf.average_pnl_per_trade
            if recent_performance > 0.01:  # > 1% average
                perf.performance_trend = 'improving'
            elif recent_performance < -0.01:  # < -1% average
                perf.performance_trend = 'declining'
            else:
                perf.performance_trend = 'stable'
    
    def get_strategy_recommendation(self, token_address: str, signal_data: Dict, 
                                  market_data: Dict) -> Tuple[StrategyType, float, Dict]:
        """Get recommended strategy for a token based on all factors"""
        
        # Detect market regime
        price_history = market_data.get('price_history', [])
        volume_history = market_data.get('volume_history', [])
        
        market_regime, confidence = self.detect_market_regime(price_history, volume_history)
        
        # Get performance scores
        performance_scores = {
            strategy: perf.performance_score
            for strategy, perf in self.strategy_performance.items()
        }
        
        # Calculate optimal allocations
        optimal_allocations = self.allocate_capital_to_strategies(market_regime, performance_scores)
        
        # Select best strategy based on signal characteristics
        momentum_score = abs(signal_data.get('momentum_strength', 0)) * 0.4
        volatility_score = signal_data.get('volatility', 0.1) * 0.3
        range_score = 1.0 - volatility_score  # Inverse of volatility
        
        strategy_scores = {
            StrategyType.MOMENTUM: momentum_score * optimal_allocations.get(StrategyType.MOMENTUM, 0.5),
            StrategyType.MEAN_REVERSION: volatility_score * optimal_allocations.get(StrategyType.MEAN_REVERSION, 0.3),
            StrategyType.GRID_TRADING: range_score * optimal_allocations.get(StrategyType.GRID_TRADING, 0.2)
        }
        
        # Select strategy with highest score
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        recommended_strategy = best_strategy[0]
        confidence_score = best_strategy[1]
        
        # Calculate position size based on strategy allocation
        strategy_allocation = self.strategy_allocations[recommended_strategy]
        available_capital = strategy_allocation.available_capital
        max_position_size = min(
            available_capital * 0.2,  # Max 20% of strategy capital per position
            self.settings.MAX_POSITION_SIZE * self.settings.PORTFOLIO_VALUE
        )
        
        recommendation_details = {
            'market_regime': market_regime.value,
            'regime_confidence': confidence,
            'strategy_scores': {s.value: score for s, score in strategy_scores.items()},
            'optimal_allocations': {s.value: alloc for s, alloc in optimal_allocations.items()},
            'max_position_size': max_position_size
        }
        
        return recommended_strategy, confidence_score, recommendation_details
    
    def rebalance_strategies(self) -> Dict:
        """Rebalance capital allocation between strategies"""
        
        current_allocations = {}
        total_exposure = 0
        
        # Calculate current exposures
        for strategy, allocation in self.strategy_allocations.items():
            current_exposure = allocation.current_exposure
            current_allocations[strategy] = current_exposure / self.settings.PORTFOLIO_VALUE
            total_exposure += current_exposure
        
        # Detect if rebalancing is needed
        target_allocations = self.allocate_capital_to_strategies(
            self.current_market_regime,
            {s: p.performance_score for s, p in self.strategy_performance.items()}
        )
        
        rebalance_actions = []
        
        for strategy, target_allocation in target_allocations.items():
            current_allocation = current_allocations.get(strategy, 0)
            allocation_drift = abs(target_allocation - current_allocation)
            
            if allocation_drift > self.REBALANCE_THRESHOLD:
                if target_allocation > current_allocation:
                    action = 'increase_allocation'
                    amount = target_allocation - current_allocation
                else:
                    action = 'decrease_allocation'
                    amount = current_allocation - target_allocation
                
                rebalance_actions.append({
                    'strategy': strategy.value,
                    'action': action,
                    'amount': amount,
                    'target_allocation': target_allocation,
                    'current_allocation': current_allocation
                })
        
        return {
            'rebalance_needed': len(rebalance_actions) > 0,
            'actions': rebalance_actions,
            'total_exposure': total_exposure / self.settings.PORTFOLIO_VALUE
        }
    
    def get_coordination_dashboard_data(self) -> Dict:
        """Get comprehensive coordination data for dashboard"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'market_regime': {
                'current': self.current_market_regime.value,
                'confidence': self.market_regime_confidence
            },
            'strategy_allocations': {
                strategy.value: {
                    'allocated_capital': alloc.allocated_capital,
                    'allocation_percentage': alloc.allocation_percentage,
                    'current_exposure': alloc.current_exposure,
                    'available_capital': alloc.available_capital,
                    'active_positions': alloc.active_positions
                } for strategy, alloc in self.strategy_allocations.items()
            },
            'strategy_performance': {
                strategy.value: {
                    'total_trades': perf.total_trades,
                    'win_rate': perf.win_rate,
                    'total_pnl': perf.total_pnl,
                    'average_pnl': perf.average_pnl_per_trade,
                    'performance_score': perf.performance_score,
                    'trend': perf.performance_trend
                } for strategy, perf in self.strategy_performance.items()
            },
            'active_positions_by_strategy': {
                strategy.value: len(positions)
                for strategy, positions in self.active_positions.items()
            },
            'rebalance_status': self.rebalance_strategies()
        }

async def create_strategy_coordinator(settings):
    """Factory function to create strategy coordinator"""
    return StrategyCoordinator(settings)

# Example usage and testing
if __name__ == "__main__":
    async def test_coordinator():
        from src.config.settings import load_settings
        
        settings = load_settings()
        coordinator = StrategyCoordinator(settings)
        
        # Test market regime detection
        price_history = [0.001 + 0.0001 * i + np.random.normal(0, 0.00005) for i in range(50)]
        volume_history = [100 + np.random.normal(0, 10) for _ in range(50)]
        
        regime, confidence = coordinator.detect_market_regime(price_history, volume_history)
        print(f"Market regime: {regime.value} (confidence: {confidence:.2f})")
        
        # Test strategy recommendation
        signal_data = {
            'momentum_strength': 0.08,
            'volatility': 0.12
        }
        market_data = {
            'price_history': price_history,
            'volume_history': volume_history
        }
        
        strategy, confidence, details = coordinator.get_strategy_recommendation(
            'TestToken', signal_data, market_data
        )
        
        print(f"Recommended strategy: {strategy.value} (confidence: {confidence:.2f})")
        print(f"Market regime: {details['market_regime']}")
    
    if __name__ == "__main__":
        asyncio.run(test_coordinator())