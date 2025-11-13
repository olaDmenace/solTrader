"""
Strategies Package - NEW for Day 3 foundation

MIGRATION NOTE: New unified strategy package for all trading strategies
CRITICAL: Foundation for massive strategy extraction in Days 4-5

This package will contain:
- base.py: Base strategy interface (COMPLETE)
- momentum.py: Momentum strategy (Day 4-5 extraction target)
- mean_reversion.py: Mean reversion strategy (Day 4-5 extraction target) 
- grid_trading.py: Grid trading strategy (Day 4-5 extraction target)
- arbitrage.py: Arbitrage strategy (Day 4-5 extraction target)
"""

from .base import (
    BaseStrategy,
    StrategyConfig,
    StrategyMetrics,
    StrategyType,
    StrategyStatus
)

__all__ = [
    'BaseStrategy',
    'StrategyConfig', 
    'StrategyMetrics',
    'StrategyType',
    'StrategyStatus'
]