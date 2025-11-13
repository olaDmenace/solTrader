"""
Models Package - NEW for Day 3 data model consolidation

MIGRATION NOTE: New unified data models package
CRITICAL: Clean data models for strategy extraction and portfolio management

This package contains:
- position.py: Position management models (MIGRATED & ENHANCED)
- signal.py: Signal generation models (MIGRATED & ENHANCED)
- trade.py: Trade tracking models (NEW)
- portfolio.py: Portfolio management models (NEW)
"""

from .position import (
    Position,
    PaperPosition,
    PositionManager,
    TradeEntry,
    TrailingStop,
    ExitReason
)

from .signal import (
    Signal,
    MarketCondition,
    SignalType,
    SignalGenerator,
    TrendAnalysis,
    VolumeAnalysis
)

from .trade import (
    Trade,
    TradeExecution,
    TradeBook,
    TradeStatus,
    TradeDirection,
    TradeType,
    ExecutionVenue
)

from .portfolio import (
    Portfolio,
    PortfolioMetrics,
    StrategyAllocation,
    PortfolioStatus,
    AllocationStrategy
)

__all__ = [
    # Position models
    'Position',
    'PaperPosition', 
    'PositionManager',
    'TradeEntry',
    'TrailingStop',
    'ExitReason',
    
    # Signal models
    'Signal',
    'MarketCondition',
    'SignalType',
    'SignalGenerator',
    'TrendAnalysis',
    'VolumeAnalysis',
    
    # Trade models
    'Trade',
    'TradeExecution',
    'TradeBook',
    'TradeStatus',
    'TradeDirection',
    'TradeType',
    'ExecutionVenue',
    
    # Portfolio models
    'Portfolio',
    'PortfolioMetrics',
    'StrategyAllocation',
    'PortfolioStatus',
    'AllocationStrategy'
]