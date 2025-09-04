"""
Enhanced Backtesting Module - Your Path to Financial Freedom! 🚀

This comprehensive backtesting system provides:
- Multi-strategy simulation (momentum + mean reversion)
- Parameter optimization for maximum profits
- Walk-forward analysis for robust validation
- Detailed performance analytics
- Risk-adjusted returns optimization

Usage:
```python
from src.backtesting import (
    EnhancedBacktestingEngine, 
    ParameterOptimizer,
    WalkForwardAnalyzer
)

# Run comprehensive backtest
engine = EnhancedBacktestingEngine(settings)
results = await engine.run_comprehensive_backtest()

# Optimize parameters for maximum profits
optimizer = ParameterOptimizer(settings)
optimization_result = await optimizer.optimize_combined_strategies()

# Validate with walk-forward analysis
analyzer = WalkForwardAnalyzer(settings)
stability = await analyzer.run_walk_forward_analysis(optimization_result.best_parameters)
```
"""

from .enhanced_backtesting_engine import (
    EnhancedBacktestingEngine,
    BacktestResults,
    BacktestTrade,
    HistoricalDataManager,
    StrategySimulator,
    PerformanceAnalyzer,
    create_enhanced_backtesting_engine
)

from .parameter_optimizer import (
    ParameterOptimizer,
    ParameterRange,
    OptimizationResult,
    WalkForwardAnalyzer,
    create_parameter_optimizer,
    create_walk_forward_analyzer
)

from .production_backtester import (
    ProductionBacktester,
    BacktestMode,
    ExecutionQuality,
    MarketCondition,
    ExecutionResult,
    BacktestResult,
    WalkForwardPeriod
)

__all__ = [
    # Enhanced Backtesting Engine
    'EnhancedBacktestingEngine',
    'BacktestResults', 
    'BacktestTrade',
    'HistoricalDataManager',
    'StrategySimulator',
    'PerformanceAnalyzer',
    'create_enhanced_backtesting_engine',
    
    # Parameter Optimization
    'ParameterOptimizer',
    'ParameterRange',
    'OptimizationResult',
    'WalkForwardAnalyzer',
    'create_parameter_optimizer',
    'create_walk_forward_analyzer',
    
    # Production Backtesting
    'ProductionBacktester',
    'BacktestMode',
    'ExecutionQuality',
    'MarketCondition',
    'ExecutionResult',
    'BacktestResult',
    'WalkForwardPeriod'
]