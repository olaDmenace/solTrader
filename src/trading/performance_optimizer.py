import asyncio
from typing import Dict, List, Optional, Any, TypeVar, cast
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class PerformanceMetrics:
    """Stores performance metrics for analysis"""
    execution_time: float
    success_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

@dataclass
class CacheEntry:
    """Cache entry data structure"""
    data: Dict[str, Any]
    timestamp: datetime

class PerformanceOptimizer:
    """Optimizes trading performance based on historical data"""

    def __init__(self, settings: Any) -> None:
        """
        Initialize Performance Optimizer
        
        Args:
            settings: Trading settings
        """
        self.settings = settings
        self.trade_history: List[Dict[str, Any]] = []
        self.execution_times: List[float] = []
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_ttl: int = 300  # 5 minutes
        self.last_optimization = datetime.now()
        self.optimization_interval: int = 3600  # 1 hour

    async def optimize_execution(self, token_address: str) -> Dict[str, Any]:
        """
        Optimize execution parameters for a token
        
        Args:
            token_address: Token to optimize for
            
        Returns:
            Dict[str, Any]: Optimized parameters
        """
        start_time = datetime.now()
        
        try:
            # Check cache
            cached = self._get_cache(token_address)
            if cached:
                return cached

            # Calculate optimal parameters
            optimal_params = await self._calculate_optimal_params(token_address)
            
            # Update execution metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_times.append(execution_time)

            # Cache results
            cache_entry = CacheEntry(
                data=optimal_params,
                timestamp=datetime.now()
            )
            self.cache[token_address] = cache_entry
            
            return optimal_params

        except Exception as e:
            logger.error(f"Optimization error for {token_address}: {str(e)}")
            return {}

    async def _calculate_optimal_params(self, token_address: str) -> Dict[str, Any]:
        """
        Calculate optimal trading parameters based on historical performance
        
        Args:
            token_address: Token to optimize for
            
        Returns:
            Dict[str, Any]: Optimized parameters
        """
        metrics = await self._get_performance_metrics()
        
        # Adjust parameters based on performance
        slippage = self._optimize_slippage(metrics)
        position_size = self._optimize_position_size(metrics)
        stop_loss = self._optimize_stop_loss(metrics)
        take_profit = self._optimize_take_profit(metrics)

        return {
            'optimal_slippage': slippage,
            'optimal_position_size': position_size,
            'optimal_stop_loss': stop_loss,
            'optimal_take_profit': take_profit,
            'timestamp': datetime.now().isoformat()
        }

    async def _get_performance_metrics(self) -> PerformanceMetrics:
        """
        Calculate performance metrics from trade history
        
        Returns:
            PerformanceMetrics: Calculated metrics
        """
        if not self.trade_history:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0)

        successful_trades = len([t for t in self.trade_history if t.get('success')])
        total_trades = len(self.trade_history)
        
        returns = [t.get('return', 0) for t in self.trade_history]
        positive_returns = [r for r in returns if r > 0]
        negative_returns = [r for r in returns if r < 0]

        return PerformanceMetrics(
            execution_time=float(np.mean(self.execution_times)) if self.execution_times else 0.0,
            success_rate=successful_trades / total_trades if total_trades > 0 else 0,
            profit_factor=sum(positive_returns) / abs(sum(negative_returns)) if negative_returns else float('inf'),
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            max_drawdown=self._calculate_max_drawdown(returns),
            win_rate=len(positive_returns) / len(returns) if returns else 0
        )

    def _optimize_slippage(self, metrics: PerformanceMetrics) -> float:
        """
        Optimize slippage tolerance based on performance
        
        Args:
            metrics: Performance metrics
            
        Returns:
            float: Optimized slippage value
        """
        base_slippage = self.settings.SLIPPAGE_TOLERANCE
        
        # Adjust based on success rate
        if metrics.success_rate < 0.8:
            base_slippage *= 1.2
        elif metrics.success_rate > 0.95:
            base_slippage *= 0.8

        # Adjust based on execution time
        if metrics.execution_time > 2.0:  # If average execution > 2 seconds
            base_slippage *= 1.1

        return float(min(max(base_slippage, 0.001), 0.05))  # Keep between 0.1% and 5%

    def _optimize_position_size(self, metrics: PerformanceMetrics) -> float:
        """
        Optimize position sizing based on performance
        
        Args:
            metrics: Performance metrics
            
        Returns:
            float: Optimized position size
        """
        base_size = self.settings.MAX_TRADE_SIZE
        
        # Reduce size if high drawdown
        if metrics.max_drawdown > 0.1:  # 10% drawdown
            base_size *= 0.8
            
        # Adjust based on Sharpe ratio
        if metrics.sharpe_ratio < 1:
            base_size *= 0.9
        elif metrics.sharpe_ratio > 2:
            base_size *= 1.1

        return float(min(base_size, self.settings.MAX_POSITION_SIZE))

    def _optimize_stop_loss(self, metrics: PerformanceMetrics) -> float:
        """
        Optimize stop loss percentage based on performance
        
        Args:
            metrics: Performance metrics
            
        Returns:
            float: Optimized stop loss percentage
        """
        base_stop = self.settings.STOP_LOSS_PERCENTAGE
        
        if metrics.win_rate < 0.5:
            base_stop *= 0.9  # Tighter stops if winning less
        elif metrics.win_rate > 0.7:
            base_stop *= 1.1  # Wider stops if winning more

        return float(min(max(base_stop, 0.01), 0.1))  # Keep between 1% and 10%

    def _optimize_take_profit(self, metrics: PerformanceMetrics) -> float:
        """
        Optimize take profit percentage based on performance
        
        Args:
            metrics: Performance metrics
            
        Returns:
            float: Optimized take profit percentage
        """
        base_tp = self.settings.TAKE_PROFIT_PERCENTAGE
        
        if metrics.profit_factor < 1.5:
            base_tp *= 0.9  # Reduce target if not profitable enough
        elif metrics.profit_factor > 2.5:
            base_tp *= 1.1  # Increase target if very profitable

        return float(min(max(base_tp, 0.02), 0.2))  # Keep between 2% and 20%

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio from returns
        
        Args:
            returns: List of returns
            risk_free_rate: Risk-free rate (default: 2%)
            
        Returns:
            float: Calculated Sharpe ratio
        """
        if not returns:
            return 0.0
            
        returns_array = np.array(returns, dtype=np.float64)
        excess_returns = returns_array - (risk_free_rate / 365)
        std_dev = float(np.std(excess_returns))
        
        if std_dev == 0:
            return 0.0
            
        return float(np.sqrt(365) * (np.mean(excess_returns) / std_dev))

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calculate maximum drawdown from returns
        
        Args:
            returns: List of returns
            
        Returns:
            float: Maximum drawdown percentage
        """
        if not returns:
            return 0.0
            
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / np.where(peak != 0, peak, 1)
        
        return float(np.max(drawdown))

    def _get_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached data if valid
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Dict[str, Any]]: Cached data or None
        """
        if key in self.cache:
            entry = self.cache[key]
            if (datetime.now() - entry.timestamp).total_seconds() < self.cache_ttl:
                return entry.data
        return None

    def _update_cache(self, key: str, data: Dict[str, Any]) -> None:
        """
        Update cache with new data
        
        Args:
            key: Cache key
            data: Data to cache
        """
        self.cache[key] = CacheEntry(
            data=data,
            timestamp=datetime.now()
        )