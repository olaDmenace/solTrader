#!/usr/bin/env python3
"""
Capital Allocator Integration Utilities
Provides integration hooks and utilities for connecting the Dynamic Capital Allocator
with existing trading strategies and the main trading system.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Import performance-based rebalancer
try:
    from .performance_based_rebalancer import PerformanceBasedRebalancer
    PERFORMANCE_REBALANCER_AVAILABLE = True
except ImportError:
    PERFORMANCE_REBALANCER_AVAILABLE = False
    logger.warning("Performance-based rebalancer not available")

@dataclass
class StrategyPerformanceData:
    """Performance data container for strategy integration"""
    strategy_name: str
    total_return: float = 0.0
    trades_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    current_position_size: float = 0.0
    last_trade_time: Optional[datetime] = None
    error_count: int = 0
    is_active: bool = True

class AllocatorIntegrationMixin:
    """
    Mixin class for integrating trading strategies with the Dynamic Capital Allocator.
    Add this to existing strategy classes to enable automatic allocation updates.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allocator = None  # Will be set during integration
        self.allocated_capital = 0.0
        self.performance_data = StrategyPerformanceData(
            strategy_name=getattr(self, 'strategy_name', self.__class__.__name__)
        )
        self._last_performance_update = datetime.now()
        self._performance_update_interval = timedelta(minutes=5)
    
    async def set_allocator(self, allocator):
        """Set the capital allocator for this strategy"""
        self.allocator = allocator
        await self._register_with_allocator()
    
    async def _register_with_allocator(self):
        """Register this strategy with the allocator"""
        if self.allocator:
            success = await self.allocator.register_strategy(
                strategy_name=self.performance_data.strategy_name,
                initial_allocation=getattr(self, 'initial_allocation', 0.1)
            )
            if success:
                logger.info(f"[INTEGRATION] Strategy {self.performance_data.strategy_name} registered with allocator")
    
    async def update_performance_metrics(self, **metrics):
        """Update performance metrics and send to allocator"""
        try:
            # Update local performance data
            for key, value in metrics.items():
                if hasattr(self.performance_data, key):
                    setattr(self.performance_data, key, value)
            
            # Send to allocator if enough time has passed
            if datetime.now() - self._last_performance_update >= self._performance_update_interval:
                if self.allocator:
                    await self.allocator.update_strategy_metrics(
                        self.performance_data.strategy_name,
                        **metrics
                    )
                self._last_performance_update = datetime.now()
                
        except Exception as e:
            logger.error(f"[INTEGRATION] Error updating performance metrics: {e}")
    
    async def on_trade_completed(self, trade_result: Dict[str, Any]):
        """Called when a trade is completed - updates performance metrics"""
        try:
            # Update trade statistics
            self.performance_data.trades_count += 1
            self.performance_data.last_trade_time = datetime.now()
            
            # Calculate performance metrics from trade result
            if 'pnl' in trade_result and 'return' in trade_result:
                # Update metrics (simplified - in production, maintain rolling calculations)
                await self.update_performance_metrics(
                    trades_count=self.performance_data.trades_count,
                    last_trade_time=self.performance_data.last_trade_time
                )
            
        except Exception as e:
            logger.error(f"[INTEGRATION] Error processing trade completion: {e}")
    
    async def on_position_size_change(self, new_size: float):
        """Called when position size changes"""
        try:
            self.performance_data.current_position_size = new_size
            await self.update_performance_metrics(
                current_position_size=new_size
            )
            
        except Exception as e:
            logger.error(f"[INTEGRATION] Error updating position size: {e}")
    
    async def get_allocated_capital(self) -> float:
        """Get current allocated capital from allocator"""
        try:
            if self.allocator and self.performance_data.strategy_name in self.allocator.strategies:
                strategy = self.allocator.strategies[self.performance_data.strategy_name]
                self.allocated_capital = strategy.allocated_capital
                return self.allocated_capital
            return self.allocated_capital
            
        except Exception as e:
            logger.error(f"[INTEGRATION] Error getting allocated capital: {e}")
            return self.allocated_capital

class PortfolioManager:
    """
    Portfolio manager that coordinates between the allocator, risk manager, and trading strategies
    """
    
    def __init__(self, settings, allocator, risk_manager=None):
        self.settings = settings
        self.allocator = allocator
        self.risk_manager = risk_manager
        self.integrated_strategies: Dict[str, Any] = {}
        self.performance_monitor_task = None
        self.rebalance_task = None
        self.risk_monitor_task = None
        
        # Initialize performance-based rebalancer
        self.performance_rebalancer = None
        if PERFORMANCE_REBALANCER_AVAILABLE:
            try:
                self.performance_rebalancer = PerformanceBasedRebalancer(settings)
                logger.info("[PORTFOLIO_MANAGER] Performance-based rebalancer initialized")
            except Exception as e:
                logger.warning(f"[PORTFOLIO_MANAGER] Failed to initialize performance rebalancer: {e}")
        else:
            logger.info("[PORTFOLIO_MANAGER] Using basic rebalancing only")
        
        # Monitoring settings
        self.performance_check_interval = timedelta(minutes=5)
        self.rebalance_check_interval = timedelta(minutes=30)
        self.risk_check_interval = timedelta(minutes=2)  # More frequent risk monitoring
        
        logger.info("[PORTFOLIO_MANAGER] Portfolio manager initialized")
    
    async def start(self):
        """Start portfolio management tasks"""
        try:
            # Start background monitoring
            self.performance_monitor_task = asyncio.create_task(self._performance_monitor_loop())
            self.rebalance_task = asyncio.create_task(self._rebalance_monitor_loop())
            
            # Start risk monitoring if risk manager available
            if self.risk_manager:
                self.risk_monitor_task = asyncio.create_task(self._risk_monitor_loop())
            
            # Start performance-based rebalancer
            if self.performance_rebalancer:
                await self.performance_rebalancer.start()
            
            logger.info("[PORTFOLIO_MANAGER] Portfolio management tasks started")
            
        except Exception as e:
            logger.error(f"[PORTFOLIO_MANAGER] Failed to start portfolio manager: {e}")
            raise
    
    async def stop(self):
        """Stop portfolio management tasks"""
        try:
            if self.performance_monitor_task:
                self.performance_monitor_task.cancel()
                
            if self.rebalance_task:
                self.rebalance_task.cancel()
                
            if self.risk_monitor_task:
                self.risk_monitor_task.cancel()
            
            # Stop performance-based rebalancer
            if self.performance_rebalancer:
                await self.performance_rebalancer.stop()
                
            logger.info("[PORTFOLIO_MANAGER] Portfolio management tasks stopped")
            
        except Exception as e:
            logger.error(f"[PORTFOLIO_MANAGER] Error stopping portfolio manager: {e}")
    
    async def integrate_strategy(self, strategy_instance, strategy_name: str):
        """Integrate a strategy instance with the portfolio manager"""
        try:
            # Set allocator reference
            if hasattr(strategy_instance, 'set_allocator'):
                await strategy_instance.set_allocator(self.allocator)
            
            # Store reference
            self.integrated_strategies[strategy_name] = strategy_instance
            
            logger.info(f"[PORTFOLIO_MANAGER] Integrated strategy: {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"[PORTFOLIO_MANAGER] Failed to integrate strategy {strategy_name}: {e}")
            return False
    
    async def update_market_data(self, price: float, volume: float):
        """Update market data for all components"""
        try:
            await self.allocator.update_market_data(price, volume)
            
        except Exception as e:
            logger.error(f"[PORTFOLIO_MANAGER] Error updating market data: {e}")
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status"""
        try:
            allocations = await self.allocator.get_current_allocations()
            recommendations = await self.allocator.get_allocation_recommendations()
            
            portfolio_status = {
                'allocations': allocations,
                'recommendations': recommendations,
                'integrated_strategies': list(self.integrated_strategies.keys()),
                'total_strategies': len(self.integrated_strategies),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add risk status if risk manager available
            if self.risk_manager:
                risk_status = self.risk_manager.get_current_risk_status()
                portfolio_status['risk_status'] = risk_status
            
            return portfolio_status
            
        except Exception as e:
            logger.error(f"[PORTFOLIO_MANAGER] Error getting portfolio status: {e}")
            return {}
    
    async def force_rebalance(self) -> Dict[str, Any]:
        """Force immediate portfolio rebalancing"""
        try:
            result = await self.allocator.rebalance_portfolio(force=True)
            
            # Update strategy allocations
            await self._update_strategy_allocations()
            
            return result
            
        except Exception as e:
            logger.error(f"[PORTFOLIO_MANAGER] Error during forced rebalance: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _performance_monitor_loop(self):
        """Background loop for performance monitoring"""
        try:
            while True:
                await self._collect_strategy_performance()
                await self._update_performance_data()  # Feed data to performance rebalancer
                await asyncio.sleep(self.performance_check_interval.total_seconds())
                
        except asyncio.CancelledError:
            logger.info("[PORTFOLIO_MANAGER] Performance monitor loop cancelled")
        except Exception as e:
            logger.error(f"[PORTFOLIO_MANAGER] Error in performance monitor loop: {e}")
    
    async def _rebalance_monitor_loop(self):
        """Background loop for rebalancing checks"""
        try:
            while True:
                await self._check_rebalancing_needed()
                await asyncio.sleep(self.rebalance_check_interval.total_seconds())
                
        except asyncio.CancelledError:
            logger.info("[PORTFOLIO_MANAGER] Rebalance monitor loop cancelled")
        except Exception as e:
            logger.error(f"[PORTFOLIO_MANAGER] Error in rebalance monitor loop: {e}")
    
    async def _collect_strategy_performance(self):
        """Collect performance data from all integrated strategies"""
        try:
            for strategy_name, strategy_instance in self.integrated_strategies.items():
                if hasattr(strategy_instance, 'performance_data'):
                    perf_data = strategy_instance.performance_data
                    
                    # Update allocator with latest performance
                    await self.allocator.update_strategy_metrics(
                        strategy_name,
                        total_return=perf_data.total_return,
                        trades_count=perf_data.trades_count,
                        win_rate=perf_data.win_rate,
                        profit_factor=perf_data.profit_factor,
                        max_drawdown=perf_data.max_drawdown,
                        is_active=perf_data.is_active
                    )
                    
        except Exception as e:
            logger.error(f"[PORTFOLIO_MANAGER] Error collecting strategy performance: {e}")
    
    async def _update_performance_data(self):
        """Update performance data for the performance-based rebalancer"""
        try:
            if not self.performance_rebalancer:
                return
            
            # Update strategy performance data
            for strategy_name, strategy_instance in self.integrated_strategies.items():
                if hasattr(strategy_instance, 'performance_data'):
                    perf_data = strategy_instance.performance_data
                    
                    # Calculate return (simplified)
                    return_value = perf_data.total_return / 100 if perf_data.total_return else 0.001
                    
                    # Create trade data if available
                    trade_data = None
                    if perf_data.trades_count > 0:
                        trade_data = {
                            'pnl': return_value * 100,  # Mock P&L
                            'return': return_value,
                            'win': return_value > 0
                        }
                    
                    # Update performance rebalancer
                    self.performance_rebalancer.update_strategy_return(
                        strategy_name, 
                        return_value, 
                        trade_data
                    )
            
            # Update benchmark (use SOL price simulation)
            benchmark_return = np.random.normal(0.0005, 0.02)  # Mock SOL return
            self.performance_rebalancer.update_benchmark_return(benchmark_return)
            
        except Exception as e:
            logger.error(f"[PORTFOLIO_MANAGER] Error updating performance data: {e}")
    
    async def _check_rebalancing_needed(self):
        """Check if portfolio rebalancing is needed"""
        try:
            # Get current allocations
            allocations = await self.allocator.get_current_allocations()
            
            # Check if any strategy needs rebalancing
            rebalance_needed = False
            for strategy_name, target in self.allocator.allocation_targets.items():
                if target.needs_rebalancing():
                    rebalance_needed = True
                    break
            
            if rebalance_needed:
                logger.info("[PORTFOLIO_MANAGER] Automatic rebalancing triggered")
                result = await self.allocator.rebalance_portfolio()
                
                if result.get('status') == 'completed':
                    await self._update_strategy_allocations()
                    
        except Exception as e:
            logger.error(f"[PORTFOLIO_MANAGER] Error checking rebalancing: {e}")
    
    async def _risk_monitor_loop(self):
        """Background loop for portfolio risk monitoring"""
        try:
            while True:
                await self._update_risk_metrics()
                await asyncio.sleep(self.risk_check_interval.total_seconds())
                
        except asyncio.CancelledError:
            logger.info("[PORTFOLIO_MANAGER] Risk monitor loop cancelled")
        except Exception as e:
            logger.error(f"[PORTFOLIO_MANAGER] Error in risk monitor loop: {e}")
    
    async def _update_risk_metrics(self):
        """Update portfolio risk metrics"""
        try:
            if not self.risk_manager:
                return
            
            # Get current allocations
            allocations = await self.allocator.get_current_allocations()
            strategy_allocations = {}
            strategy_returns = {}
            strategy_metrics = {}
            
            for strategy_name, alloc_data in allocations.get('allocations', {}).items():
                strategy_allocations[strategy_name] = alloc_data.get('allocation_percentage', 0.0)
                
                # Mock returns for demonstration (in production, get real returns)
                strategy_returns[strategy_name] = [0.01, 0.02, -0.01, 0.015, 0.005]  # Mock returns
                
                strategy_metrics[strategy_name] = {
                    'sharpe_ratio': alloc_data.get('sharpe_ratio', 0.0),
                    'volatility': 0.15,  # Mock volatility
                    'correlation': 0.3   # Mock correlation
                }
            
            # Update risk manager
            risk_metrics = await self.risk_manager.update_portfolio_data(
                strategy_allocations,
                strategy_returns,
                strategy_metrics
            )
            
            # Check if emergency action needed
            if risk_metrics.overall_risk_level.name == 'CRITICAL':
                logger.critical(f"[PORTFOLIO_MANAGER] Critical risk level detected: VaR={risk_metrics.var_95_1d:.3f}")
                
                if abs(risk_metrics.var_95_1d) > 0.08:  # 8% emergency threshold
                    await self.risk_manager.trigger_emergency_action(
                        action=self.risk_manager.EmergencyAction.REDUCE_POSITIONS,
                        affected_strategies=list(strategy_allocations.keys())
                    )
                    
                    # Force rebalance with reduced risk
                    await self.force_rebalance()
                    
        except Exception as e:
            logger.error(f"[PORTFOLIO_MANAGER] Error updating risk metrics: {e}")
    
    async def _update_strategy_allocations(self):
        """Update capital allocations for all integrated strategies"""
        try:
            for strategy_name, strategy_instance in self.integrated_strategies.items():
                if hasattr(strategy_instance, 'allocated_capital'):
                    new_capital = await strategy_instance.get_allocated_capital()
                    logger.debug(f"[PORTFOLIO_MANAGER] Updated {strategy_name} allocation: {new_capital}")
                    
        except Exception as e:
            logger.error(f"[PORTFOLIO_MANAGER] Error updating strategy allocations: {e}")

def create_portfolio_system(settings, strategies: List[Dict[str, Any]]) -> PortfolioManager:
    """
    Factory function to create a complete portfolio management system
    
    Args:
        settings: Application settings
        strategies: List of strategy configurations with 'name', 'instance', and 'allocation'
    
    Returns:
        Configured PortfolioManager instance
    """
    try:
        # Import here to avoid circular imports
        from .dynamic_capital_allocator import DynamicCapitalAllocator
        
        # Create allocator
        allocator = DynamicCapitalAllocator(settings)
        
        # Create portfolio manager
        portfolio_manager = PortfolioManager(settings, allocator)
        
        # Register strategies with allocator
        async def setup_strategies():
            await allocator.start()
            
            for strategy_config in strategies:
                await allocator.register_strategy(
                    strategy_name=strategy_config['name'],
                    initial_allocation=strategy_config.get('allocation', 0.1)
                )
                
                if 'instance' in strategy_config:
                    await portfolio_manager.integrate_strategy(
                        strategy_config['instance'],
                        strategy_config['name']
                    )
        
        # Return setup coroutine along with manager
        return portfolio_manager, setup_strategies
        
    except Exception as e:
        logger.error(f"[PORTFOLIO_FACTORY] Error creating portfolio system: {e}")
        raise

# Utility functions for integration
async def integrate_existing_strategy(strategy_instance, allocator, strategy_name: str):
    """Utility to integrate an existing strategy with the allocator"""
    try:
        # Add mixin functionality dynamically if not already present
        if not hasattr(strategy_instance, 'set_allocator'):
            # Monkey patch the mixin methods
            strategy_instance.set_allocator = AllocatorIntegrationMixin.set_allocator.__get__(strategy_instance)
            strategy_instance._register_with_allocator = AllocatorIntegrationMixin._register_with_allocator.__get__(strategy_instance)
            strategy_instance.update_performance_metrics = AllocatorIntegrationMixin.update_performance_metrics.__get__(strategy_instance)
            strategy_instance.on_trade_completed = AllocatorIntegrationMixin.on_trade_completed.__get__(strategy_instance)
            strategy_instance.get_allocated_capital = AllocatorIntegrationMixin.get_allocated_capital.__get__(strategy_instance)
            
            # Initialize mixin attributes
            strategy_instance.allocator = None
            strategy_instance.allocated_capital = 0.0
            strategy_instance.performance_data = StrategyPerformanceData(strategy_name=strategy_name)
            strategy_instance._last_performance_update = datetime.now()
            strategy_instance._performance_update_interval = timedelta(minutes=5)
        
        # Set allocator
        await strategy_instance.set_allocator(allocator)
        
        logger.info(f"[INTEGRATION] Successfully integrated existing strategy: {strategy_name}")
        return True
        
    except Exception as e:
        logger.error(f"[INTEGRATION] Failed to integrate existing strategy {strategy_name}: {e}")
        return False