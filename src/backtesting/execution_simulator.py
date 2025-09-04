#!/usr/bin/env python3
"""
Execution Simulator with Realistic Market Structure
Integrates realistic market simulation with the existing backtesting framework
to provide accurate execution modeling for strategy validation.

Features:
- Integration with RealisticMarketSimulator
- Enhanced trade execution with market impact
- Realistic slippage and latency modeling
- Cross-DEX execution simulation
- Performance metrics collection
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .market_simulator import RealisticMarketSimulator, OrderSide, OrderType, ExecutionResult
from ..trading.formal_backtesting_validator import BacktestMetrics

logger = logging.getLogger(__name__)

class ExecutionStyle(Enum):
    AGGRESSIVE = "AGGRESSIVE"  # Market orders, immediate execution
    PASSIVE = "PASSIVE"        # Limit orders, wait for fills
    ADAPTIVE = "ADAPTIVE"      # Mix based on market conditions

@dataclass
class SimulatedTrade:
    """Represents a simulated trade with all execution details"""
    
    # Trade identification
    trade_id: str
    strategy: str
    token: str
    
    # Order details
    side: OrderSide
    requested_size: float
    intended_price: float
    execution_style: ExecutionStyle
    
    # Execution results
    execution_result: Optional[ExecutionResult] = None
    actual_pnl: float = 0.0
    
    # Timing
    requested_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    
    # Context
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_executed(self) -> bool:
        return self.execution_result is not None and self.execution_result.success
    
    @property
    def execution_quality_score(self) -> float:
        """Calculate execution quality score (0-100)"""
        if not self.execution_result or not self.execution_result.success:
            return 0.0
        
        # Base score from fill ratio
        fill_ratio = self.execution_result.executed_size / max(self.requested_size, 0.001)
        fill_score = fill_ratio * 40
        
        # Price improvement/degradation
        price_diff_pct = abs(self.execution_result.executed_price - self.intended_price) / self.intended_price * 100
        price_score = max(0, 30 - price_diff_pct * 3)  # Penalty for price deviation
        
        # Speed bonus
        speed_score = max(0, 30 - (self.execution_result.execution_time_ms / 1000) * 5)
        
        return min(100, fill_score + price_score + speed_score)

@dataclass
class BacktestExecution:
    """Enhanced backtest execution with market structure simulation"""
    
    # Backtest identification
    backtest_id: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    
    # Simulation settings
    initial_capital: float
    tokens: List[str]
    base_prices: Dict[str, float]
    
    # Results
    trades: List[SimulatedTrade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    final_capital: float = 0.0
    
    # Performance metrics
    total_trades: int = 0
    successful_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Execution metrics
    avg_execution_time: float = 0.0
    avg_slippage: float = 0.0
    avg_execution_quality: float = 0.0
    total_fees: float = 0.0

class EnhancedExecutionSimulator:
    """
    Advanced execution simulator that provides realistic market structure
    modeling for backtesting trading strategies.
    """
    
    def __init__(self, settings: Any):
        self.settings = settings
        
        # Initialize market simulator
        self.market_simulator = RealisticMarketSimulator(settings)
        
        # Execution parameters
        self.default_execution_style = ExecutionStyle.ADAPTIVE
        self.max_position_size = 0.05  # 5% of capital per position
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Tracking
        self.active_backtests: Dict[str, BacktestExecution] = {}
        self.execution_history: List[SimulatedTrade] = []
        
        logger.info("[EXEC_SIM] Enhanced execution simulator initialized")
    
    async def start_backtest(
        self,
        backtest_id: str,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        tokens: List[str],
        base_prices: Dict[str, float]
    ) -> bool:
        """Start a new backtest with realistic market simulation"""
        try:
            # Create backtest execution record
            backtest = BacktestExecution(
                backtest_id=backtest_id,
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                tokens=tokens,
                base_prices=base_prices,
                final_capital=initial_capital
            )
            
            # Initialize market simulation
            self.market_simulator.initialize_market(tokens, base_prices)
            
            # Store active backtest
            self.active_backtests[backtest_id] = backtest
            
            # Initialize equity curve
            backtest.equity_curve.append((start_date, initial_capital))
            
            logger.info(f"[EXEC_SIM] Started backtest {backtest_id} for {strategy_name}")
            logger.info(f"[EXEC_SIM] Tokens: {tokens}, Capital: ${initial_capital:,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"[EXEC_SIM] Error starting backtest {backtest_id}: {e}")
            return False
    
    async def simulate_trade_execution(
        self,
        backtest_id: str,
        trade_id: str,
        token: str,
        is_buy: bool,
        size: float,
        intended_price: float,
        execution_style: Optional[ExecutionStyle] = None,
        dex_name: str = 'raydium',
        market_context: Optional[Dict[str, Any]] = None
    ) -> SimulatedTrade:
        """
        Simulate trade execution with realistic market structure effects
        """
        try:
            backtest = self.active_backtests.get(backtest_id)
            if not backtest:
                raise ValueError(f"Backtest {backtest_id} not found")
            
            # Create simulated trade
            trade = SimulatedTrade(
                trade_id=trade_id,
                strategy=backtest.strategy_name,
                token=token,
                side=OrderSide.BUY if is_buy else OrderSide.SELL,
                requested_size=size,
                intended_price=intended_price,
                execution_style=execution_style or self.default_execution_style,
                market_conditions=market_context or {}
            )
            
            # Determine order type based on execution style
            order_type = self._determine_order_type(trade.execution_style, market_context)
            limit_price = self._calculate_limit_price(intended_price, is_buy, trade.execution_style)
            
            # Execute order through market simulator
            execution_result = await self.market_simulator.simulate_order_execution(
                token=token,
                side=trade.side,
                size=size,
                order_type=order_type,
                limit_price=limit_price,
                dex_name=dex_name
            )
            
            # Update trade with execution results
            trade.execution_result = execution_result
            trade.executed_at = datetime.now()
            
            # Calculate P&L (simplified)
            if execution_result.success:
                if is_buy:
                    # For buys, P&L = (current_price - executed_price) * size
                    # Simplified: assume we can sell at intended price for demo
                    trade.actual_pnl = (intended_price - execution_result.executed_price) * execution_result.executed_size
                else:
                    # For sells, P&L = (executed_price - cost_basis) * size
                    trade.actual_pnl = (execution_result.executed_price - intended_price) * execution_result.executed_size
                
                # Subtract fees
                trade.actual_pnl -= execution_result.total_fees
            
            # Update backtest metrics
            await self._update_backtest_metrics(backtest, trade)
            
            # Store trade
            backtest.trades.append(trade)
            self.execution_history.append(trade)
            
            # Log execution details
            self._log_trade_execution(trade)
            
            return trade
            
        except Exception as e:
            logger.error(f"[EXEC_SIM] Error simulating trade {trade_id}: {e}")
            
            # Return failed trade
            return SimulatedTrade(
                trade_id=trade_id,
                strategy=backtest.strategy_name if backtest else "unknown",
                token=token,
                side=OrderSide.BUY if is_buy else OrderSide.SELL,
                requested_size=size,
                intended_price=intended_price,
                execution_style=execution_style or self.default_execution_style
            )
    
    async def complete_backtest(self, backtest_id: str) -> BacktestMetrics:
        """Complete backtest and calculate final metrics"""
        try:
            backtest = self.active_backtests.get(backtest_id)
            if not backtest:
                raise ValueError(f"Backtest {backtest_id} not found")
            
            # Calculate final metrics
            metrics = await self._calculate_comprehensive_metrics(backtest)
            
            # Update backtest with final metrics
            backtest.total_trades = len(backtest.trades)
            backtest.successful_trades = sum(1 for t in backtest.trades if t.is_executed)
            backtest.total_pnl = sum(t.actual_pnl for t in backtest.trades if t.actual_pnl)
            backtest.final_capital = backtest.initial_capital + backtest.total_pnl
            
            # Calculate execution metrics
            executed_trades = [t for t in backtest.trades if t.execution_result and t.execution_result.success]
            if executed_trades:
                backtest.avg_execution_time = np.mean([t.execution_result.execution_time_ms for t in executed_trades])
                backtest.avg_slippage = np.mean([t.execution_result.market_impact.impact_percentage for t in executed_trades])
                backtest.avg_execution_quality = np.mean([t.execution_quality_score for t in executed_trades])
                backtest.total_fees = sum(t.execution_result.total_fees for t in executed_trades)
            
            logger.info(f"[EXEC_SIM] Completed backtest {backtest_id}")
            logger.info(f"[EXEC_SIM] Final capital: ${backtest.final_capital:,.2f} (P&L: ${backtest.total_pnl:,.2f})")
            logger.info(f"[EXEC_SIM] Total trades: {backtest.total_trades}, Success rate: {backtest.successful_trades/max(backtest.total_trades,1)*100:.1f}%")
            
            # Remove from active backtests
            del self.active_backtests[backtest_id]
            
            return metrics
            
        except Exception as e:
            logger.error(f"[EXEC_SIM] Error completing backtest {backtest_id}: {e}")
            return BacktestMetrics()  # Return empty metrics on error
    
    def _determine_order_type(self, execution_style: ExecutionStyle, market_context: Optional[Dict[str, Any]]) -> OrderType:
        """Determine order type based on execution style and market conditions"""
        try:
            if execution_style == ExecutionStyle.AGGRESSIVE:
                return OrderType.MARKET
            elif execution_style == ExecutionStyle.PASSIVE:
                return OrderType.LIMIT
            else:  # ADAPTIVE
                # Use market conditions to decide
                volatility = market_context.get('volatility', 0.02) if market_context else 0.02
                spread = market_context.get('spread_percentage', 0.1) if market_context else 0.1
                
                # Use market orders in high volatility or tight spreads
                if volatility > 0.05 or spread < 0.05:
                    return OrderType.MARKET
                else:
                    return OrderType.LIMIT
                    
        except Exception:
            return OrderType.MARKET  # Default to market orders
    
    def _calculate_limit_price(self, intended_price: float, is_buy: bool, execution_style: ExecutionStyle) -> Optional[float]:
        """Calculate limit price based on execution style"""
        try:
            if execution_style == ExecutionStyle.PASSIVE:
                # Conservative limit prices
                if is_buy:
                    return intended_price * 0.995  # 0.5% below intended price
                else:
                    return intended_price * 1.005  # 0.5% above intended price
            elif execution_style == ExecutionStyle.ADAPTIVE:
                # Moderate limit prices
                if is_buy:
                    return intended_price * 0.998  # 0.2% below intended price
                else:
                    return intended_price * 1.002  # 0.2% above intended price
            else:
                return None  # No limit for aggressive orders
                
        except Exception:
            return None
    
    async def _update_backtest_metrics(self, backtest: BacktestExecution, trade: SimulatedTrade):
        """Update backtest metrics with new trade"""
        try:
            if trade.execution_result and trade.execution_result.success:
                # Update capital
                backtest.final_capital += trade.actual_pnl
                
                # Update equity curve
                backtest.equity_curve.append((trade.executed_at or datetime.now(), backtest.final_capital))
                
                # Calculate drawdown
                peak_capital = max([capital for _, capital in backtest.equity_curve])
                current_drawdown = (peak_capital - backtest.final_capital) / peak_capital
                backtest.max_drawdown = max(backtest.max_drawdown, current_drawdown)
            
        except Exception as e:
            logger.error(f"[EXEC_SIM] Error updating backtest metrics: {e}")
    
    async def _calculate_comprehensive_metrics(self, backtest: BacktestExecution) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""
        try:
            if not backtest.trades:
                return BacktestMetrics()
            
            # Extract returns from equity curve
            returns = []
            for i in range(1, len(backtest.equity_curve)):
                prev_capital = backtest.equity_curve[i-1][1]
                curr_capital = backtest.equity_curve[i][1]
                if prev_capital > 0:
                    daily_return = (curr_capital - prev_capital) / prev_capital
                    returns.append(daily_return)
            
            if not returns:
                return BacktestMetrics()
            
            # Calculate metrics
            total_return = (backtest.final_capital - backtest.initial_capital) / backtest.initial_capital
            
            # Annualized return (assuming daily returns)
            total_days = (backtest.end_date - backtest.start_date).days
            annualized_return = (1 + total_return) ** (365 / max(total_days, 1)) - 1
            
            # Volatility
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            
            # Sharpe ratio
            excess_returns = [r - (self.risk_free_rate / 252) for r in returns]
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if len(excess_returns) > 1 and np.std(excess_returns) > 0 else 0
            
            # Trading stats
            executed_trades = [t for t in backtest.trades if t.is_executed]
            winning_trades = [t for t in executed_trades if t.actual_pnl > 0]
            
            win_rate = len(winning_trades) / max(len(executed_trades), 1)
            
            # Profit factor
            gross_profit = sum(t.actual_pnl for t in winning_trades)
            gross_loss = abs(sum(t.actual_pnl for t in executed_trades if t.actual_pnl < 0))
            profit_factor = gross_profit / max(gross_loss, 0.001)
            
            # Create and return metrics
            return BacktestMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=backtest.max_drawdown,
                total_trades=len(executed_trades),
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=np.mean([t.actual_pnl for t in winning_trades]) if winning_trades else 0,
                avg_loss=np.mean([t.actual_pnl for t in executed_trades if t.actual_pnl < 0]) if any(t.actual_pnl < 0 for t in executed_trades) else 0,
                largest_win=max([t.actual_pnl for t in winning_trades]) if winning_trades else 0,
                largest_loss=min([t.actual_pnl for t in executed_trades if t.actual_pnl < 0]) if any(t.actual_pnl < 0 for t in executed_trades) else 0
            )
            
        except Exception as e:
            logger.error(f"[EXEC_SIM] Error calculating metrics: {e}")
            return BacktestMetrics()
    
    def _log_trade_execution(self, trade: SimulatedTrade):
        """Log detailed trade execution information"""
        try:
            if trade.execution_result and trade.execution_result.success:
                result = trade.execution_result
                logger.info(f"[EXEC_SIM] ✅ Trade executed: {trade.trade_id}")
                logger.info(f"[EXEC_SIM]   Token: {trade.token}, Side: {trade.side.value}")
                logger.info(f"[EXEC_SIM]   Size: {result.executed_size:.4f} (requested: {trade.requested_size:.4f})")
                logger.info(f"[EXEC_SIM]   Price: ${result.executed_price:.6f} (intended: ${trade.intended_price:.6f})")
                logger.info(f"[EXEC_SIM]   Slippage: {result.market_impact.impact_percentage:.3f}%")
                logger.info(f"[EXEC_SIM]   Execution time: {result.execution_time_ms:.1f}ms")
                logger.info(f"[EXEC_SIM]   Quality score: {trade.execution_quality_score:.1f}/100")
                logger.info(f"[EXEC_SIM]   P&L: ${trade.actual_pnl:.4f}")
            else:
                logger.warning(f"[EXEC_SIM] ❌ Trade failed: {trade.trade_id}")
                if trade.execution_result:
                    logger.warning(f"[EXEC_SIM]   Error: {trade.execution_result.error_message}")
                    
        except Exception as e:
            logger.error(f"[EXEC_SIM] Error logging trade execution: {e}")
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """Get overall simulation statistics"""
        try:
            total_trades = len(self.execution_history)
            successful_trades = sum(1 for t in self.execution_history if t.is_executed)
            
            return {
                'active_backtests': len(self.active_backtests),
                'total_simulated_trades': total_trades,
                'successful_trades': successful_trades,
                'success_rate': successful_trades / max(total_trades, 1) * 100,
                'avg_execution_quality': np.mean([t.execution_quality_score for t in self.execution_history if t.is_executed]) if successful_trades > 0 else 0,
                'market_simulator_stats': self.market_simulator.get_simulation_stats()
            }
            
        except Exception as e:
            logger.error(f"[EXEC_SIM] Error getting simulation stats: {e}")
            return {}