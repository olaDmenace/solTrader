import asyncio
import sqlite3
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import math
import random

from src.database.db_manager import DatabaseManager
from src.portfolio.portfolio_manager import PortfolioManager
from src.trading.risk_engine import RiskEngine, RiskEngineConfig, TradeRisk
from src.trading.trade_types import TradeType, TradeDirection
from src.monitoring.system_monitor import SystemMonitor
from src.arbitrage.real_dex_connector import RealDEXConnector


class BacktestMode(Enum):
    SIMPLE = "simple"
    REALISTIC = "realistic"
    PRODUCTION = "production"


class ExecutionQuality(Enum):
    PERFECT = "perfect"
    MARKET = "market"  
    REALISTIC = "realistic"
    STRESSED = "stressed"


@dataclass
class MarketCondition:
    volatility: float  # Annualized volatility
    spread: float      # Bid-ask spread as percentage
    liquidity_depth: float  # Available liquidity in units
    impact_coefficient: float  # Price impact per unit traded
    regime: str        # "trending", "ranging", "volatile", "stable"
    timestamp: datetime


@dataclass
class ExecutionResult:
    requested_price: float
    executed_price: float
    requested_quantity: float
    executed_quantity: float
    slippage: float
    impact_cost: float
    timing_delay_ms: float
    partial_fill: bool
    market_condition: MarketCondition


@dataclass
class BacktestTrade:
    trade_id: str
    strategy_name: str
    symbol: str
    direction: TradeDirection
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    entry_time: datetime
    exit_time: Optional[datetime]
    execution_result: ExecutionResult
    pnl: Optional[float]
    fees: float
    metadata: Dict[str, Any]


@dataclass
class BacktestResult:
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_duration: timedelta
    total_trades: int
    profitable_trades: int
    total_fees: float
    start_capital: float
    end_capital: float
    trades: List[BacktestTrade]
    daily_returns: List[float]
    equity_curve: List[Tuple[datetime, float]]
    metrics: Dict[str, float]


@dataclass
class WalkForwardPeriod:
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime
    optimization_params: Dict[str, Any]
    oos_performance: Dict[str, float]


class ProductionBacktester:
    def __init__(
        self,
        db_manager: DatabaseManager,
        risk_engine: RiskEngine,
        monitor: SystemMonitor,
        mode: BacktestMode = BacktestMode.PRODUCTION
    ):
        self.db_manager = db_manager
        self.risk_engine = risk_engine
        self.monitor = monitor
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        # Execution modeling parameters
        self.execution_configs = {
            ExecutionQuality.PERFECT: {
                "slippage_range": (0.0, 0.0),
                "impact_factor": 0.0,
                "timing_delay_ms": (0, 0),
                "partial_fill_probability": 0.0
            },
            ExecutionQuality.MARKET: {
                "slippage_range": (0.01, 0.03),
                "impact_factor": 0.001,
                "timing_delay_ms": (100, 500),
                "partial_fill_probability": 0.05
            },
            ExecutionQuality.REALISTIC: {
                "slippage_range": (0.02, 0.08),
                "impact_factor": 0.003,
                "timing_delay_ms": (200, 1000),
                "partial_fill_probability": 0.15
            },
            ExecutionQuality.STRESSED: {
                "slippage_range": (0.05, 0.20),
                "impact_factor": 0.008,
                "timing_delay_ms": (500, 3000),
                "partial_fill_probability": 0.35
            }
        }
        
        # Market regime modeling
        self.regime_configs = {
            "trending": {"volatility": 0.25, "spread": 0.002, "liquidity": 1000.0},
            "ranging": {"volatility": 0.15, "spread": 0.001, "liquidity": 1500.0},
            "volatile": {"volatility": 0.45, "spread": 0.004, "liquidity": 500.0},
            "stable": {"volatility": 0.10, "spread": 0.0005, "liquidity": 2000.0}
        }
        
        self.current_market_condition = None
        self.execution_quality = ExecutionQuality.REALISTIC
        
    async def initialize(self):
        """Initialize backtesting framework"""
        try:
            # Create backtesting tables
            await self._create_backtesting_tables()
            
            # Initialize market condition
            self.current_market_condition = MarketCondition(
                volatility=0.20,
                spread=0.002,
                liquidity_depth=1000.0,
                impact_coefficient=0.001,
                regime="stable",
                timestamp=datetime.now()
            )
            
            self.logger.info("Production backtester initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize backtester: {e}")
            raise
            
    async def _create_backtesting_tables(self):
        """Create database tables for backtesting"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            # Backtest results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id TEXT UNIQUE NOT NULL,
                    strategy_name TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    total_return REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    profitable_trades INTEGER NOT NULL,
                    total_fees REAL NOT NULL,
                    start_capital REAL NOT NULL,
                    end_capital REAL NOT NULL,
                    execution_quality TEXT NOT NULL,
                    market_conditions TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Backtest trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id TEXT NOT NULL,
                    trade_id TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    pnl REAL,
                    fees REAL NOT NULL,
                    slippage REAL NOT NULL,
                    impact_cost REAL NOT NULL,
                    execution_delay_ms REAL NOT NULL,
                    partial_fill INTEGER NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Walk-forward analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS walk_forward_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT UNIQUE NOT NULL,
                    strategy_name TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    in_sample_start TEXT NOT NULL,
                    in_sample_end TEXT NOT NULL,
                    out_sample_start TEXT NOT NULL,
                    out_sample_end TEXT NOT NULL,
                    optimization_params TEXT NOT NULL,
                    oos_performance TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to create backtesting tables: {e}")
            raise
            
    def simulate_market_condition(self, timestamp: datetime, regime: str = None) -> MarketCondition:
        """Simulate realistic market conditions"""
        if regime is None:
            # Randomly assign regime based on time patterns
            hour = timestamp.hour
            if 9 <= hour <= 16:  # Trading hours
                regime = random.choices(
                    ["trending", "ranging", "volatile", "stable"],
                    weights=[0.25, 0.40, 0.20, 0.15]
                )[0]
            else:  # After hours
                regime = random.choices(
                    ["ranging", "stable", "volatile"],
                    weights=[0.50, 0.35, 0.15]
                )[0]
                
        config = self.regime_configs[regime]
        
        # Add some noise to base values
        volatility = config["volatility"] * (0.8 + 0.4 * random.random())
        spread = config["spread"] * (0.7 + 0.6 * random.random())
        liquidity = config["liquidity"] * (0.6 + 0.8 * random.random())
        
        return MarketCondition(
            volatility=volatility,
            spread=spread,
            liquidity_depth=liquidity,
            impact_coefficient=0.001 * (volatility / 0.20),  # Scale with volatility
            regime=regime,
            timestamp=timestamp
        )
        
    def simulate_execution(
        self, 
        requested_price: float,
        requested_quantity: float,
        market_condition: MarketCondition,
        direction: TradeDirection
    ) -> ExecutionResult:
        """Simulate realistic trade execution"""
        config = self.execution_configs[self.execution_quality]
        
        # Calculate slippage
        base_slippage = random.uniform(*config["slippage_range"])
        volatility_adjustment = market_condition.volatility / 0.20  # Normalize to 20% vol
        regime_adjustment = {"trending": 0.8, "ranging": 1.0, "volatile": 1.5, "stable": 0.6}[market_condition.regime]
        
        total_slippage = base_slippage * volatility_adjustment * regime_adjustment
        
        # Calculate market impact
        impact_cost = config["impact_factor"] * math.sqrt(requested_quantity / market_condition.liquidity_depth)
        impact_cost *= market_condition.impact_coefficient
        
        # Determine execution delay
        timing_delay = random.uniform(*config["timing_delay_ms"])
        
        # Check for partial fill
        partial_fill = random.random() < config["partial_fill_probability"]
        if partial_fill:
            executed_quantity = requested_quantity * random.uniform(0.6, 0.95)
        else:
            executed_quantity = requested_quantity
            
        # Calculate final executed price
        price_adjustment = total_slippage + impact_cost
        if direction == TradeDirection.BUY:
            executed_price = requested_price * (1 + price_adjustment)
        else:
            executed_price = requested_price * (1 - price_adjustment)
            
        return ExecutionResult(
            requested_price=requested_price,
            executed_price=executed_price,
            requested_quantity=requested_quantity,
            executed_quantity=executed_quantity,
            slippage=total_slippage,
            impact_cost=impact_cost,
            timing_delay_ms=timing_delay,
            partial_fill=partial_fill,
            market_condition=market_condition
        )
        
    async def run_backtest(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000.0,
        execution_quality: ExecutionQuality = ExecutionQuality.REALISTIC
    ) -> BacktestResult:
        """Run comprehensive backtest with realistic execution modeling"""
        self.execution_quality = execution_quality
        backtest_id = f"{strategy_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            self.logger.info(f"Starting backtest {backtest_id}")
            
            # Initialize tracking variables
            trades = []
            current_capital = initial_capital
            equity_curve = [(start_date, current_capital)]
            daily_returns = []
            total_fees = 0.0
            
            # Get historical data for the period
            market_data = await self._get_market_data(start_date, end_date)
            
            # Simulate trading day by day
            current_date = start_date
            while current_date <= end_date:
                # Update market conditions for this day
                market_condition = self.simulate_market_condition(current_date)
                
                # Get strategy signals for this date
                signals = await self._get_strategy_signals(strategy_name, current_date, market_data)
                
                # Execute trades based on signals
                for signal in signals:
                    if await self._should_execute_trade(signal, current_capital):
                        trade = await self._execute_backtest_trade(
                            signal, market_condition, backtest_id
                        )
                        trades.append(trade)
                        
                        # Update capital
                        if trade.pnl is not None:
                            current_capital += trade.pnl
                        current_capital -= trade.fees
                        total_fees += trade.fees
                        
                # Record daily equity
                equity_curve.append((current_date, current_capital))
                
                # Calculate daily return
                if len(equity_curve) > 1:
                    daily_return = (current_capital - equity_curve[-2][1]) / equity_curve[-2][1]
                    daily_returns.append(daily_return)
                    
                current_date += timedelta(days=1)
                
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(
                trades, daily_returns, initial_capital, current_capital
            )
            
            # Create backtest result
            result = BacktestResult(
                strategy_name=strategy_name,
                total_return=(current_capital - initial_capital) / initial_capital,
                sharpe_ratio=metrics["sharpe_ratio"],
                max_drawdown=metrics["max_drawdown"],
                win_rate=metrics["win_rate"],
                avg_trade_duration=metrics["avg_trade_duration"],
                total_trades=len(trades),
                profitable_trades=len([t for t in trades if t.pnl and t.pnl > 0]),
                total_fees=total_fees,
                start_capital=initial_capital,
                end_capital=current_capital,
                trades=trades,
                daily_returns=daily_returns,
                equity_curve=equity_curve,
                metrics=metrics
            )
            
            # Store backtest results
            await self._store_backtest_results(backtest_id, result)
            
            self.logger.info(f"Backtest {backtest_id} completed. Total return: {result.total_return:.2%}")
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
            
    async def run_walk_forward_analysis(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        in_sample_months: int = 6,
        out_sample_months: int = 1,
        optimization_params: List[str] = None
    ) -> List[WalkForwardPeriod]:
        """Run walk-forward analysis with parameter optimization"""
        analysis_id = f"wf_{strategy_name}_{start_date.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            self.logger.info(f"Starting walk-forward analysis {analysis_id}")
            
            periods = []
            current_start = start_date
            
            while current_start < end_date:
                # Define in-sample period
                in_sample_end = current_start + timedelta(days=in_sample_months * 30)
                if in_sample_end > end_date:
                    break
                    
                # Define out-of-sample period
                oos_start = in_sample_end + timedelta(days=1)
                oos_end = oos_start + timedelta(days=out_sample_months * 30)
                if oos_end > end_date:
                    oos_end = end_date
                    
                self.logger.info(f"Processing period: {current_start} to {oos_end}")
                
                # Optimize parameters on in-sample data
                optimal_params = await self._optimize_parameters(
                    strategy_name, current_start, in_sample_end, optimization_params
                )
                
                # Test on out-of-sample data
                oos_result = await self.run_backtest(
                    strategy_name, oos_start, oos_end
                )
                
                period = WalkForwardPeriod(
                    in_sample_start=current_start,
                    in_sample_end=in_sample_end,
                    out_sample_start=oos_start,
                    out_sample_end=oos_end,
                    optimization_params=optimal_params,
                    oos_performance={
                        "total_return": oos_result.total_return,
                        "sharpe_ratio": oos_result.sharpe_ratio,
                        "max_drawdown": oos_result.max_drawdown,
                        "win_rate": oos_result.win_rate
                    }
                )
                periods.append(period)
                
                # Store walk-forward period
                await self._store_walk_forward_period(analysis_id, period)
                
                # Move to next period
                current_start = oos_start
                
            self.logger.info(f"Walk-forward analysis completed. {len(periods)} periods processed")
            return periods
            
        except Exception as e:
            self.logger.error(f"Walk-forward analysis failed: {e}")
            raise
            
    async def _get_market_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get historical market data for backtesting"""
        # This would typically fetch from a data provider
        # For now, return simulated data structure
        return {
            "prices": {},
            "volumes": {},
            "spreads": {}
        }
        
    async def _get_strategy_signals(
        self, strategy_name: str, date: datetime, market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get trading signals from strategy for given date"""
        # This would integrate with actual strategy implementations
        # For now, return mock signals
        return []
        
    async def _should_execute_trade(self, signal: Dict[str, Any], available_capital: float) -> bool:
        """Determine if trade should be executed based on risk management"""
        # Integrate with risk engine
        return True
        
    async def _execute_backtest_trade(
        self, signal: Dict[str, Any], market_condition: MarketCondition, backtest_id: str
    ) -> BacktestTrade:
        """Execute a single trade in backtest"""
        trade_id = f"{backtest_id}_trade_{len(signal)}"
        
        # Simulate execution
        execution = self.simulate_execution(
            requested_price=signal.get("price", 100.0),
            requested_quantity=signal.get("quantity", 1.0),
            market_condition=market_condition,
            direction=TradeDirection.BUY  # Mock
        )
        
        # Calculate fees (0.1% for example)
        fees = execution.executed_price * execution.executed_quantity * 0.001
        
        return BacktestTrade(
            trade_id=trade_id,
            strategy_name=signal.get("strategy", "mock"),
            symbol=signal.get("symbol", "SOL/USDC"),
            direction=TradeDirection.BUY,
            entry_price=execution.executed_price,
            exit_price=None,
            quantity=execution.executed_quantity,
            entry_time=market_condition.timestamp,
            exit_time=None,
            execution_result=execution,
            pnl=None,
            fees=fees,
            metadata={}
        )
        
    def _calculate_performance_metrics(
        self, trades: List[BacktestTrade], daily_returns: List[float], 
        start_capital: float, end_capital: float
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not daily_returns:
            return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "avg_trade_duration": timedelta()}
            
        # Sharpe ratio
        returns_array = np.array(daily_returns)
        sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0.0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Win rate
        profitable_trades = [t for t in trades if t.pnl and t.pnl > 0]
        win_rate = len(profitable_trades) / len(trades) if trades else 0.0
        
        # Average trade duration
        completed_trades = [t for t in trades if t.exit_time and t.entry_time]
        if completed_trades:
            durations = [(t.exit_time - t.entry_time).total_seconds() for t in completed_trades]
            avg_duration = timedelta(seconds=sum(durations) / len(durations))
        else:
            avg_duration = timedelta()
            
        return {
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": abs(max_drawdown),
            "win_rate": win_rate,
            "avg_trade_duration": avg_duration,
            "total_return": (end_capital - start_capital) / start_capital,
            "volatility": np.std(returns_array) * np.sqrt(252),
            "best_trade": max([t.pnl for t in trades if t.pnl], default=0.0),
            "worst_trade": min([t.pnl for t in trades if t.pnl], default=0.0)
        }
        
    async def _optimize_parameters(
        self, strategy_name: str, start_date: datetime, end_date: datetime,
        param_names: List[str] = None
    ) -> Dict[str, Any]:
        """Optimize strategy parameters on in-sample data"""
        # This would implement parameter optimization
        # For now, return default parameters
        return {"param1": 1.0, "param2": 2.0}
        
    async def _store_backtest_results(self, backtest_id: str, result: BacktestResult):
        """Store backtest results in database"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            # Store main result
            cursor.execute('''
                INSERT OR REPLACE INTO backtest_results 
                (backtest_id, strategy_name, start_date, end_date, total_return, sharpe_ratio,
                 max_drawdown, win_rate, total_trades, profitable_trades, total_fees,
                 start_capital, end_capital, execution_quality, market_conditions, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                backtest_id, result.strategy_name, 
                result.equity_curve[0][0].isoformat(), result.equity_curve[-1][0].isoformat(),
                result.total_return, result.sharpe_ratio, result.max_drawdown, result.win_rate,
                result.total_trades, result.profitable_trades, result.total_fees,
                result.start_capital, result.end_capital,
                self.execution_quality.value, json.dumps({}), json.dumps(result.metrics),
                datetime.now().isoformat()
            ))
            
            # Store individual trades
            for trade in result.trades:
                cursor.execute('''
                    INSERT INTO backtest_trades 
                    (backtest_id, trade_id, strategy_name, symbol, direction, entry_price,
                     exit_price, quantity, entry_time, exit_time, pnl, fees, slippage,
                     impact_cost, execution_delay_ms, partial_fill, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    backtest_id, trade.trade_id, trade.strategy_name, trade.symbol,
                    trade.direction.value, trade.entry_price, trade.exit_price,
                    trade.quantity, trade.entry_time.isoformat(),
                    trade.exit_time.isoformat() if trade.exit_time else None,
                    trade.pnl, trade.fees, trade.execution_result.slippage,
                    trade.execution_result.impact_cost, trade.execution_result.timing_delay_ms,
                    1 if trade.execution_result.partial_fill else 0,
                    json.dumps(trade.metadata), datetime.now().isoformat()
                ))
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store backtest results: {e}")
            raise
            
    async def _store_walk_forward_period(self, analysis_id: str, period: WalkForwardPeriod):
        """Store walk-forward analysis period"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO walk_forward_analysis
                (analysis_id, strategy_name, period_start, period_end, in_sample_start,
                 in_sample_end, out_sample_start, out_sample_end, optimization_params,
                 oos_performance, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id, "strategy_name", period.in_sample_start.isoformat(),
                period.out_sample_end.isoformat(), period.in_sample_start.isoformat(),
                period.in_sample_end.isoformat(), period.out_sample_start.isoformat(),
                period.out_sample_end.isoformat(), json.dumps(period.optimization_params),
                json.dumps(period.oos_performance), datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store walk-forward period: {e}")
            raise
            
    async def shutdown(self):
        """Cleanup backtesting resources"""
        self.logger.info("Shutting down production backtester")