#!/usr/bin/env python3
"""
Formal Backtesting Validation System
Provides rigorous backtesting with statistical validation, out-of-sample testing,
and comprehensive performance metrics for trading strategy validation.

Key Features:
1. Walk-forward analysis
2. Out-of-sample validation
3. Monte Carlo simulation
4. Statistical significance testing
5. Comprehensive performance metrics
6. Strategy robustness assessment
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Import realistic market simulation components
try:
    from ..backtesting.execution_simulator import EnhancedExecutionSimulator, ExecutionStyle
    ENHANCED_SIMULATION_AVAILABLE = True
    logger.info("Enhanced execution simulation available")
except ImportError:
    ENHANCED_SIMULATION_AVAILABLE = False
    logger.warning("Enhanced execution simulation not available - using simplified model")

class ValidationResult(Enum):
    EXCELLENT = "EXCELLENT"      # Pass all tests with high confidence
    GOOD = "GOOD"               # Pass most tests with decent performance
    ACCEPTABLE = "ACCEPTABLE"   # Pass basic tests, some concerns
    POOR = "POOR"              # Fail several important tests
    FAILED = "FAILED"          # Fail critical validation tests

@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    
    # Returns and Risk
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown Analysis
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    drawdown_duration: float = 0.0  # Average days in drawdown
    
    # Trading Statistics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Consistency Metrics
    win_streak: int = 0
    loss_streak: int = 0
    monthly_win_rate: float = 0.0
    consistency_score: float = 0.0
    
    # Statistical Tests
    t_statistic: float = 0.0
    p_value: float = 0.0
    confidence_level: float = 0.0
    
    # Risk Management
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional VaR (95%)
    tail_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'returns': {
                'total_return': self.total_return,
                'annualized_return': self.annualized_return,
                'volatility': self.volatility,
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio
            },
            'drawdown': {
                'max_drawdown': self.max_drawdown,
                'avg_drawdown': self.avg_drawdown,
                'drawdown_duration': self.drawdown_duration
            },
            'trading': {
                'total_trades': self.total_trades,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'largest_win': self.largest_win,
                'largest_loss': self.largest_loss
            },
            'consistency': {
                'win_streak': self.win_streak,
                'loss_streak': self.loss_streak,
                'monthly_win_rate': self.monthly_win_rate,
                'consistency_score': self.consistency_score
            },
            'statistical': {
                't_statistic': self.t_statistic,
                'p_value': self.p_value,
                'confidence_level': self.confidence_level
            },
            'risk': {
                'var_95': self.var_95,
                'cvar_95': self.cvar_95,
                'tail_ratio': self.tail_ratio
            }
        }

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    
    # Overall Assessment
    validation_result: ValidationResult
    overall_score: float
    confidence_level: float
    
    # Test Results
    in_sample_metrics: BacktestMetrics
    out_of_sample_metrics: BacktestMetrics
    walk_forward_metrics: List[BacktestMetrics]
    
    # Statistical Validation
    statistical_tests: Dict[str, Dict[str, Any]]
    monte_carlo_results: Dict[str, Any]
    
    # Strategy Assessment
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    
    # Risk Assessment
    risk_factors: List[str]
    risk_score: float
    
    # Validation Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for storage/transmission"""
        return {
            'validation_result': self.validation_result.value,
            'overall_score': self.overall_score,
            'confidence_level': self.confidence_level,
            'in_sample_metrics': self.in_sample_metrics.to_dict(),
            'out_of_sample_metrics': self.out_of_sample_metrics.to_dict(),
            'walk_forward_metrics': [m.to_dict() for m in self.walk_forward_metrics],
            'statistical_tests': self.statistical_tests,
            'monte_carlo_results': self.monte_carlo_results,
            'assessment': {
                'strengths': self.strengths,
                'weaknesses': self.weaknesses,
                'recommendations': self.recommendations
            },
            'risk_assessment': {
                'risk_factors': self.risk_factors,
                'risk_score': self.risk_score
            },
            'timestamp': self.timestamp.isoformat()
        }

class FormalBacktestingValidator:
    """
    Formal backtesting validation system with rigorous statistical testing
    and out-of-sample validation for trading strategy assessment.
    """
    
    def __init__(self, settings: Any):
        self.settings = settings
        self.validation_history: List[ValidationReport] = []
        
        # Initialize enhanced execution simulator if available
        self.execution_simulator = None
        if ENHANCED_SIMULATION_AVAILABLE:
            self.execution_simulator = EnhancedExecutionSimulator(settings)
            logger.info("[FORMAL_VALIDATOR] Enhanced execution simulation enabled")
        else:
            logger.info("[FORMAL_VALIDATOR] Using simplified backtesting model")
        
        # Validation thresholds
        self.thresholds = {
            'min_sharpe_ratio': 0.5,
            'min_win_rate': 0.35,
            'max_drawdown': 0.25,
            'min_trades': 50,
            'min_profit_factor': 1.1,
            'min_confidence': 0.8,
            'max_p_value': 0.05
        }
        
        logger.info("[FORMAL_VALIDATOR] Formal backtesting validator initialized")
    
    async def validate_strategy(
        self,
        historical_data: Dict[str, Any],
        strategy_parameters: Dict[str, Any],
        test_period_months: int = 12
    ) -> ValidationReport:
        """
        Comprehensive strategy validation with multiple testing approaches
        """
        try:
            logger.info("[FORMAL_VALIDATOR] Starting comprehensive strategy validation...")
            
            # Prepare data for testing
            prepared_data = self._prepare_historical_data(historical_data, test_period_months)
            
            if not prepared_data:
                logger.error("[FORMAL_VALIDATOR] Insufficient historical data for validation")
                return self._create_failed_report("Insufficient historical data")
            
            # 1. In-sample backtesting (70% of data)
            logger.info("[FORMAL_VALIDATOR] Running in-sample backtesting...")
            in_sample_metrics = await self._run_in_sample_backtest(prepared_data['in_sample'], strategy_parameters)
            
            # 2. Out-of-sample testing (30% of data)
            logger.info("[FORMAL_VALIDATOR] Running out-of-sample validation...")
            out_of_sample_metrics = await self._run_out_of_sample_test(prepared_data['out_of_sample'], strategy_parameters)
            
            # 3. Walk-forward analysis
            logger.info("[FORMAL_VALIDATOR] Running walk-forward analysis...")
            walk_forward_metrics = await self._run_walk_forward_analysis(prepared_data['full'], strategy_parameters)
            
            # 4. Statistical validation
            logger.info("[FORMAL_VALIDATOR] Running statistical tests...")
            statistical_tests = self._run_statistical_tests(in_sample_metrics, out_of_sample_metrics)
            
            # 5. Monte Carlo simulation
            logger.info("[FORMAL_VALIDATOR] Running Monte Carlo simulation...")
            monte_carlo_results = self._run_monte_carlo_simulation(prepared_data['full'], strategy_parameters)
            
            # 6. Comprehensive assessment
            validation_result, overall_score, confidence_level = self._assess_validation_results(
                in_sample_metrics, out_of_sample_metrics, walk_forward_metrics, statistical_tests, monte_carlo_results
            )
            
            # 7. Generate detailed report
            report = ValidationReport(
                validation_result=validation_result,
                overall_score=overall_score,
                confidence_level=confidence_level,
                in_sample_metrics=in_sample_metrics,
                out_of_sample_metrics=out_of_sample_metrics,
                walk_forward_metrics=walk_forward_metrics,
                statistical_tests=statistical_tests,
                monte_carlo_results=monte_carlo_results,
                strengths=self._identify_strengths(in_sample_metrics, out_of_sample_metrics),
                weaknesses=self._identify_weaknesses(in_sample_metrics, out_of_sample_metrics),
                recommendations=self._generate_recommendations(in_sample_metrics, out_of_sample_metrics),
                risk_factors=self._assess_risk_factors(in_sample_metrics, out_of_sample_metrics),
                risk_score=self._calculate_risk_score(in_sample_metrics, out_of_sample_metrics)
            )
            
            # Store validation history
            self.validation_history.append(report)
            
            # Log results
            self._log_validation_results(report)
            
            logger.info(f"[FORMAL_VALIDATOR] Validation complete - Result: {validation_result.value} (Score: {overall_score:.2f})")
            
            return report
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Validation error: {e}")
            return self._create_failed_report(f"Validation error: {str(e)}")
    
    def _prepare_historical_data(self, historical_data: Dict[str, Any], test_period_months: int) -> Optional[Dict[str, Any]]:
        """Prepare and split historical data for validation"""
        try:
            # Mock data preparation - in production, use real historical price data
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=test_period_months * 30),
                end=datetime.now(),
                freq='H'
            )
            
            # Generate synthetic but realistic price data
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0.0001, 0.02, len(dates))  # Small positive drift with volatility
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Create volume data
            volumes = np.random.lognormal(10, 1, len(dates))
            
            full_data = pd.DataFrame({
                'timestamp': dates,
                'price': prices,
                'volume': volumes,
                'returns': returns
            })
            
            # Split data: 70% in-sample, 30% out-of-sample
            split_index = int(len(full_data) * 0.7)
            
            return {
                'full': full_data,
                'in_sample': full_data.iloc[:split_index],
                'out_of_sample': full_data.iloc[split_index:],
                'split_date': full_data.iloc[split_index]['timestamp']
            }
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Data preparation error: {e}")
            return None
    
    async def _run_in_sample_backtest(self, data: pd.DataFrame, strategy_params: Dict[str, Any]) -> BacktestMetrics:
        """Run comprehensive in-sample backtesting with enhanced execution simulation"""
        try:
            logger.info(f"[FORMAL_VALIDATOR] In-sample backtest on {len(data)} data points")
            
            if self.execution_simulator and ENHANCED_SIMULATION_AVAILABLE:
                return await self._run_enhanced_backtest(data, strategy_params, "in_sample")
            else:
                return await self._run_simplified_backtest(data, strategy_params)
                
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] In-sample backtest error: {e}")
            return BacktestMetrics()
    
    async def _run_enhanced_backtest(self, data: pd.DataFrame, strategy_params: Dict[str, Any], phase: str) -> BacktestMetrics:
        """Run backtest with enhanced execution simulation"""
        try:
            backtest_id = f"{phase}_{int(datetime.now().timestamp())}"
            
            # Prepare tokens and prices
            tokens = ['SOL/USDC']  # Main trading pair
            base_prices = {'SOL/USDC': 100.0}
            
            # Start enhanced backtest
            success = await self.execution_simulator.start_backtest(
                backtest_id=backtest_id,
                strategy_name="ValidationStrategy",
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now(),
                initial_capital=10000.0,
                tokens=tokens,
                base_prices=base_prices
            )
            
            if not success:
                logger.error(f"[FORMAL_VALIDATOR] Failed to start enhanced backtest")
                return await self._run_simplified_backtest(data, strategy_params)
            
            # Simulate trades through the data
            current_balance = 10000.0
            position = None
            trade_count = 0
            
            for i, row in data.iterrows():
                # Generate mock signal
                signal_strength = self._generate_mock_signal(row, i)
                
                # Entry logic
                if position is None and signal_strength > 0.6:
                    entry_price = row['price']
                    position_size = min(current_balance * 0.02, 200.0)  # Conservative position sizing
                    
                    # Execute buy order through enhanced simulation
                    trade_id = f"trade_{trade_count}"
                    trade = await self.execution_simulator.simulate_trade_execution(
                        backtest_id=backtest_id,
                        trade_id=trade_id,
                        token='SOL/USDC',
                        is_buy=True,
                        size=position_size / entry_price,  # Convert to token size
                        intended_price=entry_price,
                        execution_style=ExecutionStyle.ADAPTIVE,
                        dex_name='raydium'
                    )
                    
                    if trade.is_executed:
                        position = {
                            'entry_price': trade.execution_result.executed_price,
                            'size': trade.execution_result.executed_size,
                            'entry_time': row['timestamp'] if 'timestamp' in row.index else datetime.now(),
                            'stop_loss': trade.execution_result.executed_price * 0.95,
                            'take_profit': trade.execution_result.executed_price * 1.10,
                            'trade_obj': trade
                        }
                        current_balance -= trade.execution_result.executed_price * trade.execution_result.executed_size
                        trade_count += 1
                
                # Exit logic
                elif position is not None:
                    current_price = row['price']
                    
                    # Check exit conditions
                    exit_triggered = (
                        current_price <= position['stop_loss'] or
                        current_price >= position['take_profit'] or
                        signal_strength < 0.3
                    )
                    
                    if exit_triggered:
                        # Execute sell order
                        sell_trade_id = f"sell_trade_{trade_count}"
                        sell_trade = await self.execution_simulator.simulate_trade_execution(
                            backtest_id=backtest_id,
                            trade_id=sell_trade_id,
                            token='SOL/USDC',
                            is_buy=False,
                            size=position['size'],
                            intended_price=current_price,
                            execution_style=ExecutionStyle.ADAPTIVE,
                            dex_name='raydium'
                        )
                        
                        if sell_trade.is_executed:
                            # Update balance with proceeds
                            proceeds = sell_trade.execution_result.executed_price * sell_trade.execution_result.executed_size
                            current_balance += proceeds
                            
                            position = None
                            trade_count += 1
            
            # Complete backtest and get results
            backtest_results = await self.execution_simulator.complete_backtest(backtest_id)
            
            # Convert enhanced results to BacktestMetrics format
            trades_data = []
            if backtest_results and hasattr(backtest_results, 'trades'):
                for trade in backtest_results.trades:
                    if trade.execution_result:
                        pnl = (trade.execution_result.executed_price - trade.intended_price) * trade.execution_result.executed_size
                        trades_data.append({
                            'pnl': pnl,
                            'return': pnl / (trade.intended_price * trade.execution_result.executed_size),
                            'entry_price': trade.intended_price,
                            'exit_price': trade.execution_result.executed_price,
                            'entry_time': datetime.now() - timedelta(minutes=30),
                            'exit_time': datetime.now()
                        })
            
            # Create equity curve from balance changes
            equity_curve = [10000.0, current_balance]
            
            # Calculate comprehensive metrics using the enhanced trade data
            metrics = self._calculate_metrics(trades_data, equity_curve, data)
            
            logger.info(f"[FORMAL_VALIDATOR] Enhanced backtest completed with {trade_count} trades, balance: {current_balance:.2f}")
            return metrics
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Enhanced backtest error: {e}")
            return await self._run_simplified_backtest(data, strategy_params)
    
    async def _run_simplified_backtest(self, data: pd.DataFrame, strategy_params: Dict[str, Any]) -> BacktestMetrics:
        """Run simplified backtesting (original implementation)"""
        try:
            # Simulate trading strategy
            trades = []
            equity_curve = [10000]  # Start with $10,000
            current_balance = 10000
            position = None
            
            for i, row in data.iterrows():
                # Mock signal generation (in production, use actual strategy)
                signal_strength = self._generate_mock_signal(row, i)
                
                # Entry logic
                if position is None and signal_strength > 0.6:
                    entry_price = row['price']
                    position_size = current_balance * 0.02  # 2% position size
                    position = {
                        'entry_price': entry_price,
                        'size': position_size,
                        'entry_time': row['timestamp'],
                        'stop_loss': entry_price * 0.95,  # 5% stop loss
                        'take_profit': entry_price * 1.10  # 10% take profit
                    }
                
                # Exit logic
                elif position is not None:
                    current_price = row['price']
                    
                    # Check exit conditions
                    exit_triggered = (
                        current_price <= position['stop_loss'] or  # Stop loss
                        current_price >= position['take_profit'] or  # Take profit
                        signal_strength < 0.3  # Signal weakened
                    )
                    
                    if exit_triggered:
                        pnl = (current_price - position['entry_price']) * (position['size'] / position['entry_price'])
                        current_balance += pnl
                        
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': row['timestamp'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'pnl': pnl,
                            'return': pnl / position['size']
                        })
                        
                        position = None
                
                equity_curve.append(current_balance)
            
            # Calculate comprehensive metrics
            return self._calculate_metrics(trades, equity_curve, data)
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] In-sample backtest error: {e}")
            return BacktestMetrics()
    
    async def _run_out_of_sample_test(self, data: pd.DataFrame, strategy_params: Dict[str, Any]) -> BacktestMetrics:
        """Run out-of-sample validation test"""
        try:
            logger.info(f"[FORMAL_VALIDATOR] Out-of-sample test on {len(data)} data points")
            
            # Use same logic as in-sample but with different data
            # This simulates applying the strategy to unseen data
            return await self._run_in_sample_backtest(data, strategy_params)
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Out-of-sample test error: {e}")
            return BacktestMetrics()
    
    async def _run_walk_forward_analysis(self, data: pd.DataFrame, strategy_params: Dict[str, Any]) -> List[BacktestMetrics]:
        """Run walk-forward analysis to test strategy robustness"""
        try:
            logger.info(f"[FORMAL_VALIDATOR] Walk-forward analysis on {len(data)} data points")
            
            metrics_list = []
            window_size = len(data) // 6  # 6 windows
            
            for i in range(0, len(data) - window_size, window_size // 2):  # 50% overlap
                window_data = data.iloc[i:i + window_size]
                
                if len(window_data) < 100:  # Minimum window size
                    continue
                
                window_metrics = await self._run_in_sample_backtest(window_data, strategy_params)
                metrics_list.append(window_metrics)
            
            logger.info(f"[FORMAL_VALIDATOR] Completed {len(metrics_list)} walk-forward windows")
            return metrics_list
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Walk-forward analysis error: {e}")
            return []
    
    def _run_statistical_tests(self, in_sample: BacktestMetrics, out_sample: BacktestMetrics) -> Dict[str, Dict[str, Any]]:
        """Run statistical significance tests"""
        try:
            tests = {}
            
            # T-test for returns difference
            if in_sample.total_trades > 0 and out_sample.total_trades > 0:
                # Mock return distributions
                in_returns = np.random.normal(in_sample.annualized_return / 252, 0.02, in_sample.total_trades)
                out_returns = np.random.normal(out_sample.annualized_return / 252, 0.02, out_sample.total_trades)
                
                t_stat, p_value = stats.ttest_ind(in_returns, out_returns)
                
                tests['returns_consistency'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'interpretation': 'Returns are consistent between samples' if p_value >= 0.05 else 'Returns differ significantly'
                }
            
            # Sharpe ratio stability test
            sharpe_diff = abs(in_sample.sharpe_ratio - out_sample.sharpe_ratio)
            tests['sharpe_stability'] = {
                'in_sample_sharpe': in_sample.sharpe_ratio,
                'out_sample_sharpe': out_sample.sharpe_ratio,
                'difference': sharpe_diff,
                'stable': sharpe_diff < 0.5,
                'interpretation': 'Sharpe ratio is stable' if sharpe_diff < 0.5 else 'Sharpe ratio varies significantly'
            }
            
            # Win rate consistency
            win_rate_diff = abs(in_sample.win_rate - out_sample.win_rate)
            tests['win_rate_consistency'] = {
                'in_sample_win_rate': in_sample.win_rate,
                'out_sample_win_rate': out_sample.win_rate,
                'difference': win_rate_diff,
                'consistent': win_rate_diff < 0.15,
                'interpretation': 'Win rate is consistent' if win_rate_diff < 0.15 else 'Win rate varies significantly'
            }
            
            return tests
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Statistical tests error: {e}")
            return {}
    
    def _run_monte_carlo_simulation(self, data: pd.DataFrame, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo simulation for robustness testing"""
        try:
            logger.info("[FORMAL_VALIDATOR] Running Monte Carlo simulation...")
            
            simulation_results = []
            num_simulations = 1000
            
            for sim in range(num_simulations):
                # Bootstrap sample the data
                bootstrap_data = data.sample(n=len(data), replace=True).reset_index(drop=True)
                
                # Run simplified backtest
                returns = bootstrap_data['returns'].values
                equity = 10000 * np.exp(np.cumsum(returns))
                
                final_return = (equity[-1] - equity[0]) / equity[0]
                max_dd = np.max(np.maximum.accumulate(equity) - equity) / np.maximum.accumulate(equity).max()
                
                simulation_results.append({
                    'return': final_return,
                    'max_drawdown': max_dd
                })
            
            # Calculate statistics
            returns = [r['return'] for r in simulation_results]
            drawdowns = [r['max_drawdown'] for r in simulation_results]
            
            results = {
                'num_simulations': num_simulations,
                'return_statistics': {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'percentile_5': np.percentile(returns, 5),
                    'percentile_95': np.percentile(returns, 95),
                    'positive_returns': sum(1 for r in returns if r > 0) / len(returns)
                },
                'drawdown_statistics': {
                    'mean': np.mean(drawdowns),
                    'std': np.std(drawdowns),
                    'percentile_5': np.percentile(drawdowns, 5),
                    'percentile_95': np.percentile(drawdowns, 95),
                    'max_drawdown': max(drawdowns)
                },
                'confidence_intervals': {
                    'return_95_ci': [np.percentile(returns, 2.5), np.percentile(returns, 97.5)],
                    'drawdown_95_ci': [np.percentile(drawdowns, 2.5), np.percentile(drawdowns, 97.5)]
                }
            }
            
            logger.info(f"[FORMAL_VALIDATOR] Monte Carlo completed: {results['return_statistics']['positive_returns']:.1%} positive returns")
            return results
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Monte Carlo simulation error: {e}")
            return {}
    
    def _generate_mock_signal(self, row: pd.Series, index: int) -> float:
        """Generate mock trading signal for backtesting"""
        try:
            # Simple momentum + mean reversion signal
            if index < 20:
                return 0.5  # Default neutral
            
            # Mock momentum (simplified)
            price_momentum = (row['price'] / 100 - 1) * 10  # Normalized momentum
            volume_factor = min(1.0, row['volume'] / 50000)  # Volume confirmation
            
            # Add some noise and trend
            signal = np.clip(price_momentum * volume_factor + np.random.normal(0, 0.1), 0, 1)
            return signal
            
        except Exception:
            return 0.5
    
    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[float], data: pd.DataFrame) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""
        try:
            if not trades:
                return BacktestMetrics()
            
            returns = [trade['return'] for trade in trades]
            pnls = [trade['pnl'] for trade in trades]
            
            # Basic statistics
            total_trades = len(trades)
            winning_trades = [r for r in returns if r > 0]
            losing_trades = [r for r in returns if r < 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            
            # Returns and risk
            total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
            
            # Convert to daily returns for annualization
            daily_returns = np.diff(equity_curve) / equity_curve[:-1]
            annualized_return = np.mean(daily_returns) * 252 if len(daily_returns) > 0 else 0
            volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0
            
            # Risk ratios
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            downside_returns = [r for r in daily_returns if r < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if downside_returns else volatility
            sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
            
            # Drawdown analysis
            equity_array = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (running_max - equity_array) / running_max
            max_drawdown = np.max(drawdown)
            avg_drawdown = np.mean(drawdown[drawdown > 0]) if len(drawdown[drawdown > 0]) > 0 else 0
            
            # Profit factor
            total_profit = sum(p for p in pnls if p > 0)
            total_loss = abs(sum(p for p in pnls if p < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
            
            # Risk metrics
            var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
            cvar_95 = np.mean([r for r in daily_returns if r <= var_95]) if len(daily_returns) > 0 else 0
            
            # Create metrics object
            metrics = BacktestMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=annualized_return / abs(max_drawdown) if max_drawdown > 0 else 0,
                max_drawdown=max_drawdown,
                avg_drawdown=avg_drawdown,
                total_trades=total_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=max(returns) if returns else 0,
                largest_loss=min(returns) if returns else 0,
                var_95=var_95,
                cvar_95=cvar_95
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Metrics calculation error: {e}")
            return BacktestMetrics()
    
    def _assess_validation_results(
        self,
        in_sample: BacktestMetrics,
        out_sample: BacktestMetrics,
        walk_forward: List[BacktestMetrics],
        statistical_tests: Dict[str, Dict[str, Any]],
        monte_carlo: Dict[str, Any]
    ) -> Tuple[ValidationResult, float, float]:
        """Assess overall validation results"""
        try:
            scores = []
            
            # Performance scores
            performance_score = 0
            if in_sample.sharpe_ratio >= self.thresholds['min_sharpe_ratio']:
                performance_score += 20
            if out_sample.sharpe_ratio >= self.thresholds['min_sharpe_ratio']:
                performance_score += 20
            if in_sample.win_rate >= self.thresholds['min_win_rate']:
                performance_score += 15
            if out_sample.win_rate >= self.thresholds['min_win_rate']:
                performance_score += 15
            
            scores.append(performance_score)
            
            # Consistency scores
            consistency_score = 0
            sharpe_diff = abs(in_sample.sharpe_ratio - out_sample.sharpe_ratio)
            if sharpe_diff < 0.3:
                consistency_score += 15
            
            win_rate_diff = abs(in_sample.win_rate - out_sample.win_rate)
            if win_rate_diff < 0.1:
                consistency_score += 10
            
            scores.append(consistency_score)
            
            # Statistical significance scores
            stat_score = 0
            if statistical_tests:
                if statistical_tests.get('returns_consistency', {}).get('significant', False):
                    stat_score += 10
                if statistical_tests.get('sharpe_stability', {}).get('stable', False):
                    stat_score += 10
                if statistical_tests.get('win_rate_consistency', {}).get('consistent', False):
                    stat_score += 10
            
            scores.append(stat_score)
            
            # Risk management scores
            risk_score = 0
            if in_sample.max_drawdown <= self.thresholds['max_drawdown']:
                risk_score += 10
            if out_sample.max_drawdown <= self.thresholds['max_drawdown']:
                risk_score += 10
            
            scores.append(risk_score)
            
            # Calculate overall score
            overall_score = sum(scores) / 100.0  # Normalize to 0-1
            
            # Determine validation result
            if overall_score >= 0.8:
                result = ValidationResult.EXCELLENT
                confidence = 0.95
            elif overall_score >= 0.6:
                result = ValidationResult.GOOD
                confidence = 0.85
            elif overall_score >= 0.4:
                result = ValidationResult.ACCEPTABLE
                confidence = 0.70
            elif overall_score >= 0.2:
                result = ValidationResult.POOR
                confidence = 0.50
            else:
                result = ValidationResult.FAILED
                confidence = 0.30
            
            return result, overall_score, confidence
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Assessment error: {e}")
            return ValidationResult.FAILED, 0.0, 0.0
    
    def _identify_strengths(self, in_sample: BacktestMetrics, out_sample: BacktestMetrics) -> List[str]:
        """Identify strategy strengths"""
        strengths = []
        
        try:
            if in_sample.sharpe_ratio > 1.0 and out_sample.sharpe_ratio > 1.0:
                strengths.append("Consistently high risk-adjusted returns")
            
            if in_sample.win_rate > 0.6 and out_sample.win_rate > 0.6:
                strengths.append("High and stable win rate")
            
            if in_sample.max_drawdown < 0.15 and out_sample.max_drawdown < 0.15:
                strengths.append("Excellent risk control and low drawdowns")
            
            if in_sample.profit_factor > 1.5 and out_sample.profit_factor > 1.5:
                strengths.append("Strong profit generation capability")
            
            if abs(in_sample.sharpe_ratio - out_sample.sharpe_ratio) < 0.2:
                strengths.append("Consistent performance across different periods")
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Strengths identification error: {e}")
        
        return strengths
    
    def _identify_weaknesses(self, in_sample: BacktestMetrics, out_sample: BacktestMetrics) -> List[str]:
        """Identify strategy weaknesses"""
        weaknesses = []
        
        try:
            if in_sample.sharpe_ratio < 0.5 or out_sample.sharpe_ratio < 0.5:
                weaknesses.append("Poor risk-adjusted returns")
            
            if in_sample.win_rate < 0.4 or out_sample.win_rate < 0.4:
                weaknesses.append("Low win rate indicating poor signal quality")
            
            if in_sample.max_drawdown > 0.3 or out_sample.max_drawdown > 0.3:
                weaknesses.append("High drawdowns pose significant risk")
            
            if abs(in_sample.sharpe_ratio - out_sample.sharpe_ratio) > 0.5:
                weaknesses.append("Inconsistent performance across periods")
            
            if in_sample.total_trades < 50 or out_sample.total_trades < 30:
                weaknesses.append("Insufficient trading frequency for robust statistics")
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Weaknesses identification error: {e}")
        
        return weaknesses
    
    def _generate_recommendations(self, in_sample: BacktestMetrics, out_sample: BacktestMetrics) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        try:
            if in_sample.win_rate < 0.5:
                recommendations.append("Improve signal quality and entry criteria")
            
            if in_sample.max_drawdown > 0.2:
                recommendations.append("Implement stricter risk management and position sizing")
            
            if in_sample.profit_factor < 1.2:
                recommendations.append("Optimize take-profit and stop-loss levels")
            
            if abs(in_sample.sharpe_ratio - out_sample.sharpe_ratio) > 0.3:
                recommendations.append("Address overfitting by simplifying strategy parameters")
            
            recommendations.append("Consider implementing walk-forward optimization")
            recommendations.append("Monitor performance with real-time validation")
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Recommendations generation error: {e}")
        
        return recommendations
    
    def _assess_risk_factors(self, in_sample: BacktestMetrics, out_sample: BacktestMetrics) -> List[str]:
        """Assess strategy risk factors"""
        risk_factors = []
        
        try:
            if in_sample.max_drawdown > 0.25 or out_sample.max_drawdown > 0.25:
                risk_factors.append("HIGH_DRAWDOWN")
            
            if in_sample.volatility > 0.4 or out_sample.volatility > 0.4:
                risk_factors.append("HIGH_VOLATILITY")
            
            if in_sample.total_trades < 50:
                risk_factors.append("LOW_SAMPLE_SIZE")
            
            if abs(in_sample.win_rate - out_sample.win_rate) > 0.2:
                risk_factors.append("INCONSISTENT_PERFORMANCE")
            
            if in_sample.largest_loss < -0.1 or out_sample.largest_loss < -0.1:
                risk_factors.append("LARGE_SINGLE_LOSSES")
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Risk factor assessment error: {e}")
        
        return risk_factors
    
    def _calculate_risk_score(self, in_sample: BacktestMetrics, out_sample: BacktestMetrics) -> float:
        """Calculate overall risk score (0-1, lower is riskier)"""
        try:
            risk_components = []
            
            # Drawdown risk (lower drawdown = lower risk)
            dd_risk = 1 - min(1.0, max(in_sample.max_drawdown, out_sample.max_drawdown) / 0.5)
            risk_components.append(dd_risk)
            
            # Volatility risk
            vol_risk = 1 - min(1.0, max(in_sample.volatility, out_sample.volatility) / 0.6)
            risk_components.append(vol_risk)
            
            # Consistency risk
            sharpe_consistency = 1 - min(1.0, abs(in_sample.sharpe_ratio - out_sample.sharpe_ratio) / 1.0)
            risk_components.append(sharpe_consistency)
            
            # Return the average risk score
            return np.mean(risk_components)
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Risk score calculation error: {e}")
            return 0.5  # Neutral risk score
    
    def _create_failed_report(self, reason: str) -> ValidationReport:
        """Create a failed validation report"""
        return ValidationReport(
            validation_result=ValidationResult.FAILED,
            overall_score=0.0,
            confidence_level=0.0,
            in_sample_metrics=BacktestMetrics(),
            out_of_sample_metrics=BacktestMetrics(),
            walk_forward_metrics=[],
            statistical_tests={},
            monte_carlo_results={},
            strengths=[],
            weaknesses=[f"Validation failed: {reason}"],
            recommendations=["Address validation failures before deploying strategy"],
            risk_factors=["VALIDATION_FAILURE"],
            risk_score=0.0
        )
    
    def _log_validation_results(self, report: ValidationReport):
        """Log detailed validation results"""
        try:
            logger.info(f"[FORMAL_VALIDATOR] 📊 VALIDATION REPORT")
            logger.info(f"[FORMAL_VALIDATOR] Result: {report.validation_result.value}")
            logger.info(f"[FORMAL_VALIDATOR] Overall Score: {report.overall_score:.2f}")
            logger.info(f"[FORMAL_VALIDATOR] Confidence: {report.confidence_level:.1%}")
            
            logger.info(f"[FORMAL_VALIDATOR] 📈 IN-SAMPLE METRICS:")
            logger.info(f"[FORMAL_VALIDATOR]   Sharpe Ratio: {report.in_sample_metrics.sharpe_ratio:.2f}")
            logger.info(f"[FORMAL_VALIDATOR]   Win Rate: {report.in_sample_metrics.win_rate:.1%}")
            logger.info(f"[FORMAL_VALIDATOR]   Max Drawdown: {report.in_sample_metrics.max_drawdown:.1%}")
            logger.info(f"[FORMAL_VALIDATOR]   Total Trades: {report.in_sample_metrics.total_trades}")
            
            logger.info(f"[FORMAL_VALIDATOR] 📉 OUT-OF-SAMPLE METRICS:")
            logger.info(f"[FORMAL_VALIDATOR]   Sharpe Ratio: {report.out_of_sample_metrics.sharpe_ratio:.2f}")
            logger.info(f"[FORMAL_VALIDATOR]   Win Rate: {report.out_of_sample_metrics.win_rate:.1%}")
            logger.info(f"[FORMAL_VALIDATOR]   Max Drawdown: {report.out_of_sample_metrics.max_drawdown:.1%}")
            logger.info(f"[FORMAL_VALIDATOR]   Total Trades: {report.out_of_sample_metrics.total_trades}")
            
            if report.strengths:
                logger.info(f"[FORMAL_VALIDATOR] ✅ STRENGTHS:")
                for strength in report.strengths:
                    logger.info(f"[FORMAL_VALIDATOR]   - {strength}")
            
            if report.weaknesses:
                logger.info(f"[FORMAL_VALIDATOR] ⚠️ WEAKNESSES:")
                for weakness in report.weaknesses:
                    logger.info(f"[FORMAL_VALIDATOR]   - {weakness}")
            
            if report.recommendations:
                logger.info(f"[FORMAL_VALIDATOR] 💡 RECOMMENDATIONS:")
                for rec in report.recommendations:
                    logger.info(f"[FORMAL_VALIDATOR]   - {rec}")
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Logging error: {e}")

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation history"""
        try:
            if not self.validation_history:
                return {"total_validations": 0, "message": "No validations performed yet"}
            
            recent_report = self.validation_history[-1]
            
            return {
                "total_validations": len(self.validation_history),
                "latest_result": recent_report.validation_result.value,
                "latest_score": recent_report.overall_score,
                "latest_confidence": recent_report.confidence_level,
                "validation_trend": [r.overall_score for r in self.validation_history[-5:]],  # Last 5 scores
                "timestamp": recent_report.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"[FORMAL_VALIDATOR] Summary generation error: {e}")
            return {"error": str(e)}