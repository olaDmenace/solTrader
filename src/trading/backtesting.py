"""
backtesting.py - Comprehensive backtesting engine for trading strategies
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from decimal import Decimal

logger = logging.getLogger(__name__)

@dataclass
class BacktestParameters:
    """Parameters for backtest configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    trade_size: float
    max_positions: int
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    slippage: float = 0.001
    commission: float = 0.001

@dataclass
class BacktestResult:
    """Results from backtest execution"""
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    total_return: float
    trades: List[Dict[str, Any]]
    equity_curve: List[float]
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]

@dataclass
class BacktestPosition:
    """Position tracking for backtesting"""
    token_address: str
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    high_water_mark: float = field(init=False)
    current_price: float = field(init=False)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self):
        self.current_price = self.entry_price
        self.high_water_mark = self.entry_price

    def update(self, price: float) -> None:
        """Update position with new price"""
        self.current_price = price
        if price > self.high_water_mark:
            self.high_water_mark = price
            self._update_trailing_stop()
        self._update_pnl()

    def _update_pnl(self) -> None:
        """Update unrealized PnL"""
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.size

    def _update_trailing_stop(self) -> None:
        """Update trailing stop if active"""
        if self.trailing_stop:
            self.stop_loss = self.high_water_mark * (1 - self.trailing_stop)

    def should_close(self) -> Tuple[bool, str]:
        """Check if position should be closed"""
        if self.current_price <= self.stop_loss:
            return True, "stop_loss"
        if self.current_price >= self.take_profit:
            return True, "take_profit"
        return False, ""

class BacktestEngine:
    """Engine for backtesting trading strategies"""
    
    def __init__(self, settings: Any, market_analyzer: Any, signal_generator: Any, jupiter_client: Any):
        self.settings = settings
        self.market_analyzer = market_analyzer
        self.signal_generator = signal_generator
        self.jupiter = jupiter_client
        self.last_result: Optional[BacktestResult] = None

    async def run_backtest(self, parameters: BacktestParameters) -> BacktestResult:
        """Run backtest with specified parameters"""
        try:
            # Initialize tracking variables
            equity = [parameters.initial_capital]
            current_capital = parameters.initial_capital
            positions: Dict[str, BacktestPosition] = {}
            trades: List[Dict[str, Any]] = []
            daily_returns: List[float] = []

            current_date = parameters.start_date
            while current_date <= parameters.end_date:
                try:
                    # Update existing positions
                    await self._update_positions(positions, current_date, trades)
                    current_capital = self._calculate_portfolio_value(positions, current_capital)
                    
                    # Process new signals if capacity available
                    if len(positions) < parameters.max_positions:
                        signals = await self._get_signals(current_date)
                        for signal in signals:
                            if len(positions) >= parameters.max_positions:
                                break
                                
                            success = await self._process_signal(
                                signal=signal,
                                positions=positions,
                                capital=current_capital,
                                parameters=parameters,
                                trades=trades
                            )
                            
                            if success:
                                current_capital = self._calculate_portfolio_value(positions, current_capital)

                    # Record daily equity
                    equity.append(current_capital)
                    if len(equity) > 1:
                        daily_return = (equity[-1] - equity[-2]) / equity[-2]
                        daily_returns.append(daily_return)

                    current_date += timedelta(hours=1)

                except Exception as e:
                    logger.error(f"Error processing backtest iteration: {str(e)}")
                    continue

            # Calculate final metrics
            metrics = self._calculate_metrics(equity, daily_returns, trades)
            
            result = BacktestResult(
                total_trades=len(trades),
                win_rate=metrics['win_rate'],
                profit_factor=metrics['profit_factor'],
                max_drawdown=metrics['max_drawdown'],
                sharpe_ratio=metrics['sharpe_ratio'],
                total_return=metrics['total_return'],
                trades=trades,
                equity_curve=equity,
                metrics=metrics,
                parameters=parameters.__dict__
            )
            
            self.last_result = result
            return result

        except Exception as e:
            logger.error(f"Backtest execution failed: {str(e)}")
            raise

    async def optimize_parameters(self, 
                                start_date: datetime,
                                end_date: datetime,
                                parameter_ranges: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """Optimize strategy parameters through grid search"""
        try:
            default_ranges: Dict[str, List[float]] = {
                'trade_size': [float(x) for x in np.arange(0.1, 1.1, 0.1)],
                'stop_loss': [float(x) for x in np.arange(0.02, 0.11, 0.01)],
                'take_profit': [float(x) for x in np.arange(0.03, 0.16, 0.01)],
                'trailing_stop': [float(x) for x in np.arange(0.02, 0.11, 0.01)]
            }
            
            actual_ranges = parameter_ranges if parameter_ranges is not None else default_ranges

            best_result: Optional[BacktestResult] = None
            best_params: Dict[str, float] = {}
            best_sharpe = float('-inf')

            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(actual_ranges)

            for params in param_combinations:
                parameters = BacktestParameters(
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=float(self.settings.INITIAL_CAPITAL),
                    trade_size=params['trade_size'],
                    max_positions=int(self.settings.MAX_POSITIONS),
                    stop_loss=params['stop_loss'],
                    take_profit=params['take_profit'],
                    trailing_stop=params['trailing_stop']
                )

                result = await self.run_backtest(parameters)
                
                if result.sharpe_ratio > best_sharpe:
                    best_sharpe = result.sharpe_ratio
                    best_result = result
                    best_params = params

            return {
                "parameters": best_params,
                "sharpe_ratio": best_sharpe,
                "total_return": best_result.total_return if best_result else 0,
                "max_drawdown": best_result.max_drawdown if best_result else 0,
                "win_rate": best_result.win_rate if best_result else 0,
                "profit_factor": best_result.profit_factor if best_result else 0
            }

        except Exception as e:
            logger.error(f"Parameter optimization failed: {str(e)}")
            raise

    async def _update_positions(self,
                              positions: Dict[str, BacktestPosition],
                              current_date: datetime,
                              trades: List[Dict[str, Any]]) -> None:
        """Update open positions with new prices"""
        for token_address, position in list(positions.items()):
            try:
                price = await self._get_historical_price(token_address, current_date)
                if not price:
                    continue

                position.update(price)
                should_close, reason = position.should_close()

                if should_close:
                    trades.append({
                        'token': token_address,
                        'entry_price': position.entry_price,
                        'exit_price': position.current_price,
                        'size': position.size,
                        'entry_time': position.entry_time,
                        'exit_time': current_date,
                        'pnl': position.unrealized_pnl,
                        'reason': reason
                    })
                    del positions[token_address]

            except Exception as e:
                logger.error(f"Error updating position {token_address}: {str(e)}")

    async def _get_signals(self, current_date: datetime) -> List[Dict[str, Any]]:
        """Get trading signals for current timestamp"""
        try:
            # Get market data
            market_data = await self._get_historical_market_data(current_date)
            if not market_data:
                return []

            signals = []
            for token_data in market_data:
                signal = await self.signal_generator.analyze_token(token_data)
                if signal and signal.strength >= self.settings.SIGNAL_THRESHOLD:
                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Error getting signals: {str(e)}")
            return []

    async def _process_signal(self,
                            signal: Any,
                            positions: Dict[str, BacktestPosition],
                            capital: float,
                            parameters: BacktestParameters,
                            trades: List[Dict[str, Any]]) -> bool:
        """Process trading signal in backtest"""
        try:
            # Calculate position size
            size = min(
                parameters.trade_size * capital,
                capital * self.settings.MAX_POSITION_SIZE
            )

            if size <= 0:
                return False

            # Create new position
            position = BacktestPosition(
                token_address=signal.token_address,
                entry_price=signal.price,
                size=size,
                entry_time=signal.timestamp,
                stop_loss=signal.price * (1 - parameters.stop_loss),
                take_profit=signal.price * (1 + parameters.take_profit),
                trailing_stop=parameters.trailing_stop
            )

            positions[signal.token_address] = position
            return True

        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
            return False

    def _calculate_portfolio_value(self, positions: Dict[str, BacktestPosition], 
                                 base_capital: float) -> float:
        """Calculate current portfolio value"""
        position_values = sum(pos.unrealized_pnl for pos in positions.values())
        return base_capital + position_values

    def _calculate_metrics(self,
                         equity: List[float],
                         daily_returns: List[float],
                         trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive backtest metrics"""
        try:
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            total_trades = len(trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            gross_profits = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            gross_losses = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

            # Calculate maximum drawdown
            peak = equity[0]
            max_dd = 0
            for value in equity:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)

            # Calculate Sharpe ratio
            if len(daily_returns) > 1:
                returns_array = np.array(daily_returns)
                sharpe = np.sqrt(252) * (np.mean(returns_array) / np.std(returns_array))
            else:
                sharpe = 0

            total_return = (equity[-1] - equity[0]) / equity[0] * 100

            return {
                'win_rate': win_rate * 100,
                'profit_factor': profit_factor,
                'max_drawdown': max_dd * 100,
                'sharpe_ratio': sharpe,
                'total_return': total_return,
                'volatility': np.std(daily_returns) * np.sqrt(252) if daily_returns else 0,
                'avg_trade_duration': self._calculate_avg_trade_duration(trades),
                'avg_profit_per_trade': sum(t['pnl'] for t in trades) / len(trades) if trades else 0,
                'largest_win': max((t['pnl'] for t in trades), default=0),
                'largest_loss': min((t['pnl'] for t in trades), default=0)
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def _calculate_avg_trade_duration(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate average trade duration in hours"""
        if not trades:
            return 0
            
        durations = [
            (t['exit_time'] - t['entry_time']).total_seconds() / 3600 
            for t in trades
        ]
        return float(np.mean(durations))

    async def _get_historical_price(self, token_address: str, timestamp: datetime) -> Optional[float]:
        """Get historical price for token at timestamp"""
        try:
            price_data = await self.jupiter.get_price_history(token_address)
            if not price_data:
                return None
                
            # Find closest price point to timestamp
            closest_price = None
            min_diff = timedelta.max
            
            for data_point in price_data:
                point_time = datetime.fromtimestamp(data_point['timestamp'])
                diff = abs(point_time - timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest_price = float(data_point['price'])
                    
            return closest_price

        except Exception as e:
            logger.error(f"Error getting historical price: {str(e)}")
            return None

    async def _get_historical_market_data(self, timestamp: datetime) -> Optional[List[Dict[str, Any]]]:
        """Get historical market data for timestamp"""
        try:
            # Get list of tokens
            tokens = await self.jupiter.get_tokens_list()
            if not tokens:
                return None

            market_data = []
            for token in tokens:
                try:
                    price = await self._get_historical_price(token['address'], timestamp)
                    if not price:
                        continue

                    # Get market depth data
                    depth_data = await self.jupiter.get_market_depth(token['address'])
                    if not depth_data:
                        continue

                    market_data.append({
                        'address': token['address'],
                        'symbol': token['symbol'],
                        'price': price,
                        'volume': depth_data.get('volume24h', 0),
                        'liquidity': depth_data.get('liquidity', 0),
                        'timestamp': timestamp,
                        'price_history': [{'price': price}],
                        'volume_history': depth_data.get('recent_volumes', [])
                    })

                except Exception as e:
                    logger.error(f"Error processing token {token['address']}: {str(e)}")
                    continue

            return market_data

        except Exception as e:
            logger.error(f"Error getting historical market data: {str(e)}")
            return None

    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List[float]]) -> List[Dict[str, float]]:
        """Generate all possible parameter combinations for optimization"""
        try:
            import itertools

            # Get all parameter names and their possible values
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())

            # Generate all combinations
            combinations = []
            for values in itertools.product(*param_values):
                combination = dict(zip(param_names, values))
                combinations.append(combination)

            return combinations

        except Exception as e:
            logger.error(f"Error generating parameter combinations: {str(e)}")
            return []

    async def evaluate_strategy(self, period_days: int = 30) -> Dict[str, Any]:
        """Evaluate strategy performance over recent period"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)

            # Run backtest with current settings
            parameters = BacktestParameters(
                start_date=start_date,
                end_date=end_date,
                initial_capital=float(self.settings.INITIAL_CAPITAL),
                trade_size=float(self.settings.MAX_TRADE_SIZE),
                max_positions=int(self.settings.MAX_POSITIONS),
                stop_loss=float(self.settings.STOP_LOSS_PERCENTAGE),
                take_profit=float(self.settings.TAKE_PROFIT_PERCENTAGE),
                trailing_stop=None  # Could be added based on settings
            )

            result = await self.run_backtest(parameters)

            # Additional analysis
            risk_metrics = self._calculate_risk_metrics(result)
            optimization_potential = await self._estimate_optimization_potential(result)

            return {
                "performance_metrics": {
                    "total_return": result.total_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate
                },
                "risk_metrics": risk_metrics,
                "optimization_potential": optimization_potential,
                "trade_analysis": self._analyze_trades(result.trades)
            }

        except Exception as e:
            logger.error(f"Strategy evaluation failed: {str(e)}")
            return {}

    def _calculate_risk_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate additional risk metrics"""
        try:
            returns = np.diff(result.equity_curve) / result.equity_curve[:-1]
            
            return {
                "var_95": float(np.percentile(returns, 5)),
                "cvar_95": float(np.mean(returns[returns <= np.percentile(returns, 5)])),
                "sortino_ratio": self._calculate_sortino_ratio(returns),
                "calmar_ratio": self._calculate_calmar_ratio(result),
                "omega_ratio": self._calculate_omega_ratio(returns)
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}

    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        try:
            excess_returns = returns - (risk_free_rate / 252)
            downside_returns = excess_returns[excess_returns < 0]
            downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 1e-6
            
            return float(np.mean(excess_returns) / downside_std * np.sqrt(252))
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0

    def _calculate_calmar_ratio(self, result: BacktestResult) -> float:
        """Calculate Calmar ratio"""
        try:
            if result.max_drawdown == 0:
                return 0.0
                
            return float(result.total_return / result.max_drawdown)
            
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {str(e)}")
            return 0.0

    def _calculate_omega_ratio(self, returns: np.ndarray, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        try:
            gains = returns[returns >= threshold]
            losses = returns[returns < threshold]
            
            if len(losses) == 0 or abs(np.sum(losses)) == 0:
                return float('inf')
                
            return float(np.sum(gains) / abs(np.sum(losses)))
            
        except Exception as e:
            logger.error(f"Error calculating Omega ratio: {str(e)}")
            return 0.0

    async def _estimate_optimization_potential(self, current_result: BacktestResult) -> Dict[str, Any]:
        """Estimate potential improvement through optimization"""
        try:
            # Run quick optimization with limited parameter range
            quick_params = {
                'trade_size': np.linspace(
                    current_result.parameters['trade_size'] * 0.5,
                    current_result.parameters['trade_size'] * 1.5,
                    5
                ),
                'stop_loss': np.linspace(
                    current_result.parameters['stop_loss'] * 0.5,
                    current_result.parameters['stop_loss'] * 1.5,
                    5
                ),
                'take_profit': np.linspace(
                    current_result.parameters['take_profit'] * 0.5,
                    current_result.parameters['take_profit'] * 1.5,
                    5
                )
            }

            optimal_result = await self.optimize_parameters(
                start_date=datetime.fromisoformat(current_result.parameters['start_date']),
                end_date=datetime.fromisoformat(current_result.parameters['end_date']),
                parameter_ranges=quick_params
            )

            return {
                "potential_improvement": {
                    "sharpe_ratio": (optimal_result["sharpe_ratio"] / current_result.sharpe_ratio - 1) * 100,
                    "total_return": (optimal_result["total_return"] / current_result.total_return - 1) * 100
                },
                "suggested_parameters": optimal_result["parameters"]
            }

        except Exception as e:
            logger.error(f"Error estimating optimization potential: {str(e)}")
            return {}

    def _analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trade patterns and characteristics"""
        try:
            if not trades:
                return {}

            # Convert to DataFrame for analysis
            df = pd.DataFrame(trades)
            
            return {
                "avg_win_size": float(df[df['pnl'] > 0]['pnl'].mean()),
                "avg_loss_size": float(abs(df[df['pnl'] < 0]['pnl'].mean())),
                "win_loss_ratio": float(df[df['pnl'] > 0]['pnl'].mean() / abs(df[df['pnl'] < 0]['pnl'].mean())),
                "best_performing_tokens": df.groupby('token')['pnl'].sum().nlargest(5).to_dict(),
                "worst_performing_tokens": df.groupby('token')['pnl'].sum().nsmallest(5).to_dict(),
                "common_exit_reasons": df['reason'].value_counts().to_dict()
            }

        except Exception as e:
            logger.error(f"Error analyzing trades: {str(e)}")
            return {}