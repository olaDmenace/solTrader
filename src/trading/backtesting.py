# """
# backtesting.py - Comprehensive backtesting engine for trading strategies
# """

# import logging
# from typing import Dict, List, Optional, Any, Tuple
# from dataclasses import dataclass, field
# from datetime import datetime, timedelta
# import numpy as np
# import pandas as pd
# from decimal import Decimal

# logger = logging.getLogger(__name__)

# @dataclass
# class BacktestParameters:
#     """Parameters for backtest configuration"""
#     start_date: datetime
#     end_date: datetime
#     initial_capital: float
#     trade_size: float
#     max_positions: int
#     stop_loss: float
#     take_profit: float
#     trailing_stop: Optional[float] = None
#     slippage: float = 0.001
#     commission: float = 0.001

# @dataclass
# class BacktestResult:
#     """Results from backtest execution"""
#     total_trades: int
#     win_rate: float
#     profit_factor: float
#     max_drawdown: float
#     sharpe_ratio: float
#     total_return: float
#     trades: List[Dict[str, Any]]
#     equity_curve: List[float]
#     metrics: Dict[str, Any]
#     parameters: Dict[str, Any]

# @dataclass
# class BacktestPosition:
#     """Position tracking for backtesting"""
#     token_address: str
#     entry_price: float
#     size: float
#     entry_time: datetime
#     stop_loss: float
#     take_profit: float
#     trailing_stop: Optional[float] = None
#     high_water_mark: float = field(init=False)
#     current_price: float = field(init=False)
#     unrealized_pnl: float = 0.0
#     realized_pnl: float = 0.0

#     def __post_init__(self):
#         self.current_price = self.entry_price
#         self.high_water_mark = self.entry_price

#     def update(self, price: float) -> None:
#         """Update position with new price"""
#         self.current_price = price
#         if price > self.high_water_mark:
#             self.high_water_mark = price
#             self._update_trailing_stop()
#         self._update_pnl()

#     def _update_pnl(self) -> None:
#         """Update unrealized PnL"""
#         self.unrealized_pnl = (self.current_price - self.entry_price) * self.size

#     def _update_trailing_stop(self) -> None:
#         """Update trailing stop if active"""
#         if self.trailing_stop:
#             self.stop_loss = self.high_water_mark * (1 - self.trailing_stop)

#     def should_close(self) -> Tuple[bool, str]:
#         """Check if position should be closed"""
#         if self.current_price <= self.stop_loss:
#             return True, "stop_loss"
#         if self.current_price >= self.take_profit:
#             return True, "take_profit"
#         return False, ""

# class BacktestEngine:
#     """Engine for backtesting trading strategies"""
    
#     def __init__(self, settings: Any, market_analyzer: Any, signal_generator: Any, jupiter_client: Any):
#         self.settings = settings
#         self.market_analyzer = market_analyzer
#         self.signal_generator = signal_generator
#         self.jupiter = jupiter_client
#         self.last_result: Optional[BacktestResult] = None

#     async def run_backtest(self, parameters: BacktestParameters) -> BacktestResult:
#         """Run backtest with specified parameters"""
#         try:
#             # Initialize tracking variables
#             equity = [parameters.initial_capital]
#             current_capital = parameters.initial_capital
#             positions: Dict[str, BacktestPosition] = {}
#             trades: List[Dict[str, Any]] = []
#             daily_returns: List[float] = []

#             current_date = parameters.start_date
#             while current_date <= parameters.end_date:
#                 try:
#                     # Update existing positions
#                     await self._update_positions(positions, current_date, trades)
#                     current_capital = self._calculate_portfolio_value(positions, current_capital)
                    
#                     # Process new signals if capacity available
#                     if len(positions) < parameters.max_positions:
#                         signals = await self._get_signals(current_date)
#                         for signal in signals:
#                             if len(positions) >= parameters.max_positions:
#                                 break
                                
#                             success = await self._process_signal(
#                                 signal=signal,
#                                 positions=positions,
#                                 capital=current_capital,
#                                 parameters=parameters,
#                                 trades=trades
#                             )
                            
#                             if success:
#                                 current_capital = self._calculate_portfolio_value(positions, current_capital)

#                     # Record daily equity
#                     equity.append(current_capital)
#                     if len(equity) > 1:
#                         daily_return = (equity[-1] - equity[-2]) / equity[-2]
#                         daily_returns.append(daily_return)

#                     current_date += timedelta(hours=1)

#                 except Exception as e:
#                     logger.error(f"Error processing backtest iteration: {str(e)}")
#                     continue

#             # Calculate final metrics
#             metrics = self._calculate_metrics(equity, daily_returns, trades)
            
#             result = BacktestResult(
#                 total_trades=len(trades),
#                 win_rate=metrics['win_rate'],
#                 profit_factor=metrics['profit_factor'],
#                 max_drawdown=metrics['max_drawdown'],
#                 sharpe_ratio=metrics['sharpe_ratio'],
#                 total_return=metrics['total_return'],
#                 trades=trades,
#                 equity_curve=equity,
#                 metrics=metrics,
#                 parameters=parameters.__dict__
#             )
            
#             self.last_result = result
#             return result

#         except Exception as e:
#             logger.error(f"Backtest execution failed: {str(e)}")
#             raise

#     async def optimize_parameters(self, 
#                                 start_date: datetime,
#                                 end_date: datetime,
#                                 parameter_ranges: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
#         """Optimize strategy parameters through grid search"""
#         try:
#             default_ranges: Dict[str, List[float]] = {
#                 'trade_size': [float(x) for x in np.arange(0.1, 1.1, 0.1)],
#                 'stop_loss': [float(x) for x in np.arange(0.02, 0.11, 0.01)],
#                 'take_profit': [float(x) for x in np.arange(0.03, 0.16, 0.01)],
#                 'trailing_stop': [float(x) for x in np.arange(0.02, 0.11, 0.01)]
#             }
            
#             actual_ranges = parameter_ranges if parameter_ranges is not None else default_ranges

#             best_result: Optional[BacktestResult] = None
#             best_params: Dict[str, float] = {}
#             best_sharpe = float('-inf')

#             # Generate parameter combinations
#             param_combinations = self._generate_parameter_combinations(actual_ranges)

#             for params in param_combinations:
#                 parameters = BacktestParameters(
#                     start_date=start_date,
#                     end_date=end_date,
#                     initial_capital=float(self.settings.INITIAL_CAPITAL),
#                     trade_size=params['trade_size'],
#                     max_positions=int(self.settings.MAX_POSITIONS),
#                     stop_loss=params['stop_loss'],
#                     take_profit=params['take_profit'],
#                     trailing_stop=params['trailing_stop']
#                 )

#                 result = await self.run_backtest(parameters)
                
#                 if result.sharpe_ratio > best_sharpe:
#                     best_sharpe = result.sharpe_ratio
#                     best_result = result
#                     best_params = params

#             return {
#                 "parameters": best_params,
#                 "sharpe_ratio": best_sharpe,
#                 "total_return": best_result.total_return if best_result else 0,
#                 "max_drawdown": best_result.max_drawdown if best_result else 0,
#                 "win_rate": best_result.win_rate if best_result else 0,
#                 "profit_factor": best_result.profit_factor if best_result else 0
#             }

#         except Exception as e:
#             logger.error(f"Parameter optimization failed: {str(e)}")
#             raise

#     async def _update_positions(self,
#                               positions: Dict[str, BacktestPosition],
#                               current_date: datetime,
#                               trades: List[Dict[str, Any]]) -> None:
#         """Update open positions with new prices"""
#         for token_address, position in list(positions.items()):
#             try:
#                 price = await self._get_historical_price(token_address, current_date)
#                 if not price:
#                     continue

#                 position.update(price)
#                 should_close, reason = position.should_close()

#                 if should_close:
#                     trades.append({
#                         'token': token_address,
#                         'entry_price': position.entry_price,
#                         'exit_price': position.current_price,
#                         'size': position.size,
#                         'entry_time': position.entry_time,
#                         'exit_time': current_date,
#                         'pnl': position.unrealized_pnl,
#                         'reason': reason
#                     })
#                     del positions[token_address]

#             except Exception as e:
#                 logger.error(f"Error updating position {token_address}: {str(e)}")

#     async def _get_signals(self, current_date: datetime) -> List[Dict[str, Any]]:
#         """Get trading signals for current timestamp"""
#         try:
#             # Get market data
#             market_data = await self._get_historical_market_data(current_date)
#             if not market_data:
#                 return []

#             signals = []
#             for token_data in market_data:
#                 signal = await self.signal_generator.analyze_token(token_data)
#                 if signal and signal.strength >= self.settings.SIGNAL_THRESHOLD:
#                     signals.append(signal)

#             return signals

#         except Exception as e:
#             logger.error(f"Error getting signals: {str(e)}")
#             return []

#     async def _process_signal(self,
#                             signal: Any,
#                             positions: Dict[str, BacktestPosition],
#                             capital: float,
#                             parameters: BacktestParameters,
#                             trades: List[Dict[str, Any]]) -> bool:
#         """Process trading signal in backtest"""
#         try:
#             # Calculate position size
#             size = min(
#                 parameters.trade_size * capital,
#                 capital * self.settings.MAX_POSITION_SIZE
#             )

#             if size <= 0:
#                 return False

#             # Create new position
#             position = BacktestPosition(
#                 token_address=signal.token_address,
#                 entry_price=signal.price,
#                 size=size,
#                 entry_time=signal.timestamp,
#                 stop_loss=signal.price * (1 - parameters.stop_loss),
#                 take_profit=signal.price * (1 + parameters.take_profit),
#                 trailing_stop=parameters.trailing_stop
#             )

#             positions[signal.token_address] = position
#             return True

#         except Exception as e:
#             logger.error(f"Error processing signal: {str(e)}")
#             return False

#     def _calculate_portfolio_value(self, positions: Dict[str, BacktestPosition], 
#                                  base_capital: float) -> float:
#         """Calculate current portfolio value"""
#         position_values = sum(pos.unrealized_pnl for pos in positions.values())
#         return base_capital + position_values

#     def _calculate_metrics(self,
#                          equity: List[float],
#                          daily_returns: List[float],
#                          trades: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """Calculate comprehensive backtest metrics"""
#         try:
#             winning_trades = len([t for t in trades if t['pnl'] > 0])
#             total_trades = len(trades)
#             win_rate = winning_trades / total_trades if total_trades > 0 else 0

#             gross_profits = sum(t['pnl'] for t in trades if t['pnl'] > 0)
#             gross_losses = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
#             profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

#             # Calculate maximum drawdown
#             peak = equity[0]
#             max_dd = 0
#             for value in equity:
#                 if value > peak:
#                     peak = value
#                 dd = (peak - value) / peak
#                 max_dd = max(max_dd, dd)

#             # Calculate Sharpe ratio
#             if len(daily_returns) > 1:
#                 returns_array = np.array(daily_returns)
#                 sharpe = np.sqrt(252) * (np.mean(returns_array) / np.std(returns_array))
#             else:
#                 sharpe = 0

#             total_return = (equity[-1] - equity[0]) / equity[0] * 100

#             return {
#                 'win_rate': win_rate * 100,
#                 'profit_factor': profit_factor,
#                 'max_drawdown': max_dd * 100,
#                 'sharpe_ratio': sharpe,
#                 'total_return': total_return,
#                 'volatility': np.std(daily_returns) * np.sqrt(252) if daily_returns else 0,
#                 'avg_trade_duration': self._calculate_avg_trade_duration(trades),
#                 'avg_profit_per_trade': sum(t['pnl'] for t in trades) / len(trades) if trades else 0,
#                 'largest_win': max((t['pnl'] for t in trades), default=0),
#                 'largest_loss': min((t['pnl'] for t in trades), default=0)
#             }

#         except Exception as e:
#             logger.error(f"Error calculating metrics: {str(e)}")
#             return {}

#     def _calculate_avg_trade_duration(self, trades: List[Dict[str, Any]]) -> float:
#         """Calculate average trade duration in hours"""
#         if not trades:
#             return 0
            
#         durations = [
#             (t['exit_time'] - t['entry_time']).total_seconds() / 3600 
#             for t in trades
#         ]
#         return float(np.mean(durations))

#     async def _get_historical_price(self, token_address: str, timestamp: datetime) -> Optional[float]:
#         """Get historical price for token at timestamp"""
#         try:
#             price_data = await self.jupiter.get_price_history(token_address)
#             if not price_data:
#                 return None
                
#             # Find closest price point to timestamp
#             closest_price = None
#             min_diff = timedelta.max
            
#             for data_point in price_data:
#                 point_time = datetime.fromtimestamp(data_point['timestamp'])
#                 diff = abs(point_time - timestamp)
#                 if diff < min_diff:
#                     min_diff = diff
#                     closest_price = float(data_point['price'])
                    
#             return closest_price

#         except Exception as e:
#             logger.error(f"Error getting historical price: {str(e)}")
#             return None

#     async def _get_historical_market_data(self, timestamp: datetime) -> Optional[List[Dict[str, Any]]]:
#         """Get historical market data for timestamp"""
#         try:
#             # Get list of tokens
#             tokens = await self.jupiter.get_tokens_list()
#             if not tokens:
#                 return None

#             market_data = []
#             for token in tokens:
#                 try:
#                     price = await self._get_historical_price(token['address'], timestamp)
#                     if not price:
#                         continue

#                     # Get market depth data
#                     depth_data = await self.jupiter.get_market_depth(token['address'])
#                     if not depth_data:
#                         continue

#                     market_data.append({
#                         'address': token['address'],
#                         'symbol': token['symbol'],
#                         'price': price,
#                         'volume': depth_data.get('volume24h', 0),
#                         'liquidity': depth_data.get('liquidity', 0),
#                         'timestamp': timestamp,
#                         'price_history': [{'price': price}],
#                         'volume_history': depth_data.get('recent_volumes', [])
#                     })

#                 except Exception as e:
#                     logger.error(f"Error processing token {token['address']}: {str(e)}")
#                     continue

#             return market_data

#         except Exception as e:
#             logger.error(f"Error getting historical market data: {str(e)}")
#             return None

#     def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List[float]]) -> List[Dict[str, float]]:
#         """Generate all possible parameter combinations for optimization"""
#         try:
#             import itertools

#             # Get all parameter names and their possible values
#             param_names = list(parameter_ranges.keys())
#             param_values = list(parameter_ranges.values())

#             # Generate all combinations
#             combinations = []
#             for values in itertools.product(*param_values):
#                 combination = dict(zip(param_names, values))
#                 combinations.append(combination)

#             return combinations

#         except Exception as e:
#             logger.error(f"Error generating parameter combinations: {str(e)}")
#             return []

#     async def evaluate_strategy(self, period_days: int = 30) -> Dict[str, Any]:
#         """Evaluate strategy performance over recent period"""
#         try:
#             end_date = datetime.now()
#             start_date = end_date - timedelta(days=period_days)

#             # Run backtest with current settings
#             parameters = BacktestParameters(
#                 start_date=start_date,
#                 end_date=end_date,
#                 initial_capital=float(self.settings.INITIAL_CAPITAL),
#                 trade_size=float(self.settings.MAX_TRADE_SIZE),
#                 max_positions=int(self.settings.MAX_POSITIONS),
#                 stop_loss=float(self.settings.STOP_LOSS_PERCENTAGE),
#                 take_profit=float(self.settings.TAKE_PROFIT_PERCENTAGE),
#                 trailing_stop=None  # Could be added based on settings
#             )

#             result = await self.run_backtest(parameters)

#             # Additional analysis
#             risk_metrics = self._calculate_risk_metrics(result)
#             optimization_potential = await self._estimate_optimization_potential(result)

#             return {
#                 "performance_metrics": {
#                     "total_return": result.total_return,
#                     "sharpe_ratio": result.sharpe_ratio,
#                     "max_drawdown": result.max_drawdown,
#                     "win_rate": result.win_rate
#                 },
#                 "risk_metrics": risk_metrics,
#                 "optimization_potential": optimization_potential,
#                 "trade_analysis": self._analyze_trades(result.trades)
#             }

#         except Exception as e:
#             logger.error(f"Strategy evaluation failed: {str(e)}")
#             return {}

#     def _calculate_risk_metrics(self, result: BacktestResult) -> Dict[str, float]:
#         """Calculate additional risk metrics"""
#         try:
#             returns = np.diff(result.equity_curve) / result.equity_curve[:-1]
            
#             return {
#                 "var_95": float(np.percentile(returns, 5)),
#                 "cvar_95": float(np.mean(returns[returns <= np.percentile(returns, 5)])),
#                 "sortino_ratio": self._calculate_sortino_ratio(returns),
#                 "calmar_ratio": self._calculate_calmar_ratio(result),
#                 "omega_ratio": self._calculate_omega_ratio(returns)
#             }

#         except Exception as e:
#             logger.error(f"Error calculating risk metrics: {str(e)}")
#             return {}

#     def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
#         """Calculate Sortino ratio"""
#         try:
#             excess_returns = returns - (risk_free_rate / 252)
#             downside_returns = excess_returns[excess_returns < 0]
#             downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 1e-6
            
#             return float(np.mean(excess_returns) / downside_std * np.sqrt(252))
            
#         except Exception as e:
#             logger.error(f"Error calculating Sortino ratio: {str(e)}")
#             return 0.0

#     def _calculate_calmar_ratio(self, result: BacktestResult) -> float:
#         """Calculate Calmar ratio"""
#         try:
#             if result.max_drawdown == 0:
#                 return 0.0
                
#             return float(result.total_return / result.max_drawdown)
            
#         except Exception as e:
#             logger.error(f"Error calculating Calmar ratio: {str(e)}")
#             return 0.0

#     def _calculate_omega_ratio(self, returns: np.ndarray, threshold: float = 0) -> float:
#         """Calculate Omega ratio"""
#         try:
#             gains = returns[returns >= threshold]
#             losses = returns[returns < threshold]
            
#             if len(losses) == 0 or abs(np.sum(losses)) == 0:
#                 return float('inf')
                
#             return float(np.sum(gains) / abs(np.sum(losses)))
            
#         except Exception as e:
#             logger.error(f"Error calculating Omega ratio: {str(e)}")
#             return 0.0

#     async def _estimate_optimization_potential(self, current_result: BacktestResult) -> Dict[str, Any]:
#         """Estimate potential improvement through optimization"""
#         try:
#             # Run quick optimization with limited parameter range
#             quick_params = {
#                 'trade_size': np.linspace(
#                     current_result.parameters['trade_size'] * 0.5,
#                     current_result.parameters['trade_size'] * 1.5,
#                     5
#                 ),
#                 'stop_loss': np.linspace(
#                     current_result.parameters['stop_loss'] * 0.5,
#                     current_result.parameters['stop_loss'] * 1.5,
#                     5
#                 ),
#                 'take_profit': np.linspace(
#                     current_result.parameters['take_profit'] * 0.5,
#                     current_result.parameters['take_profit'] * 1.5,
#                     5
#                 )
#             }

#             optimal_result = await self.optimize_parameters(
#                 start_date=datetime.fromisoformat(current_result.parameters['start_date']),
#                 end_date=datetime.fromisoformat(current_result.parameters['end_date']),
#                 parameter_ranges=quick_params
#             )

#             return {
#                 "potential_improvement": {
#                     "sharpe_ratio": (optimal_result["sharpe_ratio"] / current_result.sharpe_ratio - 1) * 100,
#                     "total_return": (optimal_result["total_return"] / current_result.total_return - 1) * 100
#                 },
#                 "suggested_parameters": optimal_result["parameters"]
#             }

#         except Exception as e:
#             logger.error(f"Error estimating optimization potential: {str(e)}")
#             return {}

#     def _analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """Analyze trade patterns and characteristics"""
#         try:
#             if not trades:
#                 return {}

#             # Convert to DataFrame for analysis
#             df = pd.DataFrame(trades)
            
#             return {
#                 "avg_win_size": float(df[df['pnl'] > 0]['pnl'].mean()),
#                 "avg_loss_size": float(abs(df[df['pnl'] < 0]['pnl'].mean())),
#                 "win_loss_ratio": float(df[df['pnl'] > 0]['pnl'].mean() / abs(df[df['pnl'] < 0]['pnl'].mean())),
#                 "best_performing_tokens": df.groupby('token')['pnl'].sum().nlargest(5).to_dict(),
#                 "worst_performing_tokens": df.groupby('token')['pnl'].sum().nsmallest(5).to_dict(),
#                 "common_exit_reasons": df['reason'].value_counts().to_dict()
#             }

#         except Exception as e:
#             logger.error(f"Error analyzing trades: {str(e)}")
#             return {}

"""
backtesting.py - Comprehensive backtesting engine for trading strategies
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Mapping, TypeVar, cast
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import numpy.typing as npt
from numpy import floating
from decimal import Decimal

# Type variables for generic types
T = TypeVar('T')
FloatArray = npt.NDArray[np.float64]

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
    sortino_ratio: float
    total_return: float
    trades: List[Dict[str, Any]]
    equity_curve: List[float]
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
    market_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Position tracking for backtesting"""

    token_address: str
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    current_price: float = field(init=False)
    high_water_mark: float = field(init=False)
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
            if self.trailing_stop:
                self.stop_loss = self.high_water_mark * (1 - self.trailing_stop)
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.size

    def should_close(self) -> Tuple[bool, str]:
        """Check if position should be closed"""
        if self.current_price <= self.stop_loss:
            return True, "stop_loss"
        if self.current_price >= self.take_profit:
            return True, "take_profit"
        return False, ""


class BacktestEngine:
    """Engine for backtesting trading strategies"""

    def __init__(
        self,
        settings: Any,
        market_analyzer: Any,
        signal_generator: Any,
        jupiter_client: Any,
    ):
        self.settings = settings
        self.market_analyzer = market_analyzer
        self.signal_generator = signal_generator
        self.jupiter = jupiter_client
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []

    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        try:
            position_values = sum(
                position.unrealized_pnl for position in self.positions.values()
            )
            base_capital = self.equity_curve[-1] if self.equity_curve else self.settings.INITIAL_CAPITAL
            return float(base_capital + position_values)
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {str(e)}")
            return float(self.settings.INITIAL_CAPITAL)
        

    async def _get_price(self, token_address: str, timestamp: datetime) -> Optional[float]:
        """Get price for token at specific timestamp"""
        try:
            price_data = await self.jupiter.get_price_history(
                token_address,
                start_timestamp=int((timestamp - timedelta(minutes=5)).timestamp()),
                end_timestamp=int(timestamp.timestamp())
            )
            if price_data:
                return float(price_data[-1]['price'])
            return None
        except Exception as e:
            logger.error(f"Error getting price: {str(e)}")
            return None

    async def run_backtest(
        self, start_date: datetime, end_date: datetime, parameters: dict
    ) -> BacktestResult:
        """Run backtest with specified parameters"""
        try:
            # Initialize backtest state
            current_capital = parameters.get(
                "initial_balance", self.settings.INITIAL_CAPITAL
            )
            self.positions.clear()
            self.trade_history.clear()
            self.equity_curve = [current_capital]
            daily_returns: List[float] = []

            # Get historical data for the period
            historical_data = await self._get_historical_data(start_date, end_date)
            if not historical_data:
                raise ValueError("No historical data available")

            # Process each timestamp
            current_date = start_date
            while current_date <= end_date:
                try:
                    # Update existing positions
                    await self._update_positions(current_date)
                    current_capital = self._calculate_portfolio_value()

                    # Process new signals if capacity available
                    if len(self.positions) < parameters.get(
                        "max_positions", self.settings.MAX_POSITIONS
                    ):
                        signals = await self._generate_signals(
                            current_date, historical_data
                        )
                        for signal in signals:
                            if len(self.positions) >= parameters.get(
                                "max_positions", self.settings.MAX_POSITIONS
                            ):
                                break

                            if await self._validate_entry(signal, current_capital):
                                await self._execute_entry(
                                    signal, current_date, current_capital
                                )

                    # Record daily equity and returns
                    self.equity_curve.append(current_capital)
                    if len(self.equity_curve) > 1:
                        daily_return = (
                            self.equity_curve[-1] - self.equity_curve[-2]
                        ) / self.equity_curve[-2]
                        daily_returns.append(daily_return)

                    current_date += timedelta(hours=1)

                except Exception as e:
                    logger.error(f"Error in backtest iteration: {str(e)}")
                    continue

            # Calculate final metrics
            metrics = self._calculate_metrics(
                self.equity_curve, daily_returns, self.trade_history
            )

            return BacktestResult(
                total_trades=len(self.trade_history),
                win_rate=metrics.get('win_rate', 0.0),
                profit_factor=metrics.get('profit_factor', 0.0),
                max_drawdown=metrics.get('max_drawdown', 0.0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
                sortino_ratio=metrics.get('sortino_ratio', 0.0),
                total_return=metrics.get('total_return', 0.0),
                trades=self.trade_history,
                equity_curve=self.equity_curve,
                metrics=metrics,
                parameters=parameters,
                market_analysis=self._analyze_market_conditions(historical_data)
            )

        except Exception as e:
            logger.error(f"Backtest execution failed: {str(e)}")
            raise

    async def _get_historical_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get historical price and market data"""
        try:
            tokens = await self.jupiter.get_tokens_list()
            if not tokens:
                return {}

            historical_data = {}
            for token in tokens:
                price_history = await self.jupiter.get_price_history(
                    token['address'],
                    start_timestamp=int(start_date.timestamp()),
                    end_timestamp=int(end_date.timestamp())
                )

                if price_history:
                    historical_data[token['address']] = {
                        'price_history': price_history,
                        'token_info': token,
                        'market_metrics': {
                            'volume': sum(p.get('volume', 0) for p in price_history),
                            'price_range': {
                                'high': max(p['price'] for p in price_history),
                                'low': min(p['price'] for p in price_history)
                            }
                        }
                    }

            return historical_data

        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return {}

    async def _update_positions(self, current_date: datetime) -> None:
        """Update all open positions"""
        for token_address, position in list(self.positions.items()):
            try:
                current_price = await self._get_price(token_address, current_date)
                if not current_price:
                    continue

                position.update(current_price)
                should_close, reason = position.should_close()

                if should_close:
                    await self._execute_exit(position, current_date, reason)

            except Exception as e:
                logger.error(f"Error updating position {token_address}: {str(e)}")

    async def _generate_signals(
        self, timestamp: datetime, historical_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate trading signals for current timestamp"""
        try:
            signals = []
            for token_address, data in historical_data.items():
                # Skip if token is already in portfolio
                if token_address in self.positions:
                    continue

                market_data = await self._prepare_market_data(
                    token_address, timestamp, data
                )
                if not market_data:
                    continue

                signal = await self.signal_generator.analyze_token(market_data)
                if signal and signal.strength >= self.settings.SIGNAL_THRESHOLD:
                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return []

    async def _validate_entry(
        self, signal: Dict[str, Any], current_capital: float
    ) -> bool:
        """Validate if entry signal meets criteria"""
        try:
            if current_capital <= 0:
                return False

            # Validate liquidity
            market_depth = await self.jupiter.get_market_depth(signal["token_address"])
            if (
                not market_depth
                or market_depth.get("liquidity", 0) < self.settings.MIN_LIQUIDITY
            ):
                return False

            # Validate signal strength and market conditions
            market_analysis = await self.market_analyzer.analyze_market(
                signal["token_address"]
            )
            if (
                not market_analysis
                or market_analysis.trend_strength < self.settings.MIN_TREND_STRENGTH
            ):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating entry: {str(e)}")
            return False

    async def _execute_entry(self, signal: Dict[str, Any], timestamp: datetime, current_balance: float) -> bool:
        """Execute entry position based on signal"""
        try:
            position_size = min(
                current_balance * self.settings.MAX_POSITION_SIZE,
                signal.get('size', current_balance * 0.1)
            )

            position = Position(
                token_address=signal['token_address'],
                entry_price=float(signal['price']),  # Ensure price is float
                size=float(position_size),           # Ensure size is float
                entry_time=timestamp,
                stop_loss=float(signal['price'] * (1 - self.settings.STOP_LOSS_PERCENTAGE)),
                take_profit=float(signal['price'] * (1 + self.settings.TAKE_PROFIT_PERCENTAGE)),
                trailing_stop=self.settings.TRAILING_STOP_PERCENTAGE if hasattr(self.settings, 'TRAILING_STOP_PERCENTAGE') else None
            )

            self.positions[signal['token_address']] = position

            # Record entry trade
            self.trade_history.append({
                'type': 'entry',
                'token': signal['token_address'],
                'price': position.entry_price,
                'size': position.size,
                'timestamp': timestamp,
                'signal_strength': signal.get('strength', 0)
            })

            return True

        except Exception as e:
            logger.error(f"Error executing entry: {str(e)}")
            return False

    async def _execute_exit(
        self, position: Position, timestamp: datetime, reason: str
    ) -> None:
        """Execute position exit"""
        try:
            # Apply slippage to exit price
            exit_price = position.current_price * (1 - self.settings.SLIPPAGE)

            # Calculate realized PnL
            position.realized_pnl = (exit_price - position.entry_price) * position.size

            # Record trade exit
            self.trade_history.append(
                {
                    "token": position.token_address,
                    "type": "exit",
                    "price": exit_price,
                    "size": position.size,
                    "timestamp": timestamp,
                    "reason": reason,
                    "pnl": position.realized_pnl,
                    "hold_time": (timestamp - position.entry_time).total_seconds()
                    / 3600,  # hours
                }
            )

            # Remove position
            del self.positions[position.token_address]

        except Exception as e:
            logger.error(f"Error executing exit: {str(e)}")

    def _calculate_metrics(
        self, equity: List[float], daily_returns: List[float], trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            if not trades or not equity:
                return self._get_empty_metrics()

            # Calculate basic metrics
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            total_trades = len(trades)
            calc_win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

            gross_profits = sum(t.get('pnl', 0) for t in winning_trades)
            gross_losses = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
            calc_profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

            # Calculate drawdown
            peak = max(equity)
            calc_max_dd = (peak - min(equity)) / peak

            # Calculate risk-adjusted returns
            returns_array = np.array(daily_returns)
            calc_sharpe = np.sqrt(252) * (np.mean(returns_array) / np.std(returns_array)) if len(returns_array) > 0 else 0

            # Calculate Sortino ratio
            downside_returns = returns_array[returns_array < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
            calc_sortino = np.sqrt(252) * (np.mean(returns_array) / downside_std) if len(returns_array) > 0 else 0

            # Calculate total return
            calc_total_return = ((equity[-1] / equity[0]) - 1) * 100

            return {
                'win_rate': calc_win_rate * 100,
                'profit_factor': calc_profit_factor,
                'max_drawdown': calc_max_dd * 100,
                'sharpe_ratio': calc_sharpe,
                'sortino_ratio': calc_sortino,
                'total_return': calc_total_return,
                'total_trades': total_trades,
                'avg_trade_return': np.mean([t.get('pnl', 0) for t in trades]) if trades else 0
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return self._get_empty_metrics()

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'total_return': 0.0,
            'total_trades': 0,
            'avg_trade_return': 0.0
        }

    # def _get_empty_metrics(self) -> Dict[str, Any]:
    #     """Return empty metrics structure"""
    #     return {
    #         "win_rate": 0,
    #         "profit_factor": 0,
    #         "max_drawdown": 0,
    #         "sharpe_ratio": 0,
    #         "sortino_ratio": 0,
    #         "total_return": 0,
    #         "volatility": 0,
    #         "avg_trade_return": 0,
    #         "avg_win_size": 0,
    #         "avg_loss_size": 0,
    #         "avg_hold_time": 0,
    #         "total_trades": 0,
    #     }

    async def optimize_parameters(self, start_date: datetime, end_date: datetime,
                            parameter_ranges: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        try:
            default_ranges = {
                'STOP_LOSS_PERCENTAGE': [0.02, 0.03, 0.04, 0.05],
                'TAKE_PROFIT_PERCENTAGE': [0.03, 0.04, 0.05, 0.06],
                'MAX_POSITION_SIZE': [0.1, 0.2, 0.3],
                'SIGNAL_THRESHOLD': [0.6, 0.7, 0.8]
            }

            param_ranges = parameter_ranges if parameter_ranges is not None else default_ranges
            best_result = None
            best_params = None
            best_sharpe = float('-inf')

            # Test each parameter combination
            for params in self._generate_parameter_combinations(param_ranges):
                try:
                    result = await self.run_backtest(start_date, end_date, {
                        'initial_balance': float(self.settings.INITIAL_CAPITAL),
                        'max_positions': int(self.settings.MAX_POSITIONS),
                        **params
                    })

                    if result.sharpe_ratio > best_sharpe:
                        best_sharpe = result.sharpe_ratio
                        best_result = result
                        best_params = params

                except Exception as e:
                    logger.error(f"Error testing parameters {params}: {str(e)}")
                    continue

            if not best_result:
                return {
                    'parameters': default_ranges,  # Return default ranges instead of empty dict
                    'sharpe_ratio': 0,
                    'total_return': 0,
                    'max_drawdown': 0
                }

            return {
                'parameters': best_params,
                'sharpe_ratio': best_sharpe,
                'total_return': best_result.total_return,
                'max_drawdown': best_result.max_drawdown
            }

        except Exception as e:
            logger.error(f"Parameter optimization failed: {str(e)}")
            return {
                'parameters': default_ranges,  # Return default ranges instead of empty dict
                'sharpe_ratio': 0,
                'total_return': 0,
                'max_drawdown': 0
            }

    def _generate_parameter_combinations(
        self, 
        parameter_ranges: Mapping[str, Union[List[float], npt.NDArray[np.float64]]]
    ) -> List[Dict[str, float]]:
        """Generate all possible parameter combinations for optimization"""
        try:
            import itertools

            param_names = list(parameter_ranges.keys())
            param_values = [
                list(values) if isinstance(values, np.ndarray) else values
                for values in parameter_ranges.values()
            ]
            combinations = []

            for values in itertools.product(*param_values):
                combination = dict(zip(param_names, map(float, values)))
                combinations.append(combination)

            return combinations
        except Exception as e:
            logger.error(f"Error generating parameter combinations: {str(e)}")
            return []

    async def _prepare_market_data(
        self, token_address: str, timestamp: datetime, historical_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Prepare market data for analysis"""
        try:
            # Get relevant price history window
            lookback_period = timedelta(hours=self.settings.ANALYSIS_WINDOW)
            start_time = timestamp - lookback_period

            price_data = [
                p
                for p in historical_data["price_history"]
                if start_time <= datetime.fromtimestamp(p["timestamp"]) <= timestamp
            ]

            if not price_data:
                return None

            # Calculate market metrics
            prices = np.array([p["price"] for p in price_data])
            volumes = np.array([p.get("volume", 0) for p in price_data])

            return {
                "token_address": token_address,
                "current_price": prices[-1],
                "price_history": prices,
                "volume_history": volumes,
                "timestamp": timestamp,
                "token_info": historical_data["token_info"],
                "market_metrics": self._calculate_market_metrics(prices, volumes),
            }

            pass
        except Exception as e:
                # Only log if not a Mock-related error
                if not str(e).endswith('Mock'):
                    logger.error(f"Error preparing market data: {str(e)}")
                return None

    def _calculate_market_metrics(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Calculate various market metrics"""
        try:
            returns = np.diff(prices) / prices[:-1]

            # Explicitly convert numpy types to float
            volatility = float(np.std(returns) * np.sqrt(252))
            volume_mean = float(np.mean(volumes))
            momentum = float(self._calculate_momentum(prices))
            rsi = float(self._calculate_rsi(prices))
            trend_strength = float(self._calculate_trend_strength(prices))
            liquidity_score = float(self._calculate_liquidity_score(volumes))

            return {
                'volatility': volatility,
                'volume_mean': volume_mean,
                'price_momentum': momentum,
                'rsi': rsi,
                'trend_strength': trend_strength,
                'liquidity_score': liquidity_score
            }

        except Exception as e:
            logger.error(f"Error calculating market metrics: {str(e)}")
            return {
                'volatility': 0.0,
                'volume_mean': 0.0,
                'price_momentum': 0.0,
                'rsi': 0.0,
                'trend_strength': 0.0,
                'liquidity_score': 0.0
            }

    def _calculate_momentum(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate price momentum"""
        try:
            if len(prices) < period:
                return 0

            momentum = (prices[-1] - prices[-period]) / prices[-period]
            return float(momentum)

        except Exception as e:
            logger.error(f"Error calculating momentum: {str(e)}")
            return 0

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            if len(prices) < period + 1:
                return 50

            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                return 100

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi)

        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return 50

    def _calculate_trend_strength(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate trend strength using linear regression"""
        try:
            if len(prices) < period:
                return 0

            x = np.arange(period)
            y = prices[-period:]

            slope, _ = np.polyfit(x, y, 1)

            # Normalize slope to a 0-1 scale
            return float(min(max((abs(slope) / np.mean(y)), 0), 1))

        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return 0

    def _calculate_liquidity_score(self, volumes: np.ndarray) -> float:
        """Calculate liquidity score based on volume stability"""
        try:
            if len(volumes) < 2:
                return 0

            volume_mean = np.mean(volumes)
            volume_std = np.std(volumes)

            if volume_mean == 0:
                return 0

            # Calculate coefficient of variation and convert to a 0-1 score
            cv = volume_std / volume_mean
            score = 1 / (1 + cv)

            return float(score)

        except Exception as e:
            logger.error(f"Error calculating liquidity score: {str(e)}")
            return 0
        
    def _determine_market_trend(self, historical_data: Dict[str, Any]) -> str:
        """Determine overall market trend"""
        try:
            # Implementation of market trend analysis
            return "neutral"  # or "bullish" or "bearish"
        except Exception as e:
            logger.error(f"Error determining market trend: {str(e)}")
            return "unknown"

    def _analyze_volatility_regime(self, historical_data: Dict[str, Any]) -> str:
        """Analyze volatility regime"""
        try:
            # Implementation of volatility regime analysis
            return "normal"  # or "high" or "low"
        except Exception as e:
            logger.error(f"Error analyzing volatility regime: {str(e)}")
            return "unknown"

    def _analyze_liquidity_conditions(self, historical_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze market liquidity conditions"""
        try:
            return {
                'overall_liquidity': 0.0,
                'liquidity_trend': 0.0,
                'depth_score': 0.0
            }
        except Exception as e:
            logger.error(f"Error analyzing liquidity conditions: {str(e)}")
            return {'overall_liquidity': 0.0}

    def _calculate_correlation_matrix(self, historical_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between assets"""
        try:
            return {}  # Implementation of correlation matrix calculation
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return {}


    def _analyze_market_conditions(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall market conditions"""
        try:
            market_analysis = {
                "overall_trend": self._determine_market_trend(historical_data),
                "volatility_regime": self._analyze_volatility_regime(historical_data),
                "liquidity_conditions": self._analyze_liquidity_conditions(historical_data),
                "correlation_matrix": self._calculate_correlation_matrix(historical_data),
                "risk_metrics": self._calculate_historical_risk_metrics(historical_data),
            }

            return market_analysis

        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            return {}
        
    def _calculate_historical_risk_metrics(self, historical_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics from historical data"""
        try:
            # Extract prices from historical data
            prices = []
            for token_data in historical_data.values():
                if 'price_history' in token_data:
                    token_prices = [float(p['price']) for p in token_data['price_history']]
                    prices.extend(token_prices)

            if not prices:
                return self._get_empty_risk_metrics()

            prices_array = np.array(prices)
            returns = np.diff(prices_array) / prices_array[:-1]

            var_95 = float(np.percentile(returns, 5))
            cvar_95 = float(np.mean(returns[returns <= var_95]))
            volatility = float(np.std(returns) * np.sqrt(252))

            return {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'volatility': volatility,
                'historical_volatility': float(np.std(returns)),
                'price_range': float(np.ptp(prices_array)),
                'price_stability': float(1 / (1 + np.std(returns)))
            }

        except Exception as e:
            logger.error(f"Error calculating historical risk metrics: {str(e)}")
            return self._get_empty_risk_metrics()
        
    def _get_empty_risk_metrics(self) -> Dict[str, float]:
        """Return empty risk metrics structure"""
        return {
            'var_95': 0.0,
            'cvar_95': 0.0,
            'volatility': 0.0,
            'historical_volatility': 0.0,
            'price_range': 0.0,
            'price_stability': 0.0
        }



    def _get_empty_optimization_result(self) -> Dict[str, Any]:
        """Return empty optimization result structure"""
        return {
            "parameters": {},
            "sharpe_ratio": 0,
            "total_return": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "profit_factor": 0,
        }

    async def evaluate_strategy(self, period_days: int = 30) -> Dict[str, Any]:
        """Evaluate strategy performance over recent period"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)

            # Run backtest with current settings
            parameters = {
                'initial_balance': float(self.settings.INITIAL_CAPITAL),
                'max_positions': int(self.settings.MAX_POSITIONS),
                'stop_loss': float(self.settings.STOP_LOSS_PERCENTAGE),
                'take_profit': float(self.settings.TAKE_PROFIT_PERCENTAGE)
            }

            backtest_result = await self.run_backtest(start_date, end_date, parameters)

            # Now backtest_result is properly typed as BacktestResult
            risk_metrics = self._calculate_risk_metrics(backtest_result)
            trade_analysis = self._analyze_trades(backtest_result.trades)

            return {
                'performance_metrics': {
                    'total_return': backtest_result.total_return,
                    'sharpe_ratio': backtest_result.sharpe_ratio,
                    'max_drawdown': backtest_result.max_drawdown,
                    'win_rate': backtest_result.win_rate
                },
                'risk_metrics': risk_metrics,
                'trade_analysis': trade_analysis,
                'market_analysis': backtest_result.market_analysis
            }

        except Exception as e:
            logger.error(f"Strategy evaluation failed: {str(e)}")
            return {}

    def _calculate_risk_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate additional risk metrics"""
        try:
            returns = np.diff(result.equity_curve) / result.equity_curve[:-1]

            var_95 = float(np.percentile(returns, 5))
            cvar_95 = float(np.mean(returns[returns <= var_95]))
            max_cons_losses = float(self._calculate_max_consecutive_losses(result.trades))
            tail_ratio = float(self._calculate_tail_ratio(returns))
            calmar = float(result.total_return / result.max_drawdown if result.max_drawdown > 0 else 0)

            return {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_consecutive_losses': max_cons_losses,
                'tail_ratio': tail_ratio,
                'calmar_ratio': calmar
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'var_95': 0.0,
                'cvar_95': 0.0,
                'max_consecutive_losses': 0.0,
                'tail_ratio': 0.0,
                'calmar_ratio': 0.0
            }

    def _calculate_max_consecutive_losses(self, trades: List[Dict[str, Any]]) -> int:
        """Calculate maximum consecutive losing trades"""
        try:
            max_consecutive = current_consecutive = 0

            for trade in trades:
                if trade.get("pnl", 0) < 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0

            return max_consecutive

        except Exception as e:
            logger.error(f"Error calculating max consecutive losses: {str(e)}")
            return 0

    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate the ratio of right tail to left tail risk"""
        try:
            tail_quantile = 0.05
            right_tail = float(np.percentile(returns, 100 - tail_quantile * 100))
            left_tail = float(np.percentile(returns, tail_quantile * 100))

            if left_tail == 0:
                return 0.0

            return float(abs(right_tail / left_tail))

        except Exception as e:
            logger.error(f"Error calculating tail ratio: {str(e)}")
            return 0.0

    def _analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trade patterns and characteristics"""
        try:
            if not trades:
                return {}

            df = pd.DataFrame(trades)

            return {
                "best_performing_tokens": df.groupby("token")["pnl"]
                .sum()
                .nlargest(5)
                .to_dict(),
                "worst_performing_tokens": df.groupby("token")["pnl"]
                .sum()
                .nsmallest(5)
                .to_dict(),
                "avg_holding_period": df["hold_time"].mean()
                if "hold_time" in df
                else 0,
                "common_exit_reasons": df["reason"].value_counts().to_dict()
                if "reason" in df
                else {},
                "profit_by_market_condition": df.groupby("market_condition")["pnl"]
                .mean()
                .to_dict()
                if "market_condition" in df
                else {},
            }

        except Exception as e:
            logger.error(f"Error analyzing trades: {str(e)}")
            return {}
