"""
strategy.py - Core trading strategy implementation supporting both paper and live trading
"""
import logging
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Optional, List, Any, Protocol, Union, runtime_checkable, Tuple
import numpy as np
from src.trading.market_regime import MarketRegimeDetector, MarketRegimeType, MarketState
from .backtesting import BacktestEngine
from .exceptions import (
    TradingStrategyError,
    InsufficientBalanceError,
    ValidationError,
    MarketConditionError,
    ExecutionError,
)
from .monitoring import MonitoringSystem
from .alert import AlertSystem
from ..config.settings import Settings
from .swap import SwapExecutor
from .position import Position, TradeEntry, PositionManager
from .risk import RiskManager, MarketCondition
from .performance import PerformanceMonitor
from .signals import Signal, SignalGenerator
from .market_analyzer import MarketAnalyzer

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode enumeration"""

    PAPER = "paper"
    LIVE = "live"


@dataclass
class DailyStats:
    """Daily trading statistics tracking"""

    trade_count: int = 0
    error_count: int = 0
    total_pnl: float = 0.0
    last_reset: date = field(default_factory=date.today)


@dataclass
class PriceProtection:
    """Configuration for price-related protections"""

    max_price_impact: float
    min_liquidity: float
    max_slippage: float
    volume_threshold: float


@dataclass
class CircuitBreakers:
    """Configuration for trading circuit breakers"""

    max_daily_loss: float
    max_drawdown: float
    max_position_size: float
    max_trades_per_day: int
    error_threshold: int


@dataclass
class EntrySignal:
    """Represents a trading entry signal"""

    token_address: str
    price: float
    confidence: float
    entry_type: str
    size: float
    stop_loss: float
    take_profit: float
    slippage: float = 0.01
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_signal(
        cls, signal: Signal, size: float, stop_loss: float, take_profit: float
    ) -> "EntrySignal":
        """Create an EntrySignal from a Signal object"""
        return cls(
            token_address=signal.token_address,
            price=signal.price,
            confidence=signal.strength,
            entry_type=signal.signal_type,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )


@dataclass
class TradingState:
    """Unified trading state tracking for both paper and live trading"""

    mode: TradingMode
    is_trading: bool = False
    monitor_task: Optional[asyncio.Task] = None
    last_error: Optional[str] = None
    paper_balance: float = 0.0
    paper_positions: Dict[str, Position] = field(default_factory=dict)
    pending_orders: List[EntrySignal] = field(default_factory=list)
    daily_stats: DailyStats = field(default_factory=DailyStats)


@runtime_checkable
class TradingStrategyProtocol(Protocol):
    @property
    def is_trading(self) -> bool: ...
    backtest_engine: BacktestEngine  # Add this
    async def start_trading(self) -> bool: ...
    async def stop_trading(self) -> bool: ...
    async def get_metrics(self) -> Dict[str, Dict[str, Any]]: ...

class TradingStrategy(TradingStrategyProtocol):
    backtest_engine: BacktestEngine  
    """Unified trading strategy supporting both paper and live trading"""

    @property
    def is_trading(self) -> bool:
        """Property to expose trading state"""
        return self.state.is_trading
    
    @property
    def jupiter_client(self):
        """Expose Jupiter client"""
        return self.jupiter

    def __init__(
        self,
        jupiter_client: Any,
        wallet: Any,
        settings: Settings,
        scanner: Optional[Any] = None,
        mode: TradingMode = TradingMode.PAPER,
    ) -> None:
        """Initialize trading strategy with required components"""
        self.jupiter = jupiter_client
        self.wallet = wallet
        self.settings = settings
        self.scanner = scanner
        self.market_regime_detector = MarketRegimeDetector(settings)
        self.current_regime: Optional[MarketState] = None

        self.state = TradingState(
            mode=mode,
            paper_balance=float(settings.INITIAL_PAPER_BALANCE)
            if mode == TradingMode.PAPER
            else 0.0,
        )

        self.market_analyzer = MarketAnalyzer(jupiter_client, settings)
        self.signal_generator = SignalGenerator(settings)

        self.backtest_engine = BacktestEngine(
            settings=settings,
            market_analyzer=self.market_analyzer,
            signal_generator=self.signal_generator,
            jupiter_client=jupiter_client
        )

        self.alert_system = AlertSystem(settings)
        self.monitoring = MonitoringSystem(self.alert_system, settings)
        self.swap_executor = SwapExecutor(jupiter_client, wallet)
        self.position_manager = PositionManager(self.swap_executor, settings)
        self.risk_manager = RiskManager(settings)
        self.performance_monitor = PerformanceMonitor(settings)

        self.price_protection = PriceProtection(
            max_price_impact=settings.MAX_PRICE_IMPACT,
            min_liquidity=settings.MIN_LIQUIDITY,
            max_slippage=settings.MAX_SLIPPAGE,
            volume_threshold=settings.VOLUME_THRESHOLD,
        )

        self.circuit_breakers = CircuitBreakers(
            max_daily_loss=settings.MAX_DAILY_LOSS,
            max_drawdown=settings.MAX_DRAWDOWN,
            max_position_size=settings.MAX_POSITION_SIZE,
            max_trades_per_day=settings.MAX_TRADES_PER_DAY,
            error_threshold=settings.ERROR_THRESHOLD,
        )

    async def start_trading(self) -> bool:
        try:
            if self.state.is_trading:
                return True

            if self.state.mode == TradingMode.LIVE:
                balance = await self.wallet.get_balance()
                if isinstance(balance, (int, float, Decimal)):
                    balance = float(balance)
                    if balance < self.settings.MIN_BALANCE:
                        raise InsufficientBalanceError(
                            f"Insufficient balance: {balance} (minimum: {self.settings.MIN_BALANCE})"
                        )
                else:
                    raise ValidationError(f"Invalid balance type: {type(balance)}")

            # Initialize monitoring systems first
            await self.monitoring.start_monitoring()
            
            # Reset state
            self.state.last_error = None
            self.state.daily_stats = DailyStats()
            
            # Start trading
            self.state.is_trading = True 
            self.state.monitor_task = asyncio.create_task(self._trading_loop())

            # Start market condition monitoring
            await self.update_market_conditions()

            # Emit alert
            await self.alert_system.emit_alert(
                level="info",
                type=f"{self.state.mode.value}_trading_start",
                message=f"{self.state.mode.value.capitalize()} trading started",
                data={
                    "initial_balance": (
                        self.state.paper_balance 
                        if self.state.mode == TradingMode.PAPER
                        else balance
                    )
                },
            )
            return True

        except Exception as e:
            self.state.last_error = str(e)
            logger.error(f"Error starting {self.state.mode.value} trading: {e}")
            return False

    async def stop_trading(self) -> bool:
        """Stop trading and cleanup resources"""
        try:
            if not self.state.is_trading:
                return True

            self.state.is_trading = False

            if self.state.monitor_task and not self.state.monitor_task.done():
                self.state.monitor_task.cancel()
                try:
                    await self.state.monitor_task
                except asyncio.CancelledError:
                    pass
                finally:
                    self.state.monitor_task = None

            if self.settings.CLOSE_POSITIONS_ON_STOP:
                await self._cleanup_positions()

            metrics = await self.get_metrics()
            await self.alert_system.emit_alert(
                level="info",
                type=f"{self.state.mode.value}_trading_stop",
                message=f"{self.state.mode.value.capitalize()} trading stopped",
                data={"metrics": metrics},
            )
            return True

        except Exception as e:
            self.state.last_error = str(e)
            logger.error(f"Error stopping trading: {e}")
            return False
        

    async def _get_price_history(self) -> Dict[str, List[float]]:
        """Get historical price data for active positions"""
        history = {}
        for token in self.position_manager.positions:
            prices = await self.jupiter.get_price_history(token)
            if prices:
                history[token] = [float(p['price']) for p in prices]
        return history

    async def _get_volume_data(self) -> List[float]:
        """Get volume data for market analysis"""
        market_data = await self.jupiter.get_market_depth("So11111111111111111111111111111111111111112")
        return [float(vol) for vol in market_data.get('recent_volumes', [])] if market_data else []

    def _classify_market_regime(self, analysis: Any) -> str:
        if analysis.volatility > self.settings.MAX_VOLATILITY:
            return "volatile"
        if analysis.trend_strength > 0.7:
            return "trending_up"
        if analysis.trend_strength < -0.7:
            return "trending_down"
        return "ranging"
    

    async def _adapt_to_market_regime(self, prices: List[float], volumes: List[float]) -> None:
        """Adapt trading parameters based on market regime"""
        try:
            self.current_regime = await self.market_regime_detector.detect_regime(prices, volumes)
            
            if not self.current_regime:
                return

            # Adjust trading parameters based on regime
            if self.current_regime.regime == MarketRegimeType.TRENDING_UP:
                self._apply_trending_up_parameters()
            elif self.current_regime.regime == MarketRegimeType.TRENDING_DOWN:
                self._apply_trending_down_parameters()
            elif self.current_regime.regime == MarketRegimeType.VOLATILE:
                self._apply_volatile_parameters()
            elif self.current_regime.regime == MarketRegimeType.RANGING:
                self._apply_ranging_parameters()
            elif self.current_regime.regime == MarketRegimeType.ACCUMULATION:
                self._apply_accumulation_parameters()
            elif self.current_regime.regime == MarketRegimeType.DISTRIBUTION:
                self._apply_distribution_parameters()

            await self._notify_regime_change()

        except Exception as e:
            logger.error(f"Error adapting to market regime: {str(e)}")

    def _apply_trending_up_parameters(self) -> None:
        """Apply parameters optimized for uptrend"""
        self.settings.STOP_LOSS_PERCENTAGE = 0.05  # Wider stops in trend
        self.settings.TAKE_PROFIT_PERCENTAGE = 0.15  # Higher targets
        self.settings.MAX_POSITION_SIZE = self.settings.MAX_POSITION_SIZE * 1.2  # Larger positions
        self.settings.SLIPPAGE_TOLERANCE = 0.015  # More slippage tolerance

    def _apply_trending_down_parameters(self) -> None:
        """Apply parameters optimized for downtrend"""
        self.settings.STOP_LOSS_PERCENTAGE = 0.03  # Tighter stops
        self.settings.TAKE_PROFIT_PERCENTAGE = 0.10  # Conservative targets
        self.settings.MAX_POSITION_SIZE = self.settings.MAX_POSITION_SIZE * 0.8  # Smaller positions
        self.settings.SLIPPAGE_TOLERANCE = 0.01  # Less slippage tolerance

    def _apply_volatile_parameters(self) -> None:
        """Apply parameters optimized for volatile markets"""
        self.settings.STOP_LOSS_PERCENTAGE = 0.02  # Very tight stops
        self.settings.TAKE_PROFIT_PERCENTAGE = 0.05  # Quick profits
        self.settings.MAX_POSITION_SIZE = self.settings.MAX_POSITION_SIZE * 0.5  # Much smaller positions
        self.settings.SLIPPAGE_TOLERANCE = 0.02  # Higher slippage tolerance

    def _apply_ranging_parameters(self) -> None:
        """Apply parameters optimized for ranging markets"""
        self.settings.STOP_LOSS_PERCENTAGE = 0.04  # Moderate stops
        self.settings.TAKE_PROFIT_PERCENTAGE = 0.08  # Moderate targets
        self.settings.MAX_POSITION_SIZE = self.settings.MAX_POSITION_SIZE  # Normal position size
        self.settings.SLIPPAGE_TOLERANCE = 0.01  # Normal slippage tolerance

    def _apply_accumulation_parameters(self) -> None:
        """Apply parameters optimized for accumulation phase"""
        self.settings.STOP_LOSS_PERCENTAGE = 0.03  # Moderate stops
        self.settings.TAKE_PROFIT_PERCENTAGE = 0.12  # Higher targets
        self.settings.MAX_POSITION_SIZE = self.settings.MAX_POSITION_SIZE * 1.1  # Slightly larger positions
        self.settings.SLIPPAGE_TOLERANCE = 0.012  # Moderate slippage tolerance

    def _apply_distribution_parameters(self) -> None:
        """Apply parameters optimized for distribution phase"""
        self.settings.STOP_LOSS_PERCENTAGE = 0.02  # Tight stops
        self.settings.TAKE_PROFIT_PERCENTAGE = 0.06  # Lower targets
        self.settings.MAX_POSITION_SIZE = self.settings.MAX_POSITION_SIZE * 0.7  # Smaller positions
        self.settings.SLIPPAGE_TOLERANCE = 0.008  # Lower slippage tolerance

    async def _notify_regime_change(self) -> None:
        """Notify about market regime changes"""
        if hasattr(self, 'alert_system') and self.current_regime:
            await self.alert_system.emit_alert(
                level="info",
                type="regime_change",
                message=f"Market regime changed to {self.current_regime.regime.value}",
                data={
                    "regime": self.current_regime.regime.value,
                    "confidence": self.current_regime.confidence,
                    "volatility": self.current_regime.volatility,
                    "trend_strength": self.current_regime.trend_strength
                }
            )

    # Update in the trading loop
    async def _trading_loop(self) -> None:
        """Main trading loop handling both paper and live trading"""
        while self.state.is_trading:
            try:
                # ... existing checks ...

                # Get recent price and volume data
                prices, volumes = await self._get_recent_market_data()
                
                # Adapt parameters to market regime
                await self._adapt_to_market_regime(prices, volumes)

                # ... rest of the trading loop ...

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                self.state.daily_stats.error_count += 1
                await asyncio.sleep(5)

    async def _get_recent_market_data(self) -> Tuple[List[float], List[float]]:
        """Get recent price and volume data for regime detection"""
        try:
            # Get data for SOL as reference
            sol_address = "So11111111111111111111111111111111111111112"
            
            # Get price history
            price_data = await self.jupiter.get_price_history(sol_address)
            prices = [float(p['price']) for p in price_data] if price_data else []
            
            # Get volume data
            market_data = await self.jupiter.get_market_depth(sol_address)
            volumes = market_data.get('recent_volumes', []) if market_data else []
            
            return prices, [float(v) for v in volumes]
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return [], []

    async def update_market_conditions(self) -> None:
        try:
            price_data = await self._get_price_history()
            volume_data = await self._get_volume_data()
            
            # Using SOL as reference token
            market_analysis = await self.market_analyzer.analyze_market(
                token_address="So11111111111111111111111111111111111111112",
                price_data=list(price_data.values())[0] if price_data else None,  # Convert to list
                volume_data=volume_data
            )
            
            if market_analysis:
                self.risk_manager.market_conditions = MarketCondition(
                    volatility=market_analysis.volatility,
                    trend_strength=market_analysis.trend_strength,
                    liquidity_score=market_analysis.liquidity_score,
                    market_regime=self._classify_market_regime(market_analysis)
                )
                
                await self.alert_system.emit_alert(
                    level="info",
                    type="market_update",
                    message=f"Market regime: {self.risk_manager.market_conditions.market_regime}",
                    data=market_analysis.__dict__
                )
        except Exception as e:
            logger.error(f"Market condition update error: {str(e)}")

    # async def _trading_loop(self) -> None:
    #     """Main trading loop handling both paper and live trading"""
    #     while self.state.is_trading:
    #         try:
    #             if date.today() > self.state.daily_stats.last_reset:
    #                 self.state.daily_stats = DailyStats()

    #             if not await self._check_circuit_breakers():
    #                 logger.warning("Circuit breakers triggered - pausing trading")
    #                 await self.alert_system.emit_alert(
    #                     level="warning",
    #                     type="circuit_breakers",
    #                     message="Circuit breakers triggered - trading paused",
    #                     data={"timestamp": datetime.now().isoformat()},
    #                 )
    #                 await asyncio.sleep(300)
    #                 continue

    #             if self.state.mode == TradingMode.PAPER:
    #                 await self._update_paper_positions()
    #             else:
    #                 await self._monitor_live_positions()

    #             await self._process_pending_orders()

    #             current_positions = (
    #                 len(self.state.paper_positions)
    #                 if self.state.mode == TradingMode.PAPER
    #                 else len(self.position_manager.positions)
    #             )

    #             if current_positions < self.settings.MAX_POSITIONS:
    #                 await self._scan_opportunities()

    #             await self._update_metrics()
    #             await asyncio.sleep(self.settings.SCAN_INTERVAL)

    #         except asyncio.CancelledError:
    #             raise
    #         except Exception as e:
    #             logger.error(f"Trading loop error: {e}")
    #             self.state.daily_stats.error_count += 1
    #             await asyncio.sleep(5)

    async def _check_circuit_breakers(self) -> bool:
        """Check if any circuit breakers have been triggered"""
        try:
            daily_pnl_breach = (
                self.state.daily_stats.total_pnl
                <= -self.circuit_breakers.max_daily_loss
            )
            trade_count_breach = (
                self.state.daily_stats.trade_count
                >= self.circuit_breakers.max_trades_per_day
            )
            error_count_breach = (
                self.state.daily_stats.error_count
                >= self.circuit_breakers.error_threshold
            )

            if daily_pnl_breach:
                logger.warning(
                    f"Daily loss limit breached: {self.state.daily_stats.total_pnl}"
                )
            if trade_count_breach:
                logger.warning(
                    f"Max daily trades reached: {self.state.daily_stats.trade_count}"
                )
            if error_count_breach:
                logger.warning(
                    f"Error threshold reached: {self.state.daily_stats.error_count}"
                )

            return not (daily_pnl_breach or trade_count_breach or error_count_breach)

        except Exception as e:
            logger.error(f"Error checking circuit breakers: {e}")
            return False

    async def _validate_price_conditions(self, token_address: str, size: float) -> bool:
        """Validate price-related conditions for trading"""
        try:
            market_depth = await self.jupiter.get_market_depth(token_address)
            if not market_depth:
                return False

            liquidity = float(market_depth.get("liquidity", 0))
            volume = float(market_depth.get("volume24h", 0))
            price_impact = await self._calculate_price_impact(market_depth, size)

            conditions = {
                "liquidity": liquidity >= self.price_protection.min_liquidity,
                "volume": volume >= self.price_protection.volume_threshold,
                "price_impact": price_impact <= self.price_protection.max_price_impact,
            }

            if not all(conditions.values()):
                logger.debug(
                    f"Price conditions not met for {token_address}: "
                    f"Liquidity={liquidity}, Volume={volume}, Impact={price_impact}%"
                )

            return all(conditions.values())

        except Exception as e:
            logger.error(f"Error validating price conditions: {e}")
            return False

    async def _scan_opportunities(self) -> None:
        """Scan for new trading opportunities"""
        try:
            if not self.scanner:
                return

            new_tokens = await self.scanner.scan_new_listings()
            for token in new_tokens:
                try:
                    if self._is_token_tracked(token.address):
                        continue

                    if not self._validate_token_basics(token):
                        continue

                    signal = await self.signal_generator.analyze_token(token)
                    if not signal or signal.strength < self.settings.SIGNAL_THRESHOLD:
                        continue

                    market_volatility = (
                        await self.risk_manager.calculate_market_volatility()
                    )
                    current_balance = (
                        self.state.paper_balance
                        if self.state.mode == TradingMode.PAPER
                        else await self.wallet.get_balance()
                    )

                    token_data = await self._prepare_token_data(token, signal)
                    risk_metrics = await self._calculate_risk_metrics(
                        token_data=token_data,
                        market_volatility=market_volatility,
                        current_balance=float(current_balance),
                        signal=signal,
                    )

                    if not risk_metrics:
                        continue

                    size = self._calculate_position_size(
                        risk_metrics=risk_metrics,
                        current_balance=float(current_balance),
                    )

                    if size <= 0:
                        logger.debug(
                            f"Invalid position size calculated for {token.address}"
                        )
                        continue

                    entry_signal = self._create_entry_signal(signal, size)
                    if not entry_signal:
                        continue

                    self.state.pending_orders.append(entry_signal)
                    await self._emit_opportunity_alert(
                        token=token, signal=signal, risk_metrics=risk_metrics, size=size
                    )

                except Exception as e:
                    logger.error(f"Error analyzing token {token.address}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error scanning opportunities: {e}")

    def _is_token_tracked(self, token_address: str) -> bool:
            """Check if token is already being tracked"""
            return (
                token_address in self.state.paper_positions
                or token_address in self.position_manager.positions
                or any(
                    order.token_address == token_address
                    for order in self.state.pending_orders
                )
            )

    def _validate_token_basics(self, token: Any) -> bool:
        """Validate basic token properties"""
        try:
            if not all(
                hasattr(token, attr) for attr in ["address", "volume24h", "liquidity"]
            ):
                logger.debug(
                    f"Token missing required attributes: {token.address if hasattr(token, 'address') else 'Unknown'}"
                )
                return False

            volume_24h = float(getattr(token, "volume24h", 0))
            liquidity = float(getattr(token, "liquidity", 0))

            validations = {
                "volume": volume_24h >= self.settings.MIN_VOLUME_24H,
                "liquidity": liquidity >= self.settings.MIN_LIQUIDITY,
                "volume_threshold": volume_24h >= self.settings.VOLUME_THRESHOLD,
            }

            failed_checks = [k for k, v in validations.items() if not v]
            if failed_checks:
                logger.debug(
                    f"Token {token.address} failed validations: {', '.join(failed_checks)}"
                )

            return all(validations.values())

        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return False

    async def _prepare_token_data(self, token: Any, signal: Signal) -> Dict[str, Any]:
        """Prepare comprehensive token data for analysis"""
        return {
            "address": token.address,
            "volume24h": float(getattr(token, "volume24h", 0)),
            "liquidity": float(getattr(token, "liquidity", 0)),
            "created_at": getattr(token, "created_at", None),
            "price": signal.price,
            "market_cap": float(getattr(token, "market_cap", 0)),
            "signal_strength": signal.strength,
            "signal_type": signal.signal_type,
        }

    async def _calculate_risk_metrics(
        self,
        token_data: Dict[str, Any],
        market_volatility: float,
        current_balance: float,
        signal: Signal,
    ) -> Optional[Dict[str, float]]:
        """Calculate comprehensive risk metrics"""
        try:
            risk_score = self.risk_manager.calculate_risk_score(token_data=token_data)

            # Create market conditions object
            market_conditions = MarketCondition(
                volatility=market_volatility,
                trend_strength=float(token_data.get("trend_strength", 0.5)),
                liquidity_score=min(
                    float(token_data["liquidity"]) / self.settings.MIN_LIQUIDITY, 1.0
                ),
                market_regime="trending",  # Could be derived from trend analysis
            )

            position_value = current_balance * self.settings.MAX_POSITION_SIZE
            position_risk = self.risk_manager.calculate_position_risk(
                position_size=position_value,
                entry_price=signal.price,
                market_conditions=market_conditions,
            )

            return {
                "risk_score": risk_score,
                "position_risk": position_risk,
                "market_volatility": market_volatility,
                "position_value": position_value,
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return None

    def _calculate_position_size(
        self, risk_metrics: Dict[str, float], current_balance: float
    ) -> float:
        """Calculate final position size based on risk metrics"""
        try:
            size = risk_metrics["position_value"] * (
                1 - (risk_metrics["position_risk"] / 100)
            )

            size = min(
                size,
                current_balance * self.settings.MAX_POSITION_SIZE,
                self.circuit_breakers.max_position_size,
            )

            return max(size, 0)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _create_entry_signal(
        self, signal: Signal, size: float
    ) -> Optional[EntrySignal]:
        """Create entry signal with validation"""
        try:
            return EntrySignal.from_signal(
                signal=signal,
                size=size,
                stop_loss=signal.price * (1 - self.settings.STOP_LOSS_PERCENTAGE),
                take_profit=signal.price * (1 + self.settings.TAKE_PROFIT_PERCENTAGE),
            )
        except Exception as e:
            logger.error(f"Error creating entry signal: {e}")
            return None

    async def _emit_opportunity_alert(
        self, token: Any, signal: Signal, risk_metrics: Dict[str, float], size: float
    ) -> None:
        """Emit alert for new trading opportunity"""
        try:
            await self.alert_system.emit_alert(
                level="info",
                type="new_opportunity",
                message=f"New trading opportunity detected for {token.address}",
                data={
                    "token": token.address,
                    "signal_strength": signal.strength,
                    "proposed_size": size,
                    "risk_score": risk_metrics["risk_score"],
                    "position_risk": risk_metrics["position_risk"],
                    "market_volatility": risk_metrics["market_volatility"],
                },
            )
        except Exception as e:
            logger.error(f"Error emitting opportunity alert: {e}")

    async def _update_paper_positions(self) -> None:
        """Update paper trading positions"""
        try:
            for token_address, position in list(self.state.paper_positions.items()):
                current_price = await self._get_current_price(token_address)
                if not current_price:
                    continue

                position.update_price(current_price)
                should_close, reason = position.should_close()

                if should_close:
                    await self._close_paper_position(token_address, reason)

        except Exception as e:
            logger.error(f"Error updating paper positions: {e}")

    async def _monitor_live_positions(self) -> None:
        """Monitor live trading positions"""
        try:
            positions = self.position_manager.positions.copy()
            for token_address, position in positions.items():
                try:
                    current_price = await self._get_current_price(token_address)
                    if not current_price:
                        logger.warning(
                            f"Failed to get current price for {token_address}"
                        )
                        continue

                    position.update_price(current_price)
                    should_close, reason = position.should_close()

                    if should_close:
                        await self._close_live_position(token_address, reason)

                except Exception as e:
                    logger.error(f"Error monitoring position {token_address}: {e}")

        except Exception as e:
            logger.error(f"Error in position monitoring: {e}")

    async def _process_pending_orders(self) -> None:
        """Process pending entry orders"""
        for signal in self.state.pending_orders[:]:
            try:
                current_price = await self._get_current_price(signal.token_address)
                if not current_price:
                    continue

                price_deviation = abs(current_price - signal.price) / signal.price
                if price_deviation > signal.slippage:
                    logger.warning(
                        f"Price deviation too high for {signal.token_address}: {price_deviation:.2%}"
                    )
                    self.state.pending_orders.remove(signal)
                    continue

                if await self._validate_price_conditions(
                    signal.token_address, signal.size
                ):
                    success = await self._execute_trade(
                        token_address=signal.token_address,
                        size=signal.size,
                        price=current_price,
                    )
                    if success:
                        self.state.pending_orders.remove(signal)
                        logger.info(
                            f"Successfully executed entry for {signal.token_address}"
                        )

            except Exception as e:
                logger.error(f"Error processing order for {signal.token_address}: {e}")
                self.state.pending_orders.remove(signal)

    async def _execute_trade(
        self, token_address: str, size: float, price: float
    ) -> bool:
        """Execute trade based on trading mode"""
        try:
            if self.state.mode == TradingMode.PAPER:
                return await self._execute_paper_trade(token_address, size, price)
            else:
                return await self._execute_live_trade(token_address, size, price)
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False

    async def _execute_paper_trade(
        self, token_address: str, size: float, price: float
    ) -> bool:
        """Execute paper trade"""
        try:
            cost = size * price
            if cost > self.state.paper_balance:
                logger.warning(
                    f"Insufficient paper trading balance: {self.state.paper_balance} < {cost}"
                )
                return False

            trade_entry = TradeEntry(
                token_address=token_address,
                entry_price=price,
                entry_time=datetime.now(),
                size=size,
            )

            position = Position(
                token_address=token_address,
                entry_price=price,
                # current_price=price,
                size=size,
                stop_loss=price * (1 - self.settings.STOP_LOSS_PERCENTAGE),
                take_profit=price * (1 + self.settings.TAKE_PROFIT_PERCENTAGE),
                # unrealized_pnl=0.0,
                status="open",
                trade_entry=trade_entry,
            )

            self.state.paper_balance -= cost
            self.state.paper_positions[token_address] = position
            self.state.daily_stats.trade_count += 1

            await self.alert_system.emit_alert(
                level="info",
                type="paper_trade_opened",
                message=f"Paper position opened for {token_address}",
                data={
                    "size": size,
                    "price": price,
                    "remaining_balance": self.state.paper_balance,
                    "stop_loss": position.stop_loss,
                    "take_profit": position.take_profit,
                },
            )
            return True

        except Exception as e:
            logger.error(f"Paper trade execution error: {e}")
            return False

    async def _execute_live_trade(
        self, token_address: str, size: float, price: float
    ) -> bool:
        """Execute live trade"""
        try:
            balance = await self.wallet.get_balance()
            if float(balance) < (size * price):
                logger.warning("Insufficient balance for live trade")
                return False

            success = await self.swap_executor.execute_swap(
                input_token="So11111111111111111111111111111111111111112",  # SOL
                output_token=token_address,
                amount=size,
                slippage=self.settings.MAX_SLIPPAGE,
            )

            if success:
                position = await self.position_manager.open_position(
                    token_address=token_address,
                    size=size,
                    entry_price=price,
                    stop_loss=price * (1 - self.settings.STOP_LOSS_PERCENTAGE),
                    take_profit=price * (1 + self.settings.TAKE_PROFIT_PERCENTAGE),
                )

                if position:
                    self.state.daily_stats.trade_count += 1
                    await self.alert_system.emit_alert(
                        level="info",
                        type="live_trade_opened",
                        message=f"Live position opened for {token_address}",
                        data={
                            "size": size,
                            "price": price,
                            "stop_loss": position.stop_loss,
                            "take_profit": position.take_profit,
                        },
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"Live trade execution error: {e}")
            return False

    async def _close_paper_position(self, token_address: str, reason: str) -> bool:
        """Close paper trading position"""
        try:
            position = self.state.paper_positions.get(token_address)
            if not position:
                logger.warning(f"Position not found for {token_address}")
                return False

            exit_value = position.size * position.current_price
            self.state.paper_balance += exit_value
            realized_pnl = position.unrealized_pnl

            del self.state.paper_positions[token_address]
            self.state.daily_stats.total_pnl += realized_pnl

            await self.alert_system.emit_alert(
                level="info",
                type="paper_trade_closed",
                message=f"Paper position closed for {token_address}",
                data={
                    "pnl": realized_pnl,
                    "reason": reason,
                    "new_balance": self.state.paper_balance,
                    "exit_price": position.current_price,
                },
            )
            return True

        except Exception as e:
            logger.error(f"Error closing paper position: {e}")
            return False

    async def _close_live_position(self, token_address: str, reason: str) -> bool:
        """Close live trading position"""
        try:
            position = self.position_manager.positions.get(token_address)
            if not position:
                logger.warning(f"Live position not found for {token_address}")
                return False

            success = await self.position_manager.close_position(
                token_address=token_address, reason=reason
            )

            if success:
                self.state.daily_stats.total_pnl += position.unrealized_pnl

                await self.alert_system.emit_alert(
                    level="info",
                    type="live_trade_closed",
                    message=f"Live position closed for {token_address}",
                    data={
                        "pnl": position.unrealized_pnl,
                        "reason": reason,
                        "exit_price": position.current_price,
                    },
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Error closing live position: {e}")
            return False

    async def _calculate_price_impact(
        self, market_depth: Dict[str, Any], size: float
    ) -> float:
        """Calculate expected price impact percentage"""
        try:
            bids = market_depth.get("bids", [])
            total_liquidity = sum(float(level.get("size", 0)) for level in bids)

            if total_liquidity <= 0:
                return 100.0

            impact = (size / total_liquidity) * 100
            logger.debug(f"Calculated price impact: {impact}%")
            return impact

        except Exception as e:
            logger.error(f"Error calculating price impact: {e}")
            return 100.0

    async def _get_current_price(self, token_address: str) -> Optional[float]:
        try:
            price_data = await self.jupiter.get_price(
                input_mint=token_address,
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                amount=1000000  # 1 USDC
            )
            if not price_data:
                return None

            # Convert the output amount to price
            outAmount = float(price_data.get('outAmount', 0))
            return outAmount / 1e9 if outAmount > 0 else None

        except Exception as e:
            logger.error(f"Error getting current price for {token_address}: {e}")
            return None

    async def _update_metrics(self) -> None:
        """Update trading metrics"""
        try:
            if self.state.mode == TradingMode.PAPER:
                portfolio_value = self.state.paper_balance + sum(
                    pos.size * pos.current_price
                    for pos in self.state.paper_positions.values()
                )

                self.monitoring.update_metric("paper_balance", self.state.paper_balance)
                self.monitoring.update_metric("paper_portfolio_value", portfolio_value)
                self.monitoring.update_metric(
                    "paper_position_count", len(self.state.paper_positions)
                )

                if self.settings.INITIAL_PAPER_BALANCE > 0:
                    roi = (
                        (portfolio_value / self.settings.INITIAL_PAPER_BALANCE) - 1
                    ) * 100
                    self.monitoring.update_metric("paper_roi", roi)
            else:
                self.monitoring.update_metric(
                    "live_position_count", len(self.position_manager.positions)
                )

                try:
                    current_balance = await self.wallet.get_balance()
                    self.monitoring.update_metric(
                        "wallet_balance", float(current_balance)
                    )
                except Exception as e:
                    logger.error(f"Error updating wallet balance metric: {e}")

            self.monitoring.update_metric("daily_pnl", self.state.daily_stats.total_pnl)
            self.monitoring.update_metric(
                "daily_trades", self.state.daily_stats.trade_count
            )
            self.monitoring.update_metric(
                "error_count", self.state.daily_stats.error_count
            )
            self.monitoring.update_metric(
                "pending_orders", len(self.state.pending_orders)
            )

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    async def _cleanup_positions(self) -> None:
        """Clean up all positions when stopping trading"""
        try:
            if self.state.mode == TradingMode.PAPER:
                for token_address in list(self.state.paper_positions.keys()):
                    await self._close_paper_position(token_address, "trading_stopped")
            else:
                positions = list(self.position_manager.positions.keys())
                for token_address in positions:
                    await self._close_live_position(token_address, "trading_stopped")

        except Exception as e:
            logger.error(f"Error during position cleanup: {e}")

    async def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive trading metrics"""
        try:
            # Get async values first
            performance_metrics = await self.performance_monitor.get_performance_summary()
            risk_metrics = await self.risk_manager.get_risk_metrics()

            base_metrics = {
                "trading_status": {
                    "mode": self.state.mode.value,
                    "is_trading": self.state.is_trading,
                    "pending_orders": len(self.state.pending_orders),
                    "last_error": self.state.last_error,
                    "daily_stats": {
                        "trade_count": self.state.daily_stats.trade_count,
                        "error_count": self.state.daily_stats.error_count,
                        "total_pnl": self.state.daily_stats.total_pnl,
                        "last_reset": self.state.daily_stats.last_reset.isoformat(),
                    },
                },
                "performance": performance_metrics,
                "risk": risk_metrics,
            }

            if self.state.mode == TradingMode.PAPER:
                portfolio_value = self.state.paper_balance + sum(
                    pos.size * pos.current_price
                    for pos in self.state.paper_positions.values()
                )

                base_metrics.update(
                    {
                        "paper_trading": {
                            "paper_balance": self.state.paper_balance,
                            "portfolio_value": portfolio_value,
                            "roi_percentage": (
                                (portfolio_value / self.settings.INITIAL_PAPER_BALANCE)
                                - 1
                            )
                            * 100,
                            "positions": [
                                {
                                    "token": addr,
                                    "size": pos.size,
                                    "entry_price": pos.entry_price,
                                    "current_price": pos.current_price,
                                    "unrealized_pnl": pos.unrealized_pnl,
                                    "pnl_percentage": (
                                        pos.unrealized_pnl
                                        / (pos.entry_price * pos.size)
                                    )
                                    * 100,
                                }
                                for addr, pos in self.state.paper_positions.items()
                            ],
                        }
                    }
                )
            else:
                base_metrics.update(
                    {
                        "live_trading": {
                            "positions": [
                                {
                                    "token": addr,
                                    "size": pos.size,
                                    "entry_price": pos.entry_price,
                                    "current_price": pos.current_price,
                                    "unrealized_pnl": pos.unrealized_pnl,
                                    "stop_loss": pos.stop_loss,
                                    "take_profit": pos.take_profit,
                                    "pnl_percentage": (
                                        pos.unrealized_pnl
                                        / (pos.entry_price * pos.size)
                                    )
                                    * 100,
                                }
                                for addr, pos in self.position_manager.positions.items()
                            ]
                        }
                    }
                )

            return base_metrics

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {
                "trading_status": {
                    "mode": self.state.mode.value,
                    "is_trading": self.state.is_trading,
                    "error": str(e),
                }
            }
