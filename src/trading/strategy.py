"""
strategy.py - Core trading strategy implementation supporting both paper and live trading
"""
import logging
import asyncio
import os
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
from .transaction_manager import TransactionManager, TransactionStatus

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
    slippage: float = 0.20  # 20% slippage tolerance for volatile meme tokens
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
    completed_trades: List[Dict[str, Any]] = field(default_factory=list)


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
        
        # Major tokens to exclude from trading (not new launches)
        self._excluded_tokens = {
            "So11111111111111111111111111111111111111112",  # SOL
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
            "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",  # BTC
            "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",  # ETH
            "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",   # mSOL
            "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj",  # SAMO
            "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # BONK
            "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",  # WIF
        }

        self.backtest_engine = BacktestEngine(
            settings=settings,
            market_analyzer=self.market_analyzer,
            signal_generator=self.signal_generator,
            jupiter_client=jupiter_client
        )

        self.alert_system = AlertSystem(settings)
        self.swap_executor = SwapExecutor(jupiter_client, wallet)
        self.position_manager = PositionManager(self.swap_executor, settings)
        self.transaction_manager: Optional[TransactionManager] = None
        
        # Link position manager to settings for monitoring access
        settings.position_manager = self.position_manager
        
        self.monitoring = MonitoringSystem(self.alert_system, settings)
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
                # Initialize live trading components
                live_enabled = os.getenv('LIVE_TRADING_ENABLED', 'false').lower() == 'true'
                if not live_enabled:
                    raise ValidationError("Live trading not enabled in environment")
                    
                # Initialize wallet for live mode
                if not await self.wallet.initialize_live_mode():
                    raise ValidationError("Failed to initialize wallet for live trading")
                    
                # Initialize transaction manager
                if self.wallet.rpc_client:
                    self.transaction_manager = TransactionManager(
                        rpc_client=self.wallet.rpc_client,
                        max_retries=int(os.getenv('TRANSACTION_MAX_RETRIES', '3')),
                        confirmation_timeout=int(os.getenv('TRANSACTION_TIMEOUT', '60'))
                    )
                    await self.transaction_manager.start_monitoring()
                    
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
            
            # Initialize scanner if available (CRITICAL FIX)
            if self.scanner and hasattr(self.scanner, 'start_scanning'):
                if not self.scanner.running:
                    # Don't await start_scanning as it's a blocking loop - just initialize it
                    self.scanner.running = True
                    if not self.scanner.session:
                        import aiohttp
                        self.scanner.session = aiohttp.ClientSession()
                    
                    # Initialize Birdeye client if trending is enabled
                    if (self.scanner.trending_analyzer and 
                        getattr(self.settings, 'ENABLE_TRENDING_FILTER', True)):
                        api_key = getattr(self.settings, 'BIRDEYE_API_KEY', None)
                        cache_duration = getattr(self.settings, 'TRENDING_CACHE_DURATION', 300)
                        
                        if not self.scanner.birdeye_client:
                            from ..birdeye_client import BirdeyeClient
                            self.scanner.birdeye_client = BirdeyeClient(api_key, cache_duration)
                            await self.scanner.birdeye_client.__aenter__()
                            
                            if api_key:
                                logger.info(f"[SCANNER] Birdeye trending filter enabled with API key (length: {len(api_key)})")
                            else:
                                logger.info("[SCANNER] Birdeye trending filter enabled in fallback mode (no API key)")
                    
                    logger.info("[SCANNER] Scanner initialized and ready")
                else:
                    logger.info("[SCANNER] Scanner already running")
            else:
                logger.warning("[SCANNER] No scanner available for token discovery")
            
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

            # Stop transaction manager if running
            if self.transaction_manager:
                await self.transaction_manager.stop_monitoring()
                self.transaction_manager = None
            
            # Stop scanner if running (CRITICAL FIX)
            if self.scanner and self.scanner.running:
                self.scanner.running = False
                if self.scanner.session:
                    await self.scanner.session.close()
                    self.scanner.session = None
                
                # Clean up Birdeye client
                if self.scanner.birdeye_client:
                    await self.scanner.birdeye_client.__aexit__(None, None, None)
                    self.scanner.birdeye_client = None
                
                logger.info("[SCANNER] Scanner stopped and cleaned up")

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

    async def _trading_loop(self) -> None:
        """Enhanced trading loop with high-frequency position monitoring"""
        # Start separate high-frequency position monitoring task
        position_monitor_task = asyncio.create_task(self._high_frequency_position_monitor())
        
        try:
            while self.state.is_trading:
                try:
                    # Daily reset check
                    if date.today() > self.state.daily_stats.last_reset:
                        self.state.daily_stats = DailyStats()

                    # Circuit breaker checks
                    if not await self._check_circuit_breakers():
                        logger.warning("Circuit breakers triggered - pausing trading")
                        await self.alert_system.emit_alert(
                            level="warning",
                            type="circuit_breakers",
                            message="Circuit breakers triggered - trading paused",
                            data={"timestamp": datetime.now().isoformat()},
                        )
                        await asyncio.sleep(300)  # 5 minute pause
                        continue

                    # Get recent market data for regime detection
                    prices, volumes = await self._get_recent_market_data()
                    
                    # Adapt parameters to market regime
                    await self._adapt_to_market_regime(prices, volumes)

                    # Process pending orders
                    await self._process_pending_orders()

                    # Check if we can open new positions
                    current_positions = (
                        len(self.state.paper_positions)
                        if self.state.mode == TradingMode.PAPER
                        else len(self.position_manager.positions)
                    )

                    max_positions = getattr(self.settings, 'MAX_SIMULTANEOUS_POSITIONS', self.settings.MAX_POSITIONS)
                    if current_positions < max_positions:
                        await self._scan_opportunities()

                    # Update metrics
                    await self._update_metrics()
                    
                    # Sleep for scan interval (slower loop for opportunity scanning)
                    await asyncio.sleep(self.settings.SCAN_INTERVAL)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Trading loop error: {e}")
                    self.state.daily_stats.error_count += 1
                    await asyncio.sleep(5)
                    
        finally:
            # Clean up position monitoring task
            position_monitor_task.cancel()
            try:
                await position_monitor_task
            except asyncio.CancelledError:
                pass

    async def _high_frequency_position_monitor(self) -> None:
        """High-frequency position monitoring for momentum-based exits"""
        monitor_interval = getattr(self.settings, 'POSITION_MONITOR_INTERVAL', 3.0)
        
        while self.state.is_trading:
            try:
                if self.state.mode == TradingMode.PAPER:
                    await self._monitor_paper_positions_with_momentum()
                else:
                    await self._monitor_live_positions_with_momentum()
                    
                await asyncio.sleep(monitor_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"High-frequency position monitoring error: {e}")
                await asyncio.sleep(1)  # Short sleep on error
                
    async def _monitor_paper_positions_with_momentum(self) -> None:
        """Enhanced paper position monitoring with momentum analysis"""
        try:
            if not self.state.paper_positions:
                return
                
            logger.info(f"[MONITOR] Monitoring {len(self.state.paper_positions)} paper positions...")
            
            for token_address, position in list(self.state.paper_positions.items()):
                try:
                    # Get current price and volume
                    current_price = await self._get_current_price(token_address)
                    if not current_price:
                        logger.warning(f"[MONITOR] Cannot get price for {token_address[:8]}...")
                        continue
                        
                    # Get volume data for momentum analysis
                    market_depth = await self.jupiter.get_market_depth(token_address)
                    current_volume = float(market_depth.get('volume24h', 0)) if market_depth else 0
                    
                    # Update position with price and volume
                    old_price = position.current_price
                    position.update_price(current_price, current_volume)
                    
                    # Log position status
                    pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                    age_mins = position.age_minutes
                    logger.info(f"[HOLD] {token_address[:8]}... - Age: {age_mins:.1f}m, Price: {old_price:.6f}->{current_price:.6f}, P&L: {pnl_pct:+.2f}%")
                    
                    # Update dashboard with position monitoring activity
                    await self._update_dashboard_activity("position_update", {
                        "token": token_address[:8],
                        "age_minutes": age_mins,
                        "entry_price": position.entry_price,
                        "current_price": current_price,
                        "pnl_percentage": pnl_pct,
                        "unrealized_pnl": position.unrealized_pnl,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Check for momentum-based exits
                    age_minutes = (datetime.now() - position.entry_time).total_seconds() / 60
                    
                    # Simple exit conditions for now
                    should_close = False
                    reason = ""
                    
                    # Time-based exit (3 hours max)
                    if age_minutes > 180:
                        should_close = True
                        reason = "time_limit"
                    # Take profit (10% gain)
                    elif position.current_price >= position.entry_price * 1.10:
                        should_close = True
                        reason = "take_profit"
                    # Stop loss (5% loss)
                    elif position.current_price <= position.entry_price * 0.95:
                        should_close = True
                        reason = "stop_loss"
                    
                    if should_close:
                        logger.info(f"[EXIT_TRIGGER] Exit condition met for {token_address[:8]}... - Reason: {reason}")
                        await self._close_paper_position(token_address, reason)
                        
                        # Record exit result for learning
                        profit = position.unrealized_pnl
                        success = profit > 0
                        if hasattr(self.scanner, 'record_entry_result'):
                            self.scanner.record_entry_result(token_address, success, profit)
                            
                        # Emit detailed exit alert
                        await self.alert_system.emit_alert(
                            level="info",
                            type="momentum_exit",
                            message=f"Position closed due to {reason}",
                            data={
                                "token": token_address,
                                "reason": reason,
                                "profit": profit,
                                "profit_percentage": (profit / (position.entry_price * position.size)) * 100 if position.entry_price * position.size > 0 else 0,
                                "hold_time_minutes": position.age_minutes,
                                "exit_price": current_price
                            }
                        )
                        
                except Exception as e:
                    logger.error(f"Error monitoring paper position {token_address[:8]}...: {e}")
                    
        except Exception as e:
            logger.error(f"Error in paper position monitoring: {e}")
            
    async def _monitor_live_positions_with_momentum(self) -> None:
        """Enhanced live position monitoring with momentum analysis"""
        try:
            positions = self.position_manager.positions.copy()
            for token_address, position in positions.items():
                try:
                    # Get current price and volume
                    current_price = await self._get_current_price(token_address)
                    if not current_price:
                        logger.warning(f"Failed to get current price for {token_address}")
                        continue
                        
                    # Get volume data for momentum analysis
                    market_depth = await self.jupiter.get_market_depth(token_address)
                    current_volume = float(market_depth.get('volume24h', 0)) if market_depth else 0
                    
                    # Update position with price and volume data
                    position.update_price(current_price, current_volume)
                    
                    # Check for momentum-based exit conditions
                    should_close, reason = position.should_close()
                    
                    if should_close:
                        await self._close_live_position(token_address, reason)
                        
                        # Record exit result for learning
                        profit = position.unrealized_pnl
                        success = profit > 0
                        if hasattr(self.scanner, 'record_entry_result'):
                            self.scanner.record_entry_result(token_address, success, profit)
                            
                        # Emit detailed exit alert
                        await self.alert_system.emit_alert(
                            level="info",
                            type="momentum_exit",
                            message=f"Live position closed due to {reason}",
                            data={
                                "token": token_address,
                                "reason": reason,
                                "profit": profit,
                                "profit_percentage": position.profit_percentage * 100,
                                "hold_time_minutes": position.age_minutes,
                                "exit_price": current_price,
                                "momentum": position._calculate_momentum(),
                                "rsi": position._calculate_rsi()
                            }
                        )
                        
                except Exception as e:
                    logger.error(f"Error monitoring live position {token_address}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in live position monitoring: {e}")

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
            # For paper trading, use more permissive validation
            if self.state.mode == TradingMode.PAPER:
                # For paper trading, we just need to ensure we can get a price
                current_price = await self._get_current_price(token_address)
                if current_price and current_price > 0:
                    logger.info(f"[OK] Paper trading price validation passed for {token_address[:8]}... - price: {current_price}")
                    return True
                else:
                    logger.warning(f"[ERROR] Cannot get valid price for {token_address[:8]}...")
                    return False
            
            # For live trading, use stricter validation
            market_depth = await self.jupiter.get_market_depth(token_address)
            if not market_depth:
                logger.warning(f"[ERROR] No market depth data for {token_address[:8]}...")
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
        """Scan for new trading opportunities - FIXED for dict format"""
        try:
            # Check if trading is paused
            if getattr(self.settings, 'TRADING_PAUSED', False):
                logger.info("[SCAN] Trading paused - scanning disabled")
                return
            
            if not self.scanner:
                logger.warning("No scanner available")
                return

            logger.info("[SCAN] Starting opportunity scan...")
            
            # Update dashboard that we're scanning
            await self._update_dashboard_activity("scan_started", {
                "timestamp": datetime.now().isoformat()
            })
            
            new_token = await self.scanner.scan_for_new_tokens()
            new_tokens = [new_token] if new_token else []
            logger.info(f"[DATA] Scanner returned {len(new_tokens)} tokens")
            
            # Update dashboard with scan results
            await self._update_dashboard_activity("scan_completed", {
                "tokens_found": len(new_tokens),
                "timestamp": datetime.now().isoformat()
            })
            
            for token_data in new_tokens:
                try:
                    # Handle both dict and object formats
                    if isinstance(token_data, dict):
                        token_address = token_data.get("address")
                        token_info = token_data  # FIX: Use flat structure, not nested
                        logger.info(f"[TARGET] Processing dict token: {token_address[:8] if token_address else 'Unknown'}...")
                    else:
                        token_address = getattr(token_data, "address", None)
                        token_info = token_data
                        logger.info(f"[TARGET] Processing object token: {token_address[:8] if token_address else 'Unknown'}...")
                    
                    if not token_address:
                        logger.warning("[WARN] Token missing address - skipping")
                        continue
                    
                    # Validate Solana token address format
                    if not self._is_valid_solana_token(token_address):
                        logger.warning(f"[WARN] Invalid Solana token address: {token_address[:8]}...")
                        continue
                    
                    if self._is_token_tracked(token_address):
                        logger.info(f"[DATA] Token {token_address[:8]}... already tracked - skipping")
                        continue

                    # Create a standardized token object for validation
                    token_obj = self._create_token_object(token_address, token_info)
                    
                    if not self._validate_token_basics(token_obj):
                        logger.info(f"[ERROR] Token {token_address[:8]}... failed basic validation")
                        continue

                    logger.info(f"[OK] Token {token_address[:8]}... passed validation - analyzing signal...")
                    signal = await self.signal_generator.analyze_token(token_obj)
                    
                    # Use paper trading specific threshold if in paper mode
                    is_paper_trading = (self.state.mode == TradingMode.PAPER)
                    threshold = (self.settings.PAPER_SIGNAL_THRESHOLD if is_paper_trading 
                               else self.settings.SIGNAL_THRESHOLD)
                    
                    if not signal or signal.strength < threshold:
                        mode_str = "PAPER" if is_paper_trading else "LIVE"
                        logger.info(f"[DATA] Token {token_address[:8]}... [{mode_str}] signal too weak: {signal.strength if signal else 'None'} < {threshold}")
                        continue
                    
                    logger.info(f"[SIGNAL] Created signal for {token_address[:8]}... with strength {signal.strength:.2f} (threshold: {threshold:.2f})")

                    logger.info(f"[SIGNAL] Strong signal found for {token_address[:8]}... - calculating risk...")
                    
                    market_volatility = await self.risk_manager.calculate_market_volatility()
                    current_balance = (
                        self.state.paper_balance
                        if self.state.mode == TradingMode.PAPER
                        else await self.wallet.get_balance()
                    )

                    token_data_prepared = await self._prepare_token_data(token_obj, signal)
                    risk_metrics = await self._calculate_risk_metrics(
                        token_data=token_data_prepared,
                        market_volatility=market_volatility,
                        current_balance=float(current_balance),
                        signal=signal,
                    )

                    if not risk_metrics:
                        logger.info(f"[ERROR] Risk metrics calculation failed for {token_address[:8]}...")
                        continue

                    size = self._calculate_position_size(
                        risk_metrics=risk_metrics,
                        current_balance=float(current_balance),
                    )

                    if size <= 0:
                        logger.info(f"[ERROR] Invalid position size for {token_address[:8]}...: {size}")
                        continue

                    logger.info(f"[MONEY] Creating entry signal for {token_address[:8]}... size: {size}")
                    entry_signal = self._create_entry_signal(signal, size)
                    if not entry_signal:
                        logger.warning(f"[ERROR] Failed to create entry signal for {token_address[:8]}...")
                        continue

                    self.state.pending_orders.append(entry_signal)
                    logger.info(f"[PENDING] Added entry signal to pending orders for {token_address[:8]}... (Total pending: {len(self.state.pending_orders)})")
                    
                    await self._emit_opportunity_alert(
                        token=token_obj, signal=signal, risk_metrics=risk_metrics, size=size
                    )
                    
                    logger.info(f"[OK] Successfully processed opportunity for {token_address[:8]}...")
                    
                    # Update dashboard with real-time activity
                    await self._update_dashboard_activity("signal_generated", {
                        "token": token_address[:8],
                        "size": size,
                        "price": signal.price,
                        "timestamp": datetime.now().isoformat()
                    })

                except Exception as e:
                    token_addr = "unknown"
                    try:
                        if isinstance(token_data, dict):
                            token_addr = token_data.get("address", "unknown")[:8]
                        else:
                            token_addr = getattr(token_data, "address", "unknown")[:8]
                    except:
                        pass
                    logger.error(f"[ERROR] Error analyzing token {token_addr}...: {e}")
                    continue
            
            # Add result summary logging
            pending_count = len(self.state.pending_orders)
            mode_str = "PAPER" if self.state.mode == TradingMode.PAPER else "LIVE"
            logger.info(f"[RESULT] Opportunity scan completed - Total pending orders: {pending_count} [{mode_str} mode]")
            
            if pending_count > 0:
                logger.info(f"[PROCESS] Processing {pending_count} pending orders...")
            
            logger.info("[OK] Opportunity scan completed")

        except Exception as e:
            logger.error(f"[ERROR] Error scanning opportunities: {e}")
            import traceback
            logger.error(f"[DATA] Traceback: {traceback.format_exc()}")

    def _is_valid_solana_token(self, token_address: str) -> bool:
        """Validate that this is a proper Solana token address"""
        try:
            # Solana addresses are base58 encoded and typically 32-44 characters
            if not token_address or len(token_address) < 32 or len(token_address) > 44:
                logger.debug(f"Token address length invalid: {len(token_address) if token_address else 0}")
                return False
            
            # Check for valid base58 characters (no 0, O, I, l)
            invalid_chars = set('0OIl')
            if any(char in invalid_chars for char in token_address):
                logger.debug(f"Token address contains invalid base58 characters")
                return False
            
            # Try to decode with base58 if available, otherwise basic validation
            try:
                import base58
                decoded = base58.b58decode(token_address)
                # Solana addresses decode to 32 bytes
                valid = len(decoded) == 32
                if not valid:
                    logger.debug(f"Token address decodes to {len(decoded)} bytes, expected 32")
                return valid
            except ImportError:
                # Fallback: basic character set validation
                valid_chars = set('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz')
                valid = all(char in valid_chars for char in token_address)
                if not valid:
                    logger.debug(f"Token address contains invalid characters")
                return valid
            except Exception as decode_error:
                logger.debug(f"Token address decode error: {decode_error}")
                return False
                
        except Exception as e:
            logger.debug(f"Token validation error: {e}")
            return False
    
    # def _create_token_object(self, token_address: str, token_info: dict):
    #     """Create a standardized token object from dict data"""
    #     try:
    #         # Create a simple object with required attributes for validation
    #         class TokenObject:
    #             def __init__(self, address, info):
    #                 self.address = address
    #                 self.volume24h = info.get("price_sol", 0) * 1000000  # Mock volume from price
    #                 self.liquidity = 500000  # Default above minimum threshold  
    #                 self.market_cap = info.get("market_cap_sol", info.get("market_cap", 0))  # FIX: Check both fields
    #                 self.created_at = info.get("timestamp")
    #                 self.price_sol = info.get("price_sol", 0)
    #                 self.scan_id = info.get("scan_id", 0)
    #                 self.source = info.get("source", "unknown")
            
    #         return TokenObject(token_address, token_info)
    def _create_token_object(self, token_address: str, token_info: dict):
        """Create a standardized token object from dict data - ENHANCED for better data extraction"""
        try:
            # Create a simple object with required attributes for validation
            class TokenObject:
                def __init__(self, address, info):
                    self.address = address
                    
                    # ENHANCED DATA EXTRACTION: Multiple fallback sources
                    # Volume with intelligent fallbacks and estimation
                    volume_raw = info.get("volume_24h", info.get("volume_24h_sol", info.get("volume24h", info.get("volume", 0))))
                    if volume_raw == 0:
                        # Estimate from market cap if available (assume 5% daily turnover)
                        market_cap_raw = info.get("market_cap", info.get("market_cap_sol", 0))
                        if market_cap_raw > 0:
                            volume_raw = market_cap_raw * 0.05
                            logger.info(f"[TOKEN_DATA] Estimated volume from market cap: {volume_raw:.2f} SOL")
                    self.volume24h = float(volume_raw)
                    
                    # Liquidity with smart defaults
                    liquidity_raw = info.get("liquidity", info.get("liquidity_sol", 0))
                    if liquidity_raw == 0:
                        # Default to reasonable value for paper trading
                        liquidity_raw = 500000
                        logger.info(f"[TOKEN_DATA] Using default liquidity: {liquidity_raw} SOL")
                    self.liquidity = float(liquidity_raw)
                    
                    # Market cap with fallbacks
                    market_cap_raw = info.get("market_cap", info.get("market_cap_sol", 0))
                    if market_cap_raw == 0:
                        # Estimate minimal market cap for new tokens
                        market_cap_raw = 100000
                        logger.info(f"[TOKEN_DATA] Using default market cap: {market_cap_raw} SOL")
                    self.market_cap = float(market_cap_raw)
                    
                    # Price with intelligent estimation
                    price_raw = info.get("price", info.get("price_sol", 0))
                    if price_raw == 0:
                        # Estimate price from market cap (assume 1B token supply)
                        if self.market_cap > 0:
                            price_raw = max(0.000001, self.market_cap / 1000000000)
                            logger.info(f"[TOKEN_DATA] Estimated price from market cap: {price_raw:.8f} SOL")
                        else:
                            price_raw = 0.000001  # Minimal default price
                    self.price_sol = float(price_raw)
                    
                    # Other attributes
                    self.created_at = info.get("timestamp")
                    self.scan_id = info.get("scan_id", 0)
                    self.source = info.get("source", "enhanced_scanner")
                    
                    # PAPER TRADING OPTIMIZATION: Mark as paper-friendly
                    self.paper_trading_ready = True
                    
                    logger.info(f"[TOKEN_OBJ] Created enhanced token object for {address[:8]}...")
                    logger.info(f"  Volume: {self.volume24h:.2f} SOL, Liquidity: {self.liquidity:.2f} SOL")
                    logger.info(f"  Price: {self.price_sol:.8f} SOL, Market Cap: {self.market_cap:.0f} SOL")
            
            return TokenObject(token_address, token_info)
            
        except Exception as e:
            logger.error(f"Error creating token object: {e}")
            return None

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
        """Validate basic token properties with Solana-specific filtering - FIXED for paper trading"""
        try:
            # Handle both dict and object types
            if isinstance(token, dict):
                address = token.get("address", "")
                # CRITICAL FIX: Multiple fallback field names for data extraction
                volume_24h = float(token.get("volume_24h_sol", token.get("volume24h", token.get("volume_24h", token.get("volume", 0)))))
                liquidity = float(token.get("liquidity_sol", token.get("liquidity", 500000)))  # Default safe value
                price_sol = float(token.get("price_sol", token.get("price", 0.000001)))  # Default minimal price
                market_cap_sol = float(token.get("market_cap_sol", token.get("market_cap", 100000)))  # Default safe value
                
                # PAPER TRADING BOOST: If volume/price is missing, estimate from available data
                if volume_24h == 0 and market_cap_sol > 0:
                    # Estimate volume from market cap (1% daily turnover assumption)
                    volume_24h = market_cap_sol * 0.01
                    logger.info(f"[ESTIMATE] Estimated volume from market cap: {volume_24h:.2f} SOL")
                
                if price_sol == 0 and market_cap_sol > 0:
                    # Estimate price for small cap tokens
                    price_sol = max(0.000001, market_cap_sol / 1000000000)  # Assume 1B token supply
                    logger.info(f"[ESTIMATE] Estimated price from market cap: {price_sol:.8f} SOL")
                    
            else:
                address = getattr(token, "address", "")
                volume_24h = float(getattr(token, "volume_24h", getattr(token, "volume24h", 0)))
                liquidity = float(getattr(token, "liquidity", 500000))  # Default safe value
                price_sol = float(getattr(token, "price_sol", getattr(token, "price", 0.000001)))
                market_cap_sol = float(getattr(token, "market_cap", 100000))  # Default safe value

            # Check if this is a trending token (more permissive validation)  
            is_trending = token.get('source') == 'birdeye_trending' if isinstance(token, dict) else getattr(token, 'source', '') == 'birdeye_trending'
            
            # Check if we're in paper trading mode
            is_paper_trading = (self.state.mode == TradingMode.PAPER)
            
            # PAPER TRADING OPTIMIZED VALIDATION - Much more permissive thresholds
            if is_paper_trading:
                # ULTRA-PERMISSIVE thresholds for paper trading (maximum opportunities)
                min_volume = 0.0  # Zero volume requirement for paper trading
                min_liquidity = getattr(self.settings, 'PAPER_MIN_LIQUIDITY', 10.0)  # Very low requirement
                min_price = 0.000000001  # Virtually any price
                max_price = 1000.0  # Very high max price
                min_market_cap = 1.0  # Minimal market cap
                max_market_cap = 100000000.0  # Very high max market cap
                logger.info(f"[PAPER_VALIDATION] Using ULTRA-PERMISSIVE paper trading thresholds")
                logger.info(f"  Volume: {min_volume:.1f} SOL, Liquidity: {min_liquidity:.1f} SOL")
                logger.info(f"  Price: {min_price:.9f} - {max_price:.1f} SOL")
                logger.info(f"  Market Cap: {min_market_cap:.1f} - {max_market_cap:.1f} SOL")
            elif is_trending:
                # More permissive thresholds for trending tokens (they're already validated by Birdeye)
                min_volume = max(getattr(self.settings, 'MIN_VOLUME_24H', 50) * 0.2, 5)
                min_liquidity = max(getattr(self.settings, 'MIN_LIQUIDITY', 500) * 0.5, 200)
                min_price = getattr(self.settings, 'MIN_TOKEN_PRICE_SOL', 0.000000001)
                max_price = getattr(self.settings, 'MAX_TOKEN_PRICE_SOL', 1.0)
                min_market_cap = getattr(self.settings, 'MIN_MARKET_CAP_SOL', 10000)
                max_market_cap = getattr(self.settings, 'MAX_MARKET_CAP_SOL', 10000000)
                logger.info(f"[TRENDING_VALIDATION] Using permissive thresholds - Volume: {min_volume:.1f} SOL, Liquidity: {min_liquidity:.1f} SOL")
            else:
                # Standard thresholds for non-trending tokens
                min_volume = max(getattr(self.settings, 'MIN_VOLUME_24H', 50) * 0.5, 10)
                min_liquidity = max(getattr(self.settings, 'MIN_LIQUIDITY', 500) * 0.7, 300)
                min_price = getattr(self.settings, 'MIN_TOKEN_PRICE_SOL', 0.000000001)
                max_price = getattr(self.settings, 'MAX_TOKEN_PRICE_SOL', 1.0)
                min_market_cap = getattr(self.settings, 'MIN_MARKET_CAP_SOL', 10000)
                max_market_cap = getattr(self.settings, 'MAX_MARKET_CAP_SOL', 10000000)
            
            validations = {
                "has_address": bool(address),
                "volume": volume_24h >= min_volume,
                "liquidity": liquidity >= min_liquidity,
                "price_range": (min_price <= price_sol <= max_price),
                "market_cap_range": (min_market_cap <= market_cap_sol <= max_market_cap),
                "not_excluded": address not in getattr(self, '_excluded_tokens', set()),
                "solana_only": self._is_solana_token(address) if getattr(self.settings, 'SOLANA_ONLY', True) else True,
            }

            failed_checks = [k for k, v in validations.items() if not v]
            if failed_checks:
                logger.debug(f"Token {address[:8]}... failed validations: {', '.join(failed_checks)}")
                if failed_checks != ["has_address"]:  # Don't log address issues
                    logger.info(f"[FILTER] FAIL - Token {address[:8]}... REJECTED: {', '.join(failed_checks)}")
                    logger.info(f"  Volume: {volume_24h:.2f} SOL (need: {min_volume:.2f})")
                    logger.info(f"  Liquidity: {liquidity:.2f} SOL (need: {min_liquidity:.2f})")
                    logger.info(f"  Price: {price_sol:.6f} SOL, Market Cap: {market_cap_sol:.0f} SOL")
                return False
            else:
                mode_info = ""
                if is_paper_trading:
                    mode_info = " [PAPER]"
                elif is_trending:
                    mode_info = " [TRENDING]"
                    
                logger.info(f"[FILTER] PASS - Token {address[:8]}... PASSED all validations!{mode_info}")
                logger.info(f"  Volume: {volume_24h:.2f} SOL (threshold: {min_volume:.2f})")
                logger.info(f"  Liquidity: {liquidity:.2f} SOL (threshold: {min_liquidity:.2f})")
                logger.info(f"  Price: {price_sol:.6f} SOL, Market Cap: {market_cap_sol:.0f} SOL")
                if is_paper_trading:
                    logger.info(f"  [PAPER] PAPER TRADING MODE - Used relaxed validation thresholds")
                elif is_trending:
                    logger.info(f"  [TRENDING] TRENDING TOKEN - Used permissive validation thresholds")
                return True

        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return False
    
    def _is_solana_token(self, address: str) -> bool:
        """Check if token address is a valid Solana token"""
        try:
            # Basic Solana address validation (base58, correct length)
            if not address or len(address) < 32 or len(address) > 44:
                return False
            
            # Check if it contains only valid base58 characters
            try:
                import base58
                decoded = base58.b58decode(address)
                return len(decoded) == 32  # Solana public keys are 32 bytes
            except ImportError:
                # Fallback validation without base58 package
                valid_chars = set('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz')
                return all(char in valid_chars for char in address)
            except:
                return False
                
        except Exception as e:
            logger.debug(f"Error validating Solana address: {e}")
            return False

    async def _prepare_token_data(self, token: Any, signal: Signal) -> Dict[str, Any]:
        """Prepare comprehensive token data for analysis - FIXED for new format"""
        try:
            token_data = {
                "address": getattr(token, "address", "unknown"),
                "volume24h": float(getattr(token, "volume24h", 0)),
                "liquidity": float(getattr(token, "liquidity", 500000)),  # Default safe value
                "created_at": getattr(token, "created_at", None),
                "price": signal.price if signal else getattr(token, "price_sol", 0),
                "market_cap": float(getattr(token, "market_cap", 0)),
                "signal_strength": signal.strength if signal else 0.5,
                "signal_type": signal.signal_type if signal else "basic",
                "scan_id": getattr(token, "scan_id", 0),
                "source": getattr(token, "source", "scanner"),
            }
            
            # Add trending data if available (from scanner's trending filter)
            trending_token = getattr(signal, 'trending_token', None) if signal else None
            trending_score = getattr(signal, 'trending_score', None) if signal else None
            
            if trending_token:
                token_data['trending_token'] = trending_token
                token_data['trending_score'] = trending_score
                logger.debug(f"[TRENDING] Added trending data to token preparation: score {trending_score}")
            
            return token_data
            
        except Exception as e:
            logger.error(f"Error preparing token data: {e}")
            return {
                "address": "unknown",
                "volume24h": 0,
                "liquidity": 500000,
                "created_at": None,
                "price": 0,
                "market_cap": 0,
                "signal_strength": 0.5,
                "signal_type": "error",
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
            logger.info(f"[RISK] Calculating risk metrics...")
            logger.info(f"[RISK] Token data keys: {list(token_data.keys())}")
            logger.info(f"[RISK] Market volatility: {market_volatility:.4f}")
            logger.info(f"[RISK] Signal price: {signal.price:.6f} SOL")
            
            risk_score = self.risk_manager.calculate_risk_score(token_data=token_data)
            logger.info(f"[RISK] Risk score: {risk_score:.2f}")

            # Create market conditions object with safe defaults
            trend_strength = float(token_data.get("trend_strength", 0.5))
            liquidity = float(token_data.get("liquidity", 1000000))  # Default 1M liquidity
            min_liquidity = getattr(self.settings, 'MIN_LIQUIDITY', 100000)
            
            liquidity_score = min(liquidity / min_liquidity, 1.0)
            
            logger.info(f"[RISK] Trend strength: {trend_strength:.2f}")
            logger.info(f"[RISK] Liquidity: {liquidity:.0f}")
            logger.info(f"[RISK] Liquidity score: {liquidity_score:.2f}")

            market_conditions = MarketCondition(
                volatility=market_volatility,
                trend_strength=trend_strength,
                liquidity_score=liquidity_score,
                market_regime="trending",  # Could be derived from trend analysis
            )

            # Use MAX_POSITION_SIZE as absolute SOL amount, not percentage
            position_value = min(self.settings.MAX_POSITION_SIZE, current_balance * 0.5)
            logger.info(f"[RISK] Calculated position value: {position_value:.4f} SOL")
            
            position_risk = self.risk_manager.calculate_position_risk(
                position_size=position_value,
                entry_price=signal.price,
                market_conditions=market_conditions,
            )
            logger.info(f"[RISK] Calculated position risk: {position_risk:.2f}%")

            metrics = {
                "risk_score": risk_score,
                "position_risk": position_risk,
                "market_volatility": market_volatility,
                "position_value": position_value,
            }
            
            logger.info(f"[RISK] Final risk metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _calculate_position_size(
        self, risk_metrics: Dict[str, float], current_balance: float
    ) -> float:
        """Calculate final position size based on risk metrics and trading mode"""
        try:
            position_value = risk_metrics["position_value"]
            position_risk = risk_metrics["position_risk"]
            
            # Use paper trading specific position size limits if in paper mode
            is_paper_trading = (self.state.mode == TradingMode.PAPER)
            if is_paper_trading:
                max_position_size = self.settings.PAPER_MAX_POSITION_SIZE
                base_size = self.settings.PAPER_BASE_POSITION_SIZE
                logger.info(f"[PAPER] Using paper trading position sizing - max: {max_position_size:.2%}, base: {base_size:.4f} SOL")
            else:
                max_position_size = self.settings.MAX_POSITION_SIZE
                base_size = None
            
            logger.info(f"[RISK] Position value: {position_value:.4f} SOL")
            logger.info(f"[RISK] Position risk: {position_risk:.2f}%")
            logger.info(f"[RISK] Current balance: {current_balance:.4f} SOL")
            logger.info(f"[RISK] Max position size: {max_position_size:.2%}")
            
            # Cap position risk at 50% to prevent zero sizes
            capped_risk = min(position_risk, 50.0)
            
            # Calculate size with risk adjustment
            risk_adjusted_size = position_value * (1 - (capped_risk / 100))
            
            # Apply maximum position size limit
            max_allowed_size = current_balance * max_position_size
            
            # Apply circuit breaker limit
            circuit_breaker_limit = getattr(self.circuit_breakers, 'max_position_size', float('inf'))
            
            # For paper trading, ensure we have at least the base size
            if is_paper_trading and base_size:
                # Use the larger of risk-adjusted size or base size, but not exceed limits
                preferred_size = max(risk_adjusted_size, base_size)
                size = min(preferred_size, max_allowed_size, circuit_breaker_limit)
                logger.info(f"[PAPER] Preferred size (base or risk-adjusted): {preferred_size:.4f} SOL")
            else:
                size = min(risk_adjusted_size, max_allowed_size, circuit_breaker_limit)
            
            logger.info(f"[CALC] Risk-adjusted size: {risk_adjusted_size:.4f} SOL")
            logger.info(f"[CALC] Max allowed size: {max_allowed_size:.4f} SOL")
            logger.info(f"[CALC] Final position size: {size:.4f} SOL")

            return max(size, 0)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0

    # def _create_entry_signal(
    #     self, signal: Signal, size: float
    # ) -> Optional[EntrySignal]:
    #     """Create entry signal with validation"""
    #     try:
    #         return EntrySignal.from_signal(
    #             signal=signal,
    #             size=size,
    #             stop_loss=signal.price * (1 - self.settings.STOP_LOSS_PERCENTAGE),
    #             take_profit=signal.price * (1 + self.settings.TAKE_PROFIT_PERCENTAGE),
    #         )
    #     except Exception as e:
    #         logger.error(f"Error creating entry signal: {e}")
    #         return None

    def _create_entry_signal(self, signal: Signal, size: float) -> Optional[EntrySignal]:
        """Create entry signal with dynamic slippage based on opportunity potential and trading mode"""
        try:
            # Determine if we're in paper trading mode
            is_paper_trading = (self.state.mode == TradingMode.PAPER)
            
            # Use paper trading specific slippage or live trading slippage
            if is_paper_trading:
                base_slippage = self.settings.PAPER_TRADING_SLIPPAGE  # 50% for paper trading
                logger.info(f"[PAPER] Using paper trading slippage: {base_slippage:.1%}")
            else:
                base_slippage = 0.20  # 20% base slippage for live trading
                
            # Check if this is a trending token with high potential
            trending_data = getattr(signal, 'market_data', {})
            trending_score = trending_data.get('trending_score', 0)
            
            if trending_score > 70 and not is_paper_trading:  # Only apply trending bonus for live trading
                dynamic_slippage = min(1.5, base_slippage * (1 + trending_score / 50))  # Up to 150%
                logger.info(f"[SLIPPAGE] High-potential token detected - using {dynamic_slippage:.1%} slippage tolerance")
            else:
                dynamic_slippage = base_slippage
                
            # Create EntrySignal manually instead of using from_signal
            entry_signal = EntrySignal(
                token_address=signal.token_address,
                price=signal.price,
                confidence=signal.strength,
                entry_type=signal.signal_type,
                size=size,
                stop_loss=signal.price * (1 - self.settings.STOP_LOSS_PERCENTAGE),
                take_profit=signal.price * (1 + self.settings.TAKE_PROFIT_PERCENTAGE),
                slippage=dynamic_slippage  # Set dynamic slippage directly
            )
            
            return entry_signal
            
        except Exception as e:
            logger.error(f"Error creating entry signal: {e}")
            return None

    async def _emit_opportunity_alert(
        self, token: Any, signal: Signal, risk_metrics: Dict[str, float], size: float
    ) -> None:
        """Emit alert for new trading opportunity - FIXED for new format"""
        try:
            token_address = getattr(token, "address", "unknown")
            signal_strength = signal.strength if signal else 0.5
            
            await self.alert_system.emit_alert(
                level="info",
                type="new_opportunity",
                message=f"New trading opportunity detected for {token_address[:8]}...",
                data={
                    "token": token_address,
                    "signal_strength": signal_strength,
                    "proposed_size": size,
                    "risk_score": risk_metrics.get("risk_score", 50),
                    "position_risk": risk_metrics.get("position_risk", 50),
                    "market_volatility": risk_metrics.get("market_volatility", 0.5),
                    "scan_id": getattr(token, "scan_id", 0),
                    "source": getattr(token, "source", "scanner"),
                },
            )
            logger.info(f"[DATA] Opportunity alert emitted for {token_address[:8]}...")
        except Exception as e:
            logger.error(f"Error emitting opportunity alert: {e}")

    async def _update_paper_positions(self) -> None:
        """Legacy method - now handled by high-frequency monitor"""
        # This method is now mostly handled by _monitor_paper_positions_with_momentum
        # Keeping for compatibility but functionality moved to high-frequency monitor
        pass

    async def _monitor_live_positions(self) -> None:
        """Legacy method - now handled by high-frequency monitor"""
        # This method is now mostly handled by _monitor_live_positions_with_momentum
        # Keeping for compatibility but functionality moved to high-frequency monitor
        pass

    async def _process_pending_orders(self) -> None:
        """Process pending entry orders"""
        if not self.state.pending_orders:
            return
            
        logger.info(f"[PROCESS] Processing {len(self.state.pending_orders)} pending orders...")
        
        for signal in self.state.pending_orders[:]:
            try:
                logger.info(f"[ORDER] Processing order for {signal.token_address[:8]}... size: {signal.size}")
                
                current_price = await self._get_current_price(signal.token_address)
                if not current_price:
                    logger.warning(f"[ERROR] Cannot get current price for {signal.token_address[:8]}...")
                    continue

                price_deviation = abs(current_price - signal.price) / signal.price
                logger.info(f"[PRICE] Current price: {current_price}, Signal price: {signal.price}, Deviation: {price_deviation:.2%}")
                
                if price_deviation > signal.slippage:
                    logger.warning(
                        f"[SLIPPAGE] Price deviation too high for {signal.token_address[:8]}...: {price_deviation:.2%} > {signal.slippage:.2%}"
                    )
                    self.state.pending_orders.remove(signal)
                    continue

                logger.info(f"[VALIDATE] Validating price conditions for {signal.token_address[:8]}...")
                if await self._validate_price_conditions(
                    signal.token_address, signal.size
                ):
                    logger.info(f"[EXECUTE] Executing trade for {signal.token_address[:8]}...")
                    success = await self._execute_trade(
                        token_address=signal.token_address,
                        size=signal.size,
                        price=current_price,
                    )
                    if success:
                        self.state.pending_orders.remove(signal)
                        logger.info(
                            f"[SUCCESS] Successfully executed entry for {signal.token_address[:8]}... - Position opened!"
                        )
                    else:
                        logger.error(f"[FAILED] Trade execution failed for {signal.token_address[:8]}...")
                else:
                    logger.warning(f"[BLOCKED] Price conditions not met for {signal.token_address[:8]}...")

            except Exception as e:
                logger.error(f"[ERROR] Error processing order for {signal.token_address[:8]}...: {e}")
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
            logger.info(f"[PAPER] Attempting paper trade for {token_address[:8]}... - Cost: {cost:.4f} SOL, Balance: {self.state.paper_balance:.4f} SOL")
            
            if cost > self.state.paper_balance:
                logger.warning(
                    f"[BALANCE] Insufficient paper trading balance: {self.state.paper_balance:.4f} < {cost:.4f}"
                )
                return False

            trade_entry = TradeEntry(
                token_address=token_address,
                entry_price=price,
                entry_time=datetime.now(),
                size=size,
            )

            stop_loss_price = price * (1 - self.settings.STOP_LOSS_PERCENTAGE)
            take_profit_price = price * (1 + self.settings.TAKE_PROFIT_PERCENTAGE)

            position = Position(
                token_address=token_address,
                entry_price=price,
                # current_price=price,
                size=size,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                # unrealized_pnl=0.0,
                status="open",
                trade_entry=trade_entry,
            )

            # Update balances and positions
            self.state.paper_balance -= cost
            self.state.paper_positions[token_address] = position
            self.state.daily_stats.trade_count += 1

            logger.info(f"[TRADE] 🚀 PAPER POSITION OPENED! 🚀")
            logger.info(f"  Token: {token_address[:8]}...")
            logger.info(f"  Size: {size:.4f} tokens")
            logger.info(f"  Entry Price: {price:.6f} SOL")
            logger.info(f"  Cost: {cost:.4f} SOL")
            logger.info(f"  Stop Loss: {stop_loss_price:.6f} SOL ({getattr(self.settings, 'STOP_LOSS_PERCENTAGE', 0.15):.1%})")
            logger.info(f"  Take Profit: {take_profit_price:.6f} SOL ({getattr(self.settings, 'TAKE_PROFIT_PERCENTAGE', 0.25):.1%})")
            logger.info(f"  Previous Balance: {self.state.paper_balance + cost:.4f} SOL")
            logger.info(f"  Remaining Balance: {self.state.paper_balance:.4f} SOL")
            logger.info(f"  Total Active Positions: {len(self.state.paper_positions)}")
            logger.info(f"[PAPER] 💰 TRADE EXECUTED - Balance changed from {self.state.paper_balance + cost:.4f} to {self.state.paper_balance:.4f} SOL")

            # CRITICAL: Add to completed trades for dashboard
            trade_record = {
                "token": token_address[:8],
                "type": "paper_buy",
                "entry_price": price,
                "size": size,
                "cost": cost,
                "entry_time": datetime.now().isoformat(),
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "status": "open",
                "timestamp": datetime.now().isoformat()
            }
            self.state.completed_trades.append(trade_record)
            logger.info(f"[DASHBOARD] Added trade record to completed trades (Total: {len(self.state.completed_trades)})")

            await self.alert_system.emit_alert(
                level="info",
                type="paper_trade_opened",
                message=f"💰 Paper position opened for {token_address[:8]}... - Balance: {self.state.paper_balance:.4f} SOL",
                data={
                    "token": token_address,
                    "size": size,
                    "price": price,
                    "cost": cost,
                    "remaining_balance": self.state.paper_balance,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price,
                    "total_positions": len(self.state.paper_positions)
                },
            )
            
            # Update dashboard with trade execution
            await self._update_dashboard_with_trade({
                "type": "buy",
                "token_address": token_address,
                "symbol": "UNKNOWN",  # Will be updated if available
                "size": size,
                "price": price,
                "cost": cost,
                "timestamp": datetime.now().isoformat(),
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "status": "open"
            })
            
            await self._update_dashboard_activity("trade_executed", {
                "token": token_address[:8],
                "type": "paper_buy",
                "size": size,
                "price": price,
                "cost": cost,
                "remaining_balance": self.state.paper_balance,
                "positions": len(self.state.paper_positions),
                "timestamp": datetime.now().isoformat()
            })
            
            return True

        except Exception as e:
            logger.error(f"Paper trade execution error: {e}")
            return False

    async def execute_live_trade(self, signal: EntrySignal) -> bool:
        """Execute live trade with enhanced validation and monitoring"""
        try:
            if not self.wallet.live_mode or not self.transaction_manager:
                logger.error("Live mode not properly initialized")
                return False
                
            # Pre-trade validation
            if not await self.verify_live_balance(signal.size * signal.price):
                return False
                
            # Check emergency controls
            if not await self._check_emergency_controls():
                logger.warning("Emergency controls triggered - trade blocked")
                return False
                
            # Execute swap with enhanced monitoring
            signature = await self.swap_executor.execute_swap(
                input_token="So11111111111111111111111111111111111111112",  # SOL
                output_token=signal.token_address,
                amount=signal.size,
                slippage=signal.slippage
            )
            
            if not signature:
                logger.error(f"Failed to execute swap for {signal.token_address}")
                return False
                
            # Wait for transaction confirmation
            confirmed = await self.wait_for_transaction_confirmation(signature)
            if not confirmed:
                logger.error(f"Transaction confirmation failed: {signature}")
                return False
                
            # Open position tracking
            position = await self.position_manager.open_position(
                token_address=signal.token_address,
                size=signal.size,
                entry_price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if position:
                self.state.daily_stats.trade_count += 1
                
                # Sync position with blockchain state
                await self.sync_positions_with_blockchain()
                
                await self.alert_system.emit_alert(
                    level="info",
                    type="live_trade_executed",
                    message=f"Live trade executed for {signal.token_address}",
                    data={
                        "signature": signature,
                        "token": signal.token_address,
                        "size": signal.size,
                        "price": signal.price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit
                    }
                )
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Live trade execution error: {e}")
            return False
    
    async def close_live_position(self, token_address: str, reason: str) -> bool:
        """Close live position with enhanced monitoring"""
        try:
            if not self.wallet.live_mode or not self.transaction_manager:
                logger.error("Live mode not properly initialized")
                return False
                
            position = self.position_manager.positions.get(token_address)
            if not position:
                logger.warning(f"No position found for {token_address}")
                return False
                
            # Execute closing swap
            signature = await self.swap_executor.execute_swap(
                input_token=token_address,
                output_token="So11111111111111111111111111111111111111112",  # SOL
                amount=position.size,
                slippage=self.settings.MAX_SLIPPAGE
            )
            
            if not signature:
                logger.error(f"Failed to close position for {token_address}")
                return False
                
            # Wait for confirmation
            confirmed = await self.wait_for_transaction_confirmation(signature)
            if not confirmed:
                logger.error(f"Position close confirmation failed: {signature}")
                return False
                
            # Close position in manager
            success = await self.position_manager.close_position(token_address, reason)
            
            if success:
                # Sync with blockchain
                await self.sync_positions_with_blockchain()
                
                await self.alert_system.emit_alert(
                    level="info",
                    type="live_position_closed",
                    message=f"Live position closed for {token_address}",
                    data={
                        "signature": signature,
                        "token": token_address,
                        "reason": reason,
                        "pnl": position.unrealized_pnl
                    }
                )
                
            return success
            
        except Exception as e:
            logger.error(f"Error closing live position: {e}")
            return False
    
    async def verify_live_balance(self, required_amount: float) -> bool:
        """Verify sufficient balance for live trading"""
        try:
            if not self.wallet.live_mode:
                return False
                
            current_balance = await self.wallet.get_balance()
            if not current_balance:
                logger.error("Failed to get current balance")
                return False
                
            balance = float(current_balance)
            
            # Add buffer for fees
            fee_buffer = float(os.getenv('TRANSACTION_FEE_BUFFER', '0.01'))  # 0.01 SOL buffer
            total_required = required_amount + fee_buffer
            
            if balance < total_required:
                logger.warning(
                    f"Insufficient balance: {balance} SOL available, "
                    f"{total_required} SOL required (including {fee_buffer} SOL fee buffer)"
                )
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error verifying balance: {e}")
            return False
    
    async def wait_for_transaction_confirmation(
        self, 
        signature: str, 
        timeout: int = 60
    ) -> bool:
        """Wait for transaction confirmation with timeout"""
        try:
            if not self.transaction_manager:
                logger.error("Transaction manager not initialized")
                return False
                
            status = await self.transaction_manager.wait_for_confirmation(
                signature=signature,
                timeout=timeout
            )
            
            if status == TransactionStatus.CONFIRMED or status == TransactionStatus.FINALIZED:
                logger.info(f"Transaction confirmed: {signature}")
                return True
            elif status == TransactionStatus.FAILED:
                logger.error(f"Transaction failed: {signature}")
                return False
            elif status == TransactionStatus.TIMEOUT:
                logger.warning(f"Transaction confirmation timeout: {signature}")
                return False
            else:
                logger.warning(f"Unexpected transaction status: {status}")
                return False
                
        except Exception as e:
            logger.error(f"Error waiting for confirmation: {e}")
            return False
    
    async def sync_positions_with_blockchain(self) -> None:
        """Synchronize position state with blockchain reality"""
        try:
            if not self.wallet.live_mode:
                return
                
            # Get current token balances from wallet
            token_accounts = await self.wallet.get_token_accounts()
            
            # Update position manager with actual balances
            for account in token_accounts:
                token_address = account.get('mint')
                balance = account.get('balance', 0.0)
                
                if token_address in self.position_manager.positions:
                    position = self.position_manager.positions[token_address]
                    # Update position size if different from blockchain
                    if abs(position.size - balance) > 0.001:  # Small tolerance for rounding
                        logger.info(
                            f"Syncing position size for {token_address}: "
                            f"{position.size} -> {balance}"
                        )
                        position.size = balance
                        
            logger.debug("Position synchronization completed")
            
        except Exception as e:
            logger.error(f"Error syncing positions with blockchain: {e}")
    
    async def _check_emergency_controls(self) -> bool:
        """Check emergency trading controls"""
        try:
            # Check daily loss limits
            if self.state.daily_stats.total_pnl <= -float(os.getenv('MAX_DAILY_LOSS', '100')):
                return False
                
            # Check maximum position count
            max_positions = int(os.getenv('MAX_LIVE_POSITIONS', '5'))
            if len(self.position_manager.positions) >= max_positions:
                return False
                
            # Check error count
            if self.state.daily_stats.error_count >= int(os.getenv('MAX_DAILY_ERRORS', '10')):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking emergency controls: {e}")
            return False
    
    async def _execute_live_trade(
        self, token_address: str, size: float, price: float
    ) -> bool:
        """Legacy method - redirects to new execute_live_trade"""
        signal = EntrySignal(
            token_address=token_address,
            price=price,
            confidence=0.7,  # Default confidence
            entry_type="buy",
            size=size,
            stop_loss=price * (1 - self.settings.STOP_LOSS_PERCENTAGE),
            take_profit=price * (1 + self.settings.TAKE_PROFIT_PERCENTAGE)
        )
        return await self.execute_live_trade(signal)

    async def _close_paper_position(self, token_address: str, reason: str) -> bool:
        """Close paper trading position"""
        try:
            position = self.state.paper_positions.get(token_address)
            if not position:
                logger.warning(f"[CLOSE] Position not found for {token_address[:8]}...")
                return False

            exit_value = position.size * position.current_price
            old_balance = self.state.paper_balance
            self.state.paper_balance += exit_value
            realized_pnl = position.unrealized_pnl

            # Calculate percentage gain/loss
            entry_cost = position.size * position.entry_price
            pnl_percentage = (realized_pnl / entry_cost) * 100 if entry_cost > 0 else 0

            logger.info(f"[EXIT] Closing paper position for {token_address[:8]}...")
            logger.info(f"  Reason: {reason}")
            logger.info(f"  Entry Price: {position.entry_price:.6f} SOL")
            logger.info(f"  Exit Price: {position.current_price:.6f} SOL")
            logger.info(f"  Size: {position.size:.4f} tokens")
            logger.info(f"  Entry Cost: {entry_cost:.4f} SOL")
            logger.info(f"  Exit Value: {exit_value:.4f} SOL")
            logger.info(f"  Realized P&L: {realized_pnl:.4f} SOL ({pnl_percentage:+.2f}%)")
            logger.info(f"  Balance: {old_balance:.4f} -> {self.state.paper_balance:.4f} SOL")
            logger.info(f"  Remaining Positions: {len(self.state.paper_positions)-1}")

            del self.state.paper_positions[token_address]
            self.state.daily_stats.total_pnl += realized_pnl

            await self.alert_system.emit_alert(
                level="info",
                type="paper_trade_closed",
                message=f"Paper position closed for {token_address[:8]}... - {reason}",
                data={
                    "token": token_address,
                    "pnl": realized_pnl,
                    "pnl_percentage": pnl_percentage,
                    "reason": reason,
                    "entry_price": position.entry_price,
                    "exit_price": position.current_price,
                    "size": position.size,
                    "entry_cost": entry_cost,
                    "exit_value": exit_value,
                    "new_balance": self.state.paper_balance,
                    "remaining_positions": len(self.state.paper_positions)
                },
            )
            
            # Update the existing trade record in the dashboard to closed status
            await self._update_dashboard_trade_close({
                "token_address": token_address,
                "exit_time": datetime.now().isoformat(),
                "exit_price": position.current_price,
                "pnl": realized_pnl,
                "pnl_percentage": pnl_percentage,
                "reason": reason,
                "status": "closed"
            })
            
            # Store completed trade in memory for strategy tracking
            trade_record = {
                "token": token_address,
                "entry_time": position.trade_entry.entry_time,
                "exit_time": datetime.now(),
                "entry_price": position.entry_price,
                "exit_price": position.current_price,
                "size": position.size,
                "pnl": realized_pnl,
                "pnl_percentage": pnl_percentage,
                "reason": reason,
                "type": "paper"
            }
            
            if not hasattr(self.state, 'completed_trades'):
                self.state.completed_trades = []
            self.state.completed_trades.append(trade_record)
            
            # Update dashboard to indicate position closed
            await self._update_dashboard_activity("position_closed", {
                "token": token_address[:8],
                "exit_reason": reason,
                "realized_pnl": realized_pnl,
                "pnl_percentage": pnl_percentage,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"[COMPLETE] Trade completed and recorded - Total completed trades: {len(self.state.completed_trades)}")
            
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
        """Get current token price in SOL using multiple methods"""
        try:
            # Method 1: Try Jupiter quote API first
            try:
                price_data = await self.jupiter.get_quote(
                    token_address,
                    "So11111111111111111111111111111111111111112",  # SOL
                    "1000000000",  # 1 token (adjust for decimals)
                    50  # 0.5% slippage
                )
                
                if price_data and "outAmount" in price_data:
                    price_sol = float(price_data["outAmount"]) / 1e9
                    logger.debug(f"[PRICE] Jupiter price for {token_address[:8]}...: {price_sol:.8f} SOL")
                    return price_sol
            except Exception as e:
                logger.debug(f"[PRICE] Jupiter quote failed for {token_address[:8]}...: {e}")
            
            # Method 2: Try Jupiter price API if quote fails
            try:
                if hasattr(self.jupiter, 'get_price'):
                    price_info = await self.jupiter.get_price(token_address)
                    if price_info and 'price' in price_info:
                        # Convert USD to SOL (approximate)
                        price_usd = float(price_info['price'])
                        from src.utils.price_manager import get_sol_usd_price
                        sol_price_usd = await get_sol_usd_price()
                        price_sol = price_usd / sol_price_usd
                        logger.debug(f"[PRICE] Jupiter price API for {token_address[:8]}...: {price_sol:.8f} SOL")
                        return price_sol
            except Exception as e:
                logger.debug(f"[PRICE] Jupiter price API failed for {token_address[:8]}...: {e}")
            
            # Method 3: Use Birdeye data if available
            try:
                if hasattr(self, 'scanner') and hasattr(self.scanner, 'birdeye_client') and self.scanner.birdeye_client:
                    trending_token = self.scanner.birdeye_client.get_cached_token_by_address(token_address)
                    if trending_token and trending_token.price > 0:
                        # Convert USD price to SOL
                        from src.utils.price_manager import get_sol_usd_price
                        sol_price_usd = await get_sol_usd_price()
                        price_sol = trending_token.price / sol_price_usd
                        logger.debug(f"[PRICE] Birdeye price for {token_address[:8]}...: {price_sol:.8f} SOL")
                        return price_sol
            except Exception as e:
                logger.debug(f"[PRICE] Birdeye price failed for {token_address[:8]}...: {e}")
            
            # Method 4: Try market depth API for fallback pricing
            try:
                market_depth = await self.jupiter.get_market_depth(token_address)
                if market_depth and 'price' in market_depth:
                    price_sol = float(market_depth['price'])
                    logger.debug(f"[PRICE] Market depth price for {token_address[:8]}...: {price_sol:.8f} SOL")
                    return price_sol
            except Exception as e:
                logger.debug(f"[PRICE] Market depth failed for {token_address[:8]}...: {e}")
            
            logger.warning(f"[PRICE] All price methods failed for {token_address[:8]}...")
            return None

        except Exception as e:
            logger.error(f"[PRICE] Error getting current price for {token_address[:8]}...: {e}")
            return None

    async def _save_trade_to_dashboard(self, trade_record: Dict[str, Any]) -> None:
        """Save completed trade to dashboard data file"""
        try:
            import json
            dashboard_file = "bot_data.json"
            
            # Load existing data
            try:
                with open(dashboard_file, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = {
                    "status": "running",
                    "trades": [],
                    "performance": {
                        "total_pnl": 0.0,
                        "win_rate": 0.0,
                        "total_trades": 0,
                        "balance": self.state.paper_balance
                    },
                    "last_update": datetime.now().isoformat()
                }
            
            # Format trade for dashboard
            dashboard_trade = {
                "token": trade_record["token"][:8] + "...",
                "entry_price": trade_record["entry_price"],
                "exit_price": trade_record["exit_price"],
                "entry_time": trade_record["entry_time"].strftime("%H:%M:%S"),
                "exit_time": trade_record["exit_time"].strftime("%H:%M:%S"),
                "pnl": trade_record["pnl"],
                "pnl_percentage": trade_record["pnl_percentage"],
                "reason": trade_record["reason"],
                "timestamp": trade_record["exit_time"].strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add trade
            data["trades"].append(dashboard_trade)
            
            # Update performance metrics
            data["performance"]["total_trades"] = len(data["trades"])
            data["performance"]["total_pnl"] = sum(t["pnl"] for t in data["trades"])
            data["performance"]["balance"] = self.state.paper_balance
            
            # Calculate win rate
            wins = sum(1 for t in data["trades"] if t["pnl"] > 0)
            data["performance"]["win_rate"] = (wins / len(data["trades"])) * 100 if data["trades"] else 0
            
            data["last_update"] = datetime.now().isoformat()
            
            # Save back to file
            with open(dashboard_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"[DASHBOARD] Trade saved to dashboard - Total: {data['performance']['total_trades']}, Win Rate: {data['performance']['win_rate']:.1f}%")
            
        except Exception as e:
            logger.error(f"Error saving trade to dashboard: {e}")

    async def _update_dashboard_with_trade(self, trade_data: Dict[str, Any]) -> None:
        """Update dashboard with trade information in trades array"""
        try:
            import json
            dashboard_file = "bot_data.json"
            
            # Load existing data
            try:
                with open(dashboard_file, 'r') as f:
                    dashboard_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                dashboard_data = {
                    "status": "running",
                    "trades": [],
                    "performance": {
                        "total_pnl": 0.0,
                        "win_rate": 0.0,
                        "total_trades": 0,
                        "balance": self.state.paper_balance,
                        "unrealized_pnl": 0,
                        "open_positions": 0
                    },
                    "activity": [],
                    "last_update": datetime.now().isoformat()
                }
            
            # Ensure trades array exists
            if "trades" not in dashboard_data:
                dashboard_data["trades"] = []
            
            # Add trade to trades array
            dashboard_data["trades"].append(trade_data)
            
            # Update performance metrics
            if "performance" not in dashboard_data:
                dashboard_data["performance"] = {
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "balance": self.state.paper_balance,
                    "unrealized_pnl": 0,
                    "open_positions": 0
                }
            
            # Update trade count and balance
            dashboard_data["performance"]["total_trades"] = len(dashboard_data["trades"])
            dashboard_data["performance"]["balance"] = self.state.paper_balance
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.state.paper_positions.values())
            dashboard_data["performance"]["unrealized_pnl"] = unrealized_pnl
            dashboard_data["performance"]["open_positions"] = len(self.state.paper_positions)
            
            # Update win rate
            completed_trades = [t for t in dashboard_data["trades"] if t.get("status") == "closed"]
            if completed_trades:
                winning_trades = [t for t in completed_trades if t.get("pnl", 0) > 0]
                dashboard_data["performance"]["win_rate"] = len(winning_trades) / len(completed_trades) * 100
            
            # Update total PnL
            total_pnl = sum(t.get("pnl", 0) for t in completed_trades)
            dashboard_data["performance"]["total_pnl"] = total_pnl
            
            # Update status and timestamp
            dashboard_data["status"] = "running"
            dashboard_data["last_update"] = datetime.now().isoformat()
            
            # Save updated data
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
                
            logger.info(f"[DASHBOARD] Trade added to dashboard - Total trades: {len(dashboard_data['trades'])}")
            
        except Exception as e:
            logger.error(f"Error updating dashboard with trade: {e}")

    async def _update_dashboard_trade_close(self, close_data: Dict[str, Any]) -> None:
        """Update existing trade record in dashboard when position is closed"""
        try:
            import json
            dashboard_file = "bot_data.json"
            
            # Load existing data
            try:
                with open(dashboard_file, 'r') as f:
                    dashboard_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logger.error("Dashboard file not found or corrupted when trying to close trade")
                return
            
            if "trades" not in dashboard_data:
                logger.error("No trades array found in dashboard data")
                return
                
            # Find the corresponding open trade and update it
            token_address = close_data["token_address"]
            for trade in dashboard_data["trades"]:
                if (trade.get("token_address") == token_address and 
                    trade.get("status") == "open"):
                    # Update the trade with closing information
                    trade["exit_time"] = close_data["exit_time"]
                    trade["exit_price"] = close_data["exit_price"]
                    trade["pnl"] = close_data["pnl"]
                    trade["pnl_percentage"] = close_data["pnl_percentage"]
                    trade["reason"] = close_data["reason"]
                    trade["status"] = "closed"
                    
                    logger.info(f"[DASHBOARD] Updated trade record for {token_address[:8]}... - PnL: {close_data['pnl']:.4f} SOL")
                    break
            else:
                logger.warning(f"[DASHBOARD] Could not find open trade record for {token_address[:8]}...")
            
            # Recalculate performance metrics
            completed_trades = [t for t in dashboard_data["trades"] if t.get("status") == "closed"]
            if completed_trades:
                winning_trades = [t for t in completed_trades if t.get("pnl", 0) > 0]
                dashboard_data["performance"]["win_rate"] = len(winning_trades) / len(completed_trades) * 100
                
                total_pnl = sum(t.get("pnl", 0) for t in completed_trades)
                dashboard_data["performance"]["total_pnl"] = total_pnl
            
            # Update trade count
            dashboard_data["performance"]["total_trades"] = len(dashboard_data["trades"])
            dashboard_data["performance"]["balance"] = self.state.paper_balance
            dashboard_data["performance"]["open_positions"] = len(self.state.paper_positions)
            
            # Update timestamp
            dashboard_data["last_update"] = datetime.now().isoformat()
            
            # Save updated data
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating dashboard trade close: {e}")

    async def _update_dashboard_activity(self, activity_type: str, data: Dict[str, Any]) -> None:
        """Update dashboard with real-time bot activity"""
        try:
            import json
            dashboard_file = "bot_data.json"
            
            # Load existing data
            try:
                with open(dashboard_file, 'r') as f:
                    dashboard_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                dashboard_data = {
                    "status": "running",
                    "trades": [],
                    "performance": {
                        "total_pnl": 0.0,
                        "win_rate": 0.0,
                        "total_trades": 0,
                        "balance": self.state.paper_balance
                    },
                    "activity": [],
                    "last_update": datetime.now().isoformat()
                }
            
            # Add activity entry
            if "activity" not in dashboard_data:
                dashboard_data["activity"] = []
            
            activity_entry = {
                "type": activity_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Keep only last 50 activities for performance
            dashboard_data["activity"].append(activity_entry)
            if len(dashboard_data["activity"]) > 50:
                dashboard_data["activity"] = dashboard_data["activity"][-50:]
            
            # Update status and timestamp
            dashboard_data["status"] = "running"
            dashboard_data["last_update"] = datetime.now().isoformat()
            
            # Update performance metrics with current positions
            if "performance" not in dashboard_data:
                dashboard_data["performance"] = {
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "balance": self.state.paper_balance
                }
            
            # Update current balance and add unrealized P&L
            dashboard_data["performance"]["balance"] = self.state.paper_balance
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.state.paper_positions.values())
            dashboard_data["performance"]["unrealized_pnl"] = unrealized_pnl
            dashboard_data["performance"]["open_positions"] = len(self.state.paper_positions)
            
            # Save updated data
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
                
            logger.debug(f"[DASHBOARD] Updated with {activity_type} activity")
            
        except Exception as e:
            logger.error(f"Error updating dashboard activity: {e}")

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
            # Get metrics (these are synchronous methods)
            performance_metrics = self.performance_monitor.get_performance_summary()
            risk_metrics = self.risk_manager.get_risk_metrics()

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
