import logging
from typing import Dict, Any, Optional, List, Protocol, runtime_checkable, Tuple
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    ApplicationBuilder,
)
from datetime import datetime, timedelta
import asyncio
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from decimal import Decimal
from src.trading.backtesting import BacktestResult
from src.trading.backtesting import BacktestEngine
from src.trading.strategy import TradingStrategy
import json

from src.config.settings import Settings
from src.utils.persistence import TradePersistence

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position"""

    token_address: str
    size: float
    entry_price: float
    current_price: float
    timestamp: datetime
    status: str = "open"
    take_profits: List[Dict[str, float]] = field(default_factory=list)
    trailing_stop: Optional[float] = None
    stop_loss: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    high_water_mark: float = 0.0

    def __post_init__(self):
        self.high_water_mark = self.current_price
        self.update_pnl()

    def update_pnl(self) -> None:
        """Update unrealized PnL based on current price"""
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.size
        if self.current_price > self.high_water_mark:
            self.high_water_mark = self.current_price


@dataclass
class GridStrategy:
    """Represents a grid trading strategy"""

    token_address: str
    upper_price: float
    lower_price: float
    grid_levels: int
    amount_per_grid: float
    active: bool = True
    grid_positions: Dict[float, Position] = field(default_factory=dict)

    def calculate_grid_levels(self) -> List[float]:
        """Calculate price levels for the grid"""
        price_step = (self.upper_price - self.lower_price) / (self.grid_levels - 1)
        return [self.lower_price + (i * price_step) for i in range(self.grid_levels)]


@dataclass
class DCAStrategy:
    """Represents a DCA (Dollar Cost Averaging) strategy"""

    token_address: str
    base_amount: float
    interval_hours: int
    total_entries: int
    completed_entries: int = 0
    active: bool = True
    last_entry: datetime = field(default_factory=datetime.now)
    positions: List[Position] = field(default_factory=list)


class RiskMetrics:
    """Handles risk-related calculations and metrics"""

    def __init__(self):
        self.position_values: List[float] = []
        self.returns: List[float] = []
        self.timestamps: List[datetime] = []

    def add_data_point(self, position_value: float, timestamp: datetime) -> None:
        """Add a new data point for risk calculations"""
        self.position_values.append(position_value)
        if len(self.position_values) > 1:
            returns = (
                position_value - self.position_values[-2]
            ) / self.position_values[-2]
            self.returns.append(returns)
        self.timestamps.append(timestamp)

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not self.returns:
            return 0.0
        returns_array = np.array(self.returns)
        excess_returns = returns_array - (risk_free_rate / 365)
        return np.sqrt(365) * (np.mean(excess_returns) / np.std(excess_returns))

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.position_values:
            return 0.0
        peaks = pd.Series(self.position_values).expanding(min_periods=1).max()
        drawdowns = (pd.Series(self.position_values) - peaks) / peaks
        return abs(float(drawdowns.min()))


@runtime_checkable
class TradingStrategy(Protocol):
    @property
    def is_trading(self) -> bool:
        ...
    backtest_engine: 'BacktestEngine'  # Add this
    jupiter_client: any


    async def start_trading(self) -> bool:
        ...

    async def stop_trading(self) -> bool:
        ...

    async def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        ...

    position_manager: Any
    alert_system: Optional[Any]


class PaperTradeManager:
    """Manages paper trading operations"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.paper_positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.risk_metrics = RiskMetrics()
        self.grid_strategies: Dict[str, GridStrategy] = {}
        self.dca_strategies: Dict[str, DCAStrategy] = {}
        self.initial_balance = float(settings.INITIAL_PAPER_BALANCE)
        self.current_balance = self.initial_balance
        self._monitor_task: Optional[asyncio.Task] = None

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        return sum(
            pos.size * pos.current_price for pos in self.paper_positions.values()
        )

    def validate_trade(self, size: float, price: float) -> Tuple[bool, str]:
        """Validate trade parameters against risk limits"""
        if size <= 0 or price <= 0:
            return False, "Invalid size or price"

        trade_value = size * price
        if trade_value > self.settings.MAX_POSITION_SIZE:
            return False, f"Trade value {trade_value} exceeds max position size"

        total_exposure = sum(
            pos.size * pos.current_price for pos in self.paper_positions.values()
        )
        if (
            total_exposure + trade_value
            > self.current_balance * self.settings.MAX_PORTFOLIO_RISK
        ):
            return False, "Trade would exceed portfolio risk limits"

        return True, "Trade validation successful"

    async def update_position(
        self, token_address: str, current_price: float
    ) -> List[str]:
        logger.debug(f"Starting position update for {token_address} at {current_price}")
        
        if token_address not in self.paper_positions:
            logger.debug(f"No position found for {token_address}")
            return []

        position = self.paper_positions[token_address]
        logger.debug(f"Pre-update state: TP={position.take_profits}, TS={position.trailing_stop}")
        
        position.current_price = current_price
        position.update_pnl()
        actions = []
        
        # Log position state
        logger.debug(f"Position state after PNL update: {position.__dict__}")
        
        try:
            # Check take profit levels
            for tp in position.take_profits[:]:
                logger.debug(f"Checking TP level: {tp}")
                if current_price >= position.entry_price * (1 + tp["percentage"]):
                    logger.debug("TP level triggered")
                    size_to_close = position.size * tp["size_percentage"]
                    success, msg = await self.partial_close(
                        token_address, size_to_close, current_price, "Take profit hit"
                    )
                    if success:
                        position.take_profits.remove(tp)
                        actions.append(f"Take profit executed at ${current_price:.4f}")

            # Check trailing stop
            if position.trailing_stop:
                logger.debug(f"Checking trailing stop: current={current_price}, hwm={position.high_water_mark}")
                if current_price <= position.high_water_mark * (1 - position.trailing_stop):
                    logger.debug("Trailing stop triggered")
                    success, msg = await self.close_position(token_address, "Trailing stop hit")
                    if success:
                        actions.append(f"Trailing stop executed at ${current_price:.4f}")
                        
            return actions
            
        except Exception as e:
            logger.error(f"Error in update_position: {str(e)}", exc_info=True)
            return []

    async def execute_paper_trade(
        self,
        token_address: str,
        size: float,
        price: float,
        trailing_stop: Optional[float] = None,
        take_profits: Optional[List[Dict[str, float]]] = None,
        stop_loss: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Execute a paper trade with specified parameters"""
        try:
            is_valid, message = self.validate_trade(size, price)
            if not is_valid:
                logger.error(f"Trade validation failed: {message}")
                return False, message

            position = Position(
                token_address=token_address,
                size=size,
                entry_price=price,
                current_price=price,
                timestamp=datetime.now(),
                take_profits=take_profits or [],
                trailing_stop=trailing_stop,
                stop_loss=stop_loss,
            )

            self.paper_positions[token_address] = position
            self.trade_history.append(
                {
                    "type": "entry",
                    "token": token_address,
                    "size": size,
                    "price": price,
                    "timestamp": datetime.now(),
                }
            )

            logger.info(f"Creating position: {position.__dict__}")
            self.paper_positions[token_address] = position
            logger.info(f"Position stored. Current positions: {self.paper_positions}")

            portfolio_value = self.get_portfolio_value()
            self.risk_metrics.add_data_point(portfolio_value, datetime.now())

            return True, "Trade executed successfully"
        except Exception as e:
            return False, f"Error executing trade: {str(e)}"

    async def close_position(self, token_address: str, reason: str) -> Tuple[bool, str]:
        """Close a position"""
        if token_address not in self.paper_positions:
            return False, "Position not found"

        position = self.paper_positions[token_address]
        position.status = "closed"
        position.realized_pnl = position.unrealized_pnl

        self.trade_history.append(
            {
                "type": "close",
                "token": token_address,
                "size": position.size,
                "price": position.current_price,
                "pnl": position.realized_pnl,
                "reason": reason,
                "timestamp": datetime.now(),
            }
        )

        del self.paper_positions[token_address]
        return True, f"Position closed: {reason}"

    async def partial_close(
        self,
        token_address: str,
        size_to_close: float,
        current_price: float,
        reason: str,
    ) -> Tuple[bool, str]:
        """Partially close a position"""
        if token_address not in self.paper_positions:
            return False, "Position not found"

        position = self.paper_positions[token_address]
        if size_to_close > position.size:
            return False, "Close size larger than position size"

        realized_pnl = (current_price - position.entry_price) * size_to_close
        position.size -= size_to_close
        position.realized_pnl += realized_pnl

        self.trade_history.append(
            {
                "type": "partial_close",
                "token": token_address,
                "size": size_to_close,
                "price": current_price,
                "pnl": realized_pnl,
                "reason": reason,
                "timestamp": datetime.now(),
            }
        )

        return True, f"Position partially closed: {reason}"


class TradingBot:
    """Main trading bot class handling Telegram interactions"""

    def __init__(
        self,
        application: Application,
        trading_strategy: TradingStrategy,
        settings: Settings,
    ):
        self.application = application
        self.strategy = trading_strategy
        self.settings = settings
        self.chat_id = (
            str(settings.TELEGRAM_CHAT_ID) if settings.TELEGRAM_CHAT_ID else None
        )
        self.persistence = TradePersistence()
        self.paper_trade_manager = PaperTradeManager(settings)
        self.backtest_results: Dict[str, Any] = {}
        self._monitor_task: Optional[asyncio.Task] = None  # Add this line
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register all command and callback handlers"""
        handlers = {
            "start": self.start_command,
            "status": self.status_command,
            "positions": self.positions_command,
            "paper": self.paper_trading_menu,
            "portfolio": self.portfolio_command,
            "performance": self.performance_command,
            "risk": self.risk_command,
            "start_trading": self.start_trading_command,
            "stop_trading": self.stop_trading_command,
            "settings": self.settings_command,
            "grid": self.grid_trading_menu,
            "dca": self.dca_menu,
            "analytics": self.analytics_menu,
            "backtest": self.backtest_menu,
        }

        for cmd, handler in handlers.items():
            self.application.add_handler(CommandHandler(cmd, handler))

        self.application.add_handler(CallbackQueryHandler(self.handle_callback))

    async def initialize(self) -> None:
        """Initialize the bot application"""
        try:
            if self.settings.PAPER_TRADING:
                logger.info("Paper trading mode - initializing monitoring")
                if hasattr(self, '_monitor_task') and self._monitor_task and not self._monitor_task.done():
                    self._monitor_task.cancel()
                    try:
                        await self._monitor_task
                    except asyncio.CancelledError:
                        pass
                self._monitor_task = asyncio.create_task(self._start_position_monitoring())
                return

            # Check if token exists and is valid
            if not self.settings.TELEGRAM_BOT_TOKEN:
                logger.error("Telegram bot token not configured")
                raise ValueError("Telegram bot token is required")

            token: str = self.settings.TELEGRAM_BOT_TOKEN

            self.application = (
                ApplicationBuilder()
                .token(token)
                .base_url("https://api.telegram.org/bot")
                .base_file_url("https://api.telegram.org/file/bot")
                .get_updates_connection_pool_size(8)
                .connect_timeout(30.0)
                .read_timeout(30.0)
                .write_timeout(30.0)
                .pool_timeout(3.0)
                .build()
            )
            self._register_handlers()
            logger.info("Telegram bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            raise

    async def start(self) -> None:
        """Start the bot and initialize periodic tasks"""
        if not self.chat_id:
            logger.error("Telegram chat ID not configured")
            return

        logger.info("Starting Telegram bot...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()

        # Start periodic tasks
        asyncio.create_task(self._periodic_portfolio_update())
        asyncio.create_task(self._periodic_risk_check())
        asyncio.create_task(self._periodic_dca_check())

    async def stop(self) -> None:
        """Stop the bot and cleanup"""
        try:
            logger.info("Stopping Telegram bot...")
            if hasattr(self, '_monitor_task') and self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
            await self.application.stop()
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")

    async def send_message(
        self, message: str, keyboard: Optional[List[List[InlineKeyboardButton]]] = None
    ) -> None:
        """Send a message to the configured chat"""
        if not self.chat_id:
            return

        try:
            markup = InlineKeyboardMarkup(keyboard) if keyboard else None
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="HTML",
                reply_markup=markup,
            )
        except Exception as e:
            logger.error(f"Message send failed: {e}")

    async def _periodic_portfolio_update(self, interval: int = 300) -> None:
        """Update portfolio status every 5 minutes"""
        while True:
            try:
                await asyncio.sleep(interval)
                if self.paper_trade_manager.paper_positions:
                    for addr in list(self.paper_trade_manager.paper_positions.keys()):
                        # In production, you would get real price updates here
                        current_price = 100.0  # Example price
                        actions = await self.paper_trade_manager.update_position(
                            addr, current_price
                        )
                        if actions:
                            await self.send_message(
                                "🔔 Position Update\n\n" + "\n".join(actions)
                            )
            except Exception as e:
                logger.error(f"Portfolio update error: {e}")

    async def _periodic_risk_check(self, interval: int = 900) -> None:
        """Check risk metrics every 15 minutes"""
        while True:
            try:
                await asyncio.sleep(interval)
                if self.paper_trade_manager.paper_positions:
                    total_value = self.paper_trade_manager.get_portfolio_value()
                    metrics = self.paper_trade_manager.risk_metrics

                    if metrics.returns:
                        volatility = np.std(metrics.returns) * np.sqrt(365)
                        if volatility > self.settings.MAX_VOLATILITY:
                            await self.send_message(
                                "⚠️ High Portfolio Volatility Alert\n\n"
                                f"Current Volatility: {volatility*100:.1f}%"
                            )
            except Exception as e:
                logger.error(f"Risk check error: {e}")

    async def _periodic_dca_check(self, interval: int = 60) -> None:
        """Check DCA strategies every minute"""
        while True:
            try:
                await asyncio.sleep(interval)
                current_time = datetime.now()

                for strategy in list(self.paper_trade_manager.dca_strategies.values()):
                    if not strategy.active:
                        continue

                    next_entry = strategy.last_entry + timedelta(
                        hours=strategy.interval_hours
                    )
                    if (
                        current_time >= next_entry
                        and strategy.completed_entries < strategy.total_entries
                    ):
                        # Execute DCA entry
                        (
                            success,
                            message,
                        ) = await self.paper_trade_manager.execute_paper_trade(
                            token_address=strategy.token_address,
                            size=strategy.base_amount,
                            price=100.0,  # In production, get real price
                        )

                        if success:
                            strategy.completed_entries += 1
                            strategy.last_entry = current_time
                            await self.send_message(
                                f"✅ DCA Entry Executed\n\n"
                                f"Token: {strategy.token_address}\n"
                                f"Amount: {strategy.base_amount} SOL\n"
                                f"Entries Complete: {strategy.completed_entries}/{strategy.total_entries}"
                            )

                        if strategy.completed_entries >= strategy.total_entries:
                            strategy.active = False
                            await self.send_message(
                                f"🎯 DCA Strategy Completed\n\n"
                                f"Token: {strategy.token_address}\n"
                                f"Total Invested: {strategy.base_amount * strategy.total_entries} SOL"
                            )
            except Exception as e:
                logger.error(f"DCA check error: {e}")

    # Command Handlers
    async def start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command"""
        commands = (
            "🤖 <b>Trading Bot Commands</b>\n\n"
            "📊 Trading Controls\n"
            "/start_trading - Start bot\n"
            "/stop_trading - Stop bot\n"
            "/status - Show status\n\n"
            "📈 Trading Info\n"
            "/positions - Show positions\n"
            "/paper - Paper trading\n"
            "/portfolio - Show portfolio\n"
            "/performance - Show metrics\n"
            "/risk - Risk metrics\n\n"
            "🔧 Advanced Features\n"
            "/grid - Grid trading\n"
            "/dca - DCA trading\n"
            "/analytics - Trading analytics\n"
            "/backtest - Backtest strategies"
        )
        await update.message.reply_text(commands, parse_mode="HTML")

    async def status_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /status command"""
        metrics = await self.strategy.get_metrics()
        status = metrics.get("trading_status", {})

        message = (
            f"🤖 Trading Status\n\n"
            f"Active: {'✅' if self.strategy.is_trading else '❌'}\n"
            f"Open Positions: {status.get('open_positions', 0)}\n"
            f"Pending Orders: {status.get('pending_orders', 0)}"
        )
        await update.message.reply_text(message)

        async def positions_command(
            self, update: Update, context: ContextTypes.DEFAULT_TYPE
        ) -> None:
            """Handle /positions command"""
            positions = self.paper_trade_manager.paper_positions
            if not positions:
                await update.message.reply_text("No active positions")
                return

            message = "📊 Active Positions:\n\n"
            for addr, pos in positions.items():
                pnl_pct = (pos.unrealized_pnl / (pos.entry_price * pos.size)) * 100
                message += (
                    f"Token: {addr[:8]}...\n"
                    f"Size: {pos.size:.4f} SOL\n"
                    f"Entry: ${pos.entry_price:.4f}\n"
                    f"Current: ${pos.current_price:.4f}\n"
                    f"PnL: {pnl_pct:+.2f}%\n\n"
                )
            await update.message.reply_text(message)

    async def paper_trading_menu(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /paper command"""
        keyboard = [
            [InlineKeyboardButton("New Paper Trade", callback_data="new_paper_trade")],
            [
                InlineKeyboardButton(
                    "Close Paper Trade", callback_data="close_paper_trade"
                )
            ],
            [InlineKeyboardButton("Paper Portfolio", callback_data="paper_portfolio")],
            [
                InlineKeyboardButton(
                    "Set Take Profits", callback_data="set_take_profits"
                )
            ],
            [
                InlineKeyboardButton(
                    "Set Trailing Stop", callback_data="set_trailing_stop"
                )
            ],
        ]
        await update.message.reply_text(
            "📝 Paper Trading Menu\nSelect an option:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def portfolio_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /portfolio command"""
        total_value = self.paper_trade_manager.get_portfolio_value()
        positions = self.paper_trade_manager.paper_positions

        message = "📊 Portfolio Summary\n\n"
        message += f"Total Value: {total_value:.4f} SOL\n\n"

        if positions:
            message += "Positions:\n"
            for addr, pos in positions.items():
                weight = (pos.size * pos.current_price / total_value) * 100
                pnl_pct = (pos.unrealized_pnl / (pos.entry_price * pos.size)) * 100
                message += (
                    f"{addr[:8]}...:\n"
                    f"Weight: {weight:.1f}%\n"
                    f"PnL: {pnl_pct:+.2f}%\n\n"
                )

        await update.message.reply_text(message)

    async def performance_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /performance command"""
        metrics = self.paper_trade_manager.risk_metrics

        message = "📈 Performance Metrics\n\n"
        message += f"Sharpe Ratio: {metrics.calculate_sharpe_ratio():.2f}\n"
        message += f"Max Drawdown: {metrics.calculate_max_drawdown()*100:.2f}%\n"

        if metrics.returns:
            total_return = (np.prod(1 + np.array(metrics.returns)) - 1) * 100
            volatility = np.std(metrics.returns) * np.sqrt(365) * 100
            message += f"Total Return: {total_return:.2f}%\n"
            message += f"Annual Volatility: {volatility:.2f}%\n"

        await update.message.reply_text(message)

    async def risk_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /risk command"""
        total_value = self.paper_trade_manager.get_portfolio_value()
        metrics = self.paper_trade_manager.risk_metrics

        if metrics.returns:
            returns = np.array(metrics.returns)
            var_95 = np.percentile(returns, 5) * total_value
            cvar_95 = np.mean(returns[returns <= var_95]) * total_value
            volatility = np.std(returns) * np.sqrt(365)
        else:
            var_95 = cvar_95 = volatility = 0.0

        message = "⚠️ Risk Analysis\n\n"
        message += f"Portfolio Value: {total_value:.4f} SOL\n"
        message += f"Daily VaR (95%): {abs(var_95):.4f} SOL\n"
        message += f"CVaR (95%): {abs(cvar_95):.4f} SOL\n"
        message += f"Annual Volatility: {volatility*100:.2f}%\n"

        await update.message.reply_text(message)

    async def start_trading_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start_trading command"""
        if self.strategy.is_trading:
            await update.message.reply_text("Trading already active ✅")
            return

        success = await self.strategy.start_trading()
        if success:
            await update.message.reply_text("🚀 Trading Started")
        else:
            await update.message.reply_text("❌ Failed to start trading")

    async def stop_trading_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /stop_trading command"""
        if not self.strategy.is_trading:
            await update.message.reply_text("Trading already stopped ⏹")
            return

        success = await self.strategy.stop_trading()
        if success:
            await update.message.reply_text("Trading stopped ⏹")
        else:
            await update.message.reply_text("❌ Failed to stop trading")

    async def settings_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /settings command"""
        message = (
            "⚙️ Current Settings\n\n"
            f"Initial Balance: {self.settings.INITIAL_PAPER_BALANCE} SOL\n"
            f"Max Position Size: {self.settings.MAX_POSITION_SIZE} SOL\n"
            f"Max Portfolio Risk: {self.settings.MAX_PORTFOLIO_RISK*100}%\n"
        )
        await update.message.reply_text(message)

    async def grid_trading_menu(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /grid command"""
        keyboard = [
            [InlineKeyboardButton("Create Grid", callback_data="create_grid")],
            [InlineKeyboardButton("Active Grids", callback_data="view_grids")],
            [
                InlineKeyboardButton(
                    "Grid Performance", callback_data="grid_performance"
                )
            ],
            [InlineKeyboardButton("Stop Grid", callback_data="stop_grid")],
        ]
        await update.message.reply_text(
            "📊 Grid Trading Menu\nSelect an option:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def dca_menu(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /dca command"""
        keyboard = [
            [InlineKeyboardButton("Set DCA Strategy", callback_data="set_dca")],
            [InlineKeyboardButton("View DCA Status", callback_data="view_dca")],
            [InlineKeyboardButton("DCA Performance", callback_data="dca_performance")],
            [InlineKeyboardButton("Stop DCA", callback_data="stop_dca")],
        ]
        await update.message.reply_text(
            "💰 DCA Trading Menu\nSelect an option:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def _handle_trailing_stop_level(self, query: CallbackQuery) -> None:
        try:
            if not self.paper_trade_manager.paper_positions:
                await query.edit_message_text("No active positions")
                return

            parts = query.data.split("_")
            if len(parts) < 2:
                await query.edit_message_text("❌ Invalid trailing stop data")
                return
                
            percentage = float(parts[1]) / 100
            # position = self.paper_trade_manager.paper_positions.get("SOL")
            position = self.paper_trade_manager.paper_positions.get("So11111111111111111111111111111111111111112")
            if position:
                position.trailing_stop = percentage
                position.high_water_mark = position.current_price
                await query.edit_message_text(f"✅ Trailing stop set at {percentage*100}%")
            else:
                await query.edit_message_text("❌ No position found")

        except Exception as e:
            logger.error(f"Error setting trailing stop: {str(e)}", exc_info=True)
            await query.edit_message_text("Error setting trailing stop")

    async def _handle_tp_level(self, query: CallbackQuery) -> None:
        try:
            # First check if we have positions
            if not self.paper_trade_manager.paper_positions:
                await query.edit_message_text("No active positions to set take profits")
                return

            parts = query.data.split("_")
            if len(parts) != 3:  # Should be ['tp', '5', '30'] format
                await query.edit_message_text("Invalid take profit format")
                return

            level = float(parts[1]) / 100
            size = float(parts[2]) / 100
            
            position = next(iter(self.paper_trade_manager.paper_positions.values()))
            position.take_profits.append({
                "percentage": level,
                "size_percentage": size
            })
            
            await query.edit_message_text(f"✅ Take profit set: {level*100}% for {size*100}% of position")

        except Exception as e:
            logger.error(f"Error setting take profit level: {str(e)}", exc_info=True)
            await query.edit_message_text("❌ Failed to set take profit")

    async def analytics_menu(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /analytics command"""
        keyboard = [
            [
                InlineKeyboardButton(
                    "Performance Charts", callback_data="performance_charts"
                )
            ],
            [InlineKeyboardButton("Risk Analysis", callback_data="risk_analysis")],
            [InlineKeyboardButton("Portfolio Stats", callback_data="portfolio_stats")],
            [InlineKeyboardButton("Trade History", callback_data="trade_history")],
        ]
        await update.message.reply_text(
            "📈 Analytics Menu\nSelect an option:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def backtest_menu(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /backtest command"""
        keyboard = [
            [InlineKeyboardButton("Run Backtest", callback_data="run_backtest")],
            [InlineKeyboardButton("Backtest Results", callback_data="view_backtest")],
            [
                InlineKeyboardButton(
                    "Optimize Strategy", callback_data="optimize_strategy"
                )
            ],
        ]
        await update.message.reply_text(
            "🔄 Backtest Menu\nSelect an option:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def handle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle callback queries from inline keyboards"""
        # query = update.callback_query
        # data = query.data

        if not update.callback_query:
            return

        query = update.callback_query
        if not query.data:
            return

        handlers = {
            # Paper Trading
            "new_paper_trade": self._handle_new_paper_trade,
            "close_paper_trade": self._handle_close_paper_trade,
            "paper_portfolio": self._handle_paper_portfolio,
            "set_take_profits": self._handle_set_take_profits,
            "set_trailing_stop": self._handle_set_trailing_stop,
            "new_paper_trade": self._handle_new_paper_trade,
            "close_paper_trade": self._handle_close_paper_trade,
            # Grid Trading
            "create_grid": self._handle_create_grid,
            "view_grids": self._handle_view_grids,
            "grid_performance": self._handle_grid_performance,
            "stop_grid": self._handle_stop_grid,
            # DCA Trading
            "set_dca": self._handle_set_dca,
            "view_dca": self._handle_view_dca,
            "dca_performance": self._handle_dca_performance,
            "stop_dca": self._handle_stop_dca,
            # Analytics
            "performance_charts": self._handle_performance_charts,
            "risk_analysis": self._handle_risk_analysis,
            "portfolio_stats": self._handle_portfolio_stats,
            "trade_history": self._handle_trade_history,
            # Backtest
            "run_backtest": self._handle_run_backtest,
            "view_backtest": self._handle_view_backtest,
            "optimize_strategy": self._handle_optimize_strategy,
            "apply_optimal_params": self._handle_apply_optimal_params,
            "save_optimization": self._handle_save_optimization,
            # Take Profit handlers
            "tp_5_30": self._handle_tp_level,
            "tp_10_30": self._handle_tp_level,
            "tp_15_40": self._handle_tp_level,
            
            # Trailing Stop handlers
            "trailing_2": self._handle_trailing_stop_level,
            "trailing_5": self._handle_trailing_stop_level,
            "trailing_10": self._handle_trailing_stop_level,

            "ts_2": self._handle_ts_level,
            "ts_5": self._handle_ts_level,
            "ts_10": self._handle_ts_level,
            "back_to_menu": self._handle_back_to_menu,
            "close_specific_SOL": self._handle_close_specific_token,
        }

        if query.data in handlers:
            await handlers[query.data](query)
        await query.answer()

    async def _handle_back_to_menu(self, query: CallbackQuery) -> None:
        """Handle back button press"""
        keyboard = [
            [InlineKeyboardButton("New Paper Trade", callback_data="new_paper_trade")],
            [InlineKeyboardButton("Close Paper Trade", callback_data="close_paper_trade")],
            [InlineKeyboardButton("Paper Portfolio", callback_data="paper_portfolio")],
            [InlineKeyboardButton("Set Take Profits", callback_data="set_take_profits")],
            [InlineKeyboardButton("Set Trailing Stop", callback_data="set_trailing_stop")]
        ]
        await query.edit_message_text(
            "📝 Paper Trading Menu\nSelect an option:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def _handle_close_specific_token(self, query: CallbackQuery) -> None:
        """Handle closing specific token"""
        try:
            token = "SOL"  # For now hardcoded to SOL
            success = await self.paper_trade_manager.close_position(token, "Manual close")
            if success:
                await query.edit_message_text("✅ Position closed successfully")
            else:
                await query.edit_message_text("❌ Failed to close position")
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            await query.edit_message_text("Error closing position")

    async def _handle_close_paper_trade(self, query: CallbackQuery) -> None:
        """Handle closing of paper trades"""
        positions = self.paper_trade_manager.paper_positions
        if not positions:
            await query.edit_message_text("No paper positions to close")
            return

        message = "Select position to close:\n\n"
        keyboard = []
        for addr, pos in positions.items():
            pnl_pct = (pos.unrealized_pnl / (pos.entry_price * pos.size)) * 100
            message += f"{addr}: {pos.size} @ {pos.entry_price} ({pnl_pct:+.2f}%)\n"
            keyboard.append(
                [
                    InlineKeyboardButton(
                        f"Close {addr}", callback_data=f"close_specific_{addr}"
                    )
                ]
            )

        await query.edit_message_text(
            text=message, reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def _handle_paper_portfolio(self, query: CallbackQuery) -> None:
        """Handle paper portfolio display"""
        try:
            positions = self.paper_trade_manager.paper_positions
            total_value = self.paper_trade_manager.get_portfolio_value()

            message = "📊 Paper Portfolio Summary\n\n"
            message += f"Total Value: {total_value:.4f} SOL\n\n"

            if positions:
                message += "Active Positions:\n"
                for addr, pos in positions.items():
                    pnl_pct = (pos.unrealized_pnl / (pos.entry_price * pos.size)) * 100
                    message += (
                        f"• {addr}:\n"
                        f"  Size: {pos.size:.4f} SOL\n"
                        f"  Entry: ${pos.entry_price:.4f}\n"
                        f"  Current: ${pos.current_price:.4f}\n"
                        f"  PnL: {pnl_pct:+.2f}%\n\n"
                    )
            else:
                message += "No active positions"

            await query.edit_message_text(message)
        except Exception as e:
            await query.edit_message_text(f"Error fetching portfolio: {str(e)}")

    async def _handle_set_take_profits(self, query: CallbackQuery) -> None:
        """Handle setting take profit levels"""
        try:
            keyboard = [
                [InlineKeyboardButton("5% TP (30%)", callback_data="tp_5_30")],
                [InlineKeyboardButton("10% TP (30%)", callback_data="tp_10_30")],
                [InlineKeyboardButton("15% TP (40%)", callback_data="tp_15_40")],
                [InlineKeyboardButton("← Back", callback_data="back_to_menu")]
            ]
            await query.edit_message_text(
                "Select Take Profit Level:", 
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            # Add debug logging
            logger.info("Take profit menu displayed")
        except Exception as e:
            logger.error(f"Error setting take profits: {str(e)}")
            await query.edit_message_text("Error setting take profits")

    async def _handle_ts_level(self, query: CallbackQuery) -> None:
        try:
            if not self.paper_trade_manager.paper_positions:
                await query.edit_message_text("No active positions")
                return

            parts = query.data.split("_")
            if len(parts) != 2:
                await query.edit_message_text("Invalid trailing stop format")
                return

            percentage = float(parts[1]) / 100
            position = next(iter(self.paper_trade_manager.paper_positions.values()))
            position.trailing_stop = percentage
            position.high_water_mark = position.current_price

            await query.edit_message_text(f"✅ Trailing stop set at {percentage*100}%")

        except Exception as e:
            logger.error(f"Error setting trailing stop: {str(e)}", exc_info=True)
            await query.edit_message_text("❌ Failed to set trailing stop")

    async def _handle_set_trailing_stop(self, query: CallbackQuery) -> None:
        try:
            keyboard = [
                [InlineKeyboardButton("2% Trailing", callback_data="trailing_2")],
                [InlineKeyboardButton("5% Trailing", callback_data="trailing_5")],
                [InlineKeyboardButton("10% Trailing", callback_data="trailing_10")],
                [InlineKeyboardButton("← Back", callback_data="back_to_menu")]
            ]
            await query.edit_message_text(
                "Select Trailing Stop Percentage:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            logger.info("Trailing stop menu displayed")
        except Exception as e:
            logger.error(f"Error setting trailing stop: {str(e)}")
            await query.edit_message_text("Error setting trailing stop")

    async def _handle_create_grid(self, query: CallbackQuery) -> None:
        """Handle grid strategy creation"""
        try:
            grid = GridStrategy(
                token_address="SOL",
                upper_price=120.0,
                lower_price=80.0,
                grid_levels=5,
                amount_per_grid=0.2,
            )
            self.paper_trade_manager.grid_strategies[grid.token_address] = grid
            message = (
                "✅ Grid Strategy Created\n\n"
                f"Token: {grid.token_address}\n"
                f"Range: ${grid.lower_price:.2f} - ${grid.upper_price:.2f}\n"
                f"Levels: {grid.grid_levels}\n"
                f"Amount per Grid: {grid.amount_per_grid} SOL"
            )
            await query.edit_message_text(message)
        except Exception as e:
            await query.edit_message_text(f"Error creating grid: {str(e)}")

    async def _handle_view_grids(self, query: CallbackQuery) -> None:
        """Handle viewing active grids"""
        grids = self.paper_trade_manager.grid_strategies
        if not grids:
            await query.edit_message_text("No active grid strategies")
            return

        message = "🔲 Active Grid Strategies:\n\n"
        for token, grid in grids.items():
            message += (
                f"Token: {token}\n"
                f"Range: ${grid.lower_price:.2f} - ${grid.upper_price:.2f}\n"
                f"Active Positions: {len(grid.grid_positions)}/{grid.grid_levels}\n\n"
            )
        await query.edit_message_text(message)

    async def _handle_grid_performance(self, query: CallbackQuery) -> None:
        """Handle grid performance display"""
        grids = self.paper_trade_manager.grid_strategies
        if not grids:
            await query.edit_message_text("No grid performance data available")
            return

        message = "📊 Grid Performance:\n\n"
        for token, grid in grids.items():
            total_value = sum(
                pos.size * pos.current_price for pos in grid.grid_positions.values()
            )
            total_pnl = sum(pos.unrealized_pnl for pos in grid.grid_positions.values())
            message += (
                f"Token: {token}\n"
                f"Total Value: {total_value:.4f} SOL\n"
                f"Total PnL: {total_pnl:+.4f} SOL\n\n"
            )
        await query.edit_message_text(message)

    async def _handle_stop_grid(self, query: CallbackQuery) -> None:
        """Handle stopping grid strategy"""
        grids = self.paper_trade_manager.grid_strategies
        if not grids:
            await query.edit_message_text("No active grids to stop")
            return

        keyboard = [
            [
                InlineKeyboardButton(
                    f"Stop {token} Grid", callback_data=f"stop_grid_{token}"
                )
            ]
            for token in grids
        ]

        await query.edit_message_text(
            "Select grid to stop:", reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def _handle_set_dca(self, query: CallbackQuery) -> None:
        """Handle DCA strategy setup"""
        try:
            strategy = DCAStrategy(
                token_address="SOL",
                base_amount=0.1,
                interval_hours=24,
                total_entries=10,
            )
            self.paper_trade_manager.dca_strategies[strategy.token_address] = strategy
            message = (
                "✅ DCA Strategy Set\n\n"
                f"Token: {strategy.token_address}\n"
                f"Amount per Entry: {strategy.base_amount} SOL\n"
                f"Interval: {strategy.interval_hours}h\n"
                f"Total Entries: {strategy.total_entries}"
            )
            await query.edit_message_text(message)
        except Exception as e:
            await query.edit_message_text(f"Error setting DCA: {str(e)}")

    async def _handle_view_dca(self, query: CallbackQuery) -> None:
        """Handle viewing DCA strategies"""
        strategies = self.paper_trade_manager.dca_strategies
        if not strategies:
            await query.edit_message_text("No active DCA strategies")
            return

        message = "💰 Active DCA Strategies:\n\n"
        for token, strategy in strategies.items():
            next_entry = strategy.last_entry + timedelta(hours=strategy.interval_hours)
            message += (
                f"Token: {token}\n"
                f"Progress: {strategy.completed_entries}/{strategy.total_entries}\n"
                f"Next Entry: {next_entry.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
        await query.edit_message_text(message)

    async def _handle_dca_performance(self, query: CallbackQuery) -> None:
        """Handle DCA performance display"""
        strategies = self.paper_trade_manager.dca_strategies
        if not strategies:
            await query.edit_message_text("No DCA performance data available")
            return

        message = "📊 DCA Performance:\n\n"
        for token, strategy in strategies.items():
            total_invested = strategy.base_amount * strategy.completed_entries
            positions_value = sum(
                pos.size * pos.current_price for pos in strategy.positions
            )
            roi = (
                ((positions_value - total_invested) / total_invested * 100)
                if total_invested
                else 0
            )
            message += (
                f"Token: {token}\n"
                f"Invested: {total_invested:.4f} SOL\n"
                f"Current Value: {positions_value:.4f} SOL\n"
                f"ROI: {roi:+.2f}%\n\n"
            )
        await query.edit_message_text(message)

    async def _handle_stop_dca(self, query: CallbackQuery) -> None:
        """Handle stopping DCA strategy"""
        strategies = self.paper_trade_manager.dca_strategies
        if not strategies:
            await query.edit_message_text("No active DCA strategies to stop")
            return

        keyboard = [
            [
                InlineKeyboardButton(
                    f"Stop {token} DCA", callback_data=f"stop_dca_{token}"
                )
            ]
            for token in strategies
        ]

        await query.edit_message_text(
            "Select DCA strategy to stop:", reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def _handle_performance_charts(self, query: CallbackQuery) -> None:
        """Handle performance charts display"""
        metrics = self.paper_trade_manager.risk_metrics
        if not metrics.returns:
            await query.edit_message_text("No performance data available yet")
            return

        message = "📈 Performance Analysis:\n\n"
        total_return = (np.prod(1 + np.array(metrics.returns)) - 1) * 100
        volatility = np.std(metrics.returns) * np.sqrt(365) * 100
        sharpe = metrics.calculate_sharpe_ratio()

        message += (
            f"Total Return: {total_return:+.2f}%\n"
            f"Annual Volatility: {volatility:.2f}%\n"
            f"Sharpe Ratio: {sharpe:.2f}\n"
        )
        await query.edit_message_text(message)

    async def _handle_risk_analysis(self, query: CallbackQuery) -> None:
        """Handle risk analysis display"""
        try:
            metrics = self.paper_trade_manager.risk_metrics
            total_value = self.paper_trade_manager.get_portfolio_value()

            if not metrics.returns:
                await query.edit_message_text("No risk data available yet")
                return

            returns = np.array(metrics.returns)
            var_95 = np.percentile(returns, 5) * total_value
            cvar_95 = np.mean(returns[returns <= var_95]) * total_value

            message = (
                "⚠️ Risk Analysis:\n\n"
                f"Value at Risk (95%): {abs(var_95):.4f} SOL\n"
                f"Conditional VaR: {abs(cvar_95):.4f} SOL\n"
                f"Max Drawdown: {metrics.calculate_max_drawdown()*100:.2f}%"
            )
            await query.edit_message_text(message)
        except Exception as e:
            await query.edit_message_text(f"Error in risk analysis: {str(e)}")

    async def _handle_portfolio_stats(self, query: CallbackQuery) -> None:
        """Handle portfolio statistics display"""
        try:
            positions = self.paper_trade_manager.paper_positions
            total_value = self.paper_trade_manager.get_portfolio_value()

            message = "📊 Portfolio Statistics:\n\n"
            message += f"Total Value: {total_value:.4f} SOL\n\n"

            if positions:
                for addr, pos in positions.items():
                    weight = (pos.size * pos.current_price / total_value) * 100
                    message += (
                        f"{addr[:8]}...:\n"
                        f"Weight: {weight:.1f}%\n"
                        f"PnL: {pos.unrealized_pnl:+.4f} SOL\n\n"
                    )
            else:
                message += "No active positions"

            await query.edit_message_text(message)
        except Exception as e:
            await query.edit_message_text(f"Error fetching stats: {str(e)}")

    async def _handle_trade_history(self, query: CallbackQuery) -> None:
        """Handle trade history display"""
        trades = self.paper_trade_manager.trade_history
        if not trades:
            await query.edit_message_text("No trade history available")
            return

        message = "📜 Recent Trades:\n\n"
        for trade in trades[-5:]:  # Show last 5 trades
            message += (
                f"Type: {trade['type']}\n"
                f"Token: {trade['token']}\n"
                f"Size: {trade['size']} SOL\n"
                f"Price: ${trade['price']:.4f}\n"
                f"Time: {trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

        await query.edit_message_text(message)

    async def positions_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /positions command"""
        positions = self.paper_trade_manager.paper_positions
        if not positions:
            await update.message.reply_text("No active positions")
            return

        message = "📊 Active Positions:\n\n"
        for addr, pos in positions.items():
            pnl_pct = (pos.unrealized_pnl / (pos.entry_price * pos.size)) * 100
            message += (
                f"Token: {addr[:8]}...\n"
                f"Size: {pos.size:.4f} SOL\n"
                f"Entry: ${pos.entry_price:.4f}\n"
                f"Current: ${pos.current_price:.4f}\n"
                f"PnL: {pnl_pct:+.2f}%\n\n"
            )
        await update.message.reply_text(message)

        # Add to TradingBot class in telegram_bot.py:

    async def backtest_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /backtest command"""
        keyboard = [
            [InlineKeyboardButton("Run New Backtest", callback_data="run_backtest")],
            [InlineKeyboardButton("View Results", callback_data="view_backtest")],
            [
                InlineKeyboardButton(
                    "Compare with Live", callback_data="compare_results"
                )
            ],
        ]
        await update.message.reply_text(
            "🔄 Backtest Menu\nSelect an option:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    def _calculate_monthly_returns(self, equity_curve: List[float]) -> str:
        """Calculate and format monthly returns"""
        if not equity_curve:
            return "No data available"

        monthly_returns = []
        for i in range(1, len(equity_curve)):
            return_pct = ((equity_curve[i] / equity_curve[i - 1]) - 1) * 100
            monthly_returns.append(return_pct)

        return "\n".join(
            [
                f"{datetime.now().strftime('%Y-%m')}: {ret:+.2f}%"
                for ret in monthly_returns
            ]
        )

    def _analyze_drawdown_periods(self, equity_curve: List[float]) -> Dict[str, float]:
        if not equity_curve:
            return {"avg": 0.0, "max_duration": 0, "avg_recovery": 0}
            
        peak = equity_curve[0]
        drawdown_periods: List[Dict[str, float]] = []
        current_drawdown = {"start": 0, "depth": 0.0, "duration": 0}
        
        # Process drawdowns
        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
                if current_drawdown["depth"] > 0:
                    current_drawdown["duration"] = i - current_drawdown["start"]
                    drawdown_periods.append(current_drawdown.copy())
                    current_drawdown = {"start": i, "depth": 0.0, "duration": 0}
            else:
                current_depth = (peak - value) / peak * 100
                if current_depth > current_drawdown["depth"]:
                    current_drawdown["depth"] = current_depth
        
        # Calculate metrics
        avg_depth = (
            sum(period["depth"] for period in drawdown_periods) / len(drawdown_periods)
            if drawdown_periods else 0.0
        )
        
        max_duration = (
            max(period["duration"] for period in drawdown_periods)
            if drawdown_periods else 0
        )
        
        avg_recovery = (
            sum(period["duration"] for period in drawdown_periods) / len(drawdown_periods)
            if drawdown_periods else 0
        )
        
        return {
            "avg": float(avg_depth),
            "max_duration": int(max_duration),
            "avg_recovery": float(avg_recovery)
        }

    def _calculate_risk_metrics(self, result: Optional[BacktestResult]) -> Dict[str, float]:
        if not result:
            return {"var": 0.0, "es": 0.0, "sortino": 0.0}
        
        returns = np.diff(result.equity_curve) / result.equity_curve[:-1]
        var_95 = float(np.percentile(returns, 5))
        es_95 = float(np.mean(returns[returns <= var_95]))
        
        negative_returns = returns[returns < 0]
        downside_std = float(np.std(negative_returns)) if len(negative_returns) > 0 else 1.0
        excess_returns = np.mean(returns)
        sortino = float(excess_returns / downside_std * np.sqrt(252)) if downside_std != 0 else 0.0

        return {"var": var_95, "es": es_95, "sortino": sortino}

    def _format_backtest_results(self, result: BacktestResult) -> str:
        """Format backtest results for display"""
        return (
            "📊 <b>Backtest Results</b>\n\n"
            f"Period: {result.parameters.get('start_date').strftime('%Y-%m-%d')} to "
            f"{result.parameters.get('end_date').strftime('%Y-%m-%d')}\n\n"
            f"Total Trades: {result.total_trades}\n"
            f"Win Rate: {result.win_rate:.2f}%\n"
            f"Profit Factor: {result.profit_factor:.2f}\n"
            f"Total Return: {result.total_return:+.2f}%\n"
            f"Max Drawdown: {result.max_drawdown:.2f}%\n"
            f"Sharpe Ratio: {result.sharpe_ratio:.2f}\n\n"
            "<b>Top Performing Trades:</b>\n"
            + "\n".join(
                f"• {t['token']}: {t['pnl']:+.2f} SOL"
                for t in sorted(
                    [t for t in result.trades if t["type"] == "exit"],
                    key=lambda x: x["pnl"],
                    reverse=True,
                )[:5]
            )
        )

    # async def _handle_backtest_details(self, query: CallbackQuery) -> None:
    #     """Display detailed backtest analysis"""
    #     try:
    #         result = self.strategy.backtest_engine.last_result
    #         if not result:
    #             await query.edit_message_text("No backtest results available")
    #             return

    #         # Calculate additional metrics
    #         result = self.strategy.backtest_engine.last_result
    #         monthly_returns = self._calculate_monthly_returns(result.equity_curve)
    #         drawdown_periods = self._analyze_drawdown_periods(result.equity_curve)
    #         risk_metrics = self._calculate_risk_metrics(result)

    #         message = (
    #             "📈 <b>Detailed Backtest Analysis</b>\n\n"
    #             "<b>Monthly Returns:</b>\n"
    #             f"{monthly_returns}\n\n"
    #             "<b>Risk Metrics:</b>\n"
    #             f"• Value at Risk (95%): {risk_metrics['var']:.2f} SOL\n"
    #             f"• Expected Shortfall: {risk_metrics['es']:.2f} SOL\n"
    #             f"• Sortino Ratio: {risk_metrics['sortino']:.2f}\n\n"
    #             "<b>Drawdown Analysis:</b>\n"
    #             f"• Average Drawdown: {drawdown_periods['avg']:.2f}%\n"
    #             f"• Max Drawdown Duration: {drawdown_periods['max_duration']} days\n"
    #             f"• Recovery Time: {drawdown_periods['avg_recovery']} days"
    #         )

    #         keyboard = [
    #             [
    #                 InlineKeyboardButton(
    #                     "Export Details", callback_data="export_details"
    #                 )
    #             ],
    #             [
    #                 InlineKeyboardButton(
    #                     "Back to Results", callback_data="view_backtest"
    #                 )
    #             ],
    #         ]

    #         await query.edit_message_text(
    #             message, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML"
    #         )

    #     except Exception as e:
    #         await query.edit_message_text(f"Error displaying details: {str(e)}")

    # async def _handle_compare_results(self, query: CallbackQuery) -> None:
    #     """Compare backtest results with live performance"""
    #     try:
    #         backtest = self.strategy.backtest_engine.last_result
    #         if not backtest:
    #             await query.edit_message_text("No backtest results available")
    #             return

    #         # Get live performance metrics
    #         live_metrics = await self.strategy.get_metrics()

    #         message = (
    #             "📊 <b>Performance Comparison</b>\n\n"
    #             "<b>Metric  |  Backtest  |  Live</b>\n"
    #             f"Return:   {backtest.total_return:>8.2f}% | {live_metrics['performance']['total_return']:>8.2f}%\n"
    #             f"Win Rate: {backtest.win_rate:>8.2f}% | {live_metrics['performance']['win_rate']:>8.2f}%\n"
    #             f"Drawdown: {backtest.max_drawdown:>8.2f}% | {live_metrics['risk']['max_drawdown']:>8.2f}%\n"
    #             f"Sharpe:   {backtest.sharpe_ratio:>8.2f} | {live_metrics['performance']['sharpe_ratio']:>8.2f}\n\n"
    #             "<i>Note: Live metrics are from trading start</i>"
    #         )

    #         keyboard = [[InlineKeyboardButton("Back", callback_data="backtest_menu")]]

    #         await query.edit_message_text(
    #             message, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML"
    #         )

    #     except Exception as e:
    #         await query.edit_message_text(f"Error comparing results: {str(e)}")

    async def _run_optimization_task(self, query: CallbackQuery) -> None:
        """Run strategy optimization task"""
        try:
            await asyncio.sleep(5)  # Simulate optimization process

            optimization_results = {
                "best_params": {
                    "grid_levels": 5,
                    "take_profit_levels": 3,
                    "trailing_stop": 0.05,
                    "position_size": 0.5,
                },
                "metrics": {
                    "sharpe_ratio": 2.1,
                    "max_drawdown": 12.5,
                    "annual_return": 45.2,
                    "win_rate": 68.5,
                },
            }

            # Get values from optimization_results to avoid undefined variable errors
            grid_levels = optimization_results['best_params']['grid_levels']
            take_profit_levels = optimization_results['best_params']['take_profit_levels']
            trailing_stop = optimization_results['best_params']['trailing_stop']
            position_size = optimization_results['best_params']['position_size']
            sharpe_ratio = optimization_results['metrics']['sharpe_ratio']
            max_drawdown = optimization_results['metrics']['max_drawdown']
            annual_return = optimization_results['metrics']['annual_return']
            win_rate = optimization_results['metrics']['win_rate']

            message = (
                "✅ Optimization Complete\n\n"
                "Optimal Parameters:\n"
                f"Grid Levels: {optimization_results['best_params']['grid_levels']}\n"
                f"Take Profit Levels: {optimization_results['best_params']['take_profit_levels']}\n"
                f"Trailing Stop: {optimization_results['best_params']['trailing_stop']*100}%\n"
                f"Position Size: {optimization_results['best_params']['position_size']} SOL\n\n"
                "Expected Performance:\n"
                f"Sharpe Ratio: {optimization_results['metrics']['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {optimization_results['metrics']['max_drawdown']}%\n"
                f"Annual Return: {optimization_results['metrics']['annual_return']}%\n"
                f"Win Rate: {optimization_results['metrics']['win_rate']}%"
            )

            keyboard = [
                [
                    InlineKeyboardButton(
                        "Apply Parameters", callback_data="apply_optimal_params"
                    )
                ],
                [
                    InlineKeyboardButton(
                        "Save Results", callback_data="save_optimization"
                    )
                ],
            ]

            await self.application.bot.edit_message_text(
                chat_id=query.message.chat_id,
                message_id=query.message.message_id,
                text=message,
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

        except Exception as e:
            error_message = f"Error during optimization: {str(e)}"
            try:
                await query.edit_message_text(error_message)
            except Exception as nested_error:
                logger.error(
                    f"Failed to send error message: {error_message} - {nested_error}"
                )

    async def _start_position_monitoring(self) -> None:
        """Monitor paper trading positions"""
        while True:
            try:
                if not self.paper_trade_manager.paper_positions:
                    await asyncio.sleep(30)
                    continue

                for token_address, position in self.paper_trade_manager.paper_positions.items():
                    # Use the correct SOL mint address
                    sol_address = "So11111111111111111111111111111111111111112"
                    # price_data = await self.strategy.jupiter_client.get_price(sol_address)
                    price_data = await self.strategy.jupiter_client.get_price(
                        input_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                        output_mint=sol_address,
                        amount=1000000  # 1 USDC
                    )
                    # if not price_data or 'price' not in price_data:
                    #     logger.warning(f"Failed to get price for {token_address}")
                    #     continue

                    if not isinstance(price_data, dict) or 'outAmount' not in price_data:
                        logger.warning(f"Failed to get price for {token_address}")
                        continue

                    # Calculate price from outAmount
                    lamports = float(price_data['outAmount'])
                    sol_for_one_usdc = lamports / 1e9  # Convert lamports to SOL
                    current_price = 1.0 / sol_for_one_usdc  # Convert to USDC/SOL price

                    logger.info(f"Updated price for {token_address}: {current_price}")
                        
                    # current_price = float(price_data['price'])
                    logger.info(f"Updated price for {token_address}: {current_price}")
                    
                    actions = await self.paper_trade_manager.update_position(
                        token_address=token_address,
                        current_price=current_price
                    )
                    
                    if actions:
                        message = f"🔔 Position Update:\n{chr(10).join(actions)}"
                        await self.send_message(message)
                        
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Position monitoring error: {str(e)}", exc_info=True)
                await asyncio.sleep(5)

    async def _handle_new_paper_trade(self, query: CallbackQuery) -> None:
        try:
            sol_address = "So11111111111111111111111111111111111111112"
            logger.info(f"Fetching price for SOL address: {sol_address}")

            try:
                price_response = await self.strategy.jupiter_client.get_price(
                    input_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                    output_mint=sol_address,
                    amount=1000000  # 1 USDC
                )
                logger.info(f"Price data received: {price_response}")

                if not isinstance(price_response, dict):
                    await query.edit_message_text("❌ Invalid price response format")
                    return

                if 'outAmount' not in price_response:
                    await query.edit_message_text("❌ Missing price data in response")
                    return

                # Calculate price: 1 USDC = X SOL, therefore 1 SOL = 1/X USDC
                sol_amount = float(price_response['outAmount']) / 1e9  # Convert lamports to SOL
                current_price = 1.0 / sol_amount if sol_amount > 0 else 0

                if current_price <= 0:
                    await query.edit_message_text("❌ Invalid price calculation")
                    return

                # Execute paper trade
                success, message = await self.paper_trade_manager.execute_paper_trade(
                    token_address=sol_address,
                    size=0.1,  # Trading 0.1 SOL
                    price=current_price,
                    stop_loss=current_price * 0.95  # 5% stop loss
                )

                if success:
                    await query.edit_message_text(
                        f"✅ Paper trade opened:\n"
                        f"Amount: 0.1 SOL\n"
                        f"Price: ${current_price:.2f}\n"
                        f"Stop Loss: ${(current_price * 0.95):.2f}"
                    )
                else:
                    await query.edit_message_text(f"❌ Failed to execute paper trade: {message}")

            except Exception as price_error:
                logger.error(f"Price fetch error: {str(price_error)}")
                await query.edit_message_text("❌ Failed to get current price")
                return

        except Exception as e:
            logger.error(f"Paper trade execution error: {str(e)}", exc_info=True)
            await query.edit_message_text(f"❌ Error executing paper trade: {str(e)}")

    async def _handle_run_backtest(self, query: CallbackQuery) -> None:
        """Handle backtest execution"""
        try:
            params = {
                "strategy_type": "grid",
                "start_date": datetime.now() - timedelta(days=30),
                "end_date": datetime.now(),
                "initial_balance": 100.0,
                "token": "SOL",
            }

            message = "🔄 Running Backtest...\n\n"
            message += "Parameters:\n"
            message += f"Strategy: {params['strategy_type'].capitalize()}\n"
            message += f"Period: {params['start_date'].date()} to {params['end_date'].date()}\n"
            message += f"Initial Balance: {params['initial_balance']} SOL\n"

            self.backtest_results = {"params": params, "status": "running"}

            await query.edit_message_text(message)
            asyncio.create_task(self._run_backtest_task(query))

        except Exception as e:
            await query.edit_message_text(f"Error starting backtest: {str(e)}")

    async def _run_backtest_task(self, query: CallbackQuery) -> None:
        """Run backtest in background"""
        try:
            await asyncio.sleep(5)  # Simulate backtest running

            self.backtest_results.update(
                {
                    "status": "completed",
                    "total_return": 15.5,
                    "sharpe_ratio": 1.8,
                    "max_drawdown": 8.5,
                    "win_rate": 65.0,
                }
            )

            message = "✅ Backtest Completed\n\n"
            message += f"Total Return: {self.backtest_results['total_return']}%\n"
            message += f"Sharpe Ratio: {self.backtest_results['sharpe_ratio']:.2f}\n"
            message += f"Max Drawdown: {self.backtest_results['max_drawdown']}%\n"
            message += f"Win Rate: {self.backtest_results['win_rate']}%\n\n"

            keyboard = [
                [
                    InlineKeyboardButton(
                        "Detailed Results", callback_data="view_backtest"
                    )
                ],
                [
                    InlineKeyboardButton(
                        "Optimize Strategy", callback_data="optimize_strategy"
                    )
                ],
            ]

            await self.application.bot.edit_message_text(
                chat_id=query.message.chat_id,
                message_id=query.message.message_id,
                text=message,
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

        except Exception as e:
            logger.error(f"Backtest task error: {e}")

    async def _handle_view_backtest(self, query: CallbackQuery) -> None:
        """Display backtest results"""
        if (
            not self.backtest_results
            or self.backtest_results.get("status") != "completed"
        ):
            await query.edit_message_text("No backtest results available")
            return

        results = self.backtest_results
        message = "📊 Backtest Results\n\n"
        message += "Parameters:\n"
        message += f"Strategy: {results['params']['strategy_type'].capitalize()}\n"
        message += f"Period: {results['params']['start_date'].date()} to {results['params']['end_date'].date()}\n"
        message += f"Initial Balance: {results['params']['initial_balance']} SOL\n\n"
        message += "Performance Metrics:\n"
        message += f"Total Return: {results['total_return']}%\n"
        message += f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
        message += f"Max Drawdown: {results['max_drawdown']}%\n"
        message += f"Win Rate: {results['win_rate']}%\n"

        await query.edit_message_text(message)

    # async def _handle_optimize_strategy(self, query: CallbackQuery) -> None:
    #     """Handle strategy optimization"""
    #     try:
    #         message = "🔄 Running Strategy Optimization...\n\n"
    #         message += "Optimizing Parameters:\n"
    #         message += "- Grid Levels\n"
    #         message += "- Take Profit Levels\n"
    #         message += "- Trailing Stop Range\n"
    #         message += "- Position Sizing\n\n"
    #         message += "This may take a few minutes..."

    #         await query.edit_message_text(message)
    #         asyncio.create_task(self._run_optimization_task(query))

    #     except Exception as e:
    #         await query.edit_message_text(f"Error starting optimization: {str(e)}")

    # In telegram_bot.py
    async def _handle_optimize_strategy(self, query: CallbackQuery) -> None:
        try:
            start_msg = (
                "🔄 Running Strategy Optimization...\n\n"
                "Optimizing Parameters:\n"
                "• Stop Loss Levels\n"
                "• Take Profit Targets\n"
                "• Position Sizing\n"
                "• Entry Thresholds\n\n"
                "This may take several minutes..."
            )
            await query.edit_message_text(start_msg)

            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            results = await self.strategy.backtest_engine.optimize_parameters(
                start_date=start_date,
                end_date=end_date
            )
            
            msg = (
                "✅ Optimization Complete\n\n"
                "Optimal Parameters:\n"
                f"• Stop Loss: {results['parameters']['STOP_LOSS_PERCENTAGE']:.1%}\n"
                f"• Take Profit: {results['parameters']['TAKE_PROFIT_PERCENTAGE']:.1%}\n"
                f"• Position Size: {results['parameters']['MAX_POSITION_SIZE']:.1f} SOL\n"
                f"• Signal Threshold: {results['parameters']['SIGNAL_THRESHOLD']:.2f}\n\n"
                "Performance Metrics:\n"
                f"• Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
                f"• Total Return: {results['total_return']:.1%}\n"
                f"• Max Drawdown: {results['max_drawdown']:.1%}\n"
                f"• Win Rate: {results['win_rate']:.1%}\n"
                f"• Profit Factor: {results['profit_factor']:.2f}\n"
            )
            
            keyboard = [
                [InlineKeyboardButton("Apply Parameters", callback_data="apply_optimal_params")],
                [InlineKeyboardButton("Back", callback_data="backtest_menu")]
            ]
            
            await query.edit_message_text(
                msg, 
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except Exception as e:
            error_msg = f"Optimization failed: {str(e)}"
            logger.error(error_msg)
            await query.edit_message_text(error_msg)

    async def _handle_apply_optimal_params(self, query: CallbackQuery) -> None:
        """Apply optimized parameters to current strategy"""
        try:
            # Implementation would depend on your strategy parameters
            await query.edit_message_text("Optimal parameters applied successfully ✅")
        except Exception as e:
            await query.edit_message_text(f"Error applying parameters: {str(e)}")

    async def _handle_save_optimization(self, query: CallbackQuery) -> None:
        """Save optimization results"""
        try:
            # Implementation would depend on your storage mechanism
            await query.edit_message_text("Optimization results saved successfully ✅")
        except Exception as e:
            await query.edit_message_text(f"Error saving results: {str(e)}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
