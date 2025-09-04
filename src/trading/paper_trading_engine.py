import asyncio
import sqlite3
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import math

from src.database.db_manager import DatabaseManager
from src.portfolio.portfolio_manager import PortfolioManager
from src.trading.risk_engine import RiskEngine, RiskEngineConfig, TradeRisk
from src.trading.trade_types import TradeType, TradeDirection
from src.monitoring.system_monitor import SystemMonitor
from src.arbitrage.real_dex_connector import RealDEXConnector
from src.backtesting.production_backtester import ProductionBacktester, ExecutionQuality
from src.analytics.performance_analytics import PerformanceAnalytics
from src.utils.trading_time import trading_time


class PaperTradingMode(Enum):
    SIMULATION = "simulation"  # Use simulated market data
    LIVE_DATA = "live_data"   # Use live market data, simulate execution
    HYBRID = "hybrid"         # Mix of live data and simulation


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class PaperPosition:
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    direction: TradeDirection
    open_time: datetime
    last_updated: datetime


@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    direction: TradeDirection
    order_type: TradeType
    quantity: float
    price: Optional[float]  # None for market orders
    status: OrderStatus
    filled_quantity: float
    avg_fill_price: float
    timestamp: datetime
    strategy_name: str
    metadata: Dict[str, Any]


@dataclass
class PaperAccount:
    account_id: str
    initial_balance: float
    current_balance: float
    equity: float
    margin_used: float
    free_margin: float
    total_pnl: float
    daily_pnl: float
    positions: Dict[str, PaperPosition]
    open_orders: List[PaperOrder]
    recent_orders: List[PaperOrder]
    trade_count: int
    win_rate: float
    last_updated: datetime


@dataclass
class LiveMarketData:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime
    source: str  # DEX name


class PaperTradingEngine:
    def __init__(
        self,
        db_manager: DatabaseManager,
        risk_engine: RiskEngine,
        monitor: SystemMonitor,
        analytics: Optional[PerformanceAnalytics] = None,
        real_dex_connector: Optional[RealDEXConnector] = None,
        mode: PaperTradingMode = PaperTradingMode.LIVE_DATA,
        initial_balance: float = 10000.0
    ):
        self.db_manager = db_manager
        self.risk_engine = risk_engine
        self.monitor = monitor
        self.analytics = analytics
        self.real_dex_connector = real_dex_connector
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        # Paper trading account
        self.account = PaperAccount(
            account_id=f"paper_{uuid.uuid4().hex[:8]}",
            initial_balance=initial_balance,
            current_balance=initial_balance,
            equity=initial_balance,
            margin_used=0.0,
            free_margin=initial_balance,
            total_pnl=0.0,
            daily_pnl=0.0,
            positions={},
            open_orders=[],
            recent_orders=[],
            trade_count=0,
            win_rate=0.0,
            last_updated=trading_time.now()
        )
        
        # Market data cache
        self.market_data_cache: Dict[str, LiveMarketData] = {}
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Execution simulation
        self.execution_quality = ExecutionQuality.REALISTIC
        self.slippage_factor = 0.002  # 0.2% default slippage
        self.fee_rate = 0.001  # 0.1% transaction fee
        
        # Trading state
        self.is_trading = False
        self.trading_sessions: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Strategy integration
        self.active_strategies: Set[str] = set()
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        
    async def initialize(self):
        """Initialize paper trading engine"""
        try:
            # Create paper trading tables
            await self._create_paper_trading_tables()
            
            # Initialize market data feeds
            if self.mode in [PaperTradingMode.LIVE_DATA, PaperTradingMode.HYBRID]:
                if self.real_dex_connector:
                    await self.real_dex_connector.initialize()
                    
            # Start market data updates
            asyncio.create_task(self._market_data_update_loop())
            
            # Start position monitoring
            asyncio.create_task(self._position_monitoring_loop())
            
            self.logger.info(f"Paper trading engine initialized - Mode: {self.mode.value}, Balance: ${self.account.initial_balance}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize paper trading engine: {e}")
            raise
            
    async def _create_paper_trading_tables(self):
        """Create database tables for paper trading"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            # Paper trading accounts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT UNIQUE NOT NULL,
                    initial_balance REAL NOT NULL,
                    current_balance REAL NOT NULL,
                    equity REAL NOT NULL,
                    margin_used REAL NOT NULL,
                    free_margin REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    trade_count INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Paper positions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    avg_entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    direction TEXT NOT NULL,
                    open_time TEXT NOT NULL,
                    close_time TEXT,
                    status TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Paper orders
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    order_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    status TEXT NOT NULL,
                    filled_quantity REAL NOT NULL,
                    avg_fill_price REAL NOT NULL,
                    strategy_name TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Market data snapshots
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    last REAL NOT NULL,
                    volume REAL NOT NULL,
                    source TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to create paper trading tables: {e}")
            raise
            
    async def start_trading(self):
        """Start paper trading session"""
        if self.is_trading:
            self.logger.warning("Paper trading already active")
            return
            
        try:
            self.is_trading = True
            session_id = f"session_{trading_time.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create trading session record
            session_data = {
                "session_id": session_id,
                "start_time": trading_time.now().isoformat(),
                "mode": self.mode.value,
                "initial_balance": self.account.initial_balance,
                "execution_quality": self.execution_quality.value,
                "active_strategies": list(self.active_strategies)
            }
            self.trading_sessions.append(session_data)
            
            # Reset daily PnL
            self.account.daily_pnl = 0.0
            
            # Start trading loops
            asyncio.create_task(self._order_management_loop())
            asyncio.create_task(self._performance_monitoring_loop())
            
            self.logger.info(f"Paper trading session started: {session_id}")
            
            # Update monitoring
            if self.monitor:
                await self.monitor.log_system_event(
                    "paper_trading_start",
                    {"session_id": session_id, "mode": self.mode.value}
                )
                
        except Exception as e:
            self.logger.error(f"Failed to start paper trading: {e}")
            self.is_trading = False
            raise
            
    async def stop_trading(self):
        """Stop paper trading session"""
        if not self.is_trading:
            return
            
        try:
            self.is_trading = False
            
            # Close all open positions (optional)
            # await self._close_all_positions()
            
            # Cancel all pending orders
            await self._cancel_all_orders()
            
            # Calculate final session metrics
            session_metrics = await self._calculate_session_metrics()
            
            # Update last session
            if self.trading_sessions:
                self.trading_sessions[-1].update({
                    "end_time": trading_time.now().isoformat(),
                    "final_balance": self.account.current_balance,
                    "total_pnl": self.account.total_pnl,
                    "total_trades": self.account.trade_count,
                    "win_rate": self.account.win_rate,
                    "session_metrics": session_metrics
                })
                
            self.logger.info(f"Paper trading session stopped. Final PnL: ${self.account.total_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error stopping paper trading: {e}")
            
    async def place_order(
        self,
        symbol: str,
        direction: TradeDirection,
        order_type: TradeType,
        quantity: float,
        price: Optional[float] = None,
        strategy_name: str = "manual",
        metadata: Dict[str, Any] = None
    ) -> str:
        """Place a paper trading order"""
        if not self.is_trading:
            raise ValueError("Paper trading not active")
            
        try:
            # Generate order ID
            order_id = f"paper_{uuid.uuid4().hex[:12]}"
            
            # Risk check
            risk_check = await self._perform_risk_check(symbol, direction, quantity, price)
            if not risk_check["approved"]:
                self.logger.warning(f"Order rejected by risk engine: {risk_check['reason']}")
                return None
            
            # Order collision detection
            collision_check = await self._check_order_collision(symbol, direction, quantity, strategy_name)
            if collision_check["collision_detected"]:
                self.logger.warning(f"Order collision detected: {collision_check['reason']}")
                return None
                
            # Create order
            order = PaperOrder(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                order_type=order_type,
                quantity=quantity,
                price=price,
                status=OrderStatus.PENDING,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                timestamp=trading_time.now(),
                strategy_name=strategy_name,
                metadata=metadata or {}
            )
            
            # Add to open orders
            self.account.open_orders.append(order)
            
            # Store in database
            await self._store_order(order)
            
            # Attempt immediate fill for market orders
            if order_type == TradeType.MARKET:
                await self._try_fill_order(order)
                
            self.logger.info(f"Order placed: {order_id} - {direction.value} {quantity} {symbol}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return None
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        try:
            # Find order
            order = None
            for o in self.account.open_orders:
                if o.order_id == order_id:
                    order = o
                    break
                    
            if not order:
                self.logger.warning(f"Order not found: {order_id}")
                return False
                
            if order.status != OrderStatus.PENDING:
                self.logger.warning(f"Cannot cancel order {order_id}: status is {order.status}")
                return False
                
            # Update status
            order.status = OrderStatus.CANCELLED
            
            # Remove from open orders
            self.account.open_orders = [o for o in self.account.open_orders if o.order_id != order_id]
            
            # Update database
            await self._update_order(order)
            
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False
            
    async def get_position(self, symbol: str) -> Optional[PaperPosition]:
        """Get current position for symbol"""
        return self.account.positions.get(symbol)
        
    async def get_account_status(self) -> PaperAccount:
        """Get current account status"""
        # Update equity with current positions
        await self._update_account_equity()
        return self.account
        
    async def get_market_data(self, symbol: str) -> Optional[LiveMarketData]:
        """Get current market data for symbol"""
        return self.market_data_cache.get(symbol)
        
    async def _market_data_update_loop(self):
        """Continuously update market data"""
        while True:
            try:
                if self.mode in [PaperTradingMode.LIVE_DATA, PaperTradingMode.HYBRID]:
                    await self._update_live_market_data()
                else:
                    await self._simulate_market_data()
                    
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Market data update error: {e}")
                await asyncio.sleep(5)
                
    async def _update_live_market_data(self):
        """Update with live market data from DEX APIs"""
        if not self.real_dex_connector:
            return
            
        try:
            # Get current prices from DEX
            symbols = list(set([pos.symbol for pos in self.account.positions.values()] + 
                             [order.symbol for order in self.account.open_orders]))
            
            if not symbols:
                symbols = ["SOL/USDC", "BTC/USDC", "ETH/USDC"]  # Default symbols
                
            for symbol in symbols:
                try:
                    # Get price from real DEX connector
                    price_data = await self.real_dex_connector._get_token_price_from_dex(
                        "jupiter", symbol.split("/")[0], symbol.split("/")[1]
                    )
                    
                    if price_data:
                        market_data = LiveMarketData(
                            symbol=symbol,
                            bid=price_data * 0.999,  # Simulate bid-ask spread
                            ask=price_data * 1.001,
                            last=price_data,
                            volume=1000000.0,  # Mock volume
                            timestamp=trading_time.now(),
                            source="jupiter"
                        )
                        
                        self.market_data_cache[symbol] = market_data
                        
                        # Store price history
                        if symbol not in self.price_history:
                            self.price_history[symbol] = []
                        self.price_history[symbol].append((trading_time.now(), price_data))
                        
                        # Keep only last 1000 price points
                        if len(self.price_history[symbol]) > 1000:
                            self.price_history[symbol] = self.price_history[symbol][-1000:]
                            
                except Exception as e:
                    self.logger.debug(f"Could not get price for {symbol}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to update live market data: {e}")
            
    async def _simulate_market_data(self):
        """Simulate market data for testing"""
        symbols = ["SOL/USDC", "BTC/USDC", "ETH/USDC"]
        base_prices = {"SOL/USDC": 100.0, "BTC/USDC": 45000.0, "ETH/USDC": 3000.0}
        
        for symbol in symbols:
            # Get last price or use base
            if symbol in self.market_data_cache:
                last_price = self.market_data_cache[symbol].last
            else:
                last_price = base_prices.get(symbol, 100.0)
                
            # Simulate price movement (random walk)
            change = np.random.normal(0, 0.001)  # 0.1% volatility
            new_price = last_price * (1 + change)
            
            market_data = LiveMarketData(
                symbol=symbol,
                bid=new_price * 0.999,
                ask=new_price * 1.001,
                last=new_price,
                volume=100000.0,
                timestamp=trading_time.now(),
                source="simulation"
            )
            
            self.market_data_cache[symbol] = market_data
            
            # Store price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append((trading_time.now(), new_price))
            
            if len(self.price_history[symbol]) > 1000:
                self.price_history[symbol] = self.price_history[symbol][-1000:]
                
    async def _position_monitoring_loop(self):
        """Monitor and update positions"""
        while True:
            try:
                if self.is_trading:
                    await self._update_positions()
                    await self._update_account_equity()
                    
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(10)
                
    async def _order_management_loop(self):
        """Process pending orders"""
        while self.is_trading:
            try:
                # Process pending orders
                pending_orders = [o for o in self.account.open_orders if o.status == OrderStatus.PENDING]
                
                for order in pending_orders:
                    await self._try_fill_order(order)
                    
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                self.logger.error(f"Order management error: {e}")
                await asyncio.sleep(2)
                
    async def _performance_monitoring_loop(self):
        """Monitor performance metrics"""
        while self.is_trading:
            try:
                await self._update_performance_metrics()
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _try_fill_order(self, order: PaperOrder):
        """Attempt to fill a pending order"""
        try:
            # Get current market data
            market_data = self.market_data_cache.get(order.symbol)
            if not market_data:
                return
                
            # Determine fill price
            if order.order_type == TradeType.MARKET:
                if order.direction == TradeDirection.BUY:
                    fill_price = market_data.ask
                else:
                    fill_price = market_data.bid
            elif order.order_type == TradeType.LIMIT:
                if order.direction == TradeDirection.BUY:
                    if market_data.ask <= order.price:
                        fill_price = order.price
                    else:
                        return  # No fill
                else:
                    if market_data.bid >= order.price:
                        fill_price = order.price
                    else:
                        return  # No fill
            else:
                return
                
            # Apply execution simulation (slippage, partial fills)
            execution_result = await self._simulate_execution(order, fill_price)
            fill_price = execution_result["executed_price"]
            fill_quantity = execution_result["executed_quantity"]
            
            if fill_quantity <= 0:
                return
                
            # Update order
            order.filled_quantity += fill_quantity
            order.avg_fill_price = (
                (order.avg_fill_price * (order.filled_quantity - fill_quantity) + 
                 fill_price * fill_quantity) / order.filled_quantity
            )
            
            # Check if fully filled
            if order.filled_quantity >= order.quantity * 0.99:  # 99% filled = complete
                order.status = OrderStatus.FILLED
                self.account.open_orders = [o for o in self.account.open_orders if o.order_id != order.order_id]
                # Add to recent orders
                self.account.recent_orders.append(order)
                # Keep only last 50 recent orders
                if len(self.account.recent_orders) > 50:
                    self.account.recent_orders = self.account.recent_orders[-50:]
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
                
            # Update position
            await self._update_position_from_fill(order, fill_quantity, fill_price)
            
            # Calculate and deduct fees
            fees = fill_quantity * fill_price * self.fee_rate
            self.account.current_balance -= fees
            
            # Update trade count
            if order.status == OrderStatus.FILLED:
                self.account.trade_count += 1
                
                # Report to analytics system
                await self._report_trade_to_analytics(order, fill_price, fees)
                
            # Update database
            await self._update_order(order)
            
            self.logger.info(f"Order filled: {order.order_id} - {fill_quantity} @ {fill_price}")
            
        except Exception as e:
            self.logger.error(f"Failed to fill order {order.order_id}: {e}")
            
    async def _simulate_execution(self, order: PaperOrder, base_price: float) -> Dict[str, float]:
        """Simulate realistic execution with slippage and partial fills"""
        # Apply slippage
        slippage = np.random.normal(0, self.slippage_factor)
        if order.direction == TradeDirection.BUY:
            executed_price = base_price * (1 + abs(slippage))
        else:
            executed_price = base_price * (1 - abs(slippage))
            
        # Simulate partial fills (10% chance for market orders, 30% for limit)
        partial_fill_prob = 0.1 if order.order_type == TradeType.MARKET else 0.3
        if np.random.random() < partial_fill_prob:
            executed_quantity = order.quantity * np.random.uniform(0.3, 0.8)
        else:
            executed_quantity = order.quantity
            
        return {
            "executed_price": executed_price,
            "executed_quantity": executed_quantity
        }
        
    async def _update_position_from_fill(self, order: PaperOrder, fill_quantity: float, fill_price: float):
        """Update position based on order fill"""
        symbol = order.symbol
        
        if symbol not in self.account.positions:
            # New position
            self.account.positions[symbol] = PaperPosition(
                symbol=symbol,
                quantity=fill_quantity if order.direction == TradeDirection.BUY else -fill_quantity,
                avg_entry_price=fill_price,
                current_price=fill_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                direction=order.direction,
                open_time=trading_time.now(),
                last_updated=trading_time.now()
            )
        else:
            # Update existing position
            position = self.account.positions[symbol]
            
            if order.direction == TradeDirection.BUY:
                new_quantity = position.quantity + fill_quantity
                if new_quantity != 0:
                    position.avg_entry_price = (
                        (position.avg_entry_price * position.quantity + fill_price * fill_quantity) / new_quantity
                    )
                position.quantity = new_quantity
            else:
                # Selling - check if closing or opening short
                if position.quantity > 0:
                    # Closing long position
                    close_quantity = min(fill_quantity, position.quantity)
                    realized_pnl = close_quantity * (fill_price - position.avg_entry_price)
                    position.realized_pnl += realized_pnl
                    position.quantity -= close_quantity
                    
                    if fill_quantity > close_quantity:
                        # Going short
                        short_quantity = fill_quantity - close_quantity
                        position.quantity = -short_quantity
                        position.avg_entry_price = fill_price
                        position.direction = TradeDirection.SELL
                else:
                    # Adding to short position
                    new_quantity = position.quantity - fill_quantity
                    if new_quantity != 0:
                        position.avg_entry_price = (
                            (position.avg_entry_price * abs(position.quantity) + fill_price * fill_quantity) / abs(new_quantity)
                        )
                    position.quantity = new_quantity
                    
            position.last_updated = trading_time.now()
            
    async def _update_positions(self):
        """Update all positions with current market prices"""
        for symbol, position in self.account.positions.items():
            market_data = self.market_data_cache.get(symbol)
            if market_data:
                position.current_price = market_data.last
                
                # Calculate unrealized PnL
                if position.quantity > 0:  # Long
                    position.unrealized_pnl = position.quantity * (position.current_price - position.avg_entry_price)
                else:  # Short
                    position.unrealized_pnl = abs(position.quantity) * (position.avg_entry_price - position.current_price)
                    
                position.last_updated = trading_time.now()
                
    async def _update_account_equity(self):
        """Update account equity based on positions"""
        total_unrealized_pnl = sum([pos.unrealized_pnl for pos in self.account.positions.values()])
        total_realized_pnl = sum([pos.realized_pnl for pos in self.account.positions.values()])
        
        self.account.total_pnl = total_realized_pnl + total_unrealized_pnl
        self.account.equity = self.account.current_balance + total_unrealized_pnl
        self.account.free_margin = self.account.equity * 0.8  # 80% available for trading
        self.account.last_updated = trading_time.now()
        
    async def _perform_risk_check(self, symbol: str, direction: TradeDirection, quantity: float, price: Optional[float]) -> Dict[str, Any]:
        """Perform risk checks on order"""
        try:
            # Get current price
            market_data = self.market_data_cache.get(symbol)
            check_price = price if price else (market_data.ask if market_data else 100.0)
            
            # Calculate order value
            order_value = quantity * check_price
            
            # Check account balance
            if order_value > self.account.free_margin:
                return {"approved": False, "reason": "Insufficient free margin"}
                
            # Check position size limits (max 20% of equity per position)
            max_position_value = self.account.equity * 0.2
            if order_value > max_position_value:
                return {"approved": False, "reason": "Position size too large"}
                
            # Check daily loss limit (max 5% of initial balance)
            if self.account.daily_pnl < -self.account.initial_balance * 0.05:
                return {"approved": False, "reason": "Daily loss limit reached"}
                
            return {"approved": True, "reason": "Risk checks passed"}
            
        except Exception as e:
            self.logger.error(f"Risk check error: {e}")
            return {"approved": False, "reason": f"Risk check failed: {e}"}
    
    async def _check_order_collision(self, symbol: str, direction: TradeDirection, quantity: float, strategy_name: str) -> Dict[str, Any]:
        """Check for order collisions (identical or very similar orders)"""
        try:
            current_time = trading_time.now()
            collision_window = timedelta(seconds=30)  # 30 second collision window
            
            # Check for identical orders in recent time window
            for order in self.account.open_orders:
                time_diff = current_time - order.timestamp
                
                if (time_diff < collision_window and 
                    order.symbol == symbol and 
                    order.direction == direction and 
                    abs(order.quantity - quantity) < 0.1):  # Allow small quantity differences
                    
                    return {
                        "collision_detected": True,
                        "reason": f"Identical order exists from {time_diff.total_seconds():.1f}s ago"
                    }
            
            # Check for orders from same strategy on same symbol in collision window  
            same_strategy_orders = 0
            for order in self.account.open_orders:
                time_diff = current_time - order.timestamp
                
                if (time_diff < collision_window and 
                    order.symbol == symbol and 
                    order.strategy_name == strategy_name):
                    same_strategy_orders += 1
            
            # Limit same strategy orders on same symbol
            if same_strategy_orders >= 2:
                return {
                    "collision_detected": True, 
                    "reason": f"Too many orders from {strategy_name} on {symbol} in collision window"
                }
            
            # Check recent orders (including filled ones) to prevent rapid fire
            recent_order_count = 0
            for order in self.account.recent_orders[-10:]:  # Check last 10 recent orders
                time_diff = current_time - order.timestamp
                
                if (time_diff < collision_window and 
                    order.symbol == symbol and 
                    order.strategy_name == strategy_name):
                    recent_order_count += 1
            
            if recent_order_count >= 3:
                return {
                    "collision_detected": True,
                    "reason": f"Rapid fire orders detected: {recent_order_count} orders in {collision_window.total_seconds()}s"
                }
            
            return {"collision_detected": False, "reason": "No collision detected"}
            
        except Exception as e:
            self.logger.error(f"Order collision check error: {e}")
            # On error, allow order but log the issue
            return {"collision_detected": False, "reason": f"Collision check failed: {e}"}
            
    async def _cancel_all_orders(self):
        """Cancel all pending orders"""
        for order in self.account.open_orders[:]:  # Copy list to avoid modification during iteration
            if order.status == OrderStatus.PENDING:
                await self.cancel_order(order.order_id)
                
    async def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        try:
            # Calculate metrics
            total_return = (self.account.equity - self.account.initial_balance) / self.account.initial_balance
            
            # Calculate win rate
            filled_orders = []  # Would need to track this from database
            if self.account.trade_count > 0:
                profitable_trades = len([pos for pos in self.account.positions.values() if pos.realized_pnl > 0])
                self.account.win_rate = profitable_trades / self.account.trade_count
            else:
                self.account.win_rate = 0.0
                
            # Update daily PnL
            # This would be calculated based on start-of-day balance
            self.account.daily_pnl = self.account.total_pnl  # Simplified
            
            self.performance_metrics = {
                "total_return": total_return,
                "equity": self.account.equity,
                "balance": self.account.current_balance,
                "unrealized_pnl": sum([pos.unrealized_pnl for pos in self.account.positions.values()]),
                "realized_pnl": sum([pos.realized_pnl for pos in self.account.positions.values()]),
                "trade_count": self.account.trade_count,
                "win_rate": self.account.win_rate,
                "active_positions": len([pos for pos in self.account.positions.values() if pos.quantity != 0])
            }
            
        except Exception as e:
            self.logger.error(f"Performance metrics update error: {e}")
            
    async def _calculate_session_metrics(self) -> Dict[str, float]:
        """Calculate final session performance metrics"""
        return {
            "total_return": (self.account.equity - self.account.initial_balance) / self.account.initial_balance,
            "total_trades": self.account.trade_count,
            "win_rate": self.account.win_rate,
            "final_balance": self.account.current_balance,
            "final_equity": self.account.equity,
            "max_positions": max([len(self.account.positions)] + [0])
        }
        
    async def _store_order(self, order: PaperOrder):
        """Store order in database"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO paper_orders
                (account_id, order_id, symbol, direction, order_type, quantity, price, status,
                 filled_quantity, avg_fill_price, strategy_name, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.account.account_id, order.order_id, order.symbol, order.direction.value,
                order.order_type.value, order.quantity, order.price, order.status.value,
                order.filled_quantity, order.avg_fill_price, order.strategy_name,
                json.dumps(order.metadata), trading_time.now().isoformat(), trading_time.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store order: {e}")
            
    async def _update_order(self, order: PaperOrder):
        """Update order in database"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE paper_orders
                SET status = ?, filled_quantity = ?, avg_fill_price = ?, updated_at = ?
                WHERE order_id = ?
            ''', (
                order.status.value, order.filled_quantity, order.avg_fill_price,
                trading_time.now().isoformat(), order.order_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to update order: {e}")
            
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        await self._update_performance_metrics()
        
        return {
            "account_summary": {
                "account_id": self.account.account_id,
                "initial_balance": self.account.initial_balance,
                "current_balance": self.account.current_balance,
                "equity": self.account.equity,
                "total_pnl": self.account.total_pnl,
                "total_return": (self.account.equity - self.account.initial_balance) / self.account.initial_balance
            },
            "trading_stats": {
                "total_trades": self.account.trade_count,
                "win_rate": self.account.win_rate,
                "active_positions": len([pos for pos in self.account.positions.values() if pos.quantity != 0]),
                "open_orders": len(self.account.open_orders)
            },
            "positions": {symbol: asdict(pos) for symbol, pos in self.account.positions.items()},
            "market_data": {symbol: asdict(data) for symbol, data in self.market_data_cache.items()},
            "performance_metrics": self.performance_metrics,
            "trading_sessions": self.trading_sessions
        }
        
    async def _report_trade_to_analytics(self, order: PaperOrder, fill_price: float, fees: float):
        """Report completed trade to analytics system"""
        if not self.analytics:
            return
            
        try:
            # Extract symbol information
            base_symbol = order.symbol.split('/')[0] if '/' in order.symbol else order.symbol
            
            # Determine if this is opening or closing a position
            current_position = self.account.positions.get(order.symbol)
            
            if order.direction == TradeDirection.BUY:
                # Opening or adding to long position
                trade_id = self.analytics.record_trade_entry(
                    token_address=order.symbol,  # Using symbol as address for paper trading
                    token_symbol=base_symbol,
                    entry_price=fill_price,
                    quantity=order.filled_quantity,
                    gas_fees=fees,
                    discovery_source=order.metadata.get('discovery_source', order.strategy_name)
                )
                
                # Store trade_id in order metadata for later exit reporting
                order.metadata['analytics_trade_id'] = trade_id
                
                self.logger.info(f"[ANALYTICS] Trade entry recorded: {trade_id} - BUY {order.filled_quantity} {base_symbol} @ {fill_price}")
                
            else:
                # Closing or reducing long position (or opening short - but we'll treat as exit for now)
                if current_position and current_position.quantity > 0:
                    # This is closing a position - report as exit
                    # For paper trading, we'll use order_id as trade_id since we don't track individual entries
                    exit_reason = order.metadata.get('exit_reason', 'manual_exit')
                    
                    # Calculate P&L
                    pnl = (fill_price - current_position.avg_entry_price) * order.filled_quantity - fees
                    
                    self.analytics.record_trade_exit(
                        trade_id=order.order_id,  # Using order_id as trade_id
                        exit_price=fill_price,
                        exit_reason=exit_reason,
                        gas_fees=fees
                    )
                    
                    self.logger.info(f"[ANALYTICS] Trade exit recorded: {order.order_id} - SELL {order.filled_quantity} {base_symbol} @ {fill_price} (P&L: ${pnl:.2f})")
                
        except Exception as e:
            self.logger.error(f"[ANALYTICS] Failed to report trade to analytics: {e}")
        
    async def shutdown(self):
        """Shutdown paper trading engine"""
        try:
            if self.is_trading:
                await self.stop_trading()
                
            if self.real_dex_connector:
                await self.real_dex_connector.shutdown()
                
            self.logger.info("Paper trading engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")