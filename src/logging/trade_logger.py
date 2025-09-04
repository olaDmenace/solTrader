#!/usr/bin/env python3
"""
Centralized Trade Logger with Execution Metrics Tracking
Professional-grade trade execution logging system for comprehensive performance analysis.

Features:
- Real vs theoretical price tracking
- Slippage and fill quality analysis
- Performance attribution by strategy
- Persistent storage with query capabilities
- Async logging for minimal performance impact
- Thread-safe concurrent access
"""

import asyncio
import logging
import sqlite3
import json
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from pathlib import Path
import aiosqlite

logger = logging.getLogger(__name__)

class TradeStatus(Enum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED" 
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"

class TradeType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    ARBITRAGE_BUY = "ARBITRAGE_BUY"
    ARBITRAGE_SELL = "ARBITRAGE_SELL"
    GRID_BUY = "GRID_BUY"
    GRID_SELL = "GRID_SELL"

@dataclass
class TradeRecord:
    """Comprehensive trade record with all execution metrics"""
    
    # Trade Identification
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy: str = ""
    token_address: str = ""
    token_symbol: str = ""
    
    # Trade Details
    trade_type: TradeType = TradeType.BUY
    status: TradeStatus = TradeStatus.PENDING
    
    # Pricing Information
    theoretical_price: float = 0.0
    requested_price: float = 0.0
    executed_price: float = 0.0
    market_price_at_execution: float = 0.0
    
    # Size and Value
    requested_size: float = 0.0
    executed_size: float = 0.0
    usd_value: float = 0.0
    
    # Execution Metrics
    slippage_percentage: float = 0.0
    slippage_usd: float = 0.0
    execution_time_ms: float = 0.0
    fill_quality_score: float = 0.0  # 0-100 score
    
    # Costs
    gas_fee: float = 0.0
    dex_fees: float = 0.0
    total_costs: float = 0.0
    
    # Performance Metrics
    pnl_usd: float = 0.0
    pnl_percentage: float = 0.0
    roi: float = 0.0
    
    # Context Information
    dex_name: str = ""
    order_book_spread: float = 0.0
    market_volatility: float = 0.0
    liquidity_at_execution: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Additional Data
    signal_strength: float = 0.0
    risk_score: float = 0.0
    confidence_level: float = 0.0
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_execution_metrics(self):
        """Calculate derived execution metrics"""
        try:
            # Slippage calculation
            if self.theoretical_price > 0 and self.executed_price > 0:
                if self.trade_type in [TradeType.BUY, TradeType.ARBITRAGE_BUY, TradeType.GRID_BUY]:
                    # For buys, positive slippage = paid more than expected
                    self.slippage_percentage = ((self.executed_price - self.theoretical_price) / self.theoretical_price) * 100
                else:
                    # For sells, positive slippage = received less than expected  
                    self.slippage_percentage = ((self.theoretical_price - self.executed_price) / self.theoretical_price) * 100
                
                self.slippage_usd = abs(self.slippage_percentage / 100) * self.usd_value
            
            # Fill quality score (0-100)
            fill_ratio = self.executed_size / max(self.requested_size, 0.001)
            slippage_penalty = max(0, 100 - abs(self.slippage_percentage) * 20)  # Penalty for high slippage
            speed_bonus = max(0, 100 - (self.execution_time_ms / 1000) * 10)  # Bonus for fast execution
            
            self.fill_quality_score = min(100, (fill_ratio * 50) + (slippage_penalty * 0.3) + (speed_bonus * 0.2))
            
            # Total costs
            self.total_costs = self.gas_fee + self.dex_fees + self.slippage_usd
            
            # Update timestamp
            self.updated_at = datetime.now()
            
        except Exception as e:
            logger.error(f"[TRADE_LOGGER] Error calculating metrics for trade {self.trade_id}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade record to dictionary for storage"""
        data = asdict(self)
        
        # Convert enums to strings
        data['trade_type'] = self.trade_type.value
        data['status'] = self.status.value
        
        # Convert datetimes to ISO strings
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['executed_at'] = self.executed_at.isoformat() if self.executed_at else None
        
        # Convert metadata to JSON string
        data['metadata'] = json.dumps(self.metadata)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeRecord':
        """Create trade record from dictionary"""
        try:
            # Convert string enums back
            data['trade_type'] = TradeType(data['trade_type'])
            data['status'] = TradeStatus(data['status'])
            
            # Convert ISO strings back to datetimes
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
            if data['executed_at']:
                data['executed_at'] = datetime.fromisoformat(data['executed_at'])
            
            # Convert JSON string back to dict
            if isinstance(data['metadata'], str):
                data['metadata'] = json.loads(data['metadata'])
            
            return cls(**data)
            
        except Exception as e:
            logger.error(f"[TRADE_LOGGER] Error creating TradeRecord from dict: {e}")
            raise

class TradeDatabase:
    """Async SQLite database for trade storage with high performance"""
    
    def __init__(self, db_path: str = "logs/trades.db"):
        self.db_path = db_path
        self.db_initialized = False
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self):
        """Initialize database schema"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        trade_id TEXT PRIMARY KEY,
                        strategy TEXT,
                        token_address TEXT,
                        token_symbol TEXT,
                        trade_type TEXT,
                        status TEXT,
                        theoretical_price REAL,
                        requested_price REAL,
                        executed_price REAL,
                        market_price_at_execution REAL,
                        requested_size REAL,
                        executed_size REAL,
                        usd_value REAL,
                        slippage_percentage REAL,
                        slippage_usd REAL,
                        execution_time_ms REAL,
                        fill_quality_score REAL,
                        gas_fee REAL,
                        dex_fees REAL,
                        total_costs REAL,
                        pnl_usd REAL,
                        pnl_percentage REAL,
                        roi REAL,
                        dex_name TEXT,
                        order_book_spread REAL,
                        market_volatility REAL,
                        liquidity_at_execution REAL,
                        signal_strength REAL,
                        risk_score REAL,
                        confidence_level REAL,
                        notes TEXT,
                        metadata TEXT,
                        created_at TEXT,
                        executed_at TEXT,
                        updated_at TEXT
                    )
                """)
                
                # Create indexes for common queries
                await db.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON trades(strategy)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_token ON trades(token_address)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON trades(created_at)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_status ON trades(status)")
                
                await db.commit()
                
            self.db_initialized = True
            logger.info(f"[TRADE_DB] Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"[TRADE_DB] Database initialization error: {e}")
            raise
    
    async def insert_trade(self, trade: TradeRecord):
        """Insert new trade record"""
        try:
            if not self.db_initialized:
                await self.initialize()
            
            trade_dict = trade.to_dict()
            
            async with aiosqlite.connect(self.db_path) as db:
                columns = list(trade_dict.keys())
                placeholders = ','.join(['?' for _ in columns])
                values = list(trade_dict.values())
                
                await db.execute(
                    f"INSERT OR REPLACE INTO trades ({','.join(columns)}) VALUES ({placeholders})",
                    values
                )
                await db.commit()
                
        except Exception as e:
            logger.error(f"[TRADE_DB] Error inserting trade {trade.trade_id}: {e}")
            raise
    
    async def update_trade(self, trade: TradeRecord):
        """Update existing trade record"""
        try:
            if not self.db_initialized:
                await self.initialize()
            
            trade.updated_at = datetime.now()
            trade_dict = trade.to_dict()
            
            async with aiosqlite.connect(self.db_path) as db:
                set_clause = ','.join([f"{col} = ?" for col in trade_dict.keys() if col != 'trade_id'])
                values = [val for key, val in trade_dict.items() if key != 'trade_id']
                values.append(trade.trade_id)
                
                await db.execute(
                    f"UPDATE trades SET {set_clause} WHERE trade_id = ?",
                    values
                )
                await db.commit()
                
        except Exception as e:
            logger.error(f"[TRADE_DB] Error updating trade {trade.trade_id}: {e}")
            raise
    
    async def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """Get trade by ID"""
        try:
            if not self.db_initialized:
                await self.initialize()
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
                row = await cursor.fetchone()
                
                if row:
                    return TradeRecord.from_dict(dict(row))
                return None
                
        except Exception as e:
            logger.error(f"[TRADE_DB] Error getting trade {trade_id}: {e}")
            return None
    
    async def get_trades(
        self,
        strategy: Optional[str] = None,
        token_address: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[TradeStatus] = None,
        limit: int = 1000
    ) -> List[TradeRecord]:
        """Get trades with filters"""
        try:
            if not self.db_initialized:
                await self.initialize()
            
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)
            
            if token_address:
                query += " AND token_address = ?"
                params.append(token_address)
            
            if start_date:
                query += " AND created_at >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND created_at <= ?"
                params.append(end_date.isoformat())
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                
                return [TradeRecord.from_dict(dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"[TRADE_DB] Error getting trades: {e}")
            return []
    
    async def get_trade_analytics(
        self,
        strategy: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get trade analytics and statistics"""
        try:
            if not self.db_initialized:
                await self.initialize()
            
            start_date = datetime.now() - timedelta(days=days)
            
            query = """
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN status = 'EXECUTED' THEN 1 ELSE 0 END) as executed_trades,
                    AVG(slippage_percentage) as avg_slippage,
                    AVG(execution_time_ms) as avg_execution_time,
                    AVG(fill_quality_score) as avg_fill_quality,
                    SUM(pnl_usd) as total_pnl,
                    AVG(pnl_percentage) as avg_pnl_percentage,
                    SUM(total_costs) as total_costs
                FROM trades 
                WHERE created_at >= ?
            """
            
            params = [start_date.isoformat()]
            
            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)
            
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(query, params)
                row = await cursor.fetchone()
                
                if row:
                    return {
                        'total_trades': row[0] or 0,
                        'executed_trades': row[1] or 0,
                        'success_rate': (row[1] / max(row[0], 1)) * 100,
                        'avg_slippage': row[2] or 0,
                        'avg_execution_time_ms': row[3] or 0,
                        'avg_fill_quality': row[4] or 0,
                        'total_pnl_usd': row[5] or 0,
                        'avg_pnl_percentage': row[6] or 0,
                        'total_costs': row[7] or 0,
                        'period_days': days,
                        'strategy': strategy or 'ALL'
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"[TRADE_DB] Error getting analytics: {e}")
            return {}

class CentralizedTradeLogger:
    """
    Main trade logger class with async operation and thread safety
    """
    
    def __init__(self, settings: Any):
        self.settings = settings
        self.database = TradeDatabase()
        self.pending_trades: Dict[str, TradeRecord] = {}
        self.log_queue: List[TradeRecord] = []
        self.queue_lock = threading.Lock()
        self.is_running = False
        self.flush_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.logged_trades = 0
        self.total_processing_time = 0.0
        
        logger.info("[TRADE_LOGGER] Centralized trade logger initialized")
    
    async def start(self):
        """Start the trade logger"""
        try:
            await self.database.initialize()
            self.is_running = True
            
            # Start async flush task
            self.flush_task = asyncio.create_task(self._flush_loop())
            
            logger.info("[TRADE_LOGGER] ✅ Trade logger started successfully")
            return True
            
        except Exception as e:
            logger.error(f"[TRADE_LOGGER] Failed to start: {e}")
            return False
    
    async def stop(self):
        """Stop the trade logger"""
        try:
            self.is_running = False
            
            if self.flush_task:
                self.flush_task.cancel()
                try:
                    await self.flush_task
                except asyncio.CancelledError:
                    pass
            
            # Flush any remaining trades
            await self._flush_pending_trades()
            
            logger.info("[TRADE_LOGGER] ✅ Trade logger stopped")
            
        except Exception as e:
            logger.error(f"[TRADE_LOGGER] Error stopping: {e}")
    
    def log_trade_request(
        self,
        strategy: str,
        token_address: str,
        token_symbol: str,
        trade_type: TradeType,
        theoretical_price: float,
        requested_price: float,
        requested_size: float,
        dex_name: str = "",
        signal_strength: float = 0.0,
        risk_score: float = 0.0,
        confidence_level: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a new trade request (synchronous for immediate response)"""
        try:
            trade = TradeRecord(
                strategy=strategy,
                token_address=token_address,
                token_symbol=token_symbol,
                trade_type=trade_type,
                status=TradeStatus.PENDING,
                theoretical_price=theoretical_price,
                requested_price=requested_price,
                requested_size=requested_size,
                dex_name=dex_name,
                signal_strength=signal_strength,
                risk_score=risk_score,
                confidence_level=confidence_level,
                metadata=metadata or {}
            )
            
            # Store in pending trades
            self.pending_trades[trade.trade_id] = trade
            
            # Add to queue for async persistence
            with self.queue_lock:
                self.log_queue.append(trade)
            
            logger.info(f"[TRADE_LOGGER] 📝 Logged trade request: {trade.trade_id} ({strategy})")
            return trade.trade_id
            
        except Exception as e:
            logger.error(f"[TRADE_LOGGER] Error logging trade request: {e}")
            return ""
    
    def log_trade_execution(
        self,
        trade_id: str,
        executed_price: float,
        executed_size: float,
        execution_time_ms: float,
        gas_fee: float = 0.0,
        dex_fees: float = 0.0,
        market_price_at_execution: float = 0.0,
        order_book_spread: float = 0.0,
        liquidity_at_execution: float = 0.0,
        status: TradeStatus = TradeStatus.EXECUTED,
        notes: str = ""
    ):
        """Log trade execution details (synchronous)"""
        try:
            if trade_id not in self.pending_trades:
                logger.warning(f"[TRADE_LOGGER] Trade {trade_id} not found in pending trades")
                return
            
            trade = self.pending_trades[trade_id]
            
            # Update execution details
            trade.executed_price = executed_price
            trade.executed_size = executed_size
            trade.execution_time_ms = execution_time_ms
            trade.gas_fee = gas_fee
            trade.dex_fees = dex_fees
            trade.market_price_at_execution = market_price_at_execution or executed_price
            trade.order_book_spread = order_book_spread
            trade.liquidity_at_execution = liquidity_at_execution
            trade.status = status
            trade.executed_at = datetime.now()
            trade.notes = notes
            
            # Calculate USD value and metrics
            trade.usd_value = executed_price * executed_size
            trade.calculate_execution_metrics()
            
            # Add to queue for async persistence
            with self.queue_lock:
                self.log_queue.append(trade)
            
            # Remove from pending if completed
            if status in [TradeStatus.EXECUTED, TradeStatus.FAILED, TradeStatus.CANCELLED]:
                del self.pending_trades[trade_id]
            
            logger.info(f"[TRADE_LOGGER] 🎯 Logged execution: {trade_id} - Fill: {trade.fill_quality_score:.1f}, Slippage: {trade.slippage_percentage:.3f}%")
            
        except Exception as e:
            logger.error(f"[TRADE_LOGGER] Error logging execution for {trade_id}: {e}")
    
    async def _flush_loop(self):
        """Async loop to flush trades to database"""
        while self.is_running:
            try:
                await self._flush_pending_trades()
                await asyncio.sleep(5.0)  # Flush every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[TRADE_LOGGER] Flush loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _flush_pending_trades(self):
        """Flush pending trades to database"""
        try:
            if not self.log_queue:
                return
            
            # Get trades to flush
            with self.queue_lock:
                trades_to_flush = self.log_queue.copy()
                self.log_queue.clear()
            
            # Persist to database
            for trade in trades_to_flush:
                await self.database.insert_trade(trade)
                self.logged_trades += 1
            
            if trades_to_flush:
                logger.debug(f"[TRADE_LOGGER] Flushed {len(trades_to_flush)} trades to database")
                
        except Exception as e:
            logger.error(f"[TRADE_LOGGER] Error flushing trades: {e}")
    
    async def get_recent_trades(self, strategy: Optional[str] = None, limit: int = 100) -> List[TradeRecord]:
        """Get recent trades"""
        try:
            return await self.database.get_trades(strategy=strategy, limit=limit)
        except Exception as e:
            logger.error(f"[TRADE_LOGGER] Error getting recent trades: {e}")
            return []
    
    async def get_trade_analytics(self, strategy: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive trade analytics"""
        try:
            return await self.database.get_trade_analytics(strategy=strategy, days=days)
        except Exception as e:
            logger.error(f"[TRADE_LOGGER] Error getting analytics: {e}")
            return {}
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        try:
            analytics = await self.database.get_trade_analytics(days=30)
            
            return {
                'total_logged_trades': self.logged_trades,
                'pending_trades': len(self.pending_trades),
                'queue_size': len(self.log_queue),
                'analytics_30d': analytics,
                'system_uptime': 'running' if self.is_running else 'stopped'
            }
            
        except Exception as e:
            logger.error(f"[TRADE_LOGGER] Error getting performance summary: {e}")
            return {}