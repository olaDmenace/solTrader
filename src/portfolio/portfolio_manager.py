import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.database.db_manager import DatabaseManager
from src.trading.risk_engine import RiskEngine
from src.monitoring.system_monitor import SystemMonitor
from src.trading.trade_types import Trade, TradeDirection
from src.utils.trading_time import trading_time


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    direction: TradeDirection
    unrealized_pnl: float
    realized_pnl: float
    entry_time: datetime
    last_update: datetime


class PortfolioManager:
    def __init__(
        self,
        db_manager: DatabaseManager,
        risk_engine: RiskEngine,
        monitor: SystemMonitor
    ):
        self.db_manager = db_manager
        self.risk_engine = risk_engine
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.cash_balance = 10000.0
        self.total_equity = 10000.0
        self.is_running = False
        
    async def initialize(self):
        """Initialize portfolio manager"""
        try:
            # Load existing positions
            await self._load_positions()
            
            self.logger.info("Portfolio manager initialized")
            
        except Exception as e:
            self.logger.error(f"Portfolio manager initialization failed: {e}")
            raise
            
    async def start(self):
        """Start portfolio management"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start portfolio monitoring
        asyncio.create_task(self._portfolio_monitoring_loop())
        
        self.logger.info("Portfolio manager started")
        
    async def stop(self):
        """Stop portfolio management"""
        self.is_running = False
        self.logger.info("Portfolio manager stopped")
        
    async def _load_positions(self):
        """Load existing positions from database"""
        try:
            # This would load from database in a real implementation
            # For now, initialize empty
            self.positions = {}
            
        except Exception as e:
            self.logger.error(f"Failed to load positions: {e}")
            
    async def _portfolio_monitoring_loop(self):
        """Monitor portfolio and update metrics"""
        while self.is_running:
            try:
                # Update portfolio metrics
                await self._update_portfolio_metrics()
                
                # Log portfolio value
                await self.monitor.log_system_metric("portfolio_value", self.total_equity)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Portfolio monitoring error: {e}")
                await asyncio.sleep(120)
                
    async def _update_portfolio_metrics(self):
        """Update portfolio metrics"""
        try:
            # Calculate total equity
            total_position_value = sum([
                pos.quantity * pos.current_price for pos in self.positions.values()
            ])
            
            self.total_equity = self.cash_balance + total_position_value
            
            # Update unrealized PnL for all positions
            for position in self.positions.values():
                if position.direction == TradeDirection.BUY:
                    position.unrealized_pnl = (position.current_price - position.avg_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.avg_price - position.current_price) * position.quantity
                    
                position.last_update = trading_time.now()
                
        except Exception as e:
            self.logger.error(f"Portfolio metrics update failed: {e}")
            
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        try:
            total_unrealized_pnl = sum([pos.unrealized_pnl for pos in self.positions.values()])
            total_realized_pnl = sum([pos.realized_pnl for pos in self.positions.values()])
            
            return {
                "cash_balance": self.cash_balance,
                "total_equity": self.total_equity,
                "total_positions": len(self.positions),
                "total_unrealized_pnl": total_unrealized_pnl,
                "total_realized_pnl": total_realized_pnl,
                "total_pnl": total_unrealized_pnl + total_realized_pnl,
                "positions": {
                    symbol: {
                        "quantity": pos.quantity,
                        "avg_price": pos.avg_price,
                        "current_price": pos.current_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "direction": pos.direction.value
                    }
                    for symbol, pos in self.positions.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio summary failed: {e}")
            return {}
            
    async def shutdown(self):
        """Shutdown portfolio manager"""
        try:
            await self.stop()
            self.logger.info("Portfolio manager shutdown")
            
        except Exception as e:
            self.logger.error(f"Portfolio manager shutdown error: {e}")