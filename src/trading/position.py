"""
position.py - Defines position management classes and functionality for trading
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradeEntry:
    """Trade entry information"""
    token_address: str
    entry_price: float
    entry_time: datetime
    size: float

@dataclass
class TrailingStop:
    """Trailing stop configuration and state"""
    initial_price: float
    distance_percentage: float
    highest_price: float = 0.0
    current_stop: float = 0.0
    activated: bool = False

    def update(self, current_price: float) -> None:
        """Update trailing stop with current price"""
        if not self.activated:
            self.activated = True
            self.highest_price = current_price
            self.current_stop = current_price * (1 - self.distance_percentage)
        elif current_price > self.highest_price:
            self.highest_price = current_price
            self.current_stop = current_price * (1 - self.distance_percentage)

    def is_triggered(self, current_price: float) -> bool:
        """Check if trailing stop is triggered"""
        return self.activated and current_price < self.current_stop

@dataclass
class Position:
    """Trading position with comprehensive management features"""
    token_address: str
    size: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    high_water_mark: float = field(init=False)
    current_price: float = field(init=False)
    unrealized_pnl: float = field(init=False, default=0.0)
    status: str = "open"
    trailing_stop: Optional[Dict[str, float]] = None
    scaled_take_profits: List[Dict[str, Any]] = field(default_factory=list)
    trade_entry: Optional[TradeEntry] = None
    entry_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize derived fields after creation"""
        self.current_price = self.entry_price
        self.high_water_mark = self.entry_price
        self._update_pnl()

    def _update_pnl(self) -> None:
        """Update unrealized PnL"""
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.size

    def update_price(self, new_price: float) -> None:
        """Update position with new price"""
        self.current_price = new_price
        if new_price > self.high_water_mark:
            self.high_water_mark = new_price
            self._update_trailing_stop()
        self._update_pnl()

    def _update_trailing_stop(self) -> None:
        """Update trailing stop if active"""
        if self.trailing_stop:
            self.trailing_stop['stop_price'] = self.high_water_mark * (
                1 - self.trailing_stop['distance']
            )

    def add_trailing_stop(self, distance_percentage: float) -> None:
        """Add trailing stop to position"""
        self.trailing_stop = {
            'distance': distance_percentage,
            'stop_price': self.entry_price * (1 - distance_percentage)
        }

    def add_scaled_take_profit(self, percentage: float, size_percentage: float) -> None:
        """Add a scaled take profit level"""
        self.scaled_take_profits.append({
            'percentage': percentage,
            'size_percentage': size_percentage,
            'target_price': self.entry_price * (1 + percentage),
            'triggered': False
        })
        self.scaled_take_profits.sort(key=lambda x: x['percentage'])

    def should_close(self) -> Tuple[bool, str]:
        """Check if position should be closed"""
        if self.status != "open":
            return False, ""

        # Check stop loss
        if self.stop_loss and self.current_price <= self.stop_loss:
            return True, "stop_loss"

        # Check trailing stop
        if self.trailing_stop and self.current_price <= self.trailing_stop['stop_price']:
            return True, "trailing_stop"

        # Check take profit
        if self.take_profit and self.current_price >= self.take_profit:
            return True, "take_profit"

        # Check scaled take profits
        for tp in self.scaled_take_profits:
            if not tp['triggered'] and self.current_price >= tp['target_price']:
                return True, "scaled_take_profit"

        return False, ""

    def get_position_info(self) -> Dict[str, Any]:
        """Get comprehensive position information"""
        return {
            "token_address": self.token_address,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "size": self.size,
            "unrealized_pnl": self.unrealized_pnl,
            "status": self.status,
            "high_water_mark": self.high_water_mark,
            "entry_time": self.entry_time,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop": self.trailing_stop,
            "scaled_take_profits": self.scaled_take_profits
        }

@dataclass
class PaperPosition:
    """Paper trading position implementation"""
    token_address: str
    size: float
    entry_price: float
    high_water_mark: float = field(init=False)
    take_profits: List[Dict[str, float]] = field(default_factory=list)
    trailing_stop: Optional[float] = None
    current_price: float = field(init=False)
    entry_time: datetime = field(default_factory=datetime.now)
    pnl: float = field(init=False, default=0.0)
    status: str = "open"

    def __post_init__(self):
        """Initialize derived fields"""
        self.current_price = self.entry_price
        self.high_water_mark = self.entry_price
        self._update_pnl()

    def update_price(self, new_price: float) -> None:
        """Update position with new price"""
        self.current_price = new_price
        if new_price > self.high_water_mark:
            self.high_water_mark = new_price
        self._update_pnl()

    def _update_pnl(self) -> None:
        """Update position PnL"""
        self.pnl = (self.current_price - self.entry_price) * self.size

    def check_exits(self) -> Tuple[bool, str, Optional[float]]:
        """Check if any exit conditions are met"""
        # Check trailing stop
        if self.trailing_stop:
            stop_price = self.high_water_mark * (1 - self.trailing_stop)
            if self.current_price <= stop_price:
                return True, "trailing_stop", None

        # Check take profits
        for tp in self.take_profits:
            if self.current_price >= self.entry_price * (1 + tp['percentage']):
                return True, "take_profit", self.size * tp['size_percentage']

        return False, "", None

@dataclass
class PositionManager:
    """Manages trading positions"""
    swap_executor: Any
    settings: Any
    positions: Dict[str, Position] = field(default_factory=dict)

    async def open_position(
        self,
        token_address: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> Optional[Position]:
        """Open a new position"""
        try:
            if token_address in self.positions:
                logger.warning(f"Position already exists for {token_address}")
                return None

            trade_entry = TradeEntry(
                token_address=token_address,
                entry_price=entry_price,
                entry_time=datetime.now(),
                size=size
            )

            position = Position(
                token_address=token_address,
                size=size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trade_entry=trade_entry
            )

            self.positions[token_address] = position
            logger.info(f"Opened position for {token_address} at {entry_price}")
            return position

        except Exception as e:
            logger.error(f"Error opening position: {str(e)}")
            return None

    async def close_position(self, token_address: str, reason: str = "manual") -> bool:
        """Close an existing position"""
        try:
            position = self.positions.get(token_address)
            if not position or position.status != "open":
                return False

            # Execute closing swap
            success = await self.swap_executor.execute_swap(
                input_token=token_address,
                output_token="So11111111111111111111111111111111111111112",  # SOL
                amount=position.size,
                slippage=self.settings.SLIPPAGE_TOLERANCE
            )

            if success:
                position.status = "closed"
                logger.info(f"Position closed: {token_address}, Reason: {reason}")
                return True

            logger.error("Failed to execute closing swap")
            return False

        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return False

    def get_open_positions(self) -> Dict[str, Position]:
        """Get all currently open positions"""
        return {
            addr: pos for addr, pos in self.positions.items()
            if pos.status == "open"
        }

    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions"""
        open_positions = self.get_open_positions()
        total_pnl = sum(pos.unrealized_pnl for pos in open_positions.values())

        return {
            "open_positions": len(open_positions),
            "total_pnl": total_pnl,
            "positions": [
                {
                    "token": addr,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "pnl": pos.unrealized_pnl,
                    "size": pos.size
                }
                for addr, pos in open_positions.items()
            ]
        }