from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


class TradeDirection(Enum):
    BUY = "buy"
    SELL = "sell"
    LONG = "long"
    SHORT = "short"


class TradeType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class TradeSignal:
    symbol: str
    direction: TradeDirection
    confidence: float
    entry_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    quantity: Optional[float] = None
    strategy_name: str = "unknown"
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Trade:
    trade_id: str
    symbol: str
    direction: TradeDirection
    entry_price: float
    quantity: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    fees: float = 0.0
    strategy_name: str = "unknown"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    @property
    def is_open(self) -> bool:
        return self.exit_time is None
        
    def calculate_pnl(self, current_price: float = None) -> float:
        """Calculate current PnL"""
        if self.exit_price is not None:
            # Closed position
            if self.direction == TradeDirection.BUY:
                return (self.exit_price - self.entry_price) * self.quantity - self.fees
            else:
                return (self.entry_price - self.exit_price) * self.quantity - self.fees
        elif current_price is not None:
            # Open position
            if self.direction == TradeDirection.BUY:
                return (current_price - self.entry_price) * self.quantity
            else:
                return (self.entry_price - current_price) * self.quantity
        else:
            return 0.0