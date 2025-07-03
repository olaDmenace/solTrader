# src/trading/core.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class Position:
    token_address: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    status: str = 'open'
    current_price: float = field(init=False)
    unrealized_pnl: float = field(init=False)
    realized_pnl: float = 0.0
    scaled_take_profits: List[Dict[str, float]] = field(default_factory=list)
    trailing_stop: Optional[Dict[str, float]] = None
    
    def __post_init__(self) -> None:
        self.current_price = self.entry_price
        self.unrealized_pnl = 0.0

    def update(self, current_price: float) -> None:
        self.current_price = current_price
        self.unrealized_pnl = (current_price - self.entry_price) * self.size
        
        if self.trailing_stop:
            self._update_trailing_stop(current_price)

    def should_close(self) -> tuple[bool, str]:
        if self.current_price <= self.stop_loss:
            return True, 'stop_loss'
            
        if self.current_price >= self.take_profit:
            return True, 'take_profit'
            
        if self.trailing_stop and self._check_trailing_stop():
            return True, 'trailing_stop'
            
        return False, ''

    def _update_trailing_stop(self, current_price: float) -> None:
        if current_price > self.trailing_stop['high_price']:
            self.trailing_stop['high_price'] = current_price
            self.trailing_stop['stop_price'] = current_price * (1 - self.trailing_stop['distance'])

    def _check_trailing_stop(self) -> bool:
        return self.current_price < self.trailing_stop['stop_price']