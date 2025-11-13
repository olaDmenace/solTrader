"""
position.py - Defines position management classes and functionality for trading
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
try:
    import numpy as np
except ImportError:
    # Fallback for environments without numpy
    np = None
from enum import Enum

logger = logging.getLogger(__name__)

class ExitReason(Enum):
    """Exit reason enumeration for better tracking"""
    MOMENTUM_REVERSAL = "momentum_reversal"
    OVERBOUGHT_DIVERGENCE = "overbought_divergence"
    TIME_LIMIT = "time_limit"
    PROFIT_PROTECTION = "profit_protection"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TAKE_PROFIT = "take_profit"
    SCALED_TAKE_PROFIT = "scaled_take_profit"
    MANUAL = "manual"
    CIRCUIT_BREAKER = "circuit_breaker"

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
    """Trading position with dynamic momentum-based exit management"""
    token_address: str
    size: float  # SOL amount invested
    entry_price: float
    token_balance: float = 0.0  # Actual tokens received from swap - CRITICAL for proper exit
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
    
    # Enhanced tracking for momentum-based exits
    price_history: List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    max_hold_time_minutes: int = 180  # 3 hours max hold
    momentum_exit_enabled: bool = True

    def __post_init__(self):
        """Initialize derived fields after creation"""
        self.current_price = self.entry_price
        self.high_water_mark = self.entry_price
        self._update_pnl()

    def _update_pnl(self) -> None:
        """Update unrealized PnL"""
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.size

    def update_price(self, new_price: float, volume: Optional[float] = None) -> None:
        """Update position with new price and volume data"""
        self.current_price = new_price
        self.last_update = datetime.now()
        
        # Update price history (keep last 20 data points)
        self.price_history.append(new_price)
        if len(self.price_history) > 20:
            self.price_history.pop(0)
            
        # Update volume history if provided
        if volume is not None:
            self.volume_history.append(volume)
            if len(self.volume_history) > 20:
                self.volume_history.pop(0)
        
        if new_price > self.high_water_mark:
            self.high_water_mark = new_price
            self._update_dynamic_trailing_stop()
        self._update_pnl()

    def _update_dynamic_trailing_stop(self) -> None:
        """Update trailing stop with momentum-based adjustments"""
        if not self.trailing_stop:
            return
            
        # Calculate current momentum
        momentum = self._calculate_momentum()
        
        # Adjust trailing distance based on momentum
        if momentum > 0.1:  # Strong upward momentum
            distance = 0.03   # Tighter 3% stop
        elif momentum > 0:   # Weak upward momentum  
            distance = 0.05   # Standard 5% stop
        else:                # Negative momentum
            distance = 0.02   # Very tight 2% stop
            
        self.trailing_stop['distance'] = distance
        self.trailing_stop['stop_price'] = self.high_water_mark * (1 - distance)

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
        """Check if position should be closed using dynamic momentum-based logic"""
        if self.status != "open":
            return False, ""

        # Enhanced momentum-based exit logic
        if self.momentum_exit_enabled:
            should_exit, reason = self._check_momentum_exit()
            if should_exit:
                return True, reason

        # Traditional exit checks (as backup)
        # Check stop loss
        if self.stop_loss and self.current_price <= self.stop_loss:
            return True, ExitReason.STOP_LOSS.value

        # Check trailing stop
        if self.trailing_stop and self.current_price <= self.trailing_stop['stop_price']:
            return True, ExitReason.TRAILING_STOP.value

        # Check traditional take profit (disabled by default for momentum strategy)
        if self.take_profit and self.current_price >= self.take_profit:
            return True, ExitReason.TAKE_PROFIT.value

        # Check scaled take profits
        for tp in self.scaled_take_profits:
            if not tp['triggered'] and self.current_price >= tp['target_price']:
                return True, ExitReason.SCALED_TAKE_PROFIT.value

        return False, ""

    def _check_momentum_exit(self) -> Tuple[bool, str]:
        """Advanced momentum-based exit logic - the core of the ape strategy"""
        if len(self.price_history) < 10:
            return False, ""
            
        # 1. Time-based safety exit (prevent bag holding)
        age_minutes = (datetime.now() - self.entry_time).total_seconds() / 60
        if age_minutes > self.max_hold_time_minutes:
            return True, ExitReason.TIME_LIMIT.value
            
        # 2. Calculate momentum indicators
        momentum = self._calculate_momentum()
        rsi = self._calculate_rsi()
        volume_trend = self._get_volume_trend()
        profit_percentage = (self.current_price / self.entry_price) - 1
        
        # 3. Profit protection (protect profits when still in decent profit)
        high_water_profit = (self.high_water_mark / self.entry_price) - 1
        if high_water_profit > 0.2 and profit_percentage > 0.05:  # Had 20%+ profit AND still have 5%+ profit
            trailing_stop_price = self.high_water_mark * 0.92  # 8% trailing stop for better protection
            if self.current_price < trailing_stop_price:
                return True, ExitReason.PROFIT_PROTECTION.value
        
        # 4. Momentum reversal detection (most important for unprofitable positions)
        if momentum < -0.03 and volume_trend == "declining":
            return True, ExitReason.MOMENTUM_REVERSAL.value
            
        # 5. Overbought with momentum divergence
        if rsi > 80 and momentum < 0.01:  # High RSI but slowing momentum
            return True, ExitReason.OVERBOUGHT_DIVERGENCE.value
                
        # 6. Quick loss cut for momentum breakdown
        if momentum < -0.05 and profit_percentage < -0.1:  # Strong negative momentum + 10% loss
            return True, ExitReason.MOMENTUM_REVERSAL.value
            
        return False, ""
        
    def _calculate_momentum(self) -> float:
        """Calculate 5-period momentum"""
        if len(self.price_history) < 10:
            return 0.0
        try:
            if np is not None:
                recent = np.mean(self.price_history[-5:])
                previous = np.mean(self.price_history[-10:-5])
            else:
                recent = sum(self.price_history[-5:]) / 5
                previous = sum(self.price_history[-10:-5]) / 5
            return (recent / previous) - 1 if previous > 0 else 0.0
        except (ZeroDivisionError, IndexError):
            return 0.0
            
    def _calculate_rsi(self, period: int = 14) -> float:
        """Calculate RSI for overbought detection"""
        if len(self.price_history) < period + 1:
            return 50.0  # Neutral RSI
        try:
            if np is not None:
                prices = np.array(self.price_history[-period-1:])
                deltas = np.diff(prices)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
            else:
                prices = self.price_history[-period-1:]
                deltas = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
                gains = [d for d in deltas if d > 0]
                losses = [-d for d in deltas if d < 0]
                avg_gain = sum(gains) / len(gains) if gains else 0
                avg_loss = sum(losses) / len(losses) if losses else 0
            
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi)
        except (ZeroDivisionError, IndexError):
            return 50.0
            
    def _get_volume_trend(self) -> str:
        """Determine if volume is increasing or declining"""
        if len(self.volume_history) < 6:
            return "unknown"
        try:
            if np is not None:
                recent_volume = np.mean(self.volume_history[-3:])
                previous_volume = np.mean(self.volume_history[-6:-3])
            else:
                recent_volume = sum(self.volume_history[-3:]) / 3
                previous_volume = sum(self.volume_history[-6:-3]) / 3
            return "increasing" if recent_volume > previous_volume else "declining"
        except (IndexError, ValueError, ZeroDivisionError):
            return "unknown"
            
    @property
    def age_minutes(self) -> float:
        """Get position age in minutes"""
        return (datetime.now() - self.entry_time).total_seconds() / 60
        
    @property
    def profit_percentage(self) -> float:
        """Get current profit percentage"""
        return (self.current_price / self.entry_price) - 1 if self.entry_price > 0 else 0.0

    def get_position_info(self) -> Dict[str, Any]:
        """Get comprehensive position information including momentum data"""
        return {
            "token_address": self.token_address,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "size": self.size,
            "unrealized_pnl": self.unrealized_pnl,
            "profit_percentage": self.profit_percentage,
            "status": self.status,
            "high_water_mark": self.high_water_mark,
            "entry_time": self.entry_time,
            "age_minutes": self.age_minutes,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop": self.trailing_stop,
            "scaled_take_profits": self.scaled_take_profits,
            "momentum": self._calculate_momentum(),
            "rsi": self._calculate_rsi(),
            "volume_trend": self._get_volume_trend(),
            "momentum_exit_enabled": self.momentum_exit_enabled
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

    def update_price(self, new_price: float, volume: Optional[float] = None) -> None:
        """Update position with new price and volume data"""
        self.current_price = new_price
        if new_price > self.high_water_mark:
            self.high_water_mark = new_price
        self._update_pnl()
        
    def check_exits(self) -> Tuple[bool, str, Optional[float]]:
        """Check if any exit conditions are met using enhanced logic"""
        # Time-based exit (prevent infinite holding)
        age_minutes = (datetime.now() - self.entry_time).total_seconds() / 60
        if age_minutes > 180:  # 3 hours max
            return True, ExitReason.TIME_LIMIT.value, None
            
        # Momentum-based trailing stop
        if self.trailing_stop:
            stop_price = self.high_water_mark * (1 - self.trailing_stop)
            if self.current_price <= stop_price:
                return True, ExitReason.TRAILING_STOP.value, None

        # Dynamic take profits based on momentum
        for tp in self.take_profits:
            if self.current_price >= self.entry_price * (1 + tp['percentage']):
                return True, ExitReason.TAKE_PROFIT.value, self.size * tp['size_percentage']

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

            # Execute closing swap - CRITICAL FIX: Use token_balance not size
            # position.size = SOL invested, position.token_balance = actual tokens to sell
            if position.token_balance <= 0:
                logger.error(f"No token balance to sell for {token_address}: {position.token_balance}")
                return False
                
            success = await self.swap_executor.execute_swap(
                input_token=token_address,
                output_token="So11111111111111111111111111111111111111112",  # SOL
                amount=position.token_balance,  # FIXED: sell all tokens, not SOL amount
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
    
    def update_token_balance(self, token_address: str, token_balance: float) -> bool:
        """Update the token balance for a position - CRITICAL for proper exits"""
        try:
            position = self.positions.get(token_address)
            if not position:
                logger.error(f"Cannot update token balance - position not found: {token_address}")
                return False
            
            old_balance = position.token_balance
            position.token_balance = token_balance
            logger.info(f"Updated token balance for {token_address}: {old_balance} -> {token_balance}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating token balance: {str(e)}")
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