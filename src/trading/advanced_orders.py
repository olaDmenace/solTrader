# src/trading/advanced_orders.py

from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit" 
    TRAILING_STOP = "trailing_stop"
    DCA = "dca"
    SCALED = "scaled"

@dataclass
class OrderConfig:
    trail_percent: float = 0.02
    dca_intervals: int = 5
    scale_levels: int = 3
    activation_price: float = 0.0

class AdvancedOrderManager:
    def __init__(self, swap_executor: Any, settings: Dict[str, Any]):
        self.swap_executor = swap_executor
        self.settings = settings
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self._monitor_task: Optional[asyncio.Task] = None

    async def _get_current_price(self, token_address: str) -> float:
        """Get current price for token"""
        try:
            return await self.swap_executor.get_price(token_address)
        except Exception as e:
            logger.error(f"Failed to get price for {token_address}: {e}")
            raise

    async def _process_scaled_order(
        self, 
        token_address: str, 
        order: Dict[str, Any], 
        current_price: float
    ) -> None:
        """Process scaled order based on price levels"""
        executed_levels = order.get('executed_levels', [])

        for i, price_level in enumerate(order['levels']):
            if i not in executed_levels and current_price <= price_level:
                try:
                    await self.swap_executor.execute_order(
                        token_address=token_address,
                        order_type=OrderType.LIMIT,
                        side="buy",
                        amount=order['size_per_level'],
                        price=price_level
                    )

                    executed_levels.append(i)
                    order['executed_levels'] = executed_levels
                    logger.info(
                        f"Executed scaled order level {i+1}/{len(order['levels'])} "
                        f"for {token_address} at {price_level:.4f}"
                    )

                    # Remove order if all levels executed
                    if len(executed_levels) >= len(order['levels']):
                        del self.active_orders[token_address]
                        logger.info(f"Completed all scale levels for {token_address}")

                except Exception as e:
                    logger.error(f"Failed to execute scaled order: {e}")

    async def _process_dca_order(
        self, 
        token_address: str, 
        order: Dict[str, Any]
    ) -> None:
        """Process DCA order based on schedule"""
        current_time = datetime.now()
        next_execution = order['schedule'][order.get('executed_intervals', 0)]

        if current_time >= next_execution:
            try:
                await self.swap_executor.execute_order(
                    token_address=token_address,
                    order_type=OrderType.MARKET,
                    side="buy",
                    amount=order['amount_per_trade']
                )

                order['executed_intervals'] = order.get('executed_intervals', 0) + 1
                logger.info(
                    f"Executed DCA order {order['executed_intervals']}/{len(order['schedule'])} "
                    f"for {token_address}"
                )

                # Remove order if all intervals executed
                if order['executed_intervals'] >= len(order['schedule']):
                    del self.active_orders[token_address]
                    logger.info(f"Completed all DCA intervals for {token_address}")

            except Exception as e:
                logger.error(f"Failed to execute DCA order: {e}")

    async def _update_trailing_stop(
        self, 
        token_address: str, 
        order: Dict[str, Any], 
        current_price: float
    ) -> None:
        """Update trailing stop order based on current price"""
        if current_price > order['high_price']:
            # Update high price and stop price if current price is higher
            old_stop = order['stop_price']
            order['high_price'] = current_price
            order['stop_price'] = current_price * (1 - order['trail_percent'])
            logger.info(
                f"Updated trailing stop for {token_address}: "
                f"high_price={current_price:.4f}, "
                f"stop_price={order['stop_price']:.4f} "
                f"(was {old_stop:.4f})"
            )

        elif current_price <= order['stop_price']:
            # Execute stop order
            try:
                await self.swap_executor.execute_order(
                    token_address=token_address,
                    order_type=OrderType.MARKET,
                    side="sell"
                )
                logger.info(
                    f"Executed trailing stop for {token_address} at {current_price:.4f}"
                )
                del self.active_orders[token_address]
            except Exception as e:
                logger.error(f"Failed to execute trailing stop: {e}")

    async def _check_orders(self) -> None:
        """Check and update status of all active orders"""
        for token_address, order in list(self.active_orders.items()):
            try:
                current_price = await self._get_current_price(token_address)

                if order['type'] == OrderType.TRAILING_STOP:
                    await self._update_trailing_stop(token_address, order, current_price)
                elif order['type'] == OrderType.DCA:
                    await self._process_dca_order(token_address, order)
                elif order['type'] == OrderType.SCALED:
                    await self._process_scaled_order(token_address, order, current_price)

            except Exception as e:
                logger.error(f"Error checking order {token_address}: {e}")

    async def _monitor_loop(self) -> None:
        """Monitor active orders"""
        while True:
            try:
                await self._check_orders()
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Order monitor error: {str(e)}")
                await asyncio.sleep(5)  # Back off on error

    async def start_monitoring(self) -> None:
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    def _create_dca_schedule(self, intervals: int, interval_time: float) -> List[datetime]:
        """Create DCA order schedule"""
        schedule = []
        now = datetime.now()
        for i in range(intervals):
            schedule.append(now + timedelta(hours=interval_time * i))
        return schedule

    async def place_trailing_stop(
        self, 
        token_address: str, 
        trail_percent: float
    ) -> Dict[str, Any]:
        """Place trailing stop order"""
        current_price = await self._get_current_price(token_address)
        order = {
            'type': OrderType.TRAILING_STOP,
            'token_address': token_address,
            'trail_percent': trail_percent,
            'high_price': current_price,
            'stop_price': current_price * (1 - trail_percent)
        }
        self.active_orders[token_address] = order
        return order

    async def place_dca_order(
        self, 
        token: str, 
        total_amount: float, 
        intervals: int, 
        duration_hours: float = 24
    ) -> Dict[str, Any]:
        """Place DCA order"""
        if intervals <= 0:
            raise ValueError("Intervals must be greater than 0")

        interval_size = total_amount / intervals
        interval_time = duration_hours / intervals

        schedule = self._create_dca_schedule(intervals, interval_time)
        order = {
            'type': OrderType.DCA,
            'token': token,
            'amount_per_trade': interval_size,
            'schedule': schedule,
            'executed_intervals': 0
        }
        self.active_orders[token] = order
        return order

    def _calculate_scale_levels(
        self, 
        price_range: Tuple[float, float], 
        levels: int
    ) -> List[float]:
        """Calculate price levels for scaled orders"""
        low, high = price_range
        if low >= high:
            raise ValueError("Price range low must be less than high")
        if levels < 2:
            raise ValueError("Must have at least 2 levels")

        step = (high - low) / (levels - 1)
        return [low + (step * i) for i in range(levels)]

    async def place_scaled_order(
        self, 
        token: str, 
        total_size: float, 
        levels: int, 
        price_range: Tuple[float, float]
    ) -> Dict[str, Any]:
        """Place scaled order"""
        level_size = total_size / levels
        level_prices = self._calculate_scale_levels(price_range, levels)

        order = {
            'type': OrderType.SCALED,
            'token': token,
            'size_per_level': level_size,
            'levels': level_prices,
            'executed_levels': []
        }
        self.active_orders[token] = order
        return order