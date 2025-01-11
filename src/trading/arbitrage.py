# src/trading/arbitrage.py

from dataclasses import dataclass
from typing import AsyncIterator, List, Dict, Optional, Any, AsyncContextManager, Tuple
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from decimal import Decimal
import async_timeout

logger = logging.getLogger(__name__)

@dataclass
class ArbitrageOpportunity:
    buy_exchange: str
    sell_exchange: str
    token: str
    profit_percent: float
    size: float
    buy_price: float
    sell_price: float
    routes: Dict[str, Any]
    timestamp: float

class ArbitrageManager:
    def __init__(
        self, 
        exchanges: Dict[str, Any],
        min_profit: float = 0.01,
        max_size: Optional[float] = None
    ):
        self.exchanges = exchanges
        self.min_profit = min_profit
        self.max_size = max_size
        self.gas_costs = self._initialize_gas_costs()
        self._lock = asyncio.Lock()

    def _initialize_gas_costs(self) -> Dict[str, float]:
        """Initialize estimated gas costs per exchange"""
        return {
            'jupiter': 0.000005,
            'raydium': 0.000004,
            'orca': 0.000006
        }

    async def _get_exchange_price(self, exchange: Any, token: str) -> Optional[float]:
        """Get price from single exchange with timeout"""
        try:
            async with async_timeout.timeout(2.0) as timeout:
                return await exchange.get_price(token)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting price from {exchange.name}")
            return None
        except Exception as e:
            logger.error(f"Error getting price from {exchange.name}: {e}")
            return None

    @asynccontextmanager
    async def _create_transaction_group(self) -> AsyncIterator[Any]:
        """Create a transaction group for atomic execution"""
        group = None
        try:
            # Start transaction group
            group = await self.exchanges['default'].create_transaction_group()
            yield group
        finally:
            # Cleanup if needed
            if group is not None:
                await group.cleanup()

    async def _place_order(
        self,
        exchange: str,
        token: str,
        size: float,
        side: str,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Place order on specified exchange"""
        try:
            exchange_instance = self.exchanges[exchange]
            order_type = "LIMIT" if price else "MARKET"

            order = await exchange_instance.create_order(
                token=token,
                size=size,
                side=side,
                type=order_type,
                price=price
            )

            logger.info(
                f"Placed {order_type} {side} order on {exchange}: "
                f"size={size}, price={price if price else 'market'}"
            )
            return order

        except Exception as e:
            logger.error(f"Failed to place order on {exchange}: {e}")
            raise

    async def _execute_atomic(
        self,
        group: Any,
        transactions: List[Dict[str, Any]]
    ) -> bool:
        """Execute transactions atomically"""
        try:
            # Add transactions to group
            for tx in transactions:
                await group.add_transaction(tx)

            # Execute group
            result = await group.execute()

            if result.success:
                logger.info("Atomic execution successful")
                return True
            else:
                logger.error(f"Atomic execution failed: {result.error}")
                return False

        except Exception as e:
            logger.error(f"Failed to execute atomic transaction: {e}")
            return False

    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """Execute arbitrage trades"""
        async with self._lock:
            try:
                async with self._create_transaction_group() as group:
                    buy_tx = await self._place_order(
                        opportunity.buy_exchange,
                        opportunity.token,
                        opportunity.size,
                        "buy",
                        opportunity.buy_price
                    )
                    sell_tx = await self._place_order(
                        opportunity.sell_exchange,
                        opportunity.token,
                        opportunity.size,
                        "sell",
                        opportunity.sell_price
                    )
                    return await self._execute_atomic(group, [buy_tx, sell_tx])
            except Exception as e:
                logger.error(f"Failed to execute arbitrage: {e}")
                return False

    def _calculate_arbitrage(
        self, 
        token: str,
        prices: Dict[str, float]
    ) -> Optional[ArbitrageOpportunity]:
        """Calculate arbitrage opportunity for given prices"""
        try:
            buy_exchange = min(prices.items(), key=lambda x: x[1])
            sell_exchange = max(prices.items(), key=lambda x: x[1])

            buy_price = Decimal(str(buy_exchange[1]))
            sell_price = Decimal(str(sell_exchange[1]))

            # Calculate profit after gas costs
            gas_cost = self.gas_costs[buy_exchange[0]] + self.gas_costs[sell_exchange[0]]
            profit = (sell_price - buy_price) / buy_price - Decimal(str(gas_cost))

            if profit > Decimal(str(self.min_profit)):
                size = self._calculate_optimal_size(buy_price, sell_price)

                return ArbitrageOpportunity(
                    buy_exchange=buy_exchange[0],
                    sell_exchange=sell_exchange[0],
                    token=token,
                    profit_percent=float(profit),
                    size=size,
                    buy_price=float(buy_price),
                    sell_price=float(sell_price),
                    routes=self._get_routes(buy_exchange[0], sell_exchange[0]),
                    timestamp=time.time()
                )
        except Exception as e:
            logger.error(f"Error calculating arbitrage: {e}")

        return None

    def _calculate_optimal_size(self, buy_price: Decimal, sell_price: Decimal) -> float:
        """Calculate optimal trade size based on prices and constraints"""
        if self.max_size is None:
            return float(sell_price * Decimal('0.1'))  # Default to 10% of sell price
        return min(float(self.max_size), float(sell_price * Decimal('0.1')))

    def _get_routes(self, buy_exchange: str, sell_exchange: str) -> Dict[str, Any]:
        """Get routing information for exchanges"""
        return {
            'buy': self.exchanges[buy_exchange].get_routing_info(),
            'sell': self.exchanges[sell_exchange].get_routing_info()
        }