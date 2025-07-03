import asyncio
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MarketCondition:
    price: float
    volume_24h: float
    liquidity: float
    volatility: float
    buy_pressure: float
    price_impact: float

class EnhancedOrderExecution:
    def __init__(self, jupiter_client, settings):
        self.jupiter = jupiter_client
        self.settings = settings
        self.execution_attempts = 3
        self.price_check_interval = 1.0  # seconds
        self.max_execution_time = 30.0  # seconds

    async def analyze_market_condition(self, token_address: str, size: float) -> Optional[MarketCondition]:
        try:
            quote = await self.jupiter.get_quote(token_address)
            depth = await self.jupiter.get_market_depth(token_address)
            
            if not quote or not depth:
                return None

            current_price = float(quote.get('price', 0))
            volume_24h = float(depth.get('volume24h', 0))
            liquidity = float(depth.get('liquidity', 0))

            # Calculate buy pressure from order book
            bids_volume = sum(float(bid.get('size', 0)) for bid in depth.get('bids', []))
            asks_volume = sum(float(ask.get('size', 0)) for ask in depth.get('asks', []))
            buy_pressure = bids_volume / asks_volume if asks_volume > 0 else 0

            # Calculate price impact
            price_impact = self._calculate_price_impact(depth, size)

            # Calculate volatility from recent trades
            recent_trades = depth.get('recent_trades', [])
            if recent_trades:
                prices = [float(trade.get('price', 0)) for trade in recent_trades]
                volatility = float(np.std(prices) / np.mean(prices)) if prices else 0
            else:
                volatility = 0

            return MarketCondition(
                price=current_price,
                volume_24h=volume_24h,
                liquidity=liquidity,
                volatility=volatility,
                buy_pressure=buy_pressure,
                price_impact=price_impact
            )

        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            return None

    def _calculate_price_impact(self, depth: Dict, size: float) -> float:
        total_liquidity = sum(float(bid.get('size', 0)) for bid in depth.get('bids', []))
        return (size / total_liquidity * 100) if total_liquidity > 0 else 100.0

    async def execute_order(self, 
                          token_address: str,
                          side: str,
                          size: float,
                          max_slippage: float = 0.01) -> Tuple[bool, Optional[str]]:
        """
        Execute order with optimal routing and slippage protection
        """
        try:
            # Initial market check
            condition = await self.analyze_market_condition(token_address, size)
            if not condition or not self._validate_market_condition(condition):
                return False, "Market conditions unfavorable"

            # Price monitoring
            initial_price = condition.price
            async def monitor_price():
                while True:
                    current_condition = await self.analyze_market_condition(token_address, size)
                    if current_condition:
                        price_change = abs(current_condition.price - initial_price) / initial_price
                        if price_change > max_slippage:
                            return False
                    await asyncio.sleep(self.price_check_interval)

            # Start price monitoring
            monitor_task = asyncio.create_task(monitor_price())

            # Execute with retries
            for attempt in range(self.execution_attempts):
                try:
                    # Get optimal route
                    route = await self.jupiter.get_route(
                        token_address,
                        size,
                        side,
                        max_slippage
                    )

                    if not route:
                        continue

                    # Execute swap
                    tx_hash = await self.jupiter.execute_swap(route)
                    if tx_hash:
                        monitor_task.cancel()
                        return True, tx_hash

                except Exception as e:
                    logger.error(f"Execution attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.execution_attempts - 1:
                        monitor_task.cancel()
                        return False, str(e)
                    await asyncio.sleep(1)

            monitor_task.cancel()
            return False, "Max retries exceeded"

        except Exception as e:
            logger.error(f"Order execution error: {str(e)}")
            return False, str(e)

    def _validate_market_condition(self, condition: MarketCondition) -> bool:
        return all([
            condition.volume_24h >= self.settings.MIN_VOLUME_24H,
            condition.liquidity >= self.settings.MIN_LIQUIDITY,
            condition.volatility <= self.settings.MAX_VOLATILITY,
            condition.price_impact <= self.settings.MAX_PRICE_IMPACT
        ])