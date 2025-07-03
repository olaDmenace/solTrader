from __future__ import annotations
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class OrderResult:
    filled_amount: float
    filled_price: float
    transaction_hash: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class OrderDetails:
    order_id: str
    token_address: str
    order_type: str
    side: str
    size: float
    price: float
    status: str
    timestamp: datetime
    slippage: float
    max_retries: int = 3
    retries: int = 0
    error: Optional[str] = None
    tx_hash: Optional[str] = None
    result: Optional[OrderResult] = None

class OrderManager:
    def __init__(self, swap_executor: Any, wallet: Any, settings: Any):
        self.swap_executor = swap_executor
        self.wallet = wallet
        self.settings = settings
        self.active_orders: Dict[str, OrderDetails] = {}
        self.order_history: List[OrderDetails] = []
        self._monitor_task: Optional[asyncio.Task] = None

    def generate_order_id(self) -> str:
        """Generate unique order ID"""
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"order_{timestamp}"

    def calculate_price_impact(self, depth: Dict[str, Any], size: float) -> float:
        """Calculate price impact of order"""
        try:
            total_liquidity = sum(float(bid['size']) for bid in depth.get('bids', []))
            return (size / total_liquidity * 100) if total_liquidity > 0 else 100.0
        except Exception:
            return 100.0

    async def validate_order(self, order: OrderDetails) -> bool:
        """Validate order parameters"""
        try:
            # Check balance
            balance = await self.wallet.get_balance()
            if not balance or balance < order.size:
                return False

            # Verify token
            if not await self.verify_token(order.token_address):
                return False

            # Check market order specific conditions
            if order.order_type == OrderType.MARKET:
                return await self.validate_market_order(order)

            return True

        except Exception as e:
            logger.error(f"Order validation failed: {str(e)}")
            return False

    async def submit_order(
        self,
        token_address: str,
        side: str,
        size: float,
        price: float,
        order_type: str = OrderType.MARKET
    ) -> Optional[str]:
        try:
            # Validate basic parameters
            if not self.validate_basic_params(size, side, order_type):
                return None

            # Create order
            order_id = self.generate_order_id()
            order = OrderDetails(
                order_id=order_id,
                token_address=token_address,
                order_type=order_type,
                side=side,
                size=size,
                price=price,
                status=OrderStatus.PENDING,
                timestamp=datetime.now(),
                slippage=self.settings.SLIPPAGE_TOLERANCE
            )

            # Check market conditions
            market_data = await self.get_market_data(token_address)
            if not self.check_market_conditions(order, market_data):
                logger.warning("Market conditions not suitable for order execution")
                return None

            if not await self.validate_order(order):
                return None

            success = await self.execute_order(order)
            if not success:
                return None

            self.active_orders[order_id] = order
            return order_id

        except Exception as e:
            logger.error(f"Order submission failed: {str(e)}")
            return None

    async def validate_market_order(self, order: OrderDetails) -> bool:
        """Validate market order conditions"""
        try:
            depth = await self.swap_executor.get_market_depth(order.token_address)
            if not depth:
                return False

            impact = self.calculate_price_impact(depth, order.size)
            return impact <= self.settings.MAX_PRICE_IMPACT

        except Exception as e:
            logger.error(f"Market order validation failed: {str(e)}")
            return False

    async def verify_token(self, token_address: str) -> bool:
        """Verify token exists and is valid"""
        try:
            info = await self.swap_executor.get_token_info(token_address)
            return bool(info and info.get('mint') == token_address)
        except Exception:
            return False

    def validate_basic_params(self, size: float, side: str, order_type: str) -> bool:
        """Validate basic order parameters"""
        try:
            if size <= 0:
                logger.error("Order size must be positive")
                return False

            if side not in [OrderSide.BUY, OrderSide.SELL]:
                logger.error(f"Invalid order side: {side}")
                return False

            if order_type not in [OrderType.MARKET, OrderType.LIMIT]:
                logger.error(f"Invalid order type: {order_type}")
                return False

            return True
        except Exception as e:
            logger.error(f"Error validating basic parameters: {str(e)}")
            return False

    async def get_market_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get current market data for the token"""
        try:
            depth = await self.swap_executor.get_market_depth(token_address)
            if not depth:
                return None

            price_data = await self.swap_executor.get_price(token_address)
            if not price_data:
                return None

            return {
                "depth": depth,
                "price": price_data.get("price", 0),
                "liquidity": depth.get("totalLiquidity", 0)
            }

        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return None

    def check_market_conditions(self, order: OrderDetails, market_data: Optional[Dict[str, Any]]) -> bool:
        """Check if market conditions are suitable for execution"""
        if not market_data:
            return False

        # Check minimum liquidity
        if market_data["liquidity"] < self.settings.MIN_LIQUIDITY:
            logger.warning(f"Insufficient liquidity: {market_data['liquidity']} < {self.settings.MIN_LIQUIDITY}")
            return False

        # Check price impact
        impact = self.calculate_price_impact(market_data["depth"], order.size)
        if impact > self.settings.MAX_PRICE_IMPACT:
            logger.warning(f"Price impact too high: {impact}% > {self.settings.MAX_PRICE_IMPACT}%")
            return False

        return True

    async def execute_order(self, order: OrderDetails) -> bool:
        """Execute the order"""
        try:
            order.status = OrderStatus.SUBMITTED

            if order.order_type == OrderType.MARKET:
                # Get fresh price before execution
                execution_price = await self.get_execution_price(order)
                if not execution_price:
                    order.error = "Could not determine execution price"
                    return False

                success = await self.swap_executor.execute_swap(
                    "SOL" if order.side == OrderSide.BUY else order.token_address,
                    order.token_address if order.side == OrderSide.BUY else "SOL",
                    order.size,
                    order.slippage
                )

                if success:
                    order.status = OrderStatus.CONFIRMED
                    order.result = OrderResult(
                        filled_amount=order.size,
                        filled_price=execution_price
                    )
                    return True

            order.status = OrderStatus.FAILED
            return False

        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error = str(e)
            return False

    async def get_execution_price(self, order: OrderDetails) -> Optional[float]:
        """Get current execution price for the order"""
        try:
            price_data = await self.swap_executor.get_price(order.token_address)
            if not price_data or "price" not in price_data:
                return None
            return float(price_data["price"])
        except Exception as e:
            logger.error(f"Error getting execution price: {str(e)}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            if order_id not in self.active_orders:
                logger.debug(f"Order ID {order_id} not found in active orders.")
                return False

            order = self.active_orders[order_id]
            if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.CONFIRMED]:
                logger.debug(f"Order ID {order_id} has invalid status: {order.status}.")
                return False

            order.status = OrderStatus.CANCELLED
            self.move_to_history(order_id)
            logger.debug(f"Order ID {order_id} cancelled successfully.")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return False

    async def get_order_status(self, order_id: str) -> Optional[str]:
        """Get the current status of an order"""
        if order_id in self.active_orders:
            return self.active_orders[order_id].status
        return None

    def move_to_history(self, order_id: str) -> None:
        """Move order from active to history"""
        if order_id in self.active_orders:
            self.order_history.append(self.active_orders[order_id])
            del self.active_orders[order_id]

    def get_order_summary(self) -> Dict[str, Any]:
        """Get summary of all orders"""
        return {
            "active_orders": len(self.active_orders),
            "completed_orders": len(self.order_history),
            "orders": [
                {
                    "order_id": order.order_id,
                    "token": order.token_address,
                    "side": order.side,
                    "size": order.size,
                    "status": order.status,
                    "result": asdict(order.result) if order.result else None
                }
                for order in self.active_orders.values()
            ]
        }