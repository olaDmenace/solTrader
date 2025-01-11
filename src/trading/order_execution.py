from dataclasses import dataclass
from typing import Dict, Optional, Any, List
from datetime import datetime
import logging
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    FAILED = "failed"
    CANCELLED = "cancelled"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class OrderRequest:
    token_address: str
    side: OrderSide
    amount: float
    price: Optional[float] = None  # Required for limit orders
    order_type: OrderType = OrderType.MARKET
    max_slippage: float = 0.01  # 1% default slippage tolerance
    time_in_force: int = 60  # seconds
    timestamp: datetime = datetime.now()

@dataclass
class OrderResult:
    order_id: str
    status: OrderStatus
    filled_amount: float
    filled_price: float
    transaction_hash: Optional[str] = None
    error_message: Optional[str] = None

class OrderExecutor:
    def __init__(self, swap_executor: Any, settings: Any):
        self.swap_executor = swap_executor
        self.settings = settings
        self.active_orders: Dict[str, OrderRequest] = {}
        self.order_results: Dict[str, OrderResult] = {}
        self._monitor_task: Optional[asyncio.Task] = None

    async def execute_order(self, order: OrderRequest) -> Optional[str]:
        """Execute a trade order"""
        try:
            # Generate unique order ID
            order_id = self._generate_order_id()
            self.active_orders[order_id] = order

            # Validate order parameters
            if not await self._validate_order(order):
                await self._handle_failed_order(order_id, "Order validation failed")
                return None

            # Check market conditions
            market_data = await self._get_market_data(order.token_address)
            if not self._check_market_conditions(order, market_data):
                await self._handle_failed_order(order_id, "Market conditions unfavorable")
                return None

            # Calculate execution parameters
            execution_price = await self._calculate_execution_price(order)
            if not execution_price:
                await self._handle_failed_order(order_id, "Could not determine execution price")
                return None

            # Execute the swap
            success = await self.swap_executor.execute_swap(
                input_token="So11111111111111111111111111111111111111112" if order.side == OrderSide.BUY else order.token_address,
                output_token=order.token_address if order.side == OrderSide.BUY else "So11111111111111111111111111111111111111112",
                amount=order.amount,
                slippage=order.max_slippage
            )

            if success:
                await self._handle_successful_order(order_id, execution_price)
                return order_id
            else:
                await self._handle_failed_order(order_id, "Swap execution failed")
                return None

        except Exception as e:
            logger.error(f"Order execution error: {str(e)}")
            if order_id in self.active_orders:
                await self._handle_failed_order(order_id, str(e))
            return None

    async def _validate_order(self, order: OrderRequest) -> bool:
        """Validate order parameters"""
        try:
            # Check minimum order size
            if order.amount < self.settings.MIN_TRADE_SIZE:
                logger.warning(f"Order size {order.amount} below minimum {self.settings.MIN_TRADE_SIZE}")
                return False

            # Check maximum order size
            if order.amount > self.settings.MAX_TRADE_SIZE:
                logger.warning(f"Order size {order.amount} above maximum {self.settings.MAX_TRADE_SIZE}")
                return False

            # Validate slippage
            if order.max_slippage > self.settings.MAX_SLIPPAGE:
                logger.warning(f"Slippage {order.max_slippage} exceeds maximum {self.settings.MAX_SLIPPAGE}")
                return False

            # Additional validations for limit orders
            if order.order_type == OrderType.LIMIT and not order.price:
                logger.warning("Limit order requires price")
                return False

            return True

        except Exception as e:
            logger.error(f"Order validation error: {str(e)}")
            return False

    async def _get_market_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get current market data for the token"""
        try:
            # Get market depth
            depth = await self.swap_executor.get_market_depth(token_address)
            if not depth:
                return None

            # Get current price
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

    def _check_market_conditions(self, order: OrderRequest, market_data: Optional[Dict[str, Any]]) -> bool:
        """Check if market conditions are suitable for execution"""
        if not market_data:
            return False

        # Check liquidity
        min_liquidity = self.settings.MIN_LIQUIDITY
        if market_data["liquidity"] < min_liquidity:
            logger.warning(f"Insufficient liquidity: {market_data['liquidity']} < {min_liquidity}")
            return False

        # Calculate potential price impact
        price_impact = self._calculate_price_impact(order.amount, market_data["depth"])
        if price_impact > self.settings.MAX_PRICE_IMPACT:
            logger.warning(f"Price impact too high: {price_impact}% > {self.settings.MAX_PRICE_IMPACT}%")
            return False

        return True

    async def _calculate_execution_price(self, order: OrderRequest) -> Optional[float]:
        """Calculate the expected execution price"""
        try:
            if order.order_type == OrderType.MARKET:
                price_data = await self.swap_executor.get_price(order.token_address)
                if not price_data:
                    return None
                return float(price_data.get("price", 0))
            else:
                return order.price  # For limit orders, use the specified price

        except Exception as e:
            logger.error(f"Error calculating execution price: {str(e)}")
            return None

    def _calculate_price_impact(self, amount: float, depth: Dict[str, Any]) -> float:
        """Calculate expected price impact percentage"""
        try:
            total_liquidity = sum(float(level["size"]) for level in depth.get("bids", []))
            return (amount / total_liquidity * 100) if total_liquidity > 0 else 100.0
        except Exception as e:
            logger.error(f"Error calculating price impact: {str(e)}")
            return 100.0

    async def _handle_successful_order(self, order_id: str, execution_price: float) -> None:
        """Handle successful order execution"""
        order = self.active_orders.get(order_id)
        if not order:
            return

        self.order_results[order_id] = OrderResult(
            order_id=order_id,
            status=OrderStatus.FILLED,
            filled_amount=order.amount,
            filled_price=execution_price
        )
        del self.active_orders[order_id]
        logger.info(f"Order {order_id} executed successfully at {execution_price}")

    async def _handle_failed_order(self, order_id: str, error_message: str) -> None:
        """Handle failed order execution"""
        order = self.active_orders.get(order_id)
        if not order:
            return

        self.order_results[order_id] = OrderResult(
            order_id=order_id,
            status=OrderStatus.FAILED,
            filled_amount=0,
            filled_price=0,
            error_message=error_message
        )
        del self.active_orders[order_id]
        logger.error(f"Order {order_id} failed: {error_message}")

    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"order_{timestamp}"

    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """Get the current status of an order"""
        return self.order_results.get(order_id)

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get all active orders"""
        return [
            {
                "order_id": order_id,
                "token": order.token_address,
                "side": order.side,
                "amount": order.amount,
                "type": order.order_type,
                "timestamp": order.timestamp
            }
            for order_id, order in self.active_orders.items()
        ]