"""
Enhanced swap executor with MEV protection and private transaction handling.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
import base58
import asyncio
from datetime import datetime, timedelta
from solana.transaction import Transaction
from solders.hash import Hash
import random

logger = logging.getLogger(__name__)

@dataclass
class SwapRoute:
    """Represents a swap route with execution parameters"""
    input_mint: str
    output_mint: str
    amount: Decimal
    slippage: float
    platforms: List[str]
    expected_output: Decimal
    priority: int = 1
    max_delay: float = 2.0  # seconds

@dataclass
class SwapResult:
    """Result of a swap execution"""
    success: bool
    signature: Optional[str]
    output_amount: Optional[Decimal]
    execution_time: float
    error: Optional[str] = None
    route_info: Optional[Dict[str, Any]] = None

@dataclass
class PrivateSwapConfig:
    """Configuration for private swap execution"""
    use_private_rpc: bool = True
    use_bundle: bool = False
    randomize_timing: bool = True
    min_delay: float = 0.1
    max_delay: float = 2.0
    priority_fee_multiplier: float = 1.2

class SwapExecutor:
    """Enhanced swap executor with MEV protection"""

    def __init__(self, jupiter_client: Any, wallet: Any):
        self.jupiter = jupiter_client
        self.wallet = wallet
        self.private_rpcs: List[str] = []
        self.recent_swaps: List[Dict[str, Any]] = []
        self.max_retries: int = 3
        self.retry_delay: float = 1.0
        self.default_config = PrivateSwapConfig()
        
        # Performance metrics
        self.total_swaps: int = 0
        self.successful_swaps: int = 0
        self.failed_swaps: int = 0
        self.avg_execution_time: float = 0.0

    async def execute_swap(
        self,
        input_token: str,
        output_token: str,
        amount: float,
        slippage: float = 0.01,
        config: Optional[PrivateSwapConfig] = None
    ) -> Optional[str]:
        """Execute swap with MEV protection"""
        try:
            start_time = asyncio.get_event_loop().time()
            config = config or self.default_config

            # Get optimal route
            route = await self._get_optimal_route(
                input_token,
                output_token,
                Decimal(str(amount)),
                slippage
            )
            if not route:
                return None

            # Add random delay if configured
            if config.randomize_timing:
                delay = random.uniform(config.min_delay, config.max_delay)
                await asyncio.sleep(delay)

            # Get quote with protection measures
            quote = await self._get_protected_quote(route, config)
            if not quote:
                return None

            # Build and execute protected transaction
            result = await self._execute_protected_swap(quote, config)
            
            # Update metrics
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(result.success, execution_time)

            if result.success and result.signature:
                self._record_successful_swap(route, result)
                return result.signature

            return None

        except Exception as e:
            logger.error(f"Swap execution failed: {str(e)}")
            self.failed_swaps += 1
            return None

    async def _get_optimal_route(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal,
        slippage: float
    ) -> Optional[SwapRoute]:
        """Get optimal swap route with MEV protection considerations"""
        try:
            # Get all possible routes
            routes = await self.jupiter.get_routes(
                input_mint=input_token,
                output_mint=output_token,
                amount=str(int(amount * Decimal("1e9"))),
                slippageBps=int(slippage * 10000)
            )

            if not routes or 'data' not in routes:
                return None

            # Filter and sort routes by safety and efficiency
            safe_routes = self._filter_safe_routes(routes['data'])
            if not safe_routes:
                return None

            # Select optimal route
            best_route = safe_routes[0]
            return SwapRoute(
                input_mint=input_token,
                output_mint=output_token,
                amount=amount,
                slippage=slippage,
                platforms=best_route.get('platforms', []),
                expected_output=Decimal(str(best_route.get('outAmount', 0))) / Decimal("1e9")
            )

        except Exception as e:
            logger.error(f"Error getting optimal route: {str(e)}")
            return None

    def _filter_safe_routes(self, routes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter routes based on safety criteria"""
        try:
            # Calculate safety scores for routes
            scored_routes = []
            for route in routes:
                safety_score = self._calculate_route_safety(route)
                if safety_score >= 0.7:  # Only consider reasonably safe routes
                    scored_routes.append((route, safety_score))

            # Sort by safety score and efficiency
            return [
                route for route, score in sorted(
                    scored_routes,
                    key=lambda x: (x[1], -float(x[0].get('priceImpactPct', 100)))
                )
            ]

        except Exception as e:
            logger.error(f"Error filtering routes: {str(e)}")
            return []

    def _calculate_route_safety(self, route: Dict[str, Any]) -> float:
        """Calculate safety score for a route"""
        try:
            score = 1.0
            
            # Penalize for high price impact
            price_impact = float(route.get('priceImpactPct', 0))
            if price_impact > 1.0:
                score *= (1 - min(price_impact / 10, 0.5))
                
            # Penalize for too many hops
            num_hops = len(route.get('marketInfos', []))
            if num_hops > 2:
                score *= (1 - ((num_hops - 2) * 0.1))
                
            # Penalize less established platforms
            platforms = route.get('platforms', [])
            if not any(p in ['Raydium', 'Orca', 'Serum'] for p in platforms):
                score *= 0.8
                
            return max(0.0, min(score, 1.0))

        except Exception as e:
            logger.error(f"Error calculating route safety: {str(e)}")
            return 0.0

    async def _get_protected_quote(
        self,
        route: SwapRoute,
        config: PrivateSwapConfig
    ) -> Optional[Dict[str, Any]]:
        """Get quote with protection measures"""
        try:
            quote = await self.jupiter.get_quote(
                input_mint=route.input_mint,
                output_mint=route.output_mint,
                amount=str(int(route.amount * Decimal("1e9"))),
                slippageBps=int(route.slippage * 10000)
            )

            if not quote:
                return None

            # Add protection measures
            quote['maxTimestamp'] = int(
                (datetime.now() + timedelta(seconds=config.max_delay)).timestamp()
            )
            
            if config.use_bundle:
                quote['bundle'] = True
                quote['bundlePriority'] = route.priority

            return quote

        except Exception as e:
            logger.error(f"Error getting protected quote: {str(e)}")
            return None

    async def _execute_protected_swap(
        self,
        quote: Dict[str, Any],
        config: PrivateSwapConfig
    ) -> SwapResult:
        """Execute swap with protection measures"""
        try:
            # Build transaction
            swap_ix = await self.jupiter.get_swap_ix(quote)
            if not swap_ix:
                return SwapResult(
                    success=False,
                    signature=None,
                    output_amount=None,
                    execution_time=0.0,
                    error="Failed to get swap instruction"
                )

            # Create transaction
            recent_blockhash = await self.wallet.get_recent_blockhash()
            tx = Transaction()
            tx.recent_blockhash = Hash.from_string(recent_blockhash)
            
            # Add priority fee if configured
            if config.priority_fee_multiplier > 1.0:
                priority_fee = await self._calculate_priority_fee(quote)
                tx.recent_blockhash = Hash.from_string(
                    await self.wallet.get_recent_blockhash(priority_fee)
                )

            tx.add(swap_ix)

            # Execute with retries
            start_time = asyncio.get_event_loop().time()
            signature = await self._execute_with_retries(tx, config)
            execution_time = asyncio.get_event_loop().time() - start_time

            if not signature:
                return SwapResult(
                    success=False,
                    signature=None,
                    output_amount=None,
                    execution_time=execution_time,
                    error="Transaction failed"
                )

            # Wait for confirmation
            confirmed = await self._confirm_transaction(signature)
            
            if not confirmed:
                return SwapResult(
                    success=False,
                    signature=signature,
                    output_amount=None,
                    execution_time=execution_time,
                    error="Transaction not confirmed"
                )

            return SwapResult(
                success=True,
                signature=signature,
                output_amount=Decimal(str(quote.get('outAmount', 0))) / Decimal("1e9"),
                execution_time=execution_time,
                route_info=quote.get('route')
            )

        except Exception as e:
            logger.error(f"Error executing protected swap: {str(e)}")
            return SwapResult(
                success=False,
                signature=None,
                output_amount=None,
                execution_time=0.0,
                error=str(e)
            )

    async def _execute_with_retries(
        self,
        tx: Transaction,
        config: PrivateSwapConfig
    ) -> Optional[str]:
        """Execute transaction with retries"""
        for attempt in range(self.max_retries):
            try:
                # Use private RPC if configured
                if config.use_private_rpc and self.private_rpcs:
                    rpc_url = random.choice(self.private_rpcs)
                    signature = await self.wallet.sign_and_send_transaction(
                        tx, rpc_url=rpc_url
                    )
                else:
                    signature = await self.wallet.sign_and_send_transaction(tx)
                    
                if signature:
                    return signature

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        return None

    async def _confirm_transaction(
        self,
        signature: str,
        max_retries: int = 30
    ) -> bool:
        """Confirm transaction with timeout"""
        for _ in range(max_retries):
            try:
                status = await self.wallet.get_transaction_status(signature)
                if status == 'confirmed':
                    return True
                elif status == 'failed':
                    return False
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error checking transaction status: {str(e)}")
                return False
        return False

    async def _calculate_priority_fee(self, quote: Dict[str, Any]) -> int:
        """Calculate competitive priority fee"""
        try:
            # Get recent priority fees
            recent_fees = await self.jupiter.get_recent_priority_fees()
            if not recent_fees:
                return 5000  # Default priority fee

            # Calculate competitive fee
            avg_fee = sum(fee.get('priorityFee', 0) for fee in recent_fees) / len(recent_fees)
            return int(avg_fee * self.default_config.priority_fee_multiplier)

        except Exception as e:
            logger.error(f"Error calculating priority fee: {str(e)}")
            return 5000

    def _update_metrics(self, success: bool, execution_time: float) -> None:
        """Update performance metrics"""
        self.total_swaps += 1
        if success:
            self.successful_swaps += 1
        else:
            self.failed_swaps += 1

        # Update average execution time
        self.avg_execution_time = (
            (self.avg_execution_time * (self.total_swaps - 1) + execution_time)
            / self.total_swaps
        )

    def _record_successful_swap(self, route: SwapRoute, result: SwapResult) -> None:
        """Record successful swap details"""
        self.recent_swaps.append({
            'timestamp': datetime.now().isoformat(),
            'input_token': route.input_mint,
            'output_token': route.output_mint,
            'amount': float(route.amount),
            'output_amount': float(result.output_amount) if result.output_amount else 0.0,
            'execution_time': result.execution_time,
            'route_info': result.route_info
        })

        # Keep only recent history
        if len(self.recent_swaps) > 100:
            self.recent_swaps = self.recent_swaps[-100:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get swap executor performance metrics"""
        return {
            'total_swaps': self.total_swaps,
            'successful_swaps': self.successful_swaps,
            'failed_swaps': self.failed_swaps,
            'success_rate': (
                self.successful_swaps / self.total_swaps * 100 
                if self.total_swaps > 0 else 0
            ),
            'avg_execution_time': self.avg_execution_time,
            'recent_swaps': self.recent_swaps[-10:]  # Last 10 swaps
        }