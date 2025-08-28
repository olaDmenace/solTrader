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
            logger.info(f"[SWAP_ENTRY] Starting swap execution")
            logger.info(f"[SWAP_ENTRY] Input: {input_token[:8]}... -> Output: {output_token[:8]}...")
            logger.info(f"[SWAP_ENTRY] Amount: {amount} SOL, Slippage: {slippage}")
            
            start_time = asyncio.get_event_loop().time()
            config = config or self.default_config

            # Get optimal route
            logger.info(f"[SWAP_STEP1] Getting optimal route...")
            route = await self._get_optimal_route(
                input_token,
                output_token,
                Decimal(str(amount)),
                slippage
            )
            if not route:
                logger.error(f"[SWAP_STEP1] FAILED - No route available")
                return None
            logger.info(f"[SWAP_STEP1] SUCCESS - Route obtained")

            # Add random delay if configured
            if config.randomize_timing:
                delay = random.uniform(config.min_delay, config.max_delay)
                logger.info(f"[SWAP_DELAY] Adding random delay: {delay:.2f}s")
                await asyncio.sleep(delay)

            # Get quote with protection measures
            logger.info(f"[SWAP_STEP2] Getting protected quote...")
            quote = await self._get_protected_quote(route, config)
            if not quote:
                logger.error(f"[SWAP_STEP2] FAILED - No quote available")
                return None
            logger.info(f"[SWAP_STEP2] SUCCESS - Quote obtained: {quote.get('outAmount', 'N/A')}")

            # Build and execute protected transaction
            logger.info(f"[SWAP_STEP3] Executing protected swap...")
            result = await self._execute_protected_swap(quote, config)
            
            # Update metrics
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(result.success, execution_time)

            if result.success and result.signature:
                logger.info(f"[SWAP_SUCCESS] Swap completed: {result.signature}")
                self._record_successful_swap(route, result)
                return result.signature
            else:
                logger.error(f"[SWAP_FAILED] Swap execution failed")
                logger.error(f"[SWAP_FAILED] Success: {result.success}, Signature: {result.signature}")
                logger.error(f"[SWAP_FAILED] Error: {result.error}")
                return None

        except Exception as e:
            logger.error(f"[SWAP_EXCEPTION] Swap execution failed: {str(e)}")
            import traceback
            logger.error(f"[SWAP_EXCEPTION] Traceback: {traceback.format_exc()}")
            self.failed_swaps += 1
            return None

    async def execute_meme_token_swap(
        self,
        token_address: str,
        sol_amount: float,
        slippage: float = 10.0,
        config: Optional[PrivateSwapConfig] = None
    ) -> Optional[str]:
        """Execute meme token swap using SOL routing"""
        try:
            logger.info(f"[MEME_SWAP] Attempting SOL -> {token_address[:8]}... swap with {sol_amount} SOL")
            
            # Use SOL as input, meme token as output
            return await self.execute_swap(
                input_token="So11111111111111111111111111111111111111112",  # SOL
                output_token=token_address,
                amount=sol_amount,
                slippage=slippage,
                config=config
            )
            
        except Exception as e:
            logger.error(f"[MEME_SWAP] Failed to execute meme token swap: {str(e)}")
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
            logger.info(f"[ROUTE] Getting routes from Jupiter...")
            logger.info(f"[ROUTE] Input: {input_token[:8]}..., Output: {output_token[:8]}...")
            logger.info(f"[ROUTE] Amount: {amount} SOL ({int(amount * Decimal('1e9'))} lamports)")
            slippage_bps = min(int(slippage * 1000), 5000)  # Cap at 50% for Jupiter
            logger.info(f"[ROUTE] Slippage: {slippage} ({slippage_bps} basis points)")
            
            # Get all possible routes
            routes = await self.jupiter.get_routes(
                input_mint=input_token,
                output_mint=output_token,
                amount=str(int(amount * Decimal("1e9"))),
                slippageBps=slippage_bps
            )

            logger.info(f"[ROUTE] Jupiter response type: {type(routes)}")
            logger.info(f"[ROUTE] Jupiter response: {routes}")

            if not routes:
                logger.warning(f"[ROUTE] No routes from Jupiter - trying fallback quote method")
                return await self._fallback_route_creation(input_token, output_token, amount, slippage)
                
            if not isinstance(routes, dict):
                logger.warning(f"[ROUTE] Non-dict response - trying fallback quote method")
                return await self._fallback_route_creation(input_token, output_token, amount, slippage)
                
            # Jupiter quote API returns the quote directly, not wrapped in 'data'
            # Check if this is a direct quote response (has outAmount) or a routes response (has data)
            if "outAmount" in routes:
                logger.info(f"[ROUTE] Got direct quote response - converting to route format")
                # This is a direct quote, create a route from it
                route = SwapRoute(
                    input_mint=input_token,
                    output_mint=output_token,
                    amount=amount,
                    slippage=slippage,
                    platforms=["Jupiter"],
                    expected_output=Decimal(str(routes.get('outAmount', 0))) / Decimal("1e9")
                )
                logger.info(f"[ROUTE] SUCCESS - Direct quote route created")
                return route
            elif 'data' not in routes:
                logger.warning(f"[ROUTE] No 'data' field and no 'outAmount' - trying fallback quote method")
                logger.info(f"[ROUTE] Available keys: {list(routes.keys()) if isinstance(routes, dict) else 'N/A'}")
                return await self._fallback_route_creation(input_token, output_token, amount, slippage)
            else:
                # This is a routes response with 'data' field
                routes_data = routes['data']
                logger.info(f"[ROUTE] Found {len(routes_data)} potential routes")

            # Filter and sort routes by safety and efficiency
            safe_routes = self._filter_safe_routes(routes_data)
            logger.info(f"[ROUTE] {len(safe_routes)} routes passed safety filter")
            
            if not safe_routes:
                logger.error(f"[ROUTE] FAILED - No safe routes available")
                return None

            # Select optimal route
            best_route = safe_routes[0]
            logger.info(f"[ROUTE] Selected best route with {len(best_route.get('platforms', []))} platforms")
            logger.info(f"[ROUTE] Expected output: {best_route.get('outAmount', 0)} lamports")
            
            route = SwapRoute(
                input_mint=input_token,
                output_mint=output_token,
                amount=amount,
                slippage=slippage,
                platforms=best_route.get('platforms', []),
                expected_output=Decimal(str(best_route.get('outAmount', 0))) / Decimal("1e9")
            )
            
            logger.info(f"[ROUTE] SUCCESS - Route created")
            return route

        except Exception as e:
            logger.error(f"[ROUTE] EXCEPTION - Error getting optimal route: {str(e)}")
            import traceback
            logger.error(f"[ROUTE] Traceback: {traceback.format_exc()}")
            return None

    async def _fallback_route_creation(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal,
        slippage: float
    ) -> Optional[SwapRoute]:
        """Fallback route creation when Jupiter routes fail"""
        try:
            logger.info(f"[FALLBACK] Attempting direct Jupiter quote for route creation...")
            
            # Try direct quote instead of routes
            quote = await self.jupiter.get_quote(
                input_mint=input_token,
                output_mint=output_token,
                amount=str(int(amount * Decimal("1e9"))),
                slippageBps=min(int(slippage * 1000), 5000)  # Cap at 50% for Jupiter
            )
            
            logger.info(f"[FALLBACK] Quote response: {quote}")
            
            if quote and isinstance(quote, dict) and "outAmount" in quote:
                logger.info(f"[FALLBACK] SUCCESS - Creating route from quote")
                
                # Create a route from the quote
                route = SwapRoute(
                    input_mint=input_token,
                    output_mint=output_token,
                    amount=amount,
                    slippage=slippage,
                    platforms=["Jupiter Direct"],  # Indicate this is a direct quote route
                    expected_output=Decimal(str(quote.get('outAmount', 0))) / Decimal("1e9")
                )
                
                logger.info(f"[FALLBACK] Route created with expected output: {route.expected_output}")
                return route
            else:
                logger.error(f"[FALLBACK] FAILED - No valid quote available")
                logger.error(f"[FALLBACK] This token likely cannot be traded on Jupiter")
                return None
                
        except Exception as e:
            logger.error(f"[FALLBACK] EXCEPTION - Error in fallback route creation: {str(e)}")
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
                slippageBps=min(int(route.slippage * 1000), 5000)  # Cap at 50% for Jupiter
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
            # Build transaction using Jupiter v6 API
            logger.info(f"[SWAP] Creating transaction with Jupiter for quote: {quote.get('outAmount', 'unknown')}")
            swap_tx = await self.jupiter.create_swap_transaction(quote, str(self.wallet.wallet_address))
            if not swap_tx:
                logger.error(f"[SWAP] Jupiter transaction creation returned None - no transaction created")
                return SwapResult(
                    success=False,
                    signature=None,
                    output_amount=None,
                    execution_time=0.0,
                    error="Failed to get swap transaction from Jupiter"
                )

            # Deserialize transaction from Jupiter
            import base64
            try:
                # Jupiter returns a serialized transaction as base64
                # We need to deserialize it properly for solana-py
                from solders.transaction import VersionedTransaction
                
                tx_bytes = base64.b64decode(swap_tx)
                logger.info(f"[SWAP] Decoded {len(tx_bytes)} bytes from Jupiter")
                
                # Try deserializing as VersionedTransaction first (Jupiter v6 format)
                try:
                    versioned_tx = VersionedTransaction.from_bytes(tx_bytes)
                    logger.info(f"[SWAP] Successfully deserialized as VersionedTransaction")
                    
                    # Use the VersionedTransaction directly (don't convert)
                    tx = versioned_tx
                    logger.info(f"[SWAP] Using VersionedTransaction directly")
                    
                except Exception as ve:
                    logger.warning(f"[SWAP] VersionedTransaction failed: {ve}, trying legacy format")
                    # Fallback to legacy transaction format
                    tx = Transaction.deserialize(tx_bytes)
                    logger.info(f"[SWAP] Successfully deserialized as legacy Transaction")
                    
            except Exception as e:
                logger.error(f"[SWAP] Failed to deserialize transaction: {str(e)}")
                return SwapResult(
                    success=False,
                    signature=None,
                    output_amount=None,
                    execution_time=0.0,
                    error=f"Transaction deserialization failed: {str(e)}"
                )

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
                logger.info(f"[SWAP] Attempt {attempt + 1}/{self.max_retries} to send transaction")
                # Use private RPC if configured
                if config.use_private_rpc and self.private_rpcs:
                    rpc_url = random.choice(self.private_rpcs)
                    logger.info(f"[SWAP] Using private RPC: {rpc_url}")
                    signature = await self.wallet.sign_and_send_transaction(
                        tx, rpc_url=rpc_url
                    )
                else:
                    logger.info(f"[SWAP] Using default RPC endpoint")
                    signature = await self.wallet.sign_and_send_transaction(tx)
                    
                if signature:
                    logger.info(f"[SWAP] Transaction sent successfully: {signature}")
                    return signature
                else:
                    logger.warning(f"[SWAP] Transaction returned None signature")

            except Exception as e:
                logger.error(f"[SWAP] Attempt {attempt + 1} failed with error: {str(e)}")
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