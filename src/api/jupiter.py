import logging
from typing import Dict, Any, Optional, List
import aiohttp
import json
import base58
import asyncio
import time
from solana.transaction import Transaction
from solana.rpc.types import TxOpts
from solders.instruction import Instruction
from solders.pubkey import Pubkey

logger = logging.getLogger(__name__)

class JupiterClient:
    def __init__(self) -> None:
        self.base_url = "https://quote-api.jup.ag/v6"
        self.session: Optional[aiohttp.ClientSession] = None
        self.quote_cache = {}  # Cache for recent quotes
        self.cache_duration = 60  # Cache quotes for 60 seconds (longer caching)
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_delay = 2.0  # Start with 2s delay (more conservative)
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.base_backoff_delay = 2.0

    async def ensure_session(self) -> None:
        """Ensure aiohttp session exists and is active"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers={"Content-Type": "application/json"})

    async def __aenter__(self):
        await self.ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def _get_cache_key(self, input_mint: str, output_mint: str, amount: str) -> str:
        """Generate cache key for quote requests"""
        return f"{input_mint}:{output_mint}:{amount}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cached quote is still valid"""
        return (time.time() - cache_entry['timestamp']) < self.cache_duration
    
    async def _smart_delay(self, success: bool = True, is_rate_limited: bool = False):
        """Implement intelligent delay with exponential backoff"""
        current_time = time.time()
        
        # Ensure minimum delay between requests
        elapsed = current_time - self.last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        
        # Adjust delay based on success/failure pattern
        if success:
            # Gradually reduce delay on success (but keep minimum)
            self.consecutive_failures = 0
            self.rate_limit_delay = max(2.0, self.rate_limit_delay * 0.9)
        else:
            self.consecutive_failures += 1
            if is_rate_limited:
                # More aggressive backoff for rate limits
                self.rate_limit_delay = min(30.0, self.rate_limit_delay * 2.0)
                logger.warning(f"Jupiter rate limited, increasing delay to {self.rate_limit_delay:.1f}s")
            else:
                # Standard exponential backoff for other failures
                self.rate_limit_delay = min(15.0, self.rate_limit_delay * 1.3)
                logger.warning(f"Jupiter request failed, increasing delay to {self.rate_limit_delay:.1f}s")
        
        self.last_request_time = time.time()
        self.request_count += 1

    async def test_connection(self) -> bool:
        """Test connection to Jupiter API with rate limiting awareness"""
        try:
            await self.ensure_session()
            
            # Apply rate limiting to connection test
            await self._smart_delay(success=True)
            
            # Test with a simple quote request for SOL/USDC
            params = {
                "inputMint": "So11111111111111111111111111111111111111112",  # SOL
                "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                "amount": "100000000"  # 0.1 SOL
            }
            async with self.session.get(f"{self.base_url}/quote", params=params, timeout=10) as response:
                if response.status == 200:
                    logger.info("[OK] Jupiter API connection successful")
                    await self._smart_delay(success=True)
                    return True
                elif response.status == 429:
                    logger.warning("Jupiter API rate limited during connection test")
                    await self._smart_delay(success=False, is_rate_limited=True)
                    return False
                else:
                    logger.error(f"Jupiter API failed: {response.status}")
                    await self._smart_delay(success=False)
                    return False
        except asyncio.TimeoutError:
            logger.error("Jupiter API connection test timed out")
            await self._smart_delay(success=False)
            return False
        except Exception as e:
            logger.error(f"Jupiter API error: {e}")
            await self._smart_delay(success=False)
            return False

    async def get_tokens_list(self) -> List[Dict[str, Any]]:
        """Get list of supported tokens"""
        try:
            await self.ensure_session()
            # Jupiter v6 doesn't have a direct tokens list endpoint
            # Return a mock list with common Solana tokens
            common_tokens = [
                {
                    "address": "So11111111111111111111111111111111111111112",
                    "name": "Solana",
                    "symbol": "SOL",
                    "decimals": 9
                },
                {
                    "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    "name": "USD Coin",
                    "symbol": "USDC", 
                    "decimals": 6
                },
                {
                    "address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
                    "name": "Tether",
                    "symbol": "USDT",
                    "decimals": 6
                }
            ]
            logger.debug(f"Returning {len(common_tokens)} common tokens")
            return common_tokens
        except Exception as e:
            logger.error(f"Error getting tokens list: {str(e)}")
            return []

    async def get_price(self, 
                       input_mint: str,
                       output_mint: str = "So11111111111111111111111111111111111111112",  # SOL
                       amount: int = 1000000000  # 1 SOL in lamports
                       ) -> Optional[Dict[str, Any]]:
        """Get price quote for token swap"""
        # Use the smart get_quote method instead of duplicate logic
        try:
            result = await self.get_quote(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=str(amount),
                slippageBps=50
            )
            return result if result else None
        except Exception as e:
            logger.debug(f"Error getting price quote: {str(e)}")
            return None

    async def get_market_depth(self, token_mint: str) -> Optional[Dict[str, Any]]:
        """Get market depth for a token using quote endpoint"""
        try:
            await self.ensure_session()
            params = {
                "inputMint": token_mint,
                "outputMint": "So11111111111111111111111111111111111111112",  # SOL
                "amount": "1000000"  # 0.001 tokens
            }
            async with self.session.get(f"{self.base_url}/quote", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Convert quote to market depth format
                    return {
                        "liquidity": float(data.get("outAmount", 0)) / 1000000,
                        "price": float(data.get("outAmount", 0)) / float(data.get("inAmount", 1)),
                        "available": True
                    }
                logger.debug(f"Market depth request failed with status {response.status}")
                return None
        except Exception as e:
            logger.debug(f"Error getting market depth: {str(e)}")
            return None

    async def get_price_history(self, token_address: str, interval: str = '1h') -> List[Dict[str, float]]:
        """Get token price history - mock implementation"""
        try:
            # Jupiter v6 doesn't provide price history directly
            # Return mock price data for now
            import time
            from datetime import datetime, timedelta
            
            current_time = datetime.now()
            mock_prices = []
            base_price = 1.0
            
            # Generate 24 hours of mock price data
            for i in range(24):
                timestamp = current_time - timedelta(hours=23-i)
                # Simple random walk for mock prices
                import random
                price_change = random.uniform(-0.05, 0.05)  # +/-5% change
                base_price *= (1 + price_change)
                
                mock_prices.append({
                    "timestamp": timestamp.isoformat(),
                    "price": round(base_price, 6),
                    "volume": random.randint(1000, 10000)
                })
            
            logger.debug(f"Generated {len(mock_prices)} mock price points")
            return mock_prices
            
        except Exception as e:
            logger.error(f"Error getting price history: {str(e)}")
            return []

    async def get_token_info(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get detailed token information"""
        try:
            tokens = await self.get_tokens_list()
            return next((token for token in tokens if token.get('address') == token_address), None)
        except Exception as e:
            logger.error(f"Error getting token info: {str(e)}")
            return None

    async def get_quote(self, input_mint: str, output_mint: str, amount: str, slippageBps: int, max_retries: int = 5) -> Dict[str, Any]:
        """
        Get a quote for a token swap with smart caching and retry
        
        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address  
            amount: Amount to swap (in smallest unit)
            slippageBps: Slippage tolerance in basis points
            max_retries: Maximum retry attempts
            
        Returns:
            Dict containing quote information
        """
        # Check cache first
        cache_key = self._get_cache_key(input_mint, output_mint, amount)
        if cache_key in self.quote_cache and self._is_cache_valid(self.quote_cache[cache_key]):
            logger.debug(f"Returning cached quote for {input_mint[:8]}...:{output_mint[:8]}...")
            return self.quote_cache[cache_key]['data']
        
        await self.ensure_session()
        
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount,
            "slippageBps": slippageBps
        }
        
        # Check if we've had too many consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.error(f"Too many consecutive failures ({self.consecutive_failures}), backing off for 60 seconds")
            await asyncio.sleep(60)
            self.consecutive_failures = 0
            self.rate_limit_delay = self.base_backoff_delay
        
        for attempt in range(max_retries):
            try:
                # Pre-request delay based on current rate limiting state
                await self._smart_delay(success=True)
                
                async with self.session.get(f"{self.base_url}/quote", params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check if we got a valid response (not empty dict)
                        if isinstance(data, dict) and len(data) > 0 and 'outAmount' in data:
                            logger.debug(f"Quote success: {input_mint[:8]}...:{output_mint[:8]}...")
                            
                            # Cache successful response
                            self.quote_cache[cache_key] = {
                                'data': data,
                                'timestamp': time.time()
                            }
                            await self._smart_delay(success=True)
                            return data
                        else:
                            logger.warning(f"Empty/invalid quote response (attempt {attempt + 1}/{max_retries})")
                            await self._smart_delay(success=False)
                    
                    elif response.status == 429:
                        logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries})")
                        await self._smart_delay(success=False, is_rate_limited=True)
                        # Progressive exponential backoff for rate limits
                        backoff_delay = min(60, (2 ** attempt) * 3)
                        logger.info(f"Rate limit backoff: waiting {backoff_delay}s")
                        await asyncio.sleep(backoff_delay)
                    
                    elif response.status == 400:
                        logger.warning(f"Bad request (400) - invalid parameters (attempt {attempt + 1}/{max_retries})")
                        await self._smart_delay(success=False)
                        # Don't retry on 400 errors immediately
                        await asyncio.sleep(1)
                    
                    else:
                        logger.warning(f"Quote failed with status {response.status} (attempt {attempt + 1}/{max_retries})")
                        await self._smart_delay(success=False)
                        await asyncio.sleep(min(10, 2 ** attempt))
                        
            except asyncio.TimeoutError:
                logger.warning(f"Quote request timeout (attempt {attempt + 1}/{max_retries})")
                await self._smart_delay(success=False)
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(10, 2 ** attempt))
                        
            except Exception as e:
                logger.warning(f"Quote request error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                await self._smart_delay(success=False)
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(10, 2 ** attempt))
        
        logger.error(f"Failed to get quote after {max_retries} attempts")
        return {}
    
    async def get_routes(self, input_mint: str, output_mint: str, amount: str, slippageBps: int) -> Dict[str, Any]:
        """
        Get swap routes - alias for get_quote for compatibility
        """
        return await self.get_quote(input_mint, output_mint, amount, slippageBps)

    async def create_swap_instructions(self, quote_response: Dict[str, Any], user_public_key: str) -> Optional[Dict[str, Any]]:
        """
        Create swap instructions from a quote using the modern swap-instructions endpoint
        
        Args:
            quote_response: Response from get_quote
            user_public_key: User's wallet public key
            
        Returns:
            Instructions data for building transaction manually
        """
        try:
            await self.ensure_session()
            
            logger.info(f"[JUPITER] Creating swap instructions for user: {user_public_key[:8]}...")
            logger.debug(f"[JUPITER] Quote response keys: {list(quote_response.keys())}")
            
            # Use the swap endpoint instead - it should return a swapTransaction
            swap_data = {
                "quoteResponse": quote_response,
                "userPublicKey": user_public_key,
                "wrapAndUnwrapSol": True,
                "dynamicComputeUnitLimit": True,
                "prioritizationFeeLamports": {
                    "priorityLevelWithMaxLamports": {
                        "maxLamports": 1000000,  # 0.001 SOL max priority fee
                        "priorityLevel": "medium"
                    }
                }
            }
            
            async with self.session.post(f"{self.base_url}/swap", json=swap_data) as response:
                if response.status == 200:
                    data = await response.json()
                    # Check for swapTransaction from regular swap endpoint
                    if "swapTransaction" in data:
                        logger.info(f"[JUPITER] Successfully created swap transaction")
                        logger.debug(f"[JUPITER] Response keys: {list(data.keys())}")
                        return data
                    else:
                        logger.error(f"[JUPITER] No swapTransaction in response")
                        logger.debug(f"[JUPITER] Response keys: {list(data.keys())}")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"[JUPITER] Swap instructions creation failed: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"[JUPITER] Error creating swap instructions: {str(e)}")
            return None

    async def build_transaction_from_instructions(self, instructions_data: Dict[str, Any]) -> Optional[str]:
        """
        Build a VersionedTransaction from Jupiter instruction data
        
        Args:
            instructions_data: Response from create_swap_instructions
            
        Returns:
            Serialized transaction as base64 string
        """
        try:
            from solders.instruction import Instruction
            from solders.message import MessageV0, Message
            from solders.transaction import VersionedTransaction
            from solders.pubkey import Pubkey
            from solders.address_lookup_table_account import AddressLookupTableAccount
            import base64
            import json
            
            logger.info(f"[JUPITER] Building transaction from instructions")
            
            # Collect all instructions in order
            all_instructions = []
            
            from solders.instruction import AccountMeta
            
            # Add compute budget instructions first
            for inst_data in instructions_data.get("computeBudgetInstructions", []):
                accounts = [
                    AccountMeta(
                        pubkey=Pubkey.from_string(acc["pubkey"]),
                        is_signer=acc["isSigner"], 
                        is_writable=acc["isWritable"]
                    )
                    for acc in inst_data["accounts"]
                ]
                instruction = Instruction(
                    program_id=Pubkey.from_string(inst_data["programId"]),
                    accounts=accounts,
                    data=base64.b64decode(inst_data["data"])
                )
                all_instructions.append(instruction)
            
            # Add setup instructions
            for inst_data in instructions_data.get("setupInstructions", []):
                accounts = [
                    AccountMeta(
                        pubkey=Pubkey.from_string(acc["pubkey"]),
                        is_signer=acc["isSigner"],
                        is_writable=acc["isWritable"]
                    )
                    for acc in inst_data["accounts"]
                ]
                instruction = Instruction(
                    program_id=Pubkey.from_string(inst_data["programId"]),
                    accounts=accounts,
                    data=base64.b64decode(inst_data["data"])
                )
                all_instructions.append(instruction)
            
            # Add swap instruction
            swap_inst = instructions_data["swapInstruction"]
            accounts = [
                AccountMeta(
                    pubkey=Pubkey.from_string(acc["pubkey"]),
                    is_signer=acc["isSigner"],
                    is_writable=acc["isWritable"]
                )
                for acc in swap_inst["accounts"]
            ]
            instruction = Instruction(
                program_id=Pubkey.from_string(swap_inst["programId"]),
                accounts=accounts,
                data=base64.b64decode(swap_inst["data"])
            )
            all_instructions.append(instruction)
            
            # Add cleanup instruction
            cleanup_inst = instructions_data.get("cleanupInstruction")
            if cleanup_inst:
                accounts = [
                    AccountMeta(
                        pubkey=Pubkey.from_string(acc["pubkey"]),
                        is_signer=acc["isSigner"],
                        is_writable=acc["isWritable"]
                    )
                    for acc in cleanup_inst["accounts"]
                ]
                instruction = Instruction(
                    program_id=Pubkey.from_string(cleanup_inst["programId"]),
                    accounts=accounts,
                    data=base64.b64decode(cleanup_inst["data"])
                )
                all_instructions.append(instruction)
            
            # Get blockhash
            blockhash_data = instructions_data.get("blockhashWithMetadata", {})
            if "blockhash" in blockhash_data:
                blockhash_bytes = bytes(blockhash_data["blockhash"])
                from solders.hash import Hash
                blockhash = Hash(blockhash_bytes)
            else:
                # Fallback - this will require getting a fresh blockhash
                logger.warning("[JUPITER] No blockhash in instructions data, will need fresh blockhash")
                return None
            
            # Build MessageV0 with lookup tables
            lookup_tables = []
            for table_address in instructions_data.get("addressLookupTableAddresses", []):
                # For now, create empty lookup table - in production you'd fetch the actual table
                lookup_tables.append(AddressLookupTableAccount(
                    key=Pubkey.from_string(table_address),
                    addresses=[]  # Would need to fetch actual addresses
                ))
            
            # Create message - extract payer from swap instructions
            payer = None
            for inst_data in instructions_data.get("setupInstructions", []):
                for acc in inst_data.get("accounts", []):
                    if acc.get("isSigner", False):
                        payer = Pubkey.from_string(acc["pubkey"])
                        break
                if payer:
                    break
            
            # Fallback to swap instruction signer
            if not payer:
                swap_inst = instructions_data.get("swapInstruction", {})
                for acc in swap_inst.get("accounts", []):
                    if acc.get("isSigner", False):
                        payer = Pubkey.from_string(acc["pubkey"])
                        break
            
            try:
                message = MessageV0.try_compile(
                    payer=payer,
                    instructions=all_instructions,
                    address_lookup_table_accounts=lookup_tables,
                    recent_blockhash=blockhash
                )
                
                # Create unsigned VersionedTransaction
                transaction = VersionedTransaction(message, [])
                
                # Serialize to base64
                tx_bytes = bytes(transaction)
                tx_b64 = base64.b64encode(tx_bytes).decode('utf-8')
                
            except Exception as compile_error:
                logger.error(f"[JUPITER] Error compiling message: {str(compile_error)}")
                # Try alternative approach - create a simple serializable structure
                # that the wallet can reconstruct and sign
                transaction_data = {
                    "instructions": [
                        {
                            "programId": str(inst.program_id),
                            "accounts": [
                                {
                                    "pubkey": str(acc.pubkey),
                                    "isSigner": acc.is_signer,
                                    "isWritable": acc.is_writable
                                }
                                for acc in inst.accounts
                            ],
                            "data": base64.b64encode(inst.data).decode('utf-8')
                        }
                        for inst in all_instructions
                    ],
                    "blockhash": base64.b64encode(blockhash_bytes).decode('utf-8'),
                    "payer": str(payer) if payer else "",
                    "addressLookupTableAddresses": instructions_data.get("addressLookupTableAddresses", [])
                }
                
                # Serialize the instruction data
                tx_b64 = base64.b64encode(
                    json.dumps(transaction_data).encode('utf-8')
                ).decode('utf-8')
                logger.info("[JUPITER] Created transaction data structure for manual reconstruction")
            
            logger.info(f"[JUPITER] Successfully built transaction from instructions")
            return tx_b64
            
        except Exception as e:
            logger.error(f"[JUPITER] Error building transaction from instructions: {str(e)}")
            return None

    async def create_swap_transaction(self, quote_response: Dict[str, Any], user_public_key: str) -> Optional[str]:
        """
        Create swap transaction using Jupiter's swap endpoint
        
        Args:
            quote_response: Response from get_quote
            user_public_key: User's wallet public key
            
        Returns:
            Serialized transaction as base64 string
        """
        try:
            # Get swap transaction directly from Jupiter
            swap_data = await self.create_swap_instructions(quote_response, user_public_key)
            if not swap_data:
                logger.error("[JUPITER] Failed to get swap transaction")
                return None
            
            # Extract swapTransaction
            transaction = swap_data.get("swapTransaction")
            if not transaction:
                logger.error("[JUPITER] No swapTransaction in response")
                return None
            
            logger.info("[JUPITER] Successfully got transaction from Jupiter")
            return transaction
            
        except Exception as e:
            logger.error(f"[JUPITER] Error in create_swap_transaction: {str(e)}")
            return None

    async def close(self) -> None:
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("Jupiter client session closed")

    # Additional helper methods for compatibility
    
    async def get_sol_price(self) -> Optional[float]:
        """Get current SOL price in USD"""
        try:
            quote = await self.get_quote(
                "So11111111111111111111111111111111111111112",  # SOL
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                "1000000000",  # 1 SOL
                50  # 0.5% slippage
            )
            if quote and "outAmount" in quote:
                # Convert from lamports to USD
                return float(quote["outAmount"]) / 1000000  # USDC has 6 decimals
            return None
        except Exception as e:
            logger.debug(f"Error getting SOL price: {e}")
            return None

    async def validate_token(self, token_address: str) -> bool:
        """Check if a token is valid and tradeable"""
        try:
            # Try to get a small quote for the token
            quote = await self.get_quote(
                token_address,
                "So11111111111111111111111111111111111111112",  # SOL
                "1000000",  # Small amount
                100  # 1% slippage
            )
            return bool(quote and "outAmount" in quote)
        except Exception as e:
            logger.debug(f"Token validation failed for {token_address}: {e}")
            return False
