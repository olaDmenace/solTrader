import logging
from typing import Dict, Any, Optional, List
import aiohttp
import json
import base58
from solana.transaction import Transaction
from solana.rpc.types import TxOpts
from solders.instruction import Instruction
from solders.pubkey import Pubkey

logger = logging.getLogger(__name__)

class JupiterClient:
    def __init__(self) -> None:
        self.base_url = "https://quote-api.jup.ag/v6"
        self.session: Optional[aiohttp.ClientSession] = None

    async def ensure_session(self) -> None:
        """Ensure aiohttp session exists and is active"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers={"Content-Type": "application/json"})

    async def __aenter__(self):
        await self.ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def test_connection(self) -> bool:
        """Test connection to Jupiter API"""
        try:
            await self.ensure_session()
            # Test with a simple quote request for SOL/USDC
            params = {
                "inputMint": "So11111111111111111111111111111111111111112",  # SOL
                "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                "amount": "100000000"  # 0.1 SOL
            }
            async with self.session.get(f"{self.base_url}/quote", params=params) as response:
                if response.status == 200:
                    logger.info("✅ Jupiter API connection successful")
                    return True
                logger.error(f"❌ Jupiter API failed: {response.status}")
                return False
        except Exception as e:
            logger.error(f"❌ Jupiter API error: {e}")
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
        try:
            await self.ensure_session()
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": 50
            }

            async with self.session.get(f"{self.base_url}/quote", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Price quote response: {data}")
                    return data
                logger.debug(f"Price quote failed with status {response.status}")
                return None
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
                price_change = random.uniform(-0.05, 0.05)  # ±5% change
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

    async def get_quote(self, input_mint: str, output_mint: str, amount: str, slippageBps: int) -> Dict[str, Any]:
        """
        Get a quote for a token swap
        
        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address  
            amount: Amount to swap (in smallest unit)
            slippageBps: Slippage tolerance in basis points
            
        Returns:
            Dict containing quote information
        """
        try:
            await self.ensure_session()
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount,
                "slippageBps": slippageBps
            }
            
            async with self.session.get(f"{self.base_url}/quote", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Quote response: {data}")
                    return data
                else:
                    logger.debug(f"Quote request failed with status {response.status}")
                    return {}
                    
        except Exception as e:
            logger.debug(f"Error getting quote: {str(e)}")
            return {}
    
    async def get_routes(self, input_mint: str, output_mint: str, amount: str, slippageBps: int) -> Dict[str, Any]:
        """
        Get swap routes - alias for get_quote for compatibility
        """
        return await self.get_quote(input_mint, output_mint, amount, slippageBps)

    async def create_swap_transaction(self, quote_response: Dict[str, Any], user_public_key: str) -> Optional[str]:
        """
        Create a swap transaction from a quote
        
        Args:
            quote_response: Response from get_quote
            user_public_key: User's wallet public key
            
        Returns:
            Serialized transaction as base64 string
        """
        try:
            await self.ensure_session()
            
            swap_data = {
                "quoteResponse": quote_response,
                "userPublicKey": user_public_key,
                "wrapAndUnwrapSol": True,
                "computeUnitPriceMicroLamports": "auto"
            }
            
            async with self.session.post(f"{self.base_url}/swap", json=swap_data) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("swapTransaction")
                else:
                    error_text = await response.text()
                    logger.error(f"Swap transaction creation failed: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error creating swap transaction: {str(e)}")
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
