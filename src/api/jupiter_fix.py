import aiohttp
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class JupiterClient:
    def __init__(self):
        # Use the correct Jupiter v6 API endpoint
        self.base_url = "https://quote-api.jup.ag/v6"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"}
            )
    
    async def test_connection(self) -> bool:
        """Test Jupiter API connection"""
        try:
            await self.ensure_session()
            # Test with SOL price quote
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
    
    async def get_market_depth(self, token_mint: str) -> Optional[Dict[str, Any]]:
        """Fixed market depth method - remove 404 endpoint"""
        try:
            # Instead of market-info endpoint, use price endpoint
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
                logger.debug(f"Quote request failed: {response.status}")
                return None
        except Exception as e:
            logger.debug(f"Market depth error: {e}")
            return None
    
    async def close(self):
        """Close session"""
        if self.session and not self.session.closed:
            await self.session.close()
