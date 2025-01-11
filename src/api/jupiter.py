import logging
from typing import Dict, Any, Optional, List
import aiohttp
import json

logger = logging.getLogger(__name__)

class JupiterClient:
    def __init__(self) -> None:
        self.base_url = "https://quote-api.jup.ag/v6"  # Update URL
        # self.base_url = "https://price.jup.ag/v4"
        self.session: Optional[aiohttp.ClientSession] = None

    async def ensure_session(self) -> None:
        """Ensure aiohttp session exists and is active"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers={"Content-Type": "application/json"})

    # async def test_connection(self) -> bool:
    #     try:
    #         await self.ensure_session()
    #         async with self.session.get(self.base_url) as response:
    #             if response.status == 200:
    #                 logger.info("Successfully connected to Jupiter API")
    #                 return True
    #             logger.error(f"Connection failed with status {response.status}")
    #             return False
    #     except Exception as e:
    #         logger.error(f"Jupiter connection error: {str(e)}")
    #         return False

    # Add to both AlchemyClient and JupiterClient
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
                    logger.info("Successfully connected to Jupiter API")
                    return True
                logger.error(f"Connection failed with status {response.status}")
                return False
        except Exception as e:
            logger.error(f"Jupiter connection error: {str(e)}")
            return False

    async def get_tokens_list(self) -> List[Dict[str, Any]]:
        """Get list of supported tokens"""
        try:
            await self.ensure_session()
            async with self.session.get(f"{self.base_url}/swap-token-list") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('tokens', [])
                return []
        except Exception as e:
            logger.error(f"Error getting tokens list: {str(e)}")
            return []

    # async def get_price(self, 
    #                    input_mint: str,
    #                    output_mint: str = "So11111111111111111111111111111111111111112",  # SOL
    #                    amount: int = 1000000000  # 1 SOL in lamports
    #                    ) -> Optional[Dict[str, Any]]:
    #     """Get price quote for token swap"""
    #     try:
    #         await self.ensure_session()
    #         params = {
    #             "inputMint": input_mint,
    #             "outputMint": output_mint,
    #             "amount": str(amount),
    #             "slippageBps": 50  # 0.5% slippage
    #         }

    #         async with self.session.get(f"{self.base_url}/quote", params=params) as response:
    #             if response.status == 200:
    #                 return await response.json()
    #             logger.error(f"Price quote failed with status {response.status}")
    #             return None
    #     except Exception as e:
    #         logger.error(f"Error getting price quote: {str(e)}")
    #         return None

    async def get_price(self, 
                    input_mint: str,
                    output_mint: str,
                    amount: int) -> Optional[Dict[str, Any]]:
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
                    logger.info(f"Price quote response: {data}")
                    return data
                logger.error(f"Price quote failed with status {response.status}")
                return None
        except Exception as e:
            logger.error(f"Error getting price quote: {str(e)}")
            return None

    async def get_market_depth(self, token_mint: str) -> Optional[Dict[str, Any]]:
        """Get market depth for a token"""
        try:
            await self.ensure_session()
            async with self.session.get(
                f"{self.base_url}/market-info",
                params={"inputMint": token_mint, "outputMint": "So11111111111111111111111111111111111111112"}
            ) as response:
                if response.status == 200:
                    return await response.json()
                logger.error(f"Market depth request failed with status {response.status}")
                return None
        except Exception as e:
            logger.error(f"Error getting market depth: {str(e)}")
            return None

    async def get_token_info(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get detailed token information"""
        try:
            tokens = await self.get_tokens_list()
            return next((token for token in tokens if token.get('address') == token_address), None)
        except Exception as e:
            logger.error(f"Error getting token info: {str(e)}")
            return None

    async def get_price_history(self, token_address: str, interval: str = '1h') -> List[Dict[str, float]]:
        """Get token price history"""
        try:
            await self.ensure_session()
            async with self.session.get(
                f"{self.base_url}/price-history",
                params={"address": token_address, "interval": interval}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('prices', [])
                return []
        except Exception as e:
            logger.error(f"Error getting price history: {str(e)}")
            return []

    async def close(self) -> None:
        """Close client connection"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            logger.info("Jupiter connection closed")
        except Exception as e:
            logger.error(f"Error closing Jupiter connection: {str(e)}")