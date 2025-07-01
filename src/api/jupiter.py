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
                    logger.error(f"Quote request failed with status {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error response: {error_text}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting quote: {str(e)}")
            return {}
    
    async def get_routes(self, input_mint: str, output_mint: str, amount: str, slippageBps: int) -> Dict[str, Any]:
        """
        Get swap routes - alias for get_quote for compatibility
        
        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount to swap
            slippageBps: Slippage tolerance in basis points
            
        Returns:
            Dict containing route information
        """
        return await self.get_quote(input_mint, output_mint, amount, slippageBps)
    
    async def get_swap_ix(self, quote: Dict[str, Any]) -> Any:
        """
        Get swap instruction from quote
        
        Args:
            quote: Quote object from get_quote
            
        Returns:
            Swap instruction data
        """
        try:
            await self.ensure_session()
            
            swap_request = {
                "quoteResponse": quote,
                "userPublicKey": quote.get("userPublicKey", ""),
                "wrapAndUnwrapSol": True,
                "useSharedAccounts": True,
                "feeAccount": None,
                "computeUnitPriceMicroLamports": None,
                "prioritizationFeeLamports": None
            }
            
            async with self.session.post(
                f"{self.base_url}/swap-instructions",
                json=swap_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Swap instruction request failed with status {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error response: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting swap instruction: {str(e)}")
            return None
    
    async def build_swap_transaction(
        self, 
        route: Dict[str, Any], 
        user_public_key: str
    ) -> Transaction:
        """
        Build a swap transaction from route data
        
        Args:
            route: Route/quote data from Jupiter
            user_public_key: User's wallet public key
            
        Returns:
            Transaction: Built transaction ready for signing
            
        Raises:
            ValueError: If transaction building fails
        """
        try:
            # Update route with user public key
            route["userPublicKey"] = user_public_key
            
            # Get swap instructions
            swap_data = await self.get_swap_ix(route)
            if not swap_data:
                raise ValueError("Failed to get swap instructions")
                
            # Extract transaction data
            swap_transaction = swap_data.get("swapTransaction")
            if not swap_transaction:
                raise ValueError("No swap transaction in response")
                
            # Decode the transaction
            transaction_bytes = base58.b58decode(swap_transaction)
            transaction = Transaction.deserialize(transaction_bytes)
            
            logger.debug(f"Built swap transaction with {len(transaction.instructions)} instructions")
            return transaction
            
        except Exception as e:
            logger.error(f"Error building swap transaction: {str(e)}")
            raise ValueError(f"Failed to build swap transaction: {str(e)}")
    
    async def execute_swap(
        self,
        input_token: str,
        output_token: str, 
        amount: float,
        slippage: float = 0.01,
        priority_fee: Optional[int] = None,
        max_fee: Optional[int] = None
    ) -> bool:
        """
        Execute a token swap (this method requires wallet integration)
        
        Args:
            input_token: Input token mint address
            output_token: Output token mint address
            amount: Amount to swap
            slippage: Slippage tolerance (0.01 = 1%)
            priority_fee: Priority fee in microlamports
            max_fee: Maximum fee limit
            
        Returns:
            bool: True if swap executed successfully
            
        Note:
            This method provides the interface but requires wallet integration
            to actually execute the swap. Should be called from SwapExecutor.
        """
        try:
            # Convert amount to smallest unit (assuming 9 decimals for most tokens)
            amount_lamports = str(int(amount * 1e9))
            slippage_bps = int(slippage * 10000)  # Convert to basis points
            
            # Get quote
            quote = await self.get_quote(
                input_mint=input_token,
                output_mint=output_token,
                amount=amount_lamports,
                slippageBps=slippage_bps
            )
            
            if not quote:
                logger.error("Failed to get swap quote")
                return False
                
            # Log quote details
            in_amount = quote.get("inAmount", "0")
            out_amount = quote.get("outAmount", "0")
            price_impact = quote.get("priceImpactPct", "0")
            
            logger.info(
                f"Swap quote: {in_amount} -> {out_amount}, "
                f"Price impact: {price_impact}%"
            )
            
            # This method returns True to indicate quote was successful
            # Actual execution requires wallet integration in SwapExecutor
            return True
            
        except Exception as e:
            logger.error(f"Error in execute_swap: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Close client connection"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            logger.info("Jupiter connection closed")
        except Exception as e:
            logger.error(f"Error closing Jupiter connection: {str(e)}")