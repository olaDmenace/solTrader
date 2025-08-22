#!/usr/bin/env python3
"""
Enhanced Jupiter API Client with Robust Error Handling
Production-ready version with comprehensive retry logic and error tracking
"""
import logging
from typing import Dict, Any, Optional, List
import aiohttp
import json
import base58
from solana.transaction import Transaction
from solana.rpc.types import TxOpts
from solders.instruction import Instruction
from solders.pubkey import Pubkey

# Import our robust API utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.robust_api import robust_api_call, RobustHTTPClient, RetryConfig, ErrorSeverity

logger = logging.getLogger(__name__)

class EnhancedJupiterClient:
    """Production-ready Jupiter client with comprehensive error handling"""
    
    def __init__(self):
        self.base_url = "https://quote-api.jup.ag/v6"
        
        # Custom retry config for Jupiter API
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=15.0,
            exponential_base=2.0,
            # Jupiter is generally more reliable, so fewer retryable codes
            retryable_status_codes=[429, 500, 502, 503, 504]
        )
        
        # Create robust HTTP client
        self.http_client = RobustHTTPClient(
            base_url=self.base_url,
            component_name="jupiter_api",
            headers={"Content-Type": "application/json"},
            timeout=20.0,  # Jupiter is usually faster
            retry_config=retry_config
        )
    
    async def close(self):
        """Close HTTP session"""
        await self.http_client.close()
    
    async def test_connection(self) -> bool:
        """Test connection to Jupiter API with robust error handling"""
        try:
            logger.info("Testing Jupiter API connection...")
            
            # Test with a simple quote request for SOL/USDC
            params = {
                "inputMint": "So11111111111111111111111111111111111111112",  # SOL
                "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                "amount": "100000000"  # 0.1 SOL
            }
            
            data = await self.http_client.get("quote", params=params)
            
            if data and 'outAmount' in data:
                logger.info("✅ Jupiter API connection successful")
                return True
            else:
                logger.error("❌ Jupiter API test failed - invalid response format")
                return False
                
        except Exception as e:
            logger.error(f"❌ Jupiter API connection test failed: {e}")
            return False
    
    @robust_api_call(component="jupiter_tokens")
    async def get_tokens_list(self) -> List[Dict[str, Any]]:
        """
        Get list of tokens supported by Jupiter
        Note: Jupiter v6 doesn't have a direct tokens list endpoint,
        so we return a minimal list of common tokens
        """
        try:
            logger.info("📋 Getting Jupiter tokens list...")
            
            # Jupiter v6 doesn't have a direct tokens list endpoint
            # Return common tokens that Jupiter supports
            common_tokens = [
                {
                    "address": "So11111111111111111111111111111111111111112",
                    "symbol": "SOL",
                    "name": "Solana",
                    "decimals": 9
                },
                {
                    "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", 
                    "symbol": "USDC",
                    "name": "USD Coin",
                    "decimals": 6
                },
                {
                    "address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
                    "symbol": "USDT", 
                    "name": "Tether USD",
                    "decimals": 6
                }
            ]
            
            logger.info(f"✅ Returned {len(common_tokens)} common tokens")
            return common_tokens
            
        except Exception as e:
            logger.error(f"Error getting tokens list: {str(e)}")
            return []
    
    @robust_api_call(component="jupiter_price_quote")
    async def get_price_quote(self, input_mint: str, output_mint: str, amount: str) -> Optional[Dict[str, Any]]:
        """Get price quote from Jupiter with robust error handling"""
        try:
            logger.debug(f"💰 Getting price quote: {input_mint[:8]}... -> {output_mint[:8]}... ({amount})")
            
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount,
                "slippageBps": 50  # 0.5% slippage
            }
            
            data = await self.http_client.get("quote", params=params)
            
            if data and 'outAmount' in data:
                logger.debug(f"✅ Price quote received: {data.get('outAmount')}")
                return data
            else:
                logger.warning("⚠️ Invalid price quote response format")
                return None
                
        except Exception as e:
            logger.debug(f"Error getting price quote: {str(e)}")
            return None
    
    @robust_api_call(component="jupiter_market_depth")
    async def get_market_depth(self, input_mint: str, output_mint: str) -> Optional[Dict[str, Any]]:
        """Get market depth information with robust error handling"""
        try:
            logger.debug(f"📊 Getting market depth: {input_mint[:8]}... -> {output_mint[:8]}...")
            
            # Try multiple amounts to gauge market depth
            amounts = ["1000000", "10000000", "100000000"]  # 0.001, 0.01, 0.1 SOL equivalent
            depth_data = {}
            
            for amount in amounts:
                params = {
                    "inputMint": input_mint,
                    "outputMint": output_mint, 
                    "amount": amount,
                    "slippageBps": 50
                }
                
                try:
                    quote = await self.http_client.get("quote", params=params)
                    if quote and 'outAmount' in quote:
                        depth_data[amount] = {
                            'outAmount': quote['outAmount'],
                            'priceImpactPct': quote.get('priceImpactPct', 0)
                        }
                except:
                    continue  # Skip failed amounts
            
            if depth_data:
                logger.debug(f"✅ Market depth data for {len(depth_data)} amounts")
                return depth_data
            else:
                logger.warning("⚠️ No market depth data available")
                return None
                
        except Exception as e:
            logger.debug(f"Error getting market depth: {str(e)}")
            return None
    
    @robust_api_call(component="jupiter_price_history")
    async def get_price_history(self, token_address: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get price history for a token
        Note: Jupiter v6 doesn't provide historical data, so this returns empty for now
        """
        try:
            logger.debug(f"📈 Getting price history for {token_address[:8]}... ({days} days)")
            
            # Jupiter v6 doesn't provide price history directly
            # This would need to be implemented with other data sources
            logger.debug("⚠️ Price history not available from Jupiter v6")
            return []
            
        except Exception as e:
            logger.error(f"Error getting price history: {str(e)}")
            return []
    
    @robust_api_call(component="jupiter_token_info")
    async def get_token_info(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get token information with robust error handling"""
        try:
            logger.debug(f"ℹ️ Getting token info for {token_address[:8]}...")
            
            # Jupiter doesn't have a direct token info endpoint
            # Try to validate the token by attempting a small quote
            validation_quote = await self.get_price_quote(
                token_address,
                "So11111111111111111111111111111111111111112",  # SOL
                "1000000"  # Small amount
            )
            
            if validation_quote:
                logger.debug(f"✅ Token {token_address[:8]}... is tradeable")
                return {
                    "address": token_address,
                    "tradeable": True,
                    "last_quote": validation_quote
                }
            else:
                logger.debug(f"❌ Token {token_address[:8]}... not tradeable")
                return None
                
        except Exception as e:
            logger.error(f"Error getting token info: {str(e)}")
            return None
    
    @robust_api_call(component="jupiter_quote")
    async def get_quote(self, 
                       input_mint: str,
                       output_mint: str, 
                       amount: int,
                       slippage_bps: int = 50) -> Dict[str, Any]:
        """Get trading quote with comprehensive error handling"""
        try:
            logger.debug(f"💱 Getting trading quote: {amount} {input_mint[:8]}... -> {output_mint[:8]}...")
            
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": slippage_bps,
                "onlyDirectRoutes": "false",  # Allow multi-hop routes
                "asLegacyTransaction": "false"  # Use versioned transactions
            }
            
            data = await self.http_client.get("quote", params=params)
            
            if data and 'outAmount' in data:
                logger.debug(f"✅ Quote received: {data['outAmount']} output")
                
                # Add some helpful computed fields
                data['input_amount'] = amount
                data['input_mint'] = input_mint
                data['output_mint'] = output_mint
                data['effective_price'] = float(data['outAmount']) / amount if amount > 0 else 0
                
                return data
            else:
                logger.warning("⚠️ Invalid quote response format")
                return {}
                
        except Exception as e:
            logger.debug(f"Error getting quote: {str(e)}")
            return {}
    
    @robust_api_call(component="jupiter_swap_tx")
    async def create_swap_transaction(self,
                                    quote: Dict[str, Any],
                                    user_public_key: str,
                                    wrap_unwrap_sol: bool = True) -> Optional[str]:
        """Create swap transaction with robust error handling"""
        try:
            logger.debug(f"🔨 Creating swap transaction for user {user_public_key[:8]}...")
            
            if not quote or 'outAmount' not in quote:
                logger.error("❌ Invalid quote provided for transaction creation")
                return None
            
            swap_data = {
                "quoteResponse": quote,
                "userPublicKey": user_public_key,
                "wrapUnwrapSOL": wrap_unwrap_sol,
                "computeUnitPriceMicroLamports": 1000  # Set compute unit price
            }
            
            response = await self.http_client.post("swap", json_data=swap_data)
            
            if response and 'swapTransaction' in response:
                transaction_data = response['swapTransaction']
                logger.debug("✅ Swap transaction created successfully")
                return transaction_data
            else:
                logger.error("❌ Invalid swap transaction response format")
                return None
                
        except Exception as e:
            logger.error(f"Error creating swap transaction: {str(e)}")
            return None
    
    @robust_api_call(component="jupiter_sol_price")
    async def get_sol_price_in_usdc(self) -> Optional[float]:
        """Get current SOL price in USDC with robust error handling"""
        try:
            logger.debug("💲 Getting SOL price in USDC...")
            
            quote = await self.get_quote(
                "So11111111111111111111111111111111111111112",  # SOL
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                1000000000  # 1 SOL
            )
            
            if quote and 'outAmount' in quote:
                usdc_amount = float(quote['outAmount']) / 1000000  # USDC has 6 decimals
                logger.debug(f"✅ SOL price: ${usdc_amount:.2f}")
                return usdc_amount
            else:
                logger.warning("⚠️ Could not get SOL price")
                return None
                
        except Exception as e:
            logger.debug(f"Error getting SOL price: {e}")
            return None
    
    async def validate_token_tradeable(self, token_address: str) -> bool:
        """Check if a token is tradeable on Jupiter with robust error handling"""
        try:
            logger.debug(f"🔍 Validating token tradeability: {token_address[:8]}...")
            
            # Try to get a small quote for the token
            quote = await self.get_quote(
                token_address,
                "So11111111111111111111111111111111111111112",  # SOL
                1000000  # Small amount
            )
            
            is_tradeable = bool(quote and 'outAmount' in quote)
            logger.debug(f"{'✅' if is_tradeable else '❌'} Token {token_address[:8]}... tradeable: {is_tradeable}")
            return is_tradeable
            
        except Exception as e:
            logger.debug(f"Token validation failed for {token_address}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status information"""
        return {
            "base_url": self.base_url,
            "component_name": "jupiter_api",
            "connection_active": self.http_client.session is not None and not self.http_client.session.closed
        }

# Global instance for reuse
enhanced_jupiter_client = EnhancedJupiterClient()

async def get_enhanced_jupiter_quote(input_mint: str, output_mint: str, amount: int) -> Dict[str, Any]:
    """Convenience function to get quote from enhanced client"""
    return await enhanced_jupiter_client.get_quote(input_mint, output_mint, amount)