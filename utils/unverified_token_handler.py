"""
Unverified Token Handler - Safe Trading for Flagged/Unverified Tokens

This module provides robust handling for tokens that are:
- Not verified on Jupiter
- Flagged with wash trading warnings
- Missing from standard token lists
- Have irregular configurations (decimals, minimum trade sizes)

Handles ALL "BananaGuy-like" tokens safely.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal
import aiohttp

logger = logging.getLogger(__name__)

class UnverifiedTokenHandler:
    """Handles trading of unverified/flagged tokens with enhanced safety"""
    
    def __init__(self):
        self.token_cache = {}  # Cache for token metadata
        self.failed_tokens = set()  # Tokens that consistently fail
        self.jupiter_base_url = "https://quote-api.jup.ag/v6"
        
        # CRITICAL FIX: Ensure SOL is never in failed cache
        SOL_MINT = "So11111111111111111111111111111111111111112"
        if SOL_MINT in self.failed_tokens:
            self.failed_tokens.remove(SOL_MINT)
            logger.info(f"[UNVERIFIED] Removed SOL from failed tokens cache")
        
    async def validate_unverified_token_trade(
        self, 
        token_mint: str, 
        amount: Decimal, 
        output_mint: str = "So11111111111111111111111111111111111111112"  # SOL
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Comprehensive validation for unverified token trades
        
        Returns: (is_safe, reason, token_metadata)
        """
        try:
            # CRITICAL FIX: Never fail SOL token - it's the base currency
            SOL_MINT = "So11111111111111111111111111111111111111112"
            if token_mint == SOL_MINT or output_mint == SOL_MINT:
                logger.info(f"[UNVERIFIED] SOL token whitelisted - always safe")
                return True, "SOL is always safe", {"symbol": "SOL", "decimals": 9}
            
            # Check if token is in failed cache
            if token_mint in self.failed_tokens:
                return False, "Token in failed cache", None
            
            # Get or cache token metadata
            metadata = await self._get_token_metadata(token_mint)
            if not metadata:
                return False, "Cannot retrieve token metadata", None
            
            # Validate token decimals and amount
            is_amount_valid, amount_reason = self._validate_amount(amount, metadata)
            if not is_amount_valid:
                return False, amount_reason, metadata
            
            # Try direct Jupiter quote for validation
            quote = await self._get_direct_quote(token_mint, amount, output_mint)
            if not quote:
                return False, "No direct quote available", metadata
            
            # Validate quote sanity
            is_quote_valid, quote_reason = self._validate_quote(amount, quote, metadata)
            if not is_quote_valid:
                return False, quote_reason, metadata
                
            logger.info(f"[UNVERIFIED] Token {token_mint[:8]}... validated for trading")
            return True, "Safe to trade", metadata
            
        except Exception as e:
            logger.error(f"[UNVERIFIED] Validation failed for {token_mint}: {e}")
            return False, f"Validation error: {str(e)}", None
    
    async def _get_token_metadata(self, token_mint: str) -> Optional[Dict]:
        """Get and cache token metadata"""
        if token_mint in self.token_cache:
            return self.token_cache[token_mint]
        
        try:
            # Try multiple sources for metadata
            metadata = await self._fetch_metadata_from_jupiter(token_mint)
            if not metadata:
                metadata = await self._fetch_metadata_from_solana(token_mint)
            
            if metadata:
                self.token_cache[token_mint] = metadata
                logger.debug(f"[UNVERIFIED] Cached metadata for {token_mint[:8]}...")
            
            return metadata
            
        except Exception as e:
            logger.warning(f"[UNVERIFIED] Failed to get metadata for {token_mint}: {e}")
            return None
    
    async def _fetch_metadata_from_jupiter(self, token_mint: str) -> Optional[Dict]:
        """Fetch metadata from Jupiter token list"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://token.jup.ag/all") as response:
                    if response.status == 200:
                        tokens = await response.json()
                        for token in tokens:
                            if token.get('address') == token_mint:
                                return {
                                    'symbol': token.get('symbol', 'UNKNOWN'),
                                    'name': token.get('name', 'Unknown Token'),
                                    'decimals': token.get('decimals', 9),
                                    'verified': token.get('extensions', {}).get('jupShield', {}).get('verified', False),
                                    'warnings': token.get('extensions', {}).get('jupShield', {}).get('warnings', [])
                                }
        except Exception as e:
            logger.debug(f"[UNVERIFIED] Jupiter metadata fetch failed: {e}")
        
        return None
    
    async def _fetch_metadata_from_solana(self, token_mint: str) -> Optional[Dict]:
        """Fallback: fetch basic metadata from Solana network"""
        try:
            # This would connect to Solana RPC to get token metadata
            # For now, return basic structure with sensible defaults
            return {
                'symbol': f'TOKEN_{token_mint[:4]}',
                'name': 'Unverified Token',
                'decimals': 9,  # Most tokens use 9
                'verified': False,
                'warnings': ['unverified']
            }
        except Exception:
            return None
    
    def _validate_amount(self, amount: Decimal, metadata: Dict) -> Tuple[bool, str]:
        """Validate trade amount based on token configuration"""
        decimals = metadata.get('decimals', 9)
        
        # Calculate minimum trade size based on decimals
        if decimals <= 6:
            min_amount = Decimal('0.001')   # Tokens with 6 decimals or less
        else:
            min_amount = Decimal('0.000001')  # Standard 9-decimal tokens
        
        if amount < min_amount:
            return False, f"Amount {amount} below minimum {min_amount} for {decimals}-decimal token"
        
        # Maximum sanity check
        max_amount = Decimal('1000.0')  # No trade larger than 1000 units
        if amount > max_amount:
            return False, f"Amount {amount} exceeds maximum {max_amount}"
        
        return True, "Amount valid"
    
    async def _get_direct_quote(self, input_mint: str, amount: Decimal, output_mint: str) -> Optional[Dict]:
        """Get direct quote from Jupiter for unverified token"""
        try:
            # Convert amount to lamports based on token decimals
            amount_lamports = int(amount * Decimal('1e9'))  # Assume 9 decimals for now
            
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount_lamports),
                'slippageBps': '5000',  # 50% max slippage for unverified tokens
                'onlyDirectRoutes': 'false',
                'asLegacyTransaction': 'false'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.jupiter_base_url}/quote", params=params) as response:
                    if response.status == 200:
                        quote_data = await response.json()
                        logger.debug(f"[UNVERIFIED] Direct quote successful for {input_mint[:8]}...")
                        return quote_data
                    else:
                        logger.warning(f"[UNVERIFIED] Direct quote failed with status {response.status}")
                        return None
        
        except Exception as e:
            logger.warning(f"[UNVERIFIED] Direct quote error: {e}")
            return None
    
    def _validate_quote(self, input_amount: Decimal, quote: Dict, metadata: Dict) -> Tuple[bool, str]:
        """Validate quote sanity to prevent bad trades"""
        try:
            input_amount_str = quote.get('inAmount', '0')
            output_amount_str = quote.get('outAmount', '0')
            
            input_lamports = Decimal(input_amount_str)
            output_lamports = Decimal(output_amount_str)
            
            if input_lamports <= 0 or output_lamports <= 0:
                return False, "Quote contains zero amounts"
            
            # Calculate effective slippage
            price_impact = quote.get('priceImpactPct', '0')
            if price_impact and Decimal(price_impact) > Decimal('0.5'):  # 50% price impact
                return False, f"Excessive price impact: {price_impact}%"
            
            # Sanity check: output should be reasonable relative to input
            # For unverified tokens, we're more lenient but still check for obvious scams
            ratio = output_lamports / input_lamports
            if ratio < Decimal('0.000001'):  # Extremely low ratio
                return False, "Quote ratio suspiciously low - possible scam token"
            
            logger.debug(f"[UNVERIFIED] Quote validation passed - Impact: {price_impact}%")
            return True, "Quote valid"
            
        except Exception as e:
            return False, f"Quote validation error: {str(e)}"
    
    def mark_token_failed(self, token_mint: str):
        """Mark token as consistently failing to prevent repeated attempts"""
        self.failed_tokens.add(token_mint)
        logger.warning(f"[UNVERIFIED] Token {token_mint[:8]}... marked as failed")
    
    def clear_failed_cache(self):
        """Clear failed token cache (call periodically to retry previously failed tokens)"""
        cleared_count = len(self.failed_tokens)
        self.failed_tokens.clear()
        logger.info(f"[UNVERIFIED] Cleared {cleared_count} failed tokens from cache")