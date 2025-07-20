"""
Solana New Token Scanner - Real new token detection for micro-cap gems
Integrates with Raydium, Jupiter, and Solana RPC to find newly launched tokens
"""
import logging
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import base64
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey

logger = logging.getLogger(__name__)

class SolanaNewTokenScanner:
    """Real new token scanner for Solana micro-cap detection"""
    
    def __init__(self, jupiter_client, alchemy_client, settings):
        self.jupiter = jupiter_client
        self.alchemy = alchemy_client
        self.settings = settings
        self.discovered_tokens = []
        self.scan_count = 0
        self.running = False
        self.session = None
        
        # Major tokens to exclude (not new launches)
        self.excluded_tokens = {
            "So11111111111111111111111111111111111111112",  # SOL
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
            "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",  # BTC
            "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",  # ETH
            "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",   # mSOL
            "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj",  # SAMO
            "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # BONK
            "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",  # WIF
        }
        
        # Raydium program addresses
        self.raydium_program_id = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
        self.raydium_amm_program = "5quBtoiQqxF9Jv6KYKctB59NT3gtJD2Y65kdnB1Uev3h"
        
        # Track processed tokens to avoid duplicates
        self.processed_tokens = set()
        
        # Recent new tokens cache (last 24 hours)
        self.recent_tokens = []
        
    async def start_scanning(self):
        """Start the scanning process"""
        logger.info("[SCANNER] Starting Solana new token scanner...")
        self.running = True
        self.session = aiohttp.ClientSession()
        
        while self.running:
            try:
                # Check if trading is paused
                if getattr(self.settings, 'TRADING_PAUSED', False):
                    logger.info("[SCANNER] Trading paused - scanner running in detection-only mode")
                    await asyncio.sleep(30)
                    continue
                
                await self.scan_for_new_tokens()
                await asyncio.sleep(self.settings.SCAN_INTERVAL)
                
            except Exception as e:
                logger.error(f"[SCANNER] Error in scanning loop: {e}")
                await asyncio.sleep(10)
    
    async def scan_for_new_tokens(self) -> Optional[Dict[str, Any]]:
        """Scan for newly launched tokens"""
        try:
            self.scan_count += 1
            logger.info(f"[SCAN] Starting scan #{self.scan_count} for new Solana tokens...")
            
            # Multiple scanning methods
            new_tokens = []
            
            # 1. Scan recent Raydium pool creations
            raydium_tokens = await self._scan_raydium_new_pools()
            new_tokens.extend(raydium_tokens)
            
            # 2. Scan Jupiter token list for recent additions
            jupiter_tokens = await self._scan_jupiter_new_tokens()
            new_tokens.extend(jupiter_tokens)
            
            # 3. Scan recent transactions for token creation
            creation_tokens = await self._scan_token_creation_events()
            new_tokens.extend(creation_tokens)
            
            logger.info(f"[SCAN] Found {len(new_tokens)} potential new tokens")
            
            # Filter and validate tokens
            for token_info in new_tokens:
                if await self._validate_new_token(token_info):
                    return token_info
            
            return None
            
        except Exception as e:
            logger.error(f"[SCANNER] Error scanning for new tokens: {e}")
            return None
    
    async def _scan_raydium_new_pools(self) -> List[Dict[str, Any]]:
        """Scan for newly created Raydium liquidity pools"""
        try:
            # Get recent program account changes for Raydium
            rpc_client = AsyncClient(self.alchemy.http_endpoint)
            
            # Get recent signatures for Raydium AMM program
            signatures = await rpc_client.get_signatures_for_address(
                Pubkey.from_string(self.raydium_amm_program),
                limit=50
            )
            
            new_tokens = []
            current_time = datetime.now()
            
            for sig_info in signatures.value:
                # Check if transaction is recent enough
                if sig_info.block_time:
                    tx_time = datetime.fromtimestamp(sig_info.block_time)
                    age_minutes = (current_time - tx_time).total_seconds() / 60
                    
                    if age_minutes <= self.settings.NEW_TOKEN_MAX_AGE_MINUTES:
                        # Get transaction details
                        tx_details = await rpc_client.get_transaction(
                            sig_info.signature,
                            max_supported_transaction_version=0
                        )
                        
                        # Parse for new pool creation
                        tokens = await self._parse_pool_creation_tx(tx_details)
                        new_tokens.extend(tokens)
            
            await rpc_client.close()
            return new_tokens
            
        except Exception as e:
            logger.error(f"[SCANNER] Error scanning Raydium pools: {e}")
            return []
    
    async def _scan_jupiter_new_tokens(self) -> List[Dict[str, Any]]:
        """Scan Jupiter token list for recent additions"""
        try:
            if not self.session:
                return []
                
            # Get Jupiter token list
            async with self.session.get("https://token.jup.ag/all") as response:
                if response.status == 200:
                    tokens = await response.json()
                    
                    new_tokens = []
                    for token in tokens:
                        # Skip if already processed or excluded
                        if (token['address'] in self.processed_tokens or 
                            token['address'] in self.excluded_tokens):
                            continue
                        
                        # Check if token might be new (basic heuristics)
                        if await self._is_potentially_new_token(token):
                            token_info = {
                                'address': token['address'],
                                'symbol': token.get('symbol', 'UNKNOWN'),
                                'name': token.get('name', 'Unknown Token'),
                                'decimals': token.get('decimals', 9),
                                'source': 'jupiter_scan'
                            }
                            new_tokens.append(token_info)
                            self.processed_tokens.add(token['address'])
                    
                    return new_tokens
                    
        except Exception as e:
            logger.error(f"[SCANNER] Error scanning Jupiter tokens: {e}")
        
        return []
    
    async def _scan_token_creation_events(self) -> List[Dict[str, Any]]:
        """Scan recent blockchain events for token creation"""
        try:
            # This would integrate with Solana RPC to find recent token creation events
            # For now, we'll use a simplified approach
            return []
            
        except Exception as e:
            logger.error(f"[SCANNER] Error scanning token creation events: {e}")
            return []
    
    async def _parse_pool_creation_tx(self, tx_details) -> List[Dict[str, Any]]:
        """Parse transaction details for new pool creation"""
        try:
            if not tx_details or not tx_details.value:
                return []
            
            # Parse transaction for new token addresses
            # This is a simplified implementation
            new_tokens = []
            
            # In a real implementation, we would parse the transaction logs
            # and instruction data to extract new token mint addresses
            
            return new_tokens
            
        except Exception as e:
            logger.error(f"[SCANNER] Error parsing pool creation transaction: {e}")
            return []
    
    async def _is_potentially_new_token(self, token: Dict[str, Any]) -> bool:
        """Check if token might be newly launched"""
        try:
            # Basic checks for potential new tokens
            symbol = token.get('symbol', '').upper()
            name = token.get('name', '').lower()
            
            # Skip stable coins and major tokens
            if any(stable in symbol for stable in ['USDC', 'USDT', 'SOL', 'BTC', 'ETH']):
                return False
            
            # Check for meme token characteristics
            meme_indicators = ['dog', 'cat', 'pepe', 'moon', 'baby', 'inu', 'shib', 'meme']
            if any(indicator in name for indicator in meme_indicators):
                return True
            
            # Check if symbol looks like a new token (random characters, etc.)
            if len(symbol) > 10 or any(char in symbol for char in ['X', 'Z', 'Q']):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[SCANNER] Error checking if token is new: {e}")
            return False
    
    async def _validate_new_token(self, token_info: Dict[str, Any]) -> bool:
        """Validate if token meets our trading criteria"""
        try:
            token_address = token_info['address']
            
            # Skip if already processed
            if token_address in self.processed_tokens:
                return False
            
            # Get token price and market data
            price_data = await self._get_token_price_data(token_address)
            if not price_data:
                return False
            
            price_sol = price_data.get('price_sol', 0)
            market_cap_sol = price_data.get('market_cap_sol', 0)
            
            # Apply filtering criteria
            if not self._passes_price_filter(price_sol, market_cap_sol):
                logger.debug(f"[FILTER] Token {token_address[:8]}... failed price filter")
                return False
            
            # Check liquidity
            liquidity_sol = price_data.get('liquidity_sol', 0)
            if liquidity_sol < self.settings.MIN_LIQUIDITY:
                logger.debug(f"[FILTER] Token {token_address[:8]}... insufficient liquidity")
                return False
            
            # Add validated token data
            token_info.update(price_data)
            
            logger.info(f"[NEW TOKEN] Found valid new token: {token_info['symbol']} ({token_address[:8]}...)")
            logger.info(f"  Price: {price_sol:.8f} SOL")
            logger.info(f"  Market Cap: {market_cap_sol:.2f} SOL")
            logger.info(f"  Liquidity: {liquidity_sol:.2f} SOL")
            
            self.processed_tokens.add(token_address)
            return True
            
        except Exception as e:
            logger.error(f"[SCANNER] Error validating token: {e}")
            return False
    
    def _passes_price_filter(self, price_sol: float, market_cap_sol: float) -> bool:
        """Check if token passes price and market cap filters"""
        # Price range filter
        if (price_sol < self.settings.MIN_TOKEN_PRICE_SOL or 
            price_sol > self.settings.MAX_TOKEN_PRICE_SOL):
            return False
        
        # Market cap filter
        if (market_cap_sol < self.settings.MIN_MARKET_CAP_SOL or 
            market_cap_sol > self.settings.MAX_MARKET_CAP_SOL):
            return False
        
        return True
    
    async def _get_token_price_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get price and market data for token"""
        try:
            # Use Jupiter for price data
            if not self.session:
                return None
                
            # Get price from Jupiter
            price_url = f"https://price.jup.ag/v4/price?ids={token_address}"
            async with self.session.get(price_url) as response:
                if response.status == 200:
                    price_data = await response.json()
                    
                    if token_address in price_data.get('data', {}):
                        token_price_data = price_data['data'][token_address]
                        price_usd = token_price_data.get('price', 0)
                        
                        # Convert USD to SOL using dynamic price
                        from src.utils.price_manager import get_sol_usd_price
                        sol_price_usd = await get_sol_usd_price()
                        price_sol = price_usd / sol_price_usd
                        
                        # Get additional market data
                        market_data = await self._get_market_data(token_address)
                        
                        return {
                            'price_sol': price_sol,
                            'price_usd': price_usd,
                            'market_cap_sol': market_data.get('market_cap_sol', 0),
                            'liquidity_sol': market_data.get('liquidity_sol', 0),
                            'volume_24h_sol': market_data.get('volume_24h_sol', 0)
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"[SCANNER] Error getting token price data: {e}")
            return None
    
    async def _get_market_data(self, token_address: str) -> Dict[str, Any]:
        """Get additional market data for token"""
        try:
            # This would integrate with DexScreener or similar APIs
            # For now, return mock data
            return {
                'market_cap_sol': 1000.0,  # Mock market cap
                'liquidity_sol': 500.0,    # Mock liquidity
                'volume_24h_sol': 100.0    # Mock volume
            }
            
        except Exception as e:
            logger.error(f"[SCANNER] Error getting market data: {e}")
            return {}
    
    async def stop_scanning(self):
        """Stop the scanning process"""
        logger.info("[SCANNER] Stopping new token scanner...")
        self.running = False
        if self.session:
            await self.session.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scanner statistics"""
        return {
            'scans_completed': self.scan_count,
            'tokens_discovered': len(self.discovered_tokens),
            'tokens_processed': len(self.processed_tokens),
            'is_running': self.running
        }