"""
Dynamic SOL Price Manager
Fetches real-time SOL/USD price for accurate volume calculations
"""
import logging
import asyncio
import aiohttp
import time
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DynamicPriceManager:
    """Manages dynamic SOL/USD price fetching with caching"""
    
    def __init__(self, cache_duration: int = 300):  # 5 minutes cache
        self.cache_duration = cache_duration
        self.cached_price: Optional[float] = None
        self.cache_timestamp: Optional[datetime] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.fallback_price = 200.0  # Fallback if all APIs fail
        
    async def get_sol_usd_price(self) -> float:
        """Get current SOL/USD price with caching"""
        try:
            # Check cache first
            if self._is_cache_valid():
                logger.debug(f"[PRICE] Using cached SOL price: ${self.cached_price:.2f}")
                return self.cached_price
            
            # Fetch new price
            new_price = await self._fetch_sol_price()
            if new_price:
                self.cached_price = new_price
                self.cache_timestamp = datetime.now()
                logger.info(f"[PRICE] Updated SOL price: ${new_price:.2f}")
                return new_price
            else:
                # Use cached price if available, otherwise fallback
                if self.cached_price:
                    logger.warning(f"[PRICE] API failed, using stale cached price: ${self.cached_price:.2f}")
                    return self.cached_price
                else:
                    logger.warning(f"[PRICE] API failed, using fallback price: ${self.fallback_price:.2f}")
                    return self.fallback_price
                    
        except Exception as e:
            logger.error(f"[PRICE] Error getting SOL price: {e}")
            return self.cached_price or self.fallback_price
    
    def _is_cache_valid(self) -> bool:
        """Check if cached price is still valid"""
        if not self.cached_price or not self.cache_timestamp:
            return False
        
        age = (datetime.now() - self.cache_timestamp).total_seconds()
        return age < self.cache_duration
    
    async def _fetch_sol_price(self) -> Optional[float]:
        """Fetch SOL price from multiple sources"""
        # Method 1: Jupiter Price API
        price = await self._fetch_from_jupiter()
        if price:
            return price
            
        # Method 2: CoinGecko API (backup)
        price = await self._fetch_from_coingecko()
        if price:
            return price
            
        # Method 3: Birdeye API (if available)
        price = await self._fetch_from_birdeye()
        if price:
            return price
            
        return None
    
    async def _fetch_from_jupiter(self) -> Optional[float]:
        """Fetch SOL price from Jupiter Price API"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # SOL token address
            sol_address = "So11111111111111111111111111111111111111112"
            url = f"https://price.jup.ag/v4/price?ids={sol_address}"
            
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if sol_address in data.get('data', {}):
                        price = float(data['data'][sol_address]['price'])
                        logger.debug(f"[PRICE] Jupiter SOL price: ${price:.2f}")
                        return price
                        
        except Exception as e:
            logger.debug(f"[PRICE] Jupiter price fetch failed: {e}")
        
        return None
    
    async def _fetch_from_coingecko(self) -> Optional[float]:
        """Fetch SOL price from CoinGecko API"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
            
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'solana' in data and 'usd' in data['solana']:
                        price = float(data['solana']['usd'])
                        logger.debug(f"[PRICE] CoinGecko SOL price: ${price:.2f}")
                        return price
                        
        except Exception as e:
            logger.debug(f"[PRICE] CoinGecko price fetch failed: {e}")
        
        return None
    
    async def _fetch_from_birdeye(self) -> Optional[float]:
        """Fetch SOL price from Birdeye API (if available)"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Birdeye SOL price endpoint
            url = "https://public-api.birdeye.so/public/price?address=So11111111111111111111111111111111111111112"
            
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data and 'value' in data['data']:
                        price = float(data['data']['value'])
                        logger.debug(f"[PRICE] Birdeye SOL price: ${price:.2f}")
                        return price
                        
        except Exception as e:
            logger.debug(f"[PRICE] Birdeye price fetch failed: {e}")
        
        return None
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None

# Global price manager instance
_price_manager = None

async def get_sol_usd_price() -> float:
    """Get current SOL/USD price (global function)"""
    global _price_manager
    
    if _price_manager is None:
        _price_manager = DynamicPriceManager()
    
    return await _price_manager.get_sol_usd_price()

async def cleanup_price_manager():
    """Cleanup global price manager"""
    global _price_manager
    
    if _price_manager:
        await _price_manager.close()
        _price_manager = None