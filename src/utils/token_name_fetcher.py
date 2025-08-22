#!/usr/bin/env python3
"""
Token Name Fetcher for Production Dashboard
Fetches real token names from multiple sources for better UX
"""
import logging
import aiohttp
import asyncio
from typing import Dict, Optional, Any
import json

logger = logging.getLogger(__name__)

class TokenNameFetcher:
    """Fetches token names from multiple sources for better dashboard UX"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Dict[str, str]] = {}  # Cache token info
        
    async def ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"User-Agent": "SolTrader/1.0"},
                timeout=aiohttp.ClientTimeout(total=10)
            )
    
    async def close(self):
        """Close session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_token_info(self, token_address: str) -> Dict[str, str]:
        """
        Get token information from multiple sources
        Returns dict with name, symbol, source
        """
        if token_address in self.cache:
            return self.cache[token_address]
        
        # Try multiple sources in order of reliability
        sources = [
            self._fetch_from_dexscreener,
            self._fetch_from_solscan,
            self._fetch_from_pump_fun,
            self._create_fallback_name
        ]
        
        for source_func in sources:
            try:
                result = await source_func(token_address)
                if result and result.get("name") != "Unknown":
                    self.cache[token_address] = result
                    return result
            except Exception as e:
                logger.debug(f"Token source failed: {source_func.__name__}: {e}")
                continue
        
        # Final fallback
        fallback = self._create_fallback_name(token_address)
        self.cache[token_address] = fallback
        return fallback
    
    async def _fetch_from_dexscreener(self, token_address: str) -> Optional[Dict[str, str]]:
        """Fetch from DexScreener API"""
        await self.ensure_session()
        
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                pairs = data.get("pairs", [])
                
                if pairs:
                    pair = pairs[0]
                    base_token = pair.get("baseToken", {})
                    
                    return {
                        "name": base_token.get("name", "Unknown"),
                        "symbol": base_token.get("symbol", token_address[:8]),
                        "source": "dexscreener"
                    }
        return None
    
    async def _fetch_from_solscan(self, token_address: str) -> Optional[Dict[str, str]]:
        """Fetch from Solscan API"""
        await self.ensure_session()
        
        url = f"https://api.solscan.io/token/meta"
        params = {"token": token_address}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                return {
                    "name": data.get("name", "Unknown"),
                    "symbol": data.get("symbol", token_address[:8]),
                    "source": "solscan"
                }
        return None
    
    async def _fetch_from_pump_fun(self, token_address: str) -> Optional[Dict[str, str]]:
        """Try to detect Pump.fun tokens and create meaningful names"""
        # Pump.fun tokens often end with "pump" 
        if token_address.endswith("pump"):
            return {
                "name": f"Pump Token {token_address[:8]}",
                "symbol": f"${token_address[:6].upper()}",
                "source": "pump_fun"
            }
        return None
    
    def _create_fallback_name(self, token_address: str) -> Dict[str, str]:
        """Create fallback name for unknown tokens"""
        return {
            "name": f"Token {token_address[:8]}...",
            "symbol": token_address[:6].upper(),
            "source": "fallback"
        }
    
    async def get_batch_token_info(self, token_addresses: list) -> Dict[str, Dict[str, str]]:
        """Get token info for multiple addresses in batch"""
        tasks = [self.get_token_info(addr) for addr in token_addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        batch_info = {}
        for addr, result in zip(token_addresses, results):
            if isinstance(result, dict):
                batch_info[addr] = result
            else:
                batch_info[addr] = self._create_fallback_name(addr)
        
        return batch_info

# Global instance for reuse
token_fetcher = TokenNameFetcher()

async def get_token_name_info(token_address: str) -> Dict[str, str]:
    """Convenience function to get token info"""
    return await token_fetcher.get_token_info(token_address)

async def cleanup_token_fetcher():
    """Cleanup function"""
    await token_fetcher.close()