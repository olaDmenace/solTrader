"""
Token Metadata Caching System

Provides intelligent caching for token metadata from multiple sources:
- Jupiter token list (primary)
- Solana RPC metadata (fallback)
- Birdeye API (pricing/market data)
- Custom token registry

Features:
- Multi-source data aggregation
- Persistent storage with expiration
- Background refresh mechanisms
- Batch loading for performance
- Thread-safe operations
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
import aiohttp
import aiofiles
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class TokenMetadata:
    """Token metadata structure"""
    address: str
    symbol: str = "UNKNOWN"
    name: str = "Unknown Token"
    decimals: int = 9
    verified: bool = False
    logo_uri: Optional[str] = None
    
    # Market data
    price_usd: Optional[float] = None
    market_cap_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    
    # Trust indicators
    jupiter_verified: bool = False
    birdeye_verified: bool = False
    warnings: List[str] = field(default_factory=list)
    
    # Cache metadata
    cached_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    source: str = "unknown"
    
    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at
    
    @property
    def display_name(self) -> str:
        """Get user-friendly display name"""
        if self.symbol and self.symbol != "UNKNOWN":
            return f"{self.symbol}"
        return f"{self.address[:8]}..."
    
    @property
    def full_display_name(self) -> str:
        """Get full display name with symbol and name"""
        if self.symbol and self.symbol != "UNKNOWN":
            if self.name and self.name != "Unknown Token":
                return f"{self.symbol} ({self.name})"
            return self.symbol
        return f"{self.address[:8]}..."

class TokenMetadataCache:
    """Intelligent token metadata caching system"""
    
    def __init__(self, cache_dir: str = "./data/token_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for fast access
        self.memory_cache: Dict[str, TokenMetadata] = {}
        self.cache_lock = Lock()
        
        # Configuration
        self.default_expiry_hours = 24
        self.max_memory_cache_size = 5000
        self.batch_size = 100
        
        # Data sources
        self.jupiter_token_list_url = "https://token.jup.ag/all"
        self.birdeye_api_key = os.getenv('BIRDEYE_API_KEY')
        
        # Background refresh tracking
        self.last_full_refresh = None
        self.refresh_lock = asyncio.Lock()
        
        logger.info(f"[CACHE] Token metadata cache initialized: {self.cache_dir}")
    
    async def get_token_metadata(self, token_address: str, force_refresh: bool = False) -> Optional[TokenMetadata]:
        """
        Get token metadata with intelligent caching
        
        Args:
            token_address: Token mint address
            force_refresh: Force refresh from remote sources
            
        Returns:
            TokenMetadata or None if not found
        """
        try:
            # Check memory cache first
            if not force_refresh and token_address in self.memory_cache:
                metadata = self.memory_cache[token_address]
                if not metadata.is_expired:
                    return metadata
                else:
                    logger.debug(f"[CACHE] Expired metadata for {token_address[:8]}...")
            
            # Check persistent cache
            if not force_refresh:
                metadata = await self._load_from_disk(token_address)
                if metadata and not metadata.is_expired:
                    # Update memory cache
                    with self.cache_lock:
                        self.memory_cache[token_address] = metadata
                    return metadata
            
            # Fetch fresh data
            metadata = await self._fetch_fresh_metadata(token_address)
            if metadata:
                await self._save_to_cache(metadata)
                return metadata
            
            # Return stale data if available as fallback
            if token_address in self.memory_cache:
                logger.warning(f"[CACHE] Using stale metadata for {token_address[:8]}...")
                return self.memory_cache[token_address]
                
            return None
            
        except Exception as e:
            logger.error(f"[CACHE] Failed to get metadata for {token_address}: {e}")
            return None
    
    async def get_batch_metadata(self, token_addresses: List[str]) -> Dict[str, Optional[TokenMetadata]]:
        """
        Get metadata for multiple tokens efficiently
        
        Args:
            token_addresses: List of token mint addresses
            
        Returns:
            Dict mapping addresses to metadata
        """
        results = {}
        missing_addresses = []
        
        # Check cache for existing data
        for address in token_addresses:
            metadata = await self.get_token_metadata(address, force_refresh=False)
            results[address] = metadata
            if not metadata:
                missing_addresses.append(address)
        
        # Batch fetch missing metadata
        if missing_addresses:
            logger.info(f"[CACHE] Batch fetching {len(missing_addresses)} missing tokens")
            await self._batch_fetch_metadata(missing_addresses, results)
        
        return results
    
    async def _fetch_fresh_metadata(self, token_address: str) -> Optional[TokenMetadata]:
        """Fetch fresh metadata from multiple sources"""
        try:
            # Try Jupiter first (most reliable)
            metadata = await self._fetch_from_jupiter(token_address)
            if metadata:
                # Enhance with Birdeye market data
                await self._enhance_with_birdeye(metadata)
                return metadata
            
            # Fallback to Solana RPC metadata
            metadata = await self._fetch_from_solana_rpc(token_address)
            if metadata:
                await self._enhance_with_birdeye(metadata)
                return metadata
                
            # Create minimal metadata as last resort
            return TokenMetadata(
                address=token_address,
                symbol=f"TOKEN_{token_address[:4]}",
                name="Unknown Token",
                source="minimal",
                expires_at=datetime.now() + timedelta(hours=1)  # Short expiry for unknown tokens
            )
            
        except Exception as e:
            logger.error(f"[CACHE] Error fetching metadata for {token_address}: {e}")
            return None
    
    async def _fetch_from_jupiter(self, token_address: str) -> Optional[TokenMetadata]:
        """Fetch metadata from Jupiter token list"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.jupiter_token_list_url) as response:
                    if response.status == 200:
                        tokens = await response.json()
                        for token in tokens:
                            if token.get('address') == token_address:
                                metadata = TokenMetadata(
                                    address=token_address,
                                    symbol=token.get('symbol', 'UNKNOWN'),
                                    name=token.get('name', 'Unknown Token'),
                                    decimals=token.get('decimals', 9),
                                    logo_uri=token.get('logoURI'),
                                    jupiter_verified=token.get('extensions', {}).get('jupShield', {}).get('verified', False),
                                    warnings=token.get('extensions', {}).get('jupShield', {}).get('warnings', []),
                                    verified=token.get('extensions', {}).get('jupShield', {}).get('verified', False),
                                    source="jupiter",
                                    expires_at=datetime.now() + timedelta(hours=self.default_expiry_hours)
                                )
                                logger.debug(f"[CACHE] Jupiter metadata found for {token_address[:8]}...")
                                return metadata
                        
                        logger.debug(f"[CACHE] Token {token_address[:8]}... not found in Jupiter list")
                        return None
                    else:
                        logger.warning(f"[CACHE] Jupiter API returned status {response.status}")
                        return None
                        
        except Exception as e:
            logger.warning(f"[CACHE] Jupiter fetch failed: {e}")
            return None
    
    async def _fetch_from_solana_rpc(self, token_address: str) -> Optional[TokenMetadata]:
        """Fetch basic metadata from Solana RPC (fallback)"""
        try:
            # This would typically query the Solana RPC for token metadata
            # For now, create a basic structure
            return TokenMetadata(
                address=token_address,
                symbol=f"TOKEN_{token_address[:4]}",
                name="Unverified Token",
                decimals=9,  # Most tokens use 9
                verified=False,
                warnings=['unverified'],
                source="solana_rpc",
                expires_at=datetime.now() + timedelta(hours=6)  # Shorter expiry for RPC data
            )
        except Exception as e:
            logger.warning(f"[CACHE] Solana RPC fetch failed: {e}")
            return None
    
    async def _enhance_with_birdeye(self, metadata: TokenMetadata):
        """Enhance metadata with Birdeye market data"""
        if not self.birdeye_api_key:
            return
        
        try:
            url = f"https://public-api.birdeye.so/public/price"
            params = {"address": metadata.address}
            headers = {"X-API-KEY": self.birdeye_api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success') and data.get('data'):
                            price_data = data['data']
                            metadata.price_usd = price_data.get('value')
                            metadata.birdeye_verified = True
                            logger.debug(f"[CACHE] Enhanced {metadata.address[:8]}... with Birdeye data")
                    else:
                        logger.debug(f"[CACHE] Birdeye API returned status {response.status} for {metadata.address[:8]}...")
                        
        except Exception as e:
            logger.debug(f"[CACHE] Birdeye enhancement failed: {e}")
    
    async def _batch_fetch_metadata(self, addresses: List[str], results: Dict[str, Optional[TokenMetadata]]):
        """Batch fetch metadata for multiple addresses"""
        # Process in chunks to avoid overwhelming APIs
        for i in range(0, len(addresses), self.batch_size):
            chunk = addresses[i:i + self.batch_size]
            
            # Create tasks for parallel fetching
            tasks = []
            for address in chunk:
                tasks.append(self._fetch_fresh_metadata(address))
            
            # Execute in parallel with rate limiting
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for address, metadata in zip(chunk, chunk_results):
                if isinstance(metadata, Exception):
                    logger.error(f"[CACHE] Batch fetch failed for {address}: {metadata}")
                    results[address] = None
                else:
                    results[address] = metadata
                    if metadata:
                        await self._save_to_cache(metadata)
            
            # Rate limiting between chunks
            if i + self.batch_size < len(addresses):
                await asyncio.sleep(0.1)
    
    async def _save_to_cache(self, metadata: TokenMetadata):
        """Save metadata to both memory and disk cache"""
        try:
            # Update memory cache
            with self.cache_lock:
                self.memory_cache[metadata.address] = metadata
                # Trim cache if too large
                if len(self.memory_cache) > self.max_memory_cache_size:
                    # Remove oldest entries (simple LRU)
                    oldest_keys = list(self.memory_cache.keys())[:100]
                    for key in oldest_keys:
                        del self.memory_cache[key]
            
            # Save to disk
            await self._save_to_disk(metadata)
            
        except Exception as e:
            logger.error(f"[CACHE] Failed to save metadata for {metadata.address}: {e}")
    
    async def _save_to_disk(self, metadata: TokenMetadata):
        """Save metadata to persistent disk cache"""
        try:
            cache_file = self.cache_dir / f"{metadata.address}.json"
            data = asdict(metadata)
            
            # Convert datetime objects to ISO strings
            if metadata.cached_at:
                data['cached_at'] = metadata.cached_at.isoformat()
            if metadata.expires_at:
                data['expires_at'] = metadata.expires_at.isoformat()
            
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
                
        except Exception as e:
            logger.error(f"[CACHE] Failed to save to disk: {e}")
    
    async def _load_from_disk(self, token_address: str) -> Optional[TokenMetadata]:
        """Load metadata from persistent disk cache"""
        try:
            cache_file = self.cache_dir / f"{token_address}.json"
            if not cache_file.exists():
                return None
            
            async with aiofiles.open(cache_file, 'r') as f:
                data = json.loads(await f.read())
            
            # Convert ISO strings back to datetime objects
            if data.get('cached_at'):
                data['cached_at'] = datetime.fromisoformat(data['cached_at'])
            if data.get('expires_at'):
                data['expires_at'] = datetime.fromisoformat(data['expires_at'])
            
            return TokenMetadata(**data)
            
        except Exception as e:
            logger.debug(f"[CACHE] Failed to load from disk: {e}")
            return None
    
    async def refresh_popular_tokens(self):
        """Background refresh of popular/frequently accessed tokens"""
        try:
            async with self.refresh_lock:
                # Get list of popular tokens from Jupiter
                popular_tokens = await self._get_popular_token_list()
                if popular_tokens:
                    logger.info(f"[CACHE] Refreshing {len(popular_tokens)} popular tokens")
                    await self.get_batch_metadata(popular_tokens)
                    self.last_full_refresh = datetime.now()
                    
        except Exception as e:
            logger.error(f"[CACHE] Popular token refresh failed: {e}")
    
    async def _get_popular_token_list(self) -> List[str]:
        """Get list of popular token addresses for background refresh"""
        try:
            # This could be enhanced to get actual popular tokens
            # For now, return SOL and USDC addresses
            return [
                "So11111111111111111111111111111111111111112",  # SOL
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",   # USDC
            ]
        except Exception:
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        memory_size = len(self.memory_cache)
        disk_files = len(list(self.cache_dir.glob("*.json")))
        
        return {
            "memory_cache_size": memory_size,
            "disk_cache_files": disk_files,
            "last_refresh": self.last_full_refresh.isoformat() if self.last_full_refresh else None,
            "cache_directory": str(self.cache_dir)
        }
    
    async def clear_expired_cache(self):
        """Clear expired entries from cache"""
        try:
            # Clear memory cache
            expired_keys = []
            with self.cache_lock:
                for address, metadata in self.memory_cache.items():
                    if metadata.is_expired:
                        expired_keys.append(address)
                
                for key in expired_keys:
                    del self.memory_cache[key]
            
            # Clear disk cache
            expired_files = 0
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    async with aiofiles.open(cache_file, 'r') as f:
                        data = json.loads(await f.read())
                    
                    if data.get('expires_at'):
                        expires_at = datetime.fromisoformat(data['expires_at'])
                        if datetime.now() > expires_at:
                            cache_file.unlink()
                            expired_files += 1
                            
                except Exception:
                    continue
            
            logger.info(f"[CACHE] Cleared {len(expired_keys)} memory entries and {expired_files} disk files")
            
        except Exception as e:
            logger.error(f"[CACHE] Failed to clear expired cache: {e}")

# Global cache instance
_global_cache: Optional[TokenMetadataCache] = None

def get_token_cache() -> TokenMetadataCache:
    """Get global token metadata cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = TokenMetadataCache()
    return _global_cache

async def get_token_info(token_address: str) -> Optional[TokenMetadata]:
    """Convenience function to get token info"""
    cache = get_token_cache()
    return await cache.get_token_metadata(token_address)

async def get_token_display_name(token_address: str) -> str:
    """Get user-friendly token display name"""
    metadata = await get_token_info(token_address)
    if metadata:
        return metadata.display_name
    return f"{token_address[:8]}..."