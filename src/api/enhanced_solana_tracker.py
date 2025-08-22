#!/usr/bin/env python3
"""
Enhanced Solana Tracker Client with Robust Error Handling
Production-ready version with comprehensive retry logic and error tracking
"""
import asyncio
import aiohttp
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import os

# Import our robust API utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.robust_api import robust_api_call, RobustHTTPClient, RetryConfig, ErrorSeverity

logger = logging.getLogger(__name__)

@dataclass
class TokenData:
    address: str
    symbol: str
    name: str
    price: float
    price_change_24h: float
    volume_24h: float
    market_cap: float
    liquidity: float
    age_minutes: int
    momentum_score: float
    source: str

class EnhancedSolanaTrackerClient:
    """Production-ready Solana Tracker client with comprehensive error handling"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('SOLANA_TRACKER_KEY')
        self.base_url = "https://data.solanatracker.io"
        
        # Rate limiting
        self.daily_limit = 333  # Free tier: 10k/month ÷ 30 days
        self.requests_today = 0
        self.last_reset = datetime.now().date()
        self.last_request_time = 0
        self.min_interval = 1.0  # 1 second between requests
        
        # Request tracking
        self.request_count = {
            'trending': 0,
            'volume': 0,
            'memescope': 0,
            'total': 0
        }
        
        # Caching for resilience
        self.cache = {
            'trending': {'data': [], 'timestamp': 0, 'ttl': 780},  # 13 minutes
            'volume': {'data': [], 'timestamp': 0, 'ttl': 900},    # 15 minutes
            'memescope': {'data': [], 'timestamp': 0, 'ttl': 1200} # 20 minutes
        }
        
        # Create robust HTTP client
        headers = {
            'User-Agent': 'SolTrader-Bot/1.0',
            'Accept': 'application/json'
        }
        if self.api_key:
            headers['x-api-key'] = self.api_key
        
        # Custom retry config for Solana Tracker
        retry_config = RetryConfig(
            max_attempts=4,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=1.5,
            retryable_status_codes=[429, 500, 502, 503, 504, 520, 521, 522, 523, 524]
        )
        
        self.http_client = RobustHTTPClient(
            base_url=self.base_url,
            component_name="solana_tracker",
            headers=headers,
            timeout=30.0,
            retry_config=retry_config
        )
        
        # Last request times for scheduling
        self.last_requests = {
            'trending': 0,
            'volume': 0,
            'memescope': 0
        }
        
        # Scheduling intervals (in seconds)
        self.intervals = {
            'trending': 780,   # 13 minutes
            'volume': 900,     # 15 minutes  
            'memescope': 1200  # 20 minutes
        }
    
    async def close(self):
        """Close HTTP session"""
        await self.http_client.close()
    
    def _reset_daily_counter_if_needed(self):
        """Reset request counter if new day"""
        try:
            today = datetime.now().date()
            if today > self.last_reset:
                old_count = self.requests_today
                self.requests_today = 0
                self.last_reset = today
                self.request_count = {k: 0 for k in self.request_count}
                logger.info(f"Daily request counter reset. New day: {today} (Previous: {old_count} requests)")
        except Exception as e:
            logger.error(f"Error during daily counter reset: {e}")
    
    def _can_make_request(self) -> bool:
        """Check if we can make a request without hitting rate limits"""
        self._reset_daily_counter_if_needed()
        return self.requests_today < self.daily_limit
    
    def _should_make_scheduled_request(self, endpoint_type: str) -> bool:
        """Check if enough time has passed for scheduled request"""
        last_time = self.last_requests.get(endpoint_type, 0)
        interval = self.intervals.get(endpoint_type, 600)
        return time.time() - last_time >= interval
    
    def _get_cached_data(self, cache_key: str) -> List[TokenData]:
        """Get cached data if still valid"""
        cache_entry = self.cache.get(cache_key, {})
        data = cache_entry.get('data', [])
        timestamp = cache_entry.get('timestamp', 0)
        ttl = cache_entry.get('ttl', 600)
        
        if data and (time.time() - timestamp) < ttl:
            return data
        return []
    
    def _cache_data(self, cache_key: str, data: List[TokenData]):
        """Cache data with timestamp"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': time.time(),
            'ttl': self.cache[cache_key].get('ttl', 600)
        }
    
    @robust_api_call(component="solana_tracker_api")
    async def _make_robust_request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make a robust API request with proper error handling"""
        
        # Check rate limits
        if not self._can_make_request():
            await asyncio.sleep(0.5)
            if not self._can_make_request():
                logger.warning("Rate limit active - skipping request")
                return None
        
        # Rate limiting delay
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        
        # Make the request using our robust HTTP client
        try:
            data = await self.http_client.get(endpoint, params=params)
            
            # Update counters on success
            self.last_request_time = time.time()
            self.requests_today += 1
            self.request_count['total'] += 1
            
            # Log success
            items_count = len(data.get('data', []) if isinstance(data, dict) else data if isinstance(data, list) else [])
            logger.info(f"API Response from {endpoint}: {items_count} items (Daily: {self.requests_today}/{self.daily_limit})")
            
            return data
            
        except aiohttp.ClientResponseError as e:
            if e.status == 401:
                logger.error("Solana Tracker API: 401 Unauthorized")
                logger.error("This usually means:")
                logger.error("1. API key is missing or invalid")
                logger.error("2. API key is not properly set in .env file") 
                logger.error("3. You may need to subscribe to a plan at https://solanatracker.io/")
                return None
            elif e.status == 429:
                logger.warning("Rate limit hit, backing off...")
                await asyncio.sleep(60)
                return None
            else:
                # Let the robust_api_call decorator handle other HTTP errors
                raise
    
    def _parse_token_data(self, raw_data: dict, source: str) -> List[TokenData]:
        """Parse raw API response into TokenData objects"""
        try:
            tokens = []
            data_list = raw_data.get('data', []) if isinstance(raw_data, dict) else raw_data
            
            for item in data_list:
                try:
                    token = TokenData(
                        address=item.get('address', ''),
                        symbol=item.get('symbol', 'UNK'),
                        name=item.get('name', 'Unknown'),
                        price=float(item.get('price', 0)),
                        price_change_24h=float(item.get('price_change_24h', 0)),
                        volume_24h=float(item.get('volume_24h', 0)),
                        market_cap=float(item.get('market_cap', 0)),
                        liquidity=float(item.get('liquidity', 0)),
                        age_minutes=int(item.get('age_minutes', 0)),
                        momentum_score=float(item.get('momentum_score', 0)),
                        source=source
                    )
                    tokens.append(token)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipped malformed token data: {e}")
                    continue
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error parsing token data from {source}: {e}")
            return []
    
    async def get_trending_tokens(self, limit: int = 50) -> List[TokenData]:
        """Get trending tokens with robust error handling and caching"""
        
        # Try cache first
        cached = self._get_cached_data('trending')
        if cached:
            logger.info(f"Using cached trending tokens: {len(cached)}")
            return cached[:limit]
        
        # Check if we should make scheduled request
        if not self._should_make_scheduled_request('trending'):
            if self.requests_today < 5:  # Allow first few requests for startup
                logger.info("📊 Rate limited but allowing initial trending request")
            else:
                logger.info("⏳ No cached trending tokens, waiting for next scheduled request")
                return []
        
        # Make the request
        logger.info("Fetching trending tokens from Solana Tracker...")
        endpoint = "tokens/trending"
        params = {'limit': limit}
        
        data = await self._make_robust_request(endpoint, params)
        if not data:
            # Return cached data even if expired as fallback
            cached_fallback = self.cache.get('trending', {}).get('data', [])
            if cached_fallback:
                logger.warning("API failed, using expired cache as fallback")
                return cached_fallback[:limit]
            return []
        
        # Parse and cache the results
        self.last_requests['trending'] = time.time()
        self.request_count['trending'] += 1
        
        tokens = self._parse_token_data(data, 'trending')
        if tokens:
            self._cache_data('trending', tokens)
            logger.info(f"Cached {len(tokens)} trending tokens")
        
        return tokens[:limit]
    
    async def get_volume_tokens(self, limit: int = 50) -> List[TokenData]:
        """Get high-volume tokens with robust error handling and caching"""
        
        # Try cache first  
        cached = self._get_cached_data('volume')
        if cached:
            logger.info(f"Using cached volume tokens: {len(cached)}")
            return cached[:limit]
        
        # Check scheduling
        if not self._should_make_scheduled_request('volume'):
            logger.info("⏳ No cached volume tokens, waiting for next scheduled request")
            return []
        
        # Make the request
        logger.info("Fetching volume tokens from Solana Tracker...")
        endpoint = "tokens/volume"
        params = {'limit': limit}
        
        data = await self._make_robust_request(endpoint, params)
        if not data:
            cached_fallback = self.cache.get('volume', {}).get('data', [])
            if cached_fallback:
                logger.warning("Volume API failed, using expired cache")
                return cached_fallback[:limit]
            return []
        
        # Parse and cache
        self.last_requests['volume'] = time.time()
        self.request_count['volume'] += 1
        
        tokens = self._parse_token_data(data, 'volume')
        if tokens:
            self._cache_data('volume', tokens)
            logger.info(f"Cached {len(tokens)} volume tokens")
        
        return tokens[:limit]
    
    async def get_memescope_tokens(self, limit: int = 50) -> List[TokenData]:
        """Get memescope tokens with robust error handling and caching"""
        
        # Try cache first
        cached = self._get_cached_data('memescope') 
        if cached:
            logger.info(f"Using cached memescope tokens: {len(cached)}")
            return cached[:limit]
        
        # Check scheduling
        if not self._should_make_scheduled_request('memescope'):
            logger.info("⏳ No cached memescope tokens, waiting for next scheduled request")
            return []
        
        # Make the request
        logger.info("Fetching memescope tokens from Solana Tracker...")
        endpoint = "tokens/memescope"
        params = {'limit': limit}
        
        data = await self._make_robust_request(endpoint, params)
        if not data:
            cached_fallback = self.cache.get('memescope', {}).get('data', [])
            if cached_fallback:
                logger.warning("Memescope API failed, using expired cache")
                return cached_fallback[:limit]
            return []
        
        # Parse and cache
        self.last_requests['memescope'] = time.time() 
        self.request_count['memescope'] += 1
        
        tokens = self._parse_token_data(data, 'memescope')
        if tokens:
            self._cache_data('memescope', tokens)
            logger.info(f"Cached {len(tokens)} memescope tokens")
        
        return tokens[:limit]
    
    async def get_all_tokens(self) -> List[TokenData]:
        """Get tokens from all sources, combining results intelligently"""
        logger.info("Fetching tokens from all Solana Tracker sources...")
        
        # Fetch from all endpoints concurrently
        results = await asyncio.gather(
            self.get_trending_tokens(30),
            self.get_volume_tokens(30), 
            self.get_memescope_tokens(30),
            return_exceptions=True
        )
        
        all_tokens = []
        source_names = ['trending', 'volume', 'memescope']
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {source_names[i]} tokens: {result}")
                continue
            elif isinstance(result, list):
                all_tokens.extend(result)
                logger.info(f"Added {len(result)} tokens from {source_names[i]}")
        
        # Remove duplicates by address, keeping the first occurrence
        seen_addresses = set()
        unique_tokens = []
        
        for token in all_tokens:
            if token.address not in seen_addresses:
                seen_addresses.add(token.address)
                unique_tokens.append(token)
        
        logger.info(f"Combined {len(unique_tokens)} unique tokens from {len([r for r in results if not isinstance(r, Exception)])} sources")
        
        return unique_tokens
    
    async def test_connection(self) -> bool:
        """Test API connectivity with robust error handling"""
        try:
            logger.info("Testing Solana Tracker API connection...")
            
            # Try a simple trending request
            data = await self._make_robust_request("tokens/trending", {"limit": 1})
            
            if data:
                logger.info("Solana Tracker API connection successful")
                return True
            else:
                logger.error("Solana Tracker API test failed - no data returned")
                return False
                
        except Exception as e:
            logger.error(f"Solana Tracker API connection test failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status and statistics"""
        cache_status = {}
        for key, entry in self.cache.items():
            cache_status[key] = {
                'items': len(entry.get('data', [])),
                'age_seconds': int(time.time() - entry.get('timestamp', 0)),
                'ttl_seconds': entry.get('ttl', 0)
            }
        
        return {
            'daily_requests': f"{self.requests_today}/{self.daily_limit}",
            'request_counts': self.request_count,
            'cache_status': cache_status,
            'rate_limited': not self._can_make_request(),
            'api_key_configured': bool(self.api_key)
        }

# Global instance for reuse
enhanced_solana_client = EnhancedSolanaTrackerClient()

async def get_enhanced_solana_tokens() -> List[TokenData]:
    """Convenience function to get tokens from enhanced client"""
    return await enhanced_solana_client.get_all_tokens()