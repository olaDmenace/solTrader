"""
Birdeye API Client for Solana token trending data
Provides rate-limited, cached access to Birdeye trending endpoint
"""
import logging
import asyncio
import aiohttp
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class TrendingToken:
    """Structured trending token data"""
    address: str
    rank: int
    price: float
    price_24h_change_percent: float
    volume_24h_usd: float
    volume_24h_change_percent: float
    marketcap: float
    liquidity: float
    symbol: str
    name: str
    decimals: int = 9
    
    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> 'TrendingToken':
        """Create TrendingToken from API response data"""
        def safe_float(value, default=0.0):
            """Safely convert to float, handling None values"""
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        def safe_int(value, default=0):
            """Safely convert to int, handling None values"""
            if value is None:
                return default
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        
        return cls(
            address=data.get('address', ''),
            rank=safe_int(data.get('rank'), 999),
            price=safe_float(data.get('price')),
            price_24h_change_percent=safe_float(data.get('price24hChangePercent')),
            volume_24h_usd=safe_float(data.get('volume24hUSD')),
            volume_24h_change_percent=safe_float(data.get('volume24hChangePercent')),
            marketcap=safe_float(data.get('marketcap')),
            liquidity=safe_float(data.get('liquidity')),
            symbol=data.get('symbol', 'UNKNOWN'),
            name=data.get('name', 'Unknown Token'),
            decimals=safe_int(data.get('decimals'), 9)
        )

class BirdeyeClient:
    """Birdeye API client with rate limiting and caching"""
    
    def __init__(self, api_key: Optional[str] = None, cache_duration: int = 300):
        self.api_key = api_key
        self.cache_duration = cache_duration  # 5 minutes default
        self.base_url = "https://public-api.birdeye.so"
        self.session = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 2 seconds between requests
        self.request_count = 0
        self.request_window_start = time.time()
        self.max_requests_per_hour = 90  # Conservative limit
        
        # Caching
        self.cache = {}
        self.cache_timestamps = {}
        
        # Error tracking
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.api_available = True
        self.last_error_time = 0
        self.error_cooldown = 300  # 5 minutes cooldown after errors
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with optional API key"""
        headers = {
            'accept': 'application/json',
            'x-chain': 'solana',
            'User-Agent': 'SolTrader/1.0'
        }
        if self.api_key:
            headers['X-API-KEY'] = self.api_key
        return headers
    
    async def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        
        # Reset request count every hour
        if current_time - self.request_window_start >= 3600:
            self.request_count = 0
            self.request_window_start = current_time
        
        # Check hourly limit
        if self.request_count >= self.max_requests_per_hour:
            wait_time = 3600 - (current_time - self.request_window_start)
            logger.warning(f"[BIRDEYE] Rate limit reached, waiting {wait_time:.0f}s")
            await asyncio.sleep(wait_time)
            self.request_count = 0
            self.request_window_start = time.time()
        
        # Enforce minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[cache_key]
        return time.time() - cache_time < self.cache_duration
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = time.time()
    
    def _check_api_availability(self) -> bool:
        """Check if API is available after errors"""
        if not self.api_available:
            if time.time() - self.last_error_time > self.error_cooldown:
                logger.info("[BIRDEYE] Retrying API after cooldown period")
                self.api_available = True
                self.consecutive_errors = 0
            else:
                return False
        return True
    
    async def get_trending_tokens(self, limit: int = 20, offset: int = 0) -> Optional[List[TrendingToken]]:
        """Get trending tokens from Birdeye API"""
        try:
            # Check API availability
            if not self._check_api_availability():
                logger.debug("[BIRDEYE] API unavailable, skipping request")
                return None
            
            # Check cache first
            cache_key = f"trending_{limit}_{offset}"
            if self._is_cache_valid(cache_key):
                logger.debug(f"[BIRDEYE] Using cached trending data ({limit} tokens)")
                return self.cache[cache_key]
            
            if not self.session:
                logger.error("[BIRDEYE] No active session")
                return None
            
            # Rate limiting
            await self._rate_limit()
            
            # Make API request - Correct endpoint from Birdeye documentation
            url = f"{self.base_url}/defi/token_trending"
            
            # Ensure limit is within API constraints (1-20)
            api_limit = max(1, min(limit, 20))
            
            params = {
                'sort_by': 'rank',
                'sort_type': 'asc',
                'offset': offset,
                'limit': api_limit
            }
            
            headers = self._get_headers()
            logger.debug(f"[BIRDEYE] Requesting: {url}")
            logger.debug(f"[BIRDEYE] Headers: {headers}")
            logger.debug(f"[BIRDEYE] Params: {params}")
            
            async with self.session.get(url, headers=headers, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('success', False) and 'data' in data and 'tokens' in data['data']:
                        tokens = []
                        for token_data in data['data']['tokens']:
                            try:
                                token = TrendingToken.from_api_data(token_data)
                                tokens.append(token)
                            except Exception as e:
                                logger.warning(f"[BIRDEYE] Error parsing token data: {e}")
                                continue
                        
                        logger.info(f"[BIRDEYE] Successfully fetched {len(tokens)} trending tokens")
                        
                        # Cache the results
                        self._cache_data(cache_key, tokens)
                        
                        # Reset error counter on success
                        self.consecutive_errors = 0
                        
                        return tokens
                    else:
                        logger.warning(f"[BIRDEYE] Invalid response format: {data}")
                        return None
                
                elif response.status == 429:
                    # Rate limited
                    logger.warning("[BIRDEYE] Rate limited by API")
                    retry_after = int(response.headers.get('Retry-After', 60))
                    await asyncio.sleep(retry_after)
                    return None
                
                else:
                    response_text = await response.text()
                    logger.error(f"[BIRDEYE] API error: {response.status}")
                    logger.error(f"[BIRDEYE] Response body: {response_text}")
                    logger.error(f"[BIRDEYE] Request URL: {url}")
                    logger.error(f"[BIRDEYE] Request headers: {headers}")
                    await self._handle_api_error()
                    return None
                    
        except asyncio.TimeoutError:
            logger.error("[BIRDEYE] API request timeout")
            await self._handle_api_error()
            return None
        
        except Exception as e:
            logger.error(f"[BIRDEYE] Unexpected error: {e}")
            await self._handle_api_error()
            return None
    
    async def _handle_api_error(self):
        """Handle API errors with backoff"""
        self.consecutive_errors += 1
        
        if self.consecutive_errors >= self.max_consecutive_errors:
            logger.warning(f"[BIRDEYE] Too many consecutive errors ({self.consecutive_errors}), disabling API for {self.error_cooldown}s")
            self.api_available = False
            self.last_error_time = time.time()
    
    def get_cached_token_by_address(self, address: str) -> Optional[TrendingToken]:
        """Get specific token from cache by address"""
        for cached_tokens in self.cache.values():
            if isinstance(cached_tokens, list):
                for token in cached_tokens:
                    if isinstance(token, TrendingToken) and token.address == address:
                        return token
        return None
    
    def is_token_trending(self, address: str, max_rank: int = 50) -> bool:
        """Check if token is currently trending within rank limit"""
        token = self.get_cached_token_by_address(address)
        return token is not None and token.rank <= max_rank
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        current_time = time.time()
        cache_entries = len(self.cache)
        valid_cache_entries = sum(1 for key in self.cache.keys() if self._is_cache_valid(key))
        
        return {
            'api_available': self.api_available,
            'consecutive_errors': self.consecutive_errors,
            'requests_this_hour': self.request_count,
            'time_until_rate_reset': max(0, 3600 - (current_time - self.request_window_start)),
            'cache_entries': cache_entries,
            'valid_cache_entries': valid_cache_entries,
            'has_api_key': self.api_key is not None
        }