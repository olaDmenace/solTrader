import asyncio
import aiohttp
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

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

class GeckoTerminalClient:
    """
    FREE API client for GeckoTerminal - NO API KEY REQUIRED!
    Rate limit: 30 calls/minute = 43,200/day = 1,296,000/month
    Perfect replacement for quota-exhausted Solana Tracker API
    """
    
    def __init__(self):
        self.base_url = "https://api.geckoterminal.com/api/v2"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting - Conservative: 25 calls/minute (30 is limit)
        self.max_calls_per_minute = 25
        self.calls_this_minute = []
        self.last_request_time = 0
        self.min_interval = 2.5  # 2.5 seconds between requests
        
        # Request tracking
        self.request_count = {
            'trending': 0,
            'volume': 0,
            'new_pools': 0,
            'total': 0
        }
        
        # Cache for reducing API calls
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        logger.info("GeckoTerminal client initialized - FREE API with 43K+ daily requests")

    async def __aenter__(self):
        await self.start_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def start_session(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'SolTrader-Bot/1.0',
                'Accept': 'application/json'
            }
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )

    async def close_session(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def close(self):
        """Alias for close_session for compatibility"""
        await self.close_session()

    def _can_make_request(self) -> bool:
        """Check if we can make a request based on rate limits"""
        now = time.time()
        
        # Clean up old requests (older than 1 minute)
        self.calls_this_minute = [
            call_time for call_time in self.calls_this_minute 
            if now - call_time < 60
        ]
        
        # Check if we're under rate limit
        if len(self.calls_this_minute) >= self.max_calls_per_minute:
            logger.warning(f"Rate limit hit: {len(self.calls_this_minute)}/{self.max_calls_per_minute} calls/minute")
            return False
        
        # Check minimum interval
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_interval:
            return False
        
        return True

    async def _make_request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make rate-limited request to GeckoTerminal API"""
        if not self._can_make_request():
            # Wait for rate limit to clear
            wait_time = self.min_interval - (time.time() - self.last_request_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            if not self._can_make_request():
                logger.warning("Still rate limited after waiting")
                return None

        if not self.session:
            await self.start_session()

        url = f"{self.base_url}/{endpoint}"
        
        try:
            logger.debug(f"Making request to: {url}")
            async with self.session.get(url, params=params) as response:
                now = time.time()
                self.last_request_time = now
                self.calls_this_minute.append(now)
                self.request_count['total'] += 1
                
                if response.status == 200:
                    data = await response.json()
                    items_count = len(data.get('data', []))
                    logger.info(f"GeckoTerminal API: {items_count} items from {endpoint} ({len(self.calls_this_minute)}/25 calls/min)")
                    return data
                elif response.status == 429:
                    logger.warning("Rate limit hit, backing off...")
                    await asyncio.sleep(60)
                    return None
                else:
                    logger.error(f"API request failed: {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error response: {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Request timeout for {url}")
            return None
        except Exception as e:
            logger.error(f"Request error for {url}: {str(e)}")
            return None

    async def get_trending_tokens(self, limit: int = 50) -> List[TokenData]:
        """Get trending tokens from Solana network"""
        # Check cache first
        cached = self._get_cached_data('trending')
        if cached:
            logger.info(f"Using cached trending tokens: {len(cached)}")
            return cached

        endpoint = "networks/solana/trending_pools"
        params = {'include': 'base_token'}
        
        data = await self._make_request(endpoint, params)
        if not data:
            return []

        self.request_count['trending'] += 1
        
        tokens = []
        pools = data.get('data', [])
        
        for pool in pools[:limit]:  # Limit results
            try:
                attributes = pool.get('attributes', {})
                base_token = pool.get('relationships', {}).get('base_token', {}).get('data', {})
                
                # Get token address and basic info
                token_address = base_token.get('id', '')
                if not token_address:
                    continue
                
                # Extract pool data
                name = attributes.get('name', '')
                address_in_name = attributes.get('address', token_address)
                
                # Get price and market data
                base_token_price_usd = float(attributes.get('base_token_price_usd') or 0)
                price_change_data = attributes.get('price_change_percentage', {}) or {}
                price_change_24h = float(price_change_data.get('h24') or 0)
                volume_data = attributes.get('volume_usd', {}) or {}
                volume_24h = float(volume_data.get('h24') or 0)
                reserve_in_usd = float(attributes.get('reserve_in_usd') or 0)
                market_cap_usd = float(attributes.get('market_cap_usd') or 0)
                
                # Try to extract symbol from name or use first part
                symbol = ''
                if '/' in name:
                    symbol = name.split('/')[0].strip()
                else:
                    # Use address suffix as symbol
                    symbol = f"Token{token_address[-6:].upper()}"
                
                # Calculate token age (assume new for trending)
                age_minutes = 60  # Assume trending tokens are relatively new
                
                token = TokenData(
                    address=token_address,
                    symbol=symbol,
                    name=name or symbol,
                    price=base_token_price_usd,
                    price_change_24h=price_change_24h,
                    volume_24h=volume_24h,
                    market_cap=market_cap_usd,
                    liquidity=reserve_in_usd,  # Total pool liquidity
                    age_minutes=age_minutes,
                    momentum_score=self._calculate_momentum_score(price_change_24h, volume_24h, reserve_in_usd),
                    source='trending'
                )
                tokens.append(token)
                logger.debug(f"Parsed trending token: {symbol} - Price: ${token.price:.8f}, Change: {token.price_change_24h:.1f}%, Volume: ${token.volume_24h:.0f}")
                
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error parsing trending pool data: {e}")
                continue

        self._cache_data('trending', tokens)
        logger.info(f"Retrieved {len(tokens)} trending tokens from GeckoTerminal")
        return tokens

    async def get_volume_tokens(self, limit: int = 50) -> List[TokenData]:
        """Get high volume tokens - using new pools as proxy since they often have high volume"""
        # Check cache first
        cached = self._get_cached_data('volume')
        if cached:
            logger.info(f"Using cached volume tokens: {len(cached)}")
            return cached

        endpoint = "networks/solana/new_pools"
        params = {'include': 'base_token'}
        
        data = await self._make_request(endpoint, params)
        if not data:
            return []

        self.request_count['volume'] += 1
        
        tokens = []
        pools = data.get('data', [])
        
        for pool in pools[:limit]:  # Limit results
            try:
                attributes = pool.get('attributes', {})
                base_token = pool.get('relationships', {}).get('base_token', {}).get('data', {})
                
                # Get token address and basic info
                token_address = base_token.get('id', '')
                if not token_address:
                    continue
                
                # Extract pool data
                name = attributes.get('name', '')
                
                # Get price and market data
                base_token_price_usd = float(attributes.get('base_token_price_usd') or 0)
                price_change_data = attributes.get('price_change_percentage', {}) or {}
                price_change_24h = float(price_change_data.get('h24') or 0)
                volume_data = attributes.get('volume_usd', {}) or {}
                volume_24h = float(volume_data.get('h24') or 0)
                reserve_in_usd = float(attributes.get('reserve_in_usd') or 0)
                market_cap_usd = float(attributes.get('market_cap_usd') or 0)
                
                # Calculate age from pool creation
                pool_created_at = attributes.get('pool_created_at', '')
                age_minutes = self._calculate_token_age(pool_created_at)
                
                # Try to extract symbol from name
                symbol = ''
                if '/' in name:
                    symbol = name.split('/')[0].strip()
                else:
                    symbol = f"Token{token_address[-6:].upper()}"
                
                token = TokenData(
                    address=token_address,
                    symbol=symbol,
                    name=name or symbol,
                    price=base_token_price_usd,
                    price_change_24h=price_change_24h,
                    volume_24h=volume_24h,
                    market_cap=market_cap_usd,
                    liquidity=reserve_in_usd,
                    age_minutes=age_minutes,
                    momentum_score=self._calculate_momentum_score(price_change_24h, volume_24h, reserve_in_usd),
                    source='volume'
                )
                tokens.append(token)
                logger.debug(f"Parsed volume token: {symbol} - Volume: ${token.volume_24h:.0f}, Liquidity: ${token.liquidity:.0f}, Age: {token.age_minutes}min")
                
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error parsing volume pool data: {e}")
                continue

        self._cache_data('volume', tokens)
        logger.info(f"Retrieved {len(tokens)} volume tokens from GeckoTerminal")
        return tokens

    async def get_memescope_tokens(self, limit: int = 50) -> List[TokenData]:
        """Get new pools as memescope equivalent"""
        # Check cache first  
        cached = self._get_cached_data('memescope')
        if cached:
            logger.info(f"Using cached memescope tokens: {len(cached)}")
            return cached

        endpoint = "networks/solana/new_pools" 
        params = {'include': 'base_token'}
        
        data = await self._make_request(endpoint, params)
        if not data:
            return []

        self.request_count['new_pools'] += 1
        
        tokens = []
        pools = data.get('data', [])
        
        # Filter for very new pools (memescope style)
        for pool in pools[:limit]:
            try:
                attributes = pool.get('attributes', {})
                base_token = pool.get('relationships', {}).get('base_token', {}).get('data', {})
                
                # Get token address
                token_address = base_token.get('id', '')
                if not token_address:
                    continue
                
                # Calculate age - focus on very new tokens
                pool_created_at = attributes.get('pool_created_at', '')
                age_minutes = self._calculate_token_age(pool_created_at)
                
                # Skip older tokens for memescope
                if age_minutes > 2880:  # Skip tokens older than 48 hours
                    continue
                
                name = attributes.get('name', '')
                base_token_price_usd = float(attributes.get('base_token_price_usd') or 0)
                
                price_change_data = attributes.get('price_change_percentage', {}) or {}
                price_change_24h = float(price_change_data.get('h24') or 0)
                
                volume_data = attributes.get('volume_usd', {}) or {}
                volume_24h = float(volume_data.get('h24') or 0)
                
                reserve_in_usd = float(attributes.get('reserve_in_usd') or 0)
                market_cap_usd = float(attributes.get('market_cap_usd') or 0)
                
                # Extract symbol
                symbol = ''
                if '/' in name:
                    symbol = name.split('/')[0].strip()
                else:
                    symbol = f"New{token_address[-6:].upper()}"
                
                token = TokenData(
                    address=token_address,
                    symbol=symbol,
                    name=name or symbol,
                    price=base_token_price_usd,
                    price_change_24h=price_change_24h,
                    volume_24h=volume_24h,
                    market_cap=market_cap_usd,
                    liquidity=reserve_in_usd,
                    age_minutes=age_minutes,
                    momentum_score=self._calculate_momentum_score(price_change_24h, volume_24h, reserve_in_usd),
                    source='memescope'
                )
                tokens.append(token)
                logger.debug(f"Parsed memescope token: {symbol} - Age: {age_minutes}min, Price: ${token.price:.8f}")
                
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error parsing memescope pool data: {e}")
                continue

        self._cache_data('memescope', tokens)
        logger.info(f"Retrieved {len(tokens)} memescope tokens from GeckoTerminal")
        return tokens

    async def get_all_tokens(self) -> List[TokenData]:
        """Get tokens from all sources and combine them"""
        all_tokens = []
        
        # Get from all sources
        trending = await self.get_trending_tokens()
        volume = await self.get_volume_tokens()
        memescope = await self.get_memescope_tokens()
        
        logger.info(f"GeckoTerminal results: trending={len(trending)}, volume={len(volume)}, memescope={len(memescope)}")
        
        all_tokens.extend(trending)
        all_tokens.extend(volume)
        all_tokens.extend(memescope)
        
        # Remove duplicates by address
        unique_tokens = {}
        for token in all_tokens:
            if token.address not in unique_tokens:
                unique_tokens[token.address] = token
            else:
                # Keep the one with higher momentum score
                if token.momentum_score > unique_tokens[token.address].momentum_score:
                    unique_tokens[token.address] = token
        
        result = list(unique_tokens.values())
        logger.info(f"Combined {len(result)} unique tokens from GeckoTerminal (FREE API)")
        return result

    def _calculate_token_age(self, created_at: str) -> int:
        """Calculate token age in minutes from ISO timestamp"""
        if not created_at:
            return 60  # Default to 1 hour for unknown age
        try:
            # Handle different timestamp formats
            if created_at.endswith('Z'):
                created_at = created_at[:-1] + '+00:00'
            
            created = datetime.fromisoformat(created_at)
            age = datetime.now() - created.replace(tzinfo=None)
            return int(age.total_seconds() / 60)
        except Exception as e:
            logger.debug(f"Error parsing timestamp {created_at}: {e}")
            return 60

    def _calculate_momentum_score(self, price_change_24h: float, volume_24h: float, liquidity: float) -> float:
        """Calculate momentum score from price change, volume, and liquidity"""
        try:
            # Normalize and weight the factors
            momentum = (
                abs(price_change_24h / 100) * 0.4 +  # Price change weight (absolute)
                (min(volume_24h / 100000, 1)) * 0.3 +  # Volume weight (capped at 100k)
                (min(liquidity / 50000, 1)) * 0.3   # Liquidity weight (capped at 50k)
            )
            
            return max(0, min(momentum * 10, 10))  # Scale to 0-10
        except:
            return 0

    def _cache_data(self, key: str, data: List[TokenData]):
        """Cache data with timestamp"""
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }

    def _get_cached_data(self, key: str) -> Optional[List[TokenData]]:
        """Get cached data if not expired"""
        if key not in self.cache:
            return None
        
        cached = self.cache[key]
        if time.time() - cached['timestamp'] > self.cache_duration:
            del self.cache[key]
            return None
        
        return cached['data']

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current API usage statistics"""
        now = time.time()
        calls_this_minute = len([
            call for call in self.calls_this_minute 
            if now - call < 60
        ])
        
        return {
            'api_provider': 'GeckoTerminal',
            'is_free': True,
            'calls_this_minute': calls_this_minute,
            'max_calls_per_minute': self.max_calls_per_minute,
            'daily_capacity': self.max_calls_per_minute * 60 * 24,
            'monthly_capacity': self.max_calls_per_minute * 60 * 24 * 30,
            'request_breakdown': self.request_count.copy(),
            'next_request_allowed': max(0, self.last_request_time + self.min_interval - now)
        }

    async def test_connection(self) -> bool:
        """Test API connectivity"""
        try:
            if not self.session:
                await self.start_session()
            
            # Simple test request
            data = await self._make_request("networks/solana/trending_pools", {"include": "base_token"})
            if data and data.get('data'):
                logger.info("GeckoTerminal API connection successful - FREE API working!")
                return True
            else:
                logger.error("GeckoTerminal API connection failed")
                return False
        except Exception as e:
            logger.error(f"GeckoTerminal API connection test failed: {e}")
            return False