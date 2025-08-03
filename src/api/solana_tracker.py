import asyncio
import aiohttp
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import os

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

class SolanaTrackerClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('SOLANA_TRACKER_KEY')
        self.base_url = "https://data.solanatracker.io"
        self.session: Optional[aiohttp.ClientSession] = None
        
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
        
        # Scheduling intervals (in seconds)
        self.intervals = {
            'trending': 780,   # 13 minutes
            'volume': 900,     # 15 minutes  
            'memescope': 1080  # 18 minutes
        }
        
        self.last_requests = {
            'trending': 0,
            'volume': 0,
            'memescope': 0
        }
        
        # Cache for reducing API calls
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        if self.api_key:
            # Show partial key for debugging without exposing full key
            masked_key = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
            logger.info(f"SolanaTracker client initialized with API key ({masked_key}), daily limit: {self.daily_limit}")
        else:
            logger.warning("SolanaTracker client initialized WITHOUT API key - some endpoints may not work")
            logger.info("Get your free API key from: https://solanatracker.io/")

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
            if self.api_key:
                headers['x-api-key'] = self.api_key
            
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
    
    async def _test_connection(self) -> bool:
        """Test API connectivity"""
        try:
            await self.start_session()
            # Make a simple request to test connectivity
            headers = {"X-API-KEY": self.api_key} if self.api_key else {}
            async with self.session.get(f"{self.base_url}/tokens/trending", headers=headers, timeout=10) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def _reset_daily_counter_if_needed(self):
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
            # Ensure we don't get stuck in a bad state
            try:
                self.requests_today = 0
                self.last_reset = datetime.now().date()
                self.request_count = {k: 0 for k in self.request_count}
                logger.warning("Emergency counter reset performed")
            except Exception as emergency_error:
                logger.critical(f"Emergency counter reset failed: {emergency_error}")

    def _can_make_request(self) -> bool:
        self._reset_daily_counter_if_needed()
        
        # Check daily limit
        if self.requests_today >= self.daily_limit:
            logger.warning(f"Daily API limit reached: {self.requests_today}/{self.daily_limit}")
            return False
        
        # Check rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_interval:
            return False
        
        return True

    def _should_make_scheduled_request(self, endpoint: str) -> bool:
        if not self._can_make_request():
            return False
        
        last_request = self.last_requests.get(endpoint, 0)
        interval = self.intervals.get(endpoint, 900)
        
        return (time.time() - last_request) >= interval

    async def _make_request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        if not self._can_make_request():
            await asyncio.sleep(self.min_interval)
            if not self._can_make_request():
                return None

        if not self.session:
            await self.start_session()

        # Rate limiting delay
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)

        url = f"{self.base_url}/{endpoint}"
        
        try:
            logger.debug(f"Making request to: {url}")
            async with self.session.get(url, params=params) as response:
                self.last_request_time = time.time()
                self.requests_today += 1
                self.request_count['total'] += 1
                
                if response.status == 200:
                    data = await response.json()
                    items_count = len(data.get('data', []) if isinstance(data, dict) else data if isinstance(data, list) else [])
                    logger.info(f"API Response from {endpoint}: {items_count} items (Daily: {self.requests_today}/{self.daily_limit})")
                    
                    return data
                elif response.status == 401:
                    logger.error("API request failed: 401 Unauthorized")
                    logger.error("This usually means:")
                    logger.error("1. API key is missing or invalid")
                    logger.error("2. API key is not properly set in .env file")
                    logger.error("3. You may need to subscribe to a plan at https://solanatracker.io/")
                    return None
                elif response.status == 429:
                    logger.warning("Rate limit hit, backing off...")
                    await asyncio.sleep(60)
                    return None
                else:
                    logger.error(f"API request failed: {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Request timeout for {url}")
            return None
        except Exception as e:
            logger.error(f"Request error for {url}: {str(e)}")
            return None

    async def get_trending_tokens(self, limit: int = 50) -> List[TokenData]:
        # Check cache first
        cached = self._get_cached_data('trending')
        if cached:
            logger.info(f"Using cached trending tokens: {len(cached)}")
            return cached
            
        # If no cache and rate limited, still try once for initial discovery
        if not self._should_make_scheduled_request('trending'):
            if self.requests_today < 5:  # Allow first few requests
                logger.info("Rate limited but allowing initial request for token discovery")
            else:
                logger.info("No cached trending tokens, skipping request due to rate limiting")
                return []

        endpoint = "tokens/trending"
        params = {'limit': limit}
        
        data = await self._make_request(endpoint, params)
        if not data:
            return []

        self.last_requests['trending'] = time.time()
        self.request_count['trending'] += 1
        
        tokens = []
        # Handle both dict and list response formats
        items = data.get('data', []) if isinstance(data, dict) else data if isinstance(data, list) else []
        
        for item in items:
            try:
                # Extract nested token info
                token_info = item.get('token', {})
                pools = item.get('pools', [])
                events = item.get('events', {})
                
                # Get basic token data
                address = token_info.get('mint') or token_info.get('address')
                symbol = token_info.get('symbol')
                name = token_info.get('name')
                
                if not address or not symbol:
                    logger.debug(f"Skipping trending token - missing address/symbol: address={address}, symbol={symbol}")
                    continue
                
                # Extract price and liquidity from first pool
                price_usd = 0.0
                liquidity_usd = 0.0
                volume_24h = 0.0
                market_cap_usd = 0.0
                
                if pools:
                    pool = pools[0]  # Use first pool
                    price_data = pool.get('price', {})
                    liquidity_data = pool.get('liquidity', {})
                    txns_data = pool.get('txns', {})
                    market_cap_data = pool.get('marketCap', {})
                    
                    price_usd = price_data.get('usd', 0.0) if isinstance(price_data, dict) else 0.0
                    liquidity_usd = liquidity_data.get('usd', 0.0) if isinstance(liquidity_data, dict) else 0.0
                    volume_24h = txns_data.get('volume24h', 0.0) if isinstance(txns_data, dict) else 0.0
                    market_cap_usd = market_cap_data.get('usd', 0.0) if isinstance(market_cap_data, dict) else 0.0
                
                # Extract 24h price change from events
                price_change_24h = 0.0
                if events and '24h' in events:
                    price_change_24h = events['24h'].get('priceChangePercentage', 0.0) or 0.0
                
                # Calculate token age from creation time
                created_time = token_info.get('creation', {}).get('created_time')
                age_minutes = 0
                if created_time:
                    try:
                        current_time = time.time()
                        age_seconds = current_time - created_time
                        age_minutes = int(age_seconds / 60)
                    except:
                        age_minutes = 0
                
                token = TokenData(
                    address=address,
                    symbol=symbol,
                    name=name or symbol,
                    price=float(price_usd) if price_usd else 0.0,
                    price_change_24h=float(price_change_24h) if price_change_24h else 0.0,
                    volume_24h=float(volume_24h) if volume_24h else 0.0,
                    market_cap=float(market_cap_usd) if market_cap_usd else 0.0,
                    liquidity=float(liquidity_usd) if liquidity_usd else 0.0,
                    age_minutes=age_minutes,
                    momentum_score=self._calculate_momentum_score_from_events(events),
                    source='trending'
                )
                tokens.append(token)
                logger.debug(f"Parsed trending token: {symbol} - Liquidity: ${token.liquidity:.0f}, Price change: {token.price_change_24h:.1f}%, Age: {token.age_minutes}min")
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing trending token data: {e} - Item keys: {list(item.keys()) if isinstance(item, dict) else 'not dict'}")
                continue

        self._cache_data('trending', tokens)
        logger.info(f"Retrieved {len(tokens)} trending tokens")
        return tokens

    async def get_volume_tokens(self, limit: int = 50) -> List[TokenData]:
        # Check cache first
        cached = self._get_cached_data('volume')
        if cached:
            logger.info(f"Using cached volume tokens: {len(cached)}")
            return cached
            
        # If no cache and rate limited, still try once for initial discovery
        if not self._should_make_scheduled_request('volume'):
            if self.requests_today < 5:  # Allow first few requests
                logger.info("Rate limited but allowing initial request for token discovery")
            else:
                logger.info("No cached volume tokens, skipping request due to rate limiting")
                return []

        endpoint = "tokens/volume"
        params = {'limit': limit, 'timeframe': '1h'}
        
        data = await self._make_request(endpoint, params)
        if not data:
            return []

        self.last_requests['volume'] = time.time()
        self.request_count['volume'] += 1
        
        tokens = []
        # Handle both dict and list response formats
        items = data.get('data', []) if isinstance(data, dict) else data if isinstance(data, list) else []
        
        for item in items:
            try:
                # Extract nested token info (same structure as trending)
                token_info = item.get('token', {})
                pools = item.get('pools', [])
                events = item.get('events', {})
                
                # Get basic token data
                address = token_info.get('mint') or token_info.get('address')
                symbol = token_info.get('symbol')
                name = token_info.get('name')
                
                if not address or not symbol:
                    logger.debug(f"Skipping volume token - missing address/symbol: address={address}, symbol={symbol}")
                    continue
                
                # Extract price and liquidity from first pool
                price_usd = 0.0
                liquidity_usd = 0.0
                volume_24h = 0.0
                market_cap_usd = 0.0
                
                if pools:
                    pool = pools[0]  # Use first pool
                    price_data = pool.get('price', {})
                    liquidity_data = pool.get('liquidity', {})
                    txns_data = pool.get('txns', {})
                    market_cap_data = pool.get('marketCap', {})
                    
                    price_usd = price_data.get('usd', 0.0) if isinstance(price_data, dict) else 0.0
                    liquidity_usd = liquidity_data.get('usd', 0.0) if isinstance(liquidity_data, dict) else 0.0
                    volume_24h = txns_data.get('volume24h', 0.0) if isinstance(txns_data, dict) else 0.0
                    market_cap_usd = market_cap_data.get('usd', 0.0) if isinstance(market_cap_data, dict) else 0.0
                
                # Extract 24h price change from events
                price_change_24h = 0.0
                if events and '24h' in events:
                    price_change_24h = events['24h'].get('priceChangePercentage', 0.0) or 0.0
                
                # Calculate token age from creation time
                created_time = token_info.get('creation', {}).get('created_time')
                age_minutes = 0
                if created_time:
                    try:
                        current_time = time.time()
                        age_seconds = current_time - created_time
                        age_minutes = int(age_seconds / 60)
                    except:
                        age_minutes = 0
                
                token = TokenData(
                    address=address,
                    symbol=symbol,
                    name=name or symbol,
                    price=float(price_usd) if price_usd else 0.0,
                    price_change_24h=float(price_change_24h) if price_change_24h else 0.0,
                    volume_24h=float(volume_24h) if volume_24h else 0.0,
                    market_cap=float(market_cap_usd) if market_cap_usd else 0.0,
                    liquidity=float(liquidity_usd) if liquidity_usd else 0.0,
                    age_minutes=age_minutes,
                    momentum_score=self._calculate_momentum_score_from_events(events),
                    source='volume'
                )
                tokens.append(token)
                logger.debug(f"Parsed volume token: {symbol} - Liquidity: ${token.liquidity:.0f}, Volume: ${token.volume_24h:.0f}, Age: {token.age_minutes}min")
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing volume token data: {e} - Item keys: {list(item.keys()) if isinstance(item, dict) else 'not dict'}")
                continue

        self._cache_data('volume', tokens)
        logger.info(f"Retrieved {len(tokens)} volume tokens")
        return tokens

    async def get_memescope_tokens(self, limit: int = 50) -> List[TokenData]:
        if not self._should_make_scheduled_request('memescope'):
            cached = self._get_cached_data('memescope')
            if cached:
                return cached
            return []

        endpoint = "tokens/memescope"
        params = {'limit': limit}
        
        data = await self._make_request(endpoint, params)
        if not data:
            return []

        self.last_requests['memescope'] = time.time()
        self.request_count['memescope'] += 1
        
        tokens = []
        # Handle both dict and list response formats
        items = data.get('data', []) if isinstance(data, dict) else data if isinstance(data, list) else []
        
        for item in items:
            try:
                token = TokenData(
                    address=item.get('address', ''),
                    symbol=item.get('symbol', ''),
                    name=item.get('name', ''),
                    price=float(item.get('price', 0)),
                    price_change_24h=float(item.get('priceChange24h', 0)),
                    volume_24h=float(item.get('volume24h', 0)),
                    market_cap=float(item.get('marketCap', 0)),
                    liquidity=float(item.get('liquidity', 0)),
                    age_minutes=self._calculate_token_age(item.get('createdAt')),
                    momentum_score=self._calculate_momentum_score(item),
                    source='memescope'
                )
                tokens.append(token)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing memescope token data: {e}")
                continue

        self._cache_data('memescope', tokens)
        logger.info(f"Retrieved {len(tokens)} memescope tokens")
        return tokens

    async def get_all_tokens(self) -> List[TokenData]:
        """Get tokens from all sources and combine them"""
        all_tokens = []
        
        # Get from all sources
        trending = await self.get_trending_tokens()
        volume = await self.get_volume_tokens()
        memescope = await self.get_memescope_tokens()
        
        logger.info(f"Source results: trending={len(trending)}, volume={len(volume)}, memescope={len(memescope)}")
        
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
        logger.info(f"Combined {len(result)} unique tokens from all sources")
        return result

    def _calculate_token_age(self, created_at: str) -> int:
        if not created_at:
            return 0
        try:
            created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            age = datetime.now() - created.replace(tzinfo=None)
            return int(age.total_seconds() / 60)
        except:
            return 0

    def _calculate_momentum_score(self, item: dict) -> float:
        try:
            price_change = float(item.get('priceChange24h', 0))
            volume = float(item.get('volume24h', 0))
            liquidity = float(item.get('liquidity', 0))
            
            # Normalize and weight the factors
            momentum = (
                (price_change / 100) * 0.4 +  # Price change weight
                (min(volume / 100000, 1)) * 0.3 +  # Volume weight (capped)
                (min(liquidity / 50000, 1)) * 0.3   # Liquidity weight (capped)
            )
            
            return max(0, min(momentum, 10))  # Clamp between 0-10
        except:
            return 0

    def _calculate_momentum_score_from_events(self, events: dict) -> float:
        """Calculate momentum score from events data structure"""
        score = 0.0
        
        if not events:
            return score
            
        # Get price changes at different timeframes
        timeframes = ['1h', '6h', '12h', '24h']
        changes = []
        
        for tf in timeframes:
            if tf in events:
                change = events[tf].get('priceChangePercentage')
                if change is not None:
                    changes.append(abs(float(change)))
        
        if changes:
            # Average momentum across timeframes
            avg_change = sum(changes) / len(changes)
            score += min(avg_change / 10, 8)  # Up to 8 points for price momentum
            
            # Bonus for consistent positive momentum
            positive_changes = [c for c in changes if c > 0]
            if len(positive_changes) >= len(changes) * 0.75:  # 75% positive
                score += 2
        
        return min(score, 10)  # Cap at 10

    def _cache_data(self, key: str, data: List[TokenData]):
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }

    def _get_cached_data(self, key: str) -> Optional[List[TokenData]]:
        if key not in self.cache:
            return None
        
        cached = self.cache[key]
        if time.time() - cached['timestamp'] > self.cache_duration:
            del self.cache[key]
            return None
        
        return cached['data']

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current API usage statistics"""
        try:
            self._reset_daily_counter_if_needed()
            
            return {
                'requests_today': self.requests_today,
                'daily_limit': self.daily_limit,
                'remaining_requests': max(0, self.daily_limit - self.requests_today),
                'usage_percentage': (self.requests_today / self.daily_limit) * 100 if self.daily_limit > 0 else 0,
                'request_breakdown': self.request_count.copy(),
                'last_reset': self.last_reset.isoformat() if hasattr(self.last_reset, 'isoformat') else str(self.last_reset),
                'next_scheduled': {
                    endpoint: max(0, self.last_requests.get(endpoint, 0) + self.intervals[endpoint] - time.time())
                    for endpoint in self.intervals
                }
            }
        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            # Return safe defaults
            return {
                'requests_today': 0,
                'daily_limit': self.daily_limit,
                'remaining_requests': self.daily_limit,
                'usage_percentage': 0,
                'request_breakdown': {k: 0 for k in ['trending', 'volume', 'memescope', 'total']},
                'last_reset': datetime.now().date().isoformat(),
                'next_scheduled': {endpoint: 0 for endpoint in self.intervals}
            }

    async def test_connection(self) -> bool:
        """Test API connectivity"""
        try:
            if not self.api_key:
                logger.warning("No API key provided - Solana Tracker API will not work")
                logger.info("To fix: Add SOLANA_TRACKER_KEY to your .env file")
                logger.info("Get free API key from: https://solanatracker.io/")
                return False
                
            if not self.session:
                await self.start_session()
            
            # Simple test request
            data = await self._make_request("tokens/trending", {"limit": 1})
            if data:
                logger.info("Solana Tracker API connection successful")
                return True
            else:
                logger.error("Solana Tracker API connection failed - check your API key")
                return False
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False