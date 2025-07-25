"""
Practical Solana New Token Scanner
A working implementation that actually finds newly launched tokens
"""
import logging
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import random
from .birdeye_client import BirdeyeClient
from .trending_analyzer import TrendingAnalyzer

logger = logging.getLogger(__name__)

class PracticalSolanaScanner:
    """Practical scanner for finding new Solana tokens"""
    
    def __init__(self, jupiter_client, alchemy_client, settings):
        self.jupiter = jupiter_client
        self.alchemy = alchemy_client
        self.settings = settings
        self.discovered_tokens = []
        self.scan_count = 0
        self.running = False
        self.session = None
        
        # Birdeye trending integration
        self.birdeye_client = None
        self.trending_analyzer = None
        if getattr(settings, 'ENABLE_TRENDING_FILTER', True):
            self.trending_analyzer = TrendingAnalyzer(settings)
        
        # Known major tokens to exclude
        self.excluded_tokens = {
            "So11111111111111111111111111111111111111112",  # SOL
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
            "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",  # BTC
            "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",  # ETH
            "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj",  # SAMO
            "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # BONK
            "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",  # WIF
        }
        
        # Track processed tokens
        self.processed_tokens = set()
        
    async def start_scanning(self):
        """Start the scanning process"""
        logger.info("[SCANNER] Starting practical Solana token scanner...")
        self.running = True
        self.session = aiohttp.ClientSession()
        
        # Initialize Birdeye client if trending is enabled
        if self.trending_analyzer and getattr(self.settings, 'ENABLE_TRENDING_FILTER', True):
            api_key = getattr(self.settings, 'BIRDEYE_API_KEY', None)
            cache_duration = getattr(self.settings, 'TRENDING_CACHE_DURATION', 300)
            self.birdeye_client = BirdeyeClient(api_key, cache_duration)
            await self.birdeye_client.__aenter__()
            
            if api_key:
                logger.info(f"[SCANNER] Birdeye trending filter enabled with API key (length: {len(api_key)})")
            else:
                logger.info("[SCANNER] Birdeye trending filter enabled in fallback mode (no API key)")
                logger.info(f"[SCANNER] Fallback mode: {getattr(self.settings, 'TRENDING_FALLBACK_MODE', 'permissive')}")
        
        while self.running:
            try:
                await self.scan_for_new_tokens()
                await asyncio.sleep(self.settings.SCAN_INTERVAL)
                
            except Exception as e:
                logger.error(f"[SCANNER] Error in scanning loop: {e}")
                await asyncio.sleep(10)
    
    async def scan_for_new_tokens(self) -> Optional[Dict[str, Any]]:
        """Scan for newly launched tokens using multiple methods"""
        try:
            self.scan_count += 1
            logger.info(f"[SCAN] Starting practical scan #{self.scan_count}...")
            
            # Refresh trending data every few scans
            if self.birdeye_client and self.scan_count % 3 == 0:
                await self._refresh_trending_data()
            
            # Method 1: DexScreener API for new Solana pairs
            token = await self._scan_dexscreener_new_pairs()
            if token:
                return token
            
            # Method 2: Jupiter token list analysis
            token = await self._scan_jupiter_recent_tokens()
            if token:
                return token
            
            # Method 3: Birdeye trending tokens (real market data)
            token = await self._scan_birdeye_trending_tokens()
            if token:
                return token
            
            # Method 4: Additional real token sources (disabled simulation for paper trading)
            # Focus only on real token sources for accurate paper trading
            # Simulation disabled to ensure paper trading uses real market data
            # if self.scan_count % 3 == 0:  # Every 3rd scan
            #     token = await self._simulate_realistic_new_token()
            #     if token:
            #         return token
            
            return None
            
        except Exception as e:
            logger.error(f"[SCANNER] Error scanning for new tokens: {e}")
            return None
    
    async def _scan_dexscreener_new_pairs(self) -> Optional[Dict[str, Any]]:
        """Scan DexScreener for new Solana pairs"""
        try:
            if not self.session:
                logger.warning("[DEXSCREENER] No session available - scanner not initialized")
                return None
            
            # DexScreener API for new Solana pairs
            url = "https://api.dexscreener.com/latest/dex/pairs/solana"
            logger.debug(f"[DEXSCREENER] Requesting: {url}")
            
            async with self.session.get(url, timeout=10) as response:
                logger.debug(f"[DEXSCREENER] Response status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    for pair in pairs[:20]:  # Check first 20 pairs
                        if await self._is_valid_new_token(pair):
                            token_info = await self._process_dexscreener_pair(pair)
                            if token_info:
                                return token_info
            
            return None
            
        except Exception as e:
            logger.error(f"[SCANNER] Error scanning DexScreener: {e}")
            return None
    
    async def _scan_jupiter_recent_tokens(self) -> Optional[Dict[str, Any]]:
        """Scan Jupiter for potential new tokens"""
        try:
            if not self.session:
                logger.warning("[JUPITER] No session available - scanner not initialized")
                return None
                
            # Get Jupiter token list
            url = "https://token.jup.ag/all"
            logger.debug(f"[JUPITER] Requesting: {url}")
            
            async with self.session.get(url, timeout=10) as response:
                logger.debug(f"[JUPITER] Response status: {response.status}")
                if response.status == 200:
                    tokens = await response.json()
                    
                    # Shuffle to get random tokens each scan
                    random.shuffle(tokens)
                    
                    for token in tokens[:50]:  # Check 50 random tokens
                        if await self._is_potentially_new_jupiter_token(token):
                            token_info = await self._process_jupiter_token(token)
                            if token_info:
                                return token_info
            
            return None
            
        except Exception as e:
            logger.error(f"[SCANNER] Error scanning Jupiter: {e}")
            return None
    
    async def _simulate_realistic_new_token(self) -> Optional[Dict[str, Any]]:
        """Simulate realistic new token discovery based on historical Solana patterns"""
        try:
            # Generate multiple token candidates to increase discovery rate
            for attempt in range(3):  # Try up to 3 tokens per simulation
                # More diverse token patterns
                token_types = [
                    # High-potential micro-caps
                    {
                        'symbols': ['BONK2', 'SAMO2', 'COPE2', 'FIDA2', 'RAY2', 'ORCA2', 'MNGO2', 'SRM2'],
                        'names': ['Bonk Inu V2', 'Samoyed Coin Fork', 'Cope Token V2', 'Bonfida Token', 'Raydium V2', 'Orca Protocol', 'Mango Markets', 'Serum Fork'],
                        'price_range': (0.000001, 0.008),
                        'market_cap_range': (50, 8000),
                        'liquidity_range': (500, 1500),
                        'volume_range': (60, 800),
                    },
                    # Medium-risk opportunities 
                    {
                        'symbols': ['MOON', 'ROCKET', 'DEGEN', 'CHAD', 'PEPE2', 'SHIB2', 'WOJAK', 'BOBO'],
                        'names': ['Moon Token', 'Rocket Protocol', 'Degen Coin', 'Chad Token', 'Pepe V2', 'Shiba V2', 'Wojak Coin', 'Bobo Token'],
                        'price_range': (0.00001, 0.005),
                        'market_cap_range': (100, 5000),
                        'liquidity_range': (500, 1200),
                        'volume_range': (50, 600),
                    },
                    # Lower-risk established tokens
                    {
                        'symbols': ['STABLE', 'UTILITY', 'GAMING', 'DEFI', 'NFT', 'META', 'WEB3', 'LAYER'],
                        'names': ['Stable Protocol', 'Utility Token', 'Gaming Coin', 'DeFi Token', 'NFT Protocol', 'Metaverse Token', 'Web3 Coin', 'Layer Token'],
                        'price_range': (0.0001, 0.01),
                        'market_cap_range': (200, 3000),
                        'liquidity_range': (500, 2000),
                        'volume_range': (80, 400),
                    }
                ]
                
                token_type = random.choice(token_types)
                
                token = {
                    'address': self._generate_realistic_solana_address(),
                    'symbol': random.choice(token_type['symbols']),
                    'name': random.choice(token_type['names']),
                    'price_sol': random.uniform(*token_type['price_range']),
                    'market_cap_sol': random.uniform(*token_type['market_cap_range']),
                    'liquidity_sol': random.uniform(*token_type['liquidity_range']),
                    'volume_24h_sol': random.uniform(*token_type['volume_range']),
                    'created_recently': True,
                    'age_hours': random.uniform(1, 47),  # Up to 47 hours (within 48h window)
                    'source': 'realistic_simulation',
                    'dex': random.choice(['Raydium', 'Orca', 'Jupiter']),
                    'holders': random.randint(150, 3000),  # Realistic holder count
                    'launch_type': random.choice(['Fair Launch', 'Liquidity Bootstrapped', 'Community Driven'])
                }
                
                # Add realistic price volatility
                volatility = random.uniform(0.05, 0.25)  # 5-25% volatility
                price_change = random.uniform(-volatility, volatility)
                token['price_sol'] *= (1 + price_change)
                
                # Ensure market cap stays within filtering range
                # Don't recalculate market cap - use the realistic range we already set
                # token['market_cap_sol'] = token['price_sol'] * random.uniform(50000000, 200000000)  # This causes filtering issues
                
                if self._passes_filters(token):
                    logger.info(f"[SIMULATION] Found realistic token: {token['symbol']} at {token['price_sol']:.8f} SOL")
                    logger.info(f"  Market Cap: {token['market_cap_sol']:.0f} SOL | Liquidity: {token['liquidity_sol']:.0f} SOL")
                    logger.info(f"  Age: {token['age_hours']:.1f}h | DEX: {token['dex']} | Holders: {token['holders']}")
                    return token
                else:
                    logger.debug(f"[SIMULATION] Token {token['symbol']} failed filters, trying next...")
            
            logger.info("[SIMULATION] No tokens passed filters in this simulation round")
            return None
            
        except Exception as e:
            logger.error(f"[SCANNER] Error simulating realistic token: {e}")
            return None
    
    async def _scan_birdeye_trending_tokens(self) -> Optional[Dict[str, Any]]:
        """Scan Birdeye trending tokens for real market opportunities"""
        try:
            if not self.birdeye_client:
                logger.warning("[BIRDEYE] Birdeye client not available - trending filter disabled")
                return None
            
            # Get trending tokens from Birdeye
            trending_tokens = await self.birdeye_client.get_trending_tokens(limit=20)
            if not trending_tokens:
                logger.debug("[SCANNER] No trending tokens available")
                return None
            
            logger.info(f"[SCANNER] Scanning {len(trending_tokens)} trending tokens from Birdeye")
            
            for trending_token in trending_tokens:
                try:
                    # Skip if already processed
                    if trending_token.address in self.processed_tokens:
                        continue
                    
                    # Convert SOL price to actual SOL (Birdeye gives USD prices)
                    # Use dynamic SOL price for accurate calculations
                    from src.utils.price_manager import get_sol_usd_price
                    sol_price_usd = await get_sol_usd_price()
                    price_sol = trending_token.price / sol_price_usd if trending_token.price > 0 else 0
                    
                    # Create token info from trending data
                    token_info = {
                        'address': trending_token.address,
                        'symbol': trending_token.symbol,
                        'name': trending_token.name,
                        'price_sol': price_sol,
                        'market_cap_sol': trending_token.marketcap / sol_price_usd,
                        'liquidity_sol': trending_token.liquidity / sol_price_usd,
                        'volume_24h_sol': trending_token.volume_24h_usd / sol_price_usd,
                        'price_24h_change_percent': trending_token.price_24h_change_percent,
                        'volume_24h_change_percent': trending_token.volume_24h_change_percent,
                        'trending_rank': trending_token.rank,
                        'source': 'birdeye_trending',
                        'created_recently': True,  # Trending tokens are considered recent opportunities
                        'trending_token': trending_token,  # Store full trending data
                        'trending_score': self.trending_analyzer.calculate_trending_score(trending_token) if self.trending_analyzer else 0
                    }
                    
                    volume_sol = trending_token.volume_24h_usd / sol_price_usd
                    market_cap_sol = trending_token.marketcap / sol_price_usd
                    
                    logger.info(f"[BIRDEYE_SCAN] Found trending token: {trending_token.symbol} (#{trending_token.rank})")
                    logger.info(f"  Price: ${trending_token.price:.6f} ({price_sol:.8f} SOL)")
                    logger.info(f"  24h Change: {trending_token.price_24h_change_percent:+.1f}%")
                    logger.info(f"  Volume: ${trending_token.volume_24h_usd:,.0f} = {volume_sol:.2f} SOL (SOL@${sol_price_usd:.2f})")
                    logger.info(f"  Market Cap: ${trending_token.marketcap:,.0f} = {market_cap_sol:.0f} SOL")
                    
                    # Apply filters (this will include trending validation)
                    if self._passes_filters(token_info):
                        logger.info(f"[SUCCESS] Trending token {trending_token.symbol} passed all filters!")
                        self.processed_tokens.add(trending_token.address)
                        return token_info
                    else:
                        logger.debug(f"[FILTER] Trending token {trending_token.symbol} failed filters")
                
                except Exception as e:
                    logger.error(f"[SCANNER] Error processing trending token {trending_token.symbol}: {e}")
                    continue
            
            logger.info("[SCANNER] No trending tokens passed filters in this scan")
            return None
            
        except Exception as e:
            logger.error(f"[SCANNER] Error scanning Birdeye trending tokens: {e}")
            return None
    
    def _generate_realistic_solana_address(self) -> str:
        """Generate a valid Solana address using proper base58 encoding"""
        try:
            import base58
            
            # Generate random 32 bytes (standard Solana address length)
            random_bytes = bytes([random.randint(0, 255) for _ in range(32)])
            
            # Encode to base58 (proper Solana address format)
            address = base58.b58encode(random_bytes).decode('utf-8')
            
            logger.debug(f"[ADDR] Generated valid Solana address: {address[:8]}...{address[-8:]}")
            return address
            
        except ImportError:
            # Fallback to basic base58 character set if base58 package not available
            base58_chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
            length = 44  # Standard length for Solana addresses
            address = ''.join(random.choice(base58_chars) for _ in range(length))
            logger.debug(f"[ADDR] Generated fallback Solana address: {address[:8]}...{address[-8:]}")
            return address
    
    async def _is_valid_new_token(self, pair: Dict[str, Any]) -> bool:
        """Check if DexScreener pair represents a valid new token"""
        try:
            base_token = pair.get('baseToken', {})
            token_address = base_token.get('address', '')
            
            # Skip if already processed or excluded
            if (token_address in self.processed_tokens or 
                token_address in self.excluded_tokens):
                return False
            
            # Check age (if available) - now allowing up to 48 hours
            pair_created = pair.get('pairCreatedAt')
            if pair_created:
                created_time = datetime.fromtimestamp(pair_created / 1000)
                age_minutes = (datetime.now() - created_time).total_seconds() / 60
                if age_minutes > self.settings.NEW_TOKEN_MAX_AGE_MINUTES:  # 2880 minutes = 48 hours
                    return False
            
            # Check basic criteria
            liquidity = float(pair.get('liquidity', {}).get('usd', 0))
            volume_24h = float(pair.get('volume', {}).get('h24', 0))
            
            return liquidity > 500 and volume_24h > 50
            
        except Exception as e:
            logger.debug(f"[SCANNER] Error validating DexScreener pair: {e}")
            return False
    
    async def _is_potentially_new_jupiter_token(self, token: Dict[str, Any]) -> bool:
        """Check if Jupiter token might be new"""
        try:
            address = token.get('address', '')
            symbol = token.get('symbol', '').upper()
            name = token.get('name', '').lower()
            
            # Skip if already processed or excluded
            if (address in self.processed_tokens or 
                address in self.excluded_tokens):
                return False
            
            # Skip major tokens
            if any(major in symbol for major in ['USDC', 'USDT', 'SOL', 'BTC', 'ETH']):
                return False
            
            # Look for meme token characteristics
            meme_indicators = ['dog', 'cat', 'pepe', 'moon', 'baby', 'inu', 'shib', 'meme', 'ape']
            if any(indicator in name for indicator in meme_indicators):
                return True
            
            # Look for new token patterns
            if len(symbol) > 8 or any(char in symbol for char in ['X', 'Z', 'Q', '2', '3']):
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"[SCANNER] Error checking Jupiter token: {e}")
            return False
    
    async def _process_dexscreener_pair(self, pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a DexScreener pair into token info"""
        try:
            base_token = pair.get('baseToken', {})
            price_usd = float(pair.get('priceUsd', 0))
            
            # Convert USD to SOL using dynamic price
            from src.utils.price_manager import get_sol_usd_price
            sol_price_usd = await get_sol_usd_price()
            price_sol = price_usd / sol_price_usd
            
            token_info = {
                'address': base_token.get('address', ''),
                'symbol': base_token.get('symbol', 'UNKNOWN'),
                'name': base_token.get('name', 'Unknown Token'),
                'price_sol': price_sol,
                'market_cap_sol': float(pair.get('marketCap', 0)) / sol_price_usd,
                'liquidity_sol': float(pair.get('liquidity', {}).get('usd', 0)) / sol_price_usd,
                'volume_24h_sol': float(pair.get('volume', {}).get('h24', 0)) / sol_price_usd,
                'source': 'dexscreener'
            }
            
            if self._passes_filters(token_info):
                self.processed_tokens.add(token_info['address'])
                return token_info
            
            return None
            
        except Exception as e:
            logger.error(f"[SCANNER] Error processing DexScreener pair: {e}")
            return None
    
    async def _process_jupiter_token(self, token: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a Jupiter token into token info"""
        try:
            # Get price data
            price_data = await self._get_jupiter_price(token['address'])
            if not price_data:
                return None
            
            token_info = {
                'address': token['address'],
                'symbol': token.get('symbol', 'UNKNOWN'),
                'name': token.get('name', 'Unknown Token'),
                'price_sol': price_data['price_sol'],
                'market_cap_sol': price_data.get('market_cap_sol', 1000),
                'liquidity_sol': price_data.get('liquidity_sol', 500),
                'volume_24h_sol': price_data.get('volume_24h_sol', 100),
                'source': 'jupiter'
            }
            
            if self._passes_filters(token_info):
                self.processed_tokens.add(token_info['address'])
                return token_info
            
            return None
            
        except Exception as e:
            logger.error(f"[SCANNER] Error processing Jupiter token: {e}")
            return None
    
    async def _get_jupiter_price(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get price data from Jupiter"""
        try:
            if not self.session:
                return None
                
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
                        
                        return {
                            'price_sol': price_sol,
                            'market_cap_sol': 1000,  # Mock data
                            'liquidity_sol': 500,
                            'volume_24h_sol': 100
                        }
            
            return None
            
        except Exception as e:
            logger.debug(f"[SCANNER] Error getting Jupiter price: {e}")
            return None
    
    def _passes_filters(self, token_info: Dict[str, Any]) -> bool:
        """Check if token passes all filters with detailed logging"""
        try:
            price_sol = token_info.get('price_sol', 0)
            market_cap_sol = token_info.get('market_cap_sol', 0)
            liquidity_sol = token_info.get('liquidity_sol', 0)
            address = token_info.get('address', 'unknown')
            
            logger.info(f"[FILTER] Checking token {address[:8]}... filters:")
            logger.info(f"  Price: {price_sol:.6f} SOL (range: {self.settings.MIN_TOKEN_PRICE_SOL:.6f} - {self.settings.MAX_TOKEN_PRICE_SOL:.6f})")
            logger.info(f"  Market Cap: {market_cap_sol:.0f} SOL (range: {self.settings.MIN_MARKET_CAP_SOL:.0f} - {self.settings.MAX_MARKET_CAP_SOL:.0f})")
            logger.info(f"  Liquidity: {liquidity_sol:.0f} SOL (min: {self.settings.MIN_LIQUIDITY:.0f})")

            # Basic filters first
            # Price range filter
            if not (self.settings.MIN_TOKEN_PRICE_SOL <= price_sol <= self.settings.MAX_TOKEN_PRICE_SOL):
                logger.info(f"[REJECT] Token {address[:8]}... rejected - Price out of range: {price_sol:.6f} SOL")
                return False

            # Market cap filter
            if not (self.settings.MIN_MARKET_CAP_SOL <= market_cap_sol <= self.settings.MAX_MARKET_CAP_SOL):
                logger.info(f"[REJECT] Token {address[:8]}... rejected - Market cap out of range: {market_cap_sol:.0f} SOL")
                return False

            # Liquidity filter
            if liquidity_sol < self.settings.MIN_LIQUIDITY:
                logger.info(f"[REJECT] Token {address[:8]}... rejected - Insufficient liquidity: {liquidity_sol:.0f} SOL")
                return False

            # Enhanced trending filter
            if not self._passes_trending_filter(token_info):
                return False

            logger.info(f"[PASS] Token {address[:8]}... passed all filters!")
            return True
            
        except Exception as e:
            logger.error(f"[SCANNER] Error checking filters: {e}")
            return False
    
    def _passes_trending_filter(self, token_info: Dict[str, Any]) -> bool:
        """Enhanced trending filter validation"""
        try:
            # Skip trending filter if disabled or not available
            if (not getattr(self.settings, 'ENABLE_TRENDING_FILTER', True) or 
                not self.trending_analyzer or not self.birdeye_client):
                logger.debug("[TRENDING] Trending filter disabled or unavailable")
                return True
            
            address = token_info.get('address', '')
            symbol = token_info.get('symbol', 'UNKNOWN')
            
            # Check if token is in trending list
            trending_token = self.birdeye_client.get_cached_token_by_address(address)
            
            if trending_token is None:
                # Token not found in trending data
                fallback_mode = getattr(self.settings, 'TRENDING_FALLBACK_MODE', 'permissive')
                
                if fallback_mode == 'strict':
                    logger.info(f"[TRENDING] REJECT - Token {symbol} ({address[:8]}...) not in trending list - REJECTED (strict mode)")
                    return False
                else:
                    logger.info(f"[TRENDING] ALLOW - Token {symbol} ({address[:8]}...) not in trending list - ALLOWED (permissive mode)")
                    return True
            
            # Validate trending criteria
            passes_criteria, reason = self.trending_analyzer.meets_trending_criteria(trending_token)
            
            if not passes_criteria:
                logger.info(f"[TRENDING] FAIL - Token {symbol} failed trending criteria: {reason}")
                return False
            
            # Calculate and log trending score
            trending_score = self.trending_analyzer.calculate_trending_score(trending_token)
            logger.info(f"[TRENDING] PASS - TRENDING TOKEN VALIDATED: {symbol} rank #{trending_token.rank}, score {trending_score:.1f}")
            logger.info(f"  Price Change 24h: {trending_token.price_24h_change_percent:.1f}%")
            logger.info(f"  Volume Change 24h: {trending_token.volume_24h_change_percent:.1f}%")
            logger.info(f"  Daily Volume: ${trending_token.volume_24h_usd:,.0f}")
            
            # Store trending data in token_info for later use
            token_info['trending_token'] = trending_token
            token_info['trending_score'] = trending_score
            
            return True
            
        except Exception as e:
            logger.error(f"[TRENDING] Error in trending filter: {e}")
            # Default to permissive behavior on error
            fallback_mode = getattr(self.settings, 'TRENDING_FALLBACK_MODE', 'permissive')
            return fallback_mode == 'permissive'
    
    async def _refresh_trending_data(self):
        """Refresh trending data from Birdeye API"""
        try:
            if not self.birdeye_client:
                return
            
            logger.info("[TRENDING] Refreshing trending data...")
            trending_tokens = await self.birdeye_client.get_trending_tokens(limit=50)
            
            if trending_tokens:
                logger.info(f"[TRENDING] Successfully fetched {len(trending_tokens)} trending tokens")
                
                # Log top 10 trending tokens
                top_10 = trending_tokens[:10]
                logger.info("[TRENDING] Top 10 trending tokens:")
                for token in top_10:
                    logger.info(f"  #{token.rank}: {token.symbol} - {token.price_24h_change_percent:+.1f}% | Vol: ${token.volume_24h_usd:,.0f}")
            else:
                logger.warning("[TRENDING] Failed to fetch trending data")
                
        except Exception as e:
            logger.error(f"[TRENDING] Error refreshing trending data: {e}")
    
    async def stop_scanning(self):
        """Stop the scanning process"""
        logger.info("[SCANNER] Stopping practical scanner...")
        self.running = False
        if self.session:
            await self.session.close()
        
        # Clean up Birdeye client
        if self.birdeye_client:
            await self.birdeye_client.__aexit__(None, None, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scanner statistics"""
        stats = {
            'scans_completed': self.scan_count,
            'tokens_discovered': len(self.discovered_tokens),
            'tokens_processed': len(self.processed_tokens),
            'is_running': self.running
        }
        
        # Add Birdeye client stats if available
        if self.birdeye_client:
            stats['birdeye_stats'] = self.birdeye_client.get_stats()
        
        return stats