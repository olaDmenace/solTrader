import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
import json

from .api.solana_tracker import SolanaTrackerClient, TokenData
from .api.geckoterminal_client import GeckoTerminalClient
from .api.smart_dual_api_manager import SmartDualAPIManager
from .config.settings import Settings
import os

logger = logging.getLogger(__name__)

@dataclass
class ScanResult:
    token: TokenData
    score: float
    reasons: List[str]
    bypassed_filters: List[str]
    discovery_time: datetime
    source_effectiveness: float

class EnhancedTokenScanner:
    def __init__(self, settings: Settings, analytics=None):
        self.settings = settings
        self.analytics = analytics  # Analytics system integration
        
        # API Provider Selection - Smart Dual-API Strategy
        api_strategy = os.getenv('API_STRATEGY', 'dual')
        
        if api_strategy == 'dual':
            self.api_client = SmartDualAPIManager()
            logger.info("Using Smart Dual-API Manager - Target: 2,500+ tokens/day")
        elif api_strategy == 'geckoterminal':
            self.api_client = GeckoTerminalClient()
            logger.info("Using GeckoTerminal API - FREE with 43K+ daily requests")
        elif api_strategy == 'solana_tracker':
            self.api_client = SolanaTrackerClient()
            logger.info("Using Solana Tracker API")
        else:
            # Default to dual-API for maximum discovery
            self.api_client = SmartDualAPIManager()
            logger.info("Using Smart Dual-API Manager (default) - Target: 2,500+ tokens/day")
        
        # Legacy reference for compatibility
        self.solana_tracker = self.api_client
        
        # Optimized filter settings for 40-60% approval rate
        self.min_liquidity = 100.0  # FURTHER REDUCED from 250 SOL for higher approval
        self.min_momentum_percentage = 5.0  # REDUCED from 10% to 5% for more opportunities
        self.min_volume_growth = 0.0  # Removed requirement
        self.max_token_age_hours = 24  # EXTENDED from 12 to 24 hours for more tokens
        self.high_momentum_bypass = 500.0  # LOWERED from 1000% to 500% for more bypasses
        self.medium_momentum_bypass = 100.0  # NEW: Medium momentum bypass at 100%
        
        # Discovery analytics
        self.discovered_tokens: Dict[str, ScanResult] = {}
        self.source_stats = {
            'trending': {'discovered': 0, 'approved': 0, 'effectiveness': 0.0},
            'volume': {'discovered': 0, 'approved': 0, 'effectiveness': 0.0},
            'memescope': {'discovered': 0, 'approved': 0, 'effectiveness': 0.0}
        }
        
        # Performance tracking
        self.scan_history = []
        self.approval_history = []
        self.daily_stats = {
            'tokens_scanned': 0,
            'tokens_approved': 0,
            'approval_rate': 0.0,
            'high_momentum_bypasses': 0,
            'api_requests_used': 0
        }
        
        # Scheduling state
        self.last_full_scan = 0
        self.scan_interval = 300  # 5 minutes between full scans
        
        self.is_running = False
        self.scan_task: Optional[asyncio.Task] = None
        
        logger.info("Enhanced Token Scanner initialized for 40-60% approval rate")
        logger.info(f"Min liquidity: {self.min_liquidity} SOL")
        logger.info(f"Min momentum: {self.min_momentum_percentage}%")
        logger.info(f"Max age: {self.max_token_age_hours} hours")
        logger.info(f"High momentum bypass: {self.high_momentum_bypass}%")
        logger.info(f"Medium momentum bypass: {self.medium_momentum_bypass}%")

    def _normalize_source_name(self, source: str) -> str:
        """Normalize source names to match source_stats keys"""
        # Handle compound source names like 'geckoterminal/volume'
        if '/' in source:
            source = source.split('/')[-1]  # Use the part after slash
        
        # Map known source types
        source_mapping = {
            'trending': 'trending',
            'volume': 'volume', 
            'memescope': 'memescope',
            'solana_tracker': 'trending',  # Default solana tracker to trending
            'geckoterminal': 'trending'     # Default geckoterminal to trending
        }
        
        return source_mapping.get(source, 'trending')  # Default to trending

    @property
    def running(self) -> bool:
        """Compatibility property for strategy.py"""
        return self.is_running
    
    @running.setter 
    def running(self, value: bool):
        """Compatibility setter for strategy.py"""
        self.is_running = value

    @property
    def session(self):
        """Compatibility property for strategy.py - delegates to solana_tracker session"""
        return getattr(self.solana_tracker, 'session', None)
    
    @session.setter
    def session(self, value):
        """Compatibility setter for strategy.py"""
        if hasattr(self.solana_tracker, 'session'):
            self.api_client.session = value

    async def start(self):
        """Start the scanning process"""
        if self.is_running:
            logger.warning("Scanner already running")
            return
            
        self.is_running = True
        logger.info("Starting enhanced token scanner...")
        
        try:
            # Start API session with timeout
            logger.info("Initializing API client session...")
            await asyncio.wait_for(self.api_client.start_session(), timeout=30)
            logger.info("API client session initialized")
            
            # Test API connection (with shorter timeout and retries)
            connection_attempts = 0
            max_attempts = 2  # Reduced from 3 attempts
            
            while connection_attempts < max_attempts:
                try:
                    logger.info(f"Testing API connection (attempt {connection_attempts + 1}/{max_attempts})...")
                    # Add timeout to prevent hanging
                    connection_result = await asyncio.wait_for(
                        self.api_client.test_connection(), 
                        timeout=15  # 15 second timeout
                    )
                    
                    if connection_result:
                        logger.info("Solana Tracker API connection successful")
                        break
                    else:
                        logger.warning("API connection test returned False")
                        connection_attempts += 1
                        
                except asyncio.TimeoutError:
                    logger.warning(f"API connection test timed out (attempt {connection_attempts + 1})")
                    connection_attempts += 1
                except Exception as conn_error:
                    logger.warning(f"API connection test failed: {conn_error}")
                    connection_attempts += 1
                
                if connection_attempts < max_attempts:
                    logger.info("Waiting 5 seconds before retry...")
                    await asyncio.sleep(5)  # Reduced from 10 seconds
            
            # Even if API test fails, continue with reduced functionality
            if connection_attempts >= max_attempts:
                logger.warning("API connection tests failed, but continuing with limited functionality")
            
            # Start scanning task
            logger.info("Starting scanning task...")
            self.scan_task = asyncio.create_task(self._scan_loop())
            logger.info("Enhanced token scanner started successfully")
            
        except asyncio.TimeoutError:
            logger.error("Scanner initialization timed out")
            self.is_running = False
            return
        except Exception as e:
            logger.error(f"Scanner initialization failed: {e}")
            self.is_running = False
            return

    async def stop(self):
        """Stop the scanning process"""
        if not self.is_running:
            return
            
        logger.info("Stopping enhanced token scanner...")
        self.is_running = False
        
        if self.scan_task:
            self.scan_task.cancel()
            try:
                await self.scan_task
            except asyncio.CancelledError:
                pass
        
        # Explicit session cleanup
        try:
            await self.api_client.close()
            logger.info("API client sessions closed successfully")
        except Exception as e:
            logger.warning(f"Error closing API sessions: {e}")
        
        logger.info("Enhanced token scanner stopped")

    async def _scan_loop(self):
        """Main scanning loop with smart scheduling"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Check if it's time for a full scan
                if time.time() - self.last_full_scan >= self.scan_interval:
                    await self._perform_full_scan()
                    self.last_full_scan = time.time()
                
                # Calculate sleep time
                scan_duration = time.time() - start_time
                sleep_time = max(1, self.scan_interval - scan_duration)
                
                logger.debug(f"Scan completed in {scan_duration:.2f}s, sleeping for {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scan loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    async def _perform_full_scan(self):
        """Perform a comprehensive scan of all token sources"""
        logger.info("Starting full token scan...")
        scan_start = time.time()
        
        try:
            # Get tokens from all sources
            all_tokens = await self.api_client.get_all_tokens()
            
            logger.info(f"Retrieved {len(all_tokens)} total tokens from API")
            
            if not all_tokens:
                logger.warning("No tokens retrieved from API - checking connection")
                # Try to reconnect once
                if await self.api_client.test_connection():
                    all_tokens = await self.api_client.get_all_tokens()
                    logger.info(f"Retry successful: {len(all_tokens)} tokens")
                
            if not all_tokens:
                logger.error("Still no tokens after retry")
                return []
            
            # Update API usage stats with error handling
            try:
                usage_stats = self.api_client.get_usage_stats()
                self.daily_stats['api_requests_used'] = usage_stats.get('requests_today', 0)
            except Exception as stats_error:
                logger.warning(f"Failed to get usage stats: {stats_error}")
                self.daily_stats['api_requests_used'] = 0
            
            # Filter and score tokens
            approved_tokens = []
            self.daily_stats['tokens_scanned'] = len(all_tokens)
            
            for token in all_tokens:
                try:
                    # Normalize source name for stats tracking
                    source_key = self._normalize_source_name(token.source)
                    
                    # Update source discovery stats
                    if source_key in self.source_stats:
                        self.source_stats[source_key]['discovered'] += 1
                    
                    # Apply filters and scoring
                    result = await self._evaluate_token(token)
                    
                    if result:
                        approved_tokens.append(result)
                        if source_key in self.source_stats:
                            self.source_stats[source_key]['approved'] += 1
                        self.discovered_tokens[token.address] = result
                        logger.info("APPROVED: %s - Score: %.1f - %s", 
                                   getattr(token, 'symbol', 'unknown'), 
                                   result.score, 
                                   result.reasons)
                        
                        # Update source effectiveness
                        if source_key in self.source_stats:
                            source_stats = self.source_stats[source_key]
                            if source_stats['discovered'] > 0:
                                source_stats['effectiveness'] = (
                                    source_stats['approved'] / source_stats['discovered']
                                ) * 100
                    else:
                        logger.info("REJECTED: %s - Liquidity: %s, Momentum: %s%%, Age: %smin", 
                                   getattr(token, 'symbol', 'unknown'), 
                                   getattr(token, 'liquidity', 0),
                                   getattr(token, 'price_change_24h', 0), 
                                   getattr(token, 'age_minutes', 0))
                        
                except Exception as e:
                    logger.error("Error evaluating token %s: %s", getattr(token, 'address', 'unknown'), str(e))
                    continue
            
            # Update daily stats
            self.daily_stats['tokens_approved'] = len(approved_tokens)
            if self.daily_stats['tokens_scanned'] > 0:
                self.daily_stats['approval_rate'] = (
                    self.daily_stats['tokens_approved'] / self.daily_stats['tokens_scanned']
                ) * 100
            
            # Update analytics system if available with error handling
            if self.analytics:
                try:
                    usage_stats = self.api_client.get_usage_stats()
                    api_requests = usage_stats.get('requests_today', 0)
                    
                    self.analytics.update_scanner_stats(
                        tokens_scanned=len(all_tokens),
                        tokens_approved=len(approved_tokens),
                        high_momentum_bypasses=self.daily_stats['high_momentum_bypasses'],
                        api_requests_used=api_requests
                    )
                    
                    # Update discovery analytics for each source
                    for source, stats in self.source_stats.items():
                        if stats['discovered'] > 0:
                            try:
                                avg_age = sum(token.age_minutes for token in all_tokens if token.source == source) / max(stats['discovered'], 1)
                                liquidity_values = [token.liquidity for token in all_tokens if token.source == source and token.liquidity > 0]
                                liquidity_stats = {
                                    'min': min(liquidity_values) if liquidity_values else 0,
                                    'max': max(liquidity_values) if liquidity_values else 0,
                                    'avg': sum(liquidity_values) / len(liquidity_values) if liquidity_values else 0
                                }
                                self.analytics.update_discovery_analytics(
                                    source=source,
                                    discovered=stats['discovered'],
                                    approved=stats['approved'],
                                    avg_age=avg_age,
                                    liquidity_stats=liquidity_stats
                                )
                            except Exception as source_error:
                                logger.warning(f"Failed to update analytics for source {source}: {source_error}")
                                
                except Exception as analytics_error:
                    logger.warning(f"Failed to update analytics: {analytics_error}")
            
            # Log scan results
            scan_duration = time.time() - scan_start
            logger.info(f"Scan completed: {len(approved_tokens)}/{len(all_tokens)} tokens approved "
                       f"({self.daily_stats['approval_rate']:.1f}% rate) in {scan_duration:.2f}s")
            
            # Log source effectiveness
            for source, stats in self.source_stats.items():
                if stats['discovered'] > 0:
                    logger.info(f"{source.capitalize()}: {stats['approved']}/{stats['discovered']} "
                               f"({stats['effectiveness']:.1f}% effective)")
            
            return approved_tokens
            
        except Exception as e:
            logger.error(f"Error in full scan: {e}")
            
            # Try to recover from API errors
            try:
                # Check if it's an API-related error
                if 'requests_' in str(e) or 'usage_stats' in str(e):
                    logger.warning("API stats error detected - attempting recovery")
                    # Reset API client if needed
                    if hasattr(self.api_client, 'session') and self.api_client.session:
                        await self.api_client.close()
                        await asyncio.sleep(2)
                        await self.api_client.start_session()
                    
                    # Return empty list to prevent crash but continue operation
                    logger.info("Recovery attempt completed - continuing with empty token list")
                    return []
                    
            except Exception as recovery_error:
                logger.error(f"Recovery attempt failed: {recovery_error}")
            
            return []

    async def _evaluate_token(self, token: TokenData) -> Optional[ScanResult]:
        """Evaluate a token against filters and assign a score"""
        reasons = []
        bypassed_filters = []
        score = 0.0
        
        # Check for momentum bypasses (multiple levels for higher approval rate)
        high_momentum = token.price_change_24h >= self.high_momentum_bypass
        medium_momentum = token.price_change_24h >= self.medium_momentum_bypass
        
        if high_momentum:
            bypassed_filters.append(f"High momentum bypass: {token.price_change_24h:.1f}%")
            self.daily_stats['high_momentum_bypasses'] += 1
            score += 60  # Major score boost for high momentum
            reasons.append(f"Explosive momentum: +{token.price_change_24h:.1f}%")
        elif medium_momentum:
            bypassed_filters.append(f"Medium momentum bypass: {token.price_change_24h:.1f}%")
            score += 30  # Good score boost for medium momentum  
            reasons.append(f"Strong momentum: +{token.price_change_24h:.1f}%")
        
        # Apply standard filters (unless high momentum bypassed)
        if not high_momentum:
            # ULTRA-PERMISSIVE liquidity filter for maximum opportunities
            liquidity_threshold = self.min_liquidity / 10 if medium_momentum else self.min_liquidity / 5  # Much lower
            if token.liquidity < liquidity_threshold:
                return None
            reasons.append(f"Liquidity: {token.liquidity:.0f} SOL")
            score += min(token.liquidity / 500, 15)  # More generous scoring, up to 15 points
            
            # ULTRA-PERMISSIVE momentum filter (allow even small gains)
            momentum_threshold = max(0.5, self.min_momentum_percentage / 10)  # Minimum 0.5% gain
            if token.price_change_24h < momentum_threshold:
                return None
            reasons.append(f"Momentum: +{token.price_change_24h:.1f}%")
            score += min(token.price_change_24h / 5, 25)  # More generous scoring, up to 25 points
            
            # Extended token age filter (even more permissive for medium momentum)
            age_limit_hours = self.max_token_age_hours * 1.5 if medium_momentum else self.max_token_age_hours
            if token.age_minutes > (age_limit_hours * 60):
                return None
            reasons.append(f"Age: {token.age_minutes} minutes")
            # More generous age scoring
            age_score = max(0, 15 - (token.age_minutes / 120))  # Up to 15 points, slower decay
            score += age_score
        
        # Volume scoring (no minimum requirement)
        if token.volume_24h > 0:
            volume_score = min(token.volume_24h / 10000, 15)  # Up to 15 points
            score += volume_score
            reasons.append(f"Volume: ${token.volume_24h:.0f}")
        
        # Market cap scoring
        if 1000 <= token.market_cap <= 100000:  # Sweet spot for growth
            score += 10
            reasons.append(f"Market cap: ${token.market_cap:.0f}")
        
        # Source-based scoring
        source_multipliers = {
            'trending': 1.2,  # Trending tokens get boost
            'volume': 1.1,    # Volume tokens get small boost  
            'memescope': 1.0  # Base score for memescope
        }
        score *= source_multipliers.get(token.source, 1.0)
        
        # Momentum score from API
        score += token.momentum_score * 2  # Up to 20 points
        
        # ULTRA-LOW score threshold for maximum paper trading opportunities
        if high_momentum:
            min_score = 0  # High momentum tokens always pass
        elif medium_momentum:
            min_score = 2  # Ultra-low threshold for medium momentum
        else:
            min_score = 3  # Ultra-low threshold for maximum opportunities (was 8)
            
        if score < min_score:
            return None
        
        # Calculate source effectiveness
        source_key = self._normalize_source_name(token.source)
        source_effectiveness = self.source_stats.get(source_key, {}).get('effectiveness', 0.0)
        
        return ScanResult(
            token=token,
            score=score,
            reasons=reasons,
            bypassed_filters=bypassed_filters,
            discovery_time=datetime.now(),
            source_effectiveness=source_effectiveness
        )

    async def get_approved_tokens(self) -> List[ScanResult]:
        """Get currently approved tokens"""
        current_time = datetime.now()
        valid_tokens = []
        
        for address, result in self.discovered_tokens.items():
            # Remove stale discoveries (older than 30 minutes)
            if (current_time - result.discovery_time).total_seconds() > 1800:
                continue
            valid_tokens.append(result)
        
        # Sort by score descending
        return sorted(valid_tokens, key=lambda x: x.score, reverse=True)

    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily statistics"""
        stats = self.daily_stats.copy()
        
        # Add source breakdown
        stats['source_breakdown'] = {}
        for source, data in self.source_stats.items():
            stats['source_breakdown'][source] = {
                'discovered': data['discovered'],
                'approved': data['approved'],
                'effectiveness': f"{data['effectiveness']:.1f}%"
            }
        
        # Add API usage
        usage_stats = self.api_client.get_usage_stats()
        stats['api_usage'] = {
            'requests_used': usage_stats['requests_today'],
            'daily_limit': usage_stats['daily_limit'],
            'remaining': usage_stats['remaining_requests'],
            'usage_percentage': f"{usage_stats['usage_percentage']:.1f}%"
        }
        
        return stats

    def get_discovery_analytics(self) -> Dict[str, Any]:
        """Get detailed discovery analytics"""
        current_time = datetime.now()
        
        # Calculate average discovery time by source
        source_times = {source: [] for source in self.source_stats.keys()}
        for result in self.discovered_tokens.values():
            age_minutes = (current_time - result.discovery_time).total_seconds() / 60
            if age_minutes <= 1440:  # Last 24 hours
                source_times[result.token.source].append(result.token.age_minutes)
        
        # Calculate current approved tokens without async call
        current_time = datetime.now()
        current_approved = 0
        for result in self.discovered_tokens.values():
            if (current_time - result.discovery_time).total_seconds() <= 1800:  # 30 minutes
                current_approved += 1
        
        analytics = {
            'total_discovered': len(self.discovered_tokens),
            'current_approved': current_approved,
            'source_effectiveness': {
                source: {
                    'effectiveness_rate': f"{data['effectiveness']:.1f}%",
                    'avg_discovery_age': sum(source_times[source]) / len(source_times[source]) 
                                       if source_times[source] else 0,
                    'discovered_count': data['discovered'],
                    'approved_count': data['approved']
                }
                for source, data in self.source_stats.items()
            },
            'high_momentum_bypasses': self.daily_stats['high_momentum_bypasses'],
            'filter_optimization': {
                'min_liquidity': f"{self.min_liquidity} SOL",
                'min_momentum': f"{self.min_momentum_percentage}%",
                'max_age': f"{self.max_token_age_hours} hours",
                'volume_requirement': "None" if self.min_volume_growth == 0 else f"{self.min_volume_growth}%"
            }
        }
        
        return analytics

    async def scan_for_new_tokens(self) -> Optional[Dict[str, Any]]:
        """Scan for new tokens and return the best candidate"""
        try:
            # Get approved tokens
            approved_tokens = await self.get_approved_tokens()
            
            logger.info(f"Scan result: {len(approved_tokens)} approved tokens available")
            
            if not approved_tokens:
                logger.warning("No approved tokens found - performing fresh scan")
                # Try a fresh scan if no approved tokens
                fresh_tokens = await self._perform_full_scan()
                if fresh_tokens:
                    approved_tokens = fresh_tokens
                    logger.info(f"Fresh scan found {len(approved_tokens)} tokens")
                else:
                    logger.error("Fresh scan also returned 0 tokens")
                    return None
            
            # APE-ING STRATEGY: Prioritize smaller market cap tokens for higher growth potential
            # Filter tokens by market cap for optimal selection
            small_cap_tokens = []
            medium_cap_tokens = []
            large_cap_tokens = []
            
            for token in approved_tokens:
                market_cap = token.token.market_cap
                if market_cap < 1000000:  # Under $1M - small cap
                    small_cap_tokens.append(token)
                elif market_cap < 10000000:  # $1M-$10M - medium cap  
                    medium_cap_tokens.append(token)
                else:  # Over $10M - large cap
                    large_cap_tokens.append(token)
            
            # Prioritize smaller caps for APE-ing strategy
            if small_cap_tokens:
                best_token = small_cap_tokens[0]  # Highest score among small caps
                logger.info(f"[APE-ING] Selected SMALL CAP token for maximum growth potential")
            elif medium_cap_tokens:
                best_token = medium_cap_tokens[0]  # Highest score among medium caps
                logger.info(f"[APE-ING] Selected MEDIUM CAP token (no small caps available)")
            else:
                best_token = large_cap_tokens[0] if large_cap_tokens else approved_tokens[0]
                logger.info(f"[APE-ING] Selected LARGE CAP token (no smaller caps available)")
            
            token_data = best_token.token
            market_cap_str = f"${token_data.market_cap:,.0f}" if token_data.market_cap > 0 else "Unknown"
            logger.info(f"Selected token: {token_data.symbol} (score: {best_token.score:.1f}, market cap: {market_cap_str}, source: {token_data.source})")
            
            # Convert to dict format expected by strategy
            # Clean token address format (remove "solana_" prefix if present)
            clean_address = token_data.address
            if clean_address.startswith('solana_'):
                clean_address = clean_address[7:]  # Remove "solana_" prefix
                logger.info(f"Cleaned address format: {token_data.address} -> {clean_address}")
            
            # Fix missing market cap data
            market_cap = token_data.market_cap
            if market_cap <= 0 and token_data.price > 0:
                # Estimate market cap from liquidity and price data
                # Conservative estimate: assume liquidity represents ~5-10% of total supply
                liquidity_ratio = 0.08  # Assume 8% of tokens are in liquidity pool
                estimated_supply = (token_data.liquidity * 2) / (token_data.price * liquidity_ratio)  # *2 for both sides of pool
                market_cap = token_data.price * estimated_supply
                logger.info(f"Estimated market cap for {token_data.symbol}: ${market_cap:.0f} (from price ${token_data.price:.8f} and liquidity {token_data.liquidity:.0f})")
            
            return {
                'address': clean_address,
                'symbol': token_data.symbol,
                'name': token_data.name,
                'price': token_data.price,
                'price_change_24h': token_data.price_change_24h,
                'volume_24h': token_data.volume_24h,
                'market_cap': market_cap,
                'liquidity': token_data.liquidity,
                'age_minutes': token_data.age_minutes,
                'momentum_score': token_data.momentum_score,
                'source': token_data.source,
                'score': best_token.score,
                'reasons': best_token.reasons,
                'bypassed_filters': best_token.bypassed_filters
            }
            
        except Exception as e:
            logger.error(f"Error in scan_for_new_tokens: {e}")
            return None

    async def manual_scan(self) -> List[ScanResult]:
        """Perform a manual scan outside the scheduled loop"""
        logger.info("Performing manual token scan...")
        return await self._perform_full_scan()

    def reset_daily_stats(self):
        """Reset daily statistics (typically called at midnight)"""
        self.daily_stats = {
            'tokens_scanned': 0,
            'tokens_approved': 0,
            'approval_rate': 0.0,
            'high_momentum_bypasses': 0,
            'api_requests_used': 0
        }
        
        # Reset source stats
        for source in self.source_stats:
            self.source_stats[source] = {'discovered': 0, 'approved': 0, 'effectiveness': 0.0}
        
        logger.info("Daily statistics reset")