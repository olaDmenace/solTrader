import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
import json

from .api.solana_tracker import SolanaTrackerClient, TokenData
from .config.settings import Settings

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
    def __init__(self, settings: Settings):
        self.settings = settings
        self.solana_tracker = SolanaTrackerClient()
        
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
            self.solana_tracker.session = value

    async def start(self):
        """Start the scanning process"""
        if self.is_running:
            logger.warning("Scanner already running")
            return
            
        self.is_running = True
        logger.info("Starting enhanced token scanner...")
        
        await self.solana_tracker.start_session()
        
        # Test API connection (with retry for rate limiting)
        connection_attempts = 0
        max_attempts = 3
        
        while connection_attempts < max_attempts:
            if await self.solana_tracker.test_connection():
                logger.info("Solana Tracker API connection successful")
                break
            else:
                connection_attempts += 1
                if connection_attempts < max_attempts:
                    logger.warning(f"API connection failed, retrying in 10 seconds... (attempt {connection_attempts}/{max_attempts})")
                    await asyncio.sleep(10)
                else:
                    logger.error("❌ Failed to connect to Solana Tracker API after 3 attempts")
                    self.is_running = False
                    return
        
        # Start scanning task
        self.scan_task = asyncio.create_task(self._scan_loop())
        logger.info("Enhanced token scanner started successfully")

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
        
        await self.solana_tracker.close()
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
            all_tokens = await self.solana_tracker.get_all_tokens()
            
            logger.info(f"Retrieved {len(all_tokens)} total tokens from API")
            
            if not all_tokens:
                logger.warning("No tokens retrieved from API - checking connection")
                # Try to reconnect once
                if await self.solana_tracker.test_connection():
                    all_tokens = await self.solana_tracker.get_all_tokens()
                    logger.info(f"Retry successful: {len(all_tokens)} tokens")
                
            if not all_tokens:
                logger.error("Still no tokens after retry")
                return []
            
            # Update API usage stats
            usage_stats = self.solana_tracker.get_usage_stats()
            self.daily_stats['api_requests_used'] = usage_stats['requests_today']
            
            # Filter and score tokens
            approved_tokens = []
            self.daily_stats['tokens_scanned'] = len(all_tokens)
            
            for token in all_tokens:
                try:
                    # Update source discovery stats
                    self.source_stats[token.source]['discovered'] += 1
                    
                    # Apply filters and scoring
                    result = await self._evaluate_token(token)
                    
                    if result:
                        approved_tokens.append(result)
                        self.source_stats[token.source]['approved'] += 1
                        self.discovered_tokens[token.address] = result
                        logger.info(f"APPROVED: {token.symbol} - Score: {result.score:.1f} - {result.reasons}")
                        
                        # Update source effectiveness
                        source_stats = self.source_stats[token.source]
                        if source_stats['discovered'] > 0:
                            source_stats['effectiveness'] = (
                                source_stats['approved'] / source_stats['discovered']
                            ) * 100
                    else:
                        logger.info(f"REJECTED: {token.symbol} - Liquidity: {token.liquidity}, Momentum: {token.price_change_24h}%, Age: {token.age_minutes}min")
                        
                except Exception as e:
                    logger.error(f"Error evaluating token {token.address}: {e}")
                    continue
            
            # Update daily stats
            self.daily_stats['tokens_approved'] = len(approved_tokens)
            if self.daily_stats['tokens_scanned'] > 0:
                self.daily_stats['approval_rate'] = (
                    self.daily_stats['tokens_approved'] / self.daily_stats['tokens_scanned']
                ) * 100
            
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
            # More permissive liquidity filter for medium momentum tokens
            liquidity_threshold = self.min_liquidity / 2 if medium_momentum else self.min_liquidity
            if token.liquidity < liquidity_threshold:
                return None
            reasons.append(f"Liquidity: {token.liquidity:.0f} SOL")
            score += min(token.liquidity / 500, 15)  # More generous scoring, up to 15 points
            
            # More permissive momentum filter
            momentum_threshold = self.min_momentum_percentage / 2 if medium_momentum else self.min_momentum_percentage
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
        
        # Much lower score threshold for 40-60% approval rate
        if high_momentum:
            min_score = 0  # High momentum tokens always pass
        elif medium_momentum:
            min_score = 5  # Very low threshold for medium momentum
        else:
            min_score = 8  # Significantly reduced from 15 to 8 for regular tokens
            
        if score < min_score:
            return None
        
        # Calculate source effectiveness
        source_effectiveness = self.source_stats[token.source]['effectiveness']
        
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
        usage_stats = self.solana_tracker.get_usage_stats()
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
            
            # Return the highest-scoring token
            best_token = approved_tokens[0]  # Already sorted by score
            token_data = best_token.token
            
            logger.info(f"Selected token: {token_data.symbol} (score: {best_token.score:.1f}, source: {token_data.source})")
            
            # Convert to dict format expected by strategy
            return {
                'address': token_data.address,
                'symbol': token_data.symbol,
                'name': token_data.name,
                'price': token_data.price,
                'price_change_24h': token_data.price_change_24h,
                'volume_24h': token_data.volume_24h,
                'market_cap': token_data.market_cap,
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