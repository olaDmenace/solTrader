import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from .solana_tracker import SolanaTrackerClient, TokenData
from .geckoterminal_client import GeckoTerminalClient
from .adaptive_quota_manager import AdaptiveQuotaManager

logger = logging.getLogger(__name__)

class APIProvider(Enum):
    SOLANA_TRACKER = "solana_tracker"
    GECKOTERMINAL = "geckoterminal"

class APIStatus(Enum):
    HEALTHY = "healthy"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    ERROR = "error"

@dataclass
class APIPerformance:
    provider: APIProvider
    status: APIStatus
    tokens_discovered: int = 0
    unique_tokens: int = 0
    quality_score: float = 0.0
    response_time_avg: float = 0.0
    success_rate: float = 1.0
    quota_used: int = 0
    quota_remaining: int = 0
    last_success: datetime = field(default_factory=datetime.now)
    last_error: Optional[str] = None
    daily_discoveries: List[int] = field(default_factory=list)

@dataclass
class TokenDiscoveryResult:
    tokens: List[TokenData]
    provider: APIProvider
    source: str
    discovery_time: datetime
    quality_metrics: Dict[str, float]
    api_calls_used: int

class SmartDualAPIManager:
    """
    Intelligent dual-API manager that maximizes token discovery while maintaining system stability.
    
    Key Features:
    - Dynamic API selection based on quota, performance, and token quality
    - Intelligent failover and load balancing
    - Adaptive quota management to prevent exhaustion
    - Token deduplication and quality scoring
    - Performance optimization and learning
    """
    
    def __init__(self):
        # Initialize API clients
        self.solana_tracker = SolanaTrackerClient()
        self.geckoterminal = GeckoTerminalClient()
        
        # Initialize adaptive quota manager
        self.quota_manager = AdaptiveQuotaManager()
        
        # Performance tracking
        self.performance = {
            APIProvider.SOLANA_TRACKER: APIPerformance(APIProvider.SOLANA_TRACKER, APIStatus.HEALTHY),
            APIProvider.GECKOTERMINAL: APIPerformance(APIProvider.GECKOTERMINAL, APIStatus.HEALTHY)
        }
        
        # Smart scheduling
        self.primary_provider = APIProvider.SOLANA_TRACKER  # High-volume primary
        self.fallback_provider = APIProvider.GECKOTERMINAL  # Unlimited fallback
        self.last_switch_time = 0
        self.switch_cooldown = 300  # 5 minutes before switching back
        
        # Discovery optimization
        self.token_cache = {}  # Address -> TokenData
        self.quality_threshold = 0.6
        self.max_age_hours = 24
        
        # Quota management
        self.daily_quotas = {
            APIProvider.SOLANA_TRACKER: 333,  # Conservative daily limit
            APIProvider.GECKOTERMINAL: 36000  # Very high but not unlimited
        }
        
        # Learning system
        self.discovery_history = []
        self.optimization_data = {
            'best_combinations': [],
            'provider_effectiveness': {},
            'time_based_patterns': {}
        }
        
        logger.info("Smart Dual-API Manager initialized - Target: 2,500+ tokens/day")

    async def __aenter__(self):
        await self.start_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def start_session(self):
        """Initialize both API clients"""
        await self.solana_tracker.start_session()
        await self.geckoterminal.start_session()
        logger.info("Dual-API sessions initialized")

    async def close(self):
        """Close both API clients"""
        await self.solana_tracker.close()
        await self.geckoterminal.close()
        logger.info("Dual-API sessions closed")

    def _update_performance_metrics(self, provider: APIProvider, success: bool, 
                                  tokens_found: int, response_time: float, error: str = None):
        """Update performance tracking for an API provider"""
        perf = self.performance[provider]
        
        if success:
            perf.tokens_discovered += tokens_found
            perf.last_success = datetime.now()
            perf.response_time_avg = (perf.response_time_avg + response_time) / 2
        else:
            perf.last_error = error
            
        # Update success rate (rolling average)
        current_success_rate = 1.0 if success else 0.0
        perf.success_rate = (perf.success_rate * 0.9) + (current_success_rate * 0.1)
        
        # Update status based on performance
        if error and "403" in error:
            perf.status = APIStatus.QUOTA_EXHAUSTED
        elif error and "429" in error:
            perf.status = APIStatus.RATE_LIMITED
        elif not success:
            perf.status = APIStatus.ERROR
        else:
            perf.status = APIStatus.HEALTHY

    def _calculate_provider_priority(self) -> List[APIProvider]:
        """Calculate optimal provider order based on current conditions"""
        scores = {}
        
        for provider, perf in self.performance.items():
            score = 0.0
            
            # Quota availability (40% weight)
            quota_remaining = self.daily_quotas[provider] - perf.quota_used
            quota_factor = quota_remaining / self.daily_quotas[provider]
            score += quota_factor * 0.4
            
            # Token discovery rate (30% weight) 
            discovery_rate = perf.tokens_discovered / max(1, perf.quota_used)
            normalized_rate = min(1.0, discovery_rate / 50)  # Normalize to 50 tokens per call
            score += normalized_rate * 0.3
            
            # API health (20% weight)
            health_score = 1.0 if perf.status == APIStatus.HEALTHY else 0.5
            score += health_score * 0.2
            
            # Success rate (10% weight)
            score += perf.success_rate * 0.1
            
            scores[provider] = score
            
        # Sort by score descending
        return sorted(scores.keys(), key=lambda p: scores[p], reverse=True)

    async def _discover_from_provider(self, provider: APIProvider, source: str) -> Optional[TokenDiscoveryResult]:
        """Discover tokens from a specific provider and source"""
        client = self.solana_tracker if provider == APIProvider.SOLANA_TRACKER else self.geckoterminal
        start_time = time.time()
        
        try:
            # Request quota allocation before making API call
            provider_name = 'solana_tracker' if provider == APIProvider.SOLANA_TRACKER else 'geckoterminal'
            quota_approved, allocated_quota, quota_reason = self.quota_manager.request_quota(provider_name, 1)
            
            if not quota_approved:
                logger.warning(f"Quota denied for {provider_name}/{source}: {quota_reason}")
                return None
            
            # Get tokens based on source
            if source == 'trending':
                tokens = await client.get_trending_tokens(limit=50)
            elif source == 'volume':
                tokens = await client.get_volume_tokens(limit=50) 
            elif source == 'memescope':
                if hasattr(client, 'get_memescope_tokens'):
                    tokens = await client.get_memescope_tokens(limit=50)
                else:
                    # Fallback for GeckoTerminal
                    tokens = await client.get_volume_tokens(limit=25)
            else:
                tokens = []
                
            response_time = time.time() - start_time
            
            # Record performance for quota optimization
            self.quota_manager.record_performance(
                provider_name, allocated_quota, len(tokens), response_time, True
            )
            
            # Update performance metrics
            self._update_performance_metrics(provider, True, len(tokens), response_time)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_token_quality(tokens)
            
            # Update quota usage
            self.performance[provider].quota_used += 1
            
            return TokenDiscoveryResult(
                tokens=tokens,
                provider=provider,
                source=source,
                discovery_time=datetime.now(),
                quality_metrics=quality_metrics,
                api_calls_used=1
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            
            # Record failed performance
            provider_name = 'solana_tracker' if provider == APIProvider.SOLANA_TRACKER else 'geckoterminal'
            self.quota_manager.record_performance(provider_name, 1, 0, response_time, False)
            
            self._update_performance_metrics(provider, False, 0, response_time, error_msg)
            
            logger.warning(f"Discovery failed from {provider.value}/{source}: {error_msg}")
            return None

    def _calculate_token_quality(self, tokens: List[TokenData]) -> Dict[str, float]:
        """Calculate quality metrics for discovered tokens"""
        if not tokens:
            return {'avg_score': 0.0, 'high_quality_ratio': 0.0, 'diversity_score': 0.0}
        
        scores = [token.momentum_score for token in tokens]
        high_quality = sum(1 for score in scores if score >= 5.0)
        
        # Diversity based on price ranges
        price_ranges = set()
        for token in tokens:
            if token.price < 0.0001:
                price_ranges.add('micro')
            elif token.price < 0.01:
                price_ranges.add('small')
            elif token.price < 1.0:
                price_ranges.add('medium')
            else:
                price_ranges.add('large')
        
        return {
            'avg_score': sum(scores) / len(scores),
            'high_quality_ratio': high_quality / len(tokens),
            'diversity_score': len(price_ranges) / 4.0,
            'token_count': len(tokens)
        }

    def _deduplicate_tokens(self, results: List[TokenDiscoveryResult]) -> List[TokenData]:
        """Deduplicate tokens and merge quality scores"""
        unique_tokens = {}
        provider_sources = {}
        
        for result in results:
            for token in result.tokens:
                if token.address in unique_tokens:
                    # Keep token with higher momentum score
                    existing = unique_tokens[token.address]
                    if token.momentum_score > existing.momentum_score:
                        unique_tokens[token.address] = token
                        provider_sources[token.address] = f"{result.provider.value}/{result.source}"
                else:
                    unique_tokens[token.address] = token
                    provider_sources[token.address] = f"{result.provider.value}/{result.source}"
        
        # Update token sources for tracking
        for address, token in unique_tokens.items():
            token.source = provider_sources[address]
            
        return list(unique_tokens.values())

    async def discover_tokens_intelligently(self) -> List[TokenData]:
        """
        Intelligently discover tokens using optimal dual-API strategy
        
        Strategy:
        1. Use Solana Tracker for high-volume discovery when quota available
        2. Use GeckoTerminal for consistent baseline discovery
        3. Intelligent failover when quotas exhausted
        4. Dynamic load balancing based on performance
        """
        discovery_results = []
        provider_priority = self._calculate_provider_priority()
        
        logger.info(f"Starting intelligent token discovery - Priority: {[p.value for p in provider_priority]}")
        
        # Parallel discovery strategy
        discovery_tasks = []
        
        for provider in provider_priority:
            perf = self.performance[provider]
            
            # Skip if quota exhausted or in error state
            if perf.status in [APIStatus.QUOTA_EXHAUSTED, APIStatus.ERROR]:
                continue
                
            # Skip if quota very low (save for high-value requests)
            quota_remaining = self.daily_quotas[provider] - perf.quota_used
            if quota_remaining < 10 and provider == APIProvider.SOLANA_TRACKER:
                logger.info(f"Conserving {provider.value} quota - {quota_remaining} calls remaining")
                continue
            
            # Determine sources to query based on provider and quota
            if provider == APIProvider.SOLANA_TRACKER and quota_remaining >= 3:
                # Use all sources for high-volume discovery
                sources = ['trending', 'volume', 'memescope']
            elif provider == APIProvider.GECKOTERMINAL:
                # Use available sources
                sources = ['trending', 'volume'] 
            else:
                # Conservative single source
                sources = ['trending']
            
            # Create discovery tasks
            for source in sources:
                task = self._discover_from_provider(provider, source)
                discovery_tasks.append(task)
        
        # Execute discovery tasks with longer timeout for real APIs
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*discovery_tasks, return_exceptions=True),
                timeout=120.0
            )
            
            # Process results
            for result in results:
                if isinstance(result, TokenDiscoveryResult):
                    discovery_results.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Discovery task failed: {result}")
                    
        except asyncio.TimeoutError:
            logger.warning("Token discovery timeout - using partial results")
        
        # Deduplicate and combine results
        all_tokens = self._deduplicate_tokens(discovery_results)
        
        # Update discovery history
        self.discovery_history.append({
            'timestamp': datetime.now(),
            'total_tokens': len(all_tokens),
            'unique_tokens': len(all_tokens),
            'providers_used': list(set(r.provider for r in discovery_results)),
            'api_calls_total': sum(r.api_calls_used for r in discovery_results)
        })
        
        # Log comprehensive results
        provider_breakdown = {}
        for result in discovery_results:
            provider_name = result.provider.value
            if provider_name not in provider_breakdown:
                provider_breakdown[provider_name] = {'tokens': 0, 'calls': 0}
            provider_breakdown[provider_name]['tokens'] += len(result.tokens)
            provider_breakdown[provider_name]['calls'] += result.api_calls_used
        
        logger.info(f"Intelligent discovery completed: {len(all_tokens)} unique tokens")
        for provider, stats in provider_breakdown.items():
            efficiency = stats['tokens'] / stats['calls'] if stats['calls'] > 0 else 0
            logger.info(f"  {provider}: {stats['tokens']} tokens, {stats['calls']} calls, {efficiency:.1f} tokens/call")
        
        return all_tokens

    async def get_trending_tokens(self, limit: int = 50) -> List[TokenData]:
        """Get trending tokens using intelligent provider selection"""
        provider_priority = self._calculate_provider_priority()
        
        for provider in provider_priority:
            try:
                client = self.solana_tracker if provider == APIProvider.SOLANA_TRACKER else self.geckoterminal
                tokens = await client.get_trending_tokens(limit)
                
                if tokens:
                    self._update_performance_metrics(provider, True, len(tokens), 1.0)
                    logger.info(f"Retrieved {len(tokens)} trending tokens from {provider.value}")
                    return tokens
                    
            except Exception as e:
                self._update_performance_metrics(provider, False, 0, 1.0, str(e))
                logger.warning(f"Trending tokens failed from {provider.value}: {e}")
                continue
        
        logger.error("All providers failed for trending tokens")
        return []

    async def get_volume_tokens(self, limit: int = 50) -> List[TokenData]:
        """Get volume tokens using intelligent provider selection"""
        return await self._get_tokens_with_fallback('get_volume_tokens', limit)

    async def get_memescope_tokens(self, limit: int = 50) -> List[TokenData]:
        """Get memescope tokens using intelligent provider selection"""
        return await self._get_tokens_with_fallback('get_memescope_tokens', limit)

    async def _get_tokens_with_fallback(self, method_name: str, limit: int) -> List[TokenData]:
        """Generic method with intelligent fallback"""
        provider_priority = self._calculate_provider_priority()
        
        for provider in provider_priority:
            try:
                client = self.solana_tracker if provider == APIProvider.SOLANA_TRACKER else self.geckoterminal
                
                if hasattr(client, method_name):
                    method = getattr(client, method_name)
                    tokens = await method(limit)
                    
                    if tokens:
                        self._update_performance_metrics(provider, True, len(tokens), 1.0)
                        logger.info(f"Retrieved {len(tokens)} tokens via {method_name} from {provider.value}")
                        return tokens
                else:
                    # Fallback to volume tokens if method doesn't exist
                    tokens = await client.get_volume_tokens(limit)
                    if tokens:
                        self._update_performance_metrics(provider, True, len(tokens), 1.0)
                        return tokens
                        
            except Exception as e:
                self._update_performance_metrics(provider, False, 0, 1.0, str(e))
                continue
        
        return []

    async def get_all_tokens(self) -> List[TokenData]:
        """Get all tokens using intelligent dual-API discovery"""
        return await self.discover_tokens_intelligently()

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics for both APIs"""
        stats = {
            'dual_api_status': 'active',
            'primary_provider': self.primary_provider.value,
            'fallback_provider': self.fallback_provider.value,
            'providers': {}
        }
        
        total_tokens = 0
        total_calls = 0
        
        for provider, perf in self.performance.items():
            provider_stats = {
                'status': perf.status.value,
                'tokens_discovered': perf.tokens_discovered,
                'quota_used': perf.quota_used,
                'quota_remaining': self.daily_quotas[provider] - perf.quota_used,
                'success_rate': f"{perf.success_rate:.1%}",
                'avg_response_time': f"{perf.response_time_avg:.2f}s",
                'last_success': perf.last_success.isoformat(),
                'efficiency': perf.tokens_discovered / max(1, perf.quota_used)
            }
            
            stats['providers'][provider.value] = provider_stats
            total_tokens += perf.tokens_discovered
            total_calls += perf.quota_used
        
        # Overall statistics
        stats['totals'] = {
            'tokens_discovered': total_tokens,
            'api_calls_used': total_calls,
            'overall_efficiency': total_tokens / max(1, total_calls),
            'discovery_sessions': len(self.discovery_history)
        }
        
        # Performance projections
        if self.discovery_history:
            recent_avg = sum(h['total_tokens'] for h in self.discovery_history[-10:]) / min(10, len(self.discovery_history))
            daily_scans = 96  # Based on 15-minute intervals
            stats['projections'] = {
                'tokens_per_scan': recent_avg,
                'projected_daily_tokens': recent_avg * daily_scans,
                'target_achievement': f"{(recent_avg * daily_scans / 2500) * 100:.1f}%"
            }
        
        return stats

    async def test_connection(self) -> bool:
        """Test connection for both API providers"""
        st_success = await self.solana_tracker.test_connection()
        gecko_success = await self.geckoterminal.test_connection()
        
        logger.info(f"Connection test - Solana Tracker: {st_success}, GeckoTerminal: {gecko_success}")
        
        # Update status based on connection tests
        self.performance[APIProvider.SOLANA_TRACKER].status = APIStatus.HEALTHY if st_success else APIStatus.ERROR
        self.performance[APIProvider.GECKOTERMINAL].status = APIStatus.HEALTHY if gecko_success else APIStatus.ERROR
        
        return st_success or gecko_success  # At least one working

    def reset_daily_quotas(self):
        """Reset daily quota counters (called at start of new day)"""
        for provider in self.performance:
            self.performance[provider].quota_used = 0
            self.performance[provider].tokens_discovered = 0
            
        logger.info("Daily quotas reset for new trading day")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            'discovery_efficiency': {
                provider.value: {
                    'tokens_per_call': perf.tokens_discovered / max(1, perf.quota_used),
                    'success_rate': perf.success_rate,
                    'avg_response_time': perf.response_time_avg,
                    'status': perf.status.value
                }
                for provider, perf in self.performance.items()
            },
            'quota_management': {
                provider.value: {
                    'used': perf.quota_used,
                    'limit': self.daily_quotas[provider],
                    'remaining': self.daily_quotas[provider] - perf.quota_used,
                    'utilization': f"{(perf.quota_used / self.daily_quotas[provider]) * 100:.1f}%"
                }
                for provider, perf in self.performance.items()
            },
            'discovery_history': self.discovery_history[-24:],  # Last 24 sessions
            'optimization_insights': self._get_optimization_insights()
        }

    def _get_optimization_insights(self) -> Dict[str, Any]:
        """Generate optimization insights based on historical data"""
        if len(self.discovery_history) < 5:
            return {'status': 'insufficient_data'}
        
        # Calculate trends
        recent_discoveries = [h['total_tokens'] for h in self.discovery_history[-10:]]
        avg_recent = sum(recent_discoveries) / len(recent_discoveries)
        
        # Provider effectiveness
        provider_effectiveness = {}
        for provider, perf in self.performance.items():
            effectiveness = perf.tokens_discovered / max(1, perf.quota_used)
            provider_effectiveness[provider.value] = effectiveness
        
        return {
            'avg_tokens_per_session': avg_recent,
            'provider_ranking': sorted(provider_effectiveness.items(), key=lambda x: x[1], reverse=True),
            'quota_optimization': self._suggest_quota_optimizations(),
            'performance_trends': {
                'improving': len(recent_discoveries) > 5 and recent_discoveries[-1] > recent_discoveries[0],
                'current_rate': avg_recent,
                'target_rate': 26  # For 2500 tokens/day with 96 scans
            }
        }

    def _suggest_quota_optimizations(self) -> List[str]:
        """Suggest quota usage optimizations"""
        suggestions = []
        
        for provider, perf in self.performance.items():
            utilization = perf.quota_used / self.daily_quotas[provider]
            efficiency = perf.tokens_discovered / max(1, perf.quota_used)
            
            if utilization < 0.5 and efficiency > 20:
                suggestions.append(f"Increase {provider.value} usage - high efficiency, low utilization")
            elif utilization > 0.8 and efficiency < 10:
                suggestions.append(f"Reduce {provider.value} usage - low efficiency, high utilization")
            elif perf.status == APIStatus.QUOTA_EXHAUSTED:
                suggestions.append(f"Implement {provider.value} quota conservation - exhausted too early")
        
        return suggestions