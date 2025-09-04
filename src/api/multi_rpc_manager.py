"""
Multi-RPC Manager with Intelligent Selection

Manages multiple Solana RPC providers with:
- Performance-based selection (latency + success rate)
- Automatic failover and recovery
- Health monitoring and statistics
- Rate limit management

Supports free tier providers:
- Helius (10 req/sec, 500K/month)
- QuickNode (generous free tier)
- Ankr (decentralized, 30M requests/month)
- Solana default (fallback)
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import aiohttp
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Commitment

logger = logging.getLogger(__name__)

@dataclass
class RPCStats:
    """Statistics for an RPC provider"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    blocked_until: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_latency(self) -> float:
        if self.successful_requests == 0:
            return float('inf')
        return self.total_latency / self.successful_requests
    
    @property
    def performance_score(self) -> float:
        """Combined performance score (higher is better)"""
        if self.is_blocked:
            return 0.0
        
        # Score = success_rate * (1 / normalized_latency) * availability_bonus
        latency_score = 1.0 / (1.0 + self.avg_latency)
        availability_bonus = 1.0 if self.consecutive_failures == 0 else 0.5
        
        return self.success_rate * latency_score * availability_bonus
    
    @property
    def is_blocked(self) -> bool:
        return self.blocked_until and datetime.now() < self.blocked_until

@dataclass
class RPCProvider:
    """RPC provider configuration"""
    name: str
    url: str
    max_requests_per_second: float = 10.0
    max_requests_per_day: Optional[int] = None
    priority: int = 1  # Lower = higher priority
    stats: RPCStats = field(default_factory=RPCStats)
    client: Optional[AsyncClient] = None
    last_request_time: float = 0.0

class MultiRPCManager:
    """Intelligent multi-RPC manager with performance-based selection"""
    
    def __init__(self):
        self.providers: Dict[str, RPCProvider] = {}
        self.current_provider: Optional[str] = None
        self.selection_strategy = "performance"  # "performance", "round_robin", "priority"
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = time.time()
        
        # Setup default providers (will be overridden by .env)
        self._setup_default_providers()
    
    def _setup_default_providers(self):
        """Setup default RPC providers with free tiers"""
        default_providers = [
            RPCProvider(
                name="solana_default",
                url="https://api.mainnet-beta.solana.com",
                max_requests_per_second=5.0,
                priority=4  # Lowest priority (fallback)
            ),
            # Note: These will be updated with actual API keys from .env
            RPCProvider(
                name="helius_free", 
                url="https://mainnet.helius-rpc.com/?api-key=YOUR_API_KEY",
                max_requests_per_second=10.0,
                max_requests_per_day=500000,
                priority=1
            ),
            RPCProvider(
                name="quicknode_free",
                url="https://your-endpoint.quicknode.pro/YOUR_API_KEY/",
                max_requests_per_second=15.0,
                priority=2
            ),
            RPCProvider(
                name="ankr_free",
                url="https://rpc.ankr.com/solana",
                max_requests_per_second=8.0,
                max_requests_per_day=30000000,
                priority=3
            )
        ]
        
        for provider in default_providers:
            self.providers[provider.name] = provider
            logger.info(f"[RPC] Registered provider: {provider.name} ({provider.url[:50]}...)")
    
    def update_provider_url(self, provider_name: str, new_url: str):
        """Update provider URL (for adding API keys from .env)"""
        if provider_name in self.providers:
            old_url = self.providers[provider_name].url
            self.providers[provider_name].url = new_url
            logger.info(f"[RPC] Updated {provider_name}: {old_url[:30]}... -> {new_url[:30]}...")
        else:
            logger.warning(f"[RPC] Provider {provider_name} not found for URL update")
    
    def add_provider(self, name: str, url: str, max_rps: float = 10.0, priority: int = 1):
        """Add a custom RPC provider"""
        provider = RPCProvider(
            name=name,
            url=url,
            max_requests_per_second=max_rps,
            priority=priority
        )
        self.providers[name] = provider
        logger.info(f"[RPC] Added custom provider: {name}")
    
    async def get_client(self, provider_name: Optional[str] = None) -> AsyncClient:
        """Get RPC client for specified provider or select best available"""
        if provider_name:
            if provider_name in self.providers:
                return await self._get_client_for_provider(provider_name)
            else:
                logger.warning(f"[RPC] Provider {provider_name} not found, selecting best available")
        
        # Select best provider
        best_provider = self._select_best_provider()
        if not best_provider:
            raise RuntimeError("No healthy RPC providers available")
        
        return await self._get_client_for_provider(best_provider)
    
    async def _get_client_for_provider(self, provider_name: str) -> AsyncClient:
        """Get AsyncClient for specific provider"""
        provider = self.providers[provider_name]
        
        # Rate limiting check
        current_time = time.time()
        time_since_last = current_time - provider.last_request_time
        min_interval = 1.0 / provider.max_requests_per_second
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.debug(f"[RPC] Rate limiting {provider_name}, sleeping {sleep_time:.3f}s")
            await asyncio.sleep(sleep_time)
        
        provider.last_request_time = time.time()
        
        # Create client if needed
        if not provider.client:
            provider.client = AsyncClient(provider.url)
            logger.debug(f"[RPC] Created client for {provider_name}")
        
        self.current_provider = provider_name
        return provider.client
    
    def _select_best_provider(self) -> Optional[str]:
        """Select best provider based on performance and availability"""
        available_providers = [
            (name, provider) for name, provider in self.providers.items()
            if not provider.stats.is_blocked
        ]
        
        if not available_providers:
            # All providers blocked, try oldest blocked one
            logger.warning("[RPC] All providers blocked, trying recovery...")
            return self._try_recovery()
        
        if self.selection_strategy == "performance":
            # Sort by performance score (descending)
            available_providers.sort(key=lambda x: x[1].stats.performance_score, reverse=True)
        elif self.selection_strategy == "priority":
            # Sort by priority (ascending - lower number = higher priority)
            available_providers.sort(key=lambda x: x[1].priority)
        else:  # round_robin
            # Simple round robin (not implemented fully, falls back to performance)
            available_providers.sort(key=lambda x: x[1].stats.performance_score, reverse=True)
        
        selected = available_providers[0][0]
        logger.debug(f"[RPC] Selected provider: {selected} (score: {available_providers[0][1].stats.performance_score:.3f})")
        return selected
    
    def _try_recovery(self) -> Optional[str]:
        """Try to recover from all-blocked state"""
        # Find provider with oldest block time
        oldest_block = None
        oldest_provider = None
        
        for name, provider in self.providers.items():
            if provider.stats.blocked_until:
                if not oldest_block or provider.stats.blocked_until < oldest_block:
                    oldest_block = provider.stats.blocked_until
                    oldest_provider = name
        
        if oldest_provider:
            # Clear block if enough time has passed
            provider = self.providers[oldest_provider]
            if datetime.now() > provider.stats.blocked_until:
                provider.stats.blocked_until = None
                provider.stats.consecutive_failures = 0
                logger.info(f"[RPC] Recovered provider: {oldest_provider}")
                return oldest_provider
            else:
                # Force unblock oldest provider
                provider.stats.blocked_until = None
                provider.stats.consecutive_failures = 0
                logger.warning(f"[RPC] Force recovery of provider: {oldest_provider}")
                return oldest_provider
        
        return list(self.providers.keys())[0]  # Fallback to first provider
    
    async def execute_with_fallback(self, operation, max_retries: int = 3) -> Any:
        """Execute operation with automatic RPC failover"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                provider_name = self._select_best_provider()
                if not provider_name:
                    raise RuntimeError("No healthy RPC providers available")
                
                start_time = time.time()
                client = await self.get_client(provider_name)
                
                # Execute operation
                result = await operation(client)
                
                # Record success
                execution_time = time.time() - start_time
                self._record_success(provider_name, execution_time)
                
                return result
                
            except Exception as e:
                last_error = e
                execution_time = time.time() - start_time if 'start_time' in locals() else 0
                
                if self.current_provider:
                    self._record_failure(self.current_provider, str(e))
                    
                logger.warning(f"[RPC] Attempt {attempt + 1} failed with {self.current_provider}: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 5))  # Exponential backoff
        
        logger.error(f"[RPC] All {max_retries} attempts failed")
        raise last_error
    
    def _record_success(self, provider_name: str, latency: float):
        """Record successful request"""
        provider = self.providers[provider_name]
        stats = provider.stats
        
        stats.total_requests += 1
        stats.successful_requests += 1
        stats.total_latency += latency
        stats.last_success = datetime.now()
        stats.consecutive_failures = 0
        
        # Unblock if was blocked due to temporary issues
        if stats.blocked_until and stats.consecutive_failures == 0:
            stats.blocked_until = None
        
        logger.debug(f"[RPC] Success {provider_name}: {latency:.3f}s (rate: {stats.success_rate:.2%})")
    
    def _record_failure(self, provider_name: str, error: str):
        """Record failed request and potentially block provider"""
        provider = self.providers[provider_name]
        stats = provider.stats
        
        stats.total_requests += 1
        stats.failed_requests += 1
        stats.last_failure = datetime.now()
        stats.consecutive_failures += 1
        
        # Block provider after too many consecutive failures
        if stats.consecutive_failures >= 3:
            block_duration = min(60 * (2 ** (stats.consecutive_failures - 3)), 1800)  # Max 30 minutes
            stats.blocked_until = datetime.now() + timedelta(seconds=block_duration)
            logger.warning(f"[RPC] Blocked {provider_name} for {block_duration}s after {stats.consecutive_failures} failures")
        
        logger.debug(f"[RPC] Failure {provider_name}: {error} (failures: {stats.consecutive_failures})")
    
    async def health_check_all_providers(self):
        """Perform health check on all providers"""
        logger.info("[RPC] Performing health check on all providers...")
        
        async def check_provider(name: str, provider: RPCProvider):
            try:
                client = AsyncClient(provider.url)
                start_time = time.time()
                
                # Simple health check - get slot
                result = await client.get_slot()
                latency = time.time() - start_time
                
                if result.value is not None:
                    logger.info(f"[RPC] Health check OK: {name} ({latency:.3f}s) - Slot: {result.value}")
                    return True
                else:
                    logger.warning(f"[RPC] Health check failed: {name} - No slot returned")
                    return False
                    
            except Exception as e:
                logger.warning(f"[RPC] Health check failed: {name} - {e}")
                return False
            finally:
                if 'client' in locals():
                    await client.close()
        
        # Check all providers concurrently
        tasks = []
        for name, provider in self.providers.items():
            if not provider.stats.is_blocked:
                tasks.append(check_provider(name, provider))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            healthy_count = sum(1 for r in results if r is True)
            logger.info(f"[RPC] Health check complete: {healthy_count}/{len(tasks)} providers healthy")
        
        self.last_health_check = time.time()
    
    async def close_all_clients(self):
        """Close all RPC clients"""
        for provider in self.providers.values():
            if provider.client:
                await provider.client.close()
                provider.client = None
        logger.info("[RPC] All RPC clients closed")
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all providers"""
        stats = {}
        for name, provider in self.providers.items():
            s = provider.stats
            stats[name] = {
                'url': provider.url[:50] + '...' if len(provider.url) > 50 else provider.url,
                'total_requests': s.total_requests,
                'success_rate': f"{s.success_rate:.2%}",
                'avg_latency': f"{s.avg_latency:.3f}s",
                'performance_score': f"{s.performance_score:.3f}",
                'consecutive_failures': s.consecutive_failures,
                'is_blocked': s.is_blocked,
                'last_success': s.last_success.strftime('%H:%M:%S') if s.last_success else 'Never',
                'last_failure': s.last_failure.strftime('%H:%M:%S') if s.last_failure else 'Never'
            }
        return stats
    
    def log_performance_summary(self):
        """Log performance summary for all providers"""
        logger.info("[RPC] === Performance Summary ===")
        stats = self.get_stats()
        
        for name, data in stats.items():
            status = "BLOCKED" if data['is_blocked'] else "ACTIVE"
            logger.info(f"[RPC] {name:15} | {status:7} | Rate: {data['success_rate']:6} | "
                       f"Latency: {data['avg_latency']:8} | Score: {data['performance_score']:5}")
        
        logger.info("[RPC] === End Summary ===")