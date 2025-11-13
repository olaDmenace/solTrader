#!/usr/bin/env python3
"""
UNIFIED DATA MANAGER - Day 12 Implementation
Centralizes all market data coordination, caching, and provider management

This unified data manager consolidates multiple data management systems:
1. SmartDualAPIManager - Intelligent dual-API coordination
2. TokenMetadataCache - Multi-source token metadata management  
3. MultiRPCManager - Performance-based RPC provider selection
4. AdaptiveQuotaManager - Centralized quota coordination
5. Multiple API client caching and rate limiting systems

Key Features:
- Single interface for all market data providers
- Intelligent caching with TTL, LRU eviction, and dependency tracking
- Centralized quota and rate management across all APIs
- Unified health monitoring and failover logic
- Data normalization across all providers
- Event-driven updates with pub/sub pattern
- Performance monitoring and analytics integration
"""

import asyncio
import logging
import time
import json
import os
import sqlite3
import aiosqlite
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import aiohttp
try:
    import redis
except ImportError:
    # Mock Redis for environments without redis module
    class MockRedis:
        def __init__(self, *args, **kwargs): pass
        def get(self, key): return None
        def set(self, key, value, ex=None): pass
        def delete(self, key): pass
        def exists(self, key): return False
        def ping(self): return True
    redis = MockRedis
from cachetools import TTLCache, LRUCache

# Import core components for integration
from core.rpc_manager import MultiRPCManager
from api.data_provider import SmartDualAPIManager
from cache.token_metadata_cache import TokenMetadataCache
from api.adaptive_quota_manager import AdaptiveQuotaManager

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Data source enumeration"""
    SOLANA_TRACKER = "solana_tracker"
    GECKO_TERMINAL = "gecko_terminal"
    JUPITER = "jupiter"
    BIRDEYE = "birdeye"
    ALCHEMY = "alchemy"
    HELIUS = "helius"
    QUICKNODE = "quicknode"

class DataType(Enum):
    """Data type enumeration"""
    TOKEN_METADATA = "token_metadata"
    PRICE_DATA = "price_data" 
    VOLUME_DATA = "volume_data"
    LIQUIDITY_DATA = "liquidity_data"
    MARKET_DEPTH = "market_depth"
    TRENDING_TOKENS = "trending_tokens"
    TOKEN_DISCOVERY = "token_discovery"
    TRANSACTION_DATA = "transaction_data"
    BALANCE_DATA = "balance_data"
    RPC_DATA = "rpc_data"

class CacheLevel(Enum):
    """Cache level hierarchy"""
    MEMORY = "memory"          # Fastest, smallest
    REDIS = "redis"            # Fast, shared
    SQLITE = "sqlite"          # Persistent, local
    FILE = "file"              # Backup, human-readable

@dataclass
class DataRequest:
    """Standardized data request structure"""
    data_type: DataType
    token_address: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    cache_ttl: int = 300  # 5 minutes default
    priority: int = 1  # 1=high, 2=medium, 3=low
    source_preference: Optional[List[DataSource]] = None
    cache_levels: List[CacheLevel] = field(default_factory=lambda: [CacheLevel.MEMORY, CacheLevel.SQLITE])
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataResponse:
    """Standardized data response structure"""
    request: DataRequest
    data: Any
    source: DataSource
    timestamp: datetime
    cache_hit: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProviderStatus:
    """Data provider health status"""
    source: DataSource
    available: bool = True
    latency_ms: float = 0.0
    success_rate: float = 1.0
    quota_used: int = 0
    quota_limit: int = 0
    last_error: Optional[str] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0

class UnifiedDataManager:
    """Comprehensive data manager with unified coordination"""
    
    def __init__(self, settings: Any):
        self.settings = settings
        
        # Core component integration
        self.rpc_manager: Optional[MultiRPCManager] = None
        self.api_coordinator: Optional[SmartDualAPIManager] = None
        self.token_cache: Optional[TokenMetadataCache] = None
        self.quota_manager: Optional[AdaptiveQuotaManager] = None
        
        # Unified caching system
        self.memory_cache: TTLCache = TTLCache(maxsize=10000, ttl=300)  # 5 min TTL
        self.lru_cache: LRUCache = LRUCache(maxsize=50000)
        self.redis_client: Optional[redis.Redis] = None
        
        # Provider management
        self.providers: Dict[DataSource, ProviderStatus] = {}
        self.provider_clients: Dict[DataSource, Any] = {}
        self.failover_rules: Dict[DataSource, List[DataSource]] = {}
        
        # Performance tracking
        self.request_metrics: Dict[str, Any] = defaultdict(lambda: {
            'count': 0, 'success_count': 0, 'total_latency': 0.0, 'cache_hits': 0
        })
        
        # Event system for data updates
        self.subscribers: Dict[DataType, List[callable]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        
        # Database for persistent storage
        self.db_path = "data/unified_data_manager.db"
        self.db_initialized = False
        
        # Health monitoring
        self.health_checks: Dict[str, datetime] = {}
        self.circuit_breakers: Dict[DataSource, bool] = {}
        
        logger.info("[UNIFIED_DATA] Data manager configuration loaded")
    
    async def initialize(self):
        """Initialize the unified data manager"""
        try:
            logger.info("[UNIFIED_DATA] Initializing Unified Data Manager...")
            
            # Initialize database
            await self._initialize_database()
            
            # Initialize core components
            await self._initialize_providers()
            
            # Setup caching layers
            await self._initialize_caching()
            
            # Configure failover rules
            self._configure_failover_rules()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("[UNIFIED_DATA] Unified Data Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"[UNIFIED_DATA] Data manager initialization failed: {e}")
            raise
    
    async def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            async with aiosqlite.connect(self.db_path) as db:
                # Data cache table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS data_cache (
                        cache_key TEXT PRIMARY KEY,
                        data_type TEXT NOT NULL,
                        token_address TEXT,
                        data BLOB NOT NULL,
                        source TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        last_accessed REAL NOT NULL
                    )
                """)
                
                # Provider metrics table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS provider_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        latency_ms REAL NOT NULL,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        quota_used INTEGER,
                        data_type TEXT
                    )
                """)
                
                # Data request history
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS request_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_hash TEXT NOT NULL,
                        data_type TEXT NOT NULL,
                        token_address TEXT,
                        timestamp REAL NOT NULL,
                        source TEXT NOT NULL,
                        cache_hit BOOLEAN NOT NULL,
                        latency_ms REAL NOT NULL,
                        success BOOLEAN NOT NULL
                    )
                """)
                
                await db.commit()
                
            self.db_initialized = True
            logger.info("[UNIFIED_DATA] Database initialized successfully")
            
        except Exception as e:
            logger.error(f"[UNIFIED_DATA] Database initialization failed: {e}")
            raise
    
    async def _initialize_providers(self):
        """Initialize all data providers"""
        try:
            # Initialize RPC manager
            self.rpc_manager = MultiRPCManager(self.settings)
            await self.rpc_manager.initialize()
            
            # Initialize API coordinator  
            self.api_coordinator = SmartDualAPIManager(self.settings)
            await self.api_coordinator.initialize()
            
            # Initialize token cache
            self.token_cache = TokenMetadataCache(self.settings)
            await self.token_cache.initialize()
            
            # Initialize quota manager
            self.quota_manager = AdaptiveQuotaManager(self.settings)
            await self.quota_manager.initialize()
            
            # Initialize provider status tracking
            for source in DataSource:
                self.providers[source] = ProviderStatus(
                    source=source,
                    available=True,
                    last_success=datetime.now()
                )
                self.circuit_breakers[source] = False
            
            logger.info("[UNIFIED_DATA] All data providers initialized")
            
        except Exception as e:
            logger.error(f"[UNIFIED_DATA] Provider initialization failed: {e}")
            raise
    
    async def _initialize_caching(self):
        """Initialize multi-level caching system"""
        try:
            # Try to initialize Redis for shared caching
            try:
                import redis.asyncio as redis_async
                self.redis_client = redis_async.Redis(
                    host=getattr(self.settings, 'REDIS_HOST', 'localhost'),
                    port=getattr(self.settings, 'REDIS_PORT', 6379),
                    decode_responses=True,
                    socket_timeout=2.0
                )
                await self.redis_client.ping()
                logger.info("[UNIFIED_DATA] Redis cache initialized")
            except Exception:
                logger.warning("[UNIFIED_DATA] Redis unavailable, using local cache only")
                self.redis_client = None
            
            # Initialize file-based cache backup
            self.file_cache_dir = Path("data/file_cache")
            self.file_cache_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("[UNIFIED_DATA] Multi-level caching initialized")
            
        except Exception as e:
            logger.error(f"[UNIFIED_DATA] Caching initialization failed: {e}")
            # Continue without advanced caching
    
    def _configure_failover_rules(self):
        """Configure provider failover rules"""
        self.failover_rules = {
            DataSource.SOLANA_TRACKER: [DataSource.GECKO_TERMINAL, DataSource.BIRDEYE],
            DataSource.GECKO_TERMINAL: [DataSource.SOLANA_TRACKER, DataSource.BIRDEYE],
            DataSource.JUPITER: [DataSource.BIRDEYE, DataSource.SOLANA_TRACKER],
            DataSource.BIRDEYE: [DataSource.JUPITER, DataSource.GECKO_TERMINAL],
            DataSource.ALCHEMY: [DataSource.HELIUS, DataSource.QUICKNODE],
            DataSource.HELIUS: [DataSource.ALCHEMY, DataSource.QUICKNODE],
            DataSource.QUICKNODE: [DataSource.ALCHEMY, DataSource.HELIUS]
        }
        logger.info("[UNIFIED_DATA] Failover rules configured")
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        # Health monitoring task
        asyncio.create_task(self._health_monitor_loop())
        
        # Cache maintenance task  
        asyncio.create_task(self._cache_maintenance_loop())
        
        # Metrics collection task
        asyncio.create_task(self._metrics_collection_loop())
        
        # Event processing task
        asyncio.create_task(self._event_processing_loop())
        
        logger.info("[UNIFIED_DATA] Background tasks started")
    
    async def get_data(self, request: DataRequest) -> DataResponse:
        """Main interface for data retrieval with unified coordination"""
        start_time = time.time()
        request_hash = self._generate_request_hash(request)
        
        try:
            # Check cache hierarchy
            cached_response = await self._check_cache(request, request_hash)
            if cached_response:
                latency_ms = (time.time() - start_time) * 1000
                await self._record_request(request_hash, request, True, latency_ms, True)
                return cached_response
            
            # Determine optimal data source
            optimal_source = await self._select_optimal_source(request)
            
            # Fetch data from source with failover
            response = await self._fetch_with_failover(request, optimal_source)
            
            # Cache the response
            await self._cache_response(request, response, request_hash)
            
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            await self._record_request(request_hash, request, False, latency_ms, True)
            
            # Publish data update event
            await self._publish_data_event(request.data_type, response)
            
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            await self._record_request(request_hash, request, False, latency_ms, False)
            
            # Return error response
            return DataResponse(
                request=request,
                data=None,
                source=DataSource.SOLANA_TRACKER,  # Default
                timestamp=datetime.now(),
                error=str(e),
                latency_ms=latency_ms
            )
    
    async def _check_cache(self, request: DataRequest, request_hash: str) -> Optional[DataResponse]:
        """Check cache hierarchy for existing data"""
        try:
            # Level 1: Memory cache
            if CacheLevel.MEMORY in request.cache_levels:
                if request_hash in self.memory_cache:
                    data = self.memory_cache[request_hash]
                    if self._is_cache_valid(data, request.cache_ttl):
                        self.request_metrics[request_hash]['cache_hits'] += 1
                        return data
            
            # Level 2: Redis cache  
            if CacheLevel.REDIS in request.cache_levels and self.redis_client:
                try:
                    cached_data = await self.redis_client.get(f"data:{request_hash}")
                    if cached_data:
                        data = json.loads(cached_data)
                        if self._is_cache_valid(data, request.cache_ttl):
                            # Promote to memory cache
                            self.memory_cache[request_hash] = data
                            return DataResponse(**data)
                except Exception:
                    pass  # Redis unavailable, continue
            
            # Level 3: SQLite cache
            if CacheLevel.SQLITE in request.cache_levels and self.db_initialized:
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute(
                        "SELECT data, timestamp FROM data_cache WHERE cache_key = ? AND expires_at > ?",
                        (request_hash, time.time())
                    )
                    row = await cursor.fetchone()
                    if row:
                        data_blob, timestamp = row
                        data = json.loads(data_blob)
                        response = DataResponse(**data)
                        
                        # Promote to higher cache levels
                        self.memory_cache[request_hash] = response
                        if self.redis_client:
                            await self.redis_client.setex(
                                f"data:{request_hash}", 
                                request.cache_ttl,
                                json.dumps(asdict(response), default=str)
                            )
                        
                        return response
            
        except Exception as e:
            logger.warning(f"[UNIFIED_DATA] Cache check failed: {e}")
        
        return None
    
    def _is_cache_valid(self, data: Any, max_age: int) -> bool:
        """Check if cached data is still valid"""
        try:
            if hasattr(data, 'timestamp'):
                timestamp = data.timestamp
            elif isinstance(data, dict) and 'timestamp' in data:
                timestamp = datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp']
            else:
                return False
            
            age_seconds = (datetime.now() - timestamp).total_seconds()
            return age_seconds < max_age
        except Exception:
            return False
    
    async def _select_optimal_source(self, request: DataRequest) -> DataSource:
        """Select optimal data source based on performance and availability"""
        try:
            # Use preference if specified
            if request.source_preference:
                for preferred_source in request.source_preference:
                    if (self.providers[preferred_source].available and 
                        not self.circuit_breakers.get(preferred_source, False)):
                        return preferred_source
            
            # Select based on data type and provider performance
            suitable_sources = self._get_suitable_sources(request.data_type)
            
            # Score sources based on performance
            source_scores = {}
            for source in suitable_sources:
                provider = self.providers[source]
                if not provider.available or self.circuit_breakers.get(source, False):
                    continue
                    
                # Calculate score (lower is better)
                score = (
                    provider.latency_ms * 0.4 +
                    (1 - provider.success_rate) * 1000 * 0.4 +
                    (provider.quota_used / max(provider.quota_limit, 1)) * 100 * 0.2
                )
                source_scores[source] = score
            
            if not source_scores:
                # All sources unavailable, return first suitable source
                return suitable_sources[0] if suitable_sources else DataSource.SOLANA_TRACKER
            
            # Return source with best score
            return min(source_scores.items(), key=lambda x: x[1])[0]
            
        except Exception as e:
            logger.warning(f"[UNIFIED_DATA] Source selection failed: {e}")
            return DataSource.SOLANA_TRACKER
    
    def _get_suitable_sources(self, data_type: DataType) -> List[DataSource]:
        """Get suitable data sources for given data type"""
        source_mapping = {
            DataType.TOKEN_METADATA: [DataSource.JUPITER, DataSource.BIRDEYE, DataSource.SOLANA_TRACKER],
            DataType.PRICE_DATA: [DataSource.GECKO_TERMINAL, DataSource.BIRDEYE, DataSource.SOLANA_TRACKER],
            DataType.VOLUME_DATA: [DataSource.GECKO_TERMINAL, DataSource.BIRDEYE],
            DataType.LIQUIDITY_DATA: [DataSource.BIRDEYE, DataSource.JUPITER],
            DataType.TRENDING_TOKENS: [DataSource.SOLANA_TRACKER, DataSource.GECKO_TERMINAL],
            DataType.TOKEN_DISCOVERY: [DataSource.SOLANA_TRACKER, DataSource.GECKO_TERMINAL],
            DataType.RPC_DATA: [DataSource.ALCHEMY, DataSource.HELIUS, DataSource.QUICKNODE],
            DataType.BALANCE_DATA: [DataSource.ALCHEMY, DataSource.HELIUS],
            DataType.TRANSACTION_DATA: [DataSource.ALCHEMY, DataSource.HELIUS]
        }
        
        return source_mapping.get(data_type, [DataSource.SOLANA_TRACKER])
    
    async def _fetch_with_failover(self, request: DataRequest, primary_source: DataSource) -> DataResponse:
        """Fetch data with automatic failover"""
        sources_to_try = [primary_source] + self.failover_rules.get(primary_source, [])
        
        last_error = None
        for source in sources_to_try:
            if self.circuit_breakers.get(source, False):
                continue
                
            try:
                response = await self._fetch_from_source(request, source)
                if response.error is None:
                    # Update provider status on success
                    await self._update_provider_status(source, True, response.latency_ms)
                    return response
                else:
                    last_error = response.error
                    
            except Exception as e:
                last_error = str(e)
                await self._update_provider_status(source, False, 0.0, str(e))
                
                # Activate circuit breaker if too many failures
                provider = self.providers[source]
                provider.consecutive_failures += 1
                if provider.consecutive_failures >= 3:
                    self.circuit_breakers[source] = True
                    logger.warning(f"[UNIFIED_DATA] Circuit breaker activated for {source}")
        
        # All sources failed
        raise Exception(f"All data sources failed. Last error: {last_error}")
    
    async def _fetch_from_source(self, request: DataRequest, source: DataSource) -> DataResponse:
        """Fetch data from specific source"""
        start_time = time.time()
        
        try:
            # Route to appropriate provider based on source and data type
            if source in [DataSource.ALCHEMY, DataSource.HELIUS, DataSource.QUICKNODE]:
                data = await self._fetch_rpc_data(request, source)
            elif source == DataSource.SOLANA_TRACKER:
                data = await self._fetch_solana_tracker_data(request)
            elif source == DataSource.GECKO_TERMINAL:
                data = await self._fetch_gecko_terminal_data(request)
            elif source == DataSource.JUPITER:
                data = await self._fetch_jupiter_data(request)
            elif source == DataSource.BIRDEYE:
                data = await self._fetch_birdeye_data(request)
            else:
                raise ValueError(f"Unsupported data source: {source}")
            
            latency_ms = (time.time() - start_time) * 1000
            
            return DataResponse(
                request=request,
                data=data,
                source=source,
                timestamp=datetime.now(),
                cache_hit=False,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return DataResponse(
                request=request,
                data=None,
                source=source,
                timestamp=datetime.now(),
                cache_hit=False,
                latency_ms=latency_ms,
                error=str(e)
            )
    
    async def _fetch_rpc_data(self, request: DataRequest, source: DataSource) -> Any:
        """Fetch RPC data using MultiRPCManager"""
        if not self.rpc_manager:
            raise Exception("RPC manager not initialized")
        
        # Map request to RPC call
        if request.data_type == DataType.BALANCE_DATA:
            return await self.rpc_manager.get_balance(request.token_address)
        elif request.data_type == DataType.TOKEN_METADATA:
            return await self.rpc_manager.get_token_info(request.token_address)
        elif request.data_type == DataType.TRANSACTION_DATA:
            return await self.rpc_manager.get_transaction_history(request.token_address)
        else:
            raise ValueError(f"Unsupported RPC data type: {request.data_type}")
    
    async def _fetch_solana_tracker_data(self, request: DataRequest) -> Any:
        """Fetch data from Solana Tracker using API coordinator"""
        if not self.api_coordinator:
            raise Exception("API coordinator not initialized")
        
        # Map request to API coordinator call
        if request.data_type == DataType.TRENDING_TOKENS:
            return await self.api_coordinator.get_trending_tokens()
        elif request.data_type == DataType.TOKEN_DISCOVERY:
            return await self.api_coordinator.discover_tokens(**request.parameters)
        elif request.data_type == DataType.PRICE_DATA:
            return await self.api_coordinator.get_token_price(request.token_address)
        else:
            raise ValueError(f"Unsupported Solana Tracker data type: {request.data_type}")
    
    async def _fetch_gecko_terminal_data(self, request: DataRequest) -> Any:
        """Fetch data from GeckoTerminal using API coordinator"""
        if not self.api_coordinator:
            raise Exception("API coordinator not initialized")
        
        # Use API coordinator's GeckoTerminal integration
        return await self.api_coordinator.get_gecko_terminal_data(request.data_type, request.token_address, request.parameters)
    
    async def _fetch_jupiter_data(self, request: DataRequest) -> Any:
        """Fetch data from Jupiter API"""
        if request.data_type == DataType.TOKEN_METADATA and self.token_cache:
            return await self.token_cache.get_jupiter_metadata(request.token_address)
        else:
            raise ValueError(f"Unsupported Jupiter data type: {request.data_type}")
    
    async def _fetch_birdeye_data(self, request: DataRequest) -> Any:
        """Fetch data from Birdeye API"""
        if request.data_type == DataType.TOKEN_METADATA and self.token_cache:
            return await self.token_cache.get_birdeye_metadata(request.token_address)
        else:
            raise ValueError(f"Unsupported Birdeye data type: {request.data_type}")
    
    async def _cache_response(self, request: DataRequest, response: DataResponse, request_hash: str):
        """Cache response across all cache levels"""
        try:
            # Memory cache
            if CacheLevel.MEMORY in request.cache_levels:
                self.memory_cache[request_hash] = response
            
            # Redis cache
            if CacheLevel.REDIS in request.cache_levels and self.redis_client:
                try:
                    await self.redis_client.setex(
                        f"data:{request_hash}",
                        request.cache_ttl,
                        json.dumps(asdict(response), default=str)
                    )
                except Exception:
                    pass  # Redis unavailable
            
            # SQLite cache
            if CacheLevel.SQLITE in request.cache_levels and self.db_initialized:
                try:
                    async with aiosqlite.connect(self.db_path) as db:
                        await db.execute("""
                            INSERT OR REPLACE INTO data_cache 
                            (cache_key, data_type, token_address, data, source, timestamp, expires_at, last_accessed)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            request_hash,
                            request.data_type.value,
                            request.token_address,
                            json.dumps(asdict(response), default=str),
                            response.source.value,
                            response.timestamp.timestamp(),
                            time.time() + request.cache_ttl,
                            time.time()
                        ))
                        await db.commit()
                except Exception as e:
                    logger.warning(f"[UNIFIED_DATA] SQLite cache write failed: {e}")
            
        except Exception as e:
            logger.warning(f"[UNIFIED_DATA] Cache write failed: {e}")
    
    def _generate_request_hash(self, request: DataRequest) -> str:
        """Generate unique hash for request caching"""
        hash_input = f"{request.data_type.value}:{request.token_address}:{json.dumps(request.parameters, sort_keys=True)}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    async def _update_provider_status(self, source: DataSource, success: bool, latency_ms: float, error: str = None):
        """Update provider performance status"""
        try:
            provider = self.providers[source]
            
            if success:
                provider.last_success = datetime.now()
                provider.consecutive_failures = 0
                provider.available = True
                
                # Reset circuit breaker
                self.circuit_breakers[source] = False
                
                # Update success rate (exponential moving average)
                provider.success_rate = provider.success_rate * 0.9 + 0.1
                
                # Update latency (exponential moving average)
                provider.latency_ms = provider.latency_ms * 0.8 + latency_ms * 0.2
                
            else:
                provider.last_error = error
                provider.consecutive_failures += 1
                
                # Update success rate
                provider.success_rate = provider.success_rate * 0.9
                
                # Deactivate if too many failures
                if provider.consecutive_failures >= 5:
                    provider.available = False
            
            # Record metrics to database
            if self.db_initialized:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        INSERT INTO provider_metrics 
                        (source, timestamp, latency_ms, success, error_message)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        source.value,
                        time.time(),
                        latency_ms,
                        success,
                        error
                    ))
                    await db.commit()
                    
        except Exception as e:
            logger.warning(f"[UNIFIED_DATA] Provider status update failed: {e}")
    
    async def _record_request(self, request_hash: str, request: DataRequest, cache_hit: bool, latency_ms: float, success: bool):
        """Record request metrics and history"""
        try:
            # Update memory metrics
            metrics = self.request_metrics[request_hash]
            metrics['count'] += 1
            if success:
                metrics['success_count'] += 1
            metrics['total_latency'] += latency_ms
            if cache_hit:
                metrics['cache_hits'] += 1
            
            # Record to database
            if self.db_initialized:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        INSERT INTO request_history 
                        (request_hash, data_type, token_address, timestamp, source, cache_hit, latency_ms, success)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        request_hash,
                        request.data_type.value,
                        request.token_address,
                        time.time(),
                        request.source_preference[0].value if request.source_preference else 'auto',
                        cache_hit,
                        latency_ms,
                        success
                    ))
                    await db.commit()
                    
        except Exception as e:
            logger.warning(f"[UNIFIED_DATA] Request recording failed: {e}")
    
    async def _publish_data_event(self, data_type: DataType, response: DataResponse):
        """Publish data update event to subscribers"""
        try:
            if data_type in self.subscribers:
                event = {
                    'type': 'data_update',
                    'data_type': data_type,
                    'response': response,
                    'timestamp': datetime.now()
                }
                
                # Notify all subscribers
                for callback in self.subscribers[data_type]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                    except Exception as e:
                        logger.warning(f"[UNIFIED_DATA] Subscriber notification failed: {e}")
                        
        except Exception as e:
            logger.warning(f"[UNIFIED_DATA] Event publishing failed: {e}")
    
    def subscribe(self, data_type: DataType, callback: callable):
        """Subscribe to data updates"""
        self.subscribers[data_type].append(callback)
        logger.info(f"[UNIFIED_DATA] Subscribed to {data_type} updates")
    
    def unsubscribe(self, data_type: DataType, callback: callable):
        """Unsubscribe from data updates"""
        if callback in self.subscribers[data_type]:
            self.subscribers[data_type].remove(callback)
            logger.info(f"[UNIFIED_DATA] Unsubscribed from {data_type} updates")
    
    async def _health_monitor_loop(self):
        """Background health monitoring"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check provider health
                for source in DataSource:
                    provider = self.providers[source]
                    
                    # Reset circuit breaker if enough time has passed
                    if (self.circuit_breakers.get(source, False) and 
                        provider.last_success and
                        (datetime.now() - provider.last_success).seconds > 300):  # 5 minutes
                        
                        self.circuit_breakers[source] = False
                        provider.consecutive_failures = 0
                        logger.info(f"[UNIFIED_DATA] Circuit breaker reset for {source}")
                
                # Update health check timestamp
                self.health_checks['provider_monitor'] = datetime.now()
                
            except Exception as e:
                logger.error(f"[UNIFIED_DATA] Health monitor error: {e}")
    
    async def _cache_maintenance_loop(self):
        """Background cache maintenance"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean expired SQLite cache entries
                if self.db_initialized:
                    async with aiosqlite.connect(self.db_path) as db:
                        await db.execute("DELETE FROM data_cache WHERE expires_at < ?", (time.time(),))
                        
                        # Clean old metrics (keep last 7 days)
                        cutoff = time.time() - (7 * 24 * 3600)
                        await db.execute("DELETE FROM provider_metrics WHERE timestamp < ?", (cutoff,))
                        await db.execute("DELETE FROM request_history WHERE timestamp < ?", (cutoff,))
                        
                        await db.commit()
                
                # Clean memory cache (TTL cache does this automatically)
                # Clean LRU cache by accessing it (triggers cleanup)
                list(self.lru_cache.keys())[:1]
                
                self.health_checks['cache_maintenance'] = datetime.now()
                logger.info("[UNIFIED_DATA] Cache maintenance completed")
                
            except Exception as e:
                logger.error(f"[UNIFIED_DATA] Cache maintenance error: {e}")
    
    async def _metrics_collection_loop(self):
        """Background metrics collection"""
        while True:
            try:
                await asyncio.sleep(300)  # Collect every 5 minutes
                
                # Calculate and log performance metrics
                total_requests = sum(m['count'] for m in self.request_metrics.values())
                total_success = sum(m['success_count'] for m in self.request_metrics.values())
                total_cache_hits = sum(m['cache_hits'] for m in self.request_metrics.values())
                
                if total_requests > 0:
                    success_rate = (total_success / total_requests) * 100
                    cache_hit_rate = (total_cache_hits / total_requests) * 100
                    
                    logger.info(f"[UNIFIED_DATA] Metrics - Requests: {total_requests}, "
                              f"Success Rate: {success_rate:.1f}%, Cache Hit Rate: {cache_hit_rate:.1f}%")
                
                self.health_checks['metrics_collection'] = datetime.now()
                
            except Exception as e:
                logger.error(f"[UNIFIED_DATA] Metrics collection error: {e}")
    
    async def _event_processing_loop(self):
        """Background event processing"""
        while True:
            try:
                # Process events from queue
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=10.0)
                    # Process event (placeholder for future event handling)
                    logger.debug(f"[UNIFIED_DATA] Processed event: {event}")
                except asyncio.TimeoutError:
                    pass  # No events to process
                
                self.health_checks['event_processing'] = datetime.now()
                
            except Exception as e:
                logger.error(f"[UNIFIED_DATA] Event processing error: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        try:
            # Provider health
            provider_health = {
                source.value: {
                    'available': provider.available,
                    'success_rate': provider.success_rate,
                    'latency_ms': provider.latency_ms,
                    'consecutive_failures': provider.consecutive_failures,
                    'circuit_breaker_active': self.circuit_breakers.get(source, False)
                }
                for source, provider in self.providers.items()
            }
            
            # Cache statistics
            cache_stats = {
                'memory_cache_size': len(self.memory_cache),
                'memory_cache_maxsize': self.memory_cache.maxsize,
                'lru_cache_size': len(self.lru_cache),
                'redis_available': self.redis_client is not None,
                'sqlite_available': self.db_initialized
            }
            
            # Request metrics
            total_requests = sum(m['count'] for m in self.request_metrics.values())
            total_success = sum(m['success_count'] for m in self.request_metrics.values())
            total_cache_hits = sum(m['cache_hits'] for m in self.request_metrics.values())
            
            request_stats = {
                'total_requests': total_requests,
                'success_rate': (total_success / max(total_requests, 1)) * 100,
                'cache_hit_rate': (total_cache_hits / max(total_requests, 1)) * 100,
                'active_subscribers': sum(len(subs) for subs in self.subscribers.values())
            }
            
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'providers': provider_health,
                'cache': cache_stats,
                'requests': request_stats,
                'health_checks': {k: v.isoformat() for k, v in self.health_checks.items()}
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def shutdown(self):
        """Graceful shutdown of data manager"""
        try:
            logger.info("[UNIFIED_DATA] Shutting down Unified Data Manager...")
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            # Shutdown integrated components
            if self.rpc_manager:
                await self.rpc_manager.shutdown()
            
            if self.api_coordinator:
                await self.api_coordinator.shutdown()
            
            if self.token_cache:
                await self.token_cache.shutdown()
            
            if self.quota_manager:
                await self.quota_manager.shutdown()
            
            # Save final metrics if database available
            if self.db_initialized:
                # Save final cache state
                pass  # Already persistent
            
            logger.info("[UNIFIED_DATA] Unified Data Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"[UNIFIED_DATA] Shutdown error: {e}")


# Convenience functions for common data operations
async def get_token_price(data_manager: UnifiedDataManager, token_address: str) -> Optional[float]:
    """Get current token price"""
    request = DataRequest(
        data_type=DataType.PRICE_DATA,
        token_address=token_address,
        cache_ttl=60  # 1 minute cache
    )
    
    response = await data_manager.get_data(request)
    if response.error is None and response.data:
        return response.data.get('price')
    return None

async def get_trending_tokens(data_manager: UnifiedDataManager, limit: int = 20) -> Optional[List[Dict]]:
    """Get trending tokens list"""
    request = DataRequest(
        data_type=DataType.TRENDING_TOKENS,
        parameters={'limit': limit},
        cache_ttl=300  # 5 minute cache
    )
    
    response = await data_manager.get_data(request)
    if response.error is None and response.data:
        return response.data.get('tokens', [])
    return None

async def get_token_metadata(data_manager: UnifiedDataManager, token_address: str) -> Optional[Dict]:
    """Get comprehensive token metadata"""
    request = DataRequest(
        data_type=DataType.TOKEN_METADATA,
        token_address=token_address,
        cache_ttl=3600,  # 1 hour cache for metadata
        source_preference=[DataSource.JUPITER, DataSource.BIRDEYE]
    )
    
    response = await data_manager.get_data(request)
    if response.error is None and response.data:
        return response.data
    return None