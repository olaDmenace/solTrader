# # src/utils/network_simulator.py

# import asyncio
# import random
# from datetime import datetime, timedelta
# from typing import Optional, Dict, Any, List
# import aiohttp
# from dataclasses import dataclass

# @dataclass
# class NetworkCondition:
#     latency: float  # Base latency in seconds
#     jitter: float   # Random latency variation
#     packet_loss: float  # Probability of packet loss
#     error_rate: float   # Probability of errors
#     bandwidth: float    # Simulated bandwidth in MB/s
#     timeout: float     # Connection timeout in seconds

# class NetworkError(Exception):
#     pass

# class NetworkSimulator:
#     def __init__(self):
#         self.conditions = NetworkCondition(
#             latency=0.0,
#             jitter=0.0,
#             packet_loss=0.0,
#             error_rate=0.0,
#             bandwidth=float('inf'),
#             timeout=30.0
#         )
#         self.error_types = [
#             aiohttp.ClientError("Connection refused"),
#             aiohttp.ServerTimeoutError("Request timeout"),
#             aiohttp.ClientPayloadError("Corrupted response"),
#             aiohttp.ServerDisconnectedError("Server disconnected"),
#             aiohttp.ClientOSError("Connection reset"),
#             asyncio.TimeoutError("Operation timeout")
#         ]
#         self._start_time = datetime.now()
#         self._transferred_bytes = 0
#         self._active_connections = 0
#         self._max_connections = 10

#     async def simulate_request(self, size_bytes: int = 1000) -> None:
#         """Simulate a network request with current conditions"""
#         await self._check_connection_limit()
#         await self._simulate_latency()
#         await self._simulate_bandwidth(size_bytes)
#         self._simulate_errors()
#         self._update_metrics(size_bytes)

#     async def _check_connection_limit(self) -> None:
#         """Check if new connection can be established"""
#         if self._active_connections >= self._max_connections:
#             raise NetworkError("Connection limit exceeded")
#         self._active_connections += 1

#     async def _simulate_latency(self) -> None:
#         """Simulate network latency with jitter"""
#         if self.conditions.latency > 0:
#             jitter = random.uniform(-self.conditions.jitter, self.conditions.jitter)
#             total_latency = max(0, self.conditions.latency + jitter)
#             await asyncio.sleep(total_latency)

#     async def _simulate_bandwidth(self, size_bytes: int) -> None:
#         """Simulate bandwidth limitations"""
#         if self.conditions.bandwidth != float('inf'):
#             transfer_time = size_bytes / (self.conditions.bandwidth * 1024 * 1024)
#             await asyncio.sleep(transfer_time)

#     def _simulate_errors(self) -> None:
#         """Simulate various network errors"""
#         if random.random() < self.conditions.packet_loss:
#             raise NetworkError("Packet loss")
            
#         if random.random() < self.conditions.error_rate:
#             raise random.choice(self.error_types)

#     def _update_metrics(self, size_bytes: int) -> None:
#         """Update internal metrics"""
#         self._transferred_bytes += size_bytes
#         self._active_connections -= 1

#     def get_metrics(self) -> Dict[str, Any]:
#         """Get current network metrics"""
#         uptime = (datetime.now() - self._start_time).total_seconds()
#         return {
#             'uptime': uptime,
#             'throughput': self._transferred_bytes / uptime if uptime > 0 else 0,
#             'active_connections': self._active_connections,
#             'total_transferred': self._transferred_bytes
#         }

#     @classmethod
#     def create_profile(cls, profile: str) -> 'NetworkSimulator':
#         """Create simulator with predefined network profile"""
#         simulator = cls()
#         profiles = {
#             'perfect': NetworkCondition(0, 0, 0, 0, float('inf'), 30),
#             'good': NetworkCondition(0.05, 0.01, 0.001, 0.01, 10, 10),
#             'poor': NetworkCondition(0.2, 0.05, 0.01, 0.05, 1, 5),
#             'terrible': NetworkCondition(1.0, 0.2, 0.05, 0.1, 0.1, 3),
#             'mobile': NetworkCondition(0.1, 0.05, 0.02, 0.03, 2, 15),
#             'satellite': NetworkCondition(0.5, 0.1, 0.03, 0.02, 0.5, 20)
#         }
#         simulator.conditions = profiles.get(profile, profiles['good'])
#         return simulator


"""
Enhanced network simulator with RPC diversification and realistic condition simulation.
Provides support for MEV protection and private transaction handling.
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import aiohttp
from aiohttp import TCPConnector
import logging
import backoff
from typing import Union, Callable, Awaitable

logger = logging.getLogger(__name__)

@dataclass
class NetworkCondition:
    """Network condition configuration"""
    latency: float      # Base latency in seconds
    jitter: float      # Random latency variation
    packet_loss: float # Probability of packet loss
    error_rate: float  # Probability of errors
    bandwidth: float   # Simulated bandwidth in MB/s
    timeout: float     # Connection timeout in seconds

@dataclass
class NetworkStats:
    """Network performance metrics"""
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    avg_latency: float = 0.0
    last_error: Optional[str] = None
    last_success: Optional[datetime] = None
    bytes_transferred: int = 0

@dataclass
class RPCEndpoint:
    """RPC endpoint configuration and state"""
    url: str
    priority: int
    is_private: bool = False
    weight: int = 100
    error_count: int = 0
    success_count: int = 0
    last_used: Optional[datetime] = None
    last_error: Optional[str] = None
    avg_latency: float = 0.0
    stats: NetworkStats = field(default_factory=NetworkStats)

class NetworkError(Exception):
    """Custom network error"""
    pass

class NetworkSimulator:
    """Enhanced network simulator with RPC endpoint management"""
    
    def __init__(self, max_connections: int = 10):
        self.conditions = NetworkCondition(
            latency=0.1,
            jitter=0.05,
            packet_loss=0.01,
            error_rate=0.02,
            bandwidth=10.0,
            timeout=30.0
        )
        self.endpoints: Dict[str, RPCEndpoint] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._max_connections = max_connections
        self._active_connections = 0
        self.error_types = [
            aiohttp.ClientError,
            aiohttp.ServerTimeoutError,
            aiohttp.ClientPayloadError,
            aiohttp.ServerDisconnectedError,
            aiohttp.ClientOSError,
            asyncio.TimeoutError
        ]
        self._start_time = datetime.now()
        self._transferred_bytes = 0

    async def start(self) -> None:
        """Initialize and start the network simulator"""
        try:
            # Initialize session with proper SSL and connection limits
            connector = TCPConnector(
                limit=self._max_connections,
                ssl=True,
                enable_cleanup_closed=True
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.conditions.timeout)
            )
            
            # Start endpoint monitoring
            self._monitor_task = asyncio.create_task(self._monitor_endpoints())
            logger.info("Network simulator started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start network simulator: {str(e)}")
            raise

    async def stop(self) -> None:
        """Clean up resources"""
        try:
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass

            if self.session:
                await self.session.close()
                
            logger.info("Network simulator stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping network simulator: {str(e)}")

    def add_endpoint(self, name: str, url: str, priority: int = 1, 
                    is_private: bool = False) -> None:
        """Add new RPC endpoint"""
        self.endpoints[name] = RPCEndpoint(
            url=url,
            priority=priority,
            is_private=is_private,
            weight=200 if is_private else 100
        )

    async def simulate_request(
        self,
        url: str,
        method: str,
        data: Any,
        size_bytes: int = 1000
    ) -> Tuple[bool, Optional[Dict[str, Any]], float]:
        """Simulate network request with current conditions"""
        if self._active_connections >= self._max_connections:
            raise NetworkError("Connection limit exceeded")
            
        self._active_connections += 1
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Apply network conditions
            await self._simulate_network_conditions(size_bytes)
            
            # Execute request with retries
            response = await self._execute_request_with_retry(url, method, data)
            
            # Calculate metrics
            duration = asyncio.get_event_loop().time() - start_time
            self._update_metrics(True, duration, size_bytes)
            
            return True, response, duration
            
        except Exception as e:
            self._update_metrics(False, 0.0, 0)
            logger.error(f"Request failed: {str(e)}")
            return False, None, 0.0
            
        finally:
            self._active_connections -= 1

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def _execute_request_with_retry(
        self,
        url: str,
        method: str,
        data: Any
    ) -> Optional[Dict[str, Any]]:
        """Execute request with retry logic"""
        if not self.session:
            raise RuntimeError("Session not initialized")

        try:
            async with self.session.post(
                url,
                json={"jsonrpc": "2.0", "method": method, "params": data, "id": 1},
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    raise NetworkError(f"Request failed with status {response.status}")
                return await response.json()
                
        except Exception as e:
            logger.error(f"Request execution failed: {str(e)}")
            raise

    async def _simulate_network_conditions(self, size_bytes: int) -> None:
        """Simulate network conditions"""
        try:
            # Simulate packet loss
            if random.random() < self.conditions.packet_loss:
                raise NetworkError("Simulated packet loss")

            # Simulate latency with jitter
            latency = self.conditions.latency + (
                random.random() * self.conditions.jitter
            )
            await asyncio.sleep(latency)

            # Simulate bandwidth limitations
            if self.conditions.bandwidth > 0:
                transfer_time = size_bytes / (
                    self.conditions.bandwidth * 1024 * 1024
                )
                await asyncio.sleep(transfer_time)

            # Simulate random errors
            if random.random() < self.conditions.error_rate:
                raise random.choice(self.error_types)()

        except Exception as e:
            logger.error(f"Error simulating network conditions: {str(e)}")
            raise

    def _update_metrics(self, success: bool, duration: float, 
                       size_bytes: int) -> None:
        """Update network metrics"""
        try:
            now = datetime.now()
            
            # Update global metrics
            self._transferred_bytes += size_bytes
            
            # Update endpoint metrics
            for endpoint in self.endpoints.values():
                endpoint.stats.requests_total += 1
                if success:
                    endpoint.stats.requests_success += 1
                    endpoint.stats.last_success = now
                    endpoint.stats.avg_latency = (
                        (endpoint.stats.avg_latency * 
                         (endpoint.stats.requests_total - 1) + duration)
                        / endpoint.stats.requests_total
                    )
                else:
                    endpoint.stats.requests_failed += 1
                endpoint.stats.bytes_transferred += size_bytes
                
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    async def _monitor_endpoints(self) -> None:
        """Monitor endpoint health"""
        while True:
            try:
                for name, endpoint in self.endpoints.items():
                    try:
                        # Health check request
                        success, _, duration = await self.simulate_request(
                            endpoint.url,
                            "getHealth",
                            [],
                            100
                        )
                        
                        # Update endpoint status
                        if success:
                            endpoint.success_count += 1
                            endpoint.last_used = datetime.now()
                            endpoint.avg_latency = (
                                (endpoint.avg_latency * 
                                 (endpoint.success_count - 1) + duration)
                                / endpoint.success_count
                            )
                        else:
                            endpoint.error_count += 1
                            
                    except Exception as e:
                        logger.error(f"Health check failed for {name}: {str(e)}")
                        endpoint.error_count += 1
                        endpoint.last_error = str(e)
                        
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {str(e)}")
                await asyncio.sleep(5)

    def get_best_endpoint(self) -> Optional[str]:
        """Get best performing endpoint based on health and priority"""
        if not self.endpoints:
            return None
            
        # Calculate scores for each endpoint
        scored_endpoints = []
        for name, endpoint in self.endpoints.items():
            if endpoint.stats.requests_total == 0:
                continue
                
            success_rate = (
                endpoint.stats.requests_success / 
                endpoint.stats.requests_total
            )
            latency_score = 1.0 / (1.0 + endpoint.avg_latency)
            priority_score = 1.0 / (1.0 + endpoint.priority)
            
            # Combine scores with weights
            score = (
                success_rate * 0.4 +
                latency_score * 0.3 +
                priority_score * 0.3
            ) * endpoint.weight
            
            scored_endpoints.append((name, score))
            
        if not scored_endpoints:
            return next(iter(self.endpoints.keys()))
            
        # Return endpoint with highest score
        return max(scored_endpoints, key=lambda x: x[1])[0]

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive network metrics"""
        uptime = (datetime.now() - self._start_time).total_seconds()
        
        return {
            'uptime_seconds': uptime,
            'total_bytes_transferred': self._transferred_bytes,
            'avg_throughput_mbps': (
                self._transferred_bytes / (1024 * 1024) / uptime
                if uptime > 0 else 0
            ),
            'active_connections': self._active_connections,
            'endpoints': {
                name: {
                    'success_rate': (
                        endpoint.stats.requests_success /
                        endpoint.stats.requests_total * 100
                        if endpoint.stats.requests_total > 0 else 0
                    ),
                    'avg_latency': endpoint.avg_latency,
                    'error_count': endpoint.error_count,
                    'last_used': endpoint.last_used.isoformat()
                    if endpoint.last_used else None,
                    'last_error': endpoint.last_error
                }
                for name, endpoint in self.endpoints.items()
            },
            'conditions': {
                'latency': self.conditions.latency,
                'jitter': self.conditions.jitter,
                'packet_loss': self.conditions.packet_loss,
                'error_rate': self.conditions.error_rate,
                'bandwidth': self.conditions.bandwidth
            }
        }

    @classmethod
    def create_profile(cls, profile: str) -> 'NetworkSimulator':
        """Create simulator with predefined network profile"""
        simulator = cls()
        profiles = {
            'optimal': NetworkCondition(0.05, 0.01, 0.001, 0.001, 100.0, 30.0),
            'good': NetworkCondition(0.1, 0.05, 0.01, 0.02, 10.0, 30.0),
            'poor': NetworkCondition(0.3, 0.1, 0.05, 0.05, 1.0, 30.0),
            'unstable': NetworkCondition(0.5, 0.2, 0.1, 0.1, 0.5, 30.0)
        }
        
        simulator.conditions = profiles.get(
            profile,
            NetworkCondition(0.1, 0.05, 0.01, 0.02, 10.0, 30.0)  # default
        )
        return simulator