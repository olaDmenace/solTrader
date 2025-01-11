# src/utils/network_simulator.py

import asyncio
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import aiohttp
from dataclasses import dataclass

@dataclass
class NetworkCondition:
    latency: float  # Base latency in seconds
    jitter: float   # Random latency variation
    packet_loss: float  # Probability of packet loss
    error_rate: float   # Probability of errors
    bandwidth: float    # Simulated bandwidth in MB/s
    timeout: float     # Connection timeout in seconds

class NetworkError(Exception):
    pass

class NetworkSimulator:
    def __init__(self):
        self.conditions = NetworkCondition(
            latency=0.0,
            jitter=0.0,
            packet_loss=0.0,
            error_rate=0.0,
            bandwidth=float('inf'),
            timeout=30.0
        )
        self.error_types = [
            aiohttp.ClientError("Connection refused"),
            aiohttp.ServerTimeoutError("Request timeout"),
            aiohttp.ClientPayloadError("Corrupted response"),
            aiohttp.ServerDisconnectedError("Server disconnected"),
            aiohttp.ClientOSError("Connection reset"),
            asyncio.TimeoutError("Operation timeout")
        ]
        self._start_time = datetime.now()
        self._transferred_bytes = 0
        self._active_connections = 0
        self._max_connections = 10

    async def simulate_request(self, size_bytes: int = 1000) -> None:
        """Simulate a network request with current conditions"""
        await self._check_connection_limit()
        await self._simulate_latency()
        await self._simulate_bandwidth(size_bytes)
        self._simulate_errors()
        self._update_metrics(size_bytes)

    async def _check_connection_limit(self) -> None:
        """Check if new connection can be established"""
        if self._active_connections >= self._max_connections:
            raise NetworkError("Connection limit exceeded")
        self._active_connections += 1

    async def _simulate_latency(self) -> None:
        """Simulate network latency with jitter"""
        if self.conditions.latency > 0:
            jitter = random.uniform(-self.conditions.jitter, self.conditions.jitter)
            total_latency = max(0, self.conditions.latency + jitter)
            await asyncio.sleep(total_latency)

    async def _simulate_bandwidth(self, size_bytes: int) -> None:
        """Simulate bandwidth limitations"""
        if self.conditions.bandwidth != float('inf'):
            transfer_time = size_bytes / (self.conditions.bandwidth * 1024 * 1024)
            await asyncio.sleep(transfer_time)

    def _simulate_errors(self) -> None:
        """Simulate various network errors"""
        if random.random() < self.conditions.packet_loss:
            raise NetworkError("Packet loss")
            
        if random.random() < self.conditions.error_rate:
            raise random.choice(self.error_types)

    def _update_metrics(self, size_bytes: int) -> None:
        """Update internal metrics"""
        self._transferred_bytes += size_bytes
        self._active_connections -= 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current network metrics"""
        uptime = (datetime.now() - self._start_time).total_seconds()
        return {
            'uptime': uptime,
            'throughput': self._transferred_bytes / uptime if uptime > 0 else 0,
            'active_connections': self._active_connections,
            'total_transferred': self._transferred_bytes
        }

    @classmethod
    def create_profile(cls, profile: str) -> 'NetworkSimulator':
        """Create simulator with predefined network profile"""
        simulator = cls()
        profiles = {
            'perfect': NetworkCondition(0, 0, 0, 0, float('inf'), 30),
            'good': NetworkCondition(0.05, 0.01, 0.001, 0.01, 10, 10),
            'poor': NetworkCondition(0.2, 0.05, 0.01, 0.05, 1, 5),
            'terrible': NetworkCondition(1.0, 0.2, 0.05, 0.1, 0.1, 3),
            'mobile': NetworkCondition(0.1, 0.05, 0.02, 0.03, 2, 15),
            'satellite': NetworkCondition(0.5, 0.1, 0.03, 0.02, 0.5, 20)
        }
        simulator.conditions = profiles.get(profile, profiles['good'])
        return simulator