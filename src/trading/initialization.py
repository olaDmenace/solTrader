# src/trading/initialization.py

import logging
from typing import Dict, Any, Optional
import asyncio
from dataclasses import dataclass
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    apis_healthy: bool
    wallet_connected: bool
    sufficient_balance: bool
    network_latency: float
    errors: Optional[str] = None

class StrategyInitializer:
    def __init__(self, settings: Any, wallet: Any, apis: Dict[str, Any]):
        self.settings = settings
        self.wallet = wallet
        self.apis = apis
        self.retry_attempts = 3
        self.retry_delay = 5

    async def initialize(self) -> bool:
        try:
            if not self._validate_config():
                return False

            health_status = await self._check_health()
            if not health_status.apis_healthy:
                logger.error(f"Health check failed: {health_status.errors}")
                return False

            wallet_ready = await self._initialize_wallet()
            if not wallet_ready:
                return False

            logger.info("Strategy initialization successful")
            return True

        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            return False

    def _validate_config(self) -> bool:
        required_settings = [
            'ALCHEMY_RPC_URL',
            'WALLET_ADDRESS',
            'MAX_POSITIONS',
            'MAX_TRADE_SIZE',
            'STOP_LOSS_PERCENTAGE',
            'TAKE_PROFIT_PERCENTAGE'
        ]
        return all(hasattr(self.settings, setting) for setting in required_settings)

    async def _check_health(self) -> HealthStatus:
        latencies = []
        errors = []

        for name, api in self.apis.items():
            try:
                start_time = asyncio.get_event_loop().time()
                healthy = await self._retry_api_check(api)
                latency = asyncio.get_event_loop().time() - start_time

                if not healthy:
                    errors.append(f"{name} API check failed")
                latencies.append(latency)

            except Exception as e:
                errors.append(f"{name} error: {str(e)}")

        return HealthStatus(
            apis_healthy=len(errors) == 0,
            wallet_connected=False,  # Will be updated
            sufficient_balance=False,  # Will be updated
            network_latency=sum(latencies) / len(latencies) if latencies else 0,
            errors="; ".join(errors) if errors else None
        )

    async def _retry_api_check(self, api: Any) -> bool:
        for attempt in range(self.retry_attempts):
            try:
                return await api.test_connection()
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise
                await asyncio.sleep(self.retry_delay)
        return False

    async def _initialize_wallet(self) -> bool:
        try:
            connected = await self._retry_operation(
                self.wallet.connect,
                self.settings.WALLET_ADDRESS
            )
            if not connected:
                return False

            balance = await self.wallet.get_balance()
            if balance < self.settings.MIN_BALANCE:
                logger.error(f"Insufficient balance: {balance} SOL")
                return False

            return True

        except Exception as e:
            logger.error(f"Wallet initialization error: {str(e)}")
            return False

    async def _retry_operation(self, operation, *args):
        for attempt in range(self.retry_attempts):
            try:
                return await operation(*args)
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise
                await asyncio.sleep(self.retry_delay)
        return False