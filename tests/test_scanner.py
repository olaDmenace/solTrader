import pytest
import pytest_asyncio
import logging
from src.token_scanner import TokenScanner
from src.api.jupiter import JupiterClient
from src.api.alchemy import AlchemyClient
from dotenv import load_dotenv
import os
from typing import AsyncIterator, cast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session", autouse=True)
def load_env() -> None:
    load_dotenv()
    rpc_url = os.getenv('ALCHEMY_RPC_URL')
    assert rpc_url, "ALCHEMY_RPC_URL not found"

@pytest_asyncio.fixture
async def scanner() -> AsyncIterator[TokenScanner]:
    """Create TokenScanner instance for testing"""
    rpc_url = os.getenv('ALCHEMY_RPC_URL')
    if not rpc_url:
        raise ValueError("ALCHEMY_RPC_URL not found in environment")

    jupiter_client = JupiterClient()
    alchemy_client = AlchemyClient(rpc_url)
    scanner = TokenScanner(jupiter_client, alchemy_client)

    yield scanner

    await alchemy_client.close()

@pytest.mark.asyncio
async def test_scan_new_listings(scanner: TokenScanner) -> None:
    """Test scanning for new token listings"""
    tokens = await scanner.scan_new_listings()
    assert isinstance(tokens, list), "Should return list of tokens"
    logger.info(f"Found {len(tokens)} new token listings")

@pytest.mark.asyncio
async def test_token_screening(scanner: TokenScanner) -> None:
    """Test token screening process"""
    sol_token = {
        "address": "So11111111111111111111111111111111111111112",
        "symbol": "SOL",
        "name": "Wrapped SOL",
        "decimals": 9
    }

    passed = await scanner._comprehensive_screen(sol_token)
    assert isinstance(passed, bool), "Screening should return boolean"
    logger.info(f"Token screening result: {passed}")

@pytest.mark.asyncio
async def test_get_token_metrics(scanner: TokenScanner) -> None:
    """Test getting token metrics"""
    sol_address = "So11111111111111111111111111111111111111112"
    metrics = await scanner._get_token_metrics(sol_address)
    assert metrics is not None, "Should return token metrics"
    assert hasattr(metrics, 'price'), "Should have price"
    assert hasattr(metrics, 'volume_24h'), "Should have volume"
    logger.info(f"Token metrics: {metrics}")