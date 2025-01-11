import pytest
import pytest_asyncio
import logging
from src.api.alchemy import AlchemyClient
from dotenv import load_dotenv
import os
from typing import AsyncGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session", autouse=True)
def load_env() -> None:
    """Load environment variables before tests"""
    load_dotenv()
    alchemy_url = os.getenv('ALCHEMY_RPC_URL')
    wallet_address = os.getenv('WALLET_ADDRESS')
    
    # Verify required environment variables
    assert alchemy_url and alchemy_url.startswith('https://'), "Invalid ALCHEMY_RPC_URL"
    assert wallet_address, "WALLET_ADDRESS not found in .env"

@pytest_asyncio.fixture
async def alchemy_client() -> AsyncGenerator[AlchemyClient, None]:
    """Create Alchemy client fixture"""
    rpc_url = os.getenv('ALCHEMY_RPC_URL')
    if not rpc_url:
        raise ValueError("ALCHEMY_RPC_URL not found in environment")
    client = AlchemyClient(rpc_url)
    yield client
    await client.close()

@pytest.mark.asyncio
async def test_alchemy_connection(alchemy_client: AlchemyClient) -> None:
    """Test Alchemy RPC connection"""
    try:
        logger.info("Testing Alchemy connection...")
        result = await alchemy_client.test_connection()
        assert result is True, "Alchemy connection failed"
    except Exception as e:
        logger.error(f"Unexpected error in Alchemy connection test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_get_sol_balance(alchemy_client: AlchemyClient) -> None:
    """Test getting SOL balance"""
    try:
        wallet_address = os.getenv('WALLET_ADDRESS')
        if not wallet_address:
            raise ValueError("WALLET_ADDRESS not found in environment")
            
        logger.info(f"Testing balance check for wallet: {wallet_address}")
        balance = await alchemy_client.get_balance(wallet_address)
        assert isinstance(balance, float), "Balance should be a float"
        assert balance >= 0, "Balance should be non-negative"
        logger.info(f"Wallet balance: {balance} SOL")
    except Exception as e:
        logger.error(f"Unexpected error in balance test: {str(e)}")
        raise
