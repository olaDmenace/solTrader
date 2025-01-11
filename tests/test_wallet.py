import pytest
import pytest_asyncio
import logging
from typing import AsyncGenerator, AsyncIterator
from src.phantom_wallet import PhantomWallet
from src.api.alchemy import AlchemyClient
from dotenv import load_dotenv
import os
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session", autouse=True)
def load_env() -> None:
    """Load environment variables"""
    load_dotenv()
    assert os.getenv('ALCHEMY_RPC_URL'), "ALCHEMY_RPC_URL not found"
    assert os.getenv('WALLET_ADDRESS'), "WALLET_ADDRESS not found"

@pytest_asyncio.fixture
async def wallet() -> AsyncIterator[PhantomWallet]:
    """Create PhantomWallet instance for testing"""
    rpc_url = os.getenv('ALCHEMY_RPC_URL')
    if not rpc_url:
        raise ValueError("ALCHEMY_RPC_URL not found in environment")

    alchemy_client = AlchemyClient(rpc_url)
    wallet = PhantomWallet(alchemy_client)

    yield wallet  # Provide the wallet instance to the test

    # Cleanup after the test
    await wallet.disconnect()
    await alchemy_client.close()

@pytest.mark.asyncio
async def test_wallet_connection(wallet: PhantomWallet) -> None:
    """Test wallet connection"""
    wallet_address = os.getenv('WALLET_ADDRESS')
    assert wallet_address, "WALLET_ADDRESS not found in environment"

    connected = await wallet.connect(wallet_address)
    assert connected is True, "Wallet connection failed"
    assert wallet.connected is True, "Wallet should be marked as connected"
    assert wallet.wallet_address == wallet_address, "Wallet address should be set"

@pytest.mark.asyncio
async def test_get_balance(wallet: PhantomWallet) -> None:
    """Test getting wallet balance"""
    wallet_address = os.getenv('WALLET_ADDRESS')
    assert wallet_address, "WALLET_ADDRESS not found in environment"

    await wallet.connect(wallet_address)
    balance = await wallet.get_balance()

    assert balance is not None, "Balance should not be None"
    assert isinstance(balance, float), "Balance should be float"
    assert balance >= 0, "Balance should be non-negative"
    logger.info(f"Wallet balance: {balance} SOL")

@pytest.mark.asyncio
async def test_token_accounts(wallet: PhantomWallet) -> None:
    """Test getting token accounts"""
    wallet_address = os.getenv('WALLET_ADDRESS')
    assert wallet_address, "WALLET_ADDRESS not found in environment"

    await wallet.connect(wallet_address)
    accounts = await wallet.get_token_accounts()

    assert isinstance(accounts, list), "Should return list of accounts"
    logger.info(f"Found {len(accounts)} token accounts")

@pytest.mark.asyncio
async def test_monitor_transactions(wallet: PhantomWallet) -> None:
    """Test transaction monitoring"""
    wallet_address = os.getenv('WALLET_ADDRESS')
    assert wallet_address, "WALLET_ADDRESS not found in environment"

    await wallet.connect(wallet_address)

    # Test callback function
    async def test_callback(tx_info: dict) -> None:
        logger.info(f"Transaction detected: {tx_info}")

    # Start monitoring
    monitoring_started = await wallet.monitor_transactions(callback=test_callback)
    assert monitoring_started is True, "Should be able to start monitoring"

    # Let it run briefly
    await asyncio.sleep(2)

    # Verify monitoring task
    assert wallet._monitor_task is not None, "Monitor task should exist"
    assert not wallet._monitor_task.done(), "Monitor task should be running"

    # Clean up
    if wallet._monitor_task and not wallet._monitor_task.done():
        wallet._monitor_task.cancel()
        try:
            await wallet._monitor_task
        except asyncio.CancelledError:
            pass

@pytest.mark.asyncio
async def test_wallet_state(wallet: PhantomWallet) -> None:
    """Test wallet state management"""
    wallet_address = os.getenv('WALLET_ADDRESS')
    assert wallet_address, "WALLET_ADDRESS not found in environment"

    # Test initial state
    assert not wallet.connected, "Should start disconnected"
    assert wallet.wallet_address is None, "Should start with no address"
    assert wallet.is_stale(), "New wallet should be stale"

    # Test connection
    await wallet.connect(wallet_address)
    assert wallet.connected, "Should be connected"
    assert not wallet.is_stale(), "Fresh connection should not be stale"

    # Test disconnection
    await wallet.disconnect()
    assert not wallet.connected, "Should be disconnected"
    assert wallet.wallet_address is None, "Should clear address"
    assert wallet.is_stale(), "Disconnected wallet should be stale"