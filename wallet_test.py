import asyncio
import traceback
from src.api.alchemy import AlchemyClient
import os
from dotenv import load_dotenv
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def check_wallet() -> None:
    load_dotenv()
    rpc_url = os.getenv('ALCHEMY_RPC_URL')
    wallet_address = os.getenv('WALLET_ADDRESS')

    if not rpc_url:
        raise ValueError("ALCHEMY_RPC_URL not found in environment")
    if not wallet_address:
        raise ValueError("WALLET_ADDRESS not found in environment")

    print(f"\nChecking wallet details:")
    print(f"------------------------")
    print(f"Wallet Address: {wallet_address}")
    print(f"RPC URL: {rpc_url}")

    client = AlchemyClient(rpc_url)

    try:
        # Test connection
        print("\nTesting connection...")
        connection = await client.test_connection()
        print(f"Connection test result: {connection}")

        # Get account info
        print("\nGetting account info...")
        response = await client._make_request("getAccountInfo", [wallet_address])
        print(f"\nAccount Info Response: {response}")

        # Get balance
        print("\nGetting balance...")
        balance = await client.get_balance(wallet_address)
        print(f"Balance: {balance} SOL")

    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(check_wallet())