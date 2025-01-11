
import asyncio
from src.api.alchemy import AlchemyClient
import os
from dotenv import load_dotenv
import sys

async def test_manual() -> None:
    load_dotenv()
    rpc_url = os.getenv('ALCHEMY_RPC_URL')
    wallet = os.getenv('WALLET_ADDRESS')

    if not rpc_url:
        print("Error: ALCHEMY_RPC_URL not found in environment")
        sys.exit(1)

    if not wallet:
        print("Error: WALLET_ADDRESS not found in environment")
        sys.exit(1)

    print(f"RPC URL: {rpc_url}")
    print(f"Wallet: {wallet}")

    client = AlchemyClient(rpc_url)
    balance = await client.get_balance(wallet)
    print(f"Balance: {balance} SOL")
    await client.close()

if __name__ == "__main__":
    asyncio.run(test_manual())
