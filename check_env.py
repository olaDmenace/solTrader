import os
from dotenv import load_dotenv

load_dotenv()
print(f"RPC URL: {os.getenv('ALCHEMY_RPC_URL')}")
print(f"Wallet: {os.getenv('WALLET_ADDRESS')}")