#!/usr/bin/env python3

"""
Test Market Cap Fix - Verify market cap estimation resolves validation issues
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
load_dotenv()
os.environ['API_STRATEGY'] = 'dual'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from src.config.settings import Settings
from src.enhanced_token_scanner import EnhancedTokenScanner

async def test_market_cap_fix():
    print("MARKET CAP FIX TEST")
    print("=" * 50)
    
    try:
        alchemy_rpc = os.getenv('ALCHEMY_RPC_URL', 'https://solana-mainnet.g.alchemy.com/v2/test')
        wallet_address = os.getenv('WALLET_ADDRESS', 'JxKzzx2Hif9fnpg9J6jY8XfwYnSLHF6CQZK7zT9ScNb')
        settings = Settings(ALCHEMY_RPC_URL=alchemy_rpc, WALLET_ADDRESS=wallet_address)
        scanner = EnhancedTokenScanner(settings)
        
        print("Testing market cap fix...")
        await scanner.start()
        
        selected_token = await scanner.scan_for_new_tokens()
        
        if selected_token:
            print(f"\nSelected Token: {selected_token['symbol']}")
            price = selected_token.get('price', 0)
            market_cap = selected_token.get('market_cap', 0)
            
            print(f"Price: {price:.12f}")
            print(f"Market Cap: {market_cap:.2f}")
            
            validation_pass = price > 0 and market_cap > 0
            
            if validation_pass:
                print("\nSUCCESS: Validation should now pass!")
                print(f"Expected: [OK] Token passed validation")
            else:
                print(f"\nFAIL: Still failing validation")
                print(f"Price valid: {price > 0}, Market cap valid: {market_cap > 0}")
            
            await scanner.stop()
            return validation_pass
        else:
            print("No token selected")
            await scanner.stop()
            return False
            
    except Exception as e:
        print(f"Test error: {e}")
        return False

async def main():
    success = await test_market_cap_fix()
    print(f"\n{'='*50}")
    print(f"RESULT: {'SUCCESS' if success else 'FAILED'}")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)