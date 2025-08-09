#!/usr/bin/env python3

"""
Debug Token Data Values
Check what actual values are being returned for price and market_cap
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up environment
load_dotenv()
os.environ['API_STRATEGY'] = 'dual'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.config.settings import Settings
from src.enhanced_token_scanner import EnhancedTokenScanner

async def debug_token_data():
    """Debug actual token data values being returned"""
    
    print("TOKEN DATA DEBUGGING")
    print("=" * 50)
    
    try:
        # Initialize scanner
        alchemy_rpc = os.getenv('ALCHEMY_RPC_URL', 'https://solana-mainnet.g.alchemy.com/v2/test')
        wallet_address = os.getenv('WALLET_ADDRESS', 'JxKzzx2Hif9fnpg9J6jY8XfwYnSLHF6CQZK7zT9ScNb')
        settings = Settings(ALCHEMY_RPC_URL=alchemy_rpc, WALLET_ADDRESS=wallet_address)
        scanner = EnhancedTokenScanner(settings)
        
        print("Starting scanner...")
        await scanner.start()
        
        print("Performing token scan...")
        approved_tokens = await scanner._perform_full_scan()
        
        if approved_tokens and len(approved_tokens) > 0:
            print(f"\nFound {len(approved_tokens)} approved tokens")
            
            # Show detailed data for first few tokens
            for i, scan_result in enumerate(approved_tokens[:3]):
                token = scan_result.token
                print(f"\n--- TOKEN {i+1}: {token.symbol} ---")
                print(f"Address: {token.address}")
                print(f"Price (from API): {token.price}")
                print(f"Price type: {type(token.price)}")
                print(f"Market Cap (from API): {token.market_cap}")  
                print(f"Market Cap type: {type(token.market_cap)}")
                print(f"Volume 24h: {token.volume_24h}")
                print(f"Liquidity: {token.liquidity}")
                print(f"Price Change 24h: {token.price_change_24h}%")
                print(f"Source: {token.source}")
                print(f"Score: {scan_result.score:.1f}")
        else:
            print("No approved tokens found")
        
        # Test scan_for_new_tokens method
        print(f"\n" + "=" * 30)
        print("TESTING scan_for_new_tokens() OUTPUT")
        print("=" * 30)
        
        selected_token = await scanner.scan_for_new_tokens()
        
        if selected_token:
            print(f"\nSelected Token Data Structure:")
            for key, value in selected_token.items():
                print(f"  {key}: {value} (type: {type(value).__name__})")
            
            # Specifically check problem fields
            print(f"\n--- PROBLEM FIELDS ANALYSIS ---")
            print(f"Price: {selected_token.get('price', 'MISSING')} SOL")
            print(f"Market Cap: {selected_token.get('market_cap', 'MISSING')} SOL")
            print(f"Price is zero: {selected_token.get('price', 0) == 0}")
            print(f"Market cap is zero: {selected_token.get('market_cap', 0) == 0}")
            
            # Calculate potential conversions
            price_usd = selected_token.get('price', 0)
            market_cap_usd = selected_token.get('market_cap', 0)
            
            # Rough SOL price estimate (you'd normally get this from an API)
            sol_price_usd = 150.0  # Approximate SOL price in USD
            
            if price_usd > 0:
                price_sol = price_usd / sol_price_usd
                print(f"Price converted to SOL: {price_sol:.12f} SOL (assuming SOL = ${sol_price_usd})")
            
            if market_cap_usd > 0:
                market_cap_sol = market_cap_usd / sol_price_usd  
                print(f"Market cap converted to SOL: {market_cap_sol:.2f} SOL")
            
        else:
            print("No token selected")
        
        await scanner.stop()
        return selected_token is not None
        
    except Exception as e:
        print(f"Debug error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main debug function"""
    
    print("DEBUGGING TOKEN DATA VALUES")
    print("=" * 50)
    
    success = await debug_token_data()
    
    print("\n" + "=" * 50)
    if success:
        print("DEBUG COMPLETE - Check output above for data issues")
    else:
        print("DEBUG FAILED - Check errors above")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)