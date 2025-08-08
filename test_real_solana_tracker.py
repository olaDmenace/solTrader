#!/usr/bin/env python3

"""
Test Real Solana Tracker API - No Unicode
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment
load_dotenv()

try:
    from api.solana_tracker import SolanaTrackerClient
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

async def test_solana_tracker_real():
    """Test real Solana Tracker API calls"""
    
    print("REAL SOLANA TRACKER API TEST")
    print("=" * 40)
    
    try:
        async with SolanaTrackerClient() as client:
            print(f"API key loaded: {bool(client.api_key)}")
            
            # Test trending tokens
            print("Testing trending tokens...")
            trending = await client.get_trending_tokens(limit=10)
            
            print(f"Trending tokens returned: {len(trending)}")
            
            if trending:
                for i, token in enumerate(trending[:5]):
                    print(f"  {i+1}. {token.symbol} - Price: ${token.price:.8f}, Change: {token.price_change_24h:.1f}%")
            
            # Test volume tokens  
            print("\nTesting volume tokens...")
            volume = await client.get_volume_tokens(limit=10)
            
            print(f"Volume tokens returned: {len(volume)}")
            
            if volume:
                for i, token in enumerate(volume[:5]):
                    print(f"  {i+1}. {token.symbol} - Volume: ${token.volume_24h:.0f}")
            
            # Test all tokens
            print("\nTesting combined discovery...")
            all_tokens = await client.get_all_tokens()
            
            print(f"All tokens combined: {len(all_tokens)}")
            
            return {
                'trending_count': len(trending),
                'volume_count': len(volume),
                'total_count': len(all_tokens),
                'api_working': len(all_tokens) > 0
            }
            
    except Exception as e:
        print(f"Error: {e}")
        return {'api_working': False, 'error': str(e)}

def main():
    """Main function"""
    
    try:
        results = asyncio.run(test_solana_tracker_real())
        
        print("\n" + "=" * 40)
        print("TEST RESULTS")
        print("=" * 40)
        
        if results.get('api_working'):
            print("SUCCESS: Solana Tracker API working!")
            print(f"Trending: {results.get('trending_count', 0)}")
            print(f"Volume: {results.get('volume_count', 0)}")  
            print(f"Total: {results.get('total_count', 0)}")
            return True
        else:
            print("FAILED: API not working")
            print(f"Error: {results.get('error', 'Unknown')}")
            return False
            
    except Exception as e:
        print(f"Test error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)