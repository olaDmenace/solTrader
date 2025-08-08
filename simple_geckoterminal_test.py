#!/usr/bin/env python3

"""
Simple GeckoTerminal API Test - No Unicode characters
"""

import asyncio
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from api.geckoterminal_client import GeckoTerminalClient
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def simple_test():
    """Simple API test"""
    
    print("Starting GeckoTerminal API Test...")
    
    try:
        async with GeckoTerminalClient() as client:
            print("Client initialized successfully")
            
            # Test connection
            print("Testing connection...")
            connected = await client.test_connection()
            
            if connected:
                print("SUCCESS: API connection working!")
                
                # Get some tokens
                print("Getting trending tokens...")
                tokens = await client.get_trending_tokens(limit=5)
                
                if tokens:
                    print(f"SUCCESS: Found {len(tokens)} tokens")
                    for i, token in enumerate(tokens):
                        print(f"  {i+1}. {token.symbol} - ${token.price:.6f}")
                    
                    return True
                else:
                    print("ERROR: No tokens found")
                    return False
            else:
                print("ERROR: API connection failed")
                return False
                
    except Exception as e:
        print(f"ERROR: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

def main():
    """Main function"""
    print("=" * 50)
    print("EMERGENCY GeckoTerminal API TEST")
    print("=" * 50)
    
    os.environ['API_PROVIDER'] = 'geckoterminal'
    
    try:
        result = asyncio.run(simple_test())
        
        print("=" * 50)
        if result:
            print("TEST PASSED - API is working!")
            print("Bot should now discover tokens successfully!")
        else:
            print("TEST FAILED - Need to investigate")
        print("=" * 50)
            
    except Exception as e:
        print(f"Test execution error: {e}")

if __name__ == "__main__":
    main()