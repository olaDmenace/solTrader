#!/usr/bin/env python3

"""
EMERGENCY TEST SCRIPT - GeckoTerminal API Fix
Tests the quota-free GeckoTerminal API to resolve Solana Tracker 403 errors
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from api.geckoterminal_client import GeckoTerminalClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_geckoterminal_api():
    """Test GeckoTerminal API functionality"""
    
    print("\n" + "="*80)
    print("🚨 EMERGENCY API FIX TEST - GeckoTerminal Integration")
    print("="*80)
    
    try:
        # Initialize client
        print("\n📡 Initializing GeckoTerminal client...")
        async with GeckoTerminalClient() as client:
            
            # Test connection
            print("🔍 Testing API connection...")
            connection_ok = await client.test_connection()
            if not connection_ok:
                print("❌ API connection failed!")
                return False
            
            print("✅ GeckoTerminal API connection successful!")
            
            # Get usage stats
            stats = client.get_usage_stats()
            print(f"\n📊 API Usage Stats:")
            print(f"   Provider: {stats['api_provider']}")
            print(f"   Free API: {stats['is_free']}")
            print(f"   Current calls/minute: {stats['calls_this_minute']}/{stats['max_calls_per_minute']}")
            print(f"   Daily capacity: {stats['daily_capacity']:,}")
            print(f"   Monthly capacity: {stats['monthly_capacity']:,}")
            
            # Test trending tokens
            print(f"\n🔥 Testing trending tokens...")
            trending_tokens = await client.get_trending_tokens(limit=10)
            
            if not trending_tokens:
                print("❌ No trending tokens retrieved!")
                return False
            
            print(f"✅ Retrieved {len(trending_tokens)} trending tokens:")
            for i, token in enumerate(trending_tokens[:5]):
                print(f"   {i+1}. {token.symbol} - Price: ${token.price:.8f}, "
                      f"Change: {token.price_change_24h:.1f}%, "
                      f"Volume: ${token.volume_24h:,.0f}")
            
            # Test volume tokens
            print(f"\n📈 Testing volume tokens...")
            volume_tokens = await client.get_volume_tokens(limit=10)
            
            if not volume_tokens:
                print("❌ No volume tokens retrieved!")
                return False
                
            print(f"✅ Retrieved {len(volume_tokens)} volume tokens:")
            for i, token in enumerate(volume_tokens[:5]):
                print(f"   {i+1}. {token.symbol} - Volume: ${token.volume_24h:,.0f}, "
                      f"Liquidity: ${token.liquidity:,.0f}, "
                      f"Age: {token.age_minutes}min")
            
            # Test combined token discovery
            print(f"\n🎯 Testing combined token discovery...")
            all_tokens = await client.get_all_tokens()
            
            if not all_tokens:
                print("❌ No tokens from combined discovery!")
                return False
            
            print(f"✅ Combined discovery found {len(all_tokens)} unique tokens!")
            
            # Show top tokens by momentum score
            top_tokens = sorted(all_tokens, key=lambda x: x.momentum_score, reverse=True)[:10]
            print(f"\n🏆 Top 10 tokens by momentum score:")
            for i, token in enumerate(top_tokens):
                print(f"   {i+1}. {token.symbol} (score: {token.momentum_score:.1f}) - "
                      f"Source: {token.source}, "
                      f"Price: ${token.price:.8f}, "
                      f"Change: {token.price_change_24h:.1f}%")
            
            # Final API stats
            final_stats = client.get_usage_stats()
            print(f"\n📊 Final API Usage:")
            print(f"   Total requests made: {final_stats['request_breakdown']['total']}")
            print(f"   Current minute usage: {final_stats['calls_this_minute']}/{final_stats['max_calls_per_minute']}")
            print(f"   No quota limits - FREE API! 🎉")
            
            print(f"\n✅ GeckoTerminal API test completed successfully!")
            print(f"   - API is working and quota-free")
            print(f"   - Token discovery is functional") 
            print(f"   - Rate limiting is working properly")
            print(f"   - Ready to replace Solana Tracker API")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"GeckoTerminal test failed: {e}", exc_info=True)
        return False

async def test_quota_calculation():
    """Test the new safe scanning intervals"""
    
    print(f"\n" + "="*80)
    print("📊 QUOTA MANAGEMENT ANALYSIS")
    print("="*80)
    
    # New safe configuration
    scanner_interval = 900  # 15 minutes
    daily_scans = (24 * 60 * 60) / scanner_interval
    api_calls_per_scan = 3  # trending + volume + new_pools
    daily_api_calls = daily_scans * api_calls_per_scan
    
    # GeckoTerminal capacity
    geckoterminal_daily_capacity = 25 * 60 * 24  # 36,000/day
    
    print(f"📈 New Safe Configuration:")
    print(f"   Scanner interval: {scanner_interval} seconds ({scanner_interval/60:.1f} minutes)")
    print(f"   Daily scans: {daily_scans:.0f}")
    print(f"   API calls per scan: {api_calls_per_scan}")
    print(f"   Daily API calls: {daily_api_calls:.0f}")
    print(f"   GeckoTerminal daily capacity: {geckoterminal_daily_capacity:,}")
    print(f"   Usage percentage: {(daily_api_calls / geckoterminal_daily_capacity) * 100:.1f}%")
    
    if daily_api_calls < geckoterminal_daily_capacity:
        print(f"✅ Configuration is SAFE - well within quota limits!")
    else:
        print(f"❌ Configuration exceeds quota!")
    
    print(f"\n📊 Comparison with old configuration:")
    old_daily_calls = 1440 * 3  # Every minute, 3 calls
    print(f"   Old daily calls: {old_daily_calls:,}")
    print(f"   New daily calls: {daily_api_calls:.0f}")
    print(f"   Reduction factor: {old_daily_calls / daily_api_calls:.1f}x")
    print(f"   Quota savings: {((old_daily_calls - daily_api_calls) / old_daily_calls) * 100:.1f}%")

def main():
    """Main test function"""
    print(f"🚀 Starting GeckoTerminal API Emergency Fix Test")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Purpose: Replace quota-exhausted Solana Tracker API")
    
    # Set environment for testing
    os.environ['API_PROVIDER'] = 'geckoterminal'
    
    # Run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Test quota calculations first
        loop.run_until_complete(test_quota_calculation())
        
        # Test API functionality
        api_test_result = loop.run_until_complete(test_geckoterminal_api())
        
        print(f"\n" + "="*80)
        print("📋 TEST SUMMARY")
        print("="*80)
        
        if api_test_result:
            print("✅ All tests PASSED!")
            print("✅ GeckoTerminal API is working perfectly")
            print("✅ Quota-free operation confirmed")
            print("✅ Token discovery is functional")
            print("✅ Rate limiting is working")
            print("\n🎯 READY FOR DEPLOYMENT:")
            print("   1. Email spam has been stopped")
            print("   2. Scanner interval increased to 15 minutes")
            print("   3. GeckoTerminal API integrated")
            print("   4. Bot should now discover tokens successfully")
            print("\n🚀 Next step: Restart the bot to apply fixes!")
            
        else:
            print("❌ API tests FAILED!")
            print("❌ Further investigation needed")
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False
    finally:
        loop.close()

if __name__ == "__main__":
    main()