#!/usr/bin/env python3
"""
Critical Scanner Diagnostic Tool
Tests each API endpoint individually to identify the exact issue
"""
import asyncio
import aiohttp
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging for detailed output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_dexscreener_api():
    """Test DexScreener API directly"""
    logger.info("🔍 Testing DexScreener API...")
    
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.dexscreener.com/latest/dex/pairs/solana"
            logger.info(f"Requesting: {url}")
            
            async with session.get(url, timeout=10) as response:
                logger.info(f"Status: {response.status}")
                logger.info(f"Headers: {dict(response.headers)}")
                
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    logger.info(f"✅ DexScreener API working! Found {len(pairs)} pairs")
                    
                    # Show first few pairs
                    for i, pair in enumerate(pairs[:3]):
                        base_token = pair.get('baseToken', {})
                        logger.info(f"  Pair {i+1}: {base_token.get('symbol', 'UNKNOWN')} - {base_token.get('address', 'No address')}")
                    
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ DexScreener API failed: {response.status} - {error_text}")
                    return False
                    
    except Exception as e:
        logger.error(f"❌ DexScreener API error: {e}")
        return False

async def test_jupiter_api():
    """Test Jupiter API directly"""
    logger.info("🪐 Testing Jupiter API...")
    
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://token.jup.ag/all"
            logger.info(f"Requesting: {url}")
            
            async with session.get(url, timeout=10) as response:
                logger.info(f"Status: {response.status}")
                logger.info(f"Headers: {dict(response.headers)}")
                
                if response.status == 200:
                    tokens = await response.json()
                    logger.info(f"✅ Jupiter API working! Found {len(tokens)} tokens")
                    
                    # Show first few tokens
                    for i, token in enumerate(tokens[:3]):
                        logger.info(f"  Token {i+1}: {token.get('symbol', 'UNKNOWN')} - {token.get('address', 'No address')}")
                    
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Jupiter API failed: {response.status} - {error_text}")
                    return False
                    
    except Exception as e:
        logger.error(f"❌ Jupiter API error: {e}")
        return False

async def test_birdeye_api():
    """Test Birdeye API directly"""
    logger.info("🐦 Testing Birdeye API...")
    
    try:
        from src.birdeye_client import BirdeyeClient
        
        # Test without API key first
        async with BirdeyeClient(api_key=None) as client:
            logger.info("Testing Birdeye without API key...")
            trending_tokens = await client.get_trending_tokens(limit=5)
            
            if trending_tokens:
                logger.info(f"✅ Birdeye API working! Found {len(trending_tokens)} trending tokens")
                for i, token in enumerate(trending_tokens[:3]):
                    logger.info(f"  Token {i+1}: {token.symbol} (#{token.rank}) - {token.price_24h_change_percent:+.1f}%")
                return True
            else:
                logger.warning("⚠️ Birdeye API returned no trending tokens")
                return False
                
    except Exception as e:
        logger.error(f"❌ Birdeye API error: {e}")
        return False

async def test_scanner_integration():
    """Test the actual scanner implementation"""
    logger.info("🔧 Testing Scanner Integration...")
    
    try:
        from src.practical_solana_scanner import PracticalSolanaScanner
        from src.config.settings import load_settings
        
        # Load settings
        try:
            settings = load_settings()
            logger.info("✅ Settings loaded successfully")
        except Exception as e:
            logger.error(f"❌ Settings loading failed: {e}")
            # Create mock settings
            class MockSettings:
                SCAN_INTERVAL = 5
                MIN_LIQUIDITY = 500.0
                MIN_VOLUME_24H = 50.0
                NEW_TOKEN_MAX_AGE_MINUTES = 2880
                MIN_TOKEN_PRICE_SOL = 0.000001
                MAX_TOKEN_PRICE_SOL = 0.01
                MIN_MARKET_CAP_SOL = 10.0
                MAX_MARKET_CAP_SOL = 10000.0
                ENABLE_TRENDING_FILTER = True
                MAX_TRENDING_RANK = 50
                MIN_PRICE_CHANGE_24H = 20.0
                MIN_VOLUME_CHANGE_24H = 10.0
                MIN_TRENDING_SCORE = 60.0
                TRENDING_SIGNAL_BOOST = 0.5
                TRENDING_FALLBACK_MODE = "permissive"
                TRENDING_CACHE_DURATION = 300
                BIRDEYE_API_KEY = None
            
            settings = MockSettings()
            logger.info("⚠️ Using mock settings")
        
        # Create scanner
        scanner = PracticalSolanaScanner(None, None, settings)
        logger.info("✅ Scanner created")
        
        # Initialize session manually (like our fix does)
        scanner.running = True
        scanner.session = aiohttp.ClientSession()
        logger.info("✅ Scanner session initialized")
        
        # Test individual scan methods
        logger.info("Testing DexScreener scan method...")
        dex_result = await scanner._scan_dexscreener_new_pairs()
        logger.info(f"DexScreener result: {dex_result is not None}")
        
        logger.info("Testing Jupiter scan method...")
        jupiter_result = await scanner._scan_jupiter_recent_tokens()
        logger.info(f"Jupiter result: {jupiter_result is not None}")
        
        logger.info("Testing Birdeye scan method...")
        birdeye_result = await scanner._scan_birdeye_trending_tokens()
        logger.info(f"Birdeye result: {birdeye_result is not None}")
        
        # Test main scan method
        logger.info("Testing main scan_for_new_tokens method...")
        main_result = await scanner.scan_for_new_tokens()
        logger.info(f"Main scan result: {main_result is not None}")
        
        if main_result:
            logger.info(f"✅ SUCCESS! Found token: {main_result}")
        else:
            logger.warning("⚠️ No tokens found in main scan")
        
        # Cleanup
        await scanner.session.close()
        
        return main_result is not None
        
    except Exception as e:
        logger.error(f"❌ Scanner integration test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Run all diagnostic tests"""
    logger.info("🚀 Starting Scanner Diagnostic Tests")
    logger.info("="*60)
    
    tests = [
        ("DexScreener API", test_dexscreener_api),
        ("Jupiter API", test_jupiter_api),
        ("Birdeye API", test_birdeye_api),
        ("Scanner Integration", test_scanner_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = await test_func()
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.info(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"💥 {test_name}: CRASHED - {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Scanner should be working now.")
    elif passed >= total / 2:
        logger.info("⚠️ PARTIAL SUCCESS! Some APIs working, check individual failures.")
    else:
        logger.info("💥 MAJOR ISSUES! Most APIs are failing.")
    
    logger.info(f"\n{'='*60}")
    
    # Specific recommendations
    if not results.get("DexScreener API", False):
        logger.info("🔧 DexScreener API failed - check network connectivity")
    
    if not results.get("Jupiter API", False):
        logger.info("🔧 Jupiter API failed - check network connectivity")
    
    if not results.get("Birdeye API", False):
        logger.info("🔧 Birdeye API failed - consider getting API key")
    
    if not results.get("Scanner Integration", False):
        logger.info("🔧 Scanner Integration failed - check the bot code initialization")

if __name__ == "__main__":
    asyncio.run(main())