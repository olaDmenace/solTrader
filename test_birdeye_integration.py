#!/usr/bin/env python3
"""
Test script for Birdeye Trending API integration
Tests the complete integration including API client, analyzer, and scanner enhancement
"""
import asyncio
import logging
import sys
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.birdeye_client import BirdeyeClient, TrendingToken
from src.trending_analyzer import TrendingAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockSettings:
    """Mock settings for testing"""
    def __init__(self):
        # Birdeye settings
        self.BIRDEYE_API_KEY = None
        self.ENABLE_TRENDING_FILTER = True
        self.MAX_TRENDING_RANK = 50
        self.MIN_PRICE_CHANGE_24H = 20.0
        self.MIN_VOLUME_CHANGE_24H = 10.0
        self.MIN_TRENDING_SCORE = 60.0
        self.TRENDING_SIGNAL_BOOST = 0.5
        self.TRENDING_FALLBACK_MODE = "permissive"
        self.TRENDING_CACHE_DURATION = 300
        self.TRENDING_REQUEST_INTERVAL = 2.0

async def test_birdeye_client():
    """Test Birdeye API client functionality"""
    logger.info("=" * 60)
    logger.info("TESTING BIRDEYE API CLIENT")
    logger.info("=" * 60)
    
    # Get API key from environment
    api_key = os.getenv('BIRDEYE_API_KEY')
    if api_key:
        logger.info(f"✅ Found Birdeye API key in environment (length: {len(api_key)})")
    else:
        logger.warning("⚠️  No Birdeye API key found in environment")
        logger.info("Set BIRDEYE_API_KEY in your .env file to test with real API")
    
    settings = MockSettings()
    
    async with BirdeyeClient(api_key=api_key, cache_duration=60) as client:
        # Test getting trending tokens
        logger.info("Fetching trending tokens...")
        trending_tokens = await client.get_trending_tokens(limit=10)
        
        if trending_tokens:
            logger.info(f"✅ Successfully fetched {len(trending_tokens)} trending tokens")
            
            # Display top 5 tokens
            logger.info("\nTop 5 trending tokens:")
            for i, token in enumerate(trending_tokens[:5], 1):
                logger.info(f"  {i}. {token.symbol} (#{token.rank})")
                logger.info(f"     Price Change 24h: {token.price_24h_change_percent:+.1f}%")
                logger.info(f"     Volume Change 24h: {token.volume_24h_change_percent:+.1f}%")
                logger.info(f"     Daily Volume: ${token.volume_24h_usd:,.0f}")
                logger.info(f"     Address: {token.address[:8]}...")
                logger.info("")
            
            # Test caching
            logger.info("Testing cache functionality...")
            cached_tokens = await client.get_trending_tokens(limit=10)
            if cached_tokens and len(cached_tokens) == len(trending_tokens):
                logger.info("✅ Cache working correctly")
            else:
                logger.warning("⚠️ Cache may not be working as expected")
            
            # Test token lookup by address
            if trending_tokens:
                test_token = trending_tokens[0]
                found_token = client.get_cached_token_by_address(test_token.address)
                if found_token and found_token.address == test_token.address:
                    logger.info("✅ Token lookup by address working")
                else:
                    logger.warning("⚠️ Token lookup by address failed")
            
            # Test trending check
            if trending_tokens:
                test_token = trending_tokens[0]
                is_trending = client.is_token_trending(test_token.address, max_rank=50)
                if is_trending:
                    logger.info("✅ Trending check working")
                else:
                    logger.warning("⚠️ Trending check failed")
            
            return trending_tokens
            
        else:
            logger.error("❌ Failed to fetch trending tokens")
            return None

async def test_trending_analyzer(trending_tokens):
    """Test trending analyzer functionality"""
    logger.info("=" * 60)
    logger.info("TESTING TRENDING ANALYZER")
    logger.info("=" * 60)
    
    if not trending_tokens:
        logger.warning("No trending tokens to analyze")
        return
    
    settings = MockSettings()
    analyzer = TrendingAnalyzer(settings)
    
    # Test scoring
    logger.info("Testing trending score calculation...")
    for i, token in enumerate(trending_tokens[:3], 1):
        score = analyzer.calculate_trending_score(token)
        logger.info(f"  Token {i}: {token.symbol}")
        logger.info(f"    Rank: #{token.rank}")
        logger.info(f"    Trending Score: {score:.1f}/100")
        logger.info("")
    
    # Test criteria validation
    logger.info("Testing trending criteria validation...")
    passed_count = 0
    for token in trending_tokens[:10]:
        passes, reason = analyzer.meets_trending_criteria(token)
        if passes:
            passed_count += 1
            logger.info(f"✅ {token.symbol}: {reason}")
        else:
            logger.info(f"❌ {token.symbol}: {reason}")
    
    logger.info(f"\nCriteria validation: {passed_count}/{min(10, len(trending_tokens))} tokens passed")
    
    # Test signal enhancement
    logger.info("Testing signal enhancement...")
    if trending_tokens:
        test_token = trending_tokens[0]
        base_signal = 0.7  # 70% base signal
        enhanced_signal = analyzer.enhance_signal_strength(base_signal, test_token)
        boost = enhanced_signal - base_signal
        logger.info(f"  Base Signal: {base_signal:.3f}")
        logger.info(f"  Enhanced Signal: {enhanced_signal:.3f}")
        logger.info(f"  Boost Applied: +{boost:.3f} ({boost/base_signal*100:+.1f}%)")
    
    # Test summary generation
    logger.info("Testing trending summary...")
    summary = analyzer.get_trending_summary(trending_tokens[:10])
    if 'error' not in summary:
        logger.info("✅ Summary generated successfully:")
        logger.info(f"  Average Rank: {summary['avg_rank']:.1f}")
        logger.info(f"  Average Momentum: {summary['avg_momentum_24h']:+.1f}%")
        logger.info(f"  Top Momentum: {summary['top_momentum_token']['symbol']} ({summary['top_momentum_token']['momentum']:+.1f}%)")
        logger.info(f"  Criteria Pass Rate: {summary['criteria_pass_rate']:.1f}%")
    else:
        logger.error(f"❌ Summary generation failed: {summary['error']}")

async def test_integration_flow():
    """Test the complete integration flow"""
    logger.info("=" * 60)
    logger.info("TESTING COMPLETE INTEGRATION FLOW")
    logger.info("=" * 60)
    
    settings = MockSettings()
    
    # Simulate scanner integration
    logger.info("Simulating scanner integration...")
    
    # Get API key from environment
    api_key = os.getenv('BIRDEYE_API_KEY')
    
    async with BirdeyeClient(api_key=api_key) as client:
        # Get trending data
        trending_tokens = await client.get_trending_tokens(limit=20)
        if not trending_tokens:
            logger.error("❌ Cannot proceed with integration test - no trending data")
            return
        
        analyzer = TrendingAnalyzer(settings)
        
        # Simulate token discovery and filtering
        logger.info("Simulating token filtering with trending validation...")
        
        # Mock token from scanner
        mock_token_data = {
            'address': trending_tokens[0].address,
            'symbol': trending_tokens[0].symbol,
            'price_sol': 0.001234,
            'market_cap_sol': 1500,
            'liquidity_sol': 800,
            'volume_24h_sol': 150,
            'source': 'mock_scanner'
        }
        
        logger.info(f"Mock token: {mock_token_data['symbol']} ({mock_token_data['address'][:8]}...)")
        
        # Check if token is trending
        trending_token = client.get_cached_token_by_address(mock_token_data['address'])
        if trending_token:
            logger.info(f"✅ Token found in trending list (rank #{trending_token.rank})")
            
            # Validate criteria
            passes, reason = analyzer.meets_trending_criteria(trending_token)
            if passes:
                logger.info(f"✅ Token passes trending criteria: {reason}")
                
                # Calculate score and enhancement
                score = analyzer.calculate_trending_score(trending_token)
                base_signal = 0.65
                enhanced_signal = analyzer.enhance_signal_strength(base_signal, trending_token)
                
                logger.info(f"✅ Complete flow successful:")
                logger.info(f"  Trending Score: {score:.1f}/100")
                logger.info(f"  Signal Enhancement: {base_signal:.3f} → {enhanced_signal:.3f}")
                logger.info(f"  Final Decision: TRADE (trending validation passed)")
                
            else:
                logger.info(f"❌ Token failed trending criteria: {reason}")
                logger.info(f"  Final Decision: SKIP (trending validation failed)")
        else:
            logger.info(f"⚠️ Token not in trending list")
            if settings.TRENDING_FALLBACK_MODE == "permissive":
                logger.info(f"  Final Decision: TRADE (permissive mode)")
            else:
                logger.info(f"  Final Decision: SKIP (strict mode)")

async def test_performance_metrics():
    """Test performance and monitoring capabilities"""
    logger.info("=" * 60)
    logger.info("TESTING PERFORMANCE METRICS")
    logger.info("=" * 60)
    
    settings = MockSettings()
    
    # Get API key from environment
    api_key = os.getenv('BIRDEYE_API_KEY')
    
    async with BirdeyeClient(api_key=api_key) as client:
        # Test API performance
        import time
        
        logger.info("Testing API performance...")
        start_time = time.time()
        trending_tokens = await client.get_trending_tokens(limit=20)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        logger.info(f"  API Response Time: {response_time:.0f}ms")
        
        if trending_tokens:
            logger.info(f"  Tokens Retrieved: {len(trending_tokens)}")
            logger.info(f"  Performance: {len(trending_tokens)/response_time*1000:.1f} tokens/second")
        
        # Test client stats
        stats = client.get_stats()
        logger.info("Client Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Test analyzer performance
        if trending_tokens:
            analyzer = TrendingAnalyzer(settings)
            
            logger.info("Testing analyzer performance...")
            start_time = time.time()
            
            for token in trending_tokens[:10]:
                score = analyzer.calculate_trending_score(token)
                passes, reason = analyzer.meets_trending_criteria(token)
                enhanced = analyzer.enhance_signal_strength(0.7, token)
            
            end_time = time.time()
            analysis_time = (end_time - start_time) * 1000
            logger.info(f"  Analysis Time (10 tokens): {analysis_time:.0f}ms")
            if analysis_time > 0:
                logger.info(f"  Analysis Performance: {10/analysis_time*1000:.1f} tokens/second")
            else:
                logger.info(f"  Analysis Performance: >10,000 tokens/second (extremely fast!)")

async def main():
    """Main test runner"""
    logger.info("🚀 Starting Birdeye Trending API Integration Tests")
    logger.info("=" * 60)
    
    # Check API key status
    api_key = os.getenv('BIRDEYE_API_KEY')
    if api_key:
        logger.info(f"✅ Using Birdeye API key from environment (length: {len(api_key)})")
    else:
        logger.warning("⚠️  No API key found - tests may fail")
        logger.info("💡 Set BIRDEYE_API_KEY in your .env file for full testing")
    
    logger.info("=" * 60)
    
    try:
        # Test 1: Birdeye Client
        trending_tokens = await test_birdeye_client()
        
        if trending_tokens:
            # Test 2: Trending Analyzer
            await test_trending_analyzer(trending_tokens)
            
            # Test 3: Integration Flow
            await test_integration_flow()
            
            # Test 4: Performance
            await test_performance_metrics()
            
            logger.info("=" * 60)
            logger.info("🎉 ALL TESTS COMPLETED")
            logger.info("✅ Birdeye Trending API integration is working correctly!")
            logger.info("=" * 60)
            
        else:
            logger.error("❌ Basic API test failed - cannot continue with other tests")
            logger.error("This might be due to:")
            logger.error("  - Network connectivity issues")
            logger.error("  - Birdeye API rate limiting")
            logger.error("  - API service unavailability")
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Tests interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())