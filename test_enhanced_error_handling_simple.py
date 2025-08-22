#!/usr/bin/env python3
"""
Test Enhanced Error Handling and Retry Logic (Windows Console Compatible)
Verifies the robust API utilities and enhanced clients work correctly
"""
import asyncio
import logging
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.utils.robust_api import error_tracker, get_error_summary
from src.api.enhanced_solana_tracker import EnhancedSolanaTrackerClient
from src.api.enhanced_jupiter import EnhancedJupiterClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_error_tracking():
    """Test the error tracking system"""
    print("\n" + "="*60)
    print("TESTING ERROR TRACKING SYSTEM")
    print("="*60)
    
    # Simulate some errors
    from src.utils.robust_api import ErrorSeverity
    
    # Simulate low severity errors
    for i in range(3):
        try:
            raise ConnectionError(f"Simulated connection error {i+1}")
        except Exception as e:
            error_tracker.record_error("test_component", e, ErrorSeverity.LOW, i+1)
    
    # Simulate medium severity error
    try:
        raise ValueError("Simulated parsing error")
    except Exception as e:
        error_tracker.record_error("test_component", e, ErrorSeverity.MEDIUM, 1)
    
    # Get error summary
    summary = get_error_summary(hours=1)
    
    print(f"Error Summary (last 1 hour):")
    print(f"   Total errors: {summary['total_errors']}")
    print(f"   By component: {summary['by_component']}")
    print(f"   By severity: {summary['by_severity']}")
    print(f"   Unresolved: {summary['unresolved_count']}")
    
    # Test recovery
    error_tracker.record_recovery("test_component", 3)
    
    return True

async def test_enhanced_solana_tracker():
    """Test the enhanced Solana Tracker client"""
    print("\n" + "="*60)
    print("TESTING ENHANCED SOLANA TRACKER CLIENT")
    print("="*60)
    
    client = EnhancedSolanaTrackerClient()
    
    try:
        # Test connection
        print("Testing connection...")
        connection_ok = await client.test_connection()
        print(f"   Connection: {'OK' if connection_ok else 'FAILED'}")
        
        # Test trending tokens
        print("Testing trending tokens...")
        trending = await client.get_trending_tokens(limit=5)
        print(f"   Got {len(trending)} trending tokens")
        
        for token in trending[:3]:
            print(f"   - {token.symbol}: ${token.price:.6f} ({token.price_change_24h:+.2f}%)")
        
        # Test client status
        print("Client status:")
        status = client.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced Solana Tracker test failed: {e}")
        return False
    finally:
        await client.close()

async def test_enhanced_jupiter():
    """Test the enhanced Jupiter client"""
    print("\n" + "="*60)
    print("TESTING ENHANCED JUPITER CLIENT")
    print("="*60)
    
    client = EnhancedJupiterClient()
    
    try:
        # Test connection
        print("Testing connection...")
        connection_ok = await client.test_connection()
        print(f"   Connection: {'OK' if connection_ok else 'FAILED'}")
        
        # Test SOL price
        print("Testing SOL price...")
        sol_price = await client.get_sol_price_in_usdc()
        if sol_price:
            print(f"   SOL price: ${sol_price:.2f}")
        
        # Test token validation
        print("Testing token validation...")
        sol_valid = await client.validate_token_tradeable("So11111111111111111111111111111111111111112")
        print(f"   SOL tradeable: {'YES' if sol_valid else 'NO'}")
        
        # Test quote
        print("Testing quote...")
        quote = await client.get_quote(
            "So11111111111111111111111111111111111111112",  # SOL
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            100000000  # 0.1 SOL
        )
        
        if quote:
            print(f"   Quote received: {quote.get('outAmount', 'N/A')} USDC")
            print(f"   Price impact: {quote.get('priceImpactPct', 0):.4f}%")
        
        # Test client status
        print("Client status:")
        status = client.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced Jupiter test failed: {e}")
        return False
    finally:
        await client.close()

async def test_retry_mechanism():
    """Test the retry mechanism with a mock failing function"""
    print("\n" + "="*60)
    print("TESTING RETRY MECHANISM")
    print("="*60)
    
    from src.utils.robust_api import robust_api_call, RetryConfig
    
    # Create a function that fails a few times then succeeds
    attempt_count = 0
    
    @robust_api_call(
        config=RetryConfig(max_attempts=4, base_delay=0.1),
        component="retry_test"
    )
    async def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        
        if attempt_count < 3:
            raise ConnectionError(f"Simulated failure on attempt {attempt_count}")
        
        return {"success": True, "attempts": attempt_count}
    
    try:
        print("Testing function that fails 2 times then succeeds...")
        result = await flaky_function()
        print(f"   SUCCESS after {result['attempts']} attempts")
        return True
        
    except Exception as e:
        logger.error(f"Retry test failed: {e}")
        return False

async def test_all_systems():
    """Run all error handling tests"""
    print("STARTING COMPREHENSIVE ERROR HANDLING TESTS")
    print("="*80)
    
    start_time = time.time()
    tests_passed = 0
    total_tests = 4
    
    # Run all tests
    tests = [
        ("Error Tracking", test_error_tracking),
        ("Enhanced Solana Tracker", test_enhanced_solana_tracker),
        ("Enhanced Jupiter", test_enhanced_jupiter),
        ("Retry Mechanism", test_retry_mechanism)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name}...")
            success = await test_func()
            if success:
                print(f"   PASS: {test_name}")
                tests_passed += 1
            else:
                print(f"   FAIL: {test_name}")
        except Exception as e:
            print(f"   CRASH: {test_name}: {e}")
    
    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {(tests_passed/total_tests)*100:.1f}%")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    
    # Show final error summary
    print(f"\nFinal Error Summary:")
    summary = get_error_summary(hours=1)
    print(f"   Total errors recorded: {summary['total_errors']}")
    print(f"   Components affected: {len(summary['by_component'])}")
    print(f"   Severity breakdown: {summary['by_severity']}")
    
    if tests_passed == total_tests:
        print("\nALL TESTS PASSED! Error handling is working correctly.")
        return True
    else:
        print(f"\nWARNING: {total_tests - tests_passed} tests failed. Check logs for details.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_all_systems())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nTest suite crashed: {e}")
        exit(1)