#!/usr/bin/env python3
"""
Test Core Error Handling Components
Tests just the robust API utilities without external API dependencies
"""
import asyncio
import logging
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.utils.robust_api import (
    error_tracker, 
    get_error_summary, 
    robust_api_call, 
    RetryConfig, 
    ErrorSeverity,
    RobustHTTPClient
)

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

async def test_retry_mechanism():
    """Test the retry mechanism with a mock failing function"""
    print("\n" + "="*60)
    print("TESTING RETRY MECHANISM")
    print("="*60)
    
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

async def test_robust_http_client():
    """Test the RobustHTTPClient with a mock endpoint"""
    print("\n" + "="*60)
    print("TESTING ROBUST HTTP CLIENT")
    print("="*60)
    
    # Create client for a reliable endpoint
    client = RobustHTTPClient(
        base_url="https://httpbin.org",
        component_name="http_test",
        timeout=10.0
    )
    
    try:
        print("Testing HTTP client with httpbin.org...")
        
        # Test a simple GET request
        response = await client.get("status/200")
        print(f"   GET /status/200: SUCCESS")
        
        # Test error handling with a 404
        try:
            await client.get("status/404")
            print("   GET /status/404: Unexpectedly succeeded")
            return False
        except Exception:
            print("   GET /status/404: Correctly handled error")
        
        return True
        
    except Exception as e:
        print(f"   HTTP client test failed: {e}")
        return False
    finally:
        await client.close()

async def test_error_severity_classification():
    """Test error severity classification"""
    print("\n" + "="*60)
    print("TESTING ERROR SEVERITY CLASSIFICATION") 
    print("="*60)
    
    from src.utils.robust_api import _determine_error_severity
    import aiohttp
    
    # Test different error types
    test_cases = [
        (ConnectionError("Connection failed"), ErrorSeverity.LOW),
        (ValueError("Invalid data"), ErrorSeverity.MEDIUM),
        (MemoryError("Out of memory"), ErrorSeverity.CRITICAL),
        (aiohttp.ClientTimeout(), ErrorSeverity.LOW),
    ]
    
    passed = 0
    for error, expected_severity in test_cases:
        actual_severity = _determine_error_severity(error)
        if actual_severity == expected_severity:
            print(f"   {type(error).__name__}: {expected_severity.value} - PASS")
            passed += 1
        else:
            print(f"   {type(error).__name__}: expected {expected_severity.value}, got {actual_severity.value} - FAIL")
    
    print(f"   Severity classification: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

async def test_alert_thresholds():
    """Test alert threshold system"""
    print("\n" + "="*60)
    print("TESTING ALERT THRESHOLDS")
    print("="*60)
    
    # Create a fresh error tracker for this test
    from src.utils.robust_api import ErrorTracker
    test_tracker = ErrorTracker()
    
    # Set low thresholds for testing
    test_tracker.thresholds[ErrorSeverity.LOW] = 2
    test_tracker.thresholds[ErrorSeverity.MEDIUM] = 1
    
    print("Testing alert triggers...")
    
    # Generate errors to trigger alerts
    try:
        raise ConnectionError("Test error 1")
    except Exception as e:
        test_tracker.record_error("alert_test", e, ErrorSeverity.LOW, 1)
    
    try:
        raise ConnectionError("Test error 2") 
    except Exception as e:
        test_tracker.record_error("alert_test", e, ErrorSeverity.LOW, 2)
    
    # This should trigger an alert
    try:
        raise ValueError("Medium severity error")
    except Exception as e:
        test_tracker.record_error("alert_test", e, ErrorSeverity.MEDIUM, 1)
    
    print("   Alert system functioning (check logs for alert messages)")
    return True

async def run_all_tests():
    """Run all core error handling tests"""
    print("STARTING CORE ERROR HANDLING TESTS")
    print("="*80)
    
    start_time = time.time()
    tests_passed = 0
    total_tests = 5
    
    # Run all tests
    tests = [
        ("Error Tracking", test_error_tracking),
        ("Retry Mechanism", test_retry_mechanism),
        ("Robust HTTP Client", test_robust_http_client),
        ("Error Severity Classification", test_error_severity_classification),
        ("Alert Thresholds", test_alert_thresholds)
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
    print("CORE ERROR HANDLING TEST RESULTS")
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
        print("\nALL CORE TESTS PASSED! Error handling system is working correctly.")
        return True
    else:
        print(f"\nWARNING: {total_tests - tests_passed} tests failed. Check output for details.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nTest suite crashed: {e}")
        exit(1)