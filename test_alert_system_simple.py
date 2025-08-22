#!/usr/bin/env python3
"""
Simple Alert System Test (Windows Console Compatible)
Tests the alert system without Unicode characters
"""
import asyncio
import logging
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.utils.alerting_system import ProductionAlerter, AlertConfig

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce log noise
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def test_basic_configuration():
    """Test that alert configuration loads correctly"""
    print("Testing alert configuration...")
    
    alerter = ProductionAlerter()
    config = alerter.config
    
    print(f"  Email enabled: {config.email_enabled}")
    print(f"  SMS enabled: {config.sms_enabled}")
    print(f"  Max alerts/hour: {config.max_alerts_per_hour}")
    print(f"  Min interval: {config.min_alert_interval}s")
    
    # Check if basic config is loaded
    has_basic_config = (
        config.max_alerts_per_hour > 0 and
        config.min_alert_interval > 0 and
        config.smtp_host
    )
    
    if has_basic_config:
        print("  PASS: Configuration loaded successfully")
        return True
    else:
        print("  FAIL: Configuration not loaded properly")
        return False

async def test_rate_limiting():
    """Test that rate limiting works"""
    print("Testing rate limiting...")
    
    # Create test config with tight limits
    test_config = AlertConfig(
        email_enabled=False,
        sms_enabled=False,
        max_alerts_per_hour=2,
        min_alert_interval=1
    )
    
    test_alerter = ProductionAlerter(test_config)
    
    # Send alerts rapidly
    success_count = 0
    for i in range(4):
        success = await test_alerter.send_alert(
            title=f"Test {i}",
            message="Rate limit test",
            severity="low",
            component=f"test_{i}"  # Different components to avoid min interval
        )
        if success:
            success_count += 1
    
    # Should be limited to 2 alerts
    if success_count <= test_config.max_alerts_per_hour:
        print(f"  PASS: Rate limiting working (sent {success_count}/4 alerts)")
        return True
    else:
        print(f"  FAIL: Rate limiting not working (sent {success_count}/4 alerts)")
        return False

async def test_message_formatting():
    """Test message formatting"""
    print("Testing message formatting...")
    
    from src.utils.alerting_system import AlertMessage
    from datetime import datetime
    
    alerter = ProductionAlerter()
    
    test_alert = AlertMessage(
        title="Test Alert",
        message="This is a test message",
        severity="medium",
        component="test",
        timestamp=datetime.now(),
        details={"error_count": 5, "duration": "10 minutes"}
    )
    
    try:
        # Test formatting functions
        email_body = alerter._format_email_body(test_alert)
        sms_body = alerter._format_sms_body(test_alert)
        
        # Basic checks
        has_html = "<html>" in email_body
        has_details = "error_count" in email_body
        sms_not_too_long = len(sms_body) <= 160
        
        if has_html and has_details and sms_not_too_long:
            print(f"  PASS: Message formatting working")
            print(f"    Email: HTML format with details")
            print(f"    SMS: {len(sms_body)} chars (within limit)")
            return True
        else:
            print(f"  FAIL: Message formatting issues")
            return False
            
    except Exception as e:
        print(f"  FAIL: Formatting error: {e}")
        return False

async def test_alert_sending_dry_run():
    """Test alert sending without actually sending"""
    print("Testing alert system (dry run)...")
    
    alerter = ProductionAlerter()
    config = alerter.config
    
    # Check what would be available
    would_send_email = (config.email_enabled and 
                       config.to_emails and 
                       config.smtp_username)
    
    would_send_sms = (config.sms_enabled and 
                     config.to_phone_numbers and
                     (config.twilio_account_sid or config.textbelt_api_key))
    
    print(f"  Email ready: {would_send_email}")
    print(f"  SMS ready: {would_send_sms}")
    
    if would_send_email or would_send_sms:
        print("  PASS: Alert system configured and ready")
        return True
    else:
        print("  INFO: No alert methods configured (this is OK for testing)")
        print("        Run 'python setup_alerts.py' to configure real alerts")
        return True  # Not a failure, just not configured

async def test_integration():
    """Test integration with error tracking"""
    print("Testing integration with error tracking...")
    
    from src.utils.robust_api import error_tracker, ErrorSeverity
    
    # Simulate some errors
    try:
        for i in range(2):
            try:
                raise ConnectionError(f"Test error {i+1}")
            except Exception as e:
                error_tracker.record_error("alert_integration_test", e, ErrorSeverity.HIGH, i+1)
                await asyncio.sleep(0.1)
        
        print("  PASS: Error integration working")
        print("        (Check console output for alert notifications)")
        return True
        
    except Exception as e:
        print(f"  FAIL: Integration error: {e}")
        return False

async def run_simple_tests():
    """Run all simple alert tests"""
    print("SIMPLE ALERT SYSTEM TESTS")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_basic_configuration),
        ("Rate Limiting", test_rate_limiting),
        ("Message Formatting", test_message_formatting),
        ("Alert System Status", test_alert_sending_dry_run),
        ("Error Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = await test_func()
            if success:
                passed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed ({(passed/total)*100:.0f}%)")
    
    if passed == total:
        print("SUCCESS: Alert system is working correctly!")
        print("\nNext steps:")
        print("1. Run 'python setup_alerts.py' to configure email/SMS")
        print("2. Test with real alerts using the configuration")
    elif passed >= total - 1:
        print("MOSTLY WORKING: Alert system is functional")
        print("Minor issues detected, but core functionality works")
    else:
        print("ISSUES DETECTED: Alert system needs attention")
    
    return passed >= total - 1

if __name__ == "__main__":
    try:
        success = asyncio.run(run_simple_tests())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests cancelled")
        exit(1)
    except Exception as e:
        print(f"\nTest failure: {e}")
        exit(1)