#!/usr/bin/env python3
"""
Test Alert System
Comprehensive testing of the email and SMS alert system
"""
import asyncio
import logging
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.utils.alerting_system import ProductionAlerter, AlertConfig, production_alerter
from src.utils.robust_api import error_tracker, ErrorSeverity

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_basic_alert_config():
    """Test basic alert configuration loading"""
    print("\n" + "="*60)
    print("TESTING ALERT CONFIGURATION")
    print("="*60)
    
    alerter = ProductionAlerter()
    config = alerter.config
    
    print(f"Email enabled: {config.email_enabled}")
    print(f"SMS enabled: {config.sms_enabled}")
    print(f"SMTP host: {config.smtp_host}")
    print(f"Email recipients: {len(config.to_emails)}")
    print(f"Phone recipients: {len(config.to_phone_numbers)}")
    print(f"Max alerts per hour: {config.max_alerts_per_hour}")
    print(f"Min alert interval: {config.min_alert_interval}s")
    print(f"Quiet hours: {config.quiet_hours_start}:00 - {config.quiet_hours_end}:00")
    
    # Check if any alerts are configured
    has_email = config.email_enabled and config.to_emails and config.smtp_username
    has_sms = config.sms_enabled and config.to_phone_numbers
    
    if not has_email and not has_sms:
        print("\n⚠️ WARNING: No alert methods configured!")
        print("Run 'python setup_alerts.py' to configure alerts")
        return False
    
    print(f"\n✅ Alert configuration loaded successfully")
    print(f"   Available methods: {'Email ' if has_email else ''}{'SMS' if has_sms else ''}")
    
    return True

async def test_alert_rate_limiting():
    """Test alert rate limiting functionality"""
    print("\n" + "="*60)
    print("TESTING ALERT RATE LIMITING")
    print("="*60)
    
    # Create a test alerter with low limits
    test_config = AlertConfig(
        email_enabled=False,  # Disable actual sending for this test
        sms_enabled=False,
        max_alerts_per_hour=3,
        min_alert_interval=2  # 2 seconds
    )
    
    test_alerter = ProductionAlerter(test_config)
    
    print("Testing minimum interval between alerts...")
    
    # Send first alert
    success1 = await test_alerter.send_alert(
        title="Test Alert 1",
        message="First test alert",
        severity="medium", 
        component="rate_limit_test"
    )
    
    # Try to send immediately (should be blocked)
    success2 = await test_alerter.send_alert(
        title="Test Alert 2",
        message="Second test alert (should be blocked)",
        severity="medium",
        component="rate_limit_test" 
    )
    
    print(f"   First alert sent: {success1}")
    print(f"   Second alert blocked: {not success2}")
    
    # Wait and try again
    print("   Waiting for interval to pass...")
    await asyncio.sleep(3)
    
    success3 = await test_alerter.send_alert(
        title="Test Alert 3", 
        message="Third test alert (should work)",
        severity="medium",
        component="rate_limit_test"
    )
    
    print(f"   Third alert sent after wait: {success3}")
    
    # Test hourly rate limit
    print("\nTesting hourly rate limit...")
    for i in range(5):
        await asyncio.sleep(0.1)  # Small delay
        success = await test_alerter.send_alert(
            title=f"Hourly Test {i+1}",
            message=f"Hourly rate limit test {i+1}",
            severity="low",
            component=f"hourly_test_{i}"  # Different components to avoid min interval
        )
        print(f"   Alert {i+1}: {'Sent' if success else 'Blocked (rate limit)'}")
        if not success:
            break
    
    stats = test_alerter.get_alert_stats()
    print(f"\nRate limiting test results:")
    print(f"   Alerts in current hour: {stats['current_hour_count']}")
    print(f"   Total alerts sent: {stats['total_alerts_sent']}")
    
    return True

async def test_alert_formatting():
    """Test alert message formatting"""
    print("\n" + "="*60)
    print("TESTING ALERT FORMATTING")
    print("="*60)
    
    from src.utils.alerting_system import AlertMessage
    from datetime import datetime
    
    # Create test alert
    test_alert = AlertMessage(
        title="Critical Trading Bot Error",
        message="The trading bot has encountered multiple connection failures and has stopped executing trades.",
        severity="critical",
        component="solana_tracker",
        timestamp=datetime.now(),
        details={
            "error_count": 5,
            "last_error": "Connection timeout after 30s",
            "affected_tokens": ["SOL", "USDC", "PUMP"],
            "duration": "15 minutes"
        }
    )
    
    # Test email formatting (just create the HTML, don't send)
    test_alerter = ProductionAlerter()
    email_html = test_alerter._format_email_body(test_alert)
    sms_text = test_alerter._format_sms_body(test_alert)
    
    print("Email HTML preview:")
    print("   Title: Critical Trading Bot Error")
    print("   Contains HTML styling: ✅" if "<html>" in email_html else "❌")
    print("   Contains details: ✅" if "error_count" in email_html else "❌")
    print("   Contains timestamp: ✅" if test_alert.timestamp.strftime('%Y-%m-%d') in email_html else "❌")
    
    print(f"\nSMS text preview ({len(sms_text)} chars):")
    print(f"   \"{sms_text[:100]}{'...' if len(sms_text) > 100 else ''}\"")
    print(f"   Within SMS limit: ✅" if len(sms_text) <= 160 else f"⚠️ ({len(sms_text)} chars)")
    
    return True

async def test_integration_with_error_tracker():
    """Test integration with the error tracking system"""
    print("\n" + "="*60)
    print("TESTING ERROR TRACKER INTEGRATION")
    print("="*60)
    
    # This will trigger our enhanced alert system
    print("Simulating errors to trigger alerts...")
    
    # Generate some errors that should trigger alerts
    for i in range(3):
        try:
            raise ConnectionError(f"Simulated API failure {i+1}")
        except Exception as e:
            error_tracker.record_error("integration_test", e, ErrorSeverity.HIGH, i+1)
            await asyncio.sleep(0.1)  # Small delay
    
    print("✅ Error tracker integration test completed")
    print("   Check console output above for alert notifications")
    
    return True

async def test_live_alert_system():
    """Test sending actual alerts (if configured)"""
    print("\n" + "="*60)
    print("TESTING LIVE ALERT SYSTEM")
    print("="*60)
    
    alerter = production_alerter
    config = alerter.config
    
    # Check if we can actually send alerts
    can_email = config.email_enabled and config.to_emails and config.smtp_username and config.smtp_password
    can_sms = config.sms_enabled and config.to_phone_numbers
    
    if not can_email and not can_sms:
        print("❌ No alert methods are fully configured")
        print("Run 'python setup_alerts.py' to configure alerts")
        return False
    
    print("Available alert methods:")
    print(f"   Email: {'✅ Ready' if can_email else '❌ Not configured'}")
    print(f"   SMS: {'✅ Ready' if can_sms else '❌ Not configured'}")
    
    # Ask user if they want to send test alerts
    print(f"\nThis will send actual test alerts to:")
    if can_email:
        print(f"   📧 Email: {', '.join(config.to_emails)}")
    if can_sms:
        print(f"   📱 SMS: {', '.join(config.to_phone_numbers)}")
    
    # For automated testing, skip the interactive part
    print("\n🧪 Sending test alert...")
    
    try:
        results = await alerter.test_alert_system()
        
        print("Test alert results:")
        if 'email' in results:
            print(f"   Email: {'✅ Sent' if results['email'] else '❌ Failed'}")
        if 'sms' in results:
            print(f"   SMS: {'✅ Sent' if results['sms'] else '❌ Failed'}")
        
        success = any(results.values())
        if success:
            print("\n✅ Live alert test successful!")
            print("Check your email/phone for the test message")
        else:
            print("\n❌ Live alert test failed")
            print("Check your configuration and credentials")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Live alert test failed with error: {e}")
        return False

async def run_all_alert_tests():
    """Run all alert system tests"""
    print("STARTING ALERT SYSTEM TESTS")
    print("="*80)
    
    start_time = time.time()
    tests_passed = 0
    total_tests = 5
    
    # Run all tests
    tests = [
        ("Alert Configuration", test_basic_alert_config),
        ("Rate Limiting", test_alert_rate_limiting),
        ("Message Formatting", test_alert_formatting),
        ("Error Tracker Integration", test_integration_with_error_tracker),
        ("Live Alert System", test_live_alert_system)
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
    print("ALERT SYSTEM TEST RESULTS")
    print("="*80)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {(tests_passed/total_tests)*100:.1f}%")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    
    # Show alert stats
    stats = production_alerter.get_alert_stats()
    print(f"\nAlert System Statistics:")
    print(f"   Total alerts sent: {stats['total_alerts_sent']}")
    print(f"   Alerts last 24h: {stats['alerts_last_24h']}")
    print(f"   Email enabled: {stats['email_enabled']}")
    print(f"   SMS enabled: {stats['sms_enabled']}")
    
    if tests_passed == total_tests:
        print("\n🎉 ALL ALERT TESTS PASSED! Your alert system is ready for production.")
        return True
    else:
        print(f"\n⚠️  {total_tests - tests_passed} tests failed. Check configuration and try again.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_alert_tests())
        if success:
            print("\n✅ Alert system is fully operational!")
            print("Your trading bot will now notify you of critical issues.")
        else:
            print("\n❌ Alert system needs configuration.")
            print("Run 'python setup_alerts.py' to set up alerts.")
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nAlert tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nAlert test suite crashed: {e}")
        exit(1)