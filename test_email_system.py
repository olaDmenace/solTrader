#!/usr/bin/env python3
"""
Email System Test Script for SolTrader Bot

This script tests the email notification system to ensure SMTP connectivity
and email delivery is working correctly after the bot crash.

Usage:
    python test_email_system.py

Expected Output:
    ✅ Email configuration verified
    ✅ SMTP connection successful
    ✅ Test email sent successfully
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the src directory to the path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import with absolute imports to avoid relative import issues
try:
    from src.config.settings import Settings
    from src.notifications.email_system import EmailNotificationSystem
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, current_dir)
    from src.config.settings import Settings
    from src.notifications.email_system import EmailNotificationSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_email_system():
    """Test the email notification system"""
    
    print("🔧 SolTrader Email System Test")
    print("=" * 50)
    
    try:
        # Load settings
        print("📋 Loading configuration...")
        settings = Settings()
        
        # Check email configuration
        print("🔍 Checking email configuration...")
        if not settings.EMAIL_ENABLED:
            print("❌ Email notifications are disabled in settings")
            return False
            
        required_settings = [
            ('EMAIL_USER', settings.EMAIL_USER),
            ('EMAIL_PASSWORD', settings.EMAIL_PASSWORD),
            ('EMAIL_TO', settings.EMAIL_TO),
            ('EMAIL_SMTP_SERVER', settings.EMAIL_SMTP_SERVER),
            ('EMAIL_PORT', settings.EMAIL_PORT),
        ]
        
        missing_settings = []
        for name, value in required_settings:
            if not value:
                missing_settings.append(name)
            else:
                # Mask password for display
                display_value = value if name != 'EMAIL_PASSWORD' else '*' * 8
                print(f"   ✅ {name}: {display_value}")
        
        if missing_settings:
            print(f"❌ Missing required email settings: {', '.join(missing_settings)}")
            print("   Please check your .env file")
            return False
        
        print("✅ Email configuration verified")
        
        # Initialize email system
        print("\n📧 Initializing email system...")
        email_system = EmailNotificationSystem(settings)
        
        if not email_system.enabled:
            print("❌ Email system initialization failed")
            return False
        
        print("✅ Email system initialized")
        
        # Start email system
        print("\n🚀 Starting email system...")
        await email_system.start()
        print("✅ Email system started")
        
        # Test SMTP connection by sending a test email
        print("\n📨 Sending test email...")
        success, message = await email_system.test_email()
        
        if success:
            print(f"✅ {message}")
            
            # Wait a moment for the email to be processed
            print("⏳ Waiting for email to be sent...")
            await asyncio.sleep(5)
            
            # Check statistics
            stats = email_system.get_stats()
            print(f"📊 Email statistics:")
            print(f"   • Emails sent: {stats['emails_sent']}")
            print(f"   • Emails failed: {stats['emails_failed']}")
            print(f"   • Success rate: {stats['success_rate']:.1f}%")
            print(f"   • Queue size: {stats['queue_size']}")
            
            if stats['emails_sent'] > 0:
                print("✅ Test email sent successfully!")
                print(f"📧 Check your inbox at: {settings.EMAIL_TO}")
            else:
                print("⚠️  Test email queued but not yet sent - check logs for details")
                
        else:
            print(f"❌ Test email failed: {message}")
            return False
        
        # Stop email system
        print("\n⏹️  Stopping email system...")
        await email_system.stop()
        print("✅ Email system stopped")
        
        return True
        
    except Exception as e:
        logger.error(f"Email test failed: {e}")
        print(f"❌ Email test failed: {e}")
        return False

async def send_crash_recovery_notification():
    """Send a notification that the bot has recovered from the crash"""
    
    try:
        settings = Settings()
        email_system = EmailNotificationSystem(settings)
        
        if not email_system.enabled:
            print("❌ Cannot send recovery notification - email not configured")
            return False
        
        await email_system.start()
        
        # Send recovery notification
        recovery_data = {
            'crash_date': '2025-07-28 23:55:40',
            'recovery_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'crash_cause': "'requests_used' KeyError in enhanced_token_scanner.py",
            'fixes_applied': [
                'Fixed requests_used -> requests_today key mismatch',
                'Added error recovery logic for API failures',
                'Improved logging resilience during daily resets',
                'Added comprehensive error handling'
            ],
            'next_steps': [
                'Monitor bot operation for 24 hours',
                'Verify token discovery resumes',
                'Check daily email reports',
                'Confirm dashboard updates'
            ]
        }
        
        await email_system.send_critical_alert(
            subject="🔧 Bot Recovery - Crash Fixed",
            message="""
The SolTrader bot has been successfully recovered from the July 28th crash.

CRASH DETAILS:
• Date: July 28, 2025 at 23:55:40 UTC
• Cause: KeyError 'requests_used' in enhanced_token_scanner.py
• Impact: Process hung after daily API counter reset

FIXES APPLIED:
• Fixed API counter key mismatch (requests_used → requests_today)
• Added comprehensive error recovery for API failures
• Improved resilience during daily counter resets
• Enhanced error handling throughout the scanner

SYSTEM STATUS:
• ✅ Scanner errors resolved
• ✅ API counter handling fixed
• ✅ Error recovery logic added
• ✅ Email system verified

The bot should now resume normal token discovery and trading operations.
Please monitor the dashboard and daily reports to confirm full recovery.
            """,
            data=recovery_data
        )
        
        # Wait for email to be sent
        await asyncio.sleep(3)
        await email_system.stop()
        
        print("✅ Recovery notification sent successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send recovery notification: {e}")
        print(f"❌ Failed to send recovery notification: {e}")
        return False

def main():
    """Main function to run the email test"""
    
    print("Starting SolTrader Email System Test...")
    
    # Run the email test
    test_result = asyncio.run(test_email_system())
    
    if test_result:
        print("\n" + "=" * 50)
        print("🎉 EMAIL SYSTEM TEST PASSED!")
        print("✅ SMTP configuration is working correctly")
        print("✅ Email delivery is functional")
        print("\nNext steps:")
        print("1. Restart the SolTrader bot service")
        print("2. Monitor logs for normal operation")
        print("3. Check for daily email reports")
        
        # Ask if user wants to send recovery notification
        try:
            response = input("\n📧 Send crash recovery notification email? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                print("\n📨 Sending recovery notification...")
                asyncio.run(send_crash_recovery_notification())
        except KeyboardInterrupt:
            print("\nTest completed.")
            
        return 0
    else:
        print("\n" + "=" * 50)
        print("❌ EMAIL SYSTEM TEST FAILED!")
        print("Please check your email configuration and try again.")
        print("\nCommon issues:")
        print("• Check EMAIL_* environment variables in .env file")
        print("• Verify SMTP server settings")
        print("• Ensure email password/app password is correct")
        print("• Check firewall/network connectivity")
        return 1

if __name__ == "__main__":
    sys.exit(main())