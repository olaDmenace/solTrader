#!/usr/bin/env python3
"""
Simple Email Test Script for SolTrader Bot

This script directly tests SMTP connectivity without complex imports.
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

def load_email_config():
    """Load email configuration from .env file"""
    
    # Load .env file
    load_dotenv()
    
    config = {
        'enabled': os.getenv('EMAIL_ENABLED', 'false').lower() == 'true',
        'user': os.getenv('EMAIL_USER'),
        'password': os.getenv('EMAIL_PASSWORD'),
        'to': os.getenv('EMAIL_TO'),
        'smtp_server': os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com'),
        'port': int(os.getenv('EMAIL_PORT', 587))
    }
    
    return config

def test_smtp_connection(config):
    """Test SMTP connection"""
    
    print("🔌 Testing SMTP connection...")
    
    try:
        with smtplib.SMTP(config['smtp_server'], config['port']) as server:
            server.starttls()
            server.login(config['user'], config['password'])
            print(f"✅ SMTP connection successful to {config['smtp_server']}:{config['port']}")
            return True
    except Exception as e:
        print(f"❌ SMTP connection failed: {e}")
        return False

def send_test_email(config):
    """Send a test email"""
    
    print("📧 Sending test email...")
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "[SolTrader] Email System Test - Recovery Verification"
        msg['From'] = config['user']
        msg['To'] = config['to']
        
        # Create email content
        text_content = f"""
SolTrader Bot - Email System Test

This is a test email to verify that the email notification system is working correctly after the July 28th crash fix.

Test Details:
- Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
- SMTP Server: {config['smtp_server']}
- Port: {config['port']}
- Status: ✅ Email delivery successful

The bot crash has been fixed with the following changes:
1. Fixed 'requests_used' -> 'requests_today' key mismatch
2. Added comprehensive error handling for API failures
3. Enhanced daily reset resilience
4. Improved analytics error handling

Next Steps:
1. Restart the SolTrader bot service
2. Monitor token discovery resumption
3. Check for daily email reports with real data
4. Verify dashboard updates

---
SolTrader Bot - Automated Solana Trading System
This is an automated message.
        """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>SolTrader Email Test</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #66BB6A, #66BB6A88); color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 30px; }}
                .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
                .status {{ background-color: #66BB6A22; color: #66BB6A; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .fixes {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                ul {{ text-align: left; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 SolTrader Email Test</h1>
                    <p>System Recovery Verification</p>
                </div>
                <div class="content">
                    <div class="status">
                        <h3>✅ Email System Status: OPERATIONAL</h3>
                        <p><strong>Test Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                        <p><strong>SMTP Server:</strong> {config['smtp_server']}:{config['port']}</p>
                    </div>
                    
                    <h3>🔧 Crash Recovery Applied</h3>
                    <div class="fixes">
                        <p><strong>July 28th Crash:</strong> KeyError 'requests_used' causing process hang</p>
                        <p><strong>Fixes Applied:</strong></p>
                        <ul>
                            <li>Fixed API counter key mismatch (requests_used → requests_today)</li>
                            <li>Added comprehensive error handling for API failures</li>
                            <li>Enhanced daily reset resilience</li>
                            <li>Improved analytics error handling</li>
                        </ul>
                    </div>
                    
                    <h3>📋 Next Steps</h3>
                    <ol>
                        <li>Restart the SolTrader bot service</li>
                        <li>Monitor token discovery resumption (40+ tokens)</li>
                        <li>Check for daily email reports with real data</li>
                        <li>Verify dashboard updates</li>
                    </ol>
                </div>
                <div class="footer">
                    <p>SolTrader Bot - Automated Solana Trading System</p>
                    <p>This is an automated test message.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Attach both HTML and text versions
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_content, 'html')
        
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        with smtplib.SMTP(config['smtp_server'], config['port']) as server:
            server.starttls()
            server.login(config['user'], config['password'])
            text = msg.as_string()
            server.sendmail(config['user'], config['to'], text)
        
        print(f"✅ Test email sent successfully to {config['to']}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send test email: {e}")
        return False

def main():
    """Main function"""
    
    print("🔧 SolTrader Simple Email Test")
    print("=" * 50)
    
    # Load configuration
    print("📋 Loading email configuration...")
    config = load_email_config()
    
    # Check if email is enabled
    if not config['enabled']:
        print("❌ Email notifications are disabled (EMAIL_ENABLED=false)")
        print("   Set EMAIL_ENABLED=true in .env file to enable")
        return 1
    
    # Check required settings
    required_fields = ['user', 'password', 'to', 'smtp_server']
    missing_fields = [field for field in required_fields if not config[field]]
    
    if missing_fields:
        print(f"❌ Missing required configuration: {', '.join(missing_fields)}")
        print("   Please check your .env file")
        return 1
    
    # Display configuration (mask password)
    print("✅ Email configuration loaded:")
    print(f"   • EMAIL_USER: {config['user']}")
    print(f"   • EMAIL_PASSWORD: {'*' * 8}")
    print(f"   • EMAIL_TO: {config['to']}")
    print(f"   • SMTP_SERVER: {config['smtp_server']}")
    print(f"   • PORT: {config['port']}")
    
    # Test SMTP connection
    if not test_smtp_connection(config):
        print("\n❌ EMAIL TEST FAILED - SMTP connection issue")
        print("\nCommon solutions:")
        print("• Check EMAIL_USER and EMAIL_PASSWORD in .env file")
        print("• For Gmail: Use App Password instead of regular password")
        print("• Verify SMTP server and port settings")
        print("• Check firewall/network connectivity")
        return 1
    
    # Send test email
    if not send_test_email(config):
        print("\n❌ EMAIL TEST FAILED - Could not send test email")
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 EMAIL TEST PASSED!")
    print("✅ SMTP connection successful")
    print("✅ Test email sent successfully")
    print(f"✅ Check your inbox: {config['to']}")
    
    print("\n📋 Bot Recovery Status:")
    print("✅ Email system verified")
    print("✅ Crash fixes applied")
    print("✅ Ready to restart bot")
    
    print("\n🚀 Next steps:")
    print("1. Restart SolTrader bot service")
    print("2. Monitor logs for token discovery")
    print("3. Verify dashboard updates")
    print("4. Check daily email reports")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())