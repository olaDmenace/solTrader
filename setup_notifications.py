#!/usr/bin/env python3
"""
Setup Trade Notifications
Configure email and webhook notifications for SolTrader
"""
import sys
import json
import getpass
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def setup_email_notifications():
    """Interactive setup for email notifications"""
    print("\nEmail Notification Setup")
    print("-" * 30)
    
    smtp_server = input("SMTP Server (default: smtp.gmail.com): ").strip()
    if not smtp_server:
        smtp_server = "smtp.gmail.com"
    
    smtp_port = input("SMTP Port (default: 587): ").strip()
    if not smtp_port:
        smtp_port = 587
    else:
        smtp_port = int(smtp_port)
    
    username = input("Email Username: ").strip()
    if not username:
        print("Username is required for email notifications")
        return False
    
    password = getpass.getpass("Email Password (hidden): ").strip()
    if not password:
        print("Password is required for email notifications")
        return False
    
    recipient = input("Recipient Email: ").strip()
    if not recipient:
        recipient = username  # Use sender as recipient if not specified
    
    sender_name = input("Sender Name (default: SolTrader Bot): ").strip()
    if not sender_name:
        sender_name = "SolTrader Bot"
    
    # Configure email
    try:
        from src.utils.trade_notifications import configure_email_notifications
        configure_email_notifications(smtp_server, smtp_port, username, password, recipient)
        print(f"Email notifications configured successfully!")
        print(f"Notifications will be sent to: {recipient}")
        return True
    except Exception as e:
        print(f"Failed to configure email notifications: {e}")
        return False

def setup_webhook_notifications():
    """Interactive setup for webhook notifications"""
    print("\nWebhook Notification Setup")
    print("-" * 30)
    
    url = input("Webhook URL: ").strip()
    if not url:
        print("URL is required for webhook notifications")
        return False
    
    method = input("HTTP Method (default: POST): ").strip().upper()
    if not method:
        method = "POST"
    
    # Optional custom headers
    print("Custom Headers (optional):")
    print("Enter headers in format 'Key: Value', press Enter with empty line to finish")
    headers = {'Content-Type': 'application/json'}
    
    while True:
        header = input("Header: ").strip()
        if not header:
            break
        
        if ':' in header:
            key, value = header.split(':', 1)
            headers[key.strip()] = value.strip()
        else:
            print("Invalid header format, use 'Key: Value'")
    
    # Configure webhook
    try:
        from src.utils.trade_notifications import configure_webhook_notifications
        configure_webhook_notifications(url, method, headers)
        print(f"Webhook notifications configured successfully!")
        print(f"Notifications will be sent to: {url}")
        return True
    except Exception as e:
        print(f"Failed to configure webhook notifications: {e}")
        return False

def test_notifications():
    """Send test notifications"""
    print("\nTesting Notifications")
    print("-" * 20)
    
    try:
        from src.utils.trade_notifications import (
            send_trade_opened,
            send_trade_closed,
            send_alert,
            send_status_update,
            start_notifications
        )
        
        # Start notification processing
        start_notifications()
        
        print("Sending test notifications...")
        
        # Test trade opened
        send_trade_opened(
            token_address="TEST_TOKEN_ADDRESS",
            symbol="TESTCOIN",
            entry_price=0.001234,
            quantity=50000,
            trade_type="test"
        )
        print("  Test trade opened notification sent")
        
        # Test trade closed
        send_trade_closed(
            token_address="TEST_TOKEN_ADDRESS",
            symbol="TESTCOIN",
            exit_price=0.001500,
            pnl=0.0133,
            pnl_percentage=21.55,
            duration_minutes=12.3
        )
        print("  Test trade closed notification sent")
        
        # Test alert
        send_alert(
            alert_type="Test Alert",
            message="This is a test notification to verify your setup is working correctly",
            severity="info"
        )
        print("  Test alert notification sent")
        
        # Test status update
        send_status_update(
            status="test",
            details="Notification system test completed successfully"
        )
        print("  Test status notification sent")
        
        print("\nTest notifications sent! Check your configured channels.")
        return True
        
    except Exception as e:
        print(f"Failed to send test notifications: {e}")
        return False

def show_current_config():
    """Display current notification configuration"""
    print("\nCurrent Notification Configuration")
    print("-" * 40)
    
    try:
        from src.utils.trade_notifications import trade_notifications
        
        for channel_name, channel in trade_notifications.channels.items():
            status = "ENABLED" if channel.enabled else "DISABLED"
            print(f"{channel_name.upper()}: {status}")
            
            if channel.enabled and channel_name == 'email':
                config = channel.config
                print(f"  Server: {config.get('smtp_server', 'Not set')}")
                print(f"  Port: {config.get('smtp_port', 'Not set')}")
                print(f"  Username: {config.get('username', 'Not set')}")
                print(f"  Recipient: {config.get('recipient', 'Not set')}")
            
            elif channel.enabled and channel_name == 'webhook':
                config = channel.config
                print(f"  URL: {config.get('url', 'Not set')}")
                print(f"  Method: {config.get('method', 'Not set')}")
        
        print()
        
    except Exception as e:
        print(f"Failed to load configuration: {e}")

def main():
    """Main setup function"""
    print("SolTrader Notification Setup")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Setup Email Notifications")
        print("2. Setup Webhook Notifications") 
        print("3. Show Current Configuration")
        print("4. Test Notifications")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            setup_email_notifications()
        elif choice == '2':
            setup_webhook_notifications()
        elif choice == '3':
            show_current_config()
        elif choice == '4':
            test_notifications()
        elif choice == '5':
            print("Setup complete!")
            break
        else:
            print("Invalid choice, please select 1-5")
    
    print("\nNotification Integration:")
    print("- Your trading bot will now send real-time notifications")
    print("- Notifications include trade events, alerts, and status updates")
    print("- Check your configured channels for incoming messages")
    print("- You can reconfigure notifications anytime by running this script")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup cancelled")
    except Exception as e:
        print(f"\nSetup error: {e}")
        import traceback
        traceback.print_exc()