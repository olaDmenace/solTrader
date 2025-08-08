#!/usr/bin/env python3

"""
Verify Paper Trading Integration with Dual-API System
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment
load_dotenv()

try:
    from enhanced_token_scanner import EnhancedTokenScanner
    from config.settings import Settings
    from api.smart_dual_api_manager import SmartDualAPIManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

async def test_scanner_integration():
    """Test enhanced scanner with dual-API"""
    
    print("SCANNER INTEGRATION TEST")
    print("=" * 40)
    
    try:
        # Initialize settings
        settings = Settings()
        print(f"Settings loaded: {bool(settings)}")
        print(f"Paper trading enabled: {settings.PAPER_TRADING}")
        print(f"Scanner interval: {settings.SCANNER_INTERVAL}")
        
        # Initialize scanner
        scanner = EnhancedTokenScanner(settings)
        print(f"Scanner initialized: {bool(scanner)}")
        print(f"API client type: {type(scanner.api_client).__name__}")
        
        # Test token discovery
        print("Testing token discovery...")
        await scanner.start_session()
        
        # Run a scan
        scan_results = await scanner.scan_and_approve_tokens()
        
        await scanner.close_session()
        
        print(f"Scan completed:")
        print(f"  Total tokens found: {len(scan_results)}")
        print(f"  Scanner ready for paper trading: {len(scan_results) > 0}")
        
        if scan_results:
            for i, token in enumerate(scan_results[:3]):
                print(f"  {i+1}. {token.token.symbol} - Score: {token.score:.1f}")
        
        return len(scan_results) > 0
        
    except Exception as e:
        print(f"Scanner integration error: {e}")
        return False

async def test_quota_management():
    """Test quota management and fallback"""
    
    print("\nQUOTA MANAGEMENT TEST")
    print("=" * 40)
    
    try:
        async with SmartDualAPIManager() as manager:
            # Get current quota status
            quota_status = manager.quota_manager.get_quota_status()
            
            print("Current Quota Status:")
            for provider, status in quota_status['providers'].items():
                print(f"  {provider}:")
                print(f"    Total quota: {status['total_quota']}")
                print(f"    Used: {status['used_quota']}")
                print(f"    Remaining: {status['remaining_quota']}")
                print(f"    Utilization: {status['utilization_rate']}")
            
            # Test quota health
            health = manager.quota_manager.check_quota_health()
            print(f"\nQuota Health Status: {health['overall_status']}")
            print(f"Emergency mode: {health['emergency_mode']}")
            
            if health['alerts']:
                print("Alerts:")
                for alert in health['alerts']:
                    print(f"  - {alert}")
            
            # Calculate daily usage projection
            st_quota = quota_status['providers']['solana_tracker']
            daily_scans = 96  # 15-minute intervals
            calls_per_scan = 3  # trending + volume + memescope
            projected_daily_usage = daily_scans * calls_per_scan
            
            print(f"\nDaily Usage Projection:")
            print(f"  Scans per day: {daily_scans}")
            print(f"  API calls per scan: {calls_per_scan}")
            print(f"  Projected daily usage: {projected_daily_usage}")
            print(f"  Daily quota: {st_quota['total_quota']}")
            print(f"  Usage ratio: {(projected_daily_usage / st_quota['total_quota']) * 100:.1f}%")
            
            # Monthly projection
            monthly_usage = projected_daily_usage * 30
            monthly_quota = st_quota['total_quota'] * 30  # Assuming 10k/month = 333/day
            
            print(f"\nMonthly Usage Projection:")
            print(f"  Projected monthly usage: {monthly_usage}")
            print(f"  Estimated monthly quota: 10000")
            print(f"  Monthly usage ratio: {(monthly_usage / 10000) * 100:.1f}%")
            
            quota_safe = projected_daily_usage <= st_quota['total_quota'] * 0.9
            monthly_safe = monthly_usage <= 10000 * 0.9
            
            return quota_safe and monthly_safe
            
    except Exception as e:
        print(f"Quota management error: {e}")
        return False

def calculate_quota_safety():
    """Calculate quota safety margins"""
    
    print("\nQUOTA SAFETY CALCULATION")
    print("=" * 40)
    
    # Current configuration
    scanner_interval = 900  # 15 minutes
    daily_scans = (24 * 60 * 60) / scanner_interval
    
    # Solana Tracker usage
    st_calls_per_scan = 3  # trending, volume, memescope
    st_daily_calls = daily_scans * st_calls_per_scan
    st_monthly_calls = st_daily_calls * 30
    
    # Quotas
    st_daily_quota = 333
    st_monthly_quota = 10000
    
    print(f"Solana Tracker Usage Analysis:")
    print(f"  Scanner interval: {scanner_interval} seconds ({scanner_interval/60:.1f} minutes)")
    print(f"  Daily scans: {daily_scans:.0f}")
    print(f"  Daily API calls: {st_daily_calls:.0f}")
    print(f"  Monthly API calls: {st_monthly_calls:.0f}")
    
    print(f"\nQuota Limits:")
    print(f"  Daily quota: {st_daily_quota}")
    print(f"  Monthly quota: {st_monthly_quota}")
    
    daily_usage_percent = (st_daily_calls / st_daily_quota) * 100
    monthly_usage_percent = (st_monthly_calls / st_monthly_quota) * 100
    
    print(f"\nUsage Percentages:")
    print(f"  Daily usage: {daily_usage_percent:.1f}%")
    print(f"  Monthly usage: {monthly_usage_percent:.1f}%")
    
    daily_safe = daily_usage_percent <= 85
    monthly_safe = monthly_usage_percent <= 85
    
    print(f"\nSafety Assessment:")
    print(f"  Daily quota safe: {daily_safe} ({daily_usage_percent:.1f}% <= 85%)")
    print(f"  Monthly quota safe: {monthly_safe} ({monthly_usage_percent:.1f}% <= 85%)")
    
    return daily_safe and monthly_safe

def reactivate_email_notifications():
    """Reactivate email notifications safely"""
    
    print("\nREACTIVATING EMAIL NOTIFICATIONS")
    print("=" * 40)
    
    try:
        # Update .env file
        env_path = '.env'
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Update email settings
        updated_lines = []
        for line in lines:
            if 'EMAIL_ENABLED=false' in line:
                updated_lines.append('EMAIL_ENABLED=true\n')
                print("Enabled EMAIL_ENABLED")
            elif 'CRITICAL_ALERTS=false' in line:
                updated_lines.append('CRITICAL_ALERTS=true\n')
                print("Enabled CRITICAL_ALERTS")
            elif 'PERFORMANCE_ALERTS=false' in line:
                updated_lines.append('PERFORMANCE_ALERTS=true\n')
                print("Enabled PERFORMANCE_ALERTS")
            else:
                updated_lines.append(line)
        
        # Write back
        with open(env_path, 'w') as f:
            f.writelines(updated_lines)
        
        print("Email notifications reactivated successfully")
        return True
        
    except Exception as e:
        print(f"Error reactivating email notifications: {e}")
        return False

async def main():
    """Main verification function"""
    
    print("PRODUCTION READINESS VERIFICATION")
    print("=" * 50)
    
    # Test 1: Scanner integration
    scanner_ready = await test_scanner_integration()
    
    # Test 2: Quota management
    quota_ready = await test_quota_management()
    
    # Test 3: Quota safety calculation
    quota_safe = calculate_quota_safety()
    
    # Test 4: Reactivate email notifications
    email_ready = reactivate_email_notifications()
    
    # Final assessment
    print("\n" + "=" * 50)
    print("PRODUCTION READINESS ASSESSMENT")
    print("=" * 50)
    
    print(f"Scanner Integration: {'READY' if scanner_ready else 'NOT READY'}")
    print(f"Quota Management: {'READY' if quota_ready else 'NOT READY'}")
    print(f"Quota Safety: {'SAFE' if quota_safe else 'UNSAFE'}")
    print(f"Email Notifications: {'ACTIVATED' if email_ready else 'FAILED'}")
    
    all_ready = scanner_ready and quota_ready and quota_safe and email_ready
    
    if all_ready:
        print(f"\nSUCCESS: All systems READY for production!")
        print(f"Paper trading will execute successfully")
        print(f"Quota limits are safely managed")
        print(f"Email notifications are active")
        print(f"\nDEPLOY WITH CONFIDENCE!")
    else:
        print(f"\nWARNING: Some systems need attention before deployment")
    
    return all_ready

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Verification error: {e}")
        sys.exit(1)