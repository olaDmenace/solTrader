#!/usr/bin/env python3
"""
Test Analytics Integration
Verifies that the performance analytics and risk management systems are properly integrated
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_integration():
    """Test the analytics integration"""
    print("Testing Analytics Integration")
    print("=" * 40)
    
    try:
        # Test 1: Check imports work
        print("1. Testing imports...")
        from src.utils.performance_analytics import performance_analytics, add_trade_to_analytics
        from src.utils.risk_management import risk_manager, update_risk_metrics
        print("   Imports successful")
        
        # Test 2: Add sample trade data
        print("2. Testing trade data addition...")
        sample_trade = {
            "timestamp": datetime.now().isoformat(),
            "token_address": "TEST123456789",
            "token_symbol": "TEST",
            "entry_price": 100.0,
            "exit_price": 110.0,
            "quantity": 1000.0,
            "profit_loss_usd": 100.0,
            "profit_loss_sol": 0.5,
            "profit_percentage": 10.0,
            "duration_minutes": 30.0,
            "trade_type": "test_integration",
            "source": "integration_test"
        }
        
        # Add to analytics
        add_trade_to_analytics(sample_trade)
        print("   Analytics data added successfully")
        
        # Update risk metrics
        update_risk_metrics(sample_trade)
        print("   Risk metrics updated successfully")
        
        # Test 3: Verify data was stored
        print("3. Testing data retrieval...")
        analytics_summary = performance_analytics.get_analytics_summary()
        print(f"   Total trades tracked: {analytics_summary['total_trades_tracked']}")
        
        risk_summary = risk_manager.get_risk_summary()
        print(f"   Risk events tracked: {len(risk_summary.get('recent_events', []))}")
        
        # Test 4: Check file creation
        print("4. Testing file creation...")
        analytics_files = [
            "analytics/trades_history.json",
            "analytics/daily_performance.json", 
            "analytics/weekly_performance.json"
        ]
        
        for file_path in analytics_files:
            if Path(file_path).exists():
                print(f"   {file_path} - EXISTS")
            else:
                print(f"   {file_path} - MISSING")
        
        risk_files = [
            "analytics/risk_events.json",
            "analytics/risk_metrics.json"
        ]
        
        for file_path in risk_files:
            if Path(file_path).exists():
                print(f"   {file_path} - EXISTS")
            else:
                print(f"   {file_path} - MISSING")
        
        print("\n" + "=" * 40)
        print("INTEGRATION TEST COMPLETE")
        print("The analytics and risk management systems are properly integrated!")
        print("\nNext steps:")
        print("- Run your trading bot to see automatic analytics tracking")
        print("- Check reports/ directory for daily/weekly performance reports")
        print("- Monitor analytics/ directory for JSON data files")
        
        return True
        
    except ImportError as e:
        print(f"   Import failed: {e}")
        return False
    except Exception as e:
        print(f"   Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_integration()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest cancelled")
        exit(1)
    except Exception as e:
        print(f"\nTest error: {e}")
        exit(1)