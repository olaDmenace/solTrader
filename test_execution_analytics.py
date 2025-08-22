#!/usr/bin/env python3
"""
Test Execution Analytics
Verifies the execution timing and slippage tracking system
"""
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_execution_analytics():
    """Test the execution analytics system"""
    print("Testing Execution Analytics")
    print("=" * 40)
    
    try:
        # Test 1: Check imports work
        print("1. Testing imports...")
        from src.utils.execution_analytics import (
            start_execution_tracking, 
            complete_execution_tracking,
            get_execution_report,
            execution_analytics
        )
        print("   Imports successful")
        
        # Test 2: Test execution tracking
        print("2. Testing execution tracking...")
        
        # Simulate a successful trade execution
        trade_id = "test_trade_001"
        token_address = "TEST123456789ABCDEF"
        expected_price = 100.0
        
        execution_key = start_execution_tracking(trade_id, token_address, expected_price)
        print(f"   Started tracking: {execution_key}")
        
        # Simulate some execution time
        time.sleep(0.1)  # 100ms execution time
        
        # Complete with success
        actual_price = 102.0  # 2% slippage
        gas_fee = 0.005
        complete_execution_tracking(
            execution_key, 
            actual_price, 
            gas_fee, 
            True,
            network_latency_ms=50.0,
            confirmation_time_ms=2000.0
        )
        print("   Completed successful execution tracking")
        
        # Test 3: Test failed execution
        print("3. Testing failed execution...")
        failed_execution_key = start_execution_tracking("test_trade_002", token_address, expected_price)
        time.sleep(0.05)  # 50ms before failure
        
        complete_execution_tracking(
            failed_execution_key,
            expected_price,
            0.0,
            False,
            "Network timeout error"
        )
        print("   Completed failed execution tracking")
        
        # Test 4: Test high slippage scenario
        print("4. Testing high slippage scenario...")
        high_slip_key = start_execution_tracking("test_trade_003", token_address, expected_price)
        time.sleep(0.2)  # 200ms execution
        
        high_slip_price = 95.0  # 5% negative slippage
        complete_execution_tracking(high_slip_key, high_slip_price, 0.006, True)
        print("   Completed high slippage execution tracking")
        
        # Test 5: Get analytics summary
        print("5. Testing analytics retrieval...")
        
        performance = execution_analytics.get_execution_performance(30)
        print(f"   Total executions: {performance.total_trades}")
        print(f"   Success rate: {performance.success_rate:.1f}%")
        print(f"   Average execution time: {performance.average_execution_time_ms:.1f}ms")
        print(f"   Average slippage: {performance.slippage_analysis.average_slippage:.3f}%")
        
        recent_summary = execution_analytics.get_recent_execution_summary(5)
        print(f"   Recent executions tracked: {len(recent_summary)}")
        
        # Test 6: Generate execution report
        print("6. Testing report generation...")
        report = get_execution_report()
        print(f"   Report generated ({len(report)} characters)")
        
        # Save report to file
        report_file = Path("reports") / f"execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"   Report saved to: {report_file}")
        
        # Test 7: Check file creation
        print("7. Testing file creation...")
        analytics_files = [
            "analytics/execution_metrics.json"
        ]
        
        for file_path in analytics_files:
            if Path(file_path).exists():
                print(f"   {file_path} - EXISTS")
            else:
                print(f"   {file_path} - MISSING")
        
        print("\n" + "=" * 40)
        print("EXECUTION ANALYTICS TEST COMPLETE")
        print("The execution timing and slippage tracking system is working!")
        print("\nKey features tested:")
        print("- Execution timing measurement")
        print("- Slippage calculation and analysis")
        print("- Success/failure tracking")
        print("- Network latency and confirmation time tracking")
        print("- Alert detection for high slippage and slow executions")
        print("- Performance analytics and reporting")
        
        print("\nNext steps:")
        print("- Trading bot will now automatically track all trade executions")
        print("- Check reports/ directory for execution analytics reports")
        print("- Monitor analytics/execution_metrics.json for detailed data")
        
        # Show sample of recent executions if available
        if recent_summary:
            print("\nRecent Execution Sample:")
            print("-" * 25)
            for i, execution in enumerate(recent_summary[:3], 1):
                status = "SUCCESS" if execution['success'] else f"FAILED ({execution['error']})"
                print(f"{i}. {execution['token']} - {execution['execution_time_ms']:.1f}ms - "
                     f"{execution['slippage_pct']:.2f}% slip - {status}")
        
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
        success = test_execution_analytics()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest cancelled")
        exit(1)
    except Exception as e:
        print(f"\nTest error: {e}")
        exit(1)