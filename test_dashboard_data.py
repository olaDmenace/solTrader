#!/usr/bin/env python3
"""
Test script to validate dashboard data loading
"""

import json
import os
from datetime import datetime

def test_dashboard_data():
    """Test that dashboard data loads correctly"""
    
    print("TESTING DASHBOARD DATA LOADING...")
    print("=" * 50)
    
    # Test 1: Check if bot_data.json exists
    if os.path.exists('bot_data.json'):
        print("PASS: bot_data.json file exists")
    else:
        print("FAIL: bot_data.json file missing")
        return False
    
    # Test 2: Load and validate JSON structure
    try:
        with open('bot_data.json', 'r') as f:
            data = json.load(f)
        print("PASS: JSON loads successfully")
    except json.JSONDecodeError as e:
        print(f"FAIL: JSON decode error: {e}")
        return False
    except FileNotFoundError:
        print("FAIL: File not found")
        return False
    
    # Test 3: Validate required fields
    required_fields = ['status', 'trades', 'performance']
    for field in required_fields:
        if field in data:
            print(f"PASS: Required field '{field}' present")
        else:
            print(f"FAIL: Missing required field: {field}")
    
    # Test 4: Validate trade data
    trades = data.get('trades', [])
    print(f"INFO: Found {len(trades)} trades")
    
    for i, trade in enumerate(trades, 1):
        token = trade.get('token', 'N/A')
        pnl = trade.get('pnl', 0)
        print(f"   Trade {i}: {token} - P&L: ${pnl:.6f}")
    
    # Test 5: Validate performance metrics
    performance = data.get('performance', {})
    total_pnl = performance.get('total_pnl', 0)
    win_rate = performance.get('win_rate', 0)
    total_trades = performance.get('total_trades', 0)
    
    print("\nPERFORMANCE METRICS:")
    print(f"   Total P&L: ${total_pnl:.6f}")
    print(f"   Win Rate: {win_rate}%")
    print(f"   Total Trades: {total_trades}")
    
    # Test 6: Check for token name issues
    token_names = [trade.get('token', 'N/A') for trade in trades]
    na_count = token_names.count('N/A')
    
    if na_count == 0:
        print("PASS: All trades have token names (no N/A values)")
    else:
        print(f"WARNING: {na_count} trades have 'N/A' token names")
    
    print("\nDASHBOARD SHOULD DISPLAY:")
    print(f"   - Token Names: {', '.join(token_names)}")
    print(f"   - Total P&L: ${total_pnl:.6f}")
    print(f"   - Win Rate: {win_rate}%")
    print(f"   - Open Positions: {performance.get('active_positions', 0)}")
    
    return True

if __name__ == "__main__":
    success = test_dashboard_data()
    if success:
        print("\nSUCCESS: Dashboard data validation PASSED")
        print("INFO: Dashboard should be accessible at: http://localhost:5000")
    else:
        print("\nFAILED: Dashboard data validation FAILED")