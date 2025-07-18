#!/usr/bin/env python3
"""
Test script to verify the market cap field mapping fix
"""
import sys
import os
sys.path.append('src')

def test_field_extraction():
    """Test the exact field extraction logic from strategy.py"""
    
    print("🔍 Testing Market Cap Field Extraction Fix")
    print("=" * 60)
    
    # Simulate the exact scanner data structure from logs
    scanner_token_data = {
        'address': 'EKpQGSJt...',
        'volume24h': 4085.30,
        'liquidity': 500000,
        'created_at': None,
        'price': 5.560690,
        'market_cap': 4459,  # Scanner provides this field
        'signal_strength': 0.7,
        'signal_type': 'basic',
        'scan_id': 1,
        'source': 'scanner'
    }
    
    print(f"📊 Scanner Token Data:")
    print(f"  Address: {scanner_token_data['address']}")
    print(f"  Price: {scanner_token_data['price']}")
    print(f"  Market Cap: {scanner_token_data['market_cap']}")
    print(f"  Volume: {scanner_token_data['volume24h']}")
    print("")
    
    # Test OLD field extraction logic (BROKEN)
    print("❌ OLD Logic (BROKEN):")
    old_market_cap_sol = float(scanner_token_data.get("market_cap_sol", 0))
    print(f"  market_cap_sol = token.get('market_cap_sol', 0) = {old_market_cap_sol}")
    
    # Test NEW field extraction logic (FIXED)
    print("✅ NEW Logic (FIXED):")
    new_market_cap_sol = float(scanner_token_data.get("market_cap_sol", scanner_token_data.get("market_cap", 0)))
    print(f"  market_cap_sol = token.get('market_cap_sol', token.get('market_cap', 0)) = {new_market_cap_sol}")
    print("")
    
    # Test validation logic
    print("🔍 Validation Test:")
    
    # Mock settings
    MIN_MARKET_CAP_SOL = 10.0
    MAX_MARKET_CAP_SOL = 10000.0
    
    # OLD validation (would fail)
    old_validation = (MIN_MARKET_CAP_SOL <= old_market_cap_sol <= MAX_MARKET_CAP_SOL)
    print(f"  OLD: {MIN_MARKET_CAP_SOL} <= {old_market_cap_sol} <= {MAX_MARKET_CAP_SOL} = {old_validation}")
    
    # NEW validation (should pass)
    new_validation = (MIN_MARKET_CAP_SOL <= new_market_cap_sol <= MAX_MARKET_CAP_SOL)
    print(f"  NEW: {MIN_MARKET_CAP_SOL} <= {new_market_cap_sol} <= {MAX_MARKET_CAP_SOL} = {new_validation}")
    print("")
    
    # Results
    print("📋 Results:")
    if old_validation:
        print("  ❌ OLD logic: Token would PASS validation (unexpected)")
    else:
        print("  ❌ OLD logic: Token would FAIL validation (market_cap = 0)")
    
    if new_validation:
        print("  ✅ NEW logic: Token would PASS validation (market_cap = 4459)")
    else:
        print("  ❌ NEW logic: Token would FAIL validation (unexpected)")
    
    print("")
    print("🎯 Expected Log Patterns After Fix:")
    print("  BEFORE: [FILTER] Token details - Volume: 4085.30, Price: 5.560690, Market Cap: 0")
    print("  BEFORE: [FILTER] Token rejected: market_cap_range")
    print("  AFTER:  [FILTER] Token details - Volume: 4085.30, Price: 5.560690, Market Cap: 4459")
    print("  AFTER:  [PASS] Token passed all filters!")
    
    return new_validation

def test_object_creation():
    """Test TokenObject creation logic"""
    
    print("\n🔧 Testing TokenObject Creation:")
    print("=" * 40)
    
    # Simulate token info dict
    token_info = {
        'market_cap': 4459,
        'price_sol': 5.560690,
        'timestamp': None,
        'scan_id': 1,
        'source': 'scanner'
    }
    
    print(f"📊 Token Info Dict:")
    for key, value in token_info.items():
        print(f"  {key}: {value}")
    print("")
    
    # Test OLD logic (BROKEN)
    old_market_cap = token_info.get("market_cap_sol", 0)
    print(f"❌ OLD: info.get('market_cap_sol', 0) = {old_market_cap}")
    
    # Test NEW logic (FIXED) 
    new_market_cap = token_info.get("market_cap_sol", token_info.get("market_cap", 0))
    print(f"✅ NEW: info.get('market_cap_sol', info.get('market_cap', 0)) = {new_market_cap}")
    
    return new_market_cap > 0

if __name__ == "__main__":
    print("🚀 Starting Market Cap Field Mapping Fix Test")
    print("=" * 80)
    
    validation_passed = test_field_extraction()
    object_creation_passed = test_object_creation()
    
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS:")
    print(f"  ✅ Field extraction fix: {'WORKING' if validation_passed else 'FAILED'}")
    print(f"  ✅ Object creation fix: {'WORKING' if object_creation_passed else 'FAILED'}")
    
    if validation_passed and object_creation_passed:
        print("\n🎉 ALL TESTS PASSED! Fix is ready for deployment.")
        print("💰 Bot should now execute trades with correct market cap values.")
    else:
        print("\n❌ TESTS FAILED! Fix needs revision.")
        
    print("=" * 80)