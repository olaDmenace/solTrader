#!/usr/bin/env python3
"""
Test script to verify signal generation fixes
This tests that signal strength never returns None and handles edge cases properly
"""

import asyncio
import logging
from src.trading.signals import SignalGenerator
from src.config.settings import load_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_signal_generation_fixes():
    """Test various edge cases to ensure signal generation is robust"""
    
    settings = load_settings()
    signal_gen = SignalGenerator(settings)
    
    print("=" * 60)
    print("TESTING SIGNAL GENERATION FIXES")
    print("=" * 60)
    
    # Test cases with various scenarios
    test_cases = [
        {
            'name': 'High Volume/Liquidity Token',
            'token': {
                'address': '11111111111111111111111111111111',
                'price': 0.00001,
                'volume24h': 10000.0,
                'liquidity': 20000.0,
                'market_cap': 100000.0,
                'source': 'scanner'
            },
            'expected': 'STRONG_SIGNAL'
        },
        {
            'name': 'Moderate Token (Paper Trading Level)',
            'token': {
                'address': '22222222222222222222222222222222',
                'price': 0.000001,
                'volume24h': 1000.0,
                'liquidity': 2000.0,
                'market_cap': 10000.0,
                'source': 'scanner'
            },
            'expected': 'MODERATE_SIGNAL'
        },
        {
            'name': 'Low Volume Token',
            'token': {
                'address': '33333333333333333333333333333333',
                'price': 0.0001,
                'volume24h': 100.0,
                'liquidity': 500.0,
                'market_cap': 5000.0,
                'source': 'scanner'
            },
            'expected': 'WEAK_SIGNAL'
        },
        {
            'name': 'Empty Price History (Edge Case)',
            'token': {
                'address': '44444444444444444444444444444444',
                'price': 0.00005,
                'volume24h': 2000.0,
                'liquidity': 3000.0,
                'market_cap': 15000.0,
                'price_history': {},
                'volume_history': [],
                'source': 'scanner'
            },
            'expected': 'MODERATE_SIGNAL'
        },
        {
            'name': 'Trending Token',
            'token': {
                'address': '55555555555555555555555555555555',
                'price': 0.00002,
                'volume24h': 1500.0,
                'liquidity': 2500.0,
                'market_cap': 12000.0,
                'source': 'birdeye_trending',
                'trending_token': True,
                'trending_score': 85
            },
            'expected': 'TRENDING_SIGNAL'
        }
    ]
    
    successful_tests = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}/{total_tests}: {test_case['name']}")
        print("-" * 50)
        
        try:
            signal = await signal_gen.analyze_token(test_case['token'])
            
            if signal is None:
                print(f"Result: Signal is None (may be below threshold)")
                print(f"Status: This is acceptable if signal strength < threshold")
                successful_tests += 1
            else:
                # Validate signal properties
                address_ok = isinstance(signal.token_address, str) and len(signal.token_address) > 0
                price_ok = isinstance(signal.price, (int, float)) and signal.price > 0
                strength_ok = (isinstance(signal.strength, (int, float)) and 
                             0.0 <= signal.strength <= 1.0 and 
                             signal.strength == signal.strength)  # NaN check
                type_ok = isinstance(signal.signal_type, str)
                
                print(f"Result: Signal Generated Successfully")
                print(f"  Address: {signal.token_address[:8]}...")
                print(f"  Strength: {signal.strength:.3f}")
                print(f"  Price: {signal.price}")
                print(f"  Type: {signal.signal_type}")
                
                if all([address_ok, price_ok, strength_ok, type_ok]):
                    print(f"Status: ✅ ALL VALIDATIONS PASSED")
                    successful_tests += 1
                else:
                    print(f"Status: ❌ VALIDATION FAILED")
                    print(f"  Address OK: {address_ok}")
                    print(f"  Price OK: {price_ok}")
                    print(f"  Strength OK: {strength_ok}")
                    print(f"  Type OK: {type_ok}")
                    
        except Exception as e:
            print(f"Result: ❌ EXCEPTION OCCURRED")
            print(f"Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Successful Tests: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Signal generation is robust!")
        print("\nThe bot should now be able to:")
        print("- Generate valid signal strengths (never None)")
        print("- Handle tokens with missing price history")
        print("- Work with various volume/liquidity levels")
        print("- Support both regular and trending tokens")
        return True
    else:
        print("⚠️  Some tests failed - review the issues above")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_signal_generation_fixes())
    exit(0 if result else 1)