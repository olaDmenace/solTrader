#!/usr/bin/env python3
"""
Quick test for both profit protection and momentum reversal logic
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trading.position import Position, ExitReason

def test_profit_protection():
    """Test the profit protection logic specifically"""
    print("=== Testing Profit Protection Logic ===")
    
    position = Position(
        token_address='TEST_PROFIT_PROTECTION',
        size=1.0,
        entry_price=1.0
    )
    
    # Profit protection scenario: 35% gain then decline to 12% (still profitable)
    price_sequence = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.32, 1.28, 1.25, 1.22, 1.18, 1.15, 1.12]
    volume_sequence = [100, 110, 120, 130, 140, 145, 150, 155, 150, 145, 140, 135, 130, 125, 120]
    
    for i, price in enumerate(price_sequence):
        volume = volume_sequence[i] if i < len(volume_sequence) else 100
        position.update_price(price, volume)
    
    should_exit, reason = position._check_momentum_exit()
    
    print(f"High Water Profit: {(position.high_water_mark / position.entry_price - 1) * 100:.1f}%")
    print(f"Current Profit: {(position.current_price / position.entry_price - 1) * 100:.1f}%")
    print(f"Trailing Stop: {position.high_water_mark * 0.92:.3f}")
    print(f"Current Price: {position.current_price}")
    print(f"Exit Reason: {reason}")
    
    profit_protection_works = should_exit and reason == ExitReason.PROFIT_PROTECTION.value
    print(f"✅ PROFIT PROTECTION: {'PASS' if profit_protection_works else 'FAIL'}")
    
    return profit_protection_works

def test_momentum_reversal():
    """Test the momentum reversal logic specifically"""
    print("\n=== Testing Momentum Reversal Logic ===")
    
    position = Position(
        token_address='TEST_MOMENTUM_REVERSAL',
        size=1.0,
        entry_price=1.0
    )
    
    # Momentum reversal scenario: 25% gain then crash to -10% (unprofitable)
    price_sequence = [1.0, 1.02, 1.05, 1.08, 1.1, 1.15, 1.2, 1.25, 1.22, 1.18, 1.15, 1.1, 1.05, 0.95, 0.9]
    volume_sequence = [100, 105, 110, 115, 120, 125, 130, 135, 125, 115, 105, 95, 85, 75, 65]
    
    for i, price in enumerate(price_sequence):
        volume = volume_sequence[i] if i < len(volume_sequence) else 100
        position.update_price(price, volume)
    
    should_exit, reason = position._check_momentum_exit()
    
    print(f"High Water Profit: {(position.high_water_mark / position.entry_price - 1) * 100:.1f}%")
    print(f"Current Profit: {(position.current_price / position.entry_price - 1) * 100:.1f}%")
    print(f"Momentum: {position._calculate_momentum():.4f}")
    print(f"Volume Trend: {position._get_volume_trend()}")
    print(f"Exit Reason: {reason}")
    
    momentum_reversal_works = should_exit and reason == ExitReason.MOMENTUM_REVERSAL.value
    print(f"✅ MOMENTUM REVERSAL: {'PASS' if momentum_reversal_works else 'FAIL'}")
    
    return momentum_reversal_works

if __name__ == "__main__":
    profit_test = test_profit_protection()
    momentum_test = test_momentum_reversal()
    
    if profit_test and momentum_test:
        print("\n🎉 BOTH TESTS PASS!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)