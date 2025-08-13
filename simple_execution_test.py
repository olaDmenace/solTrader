#!/usr/bin/env python3
"""
Simple test to verify execution pipeline fix works
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_token_validation_fix():
    """Test the token validation fix"""
    try:
        from src.trading.strategy import TradingStrategy, TradingMode
        from src.config.settings import Settings
        
        print("=" * 60)
        print("TESTING TOKEN VALIDATION FIX")
        print("=" * 60)
        
        # Create mock settings
        class MockSettings:
            def __init__(self):
                self.PAPER_TRADING = True
                self.INITIAL_PAPER_BALANCE = 100.0
                self.PAPER_MIN_LIQUIDITY = 10.0
                self.PAPER_SIGNAL_THRESHOLD = 0.1
                
        # Create mock components
        class MockJupiter:
            pass
            
        class MockWallet:
            pass
        
        settings = MockSettings()
        
        # Create strategy
        strategy = TradingStrategy(
            jupiter_client=MockJupiter(),
            wallet=MockWallet(),
            settings=settings,
            mode=TradingMode.PAPER
        )
        
        print(f"Strategy created in {strategy.state.mode.value} mode")
        print(f"Paper balance: {strategy.state.paper_balance:.4f} SOL")
        
        # Test token data that was failing before
        test_token_data = {
            "address": "CvGBG44dVcUKNdDXHNGWFtL8xeZnTqV9T8EkGQ4s2VeR",
            "volume_24h_sol": 0.0,  # This was the problem
            "liquidity_sol": 500000.0,
            "price_sol": 0.0,  # This was also a problem
            "market_cap_sol": 2627679.0
        }
        
        print("\nTesting problematic token data:")
        print(f"  Original volume: {test_token_data['volume_24h_sol']} SOL")
        print(f"  Original price: {test_token_data['price_sol']} SOL")
        
        # Test the enhanced token object creation
        token_obj = strategy._create_token_object(
            test_token_data["address"], 
            test_token_data
        )
        
        if token_obj:
            print(f"\nEnhanced token object created:")
            print(f"  Fixed volume: {token_obj.volume24h:.2f} SOL")
            print(f"  Fixed price: {token_obj.price_sol:.8f} SOL")
            print(f"  Liquidity: {token_obj.liquidity:.2f} SOL")
            print(f"  Market cap: {token_obj.market_cap:.0f} SOL")
            
            # Test validation
            is_valid = strategy._validate_token_basics(token_obj)
            print(f"\nValidation result: {'PASSED' if is_valid else 'FAILED'}")
            
            if is_valid:
                print("SUCCESS: Token validation fix working!")
                return True
            else:
                print("FAILED: Token still failing validation")
                return False
        else:
            print("FAILED: Could not create token object")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_token_validation_fix()
    if success:
        print("\nTEST PASSED: Execution pipeline fix works!")
    else:
        print("\nTEST FAILED: Fix needs more work")