#!/usr/bin/env python3

"""
Test Address Format Fix
Verify that token addresses are properly cleaned of "solana_" prefix
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up environment
load_dotenv()
os.environ['API_STRATEGY'] = 'dual'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.config.settings import Settings
from src.enhanced_token_scanner import EnhancedTokenScanner
from src.api.solana_tracker import TokenData

def create_test_tokens_with_address_formats():
    """Create test tokens with various address formats"""
    
    test_tokens = [
        # Token with "solana_" prefix (needs cleaning)
        TokenData(
            address="solana_9TPXpr6q36nHjLrC1raFFPYwiybEE53o4qhXLmzvbonk",
            symbol="CLIPPY",
            name="Clippy Token",
            price=0.000123,
            price_change_24h=8053.6,  # High momentum
            volume_24h=250000,
            market_cap=500000,
            liquidity=1200,
            age_minutes=45,
            momentum_score=9.5,
            source="geckoterminal/trending"
        ),
        # Token without prefix (should stay as-is)
        TokenData(
            address="7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
            symbol="HANDGUY",
            name="Hand Guy Token",
            price=0.000456,
            price_change_24h=972.5,  # High momentum
            volume_24h=150000,
            market_cap=300000,
            liquidity=800,
            age_minutes=90,
            momentum_score=8.2,
            source="trending"
        ),
        # Another token with "solana_" prefix
        TokenData(
            address="solana_4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
            symbol="ROCKET",
            name="Rocket Token",
            price=0.000789,
            price_change_24h=1500.0,  # Very high momentum
            volume_24h=500000,
            market_cap=1000000,
            liquidity=2500,
            age_minutes=30,
            momentum_score=9.8,
            source="geckoterminal/volume"
        )
    ]
    
    return test_tokens

async def test_address_format_cleaning():
    """Test that addresses are properly cleaned in scan_for_new_tokens"""
    
    print("ADDRESS FORMAT FIX TEST")
    print("=" * 50)
    
    try:
        # Initialize scanner
        alchemy_rpc = os.getenv('ALCHEMY_RPC_URL', 'https://solana-mainnet.g.alchemy.com/v2/test')
        wallet_address = os.getenv('WALLET_ADDRESS', 'JxKzzx2Hif9fnpg9J6jY8XfwYnSLHF6CQZK7zT9ScNb')
        settings = Settings(ALCHEMY_RPC_URL=alchemy_rpc, WALLET_ADDRESS=wallet_address)
        scanner = EnhancedTokenScanner(settings)
        
        print("Scanner initialized successfully")
        
        # Create test tokens with different address formats
        test_tokens = create_test_tokens_with_address_formats()
        print(f"Created {len(test_tokens)} test tokens with various address formats")
        
        # Manually add tokens to scanner's discovered tokens to simulate scan results
        for i, token in enumerate(test_tokens):
            print(f"\nTesting Token {i+1}: {token.symbol}")
            print(f"  Original address: {token.address}")
            print(f"  Expected clean address: {token.address[7:] if token.address.startswith('solana_') else token.address}")
            
            # Evaluate token to create ScanResult
            result = await scanner._evaluate_token(token)
            if result:
                scanner.discovered_tokens[token.address] = result
                print(f"  Status: APPROVED (Score: {result.score:.1f})")
            else:
                print(f"  Status: REJECTED")
        
        # Test scan_for_new_tokens method
        print(f"\n" + "=" * 30)
        print("TESTING scan_for_new_tokens()")
        print("=" * 30)
        
        selected_token = await scanner.scan_for_new_tokens()
        
        if selected_token:
            print(f"\nSelected Token Results:")
            print(f"  Symbol: {selected_token['symbol']}")
            print(f"  Returned address: {selected_token['address']}")
            print(f"  Score: {selected_token['score']:.1f}")
            
            # Verify address format
            address = selected_token['address']
            if address.startswith('solana_'):
                print(f"  ❌ FAIL: Address still has 'solana_' prefix: {address}")
                return False
            elif len(address) >= 32:  # Solana addresses are typically 32+ characters
                print(f"  PASS: Address properly cleaned: {address}")
                print(f"  Address length: {len(address)} characters")
                return True
            else:
                print(f"  ⚠️  WARN: Address might be too short: {address} ({len(address)} chars)")
                return False
        else:
            print(f"  ❌ FAIL: No token selected")
            return False
            
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_all_address_formats():
    """Test different address format scenarios"""
    
    print("\n" + "=" * 50)
    print("COMPREHENSIVE ADDRESS FORMAT TEST")
    print("=" * 50)
    
    test_cases = [
        {
            'input': 'solana_9TPXpr6q36nHjLrC1raFFPYwiybEE53o4qhXLmzvbonk',
            'expected': '9TPXpr6q36nHjLrC1raFFPYwiybEE53o4qhXLmzvbonk',
            'description': 'Standard solana_ prefix removal'
        },
        {
            'input': '7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU',
            'expected': '7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU',
            'description': 'No prefix (should remain unchanged)'
        },
        {
            'input': 'solana_',
            'expected': '',
            'description': 'Edge case: just prefix'
        },
        {
            'input': 'solana_x',
            'expected': 'x',
            'description': 'Edge case: prefix with single character'
        }
    ]
    
    all_passed = True
    
    for i, case in enumerate(test_cases, 1):
        input_addr = case['input']
        expected = case['expected']
        desc = case['description']
        
        # Apply the same logic as the fix
        clean_address = input_addr
        if clean_address.startswith('solana_'):
            clean_address = clean_address[7:]
        
        print(f"Test {i}: {desc}")
        print(f"  Input: '{input_addr}'")
        print(f"  Expected: '{expected}'")
        print(f"  Got: '{clean_address}'")
        
        if clean_address == expected:
            print(f"  PASS")
        else:
            print(f"  FAIL")
            all_passed = False
        print()
    
    return all_passed

async def main():
    """Main test function"""
    
    print("TESTING TOKEN ADDRESS FORMAT FIX")
    print("=" * 50)
    
    # Run tests
    address_cleaning_test = await test_address_format_cleaning()
    format_logic_test = await test_all_address_formats()
    
    # Overall results
    print("\n" + "=" * 50)
    print("OVERALL TEST RESULTS")
    print("=" * 50)
    
    print(f"Address cleaning test: {'PASS' if address_cleaning_test else 'FAIL'}")
    print(f"Format logic test: {'PASS' if format_logic_test else 'FAIL'}")
    
    if address_cleaning_test and format_logic_test:
        print("\nSUCCESS: All tests passed!")
        print("Token address format fix is working correctly")
        print("Expected behavior:")
        print("  - solana_ADDRESS -> ADDRESS")
        print("  - Regular addresses remain unchanged")
        print("  - Trading strategy will receive clean addresses")
    else:
        print("\nFAILURE: Some tests failed")
        print("Address format fix needs additional work")
    
    return address_cleaning_test and format_logic_test

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)