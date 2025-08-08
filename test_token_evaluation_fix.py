#!/usr/bin/env python3

"""
Test Token Evaluation Fix
Verify that string formatting errors are resolved and tokens can be evaluated
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
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

def create_test_tokens():
    """Create test tokens with potential formatting issues"""
    
    # Create tokens with brackets and special characters in address/symbol
    test_tokens = [
        TokenData(
            address="solana_[test]_address_123",
            symbol="TEST[1]",
            name="Test Token [Special]",
            price=0.000123,
            price_change_24h=15.5,
            volume_24h=50000,
            market_cap=100000,
            liquidity=200,
            age_minutes=120,
            momentum_score=6.5,
            source="trending"
        ),
        TokenData(
            address="valid_address_456",
            symbol="VALID",
            name="Valid Token",
            price=0.000456,
            price_change_24h=8.2,
            volume_24h=25000,
            market_cap=50000,
            liquidity=150,
            age_minutes=240,
            momentum_score=4.1,
            source="volume"
        ),
        TokenData(
            address="address_with_{braces}",
            symbol="BRACE{}",
            name="Token with {braces}",
            price=0.000789,
            price_change_24h=120.0,  # High momentum bypass
            volume_24h=100000,
            market_cap=200000,
            liquidity=500,
            age_minutes=60,
            momentum_score=8.5,
            source="memescope"
        )
    ]
    
    return test_tokens

async def test_token_evaluation():
    """Test token evaluation with fixed string formatting"""
    
    print("TOKEN EVALUATION FIX TEST")
    print("=" * 50)
    
    try:
        # Initialize scanner
        alchemy_rpc = os.getenv('ALCHEMY_RPC_URL', 'https://solana-mainnet.g.alchemy.com/v2/test')
        wallet_address = os.getenv('WALLET_ADDRESS', 'JxKzzx2Hif9fnpg9J6jY8XfwYnSLHF6CQZK7zT9ScNb')
        settings = Settings(ALCHEMY_RPC_URL=alchemy_rpc, WALLET_ADDRESS=wallet_address)
        scanner = EnhancedTokenScanner(settings)
        
        print("Scanner initialized successfully")
        
        # Create test tokens
        test_tokens = create_test_tokens()
        print(f"Created {len(test_tokens)} test tokens with special characters")
        
        # Test token evaluation
        evaluation_results = []
        evaluation_errors = 0
        
        for i, token in enumerate(test_tokens, 1):
            print(f"\nTesting Token {i}: {token.symbol}")
            print(f"  Address: {token.address}")
            print(f"  Momentum: {token.price_change_24h}%")
            print(f"  Liquidity: {token.liquidity} SOL")
            
            try:
                # Test evaluation
                result = await scanner._evaluate_token(token)
                
                if result:
                    evaluation_results.append(result)
                    print(f"  Status: APPROVED (Score: {result.score:.1f})")
                    print(f"  Reasons: {result.reasons}")
                else:
                    print(f"  Status: REJECTED (filters failed)")
                    
            except Exception as e:
                evaluation_errors += 1
                print(f"  Status: ERROR - {str(e)}")
        
        # Results summary
        print("\n" + "=" * 50)
        print("EVALUATION TEST RESULTS")
        print("=" * 50)
        
        print(f"Tokens tested: {len(test_tokens)}")
        print(f"Tokens approved: {len(evaluation_results)}")
        print(f"Tokens rejected: {len(test_tokens) - len(evaluation_results) - evaluation_errors}")
        print(f"Evaluation errors: {evaluation_errors}")
        
        if evaluation_errors == 0:
            print("\nSUCCESS: No string formatting errors detected!")
            print("Token evaluation is working correctly")
            approval_rate = (len(evaluation_results) / len(test_tokens)) * 100
            print(f"Approval rate: {approval_rate:.1f}%")
        else:
            print(f"\nFAILURE: {evaluation_errors} evaluation errors detected")
            print("String formatting issues may still exist")
        
        return evaluation_errors == 0
        
    except Exception as e:
        print(f"Test setup error: {e}")
        return False

async def test_dual_api_integration():
    """Test that dual API integration still works after the fix"""
    
    print("\n" + "=" * 50)
    print("DUAL API INTEGRATION TEST")
    print("=" * 50)
    
    try:
        # Initialize scanner
        alchemy_rpc = os.getenv('ALCHEMY_RPC_URL', 'https://solana-mainnet.g.alchemy.com/v2/test')
        wallet_address = os.getenv('WALLET_ADDRESS', 'JxKzzx2Hif9fnpg9J6jY8XfwYnSLHF6CQZK7zT9ScNb')
        settings = Settings(ALCHEMY_RPC_URL=alchemy_rpc, WALLET_ADDRESS=wallet_address)
        scanner = EnhancedTokenScanner(settings)
        
        print("Starting scanner...")
        await scanner.start()
        
        print("Testing token discovery...")
        
        # Perform a scan
        approved_tokens = await scanner._perform_full_scan()
        
        print(f"Scan completed:")
        print(f"  Approved tokens: {len(approved_tokens) if approved_tokens else 0}")
        
        # Check daily stats
        stats = scanner.get_daily_stats()
        print(f"  Tokens scanned: {stats['tokens_scanned']}")
        print(f"  Approval rate: {stats['approval_rate']:.1f}%")
        
        await scanner.stop()
        
        # Success criteria
        success = (
            approved_tokens is not None and 
            stats['tokens_scanned'] > 0 and 
            stats['approval_rate'] > 0
        )
        
        if success:
            print("\nSUCCESS: Dual API integration working correctly!")
            print("Token discovery and evaluation pipeline operational")
        else:
            print("\nWARNING: Issues detected in dual API integration")
            print("May need additional debugging")
        
        return success
        
    except Exception as e:
        print(f"Integration test error: {e}")
        return False

async def main():
    """Main test function"""
    
    print("TESTING TOKEN EVALUATION FIXES")
    print("=" * 50)
    
    # Run tests
    evaluation_test = await test_token_evaluation()
    integration_test = await test_dual_api_integration()
    
    # Overall results
    print("\n" + "=" * 50)
    print("OVERALL TEST RESULTS")
    print("=" * 50)
    
    print(f"Token evaluation test: {'PASS' if evaluation_test else 'FAIL'}")
    print(f"Dual API integration test: {'PASS' if integration_test else 'FAIL'}")
    
    if evaluation_test and integration_test:
        print("\nSUCCESS: All tests passed!")
        print("Token evaluation fix is working correctly")
        print("System ready for production operation")
    else:
        print("\nFAILURE: Some tests failed")
        print("Additional debugging required")
    
    return evaluation_test and integration_test

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)