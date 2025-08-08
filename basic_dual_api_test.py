#!/usr/bin/env python3

"""
Basic Dual-API Strategy Test - No Unicode
"""

import asyncio
import sys
import os
import logging
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from api.smart_dual_api_manager import SmartDualAPIManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_dual_api():
    """Test dual-API strategy"""
    
    print("DUAL-API STRATEGY TEST")
    print("=" * 50)
    
    try:
        async with SmartDualAPIManager() as manager:
            print("Manager initialized successfully")
            
            # Test discovery
            print("Testing token discovery...")
            start_time = time.time()
            tokens = await manager.discover_tokens_intelligently()
            end_time = time.time()
            
            tokens_found = len(tokens)
            response_time = end_time - start_time
            
            print(f"Results:")
            print(f"  Tokens found: {tokens_found}")
            print(f"  Response time: {response_time:.2f}s")
            
            if tokens_found > 0:
                # Calculate projections
                daily_scans = 96  # 15-minute intervals
                projected_daily = tokens_found * daily_scans
                baseline = 864  # GeckoTerminal only baseline
                improvement = projected_daily / baseline
                
                print(f"Projections:")
                print(f"  Daily tokens: {projected_daily}")
                print(f"  Improvement: {improvement:.1f}x")
                print(f"  Target (2500): {projected_daily >= 2500}")
                
                # Show sample tokens
                print(f"Sample tokens:")
                for i, token in enumerate(tokens[:3]):
                    print(f"  {i+1}. {token.symbol} (score: {token.momentum_score:.1f})")
                
                return {
                    'success': True,
                    'tokens_found': tokens_found,
                    'projected_daily': projected_daily,
                    'improvement_factor': improvement,
                    'meets_target': projected_daily >= 2500
                }
            else:
                print("No tokens found")
                return {'success': False}
                
    except Exception as e:
        print(f"Test error: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Main function"""
    os.environ['API_STRATEGY'] = 'dual'
    
    try:
        results = asyncio.run(test_dual_api())
        
        print("\n" + "=" * 50)
        print("FINAL RESULTS")
        print("=" * 50)
        
        if results.get('success'):
            improvement = results.get('improvement_factor', 0)
            projected = results.get('projected_daily', 0)
            meets_target = results.get('meets_target', False)
            
            print(f"SUCCESS: Dual-API strategy working!")
            print(f"Improvement: {improvement:.1f}x over single API")
            print(f"Projected daily: {projected:.0f} tokens")
            print(f"Meets 2500 target: {meets_target}")
            
            if meets_target:
                print("READY FOR DEPLOYMENT!")
                return True
            else:
                print("Partial success - optimization needed")
                return True
        else:
            print("FAILED: Issues detected")
            return False
            
    except Exception as e:
        print(f"Execution error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)