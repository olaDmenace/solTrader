#!/usr/bin/env python3

"""
Simplified Dual-API Strategy Test
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
logger = logging.getLogger(__name__)

async def test_dual_api_discovery():
    """Test dual-API token discovery"""
    
    print("=" * 60)
    print("DUAL-API STRATEGY TEST")
    print("=" * 60)
    
    try:
        async with SmartDualAPIManager() as manager:
            print("✓ Smart Dual-API Manager initialized")
            
            # Test discovery
            print("\nTesting intelligent token discovery...")
            start_time = time.time()
            tokens = await manager.discover_tokens_intelligently()
            end_time = time.time()
            
            print(f"Discovery Results:")
            print(f"  Total tokens: {len(tokens)}")
            print(f"  Response time: {end_time - start_time:.2f}s")
            
            if tokens:
                # Analyze sources
                sources = {}
                for token in tokens:
                    source = getattr(token, 'source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                
                print(f"  Token sources: {sources}")
                
                # Show sample tokens
                print(f"\nSample tokens:")
                for i, token in enumerate(tokens[:5]):
                    print(f"  {i+1}. {token.symbol} - Score: {token.momentum_score:.1f}")
                
                # Calculate projections
                tokens_per_scan = len(tokens)
                daily_scans = 96  # 15-minute intervals
                projected_daily = tokens_per_scan * daily_scans
                improvement_vs_baseline = projected_daily / 864  # vs GeckoTerminal only
                
                print(f"\nPerformance Projections:")
                print(f"  Tokens per scan: {tokens_per_scan}")
                print(f"  Projected daily tokens: {projected_daily}")
                print(f"  Improvement factor: {improvement_vs_baseline:.1f}x")
                print(f"  Target achievement: {'YES' if projected_daily >= 2500 else 'NO'} ({(projected_daily/2500)*100:.1f}%)")
                
                # Get usage stats
                stats = manager.get_usage_stats()
                print(f"\nAPI Usage:")
                if 'providers' in stats:
                    for provider, provider_stats in stats['providers'].items():
                        print(f"  {provider}: efficiency {provider_stats.get('efficiency', 0):.1f}")
                
                return {
                    'tokens_found': len(tokens),
                    'projected_daily': projected_daily,
                    'improvement_factor': improvement_vs_baseline,
                    'target_achieved': projected_daily >= 2500
                }
            else:
                print("No tokens discovered")
                return None
                
    except Exception as e:
        print(f"Test failed: {e}")
        logger.error(f"Dual-API test error: {e}", exc_info=True)
        return None

def main():
    """Main function"""
    
    # Set environment
    os.environ['API_STRATEGY'] = 'dual'
    
    try:
        results = asyncio.run(test_dual_api_discovery())
        
        if results and results['target_achieved']:
            print(f"\n✅ SUCCESS: Dual-API strategy achieves target!")
            print(f"   {results['improvement_factor']:.1f}x improvement over single API")
        elif results:
            print(f"\n⚠️ PARTIAL: {results['improvement_factor']:.1f}x improvement achieved")
            print(f"   {results['projected_daily']:.0f} tokens/day projected")
        else:
            print(f"\n❌ FAILED: No token discovery")
            
        return results is not None
        
    except Exception as e:
        print(f"Execution error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)