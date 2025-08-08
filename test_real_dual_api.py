#!/usr/bin/env python3

"""
Test Real Dual-API Manager - No Unicode
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment
load_dotenv()

try:
    from api.smart_dual_api_manager import SmartDualAPIManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

async def test_real_dual_api():
    """Test real dual-API discovery"""
    
    print("REAL DUAL-API MANAGER TEST")
    print("=" * 40)
    
    try:
        async with SmartDualAPIManager() as manager:
            print("Dual-API manager initialized")
            
            # Test intelligent discovery
            print("Running intelligent token discovery...")
            tokens = await manager.discover_tokens_intelligently()
            
            print(f"Total tokens discovered: {len(tokens)}")
            
            if tokens:
                # Analyze sources
                sources = {}
                for token in tokens:
                    source = getattr(token, 'source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                
                print(f"Token sources breakdown:")
                for source, count in sources.items():
                    print(f"  {source}: {count} tokens")
                
                # Show top tokens
                top_tokens = sorted(tokens, key=lambda x: x.momentum_score, reverse=True)[:10]
                print(f"\nTop 10 tokens by momentum:")
                for i, token in enumerate(top_tokens):
                    print(f"  {i+1}. {token.symbol} - Score: {token.momentum_score:.1f} - Source: {token.source}")
                
                # Calculate performance
                tokens_per_scan = len(tokens)
                daily_scans = 96  # 15-minute intervals
                projected_daily = tokens_per_scan * daily_scans
                baseline = 864  # Previous GeckoTerminal only
                improvement = projected_daily / baseline
                
                print(f"\nPerformance Analysis:")
                print(f"Tokens per scan: {tokens_per_scan}")
                print(f"Projected daily tokens: {projected_daily}")
                print(f"Improvement over baseline: {improvement:.1f}x")
                print(f"Target achievement (2500): {projected_daily >= 2500}")
                
                # Get usage stats
                stats = manager.get_usage_stats()
                print(f"\nAPI Usage Stats:")
                if 'providers' in stats:
                    for provider, pstats in stats['providers'].items():
                        efficiency = pstats.get('efficiency', 0)
                        tokens_found = pstats.get('tokens_discovered', 0)
                        print(f"  {provider}: {tokens_found} tokens, {efficiency:.1f} efficiency")
                
                return {
                    'total_tokens': len(tokens),
                    'tokens_per_scan': tokens_per_scan,
                    'projected_daily': projected_daily,
                    'improvement_factor': improvement,
                    'meets_target': projected_daily >= 2500,
                    'sources': sources
                }
            else:
                print("No tokens discovered")
                return {'total_tokens': 0}
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def main():
    """Main function"""
    
    # Ensure dual-API strategy is enabled
    os.environ['API_STRATEGY'] = 'dual'
    
    try:
        results = asyncio.run(test_real_dual_api())
        
        print("\n" + "=" * 40)
        print("FINAL RESULTS")
        print("=" * 40)
        
        if results.get('total_tokens', 0) > 0:
            improvement = results.get('improvement_factor', 0)
            projected = results.get('projected_daily', 0)
            meets_target = results.get('meets_target', False)
            
            print(f"SUCCESS: Dual-API working!")
            print(f"Total tokens: {results['total_tokens']}")
            print(f"Improvement: {improvement:.1f}x")
            print(f"Projected daily: {projected:.0f} tokens")
            print(f"Meets 2500 target: {meets_target}")
            
            # Show sources
            sources = results.get('sources', {})
            print(f"Sources used: {list(sources.keys())}")
            
            if improvement >= 3.0:
                print("EXCELLENT: 3x+ improvement achieved!")
                return True
            elif improvement >= 1.5:
                print("GOOD: Significant improvement achieved")
                return True
            else:
                print("PARTIAL: Some improvement but optimization needed")
                return True
        else:
            print("FAILED: No tokens discovered")
            if 'error' in results:
                print(f"Error: {results['error']}")
            return False
            
    except Exception as e:
        print(f"Test execution error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)