#!/usr/bin/env python3

"""
Production Dual-API Test - Real Implementation
"""

import asyncio
import sys
import os
from dotenv import load_dotenv
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment
load_dotenv()

try:
    from api.solana_tracker import SolanaTrackerClient
    from api.geckoterminal_client import GeckoTerminalClient
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

async def test_individual_apis():
    """Test each API individually first"""
    
    print("INDIVIDUAL API TESTING")
    print("=" * 40)
    
    # Test Solana Tracker
    print("Testing Solana Tracker...")
    try:
        async with SolanaTrackerClient() as st_client:
            st_tokens = await st_client.get_all_tokens()
            st_count = len(st_tokens)
            print(f"Solana Tracker: {st_count} tokens")
            
            if st_count > 0:
                st_sample = st_tokens[:3]
                for token in st_sample:
                    print(f"  {token.symbol} - Score: {token.momentum_score:.1f}")
    except Exception as e:
        print(f"Solana Tracker error: {e}")
        st_count = 0
    
    # Test GeckoTerminal
    print("\nTesting GeckoTerminal...")
    try:
        async with GeckoTerminalClient() as gt_client:
            gt_tokens = await gt_client.get_all_tokens()
            gt_count = len(gt_tokens)
            print(f"GeckoTerminal: {gt_count} tokens")
            
            if gt_count > 0:
                gt_sample = gt_tokens[:3]
                for token in gt_sample:
                    print(f"  {token.symbol} - Score: {token.momentum_score:.1f}")
    except Exception as e:
        print(f"GeckoTerminal error: {e}")
        gt_count = 0
    
    return st_count, gt_count

async def test_combined_discovery():
    """Test combined token discovery manually"""
    
    print("\nCOMBINED DISCOVERY TEST")
    print("=" * 40)
    
    all_tokens = []
    
    # Get tokens from both APIs
    try:
        async with SolanaTrackerClient() as st_client:
            st_tokens = await st_client.get_all_tokens()
            all_tokens.extend(st_tokens)
            print(f"Added {len(st_tokens)} from Solana Tracker")
    except Exception as e:
        print(f"Solana Tracker failed: {e}")
    
    try:
        async with GeckoTerminalClient() as gt_client:
            gt_tokens = await gt_client.get_all_tokens()
            all_tokens.extend(gt_tokens)
            print(f"Added {len(gt_tokens)} from GeckoTerminal")
    except Exception as e:
        print(f"GeckoTerminal failed: {e}")
    
    # Deduplicate
    unique_tokens = {}
    for token in all_tokens:
        if token.address not in unique_tokens:
            unique_tokens[token.address] = token
        else:
            # Keep higher momentum score
            if token.momentum_score > unique_tokens[token.address].momentum_score:
                unique_tokens[token.address] = token
    
    final_tokens = list(unique_tokens.values())
    print(f"Total unique tokens: {len(final_tokens)}")
    
    return final_tokens

async def calculate_performance_metrics(tokens):
    """Calculate real performance metrics"""
    
    print("\nPERFORMANCE ANALYSIS")
    print("=" * 40)
    
    tokens_per_scan = len(tokens)
    daily_scans = 96  # 15-minute intervals
    projected_daily = tokens_per_scan * daily_scans
    
    # Baselines
    gecko_only_baseline = 9 * 96  # Previous GeckoTerminal only performance
    
    improvement_factor = projected_daily / gecko_only_baseline if gecko_only_baseline > 0 else 0
    
    print(f"Tokens per scan: {tokens_per_scan}")
    print(f"Daily scans: {daily_scans}")
    print(f"Projected daily tokens: {projected_daily}")
    print(f"GeckoTerminal baseline: {gecko_only_baseline}")
    print(f"Improvement factor: {improvement_factor:.1f}x")
    
    # Target analysis
    target_daily = 2500
    target_achievement = (projected_daily / target_daily) * 100 if target_daily > 0 else 0
    
    print(f"Target daily tokens: {target_daily}")
    print(f"Target achievement: {target_achievement:.1f}%")
    
    # Token quality analysis
    if tokens:
        momentum_scores = [token.momentum_score for token in tokens]
        avg_momentum = sum(momentum_scores) / len(momentum_scores)
        high_quality = len([s for s in momentum_scores if s >= 7.0])
        
        print(f"Average momentum score: {avg_momentum:.2f}")
        print(f"High quality tokens (7+): {high_quality}")
        
        # Show top tokens
        top_tokens = sorted(tokens, key=lambda x: x.momentum_score, reverse=True)[:5]
        print(f"Top 5 tokens:")
        for i, token in enumerate(top_tokens):
            print(f"  {i+1}. {token.symbol} - Score: {token.momentum_score:.1f}")
    
    return {
        'tokens_per_scan': tokens_per_scan,
        'projected_daily': projected_daily,
        'improvement_factor': improvement_factor,
        'target_achievement': target_achievement,
        'meets_target': projected_daily >= target_daily
    }

async def main():
    """Main test function"""
    
    print("PRODUCTION DUAL-API REAL PERFORMANCE TEST")
    print("=" * 50)
    
    # Test individual APIs
    st_count, gt_count = await test_individual_apis()
    
    # Test combined discovery
    combined_tokens = await test_combined_discovery()
    
    # Calculate performance
    metrics = await calculate_performance_metrics(combined_tokens)
    
    # Final results
    print("\n" + "=" * 50)
    print("FINAL PRODUCTION RESULTS")
    print("=" * 50)
    
    print(f"API Performance:")
    print(f"  Solana Tracker: {st_count} tokens")
    print(f"  GeckoTerminal: {gt_count} tokens")
    print(f"  Combined unique: {len(combined_tokens)} tokens")
    
    print(f"\nProjected Performance:")
    print(f"  Improvement factor: {metrics['improvement_factor']:.1f}x")
    print(f"  Daily tokens: {metrics['projected_daily']}")
    print(f"  Target achievement: {metrics['target_achievement']:.1f}%")
    
    success = metrics['meets_target']
    
    if success:
        print(f"\nSUCCESS: Target achieved!")
        print(f"Production deployment recommended")
    else:
        improvement = metrics['improvement_factor']
        if improvement >= 2.0:
            print(f"\nSIGNIFICANT IMPROVEMENT: {improvement:.1f}x increase")
            print(f"Substantial benefit achieved")
        else:
            print(f"\nPARTIAL IMPROVEMENT: {improvement:.1f}x increase")
            print(f"Some benefit but optimization needed")
    
    return success or metrics['improvement_factor'] >= 2.0

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test execution error: {e}")
        sys.exit(1)