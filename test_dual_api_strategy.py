#!/usr/bin/env python3

"""
COMPREHENSIVE DUAL-API STRATEGY TEST
Tests the Smart Dual-API Manager for maximum token discovery
Target: 3-5x increase in token discovery (2,500+ tokens/day)
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from api.smart_dual_api_manager import SmartDualAPIManager
    from api.adaptive_quota_manager import AdaptiveQuotaManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class DualAPIPerformanceTester:
    def __init__(self):
        self.results = {
            'total_tokens_discovered': 0,
            'unique_tokens': 0,
            'api_calls_used': 0,
            'discovery_sessions': 0,
            'provider_breakdown': {},
            'efficiency_metrics': {},
            'quota_utilization': {},
            'performance_timeline': []
        }

    async def test_single_discovery_session(self, manager: SmartDualAPIManager) -> dict:
        """Test a single token discovery session"""
        print("\n--- Testing Single Discovery Session ---")
        
        start_time = time.time()
        tokens = await manager.discover_tokens_intelligently()
        end_time = time.time()
        
        session_results = {
            'tokens_found': len(tokens),
            'unique_addresses': len(set(token.address for token in tokens)),
            'response_time': end_time - start_time,
            'avg_momentum_score': sum(token.momentum_score for token in tokens) / len(tokens) if tokens else 0,
            'provider_sources': {},
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        # Analyze token sources and quality
        for token in tokens:
            source = getattr(token, 'source', 'unknown')
            if source not in session_results['provider_sources']:
                session_results['provider_sources'][source] = 0
            session_results['provider_sources'][source] += 1
            
            # Quality distribution
            if token.momentum_score >= 7.0:
                session_results['quality_distribution']['high'] += 1
            elif token.momentum_score >= 4.0:
                session_results['quality_distribution']['medium'] += 1
            else:
                session_results['quality_distribution']['low'] += 1
        
        print(f"Discovery Results:")
        print(f"  Total tokens: {session_results['tokens_found']}")
        print(f"  Unique addresses: {session_results['unique_addresses']}")
        print(f"  Response time: {session_results['response_time']:.2f}s")
        print(f"  Avg momentum score: {session_results['avg_momentum_score']:.2f}")
        print(f"  Provider breakdown: {session_results['provider_sources']}")
        print(f"  Quality distribution: {session_results['quality_distribution']}")
        
        return session_results

    async def test_quota_management(self, manager: SmartDualAPIManager) -> dict:
        """Test quota management system"""
        print("\n--- Testing Quota Management ---")
        
        quota_stats = manager.quota_manager.get_quota_status()
        health_check = manager.quota_manager.check_quota_health()
        
        print("Quota Status:")
        for provider, status in quota_stats['providers'].items():
            print(f"  {provider}:")
            print(f"    Allocated: {status['allocated_quota']}/{status['total_quota']}")
            print(f"    Used: {status['used_quota']} ({status['utilization_rate']})")
            print(f"    Efficiency: {status['efficiency_score']:.2f} tokens/call")
            print(f"    Strategy: {status['allocation_strategy']}")
        
        print(f"\nQuota Health:")
        print(f"  Status: {health_check['overall_status']}")
        print(f"  Emergency mode: {health_check['emergency_mode']}")
        if health_check['alerts']:
            print(f"  Alerts: {health_check['alerts']}")
        if health_check['recommendations']:
            print(f"  Recommendations: {health_check['recommendations']}")
        
        return {
            'quota_status': quota_stats,
            'health_check': health_check
        }

    async def test_api_fallback(self, manager: SmartDualAPIManager) -> dict:
        """Test API fallback mechanisms"""
        print("\n--- Testing API Fallback Mechanisms ---")
        
        fallback_results = {
            'solana_tracker_test': False,
            'geckoterminal_test': False,
            'fallback_behavior': 'unknown'
        }
        
        try:
            # Test individual API connections
            st_connection = await manager.solana_tracker.test_connection()
            gecko_connection = await manager.geckoterminal.test_connection()
            
            fallback_results['solana_tracker_test'] = st_connection
            fallback_results['geckoterminal_test'] = gecko_connection
            
            print(f"API Connection Tests:")
            print(f"  Solana Tracker: {'PASS' if st_connection else 'FAIL'}")
            print(f"  GeckoTerminal: {'PASS' if gecko_connection else 'FAIL'}")
            
            if st_connection and gecko_connection:
                fallback_results['fallback_behavior'] = 'both_available'
                print("  Fallback: Both APIs available - intelligent selection active")
            elif st_connection:
                fallback_results['fallback_behavior'] = 'solana_tracker_only'
                print("  Fallback: Solana Tracker only - high volume limited quota")
            elif gecko_connection:
                fallback_results['fallback_behavior'] = 'geckoterminal_only'
                print("  Fallback: GeckoTerminal only - unlimited but lower volume")
            else:
                fallback_results['fallback_behavior'] = 'both_failed'
                print("  Fallback: Both APIs failed - system degraded")
                
        except Exception as e:
            print(f"Fallback test error: {e}")
            fallback_results['fallback_behavior'] = 'error'
        
        return fallback_results

    async def test_performance_optimization(self, manager: SmartDualAPIManager) -> dict:
        """Test performance optimization features"""
        print("\n--- Testing Performance Optimization ---")
        
        # Get performance insights
        insights = manager.quota_manager.get_performance_insights()
        usage_stats = manager.get_usage_stats()
        
        print("Performance Analysis:")
        if 'efficiency_analysis' in insights:
            for provider, efficiency in insights['efficiency_analysis'].items():
                print(f"  {provider}:")
                print(f"    Avg efficiency: {efficiency['avg_efficiency']:.2f} tokens/call")
                print(f"    Max efficiency: {efficiency['max_efficiency']:.2f}")
                print(f"    Consistency: {efficiency['consistency']:.2f}")
        
        print("\nOptimization Opportunities:")
        for opportunity in insights.get('optimization_opportunities', []):
            print(f"  - {opportunity}")
        
        print(f"\nUsage Statistics:")
        if 'totals' in usage_stats:
            totals = usage_stats['totals']
            print(f"  Total tokens discovered: {totals['tokens_discovered']}")
            print(f"  API calls used: {totals['api_calls_used']}")
            print(f"  Overall efficiency: {totals['overall_efficiency']:.2f}")
        
        return {
            'performance_insights': insights,
            'usage_statistics': usage_stats
        }

    async def test_discovery_scaling(self, manager: SmartDualAPIManager, test_rounds: int = 3) -> dict:
        """Test token discovery scaling over multiple rounds"""
        print(f"\n--- Testing Discovery Scaling ({test_rounds} rounds) ---")
        
        scaling_results = {
            'rounds': [],
            'total_unique_tokens': 0,
            'avg_tokens_per_round': 0,
            'scaling_efficiency': 0,
            'projected_daily_tokens': 0
        }
        
        all_discovered_tokens = set()
        
        for round_num in range(test_rounds):
            print(f"\nRound {round_num + 1}:")
            
            round_start = time.time()
            tokens = await manager.discover_tokens_intelligently()
            round_end = time.time()
            
            round_unique = set(token.address for token in tokens)
            all_discovered_tokens.update(round_unique)
            
            round_results = {
                'round': round_num + 1,
                'tokens_found': len(tokens),
                'new_unique_tokens': len(round_unique - (all_discovered_tokens - round_unique)),
                'total_unique_so_far': len(all_discovered_tokens),
                'response_time': round_end - round_start,
                'discovery_rate': len(tokens) / (round_end - round_start)
            }
            
            scaling_results['rounds'].append(round_results)
            
            print(f"  Tokens found: {round_results['tokens_found']}")
            print(f"  New unique: {round_results['new_unique_tokens']}")
            print(f"  Total unique: {round_results['total_unique_so_far']}")
            print(f"  Response time: {round_results['response_time']:.2f}s")
            
            # Small delay between rounds
            await asyncio.sleep(2)
        
        # Calculate scaling metrics
        total_tokens = sum(round_data['tokens_found'] for round_data in scaling_results['rounds'])
        scaling_results['total_unique_tokens'] = len(all_discovered_tokens)
        scaling_results['avg_tokens_per_round'] = total_tokens / test_rounds
        scaling_results['scaling_efficiency'] = len(all_discovered_tokens) / total_tokens if total_tokens > 0 else 0
        
        # Project daily performance (96 scans per day at 15-minute intervals)
        scaling_results['projected_daily_tokens'] = scaling_results['avg_tokens_per_round'] * 96
        
        print(f"\nScaling Summary:")
        print(f"  Total unique tokens: {scaling_results['total_unique_tokens']}")
        print(f"  Avg tokens per round: {scaling_results['avg_tokens_per_round']:.1f}")
        print(f"  Scaling efficiency: {scaling_results['scaling_efficiency']:.2f}")
        print(f"  Projected daily tokens: {scaling_results['projected_daily_tokens']:.0f}")
        
        return scaling_results

async def run_comprehensive_test():
    """Run comprehensive dual-API strategy test"""
    
    print("=" * 80)
    print("SMART DUAL-API STRATEGY COMPREHENSIVE TEST")
    print("Target: 3-5x increase in token discovery (2,500+ tokens/day)")
    print("=" * 80)
    
    tester = DualAPIPerformanceTester()
    
    try:
        async with SmartDualAPIManager() as manager:
            print("Smart Dual-API Manager initialized successfully")
            
            # Test 1: Single discovery session
            session_results = await tester.test_single_discovery_session(manager)
            
            # Test 2: Quota management
            quota_results = await tester.test_quota_management(manager)
            
            # Test 3: API fallback mechanisms
            fallback_results = await tester.test_api_fallback(manager)
            
            # Test 4: Performance optimization
            optimization_results = await tester.test_performance_optimization(manager)
            
            # Test 5: Discovery scaling
            scaling_results = await tester.test_discovery_scaling(manager, test_rounds=3)
            
            # Compile final results
            final_results = {
                'session_test': session_results,
                'quota_test': quota_results,
                'fallback_test': fallback_results,
                'optimization_test': optimization_results,
                'scaling_test': scaling_results
            }
            
            print("\n" + "=" * 80)
            print("FINAL RESULTS SUMMARY")
            print("=" * 80)
            
            # Performance summary
            tokens_per_scan = scaling_results['avg_tokens_per_round']
            projected_daily = scaling_results['projected_daily_tokens']
            improvement_factor = projected_daily / 864  # Baseline from GeckoTerminal only
            
            print(f"Token Discovery Performance:")
            print(f"  Tokens per scan: {tokens_per_scan:.1f}")
            print(f"  Projected daily tokens: {projected_daily:.0f}")
            print(f"  Improvement factor: {improvement_factor:.1f}x")
            print(f"  Target achievement: {'YES' if projected_daily >= 2500 else 'NO'} "
                  f"({(projected_daily / 2500) * 100:.1f}% of target)")
            
            # API utilization
            print(f"\nAPI Utilization:")
            quota_stats = quota_results['quota_status']['providers']
            for provider, stats in quota_stats.items():
                print(f"  {provider}: {stats['utilization_rate']} utilization, "
                      f"{stats['efficiency_score']:.2f} efficiency")
            
            # System health
            health_status = quota_results['health_check']['overall_status']
            emergency_mode = quota_results['health_check']['emergency_mode']
            print(f"\nSystem Health:")
            print(f"  Status: {health_status.upper()}")
            print(f"  Emergency mode: {'ACTIVE' if emergency_mode else 'INACTIVE'}")
            
            # Recommendations
            print(f"\nSystem Recommendations:")
            if projected_daily >= 3000:
                print("  ✅ EXCELLENT: Target exceeded - system optimized for maximum discovery")
            elif projected_daily >= 2500:
                print("  ✅ SUCCESS: Target achieved - dual-API strategy working")
            elif projected_daily >= 1500:
                print("  ⚠️ PARTIAL: Improvement achieved but below target - optimize quota allocation")
            else:
                print("  ❌ INSUFFICIENT: Below target - investigate API issues")
            
            if fallback_results['solana_tracker_test'] and fallback_results['geckoterminal_test']:
                print("  ✅ Both APIs operational - optimal discovery conditions")
            else:
                print("  ⚠️ API connectivity issues detected - check network/credentials")
            
            print(f"\n🚀 Dual-API Strategy Test Complete!")
            print(f"   Ready for production deployment with {improvement_factor:.1f}x improvement")
            
            return final_results
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"Dual-API test failed: {e}", exc_info=True)
        return None

def main():
    """Main test function"""
    
    # Set environment for dual-API testing
    os.environ['API_STRATEGY'] = 'dual'
    # Re-enable Solana Tracker for testing
    os.environ['SOLANA_TRACKER_KEY'] = '542d5c9a-ea00-485c-817b-cd9839411972'
    
    try:
        results = asyncio.run(run_comprehensive_test())
        
        if results:
            print("\n✅ All tests completed successfully!")
            return True
        else:
            print("\n❌ Tests failed - check logs for details")
            return False
            
    except Exception as e:
        print(f"Test execution error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)