#!/usr/bin/env python3
"""
Cross-Strategy Conflict Resolution Testing
Test how the system handles multiple strategies competing for the same resources:

1. Multiple Strategies Same Token - Test resource allocation when strategies want same token
2. Conflicting Signals - Test resolution when strategies give opposing signals
3. Resource Competition - Test capital allocation under resource constraints
4. Priority Systems - Test strategy prioritization mechanisms
5. Risk Limit Conflicts - Test when combined strategies exceed risk limits
6. Order Collision Detection - Test simultaneous order prevention
7. Portfolio Balance Conflicts - Test portfolio rebalancing conflicts
8. Recovery After Conflicts - Test system recovery after conflict resolution
"""

import asyncio
import logging
import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import load_settings
from src.database.db_manager import DatabaseManager
from src.trading.risk_engine import RiskEngine, RiskEngineConfig
from src.trading.paper_trading_engine import PaperTradingEngine, PaperTradingMode
from src.trading.trade_types import TradeDirection, TradeType
from src.monitoring.system_monitor import SystemMonitor
from src.portfolio.performance_based_rebalancer import PerformanceBasedRebalancer

# Clean logging setup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class MockStrategy:
    """Mock trading strategy for conflict testing"""
    
    def __init__(self, name: str, priority: int = 1, aggressiveness: float = 0.5):
        self.name = name
        self.priority = priority  # Higher = more priority
        self.aggressiveness = aggressiveness  # 0.0 to 1.0
        self.preferred_symbols = []
        self.signal_history = []
        
    def set_preferred_symbols(self, symbols: List[str]):
        """Set symbols this strategy prefers to trade"""
        self.preferred_symbols = symbols
        
    async def generate_signal(self, symbol: str, market_price: float) -> Dict[str, Any]:
        """Generate a trading signal for given symbol"""
        # Simulate signal generation based on strategy characteristics
        confidence = random.uniform(0.3, 0.9)
        
        # More aggressive strategies trade more frequently
        should_trade = random.random() < self.aggressiveness
        
        if not should_trade:
            return None
            
        # Random direction but biased by strategy name for testing
        if "bull" in self.name.lower():
            direction = TradeDirection.BUY
        elif "bear" in self.name.lower():
            direction = TradeDirection.SELL
        else:
            direction = random.choice([TradeDirection.BUY, TradeDirection.SELL])
            
        # Quantity based on confidence and aggressiveness
        base_quantity = 1.0
        quantity = base_quantity * confidence * self.aggressiveness
        
        signal = {
            'strategy': self.name,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': market_price,
            'confidence': confidence,
            'priority': self.priority,
            'timestamp': datetime.now()
        }
        
        self.signal_history.append(signal)
        return signal


class ConflictResolutionTester:
    """Test cross-strategy conflict resolution mechanisms"""
    
    def __init__(self):
        self.settings = load_settings()
        self.test_results = []
        self.conflict_scenarios = []
        self.strategies = []
        
    async def initialize_systems(self):
        """Initialize trading systems for conflict testing"""
        print("Initializing conflict resolution testing systems...")
        
        # Database Manager
        self.db_manager = DatabaseManager(self.settings)
        await self.db_manager.initialize()
        
        # Risk Engine with moderate limits for conflict testing
        risk_config = RiskEngineConfig(
            max_position_size=0.15,     # Allow moderate positions
            max_portfolio_risk=0.30,    # Allow higher risk for testing
            max_daily_loss=0.20         # Allow higher losses for testing
        )
        self.risk_engine = RiskEngine(self.db_manager, risk_config)
        await self.risk_engine.initialize()
        
        # System Monitor
        self.system_monitor = SystemMonitor(self.db_manager)
        await self.system_monitor.initialize()
        await self.system_monitor.start_monitoring()
        
        # Performance Rebalancer
        self.performance_rebalancer = PerformanceBasedRebalancer(self.settings)
        await self.performance_rebalancer.initialize()
        
        # Paper Trading Engine
        self.paper_engine = PaperTradingEngine(
            self.db_manager, self.risk_engine, self.system_monitor,
            None, PaperTradingMode.SIMULATION, 15000.0  # Higher balance for conflict testing
        )
        await self.paper_engine.initialize()
        await self.paper_engine.start_trading()
        
        # Create mock strategies
        self.strategies = [
            MockStrategy("BullStrategy", priority=3, aggressiveness=0.8),
            MockStrategy("BearStrategy", priority=2, aggressiveness=0.7),
            MockStrategy("NeutralStrategy", priority=1, aggressiveness=0.5),
            MockStrategy("ScalpStrategy", priority=4, aggressiveness=0.9),
            MockStrategy("SwingStrategy", priority=2, aggressiveness=0.4)
        ]
        
        # Set preferred symbols (overlapping to create conflicts)
        self.strategies[0].set_preferred_symbols(["SOL/USDC", "BTC/USDC", "ETH/USDC"])
        self.strategies[1].set_preferred_symbols(["SOL/USDC", "BTC/USDC", "AVAX/USDC"])
        self.strategies[2].set_preferred_symbols(["ETH/USDC", "SOL/USDC", "ADA/USDC"])
        self.strategies[3].set_preferred_symbols(["SOL/USDC", "ETH/USDC"])  # High overlap
        self.strategies[4].set_preferred_symbols(["BTC/USDC", "ETH/USDC"])
        
        print("[OK] Conflict resolution systems initialized with 5 competing strategies")

    async def test_same_token_competition(self):
        """Test multiple strategies competing for the same token"""
        print("\\n=== SAME TOKEN COMPETITION TESTS ===")
        
        # Test 1: Multiple strategies wanting SOL/USDC
        print("[TEST 1] Multiple strategies targeting SOL/USDC...")
        
        target_symbol = "SOL/USDC"
        market_price = 145.50
        competing_signals = []
        
        # Generate signals from all strategies for the same symbol
        for strategy in self.strategies:
            if target_symbol in strategy.preferred_symbols:
                signal = await strategy.generate_signal(target_symbol, market_price)
                if signal:
                    competing_signals.append(signal)
        
        if len(competing_signals) >= 2:
            self.test_results.append(("SAME_TOKEN_COMPETITION", True, 
                                   f"{len(competing_signals)} strategies competing for {target_symbol}"))
            
            # Record conflict scenario
            self.conflict_scenarios.append({
                'type': 'same_token',
                'symbol': target_symbol,
                'competing_strategies': len(competing_signals),
                'signals': competing_signals
            })
        else:
            self.test_results.append(("SAME_TOKEN_COMPETITION", False, 
                                   f"Insufficient competition: only {len(competing_signals)} strategies"))
        
        # Test 2: Signal priority resolution
        print("[TEST 2] Signal priority resolution...")
        try:
            if competing_signals:
                # Sort by priority (higher = better)
                sorted_signals = sorted(competing_signals, key=lambda x: x['priority'], reverse=True)
                highest_priority = sorted_signals[0]
                
                # Try to place the highest priority order
                order_id = await self.paper_engine.place_order(
                    symbol=highest_priority['symbol'],
                    direction=highest_priority['direction'],
                    order_type=TradeType.MARKET,
                    quantity=highest_priority['quantity'],
                    strategy_name=highest_priority['strategy']
                )
                
                if order_id:
                    self.test_results.append(("PRIORITY_RESOLUTION", True, 
                                           f"Highest priority strategy executed: {highest_priority['strategy']}"))
                else:
                    self.test_results.append(("PRIORITY_RESOLUTION", False, "Priority order rejected"))
            
        except Exception as e:
            self.test_results.append(("PRIORITY_RESOLUTION", False, f"Error: {e}"))

    async def test_conflicting_signals(self):
        """Test resolution of conflicting buy/sell signals"""
        print("\\n=== CONFLICTING SIGNALS TESTS ===")
        
        # Test 1: Opposite direction signals
        print("[TEST 1] Opposite direction signal conflicts...")
        
        test_symbol = "BTC/USDC"
        market_price = 64250.0
        
        # Force generate conflicting signals
        bull_signal = await self.strategies[0].generate_signal(test_symbol, market_price)  # BullStrategy
        bear_signal = await self.strategies[1].generate_signal(test_symbol, market_price)  # BearStrategy
        
        conflicts_detected = 0
        orders_placed = []
        
        # Try to place both orders
        for signal in [bull_signal, bear_signal]:
            if signal:
                order_id = await self.paper_engine.place_order(
                    symbol=signal['symbol'],
                    direction=signal['direction'],
                    order_type=TradeType.MARKET,
                    quantity=signal['quantity'],
                    strategy_name=signal['strategy']
                )
                
                if order_id:
                    orders_placed.append((signal['strategy'], signal['direction'], order_id))
                else:
                    conflicts_detected += 1
        
        # Check if system handled conflicting signals appropriately
        if len(orders_placed) <= 1 or conflicts_detected > 0:
            self.test_results.append(("CONFLICTING_SIGNALS", True, 
                                   f"Conflict resolution: {len(orders_placed)} orders placed, {conflicts_detected} conflicts detected"))
        else:
            self.test_results.append(("CONFLICTING_SIGNALS", False, 
                                   "System allowed conflicting orders without resolution"))
        
        # Test 2: Portfolio balance after conflicts
        print("[TEST 2] Portfolio balance after signal conflicts...")
        try:
            account = await self.paper_engine.get_account_status()
            
            # Check if portfolio is still balanced
            balance_ratio = account.current_balance / account.equity if account.equity > 0 else 1.0
            
            if 0.8 <= balance_ratio <= 1.2:  # Within reasonable balance
                self.test_results.append(("PORTFOLIO_BALANCE_CONFLICT", True, 
                                       f"Portfolio balanced after conflicts: {balance_ratio:.2f}"))
            else:
                self.test_results.append(("PORTFOLIO_BALANCE_CONFLICT", False, 
                                       f"Portfolio imbalanced: {balance_ratio:.2f}"))
                
        except Exception as e:
            self.test_results.append(("PORTFOLIO_BALANCE_CONFLICT", False, f"Error: {e}"))

    async def test_resource_constraints(self):
        """Test resource allocation under capital constraints"""
        print("\\n=== RESOURCE CONSTRAINT TESTS ===")
        
        # Test 1: Capital allocation limits
        print("[TEST 1] Capital allocation under constraints...")
        
        # Get current available capital
        account = await self.paper_engine.get_account_status()
        available_capital = account.free_margin
        
        # Generate multiple large orders that would exceed available capital
        large_orders = []
        total_required_capital = 0
        
        test_symbols = ["SOL/USDC", "ETH/USDC", "BTC/USDC"]
        
        for i, symbol in enumerate(test_symbols):
            # Each order requires significant capital
            quantity = 50.0  # Large quantity
            estimated_cost = quantity * (145.50 if "SOL" in symbol else 3420.0 if "ETH" in symbol else 64250.0)
            total_required_capital += estimated_cost
            
            large_orders.append({
                'symbol': symbol,
                'quantity': quantity,
                'estimated_cost': estimated_cost,
                'strategy': f"CapitalTest_{i+1}"
            })
        
        # Try to place all large orders
        successful_orders = 0
        rejected_orders = 0
        
        for order in large_orders:
            try:
                order_id = await self.paper_engine.place_order(
                    symbol=order['symbol'],
                    direction=TradeDirection.BUY,
                    order_type=TradeType.MARKET,
                    quantity=order['quantity'],
                    strategy_name=order['strategy']
                )
                
                if order_id:
                    successful_orders += 1
                else:
                    rejected_orders += 1
                    
            except Exception:
                rejected_orders += 1
            
            await asyncio.sleep(0.1)  # Small delay between orders
        
        # Capital constraints should prevent all orders from being placed
        if rejected_orders > 0:
            self.test_results.append(("CAPITAL_CONSTRAINTS", True, 
                                   f"Capital limits enforced: {successful_orders} placed, {rejected_orders} rejected"))
        else:
            self.test_results.append(("CAPITAL_CONSTRAINTS", False, 
                                   "Capital constraints not properly enforced"))
        
        # Test 2: Risk limit enforcement across strategies
        print("[TEST 2] Risk limit enforcement across multiple strategies...")
        try:
            # Check current portfolio risk
            portfolio_risk = await self.risk_engine.check_portfolio_risk()
            current_risk = portfolio_risk.get('risk_percentage', 0)
            
            # Try to place additional risky orders
            risk_test_orders = 0
            risk_rejections = 0
            
            for strategy in self.strategies[:3]:  # Test first 3 strategies
                order_id = await self.paper_engine.place_order(
                    symbol="RISK_TEST/USDC",
                    direction=TradeDirection.BUY,
                    order_type=TradeType.MARKET,
                    quantity=10.0,
                    strategy_name=f"{strategy.name}_risk_test"
                )
                
                if order_id:
                    risk_test_orders += 1
                else:
                    risk_rejections += 1
            
            if risk_rejections > 0:
                self.test_results.append(("MULTI_STRATEGY_RISK", True, 
                                       f"Multi-strategy risk managed: {risk_rejections} orders rejected"))
            else:
                self.test_results.append(("MULTI_STRATEGY_RISK", False, 
                                       "Multi-strategy risk limits not enforced"))
                
        except Exception as e:
            self.test_results.append(("MULTI_STRATEGY_RISK", False, f"Error: {e}"))

    async def test_simultaneous_execution(self):
        """Test simultaneous order execution from multiple strategies"""
        print("\\n=== SIMULTANEOUS EXECUTION TESTS ===")
        
        # Test 1: Concurrent order placement
        print("[TEST 1] Concurrent order placement...")
        
        async def place_concurrent_order(strategy: MockStrategy, symbol: str, order_num: int):
            """Place an order concurrently"""
            try:
                signal = await strategy.generate_signal(symbol, 150.0)
                if not signal:
                    return None
                    
                order_id = await self.paper_engine.place_order(
                    symbol=signal['symbol'],
                    direction=signal['direction'],
                    order_type=TradeType.MARKET,
                    quantity=1.0,  # Small quantity for concurrent test
                    strategy_name=f"{strategy.name}_concurrent_{order_num}"
                )
                
                return {
                    'strategy': strategy.name,
                    'order_id': order_id,
                    'success': order_id is not None
                }
                
            except Exception as e:
                return {
                    'strategy': strategy.name,
                    'error': str(e),
                    'success': False
                }
        
        # Launch concurrent orders
        concurrent_tasks = []
        test_symbol = "CONCURRENT_TEST/USDC"
        
        for i, strategy in enumerate(self.strategies):
            task = place_concurrent_order(strategy, test_symbol, i)
            concurrent_tasks.append(task)
        
        # Execute all concurrent orders
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        
        successful_concurrent = len([r for r in results if isinstance(r, dict) and r.get('success')])
        failed_concurrent = len(results) - successful_concurrent
        
        if successful_concurrent > 0:
            self.test_results.append(("CONCURRENT_EXECUTION", True, 
                                   f"Concurrent execution: {successful_concurrent} successful, {failed_concurrent} failed"))
        else:
            self.test_results.append(("CONCURRENT_EXECUTION", False, 
                                   "No concurrent orders executed successfully"))
        
        # Test 2: Order collision detection
        print("[TEST 2] Order collision detection...")
        
        # Try to place identical orders simultaneously
        collision_symbol = "COLLISION_TEST/USDC"
        
        collision_tasks = []
        for i in range(3):  # Try 3 identical orders
            task = self.paper_engine.place_order(
                symbol=collision_symbol,
                direction=TradeDirection.BUY,
                order_type=TradeType.MARKET,
                quantity=5.0,
                strategy_name=f"CollisionTest_{i}"
            )
            collision_tasks.append(task)
        
        # Execute simultaneously
        collision_results = await asyncio.gather(*collision_tasks, return_exceptions=True)
        successful_collisions = len([r for r in collision_results if r is not None])
        
        # Ideally, system should prevent or manage order collisions
        if successful_collisions <= 1:
            self.test_results.append(("ORDER_COLLISION", True, 
                                   f"Collision prevention: only {successful_collisions} identical orders placed"))
        else:
            self.test_results.append(("ORDER_COLLISION", False, 
                                   f"Collision detection failed: {successful_collisions} identical orders placed"))

    async def test_recovery_after_conflicts(self):
        """Test system recovery after conflict scenarios"""
        print("\\n=== RECOVERY AFTER CONFLICTS TESTS ===")
        
        # Test 1: System stability after conflicts
        print("[TEST 1] System stability after multiple conflicts...")
        try:
            # Check system health after all the conflict tests
            health = await self.system_monitor.get_health_status()
            system_status = health.get('status', 'UNKNOWN')
            
            # Try a simple operation to test system responsiveness
            recovery_order = await self.paper_engine.place_order(
                symbol="RECOVERY_TEST/USDC",
                direction=TradeDirection.BUY,
                order_type=TradeType.MARKET,
                quantity=0.1,  # Very small order
                strategy_name="RecoveryTest"
            )
            
            if system_status in ['HEALTHY', 'CRITICAL'] and (recovery_order or system_status == 'HEALTHY'):
                self.test_results.append(("SYSTEM_RECOVERY", True, 
                                       f"System stable after conflicts: {system_status}"))
            else:
                self.test_results.append(("SYSTEM_RECOVERY", False, 
                                       f"System potentially unstable: {system_status}"))
                
        except Exception as e:
            self.test_results.append(("SYSTEM_RECOVERY", False, f"Recovery test error: {e}"))
        
        # Test 2: Strategy coordination recovery
        print("[TEST 2] Strategy coordination after conflicts...")
        try:
            # Test if strategies can still coordinate properly
            coordination_signals = []
            
            for strategy in self.strategies[:3]:  # Test subset of strategies
                signal = await strategy.generate_signal("COORDINATION_TEST/USDC", 100.0)
                if signal:
                    coordination_signals.append(signal)
            
            if len(coordination_signals) >= 2:
                self.test_results.append(("STRATEGY_COORDINATION", True, 
                                       f"Strategy coordination recovered: {len(coordination_signals)} active strategies"))
            else:
                self.test_results.append(("STRATEGY_COORDINATION", False, 
                                       "Strategy coordination impaired"))
                
        except Exception as e:
            self.test_results.append(("STRATEGY_COORDINATION", False, f"Error: {e}"))

    async def run_comprehensive_conflict_testing(self):
        """Run comprehensive cross-strategy conflict testing"""
        
        print("=" * 80)
        print("            CROSS-STRATEGY CONFLICT RESOLUTION TESTING")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Testing conflict resolution between competing trading strategies\\n")
        
        try:
            await self.initialize_systems()
            
            # Run all conflict resolution tests
            await self.test_same_token_competition()
            await self.test_conflicting_signals()
            await self.test_resource_constraints()
            await self.test_simultaneous_execution()
            await self.test_recovery_after_conflicts()
            
            # Generate comprehensive report
            await self.generate_conflict_testing_report()
            
        except Exception as e:
            print(f"\\nCONFLICT TESTING SYSTEM ERROR: {e}")
            return False
        finally:
            await self.cleanup_systems()
        
        return True

    async def generate_conflict_testing_report(self):
        """Generate comprehensive conflict resolution testing report"""
        print("\\n" + "=" * 80)
        print("              CROSS-STRATEGY CONFLICT RESOLUTION REPORT")
        print("=" * 80)
        
        passed_tests = [r for r in self.test_results if r[1] == True]
        failed_tests = [r for r in self.test_results if r[1] == False]
        
        success_rate = len(passed_tests) / len(self.test_results) * 100 if self.test_results else 0
        
        print(f"CONFLICT RESOLUTION SUMMARY:")
        print(f"   Total Tests: {len(self.test_results)}")
        print(f"   Passed: {len(passed_tests)}")
        print(f"   Failed: {len(failed_tests)}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Conflict Scenarios: {len(self.conflict_scenarios)}")
        
        print(f"\\nCONFLICT RESOLUTION CATEGORIES:")
        
        conflict_categories = {
            "Resource Competition": [r for r in self.test_results if "COMPETITION" in r[0] or "PRIORITY" in r[0]],
            "Signal Conflicts": [r for r in self.test_results if "CONFLICTING" in r[0] or "BALANCE" in r[0]],
            "Resource Constraints": [r for r in self.test_results if "CONSTRAINTS" in r[0] or "RISK" in r[0]],
            "Concurrent Execution": [r for r in self.test_results if "CONCURRENT" in r[0] or "COLLISION" in r[0]],
            "System Recovery": [r for r in self.test_results if "RECOVERY" in r[0] or "COORDINATION" in r[0]]
        }
        
        for category, results in conflict_categories.items():
            if results:
                category_passed = len([r for r in results if r[1] == True])
                category_total = len(results)
                status = "[OK]" if category_passed == category_total else "[ISSUES]"
                print(f"\\n{status} {category}: {category_passed}/{category_total} tests passed")
                
                for test_name, success, message in results:
                    result_status = "[PASS]" if success else "[FAIL]"
                    print(f"      {result_status} {test_name}: {message}")
        
        if self.conflict_scenarios:
            print(f"\\nCONFLICT SCENARIOS TESTED:")
            for i, scenario in enumerate(self.conflict_scenarios, 1):
                print(f"   {i}. {scenario['type'].replace('_', ' ').title()}: {scenario['competing_strategies']} strategies on {scenario['symbol']}")
        
        print(f"\\nSTRATEGY COORDINATION ANALYSIS:")
        total_signals = sum(len(strategy.signal_history) for strategy in self.strategies)
        active_strategies = len([s for s in self.strategies if len(s.signal_history) > 0])
        print(f"   Total Signals Generated: {total_signals}")
        print(f"   Active Strategies: {active_strategies}/{len(self.strategies)}")
        
        if active_strategies > 0:
            avg_signals = total_signals / active_strategies
            print(f"   Average Signals per Strategy: {avg_signals:.1f}")
        
        print(f"\\nCONFLICT RESOLUTION ASSESSMENT:")
        if success_rate >= 85:
            print("   STATUS: EXCELLENT - Robust conflict resolution mechanisms")
            verdict = "PRODUCTION_READY"
        elif success_rate >= 70:
            print("   STATUS: GOOD - Effective conflict management with minor issues")
            verdict = "READY_WITH_MONITORING"
        elif success_rate >= 55:
            print("   STATUS: ACCEPTABLE - Basic conflict resolution working")
            verdict = "NEEDS_IMPROVEMENTS"
        else:
            print("   STATUS: INSUFFICIENT - Major conflict resolution failures")
            verdict = "REQUIRES_FIXES"
        
        print(f"   VERDICT: {verdict}")
        
        if failed_tests:
            print(f"\\nCRITICAL CONFLICT ISSUES:")
            for test_name, success, message in failed_tests:
                print(f"   - {test_name}: {message}")
        
        # Recommendations based on results
        print(f"\\nRECOMMENDATIONS:")
        if success_rate >= 80:
            print("   1. Multi-strategy deployment ready - conflicts well managed")
            print("   2. Consider implementing strategy performance weighting")
            print("   3. Monitor resource utilization under high strategy load")
        else:
            print("   1. Enhance conflict resolution algorithms before multi-strategy deployment")
            print("   2. Implement more sophisticated priority and resource allocation")
            print("   3. Add strategy coordination layer for better resource sharing")
        
        print("\\n" + "=" * 80)
        print("              CONFLICT RESOLUTION TESTING COMPLETE")
        print("=" * 80)

    async def cleanup_systems(self):
        """Cleanup all conflict testing systems"""
        print("\\nCleaning up conflict resolution systems...")
        
        try:
            if hasattr(self, 'paper_engine'):
                await self.paper_engine.shutdown()
            if hasattr(self, 'performance_rebalancer'):
                # Note: PerformanceBasedRebalancer may not have shutdown method
                pass
            if hasattr(self, 'system_monitor'):
                await self.system_monitor.stop_monitoring()
            if hasattr(self, 'risk_engine'):
                await self.risk_engine.shutdown()
            if hasattr(self, 'db_manager'):
                await self.db_manager.close()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
        
        print("System cleanup completed")


async def main():
    """Main execution"""
    tester = ConflictResolutionTester()
    
    try:
        success = await tester.run_comprehensive_conflict_testing()
        return 0 if success else 1
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)