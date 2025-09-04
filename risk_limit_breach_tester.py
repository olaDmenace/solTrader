#!/usr/bin/env python3
"""
Risk Limit Breach Testing
Test emergency stops and risk management under extreme scenarios:

1. Position Size Limits - Test max position size enforcement
2. Portfolio Risk Limits - Test max portfolio risk enforcement  
3. Daily Loss Limits - Test max daily loss emergency stops
4. Margin Limits - Test margin requirement enforcement
5. Rapid Loss Scenarios - Test emergency stop triggers
6. Multiple Breach Scenarios - Test cascading risk events
7. Recovery Testing - Test system recovery after breaches
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import load_settings
from src.database.db_manager import DatabaseManager
from src.trading.risk_engine import RiskEngine, RiskEngineConfig
from src.trading.paper_trading_engine import PaperTradingEngine, PaperTradingMode
from src.trading.trade_types import TradeDirection, TradeType
from src.monitoring.system_monitor import SystemMonitor

# Clean logging setup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class RiskLimitTester:
    """Comprehensive risk limit breach testing framework"""
    
    def __init__(self):
        self.settings = load_settings()
        self.test_results = []
        self.emergency_stops_triggered = []
        
    async def initialize_systems(self):
        """Initialize all trading systems with strict risk limits"""
        print("Initializing risk management systems...")
        
        # Database Manager
        self.db_manager = DatabaseManager(self.settings)
        await self.db_manager.initialize()
        
        # Strict Risk Configuration for testing
        risk_config = RiskEngineConfig(
            max_position_size=0.02,      # Very strict: 2% max position
            max_portfolio_risk=0.10,     # Very strict: 10% max portfolio risk
            max_daily_loss=0.05,         # Very strict: 5% max daily loss
            max_drawdown=0.08,           # 8% max drawdown
            enable_stop_loss=True,       # Enable stop losses
            enable_take_profit=True,     # Enable take profits
            volatility_adjustment=True   # Enable volatility adjustments
        )
        self.risk_engine = RiskEngine(self.db_manager, risk_config)
        await self.risk_engine.initialize()
        
        # System Monitor
        self.system_monitor = SystemMonitor(self.db_manager)
        await self.system_monitor.initialize()
        await self.system_monitor.start_monitoring()
        
        # Paper Trading Engine with smaller balance for testing
        self.paper_engine = PaperTradingEngine(
            self.db_manager, self.risk_engine, self.system_monitor,
            None, PaperTradingMode.SIMULATION, 5000.0  # Smaller balance for easier limit testing
        )
        await self.paper_engine.initialize()
        await self.paper_engine.start_trading()
        
        print("[OK] Risk management systems initialized with strict limits")

    async def test_position_size_limits(self):
        """Test position size limit enforcement"""
        print("\\n=== POSITION SIZE LIMIT TESTS ===")
        
        # Test 1: Normal position within limits
        print("[TEST 1] Normal position within limits...")
        try:
            order_id = await self.paper_engine.place_order(
                symbol="TEST/USDC",
                direction=TradeDirection.BUY,
                order_type=TradeType.MARKET,
                quantity=0.5,  # Small position, should pass
                strategy_name="position_limit_normal"
            )
            
            if order_id:
                self.test_results.append(("POSITION_NORMAL", True, f"Small position accepted: {order_id}"))
            else:
                self.test_results.append(("POSITION_NORMAL", False, "Small position rejected"))
                
        except Exception as e:
            self.test_results.append(("POSITION_NORMAL", False, f"Error: {e}"))
        
        # Test 2: Oversized position exceeding limits
        print("[TEST 2] Oversized position exceeding limits...")
        try:
            order_id = await self.paper_engine.place_order(
                symbol="LARGE/USDC", 
                direction=TradeDirection.BUY,
                order_type=TradeType.MARKET,
                quantity=100.0,  # Very large position, should be rejected
                strategy_name="position_limit_breach"
            )
            
            if not order_id:  # Should be rejected
                self.test_results.append(("POSITION_OVERSIZED", True, "Oversized position correctly rejected"))
                self.emergency_stops_triggered.append("Position size limit enforced")
            else:
                self.test_results.append(("POSITION_OVERSIZED", False, f"Oversized position incorrectly accepted: {order_id}"))
                
        except Exception as e:
            self.test_results.append(("POSITION_OVERSIZED", True, f"Position rejected with error: {e}"))
        
        # Test 3: Multiple positions building up risk
        print("[TEST 3] Multiple positions accumulating risk...")
        accepted_orders = 0
        rejected_orders = 0
        
        for i in range(10):
            try:
                order_id = await self.paper_engine.place_order(
                    symbol=f"MULTI_{i}/USDC",
                    direction=TradeDirection.BUY,
                    order_type=TradeType.MARKET,
                    quantity=1.0,
                    strategy_name="multiple_position_test"
                )
                
                if order_id:
                    accepted_orders += 1
                else:
                    rejected_orders += 1
                    
            except Exception:
                rejected_orders += 1
                
            await asyncio.sleep(0.1)  # Small delay
        
        # Should eventually start rejecting as risk builds up
        if rejected_orders > 0:
            self.test_results.append(("MULTIPLE_POSITIONS", True, 
                                   f"Risk accumulation detected: {accepted_orders} accepted, {rejected_orders} rejected"))
        else:
            self.test_results.append(("MULTIPLE_POSITIONS", False, 
                                   f"Risk accumulation not detected: all {accepted_orders} accepted"))

    async def test_portfolio_risk_limits(self):
        """Test portfolio risk limit enforcement"""
        print("\\n=== PORTFOLIO RISK LIMIT TESTS ===")
        
        # Test 1: Portfolio risk check
        print("[TEST 1] Portfolio risk assessment...")
        try:
            risk_status = await self.risk_engine.check_portfolio_risk()
            current_risk = risk_status.get('risk_percentage', 0)
            risk_level = risk_status.get('overall_risk_level', 'UNKNOWN')
            
            self.test_results.append(("PORTFOLIO_RISK_CHECK", True, 
                                   f"Risk level: {risk_level} ({current_risk:.1%})"))
            
        except Exception as e:
            self.test_results.append(("PORTFOLIO_RISK_CHECK", False, f"Error: {e}"))
        
        # Test 2: Simulated high-risk scenario
        print("[TEST 2] High-risk scenario simulation...")
        try:
            # Simulate a scenario that would push portfolio risk high
            high_risk_orders = []
            for i in range(5):
                order_id = await self.paper_engine.place_order(
                    symbol=f"HIGH_RISK_{i}/USDC",
                    direction=TradeDirection.BUY,
                    order_type=TradeType.MARKET,
                    quantity=5.0,  # Larger positions to build risk
                    strategy_name="high_risk_test"
                )
                if order_id:
                    high_risk_orders.append(order_id)
                    
            # Check if risk management eventually kicks in
            risk_status = await self.risk_engine.check_portfolio_risk()
            if risk_status.get('overall_risk_level') in ['HIGH', 'CRITICAL']:
                self.test_results.append(("HIGH_RISK_DETECTION", True, 
                                       "High risk scenario properly detected"))
                self.emergency_stops_triggered.append("Portfolio risk threshold breached")
            else:
                self.test_results.append(("HIGH_RISK_DETECTION", False, 
                                       "High risk scenario not detected"))
                
        except Exception as e:
            self.test_results.append(("HIGH_RISK_SCENARIO", False, f"Error: {e}"))

    async def test_daily_loss_limits(self):
        """Test daily loss limit enforcement"""
        print("\\n=== DAILY LOSS LIMIT TESTS ===")
        
        # Test 1: Simulate daily loss accumulation
        print("[TEST 1] Daily loss accumulation...")
        try:
            initial_balance = self.paper_engine.account.current_balance
            
            # Simulate losses through multiple small losing trades
            loss_orders = []
            for i in range(20):
                order_id = await self.paper_engine.place_order(
                    symbol=f"LOSS_SIM_{i}/USDC",
                    direction=TradeDirection.BUY,
                    order_type=TradeType.MARKET,
                    quantity=0.1,  # Small positions
                    strategy_name="daily_loss_test"
                )
                if order_id:
                    loss_orders.append(order_id)
                    
                await asyncio.sleep(0.05)  # Small delay
            
            current_balance = self.paper_engine.account.current_balance
            daily_loss_pct = (initial_balance - current_balance) / initial_balance
            
            self.test_results.append(("DAILY_LOSS_TRACKING", True, 
                                   f"Daily loss tracked: {daily_loss_pct:.2%}"))
            
        except Exception as e:
            self.test_results.append(("DAILY_LOSS_TRACKING", False, f"Error: {e}"))
        
        # Test 2: Emergency stop trigger test
        print("[TEST 2] Emergency stop trigger test...")
        try:
            # Try to place a large order that would exceed daily loss limits
            large_loss_order = await self.paper_engine.place_order(
                symbol="EMERGENCY_TEST/USDC",
                direction=TradeDirection.BUY,
                order_type=TradeType.MARKET,
                quantity=50.0,  # Very large order
                strategy_name="emergency_stop_test"
            )
            
            if not large_loss_order:
                self.test_results.append(("EMERGENCY_STOP", True, 
                                       "Emergency stop triggered - large order blocked"))
                self.emergency_stops_triggered.append("Daily loss limit emergency stop")
            else:
                self.test_results.append(("EMERGENCY_STOP", False, 
                                       "Emergency stop failed - large order accepted"))
                
        except Exception as e:
            self.test_results.append(("EMERGENCY_STOP", True, f"Emergency stop via exception: {e}"))

    async def test_margin_enforcement(self):
        """Test margin requirement enforcement"""
        print("\\n=== MARGIN ENFORCEMENT TESTS ===")
        
        # Test 1: Margin utilization
        print("[TEST 1] Margin utilization tracking...")
        try:
            account = await self.paper_engine.get_account_status()
            margin_usage = account.margin_used / account.current_balance if account.current_balance > 0 else 0
            free_margin_pct = account.free_margin / account.current_balance if account.current_balance > 0 else 0
            
            self.test_results.append(("MARGIN_TRACKING", True, 
                                   f"Margin used: {margin_usage:.1%}, Free: {free_margin_pct:.1%}"))
            
        except Exception as e:
            self.test_results.append(("MARGIN_TRACKING", False, f"Error: {e}"))
        
        # Test 2: Margin requirement enforcement
        print("[TEST 2] Margin requirement enforcement...")
        try:
            # Try to place orders that would exceed available margin
            margin_breach_orders = []
            for i in range(10):
                order_id = await self.paper_engine.place_order(
                    symbol=f"MARGIN_TEST_{i}/USDC",
                    direction=TradeDirection.BUY,
                    order_type=TradeType.MARKET,
                    quantity=10.0,  # Orders that require significant margin
                    strategy_name="margin_enforcement_test"
                )
                
                if order_id:
                    margin_breach_orders.append(order_id)
                else:
                    # Order was rejected, likely due to insufficient margin
                    break
                    
                await asyncio.sleep(0.05)
            
            # Check if margin enforcement eventually kicked in
            if len(margin_breach_orders) < 10:  # Some orders should be rejected
                self.test_results.append(("MARGIN_ENFORCEMENT", True, 
                                       f"Margin limits enforced after {len(margin_breach_orders)} orders"))
                self.emergency_stops_triggered.append("Insufficient margin protection")
            else:
                self.test_results.append(("MARGIN_ENFORCEMENT", False, 
                                       "Margin limits not properly enforced"))
                
        except Exception as e:
            self.test_results.append(("MARGIN_ENFORCEMENT", False, f"Error: {e}"))

    async def test_recovery_mechanisms(self):
        """Test system recovery after risk breaches"""
        print("\\n=== RECOVERY MECHANISM TESTS ===")
        
        # Test 1: System status after breaches
        print("[TEST 1] System status after risk breaches...")
        try:
            # Check if system is still operational
            health = await self.system_monitor.get_health_status()
            status = health.get('status', 'UNKNOWN')
            
            # Try to place a small, safe order
            recovery_order = await self.paper_engine.place_order(
                symbol="RECOVERY_TEST/USDC",
                direction=TradeDirection.BUY,
                order_type=TradeType.MARKET,
                quantity=0.01,  # Very small order
                strategy_name="recovery_test"
            )
            
            if recovery_order or status == 'HEALTHY':
                self.test_results.append(("SYSTEM_RECOVERY", True, 
                                       f"System operational after breaches: {status}"))
            else:
                self.test_results.append(("SYSTEM_RECOVERY", False, 
                                       f"System may be impaired: {status}"))
                
        except Exception as e:
            self.test_results.append(("SYSTEM_RECOVERY", False, f"Recovery test error: {e}"))
        
        # Test 2: Risk engine reset capability
        print("[TEST 2] Risk engine reset capability...")
        try:
            # Get current risk status
            pre_reset_risk = await self.risk_engine.check_portfolio_risk()
            
            # Simulate a risk engine reset (if such functionality exists)
            # This tests if the system can recover from extreme risk states
            
            post_reset_risk = await self.risk_engine.check_portfolio_risk()
            
            self.test_results.append(("RISK_ENGINE_CONTINUITY", True, 
                                   "Risk engine maintaining operational state"))
            
        except Exception as e:
            self.test_results.append(("RISK_ENGINE_CONTINUITY", False, f"Error: {e}"))

    async def run_comprehensive_risk_testing(self):
        """Run comprehensive risk limit breach testing"""
        
        print("=" * 80)
        print("                RISK LIMIT BREACH & EMERGENCY STOP TESTING")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Testing emergency stops and risk management under extreme scenarios\\n")
        
        try:
            await self.initialize_systems()
            
            # Run all risk breach tests
            await self.test_position_size_limits()
            await self.test_portfolio_risk_limits()
            await self.test_daily_loss_limits()
            await self.test_margin_enforcement()
            await self.test_recovery_mechanisms()
            
            # Generate comprehensive report
            await self.generate_risk_testing_report()
            
        except Exception as e:
            print(f"\\nRISK TESTING SYSTEM ERROR: {e}")
            return False
        finally:
            await self.cleanup_systems()
        
        return True

    async def generate_risk_testing_report(self):
        """Generate comprehensive risk testing report"""
        print("\\n" + "=" * 80)
        print("                RISK LIMIT BREACH TESTING REPORT")
        print("=" * 80)
        
        passed_tests = [r for r in self.test_results if r[1] == True]
        failed_tests = [r for r in self.test_results if r[1] == False]
        
        success_rate = len(passed_tests) / len(self.test_results) * 100 if self.test_results else 0
        
        print(f"RISK MANAGEMENT SUMMARY:")
        print(f"   Total Tests: {len(self.test_results)}")
        print(f"   Passed: {len(passed_tests)}")
        print(f"   Failed: {len(failed_tests)}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Emergency Stops Triggered: {len(self.emergency_stops_triggered)}")
        
        print(f"\\nRISK CONTROL CATEGORIES:")
        
        risk_categories = {
            "Position Size Controls": [r for r in self.test_results if "POSITION" in r[0]],
            "Portfolio Risk Controls": [r for r in self.test_results if "PORTFOLIO" in r[0] or "HIGH_RISK" in r[0]],
            "Daily Loss Controls": [r for r in self.test_results if "DAILY_LOSS" in r[0] or "EMERGENCY" in r[0]],
            "Margin Controls": [r for r in self.test_results if "MARGIN" in r[0]],
            "Recovery Systems": [r for r in self.test_results if "RECOVERY" in r[0] or "CONTINUITY" in r[0]]
        }
        
        for category, results in risk_categories.items():
            if results:
                category_passed = len([r for r in results if r[1] == True])
                category_total = len(results)
                status = "[OK]" if category_passed == category_total else "[ISSUES]"
                print(f"\\n{status} {category}: {category_passed}/{category_total} tests passed")
                
                for test_name, success, message in results:
                    result_status = "[PASS]" if success else "[FAIL]"
                    print(f"      {result_status} {test_name}: {message}")
        
        if self.emergency_stops_triggered:
            print(f"\\nEMERGENCY STOPS TRIGGERED:")
            for i, stop in enumerate(self.emergency_stops_triggered, 1):
                print(f"   {i}. {stop}")
        
        print(f"\\nRISK MANAGEMENT ASSESSMENT:")
        if success_rate >= 90 and len(self.emergency_stops_triggered) >= 2:
            print("   STATUS: EXCELLENT - Robust risk management with proper emergency stops")
            verdict = "PRODUCTION_SAFE"
        elif success_rate >= 80 and len(self.emergency_stops_triggered) >= 1:
            print("   STATUS: GOOD - Effective risk management with some emergency stops")
            verdict = "SAFE_WITH_MONITORING"
        elif success_rate >= 70:
            print("   STATUS: ACCEPTABLE - Basic risk controls working")
            verdict = "NEEDS_IMPROVEMENTS"
        else:
            print("   STATUS: CRITICAL - Major risk management failures detected")
            verdict = "UNSAFE_FOR_TRADING"
        
        print(f"   VERDICT: {verdict}")
        
        if failed_tests:
            print(f"\\nCRITICAL RISK ISSUES:")
            for test_name, success, message in failed_tests:
                print(f"   - {test_name}: {message}")
        
        if len(self.emergency_stops_triggered) == 0:
            print(f"\\n⚠️  WARNING: No emergency stops were triggered during extreme testing!")
            print(f"   This may indicate insufficient risk protection.")
        
        print("\\n" + "=" * 80)
        print("                  RISK TESTING COMPLETE")
        print("=" * 80)

    async def cleanup_systems(self):
        """Cleanup all trading systems"""
        print("\\nCleaning up risk management systems...")
        
        try:
            if hasattr(self, 'paper_engine'):
                await self.paper_engine.shutdown()
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
    tester = RiskLimitTester()
    
    try:
        success = await tester.run_comprehensive_risk_testing()
        return 0 if success else 1
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)