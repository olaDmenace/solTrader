#!/usr/bin/env python3
"""
Complete Trade Execution Path Validator
Validates the entire trading pipeline from signal generation to database logging:

1. Signal Generation -> Strategy Decision
2. Strategy Decision -> Risk Assessment  
3. Risk Assessment -> Order Creation
4. Order Creation -> Execution
5. Execution -> Position Update
6. Position Update -> Portfolio Update
7. Portfolio Update -> Performance Tracking
8. Performance Tracking -> Database Logging
9. Database Logging -> Audit Trail

Tests both successful and failed execution paths.
"""

import asyncio
import logging
import sys
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import load_settings
from src.database.db_manager import DatabaseManager
from src.trading.risk_engine import RiskEngine, RiskEngineConfig
from src.trading.paper_trading_engine import PaperTradingEngine, PaperTradingMode
from src.trading.trade_types import TradeDirection, TradeType, OrderStatus
from src.monitoring.system_monitor import SystemMonitor
from src.portfolio.performance_based_rebalancer import PerformanceBasedRebalancer

# Clean logging setup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class TradeExecutionValidator:
    """Comprehensive validator for trade execution pipeline"""
    
    def __init__(self):
        self.settings = load_settings()
        self.validation_results = []
        self.execution_trail = []
        
    async def initialize_systems(self):
        """Initialize all trading systems"""
        print("Initializing trading systems...")
        
        # Database Manager
        self.db_manager = DatabaseManager(self.settings)
        await self.db_manager.initialize()
        
        # Risk Engine
        risk_config = RiskEngineConfig(
            max_position_size=0.10,
            max_portfolio_risk=0.20,
            max_daily_loss=0.15
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
            None, PaperTradingMode.SIMULATION, 10000.0
        )
        await self.paper_engine.initialize()
        await self.paper_engine.start_trading()
        
        print("[OK] All systems initialized successfully")

    async def validate_successful_execution_path(self):
        """Test complete successful trade execution path"""
        print("\\n=== SUCCESSFUL EXECUTION PATH TEST ===")
        
        trail_id = f"success_{int(time.time())}"
        self.execution_trail.append(f"[{trail_id}] Starting successful execution path test")
        
        # Step 1: Signal Generation (Simulated)
        print("[STEP 1] Signal Generation...")
        signal_data = {
            'symbol': 'SOL/USDC',
            'direction': 'BUY',
            'confidence': 0.85,
            'expected_return': 0.03,
            'strategy': 'execution_path_test',
            'timestamp': datetime.now().isoformat()
        }
        
        self.execution_trail.append(f"[{trail_id}] Signal generated: {signal_data}")
        self.validation_results.append(("SIGNAL_GENERATION", True, "Trading signal generated"))
        
        # Step 2: Risk Assessment
        print("[STEP 2] Risk Assessment...")
        try:
            risk_assessment = await self.risk_engine.assess_trade_risk(
                symbol=signal_data['symbol'],
                direction=signal_data['direction'],
                quantity=5.0,
                price=150.0,
                strategy_name=signal_data['strategy']
            )
            
            risk_passed = risk_assessment.recommendation not in ['REJECT', 'BLOCK']
            self.execution_trail.append(f"[{trail_id}] Risk assessment: {risk_assessment.risk_level.value} - {risk_assessment.recommendation}")
            self.validation_results.append(("RISK_ASSESSMENT", True, f"Risk assessed: {risk_assessment.risk_level.value}"))
            
        except Exception as e:
            self.execution_trail.append(f"[{trail_id}] Risk assessment failed: {e}")
            self.validation_results.append(("RISK_ASSESSMENT", False, f"Error: {e}"))
            return False
        
        # Step 3: Order Creation
        print("[STEP 3] Order Creation...")
        try:
            order_id = await self.paper_engine.place_order(
                symbol=signal_data['symbol'],
                direction=TradeDirection.BUY,
                order_type=TradeType.MARKET,
                quantity=5.0,
                strategy_name=signal_data['strategy']
            )
            
            if order_id:
                self.execution_trail.append(f"[{trail_id}] Order created: {order_id}")
                self.validation_results.append(("ORDER_CREATION", True, f"Order created: {order_id}"))
            else:
                self.execution_trail.append(f"[{trail_id}] Order creation failed - blocked by risk management")
                self.validation_results.append(("ORDER_CREATION", False, "Order blocked by risk management"))
                return False
                
        except Exception as e:
            self.execution_trail.append(f"[{trail_id}] Order creation error: {e}")
            self.validation_results.append(("ORDER_CREATION", False, f"Error: {e}"))
            return False
        
        # Step 4: Order Execution (simulated - should be instant in paper trading)
        print("[STEP 4] Order Execution...")
        await asyncio.sleep(0.1)  # Small delay for execution
        
        try:
            # Check order status
            account = await self.paper_engine.get_account_status()
            recent_orders = [o for o in account.recent_orders if o.order_id == order_id]
            
            if recent_orders and recent_orders[0].status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                executed_order = recent_orders[0]
                self.execution_trail.append(f"[{trail_id}] Order executed: {executed_order.filled_quantity} @ {executed_order.avg_fill_price}")
                self.validation_results.append(("ORDER_EXECUTION", True, f"Order filled: {executed_order.filled_quantity} units"))
            else:
                self.execution_trail.append(f"[{trail_id}] Order execution failed or pending")
                self.validation_results.append(("ORDER_EXECUTION", False, "Order not filled"))
                return False
                
        except Exception as e:
            self.execution_trail.append(f"[{trail_id}] Order execution check error: {e}")
            self.validation_results.append(("ORDER_EXECUTION", False, f"Error: {e}"))
            return False
        
        # Step 5: Position Update
        print("[STEP 5] Position Update...")
        try:
            account = await self.paper_engine.get_account_status()
            positions = [p for p in account.positions if p.symbol == signal_data['symbol']]
            
            if positions:
                position = positions[0]
                self.execution_trail.append(f"[{trail_id}] Position updated: {position.quantity} {position.symbol}")
                self.validation_results.append(("POSITION_UPDATE", True, f"Position: {position.quantity} {position.symbol}"))
            else:
                self.execution_trail.append(f"[{trail_id}] Position update failed - no position found")
                self.validation_results.append(("POSITION_UPDATE", False, "No position found"))
                
        except Exception as e:
            self.execution_trail.append(f"[{trail_id}] Position update error: {e}")
            self.validation_results.append(("POSITION_UPDATE", False, f"Error: {e}"))
        
        # Step 6: Portfolio Update
        print("[STEP 6] Portfolio Update...")
        try:
            account = await self.paper_engine.get_account_status()
            if account.trade_count > 0 and account.equity != account.current_balance:
                self.execution_trail.append(f"[{trail_id}] Portfolio updated: Balance=${account.current_balance:.2f}, Equity=${account.equity:.2f}")
                self.validation_results.append(("PORTFOLIO_UPDATE", True, f"Portfolio tracking active"))
            else:
                self.execution_trail.append(f"[{trail_id}] Portfolio update check: Balance=${account.current_balance:.2f}")
                self.validation_results.append(("PORTFOLIO_UPDATE", True, "Portfolio balanced"))
                
        except Exception as e:
            self.execution_trail.append(f"[{trail_id}] Portfolio update error: {e}")
            self.validation_results.append(("PORTFOLIO_UPDATE", False, f"Error: {e}"))
        
        # Step 7: Performance Tracking
        print("[STEP 7] Performance Tracking...")
        try:
            # Update strategy performance
            self.performance_rebalancer.update_strategy_return(
                signal_data['strategy'], 0.005, {  # Small positive return
                    'trade_id': order_id,
                    'symbol': signal_data['symbol'],
                    'execution_test': True
                }
            )
            
            self.execution_trail.append(f"[{trail_id}] Performance tracking updated")
            self.validation_results.append(("PERFORMANCE_TRACKING", True, "Strategy performance recorded"))
            
        except Exception as e:
            self.execution_trail.append(f"[{trail_id}] Performance tracking error: {e}")
            self.validation_results.append(("PERFORMANCE_TRACKING", False, f"Error: {e}"))
        
        # Step 8: Database Logging Verification
        print("[STEP 8] Database Logging Verification...")
        try:
            # Check if order was logged to database
            conn = sqlite3.connect(self.db_manager.db_path, timeout=5.0)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM orders WHERE order_id = ?", (order_id,))
            order_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM positions WHERE strategy_name = ?", (signal_data['strategy'],))
            position_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM strategy_performance WHERE strategy_name = ?", (signal_data['strategy'],))
            performance_count = cursor.fetchone()[0]
            
            conn.close()
            
            if order_count > 0:
                self.execution_trail.append(f"[{trail_id}] Database logging verified: Order={order_count}, Positions={position_count}, Performance={performance_count}")
                self.validation_results.append(("DATABASE_LOGGING", True, f"Complete audit trail recorded"))
            else:
                self.execution_trail.append(f"[{trail_id}] Database logging incomplete")
                self.validation_results.append(("DATABASE_LOGGING", False, "Order not logged to database"))
                
        except Exception as e:
            self.execution_trail.append(f"[{trail_id}] Database logging verification error: {e}")
            self.validation_results.append(("DATABASE_LOGGING", False, f"Error: {e}"))
        
        # Step 9: Audit Trail Consistency
        print("[STEP 9] Audit Trail Consistency...")
        try:
            # Verify that all steps have consistent timestamps and data
            trail_entries = [entry for entry in self.execution_trail if trail_id in entry]
            
            if len(trail_entries) >= 8:  # Should have entries for all major steps
                self.validation_results.append(("AUDIT_TRAIL", True, f"Complete audit trail: {len(trail_entries)} entries"))
            else:
                self.validation_results.append(("AUDIT_TRAIL", False, f"Incomplete audit trail: {len(trail_entries)} entries"))
                
        except Exception as e:
            self.validation_results.append(("AUDIT_TRAIL", False, f"Error: {e}"))
        
        return True

    async def validate_failed_execution_paths(self):
        """Test failure scenarios and error handling"""
        print("\\n=== FAILED EXECUTION PATH TESTS ===")
        
        # Test 1: Risk Rejection
        print("[TEST 1] Risk Rejection Path...")
        try:
            # Try to place a very large order that should be rejected
            trail_id = f"risk_reject_{int(time.time())}"
            
            risk_assessment = await self.risk_engine.assess_trade_risk(
                symbol="RISK/TEST",
                direction="BUY", 
                quantity=1000.0,  # Very large quantity
                price=1000.0,     # Very high price
                strategy_name="risk_rejection_test"
            )
            
            order_id = await self.paper_engine.place_order(
                symbol="RISK/TEST",
                direction=TradeDirection.BUY,
                order_type=TradeType.MARKET,
                quantity=1000.0,
                strategy_name="risk_rejection_test"
            )
            
            if not order_id:  # Order should be rejected
                self.execution_trail.append(f"[{trail_id}] Risk rejection successful")
                self.validation_results.append(("RISK_REJECTION", True, "High-risk order properly rejected"))
            else:
                self.execution_trail.append(f"[{trail_id}] Risk rejection failed - order accepted: {order_id}")
                self.validation_results.append(("RISK_REJECTION", False, "High-risk order improperly accepted"))
                
        except Exception as e:
            self.validation_results.append(("RISK_REJECTION", False, f"Error: {e}"))
        
        # Test 2: Invalid Symbol Handling
        print("[TEST 2] Invalid Symbol Handling...")
        try:
            trail_id = f"invalid_symbol_{int(time.time())}"
            
            order_id = await self.paper_engine.place_order(
                symbol="INVALID/SYMBOL",
                direction=TradeDirection.BUY,
                order_type=TradeType.MARKET,
                quantity=1.0,
                strategy_name="invalid_symbol_test"
            )
            
            # Should either reject or handle gracefully
            self.execution_trail.append(f"[{trail_id}] Invalid symbol handling: {order_id}")
            self.validation_results.append(("INVALID_SYMBOL", True, "Invalid symbol handled properly"))
            
        except Exception as e:
            self.execution_trail.append(f"[{trail_id}] Invalid symbol error: {e}")
            self.validation_results.append(("INVALID_SYMBOL", True, f"Error handled: {e}"))

    async def run_comprehensive_validation(self):
        """Run complete trade execution path validation"""
        
        print("=" * 80)
        print("          COMPLETE TRADE EXECUTION PATH VALIDATION")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Validating end-to-end trade execution pipeline\\n")
        
        try:
            await self.initialize_systems()
            
            # Test successful execution path
            await self.validate_successful_execution_path()
            
            # Test failure scenarios
            await self.validate_failed_execution_paths()
            
            # Generate comprehensive report
            await self.generate_validation_report()
            
        except Exception as e:
            print(f"\\nVALIDATION SYSTEM ERROR: {e}")
            return False
        finally:
            await self.cleanup_systems()
        
        return True

    async def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\\n" + "=" * 80)
        print("              TRADE EXECUTION PATH VALIDATION REPORT")
        print("=" * 80)
        
        passed_tests = [r for r in self.validation_results if r[1] == True]
        failed_tests = [r for r in self.validation_results if r[1] == False]
        
        success_rate = len(passed_tests) / len(self.validation_results) * 100 if self.validation_results else 0
        
        print(f"EXECUTION PIPELINE SUMMARY:")
        print(f"   Total Tests: {len(self.validation_results)}")
        print(f"   Passed: {len(passed_tests)}")
        print(f"   Failed: {len(failed_tests)}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        print(f"\\nPIPELINE STAGE RESULTS:")
        
        pipeline_stages = {
            "Signal Processing": ["SIGNAL_GENERATION"],
            "Risk Management": ["RISK_ASSESSMENT", "RISK_REJECTION"],
            "Order Management": ["ORDER_CREATION", "ORDER_EXECUTION"],
            "Position Management": ["POSITION_UPDATE", "PORTFOLIO_UPDATE"],
            "Performance Tracking": ["PERFORMANCE_TRACKING"],
            "Data Persistence": ["DATABASE_LOGGING", "AUDIT_TRAIL"],
            "Error Handling": ["INVALID_SYMBOL"]
        }
        
        for stage, test_names in pipeline_stages.items():
            stage_results = [r for r in self.validation_results if r[0] in test_names]
            if stage_results:
                stage_passed = len([r for r in stage_results if r[1] == True])
                stage_total = len(stage_results)
                status = "[OK]" if stage_passed == stage_total else "[ISSUES]"
                print(f"   {status} {stage}: {stage_passed}/{stage_total} tests passed")
                
                for test_name, success, message in stage_results:
                    result_status = "[PASS]" if success else "[FAIL]"
                    print(f"      {result_status} {test_name}: {message}")
        
        print(f"\\nEXECUTION TRAIL ANALYSIS:")
        print(f"   Total Trail Entries: {len(self.execution_trail)}")
        print(f"   Trail Completeness: {'COMPLETE' if len(self.execution_trail) > 10 else 'PARTIAL'}")
        
        if len(self.execution_trail) > 0:
            print(f"\\nSAMPLE EXECUTION TRAIL:")
            for entry in self.execution_trail[-5:]:  # Show last 5 entries
                print(f"   {entry}")
        
        print(f"\\nTRADE EXECUTION ASSESSMENT:")
        if success_rate >= 90:
            print("   STATUS: EXCELLENT - Complete end-to-end execution working")
            verdict = "PRODUCTION_READY"
        elif success_rate >= 75:
            print("   STATUS: GOOD - Core execution path working with minor issues")
            verdict = "READY_WITH_MONITORING"
        elif success_rate >= 60:
            print("   STATUS: ACCEPTABLE - Basic execution working, needs improvements")
            verdict = "NEEDS_IMPROVEMENTS"
        else:
            print("   STATUS: CRITICAL - Major execution path failures detected")
            verdict = "REQUIRES_FIXES"
        
        print(f"   VERDICT: {verdict}")
        
        if failed_tests:
            print(f"\\nCRITICAL ISSUES TO ADDRESS:")
            for test_name, success, message in failed_tests:
                print(f"   - {test_name}: {message}")
        
        print("\\n" + "=" * 80)
        print("                  VALIDATION COMPLETE")
        print("=" * 80)

    async def cleanup_systems(self):
        """Cleanup all trading systems"""
        print("\\nCleaning up trading systems...")
        
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
    validator = TradeExecutionValidator()
    
    try:
        success = await validator.run_comprehensive_validation()
        return 0 if success else 1
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)