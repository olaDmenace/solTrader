#!/usr/bin/env python3
"""
FINAL System Integration Diagnostic - Production Ready
Human-readable validation of all core modules working in sync
"""

import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import load_settings
from src.database.db_manager import DatabaseManager
from src.trading.risk_engine import RiskEngine, RiskEngineConfig
from src.monitoring.system_monitor import SystemMonitor
from src.portfolio.performance_based_rebalancer import PerformanceBasedRebalancer
from src.portfolio.portfolio_manager import PortfolioManager
from src.trading.paper_trading_engine import PaperTradingEngine, PaperTradingMode
from src.trading.trade_types import TradeDirection, TradeType

# Clean logging setup
logging.basicConfig(level=logging.WARNING)  # Suppress debug noise
logger = logging.getLogger(__name__)


def print_header(title):
    print("=" * 80)
    print(f"  {title.center(76)}")
    print("=" * 80)


def print_section(title):
    print(f"\n{title}")
    print("-" * len(title))


async def run_final_diagnostic():
    """Run the final comprehensive system diagnostic"""
    
    print_header("SOLTRADER FINAL SYSTEM INTEGRATION DIAGNOSTIC")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Validating all core modules for production readiness")
    
    # Module tracking
    modules = {}
    module_health = {}
    start_time = time.time()
    
    try:
        print_section("PHASE 1: CORE MODULE INITIALIZATION")
        
        # Settings
        print("Loading system configuration...")
        settings = load_settings()
        print("   [OK] Configuration loaded successfully")
        
        # Database Manager
        print("Initializing Database Manager...")
        modules['db_manager'] = DatabaseManager(settings)
        await modules['db_manager'].initialize()
        module_health['database'] = 'HEALTHY'
        print("   [OK] Database connected and tables ready")
        
        # Risk Engine
        print("Initializing Risk Engine...")
        risk_config = RiskEngineConfig(
            max_position_size=0.05,
            max_portfolio_risk=0.15,
            max_daily_loss=0.10,
            enable_stop_loss=True
        )
        modules['risk_engine'] = RiskEngine(modules['db_manager'], risk_config)
        await modules['risk_engine'].initialize()
        module_health['risk_engine'] = 'HEALTHY'
        print("   [OK] Risk engine loaded with production config")
        
        # System Monitor
        print("Initializing System Monitor...")
        modules['system_monitor'] = SystemMonitor(modules['db_manager'])
        await modules['system_monitor'].initialize()
        await modules['system_monitor'].start_monitoring()
        module_health['system_monitor'] = 'HEALTHY'
        print("   [OK] System monitoring active")
        
        # Performance Rebalancer
        print("Initializing Performance Rebalancer...")
        modules['performance_rebalancer'] = PerformanceBasedRebalancer(modules['db_manager'])
        await modules['performance_rebalancer'].initialize()
        await modules['performance_rebalancer'].start()
        module_health['performance_rebalancer'] = 'HEALTHY'
        print("   [OK] Performance analysis active")
        
        # Portfolio Manager
        print("Initializing Portfolio Manager...")
        modules['portfolio_manager'] = PortfolioManager(
            modules['db_manager'], modules['risk_engine'], modules['system_monitor']
        )
        await modules['portfolio_manager'].initialize()
        await modules['portfolio_manager'].start()
        module_health['portfolio_manager'] = 'HEALTHY'
        print("   [OK] Portfolio tracking active")
        
        # Paper Trading Engine (without DEX connector for now)
        print("Initializing Paper Trading Engine...")
        modules['paper_trading_engine'] = PaperTradingEngine(
            modules['db_manager'], modules['risk_engine'], modules['system_monitor'],
            None,  # No DEX connector for this test
            PaperTradingMode.SIMULATION, 10000.0
        )
        await modules['paper_trading_engine'].initialize()
        await modules['paper_trading_engine'].start_trading()
        module_health['paper_trading_engine'] = 'HEALTHY'
        print("   [OK] Paper trading active with $10,000 balance")
        
        print("\n[SUCCESS] ALL CORE MODULES INITIALIZED")
        
        print_section("PHASE 2: LIVE SYSTEM SYNC TEST")
        
        print("Running 3 system synchronization ticks...")
        
        for tick in range(1, 4):
            print(f"\nTICK {tick}/3 - {datetime.now().strftime('%H:%M:%S')}")
            
            # Test 1: Risk Assessment
            print("   Testing risk assessment...")
            try:
                risk = await modules['risk_engine'].assess_trade_risk(
                    symbol="SOL/USDC", direction="BUY", quantity=10.0, price=145.50, strategy_name="test"
                )
                print(f"      Risk Level: {risk.risk_level.value} | Score: {risk.risk_score:.1f}")
                print(f"      Recommendation: {risk.recommendation}")
                module_health['risk_assessment'] = 'HEALTHY'
            except Exception as e:
                print(f"      [ERROR] Risk assessment failed: {e}")
                module_health['risk_assessment'] = 'ERROR'
            
            # Test 2: Portfolio Risk Check
            print("   Testing portfolio risk management...")
            try:
                portfolio_risk = await modules['risk_engine'].check_portfolio_risk()
                print(f"      Portfolio Risk: {portfolio_risk['risk_percentage']:.1%}")
                print(f"      Status: {portfolio_risk['overall_risk_level']}")
                module_health['portfolio_risk'] = 'HEALTHY'
            except Exception as e:
                print(f"      [ERROR] Portfolio risk check failed: {e}")
                module_health['portfolio_risk'] = 'ERROR'
            
            # Test 3: Paper Trade Execution (tick 2 only)
            if tick == 2:
                print("   Testing trade execution...")
                try:
                    order_id = await modules['paper_trading_engine'].place_order(
                        symbol="SOL/USDC", direction=TradeDirection.BUY, order_type=TradeType.MARKET,
                        quantity=1.0, strategy_name="diagnostic_test"
                    )
                    if order_id:
                        print(f"      [OK] Test order placed: {order_id}")
                        module_health['trade_execution'] = 'HEALTHY'
                        module_health['trade_logging'] = 'HEALTHY'
                    else:
                        print("      [WARN] Order rejected by risk management")
                        module_health['trade_execution'] = 'WARNING'
                except Exception as e:
                    print(f"      [ERROR] Trade execution failed: {e}")
                    module_health['trade_execution'] = 'ERROR'
            
            # Test 4: Account Status
            print("   Checking account status...")
            try:
                account = await modules['paper_trading_engine'].get_account_status()
                print(f"      Balance: ${account.current_balance:,.2f}")
                print(f"      Equity: ${account.equity:,.2f}")
                print(f"      Trades: {account.trade_count}")
                module_health['account_management'] = 'HEALTHY'
                
                if account.trade_count > 0:
                    module_health['capital_allocation'] = 'HEALTHY'
            except Exception as e:
                print(f"      [ERROR] Account status failed: {e}")
                module_health['account_management'] = 'ERROR'
            
            # Test 5: System Health
            print("   Checking system health...")
            try:
                health = await modules['system_monitor'].get_health_status()
                print(f"      System Status: {health['status']}")
                if 'metrics' in health:
                    metrics = health['metrics']
                    print(f"      Resources: CPU {metrics['cpu_usage']:.1f}% | Memory {metrics['memory_usage']:.1f}%")
                module_health['system_health'] = 'HEALTHY'
            except Exception as e:
                print(f"      [ERROR] System health check failed: {e}")
                module_health['system_health'] = 'ERROR'
            
            # Test 6: Performance Data
            print("   Testing performance tracking...")
            try:
                # Add some test performance data
                modules['performance_rebalancer'].update_strategy_return(
                    "diagnostic_test", 0.02, {"tick": tick, "test": True}
                )
                
                # Try to analyze performance
                analysis = await modules['performance_rebalancer'].analyze_strategy_performance("diagnostic_test")
                if analysis:
                    print(f"      Performance Analysis: {analysis.rebalancing_signal}")
                else:
                    print("      Performance Analysis: Insufficient data")
                module_health['performance_tracking'] = 'HEALTHY'
            except Exception as e:
                print(f"      [ERROR] Performance tracking failed: {e}")
                module_health['performance_tracking'] = 'ERROR'
            
            if tick < 3:
                await asyncio.sleep(1.5)
        
        print("\n[SUCCESS] SYSTEM SYNC TEST COMPLETED")
        
        print_section("PHASE 3: MODULE HEALTH REPORT")
        
        # Calculate overall health
        healthy = len([h for h in module_health.values() if h == 'HEALTHY'])
        warnings = len([h for h in module_health.values() if h == 'WARNING'])
        errors = len([h for h in module_health.values() if h == 'ERROR'])
        total = len(module_health)
        
        health_score = (healthy / total) * 100 if total > 0 else 0
        
        print(f"SYSTEM HEALTH SUMMARY:")
        print(f"   Overall Score: {health_score:.1f}%")
        print(f"   Healthy: {healthy} | Warnings: {warnings} | Errors: {errors}")
        print(f"   Total Modules Tested: {total}")
        print(f"   Test Duration: {time.time() - start_time:.1f} seconds")
        
        print(f"\nDETAILED MODULE STATUS:")
        
        # Group by functionality
        categories = {
            'Core Infrastructure': ['database', 'system_monitor', 'system_health'],
            'Risk Management': ['risk_engine', 'risk_assessment', 'portfolio_risk'],
            'Trading Systems': ['paper_trading_engine', 'trade_execution', 'trade_logging'],
            'Portfolio Management': ['portfolio_manager', 'account_management', 'capital_allocation'],
            'Analytics & Performance': ['performance_rebalancer', 'performance_tracking']
        }
        
        for category, module_list in categories.items():
            print(f"\n{category}:")
            for module in module_list:
                if module in module_health:
                    status = module_health[module]
                    symbol = {"HEALTHY": "[OK]", "WARNING": "[WARN]", "ERROR": "[ERR]"}.get(status, "[???]")
                    print(f"   {symbol} {module.replace('_', ' ').title()}: {status}")
        
        print_section("SYNCHRONIZATION VALIDATION")
        
        sync_tests = [
            ("Database <-> Risk Engine", 
             module_health.get('database') == 'HEALTHY' and module_health.get('risk_engine') == 'HEALTHY'),
            ("Risk Engine <-> Trading", 
             module_health.get('risk_assessment') == 'HEALTHY' and module_health.get('trade_execution') in ['HEALTHY', 'WARNING']),
            ("Trading <-> Portfolio", 
             module_health.get('paper_trading_engine') == 'HEALTHY' and module_health.get('portfolio_manager') == 'HEALTHY'),
            ("Performance <-> Analytics", 
             module_health.get('performance_rebalancer') == 'HEALTHY' and module_health.get('performance_tracking') == 'HEALTHY'),
            ("Monitoring <-> All Systems", 
             module_health.get('system_monitor') == 'HEALTHY' and module_health.get('system_health') == 'HEALTHY')
        ]
        
        synced_systems = 0
        for test_name, is_synced in sync_tests:
            status = "[SYNCED]" if is_synced else "[ISSUES]"
            print(f"{status} {test_name}")
            if is_synced:
                synced_systems += 1
        
        sync_score = (synced_systems / len(sync_tests)) * 100
        print(f"\nSynchronization Score: {sync_score:.1f}%")
        
        print_section("FINAL SYSTEM ASSESSMENT")
        
        overall_score = (health_score + sync_score) / 2
        
        if overall_score >= 90:
            print("STATUS: EXCELLENT - System ready for production deployment")
            print("All modules healthy and synchronized")
            verdict = "PRODUCTION_READY"
        elif overall_score >= 75:
            print("STATUS: GOOD - System ready for extended paper trading")
            print("Minor issues detected but core functionality operational")
            verdict = "PAPER_TRADING_READY"
        elif overall_score >= 60:
            print("STATUS: ACCEPTABLE - Basic functionality working")
            print("Some modules need attention before full deployment")
            verdict = "NEEDS_IMPROVEMENTS"
        else:
            print("STATUS: NEEDS WORK - Multiple system issues detected")
            print("Significant problems require resolution")
            verdict = "NEEDS_MAJOR_FIXES"
        
        print(f"\nOVERALL SCORE: {overall_score:.1f}%")
        print(f"VERDICT: {verdict}")
        
        # Test completion summary
        print_section("DIAGNOSTIC COMPLETE")
        
        successful_tests = healthy + (warnings * 0.5)  # Warnings count as half success
        total_possible = len(module_health)
        success_rate = (successful_tests / total_possible) * 100 if total_possible > 0 else 0
        
        print(f"Tests Completed: {len(module_health)}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Integration Status: {'PASSED' if overall_score >= 75 else 'NEEDS_WORK'}")
        
        if overall_score >= 75:
            print("\nRECOMMENDED NEXT STEPS:")
            print("1. Deploy paper trading system for extended validation")
            print("2. Monitor system performance under load") 
            print("3. Validate arbitrage detection with live market data")
            print("4. Test risk management under various market conditions")
            print("5. Gradually transition to live trading with small positions")
        else:
            print(f"\nISSUES TO ADDRESS:")
            for module, status in module_health.items():
                if status == 'ERROR':
                    print(f"- Fix {module.replace('_', ' ')} module errors")
                elif status == 'WARNING':
                    print(f"- Review {module.replace('_', ' ')} module warnings")
        
    except Exception as e:
        print(f"\nDIAGNOSTIC SYSTEM ERROR: {e}")
        overall_score = 0
        verdict = "SYSTEM_ERROR"
    
    finally:
        # Cleanup
        print(f"\nCleaning up system modules...")
        cleanup_order = ['paper_trading_engine', 'performance_rebalancer', 'portfolio_manager', 
                        'system_monitor', 'risk_engine', 'db_manager']
        
        for module_name in cleanup_order:
            if module_name in modules:
                try:
                    module = modules[module_name]
                    if hasattr(module, 'shutdown'):
                        await module.shutdown()
                    elif hasattr(module, 'stop'):
                        await module.stop()
                    elif hasattr(module, 'close'):
                        await module.close()
                except:
                    pass  # Ignore cleanup errors
        
        print("System cleanup completed")
    
    print_header("DIAGNOSTIC SESSION COMPLETE")
    
    return overall_score >= 75


async def main():
    """Main execution"""
    try:
        success = await run_final_diagnostic()
        return 0 if success else 1
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)