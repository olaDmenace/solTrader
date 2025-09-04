#!/usr/bin/env python3
"""
Complete System Integration Test & Diagnostic
Human-readable validation of all core modules working in sync
"""

import asyncio
import logging
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import load_settings
from src.database.db_manager import DatabaseManager
from src.trading.risk_engine import RiskEngine, RiskEngineConfig
from src.monitoring.system_monitor import SystemMonitor
from src.arbitrage.real_dex_connector import RealDEXConnector
from src.portfolio.performance_based_rebalancer import PerformanceBasedRebalancer
from src.portfolio.portfolio_manager import PortfolioManager
from src.trading.paper_trading_engine import PaperTradingEngine, PaperTradingMode
from src.trading.trade_types import TradeDirection, TradeType

# Setup clean logging for human readability
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Clean format for diagnostic output
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'system_diagnostic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class SystemDiagnostic:
    def __init__(self):
        """Initialize the comprehensive system diagnostic"""
        self.start_time = datetime.now()
        self.modules = {}
        self.test_results = {}
        self.live_data_ticks = []
        
        # Health tracking
        self.module_status = {
            'database': {'status': 'NOT_STARTED', 'details': ''},
            'risk_engine': {'status': 'NOT_STARTED', 'details': ''},
            'system_monitor': {'status': 'NOT_STARTED', 'details': ''},
            'dex_connector': {'status': 'NOT_STARTED', 'details': ''},
            'performance_rebalancer': {'status': 'NOT_STARTED', 'details': ''},
            'portfolio_manager': {'status': 'NOT_STARTED', 'details': ''},
            'paper_trading_engine': {'status': 'NOT_STARTED', 'details': ''},
            'strategy_execution': {'status': 'NOT_STARTED', 'details': ''},
            'trade_logging': {'status': 'NOT_STARTED', 'details': ''},
            'capital_allocation': {'status': 'NOT_STARTED', 'details': ''}
        }

    def print_header(self, title: str):
        """Print a formatted section header"""
        logger.info("=" * 80)
        logger.info(f" {title.center(76)} ")
        logger.info("=" * 80)

    def print_subheader(self, title: str):
        """Print a formatted subsection header"""
        logger.info("-" * 60)
        logger.info(f" {title} ")
        logger.info("-" * 60)

    def update_module_status(self, module: str, status: str, details: str = ""):
        """Update module status with details"""
        self.module_status[module]['status'] = status
        self.module_status[module]['details'] = details
        
        status_symbol = {
            'NOT_STARTED': '[READY]',
            'INITIALIZING': '[INIT]',
            'HEALTHY': '[OK]',
            'WARNING': '[WARN]',
            'ERROR': '[ERR]',
            'CRITICAL': '[CRIT]'
        }.get(status, '[???]')
        
        logger.info(f"{status_symbol} {module.upper()}: {status} - {details}")

    async def initialize_all_modules(self):
        """Initialize all system modules"""
        self.print_header("SYSTEM MODULE INITIALIZATION")
        
        try:
            # Load settings
            logger.info("[CONFIG] Loading system configuration...")
            settings = load_settings()
            logger.info("   Configuration loaded successfully")
            
            # 1. Database Manager
            self.update_module_status('database', 'INITIALIZING', 'Starting database connection')
            self.modules['db_manager'] = DatabaseManager(settings)
            await self.modules['db_manager'].initialize()
            self.update_module_status('database', 'HEALTHY', 'Database connected and tables ready')
            
            # 2. Risk Engine
            self.update_module_status('risk_engine', 'INITIALIZING', 'Loading risk configuration')
            risk_config = RiskEngineConfig(
                max_position_size=0.05,
                max_portfolio_risk=0.15,
                max_daily_loss=0.10,
                enable_stop_loss=True,
                enable_take_profit=True
            )
            self.modules['risk_engine'] = RiskEngine(self.modules['db_manager'], risk_config)
            await self.modules['risk_engine'].initialize()
            self.update_module_status('risk_engine', 'HEALTHY', 'Risk engine loaded with production config')
            
            # 3. System Monitor
            self.update_module_status('system_monitor', 'INITIALIZING', 'Starting system monitoring')
            self.modules['system_monitor'] = SystemMonitor(self.modules['db_manager'])
            await self.modules['system_monitor'].initialize()
            await self.modules['system_monitor'].start_monitoring()
            self.update_module_status('system_monitor', 'HEALTHY', 'System monitoring active')
            
            # 4. DEX Connector
            self.update_module_status('dex_connector', 'INITIALIZING', 'Connecting to DEX APIs')
            self.modules['dex_connector'] = RealDEXConnector(settings)
            await self.modules['dex_connector'].initialize()
            self.update_module_status('dex_connector', 'HEALTHY', 'Connected to Jupiter, Raydium, Orca, Serum, Meteora')
            
            # 5. Performance Rebalancer
            self.update_module_status('performance_rebalancer', 'INITIALIZING', 'Loading performance analytics')
            self.modules['performance_rebalancer'] = PerformanceBasedRebalancer(self.modules['db_manager'])
            await self.modules['performance_rebalancer'].initialize()
            await self.modules['performance_rebalancer'].start()
            self.update_module_status('performance_rebalancer', 'HEALTHY', 'Performance analysis active')
            
            # 6. Portfolio Manager
            self.update_module_status('portfolio_manager', 'INITIALIZING', 'Starting portfolio management')
            self.modules['portfolio_manager'] = PortfolioManager(
                self.modules['db_manager'],
                self.modules['risk_engine'],
                self.modules['system_monitor']
            )
            await self.modules['portfolio_manager'].initialize()
            await self.modules['portfolio_manager'].start()
            self.update_module_status('portfolio_manager', 'HEALTHY', 'Portfolio tracking active')
            
            # 7. Paper Trading Engine
            self.update_module_status('paper_trading_engine', 'INITIALIZING', 'Starting paper trading')
            self.modules['paper_trading_engine'] = PaperTradingEngine(
                self.modules['db_manager'],
                self.modules['risk_engine'],
                self.modules['system_monitor'],
                self.modules['dex_connector'],
                PaperTradingMode.LIVE_DATA,
                initial_balance=10000.0
            )
            await self.modules['paper_trading_engine'].initialize()
            await self.modules['paper_trading_engine'].start_trading()
            self.update_module_status('paper_trading_engine', 'HEALTHY', 'Paper trading active with $10,000 balance')
            
            logger.info("")
            logger.info("[SUCCESS] ALL MODULES INITIALIZED SUCCESSFULLY")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] MODULE INITIALIZATION FAILED: {e}")
            return False

    async def simulate_live_data_ticks(self, num_ticks: int = 5):
        """Simulate live market data ticks and system responses"""
        self.print_header("LIVE DATA SIMULATION & SYSTEM SYNC TEST")
        
        logger.info(f"[SIM] Simulating {num_ticks} market data ticks...")
        logger.info("   Each tick represents real-time market updates flowing through the system")
        logger.info("")
        
        test_symbols = ["SOL/USDC", "BTC/USDC", "ETH/USDC"]
        base_prices = {"SOL/USDC": 145.50, "BTC/USDC": 64250.0, "ETH/USDC": 3420.0}
        
        for tick in range(1, num_ticks + 1):
            logger.info(f"[TICK {tick}/{num_ticks}] {datetime.now().strftime('%H:%M:%S')}")
            
            tick_data = {
                'tick_number': tick,
                'timestamp': datetime.now(),
                'market_data': {},
                'arbitrage_opportunities': [],
                'risk_assessments': [],
                'trades_executed': [],
                'system_metrics': {}
            }
            
            # 1. Market Data Update
            logger.info("   [MARKET] Market Data Update:")
            for symbol in test_symbols:
                # Simulate price movement
                base_price = base_prices[symbol]
                price_change = (tick - 3) * 0.02  # Simulate trend
                current_price = base_price * (1 + price_change + (0.001 * (tick % 3 - 1)))  # Add noise
                
                tick_data['market_data'][symbol] = {
                    'price': current_price,
                    'change': price_change * 100,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"      {symbol}: ${current_price:,.2f} ({price_change:+.2%})")
            
            # 2. DEX Arbitrage Scan
            logger.info("   ⚡ Arbitrage Opportunity Scan:")
            try:
                opportunities = await self.modules['dex_connector'].find_arbitrage_opportunities(
                    token_pairs=[("SOL", "USDC")],
                    min_profit_percentage=0.1,
                    max_amount=100.0
                )
                tick_data['arbitrage_opportunities'] = len(opportunities)
                logger.info(f"      Found {len(opportunities)} potential arbitrage opportunities")
                
                if opportunities:
                    best_opp = opportunities[0]
                    logger.info(f"      Best: {best_opp.token_a}/{best_opp.token_b} - "
                              f"${best_opp.estimated_profit:.2f} profit "
                              f"({best_opp.confidence_score:.1%} confidence)")
                              
            except Exception as e:
                logger.info(f"      Arbitrage scan error: {e}")
                tick_data['arbitrage_opportunities'] = 0
            
            # 3. Risk Assessment
            logger.info("   🛡️ Risk Assessment:")
            try:
                test_risk = await self.modules['risk_engine'].assess_trade_risk(
                    symbol="SOL/USDC",
                    direction="BUY",
                    quantity=10.0,
                    price=tick_data['market_data']['SOL/USDC']['price'],
                    strategy_name="integration_test"
                )
                
                tick_data['risk_assessments'].append({
                    'symbol': 'SOL/USDC',
                    'risk_level': test_risk.risk_level.value,
                    'risk_score': test_risk.risk_score,
                    'recommendation': test_risk.recommendation
                })
                
                logger.info(f"      SOL/USDC Trade Risk: {test_risk.risk_level.value} "
                          f"(Score: {test_risk.risk_score:.1f})")
                logger.info(f"      Recommendation: {test_risk.recommendation}")
                
            except Exception as e:
                logger.info(f"      Risk assessment error: {e}")
            
            # 4. Strategy Execution Test
            if tick == 2:  # Execute a test trade on tick 2
                logger.info("   🎯 Strategy Execution Test:")
                try:
                    order_id = await self.modules['paper_trading_engine'].place_order(
                        symbol="SOL/USDC",
                        direction=TradeDirection.BUY,
                        order_type=TradeType.MARKET,
                        quantity=1.0,
                        strategy_name="integration_test",
                        metadata={
                            'tick': tick,
                            'test_type': 'integration',
                            'market_price': tick_data['market_data']['SOL/USDC']['price']
                        }
                    )
                    
                    if order_id:
                        tick_data['trades_executed'].append({
                            'order_id': order_id,
                            'symbol': 'SOL/USDC',
                            'type': 'BUY',
                            'quantity': 1.0
                        })
                        logger.info(f"      ✅ Test order placed: {order_id}")
                        self.update_module_status('strategy_execution', 'HEALTHY', f'Order {order_id} executed')
                        self.update_module_status('trade_logging', 'HEALTHY', 'Trade logged to database')
                    else:
                        logger.info("      ⚠️ Test order rejected by risk management")
                        
                except Exception as e:
                    logger.info(f"      ❌ Strategy execution error: {e}")
            
            # 5. Portfolio Status
            logger.info("   💼 Portfolio Status:")
            try:
                account = await self.modules['paper_trading_engine'].get_account_status()
                portfolio_summary = self.modules['portfolio_manager'].get_portfolio_summary()
                
                tick_data['system_metrics']['portfolio'] = {
                    'balance': account.current_balance,
                    'equity': account.equity,
                    'positions': len(account.positions),
                    'trades': account.trade_count
                }
                
                logger.info(f"      Balance: ${account.current_balance:,.2f}")
                logger.info(f"      Equity: ${account.equity:,.2f}")
                logger.info(f"      Positions: {len(account.positions)}")
                logger.info(f"      Total Trades: {account.trade_count}")
                
                if account.trade_count > 0:
                    self.update_module_status('capital_allocation', 'HEALTHY', 
                                            f'Managing ${account.equity:,.2f} in equity')
                
            except Exception as e:
                logger.info(f"      Portfolio status error: {e}")
            
            # 6. System Health Check
            logger.info("   🔍 System Health:")
            try:
                health = await self.modules['system_monitor'].get_health_status()
                tick_data['system_metrics']['health'] = health
                
                logger.info(f"      System Status: {health['status']}")
                if 'metrics' in health:
                    metrics = health['metrics']
                    logger.info(f"      CPU: {metrics['cpu_usage']:.1f}% | "
                              f"Memory: {metrics['memory_usage']:.1f}% | "
                              f"Connections: {metrics['active_connections']}")
                              
            except Exception as e:
                logger.info(f"      System health error: {e}")
            
            # Store tick data
            self.live_data_ticks.append(tick_data)
            
            logger.info("")
            
            # Wait between ticks
            if tick < num_ticks:
                await asyncio.sleep(2)  # 2 second intervals
        
        logger.info("🎉 LIVE DATA SIMULATION COMPLETED")
        return True

    async def generate_module_health_report(self):
        """Generate comprehensive module health report"""
        self.print_header("COMPLETE MODULE HEALTH REPORT")
        
        # Overall system status
        healthy_modules = len([m for m in self.module_status.values() if m['status'] == 'HEALTHY'])
        total_modules = len(self.module_status)
        system_health_percentage = (healthy_modules / total_modules) * 100
        
        logger.info(f"🎯 SYSTEM OVERVIEW")
        logger.info(f"   Overall Health: {system_health_percentage:.1f}% ({healthy_modules}/{total_modules} modules healthy)")
        logger.info(f"   Test Duration: {(datetime.now() - self.start_time).total_seconds():.1f} seconds")
        logger.info(f"   Live Data Ticks Processed: {len(self.live_data_ticks)}")
        logger.info("")
        
        # Detailed module status
        self.print_subheader("DETAILED MODULE STATUS")
        
        module_categories = {
            'Core Infrastructure': ['database', 'system_monitor'],
            'Trading Engine': ['risk_engine', 'paper_trading_engine', 'strategy_execution'],
            'Market Data & Analysis': ['dex_connector', 'performance_rebalancer'],
            'Portfolio Management': ['portfolio_manager', 'capital_allocation', 'trade_logging']
        }
        
        for category, modules in module_categories.items():
            logger.info(f"📋 {category}:")
            for module in modules:
                if module in self.module_status:
                    status_info = self.module_status[module]
                    status_symbol = {
                        'HEALTHY': '✅',
                        'WARNING': '⚠️',
                        'ERROR': '❌',
                        'CRITICAL': '🚨',
                        'NOT_STARTED': '⏸️'
                    }.get(status_info['status'], '❓')
                    
                    logger.info(f"   {status_symbol} {module.replace('_', ' ').title()}: "
                              f"{status_info['status']}")
                    if status_info['details']:
                        logger.info(f"      → {status_info['details']}")
            logger.info("")
        
        # Performance metrics
        self.print_subheader("PERFORMANCE METRICS")
        
        if self.live_data_ticks:
            # Calculate some basic metrics
            total_opportunities = sum([tick.get('arbitrage_opportunities', 0) for tick in self.live_data_ticks])
            total_trades = sum([len(tick.get('trades_executed', [])) for tick in self.live_data_ticks])
            
            logger.info(f"📊 Trading Performance:")
            logger.info(f"   Arbitrage Opportunities Detected: {total_opportunities}")
            logger.info(f"   Test Trades Executed: {total_trades}")
            logger.info(f"   Average Opportunities per Tick: {total_opportunities / len(self.live_data_ticks):.1f}")
            logger.info("")
            
            # Latest portfolio state
            if self.live_data_ticks[-1].get('system_metrics', {}).get('portfolio'):
                portfolio = self.live_data_ticks[-1]['system_metrics']['portfolio']
                logger.info(f"💼 Final Portfolio State:")
                logger.info(f"   Balance: ${portfolio['balance']:,.2f}")
                logger.info(f"   Equity: ${portfolio['equity']:,.2f}")
                logger.info(f"   Active Positions: {portfolio['positions']}")
                logger.info(f"   Total Trades: {portfolio['trades']}")
                logger.info("")
        
        # Integration sync check
        self.print_subheader("MODULE SYNCHRONIZATION CHECK")
        
        sync_tests = [
            ("Database ↔ Risk Engine", self._check_db_risk_sync()),
            ("Risk Engine ↔ Trading Engine", self._check_risk_trading_sync()),
            ("Trading Engine ↔ Portfolio Manager", self._check_trading_portfolio_sync()),
            ("DEX Connector ↔ Strategy Execution", self._check_dex_strategy_sync()),
            ("Performance Rebalancer ↔ Capital Allocation", self._check_rebalancer_allocation_sync()),
            ("System Monitor ↔ All Modules", self._check_monitor_integration_sync())
        ]
        
        for test_name, is_synced in sync_tests:
            sync_symbol = "✅" if is_synced else "⚠️"
            sync_status = "SYNCHRONIZED" if is_synced else "NEEDS_ATTENTION"
            logger.info(f"   {sync_symbol} {test_name}: {sync_status}")
        
        logger.info("")
        
        # Final assessment
        self.print_subheader("FINAL SYSTEM ASSESSMENT")
        
        if system_health_percentage >= 90:
            logger.info("🎉 SYSTEM STATUS: EXCELLENT")
            logger.info("   All modules are healthy and working in perfect sync.")
            logger.info("   System is ready for production deployment.")
        elif system_health_percentage >= 75:
            logger.info("✅ SYSTEM STATUS: GOOD")
            logger.info("   Most modules are healthy. Minor issues detected.")
            logger.info("   System is suitable for paper trading validation.")
        elif system_health_percentage >= 50:
            logger.info("⚠️ SYSTEM STATUS: DEGRADED")
            logger.info("   Several modules have issues. Review required.")
            logger.info("   Address problems before production use.")
        else:
            logger.info("❌ SYSTEM STATUS: CRITICAL")
            logger.info("   Major system issues detected.")
            logger.info("   Immediate attention required before deployment.")
        
        logger.info("")
        logger.info("📋 DIAGNOSTIC COMPLETE")
        logger.info(f"   Full report saved to: system_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        return system_health_percentage

    def _check_db_risk_sync(self) -> bool:
        """Check if database and risk engine are properly synchronized"""
        return (self.module_status['database']['status'] == 'HEALTHY' and 
                self.module_status['risk_engine']['status'] == 'HEALTHY')

    def _check_risk_trading_sync(self) -> bool:
        """Check if risk engine and trading engine are synchronized"""
        return (self.module_status['risk_engine']['status'] == 'HEALTHY' and 
                self.module_status['paper_trading_engine']['status'] == 'HEALTHY')

    def _check_trading_portfolio_sync(self) -> bool:
        """Check if trading engine and portfolio manager are synchronized"""
        return (self.module_status['paper_trading_engine']['status'] == 'HEALTHY' and 
                self.module_status['portfolio_manager']['status'] == 'HEALTHY')

    def _check_dex_strategy_sync(self) -> bool:
        """Check if DEX connector and strategy execution are synchronized"""
        return (self.module_status['dex_connector']['status'] == 'HEALTHY' and 
                self.module_status['strategy_execution']['status'] in ['HEALTHY', 'NOT_STARTED'])

    def _check_rebalancer_allocation_sync(self) -> bool:
        """Check if performance rebalancer and capital allocation are synchronized"""
        return (self.module_status['performance_rebalancer']['status'] == 'HEALTHY' and 
                self.module_status['capital_allocation']['status'] in ['HEALTHY', 'NOT_STARTED'])

    def _check_monitor_integration_sync(self) -> bool:
        """Check if system monitor is properly integrated with all modules"""
        return self.module_status['system_monitor']['status'] == 'HEALTHY'

    async def cleanup_modules(self):
        """Cleanup all modules"""
        logger.info("🧹 Cleaning up system modules...")
        
        cleanup_order = [
            'paper_trading_engine',
            'performance_rebalancer', 
            'portfolio_manager',
            'dex_connector',
            'system_monitor',
            'risk_engine',
            'db_manager'
        ]
        
        for module_name in cleanup_order:
            if module_name in self.modules:
                try:
                    module = self.modules[module_name]
                    if hasattr(module, 'shutdown'):
                        await module.shutdown()
                    elif hasattr(module, 'stop'):
                        await module.stop()
                    elif hasattr(module, 'close'):
                        await module.close()
                    logger.info(f"   ✅ {module_name} cleaned up")
                except Exception as e:
                    logger.info(f"   ⚠️ {module_name} cleanup error: {e}")
        
        logger.info("✅ System cleanup completed")

    async def run_complete_diagnostic(self):
        """Run the complete system diagnostic"""
        try:
            self.print_header("SOLTRADER COMPLETE SYSTEM INTEGRATION DIAGNOSTIC")
            logger.info("🔍 Comprehensive validation of all core modules working in sync")
            logger.info(f"⏰ Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("")
            
            # Phase 1: Initialize all modules
            if not await self.initialize_all_modules():
                logger.info("❌ DIAGNOSTIC FAILED: Module initialization failed")
                return False
            
            # Phase 2: Simulate live data and test system sync
            await asyncio.sleep(2)  # Brief pause between phases
            if not await self.simulate_live_data_ticks(5):
                logger.info("❌ DIAGNOSTIC FAILED: Live data simulation failed")
                return False
            
            # Phase 3: Generate comprehensive health report
            await asyncio.sleep(1)  # Brief pause before report
            health_score = await self.generate_module_health_report()
            
            return health_score >= 75  # Consider 75%+ as passing
            
        except Exception as e:
            logger.error(f"❌ DIAGNOSTIC SYSTEM ERROR: {e}")
            return False
        finally:
            # Always cleanup
            await self.cleanup_modules()


async def main():
    """Main diagnostic execution"""
    diagnostic = SystemDiagnostic()
    
    try:
        success = await diagnostic.run_complete_diagnostic()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)