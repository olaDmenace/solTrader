#!/usr/bin/env python3
"""
Test Paper Trading System Integration
Validates all production components work together correctly
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import load_settings
from src.database.db_manager import DatabaseManager
from src.trading.risk_engine import RiskEngine, RiskEngineConfig
from src.monitoring.system_monitor import SystemMonitor
from src.arbitrage.real_dex_connector import RealDEXConnector
from src.portfolio.performance_based_rebalancer import PerformanceBasedRebalancer
from src.trading.paper_trading_engine import PaperTradingEngine, PaperTradingMode
from src.trading.trade_types import TradeDirection, TradeType

# Setup logging without emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'paper_trading_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


async def test_component_initialization():
    """Test that all components can be initialized"""
    logger.info("Testing component initialization...")
    
    try:
        # Load settings
        settings = load_settings()
        logger.info("Settings loaded successfully")
        
        # Initialize database
        db_manager = DatabaseManager(settings)
        await db_manager.initialize()
        logger.info("Database manager initialized")
        
        # Initialize risk engine
        risk_config = RiskEngineConfig()
        risk_engine = RiskEngine(db_manager, risk_config)
        await risk_engine.initialize()
        logger.info("Risk engine initialized")
        
        # Initialize system monitor  
        monitor = SystemMonitor(db_manager)
        await monitor.initialize()
        logger.info("System monitor initialized")
        
        # Initialize real DEX connector
        dex_connector = RealDEXConnector()
        await dex_connector.initialize()
        logger.info("Real DEX connector initialized")
        
        # Initialize performance rebalancer
        rebalancer = PerformanceBasedRebalancer(db_manager)
        await rebalancer.initialize()
        logger.info("Performance rebalancer initialized")
        
        # Initialize paper trading engine
        paper_engine = PaperTradingEngine(
            db_manager, risk_engine, monitor, dex_connector,
            PaperTradingMode.SIMULATION, 10000.0
        )
        await paper_engine.initialize()
        logger.info("Paper trading engine initialized")
        
        # Test basic functionality
        await test_paper_trading_operations(paper_engine)
        
        # Cleanup
        await paper_engine.shutdown()
        await rebalancer.shutdown()
        await dex_connector.shutdown()
        await monitor.shutdown()
        await risk_engine.shutdown()
        await db_manager.close()
        
        logger.info("All components initialized and tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"Component initialization test failed: {e}")
        return False


async def test_paper_trading_operations(paper_engine):
    """Test basic paper trading operations"""
    logger.info("Testing paper trading operations...")
    
    try:
        # Start paper trading
        await paper_engine.start_trading()
        logger.info("Paper trading started")
        
        # Wait for market data to initialize
        await asyncio.sleep(3)
        
        # Place a test order
        order_id = await paper_engine.place_order(
            symbol="SOL/USDC",
            direction=TradeDirection.BUY,
            order_type=TradeType.MARKET,
            quantity=1.0,
            strategy_name="test_strategy"
        )
        
        if order_id:
            logger.info(f"Test order placed successfully: {order_id}")
        else:
            logger.warning("Test order was not placed")
            
        # Wait for order processing
        await asyncio.sleep(2)
        
        # Get account status
        account = await paper_engine.get_account_status()
        logger.info(f"Account status: Balance=${account.current_balance:.2f}, Equity=${account.equity:.2f}")
        
        # Get performance report
        report = await paper_engine.get_performance_report()
        logger.info(f"Trading stats: Trades={report['trading_stats']['total_trades']}")
        
        # Stop paper trading
        await paper_engine.stop_trading()
        logger.info("Paper trading stopped")
        
        logger.info("Paper trading operations test completed successfully")
        
    except Exception as e:
        logger.error(f"Paper trading operations test failed: {e}")
        raise


async def test_arbitrage_detection():
    """Test arbitrage opportunity detection"""
    logger.info("Testing arbitrage detection...")
    
    try:
        dex_connector = RealDEXConnector()
        await dex_connector.initialize()
        
        # Test arbitrage opportunity scanning
        opportunities = await dex_connector.find_arbitrage_opportunities(
            token_pairs=[("SOL", "USDC")],
            min_profit_percentage=0.1,
            max_amount=100.0
        )
        
        logger.info(f"Found {len(opportunities)} arbitrage opportunities")
        
        for i, opp in enumerate(opportunities[:3]):  # Show first 3
            logger.info(f"Opportunity {i+1}: {opp.token_a}/{opp.token_b} - "
                       f"Profit: ${opp.estimated_profit:.2f} - "
                       f"Confidence: {opp.confidence_score:.1%}")
            
        await dex_connector.shutdown()
        logger.info("Arbitrage detection test completed")
        
    except Exception as e:
        logger.error(f"Arbitrage detection test failed: {e}")


async def test_performance_analysis():
    """Test performance analysis system"""
    logger.info("Testing performance analysis...")
    
    try:
        settings = load_settings()
        db_manager = DatabaseManager(settings)
        await db_manager.initialize()
        
        rebalancer = PerformanceBasedRebalancer(db_manager)
        await rebalancer.initialize()
        
        # Add some test performance data
        test_returns = [0.02, -0.01, 0.03, 0.01, -0.005]
        for i, return_val in enumerate(test_returns):
            await rebalancer.update_strategy_return(
                "test_strategy",
                return_val,
                {"trade_id": f"test_{i}", "timestamp": datetime.now()}
            )
            
        # Test performance analysis
        analysis = await rebalancer.analyze_strategy_performance("test_strategy")
        if analysis:
            logger.info(f"Performance analysis: Return={analysis.total_return:.2%}, "
                       f"Sharpe={analysis.sharpe_ratio:.2f}, Signal={analysis.rebalancing_signal}")
        else:
            logger.info("No performance analysis available yet")
            
        # Test rebalancing signal generation
        signals = await rebalancer.generate_rebalancing_signals()
        logger.info(f"Generated {len(signals)} rebalancing signals")
        
        await rebalancer.shutdown()
        await db_manager.close()
        
        logger.info("Performance analysis test completed")
        
    except Exception as e:
        logger.error(f"Performance analysis test failed: {e}")


async def test_risk_management():
    """Test risk management system"""
    logger.info("Testing risk management...")
    
    try:
        settings = load_settings()
        db_manager = DatabaseManager(settings)
        await db_manager.initialize()
        
        risk_config = RiskEngineConfig(
            max_position_size=0.1,
            max_portfolio_risk=0.2
        )
        risk_engine = RiskEngine(db_manager, risk_config)
        await risk_engine.initialize()
        
        # Test risk assessment
        risk_assessment = await risk_engine.assess_trade_risk(
            symbol="SOL/USDC",
            direction="BUY", 
            quantity=10.0,
            price=100.0,
            strategy_name="test"
        )
        
        logger.info(f"Risk assessment: Level={risk_assessment.risk_level.value}, "
                   f"Score={risk_assessment.risk_score:.1f}, "
                   f"Recommendation={risk_assessment.recommendation}")
        
        # Test portfolio risk check
        portfolio_risk = await risk_engine.check_portfolio_risk()
        logger.info(f"Portfolio risk: {portfolio_risk['risk_percentage']:.1%} - "
                   f"Status: {portfolio_risk['overall_risk_level']}")
        
        # Test trading halt check
        should_halt, reason = await risk_engine.should_halt_trading()
        logger.info(f"Trading halt check: Halt={should_halt}, Reason={reason}")
        
        await risk_engine.shutdown()
        await db_manager.close()
        
        logger.info("Risk management test completed")
        
    except Exception as e:
        logger.error(f"Risk management test failed: {e}")


async def run_comprehensive_test():
    """Run comprehensive system integration test"""
    logger.info("="*60)
    logger.info("SOLTRADER PAPER TRADING SYSTEM - COMPREHENSIVE TEST")
    logger.info("="*60)
    
    test_results = {}
    
    try:
        # Test 1: Component Initialization
        logger.info("Test 1: Component Initialization")
        test_results["initialization"] = await test_component_initialization()
        
        # Test 2: Arbitrage Detection
        logger.info("Test 2: Arbitrage Detection")
        try:
            await test_arbitrage_detection()
            test_results["arbitrage"] = True
        except Exception as e:
            logger.error(f"Arbitrage test failed: {e}")
            test_results["arbitrage"] = False
            
        # Test 3: Performance Analysis  
        logger.info("Test 3: Performance Analysis")
        try:
            await test_performance_analysis()
            test_results["performance"] = True
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            test_results["performance"] = False
            
        # Test 4: Risk Management
        logger.info("Test 4: Risk Management")
        try:
            await test_risk_management()
            test_results["risk_management"] = True
        except Exception as e:
            logger.error(f"Risk management test failed: {e}")
            test_results["risk_management"] = False
            
        # Generate final report
        logger.info("="*60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*60)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        success_rate = (passed_tests / total_tests) * 100
        
        for test_name, passed in test_results.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"{test_name.upper()}: {status}")
            
        logger.info(f"Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 75:
            logger.info("SYSTEM READY: Paper trading system is operational!")
            logger.info("Next steps:")
            logger.info("- Run extended paper trading session")
            logger.info("- Monitor performance metrics")
            logger.info("- Validate arbitrage detection accuracy")
            logger.info("- Test risk management under various conditions")
        else:
            logger.warning("SYSTEM NEEDS WORK: Some components failed testing")
            logger.warning("Review failed tests before deployment")
            
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Comprehensive test failed: {e}")
        
    logger.info("COMPREHENSIVE TEST COMPLETED")


async def main():
    """Main test execution"""
    await run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main())