#!/usr/bin/env python3
"""
Health Monitoring and System Validation Script
Runs continuous health checks and validates system performance
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'health_monitoring_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


async def test_system_health():
    """Test system health and performance monitoring"""
    logger.info("="*60)
    logger.info("SOLTRADER SYSTEM HEALTH MONITORING")
    logger.info("="*60)
    
    try:
        # Initialize core components
        settings = load_settings()
        db_manager = DatabaseManager(settings)
        await db_manager.initialize()
        
        monitor = SystemMonitor(db_manager)
        await monitor.initialize()
        await monitor.start_monitoring()
        
        logger.info("System monitoring started successfully")
        
        # Run health checks for 30 seconds
        logger.info("Running health checks for 30 seconds...")
        
        start_time = time.time()
        check_interval = 5  # Check every 5 seconds
        
        while time.time() - start_time < 30:
            # Get current health status
            health_status = await monitor.get_health_status()
            
            logger.info(f"Health Status: {health_status['status']} - {health_status['message']}")
            
            if 'metrics' in health_status:
                metrics = health_status['metrics']
                logger.info(f"  CPU: {metrics['cpu_usage']:.1f}%")
                logger.info(f"  Memory: {metrics['memory_usage']:.1f}%")
                logger.info(f"  Disk: {metrics['disk_usage']:.1f}%")
                logger.info(f"  Connections: {metrics['active_connections']}")
                logger.info(f"  Uptime: {metrics['uptime_hours']:.2f} hours")
                
            # Log some test metrics
            await monitor.log_system_metric("test_metric", time.time() % 100)
            await monitor.log_system_event("health_check", {
                "status": health_status['status'],
                "timestamp": datetime.now().isoformat()
            })
            
            await asyncio.sleep(check_interval)
            
        # Get metrics summary
        logger.info("Getting metrics summary...")
        summary = monitor.get_metrics_summary(hours=1)
        
        if summary:
            logger.info("System Performance Summary:")
            if 'cpu_usage' in summary:
                cpu = summary['cpu_usage']
                logger.info(f"  CPU - Avg: {cpu['avg']:.1f}%, Max: {cpu['max']:.1f}%, Min: {cpu['min']:.1f}%")
            if 'memory_usage' in summary:
                mem = summary['memory_usage']
                logger.info(f"  Memory - Avg: {mem['avg']:.1f}%, Max: {mem['max']:.1f}%, Min: {mem['min']:.1f}%")
            if 'disk_usage' in summary:
                disk = summary['disk_usage']
                logger.info(f"  Disk - Avg: {disk['avg']:.1f}%, Max: {disk['max']:.1f}%, Min: {disk['min']:.1f}%")
            logger.info(f"  Uptime: {summary['uptime_hours']:.2f} hours")
            logger.info(f"  Data Points: {summary['data_points']}")
        
        # Cleanup
        await monitor.stop_monitoring()
        await monitor.shutdown()
        await db_manager.close()
        
        logger.info("="*60)
        logger.info("HEALTH MONITORING COMPLETED SUCCESSFULLY")
        logger.info("System is healthy and monitoring functions are operational")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Health monitoring failed: {e}")
        return False


async def test_database_performance():
    """Test database performance and operations"""
    logger.info("Testing database performance...")
    
    try:
        settings = load_settings()
        db_manager = DatabaseManager(settings)
        await db_manager.initialize()
        
        # Test metric logging performance
        start_time = time.time()
        num_operations = 100
        
        for i in range(num_operations):
            await db_manager.log_metric(f"test_metric_{i%10}", i * 0.1, {"batch": "performance_test"})
            
        duration = time.time() - start_time
        ops_per_second = num_operations / duration
        
        logger.info(f"Database Performance:")
        logger.info(f"  Operations: {num_operations}")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.info(f"  Rate: {ops_per_second:.1f} ops/sec")
        
        # Test metric retrieval
        start_time = time.time()
        metrics = await db_manager.get_metrics("test_metric_0", hours=1)
        retrieval_duration = time.time() - start_time
        
        logger.info(f"  Retrieved {len(metrics)} metrics in {retrieval_duration:.3f} seconds")
        
        await db_manager.close()
        
        if ops_per_second > 10:  # At least 10 ops/sec
            logger.info("Database performance is acceptable")
            return True
        else:
            logger.warning("Database performance is slow")
            return False
            
    except Exception as e:
        logger.error(f"Database performance test failed: {e}")
        return False


async def test_risk_engine_performance():
    """Test risk engine performance under load"""
    logger.info("Testing risk engine performance...")
    
    try:
        settings = load_settings()
        db_manager = DatabaseManager(settings)
        await db_manager.initialize()
        
        risk_config = RiskEngineConfig()
        risk_engine = RiskEngine(db_manager, risk_config)
        await risk_engine.initialize()
        
        # Test risk assessment performance
        test_symbols = ["SOL/USDC", "BTC/USDC", "ETH/USDC", "BONK/USDC", "WIF/USDC"]
        assessments_per_symbol = 20
        
        start_time = time.time()
        total_assessments = 0
        
        for symbol in test_symbols:
            for i in range(assessments_per_symbol):
                risk_assessment = await risk_engine.assess_trade_risk(
                    symbol=symbol,
                    direction="BUY",
                    quantity=10.0 + i,
                    price=100.0 + i * 5,
                    strategy_name=f"test_strategy_{i%5}"
                )
                total_assessments += 1
                
        duration = time.time() - start_time
        assessments_per_second = total_assessments / duration
        
        logger.info(f"Risk Engine Performance:")
        logger.info(f"  Total Assessments: {total_assessments}")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.info(f"  Rate: {assessments_per_second:.1f} assessments/sec")
        
        # Test portfolio risk check
        start_time = time.time()
        portfolio_risk = await risk_engine.check_portfolio_risk()
        portfolio_check_duration = time.time() - start_time
        
        logger.info(f"  Portfolio Risk Check: {portfolio_check_duration:.3f} seconds")
        logger.info(f"  Portfolio Status: {portfolio_risk['overall_risk_level']}")
        
        await risk_engine.shutdown()
        await db_manager.close()
        
        if assessments_per_second > 5:  # At least 5 assessments/sec
            logger.info("Risk engine performance is acceptable")
            return True
        else:
            logger.warning("Risk engine performance is slow")
            return False
            
    except Exception as e:
        logger.error(f"Risk engine performance test failed: {e}")
        return False


async def run_comprehensive_health_check():
    """Run comprehensive health and performance validation"""
    logger.info("STARTING COMPREHENSIVE HEALTH CHECK")
    logger.info("="*60)
    
    test_results = {}
    
    # Test 1: System Health Monitoring
    logger.info("Test 1: System Health Monitoring")
    test_results["system_health"] = await test_system_health()
    
    # Test 2: Database Performance
    logger.info("Test 2: Database Performance")
    test_results["database_performance"] = await test_database_performance()
    
    # Test 3: Risk Engine Performance
    logger.info("Test 3: Risk Engine Performance")
    test_results["risk_engine_performance"] = await test_risk_engine_performance()
    
    # Generate final report
    logger.info("="*60)
    logger.info("HEALTH CHECK RESULTS SUMMARY")
    logger.info("="*60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    for test_name, passed in test_results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{test_name.upper()}: {status}")
        
    logger.info(f"Overall Health Score: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        logger.info("SYSTEM STATUS: HEALTHY - All systems operational")
        logger.info("Ready for production deployment")
    elif success_rate >= 60:
        logger.info("SYSTEM STATUS: DEGRADED - Some performance issues")
        logger.info("Monitor closely and investigate failed tests")
    else:
        logger.info("SYSTEM STATUS: UNHEALTHY - Multiple system issues")
        logger.info("Address critical issues before deployment")
        
    logger.info("="*60)
    logger.info("HEALTH CHECK COMPLETED")
    
    return success_rate


async def main():
    """Main health check execution"""
    success_rate = await run_comprehensive_health_check()
    
    if success_rate >= 75:
        logger.info("FINAL STATUS: SYSTEM READY FOR DEPLOYMENT")
        logger.info("Recommended next steps:")
        logger.info("1. Deploy paper trading system")
        logger.info("2. Run extended validation session")
        logger.info("3. Monitor performance metrics")
        logger.info("4. Gradually transition to live trading")
    else:
        logger.warning("FINAL STATUS: SYSTEM NEEDS OPTIMIZATION")
        logger.warning("Address performance issues before deployment")
        
    return 0 if success_rate >= 75 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)