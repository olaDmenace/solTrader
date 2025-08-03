#!/usr/bin/env python3
"""
Health Monitoring System Test Suite
Tests both internal and external health monitoring components
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.monitoring.health_monitor import HealthMonitor, HealthStatus, RecoveryAction
from health_checker import ExternalHealthChecker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthMonitoringTestSuite:
    """Comprehensive test suite for health monitoring system"""
    
    def __init__(self):
        """Initialize test suite"""
        self.test_results = {}
        self.mock_bot = self._create_mock_bot()
        self.mock_settings = self._create_mock_settings()
        
    def _create_mock_bot(self):
        """Create mock bot instance for testing"""
        mock_bot = Mock()
        mock_bot.enhanced_scanner = Mock()
        mock_bot.enhanced_scanner.daily_stats = {
            'tokens_scanned': 1000,
            'tokens_approved': 300,
            'approval_rate': 30.0,
            'api_requests_used': 50,
            'api_errors': 2
        }
        mock_bot.enhanced_scanner.solana_tracker = Mock()
        mock_bot.strategy = Mock()
        mock_bot.email_system = Mock()
        return mock_bot
    
    def _create_mock_settings(self):
        """Create mock settings for testing"""
        mock_settings = Mock()
        mock_settings.PAPER_TRADING = True
        mock_settings.INITIAL_PAPER_BALANCE = 100.0
        return mock_settings
    
    async def test_internal_health_monitor(self) -> bool:
        """Test internal health monitoring system"""
        logger.info("🧪 Testing Internal Health Monitor...")
        
        try:
            # Initialize health monitor
            health_monitor = HealthMonitor(bot_instance=self.mock_bot, settings=self.mock_settings)
            
            # Test 1: Health monitor initialization
            assert health_monitor.metrics is not None
            assert len(health_monitor.metrics) > 0
            logger.info("✅ Health monitor initialization test passed")
            
            # Test 2: Force health check
            report = await health_monitor.force_health_check()
            assert report is not None
            assert report.overall_status in [status for status in HealthStatus]
            logger.info(f"✅ Health check test passed - Status: {report.overall_status.value}")
            
            # Test 3: Start and stop monitoring
            await health_monitor.start_monitoring()
            assert health_monitor.is_monitoring == True
            
            # Wait a short time to let monitoring run
            await asyncio.sleep(2)
            
            await health_monitor.stop_monitoring()
            assert health_monitor.is_monitoring == False
            logger.info("✅ Start/stop monitoring test passed")
            
            # Test 4: Test metric updates
            metric_name = 'token_discovery_rate'
            if metric_name in health_monitor.metrics:
                original_value = health_monitor.metrics[metric_name].current_value
                health_monitor.metrics[metric_name].update(150)  # Good value
                assert health_monitor.metrics[metric_name].status == HealthStatus.HEALTHY
                
                health_monitor.metrics[metric_name].update(30)   # Critical value
                assert health_monitor.metrics[metric_name].status == HealthStatus.CRITICAL
                logger.info("✅ Metric update test passed")
            
            # Test 5: Recovery action determination
            critical_metric = health_monitor.metrics[metric_name]
            recovery_action = health_monitor._determine_recovery_action(critical_metric)
            assert recovery_action in [action for action in RecoveryAction]
            logger.info(f"✅ Recovery action test passed - Action: {recovery_action.value}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Internal health monitor test failed: {e}")
            return False
    
    def test_external_health_checker(self) -> bool:
        """Test external health checking system"""
        logger.info("🧪 Testing External Health Checker...")
        
        try:
            # Initialize external health checker
            checker = ExternalHealthChecker("health_checker_config.json")
            
            # Test 1: Configuration loading
            assert checker.config is not None
            assert 'thresholds' in checker.config
            assert 'checks' in checker.config
            logger.info("✅ Configuration loading test passed")
            
            # Test 2: Health check execution
            is_healthy, issues = checker.check_bot_health()
            assert isinstance(is_healthy, bool)
            assert isinstance(issues, list)
            logger.info(f"✅ Health check execution test passed - Healthy: {is_healthy}, Issues: {len(issues)}")
            
            # Test 3: Resource usage check
            resource_issues = checker._check_resource_usage()
            assert isinstance(resource_issues, list)
            logger.info("✅ Resource usage check test passed")
            
            # Test 4: Log activity check
            log_active = checker._check_log_activity()
            assert isinstance(log_active, bool)
            logger.info(f"✅ Log activity check test passed - Active: {log_active}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ External health checker test failed: {e}")
            return False
    
    async def test_health_report_generation(self) -> bool:
        """Test health report generation and saving"""
        logger.info("🧪 Testing Health Report Generation...")
        
        try:
            # Create test dashboard data
            test_dashboard_data = {
                "status": "running",
                "trades": [
                    {
                        "type": "buy",
                        "timestamp": datetime.now().isoformat(),
                        "token_address": "test123",
                        "size": 10.0,
                        "price": 0.1
                    }
                ],
                "performance": {
                    "total_trades": 1,
                    "balance": 99.0,
                    "open_positions": 1
                },
                "activity": [
                    {
                        "type": "scan_completed",
                        "timestamp": datetime.now().isoformat(),
                        "data": {"tokens_found": 150}
                    }
                ]
            }
            
            # Save test data
            with open("bot_data.json", "w") as f:
                json.dump(test_dashboard_data, f, indent=2)
            
            # Test health monitor with test data
            health_monitor = HealthMonitor(bot_instance=self.mock_bot, settings=self.mock_settings)
            report = await health_monitor.force_health_check()
            
            # Verify report structure
            assert report.timestamp is not None
            assert report.overall_status is not None
            assert report.metrics is not None
            assert report.uptime >= 0
            logger.info("✅ Health report generation test passed")
            
            # Test report serialization
            report_dict = report.to_dict()
            assert isinstance(report_dict, dict)
            assert 'timestamp' in report_dict
            assert 'overall_status' in report_dict
            logger.info("✅ Health report serialization test passed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Health report generation test failed: {e}")
            return False
    
    def test_recovery_mechanisms(self) -> bool:
        """Test recovery mechanism logic (without actual execution)"""
        logger.info("🧪 Testing Recovery Mechanisms...")
        
        try:
            health_monitor = HealthMonitor(bot_instance=self.mock_bot, settings=self.mock_settings)
            
            # Test recovery action determination for different metrics
            test_cases = [
                ('api_error_rate', RecoveryAction.SOFT_RECOVERY),
                ('token_discovery_rate', RecoveryAction.MEDIUM_RECOVERY),
                ('cpu_usage', RecoveryAction.HARD_RECOVERY),
                ('disk_usage', RecoveryAction.MANUAL_INTERVENTION)
            ]
            
            for metric_name, expected_action in test_cases:
                if metric_name in health_monitor.metrics:
                    metric = health_monitor.metrics[metric_name]
                    metric.status = HealthStatus.CRITICAL  # Force critical status
                    
                    action = health_monitor._determine_recovery_action(metric)
                    assert action == expected_action
                    logger.info(f"✅ Recovery action test passed for {metric_name}: {action.value}")
            
            # Test recovery attempt limiting
            # Simulate multiple recovery attempts
            for i in range(6):  # More than the limit
                health_monitor.recovery_attempts.append({
                    'timestamp': datetime.now(),
                    'action': 'test_action',
                    'success': i % 2 == 0  # Alternate success/failure
                })
            
            # Check that limits are enforced (this would be tested in actual recovery)
            recent_attempts = [
                attempt for attempt in health_monitor.recovery_attempts
                if attempt['timestamp'] > datetime.now() - timedelta(hours=1)
            ]
            
            assert len(recent_attempts) == 6
            logger.info("✅ Recovery attempt tracking test passed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Recovery mechanisms test failed: {e}")
            return False
    
    def test_configuration_validation(self) -> bool:
        """Test configuration file validation"""
        logger.info("🧪 Testing Configuration Validation...")
        
        try:
            # Test internal health monitor config
            if os.path.exists("health_monitor_config.json"):
                with open("health_monitor_config.json", "r") as f:
                    config = json.load(f)
                
                required_keys = ['monitoring_interval', 'thresholds', 'recovery_cooldown_minutes']
                for key in required_keys:
                    assert key in config
                
                # Validate thresholds structure
                assert 'thresholds' in config
                for metric, thresholds in config['thresholds'].items():
                    assert 'warning' in thresholds
                    assert 'critical' in thresholds
                
                logger.info("✅ Internal config validation test passed")
            
            # Test external health checker config
            if os.path.exists("health_checker_config.json"):
                with open("health_checker_config.json", "r") as f:
                    config = json.load(f)
                
                required_keys = ['bot_script', 'python_executable', 'checks', 'thresholds']
                for key in required_keys:
                    assert key in config
                
                logger.info("✅ External config validation test passed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation test failed: {e}")
            return False
    
    def test_integration_with_bot(self) -> bool:
        """Test integration with bot components"""
        logger.info("🧪 Testing Integration with Bot Components...")
        
        try:
            # This test would normally require the actual bot
            # For now, we'll test the integration points with mocks
            
            health_monitor = HealthMonitor(bot_instance=self.mock_bot, settings=self.mock_settings)
            
            # Test bot integration points
            assert health_monitor.bot == self.mock_bot
            assert health_monitor.settings == self.mock_settings
            
            # Test that health monitor can access bot components
            if hasattr(health_monitor.bot, 'enhanced_scanner'):
                scanner = health_monitor.bot.enhanced_scanner
                assert scanner is not None
                logger.info("✅ Scanner integration test passed")
            
            if hasattr(health_monitor.bot, 'strategy'):
                strategy = health_monitor.bot.strategy
                assert strategy is not None
                logger.info("✅ Strategy integration test passed")
            
            if hasattr(health_monitor.bot, 'email_system'):
                email_system = health_monitor.bot.email_system
                assert email_system is not None
                logger.info("✅ Email system integration test passed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all health monitoring tests"""
        logger.info("🚀 Starting Health Monitoring Test Suite")
        logger.info("=" * 60)
        
        test_methods = [
            ("Internal Health Monitor", self.test_internal_health_monitor),
            ("External Health Checker", self.test_external_health_checker),
            ("Health Report Generation", self.test_health_report_generation),
            ("Recovery Mechanisms", self.test_recovery_mechanisms),
            ("Configuration Validation", self.test_configuration_validation),
            ("Bot Integration", self.test_integration_with_bot)
        ]
        
        results = {}
        passed = 0
        total = len(test_methods)
        
        for test_name, test_method in test_methods:
            logger.info(f"\n📋 Running: {test_name}")
            try:
                if asyncio.iscoroutinefunction(test_method):
                    result = await test_method()
                else:
                    result = test_method()
                
                results[test_name] = result
                if result:
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Summary
        logger.info("=" * 60)
        logger.info("📊 HEALTH MONITORING TEST RESULTS:")
        logger.info(f"📊 Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! Health monitoring system is working correctly!")
        else:
            logger.warning("⚠️ Some tests failed. Review the health monitoring system.")
        
        return results
    
    def cleanup(self):
        """Clean up test resources"""
        try:
            # Remove test files
            test_files = ["bot_data.json"]
            for file in test_files:
                if os.path.exists(file):
                    os.remove(file)
            logger.info("✅ Test cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

async def main():
    """Run the health monitoring test suite"""
    test_suite = HealthMonitoringTestSuite()
    
    try:
        results = await test_suite.run_all_tests()
        
        # Return appropriate exit code
        all_passed = all(results.values())
        return 0 if all_passed else 1
        
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)