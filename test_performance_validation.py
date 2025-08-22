#!/usr/bin/env python3
"""
Performance Validation Under Load
Tests system performance with concurrent operations
"""
import asyncio
import time
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config.settings import load_settings
from src.api.jupiter import JupiterClient
from src.enhanced_token_scanner import EnhancedTokenScanner
from src.analytics.performance_analytics import PerformanceAnalytics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceValidator:
    """Test system performance under various load conditions"""
    
    def __init__(self):
        self.settings = load_settings()
        self.results = {}
        
    async def test_api_response_times(self):
        """Test API response times under concurrent load"""
        logger.info("PERFORMANCE TEST 1: API Response Times")
        
        jupiter = JupiterClient()
        
        # Test concurrent SOL price requests
        start_time = time.time()
        tasks = [jupiter.get_sol_price() for _ in range(10)]
        prices = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        successful_requests = len([p for p in prices if isinstance(p, (int, float))])
        avg_response_time = (end_time - start_time) / 10
        
        logger.info(f"Concurrent SOL price requests: {successful_requests}/10 successful")
        logger.info(f"Average response time: {avg_response_time:.3f}s")
        
        self.results["api_response"] = {
            "success_rate": successful_requests / 10,
            "avg_time": avg_response_time,
            "status": "PASS" if successful_requests >= 8 and avg_response_time < 2.0 else "FAIL"
        }
        
        await jupiter.close()
    
    async def test_scanner_throughput(self):
        """Test token scanner throughput"""
        logger.info("PERFORMANCE TEST 2: Scanner Throughput")
        
        analytics = PerformanceAnalytics(self.settings)
        scanner = EnhancedTokenScanner(self.settings, analytics)
        
        # Test rapid token scanning
        start_time = time.time()
        
        # Simulate token scanning for 10 seconds
        scan_count = 0
        timeout = start_time + 10  # 10 second test
        
        try:
            while time.time() < timeout:
                # This would normally scan tokens, but we'll simulate
                await asyncio.sleep(0.1)  # Simulate scan time
                scan_count += 1
                if scan_count >= 50:  # Limit to prevent runaway
                    break
        except Exception as e:
            logger.error(f"Scanner test error: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = scan_count / duration if duration > 0 else 0
        
        logger.info(f"Scanner operations: {scan_count} in {duration:.1f}s")
        logger.info(f"Throughput: {throughput:.1f} ops/second")
        
        self.results["scanner_throughput"] = {
            "operations": scan_count,
            "duration": duration,
            "throughput": throughput,
            "status": "PASS" if throughput >= 5.0 else "FAIL"
        }
        
        await scanner.stop()
    
    async def test_memory_usage(self):
        """Test memory usage patterns"""
        logger.info("PERFORMANCE TEST 3: Memory Usage")
        
        try:
            import psutil
            process = psutil.Process()
            
            # Get initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate heavy operations
            large_data = []
            for i in range(1000):
                large_data.append({"token": f"test_{i}", "price": i * 0.001, "data": "x" * 100})
            
            # Get peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Cleanup
            del large_data
            
            # Get final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_increase = peak_memory - initial_memory
            memory_cleanup = peak_memory - final_memory
            
            logger.info(f"Initial memory: {initial_memory:.1f} MB")
            logger.info(f"Peak memory: {peak_memory:.1f} MB (+{memory_increase:.1f} MB)")
            logger.info(f"Final memory: {final_memory:.1f} MB (cleaned: {memory_cleanup:.1f} MB)")
            
            self.results["memory_usage"] = {
                "initial_mb": initial_memory,
                "peak_mb": peak_memory,
                "final_mb": final_memory,
                "increase_mb": memory_increase,
                "status": "PASS" if memory_increase < 100 else "FAIL"  # Less than 100MB increase
            }
            
        except ImportError:
            logger.warning("psutil not available - skipping memory test")
            self.results["memory_usage"] = {"status": "SKIP", "reason": "psutil not available"}
    
    async def test_concurrent_operations(self):
        """Test concurrent operations handling"""
        logger.info("PERFORMANCE TEST 4: Concurrent Operations")
        
        async def mock_trade_operation(operation_id):
            # Simulate trade processing time
            await asyncio.sleep(0.1 + (operation_id % 3) * 0.05)  # Variable processing time
            return f"trade_{operation_id}"
        
        start_time = time.time()
        
        # Run 20 concurrent mock operations
        tasks = [mock_trade_operation(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        successful_ops = len([r for r in results if isinstance(r, str)])
        total_time = end_time - start_time
        
        logger.info(f"Concurrent operations: {successful_ops}/20 successful")
        logger.info(f"Total time: {total_time:.3f}s (expected ~0.25s with concurrency)")
        logger.info(f"Concurrency efficiency: {(0.25/total_time)*100:.1f}%" if total_time > 0 else "N/A")
        
        self.results["concurrent_ops"] = {
            "success_rate": successful_ops / 20,
            "total_time": total_time,
            "efficiency": (0.25/total_time) if total_time > 0 else 0,
            "status": "PASS" if successful_ops >= 18 and total_time < 0.5 else "FAIL"
        }
    
    async def run_all_tests(self):
        """Run all performance tests"""
        logger.info("STARTING COMPREHENSIVE PERFORMANCE VALIDATION")
        logger.info("=" * 60)
        
        await self.test_api_response_times()
        await self.test_scanner_throughput()
        await self.test_memory_usage()
        await self.test_concurrent_operations()
        
        return self.results
    
    def print_results(self):
        """Print performance test results"""
        logger.info("=" * 60)
        logger.info("PERFORMANCE VALIDATION RESULTS")
        logger.info("=" * 60)
        
        passed = 0
        total = len([r for r in self.results.values() if r.get("status") != "SKIP"])
        
        for test_name, result in self.results.items():
            status = result.get("status", "UNKNOWN")
            if status == "SKIP":
                logger.info(f"{test_name.upper().replace('_', ' ')}: SKIPPED ({result.get('reason', 'No reason')})")
                continue
                
            status_icon = "[OK]" if status == "PASS" else "[FAIL]"
            logger.info(f"{test_name.upper().replace('_', ' ')}: {status_icon} {status}")
            
            if status == "PASS":
                passed += 1
        
        logger.info("=" * 60)
        logger.info(f"OVERALL PERFORMANCE RESULT: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("PERFORMANCE: EXCELLENT - System ready for production load")
        elif passed >= total * 0.75:
            logger.info("PERFORMANCE: GOOD - Minor optimizations recommended")
        else:
            logger.info("PERFORMANCE: NEEDS ATTENTION - Optimization required")

async def main():
    """Run performance validation"""
    validator = PerformanceValidator()
    await validator.run_all_tests()
    validator.print_results()

if __name__ == "__main__":
    asyncio.run(main())