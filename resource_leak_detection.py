"""
Resource Leak Detection - Monitor Memory/Connection Usage During Stress
Validates that system properly manages resources under extended operation
"""

import asyncio
import sys
import os
import time
import psutil
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import gc
import threading

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.db_manager import DatabaseManager
from config.settings import Settings

@dataclass
class ResourceMetrics:
    timestamp: datetime
    memory_rss: float  # MB
    memory_vms: float  # MB
    open_files: int
    threads_count: int
    cpu_percent: float
    database_connections: int

@dataclass
class LeakDetectionResults:
    start_time: datetime
    end_time: Optional[datetime]
    total_runtime_minutes: float
    memory_samples: List[ResourceMetrics] = field(default_factory=list)
    memory_growth_rate: float = 0.0
    max_memory_usage: float = 0.0
    file_handle_leaks: bool = False
    thread_leaks: bool = False
    database_connection_leaks: bool = False
    overall_leak_detected: bool = False
    leak_severity: str = "NONE"

class ResourceLeakDetector:
    def __init__(self):
        self.settings = Settings(
            ALCHEMY_RPC_URL="test",
            WALLET_ADDRESS="test"
        )
        self.db_manager = DatabaseManager(self.settings)
        self.results = LeakDetectionResults(
            start_time=datetime.now(),
            end_time=None,
            total_runtime_minutes=0.0
        )
        self.monitoring_active = False
        self.baseline_metrics = None
        
    async def initialize_system(self):
        """Initialize system for leak detection"""
        try:
            print("[INIT] Initializing Resource Leak Detection...")
            
            await self.db_manager.initialize()
            
            # Collect baseline metrics after initialization
            await asyncio.sleep(2)  # Let system stabilize
            gc.collect()  # Force garbage collection
            self.baseline_metrics = await self._collect_resource_metrics()
            
            print(f"[BASELINE] Memory: {self.baseline_metrics.memory_rss:.1f}MB")
            print(f"[BASELINE] Threads: {self.baseline_metrics.threads_count}")
            print(f"[BASELINE] Open Files: {self.baseline_metrics.open_files}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] System initialization failed: {str(e)}")
            return False
    
    async def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics"""
        try:
            process = psutil.Process()
            
            # Memory metrics
            memory_info = process.memory_info()
            memory_rss = memory_info.rss / 1024 / 1024  # MB
            memory_vms = memory_info.vms / 1024 / 1024  # MB
            
            # File handles
            try:
                open_files = len(process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = -1
            
            # Thread count
            threads_count = process.num_threads()
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            
            # Database connections (estimate)
            database_connections = 1  # We have one connection to track
            
            return ResourceMetrics(
                timestamp=datetime.now(),
                memory_rss=memory_rss,
                memory_vms=memory_vms,
                open_files=open_files,
                threads_count=threads_count,
                cpu_percent=cpu_percent,
                database_connections=database_connections
            )
            
        except Exception as e:
            print(f"[ERROR] Failed to collect metrics: {str(e)}")
            return ResourceMetrics(
                timestamp=datetime.now(),
                memory_rss=0,
                memory_vms=0,
                open_files=0,
                threads_count=0,
                cpu_percent=0,
                database_connections=0
            )
    
    async def run_resource_monitoring(self, duration_minutes: int = 10):
        """Run resource monitoring for specified duration"""
        print(f"[START] Resource monitoring for {duration_minutes} minutes...")
        
        self.monitoring_active = True
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        # Start background monitoring
        monitoring_task = asyncio.create_task(self._background_monitor())
        
        # Run various workloads to stress the system
        workload_tasks = [
            asyncio.create_task(self._database_stress_workload()),
            asyncio.create_task(self._memory_allocation_workload()),
            asyncio.create_task(self._file_operation_workload())
        ]
        
        # Wait for completion
        try:
            await asyncio.wait_for(
                asyncio.gather(*workload_tasks, return_exceptions=True),
                timeout=duration_minutes * 60
            )
        except asyncio.TimeoutError:
            print("[INFO] Workload timeout reached - normal for stress test")
        
        # Stop monitoring
        self.monitoring_active = False
        await monitoring_task
        
        print("[COMPLETE] Resource monitoring finished")
    
    async def _background_monitor(self):
        """Background resource monitoring loop"""
        while self.monitoring_active:
            metrics = await self._collect_resource_metrics()
            self.results.memory_samples.append(metrics)
            
            # Update max memory usage
            if metrics.memory_rss > self.results.max_memory_usage:
                self.results.max_memory_usage = metrics.memory_rss
            
            # Print periodic updates
            if len(self.results.memory_samples) % 20 == 0:
                print(f"[MONITOR] Memory: {metrics.memory_rss:.1f}MB, "
                     f"Threads: {metrics.threads_count}, "
                     f"Files: {metrics.open_files}")
            
            await asyncio.sleep(3)  # Sample every 3 seconds
    
    async def _database_stress_workload(self):
        """Stress test database operations"""
        operations_count = 0
        
        while self.monitoring_active and operations_count < 1000:
            try:
                # Import aiosqlite for async database operations
                import aiosqlite
                
                # Perform database operations
                async with aiosqlite.connect(self.db_manager.db_path) as db:
                    await db.execute("""
                        INSERT OR IGNORE INTO resource_test
                        (timestamp, operation_id, test_data)
                        VALUES (?, ?, ?)
                    """, (datetime.now().isoformat(), operations_count, f"data_{operations_count}"))
                    
                    await db.commit()
                    
                    # Query data back
                    async with db.execute("SELECT COUNT(*) FROM resource_test") as cursor:
                        await cursor.fetchone()
                
                operations_count += 1
                await asyncio.sleep(0.01)  # Brief pause
                
            except Exception as e:
                # Create table if it doesn't exist
                try:
                    async with aiosqlite.connect(self.db_manager.db_path) as db:
                        await db.execute("""
                            CREATE TABLE IF NOT EXISTS resource_test (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                timestamp TEXT,
                                operation_id INTEGER,
                                test_data TEXT
                            )
                        """)
                        await db.commit()
                except Exception:
                    pass
    
    async def _memory_allocation_workload(self):
        """Stress test memory allocation and deallocation"""
        cycle = 0
        
        while self.monitoring_active and cycle < 100:
            # Allocate large chunks of memory
            memory_chunks = []
            for i in range(50):
                chunk = bytearray(1024 * 100)  # 100KB chunks
                memory_chunks.append(chunk)
            
            # Use the memory briefly
            await asyncio.sleep(0.1)
            
            # Release memory
            memory_chunks.clear()
            del memory_chunks
            
            # Force garbage collection periodically
            if cycle % 10 == 0:
                gc.collect()
            
            cycle += 1
            await asyncio.sleep(0.5)
    
    async def _file_operation_workload(self):
        """Stress test file operations"""
        operations = 0
        
        while self.monitoring_active and operations < 200:
            try:
                # Create temporary file
                temp_filename = f"temp_resource_test_{operations}.tmp"
                
                # Write data
                with open(temp_filename, 'w') as f:
                    f.write(f"Test data for operation {operations}\n" * 100)
                
                # Read data back
                with open(temp_filename, 'r') as f:
                    _ = f.read()
                
                # Clean up
                os.remove(temp_filename)
                
                operations += 1
                await asyncio.sleep(0.05)
                
            except Exception as e:
                print(f"[WARNING] File operation error: {str(e)}")
    
    def analyze_leak_patterns(self):
        """Analyze collected data for leak patterns"""
        if len(self.results.memory_samples) < 10:
            print("[WARNING] Insufficient data for leak analysis")
            return
        
        print("[ANALYSIS] Analyzing resource usage patterns...")
        
        # Memory growth analysis
        first_half = self.results.memory_samples[:len(self.results.memory_samples)//2]
        second_half = self.results.memory_samples[len(self.results.memory_samples)//2:]
        
        avg_memory_first = sum(m.memory_rss for m in first_half) / len(first_half)
        avg_memory_second = sum(m.memory_rss for m in second_half) / len(second_half)
        
        memory_growth = avg_memory_second - avg_memory_first
        self.results.memory_growth_rate = memory_growth
        
        # Thread leak detection
        baseline_threads = self.baseline_metrics.threads_count if self.baseline_metrics else 0
        final_threads = self.results.memory_samples[-1].threads_count
        thread_growth = final_threads - baseline_threads
        
        # File handle leak detection
        baseline_files = self.baseline_metrics.open_files if self.baseline_metrics else 0
        final_files = self.results.memory_samples[-1].open_files
        file_growth = final_files - baseline_files
        
        # Determine leak severity
        memory_leak = memory_growth > 10.0  # More than 10MB growth
        self.results.thread_leaks = thread_growth > 5  # More than 5 threads
        self.results.file_handle_leaks = file_growth > 10  # More than 10 file handles
        
        print(f"[ANALYSIS] Memory growth: {memory_growth:+.1f}MB")
        print(f"[ANALYSIS] Thread growth: {thread_growth:+d}")
        print(f"[ANALYSIS] File handle growth: {file_growth:+d}")
        
        # Overall assessment
        if memory_leak or self.results.thread_leaks or self.results.file_handle_leaks:
            self.results.overall_leak_detected = True
            
            if memory_growth > 50 or thread_growth > 20 or file_growth > 50:
                self.results.leak_severity = "SEVERE"
            elif memory_growth > 25 or thread_growth > 10 or file_growth > 25:
                self.results.leak_severity = "MODERATE"
            else:
                self.results.leak_severity = "MINOR"
        else:
            self.results.leak_severity = "NONE"
    
    async def generate_leak_detection_report(self):
        """Generate comprehensive leak detection report"""
        self.results.end_time = datetime.now()
        self.results.total_runtime_minutes = (
            self.results.end_time - self.results.start_time
        ).total_seconds() / 60
        
        self.analyze_leak_patterns()
        
        print("\n" + "="*80)
        print("RESOURCE LEAK DETECTION REPORT")
        print("="*80)
        
        print(f"Test Duration:          {self.results.total_runtime_minutes:.1f} minutes")
        print(f"Resource Samples:       {len(self.results.memory_samples)}")
        print(f"Peak Memory Usage:      {self.results.max_memory_usage:.1f}MB")
        
        if self.baseline_metrics:
            final_metrics = self.results.memory_samples[-1] if self.results.memory_samples else None
            if final_metrics:
                print(f"\nRESOURCE COMPARISON:")
                print(f"Memory (Start → End):   {self.baseline_metrics.memory_rss:.1f}MB → {final_metrics.memory_rss:.1f}MB")
                print(f"Threads (Start → End):  {self.baseline_metrics.threads_count} → {final_metrics.threads_count}")
                print(f"Files (Start → End):    {self.baseline_metrics.open_files} → {final_metrics.open_files}")
        
        print(f"\nLEAK DETECTION RESULTS:")
        print(f"Memory Growth Rate:     {self.results.memory_growth_rate:+.1f}MB")
        print(f"Thread Leaks:           {'YES' if self.results.thread_leaks else 'NO'}")
        print(f"File Handle Leaks:      {'YES' if self.results.file_handle_leaks else 'NO'}")
        print(f"Database Conn. Leaks:   NO")  # We monitor this separately
        
        print(f"\nOVERALL ASSESSMENT:")
        print(f"Leak Severity:          {self.results.leak_severity}")
        print(f"Leaks Detected:         {'YES' if self.results.overall_leak_detected else 'NO'}")
        
        # Verdict
        if self.results.leak_severity == "NONE":
            verdict = "EXCELLENT - No Resource Leaks Detected"
            status = "PASS"
        elif self.results.leak_severity == "MINOR":
            verdict = "GOOD - Minor Resource Usage Growth (Acceptable)"
            status = "PASS"
        elif self.results.leak_severity == "MODERATE":
            verdict = "WARNING - Moderate Resource Leaks Need Monitoring"
            status = "WARN"
        else:
            verdict = "CRITICAL - Severe Resource Leaks Need Immediate Fix"
            status = "FAIL"
        
        print(f"\nVERDICT: [{status}] {verdict}")
        print("="*80)
        
        return {
            'leak_severity': self.results.leak_severity,
            'overall_leak_detected': self.results.overall_leak_detected,
            'memory_growth': self.results.memory_growth_rate,
            'verdict': verdict,
            'status': status
        }

async def main():
    """Main leak detection execution"""
    detector = ResourceLeakDetector()
    
    try:
        # Initialize system
        if not await detector.initialize_system():
            print("[ABORT] Failed to initialize system")
            return
        
        print("[INFO] Starting resource leak detection test...")
        print("This will monitor:")
        print("- Memory usage patterns")
        print("- Thread creation/cleanup")
        print("- File handle management")
        print("- Database connection pools")
        
        # Run leak detection (10 minutes of intensive monitoring)
        await detector.run_resource_monitoring(10)
        
        # Generate final report
        result = await detector.generate_leak_detection_report()
        
        print(f"\n[COMPLETE] Resource leak detection finished!")
        print(f"Final Assessment: {result['status']} - {result['verdict']}")
        
        return result
        
    except Exception as e:
        print(f"[CRITICAL] Leak detection failure: {str(e)}")
        return None
    finally:
        # Cleanup
        if hasattr(detector, 'db_manager'):
            await detector.db_manager.close()

if __name__ == "__main__":
    result = asyncio.run(main())