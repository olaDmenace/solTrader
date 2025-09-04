"""
Time Synchronization Verification - Ensure All Modules Use Synced Time Sources
Critical for trading systems where timing accuracy affects execution
"""

import asyncio
import sys
import os
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import threading
from collections import defaultdict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.db_manager import DatabaseManager
from config.settings import Settings

@dataclass
class TimeSource:
    name: str
    timestamp: datetime
    source_type: str
    precision_ms: float
    offset_from_reference: float = 0.0

@dataclass
class TimeSyncResults:
    start_time: datetime
    end_time: Optional[datetime]
    test_duration_seconds: float
    reference_time_source: str
    time_sources_tested: int
    max_time_drift: float
    avg_time_drift: float
    sync_violations_count: int
    critical_violations_count: int
    time_sources: List[TimeSource] = field(default_factory=list)
    sync_status: str = "UNKNOWN"

class TimeSynchronizationVerifier:
    def __init__(self):
        self.settings = Settings(
            ALCHEMY_RPC_URL="test",
            WALLET_ADDRESS="test"
        )
        self.db_manager = DatabaseManager(self.settings)
        self.results = TimeSyncResults(
            start_time=datetime.now(),
            end_time=None,
            test_duration_seconds=0.0,
            reference_time_source="SYSTEM_UTC",
            time_sources_tested=0,
            max_time_drift=0.0,
            avg_time_drift=0.0,
            sync_violations_count=0,
            critical_violations_count=0
        )
        self.reference_time = None
        
    async def initialize_system(self):
        """Initialize system for time sync verification"""
        try:
            print("[INIT] Initializing Time Synchronization Verification...")
            
            await self.db_manager.initialize()
            
            # Establish reference time (high precision UTC)
            self.reference_time = datetime.now(timezone.utc)
            
            print(f"[REFERENCE] System UTC time established: {self.reference_time}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] System initialization failed: {str(e)}")
            return False
    
    async def test_system_time_sources(self):
        """Test various system time sources for synchronization"""
        print("[TEST] Testing system time sources...")
        
        # Test 1: Python datetime sources
        await self._test_python_time_sources()
        
        # Test 2: Database timestamp consistency
        await self._test_database_time_consistency()
        
        # Test 3: Multi-threaded time consistency
        await self._test_multithreaded_time_consistency()
        
        # Test 4: Trading timestamp precision
        await self._test_trading_timestamp_precision()
        
        # Test 5: Cross-module time synchronization
        await self._test_cross_module_time_sync()
    
    async def _test_python_time_sources(self):
        """Test various Python time sources"""
        print("[TIME_SOURCES] Testing Python time sources...")
        
        # Reference time (high precision)
        ref_time = time.time_ns() / 1_000_000_000  # Convert to seconds with ns precision
        
        # Test different time sources
        time_sources = [
            ("datetime.now()", lambda: datetime.now().timestamp()),
            ("datetime.utcnow()", lambda: datetime.utcnow().timestamp()),
            ("datetime.now(UTC)", lambda: datetime.now(timezone.utc).timestamp()),
            ("time.time()", lambda: time.time()),
            ("time.perf_counter()", lambda: time.perf_counter() + (ref_time - time.perf_counter())),
        ]
        
        for source_name, time_func in time_sources:
            try:
                # Take multiple samples to check precision
                samples = []
                for _ in range(10):
                    sample_time = time_func()
                    samples.append(sample_time)
                    await asyncio.sleep(0.001)  # 1ms between samples
                
                # Calculate precision and offset
                avg_time = sum(samples) / len(samples)
                precision = max(samples) - min(samples)
                offset = abs(avg_time - ref_time)
                
                time_source = TimeSource(
                    name=source_name,
                    timestamp=datetime.fromtimestamp(avg_time, timezone.utc),
                    source_type="PYTHON_BUILTIN",
                    precision_ms=precision * 1000,
                    offset_from_reference=offset * 1000  # Convert to milliseconds
                )
                
                self.results.time_sources.append(time_source)
                self.results.time_sources_tested += 1
                
                print(f"  {source_name}: Offset {offset*1000:.2f}ms, Precision {precision*1000:.2f}ms")
                
            except Exception as e:
                print(f"  {source_name}: ERROR - {str(e)}")
    
    async def _test_database_time_consistency(self):
        """Test database timestamp consistency"""
        print("[DATABASE] Testing database timestamp consistency...")
        
        try:
            import aiosqlite
            
            # Create test table
            async with aiosqlite.connect(self.db_manager.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS time_sync_test (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        python_timestamp REAL,
                        sqlite_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                        test_id INTEGER
                    )
                """)
                await db.commit()
                
                # Insert multiple records with Python timestamps
                python_times = []
                sqlite_times = []
                
                for i in range(10):
                    python_time = time.time()
                    python_times.append(python_time)
                    
                    await db.execute("""
                        INSERT INTO time_sync_test (python_timestamp, test_id)
                        VALUES (?, ?)
                    """, (python_time, i))
                    await db.commit()
                    
                    # Retrieve the SQLite timestamp
                    async with db.execute("""
                        SELECT sqlite_timestamp FROM time_sync_test WHERE test_id = ?
                    """, (i,)) as cursor:
                        row = await cursor.fetchone()
                        if row:
                            # Convert SQLite timestamp to Python timestamp
                            sqlite_dt = datetime.fromisoformat(row[0].replace(' ', 'T'))
                            sqlite_times.append(sqlite_dt.timestamp())
                    
                    await asyncio.sleep(0.01)  # 10ms between operations
                
                # Analyze time differences
                if python_times and sqlite_times:
                    time_diffs = [abs(p - s) for p, s in zip(python_times, sqlite_times)]
                    max_diff = max(time_diffs) * 1000  # Convert to milliseconds
                    avg_diff = sum(time_diffs) / len(time_diffs) * 1000
                    
                    db_time_source = TimeSource(
                        name="SQLite_CURRENT_TIMESTAMP",
                        timestamp=datetime.fromtimestamp(sqlite_times[-1], timezone.utc),
                        source_type="DATABASE",
                        precision_ms=max_diff,
                        offset_from_reference=avg_diff
                    )
                    
                    self.results.time_sources.append(db_time_source)
                    self.results.time_sources_tested += 1
                    
                    print(f"  Database time sync: Max diff {max_diff:.2f}ms, Avg diff {avg_diff:.2f}ms")
                
        except Exception as e:
            print(f"[ERROR] Database time test failed: {str(e)}")
    
    async def _test_multithreaded_time_consistency(self):
        """Test time consistency across threads"""
        print("[THREADS] Testing multithreaded time consistency...")
        
        try:
            thread_times = {}
            sync_event = threading.Event()
            
            def thread_time_capture(thread_id):
                # Wait for synchronization signal
                sync_event.wait()
                thread_times[thread_id] = time.time()
            
            # Create multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=thread_time_capture, args=(i,))
                thread.start()
                threads.append(thread)
            
            # Signal all threads to capture time simultaneously
            await asyncio.sleep(0.1)  # Brief delay
            sync_event.set()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=1.0)
            
            # Analyze thread time consistency
            if thread_times:
                times = list(thread_times.values())
                time_spread = (max(times) - min(times)) * 1000  # Convert to milliseconds
                
                thread_time_source = TimeSource(
                    name="Multithreaded_Sync",
                    timestamp=datetime.fromtimestamp(sum(times) / len(times), timezone.utc),
                    source_type="THREADING",
                    precision_ms=time_spread,
                    offset_from_reference=0.0
                )
                
                self.results.time_sources.append(thread_time_source)
                self.results.time_sources_tested += 1
                
                print(f"  Thread time spread: {time_spread:.2f}ms across {len(times)} threads")
            
        except Exception as e:
            print(f"[ERROR] Multithreaded time test failed: {str(e)}")
    
    async def _test_trading_timestamp_precision(self):
        """Test timestamp precision for trading operations"""
        print("[TRADING] Testing trading timestamp precision...")
        
        try:
            # Simulate rapid trading operations
            trade_timestamps = []
            operation_times = []
            
            for i in range(100):
                start_time = time.perf_counter_ns()
                
                # Simulate timestamp capture during trade
                trade_time = datetime.now(timezone.utc)
                trade_timestamps.append(trade_time.timestamp())
                
                end_time = time.perf_counter_ns()
                operation_times.append((end_time - start_time) / 1_000_000)  # Convert to milliseconds
                
                if i < 99:  # Don't sleep after last iteration
                    await asyncio.sleep(0.001)  # 1ms between operations
            
            # Analyze trading timestamp precision
            if trade_timestamps:
                time_diffs = []
                for i in range(1, len(trade_timestamps)):
                    diff = (trade_timestamps[i] - trade_timestamps[i-1]) * 1000
                    time_diffs.append(diff)
                
                min_interval = min(time_diffs)
                max_interval = max(time_diffs)
                avg_interval = sum(time_diffs) / len(time_diffs)
                avg_operation_time = sum(operation_times) / len(operation_times)
                
                trading_time_source = TimeSource(
                    name="Trading_Timestamps",
                    timestamp=datetime.fromtimestamp(trade_timestamps[-1], timezone.utc),
                    source_type="TRADING_OPERATION",
                    precision_ms=avg_operation_time,
                    offset_from_reference=0.0
                )
                
                self.results.time_sources.append(trading_time_source)
                self.results.time_sources_tested += 1
                
                print(f"  Trading intervals: {min_interval:.2f}-{max_interval:.2f}ms")
                print(f"  Average operation: {avg_operation_time:.3f}ms")
            
        except Exception as e:
            print(f"[ERROR] Trading timestamp test failed: {str(e)}")
    
    async def _test_cross_module_time_sync(self):
        """Test time synchronization across different modules"""
        print("[MODULES] Testing cross-module time synchronization...")
        
        try:
            # Simulate different module timestamp styles
            module_times = {
                "risk_engine": datetime.now(timezone.utc),
                "portfolio_manager": datetime.utcnow(),
                "paper_trading": datetime.now(),
                "database_logger": datetime.now(timezone.utc),
                "market_data": time.time()
            }
            
            # Convert all to timestamps for comparison
            timestamps = {}
            for module, time_val in module_times.items():
                if isinstance(time_val, datetime):
                    if time_val.tzinfo is None:
                        # Assume UTC for naive datetime
                        time_val = time_val.replace(tzinfo=timezone.utc)
                    timestamps[module] = time_val.timestamp()
                else:
                    timestamps[module] = time_val
            
            # Calculate time synchronization across modules
            reference_time = timestamps["risk_engine"]  # Use risk engine as reference
            
            for module, timestamp in timestamps.items():
                offset = abs(timestamp - reference_time) * 1000  # Convert to milliseconds
                
                module_time_source = TimeSource(
                    name=f"Module_{module}",
                    timestamp=datetime.fromtimestamp(timestamp, timezone.utc),
                    source_type="MODULE_TIMESTAMP",
                    precision_ms=1.0,  # Assume 1ms precision
                    offset_from_reference=offset
                )
                
                self.results.time_sources.append(module_time_source)
                self.results.time_sources_tested += 1
                
                print(f"  {module}: Offset {offset:.2f}ms from reference")
            
        except Exception as e:
            print(f"[ERROR] Cross-module time test failed: {str(e)}")
    
    def analyze_time_synchronization(self):
        """Analyze collected time synchronization data"""
        print("[ANALYSIS] Analyzing time synchronization results...")
        
        if not self.results.time_sources:
            print("[WARNING] No time sources to analyze")
            return
        
        # Calculate overall statistics
        offsets = [ts.offset_from_reference for ts in self.results.time_sources if ts.offset_from_reference > 0]
        
        if offsets:
            self.results.max_time_drift = max(offsets)
            self.results.avg_time_drift = sum(offsets) / len(offsets)
            
            # Count synchronization violations
            # Minor violation: > 10ms offset
            # Critical violation: > 100ms offset
            for offset in offsets:
                if offset > 10.0:  # 10ms
                    self.results.sync_violations_count += 1
                if offset > 100.0:  # 100ms
                    self.results.critical_violations_count += 1
        
        print(f"[ANALYSIS] Max time drift: {self.results.max_time_drift:.2f}ms")
        print(f"[ANALYSIS] Avg time drift: {self.results.avg_time_drift:.2f}ms")
        print(f"[ANALYSIS] Sync violations (>10ms): {self.results.sync_violations_count}")
        print(f"[ANALYSIS] Critical violations (>100ms): {self.results.critical_violations_count}")
    
    async def generate_time_sync_report(self):
        """Generate comprehensive time synchronization report"""
        self.results.end_time = datetime.now()
        self.results.test_duration_seconds = (
            self.results.end_time - self.results.start_time
        ).total_seconds()
        
        self.analyze_time_synchronization()
        
        print("\n" + "="*80)
        print("TIME SYNCHRONIZATION VERIFICATION REPORT")
        print("="*80)
        
        print(f"Test Duration:          {self.results.test_duration_seconds:.1f} seconds")
        print(f"Time Sources Tested:    {self.results.time_sources_tested}")
        print(f"Reference Time Source:  {self.results.reference_time_source}")
        
        print(f"\nSYNCHRONIZATION METRICS:")
        print(f"Maximum Time Drift:     {self.results.max_time_drift:.2f}ms")
        print(f"Average Time Drift:     {self.results.avg_time_drift:.2f}ms")
        print(f"Minor Violations:       {self.results.sync_violations_count} (>10ms)")
        print(f"Critical Violations:    {self.results.critical_violations_count} (>100ms)")
        
        print(f"\nTIME SOURCE BREAKDOWN:")
        for ts in self.results.time_sources[:10]:  # Show first 10 sources
            print(f"  {ts.name:<20} | {ts.source_type:<15} | {ts.offset_from_reference:>7.2f}ms | {ts.precision_ms:>7.2f}ms")
        
        # Determine sync status
        if self.results.critical_violations_count > 0:
            self.results.sync_status = "CRITICAL_DESYNC"
            verdict = "CRITICAL - Severe Time Synchronization Issues"
            status = "FAIL"
        elif self.results.sync_violations_count > self.results.time_sources_tested * 0.3:
            self.results.sync_status = "POOR_SYNC"
            verdict = "POOR - Multiple Time Synchronization Issues"
            status = "WARN"
        elif self.results.max_time_drift > 50.0:
            self.results.sync_status = "MODERATE_DRIFT"
            verdict = "ACCEPTABLE - Some Time Drift Within Limits"
            status = "PASS"
        elif self.results.max_time_drift > 10.0:
            self.results.sync_status = "MINOR_DRIFT"
            verdict = "GOOD - Minor Time Drift Detected"
            status = "PASS"
        else:
            self.results.sync_status = "EXCELLENT_SYNC"
            verdict = "EXCELLENT - All Time Sources Properly Synchronized"
            status = "PASS"
        
        print(f"\nSYNCHRONIZATION STATUS: {self.results.sync_status}")
        print(f"VERDICT: [{status}] {verdict}")
        print("="*80)
        
        return {
            'sync_status': self.results.sync_status,
            'max_time_drift': self.results.max_time_drift,
            'avg_time_drift': self.results.avg_time_drift,
            'critical_violations': self.results.critical_violations_count,
            'verdict': verdict,
            'status': status
        }

async def main():
    """Main time sync verification execution"""
    verifier = TimeSynchronizationVerifier()
    
    try:
        # Initialize system
        if not await verifier.initialize_system():
            print("[ABORT] Failed to initialize system")
            return
        
        print("[INFO] Starting comprehensive time synchronization verification...")
        print("This will test:")
        print("- Python datetime sources")
        print("- Database timestamp consistency")
        print("- Multithreaded time synchronization")
        print("- Trading timestamp precision")
        print("- Cross-module time coordination")
        
        # Run time synchronization tests
        await verifier.test_system_time_sources()
        
        # Generate final report
        result = await verifier.generate_time_sync_report()
        
        print(f"\n[COMPLETE] Time synchronization verification finished!")
        print(f"Final Assessment: {result['status']} - {result['verdict']}")
        
        return result
        
    except Exception as e:
        print(f"[CRITICAL] Time sync verification failure: {str(e)}")
        return None
    finally:
        # Cleanup
        if hasattr(verifier, 'db_manager'):
            await verifier.db_manager.close()

if __name__ == "__main__":
    result = asyncio.run(main())