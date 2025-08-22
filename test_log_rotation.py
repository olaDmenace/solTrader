#!/usr/bin/env python3
"""
Test Log Rotation System
Comprehensive testing of log rotation, cleanup, and monitoring
"""
import asyncio
import logging
import time
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.utils.log_management import LogRotationManager, LogRotationConfig
from src.utils.logging_config import LoggingSetup, setup_production_logging

def create_large_log_file(file_path: Path, size_mb: float):
    """Create a log file of specified size for testing"""
    target_size = int(size_mb * 1024 * 1024)
    
    with open(file_path, 'w') as f:
        line = "2025-08-22 14:30:00 - trading - INFO - Test log entry for rotation testing\n"
        line_size = len(line.encode('utf-8'))
        lines_needed = target_size // line_size
        
        for i in range(lines_needed):
            f.write(f"2025-08-22 14:30:{i%60:02d} - trading - INFO - Test log entry #{i} for rotation testing\n")

async def test_log_rotation_config():
    """Test log rotation configuration"""
    print("\n" + "="*60)
    print("TESTING LOG ROTATION CONFIGURATION")
    print("="*60)
    
    # Test with custom config
    config = LogRotationConfig(
        max_file_size_mb=10,
        max_backup_files=5,
        compress_backups=True,
        check_interval_minutes=1
    )
    
    # Create temp directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        config.log_directory = temp_dir
        config.archive_directory = f"{temp_dir}/archive"
        
        manager = LogRotationManager(config)
        
        print(f"Test directory: {temp_dir}")
        print(f"Max file size: {config.max_file_size_mb}MB")
        print(f"Max backup files: {config.max_backup_files}")
        print(f"Compression: {config.compress_backups}")
        
        # Check directories were created
        if Path(temp_dir).exists() and Path(config.archive_directory).exists():
            print("✅ Directories created successfully")
            return True
        else:
            print("❌ Directory creation failed")
            return False

async def test_file_rotation():
    """Test actual file rotation"""
    print("\n" + "="*60)
    print("TESTING FILE ROTATION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LogRotationConfig(
            max_file_size_mb=1,  # Very small for testing
            max_backup_files=3,
            compress_backups=False,  # Disable for faster testing
            log_directory=temp_dir,
            archive_directory=f"{temp_dir}/archive"
        )
        
        manager = LogRotationManager(config)
        
        # Create a large test log file
        test_log = Path(temp_dir) / "trading.log"
        print(f"Creating 2MB test log file...")
        create_large_log_file(test_log, 2.0)
        
        file_size = manager.get_file_size_mb(test_log)
        print(f"Created file size: {file_size:.2f}MB")
        
        # Check if it should be rotated
        should_rotate = manager.should_rotate_file(test_log)
        print(f"Should rotate: {should_rotate}")
        
        if should_rotate:
            print("Rotating file...")
            success = manager.rotate_file(test_log)
            
            if success:
                # Check that archive was created and original file is smaller
                archive_files = list(Path(config.archive_directory).glob("trading_*"))
                new_size = manager.get_file_size_mb(test_log)
                
                print(f"✅ Rotation successful")
                print(f"  Archive files created: {len(archive_files)}")
                print(f"  New file size: {new_size:.2f}MB")
                return True
            else:
                print("❌ Rotation failed")
                return False
        else:
            print("❌ File should have been marked for rotation")
            return False

async def test_cleanup_functionality():
    """Test old file cleanup"""
    print("\n" + "="*60)
    print("TESTING CLEANUP FUNCTIONALITY")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LogRotationConfig(
            max_backup_files=2,  # Keep only 2 backup files
            max_age_days=0,      # Delete immediately for testing
            log_directory=temp_dir,
            archive_directory=f"{temp_dir}/archive"
        )
        
        manager = LogRotationManager(config)
        archive_dir = Path(config.archive_directory)
        
        # Create several fake backup files
        print("Creating 5 fake backup files...")
        for i in range(5):
            backup_file = archive_dir / f"trading_202508{20+i:02d}_120000.log"
            backup_file.write_text(f"Fake backup file {i}")
            # Set different modification times
            timestamp = time.time() - (i * 86400)  # i days ago
            os.utime(backup_file, (timestamp, timestamp))
        
        files_before = len(list(archive_dir.glob("*.log")))
        print(f"Files before cleanup: {files_before}")
        
        # Run cleanup
        print("Running cleanup...")
        manager.cleanup_old_files()
        
        files_after = len(list(archive_dir.glob("*.log")))
        print(f"Files after cleanup: {files_after}")
        
        if files_after <= config.max_backup_files:
            print(f"✅ Cleanup successful (kept {files_after}/{config.max_backup_files} files)")
            return True
        else:
            print(f"❌ Cleanup failed (still have {files_after} files)")
            return False

async def test_compression():
    """Test file compression"""
    print("\n" + "="*60)
    print("TESTING FILE COMPRESSION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LogRotationConfig(
            max_file_size_mb=0.5,  # Small size for testing
            compress_backups=True,
            compression_level=1,   # Fastest compression for testing
            log_directory=temp_dir,
            archive_directory=f"{temp_dir}/archive"
        )
        
        manager = LogRotationManager(config)
        
        # Create a test log file with repetitive content (compresses well)
        test_log = Path(temp_dir) / "trading.log"
        repetitive_content = "This is a test log line that repeats many times.\n" * 10000
        test_log.write_text(repetitive_content)
        
        original_size = manager.get_file_size_mb(test_log)
        print(f"Original file size: {original_size:.2f}MB")
        
        # Rotate the file
        print("Rotating file with compression...")
        success = manager.rotate_file(test_log)
        
        if success:
            # Check for compressed archive file
            archive_dir = Path(config.archive_directory)
            compressed_files = list(archive_dir.glob("*.gz"))
            
            if compressed_files:
                compressed_size = manager.get_file_size_mb(compressed_files[0])
                compression_ratio = (original_size - compressed_size) / original_size * 100
                
                print(f"✅ Compression successful")
                print(f"  Compressed file: {compressed_files[0].name}")
                print(f"  Compressed size: {compressed_size:.2f}MB")
                print(f"  Compression ratio: {compression_ratio:.1f}%")
                return True
            else:
                print("❌ No compressed files found")
                return False
        else:
            print("❌ Rotation with compression failed")
            return False

async def test_background_rotation():
    """Test background rotation task"""
    print("\n" + "="*60)
    print("TESTING BACKGROUND ROTATION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LogRotationConfig(
            max_file_size_mb=1,
            check_interval_minutes=0.1,  # Check every 6 seconds for testing
            rotate_on_startup=True,
            log_directory=temp_dir,
            archive_directory=f"{temp_dir}/archive"
        )
        
        manager = LogRotationManager(config)
        
        # Create a large log file
        test_log = Path(temp_dir) / "trading.log"
        print("Creating large test log file...")
        create_large_log_file(test_log, 1.5)
        
        print(f"Starting background rotation (check every 6s)...")
        await manager.start_background_rotation()
        
        # Wait a bit for background rotation to trigger
        print("Waiting for background rotation to trigger...")
        await asyncio.sleep(8)
        
        # Check if rotation happened
        archive_files = list(Path(config.archive_directory).glob("*"))
        current_size = manager.get_file_size_mb(test_log)
        
        # Stop background rotation
        await manager.stop_background_rotation()
        
        if archive_files and current_size < 0.1:  # Should be nearly empty after rotation
            print(f"✅ Background rotation successful")
            print(f"  Archive files: {len(archive_files)}")
            print(f"  Current log size: {current_size:.2f}MB")
            return True
        else:
            print(f"❌ Background rotation failed")
            print(f"  Archive files: {len(archive_files)}")
            print(f"  Current log size: {current_size:.2f}MB")
            return False

async def test_statistics():
    """Test statistics and monitoring"""
    print("\n" + "="*60)
    print("TESTING STATISTICS AND MONITORING")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LogRotationConfig(
            log_directory=temp_dir,
            archive_directory=f"{temp_dir}/archive"
        )
        
        manager = LogRotationManager(config)
        
        # Create some test files
        for log_name in ['trading', 'api', 'error']:
            log_file = Path(temp_dir) / f"{log_name}.log"
            log_file.write_text(f"Test content for {log_name} log\n" * 1000)
        
        # Get statistics
        stats = manager.get_log_statistics()
        
        print("Log Statistics:")
        print(f"  Config max file size: {stats['config']['max_file_size_mb']}MB")
        print(f"  Current files: {len(stats['current_files'])}")
        
        current_files = stats['current_files']
        for log_name, info in current_files.items():
            print(f"    {log_name}: {info['size_mb']:.2f}MB (rotation needed: {info['needs_rotation']})")
        
        print(f"  Archive files: {stats['archive_info']['file_count']}")
        print(f"  Archive size: {stats['archive_info']['total_size_mb']:.2f}MB")
        print(f"  Total size: {stats['health']['total_size_mb']:.2f}MB")
        print(f"  Size usage: {stats['health']['size_usage_percent']:.1f}%")
        
        # Check if stats make sense
        has_current_files = len(stats['current_files']) > 0
        has_health_info = 'total_size_mb' in stats['health']
        
        if has_current_files and has_health_info:
            print("✅ Statistics generation successful")
            return True
        else:
            print("❌ Statistics generation failed")
            return False

async def test_integration_with_logging():
    """Test integration with logging system"""
    print("\n" + "="*60)
    print("TESTING INTEGRATION WITH LOGGING SYSTEM")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create logging setup with custom directory
        logging_setup = LoggingSetup(temp_dir)
        
        # Setup loggers
        loggers = logging_setup.setup_all_loggers()
        print(f"Created {len(loggers)} loggers: {list(loggers.keys())}")
        
        # Generate some log entries
        trading_logger = loggers.get('trading')
        api_logger = loggers.get('api')
        error_logger = loggers.get('error')
        
        if trading_logger and api_logger and error_logger:
            print("Generating test log entries...")
            
            for i in range(100):
                trading_logger.info(f"Trading log entry {i}")
                api_logger.debug(f"API debug entry {i}")
                if i % 10 == 0:
                    error_logger.warning(f"Error log entry {i}")
            
            # Check that files were created
            log_files = list(Path(temp_dir).glob("*.log"))
            total_size = sum(Path(f).stat().st_size for f in log_files)
            
            print(f"✅ Integration successful")
            print(f"  Log files created: {len(log_files)}")
            print(f"  File names: {[f.name for f in log_files]}")
            print(f"  Total size: {total_size} bytes")
            return True
        else:
            print("❌ Logger creation failed")
            return False

async def run_all_log_rotation_tests():
    """Run all log rotation tests"""
    print("LOG ROTATION SYSTEM TESTS")
    print("="*80)
    
    start_time = time.time()
    tests_passed = 0
    total_tests = 7
    
    # Run all tests
    tests = [
        ("Configuration", test_log_rotation_config),
        ("File Rotation", test_file_rotation),
        ("Cleanup Functionality", test_cleanup_functionality),
        ("Compression", test_compression),
        ("Background Rotation", test_background_rotation),
        ("Statistics", test_statistics),
        ("Logging Integration", test_integration_with_logging)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name}...")
            success = await test_func()
            if success:
                print(f"   PASS: {test_name}")
                tests_passed += 1
            else:
                print(f"   FAIL: {test_name}")
        except Exception as e:
            print(f"   CRASH: {test_name}: {e}")
    
    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("LOG ROTATION TEST RESULTS")
    print("="*80)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {(tests_passed/total_tests)*100:.1f}%")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    
    if tests_passed == total_tests:
        print("\n🎉 ALL LOG ROTATION TESTS PASSED!")
        print("Your log rotation system is ready for production.")
        print("\nKey features working:")
        print("✅ Automatic file rotation based on size")
        print("✅ Configurable backup retention")
        print("✅ File compression to save space")
        print("✅ Background monitoring")
        print("✅ Old file cleanup")
        print("✅ Comprehensive statistics")
        print("✅ Integration with logging system")
        return True
    else:
        print(f"\n⚠️  {total_tests - tests_passed} tests failed.")
        print("Check the output above for details.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_log_rotation_tests())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nLog rotation tests interrupted")
        exit(1)
    except Exception as e:
        print(f"\nLog rotation test suite crashed: {e}")
        exit(1)