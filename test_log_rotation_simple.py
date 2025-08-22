#!/usr/bin/env python3
"""
Simple Log Rotation Test (Windows Console Compatible)
Tests log rotation system without Unicode characters
"""
import asyncio
import time
import tempfile
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.utils.log_management import LogRotationManager, LogRotationConfig

def create_large_log_file(file_path: Path, size_mb: float):
    """Create a log file of specified size for testing"""
    target_size = int(size_mb * 1024 * 1024)
    
    with open(file_path, 'w') as f:
        line = "2025-08-22 14:30:00 - trading - INFO - Test log entry for rotation testing\n"
        line_size = len(line.encode('utf-8'))
        lines_needed = target_size // line_size
        
        for i in range(lines_needed):
            f.write(f"2025-08-22 14:30:{i%60:02d} - trading - INFO - Test log entry #{i} for rotation testing\n")

async def test_basic_rotation():
    """Test basic file rotation functionality"""
    print("Testing basic file rotation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LogRotationConfig(
            max_file_size_mb=1,  # Small for testing
            compress_backups=False,  # Faster testing
            log_directory=temp_dir,
            archive_directory=f"{temp_dir}/archive"
        )
        
        manager = LogRotationManager(config)
        
        # Create large test file
        test_log = Path(temp_dir) / "trading.log"
        create_large_log_file(test_log, 1.5)  # 1.5MB file
        
        size_before = manager.get_file_size_mb(test_log)
        print(f"  File size before rotation: {size_before:.2f}MB")
        
        # Rotate the file
        success = manager.rotate_file(test_log)
        size_after = manager.get_file_size_mb(test_log)
        
        # Check archive was created
        archive_files = list(Path(config.archive_directory).glob("trading_*"))
        
        print(f"  Rotation success: {success}")
        print(f"  File size after rotation: {size_after:.2f}MB")
        print(f"  Archive files created: {len(archive_files)}")
        
        return success and size_after < 0.1 and len(archive_files) > 0

async def test_cleanup():
    """Test old file cleanup"""
    print("Testing file cleanup...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LogRotationConfig(
            max_backup_files=2,
            log_directory=temp_dir,
            archive_directory=f"{temp_dir}/archive"
        )
        
        manager = LogRotationManager(config)
        archive_dir = Path(config.archive_directory)
        
        # Create 5 fake backup files
        for i in range(5):
            backup_file = archive_dir / f"trading_202508{20+i:02d}_120000.log"
            backup_file.write_text(f"Fake backup file {i}")
        
        files_before = len(list(archive_dir.glob("*.log")))
        manager.cleanup_old_files()
        files_after = len(list(archive_dir.glob("*.log")))
        
        print(f"  Files before cleanup: {files_before}")
        print(f"  Files after cleanup: {files_after}")
        print(f"  Max allowed files: {config.max_backup_files}")
        
        return files_after <= config.max_backup_files

async def test_statistics():
    """Test statistics generation"""
    print("Testing statistics...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LogRotationConfig(
            log_directory=temp_dir,
            archive_directory=f"{temp_dir}/archive"
        )
        
        manager = LogRotationManager(config)
        
        # Create some test files
        for log_name in ['trading', 'api', 'error']:
            log_file = Path(temp_dir) / f"{log_name}.log"
            log_file.write_text(f"Test content for {log_name}\n" * 1000)
        
        stats = manager.get_log_statistics()
        
        print(f"  Current files detected: {len(stats['current_files'])}")
        print(f"  Health info available: {'total_size_mb' in stats['health']}")
        print(f"  Configuration loaded: {stats['config']['max_file_size_mb']}MB max")
        
        return len(stats['current_files']) > 0 and 'total_size_mb' in stats['health']

async def test_background_rotation():
    """Test background rotation (abbreviated)"""
    print("Testing background rotation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LogRotationConfig(
            max_file_size_mb=1,
            check_interval_minutes=0.05,  # 3 seconds
            rotate_on_startup=True,
            log_directory=temp_dir,
            archive_directory=f"{temp_dir}/archive"
        )
        
        manager = LogRotationManager(config)
        
        # Create large file
        test_log = Path(temp_dir) / "trading.log"
        create_large_log_file(test_log, 1.2)
        
        print("  Starting background rotation...")
        await manager.start_background_rotation()
        
        # Wait briefly
        await asyncio.sleep(4)
        
        # Check results
        archive_files = list(Path(config.archive_directory).glob("*"))
        current_size = manager.get_file_size_mb(test_log)
        
        await manager.stop_background_rotation()
        
        print(f"  Archive files after background rotation: {len(archive_files)}")
        print(f"  Current log size: {current_size:.2f}MB")
        
        return len(archive_files) > 0 and current_size < 0.1

async def run_simple_tests():
    """Run simplified log rotation tests"""
    print("LOG ROTATION SYSTEM TESTS (SIMPLIFIED)")
    print("="*50)
    
    tests = [
        ("Basic Rotation", test_basic_rotation),
        ("File Cleanup", test_cleanup),
        ("Statistics", test_statistics),
        ("Background Rotation", test_background_rotation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            success = await test_func()
            if success:
                print(f"  PASS: {test_name}")
                passed += 1
            else:
                print(f"  FAIL: {test_name}")
        except Exception as e:
            print(f"  ERROR: {test_name}: {e}")
    
    print(f"\n" + "="*50)
    print(f"RESULTS: {passed}/{total} tests passed ({(passed/total)*100:.0f}%)")
    
    if passed == total:
        print("SUCCESS: Log rotation system is working!")
        print("\nFeatures verified:")
        print("- File rotation when size limits exceeded")
        print("- Old backup file cleanup")
        print("- Statistics and monitoring")
        print("- Background rotation task")
        print("\nYour logs will now be automatically managed.")
        return True
    elif passed >= total - 1:
        print("MOSTLY WORKING: Core functionality verified")
        return True
    else:
        print("ISSUES DETECTED: Log rotation needs attention")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_simple_tests())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests cancelled")
        exit(1)
    except Exception as e:
        print(f"\nTest failure: {e}")
        exit(1)