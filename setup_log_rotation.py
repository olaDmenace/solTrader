#!/usr/bin/env python3
"""
Setup Log Rotation for SolTrader Production System
Configures and starts log rotation for the existing trading bot
"""
import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.utils.log_management import LogRotationManager, LogRotationConfig, start_log_rotation
from src.utils.logging_config import setup_production_logging

async def setup_production_log_rotation():
    """Setup log rotation for production use"""
    print("Setting up SolTrader Log Rotation")
    print("="*50)
    
    # Configure log rotation
    config = LogRotationConfig(
        # File size limits
        max_file_size_mb=50,     # Rotate when files reach 50MB
        max_total_size_mb=500,   # Keep total log size under 500MB
        max_backup_files=10,     # Keep 10 backup files per log
        max_age_days=30,         # Delete files older than 30 days
        
        # Rotation behavior
        rotate_on_startup=True,  # Check logs on startup
        check_interval_minutes=60,  # Check every hour
        
        # Compression to save space
        compress_backups=True,
        compression_level=6,
        
        # Use existing logs directory
        log_directory="logs",
        archive_directory="logs/archive"
    )
    
    # Setup enhanced logging
    print("1. Setting up enhanced logging system...")
    loggers = setup_production_logging()
    print(f"   Created loggers: {list(loggers.keys())}")
    
    # Setup log rotation manager
    print("2. Configuring log rotation...")
    manager = LogRotationManager(config)
    
    # Show current log status
    stats = manager.get_log_statistics()
    print(f"   Current log files: {len(stats['current_files'])}")
    
    for log_name, info in stats['current_files'].items():
        status = "NEEDS ROTATION" if info['needs_rotation'] else "OK"
        print(f"     {log_name}: {info['size_mb']:.1f}MB ({status})")
    
    print(f"   Archive files: {stats['archive_info']['file_count']}")
    print(f"   Total size: {stats['health']['total_size_mb']:.1f}MB")
    
    # Start background rotation
    print("3. Starting background log rotation...")
    await start_log_rotation()
    
    print("✓ Log rotation setup complete!")
    print(f"\nConfiguration:")
    print(f"  Max file size: {config.max_file_size_mb}MB")
    print(f"  Max total size: {config.max_total_size_mb}MB") 
    print(f"  Backup retention: {config.max_backup_files} files, {config.max_age_days} days")
    print(f"  Check interval: {config.check_interval_minutes} minutes")
    print(f"  Compression: {'Enabled' if config.compress_backups else 'Disabled'}")
    
    print(f"\nLog files being managed:")
    print(f"  📊 trading.log - Main trading activity")
    print(f"  🔗 api.log - API communications") 
    print(f"  ⚠️ error.log - Errors and warnings")
    print(f"  📈 performance.log - Performance metrics")
    
    print(f"\nArchive directory: logs/archive/")
    print(f"Your logs will be automatically rotated and old files cleaned up.")
    
    return manager

async def check_log_status():
    """Check current log status"""
    from src.utils.log_management import log_rotation_manager
    
    print("\nCurrent Log Status:")
    print("-" * 30)
    
    stats = log_rotation_manager.get_log_statistics()
    
    print(f"Log Directory: {Path('logs').absolute()}")
    print(f"Archive Directory: {Path('logs/archive').absolute()}")
    
    if stats['current_files']:
        print(f"\nActive Log Files:")
        for log_name, info in stats['current_files'].items():
            print(f"  {log_name}: {info['size_mb']:.2f}MB")
    
    if stats['archive_info']['file_count'] > 0:
        print(f"\nArchive Status:")
        print(f"  Files: {stats['archive_info']['file_count']}")
        print(f"  Size: {stats['archive_info']['total_size_mb']:.2f}MB")
        
        if stats['archive_info'].get('oldest_file'):
            print(f"  Oldest: {stats['archive_info']['oldest_file']['name']} ({stats['archive_info']['oldest_file']['age_days']:.1f} days old)")
    
    print(f"\nSystem Health:")
    print(f"  Total size: {stats['health']['total_size_mb']:.2f}MB / {stats['config']['max_total_size_mb']}MB ({stats['health']['size_usage_percent']:.1f}%)")
    print(f"  Files needing rotation: {stats['health']['files_need_rotation']}")
    print(f"  Background rotation: {'Active' if stats['health']['background_rotation_active'] else 'Inactive'}")
    
    return stats

if __name__ == "__main__":
    try:
        print("SolTrader Log Rotation Setup")
        print("="*50)
        
        # Setup log rotation
        manager = asyncio.run(setup_production_log_rotation())
        
        # Show status
        asyncio.run(check_log_status())
        
        print(f"\n🎉 Setup complete! Your trading bot logs are now managed automatically.")
        print(f"\nTo monitor log rotation:")
        print(f"  - Check logs/archive/ directory for rotated files") 
        print(f"  - Logs will be rotated when they exceed 50MB")
        print(f"  - Old files will be deleted after 30 days")
        print(f"  - Compressed backups save disk space")
        
        # Keep running to demonstrate background rotation
        print(f"\nLog rotation is running in the background.")
        print(f"Press Ctrl+C to exit this demo (rotation will continue in your trading bot).")
        
        try:
            while True:
                await asyncio.sleep(60)  # Sleep 1 minute
                stats = await check_log_status()
                print(f"\n[{asyncio.get_event_loop().time():.0f}s] Log rotation check complete")
        except KeyboardInterrupt:
            print(f"\nStopping demo...")
            from src.utils.log_management import stop_log_rotation
            await stop_log_rotation()
            print(f"Demo stopped. Log rotation will resume when you start your trading bot.")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)