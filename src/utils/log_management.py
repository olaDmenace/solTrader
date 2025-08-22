#!/usr/bin/env python3
"""
Log Management and Rotation System
Handles automatic log rotation, cleanup, and monitoring for SolTrader
"""
import os
import gzip
import shutil
import logging
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import threading
import glob
import json

@dataclass
class LogRotationConfig:
    """Configuration for log rotation"""
    # File size limits
    max_file_size_mb: int = 50  # Rotate when file reaches this size
    max_total_size_mb: int = 500  # Max total size for all log files
    
    # File count limits
    max_backup_files: int = 10  # Keep this many backup files
    max_age_days: int = 30  # Delete files older than this
    
    # Rotation timing
    rotate_on_startup: bool = True
    check_interval_minutes: int = 60  # Check every hour
    
    # Compression
    compress_backups: bool = True
    compression_level: int = 6
    
    # Paths
    log_directory: str = "logs"
    archive_directory: str = "logs/archive"

class LogRotationManager:
    """Manages log rotation and cleanup for all log files"""
    
    def __init__(self, config: LogRotationConfig = None):
        self.config = config or LogRotationConfig()
        
        # Ensure directories exist
        self.log_dir = Path(self.config.log_directory)
        self.archive_dir = Path(self.config.archive_directory)
        self.log_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
        
        # Track managed log files
        self.managed_files = {
            "trading.log": self.log_dir / "trading.log",
            "error.log": self.log_dir / "error.log", 
            "api.log": self.log_dir / "api.log",
            "performance.log": self.log_dir / "performance.log"
        }
        
        # Background task management
        self._rotation_task = None
        self._stop_event = asyncio.Event()
        self._last_rotation_check = {}
        
        # Statistics
        self.stats = {
            "total_rotations": 0,
            "files_compressed": 0,
            "files_deleted": 0,
            "bytes_saved": 0,
            "last_rotation": None
        }
        
        # Setup logging for this module
        self.logger = logging.getLogger(__name__)
    
    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in megabytes"""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except (OSError, FileNotFoundError):
            return 0.0
    
    def get_directory_size_mb(self, directory: Path) -> float:
        """Get total size of all files in directory in megabytes"""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except (OSError, FileNotFoundError):
            pass
        return total_size / (1024 * 1024)
    
    def should_rotate_file(self, file_path: Path) -> bool:
        """Check if a file should be rotated based on size"""
        if not file_path.exists():
            return False
        
        file_size_mb = self.get_file_size_mb(file_path)
        return file_size_mb >= self.config.max_file_size_mb
    
    def rotate_file(self, file_path: Path) -> bool:
        """Rotate a single log file"""
        if not file_path.exists():
            self.logger.debug(f"File {file_path} does not exist, skipping rotation")
            return False
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = self.archive_dir / backup_name
            
            # Move current file to archive
            shutil.move(str(file_path), str(backup_path))
            
            # Create new empty log file
            file_path.touch()
            
            # Compress if enabled
            if self.config.compress_backups:
                compressed_path = backup_path.with_suffix(backup_path.suffix + ".gz")
                self._compress_file(backup_path, compressed_path)
                backup_path.unlink()  # Remove uncompressed version
                self.stats["files_compressed"] += 1
                self.stats["bytes_saved"] += self.get_file_size_mb(backup_path) * 1024 * 1024
            
            self.stats["total_rotations"] += 1
            self.stats["last_rotation"] = datetime.now().isoformat()
            
            self.logger.info(f"Rotated log file: {file_path} -> {backup_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rotate {file_path}: {e}")
            return False
    
    def _compress_file(self, source_path: Path, target_path: Path):
        """Compress a file using gzip"""
        try:
            with open(source_path, 'rb') as f_in:
                with gzip.open(target_path, 'wb', compresslevel=self.config.compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            self.logger.error(f"Failed to compress {source_path}: {e}")
            raise
    
    def cleanup_old_files(self):
        """Remove old backup files based on age and count limits"""
        try:
            # Get all backup files sorted by modification time (newest first)
            pattern = f"{self.archive_dir}/*"
            backup_files = []
            
            for file_path in glob.glob(pattern):
                path_obj = Path(file_path)
                if path_obj.is_file():
                    mtime = path_obj.stat().st_mtime
                    backup_files.append((path_obj, mtime))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove files beyond the count limit
            files_to_remove = []
            if len(backup_files) > self.config.max_backup_files:
                files_to_remove.extend(backup_files[self.config.max_backup_files:])
            
            # Remove files older than max age
            cutoff_time = time.time() - (self.config.max_age_days * 24 * 3600)
            for file_path, mtime in backup_files:
                if mtime < cutoff_time:
                    files_to_remove.append((file_path, mtime))
            
            # Remove duplicate entries
            files_to_remove = list(set(files_to_remove))
            
            # Delete the files
            deleted_count = 0
            for file_path, _ in files_to_remove:
                try:
                    file_size = self.get_file_size_mb(file_path)
                    file_path.unlink()
                    self.stats["files_deleted"] += 1
                    self.stats["bytes_saved"] += file_size * 1024 * 1024
                    deleted_count += 1
                    self.logger.debug(f"Deleted old backup: {file_path}")
                except Exception as e:
                    self.logger.error(f"Failed to delete {file_path}: {e}")
            
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old backup files")
                
        except Exception as e:
            self.logger.error(f"Failed during cleanup: {e}")
    
    def check_disk_space_limits(self):
        """Check if total log directory size exceeds limits"""
        total_size_mb = self.get_directory_size_mb(self.log_dir)
        
        if total_size_mb > self.config.max_total_size_mb:
            self.logger.warning(f"Log directory size ({total_size_mb:.2f}MB) exceeds limit ({self.config.max_total_size_mb}MB)")
            
            # Force cleanup of older files
            self.cleanup_old_files()
            
            # If still over limit, reduce backup count temporarily
            new_size = self.get_directory_size_mb(self.log_dir)
            if new_size > self.config.max_total_size_mb:
                self.logger.warning("Forcing aggressive cleanup due to disk space limits")
                self._aggressive_cleanup()
    
    def _aggressive_cleanup(self):
        """More aggressive cleanup when disk space is critical"""
        # Get all files in archive directory sorted by age
        archive_files = []
        for file_path in self.archive_dir.rglob("*"):
            if file_path.is_file():
                mtime = file_path.stat().st_mtime
                archive_files.append((file_path, mtime))
        
        # Sort by age (oldest first for deletion)
        archive_files.sort(key=lambda x: x[1])
        
        # Delete oldest files until we're under the limit
        deleted_count = 0
        for file_path, _ in archive_files:
            current_size = self.get_directory_size_mb(self.log_dir)
            if current_size <= self.config.max_total_size_mb * 0.8:  # Leave 20% buffer
                break
            
            try:
                file_path.unlink()
                deleted_count += 1
                self.stats["files_deleted"] += 1
            except Exception as e:
                self.logger.error(f"Failed to delete {file_path}: {e}")
        
        if deleted_count > 0:
            self.logger.warning(f"Aggressively deleted {deleted_count} files due to disk space limits")
    
    def rotate_all_logs(self):
        """Check and rotate all managed log files if needed"""
        rotated_count = 0
        
        for log_name, log_path in self.managed_files.items():
            if self.should_rotate_file(log_path):
                if self.rotate_file(log_path):
                    rotated_count += 1
        
        if rotated_count > 0:
            self.logger.info(f"Rotated {rotated_count} log files")
            
        # Always run cleanup after rotation
        self.cleanup_old_files()
        self.check_disk_space_limits()
        
        return rotated_count
    
    async def start_background_rotation(self):
        """Start background task for automatic log rotation"""
        if self._rotation_task is not None:
            self.logger.warning("Background rotation already running")
            return
        
        self.logger.info(f"Starting background log rotation (check every {self.config.check_interval_minutes} minutes)")
        
        # Initial rotation if configured
        if self.config.rotate_on_startup:
            self.rotate_all_logs()
        
        # Start background task
        self._stop_event.clear()
        self._rotation_task = asyncio.create_task(self._background_rotation_loop())
    
    async def stop_background_rotation(self):
        """Stop background log rotation"""
        if self._rotation_task is None:
            return
        
        self.logger.info("Stopping background log rotation")
        self._stop_event.set()
        
        try:
            await asyncio.wait_for(self._rotation_task, timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("Background rotation task did not stop gracefully")
            self._rotation_task.cancel()
        
        self._rotation_task = None
    
    async def _background_rotation_loop(self):
        """Background loop for log rotation"""
        try:
            while not self._stop_event.is_set():
                # Wait for next check interval
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.check_interval_minutes * 60
                    )
                    # If we get here, stop event was set
                    break
                except asyncio.TimeoutError:
                    # Timeout is normal, time to check logs
                    pass
                
                # Perform rotation check
                try:
                    self.rotate_all_logs()
                except Exception as e:
                    self.logger.error(f"Error during background rotation: {e}")
                    
                    # Send alert if available
                    try:
                        from .alerting_system import production_alerter
                        await production_alerter.send_alert(
                            title="Log Rotation Error",
                            message=f"Background log rotation failed: {e}",
                            severity="medium",
                            component="log_rotation"
                        )
                    except:
                        pass  # Don't break rotation loop if alerts fail
                
        except asyncio.CancelledError:
            self.logger.info("Background rotation task cancelled")
        except Exception as e:
            self.logger.error(f"Background rotation loop error: {e}")
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get comprehensive log statistics"""
        stats = {
            "config": {
                "max_file_size_mb": self.config.max_file_size_mb,
                "max_total_size_mb": self.config.max_total_size_mb,
                "max_backup_files": self.config.max_backup_files,
                "max_age_days": self.config.max_age_days
            },
            "current_files": {},
            "archive_info": {},
            "rotation_stats": self.stats.copy()
        }
        
        # Current file sizes
        total_current_size = 0
        for log_name, log_path in self.managed_files.items():
            if log_path.exists():
                size_mb = self.get_file_size_mb(log_path)
                stats["current_files"][log_name] = {
                    "size_mb": round(size_mb, 2),
                    "needs_rotation": size_mb >= self.config.max_file_size_mb,
                    "path": str(log_path)
                }
                total_current_size += size_mb
        
        # Archive directory info
        archive_files = list(self.archive_dir.glob("*"))
        archive_size = self.get_directory_size_mb(self.archive_dir)
        
        stats["archive_info"] = {
            "file_count": len(archive_files),
            "total_size_mb": round(archive_size, 2),
            "oldest_file": None,
            "newest_file": None
        }
        
        if archive_files:
            # Find oldest and newest files
            files_with_mtime = [(f, f.stat().st_mtime) for f in archive_files if f.is_file()]
            if files_with_mtime:
                oldest = min(files_with_mtime, key=lambda x: x[1])
                newest = max(files_with_mtime, key=lambda x: x[1])
                stats["archive_info"]["oldest_file"] = {
                    "name": oldest[0].name,
                    "age_days": round((time.time() - oldest[1]) / (24 * 3600), 1)
                }
                stats["archive_info"]["newest_file"] = {
                    "name": newest[0].name,
                    "age_days": round((time.time() - newest[1]) / (24 * 3600), 1)
                }
        
        # Overall health
        stats["health"] = {
            "total_size_mb": round(total_current_size + archive_size, 2),
            "size_limit_mb": self.config.max_total_size_mb,
            "size_usage_percent": round(((total_current_size + archive_size) / self.config.max_total_size_mb) * 100, 1),
            "files_need_rotation": sum(1 for f in stats["current_files"].values() if f.get("needs_rotation", False)),
            "background_rotation_active": self._rotation_task is not None
        }
        
        return stats
    
    def force_rotate_all(self):
        """Force rotation of all log files regardless of size"""
        self.logger.info("Forcing rotation of all log files")
        rotated_count = 0
        
        for log_name, log_path in self.managed_files.items():
            if log_path.exists() and self.get_file_size_mb(log_path) > 0:
                if self.rotate_file(log_path):
                    rotated_count += 1
        
        self.cleanup_old_files()
        self.check_disk_space_limits()
        
        self.logger.info(f"Force rotation completed: {rotated_count} files rotated")
        return rotated_count

# Global log rotation manager
log_rotation_manager = LogRotationManager()

def setup_log_rotation(config: LogRotationConfig = None) -> LogRotationManager:
    """Setup and return log rotation manager"""
    global log_rotation_manager
    if config:
        log_rotation_manager = LogRotationManager(config)
    return log_rotation_manager

async def start_log_rotation():
    """Start automatic log rotation"""
    await log_rotation_manager.start_background_rotation()

async def stop_log_rotation():
    """Stop automatic log rotation"""
    await log_rotation_manager.stop_background_rotation()

def rotate_logs_now():
    """Manually rotate logs now"""
    return log_rotation_manager.rotate_all_logs()

def get_log_stats():
    """Get log rotation statistics"""
    return log_rotation_manager.get_log_statistics()