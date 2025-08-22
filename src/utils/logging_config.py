#!/usr/bin/env python3
"""
Enhanced Logging Configuration for SolTrader
Configures Python logging with rotation, formatting, and monitoring
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import threading
import queue
import time

class RotatingFileHandlerWithCallback(logging.handlers.RotatingFileHandler):
    """Custom rotating file handler that integrates with our log management"""
    
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, 
                 encoding=None, delay=False, rotation_callback=None):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.rotation_callback = rotation_callback
    
    def doRollover(self):
        """Override to call our callback after rotation"""
        super().doRollover()
        if self.rotation_callback:
            try:
                self.rotation_callback(self.baseFilename)
            except Exception as e:
                # Don't let callback errors break logging
                print(f"Warning: Log rotation callback failed: {e}", file=sys.stderr)

class LoggingSetup:
    """Centralized logging setup for SolTrader"""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_dir = Path(log_directory)
        self.log_dir.mkdir(exist_ok=True)
        
        self.handlers = {}
        self.loggers = {}
        
        # Rotation settings
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.backup_count = 5
        
        # Custom formatter
        self.formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Performance formatter (more detailed)
        self.performance_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console formatter (simpler)
        self.console_formatter = logging.Formatter(
            fmt='%(levelname)s - %(name)s - %(message)s'
        )
    
    def _rotation_callback(self, filename: str):
        """Callback when log file rotates"""
        # This integrates with our log_management system
        try:
            from .log_management import log_rotation_manager
            # Update stats when Python's built-in rotation happens
            log_rotation_manager.stats["total_rotations"] += 1
            log_rotation_manager.stats["last_rotation"] = time.time()
        except ImportError:
            pass
    
    def create_file_handler(self, 
                           log_name: str, 
                           level: int = logging.INFO,
                           max_bytes: Optional[int] = None,
                           backup_count: Optional[int] = None) -> logging.Handler:
        """Create a rotating file handler"""
        
        log_file = self.log_dir / f"{log_name}.log"
        
        handler = RotatingFileHandlerWithCallback(
            filename=str(log_file),
            maxBytes=max_bytes or self.max_file_size,
            backupCount=backup_count or self.backup_count,
            encoding='utf-8',
            rotation_callback=self._rotation_callback
        )
        
        handler.setLevel(level)
        
        # Use performance formatter for certain logs
        if log_name in ['performance', 'trading']:
            handler.setFormatter(self.performance_formatter)
        else:
            handler.setFormatter(self.formatter)
        
        self.handlers[log_name] = handler
        return handler
    
    def create_console_handler(self, level: int = logging.WARNING) -> logging.Handler:
        """Create console handler for important messages"""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(self.console_formatter)
        return handler
    
    def setup_logger(self, 
                     name: str,
                     log_to_file: bool = True,
                     log_to_console: bool = True,
                     file_level: int = logging.INFO,
                     console_level: int = logging.WARNING) -> logging.Logger:
        """Setup a logger with file and console handlers"""
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Capture everything, handlers will filter
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Add file handler
        if log_to_file:
            file_handler = self.create_file_handler(name, file_level)
            logger.addHandler(file_handler)
        
        # Add console handler
        if log_to_console:
            console_handler = self.create_console_handler(console_level)
            logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        self.loggers[name] = logger
        return logger
    
    def setup_all_loggers(self):
        """Setup all standard SolTrader loggers"""
        
        # Main trading logger
        self.setup_logger(
            'trading', 
            file_level=logging.INFO,
            console_level=logging.WARNING
        )
        
        # API communication logger
        self.setup_logger(
            'api',
            file_level=logging.DEBUG, 
            console_level=logging.ERROR
        )
        
        # Error logger (high priority items)
        self.setup_logger(
            'error',
            file_level=logging.WARNING,
            console_level=logging.ERROR
        )
        
        # Performance logger
        self.setup_logger(
            'performance',
            file_level=logging.INFO,
            console_level=logging.CRITICAL,  # Only show critical perf issues on console
            log_to_console=False  # Usually only log to file
        )
        
        # Configure root logger to catch uncaught exceptions
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
        
        # Add error handler to root logger
        if 'error' not in self.handlers:
            error_handler = self.create_file_handler('error', logging.WARNING)
            root_logger.addHandler(error_handler)
        
        return self.loggers
    
    def get_logger_stats(self) -> Dict[str, any]:
        """Get statistics about logging setup"""
        stats = {
            'handlers_created': len(self.handlers),
            'loggers_setup': len(self.loggers),
            'log_directory': str(self.log_dir),
            'max_file_size_mb': self.max_file_size / (1024 * 1024),
            'backup_count': self.backup_count,
            'log_files': {}
        }
        
        # Check actual log files
        for log_file in self.log_dir.glob("*.log"):
            try:
                size_mb = log_file.stat().st_size / (1024 * 1024)
                stats['log_files'][log_file.name] = {
                    'size_mb': round(size_mb, 2),
                    'exists': True
                }
            except OSError:
                stats['log_files'][log_file.name] = {
                    'size_mb': 0,
                    'exists': False
                }
        
        return stats

# Global logging setup instance
logging_setup = LoggingSetup()

def setup_production_logging():
    """Setup all production logging"""
    return logging_setup.setup_all_loggers()

# Legacy compatibility function
def setup_logging(log_level: str = "INFO") -> None:
    """Legacy logging setup - now uses enhanced system"""
    setup_production_logging()
    # Set third-party loggers to WARNING level
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)