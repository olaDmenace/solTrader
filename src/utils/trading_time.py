"""
Centralized Time Utility for Trading System
Provides consistent, timezone-aware datetime objects across all modules
"""

from datetime import datetime, timezone
from typing import Optional
import time

class TradingTimeManager:
    """Centralized time management for trading system"""
    
    @staticmethod
    def now() -> datetime:
        """Get current UTC time with timezone awareness"""
        return datetime.now(timezone.utc)
    
    @staticmethod  
    def timestamp() -> float:
        """Get current timestamp (seconds since epoch)"""
        return time.time()
    
    @staticmethod
    def timestamp_ms() -> int:
        """Get current timestamp in milliseconds"""
        return int(time.time() * 1000)
    
    @staticmethod
    def timestamp_ns() -> int:
        """Get current timestamp in nanoseconds (high precision)"""
        return time.time_ns()
    
    @staticmethod
    def from_timestamp(ts: float, tz: Optional[timezone] = None) -> datetime:
        """Convert timestamp to timezone-aware datetime"""
        if tz is None:
            tz = timezone.utc
        return datetime.fromtimestamp(ts, tz)
    
    @staticmethod
    def trading_timestamp() -> datetime:
        """Get trading-specific timestamp (high precision UTC)"""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def format_trading_time(dt: datetime) -> str:
        """Format datetime for trading logs (ISO format with timezone)"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    
    @staticmethod
    def parse_trading_time(time_str: str) -> datetime:
        """Parse trading time string back to datetime"""
        try:
            return datetime.fromisoformat(time_str)
        except ValueError:
            # Fallback for strings without timezone info
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

# Global instance for easy importing
trading_time = TradingTimeManager()

# Backwards compatibility functions (deprecated but available)
def utc_now():
    """DEPRECATED: Use trading_time.now() instead"""
    return trading_time.now()

def get_timestamp():
    """DEPRECATED: Use trading_time.timestamp() instead"""
    return trading_time.timestamp()
