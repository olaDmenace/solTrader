import asyncio
import sqlite3
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from src.utils.trading_time import trading_time


class DatabaseManager:
    def __init__(self, settings):
        self.settings = settings
        self.db_path = getattr(settings, 'DATABASE_PATH', 'soltrader.db')
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize database and create tables"""
        try:
            # Ensure database directory exists
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            # Create core tables
            await self._create_core_tables()
            
            self.logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def _handle_corrupted_database(self):
        """Handle corrupted database by backing up and creating new one"""
        try:
            import shutil
            from datetime import datetime
            
            # Create backup of corrupted database
            timestamp = trading_time.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{self.db_path}.corrupted_{timestamp}"
            
            if Path(self.db_path).exists():
                shutil.move(self.db_path, backup_path)
                self.logger.warning(f"Corrupted database backed up to: {backup_path}")
            
            # Remove any existing file
            if Path(self.db_path).exists():
                Path(self.db_path).unlink()
                
            self.logger.info(f"Creating new database: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle corrupted database: {e}")
            # As last resort, just try to remove the file
            try:
                if Path(self.db_path).exists():
                    Path(self.db_path).unlink()
            except:
                pass
            
    async def _create_core_tables(self):
        """Create essential database tables"""
        try:
            # Test database integrity first
            try:
                conn = sqlite3.connect(self.db_path, timeout=5.0)  # 5 second timeout
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                if result[0] != 'ok':
                    self.logger.warning(f"Database integrity check failed: {result[0]}")
                    conn.close()
                    # Backup corrupted database and create new one
                    self._handle_corrupted_database()
                    conn = sqlite3.connect(self.db_path, timeout=5.0)
                    cursor = conn.cursor()
            except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
                self.logger.warning(f"Database corruption detected: {e}")
                if 'conn' in locals():
                    conn.close()
                # Handle corrupted database
                self._handle_corrupted_database()
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                cursor = conn.cursor()
            
            # System metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # System events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    severity TEXT DEFAULT 'info'
                )
            ''')
            
            # Trading positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    status TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Trading orders table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    status TEXT NOT NULL,
                    filled_quantity REAL DEFAULT 0,
                    avg_fill_price REAL DEFAULT 0,
                    strategy_name TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Strategy performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    return_value REAL NOT NULL,
                    trade_data TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to create core tables: {e}")
            raise
            
    async def log_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Log a system metric"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)  # 5 second timeout
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_metrics (metric_name, metric_value, timestamp, metadata)
                VALUES (?, ?, ?, ?)
            ''', (
                metric_name, value, trading_time.now().isoformat(),
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log metric: {e}")
            
    async def log_event(self, event_type: str, event_data: Dict[str, Any], severity: str = 'info'):
        """Log a system event"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)  # 5 second timeout
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_events (event_type, event_data, timestamp, severity)
                VALUES (?, ?, ?, ?)
            ''', (
                event_type, json.dumps(event_data),
                trading_time.now().isoformat(), severity
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log event: {e}")
            
    async def get_metrics(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics for the last N hours"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)  # 5 second timeout
            cursor = conn.cursor()
            
            since_time = trading_time.now().replace(hour=trading_time.now().hour - hours)
            
            cursor.execute('''
                SELECT metric_name, metric_value, timestamp, metadata
                FROM system_metrics
                WHERE metric_name = ? AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (metric_name, since_time.isoformat()))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'metric_name': row[0],
                    'metric_value': row[1],
                    'timestamp': row[2],
                    'metadata': json.loads(row[3]) if row[3] else None
                })
                
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            return []
            
    async def close(self):
        """Close database connections"""
        self.logger.info("Database manager closed")