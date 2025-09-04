#!/usr/bin/env python3
"""
Database Schema Validation - Windows Compatible
Comprehensive validation of all database tables and their structure
"""

import asyncio
import sqlite3
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import load_settings
from src.database.db_manager import DatabaseManager

logging.basicConfig(level=logging.WARNING)  # Suppress debug noise
logger = logging.getLogger(__name__)


async def validate_database_schema():
    """Validate all database schemas"""
    print("=" * 80)
    print("DATABASE SCHEMA VALIDATION")
    print("=" * 80)
    
    settings = load_settings()
    db_manager = DatabaseManager(settings)
    
    # Expected table schemas
    expected_schemas = {
        'system_metrics': ['id', 'metric_name', 'metric_value', 'timestamp', 'metadata'],
        'system_events': ['id', 'event_type', 'event_data', 'timestamp', 'severity'],
        'positions': ['id', 'position_id', 'symbol', 'direction', 'quantity', 'entry_price', 'current_price', 'unrealized_pnl', 'realized_pnl', 'status', 'strategy_name', 'entry_time', 'exit_time', 'metadata', 'created_at', 'updated_at'],
        'orders': ['id', 'order_id', 'symbol', 'direction', 'order_type', 'quantity', 'price', 'status', 'filled_quantity', 'avg_fill_price', 'strategy_name', 'metadata', 'created_at', 'updated_at'],
        'risk_assessments': ['id', 'symbol', 'risk_level', 'risk_score', 'max_loss', 'confidence', 'recommendation', 'metadata', 'timestamp'],
        'portfolio_risk_metrics': ['id', 'metric_name', 'metric_value', 'timestamp'],
        'strategy_performance': ['id', 'strategy_name', 'return_value', 'trade_data', 'timestamp']
    }
    
    try:
        # Initialize database manager
        await db_manager.initialize()
        print("[OK] Database connection established")
        
        # Get all existing tables
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_table_names = [row[0] for row in cursor.fetchall()]
        
        print(f"[INFO] Found {len(existing_table_names)} existing tables: {existing_table_names}")
        
        # Validate each expected table
        validation_results = {}
        missing_tables = []
        
        for table_name, expected_columns in expected_schemas.items():
            print(f"\n[CHECKING] Table: {table_name}")
            
            if table_name not in existing_table_names:
                print(f"   [MISSING] Table '{table_name}' does not exist")
                validation_results[table_name] = False
                missing_tables.append(table_name)
                continue
                
            # Get column info for this table
            cursor.execute(f"PRAGMA table_info({table_name})")
            actual_columns = [col[1] for col in cursor.fetchall()]
            
            print(f"   Expected columns: {len(expected_columns)}")
            print(f"   Actual columns: {len(actual_columns)}")
            
            # Check for missing columns
            missing_columns = set(expected_columns) - set(actual_columns)
            extra_columns = set(actual_columns) - set(expected_columns)
            
            if missing_columns:
                print(f"   [MISSING COLS] {missing_columns}")
                validation_results[table_name] = False
            elif extra_columns:
                print(f"   [EXTRA COLS] {extra_columns}")
                validation_results[table_name] = True  # Extra columns are OK
            else:
                print(f"   [OK] Schema is valid")
                validation_results[table_name] = True
        
        conn.close()
        
        # Create missing tables if needed
        if missing_tables:
            print(f"\n[ACTION] Creating {len(missing_tables)} missing tables...")
            await create_missing_tables(db_manager, settings)
            
            # Re-validate
            print(f"\n[RECHECK] Re-validating schemas...")
            conn = sqlite3.connect(db_manager.db_path)
            cursor = conn.cursor()
            
            for table_name in missing_tables:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                if cursor.fetchone():
                    print(f"   [OK] {table_name} now exists")
                    validation_results[table_name] = True
                else:
                    print(f"   [FAIL] {table_name} still missing")
                    
            conn.close()
        
        # Generate summary report
        print("\n" + "=" * 80)
        print("SCHEMA VALIDATION REPORT")
        print("=" * 80)
        
        total_tables = len(expected_schemas)
        valid_tables = sum(1 for result in validation_results.values() if result)
        
        print(f"Total Tables Expected: {total_tables}")
        print(f"Valid Tables: {valid_tables}")
        print(f"Invalid Tables: {total_tables - valid_tables}")
        print(f"Success Rate: {(valid_tables/total_tables)*100:.1f}%")
        
        print(f"\nTABLE STATUS:")
        for table_name, is_valid in validation_results.items():
            status = "[VALID]" if is_valid else "[INVALID]"
            print(f"   {status} {table_name}")
            
        all_valid = valid_tables == total_tables
        
        if all_valid:
            print(f"\n[SUCCESS] ALL DATABASE SCHEMAS ARE VALID!")
            print(f"Database is ready for integration testing")
        else:
            print(f"\n[WARNING] SOME SCHEMA ISSUES DETECTED!")
            print(f"Some tables may need manual creation")
            
        await db_manager.close()
        return all_valid
        
    except Exception as e:
        print(f"[ERROR] Schema validation failed: {e}")
        await db_manager.close()
        return False


async def create_missing_tables(db_manager, settings):
    """Create any missing tables by initializing all components"""
    try:
        # Import and initialize all components that create tables
        from src.trading.risk_engine import RiskEngine, RiskEngineConfig
        from src.portfolio.performance_based_rebalancer import PerformanceBasedRebalancer
        from src.trading.paper_trading_engine import PaperTradingEngine, PaperTradingMode
        from src.monitoring.system_monitor import SystemMonitor
        
        # Risk engine creates risk tables
        print("   Creating risk management tables...")
        risk_config = RiskEngineConfig()
        risk_engine = RiskEngine(db_manager, risk_config)
        await risk_engine.initialize()
        
        # Performance rebalancer creates performance tables  
        print("   Creating performance tracking tables...")
        rebalancer = PerformanceBasedRebalancer(settings)
        await rebalancer.initialize()
        
        # System monitor creates monitoring tables (already done by db_manager)
        print("   System monitoring tables ready...")
        
        # Paper trading engine creates paper trading tables
        print("   Creating paper trading tables...")
        paper_engine = PaperTradingEngine(
            db_manager, risk_engine, None, None, PaperTradingMode.SIMULATION, 10000.0
        )
        await paper_engine.initialize()
        
        # Cleanup
        await paper_engine.shutdown()
        await rebalancer.shutdown() 
        await risk_engine.shutdown()
        
        print("   [OK] All missing tables created successfully")
        
    except Exception as e:
        print(f"   [ERROR] Failed to create missing tables: {e}")


async def main():
    """Main validation execution"""
    print("Starting database schema validation...")
    is_valid = await validate_database_schema()
    
    if is_valid:
        print("\n[RESULT] Database schema validation PASSED")
    else:
        print("\n[RESULT] Database schema validation FAILED")
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)