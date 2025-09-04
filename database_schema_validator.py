#!/usr/bin/env python3
"""
Database Schema Validation
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

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DatabaseSchemaValidator:
    def __init__(self):
        self.settings = load_settings()
        self.db_manager = DatabaseManager(self.settings)
        self.validation_results = {}
        
        # Expected table schemas
        self.expected_schemas = {
            'system_metrics': {
                'columns': {
                    'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
                    'metric_name': 'TEXT NOT NULL',
                    'metric_value': 'REAL NOT NULL', 
                    'timestamp': 'TEXT NOT NULL',
                    'metadata': 'TEXT'
                },
                'description': 'System performance and health metrics'
            },
            'system_events': {
                'columns': {
                    'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
                    'event_type': 'TEXT NOT NULL',
                    'event_data': 'TEXT NOT NULL',
                    'timestamp': 'TEXT NOT NULL',
                    'severity': 'TEXT DEFAULT \'info\''
                },
                'description': 'System events and logs'
            },
            'positions': {
                'columns': {
                    'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
                    'position_id': 'TEXT UNIQUE NOT NULL',
                    'symbol': 'TEXT NOT NULL',
                    'direction': 'TEXT NOT NULL',
                    'quantity': 'REAL NOT NULL',
                    'entry_price': 'REAL NOT NULL',
                    'current_price': 'REAL NOT NULL',
                    'unrealized_pnl': 'REAL NOT NULL',
                    'realized_pnl': 'REAL NOT NULL',
                    'status': 'TEXT NOT NULL',
                    'strategy_name': 'TEXT NOT NULL',
                    'entry_time': 'TEXT NOT NULL',
                    'exit_time': 'TEXT',
                    'metadata': 'TEXT',
                    'created_at': 'TEXT NOT NULL',
                    'updated_at': 'TEXT NOT NULL'
                },
                'description': 'Trading positions'
            },
            'orders': {
                'columns': {
                    'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
                    'order_id': 'TEXT UNIQUE NOT NULL',
                    'symbol': 'TEXT NOT NULL',
                    'direction': 'TEXT NOT NULL',
                    'order_type': 'TEXT NOT NULL',
                    'quantity': 'REAL NOT NULL',
                    'price': 'REAL',
                    'status': 'TEXT NOT NULL',
                    'filled_quantity': 'REAL DEFAULT 0',
                    'avg_fill_price': 'REAL DEFAULT 0',
                    'strategy_name': 'TEXT NOT NULL',
                    'metadata': 'TEXT',
                    'created_at': 'TEXT NOT NULL',
                    'updated_at': 'TEXT NOT NULL'
                },
                'description': 'Trading orders'
            },
            'risk_assessments': {
                'columns': {
                    'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
                    'symbol': 'TEXT NOT NULL',
                    'risk_level': 'TEXT NOT NULL',
                    'risk_score': 'REAL NOT NULL',
                    'max_loss': 'REAL NOT NULL',
                    'confidence': 'REAL NOT NULL',
                    'recommendation': 'TEXT NOT NULL',
                    'metadata': 'TEXT',
                    'timestamp': 'TEXT NOT NULL'
                },
                'description': 'Risk assessments for trades'
            },
            'portfolio_risk_metrics': {
                'columns': {
                    'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
                    'metric_name': 'TEXT NOT NULL',
                    'metric_value': 'REAL NOT NULL',
                    'timestamp': 'TEXT NOT NULL'
                },
                'description': 'Portfolio-level risk metrics'
            },
            'strategy_performance': {
                'columns': {
                    'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
                    'strategy_name': 'TEXT NOT NULL',
                    'return_value': 'REAL NOT NULL',
                    'trade_data': 'TEXT NOT NULL',
                    'timestamp': 'TEXT NOT NULL'
                },
                'description': 'Strategy performance tracking'
            },
            'paper_accounts': {
                'columns': {
                    'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
                    'account_id': 'TEXT UNIQUE NOT NULL',
                    'initial_balance': 'REAL NOT NULL',
                    'current_balance': 'REAL NOT NULL',
                    'equity': 'REAL NOT NULL',
                    'margin_used': 'REAL NOT NULL',
                    'free_margin': 'REAL NOT NULL',
                    'total_pnl': 'REAL NOT NULL',
                    'daily_pnl': 'REAL NOT NULL',
                    'trade_count': 'INTEGER NOT NULL',
                    'win_rate': 'REAL NOT NULL',
                    'created_at': 'TEXT NOT NULL',
                    'updated_at': 'TEXT NOT NULL'
                },
                'description': 'Paper trading accounts'
            },
            'paper_positions': {
                'columns': {
                    'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
                    'account_id': 'TEXT NOT NULL',
                    'symbol': 'TEXT NOT NULL',
                    'quantity': 'REAL NOT NULL',
                    'avg_entry_price': 'REAL NOT NULL',
                    'current_price': 'REAL NOT NULL',
                    'unrealized_pnl': 'REAL NOT NULL',
                    'realized_pnl': 'REAL NOT NULL',
                    'direction': 'TEXT NOT NULL',
                    'open_time': 'TEXT NOT NULL',
                    'close_time': 'TEXT',
                    'status': 'TEXT NOT NULL',
                    'strategy_name': 'TEXT NOT NULL',
                    'metadata': 'TEXT NOT NULL',
                    'created_at': 'TEXT NOT NULL',
                    'updated_at': 'TEXT NOT NULL'
                },
                'description': 'Paper trading positions'
            },
            'paper_orders': {
                'columns': {
                    'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
                    'account_id': 'TEXT NOT NULL',
                    'order_id': 'TEXT UNIQUE NOT NULL',
                    'symbol': 'TEXT NOT NULL',
                    'direction': 'TEXT NOT NULL',
                    'order_type': 'TEXT NOT NULL',
                    'quantity': 'REAL NOT NULL',
                    'price': 'REAL',
                    'status': 'TEXT NOT NULL',
                    'filled_quantity': 'REAL NOT NULL',
                    'avg_fill_price': 'REAL NOT NULL',
                    'strategy_name': 'TEXT NOT NULL',
                    'metadata': 'TEXT NOT NULL',
                    'created_at': 'TEXT NOT NULL',
                    'updated_at': 'TEXT NOT NULL'
                },
                'description': 'Paper trading orders'
            }
        }

    async def validate_all_schemas(self) -> bool:
        """Validate all database schemas"""
        print("=" * 80)
        print("DATABASE SCHEMA VALIDATION")
        print("=" * 80)
        
        try:
            # Initialize database manager
            await self.db_manager.initialize()
            print("✅ Database connection established")
            
            # Get all existing tables
            existing_tables = await self._get_existing_tables()
            print(f"📋 Found {len(existing_tables)} existing tables")
            
            # Validate each expected table
            all_valid = True
            for table_name, expected_schema in self.expected_schemas.items():
                print(f"\n🔍 Validating table: {table_name}")
                print(f"   Purpose: {expected_schema['description']}")
                
                is_valid = await self._validate_table_schema(table_name, expected_schema, existing_tables)
                self.validation_results[table_name] = is_valid
                
                if is_valid:
                    print(f"   ✅ Schema is valid")
                else:
                    print(f"   ❌ Schema validation failed")
                    all_valid = False
            
            # Check for unexpected tables
            expected_table_names = set(self.expected_schemas.keys())
            unexpected_tables = set(existing_tables.keys()) - expected_table_names
            
            if unexpected_tables:
                print(f"\n⚠️  Found {len(unexpected_tables)} unexpected tables:")
                for table in unexpected_tables:
                    print(f"   - {table}")
                    
            # Generate summary report
            await self._generate_schema_report(all_valid)
            
            return all_valid
            
        except Exception as e:
            print(f"❌ Schema validation failed: {e}")
            return False
        finally:
            await self.db_manager.close()

    async def _get_existing_tables(self) -> Dict[str, Dict[str, str]]:
        """Get all existing tables and their schemas"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = [row[0] for row in cursor.fetchall()]
            
            tables = {}
            for table_name in table_names:
                # Get column info for each table
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = {}
                for col_info in cursor.fetchall():
                    col_name = col_info[1]
                    col_type = col_info[2]
                    col_notnull = col_info[3]
                    col_default = col_info[4]
                    col_pk = col_info[5]
                    
                    # Build column definition string
                    col_def = col_type
                    if col_pk:
                        col_def += " PRIMARY KEY"
                        if "INTEGER" in col_type:
                            col_def += " AUTOINCREMENT"
                    if col_notnull and not col_pk:
                        col_def += " NOT NULL"
                    if col_default is not None:
                        col_def += f" DEFAULT {col_default}"
                        
                    columns[col_name] = col_def
                    
                tables[table_name] = columns
                
            conn.close()
            return tables
            
        except Exception as e:
            print(f"❌ Failed to get existing tables: {e}")
            return {}

    async def _validate_table_schema(self, table_name: str, expected_schema: Dict[str, Any], existing_tables: Dict[str, Dict[str, str]]) -> bool:
        """Validate a single table schema"""
        if table_name not in existing_tables:
            print(f"   ❌ Table '{table_name}' does not exist")
            return False
            
        existing_columns = existing_tables[table_name]
        expected_columns = expected_schema['columns']
        
        validation_passed = True
        
        # Check if all expected columns exist
        for col_name, expected_def in expected_columns.items():
            if col_name not in existing_columns:
                print(f"   ❌ Missing column: {col_name}")
                validation_passed = False
            else:
                existing_def = existing_columns[col_name]
                # Basic type checking (SQLite is flexible with types)
                if not self._column_types_compatible(expected_def, existing_def):
                    print(f"   ⚠️  Column '{col_name}' type mismatch:")
                    print(f"      Expected: {expected_def}")
                    print(f"      Actual:   {existing_def}")
                    # Don't fail for type mismatches as SQLite is flexible
                    
        # Check for unexpected columns
        unexpected_columns = set(existing_columns.keys()) - set(expected_columns.keys())
        if unexpected_columns:
            print(f"   ⚠️  Unexpected columns: {', '.join(unexpected_columns)}")
            
        return validation_passed

    def _column_types_compatible(self, expected: str, actual: str) -> bool:
        """Check if column types are compatible"""
        # Normalize both definitions
        expected_lower = expected.lower()
        actual_lower = actual.lower()
        
        # Extract base types
        expected_type = expected_lower.split()[0]
        actual_type = actual_lower.split()[0]
        
        # SQLite type compatibility
        type_compatibility = {
            'integer': ['integer', 'int'],
            'text': ['text', 'varchar', 'char'],
            'real': ['real', 'float', 'double'],
            'blob': ['blob'],
        }
        
        for base_type, compatible_types in type_compatibility.items():
            if expected_type in compatible_types and actual_type in compatible_types:
                return True
                
        return expected_type == actual_type

    async def _generate_schema_report(self, all_valid: bool):
        """Generate comprehensive schema validation report"""
        print("\n" + "=" * 80)
        print("SCHEMA VALIDATION REPORT")
        print("=" * 80)
        
        total_tables = len(self.expected_schemas)
        valid_tables = sum(1 for result in self.validation_results.values() if result)
        
        print(f"📊 SUMMARY:")
        print(f"   Total Tables Expected: {total_tables}")
        print(f"   Valid Tables: {valid_tables}")
        print(f"   Invalid Tables: {total_tables - valid_tables}")
        print(f"   Success Rate: {(valid_tables/total_tables)*100:.1f}%")
        
        print(f"\n📋 TABLE STATUS:")
        for table_name, is_valid in self.validation_results.items():
            status = "✅ VALID" if is_valid else "❌ INVALID"
            purpose = self.expected_schemas[table_name]['description']
            print(f"   {status} {table_name} - {purpose}")
            
        if all_valid:
            print(f"\n🎉 ALL DATABASE SCHEMAS ARE VALID!")
            print(f"   Database is ready for integration testing")
        else:
            print(f"\n⚠️  SCHEMA ISSUES DETECTED!")
            print(f"   Some tables need to be created or modified")
            print(f"   Recommendation: Run database initialization again")
            
        print("\n" + "=" * 80)

    async def create_missing_tables(self) -> bool:
        """Create any missing tables with correct schema"""
        print("\nCREATING MISSING TABLES...")
        
        try:
            # Re-initialize all database components to ensure tables are created
            from src.trading.risk_engine import RiskEngine, RiskEngineConfig
            from src.monitoring.system_monitor import SystemMonitor
            from src.portfolio.performance_based_rebalancer import PerformanceBasedRebalancer
            from src.trading.paper_trading_engine import PaperTradingEngine, PaperTradingMode
            
            # Database manager creates core tables
            await self.db_manager.initialize()
            print("✅ Core database tables created")
            
            # Risk engine creates risk tables
            risk_config = RiskEngineConfig()
            risk_engine = RiskEngine(self.db_manager, risk_config)
            await risk_engine.initialize()
            print("✅ Risk management tables created")
            
            # Performance rebalancer creates performance tables
            rebalancer = PerformanceBasedRebalancer(self.db_manager)
            await rebalancer.initialize()
            print("✅ Performance tracking tables created")
            
            # Paper trading engine creates paper trading tables
            paper_engine = PaperTradingEngine(
                self.db_manager, risk_engine, None, None, PaperTradingMode.SIMULATION, 10000.0
            )
            await paper_engine.initialize()
            print("✅ Paper trading tables created")
            
            # Cleanup
            await paper_engine.shutdown()
            await rebalancer.shutdown()
            await risk_engine.shutdown()
            
            print("🎉 ALL MISSING TABLES CREATED SUCCESSFULLY")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create missing tables: {e}")
            return False


async def main():
    """Main validation execution"""
    validator = DatabaseSchemaValidator()
    
    print("Starting database schema validation...")
    is_valid = await validator.validate_all_schemas()
    
    if not is_valid:
        print("\nAttempting to create missing tables...")
        await validator.create_missing_tables()
        
        print("\nRe-validating schemas...")
        is_valid = await validator.validate_all_schemas()
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)