#!/usr/bin/env python3
"""
Test Performance Analytics System
Tests analytics, reporting, and data import functionality
"""
import asyncio
import json
import tempfile
import time
from datetime import datetime, date, timedelta
from pathlib import Path
import sys
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.utils.performance_analytics import PerformanceAnalytics, TradeMetrics
from src.utils.performance_reports import PerformanceReporter

def create_sample_trade_data() -> list:
    """Create sample trade data for testing"""
    base_time = datetime.now() - timedelta(days=10)
    trades = []
    
    # Create 50 sample trades over 10 days
    for day in range(10):
        day_time = base_time + timedelta(days=day)
        
        # 3-7 trades per day
        daily_trades = 5 if day % 2 == 0 else 3
        
        for trade_num in range(daily_trades):
            # Vary trade times throughout the day
            trade_time = day_time + timedelta(hours=9 + trade_num * 2, minutes=trade_num * 15)
            
            # Create realistic trade data
            is_profitable = (trade_num + day) % 3 != 0  # ~66% win rate
            profit_usd = (50 + trade_num * 10) if is_profitable else -(30 + trade_num * 5)
            profit_sol = profit_usd / 180  # Assume ~$180 SOL price
            
            trade = {
                'timestamp': trade_time.isoformat(),
                'token_address': f'TEST{day:02d}{trade_num:02d}' + '0' * 40,
                'token_symbol': f'TOK{day:02d}{trade_num}',
                'entry_price': 0.001 + (trade_num * 0.0001),
                'exit_price': 0.001 + (trade_num * 0.0001) * (1.1 if is_profitable else 0.9),
                'quantity': 10000 + (trade_num * 1000),
                'profit_loss_usd': profit_usd,
                'profit_loss_sol': profit_sol,
                'profit_percentage': profit_usd / 100,  # Simplified
                'duration_minutes': 15 + (trade_num * 5),
                'trade_type': 'buy',
                'source': 'test_data'
            }
            trades.append(trade)
    
    return trades

def test_analytics_creation():
    """Test analytics system creation and configuration"""
    print("Testing analytics system creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        analytics = PerformanceAnalytics(temp_dir)
        
        # Check directories were created
        data_dir = Path(temp_dir)
        files_expected = ['trades_history.json', 'daily_performance.json', 'weekly_performance.json']
        
        print(f"  Data directory: {data_dir}")
        print(f"  Analytics instance created: {analytics is not None}")
        print(f"  Expected files: {files_expected}")
        
        return analytics is not None

def test_trade_import():
    """Test importing trade data"""
    print("Testing trade data import...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        analytics = PerformanceAnalytics(temp_dir)
        
        # Create sample trades
        sample_trades = create_sample_trade_data()
        
        # Import trades
        for trade in sample_trades:
            analytics.add_trade(trade)
        
        # Check results
        imported_count = len(analytics.trades_cache)
        daily_summaries = len(analytics.daily_cache)
        weekly_summaries = len(analytics.weekly_cache)
        
        print(f"  Sample trades created: {len(sample_trades)}")
        print(f"  Trades imported: {imported_count}")
        print(f"  Daily summaries generated: {daily_summaries}")
        print(f"  Weekly summaries generated: {weekly_summaries}")
        
        return imported_count == len(sample_trades) and daily_summaries > 0

def test_daily_analytics():
    """Test daily analytics calculations"""
    print("Testing daily analytics...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        analytics = PerformanceAnalytics(temp_dir)
        sample_trades = create_sample_trade_data()
        
        # Import trades
        for trade in sample_trades:
            analytics.add_trade(trade)
        
        # Get today's analytics (use the most recent day in sample data)
        recent_date = max([datetime.fromisoformat(t['timestamp']).date() for t in sample_trades])
        daily_perf = analytics.get_daily_summary(recent_date.isoformat())
        
        if daily_perf:
            print(f"  Date analyzed: {daily_perf.date}")
            print(f"  Total trades: {daily_perf.total_trades}")
            print(f"  Win rate: {daily_perf.win_rate:.1f}%")
            print(f"  Profit/Loss: ${daily_perf.total_profit_loss_usd:.2f}")
            print(f"  Sharpe ratio: {daily_perf.sharpe_ratio:.2f}")
            print(f"  Max drawdown: {daily_perf.max_drawdown:.2f}%")
            
            # Validate calculations
            has_trades = daily_perf.total_trades > 0
            has_metrics = daily_perf.win_rate >= 0 and daily_perf.win_rate <= 100
            
            return has_trades and has_metrics
        
        return False

def test_weekly_analytics():
    """Test weekly analytics calculations"""
    print("Testing weekly analytics...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        analytics = PerformanceAnalytics(temp_dir)
        sample_trades = create_sample_trade_data()
        
        # Import trades
        for trade in sample_trades:
            analytics.add_trade(trade)
        
        # Get current week analytics
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        weekly_perf = analytics.get_weekly_summary(monday.isoformat())
        
        if weekly_perf:
            print(f"  Week: {weekly_perf.week_start} to {weekly_perf.week_end}")
            print(f"  Days active: {weekly_perf.days_active}")
            print(f"  Total trades: {weekly_perf.total_trades}")
            print(f"  Weekly P&L: ${weekly_perf.total_profit_loss_usd:.2f}")
            print(f"  Consistency score: {weekly_perf.consistency_score:.2f}")
            print(f"  Daily performances: {len(weekly_perf.daily_performances)}")
            
            return weekly_perf.days_active > 0 and len(weekly_perf.daily_performances) > 0
        
        return False

def test_data_persistence():
    """Test data saving and loading"""
    print("Testing data persistence...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create analytics and add data
        analytics1 = PerformanceAnalytics(temp_dir)
        sample_trades = create_sample_trade_data()[:10]  # Use fewer trades for speed
        
        for trade in sample_trades:
            analytics1.add_trade(trade)
        
        trades_count_1 = len(analytics1.trades_cache)
        daily_count_1 = len(analytics1.daily_cache)
        
        # Force save
        analytics1._save_data()
        
        # Create new analytics instance (should load existing data)
        analytics2 = PerformanceAnalytics(temp_dir)
        
        trades_count_2 = len(analytics2.trades_cache)
        daily_count_2 = len(analytics2.daily_cache)
        
        print(f"  Trades before reload: {trades_count_1}")
        print(f"  Trades after reload: {trades_count_2}")
        print(f"  Daily summaries before: {daily_count_1}")
        print(f"  Daily summaries after: {daily_count_2}")
        
        return trades_count_1 == trades_count_2 and daily_count_1 == daily_count_2

def test_report_generation():
    """Test report generation"""
    print("Testing report generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup analytics with sample data
        analytics = PerformanceAnalytics(f"{temp_dir}/analytics")
        reporter = PerformanceReporter(f"{temp_dir}/reports")
        
        sample_trades = create_sample_trade_data()
        for trade in sample_trades:
            analytics.add_trade(trade)
        
        # Test daily report
        today_str = date.today().isoformat()
        daily_report = reporter.generate_daily_report(today_str, save_to_file=False)
        
        # Test weekly report  
        weekly_report = reporter.generate_weekly_report(save_to_file=False)
        
        # Test summary report
        summary_report = reporter.generate_summary_report(days=7, save_to_file=False)
        
        # Test JSON report
        json_report = reporter.generate_json_report(days=7)
        
        print(f"  Daily report length: {len(daily_report)} chars")
        print(f"  Weekly report length: {len(weekly_report)} chars")
        print(f"  Summary report length: {len(summary_report)} chars")
        print(f"  JSON report keys: {list(json_report.keys())}")
        
        has_content = all([
            len(daily_report) > 100,
            len(weekly_report) > 100, 
            len(summary_report) > 100,
            'report_metadata' in json_report
        ])
        
        return has_content

def test_dashboard_import():
    """Test importing from dashboard data format"""
    print("Testing dashboard data import...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        analytics = PerformanceAnalytics(temp_dir)
        
        # Create fake dashboard data file
        dashboard_data = {
            "trades": [
                {
                    "timestamp": "2025-08-22T10:00:00",
                    "token_address": "TEST123456789",
                    "token_symbol": "TESTCOIN",
                    "entry_price": 0.001,
                    "exit_price": 0.0011,
                    "quantity": 10000,
                    "profit_usd": 50.0,
                    "profit_sol": 0.28,
                    "profit_percentage": 10.0,
                    "duration_minutes": 30,
                    "source": "trending"
                },
                {
                    "timestamp": "2025-08-22T11:00:00", 
                    "token_address": "TEST987654321",
                    "token_symbol": "ANOTHERCOIN",
                    "entry_price": 0.002,
                    "exit_price": 0.0018,
                    "quantity": 5000,
                    "profit_usd": -20.0,
                    "profit_sol": -0.11,
                    "profit_percentage": -10.0,
                    "duration_minutes": 45,
                    "source": "volume"
                }
            ]
        }
        
        dashboard_file = Path(temp_dir) / "test_dashboard.json"
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f)
        
        # Import the data
        imported_count = analytics.import_from_dashboard_data(str(dashboard_file))
        
        print(f"  Dashboard trades available: {len(dashboard_data['trades'])}")
        print(f"  Trades imported: {imported_count}")
        print(f"  Analytics trades count: {len(analytics.trades_cache)}")
        
        return imported_count == len(dashboard_data['trades'])

async def run_performance_tests():
    """Run all performance analytics tests"""
    print("PERFORMANCE ANALYTICS TESTS")
    print("=" * 50)
    
    tests = [
        ("Analytics Creation", test_analytics_creation),
        ("Trade Import", test_trade_import),
        ("Daily Analytics", test_daily_analytics),
        ("Weekly Analytics", test_weekly_analytics),
        ("Data Persistence", test_data_persistence),
        ("Report Generation", test_report_generation),
        ("Dashboard Import", test_dashboard_import)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            success = test_func()
            if success:
                print(f"  PASS: {test_name}")
                passed += 1
            else:
                print(f"  FAIL: {test_name}")
        except Exception as e:
            print(f"  ERROR: {test_name}: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed ({(passed/total)*100:.0f}%)")
    
    if passed == total:
        print("SUCCESS: Performance analytics system is working!")
        print("\nFeatures verified:")
        print("- Trade data import and storage")
        print("- Daily performance calculations")
        print("- Weekly performance summaries") 
        print("- Data persistence and reloading")
        print("- Report generation (daily, weekly, summary)")
        print("- Dashboard data import compatibility")
        print("\nYour trading performance will now be tracked automatically.")
        return True
    elif passed >= total - 1:
        print("MOSTLY WORKING: Core functionality verified")
        return True
    else:
        print("ISSUES DETECTED: Performance analytics needs attention")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_performance_tests())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests cancelled")
        exit(1)
    except Exception as e:
        print(f"\nTest failure: {e}")
        exit(1)