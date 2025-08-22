#!/usr/bin/env python3
"""
Setup Performance Analytics for SolTrader
Initializes analytics system and imports existing trade data
"""
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.utils.performance_analytics import performance_analytics, import_existing_trades
from src.utils.performance_reports import generate_daily_report, generate_weekly_report, generate_summary_report

def setup_analytics_directories():
    """Create necessary directories for analytics"""
    print("Setting up analytics directories...")
    
    directories = [
        "analytics",
        "analytics/backups", 
        "reports",
        "reports/daily",
        "reports/weekly"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  Created: {directory}/")
    
    return True

def import_dashboard_trades():
    """Import existing trades from dashboard data"""
    print("Importing existing trade data...")
    
    # Check if dashboard_data.json exists
    dashboard_file = Path("dashboard_data.json")
    if not dashboard_file.exists():
        print("  No dashboard_data.json found - skipping import")
        return 0
    
    # Import the trades
    imported_count = import_existing_trades()
    print(f"  Imported {imported_count} trades from dashboard data")
    
    return imported_count

def generate_initial_reports():
    """Generate initial performance reports"""
    print("Generating initial performance reports...")
    
    try:
        # Generate today's report
        today_report = generate_daily_report()
        print(f"  Daily report generated ({len(today_report)} chars)")
        
        # Generate this week's report
        weekly_report = generate_weekly_report()
        print(f"  Weekly report generated ({len(weekly_report)} chars)")
        
        # Generate 30-day summary
        summary_report = generate_summary_report(30)
        print(f"  Summary report generated ({len(summary_report)} chars)")
        
        return True
        
    except Exception as e:
        print(f"  Error generating reports: {e}")
        return False

def show_analytics_status():
    """Display current analytics status"""
    print("Current Analytics Status:")
    print("-" * 30)
    
    # Get analytics statistics
    stats = performance_analytics.get_analytics_summary()
    
    print(f"Total trades tracked: {stats['total_trades_tracked']}")
    print(f"Daily summaries: {stats['daily_summaries_count']}")
    print(f"Weekly summaries: {stats['weekly_summaries_count']}")
    
    if stats['date_range']['first_trade']:
        first_trade = datetime.fromisoformat(stats['date_range']['first_trade'])
        last_trade = datetime.fromisoformat(stats['date_range']['last_trade'])
        print(f"Date range: {first_trade.date()} to {last_trade.date()}")
    
    # Show recent performance
    recent_perf = performance_analytics.get_recent_performance(7)
    print(f"\nRecent 7-day performance:")
    print(f"  Active days: {recent_perf['active_days']}/7")
    print(f"  Total trades: {recent_perf['total_trades']}")
    print(f"  Total P&L: ${recent_perf['total_profit_usd']:.2f}")
    print(f"  Average daily: ${recent_perf['average_daily_profit']:.2f}")
    
    # Show today's performance
    today_perf = performance_analytics.get_daily_summary()
    if today_perf:
        print(f"\nToday's performance:")
        print(f"  Trades: {today_perf.total_trades}")
        print(f"  Win rate: {today_perf.win_rate:.1f}%")
        print(f"  P&L: ${today_perf.total_profit_loss_usd:.2f}")
    else:
        print(f"\nNo trades today yet.")

def setup_analytics_integration():
    """Information about integrating analytics with trading bot"""
    print("\nIntegrating with your trading bot:")
    print("-" * 40)
    
    print("1. Import the analytics system in your trading code:")
    print("   from src.utils.performance_analytics import add_trade_to_analytics")
    print("")
    print("2. Add this line after each completed trade:")
    print("   add_trade_to_analytics({")
    print("       'timestamp': trade_time.isoformat(),")
    print("       'token_address': token_address,")
    print("       'token_symbol': token_symbol,")
    print("       'entry_price': entry_price,")
    print("       'exit_price': exit_price,")
    print("       'quantity': quantity,")
    print("       'profit_loss_usd': profit_usd,")
    print("       'profit_loss_sol': profit_sol,")
    print("       'profit_percentage': profit_percentage,")
    print("       'duration_minutes': duration_minutes,")
    print("       'trade_type': 'buy',  # or 'sell'")
    print("       'source': 'trending'  # or 'volume', 'memescope'")
    print("   })")
    print("")
    print("3. Generate reports anytime with:")
    print("   from src.utils.performance_reports import generate_daily_report")
    print("   report = generate_daily_report()")
    print("   print(report)")

async def main():
    """Main setup function"""
    print("SolTrader Performance Analytics Setup")
    print("=" * 50)
    
    # Setup directories
    setup_analytics_directories()
    print("")
    
    # Import existing trades
    imported_count = import_dashboard_trades()
    print("")
    
    # Generate initial reports if we have data
    if imported_count > 0:
        generate_initial_reports()
        print("")
    
    # Show current status
    show_analytics_status()
    print("")
    
    # Show integration instructions
    setup_analytics_integration()
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("")
    print("Your performance analytics system is now ready.")
    print("Key features:")
    print("  - Automatic daily and weekly P&L summaries")
    print("  - Win rate, Sharpe ratio, and drawdown tracking")
    print("  - Performance reports saved to reports/ directory")
    print("  - JSON data for API integration")
    print("")
    print("Reports generated:")
    print(f"  - reports/daily_report_*.txt")
    print(f"  - reports/weekly_report_*.txt")
    print(f"  - reports/summary_report_*.txt")
    print("")
    print("Data stored in:")
    print(f"  - analytics/trades_history.json")
    print(f"  - analytics/daily_performance.json")
    print(f"  - analytics/weekly_performance.json")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSetup cancelled")
    except Exception as e:
        print(f"\nSetup failed: {e}")
        import traceback
        traceback.print_exc()