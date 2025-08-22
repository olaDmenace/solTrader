#!/usr/bin/env python3
"""
Generate Performance Charts
Creates interactive charts dashboard for SolTrader performance analysis
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Generate performance charts dashboard"""
    print("SolTrader Performance Charts Generator")
    print("=" * 50)
    
    try:
        from src.utils.performance_charts import generate_performance_charts
        
        print("Generating performance charts...")
        charts_file = generate_performance_charts()
        
        print(f"Charts dashboard created: {charts_file}")
        print("\nTo view your performance charts:")
        print(f"1. Open file: {charts_file}")
        print("2. Or open in browser: file://" + str(Path(charts_file).absolute()))
        
        print("\nCharts included:")
        print("- Cumulative P&L over time")
        print("- Daily P&L breakdown")
        print("- Rolling win rate analysis")
        print("- Trade P&L distribution")
        print("- Token performance ranking")
        
        return True
        
    except Exception as e:
        print(f"Failed to generate charts: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nChart generation cancelled")
        exit(1)