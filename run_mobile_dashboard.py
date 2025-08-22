#!/usr/bin/env python3
"""
Mobile Dashboard Launcher
Quick script to start the mobile-responsive SolTrader dashboard
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from src.dashboard.mobile_dashboard import MobileDashboard
    
    print("SolTrader Mobile Dashboard Launcher")
    print("=" * 50)
    
    # Create and run mobile dashboard
    dashboard = MobileDashboard()
    dashboard.run(host='0.0.0.0', port=5001, debug=False)
    
except ImportError as e:
    print(f"Import error: {e}")
    print("\nMake sure Flask is installed:")
    print("   pip install flask")
except KeyboardInterrupt:
    print("\nDashboard stopped by user")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()