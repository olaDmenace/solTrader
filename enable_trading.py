#!/usr/bin/env python3
"""
Trading Control Script
Use this to enable/disable trading after fixing the scanner
"""
import sys
from pathlib import Path

def toggle_trading(enable: bool = True):
    """Enable or disable trading"""
    settings_file = Path("src/config/settings.py")
    
    if not settings_file.exists():
        print("❌ Settings file not found!")
        return False
    
    # Read current settings
    with open(settings_file, 'r') as f:
        content = f.read()
    
    # Update trading pause setting
    if enable:
        new_content = content.replace(
            "TRADING_PAUSED: bool = True  # Pause trading until scanner is fixed",
            "TRADING_PAUSED: bool = False  # Trading enabled with new scanner"
        )
        status = "ENABLED"
        emoji = "✅"
    else:
        new_content = content.replace(
            "TRADING_PAUSED: bool = False  # Trading enabled with new scanner",
            "TRADING_PAUSED: bool = True  # Pause trading until scanner is fixed"
        )
        status = "DISABLED"
        emoji = "🛑"
    
    # Write updated settings
    with open(settings_file, 'w') as f:
        f.write(new_content)
    
    print(f"{emoji} Trading {status}")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['enable', 'on', 'true', '1']:
            toggle_trading(True)
        elif arg in ['disable', 'off', 'false', '0']:
            toggle_trading(False)
        else:
            print("Usage: python enable_trading.py [enable|disable]")
    else:
        print("Current status: Check TRADING_PAUSED in src/config/settings.py")
        print("Usage: python enable_trading.py [enable|disable]")