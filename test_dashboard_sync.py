#!/usr/bin/env python3
"""
Test dashboard data synchronization
"""
import asyncio
import sys
import json
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path.cwd()))

from src.config.settings import load_settings
from src.enhanced_token_scanner import EnhancedTokenScanner
from src.phantom_wallet import PhantomWallet
from src.trading.strategy import TradingStrategy

async def test_dashboard_sync():
    """Test that dashboard syncs with real trades"""
    print("DASHBOARD SYNCHRONIZATION TEST")
    print("=" * 50)
    
    # Check current dashboard state
    try:
        with open("dashboard_data.json", "r") as f:
            before_data = json.load(f)
        print(f"BEFORE: {len(before_data.get('trades', []))} trades in dashboard")
        print(f"BEFORE: Balance = {before_data.get('performance', {}).get('balance', 0):.6f} SOL")
    except:
        print("BEFORE: No dashboard data file found")
        before_data = {"trades": [], "performance": {"balance": 100.0}}
    
    # Execute a test trade
    settings = load_settings()
    scanner = EnhancedTokenScanner(settings)
    wallet = PhantomWallet(settings)
    strategy = TradingStrategy(scanner, wallet, settings)
    
    try:
        await scanner.api_client.start_session()
        
        # Get a token
        token = await scanner.scan_for_new_tokens()
        if not token:
            print("ERROR: No tokens found for test")
            return False
            
        print(f"\nExecuting test trade: {token['symbol']} @ ${token['price']:.8f}")
        
        # Execute trade
        success = await strategy._execute_paper_trade(
            token['address'],
            1.0,  # Small 1 token trade
            token['price']
        )
        
        if not success:
            print("ERROR: Trade execution failed")
            return False
            
        print("Trade executed successfully")
        
        # Check dashboard was updated
        with open("dashboard_data.json", "r") as f:
            after_data = json.load(f)
            
        print(f"\nAFTER: {len(after_data.get('trades', []))} trades in dashboard")
        print(f"AFTER: Balance = {after_data.get('performance', {}).get('balance', 0):.6f} SOL")
        
        # Verify the change
        new_trades = len(after_data.get('trades', [])) - len(before_data.get('trades', []))
        balance_change = before_data.get('performance', {}).get('balance', 100) - after_data.get('performance', {}).get('balance', 100)
        
        if new_trades > 0 and balance_change > 0:
            print(f"\nSUCCESS: Dashboard synchronized!")
            print(f"  New trades: +{new_trades}")
            print(f"  Balance change: -{balance_change:.6f} SOL")
            
            # Show latest trade
            latest_trade = after_data['trades'][-1]
            print(f"  Latest trade: {latest_trade.get('size')} tokens @ ${latest_trade.get('price'):.8f}")
            return True
        else:
            print(f"\nFAILED: Dashboard not synchronized")
            print(f"  New trades: +{new_trades}")
            print(f"  Balance change: -{balance_change:.6f} SOL")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        await scanner.stop()

if __name__ == "__main__":
    result = asyncio.run(test_dashboard_sync())
    print(f"\nDashboard Sync Test: {'PASS' if result else 'FAIL'}")