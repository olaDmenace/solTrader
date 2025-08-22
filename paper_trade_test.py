#!/usr/bin/env python3
"""
Test actual paper trade execution
"""
import asyncio
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path.cwd()))

from src.config.settings import load_settings
from src.enhanced_token_scanner import EnhancedTokenScanner
from src.phantom_wallet import PhantomWallet
from src.trading.strategy import TradingStrategy, TradingMode

async def test_paper_trade_execution():
    """Test complete paper trade execution"""
    print("=== PAPER TRADE EXECUTION TEST ===")
    
    settings = load_settings()
    print(f"Paper Trading: {settings.PAPER_TRADING}")
    print(f"Initial Balance: {settings.INITIAL_PAPER_BALANCE} SOL")
    
    try:
        # Initialize components
        print("\n1. Initializing components...")
        scanner = EnhancedTokenScanner(settings)
        wallet = PhantomWallet(settings)
        
        # Initialize strategy
        strategy = TradingStrategy(scanner, wallet, settings)
        print("   Strategy initialized")
        
        # Test token discovery
        print("\n2. Testing token discovery...")
        token = await scanner.scan_for_new_tokens()
        if not token:
            print("   ERROR: No tokens discovered")
            return False
        
        print(f"   Token discovered: {token['symbol']} at ${token['price']:.8f}")
        print(f"   Address: {token['address'][:8]}...")
        
        # Check initial paper balance
        print(f"\n3. Initial paper balance: {strategy.state.paper_balance:.6f} SOL")
        
        # Attempt paper trade
        print("\n4. Attempting paper trade...")
        success = await strategy._execute_paper_trade(
            token['address'], 
            10.0,  # 10 token quantity
            token['price']
        )
        
        if success:
            print("   SUCCESS: Paper trade executed!")
            print(f"   New balance: {strategy.state.paper_balance:.6f} SOL")
            print(f"   Active positions: {len(strategy.state.paper_positions)}")
            
            # Show position details
            if strategy.state.paper_positions:
                pos = list(strategy.state.paper_positions.values())[0]
                print(f"   Position: {pos.size} tokens at {pos.entry_price:.8f} SOL each")
        else:
            print("   FAILED: Paper trade execution failed")
        
        return success
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    result = asyncio.run(test_paper_trade_execution())
    print(f"\nPaper Trade Test: {'PASS' if result else 'FAIL'}")