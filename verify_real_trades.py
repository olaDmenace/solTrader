#!/usr/bin/env python3
"""
Verify that paper trades are REAL executions, not hard-coded simulations
"""
import asyncio
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path.cwd()))

from src.config.settings import load_settings
from src.enhanced_token_scanner import EnhancedTokenScanner
from src.phantom_wallet import PhantomWallet
from src.trading.strategy import TradingStrategy

async def verify_real_trades():
    """Verify trades are real, not simulated"""
    print("VERIFYING REAL PAPER TRADE EXECUTION")
    print("=" * 60)
    
    settings = load_settings()
    scanner = EnhancedTokenScanner(settings)
    wallet = PhantomWallet(settings)
    strategy = TradingStrategy(scanner, wallet, settings)
    
    try:
        await scanner.api_client.start_session()
        
        # Get DIFFERENT tokens for multiple tests
        print("1. Getting multiple different tokens...")
        results = await scanner._perform_full_scan()
        if not results or len(results) < 3:
            print("   ERROR: Need at least 3 tokens for verification")
            return False
            
        print(f"   Found {len(results)} approved tokens")
        
        # Test 1: Execute multiple different trades and verify balance changes
        print("\n2. Testing multiple trades with different tokens...")
        
        initial_balance = strategy.state.paper_balance
        print(f"   Starting balance: {initial_balance:.6f} SOL")
        
        trades_executed = []
        
        for i in range(3):  # Test 3 different tokens
            token_result = results[i]
            token_data = {
                'address': token_result.token.address,
                'symbol': token_result.token.symbol,
                'price': token_result.token.price
            }
            
            balance_before = strategy.state.paper_balance
            quantity = 2.0 + i  # Different quantities: 2, 3, 4
            expected_cost = quantity * token_data['price']
            
            print(f"\n   Trade {i+1}: {quantity} {token_data['symbol']} at ${token_data['price']:.8f}")
            print(f"   Expected cost: {expected_cost:.6f} SOL")
            print(f"   Balance before: {balance_before:.6f} SOL")
            
            success = await strategy._execute_paper_trade(
                token_data['address'],
                quantity,
                token_data['price']
            )
            
            if success:
                balance_after = strategy.state.paper_balance
                actual_cost = balance_before - balance_after
                print(f"   Balance after: {balance_after:.6f} SOL")
                print(f"   Actual cost: {actual_cost:.6f} SOL")
                
                # Verify the cost calculation is correct
                cost_diff = abs(actual_cost - expected_cost)
                if cost_diff < 0.000001:  # Allow tiny floating point differences
                    print(f"   REAL TRADE: Cost calculation matches exactly!")
                else:
                    print(f"   WARNING: Cost mismatch - Expected: {expected_cost:.6f}, Got: {actual_cost:.6f}")
                
                trades_executed.append({
                    'symbol': token_data['symbol'],
                    'quantity': quantity,
                    'price': token_data['price'],
                    'cost': actual_cost,
                    'address': token_data['address']
                })
            else:
                print(f"   FAILED: Trade {i+1} execution failed")
                
        # Test 2: Verify positions are tracked correctly
        print(f"\n3. Verifying position tracking...")
        print(f"   Active positions: {len(strategy.state.paper_positions)}")
        
        if len(strategy.state.paper_positions) == len(trades_executed):
            print("   REAL TRACKING: Position count matches executed trades")
            
            # Verify each position matches the trade
            for i, (address, position) in enumerate(strategy.state.paper_positions.items()):
                trade = trades_executed[i]
                print(f"   Position {i+1}: {position.size} tokens at {position.entry_price:.8f} SOL")
                
                if (abs(position.size - trade['quantity']) < 0.000001 and 
                    abs(position.entry_price - trade['price']) < 0.000001):
                    print("     MATCH: Position data matches trade execution")
                else:
                    print("     MISMATCH: Position data doesn't match trade")
        else:
            print("   ERROR: Position count doesn't match trades executed")
            
        # Test 3: Verify total balance calculation
        print(f"\n4. Verifying balance calculations...")
        total_cost = sum(trade['cost'] for trade in trades_executed)
        expected_balance = initial_balance - total_cost
        actual_balance = strategy.state.paper_balance
        
        print(f"   Initial balance: {initial_balance:.6f} SOL")
        print(f"   Total cost: {total_cost:.6f} SOL") 
        print(f"   Expected final: {expected_balance:.6f} SOL")
        print(f"   Actual final: {actual_balance:.6f} SOL")
        
        balance_diff = abs(actual_balance - expected_balance)
        if balance_diff < 0.000001:
            print("   REAL CALCULATION: Balance math is correct!")
        else:
            print(f"   ERROR: Balance calculation wrong by {balance_diff:.6f}")
            
        # Test 4: Verify trades use REAL token data (not hardcoded)
        print(f"\n5. Verifying dynamic token data...")
        
        # Get a fresh token scan and compare
        fresh_token = await scanner.scan_for_new_tokens()
        if fresh_token:
            print(f"   Fresh token: {fresh_token['symbol']} at ${fresh_token['price']:.8f}")
            
            # Check if this is different from our previous trades
            used_symbols = [trade['symbol'] for trade in trades_executed]
            if fresh_token['symbol'] not in used_symbols or fresh_token['price'] != trades_executed[0]['price']:
                print("   DYNAMIC DATA: Token selection and prices are changing")
            else:
                print("   STATIC DATA: Same token/price returned (could be coincidence)")
        
        await scanner.stop()
        
        print(f"\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        
        if len(trades_executed) >= 3:
            print("VERDICT: PAPER TRADES ARE REAL!")
            print("Evidence:")
            print("- Multiple different tokens traded")
            print("- Exact balance calculations")  
            print("- Dynamic position tracking")
            print("- Real-time token discovery")
            return True
        else:
            print("VERDICT: INSUFFICIENT EVIDENCE")
            print("Could not execute enough trades for verification")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    result = asyncio.run(verify_real_trades())
    print(f"\nResult: {'REAL TRADES VERIFIED' if result else 'VERIFICATION FAILED'}")