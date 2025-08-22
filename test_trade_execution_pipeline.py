#!/usr/bin/env python3
"""
DIRECT TEST: Trade Execution Pipeline
Test if trades actually execute or just get stuck in pending orders
"""
import asyncio
import logging
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config.settings import load_settings
from src.api.jupiter import JupiterClient
from src.phantom_wallet import PhantomWallet
from src.api.alchemy import AlchemyClient
from src.trading.strategy import TradingStrategy, TradingMode, EntrySignal
from src.enhanced_token_scanner import EnhancedTokenScanner
from src.analytics.performance_analytics import PerformanceAnalytics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_direct_execution_pipeline():
    """Test if the execution pipeline actually executes trades"""
    logger.info("🔬 TESTING TRADE EXECUTION PIPELINE")
    logger.info("=" * 60)
    
    # Initialize components
    settings = load_settings()
    alchemy = AlchemyClient(settings.ALCHEMY_RPC_URL)
    jupiter = JupiterClient()
    wallet = PhantomWallet(alchemy)
    analytics = PerformanceAnalytics(settings)
    scanner = EnhancedTokenScanner(settings, analytics)
    
    # Create strategy in PAPER mode
    strategy = TradingStrategy(
        jupiter_client=jupiter,
        wallet=wallet,
        settings=settings,
        scanner=scanner,
        mode=TradingMode.PAPER
    )
    
    initial_balance = strategy.state.paper_balance
    initial_positions = len(strategy.state.paper_positions)
    initial_pending = len(strategy.state.pending_orders)
    
    logger.info(f"INITIAL STATE:")
    logger.info(f"  Balance: {initial_balance:.4f} SOL")
    logger.info(f"  Positions: {initial_positions}")
    logger.info(f"  Pending Orders: {initial_pending}")
    
    # Create a mock pending order directly
    test_signal = EntrySignal(
        token_address="TEST_EXECUTION_12345",
        price=0.001234,  # $0.001234 USD
        confidence=0.85,
        entry_type="test",
        size=100,  # 100 tokens  
        stop_loss=0.001234 * 0.8,  # 20% stop loss
        take_profit=0.001234 * 1.3,  # 30% take profit
        slippage=0.1,
        timestamp=datetime.now()
    )
    
    # Add to pending orders
    strategy.state.pending_orders.append(test_signal)
    logger.info(f"ADDED TEST SIGNAL to pending orders")
    logger.info(f"  Token: {test_signal.token_address}")
    logger.info(f"  Price: ${test_signal.price:.6f} USD")
    logger.info(f"  Size: {test_signal.size} tokens")
    logger.info(f"  Pending Orders: {len(strategy.state.pending_orders)}")
    
    # DIRECT TEST: Call _process_pending_orders
    logger.info(f"\n🚀 EXECUTING _process_pending_orders()...")
    try:
        await strategy._process_pending_orders()
        logger.info("✅ _process_pending_orders() completed without exceptions")
    except Exception as e:
        logger.error(f"❌ _process_pending_orders() FAILED: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Check results
    final_balance = strategy.state.paper_balance
    final_positions = len(strategy.state.paper_positions)
    final_pending = len(strategy.state.pending_orders)
    
    logger.info(f"\nFINAL STATE:")
    logger.info(f"  Balance: {final_balance:.4f} SOL (Change: {final_balance - initial_balance:.6f})")
    logger.info(f"  Positions: {final_positions} (Change: +{final_positions - initial_positions})")
    logger.info(f"  Pending Orders: {final_pending} (Change: {final_pending - initial_pending})")
    
    # Determine if execution worked
    balance_changed = abs(final_balance - initial_balance) > 0.000001
    positions_added = final_positions > initial_positions
    pending_cleared = final_pending < initial_pending
    
    logger.info(f"\nEXECUTION ANALYSIS:")
    logger.info(f"  Balance Changed: {balance_changed} ({'✅' if balance_changed else '❌'})")
    logger.info(f"  Positions Added: {positions_added} ({'✅' if positions_added else '❌'})")  
    logger.info(f"  Pending Cleared: {pending_cleared} ({'✅' if pending_cleared else '❌'})")
    
    if balance_changed and positions_added and pending_cleared:
        logger.info("🎉 TRADE EXECUTION: WORKING!")
        result = "EXECUTION_WORKING"
    elif pending_cleared and not (balance_changed or positions_added):
        logger.error("🚨 TRADE EXECUTION: ORDERS CLEARED BUT NO TRADES EXECUTED!")
        result = "ORDERS_CLEARED_NO_EXECUTION"
    elif not pending_cleared:
        logger.error("🚨 TRADE EXECUTION: ORDERS NOT EVEN PROCESSED!")
        result = "ORDERS_NOT_PROCESSED"
    else:
        logger.warning("⚠️ TRADE EXECUTION: PARTIAL/UNKNOWN STATE")
        result = "UNKNOWN_STATE"
    
    # Check positions details if any were created
    if final_positions > 0:
        logger.info(f"\nPOSITION DETAILS:")
        for addr, pos in strategy.state.paper_positions.items():
            logger.info(f"  {addr[:12]}... - {pos.size} tokens @ ${pos.entry_price:.6f}")
    
    # Cleanup
    await scanner.stop()
    await jupiter.close() 
    await alchemy.close()
    
    return result

if __name__ == "__main__":
    result = asyncio.run(test_direct_execution_pipeline())
    print(f"\n🔍 EXECUTION TEST RESULT: {result}")
    
    if result == "EXECUTION_WORKING":
        print("✅ Trade execution pipeline is working correctly!")
    elif result == "ORDERS_CLEARED_NO_EXECUTION":
        print("❌ Orders are being cleared but trades are not executing!")
        print("   → Issue is in the trade execution logic")
    elif result == "ORDERS_NOT_PROCESSED": 
        print("❌ Pending orders are not even being processed!")
        print("   → Issue is in the order processing loop")
    else:
        print("⚠️ Unknown execution state - needs deeper investigation")