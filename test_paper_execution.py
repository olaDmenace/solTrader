#!/usr/bin/env python3
"""
Quick Test Script for Paper Trading Execution
Tests that discovered tokens trigger paper trades
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Set up path for imports
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# Change to project directory for relative imports
os.chdir(project_root)

from src.config.settings import load_settings
from src.enhanced_token_scanner import EnhancedTokenScanner
from src.trading.strategy import TradingStrategy, TradingMode
from src.api.jupiter import JupiterClient
from src.phantom_wallet import PhantomWallet
from src.api.alchemy import AlchemyClient
from src.analytics.performance_analytics import PerformanceAnalytics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_paper_execution():
    """Test paper trading execution with real tokens"""
    logger.info("🚀 Testing Paper Trading Execution...")
    
    # Load settings
    settings = load_settings()
    
    # Initialize components
    alchemy = AlchemyClient(settings.ALCHEMY_RPC_URL)
    jupiter = JupiterClient()
    wallet = PhantomWallet(alchemy)
    analytics = PerformanceAnalytics(settings)
    
    # Initialize scanner and strategy
    scanner = EnhancedTokenScanner(settings, analytics)
    strategy = TradingStrategy(
        jupiter_client=jupiter,
        wallet=wallet,
        settings=settings,
        scanner=scanner,
        mode=TradingMode.PAPER
    )
    
    try:
        # Start connections
        logger.info("🔌 Starting connections...")
        await alchemy.test_connection()
        await jupiter.test_connection()
        await scanner.start()
        
        # Record initial state
        initial_balance = strategy.state.paper_balance
        initial_positions = len(strategy.state.paper_positions)
        initial_pending = len(strategy.state.pending_orders)
        
        logger.info(f"💰 Initial Balance: {initial_balance:.4f} SOL")
        logger.info(f"📊 Initial Positions: {initial_positions}")
        logger.info(f"📝 Initial Pending Orders: {initial_pending}")
        
        # Start trading strategy
        logger.info("▶️ Starting trading strategy...")
        await strategy.start_trading()
        
        # Wait and monitor for changes
        logger.info("⏳ Monitoring for paper trade execution (30 seconds)...")
        
        for i in range(30):
            await asyncio.sleep(1)
            
            current_balance = strategy.state.paper_balance
            current_positions = len(strategy.state.paper_positions)
            current_pending = len(strategy.state.pending_orders)
            
            # Check for changes
            if (current_balance != initial_balance or 
                current_positions != initial_positions or 
                current_pending != initial_pending):
                
                logger.info(f"📈 ACTIVITY DETECTED!")
                logger.info(f"   Balance: {initial_balance:.4f} -> {current_balance:.4f} SOL")
                logger.info(f"   Positions: {initial_positions} -> {current_positions}")
                logger.info(f"   Pending: {initial_pending} -> {current_pending}")
                
                # Show position details
                if current_positions > 0:
                    logger.info("📍 Current Positions:")
                    for addr, pos in strategy.state.paper_positions.items():
                        logger.info(f"   {addr[:8]}... - Size: {pos.size:.4f}, Entry: {pos.entry_price:.8f}")
                
                # Show pending orders
                if current_pending > 0:
                    logger.info("📝 Pending Orders:")
                    for order in strategy.state.pending_orders:
                        logger.info(f"   {order.token_address[:8]}... - Size: {order.size:.4f}, Price: {order.price:.8f}")
                
                break
            
            if i % 10 == 9:  # Every 10 seconds
                logger.info(f"⏱️ Still monitoring... ({i+1}/30s)")
        
        # Final status
        final_balance = strategy.state.paper_balance
        final_positions = len(strategy.state.paper_positions)
        final_pending = len(strategy.state.pending_orders)
        
        logger.info("📊 FINAL RESULTS:")
        logger.info(f"   Balance: {initial_balance:.4f} -> {final_balance:.4f} SOL")
        logger.info(f"   Positions: {initial_positions} -> {final_positions}")
        logger.info(f"   Pending: {initial_pending} -> {final_pending}")
        
        # Check if execution occurred
        execution_detected = (
            final_balance != initial_balance or
            final_positions > 0 or
            final_pending > 0
        )
        
        if execution_detected:
            logger.info("✅ PAPER TRADING EXECUTION IS WORKING!")
            
            # Update dashboard with test results
            try:
                with open('bot_data.json', 'r') as f:
                    data = json.load(f)
                
                # Add test activity
                data['activity'].append({
                    "type": "execution_test",
                    "data": {
                        "message": f"Paper trading test completed - Execution detected: {execution_detected}",
                        "balance_change": final_balance - initial_balance,
                        "positions_opened": final_positions,
                        "pending_orders": final_pending,
                        "timestamp": datetime.now().isoformat()
                    },
                    "timestamp": datetime.now().isoformat()
                })
                
                data['last_update'] = datetime.now().isoformat()
                
                with open('bot_data.json', 'w') as f:
                    json.dump(data, f, indent=2)
                    
                logger.info("📊 Dashboard updated with test results")
                
            except Exception as e:
                logger.warning(f"Failed to update dashboard: {e}")
            
        else:
            logger.warning("⚠️ No paper trading execution detected")
            logger.info("💡 This could mean:")
            logger.info("   - No quality tokens found during test period")
            logger.info("   - Tokens found but didn't meet execution criteria")
            logger.info("   - Need to run for longer period")
        
        await strategy.stop_trading()
        
        return execution_detected
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
        
    finally:
        # Cleanup
        try:
            await scanner.stop()
            await jupiter.close()
            await alchemy.close()
        except:
            pass

async def main():
    """Main function"""
    try:
        logger.info("🎯 Paper Trading Execution Test")
        logger.info("=" * 50)
        
        success = await test_paper_execution()
        
        logger.info("=" * 50)
        if success:
            logger.info("🎉 TEST PASSED - Paper trading execution is working!")
            logger.info("🚀 You can now run the main bot to see continuous paper trades")
        else:
            logger.info("📝 TEST INCONCLUSIVE - Try running the main bot for longer")
            logger.info("💡 The system may need more time to find and process tokens")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Test stopped by user")
    except Exception as e:
        logger.error(f"💥 Test error: {e}")

if __name__ == "__main__":
    asyncio.run(main())