#!/usr/bin/env python3
"""
Test script to isolate pending order processing issue
"""
import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config.settings import Settings, load_settings
from src.api.alchemy import AlchemyClient
from src.api.jupiter import JupiterClient
from src.phantom_wallet import PhantomWallet
from src.trading.strategy import TradingStrategy, TradingMode
from src.enhanced_token_scanner import EnhancedTokenScanner
from src.analytics.performance_analytics import PerformanceAnalytics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_pending_order_processing():
    """Test pending order processing in isolation"""
    try:
        logger.info("Loading settings...")
        settings = load_settings()
        
        logger.info("Initializing components...")
        alchemy = AlchemyClient(settings.ALCHEMY_RPC_URL)
        jupiter = JupiterClient()
        wallet = PhantomWallet(alchemy)
        analytics = PerformanceAnalytics(settings)
        scanner = EnhancedTokenScanner(settings, analytics)
        
        logger.info("Initializing trading strategy...")
        strategy = TradingStrategy(
            jupiter_client=jupiter,
            wallet=wallet,
            settings=settings,
            scanner=scanner,
            mode=TradingMode.PAPER
        )
        
        # Check if there are pending orders
        logger.info(f"Current pending orders: {len(strategy.state.pending_orders)}")
        
        if strategy.state.pending_orders:
            logger.info("Found pending orders - processing them...")
            logger.info(f"Pending orders: {strategy.state.pending_orders}")
            
            # Try to process pending orders
            await strategy._process_pending_orders()
            
            logger.info(f"After processing - remaining orders: {len(strategy.state.pending_orders)}")
        else:
            logger.info("No pending orders found")
            
        # Close connections
        await scanner.stop()
        await jupiter.close()
        await alchemy.close()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(test_pending_order_processing())