#!/usr/bin/env python3
"""
Test Price Fetching Methods
Find out why price fetching is failing during trade execution
"""
import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config.settings import load_settings
from src.api.jupiter import JupiterClient
from src.phantom_wallet import PhantomWallet
from src.api.alchemy import AlchemyClient
from src.trading.strategy import TradingStrategy, TradingMode
from src.enhanced_token_scanner import EnhancedTokenScanner
from src.analytics.performance_analytics import PerformanceAnalytics

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_price_fetching_methods():
    """Test why price fetching is failing during trade execution"""
    logger.info("TESTING PRICE FETCHING METHODS")
    logger.info("=" * 50)
    
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
    
    # Test tokens that we know are generating signals
    test_tokens = [
        "12Z5CsL5kCPRmnLQcwpY2QrEub3MahHCtRHxoQEdpump",  # HODLess - from recent logs
        "TEST_EXECUTION_12345",  # Our test token
        "So11111111111111111111111111111111111111112"  # SOL - should always work
    ]
    
    for token_address in test_tokens:
        logger.info(f"\n🔍 TESTING PRICE FETCH: {token_address[:12]}...")
        
        # Test the strategy's price fetching method
        try:
            current_price = await strategy._get_current_price(token_address)
            if current_price:
                logger.info(f"✅ Strategy price fetch: ${current_price:.8f}")
            else:
                logger.error(f"❌ Strategy price fetch: FAILED")
        except Exception as e:
            logger.error(f"❌ Strategy price fetch: EXCEPTION - {e}")
        
        # Test Jupiter directly
        try:
            quote = await jupiter.get_quote(
                token_address,
                "So11111111111111111111111111111111111111112",  # SOL
                "1000000000",  # 1 token (assuming 9 decimals)
                50  # 0.5% slippage
            )
            if quote and "outAmount" in quote:
                out_amount = int(quote["outAmount"]) / 1e9  # Convert lamports to SOL
                logger.info(f"✅ Direct Jupiter quote: {out_amount:.8f} SOL")
            else:
                logger.error(f"❌ Direct Jupiter quote: NO DATA")
        except Exception as e:
            logger.error(f"❌ Direct Jupiter quote: EXCEPTION - {e}")
        
        # Test market depth
        try:
            market_depth = await jupiter.get_market_depth(token_address)
            if market_depth and "price" in market_depth:
                logger.info(f"✅ Market depth: ${market_depth['price']:.8f}")
            else:
                logger.error(f"❌ Market depth: NO PRICE DATA")
                if market_depth:
                    logger.error(f"   Available keys: {list(market_depth.keys())}")
        except Exception as e:
            logger.error(f"❌ Market depth: EXCEPTION - {e}")
    
    # Cleanup
    await scanner.stop()
    await jupiter.close()
    await alchemy.close()
    
    logger.info(f"\n💡 ANALYSIS:")
    logger.info(f"If SOL price fetching works but token prices fail,")
    logger.info(f"the issue is that we're trying to fetch prices for tokens")
    logger.info(f"that don't exist or aren't tradeable on Jupiter/DEX.")

if __name__ == "__main__":
    asyncio.run(test_price_fetching_methods())