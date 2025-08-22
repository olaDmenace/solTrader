#!/usr/bin/env python3
"""
Test Currency Conversion in Trading System
Verify USD token prices are properly converted to SOL costs
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_currency_conversion():
    """Test that USD token prices are properly converted to SOL costs"""
    logger.info("Testing currency conversion in paper trading...")
    
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
    
    # Get current SOL price
    sol_price_usd = await jupiter.get_sol_price()
    logger.info(f"Current SOL price: ${sol_price_usd:.2f} USD")
    
    initial_balance = strategy.state.paper_balance
    logger.info(f"Initial paper balance: {initial_balance:.4f} SOL")
    
    # Test trade with a small USD token
    token_price_usd = 0.000123  # $0.000123 USD
    token_size = 1000  # 1000 tokens
    
    logger.info(f"\nTest Trade Parameters:")
    logger.info(f"Token price: ${token_price_usd:.6f} USD")
    logger.info(f"Token size: {token_size} tokens")
    logger.info(f"Total USD value: ${token_price_usd * token_size:.4f}")
    
    # Calculate expected SOL cost
    expected_sol_cost = (token_price_usd * token_size) / sol_price_usd
    logger.info(f"Expected SOL cost: {expected_sol_cost:.6f} SOL")
    
    # Execute paper trade
    logger.info(f"\nExecuting paper trade...")
    success = await strategy._execute_paper_trade(
        token_address="TEST123456789abcdef",
        size=token_size, 
        price=token_price_usd
    )
    
    final_balance = strategy.state.paper_balance
    actual_cost = initial_balance - final_balance
    
    logger.info(f"\nResults:")
    logger.info(f"Trade successful: {success}")
    logger.info(f"Initial balance: {initial_balance:.6f} SOL")
    logger.info(f"Final balance: {final_balance:.6f} SOL")
    logger.info(f"Actual cost: {actual_cost:.6f} SOL")
    logger.info(f"Expected cost: {expected_sol_cost:.6f} SOL")
    logger.info(f"Difference: {abs(actual_cost - expected_sol_cost):.8f} SOL")
    
    # Verify conversion accuracy
    tolerance = 0.000001  # 1 micro-SOL tolerance
    if abs(actual_cost - expected_sol_cost) < tolerance:
        logger.info("✅ Currency conversion ACCURATE - USD to SOL conversion working correctly")
        result = "PASS"
    else:
        logger.error("❌ Currency conversion INACCURATE - USD to SOL conversion has errors")
        result = "FAIL"
    
    # Check position tracking
    if strategy.state.paper_positions:
        position = list(strategy.state.paper_positions.values())[0]
        logger.info(f"\nPosition Details:")
        logger.info(f"Position entry price: ${position.entry_price:.6f} USD")
        logger.info(f"Position size: {position.size} tokens")
        logger.info(f"Position value: ${position.entry_price * position.size:.4f} USD")
    
    # Cleanup
    await scanner.stop()
    await jupiter.close()
    await alchemy.close()
    
    return result == "PASS"

if __name__ == "__main__":
    result = asyncio.run(test_currency_conversion())
    if result:
        print("\n🎉 CURRENCY CONVERSION TEST PASSED!")
        print("USD token prices are properly converted to SOL costs for balance management.")
    else:
        print("\n🚨 CURRENCY CONVERSION TEST FAILED!")
        print("USD to SOL conversion needs debugging.")