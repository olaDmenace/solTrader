#!/usr/bin/env python3
"""
Test the monitoring system fix for portfolio value calculation
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

async def test_monitoring_fix():
    """Test that the monitoring system now properly calculates portfolio value"""
    logger.info("Testing monitoring system portfolio value fix...")
    
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
    
    # Check portfolio value calculation
    portfolio_value = await strategy.monitoring._get_portfolio_value()
    logger.info(f"Portfolio value: {portfolio_value} SOL")
    
    # Verify it's above the minimum threshold
    min_threshold = settings.MIN_PORTFOLIO_VALUE
    logger.info(f"Minimum threshold: {min_threshold} SOL")
    
    if portfolio_value > min_threshold:
        logger.info("✅ Portfolio value is above threshold - trades should execute!")
        result = "PASS"
    else:
        logger.error("❌ Portfolio value still below threshold - problem not fixed")
        result = "FAIL"
    
    # Test alert system
    await strategy.monitoring._check_alerts()
    recent_alerts = strategy.monitoring.alerts[-3:] if strategy.monitoring.alerts else []
    
    portfolio_alerts = [alert for alert in recent_alerts if "portfolio_value" in alert.message.lower()]
    if portfolio_alerts:
        logger.warning(f"⚠️ Still getting portfolio alerts: {len(portfolio_alerts)}")
        for alert in portfolio_alerts:
            logger.warning(f"   Alert: {alert.message}")
    else:
        logger.info("✅ No portfolio value alerts - fix successful!")
    
    # Cleanup
    await scanner.stop()
    await jupiter.close()
    await alchemy.close()
    
    logger.info(f"Test result: {result}")
    return result == "PASS"

if __name__ == "__main__":
    result = asyncio.run(test_monitoring_fix())
    if result:
        print("\n🎉 MONITORING FIX SUCCESSFUL! System should now execute trades.")
    else:
        print("\n🚨 MONITORING FIX FAILED! Additional debugging needed.")