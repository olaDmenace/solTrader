#!/usr/bin/env python3
"""
Integration test for paper trading system
Tests the complete flow from token discovery to paper trade execution and dashboard updates
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import Settings, load_settings
from src.enhanced_token_scanner import EnhancedTokenScanner
from src.trading.strategy import TradingStrategy, TradingMode
from src.analytics.performance_analytics import PerformanceAnalytics
from src.api.jupiter import JupiterClient
from src.phantom_wallet import PhantomWallet
from src.api.alchemy import AlchemyClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PaperTradingIntegrationTest:
    def __init__(self):
        """Initialize test components"""
        logger.info("🧪 Initializing Paper Trading Integration Test...")
        
        # Load settings
        try:
            self.settings = load_settings()
            logger.info("✅ Settings loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load settings: {e}")
            raise
        
        # Force paper trading mode
        self.settings.PAPER_TRADING = True
        self.settings.INITIAL_PAPER_BALANCE = 100.0
        
        # Initialize components
        self.alchemy = AlchemyClient(self.settings.ALCHEMY_RPC_URL)
        self.jupiter = JupiterClient()
        self.wallet = PhantomWallet(self.alchemy)
        self.analytics = PerformanceAnalytics(self.settings)
        self.scanner = EnhancedTokenScanner(self.settings, self.analytics)
        
        # Initialize trading strategy
        self.strategy = TradingStrategy(
            jupiter_client=self.jupiter,
            wallet=self.wallet,
            settings=self.settings,
            scanner=self.scanner,
            mode=TradingMode.PAPER
        )
        
        logger.info("✅ Test components initialized")
    
    async def test_paper_balance_initialization(self):
        """Test that paper balance is properly initialized to $100"""
        logger.info("🧪 Testing paper balance initialization...")
        
        expected_balance = 100.0
        actual_balance = self.strategy.state.paper_balance
        
        if actual_balance == expected_balance:
            logger.info(f"✅ Paper balance correctly initialized: {actual_balance} SOL")
            return True
        else:
            logger.error(f"❌ Paper balance incorrect: Expected {expected_balance}, got {actual_balance}")
            return False
    
    async def test_token_discovery(self):
        """Test that the enhanced scanner can discover approved tokens"""
        logger.info("🧪 Testing token discovery...")
        
        try:
            # Start scanner session
            await self.scanner.start()
            
            # Try to get approved tokens
            approved_tokens = await self.scanner.get_approved_tokens()
            logger.info(f"📊 Found {len(approved_tokens)} approved tokens")
            
            if approved_tokens:
                best_token = approved_tokens[0]
                logger.info(f"✅ Best token: {best_token.token.symbol} (score: {best_token.score:.1f})")
                return True, best_token
            else:
                # Try a fresh scan
                logger.info("🔄 No approved tokens found, trying fresh scan...")
                fresh_tokens = await self.scanner._perform_full_scan()
                if fresh_tokens:
                    logger.info(f"✅ Fresh scan found {len(fresh_tokens)} tokens")
                    return True, fresh_tokens[0]
                else:
                    logger.warning("⚠️ No tokens found in fresh scan either")
                    return False, None
                    
        except Exception as e:
            logger.error(f"❌ Error in token discovery: {e}")
            return False, None
    
    async def test_paper_trade_execution(self, test_token_data=None):
        """Test manual paper trade execution"""
        logger.info("🧪 Testing paper trade execution...")
        
        # Use test token or create mock data
        if test_token_data:
            token_address = test_token_data.token.address
            price = test_token_data.token.price
            symbol = test_token_data.token.symbol
        else:
            # Use mock token for testing
            token_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # Mock USDC address
            price = 0.001  # Mock price in SOL
            symbol = "TEST"
        
        size = 10.0  # 10 tokens
        
        try:
            logger.info(f"🔄 Executing paper trade: {size} {symbol} at {price} SOL per token")
            
            # Execute the paper trade
            success = await self.strategy._execute_paper_trade(token_address, size, price)
            
            if success:
                logger.info("✅ Paper trade executed successfully!")
                
                # Check if position was created
                if token_address in self.strategy.state.paper_positions:
                    position = self.strategy.state.paper_positions[token_address]
                    logger.info(f"✅ Position created: {position.size} tokens at {position.entry_price} SOL")
                    
                    # Check balance was updated
                    cost = size * price
                    expected_balance = 100.0 - cost
                    actual_balance = self.strategy.state.paper_balance
                    
                    if abs(actual_balance - expected_balance) < 0.0001:
                        logger.info(f"✅ Balance correctly updated: {actual_balance} SOL")
                        return True, token_address
                    else:
                        logger.error(f"❌ Balance incorrect: Expected {expected_balance}, got {actual_balance}")
                        return False, None
                else:
                    logger.error("❌ Position was not created")
                    return False, None
            else:
                logger.error("❌ Paper trade execution failed")
                return False, None
                
        except Exception as e:
            logger.error(f"❌ Error executing paper trade: {e}")
            return False, None
    
    async def test_dashboard_updates(self):
        """Test that bot_data.json is properly updated"""
        logger.info("🧪 Testing dashboard updates...")
        
        try:
            # Check if bot_data.json exists and has correct structure
            dashboard_file = "bot_data.json"
            
            if os.path.exists(dashboard_file):
                with open(dashboard_file, 'r') as f:
                    data = json.load(f)
                
                # Check structure
                required_keys = ["status", "trades", "performance", "activity"]
                for key in required_keys:
                    if key not in data:
                        logger.error(f"❌ Missing key in dashboard: {key}")
                        return False
                
                # Check performance data
                perf = data["performance"]
                logger.info(f"📊 Dashboard Performance:")
                logger.info(f"  - Balance: {perf.get('balance', 'N/A')} SOL")
                logger.info(f"  - Total Trades: {perf.get('total_trades', 0)}")
                logger.info(f"  - Open Positions: {perf.get('open_positions', 0)}")
                logger.info(f"  - Total P&L: {perf.get('total_pnl', 0)} SOL")
                
                # Check if we have trades
                trades_count = len(data.get("trades", []))
                logger.info(f"📊 Total trades in dashboard: {trades_count}")
                
                if trades_count > 0:
                    latest_trade = data["trades"][-1]
                    logger.info(f"📊 Latest trade: {latest_trade.get('type', 'unknown')} - {latest_trade.get('status', 'unknown')}")
                
                logger.info("✅ Dashboard structure is correct")
                return True
            else:
                logger.warning("⚠️ Dashboard file does not exist yet")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking dashboard: {e}")
            return False
    
    async def test_position_monitoring(self, token_address):
        """Test position monitoring and price updates"""
        logger.info("🧪 Testing position monitoring...")
        
        try:
            if token_address not in self.strategy.state.paper_positions:
                logger.error("❌ No position found to monitor")
                return False
            
            position = self.strategy.state.paper_positions[token_address]
            initial_price = position.current_price
            
            logger.info(f"📊 Initial position: {position.size} tokens at {initial_price} SOL")
            logger.info(f"📊 Initial unrealized P&L: {position.unrealized_pnl} SOL")
            
            # Simulate price update
            new_price = initial_price * 1.05  # 5% increase
            position.update_price(new_price)
            
            logger.info(f"📊 After price update: {new_price} SOL")
            logger.info(f"📊 New unrealized P&L: {position.unrealized_pnl} SOL")
            
            # Check if P&L calculation is correct
            expected_pnl = (new_price - initial_price) * position.size
            if abs(position.unrealized_pnl - expected_pnl) < 0.0001:
                logger.info("✅ P&L calculation is correct")
                return True
            else:
                logger.error(f"❌ P&L calculation incorrect: Expected {expected_pnl}, got {position.unrealized_pnl}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in position monitoring test: {e}")
            return False
    
    async def test_complete_integration(self):
        """Run complete integration test"""
        logger.info("🚀 Starting Complete Paper Trading Integration Test")
        logger.info("=" * 60)
        
        results = {}
        
        # Test 1: Paper balance initialization
        results["balance_init"] = await self.test_paper_balance_initialization()
        
        # Test 2: Token discovery
        discovery_success, test_token = await self.test_token_discovery()
        results["token_discovery"] = discovery_success
        
        # Test 3: Paper trade execution
        trade_success, token_address = await self.test_paper_trade_execution(test_token)
        results["trade_execution"] = trade_success
        
        # Test 4: Dashboard updates
        results["dashboard_updates"] = await self.test_dashboard_updates()
        
        # Test 5: Position monitoring (only if we have a position)
        if token_address:
            results["position_monitoring"] = await self.test_position_monitoring(token_address)
        else:
            results["position_monitoring"] = False
            logger.warning("⚠️ Skipping position monitoring test - no position created")
        
        # Summary
        logger.info("=" * 60)
        logger.info("📊 INTEGRATION TEST RESULTS:")
        passed = 0
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
            if passed_test:
                passed += 1
        
        logger.info(f"📊 Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! Paper trading system is working correctly!")
            return True
        else:
            logger.error("⚠️ Some tests failed. Paper trading system needs fixes.")
            return False
    
    async def cleanup(self):
        """Clean up test resources"""
        try:
            if self.scanner:
                await self.scanner.stop()
            logger.info("✅ Test cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

async def main():
    """Run the integration test"""
    test = PaperTradingIntegrationTest()
    
    try:
        success = await test.test_complete_integration()
        return 0 if success else 1
    finally:
        await test.cleanup()

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)