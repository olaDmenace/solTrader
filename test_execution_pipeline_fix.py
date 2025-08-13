#!/usr/bin/env python3
"""
MISSION-CRITICAL TEST: Paper Trading Execution Pipeline Fix Verification
This test script verifies that the execution pipeline fix works properly.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.settings import Settings
from src.trading.strategy import TradingStrategy, TradingMode, EntrySignal
from src.trading.signals import Signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/execution_test.log')
    ]
)

logger = logging.getLogger(__name__)

class MockJupiterClient:
    """Mock Jupiter client for testing"""
    def __init__(self):
        self.connected = True
        
    async def get_price(self, token_address: str) -> float:
        """Mock price fetching"""
        return 0.000042  # Sample price
        
    async def close(self):
        pass

class MockWallet:
    """Mock wallet for testing"""
    def __init__(self):
        self.balance = 1000.0
        
    async def get_balance(self) -> float:
        return self.balance

class MockScanner:
    """Mock scanner that returns test tokens"""
    def __init__(self):
        self.test_tokens = [
            {
                "address": "CvGBG44dVcUKNdDXHNGWFtL8xeZnTqV9T8EkGQ4s2VeR",
                "symbol": "TESTTOKEN",
                "price_sol": 0.000042,
                "volume_24h_sol": 0.0,  # This was causing the validation failure
                "liquidity_sol": 500000.0,
                "market_cap_sol": 2627679.0,
                "momentum": 156.4,
                "age_minutes": 5,
                "source": "enhanced_scanner",
                "score": 129.9
            }
        ]
        
    async def scan_for_new_tokens(self):
        """Return test token"""
        if self.test_tokens:
            token = self.test_tokens[0]
            logger.info(f"[MOCK_SCANNER] Returning test token: {token['symbol']} ({token['address'][:8]}...)")
            return token
        return None

async def test_execution_pipeline():
    """Test the fixed execution pipeline"""
    logger.info("=" * 60)
    logger.info("🔬 STARTING EXECUTION PIPELINE FIX TEST")
    logger.info("=" * 60)
    
    try:
        # Initialize settings
        settings = Settings()
        settings.PAPER_TRADING = True
        settings.INITIAL_PAPER_BALANCE = 100.0
        settings.PAPER_MIN_LIQUIDITY = 10.0  # Very low for testing
        settings.PAPER_SIGNAL_THRESHOLD = 0.1  # Very low threshold
        settings.SCAN_INTERVAL = 1  # Fast for testing
        settings.MAX_POSITIONS = 5
        settings.STOP_LOSS_PERCENTAGE = 0.15
        settings.TAKE_PROFIT_PERCENTAGE = 0.25
        
        # Initialize components
        jupiter_client = MockJupiterClient()
        wallet = MockWallet()
        scanner = MockScanner()
        
        # Initialize strategy in paper trading mode
        strategy = TradingStrategy(
            jupiter_client=jupiter_client,
            wallet=wallet,
            settings=settings,
            scanner=scanner,
            mode=TradingMode.PAPER
        )
        
        logger.info(f"[TEST] Strategy initialized in {strategy.state.mode.value} mode")
        logger.info(f"[TEST] Initial paper balance: {strategy.state.paper_balance:.4f} SOL")
        logger.info(f"[TEST] Pending orders: {len(strategy.state.pending_orders)}")
        logger.info(f"[TEST] Active positions: {len(strategy.state.paper_positions)}")
        
        # Test 1: Token validation with the fix
        logger.info("\n" + "=" * 50)
        logger.info("🧪 TEST 1: Token Validation Fix")
        logger.info("=" * 50)
        
        test_token = scanner.test_tokens[0]
        logger.info(f"[TEST] Testing token with ZERO volume (this was the bug): {test_token['volume_24h_sol']} SOL")
        
        # Create token object using the fixed method
        token_obj = strategy._create_token_object(test_token["address"], test_token)
        if not token_obj:
            logger.error("[FAIL] Token object creation failed!")
            return False
            
        logger.info(f"[SUCCESS] Token object created with enhanced data:")
        logger.info(f"  Volume: {token_obj.volume24h:.2f} SOL")
        logger.info(f"  Liquidity: {token_obj.liquidity:.2f} SOL") 
        logger.info(f"  Price: {token_obj.price_sol:.8f} SOL")
        logger.info(f"  Market Cap: {token_obj.market_cap:.0f} SOL")
        
        # Test validation
        is_valid = strategy._validate_token_basics(token_obj)
        if not is_valid:
            logger.error("[FAIL] Token validation failed even with the fix!")
            return False
            
        logger.info("[SUCCESS] ✅ Token validation passed with fix!")
        
        # Test 2: Signal generation and pending order creation
        logger.info("\n" + "=" * 50)
        logger.info("🧪 TEST 2: Signal Generation and Pending Orders")
        logger.info("=" * 50)
        
        # Manually scan opportunities to test the fixed pipeline
        await strategy._scan_opportunities()
        
        logger.info(f"[TEST] Pending orders after scan: {len(strategy.state.pending_orders)}")
        if len(strategy.state.pending_orders) == 0:
            logger.error("[FAIL] No pending orders created - signal generation failed!")
            return False
            
        logger.info("[SUCCESS] ✅ Pending orders created!")
        for i, order in enumerate(strategy.state.pending_orders):
            logger.info(f"  Order {i+1}: {order.token_address[:8]}... size: {order.size:.4f}")
        
        # Test 3: Order processing and execution
        logger.info("\n" + "=" * 50)
        logger.info("🧪 TEST 3: Order Processing and Execution")
        logger.info("=" * 50)
        
        initial_balance = strategy.state.paper_balance
        initial_positions = len(strategy.state.paper_positions)
        initial_trades = len(strategy.state.completed_trades)
        
        logger.info(f"[BEFORE] Balance: {initial_balance:.4f} SOL")
        logger.info(f"[BEFORE] Positions: {initial_positions}")
        logger.info(f"[BEFORE] Completed trades: {initial_trades}")
        
        # Process pending orders
        await strategy._process_pending_orders()
        
        final_balance = strategy.state.paper_balance
        final_positions = len(strategy.state.paper_positions)
        final_trades = len(strategy.state.completed_trades)
        
        logger.info(f"[AFTER] Balance: {final_balance:.4f} SOL")
        logger.info(f"[AFTER] Positions: {final_positions}")
        logger.info(f"[AFTER] Completed trades: {final_trades}")
        
        # Verify execution
        if final_balance == initial_balance:
            logger.error("[FAIL] Balance didn't change - trade execution failed!")
            return False
            
        if final_positions == initial_positions:
            logger.error("[FAIL] No new positions created - execution failed!")
            return False
            
        if final_trades == initial_trades:
            logger.error("[FAIL] No trades recorded - dashboard update failed!")
            return False
            
        balance_change = initial_balance - final_balance
        logger.info(f"[SUCCESS] ✅ Trade executed! Balance changed by {balance_change:.4f} SOL")
        logger.info(f"[SUCCESS] ✅ New position created! ({final_positions - initial_positions} new)")
        logger.info(f"[SUCCESS] ✅ Trade recorded! ({final_trades - initial_trades} new)")
        
        # Test 4: Verify position details
        logger.info("\n" + "=" * 50)
        logger.info("🧪 TEST 4: Position Details Verification")
        logger.info("=" * 50)
        
        for token_addr, position in strategy.state.paper_positions.items():
            logger.info(f"[POSITION] Token: {token_addr[:8]}...")
            logger.info(f"  Entry Price: {position.entry_price:.8f} SOL")
            logger.info(f"  Size: {position.size:.4f} tokens")
            logger.info(f"  Stop Loss: {position.stop_loss:.8f} SOL")
            logger.info(f"  Take Profit: {position.take_profit:.8f} SOL")
            logger.info(f"  Status: {position.status}")
            
        # Test 5: Dashboard data verification
        logger.info("\n" + "=" * 50)
        logger.info("🧪 TEST 5: Dashboard Data Verification")
        logger.info("=" * 50)
        
        for i, trade in enumerate(strategy.state.completed_trades):
            logger.info(f"[TRADE {i+1}] {trade}")
            
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ALL TESTS PASSED! EXECUTION PIPELINE FIXED!")
        logger.info("=" * 60)
        logger.info("🚀 Key Fixes Applied:")
        logger.info("  ✅ Ultra-permissive paper trading validation")
        logger.info("  ✅ Enhanced token data estimation from market cap")
        logger.info("  ✅ Multiple fallback data sources")
        logger.info("  ✅ Improved balance tracking and logging")
        logger.info("  ✅ Proper dashboard trade recording")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Test failed with exception: {e}")
        import traceback
        logger.error(f"[TRACEBACK] {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_execution_pipeline())
    if result:
        print("\n✅ EXECUTION PIPELINE FIX TEST: PASSED")
        sys.exit(0)
    else:
        print("\n❌ EXECUTION PIPELINE FIX TEST: FAILED")
        sys.exit(1)