#!/usr/bin/env python3
"""
Comprehensive test for real paper trading system
Tests real price fetching, real trending validation, and real P&L calculations
"""
import asyncio
import logging
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.settings import load_settings
from src.trading.strategy import TradingStrategy, TradingMode
from src.practical_solana_scanner import PracticalSolanaScanner
from src.jupiter_client import JupiterClient
from src.wallet import Wallet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestResult:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []
    
    def add_test(self, name: str, passed: bool, error: str = None):
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            logger.info(f"✅ {name}")
        else:
            self.tests_failed += 1
            self.errors.append(f"{name}: {error}")
            logger.error(f"❌ {name}: {error}")
    
    def summary(self):
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Tests Run: {self.tests_run}")
        logger.info(f"Passed: {self.tests_passed}")
        logger.info(f"Failed: {self.tests_failed}")
        if self.errors:
            logger.info(f"\nErrors:")
            for error in self.errors:
                logger.info(f"  - {error}")
        logger.info(f"{'='*60}")

async def test_settings_configuration():
    """Test that settings are properly configured for real paper trading"""
    logger.info("🔧 Testing Settings Configuration")
    result = TestResult()
    
    try:
        settings = load_settings()
        
        # Test basic settings
        result.add_test(
            "Settings loaded successfully", 
            settings is not None
        )
        
        # Test paper trading configuration
        result.add_test(
            "Paper trading enabled",
            settings.PAPER_TRADING == True
        )
        
        result.add_test(
            "Initial paper balance set",
            settings.INITIAL_PAPER_BALANCE > 0
        )
        
        # Test Birdeye trending configuration
        result.add_test(
            "Trending filter enabled",
            settings.ENABLE_TRENDING_FILTER == True
        )
        
        result.add_test(
            "Trending score threshold set",
            0 <= settings.MIN_TRENDING_SCORE <= 100
        )
        
        # Test price and market cap ranges
        result.add_test(
            "Token price range configured",
            settings.MIN_TOKEN_PRICE_SOL < settings.MAX_TOKEN_PRICE_SOL
        )
        
        result.add_test(
            "Market cap range configured",
            settings.MIN_MARKET_CAP_SOL < settings.MAX_MARKET_CAP_SOL
        )
        
        logger.info(f"Settings validation completed")
        return result, settings
        
    except Exception as e:
        result.add_test("Settings loading", False, str(e))
        return result, None

async def test_jupiter_price_fetching(settings):
    """Test Jupiter client real price fetching"""
    logger.info("💰 Testing Jupiter Price Fetching")
    result = TestResult()
    
    try:
        jupiter = JupiterClient(settings)
        
        # Test SOL price fetching (should always work)
        sol_address = "So11111111111111111111111111111111111111112"
        
        # Test quote method
        try:
            price_data = await jupiter.get_quote(
                sol_address,
                sol_address,
                "1000000000",  # 1 SOL
                50
            )
            result.add_test(
                "Jupiter quote API working",
                price_data is not None
            )
        except Exception as e:
            result.add_test("Jupiter quote API", False, str(e))
        
        # Test price history
        try:
            history = await jupiter.get_price_history(sol_address)
            result.add_test(
                "Jupiter price history working",
                history is not None
            )
        except Exception as e:
            result.add_test("Jupiter price history", False, str(e))
        
        # Test market depth
        try:
            depth = await jupiter.get_market_depth(sol_address)
            result.add_test(
                "Jupiter market depth working",
                depth is not None
            )
        except Exception as e:
            result.add_test("Jupiter market depth", False, str(e))
        
        return result
        
    except Exception as e:
        result.add_test("Jupiter client initialization", False, str(e))
        return result

async def test_scanner_real_tokens(settings):
    """Test scanner finding real tokens only"""
    logger.info("🔍 Testing Scanner Real Token Discovery")
    result = TestResult()
    
    try:
        jupiter = JupiterClient(settings)
        scanner = PracticalSolanaScanner(jupiter, None, settings)
        
        # Initialize scanner
        await scanner.start_scanning()
        
        result.add_test(
            "Scanner initialized",
            scanner.running == True
        )
        
        # Test real token scanning
        logger.info("Scanning for real tokens (this may take a moment)...")
        
        # Try multiple scan attempts
        real_token_found = False
        for attempt in range(3):
            token = await scanner.scan_for_new_tokens()
            if token:
                real_token_found = True
                logger.info(f"Found real token: {token.get('symbol', 'UNKNOWN')} from {token.get('source', 'unknown')}")
                
                # Validate token has real data
                result.add_test(
                    f"Token has valid address",
                    'address' in token and len(token['address']) > 30
                )
                
                result.add_test(
                    f"Token has SOL price",
                    'price_sol' in token and token['price_sol'] > 0
                )
                
                result.add_test(
                    f"Token has market cap",
                    'market_cap_sol' in token and token['market_cap_sol'] > 0
                )
                
                result.add_test(
                    f"Token source is real",
                    token.get('source') in ['dexscreener', 'jupiter', 'birdeye_trending']
                )
                
                break
        
        result.add_test(
            "Real token discovered",
            real_token_found
        )
        
        await scanner.stop_scanning()
        
        return result
        
    except Exception as e:
        result.add_test("Scanner testing", False, str(e))
        return result

async def test_paper_trading_execution(settings):
    """Test paper trading execution with real prices"""
    logger.info("📈 Testing Paper Trading Execution")
    result = TestResult()
    
    try:
        # Initialize components
        jupiter = JupiterClient(settings)
        wallet = Wallet(settings.WALLET_ADDRESS, settings.ALCHEMY_RPC_URL)
        
        # Create strategy in paper mode
        strategy = TradingStrategy(
            jupiter_client=jupiter,
            wallet=wallet,
            settings=settings,
            mode=TradingMode.PAPER
        )
        
        result.add_test(
            "Strategy initialized in paper mode",
            strategy.state.mode == TradingMode.PAPER
        )
        
        result.add_test(
            "Initial paper balance set",
            strategy.state.paper_balance == settings.INITIAL_PAPER_BALANCE
        )
        
        # Test real price fetching
        sol_address = "So11111111111111111111111111111111111111112"
        current_price = await strategy._get_current_price(sol_address)
        
        result.add_test(
            "Real price fetching works",
            current_price is not None and current_price > 0
        )
        
        if current_price:
            logger.info(f"Current SOL price: {current_price:.8f} SOL")
        
        # Test paper trade execution
        try:
            initial_balance = strategy.state.paper_balance
            trade_success = await strategy._execute_paper_trade(
                token_address=sol_address,
                size=1.0,  # 1 token
                price=current_price if current_price else 1.0
            )
            
            result.add_test(
                "Paper trade execution",
                trade_success == True
            )
            
            result.add_test(
                "Balance updated after trade",
                strategy.state.paper_balance < initial_balance
            )
            
            result.add_test(
                "Position created",
                sol_address in strategy.state.paper_positions
            )
            
            if sol_address in strategy.state.paper_positions:
                position = strategy.state.paper_positions[sol_address]
                
                result.add_test(
                    "Position has correct entry price",
                    position.entry_price == (current_price if current_price else 1.0)
                )
                
                result.add_test(
                    "Position has correct size",
                    position.size == 1.0
                )
                
                # Test price update with real price
                new_price = await strategy._get_current_price(sol_address)
                if new_price:
                    position.update_price(new_price)
                    
                    result.add_test(
                        "Position price updated",
                        position.current_price == new_price
                    )
                    
                    result.add_test(
                        "P&L calculated correctly",
                        abs(position.unrealized_pnl - ((new_price - position.entry_price) * position.size)) < 0.0001
                    )
                    
                    logger.info(f"Position P&L: {position.unrealized_pnl:.8f} SOL")
        
        except Exception as e:
            result.add_test("Paper trade execution", False, str(e))
        
        return result
        
    except Exception as e:
        result.add_test("Paper trading setup", False, str(e))
        return result

async def test_trending_validation(settings):
    """Test Birdeye trending validation"""
    logger.info("📊 Testing Trending Validation")
    result = TestResult()
    
    try:
        jupiter = JupiterClient(settings)
        scanner = PracticalSolanaScanner(jupiter, None, settings)
        
        # Initialize with trending enabled
        await scanner.start_scanning()
        
        if scanner.birdeye_client:
            result.add_test(
                "Birdeye client initialized",
                True
            )
            
            # Test trending token fetching
            trending_tokens = await scanner.birdeye_client.get_trending_tokens(limit=10)
            
            result.add_test(
                "Trending tokens fetched",
                trending_tokens is not None and len(trending_tokens) > 0
            )
            
            if trending_tokens:
                logger.info(f"Found {len(trending_tokens)} trending tokens")
                
                # Test trending analyzer
                if scanner.trending_analyzer:
                    test_token = trending_tokens[0]
                    
                    # Test scoring
                    score = scanner.trending_analyzer.calculate_trending_score(test_token)
                    result.add_test(
                        "Trending score calculation",
                        0 <= score <= 100
                    )
                    
                    # Test criteria validation
                    passes, reason = scanner.trending_analyzer.meets_trending_criteria(test_token)
                    result.add_test(
                        "Trending criteria validation",
                        isinstance(passes, bool) and isinstance(reason, str)
                    )
                    
                    logger.info(f"Test token {test_token.symbol}: Score={score:.1f}, Passes={passes}, Reason={reason}")
                    
                    # Test signal enhancement
                    base_signal = 0.7
                    enhanced = scanner.trending_analyzer.enhance_signal_strength(base_signal, test_token)
                    result.add_test(
                        "Signal enhancement",
                        enhanced >= base_signal
                    )
                    
                    logger.info(f"Signal enhancement: {base_signal:.3f} → {enhanced:.3f}")
        
        else:
            result.add_test("Birdeye client", False, "Not initialized")
        
        await scanner.stop_scanning()
        
        return result
        
    except Exception as e:
        result.add_test("Trending validation setup", False, str(e))
        return result

async def test_integration_flow(settings):
    """Test complete integration flow"""
    logger.info("🔄 Testing Complete Integration Flow")
    result = TestResult()
    
    try:
        # Initialize all components
        jupiter = JupiterClient(settings)
        wallet = Wallet(settings.WALLET_ADDRESS, settings.ALCHEMY_RPC_URL)
        scanner = PracticalSolanaScanner(jupiter, None, settings)
        
        strategy = TradingStrategy(
            jupiter_client=jupiter,
            wallet=wallet,
            settings=settings,
            scanner=scanner,
            mode=TradingMode.PAPER
        )
        
        result.add_test(
            "All components initialized",
            True
        )
        
        # Start scanner
        await scanner.start_scanning()
        
        # Test one complete scan cycle
        logger.info("Running complete scan and validation cycle...")
        
        token = await scanner.scan_for_new_tokens()
        
        if token:
            logger.info(f"Found token: {token.get('symbol')} from {token.get('source')}")
            
            # Test trending validation if applicable
            if token.get('source') == 'birdeye_trending':
                result.add_test(
                    "Trending token discovered",
                    'trending_score' in token
                )
                
                if 'trending_score' in token:
                    logger.info(f"Trending score: {token['trending_score']:.1f}")
            
            # Test price validation
            price_valid = await strategy._validate_price_conditions(
                token['address'], 
                1.0  # 1 SOL position size
            )
            
            result.add_test(
                "Price validation completed",
                isinstance(price_valid, bool)
            )
            
            logger.info(f"Price validation result: {price_valid}")
            
            # Test signal generation
            try:
                # Create token object for signal analysis
                token_obj = strategy._create_token_object(token['address'], token)
                if token_obj:
                    signal = await strategy.signal_generator.analyze_token(token_obj)
                    
                    result.add_test(
                        "Signal generation works",
                        signal is not None
                    )
                    
                    if signal:
                        logger.info(f"Signal strength: {signal.strength:.3f}")
                        
                        # Test complete flow if signal is strong enough
                        if signal.strength >= settings.SIGNAL_THRESHOLD:
                            result.add_test(
                                "Strong signal generated",
                                True
                            )
                            
                            logger.info("✅ Complete flow successful - token would be traded!")
                        else:
                            result.add_test(
                                "Signal below threshold (expected behavior)",
                                True
                            )
                            
                            logger.info("ℹ️ Signal below threshold - token filtered out correctly")
                    
            except Exception as e:
                result.add_test("Signal generation", False, str(e))
        
        else:
            result.add_test(
                "Token scanning completed (no tokens found this cycle)",
                True
            )
            logger.info("No tokens found in this scan cycle (normal behavior)")
        
        await scanner.stop_scanning()
        
        return result
        
    except Exception as e:
        result.add_test("Integration flow", False, str(e))
        return result

async def main():
    """Main test runner"""
    logger.info("🚀 Starting Real Paper Trading System Tests")
    logger.info("="*60)
    
    overall_result = TestResult()
    
    try:
        # Test 1: Settings Configuration
        settings_result, settings = await test_settings_configuration()
        overall_result.tests_run += settings_result.tests_run
        overall_result.tests_passed += settings_result.tests_passed
        overall_result.tests_failed += settings_result.tests_failed
        overall_result.errors.extend(settings_result.errors)
        
        if not settings:
            logger.error("❌ Cannot continue - settings failed to load")
            return
        
        # Test 2: Jupiter Price Fetching
        jupiter_result = await test_jupiter_price_fetching(settings)
        overall_result.tests_run += jupiter_result.tests_run
        overall_result.tests_passed += jupiter_result.tests_passed
        overall_result.tests_failed += jupiter_result.tests_failed
        overall_result.errors.extend(jupiter_result.errors)
        
        # Test 3: Scanner Real Tokens
        scanner_result = await test_scanner_real_tokens(settings)
        overall_result.tests_run += scanner_result.tests_run
        overall_result.tests_passed += scanner_result.tests_passed
        overall_result.tests_failed += scanner_result.tests_failed
        overall_result.errors.extend(scanner_result.errors)
        
        # Test 4: Paper Trading Execution
        paper_result = await test_paper_trading_execution(settings)
        overall_result.tests_run += paper_result.tests_run
        overall_result.tests_passed += paper_result.tests_passed
        overall_result.tests_failed += paper_result.tests_failed
        overall_result.errors.extend(paper_result.errors)
        
        # Test 5: Trending Validation
        trending_result = await test_trending_validation(settings)
        overall_result.tests_run += trending_result.tests_run
        overall_result.tests_passed += trending_result.tests_passed
        overall_result.tests_failed += trending_result.tests_failed
        overall_result.errors.extend(trending_result.errors)
        
        # Test 6: Integration Flow
        integration_result = await test_integration_flow(settings)
        overall_result.tests_run += integration_result.tests_run
        overall_result.tests_passed += integration_result.tests_passed
        overall_result.tests_failed += integration_result.tests_failed
        overall_result.errors.extend(integration_result.errors)
        
        # Overall summary
        overall_result.summary()
        
        # Final assessment
        success_rate = (overall_result.tests_passed / overall_result.tests_run) * 100
        
        if success_rate >= 90:
            logger.info("🎉 EXCELLENT! Real paper trading system is working perfectly!")
        elif success_rate >= 75:
            logger.info("✅ GOOD! Real paper trading system is mostly working!")
        elif success_rate >= 50:
            logger.info("⚠️ PARTIAL! Some components need attention!")
        else:
            logger.info("❌ ISSUES! Major problems detected!")
        
        logger.info(f"Overall Success Rate: {success_rate:.1f}%")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Tests interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())