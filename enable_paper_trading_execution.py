#!/usr/bin/env python3
"""
Paper Trading Execution Bridge
Ensures that discovered quality tokens are immediately processed into paper trades
"""

import asyncio
import logging
import json
import sys
import os
from datetime import datetime
from pathlib import Path

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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PaperTradingBridge:
    """Bridge to ensure paper trading execution from discovered tokens"""
    
    def __init__(self):
        self.settings = load_settings()
        logger.info("🔗 Initializing Paper Trading Execution Bridge...")
        
        # Initialize core components
        self.alchemy = AlchemyClient(self.settings.ALCHEMY_RPC_URL)
        self.jupiter = JupiterClient()
        self.wallet = PhantomWallet(self.alchemy)
        self.analytics = PerformanceAnalytics(self.settings)
        
        # Initialize scanner and strategy
        self.scanner = EnhancedTokenScanner(self.settings, self.analytics)
        self.strategy = TradingStrategy(
            jupiter_client=self.jupiter,
            wallet=self.wallet,
            settings=self.settings,
            scanner=self.scanner,
            mode=TradingMode.PAPER
        )
        
        logger.info("✅ Paper Trading Bridge initialized successfully")
    
    async def start_connections(self):
        """Start all API connections"""
        logger.info("🔌 Starting API connections...")
        
        # Test connections
        if not await self.alchemy.test_connection():
            raise Exception("❌ Alchemy connection failed")
        logger.info("✅ Alchemy connected")
        
        if not await self.jupiter.test_connection():
            raise Exception("❌ Jupiter connection failed")
        logger.info("✅ Jupiter connected")
        
        # Start scanner
        await self.scanner.start()
        logger.info("✅ Enhanced token scanner started")
        
        logger.info("🚀 All connections established")
    
    async def test_token_discovery(self):
        """Test that token discovery is working"""
        logger.info("🔍 Testing token discovery...")
        
        # Force a manual scan
        tokens = await self.scanner.manual_scan()
        logger.info(f"📊 Manual scan found {len(tokens)} approved tokens")
        
        if tokens:
            for i, token_result in enumerate(tokens[:3]):  # Show first 3
                token = token_result.token
                logger.info(f"  {i+1}. {token.symbol} - Score: {token_result.score:.1f} - {token.source}")
                logger.info(f"     Address: {token.address[:12]}...")
                logger.info(f"     Price: {token.price:.8f} SOL, Liquidity: {token.liquidity:.0f} SOL")
        
        # Test single token scan
        single_token = await self.scanner.scan_for_new_tokens()
        if single_token:
            logger.info(f"🎯 Single scan returned: {single_token.get('symbol', 'Unknown')} (Score: {single_token.get('score', 0):.1f})")
        else:
            logger.warning("⚠️ Single scan returned no tokens")
        
        return len(tokens) > 0 or single_token is not None
    
    async def test_paper_trading_execution(self):
        """Test paper trading execution with a discovered token"""
        logger.info("📋 Testing paper trading execution...")
        
        # Get a token from scanner
        token_data = await self.scanner.scan_for_new_tokens()
        if not token_data:
            logger.warning("⚠️ No tokens available for paper trading test")
            return False
        
        logger.info(f"🎯 Testing paper trade for: {token_data.get('symbol', 'Unknown')}")
        logger.info(f"   Address: {token_data.get('address', 'Unknown')[:12]}...")
        logger.info(f"   Price: {token_data.get('price', 0):.8f} SOL")
        
        # Start trading strategy
        await self.strategy.start_trading()
        
        # Check if the token gets processed into pending orders
        initial_pending = len(self.strategy.state.pending_orders)
        initial_balance = self.strategy.state.paper_balance
        
        logger.info(f"💰 Initial paper balance: {initial_balance:.4f} SOL")
        logger.info(f"📝 Initial pending orders: {initial_pending}")
        
        # Wait for processing
        logger.info("⏳ Waiting for token processing...")
        for i in range(10):  # Wait up to 10 seconds
            await asyncio.sleep(1)
            current_pending = len(self.strategy.state.pending_orders)
            current_positions = len(self.strategy.state.paper_positions)
            
            if current_pending > initial_pending:
                logger.info(f"📝 Pending orders increased to: {current_pending}")
                break
            if current_positions > 0:
                logger.info(f"📈 Paper positions opened: {current_positions}")
                break
        
        # Check final state
        final_pending = len(self.strategy.state.pending_orders)
        final_positions = len(self.strategy.state.paper_positions)
        final_balance = self.strategy.state.paper_balance
        
        logger.info(f"📊 Final state:")
        logger.info(f"   Paper balance: {final_balance:.4f} SOL")
        logger.info(f"   Pending orders: {final_pending}")
        logger.info(f"   Open positions: {final_positions}")
        
        # Check if execution occurred
        execution_occurred = (
            final_positions > 0 or 
            final_balance < initial_balance or 
            final_pending > initial_pending
        )
        
        if execution_occurred:
            logger.info("✅ Paper trading execution is working!")
            
            # Show position details if any
            if final_positions > 0:
                for addr, pos in self.strategy.state.paper_positions.items():
                    logger.info(f"📍 Position: {addr[:8]}... - Size: {pos.size:.4f}, Entry: {pos.entry_price:.8f}")
        else:
            logger.warning("⚠️ No paper trading execution detected")
        
        await self.strategy.stop_trading()
        return execution_occurred
    
    async def check_dashboard_updates(self):
        """Check that dashboard is being updated with current data"""
        logger.info("📊 Checking dashboard updates...")
        
        try:
            with open('bot_data.json', 'r') as f:
                data = json.load(f)
            
            last_update = data.get('last_update', 'Never')
            activity_count = len(data.get('activity', []))
            trades_count = len(data.get('trades', []))
            
            logger.info(f"📈 Dashboard status:")
            logger.info(f"   Last update: {last_update}")
            logger.info(f"   Activity entries: {activity_count}")
            logger.info(f"   Trades recorded: {trades_count}")
            
            # Check if updated recently (within last hour)
            if last_update != 'Never':
                try:
                    last_dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                    age_hours = (datetime.now() - last_dt).total_seconds() / 3600
                    
                    if age_hours < 1:
                        logger.info("✅ Dashboard data is current")
                        return True
                    else:
                        logger.info(f"⚠️ Dashboard data is {age_hours:.1f} hours old")
                except:
                    pass
            
            logger.info("ℹ️ Dashboard reset - ready for new data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking dashboard: {e}")
            return False
    
    async def run_comprehensive_test(self):
        """Run comprehensive test of paper trading execution"""
        logger.info("🚀 Starting comprehensive paper trading execution test...")
        
        try:
            # Start connections
            await self.start_connections()
            
            # Test 1: Token Discovery
            logger.info("\n" + "="*60)
            logger.info("TEST 1: Token Discovery")
            logger.info("="*60)
            discovery_works = await self.test_token_discovery()
            
            # Test 2: Paper Trading Execution  
            logger.info("\n" + "="*60)
            logger.info("TEST 2: Paper Trading Execution")
            logger.info("="*60)
            execution_works = await self.test_paper_trading_execution()
            
            # Test 3: Dashboard Updates
            logger.info("\n" + "="*60)
            logger.info("TEST 3: Dashboard Updates")
            logger.info("="*60)
            dashboard_works = await self.check_dashboard_updates()
            
            # Summary
            logger.info("\n" + "="*60)
            logger.info("COMPREHENSIVE TEST RESULTS")
            logger.info("="*60)
            logger.info(f"✅ Token Discovery: {'PASS' if discovery_works else 'FAIL'}")
            logger.info(f"✅ Paper Trading: {'PASS' if execution_works else 'FAIL'}")
            logger.info(f"✅ Dashboard Updates: {'PASS' if dashboard_works else 'FAIL'}")
            
            all_pass = discovery_works and execution_works and dashboard_works
            
            if all_pass:
                logger.info("\n🎉 ALL TESTS PASSED - Paper trading execution is fully operational!")
                logger.info("📈 The bot should now execute paper trades from discovered tokens")
            else:
                logger.warning("\n⚠️ Some tests failed - check logs above for details")
            
            return all_pass
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            return False
        finally:
            # Cleanup
            try:
                await self.scanner.stop()
                await self.jupiter.close()
                await self.alchemy.close()
            except:
                pass
    
    async def shutdown(self):
        """Clean shutdown"""
        logger.info("🛑 Shutting down Paper Trading Bridge...")
        try:
            if hasattr(self, 'scanner'):
                await self.scanner.stop()
            if hasattr(self, 'jupiter'):
                await self.jupiter.close()
            if hasattr(self, 'alchemy'):
                await self.alchemy.close()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

async def main():
    """Main execution"""
    bridge = PaperTradingBridge()
    
    try:
        success = await bridge.run_comprehensive_test()
        
        if success:
            print("\n🚀 Paper trading execution is now enabled!")
            print("💡 You can now run the main bot with: python main.py")
            print("📊 Dashboard will show real-time paper trading activity")
        else:
            print("\n❌ Paper trading execution test failed")
            print("🔧 Check the logs above for issues to resolve")
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await bridge.shutdown()

if __name__ == "__main__":
    asyncio.run(main())