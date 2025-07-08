import asyncio
import logging
import aiohttp
import os
from pathlib import Path
from src.config.settings import Settings, load_settings
from src.api.alchemy import AlchemyClient
from src.api.jupiter import JupiterClient
from src.practical_solana_scanner import PracticalSolanaScanner
from src.phantom_wallet import PhantomWallet
from src.trading.strategy import TradingStrategy, TradingMode
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Setup logging FIRST
def setup_logging():
    """Configure logging for the bot"""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set third-party loggers to WARNING
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

# Setup logging immediately
setup_logging()
logger = logging.getLogger(__name__)

# Load settings
try:
    settings = load_settings()
    logger.info(f"Settings loaded successfully")
    logger.info(f"Paper trading: {settings.PAPER_TRADING}")
except Exception as e:
    logger.error(f"Failed to load settings: {e}")
    exit(1)

class TradingBot:
    def __init__(self):
        """Initialize the trading bot"""
        logger.info("[INIT] Initializing SolTrader APE Bot...")
        
        # Load settings and initialize core components
        self.settings = settings
        self.alchemy = AlchemyClient(self.settings.ALCHEMY_RPC_URL)
        self.jupiter = JupiterClient()
        self.wallet = PhantomWallet(self.alchemy)
        self.scanner = PracticalSolanaScanner(self.jupiter, self.alchemy, self.settings)

        # Initialize trading strategy
        mode = TradingMode.PAPER if self.settings.PAPER_TRADING else TradingMode.LIVE
        self.strategy = TradingStrategy(
            jupiter_client=self.jupiter,
            wallet=self.wallet,
            settings=self.settings,
            scanner=self.scanner,
            mode=mode
        )

        self.telegram_bot: Optional[Any] = None
        logger.info("[OK] Bot components initialized")

    async def startup(self) -> bool:
        """Initialize all bot components"""
        try:
            logger.info("[SETUP] Starting bot initialization...")
            
            # Test connections
            logger.info("Testing API connections...")
            if not await self.alchemy.test_connection():
                logger.error("[ERROR] Alchemy connection failed")
                return False
            logger.info("[OK] Alchemy connection successful")

            if not await self.jupiter.test_connection():
                logger.error("[ERROR] Jupiter connection failed")
                return False
            logger.info("[OK] Jupiter connection successful")

            if not await self.connect_wallet():
                logger.error("[ERROR] Wallet connection failed")
                return False
            logger.info("[OK] Wallet connected")

            # Mode announcement
            mode = "Paper" if self.settings.PAPER_TRADING else "Live"
            logger.info(f"[MODE] Bot initialized in {mode} trading mode")
            logger.info(f"[BALANCE] Initial balance: {self.settings.INITIAL_PAPER_BALANCE} SOL (paper)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Startup failed: {str(e)}")
            return False

    async def connect_wallet(self) -> bool:
        """Connect and validate wallet"""
        try:
            if self.settings.PAPER_TRADING:
                # For paper trading, just validate address format
                if len(self.settings.WALLET_ADDRESS) < 32:
                    logger.error("Invalid wallet address format")
                    return False
                logger.info("[WALLET] Paper trading wallet validated")
                return True
            else:
                # For live trading, actually connect wallet
                return await self.wallet.connect()
        except Exception as e:
            logger.error(f"Wallet connection error: {e}")
            return False

    async def run(self) -> None:
        """Main bot execution loop"""
        try:
            logger.info("[START] Starting SolTrader APE Bot...")
            
            # Initialize all components
            if not await self.startup():
                logger.error("[ERROR] Startup failed - exiting")
                return

            logger.info("[READY] Bot startup complete - starting trading strategy...")
            
            # Start the trading strategy
            await self.strategy.start_trading()
            
            logger.info("[LOOP] Entering main event loop...")
            
            # Keep bot running indefinitely
            while True:
                try:
                    await asyncio.sleep(60)  # Check every minute
                    
                    # Basic health check
                    if not self.strategy.is_trading:
                        logger.warning("[WARN] Strategy is not trading - checking status...")
                        # Could add restart logic here if needed
                    
                except asyncio.CancelledError:
                    logger.info("[STOP] Bot shutdown requested")
                    break
                except Exception as e:
                    logger.error(f"[ERROR] Error in main loop: {str(e)}")
                    await asyncio.sleep(5)  # Brief pause before continuing
                    
        except KeyboardInterrupt:
            logger.info("[INTERRUPT] Keyboard interrupt - shutting down...")
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error: {str(e)}")
        finally:
            logger.info("[SHUTDOWN] Starting shutdown sequence...")
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown"""
        logger.info("[CLEANUP] Starting shutdown...")
        try:
            # Stop strategy first
            if hasattr(self.strategy, 'stop_trading'):
                await self.strategy.stop_trading()
                logger.info("[OK] Trading strategy stopped")
            
            # Close connections
            if self.jupiter:
                await self.jupiter.close()
                logger.info("[OK] Jupiter client closed")
                
            if self.alchemy:
                await self.alchemy.close()
                logger.info("[OK] Alchemy client closed")
                
            if self.wallet and not self.settings.PAPER_TRADING:
                await self.wallet.disconnect()
                logger.info("[OK] Wallet disconnected")
                
            # Cancel remaining tasks
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if tasks:
                logger.info(f"[CLEANUP] Cancelling {len(tasks)} remaining tasks...")
                for task in tasks:
                    task.cancel()
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    logger.debug(f"Task cancellation error: {e}")
                    
            logger.info("[OK] Shutdown complete")
                    
        except Exception as e:
            logger.error(f"[ERROR] Error during shutdown: {e}")

async def main():
    """Main entry point"""
    bot = None
    try:
        bot = TradingBot()
        await bot.run()
    except Exception as e:
        logger.error(f"💥 Fatal error: {str(e)}")
    finally:
        if bot:
            await bot.shutdown()

if __name__ == "__main__":
    try:
        # Run the bot
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"💥 Failed to start bot: {str(e)}")
        exit(1)
