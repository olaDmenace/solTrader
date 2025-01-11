import asyncio
import logging
from src.config.settings import Settings, load_settings
from src.api.alchemy import AlchemyClient
from src.api.jupiter import JupiterClient
from src.token_scanner import TokenScanner
from src.phantom_wallet import PhantomWallet
from src.telegram_bot import TradingBot as TgramBot
from src.trading.strategy import TradingStrategy, TradingMode
from typing import Dict, Any, Optional
from telegram.ext import ApplicationBuilder
from dotenv import load_dotenv
load_dotenv(override=True)

import os
print("ENV URL:", os.getenv('ALCHEMY_RPC_URL'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load settings first
settings = load_settings()
logger.info(f"Bot token: {settings.TELEGRAM_BOT_TOKEN[:5] if settings.TELEGRAM_BOT_TOKEN else 'Not Set'}...")
logger.info(f"Chat ID: {settings.TELEGRAM_CHAT_ID}")

class TradingBot:
    def __init__(self):
        # Load settings and initialize core components
        self.settings = settings
        self.alchemy = AlchemyClient(self.settings.ALCHEMY_RPC_URL)
        self.jupiter = JupiterClient()
        self.wallet = PhantomWallet(self.alchemy)
        self.scanner = TokenScanner(self.jupiter, self.alchemy)

        # Initialize trading strategy
        mode = TradingMode.PAPER if self.settings.PAPER_TRADING else TradingMode.LIVE
        self.strategy = TradingStrategy(
            jupiter_client=self.jupiter,
            wallet=self.wallet,
            settings=self.settings,
            scanner=self.scanner,
            mode=mode
        )

        self.telegram_bot: Optional[TgramBot] = None

        logger.info(f"Paper trading settings:")
        logger.info(f"Initial balance: {self.settings.INITIAL_PAPER_BALANCE}")
        logger.info(f"Max position size: {self.settings.MAX_POSITION_SIZE}")

    async def setup_telegram_bot(self) -> bool:
        try:
            if not self.settings.TELEGRAM_BOT_TOKEN or not self.settings.TELEGRAM_CHAT_ID:
                logger.info("Telegram bot disabled - no token or chat ID provided")
                return True

            application = (
                ApplicationBuilder()
                .token(self.settings.TELEGRAM_BOT_TOKEN)
                .base_url("https://api.telegram.org/bot")
                .base_file_url("https://api.telegram.org/file/bot")
                .get_updates_connection_pool_size(8)
                .connect_timeout(30.0)
                .read_timeout(30.0)
                .write_timeout(30.0)
                .pool_timeout(3.0)
                .build()
            )

            self.telegram_bot = TgramBot(
                application=application,
                trading_strategy=self.strategy,
                settings=self.settings
            )

            await self.telegram_bot.initialize()
            await self.telegram_bot.start()  # Add this line
            return True

        except Exception as e:
            logger.error(f"Error setting up Telegram bot: {str(e)}")
            return False

    async def test_connections(self) -> bool:
        """Test API connections"""
        try:
            # Test Alchemy connection
            alchemy_ok = await self.alchemy.test_connection()
            if not alchemy_ok:
                logger.error("Alchemy connection failed")
                return False

            # Test Jupiter connection
            jupiter_ok = await self.jupiter.test_connection()
            if not jupiter_ok:
                logger.error("Jupiter connection failed")
                return False

            # Connect wallet
            wallet_ok = await self.connect_wallet()
            if not wallet_ok:
                logger.error("Wallet connection failed")
                return False

            return True

        except Exception as e:
            logger.error(f"Connection test error: {str(e)}")
            return False

    async def connect_wallet(self) -> bool:
        """Connect to wallet and validate"""
        try:
            logger.info("Connecting to wallet...")
            if not self.settings.WALLET_ADDRESS:
                logger.error("No wallet address configured")
                return False

            connected = await self.wallet.connect(self.settings.WALLET_ADDRESS)
            if not connected:
                return False

            # Get and log balance
            balance = await self.wallet.get_balance()
            logger.info(f"Wallet connected. Balance: {balance} SOL")

            # Start monitoring wallet transactions
            await self.start_wallet_monitoring()
            return True

        except Exception as e:
            logger.error(f"Wallet connection error: {str(e)}")
            return False

    async def start_wallet_monitoring(self) -> bool:
        """Start monitoring wallet transactions"""
        async def transaction_callback(tx_info: Dict[str, Any]) -> None:
            try:
                logger.info(f"New transaction: {tx_info}")
                balance = await self.wallet.get_balance()
                if balance is not None:
                    logger.info(f"Current balance: {balance} SOL")
            except Exception as e:
                logger.error(f"Transaction callback error: {str(e)}")

        return await self.wallet.monitor_transactions(callback=transaction_callback)

    # async def startup(self) -> bool:
    #     """Initialize all components"""
    #     logger.info("Starting trading bot...")

    #     if not await self.test_connections():
    #         logger.error("Connection tests failed")
    #         return False

    #     if not await self.setup_telegram_bot():
    #         logger.error("Telegram bot setup failed")
    #         return False

    #     mode = "Paper" if self.settings.PAPER_TRADING else "Live"
    #     logger.info(f"Bot initialized in {mode} trading mode")
    #     return True

    async def startup(self) -> bool:
        logger.info("Starting trading bot...")

        if not await self.test_connections():
            return False

        if not await self.connect_wallet():
            return False

        if self.settings.TELEGRAM_BOT_TOKEN and self.settings.TELEGRAM_CHAT_ID:
            if not await self.setup_telegram_bot():  # Changed from telegram_bot.start()
                return False

        mode = "Paper" if self.settings.PAPER_TRADING else "Live"
        logger.info(f"Bot initialized in {mode} trading mode")
        return True

    async def run(self) -> None:
        """Main bot execution"""
        try:
            if await self.startup():
                await self.strategy.start_trading()
                # Keep bot running
                while True:
                    await asyncio.sleep(1)
            else:
                logger.error("Startup failed")
                return

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await self.shutdown()
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            await self.shutdown()

    # async def shutdown(self) -> None:
    #     """Clean shutdown of all components"""
    #     logger.info("Starting shutdown...")
    #     try:
    #         # Stop trading
    #         if hasattr(self, 'strategy'):
    #             await self.strategy.stop_trading()

    #         # Stop Telegram bot
    #         if self.telegram_bot:
    #             await self.telegram_bot.stop()

    #         # Disconnect wallet
    #         if hasattr(self, 'wallet'):
    #             await self.wallet.disconnect()

    #         # Close API clients
    #         if hasattr(self, 'jupiter'):
    #             await self.jupiter.close()
    #         if hasattr(self, 'alchemy'):
    #             await self.alchemy.close()

    #         # Wait a bit for connections to close
    #         await asyncio.sleep(0.1)

    #         # Cancel any remaining tasks
    #         tasks = [t for t in asyncio.all_tasks() 
    #                 if t is not asyncio.current_task()]
    #         for task in tasks:
    #             task.cancel()
    #             try:
    #                 await task
    #             except asyncio.CancelledError:
    #                 pass

    #     except Exception as e:
    #         logger.error(f"Error during shutdown: {e}")
    #     finally:
    #         logger.info("Shutdown complete")
    
    
    
    # async def shutdown(self) -> None:
    #     logger.info("Starting shutdown...")
    #     try:
    #         if self.telegram_bot:
    #             await self.telegram_bot.stop()
                
    #         await self.wallet.disconnect()
    #         await self.jupiter.close()
    #         await self.alchemy.close()
                
    #         # Cancel remaining tasks
    #         tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    #         for task in tasks:
    #             task.cancel()
    #     except Exception as e:
    #         logger.error(f"Error during shutdown: {e}")

    async def shutdown(self) -> None:
        logger.info("Starting shutdown...")
        try:
            if self.telegram_bot:
                await self.telegram_bot.stop()
                    
            await self.wallet.disconnect()
            await self.jupiter.close()
            await self.alchemy.close()
                    
            # Cancel remaining tasks
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    
async def main():
    bot = None
    try:
        bot = TradingBot()
        await bot.run()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        if bot:
            await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

# def main():
#     bot = TradingBot()
#     asyncio.run(bot.run())

# if __name__ == "__main__":
#     main()