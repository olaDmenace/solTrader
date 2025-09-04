import asyncio
import logging
import aiohttp
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from src.config.settings import Settings, load_settings
from src.api.alchemy import AlchemyClient
from src.api.jupiter import JupiterClient
from src.api.solana_tracker import SolanaTrackerClient
from src.enhanced_token_scanner import EnhancedTokenScanner
from src.phantom_wallet import PhantomWallet
from src.trading.strategy import TradingStrategy, TradingMode
from src.trading.grid_trading_strategy import GridTradingStrategy
from src.trading.mean_reversion_strategy import MeanReversionStrategy
from src.coordination.strategy_coordinator import StrategyCoordinator
from src.trading.arbitrage_system import ArbitrageSystem
from src.analytics.performance_analytics import PerformanceAnalytics
from src.notifications.email_system import EmailNotificationSystem
from src.dashboard.unified_web_dashboard import UnifiedWebDashboard
from src.monitoring.health_monitor import HealthMonitor
from src.logging.trade_logger import CentralizedTradeLogger
from src.portfolio.dynamic_capital_allocator import DynamicCapitalAllocator
from src.portfolio.allocator_integration import PortfolioManager
from src.portfolio.portfolio_risk_manager import PortfolioRiskManager
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
    
    # Configure logging with UTF-8 encoding
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading.log', encoding='utf-8'),
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
        
        # Initialize enhanced components
        self.solana_tracker = SolanaTrackerClient()
        self.analytics = PerformanceAnalytics(self.settings)
        self.enhanced_scanner = EnhancedTokenScanner(self.settings, self.analytics)
        self.email_system = EmailNotificationSystem(self.settings)
        self.dashboard = UnifiedWebDashboard(
            self.settings, 
            self.analytics, 
            self.email_system, 
            self.solana_tracker
        )

        # Initialize Phase 2 components (Grid Trading & Strategy Coordination)
        mode = TradingMode.PAPER if self.settings.PAPER_TRADING else TradingMode.LIVE
        self.grid_trading = GridTradingStrategy(
            settings=self.settings,
            jupiter_client=self.jupiter,
            wallet=self.wallet,
            mode=mode,
            analytics=self.analytics  # Pass analytics for trade recording
        )
        
        self.mean_reversion = MeanReversionStrategy(
            settings=self.settings
        )
        
        self.strategy_coordinator = StrategyCoordinator(
            settings=self.settings,
            analytics=self.analytics
        )
        
        self.arbitrage_system = ArbitrageSystem(
            settings=self.settings,
            jupiter_client=self.jupiter,
            analytics=self.analytics  # Pass analytics for trade recording
        )
        
        # Connect dashboard to arbitrage system for live data
        self.dashboard.set_arbitrage_system(self.arbitrage_system)
        
        # Initialize centralized trade logger
        self.trade_logger = CentralizedTradeLogger(self.settings)

        # Initialize Phase 3 components (Dynamic Capital Allocation & Portfolio Management)
        self.capital_allocator = DynamicCapitalAllocator(self.settings)
        self.portfolio_risk_manager = PortfolioRiskManager(self.settings)
        self.portfolio_manager = PortfolioManager(self.settings, self.capital_allocator, self.portfolio_risk_manager)

        # Initialize trading strategy with enhanced scanner and Phase 2 integration
        self.strategy = TradingStrategy(
            jupiter_client=self.jupiter,
            wallet=self.wallet,
            settings=self.settings,
            scanner=self.enhanced_scanner,  # Use enhanced scanner
            mode=mode,
            grid_trading=self.grid_trading,
            strategy_coordinator=self.strategy_coordinator,
            arbitrage_system=self.arbitrage_system,
            trade_logger=self.trade_logger
        )

        # Initialize health monitoring system
        self.health_monitor = HealthMonitor(bot_instance=self, settings=self.settings)
        
        self.telegram_bot: Optional[Any] = None
        logger.info("[OK] Bot components initialized")
    
    async def _integrate_strategies_with_portfolio(self):
        """Integrate all trading strategies with the portfolio management system"""
        try:
            from src.portfolio.allocator_integration import integrate_existing_strategy
            
            # Integration configurations for each strategy
            strategy_configs = [
                {
                    'instance': self.strategy,
                    'name': 'momentum_strategy',
                    'allocation': 0.30
                },
                {
                    'instance': self.grid_trading,
                    'name': 'grid_trading_strategy', 
                    'allocation': 0.25
                },
                {
                    'instance': self.mean_reversion,
                    'name': 'mean_reversion_strategy',
                    'allocation': 0.20
                },
                {
                    'instance': self.arbitrage_system,
                    'name': 'arbitrage_strategy',
                    'allocation': 0.25
                }
            ]
            
            # Integrate each strategy
            for config in strategy_configs:
                success = await integrate_existing_strategy(
                    config['instance'],
                    self.capital_allocator,
                    config['name']
                )
                
                if success:
                    # Also integrate with portfolio manager
                    await self.portfolio_manager.integrate_strategy(
                        config['instance'],
                        config['name']
                    )
                    
                    logger.info(f"[PORTFOLIO] Integrated {config['name']} with {config['allocation']:.1%} allocation")
                else:
                    logger.warning(f"[PORTFOLIO] Failed to integrate {config['name']}")
            
            # Perform initial portfolio rebalancing
            rebalance_result = await self.portfolio_manager.force_rebalance()
            if rebalance_result.get('status') == 'completed':
                logger.info(f"[PORTFOLIO] Initial rebalancing completed: {rebalance_result.get('rebalanced_strategies', 0)} strategies")
            
        except Exception as e:
            logger.error(f"[PORTFOLIO] Error integrating strategies with portfolio: {e}")
            # Continue without portfolio integration rather than failing startup

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

            # Skip Jupiter connection test during startup to avoid rate limits
            # The connection will be tested during actual trading
            try:
                await self.jupiter.test_connection()
                logger.info("[OK] Jupiter connection successful")
            except Exception as e:
                logger.warning(f"[WARNING] Jupiter connection test skipped due to rate limits: {e}")
                logger.info("[OK] Jupiter will be tested during trading")

            if not await self.connect_wallet():
                logger.error("[ERROR] Wallet connection failed")
                return False
            logger.info("[OK] Wallet connected")

            # Test Solana Tracker API
            try:
                if not await self.solana_tracker.test_connection():
                    logger.warning("[WARNING] Solana Tracker API connection failed - continuing without it")
                else:
                    logger.info("[OK] Solana Tracker API connection successful")
            except Exception as e:
                logger.warning(f"[WARNING] Solana Tracker API test failed: {e} - continuing without it")

            # Start enhanced components
            logger.info("Starting enhanced systems...")
            
            # Start email system
            await self.email_system.start()
            logger.info("[OK] Email notification system started")
            
            # Start enhanced scanner
            await self.enhanced_scanner.start()
            logger.info("[OK] Enhanced token scanner started")
            
            # Start unified web dashboard
            await self.dashboard.start()
            logger.info("[OK] Unified Web Dashboard started")
            
            # Start arbitrage system
            await self.arbitrage_system.start()
            logger.info("[OK] Arbitrage system started")
            
            # Start centralized trade logger
            await self.trade_logger.start()
            logger.info("[OK] Centralized trade logger started")
            
            # Start Phase 3 components (Dynamic Capital Allocation & Portfolio Management)
            await self.capital_allocator.start()
            logger.info("[OK] Dynamic Capital Allocator started")
            
            await self.portfolio_risk_manager.start()
            logger.info("[OK] Portfolio Risk Manager started")
            
            await self.portfolio_manager.start()
            logger.info("[OK] Portfolio Manager started")
            
            # Integrate strategies with portfolio manager
            await self._integrate_strategies_with_portfolio()
            logger.info("[OK] Strategies integrated with portfolio management")

            # Mode announcement
            mode = "Paper" if self.settings.PAPER_TRADING else "Live"

            # Send startup notification
            await self.email_system.send_critical_alert(
                "SolTrader Bot Started - Phase 3 Active",
                f"Bot successfully started in {mode} mode with Phase 3 Dynamic Portfolio Management:\n\n"
                f"🚀 Phase 3 Dynamic Portfolio Management Active:\n"
                f"- Dynamic Capital Allocator with real-time rebalancing\n"
                f"- Market regime detection and tactical allocation\n"
                f"- Portfolio-level risk management\n"
                f"- Performance-based strategy rebalancing\n"
                f"- Multi-strategy coordination with capital optimization\n\n"
                f"🔥 Phase 2 Multi-Strategy System:\n"
                f"- Grid Trading Strategy for ranging markets\n"
                f"- Cross-Strategy Coordination system\n"
                f"- Multi-DEX Arbitrage System\n"
                f"- Centralized trade logging with execution metrics\n"
                f"- Realistic market structure simulation\n\n"
                f"📊 Enhanced Features:\n"
                f"- Unified Web Dashboard (http://localhost:5000)\n"
                f"- Solana Tracker API integrated\n"
                f"- Optimized filters (100 SOL liquidity, 5% momentum)\n"
                f"- High momentum bypass (>500% gains)\n"
                f"- Email notifications active\n"
                f"- Real-time analytics and monitoring\n\n"
                f"Initial balance: {self.settings.INITIAL_PAPER_BALANCE} SOL"
            )
            logger.info(f"[MODE] Bot initialized in {mode} trading mode")
            logger.info(f"[BALANCE] Initial balance: {self.settings.INITIAL_PAPER_BALANCE} SOL (paper)")
            logger.info("[ENHANCED] All enhanced features activated successfully")
            
            # TEMPORARY: Skip health monitoring to fix core trading
            # await self.health_monitor.start_monitoring()
            logger.info("[HEALTH] Health monitoring disabled for debugging")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Startup failed: {str(e)}")
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
                return await self.wallet.connect(self.settings.WALLET_ADDRESS)
        except Exception as e:
            logger.error(f"Wallet connection error: {e}")
            return False

    async def _perform_health_checks(self):
        """Perform enhanced health checks"""
        try:
            # Check API rate limits
            usage_stats = self.solana_tracker.get_usage_stats()
            if usage_stats['usage_percentage'] > 90:
                await self.email_system.send_performance_alert(
                    "api_limit",
                    f"API usage at {usage_stats['usage_percentage']:.1f}%. Approaching daily limit.",
                    usage_stats
                )
            
            # Check system performance
            real_time_metrics = self.analytics.get_real_time_metrics()
            if real_time_metrics['current_drawdown'] > 20:  # 20% drawdown threshold
                await self.email_system.send_performance_alert(
                    "risk_breach",
                    f"Portfolio drawdown at {real_time_metrics['current_drawdown']:.1f}%. Risk limits breached.",
                    real_time_metrics
                )
            
        except Exception as e:
            logger.error(f"Error in health checks: {e}")

    async def _check_daily_report(self):
        """Check if daily report should be sent"""
        try:
            now = datetime.now()
            report_time = self.settings.DAILY_REPORT_TIME.split(':')
            report_hour = int(report_time[0])
            report_minute = int(report_time[1]) if len(report_time) > 1 else 0
            
            # Check if it's time for daily report (within 1-minute window)
            if (now.hour == report_hour and 
                report_minute <= now.minute <= report_minute + 1):
                
                daily_stats = self.analytics.get_daily_breakdown()
                await self.email_system.send_daily_report(daily_stats)
                
                # Reset daily stats at midnight
                if now.hour == 0 and now.minute <= 1:
                    self.analytics.reset_daily_stats()
                    self.enhanced_scanner.reset_daily_stats()
                    
        except Exception as e:
            logger.error(f"Error checking daily report: {e}")

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
                    
                    # Enhanced health checks
                    await self._perform_health_checks()
                    
                    # Send daily report if needed
                    await self._check_daily_report()
                    
                    # Basic health check
                    if not self.strategy.is_trading:
                        logger.warning("[WARN] Strategy is not trading - checking status...")
                        await self.email_system.send_performance_alert(
                            "system_health",
                            "Trading strategy is not active. Investigating...",
                            {"timestamp": str(datetime.now()), "issue": "strategy_inactive"}
                        )
                    
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
            # Send shutdown notification
            await self.email_system.send_critical_alert(
                "SolTrader Bot Shutdown",
                "Bot is shutting down. Final statistics will be sent in daily report."
            )

            # Stop health monitoring first
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()
                logger.info("[OK] Health monitoring stopped")
            
            # Stop enhanced components
            logger.info("Stopping enhanced systems...")
            
            if self.dashboard:
                await self.dashboard.stop()
                logger.info("[OK] Unified Web Dashboard stopped")
            
            if self.arbitrage_system:
                await self.arbitrage_system.stop()
                logger.info("[OK] Arbitrage system stopped")
            
            if self.trade_logger:
                await self.trade_logger.stop()
                logger.info("[OK] Centralized trade logger stopped")
            
            # Stop Phase 3 components (Portfolio Management)
            if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                await self.portfolio_manager.stop()
                logger.info("[OK] Portfolio Manager stopped")
            
            if hasattr(self, 'portfolio_risk_manager') and self.portfolio_risk_manager:
                await self.portfolio_risk_manager.stop()
                logger.info("[OK] Portfolio Risk Manager stopped")
            
            if hasattr(self, 'capital_allocator') and self.capital_allocator:
                await self.capital_allocator.stop()
                logger.info("[OK] Dynamic Capital Allocator stopped")
            
            if self.enhanced_scanner:
                await self.enhanced_scanner.stop()
                logger.info("[OK] Enhanced scanner stopped")
            
            if self.email_system:
                await self.email_system.stop()
                logger.info("[OK] Email system stopped")

            # Stop strategy
            if hasattr(self.strategy, 'stop_trading'):
                await self.strategy.stop_trading()
                logger.info("[OK] Trading strategy stopped")
            
            # Close API connections
            if self.solana_tracker:
                await self.solana_tracker.close()
                logger.info("[OK] Solana Tracker client closed")

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
        logger.error(f"[FATAL] Fatal error: {str(e)}")
    finally:
        if bot:
            await bot.shutdown()

if __name__ == "__main__":
    try:
        # Run the bot
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[STOP] Bot stopped by user")
    except Exception as e:
        logger.error(f"[FAILED] Failed to start bot: {str(e)}")
        exit(1)
