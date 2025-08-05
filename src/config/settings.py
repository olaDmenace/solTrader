
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

@dataclass
class Settings:
    # Required settings
    ALCHEMY_RPC_URL: str
    WALLET_ADDRESS: str

    # Paper Trading settings - OPTIMIZED
    position_manager: Any = None
    PAPER_TRADING: bool = True
    INITIAL_PAPER_BALANCE: float = 100.0
    MAX_POSITION_SIZE: float = 0.35  # Optimized for better risk management
    MAX_SLIPPAGE: float = 0.30  # Higher slippage tolerance for meme tokens
    MAX_TRADES_PER_DAY: int = 20  # More trades for ape strategy
    
    # Paper Trading Specific Parameters
    PAPER_MIN_MOMENTUM_THRESHOLD: float = 3.0  # Lower momentum threshold for paper trading
    PAPER_MIN_LIQUIDITY: float = 50.0  # Lower liquidity requirement for paper trading
    PAPER_TRADING_SLIPPAGE: float = 0.50  # Higher slippage tolerance for paper trading (50%)
    PAPER_BASE_POSITION_SIZE: float = 0.1  # Base position size for paper trading
    PAPER_MAX_POSITION_SIZE: float = 0.5  # Max position size for paper trading
    PAPER_SIGNAL_THRESHOLD: float = 0.3  # Lower signal threshold for paper trading
    
    # Trading pause (disable while fixing scanner)
    TRADING_PAUSED: bool = False  # Trading enabled with new scanner

    # Trading parameters (optimized for new token sniping)
    MIN_BALANCE: float = 0.1  # Minimum SOL balance to maintain
    MAX_TRADE_SIZE: float = 2.0  # Smaller trades for more opportunities
    SLIPPAGE_TOLERANCE: float = 0.25  # Higher tolerance for meme tokens

    # Position Management - OPTIMIZED
    MAX_POSITIONS: int = 3  # Maximum number of open positions
    MAX_SIMULTANEOUS_POSITIONS: int = 3  # Aligned with MAX_POSITIONS
    MIN_TRADE_SIZE: float = 0.1  # Minimum trade size  
    INITIAL_CAPITAL: float = 100.0  # Starting capital
    PORTFOLIO_VALUE: float = 100.0  # Current portfolio value
    MIN_PORTFOLIO_VALUE: float = 10.0  # Minimum portfolio value threshold

    # Monitoring settings (high-frequency for momentum trading)
    MONITOR_INTERVAL: float = 1.0  # Faster monitoring for position updates
    POSITION_MONITOR_INTERVAL: float = 3.0  # Very fast position monitoring
    STALE_THRESHOLD: int = 300  # Consider data stale after 5 minutes

    # Scanner settings - OPTIMIZED FOR 40-60% APPROVAL RATE
    SCAN_INTERVAL: int = 5  # Very fast scanning for new tokens
    MIN_LIQUIDITY: float = 100.0  # FURTHER REDUCED from 250 to 100 SOL for higher approval
    MIN_VOLUME_24H: float = 50.0  # Lower volume requirement for new launches
    VOLUME_THRESHOLD: float = 50.0  # Lower volume threshold
    MIN_VOLUME_GROWTH: float = 0.0  # REMOVED volume growth requirement
    MIN_MOMENTUM_PERCENTAGE: float = 5.0  # REDUCED from 10% to 5% for more opportunities
    MAX_TOKEN_AGE_HOURS: float = 24.0  # EXTENDED from 12 to 24 hours for more tokens
    HIGH_MOMENTUM_BYPASS: float = 500.0  # LOWERED from 1000% to 500% for more bypasses  
    MEDIUM_MOMENTUM_BYPASS: float = 100.0  # NEW: Medium momentum bypass at 100%
    MAX_PRICE_IMPACT: float = 2.0  # Higher impact tolerance for new tokens
    
    # New token sniping settings - Solana specific
    NEW_TOKEN_MAX_AGE_MINUTES: int = 2880  # Consider tokens new if < 48 hours old (48 * 60 = 2880)
    MIN_CONTRACT_SCORE: int = 70  # Minimum security score for entry
    MAX_TOKEN_PRICE_SOL: float = 0.01  # Max price in SOL (target micro-cap)
    MIN_TOKEN_PRICE_SOL: float = 0.000001  # Min price in SOL (avoid dust)
    MAX_MARKET_CAP_SOL: float = 50000.0  # Max market cap in SOL (~$7.5M)
    MIN_MARKET_CAP_SOL: float = 10.0  # Min market cap in SOL (~$1.5K)
    
    # Solana blockchain specific
    SOLANA_ONLY: bool = True  # Only trade Solana tokens
    EXCLUDE_MAJOR_TOKENS: bool = True  # Exclude BTC, ETH, USDC, etc.
    
    # Raydium/Jupiter DEX settings for new token detection
    RAYDIUM_PROGRAM_ID: str = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
    JUPITER_API_URL: str = "https://quote-api.jup.ag/v6"
    MAX_HOLD_TIME_MINUTES: int = 180  # Maximum hold time (3 hours)
    FAST_EXIT_THRESHOLD: float = -0.1  # Quick exit at 10% loss
    
    # Momentum trading settings
    MOMENTUM_EXIT_ENABLED: bool = True  # Enable dynamic momentum exits
    MOMENTUM_THRESHOLD: float = -0.03  # Exit on 3% negative momentum
    RSI_OVERBOUGHT: float = 80.0  # RSI overbought level
    PROFIT_PROTECTION_THRESHOLD: float = 0.2  # Start trailing at 20% profit

    # Notification settings
    DISCORD_WEBHOOK_URL: Optional[str] = None
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

    # Risk management (enhanced for ape trading)
    MAX_DAILY_TRADES: int = 15  # More trades for aggressive strategy
    MAX_DAILY_LOSS: float = 2.0  # Higher loss tolerance
    STOP_LOSS_PERCENTAGE: float = 0.15  # 15% stop loss (wider for volatility)
    TAKE_PROFIT_PERCENTAGE: float = 0.5  # 50% take profit (let winners run)
    MAX_DRAWDOWN: float = 10.0  # Higher drawdown tolerance
    MAX_PORTFOLIO_RISK: float = 10.0  # Higher portfolio risk
    ERROR_THRESHOLD: int = 10  # More errors allowed
    MAX_VOLATILITY: float = 0.8  # Higher volatility tolerance for new tokens
    
    # Position management
    MAX_POSITION_PER_TOKEN: float = 1.0  # Max 1 SOL per token
    MAX_SIMULTANEOUS_POSITIONS: int = 5  # More concurrent positions

    # Signal settings (optimized for new token detection)
    SIGNAL_THRESHOLD: float = 0.5  # Lower threshold for faster entries
    MIN_SIGNAL_INTERVAL: int = 60  # Faster signal processing
    MAX_SIGNALS_PER_HOUR: int = 15  # More signals per hour

    # Technical Analysis settings
    VOLUME_WEIGHT: float = 0.3
    LIQUIDITY_WEIGHT: float = 0.3
    MOMENTUM_WEIGHT: float = 0.2
    MARKET_IMPACT_WEIGHT: float = 0.2

    # Price movement thresholds
    MIN_PRICE_CHANGE: float = 0.02  # 2% minimum price movement
    MAX_PRICE_CHANGE: float = 0.20  # 20% maximum price movement
    MOMENTUM_LOOKBACK: int = 12  # Hours for momentum calculation

    # Grid trading settings
    MIN_GRID_LEVELS: int = 3
    MAX_GRID_LEVELS: int = 10
    MIN_GRID_SPACING: float = 0.01
    GRID_TAKE_PROFIT: float = 0.02  # 2% profit per grid

    # DCA settings
    MIN_DCA_INTERVAL: int = 1  # hours
    MAX_DCA_ENTRIES: int = 100
    DEFAULT_DCA_AMOUNT: float = 0.1  # SOL

    # Operation settings
    CLOSE_POSITIONS_ON_STOP: bool = True  # Close positions when stopping
    
    # Gas optimization for fast execution
    PRIORITY_FEE_MULTIPLIER: float = 2.0  # Higher priority for fast execution
    MAX_GAS_PRICE: int = 200  # Higher gas limit for speed
    
    # Solana Tracker API Configuration - REPLACES BIRDEYE
    SOLANA_TRACKER_KEY: Optional[str] = None  # API key for Solana Tracker
    SOLANA_TRACKER_BASE_URL: str = "https://data.solanatracker.io"
    DAILY_REQUEST_LIMIT: int = 333  # Free tier: 10k/month ÷ 30 days
    TRENDING_INTERVAL: int = 780     # 13 minutes (4.6 req/hour)
    VOLUME_INTERVAL: int = 900       # 15 minutes (4 req/hour)
    MEMESCOPE_INTERVAL: int = 1080   # 18 minutes (3.3 req/hour)
    
    # Email Notification Configuration
    EMAIL_ENABLED: bool = True
    EMAIL_SMTP_SERVER: str = "smtp.gmail.com"
    EMAIL_PORT: int = 587
    EMAIL_USER: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None
    EMAIL_TO: Optional[str] = None
    DAILY_REPORT_TIME: str = "20:00"  # 8 PM daily report
    CRITICAL_ALERTS: bool = True
    PERFORMANCE_ALERTS: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            key: getattr(self, key)
            for key in self.__annotations__
            if hasattr(self, key)
        }

    def validate(self) -> bool:
        """Validate settings with enhanced checks for ape trading"""
        try:
            # Required settings
            if not self.ALCHEMY_RPC_URL or not self.WALLET_ADDRESS:
                return False

            # Trading parameters
            if (self.MIN_BALANCE <= 0 or 
                self.MAX_TRADE_SIZE <= 0 or 
                self.SLIPPAGE_TOLERANCE <= 0 or
                self.MAX_POSITION_SIZE <= 0):
                return False

            # Position limits
            if (self.MAX_POSITIONS <= 0 or
                self.MIN_TRADE_SIZE <= 0 or
                self.INITIAL_CAPITAL <= 0):
                return False

            # Monitoring settings
            if (self.MONITOR_INTERVAL <= 0 or 
                self.POSITION_MONITOR_INTERVAL <= 0 or
                self.STALE_THRESHOLD <= 0):
                return False

            # Scanner settings
            if (self.SCAN_INTERVAL <= 0 or 
                self.MIN_LIQUIDITY <= 0 or 
                self.MIN_VOLUME_24H <= 0 or
                self.VOLUME_THRESHOLD <= 0):
                return False

            # Risk management
            if (self.MAX_DAILY_TRADES <= 0 or 
                self.MAX_DAILY_LOSS <= 0 or 
                self.STOP_LOSS_PERCENTAGE <= 0 or 
                self.TAKE_PROFIT_PERCENTAGE <= 0 or
                self.MAX_DRAWDOWN <= 0 or
                self.MAX_PORTFOLIO_RISK <= 0):
                return False

            # New token settings validation
            if (self.NEW_TOKEN_MAX_AGE_MINUTES <= 0 or
                self.MIN_CONTRACT_SCORE < 0 or self.MIN_CONTRACT_SCORE > 100 or
                self.MAX_HOLD_TIME_MINUTES <= 0):
                return False

            # Grid trading validation
            if (self.MIN_GRID_LEVELS <= 0 or
                self.MAX_GRID_LEVELS < self.MIN_GRID_LEVELS or
                self.MIN_GRID_SPACING <= 0):
                return False

            # DCA validation
            if (self.MIN_DCA_INTERVAL <= 0 or
                self.MAX_DCA_ENTRIES <= 0 or
                self.DEFAULT_DCA_AMOUNT <= 0):
                return False

            return True
        except Exception:
            return False

def load_settings() -> Settings:
    """Load settings from environment variables"""
    load_dotenv()

    # Required settings
    alchemy_url = os.getenv('ALCHEMY_RPC_URL')
    if not alchemy_url:
        raise ValueError("ALCHEMY_RPC_URL not found in environment variables")

    wallet_address = os.getenv('WALLET_ADDRESS')
    if not wallet_address:
        raise ValueError("WALLET_ADDRESS not found in environment variables")

    settings = Settings(
        ALCHEMY_RPC_URL=alchemy_url,
        WALLET_ADDRESS=wallet_address,
        DISCORD_WEBHOOK_URL=os.getenv('DISCORD_WEBHOOK_URL'),
        TELEGRAM_BOT_TOKEN=os.getenv('TELEGRAM_BOT_TOKEN'),
        TELEGRAM_CHAT_ID=os.getenv('TELEGRAM_CHAT_ID'),
        SOLANA_TRACKER_KEY=os.getenv('SOLANA_TRACKER_KEY'),
        EMAIL_USER=os.getenv('EMAIL_USER'),
        EMAIL_PASSWORD=os.getenv('EMAIL_PASSWORD'),
        EMAIL_TO=os.getenv('EMAIL_TO'),
    )

    # Add new environment mappings for grid and DCA settings
    env_mappings = {
        # ... (your existing mappings)
        'MIN_GRID_LEVELS': ('MIN_GRID_LEVELS', int),
        'MAX_GRID_LEVELS': ('MAX_GRID_LEVELS', int),
        'MIN_GRID_SPACING': ('MIN_GRID_SPACING', float),
        'GRID_TAKE_PROFIT': ('GRID_TAKE_PROFIT', float),
        'MIN_DCA_INTERVAL': ('MIN_DCA_INTERVAL', int),
        'MAX_DCA_ENTRIES': ('MAX_DCA_ENTRIES', int),
        'DEFAULT_DCA_AMOUNT': ('DEFAULT_DCA_AMOUNT', float),
        'MAX_VOLATILITY': ('MAX_VOLATILITY', float),
    }

    # Expanded environment mappings
    env_mappings = {
        # Existing mappings
        'MIN_BALANCE': ('TRADING_MIN_BALANCE', float),
        'MAX_TRADE_SIZE': ('TRADING_MAX_SIZE', float),
        'MAX_POSITION_SIZE': ('MAX_POSITION_SIZE', float),
        'MAX_POSITIONS': ('MAX_POSITIONS', int),
        'SLIPPAGE_TOLERANCE': ('TRADING_SLIPPAGE', float),
        'MONITOR_INTERVAL': ('MONITOR_INTERVAL', float),
        'POSITION_MONITOR_INTERVAL': ('POSITION_MONITOR_INTERVAL', float),
        'SCAN_INTERVAL': ('SCANNER_INTERVAL', int),
        'STALE_THRESHOLD': ('STALE_THRESHOLD', int),
        'MIN_LIQUIDITY': ('MIN_LIQUIDITY', float),
        'MIN_VOLUME_24H': ('MIN_VOLUME_24H', float),
        'MAX_DAILY_TRADES': ('MAX_DAILY_TRADES', int),
        'MAX_DAILY_LOSS': ('MAX_DAILY_LOSS', float),
        'STOP_LOSS_PERCENTAGE': ('STOP_LOSS_PERCENTAGE', float),
        'TAKE_PROFIT_PERCENTAGE': ('TAKE_PROFIT_PERCENTAGE', float),
        
        # New token sniping settings
        'NEW_TOKEN_MAX_AGE_MINUTES': ('NEW_TOKEN_MAX_AGE_MINUTES', int),
        'MIN_CONTRACT_SCORE': ('MIN_CONTRACT_SCORE', int),
        'MAX_HOLD_TIME_MINUTES': ('MAX_HOLD_TIME_MINUTES', int),
        'MOMENTUM_EXIT_ENABLED': ('MOMENTUM_EXIT_ENABLED', lambda x: x.lower() == 'true'),
        'MOMENTUM_THRESHOLD': ('MOMENTUM_THRESHOLD', float),
        'RSI_OVERBOUGHT': ('RSI_OVERBOUGHT', float),
        'PROFIT_PROTECTION_THRESHOLD': ('PROFIT_PROTECTION_THRESHOLD', float),
        'MAX_POSITION_PER_TOKEN': ('MAX_POSITION_PER_TOKEN', float),
        'MAX_SIMULTANEOUS_POSITIONS': ('MAX_SIMULTANEOUS_POSITIONS', int),

        # New mappings
        'PAPER_TRADING': ('PAPER_TRADING', lambda x: x.lower() == 'true'),
        'INITIAL_PAPER_BALANCE': ('INITIAL_PAPER_BALANCE', float),
        
        # Paper Trading Specific Parameters
        'PAPER_MIN_MOMENTUM_THRESHOLD': ('PAPER_MIN_MOMENTUM_THRESHOLD', float),
        'PAPER_MIN_LIQUIDITY': ('PAPER_MIN_LIQUIDITY', float),
        'PAPER_TRADING_SLIPPAGE': ('PAPER_TRADING_SLIPPAGE', float),
        'PAPER_BASE_POSITION_SIZE': ('PAPER_BASE_POSITION_SIZE', float),
        'PAPER_MAX_POSITION_SIZE': ('PAPER_MAX_POSITION_SIZE', float),
        'PAPER_SIGNAL_THRESHOLD': ('PAPER_SIGNAL_THRESHOLD', float),
        'MIN_GRID_LEVELS': ('MIN_GRID_LEVELS', int),
        'MAX_GRID_LEVELS': ('MAX_GRID_LEVELS', int),
        'MIN_GRID_SPACING': ('MIN_GRID_SPACING', float),
        'GRID_TAKE_PROFIT': ('GRID_TAKE_PROFIT', float),
        'MIN_DCA_INTERVAL': ('MIN_DCA_INTERVAL', int),
        'MAX_DCA_ENTRIES': ('MAX_DCA_ENTRIES', int),
        'DEFAULT_DCA_AMOUNT': ('DEFAULT_DCA_AMOUNT', float),
        'MAX_VOLATILITY': ('MAX_VOLATILITY', float),
        'SIGNAL_THRESHOLD': ('SIGNAL_THRESHOLD', float),
        'MIN_SIGNAL_INTERVAL': ('MIN_SIGNAL_INTERVAL', int),
        'MAX_SIGNALS_PER_HOUR': ('MAX_SIGNALS_PER_HOUR', int),
        'VOLUME_WEIGHT': ('VOLUME_WEIGHT', float),
        'LIQUIDITY_WEIGHT': ('LIQUIDITY_WEIGHT', float),
        'MOMENTUM_WEIGHT': ('MOMENTUM_WEIGHT', float),
        'MARKET_IMPACT_WEIGHT': ('MARKET_IMPACT_WEIGHT', float),
        'MIN_PRICE_CHANGE': ('MIN_PRICE_CHANGE', float),
        'MAX_PRICE_CHANGE': ('MAX_PRICE_CHANGE', float),
        'MOMENTUM_LOOKBACK': ('MOMENTUM_LOOKBACK', int),
        'CLOSE_POSITIONS_ON_STOP': ('CLOSE_POSITIONS_ON_STOP', lambda x: x.lower() == 'true'),
        'PRIORITY_FEE_MULTIPLIER': ('PRIORITY_FEE_MULTIPLIER', float),
        'MAX_GAS_PRICE': ('MAX_GAS_PRICE', int),
        
        # Optimized filter settings
        'MIN_VOLUME_GROWTH': ('MIN_VOLUME_GROWTH', float),
        'MIN_MOMENTUM_PERCENTAGE': ('MIN_MOMENTUM_PERCENTAGE', float),
        'MAX_TOKEN_AGE_HOURS': ('MAX_TOKEN_AGE_HOURS', float),
        'HIGH_MOMENTUM_BYPASS': ('HIGH_MOMENTUM_BYPASS', float),
        'MEDIUM_MOMENTUM_BYPASS': ('MEDIUM_MOMENTUM_BYPASS', float),
        
        # Solana Tracker API settings
        'SOLANA_TRACKER_KEY': ('SOLANA_TRACKER_KEY', str),
        'SOLANA_TRACKER_BASE_URL': ('SOLANA_TRACKER_BASE_URL', str),
        'DAILY_REQUEST_LIMIT': ('DAILY_REQUEST_LIMIT', int),
        'TRENDING_INTERVAL': ('TRENDING_INTERVAL', int),
        'VOLUME_INTERVAL': ('VOLUME_INTERVAL', int),
        'MEMESCOPE_INTERVAL': ('MEMESCOPE_INTERVAL', int),
        
        # Email notification settings
        'EMAIL_ENABLED': ('EMAIL_ENABLED', lambda x: x.lower() == 'true'),
        'EMAIL_SMTP_SERVER': ('EMAIL_SMTP_SERVER', str),
        'EMAIL_PORT': ('EMAIL_PORT', int),
        'EMAIL_USER': ('EMAIL_USER', str),
        'EMAIL_PASSWORD': ('EMAIL_PASSWORD', str),
        'EMAIL_TO': ('EMAIL_TO', str),
        'DAILY_REPORT_TIME': ('DAILY_REPORT_TIME', str),
        'CRITICAL_ALERTS': ('CRITICAL_ALERTS', lambda x: x.lower() == 'true'),
        'PERFORMANCE_ALERTS': ('PERFORMANCE_ALERTS', lambda x: x.lower() == 'true'),
    }

    for attr, (env_var, type_func) in env_mappings.items():
        if value := os.getenv(env_var):
            try:
                setattr(settings, attr, type_func(value))
            except ValueError:
                logger.warning(f"Invalid value for {env_var}, using default")

    if not settings.validate():
        raise ValueError("Invalid settings configuration")

    return settings