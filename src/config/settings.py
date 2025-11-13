
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

    # Paper Trading settings - RISK MITIGATION APPLIED
    position_manager: Any = None
    PAPER_TRADING: bool = False  # FIXED: Default to live trading, override via .env
    TRADING_MODE: str = "live"  # Trading mode: "paper" or "live"  
    INITIAL_PAPER_BALANCE: float = 100.0
    
    # Paper Trading Specific Parameters
    PAPER_MIN_MOMENTUM_THRESHOLD: float = 1.5  # Very low threshold for more trades
    PAPER_MIN_LIQUIDITY: float = 2.0  # Extremely low liquidity requirement
    PAPER_TRADING_SLIPPAGE: float = 0.05  # 5% slippage tolerance
    PAPER_BASE_POSITION_SIZE: float = 0.5  # Base position size
    PAPER_MAX_POSITION_SIZE: float = 0.8  # Max position size
    PAPER_SIGNAL_THRESHOLD: float = 0.05  # Low threshold for max frequency
    
    MAX_POSITION_SIZE: float = 0.05  # CRITICAL FIX: 5% max position (was suicide 35%)
    MAX_SLIPPAGE: float = 3.0  # CRITICAL FIX: 3% slippage (was destroying profits at 10%)
    MAX_TRADES_PER_DAY: int = 15  # Reduced for better quality trades
    
    # Trading pause (disable while fixing scanner)
    TRADING_PAUSED: bool = False  # Trading enabled with new scanner

    # Trading parameters - RISK MITIGATION APPLIED
    MIN_BALANCE: float = 0.005  # Minimum SOL balance to maintain (for micro-capital)
    MAX_TRADE_SIZE: float = 0.02  # CRITICAL FIX: Max 0.02 SOL per trade for micro-capital
    SLIPPAGE_TOLERANCE: float = 0.03  # CRITICAL FIX: 3% slippage (was too high 25%)

    # Position Management - OPTIMIZED
    MAX_POSITIONS: int = 1  # Maximum number of open positions (micro-capital)
    MAX_SIMULTANEOUS_POSITIONS: int = 1  # Aligned with MAX_POSITIONS
    MIN_TRADE_SIZE: float = 0.005  # Minimum trade size (micro-capital)  
    INITIAL_CAPITAL: float = 100.0  # Starting capital
    PORTFOLIO_VALUE: float = 100.0  # Current portfolio value
    MIN_PORTFOLIO_VALUE: float = 10.0  # Minimum portfolio value threshold

    # Monitoring settings (high-frequency for momentum trading)
    MONITOR_INTERVAL: float = 1.0  # Faster monitoring for position updates
    POSITION_MONITOR_INTERVAL: float = 3.0  # Very fast position monitoring
    STALE_THRESHOLD: int = 300  # Consider data stale after 5 minutes

    # Scanner settings - OPTIMIZED FOR MICRO-CAPITAL ($2.79)
    SCAN_INTERVAL: int = 5  # Very fast scanning for new tokens
    MIN_LIQUIDITY: float = 5.0  # MICRO-CAPITAL: Reduced from 100 to 5 SOL for tiny trades
    MIN_VOLUME_24H: float = 10.0  # MICRO-CAPITAL: Very low volume for micro trades
    VOLUME_THRESHOLD: float = 10.0  # MICRO-CAPITAL: Very low volume threshold
    MIN_VOLUME_GROWTH: float = 0.0  # REMOVED volume growth requirement
    MIN_MOMENTUM_PERCENTAGE: float = 2.0  # MICRO-CAPITAL: Reduced from 5% to 2% for more opportunities
    MAX_TOKEN_AGE_HOURS: float = 24.0  # EXTENDED from 12 to 24 hours for more tokens
    HIGH_MOMENTUM_BYPASS: float = 500.0  # LOWERED from 1000% to 500% for more bypasses  
    MEDIUM_MOMENTUM_BYPASS: float = 100.0  # NEW: Medium momentum bypass at 100%
    MAX_PRICE_IMPACT: float = 2.0  # Higher impact tolerance for new tokens
    
    # New token sniping settings - USD-based for better UX
    NEW_TOKEN_MAX_AGE_MINUTES: int = 2880  # Consider tokens new if < 48 hours old (48 * 60 = 2880)
    MIN_CONTRACT_SCORE: int = 70  # Minimum security score for entry
    
    # USD-based pricing (professional standard)
    MAX_TOKEN_PRICE_USD: float = 2.0  # Max price in USD (target micro-cap tokens)
    MIN_TOKEN_PRICE_USD: float = 0.00001  # Min price in USD (avoid dust tokens)
    MAX_MARKET_CAP_USD: float = 10000000.0  # Max market cap in USD ($10M - small cap)
    MIN_MARKET_CAP_USD: float = 100000.0  # Min market cap in USD ($100K - avoid very new)
    
    # Legacy SOL-based settings (deprecated but maintained for compatibility)
    MAX_TOKEN_PRICE_SOL: float = 0.01  # DEPRECATED: Use MAX_TOKEN_PRICE_USD
    MIN_TOKEN_PRICE_SOL: float = 0.000001  # DEPRECATED: Use MIN_TOKEN_PRICE_USD  
    MAX_MARKET_CAP_SOL: float = 50000.0  # DEPRECATED: Use MAX_MARKET_CAP_USD
    MIN_MARKET_CAP_SOL: float = 10.0  # DEPRECATED: Use MIN_MARKET_CAP_USD
    
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
    
    # Mean Reversion Strategy Settings
    ENABLE_MEAN_REVERSION: bool = False  # Enable mean reversion strategy (Phase 2)
    MEAN_REVERSION_RSI_OVERSOLD: float = 20.0  # RSI oversold threshold
    MEAN_REVERSION_RSI_OVERBOUGHT: float = 80.0  # RSI overbought threshold
    MEAN_REVERSION_Z_SCORE_THRESHOLD: float = -2.0  # Z-score buy threshold
    MEAN_REVERSION_MIN_LIQUIDITY_USD: float = 25000.0  # Minimum liquidity for mean reversion
    MEAN_REVERSION_CONFIDENCE_THRESHOLD: float = 0.6  # Minimum confidence for signal
    
    # Enhanced Exit Manager Settings - CONFIGURABLE VIA .ENV
    DYNAMIC_STOP_LOSS_BASE: float = 0.10  # Base stop loss percentage (10%)
    VOLATILITY_MULTIPLIER: float = 1.5  # Stop loss volatility adjustment multiplier
    TRAILING_STOP_ACTIVATION: float = 0.15  # Profit threshold to activate trailing stop (15%)
    TRAILING_STOP_PERCENTAGE: float = 0.08  # Trailing stop distance (8%)
    MAX_HOLD_TIME_HOURS: int = 6  # Maximum hold time in hours
    TAKE_PROFIT_LEVEL_1: float = 0.20  # First take profit level (20%)
    TAKE_PROFIT_LEVEL_2: float = 0.35  # Second take profit level (35%)  
    TAKE_PROFIT_LEVEL_3: float = 0.50  # Third take profit level (50%)
    POSITION_SCALE_1: float = 0.30  # First exit percentage (30%)
    POSITION_SCALE_2: float = 0.40  # Second exit percentage (40%)
    POSITION_SCALE_3: float = 0.30  # Third exit percentage (30%)

    # Notification settings
    DISCORD_WEBHOOK_URL: Optional[str] = None
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

    # Risk management - CRITICAL RISK MITIGATION APPLIED
    MAX_DAILY_TRADES: int = 5  # Reduced for micro-capital
    MAX_DAILY_LOSS: float = 0.005  # CRITICAL FIX: 0.005 SOL daily loss limit for micro-capital
    STOP_LOSS_PERCENTAGE: float = 0.15  # CRITICAL FIX: 15% stop loss (from .env)
    TAKE_PROFIT_PERCENTAGE: float = 0.25  # 25% take profit (from .env)
    MAX_DRAWDOWN: float = 15.0  # CRITICAL FIX: 15% max drawdown (from .env)
    MAX_PORTFOLIO_RISK: float = 10.0  # CRITICAL FIX: 10% portfolio risk (from .env)
    ERROR_THRESHOLD: int = 5  # Stricter error tolerance
    MAX_VOLATILITY: float = 0.4  # From .env: 40% volatility limit
    
    # Position management - RISK MITIGATION
    MAX_POSITION_PER_TOKEN: float = 0.003  # CRITICAL FIX: Max 0.003 SOL per token (from .env)
    MAX_SIMULTANEOUS_POSITIONS: int = 1  # CRITICAL FIX: Max 1 position for micro-capital

    # Signal settings (optimized for new token detection)
    SIGNAL_THRESHOLD: float = 0.1   # TEST: Ultra-low threshold for maximum trade opportunities
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
        'TRADING_MODE': ('TRADING_MODE', str),
        'INITIAL_PAPER_BALANCE': ('INITIAL_PAPER_BALANCE', float),
        
        # Paper Trading Specific Parameters
        'PAPER_MIN_MOMENTUM_THRESHOLD': ('PAPER_MIN_MOMENTUM_THRESHOLD', float),
        'PAPER_MIN_LIQUIDITY': ('PAPER_MIN_LIQUIDITY', float),
        'PAPER_TRADING_SLIPPAGE': ('PAPER_TRADING_SLIPPAGE', float),
        'PAPER_BASE_POSITION_SIZE': ('PAPER_BASE_POSITION_SIZE', float),
        'PAPER_MAX_POSITION_SIZE': ('PAPER_MAX_POSITION_SIZE', float),
        'PAPER_SIGNAL_THRESHOLD': ('PAPER_SIGNAL_THRESHOLD', float),
        
        # Token Price and Market Cap Settings
        # USD-based settings (primary)
        'MAX_TOKEN_PRICE_USD': ('MAX_TOKEN_PRICE_USD', float),
        'MIN_TOKEN_PRICE_USD': ('MIN_TOKEN_PRICE_USD', float),
        'MAX_MARKET_CAP_USD': ('MAX_MARKET_CAP_USD', float),
        'MIN_MARKET_CAP_USD': ('MIN_MARKET_CAP_USD', float),
        
        # Legacy SOL-based settings (backward compatibility)
        'MAX_TOKEN_PRICE_SOL': ('MAX_TOKEN_PRICE_SOL', float),
        'MIN_TOKEN_PRICE_SOL': ('MIN_TOKEN_PRICE_SOL', float),
        'MAX_MARKET_CAP_SOL': ('MAX_MARKET_CAP_SOL', float),
        'MIN_MARKET_CAP_SOL': ('MIN_MARKET_CAP_SOL', float),
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
        
        # Enhanced Exit Manager Settings
        'DYNAMIC_STOP_LOSS_BASE': ('DYNAMIC_STOP_LOSS_BASE', float),
        'VOLATILITY_MULTIPLIER': ('VOLATILITY_MULTIPLIER', float),
        'TRAILING_STOP_ACTIVATION': ('TRAILING_STOP_ACTIVATION', float),
        'TRAILING_STOP_PERCENTAGE': ('TRAILING_STOP_PERCENTAGE', float),
        'MAX_HOLD_TIME_HOURS': ('MAX_HOLD_TIME_HOURS', int),
        'TAKE_PROFIT_LEVEL_1': ('TAKE_PROFIT_LEVEL_1', float),
        'TAKE_PROFIT_LEVEL_2': ('TAKE_PROFIT_LEVEL_2', float),
        'TAKE_PROFIT_LEVEL_3': ('TAKE_PROFIT_LEVEL_3', float),
        'POSITION_SCALE_1': ('POSITION_SCALE_1', float),
        'POSITION_SCALE_2': ('POSITION_SCALE_2', float),
        'POSITION_SCALE_3': ('POSITION_SCALE_3', float),
        
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