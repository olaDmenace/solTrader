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

    # Paper Trading settings - LOADED FROM .ENV
    position_manager: Any = None
    PAPER_TRADING: bool = False  # Loaded from .env
    INITIAL_PAPER_BALANCE: float = 100.0  # Loaded from .env
    MAX_POSITION_SIZE: float = 0.05  # Loaded from .env
    
    # Helius configuration (fallback RPC)
    HELIUS_RPC_URL: Optional[str] = None
    
    # Jupiter configuration
    JUPITER_QUOTE_API_URL: str = "https://quote-api.jup.ag/v6/quote"
    JUPITER_SWAP_API_URL: str = "https://quote-api.jup.ag/v6/swap"
    JUPITER_PRIORITY_FEE_API_URL: str = "https://quote-api.jup.ag/v6/swap-instructions"
    
    # Trading parameters - LOADED FROM .ENV (defaults are fallbacks only)
    MIN_LIQUIDITY_USD: float = 10000.0
    MIN_LIQUIDITY: float = 5.0
    MIN_VOLUME_24H: float = 10.0
    VOLUME_THRESHOLD: float = 10.0
    MIN_MOMENTUM_PERCENTAGE: float = 2.0
    MAX_TOKEN_AGE_HOURS: float = 24.0
    
    # Scanner settings  
    SCAN_INTERVAL: int = 5
    SCANNER_INTERVAL: int = 300  # From .env
    NEW_TOKEN_MAX_AGE_MINUTES: int = 2880
    MIN_CONTRACT_SCORE: int = 70
    
    # Price ranges
    MAX_TOKEN_PRICE_USD: float = 2.0
    MIN_TOKEN_PRICE_USD: float = 0.00001
    MAX_MARKET_CAP_USD: float = 10000000.0
    MIN_MARKET_CAP_USD: float = 100000.0
    
    # Legacy SOL-based for compatibility
    MAX_TOKEN_PRICE_SOL: float = 0.01
    MIN_TOKEN_PRICE_SOL: float = 0.000001
    
    # Trading timeouts and exits
    MAX_HOLD_TIME_MINUTES: int = 180
    FAST_EXIT_THRESHOLD: float = -0.1
    MOMENTUM_EXIT_ENABLED: bool = True
    MOMENTUM_THRESHOLD: float = -0.03
    RSI_OVERBOUGHT: float = 80.0
    
    # Mean reversion strategy
    ENABLE_MEAN_REVERSION: bool = False
    MEAN_REVERSION_RSI_OVERSOLD: float = 20.0
    MEAN_REVERSION_RSI_OVERBOUGHT: float = 80.0
    
    # Signal generation settings
    SIGNAL_THRESHOLD: float = 0.5
    PAPER_SIGNAL_THRESHOLD: float = 0.3
    MOMENTUM_LOOKBACK_HOURS: int = 24
    
    # Risk management settings (loaded from .env)
    MAX_DAILY_TRADES: int = 50
    MAX_TRADES_PER_DAY: int = 15
    MAX_DAILY_LOSS: float = 10.0
    MAX_DRAWDOWN: float = 0.20
    MAX_PORTFOLIO_RISK: float = 10.0
    POSITION_MONITOR_INTERVAL: float = 3.0
    STALE_THRESHOLD: int = 300
    ERROR_THRESHOLD: float = 0.1
    STOP_LOSS_PERCENTAGE: float = 0.15
    MAX_POSITIONS: int = 10
    MAX_VOLATILITY: float = 0.4
    
    # Trading limits (loaded from .env)
    MIN_BALANCE: float = 1.0
    MAX_TRADE_SIZE: float = 0.1
    SLIPPAGE_TOLERANCE: float = 0.005
    MIN_TRADE_SIZE: float = 0.001
    INITIAL_CAPITAL: float = 200.0
    MONITOR_INTERVAL: float = 3.0
    
    # Technical analysis weights (loaded from .env)
    VOLUME_WEIGHT: float = 0.3
    LIQUIDITY_WEIGHT: float = 0.3  
    MOMENTUM_WEIGHT: float = 0.2
    MARKET_IMPACT_WEIGHT: float = 0.2
    
    # Signal settings (loaded from .env)
    MAX_SIGNALS_PER_HOUR: int = 15
    
    # Risk management settings
    MIN_BALANCE: float = 1.0
    MAX_TRADE_SIZE: float = 0.1
    SLIPPAGE_TOLERANCE: float = 0.005
    MAX_POSITIONS: int = 10
    MIN_TRADE_SIZE: float = 0.001
    INITIAL_CAPITAL: float = 200.0
    MONITOR_INTERVAL: float = 3.0
    
    # Technical analysis weights
    VOLUME_WEIGHT: float = 0.3
    LIQUIDITY_WEIGHT: float = 0.3
    MOMENTUM_WEIGHT: float = 0.2
    MARKET_IMPACT_WEIGHT: float = 0.2
    
    # Price movement thresholds
    MIN_PRICE_CHANGE: float = 0.02
    MAX_PRICE_CHANGE: float = 0.20
    MOMENTUM_LOOKBACK: int = 12
    
    # Operations settings
    CLOSE_POSITIONS_ON_STOP: bool = True
    PRIORITY_FEE_MULTIPLIER: float = 2.0
    MAX_GAS_PRICE: int = 200
    ERROR_THRESHOLD: float = 0.1  # 10% error threshold
    STOP_LOSS_PERCENTAGE: float = 0.15  # 15% stop loss
    MAX_PORTFOLIO_RISK: float = 10.0  # 10% portfolio risk
    MAX_SIGNALS_PER_HOUR: int = 15  # Signals per hour limit
    
    # Email notification settings
    EMAIL_ENABLED: bool = False
    EMAIL_USER: str = ""
    EMAIL_FROM: str = ""
    EMAIL_TO: str = ""
    EMAIL_PASSWORD: str = ""
    EMAIL_SMTP_SERVER: str = "smtp.gmail.com"
    EMAIL_SMTP_PORT: int = 587
    
    # Additional legacy compatibility settings
    VOLUME_GROWTH_THRESHOLD: float = 0.0
    HIGH_MOMENTUM_BYPASS: float = 500.0
    MEDIUM_MOMENTUM_BYPASS: float = 100.0
    PROFIT_PROTECTION_THRESHOLD: float = 0.2
    MAX_PRICE_IMPACT: float = 2.0
    RAYDIUM_PROGRAM_ID: str = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
    JUPITER_API_URL: str = "https://quote-api.jup.ag/v6"
    SOLANA_ONLY: bool = True
    EXCLUDE_MAJOR_TOKENS: bool = True
    MAX_SLIPPAGE: float = 0.005  # 0.5%
    MIN_TOKEN_AGE_MINUTES: int = 30
    MOMENTUM_THRESHOLD: float = 0.05  # 5%
    
    # Position sizing
    SMALL_POSITION_THRESHOLD: float = 0.001  # $1
    MEDIUM_POSITION_THRESHOLD: float = 0.005  # $5
    LARGE_POSITION_THRESHOLD: float = 0.01   # $10
    
    # Risk management
    STOP_LOSS_THRESHOLD: float = -0.15  # -15%
    TAKE_PROFIT_THRESHOLD: float = 0.25  # +25%
    EMERGENCY_LOSS_THRESHOLD: float = -0.9999  # Emergency circuit breaker
    
    # Strategy parameters
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD_DEV: float = 2.0
    
    # Timing settings
    SCAN_INTERVAL: int = 30  # seconds
    TOKEN_AGE_THRESHOLD_MINUTES: int = 30
    MAX_SIMULTANEOUS_TRADES: int = 5
    TRADE_COOLDOWN_SECONDS: int = 60
    
    # Performance settings
    ENABLE_CACHING: bool = True
    CACHE_DURATION: int = 300  # 5 minutes
    MAX_RETRY_ATTEMPTS: int = 3
    REQUEST_TIMEOUT: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    ENABLE_TRADE_LOGGING: bool = True
    ENABLE_PERFORMANCE_LOGGING: bool = True
    
    # Monitoring
    HEALTH_CHECK_INTERVAL: int = 300  # 5 minutes
    ENABLE_EMAIL_NOTIFICATIONS: bool = False
    DASHBOARD_HOST: str = "0.0.0.0"
    DASHBOARD_PORT: int = 5000
    MIN_PORTFOLIO_VALUE: float = 10.0
    PORTFOLIO_VALUE: float = 200.0
    EMAIL_PORT: int = 587
    TAKE_PROFIT_PERCENTAGE: float = 0.25  # 25% take profit
    STOP_LOSS_PERCENTAGE: float = 0.15   # 15% stop loss
    
    # Database settings
    DB_PATH: str = "data/trading.db"
    BACKUP_INTERVAL_MINUTES: int = 60
    
    def __post_init__(self):
        """Validate settings after initialization"""
        if not self.ALCHEMY_RPC_URL:
            raise ValueError("ALCHEMY_RPC_URL is required")
        if not self.WALLET_ADDRESS:
            raise ValueError("WALLET_ADDRESS is required")
        
        # Ensure reasonable limits
        if self.MAX_POSITION_SIZE > 0.1:  # 10%
            logger.warning(f"MAX_POSITION_SIZE of {self.MAX_POSITION_SIZE} is very high - consider reducing")
        
        if self.MAX_SLIPPAGE > 0.02:  # 2%
            logger.warning(f"MAX_SLIPPAGE of {self.MAX_SLIPPAGE} is high - expect poor execution")

def load_settings() -> Settings:
    """Load settings from environment variables"""
    load_dotenv(override=True)
    
    # Required environment variables
    alchemy_rpc_url = os.getenv("ALCHEMY_RPC_URL")
    wallet_address = os.getenv("WALLET_ADDRESS")
    
    if not alchemy_rpc_url:
        raise ValueError("ALCHEMY_RPC_URL environment variable is required")
    if not wallet_address:
        raise ValueError("WALLET_ADDRESS environment variable is required")
    
    # Create settings with environment overrides
    settings = Settings(
        ALCHEMY_RPC_URL=alchemy_rpc_url,
        WALLET_ADDRESS=wallet_address,
        HELIUS_RPC_URL=os.getenv("HELIUS_RPC_URL"),
        
        # Paper trading override
        PAPER_TRADING=os.getenv("PAPER_TRADING", "false").lower() == "true",
        INITIAL_PAPER_BALANCE=float(os.getenv("INITIAL_PAPER_BALANCE", "200.0")),
        MAX_POSITION_SIZE=float(os.getenv("MAX_POSITION_SIZE", "0.05")),
        
        # Load ALL missing settings from .env
        MAX_SLIPPAGE=float(os.getenv("MAX_SLIPPAGE", "1.0")),
        MAX_TRADES_PER_DAY=int(os.getenv("MAX_TRADES_PER_DAY", "15")),
        MAX_DAILY_TRADES=int(os.getenv("MAX_TRADES_PER_DAY", "15")),  # Legacy alias
        MAX_DAILY_LOSS=float(os.getenv("MAX_DAILY_LOSS", "10.0")),
        MAX_DRAWDOWN=float(os.getenv("MAX_DRAWDOWN", "15.0")),
        MAX_PORTFOLIO_RISK=float(os.getenv("MAX_PORTFOLIO_RISK", "10.0")),
        
        # Scanner settings from .env
        MIN_LIQUIDITY=float(os.getenv("MIN_LIQUIDITY", "0.5")),
        MIN_VOLUME_24H=float(os.getenv("MIN_VOLUME_24H", "1.0")),
        SCANNER_INTERVAL=int(os.getenv("SCANNER_INTERVAL", "300")),
        
        # Risk management from .env
        STOP_LOSS_PERCENTAGE=float(os.getenv("STOP_LOSS_PERCENTAGE", "0.15")),
        MAX_POSITIONS=int(os.getenv("MAX_POSITIONS", "3")),
        MAX_VOLATILITY=float(os.getenv("MAX_VOLATILITY", "0.4")),
        
        # Technical analysis weights from .env
        VOLUME_WEIGHT=float(os.getenv("VOLUME_WEIGHT", "0.3")),
        LIQUIDITY_WEIGHT=float(os.getenv("LIQUIDITY_WEIGHT", "0.3")),
        MOMENTUM_WEIGHT=float(os.getenv("MOMENTUM_WEIGHT", "0.2")),
        MARKET_IMPACT_WEIGHT=float(os.getenv("MARKET_IMPACT_WEIGHT", "0.2")),
        
        # Signal settings from .env
        MAX_SIGNALS_PER_HOUR=int(os.getenv("MAX_SIGNALS_PER_HOUR", "10")),
        
        # Trading limits from .env
        MIN_BALANCE=float(os.getenv("TRADING_MIN_BALANCE", "1.0")),
        MAX_TRADE_SIZE=float(os.getenv("TRADING_MAX_SIZE", "0.1")),
        SLIPPAGE_TOLERANCE=float(os.getenv("TRADING_SLIPPAGE", "0.005")),
        MIN_TRADE_SIZE=float(os.getenv("MIN_TRADE_SIZE", "0.001")),
        INITIAL_CAPITAL=float(os.getenv("INITIAL_PAPER_BALANCE", "200.0")),
        MONITOR_INTERVAL=float(os.getenv("POSITION_MONITOR_INTERVAL", "3.0")),
        
        # Load ALL remaining settings from .env (no more hardcoding!)
        
        # Core Trading Parameters
        MIN_LIQUIDITY_USD=float(os.getenv("MIN_LIQUIDITY_USD", "10000.0")),
        VOLUME_THRESHOLD=float(os.getenv("VOLUME_THRESHOLD", "10.0")),
        MIN_MOMENTUM_PERCENTAGE=float(os.getenv("MIN_MOMENTUM_PERCENTAGE", "2.0")),
        MAX_TOKEN_AGE_HOURS=float(os.getenv("MAX_TOKEN_AGE_HOURS", "24.0")),
        
        # Scanner Settings
        SCAN_INTERVAL=int(os.getenv("SCAN_INTERVAL", "5")),
        NEW_TOKEN_MAX_AGE_MINUTES=int(os.getenv("NEW_TOKEN_MAX_AGE_MINUTES", "2880")),
        MIN_CONTRACT_SCORE=int(os.getenv("MIN_CONTRACT_SCORE", "70")),
        
        # Price Ranges
        MAX_TOKEN_PRICE_USD=float(os.getenv("MAX_TOKEN_PRICE_USD", "2.0")),
        MIN_TOKEN_PRICE_USD=float(os.getenv("MIN_TOKEN_PRICE_USD", "0.00001")),
        MAX_MARKET_CAP_USD=float(os.getenv("MAX_MARKET_CAP_USD", "10000000.0")),
        MIN_MARKET_CAP_USD=float(os.getenv("MIN_MARKET_CAP_USD", "100000.0")),
        
        # Legacy SOL-based
        MAX_TOKEN_PRICE_SOL=float(os.getenv("MAX_TOKEN_PRICE_SOL", "0.01")),
        MIN_TOKEN_PRICE_SOL=float(os.getenv("MIN_TOKEN_PRICE_SOL", "0.000001")),
        
        # Trading Timeouts and Exits
        MAX_HOLD_TIME_MINUTES=int(os.getenv("MAX_HOLD_TIME_MINUTES", "180")),
        FAST_EXIT_THRESHOLD=float(os.getenv("FAST_EXIT_THRESHOLD", "-0.1")),
        MOMENTUM_EXIT_ENABLED=os.getenv("MOMENTUM_EXIT_ENABLED", "true").lower() == "true",
        MOMENTUM_THRESHOLD=float(os.getenv("MOMENTUM_THRESHOLD", "-0.03")),
        RSI_OVERBOUGHT=float(os.getenv("RSI_OVERBOUGHT", "80.0")),
        
        # Mean Reversion Strategy
        ENABLE_MEAN_REVERSION=os.getenv("ENABLE_MEAN_REVERSION", "false").lower() == "true",
        MEAN_REVERSION_RSI_OVERSOLD=float(os.getenv("MEAN_REVERSION_RSI_OVERSOLD", "20.0")),
        MEAN_REVERSION_RSI_OVERBOUGHT=float(os.getenv("MEAN_REVERSION_RSI_OVERBOUGHT", "80.0")),
        
        # Signal Generation
        SIGNAL_THRESHOLD=float(os.getenv("SIGNAL_THRESHOLD", "0.5")),
        PAPER_SIGNAL_THRESHOLD=float(os.getenv("PAPER_SIGNAL_THRESHOLD", "0.3")),
        MOMENTUM_LOOKBACK_HOURS=int(os.getenv("MOMENTUM_LOOKBACK_HOURS", "24")),
        
        # System Monitoring
        ERROR_THRESHOLD=float(os.getenv("ERROR_THRESHOLD", "0.1")),
        STALE_THRESHOLD=int(os.getenv("STALE_THRESHOLD", "300")),
        
        # Email Configuration
        EMAIL_ENABLED=os.getenv("EMAIL_ENABLED", "false").lower() == "true",
        EMAIL_USER=os.getenv("EMAIL_USER", ""),
        EMAIL_FROM=os.getenv("EMAIL_FROM", ""),
        EMAIL_TO=os.getenv("EMAIL_TO", ""),
        EMAIL_PASSWORD=os.getenv("EMAIL_PASSWORD", ""),
        EMAIL_SMTP_SERVER=os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com"),
        EMAIL_SMTP_PORT=int(os.getenv("EMAIL_SMTP_PORT", "587")),
        
        # Legacy Compatibility
        VOLUME_GROWTH_THRESHOLD=float(os.getenv("VOLUME_GROWTH_THRESHOLD", "0.0")),
        HIGH_MOMENTUM_BYPASS=float(os.getenv("HIGH_MOMENTUM_BYPASS", "500.0")),
        MEDIUM_MOMENTUM_BYPASS=float(os.getenv("MEDIUM_MOMENTUM_BYPASS", "100.0")),
        PROFIT_PROTECTION_THRESHOLD=float(os.getenv("PROFIT_PROTECTION_THRESHOLD", "0.2")),
        MAX_PRICE_IMPACT=float(os.getenv("MAX_PRICE_IMPACT", "2.0")),
        RAYDIUM_PROGRAM_ID=os.getenv("RAYDIUM_PROGRAM_ID", "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"),
        JUPITER_API_URL=os.getenv("JUPITER_API_URL", "https://quote-api.jup.ag/v6"),
        SOLANA_ONLY=os.getenv("SOLANA_ONLY", "true").lower() == "true",
        EXCLUDE_MAJOR_TOKENS=os.getenv("EXCLUDE_MAJOR_TOKENS", "true").lower() == "true",
        MIN_TOKEN_AGE_MINUTES=int(os.getenv("MIN_TOKEN_AGE_MINUTES", "30")),
        
        # Price Movement Thresholds
        MIN_PRICE_CHANGE=float(os.getenv("MIN_PRICE_CHANGE", "0.02")),
        MAX_PRICE_CHANGE=float(os.getenv("MAX_PRICE_CHANGE", "0.20")),
        MOMENTUM_LOOKBACK=int(os.getenv("MOMENTUM_LOOKBACK", "12")),
        
        # Operations Settings
        CLOSE_POSITIONS_ON_STOP=os.getenv("CLOSE_POSITIONS_ON_STOP", "true").lower() == "true",
        PRIORITY_FEE_MULTIPLIER=float(os.getenv("PRIORITY_FEE_MULTIPLIER", "2.0")),
        MAX_GAS_PRICE=int(os.getenv("MAX_GAS_PRICE", "200")),
        
        # Strategy Parameters
        RSI_PERIOD=int(os.getenv("RSI_PERIOD", "14")),
        RSI_OVERSOLD=float(os.getenv("RSI_OVERSOLD", "30.0")),
        
        # Position Sizing
        SMALL_POSITION_THRESHOLD=float(os.getenv("SMALL_POSITION_THRESHOLD", "0.001")),
        MEDIUM_POSITION_THRESHOLD=float(os.getenv("MEDIUM_POSITION_THRESHOLD", "0.005")),
        LARGE_POSITION_THRESHOLD=float(os.getenv("LARGE_POSITION_THRESHOLD", "0.01")),
        
        # Risk Thresholds
        STOP_LOSS_THRESHOLD=float(os.getenv("STOP_LOSS_THRESHOLD", "-0.15")),
        TAKE_PROFIT_THRESHOLD=float(os.getenv("TAKE_PROFIT_THRESHOLD", "0.25")),
        EMERGENCY_LOSS_THRESHOLD=float(os.getenv("EMERGENCY_LOSS_THRESHOLD", "-0.9999")),
        
        # Timing
        MAX_SIMULTANEOUS_TRADES=int(os.getenv("MAX_SIMULTANEOUS_TRADES", "5")),
        TRADE_COOLDOWN_SECONDS=int(os.getenv("TRADE_COOLDOWN_SECONDS", "60")),
        
        # Logging
        LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
        ENABLE_TRADE_LOGGING=os.getenv("ENABLE_TRADE_LOGGING", "true").lower() == "true",
        ENABLE_PERFORMANCE_LOGGING=os.getenv("ENABLE_PERFORMANCE_LOGGING", "true").lower() == "true",
        
        # Monitoring
        HEALTH_CHECK_INTERVAL=int(os.getenv("HEALTH_CHECK_INTERVAL", "300")),
        ENABLE_EMAIL_NOTIFICATIONS=os.getenv("ENABLE_EMAIL_NOTIFICATIONS", "false").lower() == "true",
        DASHBOARD_HOST=os.getenv("DASHBOARD_HOST", "0.0.0.0"),
        DASHBOARD_PORT=int(os.getenv("DASHBOARD_PORT", "5000")),
        MIN_PORTFOLIO_VALUE=float(os.getenv("MIN_PORTFOLIO_VALUE", "10.0")),
        PORTFOLIO_VALUE=float(os.getenv("PORTFOLIO_VALUE", "200.0")),
        EMAIL_PORT=int(os.getenv("EMAIL_PORT", "587")),
        TAKE_PROFIT_PERCENTAGE=float(os.getenv("TAKE_PROFIT_PERCENTAGE", "0.25")),
        
        # Database
        DB_PATH=os.getenv("DB_PATH", "data/trading.db"),
        BACKUP_INTERVAL_MINUTES=int(os.getenv("BACKUP_INTERVAL_MINUTES", "60")),
        
        # Jupiter Configuration URLs (loaded from .env)
        JUPITER_QUOTE_API_URL=os.getenv("JUPITER_QUOTE_API_URL", "https://quote-api.jup.ag/v6/quote"),
        JUPITER_SWAP_API_URL=os.getenv("JUPITER_SWAP_API_URL", "https://quote-api.jup.ag/v6/swap"),
        JUPITER_PRIORITY_FEE_API_URL=os.getenv("JUPITER_PRIORITY_FEE_API_URL", "https://quote-api.jup.ag/v6/swap-instructions")
    )
    
    logger.info(f"Settings loaded - Paper Trading: {settings.PAPER_TRADING}, Max Position: {settings.MAX_POSITION_SIZE}")
    return settings