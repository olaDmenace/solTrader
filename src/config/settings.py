
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

    # Paper Trading settings
    position_manager: Any = None
    PAPER_TRADING: bool = True
    INITIAL_PAPER_BALANCE: float = 100.0  # Test with small amount
    MAX_POSITION_SIZE: float = 20.0
    MAX_SLIPPAGE=0.02
    MAX_TRADES_PER_DAY=10

    # Trading parameters
    MIN_BALANCE: float = 0.1  # Minimum SOL balance to maintain
    MAX_TRADE_SIZE: float = 20.0  # Maximum SOL per trade
    SLIPPAGE_TOLERANCE: float = 0.01  # 1% slippage tolerance

    # Position Management
    MAX_POSITIONS: int = 3  # Maximum number of open positions
    MIN_TRADE_SIZE: float = 0.1  # Minimum trade size
    INITIAL_CAPITAL: float = 100.0  # Starting capital
    PORTFOLIO_VALUE: float = 100.0  # Current portfolio value

    # Monitoring settings
    MONITOR_INTERVAL: float = 2.0  # Seconds between transaction checks
    STALE_THRESHOLD: int = 300  # Consider data stale after 5 minutes

    # Scanner settings
    SCAN_INTERVAL: int = 60  # Seconds between token scans
    MIN_LIQUIDITY: float = 1000.0  # Minimum liquidity in SOL
    MIN_VOLUME_24H: float = 100.0  # Minimum 24h volume in SOL
    VOLUME_THRESHOLD: float = 100.0  # Volume threshold for entry
    MAX_PRICE_IMPACT: float = 1.0  # Maximum allowable price impact

    # Notification settings
    DISCORD_WEBHOOK_URL: Optional[str] = None
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

    # Risk management
    MAX_DAILY_TRADES: int = 5
    MAX_DAILY_LOSS: float = 0.5  # Maximum loss in SOL per day
    STOP_LOSS_PERCENTAGE: float = 0.05  # 5% stop loss
    TAKE_PROFIT_PERCENTAGE: float = 0.1  # 10% take profit
    MAX_DRAWDOWN: float = 5.0  # Maximum drawdown percentage
    MAX_PORTFOLIO_RISK: float = 5.0  # Maximum portfolio risk
    ERROR_THRESHOLD: int = 5  # Maximum errors before stopping
    MAX_VOLATILITY: float = 0.3  # 30% annualized volatility threshold

    # Signal settings
    SIGNAL_THRESHOLD: float = 0.7  # Minimum signal strength (0-1)
    MIN_SIGNAL_INTERVAL: int = 300  # Minimum seconds between signals
    MAX_SIGNALS_PER_HOUR: int = 5

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
    CLOSE_POSITIONS_ON_STOP: bool = False  # Whether to close positions when stopping

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            key: getattr(self, key)
            for key in self.__annotations__
            if hasattr(self, key)
        }

    def validate(self) -> bool:
        """Validate settings"""
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
            if self.MONITOR_INTERVAL <= 0 or self.STALE_THRESHOLD <= 0:
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
        'SCAN_INTERVAL': ('SCANNER_INTERVAL', int),
        'STALE_THRESHOLD': ('STALE_THRESHOLD', int),
        'MIN_LIQUIDITY': ('MIN_LIQUIDITY', float),
        'MIN_VOLUME_24H': ('MIN_VOLUME_24H', float),
        'MAX_DAILY_TRADES': ('MAX_DAILY_TRADES', int),
        'MAX_DAILY_LOSS': ('MAX_DAILY_LOSS', float),
        'STOP_LOSS_PERCENTAGE': ('STOP_LOSS_PERCENTAGE', float),
        'TAKE_PROFIT_PERCENTAGE': ('TAKE_PROFIT_PERCENTAGE', float),

        # New mappings
        'PAPER_TRADING': ('PAPER_TRADING', lambda x: x.lower() == 'true'),
        'INITIAL_PAPER_BALANCE': ('INITIAL_PAPER_BALANCE', float),
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