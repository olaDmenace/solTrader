import logging
from typing import Any
# from .strategy import TradingSettings  # Import from your settings module
from ..config.settings import Settings  # Update the import path based on your project structure


logger = logging.getLogger(__name__)

class PositionSizer:
    """Handles position size calculations based on various factors"""
    
    def __init__(self, settings: Settings) -> None:
        """Initialize position sizer with settings"""
        self.settings = settings
        self.max_position_size = float(settings.MAX_POSITION_SIZE)

    def calculate_size(self, 
                      price: float, 
                      volatility: float, 
                      balance: float,
                      risk_score: float) -> float:
        """Calculate position size based on multiple factors"""
        try:
            if any(v < 0 for v in [price, volatility, balance, risk_score]):
                raise ValueError("All parameters must be non-negative")
            if risk_score > 100:
                raise ValueError("Risk score must be between 0 and 100")

            base_size = balance * float(self.settings.MAX_TRADE_SIZE)
            volatility_factor = max(0, 1 - (volatility / 100))
            risk_factor = max(0, 1 - (risk_score / 100))

            final_size = base_size * volatility_factor * risk_factor
            return min(final_size, self.max_position_size)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0