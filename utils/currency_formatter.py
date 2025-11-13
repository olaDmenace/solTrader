"""
Centralized Currency Formatting Utility
Ensures all currency display is consistently in USD throughout the system
"""
import logging
from typing import Union, Optional
from src.utils.price_manager import DynamicPriceManager

logger = logging.getLogger(__name__)

class CurrencyFormatter:
    """Centralized currency formatting to ensure consistent USD display"""
    
    def __init__(self):
        self.price_manager = DynamicPriceManager()
        self._current_sol_price: Optional[float] = None
    
    async def ensure_sol_price(self):
        """Ensure we have current SOL price"""
        if not self._current_sol_price:
            self._current_sol_price = await self.price_manager.get_sol_usd_price()
    
    async def sol_to_usd(self, sol_amount: Union[float, int, str]) -> float:
        """Convert SOL amount to USD"""
        try:
            await self.ensure_sol_price()
            sol_value = float(sol_amount) if sol_amount else 0.0
            return sol_value * self._current_sol_price
        except Exception as e:
            logger.error(f"Error converting SOL to USD: {e}")
            return 0.0
    
    async def format_currency_usd(self, amount: Union[float, int, str], 
                                 currency_type: str = "USD", 
                                 decimal_places: int = 2) -> str:
        """
        Format any currency amount to USD display
        Args:
            amount: The amount to format
            currency_type: "USD", "SOL", or "TOKEN" 
            decimal_places: Number of decimal places to show
        """
        try:
            if not amount:
                return "$0.00"
            
            value = float(amount)
            
            if currency_type == "SOL":
                # Convert SOL to USD
                value = await self.sol_to_usd(value)
            elif currency_type == "TOKEN":
                # For tokens, assume the value is already in USD equivalent
                pass
            # For USD, value is already correct
            
            return f"${value:,.{decimal_places}f}"
            
        except Exception as e:
            logger.error(f"Error formatting currency: {e}")
            return "$0.00"
    
    async def format_position_value(self, token_amount: float, token_price_sol: float) -> str:
        """Format position value in USD"""
        try:
            await self.ensure_sol_price()
            sol_value = token_amount * token_price_sol
            usd_value = sol_value * self._current_sol_price
            return f"${usd_value:,.2f}"
        except Exception as e:
            logger.error(f"Error formatting position value: {e}")
            return "$0.00"
    
    async def format_price_usd(self, price_sol: float) -> str:
        """Format token price in USD"""
        try:
            await self.ensure_sol_price()
            price_usd = price_sol * self._current_sol_price
            
            # Use appropriate decimal places based on price magnitude
            if price_usd >= 1.0:
                return f"${price_usd:.4f}"
            elif price_usd >= 0.01:
                return f"${price_usd:.6f}"
            else:
                return f"${price_usd:.8f}"
        except Exception as e:
            logger.error(f"Error formatting price: {e}")
            return "$0.00"
    
    async def format_volume_usd(self, volume_sol: float) -> str:
        """Format volume in USD"""
        try:
            await self.ensure_sol_price()
            volume_usd = volume_sol * self._current_sol_price
            
            # Use K, M, B suffixes for large volumes
            if volume_usd >= 1_000_000_000:
                return f"${volume_usd/1_000_000_000:.1f}B"
            elif volume_usd >= 1_000_000:
                return f"${volume_usd/1_000_000:.1f}M"
            elif volume_usd >= 1_000:
                return f"${volume_usd/1_000:.1f}K"
            else:
                return f"${volume_usd:.2f}"
        except Exception as e:
            logger.error(f"Error formatting volume: {e}")
            return "$0.00"
    
    async def format_market_cap_usd(self, market_cap_sol: float) -> str:
        """Format market cap in USD"""
        return await self.format_volume_usd(market_cap_sol)  # Same formatting logic
    
    async def get_current_sol_price(self) -> float:
        """Get current SOL price in USD"""
        await self.ensure_sol_price()
        return self._current_sol_price or 200.0

# Global formatter instance
currency_formatter = CurrencyFormatter()

# Convenience functions for easy import
async def format_usd(amount: float, currency_type: str = "USD") -> str:
    """Quick USD formatting"""
    return await currency_formatter.format_currency_usd(amount, currency_type)

async def format_price_usd(price_sol: float) -> str:
    """Quick price formatting"""
    return await currency_formatter.format_price_usd(price_sol)

async def format_volume_usd(volume_sol: float) -> str:
    """Quick volume formatting"""
    return await currency_formatter.format_volume_usd(volume_sol)