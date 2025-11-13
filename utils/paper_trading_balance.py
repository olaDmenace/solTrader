"""
Paper Trading Balance Manager
Maintains paper trading balance in USD terms while handling SOL conversions
"""
import logging
from typing import Optional
from utils.currency_formatter import currency_formatter
from src.utils.price_manager import DynamicPriceManager

logger = logging.getLogger(__name__)

class PaperTradingBalance:
    """Manages paper trading balance in USD with SOL conversion capabilities"""
    
    def __init__(self, initial_usd_balance: float = 200.0):
        self.usd_balance = initial_usd_balance
        self.price_manager = DynamicPriceManager()
        self._current_sol_price: Optional[float] = None
    
    async def get_current_sol_price(self) -> float:
        """Get current SOL price in USD"""
        if not self._current_sol_price:
            self._current_sol_price = await self.price_manager.get_sol_usd_price()
        return self._current_sol_price
    
    async def get_balance_usd(self) -> float:
        """Get balance in USD"""
        return self.usd_balance
    
    async def get_balance_sol(self) -> float:
        """Get balance in SOL equivalent"""
        sol_price = await self.get_current_sol_price()
        return self.usd_balance / sol_price
    
    async def can_afford_usd(self, amount_usd: float) -> bool:
        """Check if can afford amount in USD"""
        return amount_usd <= self.usd_balance
    
    async def can_afford_sol(self, amount_sol: float) -> bool:
        """Check if can afford amount in SOL (converted to USD)"""
        sol_price = await self.get_current_sol_price()
        amount_usd = amount_sol * sol_price
        return amount_usd <= self.usd_balance
    
    async def deduct_usd(self, amount_usd: float) -> bool:
        """Deduct USD amount from balance"""
        if not await self.can_afford_usd(amount_usd):
            return False
        self.usd_balance -= amount_usd
        logger.info(f"[PAPER] Balance: ${self.usd_balance:.2f} (deducted ${amount_usd:.2f})")
        return True
    
    async def deduct_sol(self, amount_sol: float) -> bool:
        """Deduct SOL amount (converted to USD) from balance"""
        sol_price = await self.get_current_sol_price()
        amount_usd = amount_sol * sol_price
        return await self.deduct_usd(amount_usd)
    
    async def add_usd(self, amount_usd: float):
        """Add USD amount to balance"""
        self.usd_balance += amount_usd
        logger.info(f"[PAPER] Balance: ${self.usd_balance:.2f} (added ${amount_usd:.2f})")
    
    async def add_sol(self, amount_sol: float):
        """Add SOL amount (converted to USD) to balance"""
        sol_price = await self.get_current_sol_price()
        amount_usd = amount_sol * sol_price
        await self.add_usd(amount_usd)
    
    async def get_formatted_balance(self) -> str:
        """Get formatted balance string"""
        sol_price = await self.get_current_sol_price()
        sol_balance = self.usd_balance / sol_price
        return f"${self.usd_balance:.2f} (~{sol_balance:.4f} SOL)"
    
    def reset_balance(self, new_usd_balance: float = 200.0):
        """Reset balance to new USD amount"""
        self.usd_balance = new_usd_balance
        logger.info(f"[PAPER] Balance reset to ${self.usd_balance:.2f}")

# Global instance for use throughout the system
paper_balance = PaperTradingBalance(200.0)  # Start with $200