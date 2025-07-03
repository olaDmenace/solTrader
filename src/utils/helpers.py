import logging
from typing import Dict, Any, Optional
from decimal import Decimal
import time

logger = logging.getLogger(__name__)

class TradingHelpers:
    @staticmethod
    def calculate_position_size(
        balance: float,
        token_price: float,
        risk_percentage: float = 1.0,
        max_slippage: float = 0.5
    ) -> float:
        """
        Calculate safe position size based on balance and risk parameters
        
        Args:
            balance: Available balance in SOL
            token_price: Current token price in USD
            risk_percentage: Percentage of balance to risk (default 1%)
            max_slippage: Maximum allowed slippage (default 0.5%)
        """
        try:
            # Calculate maximum position based on risk
            max_position = balance * (risk_percentage / 100)
            
            # Adjust for slippage
            safe_position = max_position * (1 - (max_slippage / 100))
            
            return safe_position
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    @staticmethod
    def analyze_price_impact(
        market_depth: Dict[str, Any],
        trade_size: float
    ) -> Dict[str, float]:
        """
        Analyze price impact of a trade
        
        Args:
            market_depth: Market depth data from Jupiter
            trade_size: Intended trade size in input token
        """
        try:
            bids = market_depth.get('bids', [])
            total_liquidity = sum(bid['size'] for bid in bids)
            
            impact = (trade_size / total_liquidity) * 100 if total_liquidity > 0 else 100
            
            return {
                "price_impact": impact,
                "is_safe": impact < 1.0,  # Consider safe if impact < 1%
                "total_liquidity": total_liquidity
            }
        except Exception as e:
            logger.error(f"Error analyzing price impact: {str(e)}")
            return {
                "price_impact": 100,
                "is_safe": False,
                "total_liquidity": 0
            }

    @staticmethod
    def calculate_profit_targets(
        entry_price: float,
        position_size: float,
        risk_reward_ratio: float = 2.0
    ) -> Dict[str, float]:
        """
        Calculate take profit and stop loss levels
        
        Args:
            entry_price: Entry price of the token
            position_size: Position size in token amount
            risk_reward_ratio: Desired risk/reward ratio (default 2.0)
        """
        try:
            # Default to 5% stop loss
            stop_loss_pct = 5.0
            take_profit_pct = stop_loss_pct * risk_reward_ratio
            
            return {
                "entry_price": entry_price,
                "position_size": position_size,
                "stop_loss": entry_price * (1 - stop_loss_pct/100),
                "take_profit": entry_price * (1 + take_profit_pct/100),
                "max_loss": position_size * entry_price * (stop_loss_pct/100),
                "potential_profit": position_size * entry_price * (take_profit_pct/100)
            }
        except Exception as e:
            logger.error(f"Error calculating profit targets: {str(e)}")
            return {}

    @staticmethod
    def is_token_tradeable(
        token_data: Dict[str, Any],
        min_liquidity_usd: float = 10000,
        min_holders: int = 100,
        min_age_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Validate if a token is safe to trade
        
        Args:
            token_data: Token information from Jupiter
            min_liquidity_usd: Minimum USD liquidity required
            min_holders: Minimum number of holders required
            min_age_hours: Minimum token age in hours
        """
        try:
            current_time = time.time()
            token_age = (current_time - token_data.get('created_at', current_time)) / 3600
            
            checks = {
                "sufficient_liquidity": token_data.get('liquidity', 0) >= min_liquidity_usd,
                "enough_holders": token_data.get('holder_count', 0) >= min_holders,
                "age_requirement": token_age >= min_age_hours,
                "has_verified_contract": token_data.get('is_verified', False),
            }
            
            return {
                "is_tradeable": all(checks.values()),
                "checks": checks,
                "risk_score": sum(1 for check in checks.values() if check) / len(checks) * 100
            }
        except Exception as e:
            logger.error(f"Error validating token: {str(e)}")
            return {
                "is_tradeable": False,
                "checks": {},
                "risk_score": 0
            }