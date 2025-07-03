import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class EntryPoint:
    """Data class for trade entry points"""
    price: float
    confidence: float
    volume: float
    timestamp: datetime
    reason: str

@dataclass
class ExitPoint:
    """Data class for trade exit points"""
    price: float
    type: str  # 'take_profit' or 'stop_loss'
    urgency: str  # 'low', 'medium', 'high'
    reason: str

class TradeAnalyzer:
    """Analyzes trading opportunities and manages trade execution"""

    def __init__(self, jupiter_client: Any, trend_detector: Any) -> None:
        """
        Initialize TradeAnalyzer

        Args:
            jupiter_client: Jupiter API client
            trend_detector: TrendDetector instance
        """
        self.jupiter = jupiter_client
        self.trend_detector = trend_detector

        # Configuration parameters
        self.min_confidence: float = 75.0
        self.min_volume: float = 1000.0
        self.default_slippage: float = 0.02  # 2%
        self.take_profit_percentage: float = 0.15  # 15%
        self.stop_loss_percentage: float = 0.05  # 5%

    async def analyze_entry_point(self, token_address: str) -> Optional[EntryPoint]:
        """
        Analyze potential entry point for a token

        Args:
            token_address: Token mint address

        Returns:
            Optional[EntryPoint]: Entry point details if favorable, None otherwise
        """
        try:
            # Get token data and trend analysis
            token_data = await self.jupiter.get_token_info(token_address)
            trend = await self.trend_detector.analyze_trend(token_address)

            if not token_data or not trend:
                logger.debug(f"Missing data for token {token_address}")
                return None

            # Calculate confidence score
            confidence = self._calculate_entry_confidence(token_data, trend)
            volume_24h = float(token_data.get('volume24h', 0))

            # Check minimum requirements
            if volume_24h < self.min_volume:
                logger.info(f"Volume too low: ${volume_24h}")
                return None

            if confidence < self.min_confidence:
                logger.info(f"Confidence too low: {confidence}")
                return None

            # Collect entry reasons
            entry_reasons: List[str] = []
            if trend.trend_direction == "uptrend":
                entry_reasons.append("Upward trend detected")
            if trend.momentum_score > 0.7:
                entry_reasons.append("Strong momentum")
            if trend.volume_trend == "increasing":
                entry_reasons.append("Increasing volume")

            if not entry_reasons:
                logger.debug("No compelling reasons for entry")
                return None

            return EntryPoint(
                price=float(token_data.get('price', 0)),
                confidence=confidence,
                volume=volume_24h,
                timestamp=datetime.now(),
                reason=", ".join(entry_reasons)
            )

        except Exception as e:
            logger.error(f"Error analyzing entry point: {str(e)}")
            return None

    async def calculate_exit_points(self, entry_price: float) -> Dict[str, float]:
        """
        Calculate basic exit points for a trade

        Args:
            entry_price: Entry price in SOL

        Returns:
            Dict with take_profit and stop_loss levels
        """
        try:
            take_profit = entry_price * (1 + self.take_profit_percentage)
            stop_loss = entry_price * (1 - self.stop_loss_percentage)

            return {
                'take_profit': take_profit,
                'stop_loss': stop_loss
            }

        except Exception as e:
            logger.error(f"Error calculating exit points: {str(e)}")
            return {
                'take_profit': 0.0,
                'stop_loss': 0.0
            }

    async def determine_exit_strategy(self, 
                                    entry_price: float,
                                    token_address: str) -> Dict[str, ExitPoint]:
        """
        Determine comprehensive exit strategy

        Args:
            entry_price: Entry price in SOL
            token_address: Token mint address

        Returns:
            Dict containing exit points for different scenarios
        """
        try:
            trend = await self.trend_detector.analyze_trend(token_address)
            exit_points: Dict[str, ExitPoint] = {}

            # Calculate take profit based on trend strength
            if trend and trend.confidence >= 80:
                # Strong trend - higher target
                take_profit = entry_price * 1.3
                exit_points['take_profit'] = ExitPoint(
                    price=take_profit,
                    type='take_profit',
                    urgency='low',
                    reason="Strong trend, higher profit target"
                )
            else:
                # Normal trend - conservative target
                take_profit = entry_price * 1.15
                exit_points['take_profit'] = ExitPoint(
                    price=take_profit,
                    type='take_profit',
                    urgency='medium',
                    reason="Normal trend, conservative target"
                )

            # Set stop loss based on market conditions
            stop_loss = entry_price * 0.95

            if trend and trend.momentum_score < 0.3:
                # Tighter stop loss in weak momentum
                stop_loss = entry_price * 0.97
                urgency = 'high'
                reason = "Weak momentum, tight stop loss"
            else:
                urgency = 'medium'
                reason = "Normal market conditions"

            exit_points['stop_loss'] = ExitPoint(
                price=stop_loss,
                type='stop_loss',
                urgency=urgency,
                reason=reason
            )

            return exit_points

        except Exception as e:
            logger.error(f"Error determining exit strategy: {str(e)}")
            return {}

    def _calculate_entry_confidence(self, token_data: Dict[str, Any], trend: Any) -> float:
        """
        Calculate confidence score for entry

        Args:
            token_data: Token market data
            trend: Trend analysis results

        Returns:
            float: Confidence score (0-100)
        """
        score = 0.0

        # Trend-based scoring (0-40 points)
        if trend.trend_direction == "uptrend":
            score += 20
        if trend.momentum_score > 0.7:
            score += 20

        # Volume-based scoring (0-30 points)
        daily_volume = float(token_data.get('volume24h', 0))
        if daily_volume > 10000:
            score += 30
        elif daily_volume > 5000:
            score += 20
        elif daily_volume > 1000:
            score += 10

        # Liquidity scoring (0-30 points)
        liquidity = float(token_data.get('liquidity', 0))
        if liquidity > 50000:
            score += 30
        elif liquidity > 25000:
            score += 20
        elif liquidity > 10000:
            score += 10

        return score

    async def validate_entry_conditions(self, 
                                      token_address: str,
                                      max_slippage: Optional[float] = None) -> bool:
        """
        Validate additional entry conditions

        Args:
            token_address: Token mint address
            max_slippage: Maximum acceptable slippage, uses default if None

        Returns:
            bool: True if conditions are valid
        """
        try:
            if max_slippage is None:
                max_slippage = self.default_slippage

            # Get market depth
            market_depth = await self.jupiter.get_market_depth(token_address)

            if not market_depth:
                return False

            # Check if slippage would be acceptable
            estimated_slippage = self._estimate_slippage(market_depth)
            if estimated_slippage > max_slippage:
                logger.info(f"Slippage too high: {estimated_slippage:.2%}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating entry conditions: {str(e)}")
            return False

    def _estimate_slippage(self, market_depth: Dict[str, Any]) -> float:
        """
        Estimate potential slippage based on market depth

        Args:
            market_depth: Market depth data

        Returns:
            float: Estimated slippage ratio
        """
        try:
            bids = market_depth.get('bids', [])
            total_liquidity = sum(float(bid.get('size', 0)) for bid in bids)

            if total_liquidity == 0:
                return float('inf')

            return 1 / total_liquidity

        except Exception as e:
            logger.error(f"Error estimating slippage: {str(e)}")
            return float('inf')