import asyncio
import logging
from typing import Dict, List, Optional, TypeVar, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class MarketIndicators:
    """Data class for market indicators"""
    rsi: float
    volume_profile: float
    price_momentum: float
    liquidity_score: float
    volatility: float
    trend_strength: float

class MarketAnalyzer:
    """Analyzes market data for trading opportunities"""
    
    def __init__(self, jupiter_client: Any, settings: Any) -> None:
        """
        Initialize Market Analyzer
        
        Args:
            jupiter_client: Jupiter API client
            settings: Trading settings
        """
        self.jupiter = jupiter_client
        self.settings = settings
        self.lookback_periods: Dict[str, int] = {
            'short': 12,
            'medium': 26,
            'long': 50
        }
        self.price_cache: Dict[str, List[float]] = {}
        self.volume_cache: Dict[str, List[float]] = {}
        self.cache_expiry: int = 300  # 5 minutes

    async def analyze_market(self, token_address: str) -> Optional[MarketIndicators]:
        """
        Analyze market conditions for a token
        
        Args:
            token_address: Token address to analyze
            
        Returns:
            Optional[MarketIndicators]: Market analysis results or None if error
        """
        try:
            price_data = await self._get_price_data(token_address)
            volume_data = await self._get_volume_data(token_address)
            
            if not price_data or not volume_data:
                return None

            return MarketIndicators(
                rsi=self._calculate_rsi(price_data),
                volume_profile=self._analyze_volume_profile(volume_data),
                price_momentum=self._calculate_momentum(price_data),
                liquidity_score=self._calculate_liquidity_score(volume_data),
                volatility=self._calculate_volatility(price_data),
                trend_strength=self._calculate_trend_strength(price_data)
            )

        except Exception as e:
            logger.error(f"Market analysis error for {token_address}: {str(e)}")
            return None

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0

        np_prices = np.array(prices, dtype=np.float64)
        deltas = np.diff(np_prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def _analyze_volume_profile(self, volumes: List[float]) -> float:
        """Analyze volume profile and momentum"""
        if not volumes:
            return 0.0

        np_volumes = np.array(volumes, dtype=np.float64)
        recent_vol = float(np.mean(np_volumes[-5:]))
        hist_vol = float(np.mean(np_volumes[:-5])) if len(volumes) > 5 else recent_vol
        
        if hist_vol == 0:
            return 0.0
            
        return float(min(recent_vol / hist_vol, 5.0))

    def _calculate_momentum(self, prices: List[float], window: int = 10) -> float:
        """Calculate price momentum"""
        if len(prices) < window:
            return 0.0

        momentum = (prices[-1] - prices[-window]) / prices[-window]
        return float(min(max(momentum, -1.0), 1.0))

    def _calculate_liquidity_score(self, volumes: List[float]) -> float:
        """Calculate liquidity score based on volume"""
        if not volumes:
            return 0.0

        np_volumes = np.array(volumes, dtype=np.float64)
        avg_volume = float(np.mean(np_volumes))
        
        if avg_volume == 0:
            return 0.0
            
        consistency = 1 - (float(np.std(np_volumes)) / avg_volume)
        normalized_volume = min(avg_volume / self.settings.MIN_VOLUME_24H, 1.0)
        
        return float((consistency + normalized_volume) / 2)

    def _calculate_volatility(self, prices: List[float], window: int = 20) -> float:
        """Calculate price volatility"""
        if len(prices) < window:
            return 0.0

        np_prices = np.array(prices, dtype=np.float64)
        returns = np.diff(np_prices) / np_prices[:-1]
        volatility = float(np.std(returns) * np.sqrt(365))
        return float(min(volatility, 1.0))

    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength using moving averages"""
        if len(prices) < self.lookback_periods['long']:
            return 0.0

        np_prices = np.array(prices, dtype=np.float64)
        short_ma = float(np.mean(np_prices[-self.lookback_periods['short']:]))
        medium_ma = float(np.mean(np_prices[-self.lookback_periods['medium']:]))
        long_ma = float(np.mean(np_prices[-self.lookback_periods['long']:]))

        if short_ma > medium_ma > long_ma:
            trend = 1.0  # Strong uptrend
        elif short_ma < medium_ma < long_ma:
            trend = -1.0  # Strong downtrend
        else:
            trend = 0.0  # No clear trend

        return float(abs(trend))

    async def _get_price_data(self, token_address: str) -> Optional[List[float]]:
        """Get historical price data with caching"""
        try:
            if token_address in self.price_cache:
                return self.price_cache[token_address]

            history = await self.jupiter.get_price_history(token_address, interval='5m')
            if not history:
                return None

            prices = [float(price['price']) for price in history]
            self.price_cache[token_address] = prices
            
            self._schedule_cache_cleanup(token_address)
            return prices

        except Exception as e:
            logger.error(f"Error fetching price data: {str(e)}")
            return None

    async def _get_volume_data(self, token_address: str) -> Optional[List[float]]:
        """Get historical volume data with caching"""
        try:
            if token_address in self.volume_cache:
                return self.volume_cache[token_address]

            market_data = await self.jupiter.get_market_depth(token_address)
            if not market_data:
                return None

            # Note: Changed from 'recent_volume' to 'recent_volumes'
            volumes = []
            if 'recent_volumes' in market_data:
                volumes = [float(vol) for vol in market_data['recent_volumes']]
            
            if not volumes:
                return None

            self.volume_cache[token_address] = volumes
            self._schedule_cache_cleanup(token_address)
            return volumes

        except Exception as e:
            logger.error(f"Error fetching volume data: {str(e)}")
            return None

    def _schedule_cache_cleanup(self, token_address: str) -> None:
        """Schedule cleanup of cached data"""
        async def cleanup() -> None:
            await asyncio.sleep(self.cache_expiry)
            self.price_cache.pop(token_address, None)
            self.volume_cache.pop(token_address, None)

        asyncio.create_task(cleanup())