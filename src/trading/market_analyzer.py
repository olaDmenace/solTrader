import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta

from .technical_indicators import TechnicalIndicators, BollingerResult, MACDResult, IndicatorResult

logger = logging.getLogger(__name__)

# @dataclass
# class MarketIndicators:
#     rsi: float
#     volume_profile: float
#     price_momentum: float
#     liquidity_score: float
#     volatility: float
#     trend_strength: float

@dataclass
class MarketIndicators:
    rsi: float
    volume_profile: float
    price_momentum: float
    liquidity_score: float
    volatility: float
    trend_strength: float
    macd: Optional[MACDResult] = None
    bollinger: Optional[BollingerResult] = None
    signal_strength: float = 0.0

class MarketConditions:
    def __init__(self):
        self.volatility_threshold = 0.3
        self.volume_threshold = 1000.0
        self.min_liquidity = 5000.0
        self.trend_window = 20

    async def detect_market_regime(self, price_data: List[float], volume_data: List[float]) -> str:
        volatility = self._calculate_volatility(price_data)
        volume_trend = self._analyze_volume_trend(volume_data)
        price_trend = self._analyze_price_trend(price_data)
        
        if volatility > self.volatility_threshold:
            return "volatile"
        elif price_trend > 0.7 and volume_trend > 0:
            return "trending_up"
        elif price_trend < -0.7 and volume_trend < 0:
            return "trending_down"
        return "ranging"

    def _calculate_volatility(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns) * np.sqrt(365))

    def _analyze_volume_trend(self, volumes: List[float]) -> float:
        if len(volumes) < self.trend_window:
            return 0.0
        recent_vol = np.mean(volumes[-5:])
        hist_vol = np.mean(volumes[:-5])
        return float((recent_vol - hist_vol) / hist_vol) if hist_vol > 0 else 0.0

    def _analyze_price_trend(self, prices: List[float]) -> float:
        if len(prices) < self.trend_window:
            return 0.0
        ma_short = np.mean(prices[-5:])
        ma_long = np.mean(prices[-self.trend_window:])
        return float((ma_short - ma_long) / ma_long) if ma_long > 0 else 0.0

class MarketAnalyzer:
    def __init__(self, jupiter_client: Any, settings: Any) -> None:
        self.jupiter = jupiter_client
        self.settings = settings
        self.indicators = TechnicalIndicators()  # Initialize technical indicators
        self.lookback_periods = {
            'short': 12,
            'medium': 26,
            'long': 50
        }
        self.price_cache: Dict[str, List[float]] = {}
        self.volume_cache: Dict[str, List[float]] = {}
        self.cache_expiry = 300  # 5 minutes

    def _calculate_signal_strength(self, macd: Optional[MACDResult], bollinger: Optional[BollingerResult], rsi: float) -> float:
        """Calculate overall signal strength from technical indicators"""
        signal_strength = 0.0
        
        # MACD contribution (40%)
        if macd:
            macd_signal = 1 if macd.signal == "buy" else -1 if macd.signal == "sell" else 0
            signal_strength += macd_signal * 0.4
            
        # Bollinger Bands contribution (30%)
        if bollinger:
            bb_signal = 1 if bollinger.signal == "buy" else -1 if bollinger.signal == "sell" else 0
            signal_strength += bb_signal * 0.3
            
        # RSI contribution (30%)
        rsi_signal = 1 if rsi < 30 else -1 if rsi > 70 else 0
        signal_strength += rsi_signal * 0.3
        
        return signal_strength

    async def analyze_market(self, token_address: str, price_data: Optional[List[float]] = None, volume_data: Optional[List[float]] = None) -> Optional[MarketIndicators]:
        """Analyze market with comprehensive validation"""
        try:
            # Get or validate price data
            if not price_data:
                price_data = await self._get_price_data(token_address)

            # Require minimum 5 data points instead of 30 for new tokens
            if not price_data or len(price_data) < 5:
                logger.warning(f"Insufficient price data for {token_address}: {len(price_data) if price_data else 0} points")
                # Create fallback data for basic analysis
                price_data = [1.0] * 10  # Default price series for new tokens
            
            # Pad price data if we have some but not enough
            if len(price_data) < 20:
                # Pad with the last known price to reach minimum for indicators
                last_price = price_data[-1] if price_data else 1.0
                price_data.extend([last_price] * (20 - len(price_data)))

            # Get or validate volume data
            if not volume_data:
                volume_data = await self._get_volume_data(token_address)

            # Be more lenient with volume data for new tokens
            if not volume_data or len(volume_data) < 5:
                logger.warning(f"Insufficient volume data for {token_address}: {len(volume_data) if volume_data else 0} points")
                # Create fallback volume data
                volume_data = [1000.0] * 10  # Default volume series
            
            # Pad volume data if needed
            if len(volume_data) < 20:
                last_volume = volume_data[-1] if volume_data else 1000.0
                volume_data.extend([last_volume] * (20 - len(volume_data)))

            # Calculate all indicators first
            macd_result = self.indicators.calculate_macd(price_data)
            bollinger_result = self.indicators.calculate_bollinger_bands(price_data)
            rsi = self._calculate_rsi(price_data)

            # Additional validation
            if macd_result is None or bollinger_result is None:
                logger.error("Failed to calculate technical indicators")
                return None

            # Create market indicators object
            return MarketIndicators(
                rsi=rsi,
                volume_profile=self._analyze_volume_profile(volume_data),
                price_momentum=self._calculate_momentum(price_data),
                liquidity_score=self._calculate_liquidity_score(volume_data),
                volatility=self._calculate_volatility(price_data),
                trend_strength=self._calculate_trend_strength(price_data),
                macd=macd_result,
                bollinger=bollinger_result,
                signal_strength=self._calculate_signal_strength(macd_result, bollinger_result, rsi)
            )

        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            return None

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0

        price_array = np.array(prices)
        deltas = np.diff(price_array)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def _analyze_volume_profile(self, volumes: List[float]) -> float:
        """Analyze volume profile"""
        if not volumes:
            return 0.0

        recent_vol = np.mean(volumes[-5:])
        hist_vol = np.mean(volumes[:-5]) if len(volumes) > 5 else recent_vol
        
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
        """Calculate liquidity score"""
        if not volumes:
            return 0.0

        avg_volume = np.mean(volumes)
        if avg_volume == 0:
            return 0.0
            
        consistency = 1 - (np.std(volumes) / avg_volume)
        normalized_volume = min(avg_volume / self.settings.MIN_VOLUME_24H, 1.0)
        
        return float((consistency + normalized_volume) / 2)

    def _calculate_volatility(self, prices: List[float], window: int = 20) -> float:
        """Calculate price volatility"""
        if len(prices) < window:
            return 0.0

        returns = np.diff(prices) / prices[:-1]
        volatility = float(np.std(returns) * np.sqrt(365))
        return min(volatility, 1.0)

    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength"""
        if len(prices) < self.lookback_periods['long']:
            return 0.0

        short_ma = np.mean(prices[-self.lookback_periods['short']:])
        medium_ma = np.mean(prices[-self.lookback_periods['medium']:])
        long_ma = np.mean(prices[-self.lookback_periods['long']:])

        if short_ma > medium_ma > long_ma:
            trend = 1.0  # Strong uptrend
        elif short_ma < medium_ma < long_ma:
            trend = -1.0  # Strong downtrend
        else:
            trend = 0.0  # No clear trend

        return float(abs(trend))

    async def _get_price_data(self, token_address: str) -> Optional[List[float]]:
        """Get historical price data"""
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
        """Get historical volume data"""
        try:
            if token_address in self.volume_cache:
                return self.volume_cache[token_address]

            market_data = await self.jupiter.get_market_depth(token_address)
            if not market_data or 'recent_volumes' not in market_data:
                return None

            volumes = [float(vol) for vol in market_data['recent_volumes']]
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