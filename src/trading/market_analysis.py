import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"

@dataclass
class MarketMetrics:
    price: float
    volume: float
    liquidity: float
    volatility: float
    momentum: float
    trend_strength: float
    regime: MarketRegime
    rsi: float
    macd: Tuple[float, float]  # (macd_line, signal_line)
    support_levels: List[float]
    resistance_levels: List[float]
    timestamp: datetime = datetime.now()

class MarketAnalyzer:
    def __init__(self, settings: Any):
        self.settings = settings
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lookback_periods = {
            'short': 12,
            'medium': 24,
            'long': 50,
            'rsi': 14,
            'macd': (12, 26, 9)  # (fast, slow, signal)
        }

    async def analyze_market(self, 
                           token_address: str, 
                           price_data: List[float], 
                           volume_data: List[float]) -> Optional[MarketMetrics]:
        try:
            if not self._validate_data(price_data, volume_data):
                return None

            support_resistance = self._calculate_support_resistance(price_data)
            regime = self._determine_market_regime(price_data, volume_data)
            
            metrics = MarketMetrics(
                price=self._get_current_price(price_data),
                volume=self._calculate_volume_profile(volume_data),
                liquidity=self._calculate_liquidity_score(volume_data),
                volatility=self._calculate_volatility(price_data),
                momentum=self._calculate_momentum(price_data),
                trend_strength=self._calculate_trend_strength(price_data),
                regime=regime,
                rsi=self._calculate_rsi(price_data),
                macd=self._calculate_macd(price_data),
                support_levels=support_resistance['support'],
                resistance_levels=support_resistance['resistance']
            )

            self._update_cache(token_address, metrics)
            return metrics

        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            return None

    def _validate_data(self, price_data: List[float], volume_data: List[float]) -> bool:
        min_required = max(self.lookback_periods.values())
        return len(price_data) >= min_required and len(volume_data) >= min_required

    def _get_current_price(self, price_data: List[float]) -> float:
        return float(price_data[-1]) if price_data else 0.0

    def _calculate_volume_profile(self, volumes: List[float]) -> float:
        if not volumes:
            return 0.0
        recent_vol = np.mean(volumes[-5:])
        hist_vol = np.mean(volumes[:-5]) if len(volumes) > 5 else recent_vol
        return float(min(recent_vol / hist_vol if hist_vol > 0 else 0, 5.0))

    def _calculate_liquidity_score(self, volumes: List[float]) -> float:
        if not volumes:
            return 0.0
        avg_volume = np.mean(volumes)
        consistency = 1 - (np.std(volumes) / avg_volume) if avg_volume > 0 else 0
        normalized_volume = min(avg_volume / self.settings.MIN_VOLUME_24H, 1.0)
        return float((consistency + normalized_volume) / 2)

    def _calculate_volatility(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns) * np.sqrt(365))

    def _calculate_momentum(self, prices: List[float]) -> float:
        if len(prices) < self.lookback_periods['short']:
            return 0.0
        momentum = (prices[-1] - prices[-self.lookback_periods['short']]) / prices[-self.lookback_periods['short']]
        return float(np.clip(momentum, -1, 1))

    def _calculate_trend_strength(self, prices: List[float]) -> float:
        if len(prices) < self.lookback_periods['long']:
            return 0.0
        
        mas = {
            'short': np.mean(prices[-self.lookback_periods['short']:]),
            'medium': np.mean(prices[-self.lookback_periods['medium']:]),
            'long': np.mean(prices[-self.lookback_periods['long']:])
        }
        
        if mas['short'] > mas['medium'] > mas['long']:
            trend = 1.0  # Strong uptrend
        elif mas['short'] < mas['medium'] < mas['long']:
            trend = -1.0  # Strong downtrend
        else:
            trend = 0.0  # No clear trend

        return float(abs(trend))

    def _calculate_rsi(self, prices: List[float]) -> float:
        if len(prices) < self.lookback_periods['rsi']:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:self.lookback_periods['rsi']])
        avg_loss = np.mean(losses[:self.lookback_periods['rsi']])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def _calculate_macd(self, prices: List[float]) -> Tuple[float, float]:
        fast, slow, signal = self.lookback_periods['macd']
        
        if len(prices) < max(fast, slow, signal):
            return (0.0, 0.0)

        exp1 = np.exp(prices[-fast:]).mean()
        exp2 = np.exp(prices[-slow:]).mean()
        macd_line = exp1 - exp2
        signal_line = np.exp(prices[-signal:]).mean()

        return (float(macd_line), float(signal_line))

    def _calculate_support_resistance(self, prices: List[float]) -> Dict[str, List[float]]:
        window = min(20, len(prices) // 3)
        levels = {'support': [], 'resistance': []}
        
        for i in range(window, len(prices) - window):
            price_window = prices[i-window:i+window+1]
            current_price = prices[i]
            
            if current_price == min(price_window):
                levels['support'].append(current_price)
            if current_price == max(price_window):
                levels['resistance'].append(current_price)
        
        return {
            'support': sorted(list(set(levels['support'])))[-3:],  # Keep last 3 levels
            'resistance': sorted(list(set(levels['resistance'])))[:3]  # Keep first 3 levels
        }

    def _determine_market_regime(self, prices: List[float], volumes: List[float]) -> MarketRegime:
        volatility = self._calculate_volatility(prices)
        trend = self._calculate_trend_strength(prices)
        volume_trend = self._calculate_volume_profile(volumes)
        
        if volatility > 0.8:
            return MarketRegime.VOLATILE
            
        if trend > 0.7:
            return MarketRegime.TRENDING_UP if prices[-1] > prices[-2] else MarketRegime.TRENDING_DOWN
            
        if volume_trend > 1.5:
            return MarketRegime.ACCUMULATION if prices[-1] > prices[-2] else MarketRegime.DISTRIBUTION
            
        return MarketRegime.RANGING

    def _update_cache(self, token_address: str, metrics: MarketMetrics) -> None:
        self.cache[token_address] = {
            'metrics': metrics,
            'timestamp': datetime.now()
        }
        
        # Clear old cache entries
        current_time = datetime.now()
        self.cache = {
            addr: data for addr, data in self.cache.items()
            if (current_time - data['timestamp']).seconds < 300  # 5 minutes
        }