"""
market_regime.py - Advanced market regime detection and adaptation
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class MarketRegimeType(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    UNCERTAIN = "uncertain"

@dataclass
class MarketState:
    """Current market state information"""
    regime: MarketRegimeType
    volatility: float
    trend_strength: float
    volume_profile: float
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    confidence: float = 0.0
    timestamp: datetime = datetime.now()

@dataclass
class VolumeProfile:
    """Volume analysis results"""
    volume_trend: float
    buying_pressure: float
    selling_pressure: float
    volume_concentration: List[Tuple[float, float]]  # (price_level, volume)

class MarketRegimeDetector:
    """Advanced market regime detection and analysis"""
    
    def __init__(self, settings: Any):
        self.settings = settings
        self.lookback_periods = {
            'short': 12,   # 1 hour
            'medium': 24,  # 2 hours
            'long': 72     # 6 hours
        }
        self.regime_history: List[MarketState] = []
        self.volume_threshold = float(getattr(settings, 'MIN_VOLUME_24H', 1000.0))
        self.volatility_threshold = float(getattr(settings, 'MAX_VOLATILITY', 0.5))
        self.trend_threshold = float(getattr(settings, 'MIN_TREND_STRENGTH', 0.3))
        self.min_data_points = 2
        
    async def detect_regime(self, price_data: List[float], volume_data: List[float]) -> MarketState:
        if len(price_data) < 2 or len(volume_data) < 2:
            return MarketState(
                regime=MarketRegimeType.UNCERTAIN,
                volatility=0.0,
                trend_strength=0.0,
                volume_profile=0.0
            )
        try:
            volatility = self._calculate_volatility(price_data)
            trend_metrics = self._analyze_trend(price_data)
            volume_profile = self._analyze_volume(volume_data, price_data)
            
            regime = self._classify_regime(
                volatility=volatility,
                trend_metrics=trend_metrics,
                volume_profile=volume_profile
            )

            return MarketState(
                regime=regime,
                volatility=volatility,
                trend_strength=trend_metrics['strength'],
                volume_profile=volume_profile.volume_trend,
                support_level=self._find_support_resistance(price_data)[0],
                resistance_level=self._find_support_resistance(price_data)[1],
                confidence=trend_metrics['consistency']
            )

        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return MarketState(
                regime=MarketRegimeType.UNCERTAIN,
                volatility=0.0,
                trend_strength=0.0,
                volume_profile=0.0
            )

    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        try:
            if len(prices) < 2:
                return 0.0
                
            returns = np.diff(prices) / prices[:-1]
            return float(np.std(returns) * np.sqrt(365 * 24))  # Annualized hourly volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0

    def _analyze_volume(self, volumes: List[float], prices: List[float]) -> VolumeProfile:
        if len(volumes) < 20:
            return VolumeProfile(0.0, 0.0, 0.0, [])

        # Volume trend using end-to-start ratio
        end_vol = np.mean(volumes[-10:])
        start_vol = np.mean(volumes[:10])
        volume_trend = (end_vol / start_vol - 1) if start_vol > 0 else 0

        # Use price direction for pressure
        price_direction = np.sign(prices[-1] - prices[-20])
        volume_weight = 1 if price_direction >= 0 else -1
        pressure = 0.5 * (1 + volume_weight)

        return VolumeProfile(
            volume_trend=float(volume_trend),
            buying_pressure=float(pressure),
            selling_pressure=float(1 - pressure),
            volume_concentration=[]
        )

    def _find_volume_concentration(self, 
                                 volumes: List[float], 
                                 prices: List[float]) -> List[Tuple[float, float]]:
        """Find price levels with high volume concentration"""
        try:
            if len(prices) < 2 or len(volumes) < 2:
                return []

            # Create price bins
            min_price, max_price = min(prices), max(prices)
            if min_price == max_price:  # Handle edge case
                return [(float(min_price), float(sum(volumes)))]
                
            price_range = max_price - min_price
            if price_range == 0:
                return []
                
            bin_size = price_range / 10  # 10 bins
            if bin_size == 0:
                return []
                
            bins = np.arange(min_price, max_price + bin_size, bin_size)
            
            # Calculate volume per price level
            volume_concentration = []
            for i in range(len(bins)-1):
                mask = (np.array(prices) >= bins[i]) & (np.array(prices) < bins[i+1])
                if any(mask):
                    volume_sum = sum(vol for j, vol in enumerate(volumes) if mask[j])
                    price_level = (bins[i] + bins[i+1]) / 2
                    volume_concentration.append((float(price_level), float(volume_sum)))
                    
            return sorted(volume_concentration, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding volume concentration: {str(e)}")
            return []

    def _find_support_resistance(self, prices: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if len(prices) < 10:  # Need minimum data
            return None, None
        
        price_arr = np.array(prices)
        window = min(20, len(prices) // 4)
        
        peaks = []
        troughs = []
        
        for i in range(window, len(prices) - window):
            if all(prices[i] > prices[j] for j in range(i-window, i+window) if j != i):
                peaks.append(prices[i])
            if all(prices[i] < prices[j] for j in range(i-window, i+window) if j != i):
                troughs.append(prices[i])
        
        current = prices[-1]
        support = max((p for p in troughs if p < current), default=min(prices))
        resistance = min((p for p in peaks if p > current), default=max(prices))
        
        return float(support), float(resistance)
    
    def _analyze_trend(self, prices: List[float]) -> Dict[str, float]:
        if len(prices) < 2:
            return {'strength': 0.0, 'direction': 0.0, 'consistency': 0.0}

        try:
            # Linear regression for trend
            x = np.arange(len(prices))
            y = np.array(prices)
            slope, _ = np.polyfit(x, y, 1)
            
            # Normalize slope by average price for comparable strength
            avg_price = np.mean(prices)
            strength = abs(slope) / avg_price if avg_price > 0 else 0
            direction = np.sign(slope)
            
            # Calculate consistency
            returns = np.diff(prices) / prices[:-1]
            consistent_moves = np.sum(np.sign(returns) == direction)
            consistency = consistent_moves / len(returns) if len(returns) > 0 else 0
            
            return {
                'strength': float(strength * 100),  # Scale up for better sensitivity
                'direction': float(direction),
                'consistency': float(consistency)
            }
        except Exception:
            return {'strength': 0.0, 'direction': 0.0, 'consistency': 0.0}
    


    def _classify_regime(self,
                    volatility: float,
                    trend_metrics: Dict[str, float],
                    volume_profile: VolumeProfile,
                    market_data: Optional[Dict[str, Any]] = None) -> MarketRegimeType:
        try:
            # Check for accumulation first (low volatility, high volume trend)
            if (volatility < 0.15 and 
                volume_profile.volume_trend > 0.1 and 
                volume_profile.buying_pressure >= 0.5):
                return MarketRegimeType.ACCUMULATION

            # Rest of the classification logic remains same
            if trend_metrics['strength'] > self.trend_threshold and trend_metrics['consistency'] > 0.6:
                return MarketRegimeType.TRENDING_UP if trend_metrics['direction'] > 0 else MarketRegimeType.TRENDING_DOWN

            if volatility > self.volatility_threshold:
                return MarketRegimeType.VOLATILE

            if volatility < self.volatility_threshold * 0.2 and trend_metrics['strength'] < self.trend_threshold / 2:
                return MarketRegimeType.RANGING

            return MarketRegimeType.UNCERTAIN

        except Exception as e:
            logger.error(f"Error classifying regime: {str(e)}")
            return MarketRegimeType.UNCERTAIN

    def _update_history(self, state: MarketState) -> None:
        """Update regime history"""
        self.regime_history.append(state)
        if len(self.regime_history) > 100:  # Keep last 100 states
            self.regime_history.pop(0)

    def get_recent_regimes(self, hours: int = 24) -> List[MarketState]:
        """Get regime history for specified period"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            state for state in self.regime_history 
            if state.timestamp > cutoff
        ]

    def get_regime_metrics(self) -> Dict[str, Any]:
        """Get metrics about regime changes and characteristics"""
        try:
            if not self.regime_history:
                return {}

            recent_states = self.get_recent_regimes()
            if not recent_states:
                return {}

            regime_counts = {}
            volatilities = []
            trend_strengths = []
            volume_profiles = []

            for state in recent_states:
                regime_counts[state.regime] = regime_counts.get(state.regime, 0) + 1
                volatilities.append(state.volatility)
                trend_strengths.append(state.trend_strength)
                volume_profiles.append(state.volume_profile)

            total_states = len(recent_states)
            return {
                'regime_distribution': {
                    regime.name: count/total_states 
                    for regime, count in regime_counts.items()
                },
                'avg_volatility': float(np.mean(volatilities)),
                'avg_trend_strength': float(np.mean(trend_strengths)),
                'avg_volume_profile': float(np.mean(volume_profiles)),
                'regime_changes': self._count_regime_changes(recent_states),
                'current_regime_duration': self._get_current_regime_duration()
            }

        except Exception as e:
            logger.error(f"Error getting regime metrics: {str(e)}")
            return {}

    def _count_regime_changes(self, states: List[MarketState]) -> int:
        """Count number of regime changes"""
        if len(states) < 2:
            return 0
            
        changes = sum(
            1 for i in range(1, len(states))
            if states[i].regime != states[i-1].regime
        )
        return changes

    def _get_current_regime_duration(self) -> int:
        """Calculate duration of current regime in hours"""
        if not self.regime_history:
            return 0
            
        current_regime = self.regime_history[-1].regime
        duration = 0
        
        # Count backwards until we find a different regime
        for state in reversed(self.regime_history[:-1]):
            if state.regime != current_regime:
                break
            duration += 1
            
        return duration