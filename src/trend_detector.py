
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TrendAnalysis:
    token_address: str
    momentum_score: float
    trend_direction: str
    volume_trend: str
    price_pattern: str
    confidence: float
    timestamp: datetime

class TrendDetector:
    def __init__(self, jupiter_client):
        self.jupiter = jupiter_client
        self.trend_history: Dict[str, List[TrendAnalysis]] = {}
        
    async def analyze_trend(self, token_address: str) -> Optional[TrendAnalysis]:
        try:
            price_data = await self._get_price_history(token_address)
            if not price_data:
                return None

            momentum = self._calculate_momentum(price_data)
            trend_direction = self._detect_trend_direction(price_data)
            volume_trend = await self._analyze_volume_trend(token_address)
            pattern = self._identify_price_pattern(price_data)
            
            confidence = self._calculate_confidence(
                momentum,
                trend_direction,
                volume_trend,
                pattern
            )
            
            analysis = TrendAnalysis(
                token_address=token_address,
                momentum_score=float(momentum),
                trend_direction=trend_direction,
                volume_trend=volume_trend,
                price_pattern=pattern,
                confidence=float(confidence),
                timestamp=datetime.now()
            )
            
            self._update_trend_history(token_address, analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return None

    async def _get_price_history(self, token_address: str) -> Optional[Dict[str, List[Dict[str, float]]]]:
        try:
            intervals = ['1m', '5m', '15m', '1h']
            price_data = {}
            
            for interval in intervals:
                data = await self.jupiter.get_price_history(token_address, interval)
                if data:
                    price_data[interval] = [
                        {'price': p['price']} for p in data 
                        if isinstance(p, dict) and 'price' in p
                    ]
                    
            return price_data if price_data else None
            
        except Exception as e:
            logger.error(f"Error getting price history: {str(e)}")
            return None

    def _calculate_momentum(self, price_history: Dict[str, List[Dict[str, float]]]) -> float:
        momentum_scores = []
        
        for prices in price_history.values():
            if len(prices) < 2:
                continue
                
            price_values = [p['price'] for p in prices]
            changes = np.diff(price_values)
            momentum_scores.append(float(np.mean(changes)))
            
        return float(np.mean(momentum_scores)) if momentum_scores else 0.0

    def _detect_trend_direction(self, price_history: Dict[str, List[Dict[str, float]]]) -> str:
        if '15m' not in price_history:
            return "sideways"
            
        prices = price_history['15m']
        price_values = [p['price'] for p in prices]
        
        if len(price_values) < 20:
            return "sideways"
            
        short_ma = float(np.mean(price_values[-5:]))
        long_ma = float(np.mean(price_values[-20:]))
        
        if short_ma > long_ma * 1.02:
            return "uptrend"
        elif short_ma < long_ma * 0.98:
            return "downtrend"
        return "sideways"

    async def _analyze_volume_trend(self, token_address: str) -> str:
        try:
            data = await self.jupiter.get_market_depth(token_address)
            if not data or not isinstance(data, dict):
                return "neutral"
                
            volumes = data.get('recentVolumes', [])
            if not volumes:
                return "neutral"
                
            # Convert to numpy array and ensure float type
            volumes_array = np.array(volumes, dtype=float)
            avg_volume = float(np.mean(volumes_array))
            recent_volume = float(volumes_array[-1])
            
            if recent_volume > avg_volume * 1.5:
                return "increasing"
            elif recent_volume < avg_volume * 0.5:
                return "decreasing"
            return "neutral"
        
        except Exception as e:
            logger.error(f"Error analyzing volume: {str(e)}")
            return "neutral"

    def _identify_price_pattern(self, price_history: Dict[str, List[Dict[str, float]]]) -> str:
        try:
            if '5m' not in price_history:
                return "unknown"
                
            prices = price_history['5m']
            price_values = [p['price'] for p in prices]
            
            if len(price_values) < 20:
                return "unknown"
                
            price_values = np.array(price_values, dtype=float)
            
            if self._is_breakout(price_values):
                return "breakout"
            elif self._is_consolidation(price_values):
                return "consolidation"
            elif self._is_pullback(price_values):
                return "pullback"
            return "no_pattern"
            
        except Exception as e:
            logger.error(f"Error identifying pattern: {str(e)}")
            return "unknown"

    def _is_breakout(self, prices: np.ndarray) -> bool:
        if len(prices) < 20:
            return False
            
        recent_std = float(np.std(prices[-5:]))
        historical_std = float(np.std(prices[:-5]))
        historical_high = float(np.max(prices[:-5]))
        recent_price = float(prices[-1])
        
        return bool(recent_std > historical_std * 1.5 and recent_price > historical_high)

    def _is_consolidation(self, prices: np.ndarray) -> bool:
        if len(prices) < 20:
            return False
            
        recent = prices[-10:]
        price_range = float(np.max(recent) - np.min(recent))
        avg_price = float(np.mean(recent))
        
        return bool(price_range <= avg_price * 0.02)

    def _is_pullback(self, prices: np.ndarray) -> bool:
        if len(prices) < 20:
            return False
            
        recent_high = float(np.max(prices[-20:]))
        current_price = float(prices[-1])
        pullback = (recent_high - current_price) / recent_high
        
        return bool(0.03 <= pullback <= 0.05)

    def _calculate_confidence(self, 
                            momentum: float,
                            trend: str,
                            volume_trend: str,
                            pattern: str) -> float:
        confidence = 0.0
        
        if momentum > 0.7:
            confidence += 30
        elif momentum > 0.3:
            confidence += 20
        elif momentum > 0:
            confidence += 10
            
        if trend == "uptrend":
            confidence += 30
        elif trend == "sideways":
            confidence += 15
            
        if volume_trend == "increasing":
            confidence += 20
        elif volume_trend == "neutral":
            confidence += 10
            
        if pattern == "breakout":
            confidence += 20
        elif pattern == "consolidation":
            confidence += 15
        elif pattern == "pullback":
            confidence += 10
            
        return float(confidence)

    def _update_trend_history(self, token_address: str, analysis: TrendAnalysis) -> None:
        if token_address not in self.trend_history:
            self.trend_history[token_address] = []
            
        self.trend_history[token_address].append(analysis)
        
        cutoff = datetime.now() - timedelta(hours=24)
        self.trend_history[token_address] = [
            a for a in self.trend_history[token_address]
            if a.timestamp > cutoff
        ]

    def get_trend_summary(self, token_address: str) -> Dict[str, Any]:
        if token_address not in self.trend_history:
            return {"error": "No trend history available"}
            
        history = self.trend_history[token_address]
        if not history:
            return {"error": "No trend history available"}
            
        stability = np.std([a.momentum_score for a in history])
        confidence_trend = np.mean([a.confidence for a in history])
            
        return {
            "latest_analysis": history[-1],
            "trend_stability": float(stability),
            "confidence_trend": float(confidence_trend),
            "analysis_count": len(history)
        }
