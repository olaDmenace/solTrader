# signals.py

import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from typing import Mapping, TypedDict

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    token_address: str
    price: float
    strength: float
    market_data: Dict[str, Any]
    signal_type: str
    timestamp: datetime = datetime.now()

@dataclass
class MarketCondition:
    volume_24h: float
    liquidity: float
    price_change_24h: float
    volatility: float
    momentum_score: float

class TrendData(TypedDict):
    momentum: float
    volatility: float
    trend_strength: float

class TrendAnalysis:
    def __init__(self):
        self.timeframes = ['5m', '15m', '1h', '4h']
        self.window_sizes = {'5m': 12, '15m': 24, '1h': 24, '4h': 24}

    def analyze(self, price_history: Dict[str, List[float]]) -> Dict[str, TrendData]:
        trends = {}
        for timeframe in self.timeframes:
            if timeframe in price_history and len(price_history[timeframe]) >= 2:
                prices = price_history[timeframe]
                trends[timeframe] = {
                    'momentum': self._calculate_momentum(prices),
                    'volatility': self._calculate_volatility(prices),
                    'trend_strength': self._calculate_trend_strength(prices)
                }
        return trends

    def _calculate_momentum(self, prices: List[float]) -> float:
        changes = np.diff(prices)
        return float(np.mean(changes))

    def _calculate_volatility(self, prices: List[float]) -> float:
        return float(np.std(prices) / np.mean(prices) if len(prices) > 0 else 0)

    def _calculate_trend_strength(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        linear_reg = np.polyfit(range(len(prices)), prices, 1)
        return float(abs(linear_reg[0]))

class VolumeAnalysis:
    def analyze(self, volume_data: List[float]) -> Dict[str, float]:
        if not volume_data:
            return {'volume_trend': 0.0, 'volume_consistency': 0.0}

        return {
            'volume_trend': self._calculate_volume_trend(volume_data),
            'volume_consistency': self._calculate_volume_consistency(volume_data)
        }

    def _calculate_volume_trend(self, volumes: List[float]) -> float:
        if len(volumes) < 2:
            return 0.0
        return float(volumes[-1] / np.mean(volumes) if volumes else 0)

    def _calculate_volume_consistency(self, volumes: List[float]) -> float:
        if not volumes:
            return 0.0
        return 1.0 - float(np.std(volumes) / np.mean(volumes) if volumes else 0)

class SignalGenerator:
    def __init__(self, settings: Any):
        self.settings = settings
        self.min_volume = settings.MIN_VOLUME_24H
        self.min_liquidity = settings.MIN_LIQUIDITY
        self.signal_threshold = settings.SIGNAL_THRESHOLD
        self.trend_analyzer = TrendAnalysis()
        self.volume_analyzer = VolumeAnalysis()
        self.analyzed_tokens: Dict[str, datetime] = {}

    async def analyze_token(self, token_data) -> Optional[Signal]:
        try:
            # Convert TokenObject to dict if needed
            if hasattr(token_data, 'address'):
                # It's a TokenObject, convert to dict format
                token_dict = {
                    'address': getattr(token_data, 'address', ''),
                    'price': getattr(token_data, 'price_sol', 0),
                    'volume24h': getattr(token_data, 'volume24h', 0),
                    'liquidity': getattr(token_data, 'liquidity', 0),
                    'market_cap': getattr(token_data, 'market_cap', 0),
                    'created_at': getattr(token_data, 'created_at', None),
                    'scan_id': getattr(token_data, 'scan_id', 0),
                    'source': getattr(token_data, 'source', 'unknown')
                }
            else:
                # It's already a dict - normalize field names
                token_dict = self._normalize_token_fields(token_data)

            if not self._validate_token_data(token_dict):
                return None

            market_condition = await self._analyze_market_condition(token_dict)
            if not market_condition:
                return None

            signal_strength = self._calculate_signal_strength(market_condition)
            
            logger.info(f"[SIGNAL] Token {token_dict['address'][:8]}... signal strength: {signal_strength:.3f} (threshold: {self.signal_threshold})")
            
            if signal_strength < self.signal_threshold:
                logger.info(f"[SIGNAL] Signal too weak: {signal_strength:.3f} < {self.signal_threshold}")
                return None

            return Signal(
                token_address=token_dict['address'],
                price=float(token_dict['price']),
                strength=signal_strength,
                market_data=token_dict,
                signal_type=self._determine_signal_type(market_condition)
            )

        except Exception as e:
            logger.error(f"Error analyzing token: {str(e)}")
            logger.error(f"Token data keys: {list(token_data.keys()) if isinstance(token_data, dict) else 'not a dict'}")
            return None
    
    def _normalize_token_fields(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize field names from scanner format to signal generator format"""
        normalized = token_data.copy()
        
        # Map scanner fields to expected signal generator fields
        field_mappings = {
            'price_sol': 'price',
            'volume_24h_sol': 'volume24h', 
            'liquidity_sol': 'liquidity',
            'market_cap_sol': 'market_cap'
        }
        
        for scanner_field, signal_field in field_mappings.items():
            if scanner_field in token_data and signal_field not in token_data:
                normalized[signal_field] = token_data[scanner_field]
                logger.debug(f"[NORMALIZE] Mapped {scanner_field} -> {signal_field}: {token_data[scanner_field]}")
        
        # Copy trending data if available
        if 'trending_token' in token_data:
            normalized['trending_token'] = token_data['trending_token']
        if 'trending_score' in token_data:
            normalized['trending_score'] = token_data['trending_score']
            
        return normalized

    def _validate_token_data(self, token_data: Dict[str, Any]) -> bool:
        # Check for scanner format fields (from practical_solana_scanner.py)
        scanner_fields = ['address', 'price_sol', 'volume_24h_sol', 'liquidity_sol']
        legacy_fields = ['address', 'price', 'volume24h', 'liquidity']
        
        # Accept either scanner format or legacy format
        has_scanner_fields = all(field in token_data for field in scanner_fields)
        has_legacy_fields = all(field in token_data for field in legacy_fields)
        
        return has_scanner_fields or has_legacy_fields

    async def _analyze_market_condition(self, token_data: Dict[str, Any]) -> Optional[MarketCondition]:
        try:
            volume_24h = float(token_data.get('volume24h', 0))
            liquidity = float(token_data.get('liquidity', 0))
            
            # Check if this is a trending token (apply permissive thresholds)
            is_trending = token_data.get('source') == 'birdeye_trending'
            
            if is_trending:
                # More permissive thresholds for trending tokens (same as strategy.py)
                min_volume = max(self.min_volume * 0.2, 5)  # 20% of min volume, min 5 SOL for trending
                min_liquidity = max(self.min_liquidity * 0.5, 200)  # 50% of min liquidity, min 200 SOL for trending
                logger.info(f"[SIGNALS] Trending token detected - using permissive thresholds")
                logger.info(f"[SIGNALS] Volume: {volume_24h:.1f} SOL (trending threshold: {min_volume:.1f})")
                logger.info(f"[SIGNALS] Liquidity: {liquidity:.1f} SOL (trending threshold: {min_liquidity:.1f})")
            else:
                # Standard thresholds for non-trending tokens
                min_volume = self.min_volume
                min_liquidity = self.min_liquidity
                logger.debug(f"[MARKET] Analyzing: volume={volume_24h}, liquidity={liquidity}")
                logger.debug(f"[MARKET] Thresholds: min_volume={min_volume}, min_liquidity={min_liquidity}")
            
            if volume_24h < min_volume or liquidity < min_liquidity:
                logger.info(f"[MARKET] Token failed volume/liquidity check: vol={volume_24h:.1f} (min={min_volume:.1f}), liq={liquidity:.1f} (min={min_liquidity:.1f})")
                return None

            price_history = token_data.get('price_history', {})
            volume_data = token_data.get('volume_history', [])
            
            trend_analysis = self.trend_analyzer.analyze(price_history)
            volume_analysis = self.volume_analyzer.analyze(volume_data)
            
            momentum_score = self._aggregate_momentum(trend_analysis)
            volatility = self._aggregate_volatility(trend_analysis)
            
            market_condition = MarketCondition(
                volume_24h=volume_24h,
                liquidity=liquidity,
                price_change_24h=self._calculate_price_change(price_history),
                volatility=volatility,
                momentum_score=momentum_score
            )
            
            # Pass through trending data if available
            if 'trending_token' in token_data:
                market_condition.trending_token = token_data['trending_token']
            if 'trending_score' in token_data:
                market_condition.trending_score = token_data['trending_score']
                
            return market_condition

        except Exception as e:
            logger.error(f"Error analyzing market condition: {str(e)}")
            return None

    def _calculate_signal_strength(self, market_condition: MarketCondition) -> float:
        try:
            # Volume weight: 30%
            volume_score = min(market_condition.volume_24h / (self.min_volume * 10), 1.0) * 0.3
            
            # Liquidity weight: 30%
            liquidity_score = min(market_condition.liquidity / (self.min_liquidity * 10), 1.0) * 0.3
            
            # Momentum weight: 20%
            momentum_score = market_condition.momentum_score * 0.2
            
            # Volatility weight: 20% (inverse - lower is better)
            volatility_score = (1 - min(market_condition.volatility / 100, 1.0)) * 0.2
            
            base_signal = float(volume_score + liquidity_score + momentum_score + volatility_score)
            
            # Apply trending signal enhancement if available
            enhanced_signal = self._apply_trending_boost(base_signal, market_condition)
            
            return enhanced_signal

        except Exception as e:
            logger.error(f"Error calculating signal strength: {str(e)}")
            return 0.0
    
    def _apply_trending_boost(self, base_signal: float, market_condition: MarketCondition) -> float:
        """Apply trending signal boost if token has trending data"""
        try:
            # Check if token has trending information (added by scanner)
            trending_token = getattr(market_condition, 'trending_token', None)
            trending_score = getattr(market_condition, 'trending_score', None)
            
            if not trending_token or trending_score is None:
                logger.debug("No trending data available for signal boost")
                return base_signal
            
            # Import here to avoid circular imports
            try:
                from ..trending_analyzer import TrendingAnalyzer
                from ..config.settings import Settings
                
                # Create a temporary settings object with trending boost factor
                class TempSettings:
                    def __init__(self):
                        self.TRENDING_SIGNAL_BOOST = getattr(self.settings, 'TRENDING_SIGNAL_BOOST', 0.5)
                
                temp_settings = TempSettings()
                temp_settings.TRENDING_SIGNAL_BOOST = getattr(self.settings, 'TRENDING_SIGNAL_BOOST', 0.5)
                
                analyzer = TrendingAnalyzer(temp_settings)
                enhanced_signal = analyzer.enhance_signal_strength(base_signal, trending_token)
                
                if enhanced_signal != base_signal:
                    logger.info(f"[TRENDING] Signal enhanced: {base_signal:.3f} → {enhanced_signal:.3f} (boost: +{(enhanced_signal - base_signal):.3f})")
                
                return enhanced_signal
                
            except ImportError as e:
                logger.debug(f"Trending analyzer not available: {e}")
                return base_signal
            
        except Exception as e:
            logger.error(f"Error applying trending boost: {e}")
            return base_signal

    def _aggregate_momentum(self, trend_analysis: Mapping[str, TrendData]) -> float:
        weights = {'5m': 0.1, '15m': 0.2, '1h': 0.3, '4h': 0.4}
        weighted_momentum = 0.0
        for timeframe, weight in weights.items():
            if timeframe in trend_analysis:
                weighted_momentum += trend_analysis[timeframe]['momentum'] * weight
        return float(weighted_momentum)

    def _aggregate_volatility(self, trend_analysis: Mapping[str, TrendData]) -> float:
        weights = {'5m': 0.1, '15m': 0.2, '1h': 0.3, '4h': 0.4}
        weighted_volatility = 0.0
        for timeframe, weight in weights.items():
            if timeframe in trend_analysis:
                weighted_volatility += trend_analysis[timeframe]['volatility'] * weight
        return float(weighted_volatility)

    def _calculate_price_change(self, price_history: Dict[str, List[float]]) -> float:
        if '1h' not in price_history or not price_history['1h']:
            return 0.0
        prices = price_history['1h']
        if len(prices) < 2:
            return 0.0
        return float((prices[-1] - prices[0]) / prices[0] * 100)

    def _determine_signal_type(self, market_condition: MarketCondition) -> str:
        momentum = market_condition.momentum_score
        price_change = market_condition.price_change_24h
        volatility = market_condition.volatility

        if momentum > 0.7 and price_change > 0 and volatility < 50:
            return "strong_buy"
        elif momentum > 0.5 and price_change > 0:
            return "buy"
        elif momentum < 0.3 and price_change < 0:
            return "sell"
        else:
            return "neutral"