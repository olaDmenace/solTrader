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
        try:
            if not prices or len(prices) < 2:
                return 0.0
            changes = np.diff(prices)
            if len(changes) == 0:
                return 0.0
            result = float(np.mean(changes))
            return result if not np.isnan(result) and np.isfinite(result) else 0.0
        except Exception:
            return 0.0

    def _calculate_volatility(self, prices: List[float]) -> float:
        try:
            if not prices or len(prices) == 0:
                return 0.0
            mean_price = np.mean(prices)
            if mean_price == 0:
                return 0.0
            std_price = np.std(prices)
            result = float(std_price / mean_price)
            return result if not np.isnan(result) and np.isfinite(result) else 0.0
        except Exception:
            return 0.0

    def _calculate_trend_strength(self, prices: List[float]) -> float:
        try:
            if not prices or len(prices) < 2:
                return 0.0
            linear_reg = np.polyfit(range(len(prices)), prices, 1)
            if len(linear_reg) == 0:
                return 0.0
            result = float(abs(linear_reg[0]))
            return result if not np.isnan(result) and np.isfinite(result) else 0.0
        except Exception:
            return 0.0

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
            
            # Validate signal strength is a valid number
            if signal_strength is None or not isinstance(signal_strength, (int, float)) or signal_strength != signal_strength:  # NaN check
                logger.warning(f"[SIGNAL] Invalid signal strength returned: {signal_strength}, using fallback (0.1)")
                signal_strength = 0.1
            
            # Ensure signal strength is within valid range
            signal_strength = max(0.0, min(1.0, float(signal_strength)))
            
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
            
            # Validate calculated values
            if momentum_score is None or not isinstance(momentum_score, (int, float)):
                logger.warning(f"[MARKET] Invalid momentum_score: {momentum_score}, using neutral (0.5)")
                momentum_score = 0.5
                
            if volatility is None or not isinstance(volatility, (int, float)):
                logger.warning(f"[MARKET] Invalid volatility: {volatility}, using moderate (25.0)")
                volatility = 25.0
            
            logger.debug(f"[MARKET] Creating market condition with momentum={momentum_score:.3f}, volatility={volatility:.3f}")
            
            market_condition = MarketCondition(
                volume_24h=volume_24h,
                liquidity=liquidity,
                price_change_24h=self._calculate_price_change(price_history),
                volatility=float(volatility),
                momentum_score=float(momentum_score)
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
        """Calculate signal strength with robust error handling and validation"""
        try:
            # Validate all inputs with safe defaults
            volume_24h = float(market_condition.volume_24h) if market_condition.volume_24h is not None else 0.0
            liquidity = float(market_condition.liquidity) if market_condition.liquidity is not None else 0.0
            momentum_score = float(market_condition.momentum_score) if market_condition.momentum_score is not None else 0.5
            volatility = float(market_condition.volatility) if market_condition.volatility is not None else 25.0
            
            logger.debug(f"[SIGNAL_CALC] Inputs - Volume: {volume_24h}, Liquidity: {liquidity}, Momentum: {momentum_score:.3f}, Volatility: {volatility:.3f}")
            
            # Volume weight: 30% - normalize against expected volume
            min_vol_threshold = max(self.min_volume, 1.0)  # Avoid division by zero
            volume_score = min(volume_24h / (min_vol_threshold * 10), 1.0) * 0.3
            
            # Liquidity weight: 30% - normalize against expected liquidity  
            min_liq_threshold = max(self.min_liquidity, 1.0)  # Avoid division by zero
            liquidity_score = min(liquidity / (min_liq_threshold * 10), 1.0) * 0.3
            
            # Momentum weight: 20% - already normalized to 0-1 from _aggregate_momentum
            momentum_component = max(0.0, min(1.0, momentum_score)) * 0.2
            
            # Volatility weight: 20% (inverse - lower volatility is better)
            # Cap volatility at 100 for normalization
            normalized_volatility = min(volatility / 100.0, 1.0)
            volatility_component = (1.0 - normalized_volatility) * 0.2
            
            # Calculate base signal
            base_signal = volume_score + liquidity_score + momentum_component + volatility_component
            
            logger.debug(f"[SIGNAL_CALC] Components - Vol: {volume_score:.3f}, Liq: {liquidity_score:.3f}, Mom: {momentum_component:.3f}, Vol: {volatility_component:.3f}")
            logger.debug(f"[SIGNAL_CALC] Base signal: {base_signal:.3f}")
            
            # Ensure base signal is within valid range
            base_signal = max(0.0, min(1.0, base_signal))
            
            # Apply trending signal enhancement if available
            enhanced_signal = self._apply_trending_boost(base_signal, market_condition)
            
            # Final validation - ensure we never return None or invalid values
            final_signal = max(0.0, min(1.0, float(enhanced_signal)))
            
            logger.debug(f"[SIGNAL_CALC] Final signal strength: {final_signal:.3f}")
            return final_signal

        except Exception as e:
            logger.error(f"Error calculating signal strength: {str(e)}")
            logger.error(f"Market condition values: volume={getattr(market_condition, 'volume_24h', 'None')}, liquidity={getattr(market_condition, 'liquidity', 'None')}, momentum={getattr(market_condition, 'momentum_score', 'None')}, volatility={getattr(market_condition, 'volatility', 'None')}")
            # Return a low but valid signal strength instead of 0.0 to allow some trading in paper mode
            return 0.1
    
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
        """Aggregate momentum across timeframes with safe defaults"""
        if not trend_analysis:
            logger.debug("[MOMENTUM] No trend analysis data - returning neutral momentum (0.5)")
            return 0.5  # Neutral momentum when no data available
            
        weights = {'5m': 0.1, '15m': 0.2, '1h': 0.3, '4h': 0.4}
        weighted_momentum = 0.0
        total_weight_used = 0.0
        
        for timeframe, weight in weights.items():
            if timeframe in trend_analysis:
                momentum_val = trend_analysis[timeframe].get('momentum', 0.0)
                if momentum_val is not None:
                    weighted_momentum += float(momentum_val) * weight
                    total_weight_used += weight
        
        # If no valid momentum data was found, return neutral
        if total_weight_used == 0.0:
            logger.debug("[MOMENTUM] No valid momentum data found - returning neutral (0.5)")
            return 0.5
            
        # Normalize by actual weights used and convert to 0-1 scale
        normalized_momentum = weighted_momentum / total_weight_used if total_weight_used > 0 else 0.0
        # Convert to 0-1 scale (assuming momentum can be negative)
        result = max(0.0, min(1.0, (normalized_momentum + 1.0) / 2.0))
        
        logger.debug(f"[MOMENTUM] Calculated momentum: {result:.3f} from {len(trend_analysis)} timeframes")
        return float(result)

    def _aggregate_volatility(self, trend_analysis: Mapping[str, TrendData]) -> float:
        """Aggregate volatility across timeframes with safe defaults"""
        if not trend_analysis:
            logger.debug("[VOLATILITY] No trend analysis data - returning moderate volatility (25.0)")
            return 25.0  # Moderate volatility when no data available
            
        weights = {'5m': 0.1, '15m': 0.2, '1h': 0.3, '4h': 0.4}
        weighted_volatility = 0.0
        total_weight_used = 0.0
        
        for timeframe, weight in weights.items():
            if timeframe in trend_analysis:
                volatility_val = trend_analysis[timeframe].get('volatility', 0.0)
                if volatility_val is not None:
                    weighted_volatility += float(volatility_val) * weight
                    total_weight_used += weight
        
        # If no valid volatility data was found, return moderate volatility
        if total_weight_used == 0.0:
            logger.debug("[VOLATILITY] No valid volatility data found - returning moderate (25.0)")
            return 25.0
            
        # Normalize by actual weights used
        result = weighted_volatility / total_weight_used if total_weight_used > 0 else 25.0
        
        logger.debug(f"[VOLATILITY] Calculated volatility: {result:.3f} from {len(trend_analysis)} timeframes")
        return float(result)

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