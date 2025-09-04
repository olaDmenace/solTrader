#!/usr/bin/env python3
"""
Enhanced Signal Generation System - Transparent and Data-Driven
Addresses signal quality improvements for more reliable trading decisions.

Key Improvements:
1. Transparent signal scoring with detailed breakdowns
2. Data-driven feature engineering
3. Signal confidence measurement
4. Historical validation tracking
5. Multi-factor signal composition
6. Real-time signal quality metrics
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)

class SignalQuality(Enum):
    EXCELLENT = "EXCELLENT"  # 0.8+ confidence
    GOOD = "GOOD"           # 0.6-0.8 confidence  
    FAIR = "FAIR"           # 0.4-0.6 confidence
    POOR = "POOR"           # 0.2-0.4 confidence
    REJECT = "REJECT"       # <0.2 confidence

@dataclass
class SignalComponent:
    """Individual signal component with transparent scoring"""
    name: str
    value: float
    weight: float
    confidence: float
    explanation: str
    raw_data: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class EnhancedSignal:
    """Enhanced signal with full transparency and validation data"""
    token_address: str
    signal_type: str
    overall_confidence: float
    quality: SignalQuality
    components: List[SignalComponent] = field(default_factory=list)
    
    # Market data context
    price: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    market_cap: float = 0.0
    
    # Signal metadata
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(minutes=30))
    
    # Validation data
    historical_accuracy: float = 0.0
    similar_signals_performance: float = 0.0
    
    # Risk assessment
    risk_factors: List[str] = field(default_factory=list)
    recommended_position_size: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and storage"""
        return {
            'token_address': self.token_address,
            'signal_type': self.signal_type,
            'overall_confidence': self.overall_confidence,
            'quality': self.quality.value,
            'components': [
                {
                    'name': c.name,
                    'value': c.value,
                    'weight': c.weight,
                    'confidence': c.confidence,
                    'explanation': c.explanation
                }
                for c in self.components
            ],
            'market_data': {
                'price': self.price,
                'volume_24h': self.volume_24h,
                'liquidity': self.liquidity,
                'market_cap': self.market_cap
            },
            'validation': {
                'historical_accuracy': self.historical_accuracy,
                'similar_signals_performance': self.similar_signals_performance
            },
            'risk_assessment': {
                'risk_factors': self.risk_factors,
                'recommended_position_size': self.recommended_position_size
            },
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat()
        }

class SignalValidator:
    """Validates and tracks signal performance for continuous improvement"""
    
    def __init__(self):
        self.signal_history: List[Dict[str, Any]] = []
        self.performance_tracking: Dict[str, List[float]] = {}
    
    def validate_signal(self, signal: EnhancedSignal) -> bool:
        """Validate signal quality and log for tracking"""
        try:
            # Basic validation checks
            if signal.overall_confidence < 0.2:
                return False
                
            if signal.quality == SignalQuality.REJECT:
                return False
                
            if not signal.components:
                logger.warning(f"[SIGNAL_VALIDATOR] No components in signal for {signal.token_address[:8]}...")
                return False
            
            # Store for performance tracking
            self.signal_history.append({
                'token_address': signal.token_address,
                'confidence': signal.overall_confidence,
                'quality': signal.quality.value,
                'timestamp': signal.timestamp.isoformat(),
                'components_count': len(signal.components)
            })
            
            # Keep only recent history (last 1000 signals)
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            return True
            
        except Exception as e:
            logger.error(f"[SIGNAL_VALIDATOR] Validation error: {e}")
            return False
    
    def get_historical_accuracy(self, signal_type: str) -> float:
        """Get historical accuracy for similar signals"""
        try:
            if signal_type not in self.performance_tracking:
                return 0.5  # Default 50% assumed accuracy
                
            accuracies = self.performance_tracking[signal_type]
            if not accuracies:
                return 0.5
                
            return np.mean(accuracies)
            
        except Exception:
            return 0.5

class EnhancedSignalGenerator:
    """
    Next-generation signal generator with transparency and data-driven approach
    """
    
    def __init__(self, settings: Any):
        self.settings = settings
        self.validator = SignalValidator()
        
        # Feature extractors
        self.momentum_analyzer = MomentumAnalyzer()
        self.volume_analyzer = VolumeProfileAnalyzer() 
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.market_structure_analyzer = MarketStructureAnalyzer()
        
        # Signal composition weights (can be optimized through backtesting)
        self.component_weights = {
            'momentum': 0.25,
            'volume_profile': 0.20,
            'liquidity_depth': 0.15,
            'volatility_pattern': 0.15,
            'market_structure': 0.15,
            'risk_assessment': 0.10
        }
        
        logger.info("[ENHANCED_SIGNALS] Enhanced signal generator initialized with transparent scoring")
    
    async def analyze_token(self, token_data: Any) -> Optional[EnhancedSignal]:
        """
        Generate enhanced signal with full transparency and component breakdown
        """
        try:
            # Normalize token data
            token_dict = self._normalize_token_data(token_data)
            if not self._validate_basic_requirements(token_dict):
                return None
            
            logger.info(f"[ENHANCED_SIGNALS] Analyzing {token_dict['address'][:8]}... for enhanced signal generation")
            
            # Generate all signal components
            components = await self._generate_signal_components(token_dict)
            
            if not components:
                logger.info(f"[ENHANCED_SIGNALS] No valid components generated for {token_dict['address'][:8]}...")
                return None
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(components)
            
            # Determine signal quality
            quality = self._determine_signal_quality(overall_confidence, components)
            
            # Get historical validation data
            historical_accuracy = self.validator.get_historical_accuracy("momentum")
            
            # Create enhanced signal
            enhanced_signal = EnhancedSignal(
                token_address=token_dict['address'],
                signal_type="ENHANCED_MOMENTUM",
                overall_confidence=overall_confidence,
                quality=quality,
                components=components,
                price=float(token_dict.get('price', 0)),
                volume_24h=float(token_dict.get('volume24h', 0)),
                liquidity=float(token_dict.get('liquidity', 0)),
                market_cap=float(token_dict.get('market_cap', 0)),
                historical_accuracy=historical_accuracy,
                recommended_position_size=self._calculate_recommended_position_size(overall_confidence, token_dict)
            )
            
            # Add risk factors
            enhanced_signal.risk_factors = self._identify_risk_factors(token_dict, components)
            
            # Validate signal
            if not self.validator.validate_signal(enhanced_signal):
                logger.info(f"[ENHANCED_SIGNALS] Signal validation failed for {token_dict['address'][:8]}...")
                return None
            
            # Log detailed signal information
            self._log_signal_details(enhanced_signal)
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"[ENHANCED_SIGNALS] Error analyzing token: {e}")
            return None
    
    async def _generate_signal_components(self, token_dict: Dict[str, Any]) -> List[SignalComponent]:
        """Generate all signal components with detailed analysis"""
        components = []
        
        try:
            # 1. Momentum Analysis
            momentum_component = await self._analyze_momentum(token_dict)
            if momentum_component:
                components.append(momentum_component)
            
            # 2. Volume Profile Analysis
            volume_component = await self._analyze_volume_profile(token_dict)
            if volume_component:
                components.append(volume_component)
            
            # 3. Liquidity Depth Analysis
            liquidity_component = await self._analyze_liquidity_depth(token_dict)
            if liquidity_component:
                components.append(liquidity_component)
            
            # 4. Volatility Pattern Analysis
            volatility_component = await self._analyze_volatility_pattern(token_dict)
            if volatility_component:
                components.append(volatility_component)
            
            # 5. Market Structure Analysis
            structure_component = await self._analyze_market_structure(token_dict)
            if structure_component:
                components.append(structure_component)
            
            # 6. Risk Assessment
            risk_component = await self._analyze_risk_factors(token_dict)
            if risk_component:
                components.append(risk_component)
            
        except Exception as e:
            logger.error(f"[ENHANCED_SIGNALS] Error generating components: {e}")
        
        return components
    
    async def _analyze_momentum(self, token_dict: Dict[str, Any]) -> Optional[SignalComponent]:
        """Analyze momentum with transparent scoring"""
        try:
            price = float(token_dict.get('price', 0))
            volume = float(token_dict.get('volume24h', 0))
            
            if price <= 0 or volume <= 0:
                return None
            
            # Calculate momentum score
            momentum_score = self.momentum_analyzer.calculate_momentum_score(
                current_price=price,
                volume_24h=volume,
                market_cap=float(token_dict.get('market_cap', 0))
            )
            
            # Calculate confidence based on data quality
            confidence = min(0.9, momentum_score['data_quality'] * 0.8 + 0.2)
            
            explanation = (
                f"Price momentum: {momentum_score['price_momentum']:.2f}, "
                f"Volume spike: {momentum_score['volume_spike']:.2f}, "
                f"Market cap growth: {momentum_score['mcap_growth']:.2f}"
            )
            
            return SignalComponent(
                name="momentum",
                value=momentum_score['overall_score'],
                weight=self.component_weights['momentum'],
                confidence=confidence,
                explanation=explanation,
                raw_data=momentum_score
            )
            
        except Exception as e:
            logger.error(f"[ENHANCED_SIGNALS] Momentum analysis error: {e}")
            return None
    
    async def _analyze_volume_profile(self, token_dict: Dict[str, Any]) -> Optional[SignalComponent]:
        """Analyze volume profile patterns"""
        try:
            volume = float(token_dict.get('volume24h', 0))
            
            if volume <= 0:
                return None
            
            volume_analysis = self.volume_analyzer.analyze_volume_profile(
                volume_24h=volume,
                liquidity=float(token_dict.get('liquidity', 0))
            )
            
            confidence = volume_analysis['confidence']
            
            explanation = (
                f"Volume consistency: {volume_analysis['consistency']:.2f}, "
                f"Liquidity ratio: {volume_analysis['liquidity_ratio']:.2f}, "
                f"Sustainability: {volume_analysis['sustainability']:.2f}"
            )
            
            return SignalComponent(
                name="volume_profile",
                value=volume_analysis['score'],
                weight=self.component_weights['volume_profile'],
                confidence=confidence,
                explanation=explanation,
                raw_data=volume_analysis
            )
            
        except Exception as e:
            logger.error(f"[ENHANCED_SIGNALS] Volume profile analysis error: {e}")
            return None
    
    async def _analyze_liquidity_depth(self, token_dict: Dict[str, Any]) -> Optional[SignalComponent]:
        """Analyze liquidity depth and market impact"""
        try:
            liquidity = float(token_dict.get('liquidity', 0))
            
            if liquidity <= 0:
                return None
            
            liquidity_analysis = self.liquidity_analyzer.analyze_depth(
                liquidity=liquidity,
                volume_24h=float(token_dict.get('volume24h', 0))
            )
            
            confidence = liquidity_analysis['confidence']
            
            explanation = (
                f"Depth score: {liquidity_analysis['depth_score']:.2f}, "
                f"Market impact: {liquidity_analysis['market_impact']:.2f}, "
                f"Slippage estimate: {liquidity_analysis['estimated_slippage']:.2%}"
            )
            
            return SignalComponent(
                name="liquidity_depth",
                value=liquidity_analysis['score'],
                weight=self.component_weights['liquidity_depth'],
                confidence=confidence,
                explanation=explanation,
                raw_data=liquidity_analysis
            )
            
        except Exception as e:
            logger.error(f"[ENHANCED_SIGNALS] Liquidity analysis error: {e}")
            return None
    
    async def _analyze_volatility_pattern(self, token_dict: Dict[str, Any]) -> Optional[SignalComponent]:
        """Analyze volatility patterns for signal quality"""
        try:
            price = float(token_dict.get('price', 0))
            volume = float(token_dict.get('volume24h', 0))
            
            if price <= 0:
                return None
            
            volatility_analysis = self.volatility_analyzer.analyze_pattern(
                price=price,
                volume=volume,
                market_cap=float(token_dict.get('market_cap', 0))
            )
            
            confidence = volatility_analysis['confidence']
            
            explanation = (
                f"Volatility trend: {volatility_analysis['trend']}, "
                f"Pattern strength: {volatility_analysis['pattern_strength']:.2f}, "
                f"Predictability: {volatility_analysis['predictability']:.2f}"
            )
            
            return SignalComponent(
                name="volatility_pattern",
                value=volatility_analysis['score'],
                weight=self.component_weights['volatility_pattern'],
                confidence=confidence,
                explanation=explanation,
                raw_data=volatility_analysis
            )
            
        except Exception as e:
            logger.error(f"[ENHANCED_SIGNALS] Volatility analysis error: {e}")
            return None
    
    async def _analyze_market_structure(self, token_dict: Dict[str, Any]) -> Optional[SignalComponent]:
        """Analyze overall market structure and health"""
        try:
            structure_analysis = self.market_structure_analyzer.analyze(
                price=float(token_dict.get('price', 0)),
                volume=float(token_dict.get('volume24h', 0)),
                liquidity=float(token_dict.get('liquidity', 0)),
                market_cap=float(token_dict.get('market_cap', 0))
            )
            
            confidence = structure_analysis['confidence']
            
            explanation = (
                f"Market health: {structure_analysis['health_score']:.2f}, "
                f"Structure quality: {structure_analysis['quality']}, "
                f"Sustainability: {structure_analysis['sustainability']:.2f}"
            )
            
            return SignalComponent(
                name="market_structure",
                value=structure_analysis['score'],
                weight=self.component_weights['market_structure'],
                confidence=confidence,
                explanation=explanation,
                raw_data=structure_analysis
            )
            
        except Exception as e:
            logger.error(f"[ENHANCED_SIGNALS] Market structure analysis error: {e}")
            return None
    
    async def _analyze_risk_factors(self, token_dict: Dict[str, Any]) -> Optional[SignalComponent]:
        """Comprehensive risk factor analysis"""
        try:
            risk_factors = []
            risk_score = 0.8  # Start with high score, deduct for risks
            
            # Check liquidity risk
            liquidity = float(token_dict.get('liquidity', 0))
            if liquidity < self.settings.MIN_LIQUIDITY:
                risk_factors.append("LOW_LIQUIDITY")
                risk_score -= 0.2
            
            # Check volume sustainability
            volume = float(token_dict.get('volume24h', 0))
            if volume < self.settings.MIN_VOLUME_24H:
                risk_factors.append("LOW_VOLUME")
                risk_score -= 0.15
            
            # Check market cap size
            market_cap = float(token_dict.get('market_cap', 0))
            if market_cap < 100000:  # Less than 100K market cap
                risk_factors.append("MICRO_CAP")
                risk_score -= 0.1
            
            # Check price volatility
            price = float(token_dict.get('price', 0))
            if price < 0.0001:  # Very low price tokens
                risk_factors.append("EXTREMELY_LOW_PRICE")
                risk_score -= 0.1
            
            risk_score = max(0.0, min(1.0, risk_score))
            confidence = 0.9  # High confidence in risk assessment
            
            explanation = f"Risk factors: {', '.join(risk_factors) if risk_factors else 'None identified'}"
            
            return SignalComponent(
                name="risk_assessment",
                value=risk_score,
                weight=self.component_weights['risk_assessment'],
                confidence=confidence,
                explanation=explanation,
                raw_data={'risk_factors': risk_factors, 'risk_score': risk_score}
            )
            
        except Exception as e:
            logger.error(f"[ENHANCED_SIGNALS] Risk analysis error: {e}")
            return None
    
    def _calculate_overall_confidence(self, components: List[SignalComponent]) -> float:
        """Calculate weighted overall confidence"""
        try:
            if not components:
                return 0.0
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for component in components:
                weighted_contribution = component.value * component.weight * component.confidence
                weighted_sum += weighted_contribution
                total_weight += component.weight
            
            if total_weight == 0:
                return 0.0
            
            overall_confidence = weighted_sum / total_weight
            return max(0.0, min(1.0, overall_confidence))
            
        except Exception as e:
            logger.error(f"[ENHANCED_SIGNALS] Confidence calculation error: {e}")
            return 0.0
    
    def _determine_signal_quality(self, confidence: float, components: List[SignalComponent]) -> SignalQuality:
        """Determine signal quality based on confidence and component analysis"""
        try:
            # Base quality from confidence
            if confidence >= 0.8:
                quality = SignalQuality.EXCELLENT
            elif confidence >= 0.6:
                quality = SignalQuality.GOOD
            elif confidence >= 0.25:
                quality = SignalQuality.FAIR
            elif confidence >= 0.2:
                quality = SignalQuality.POOR
            else:
                quality = SignalQuality.REJECT
            
            # Adjust based on component consistency
            component_confidences = [c.confidence for c in components]
            if component_confidences:
                std_dev = np.std(component_confidences)
                if std_dev > 0.3:  # High variance in component confidence
                    # Downgrade quality by one level
                    quality_levels = [SignalQuality.EXCELLENT, SignalQuality.GOOD, SignalQuality.FAIR, SignalQuality.POOR, SignalQuality.REJECT]
                    current_index = quality_levels.index(quality)
                    if current_index < len(quality_levels) - 1:
                        quality = quality_levels[current_index + 1]
            
            return quality
            
        except Exception as e:
            logger.error(f"[ENHANCED_SIGNALS] Quality determination error: {e}")
            return SignalQuality.REJECT
    
    def _calculate_recommended_position_size(self, confidence: float, token_dict: Dict[str, Any]) -> float:
        """Calculate recommended position size based on confidence and risk"""
        try:
            base_size = 0.02  # 2% base position size
            
            # Adjust based on confidence
            confidence_multiplier = confidence
            
            # Adjust based on liquidity
            liquidity = float(token_dict.get('liquidity', 0))
            if liquidity < 500000:  # Less than 500K liquidity
                liquidity_multiplier = 0.5
            elif liquidity < 1000000:  # Less than 1M liquidity
                liquidity_multiplier = 0.75
            else:
                liquidity_multiplier = 1.0
            
            recommended_size = base_size * confidence_multiplier * liquidity_multiplier
            return max(0.005, min(0.05, recommended_size))  # Clamp between 0.5% and 5%
            
        except Exception as e:
            logger.error(f"[ENHANCED_SIGNALS] Position size calculation error: {e}")
            return 0.01  # Default 1%
    
    def _identify_risk_factors(self, token_dict: Dict[str, Any], components: List[SignalComponent]) -> List[str]:
        """Identify specific risk factors for the signal"""
        risk_factors = []
        
        try:
            # Extract risk factors from risk assessment component
            risk_component = next((c for c in components if c.name == "risk_assessment"), None)
            if risk_component and 'risk_factors' in risk_component.raw_data:
                risk_factors.extend(risk_component.raw_data['risk_factors'])
            
            # Add additional contextual risk factors
            if float(token_dict.get('volume24h', 0)) < 10000:
                risk_factors.append("VERY_LOW_VOLUME")
            
            if len(components) < 4:
                risk_factors.append("INSUFFICIENT_DATA")
            
            # Check component consistency
            component_values = [c.value for c in components]
            if component_values and np.std(component_values) > 0.4:
                risk_factors.append("INCONSISTENT_SIGNALS")
            
        except Exception as e:
            logger.error(f"[ENHANCED_SIGNALS] Risk factor identification error: {e}")
        
        return list(set(risk_factors))  # Remove duplicates
    
    def _log_signal_details(self, signal: EnhancedSignal):
        """Log detailed signal information for transparency"""
        try:
            logger.info(f"[ENHANCED_SIGNALS] 📊 Signal Generated for {signal.token_address[:8]}...")
            logger.info(f"[ENHANCED_SIGNALS]   Overall Confidence: {signal.overall_confidence:.3f}")
            logger.info(f"[ENHANCED_SIGNALS]   Quality: {signal.quality.value}")
            logger.info(f"[ENHANCED_SIGNALS]   Components: {len(signal.components)}")
            
            for component in signal.components:
                logger.info(f"[ENHANCED_SIGNALS]     {component.name}: {component.value:.3f} "
                           f"(weight: {component.weight:.2f}, confidence: {component.confidence:.3f})")
                logger.info(f"[ENHANCED_SIGNALS]       → {component.explanation}")
            
            if signal.risk_factors:
                logger.info(f"[ENHANCED_SIGNALS]   Risk Factors: {', '.join(signal.risk_factors)}")
            
            logger.info(f"[ENHANCED_SIGNALS]   Recommended Size: {signal.recommended_position_size:.2%}")
            
        except Exception as e:
            logger.error(f"[ENHANCED_SIGNALS] Logging error: {e}")
    
    def _normalize_token_data(self, token_data: Any) -> Dict[str, Any]:
        """Normalize token data to standard format"""
        if hasattr(token_data, 'address'):
            return {
                'address': getattr(token_data, 'address', ''),
                'price': getattr(token_data, 'price_sol', 0),
                'volume24h': getattr(token_data, 'volume24h', 0),
                'liquidity': getattr(token_data, 'liquidity', 0),
                'market_cap': getattr(token_data, 'market_cap', 0)
            }
        else:
            return token_data
    
    def _validate_basic_requirements(self, token_dict: Dict[str, Any]) -> bool:
        """Validate basic token requirements"""
        try:
            if not token_dict.get('address'):
                return False
            
            price = float(token_dict.get('price', 0))
            volume = float(token_dict.get('volume24h', 0))
            liquidity = float(token_dict.get('liquidity', 0))
            
            if price <= 0 or volume <= 0 or liquidity <= 0:
                return False
            
            return True
            
        except Exception:
            return False


# Component analyzers for enhanced signal generation
class MomentumAnalyzer:
    def calculate_momentum_score(self, current_price: float, volume_24h: float, market_cap: float) -> Dict[str, float]:
        try:
            # Price momentum (simplified - in production, use historical data)
            price_momentum = min(1.0, current_price * 1000)  # Normalize price
            
            # Volume spike analysis
            volume_spike = min(1.0, volume_24h / 100000)  # Normalize volume
            
            # Market cap growth potential
            mcap_growth = min(1.0, (1000000 - market_cap) / 1000000) if market_cap < 1000000 else 0.1
            
            overall_score = (price_momentum * 0.4 + volume_spike * 0.4 + mcap_growth * 0.2)
            
            return {
                'price_momentum': price_momentum,
                'volume_spike': volume_spike,
                'mcap_growth': mcap_growth,
                'overall_score': overall_score,
                'data_quality': 0.8  # Assume good data quality
            }
        except Exception:
            return {'price_momentum': 0, 'volume_spike': 0, 'mcap_growth': 0, 'overall_score': 0, 'data_quality': 0.1}

class VolumeProfileAnalyzer:
    def analyze_volume_profile(self, volume_24h: float, liquidity: float) -> Dict[str, float]:
        try:
            consistency = min(1.0, volume_24h / 50000)  # Volume consistency
            liquidity_ratio = min(1.0, liquidity / volume_24h) if volume_24h > 0 else 0
            sustainability = (consistency + liquidity_ratio) / 2
            
            return {
                'consistency': consistency,
                'liquidity_ratio': liquidity_ratio,
                'sustainability': sustainability,
                'score': sustainability,
                'confidence': 0.7
            }
        except Exception:
            return {'consistency': 0, 'liquidity_ratio': 0, 'sustainability': 0, 'score': 0, 'confidence': 0.1}

class LiquidityAnalyzer:
    def analyze_depth(self, liquidity: float, volume_24h: float) -> Dict[str, float]:
        try:
            depth_score = min(1.0, liquidity / 500000)  # Normalize to 500K liquidity
            market_impact = max(0.1, min(1.0, liquidity / (volume_24h * 0.1))) if volume_24h > 0 else 0.1
            estimated_slippage = max(0.01, min(0.1, 1.0 / depth_score * 0.02))
            
            return {
                'depth_score': depth_score,
                'market_impact': market_impact,
                'estimated_slippage': estimated_slippage,
                'score': (depth_score + market_impact) / 2,
                'confidence': 0.8
            }
        except Exception:
            return {'depth_score': 0, 'market_impact': 0.1, 'estimated_slippage': 0.05, 'score': 0, 'confidence': 0.1}

class VolatilityAnalyzer:
    def analyze_pattern(self, price: float, volume: float, market_cap: float) -> Dict[str, float]:
        try:
            # Simplified volatility analysis
            trend = "STABLE"  # Would use historical data in production
            pattern_strength = min(1.0, volume / 100000)
            predictability = 0.6  # Default predictability
            
            return {
                'trend': trend,
                'pattern_strength': pattern_strength,
                'predictability': predictability,
                'score': pattern_strength * predictability,
                'confidence': 0.6
            }
        except Exception:
            return {'trend': 'UNKNOWN', 'pattern_strength': 0, 'predictability': 0, 'score': 0, 'confidence': 0.1}

class MarketStructureAnalyzer:
    def analyze(self, price: float, volume: float, liquidity: float, market_cap: float) -> Dict[str, float]:
        try:
            health_score = min(1.0, (liquidity + volume + market_cap) / 1000000)
            quality = "GOOD" if health_score > 0.5 else "FAIR"
            sustainability = min(1.0, liquidity / (volume * 0.1)) if volume > 0 else 0
            
            return {
                'health_score': health_score,
                'quality': quality,
                'sustainability': sustainability,
                'score': health_score * sustainability,
                'confidence': 0.7
            }
        except Exception:
            return {'health_score': 0, 'quality': 'POOR', 'sustainability': 0, 'score': 0, 'confidence': 0.1}