import logging
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from .market_analysis import MarketRegime, MarketAnalysis

logger = logging.getLogger(__name__)

@dataclass
class EntryRule:
    name: str
    condition: str
    timeframe: str
    weight: float
    is_active: bool = True

@dataclass
class ExitRule:
    name: str
    condition: str
    timeframe: str
    weight: float
    is_active: bool = True

@dataclass
class EntryPoint:
    price: float
    size: float
    stop_loss: float
    take_profit: List[Dict[str, float]]  # Multiple take profit levels
    confidence: float
    entry_type: str
    timeframe: str

@dataclass
class ExitPoint:
    price: float
    size: float  # How much of the position to exit
    reason: str
    urgency: str  # 'low', 'medium', 'high'
    confidence: float

class AdaptiveRules:
    def __init__(self, settings: Any):
        self.settings = settings
        self.entry_rules = self._initialize_entry_rules()
        self.exit_rules = self._initialize_exit_rules()
        self.current_regime = None

    def _initialize_entry_rules(self) -> List[EntryRule]:
        """Initialize default entry rules"""
        return [
            EntryRule(
                name="trend_following",
                condition="price > ma_20 and rsi > 50",
                timeframe="1h",
                weight=0.3
            ),
            EntryRule(
                name="momentum_breakout",
                condition="new_high and volume_surge",
                timeframe="15m",
                weight=0.3
            ),
            EntryRule(
                name="support_bounce",
                condition="near_support and oversold",
                timeframe="1h",
                weight=0.2
            ),
            EntryRule(
                name="volume_confirmation",
                condition="increasing_volume and positive_delta",
                timeframe="5m",
                weight=0.2
            )
        ]

    def _initialize_exit_rules(self) -> List[ExitRule]:
        """Initialize default exit rules"""
        return [
            ExitRule(
                name="trend_reversal",
                condition="price < ma_20 and rsi < 40",
                timeframe="1h",
                weight=0.3
            ),
            ExitRule(
                name="momentum_exhaustion",
                condition="divergence and overbought",
                timeframe="15m",
                weight=0.3
            ),
            ExitRule(
                name="resistance_hit",
                condition="near_resistance and decreasing_volume",
                timeframe="1h",
                weight=0.2
            ),
            ExitRule(
                name="volume_divergence",
                condition="price_up and volume_down",
                timeframe="5m",
                weight=0.2
            )
        ]
    
    def _adapt_for_downtrend(self, analysis: MarketAnalysis) -> None:
        self.settings.TAKE_PROFIT_LEVELS = [0.02, 0.03, 0.05]
        self.settings.STOP_LOSS_PERCENTAGE = 0.015
        self.settings.POSITION_SIZE_FACTOR = 0.8

    def _adapt_for_ranging(self, analysis: MarketAnalysis) -> None:
        self.settings.TAKE_PROFIT_LEVELS = [0.02, 0.04]
        self.settings.STOP_LOSS_PERCENTAGE = 0.02
        self.settings.POSITION_SIZE_FACTOR = 1.0

    def _adapt_for_accumulation(self, analysis: MarketAnalysis) -> None:
        self.settings.TAKE_PROFIT_LEVELS = [0.03, 0.05, 0.08]
        self.settings.STOP_LOSS_PERCENTAGE = 0.02
        self.settings.POSITION_SIZE_FACTOR = 1.1

    def _adapt_for_distribution(self, analysis: MarketAnalysis) -> None:
        self.settings.TAKE_PROFIT_LEVELS = [0.02, 0.03]
        self.settings.STOP_LOSS_PERCENTAGE = 0.02
        self.settings.POSITION_SIZE_FACTOR = 0.7

    def _evaluate_entry_rule(self, rule: EntryRule, price: float, analysis: MarketAnalysis) -> float:
        # Implement rule evaluation logic
        return 0.5  # Default implementation

    def _evaluate_exit_rule(self, rule: ExitRule, position: Any, analysis: MarketAnalysis) -> float:
        # Implement rule evaluation logic
        return 0.5  # Default implementation

    def adapt_to_market_regime(self, market_analysis: MarketAnalysis) -> None:
        """Adapt rules based on market regime"""
        self.current_regime = market_analysis.regime
        
        if market_analysis.regime == MarketRegime.TRENDING_UP:
            self._adapt_for_uptrend(market_analysis)
        elif market_analysis.regime == MarketRegime.TRENDING_DOWN:
            self._adapt_for_downtrend(market_analysis)
        elif market_analysis.regime == MarketRegime.VOLATILE:
            self._adapt_for_volatile(market_analysis)
        elif market_analysis.regime == MarketRegime.RANGING:
            self._adapt_for_ranging(market_analysis)
        elif market_analysis.regime == MarketRegime.ACCUMULATION:
            self._adapt_for_accumulation(market_analysis)
        elif market_analysis.regime == MarketRegime.DISTRIBUTION:
            self._adapt_for_distribution(market_analysis)

    def _adapt_for_uptrend(self, analysis: MarketAnalysis) -> None:
        """Adapt rules for uptrending market"""
        # Adjust take profit levels wider
        self.settings.TAKE_PROFIT_LEVELS = [0.03, 0.05, 0.08]
        # Tighten stop loss
        self.settings.STOP_LOSS_PERCENTAGE = 0.02
        # Increase position sizes
        self.settings.POSITION_SIZE_FACTOR = 1.2
        
        # Activate trend following rules
        for rule in self.entry_rules:
            if rule.name == "trend_following":
                rule.weight = 0.4
            elif rule.name == "momentum_breakout":
                rule.weight = 0.3
            else:
                rule.weight = 0.15

    def _adapt_for_volatile(self, analysis: MarketAnalysis) -> None:
        """Adapt rules for volatile market"""
        # Tighter take profits
        self.settings.TAKE_PROFIT_LEVELS = [0.02, 0.03, 0.04]
        # Wider stops
        self.settings.STOP_LOSS_PERCENTAGE = 0.03
        # Reduce position sizes
        self.settings.POSITION_SIZE_FACTOR = 0.7
        
        # Adjust rule weights
        for rule in self.entry_rules:
            if rule.name == "support_bounce":
                rule.weight = 0.4
            else:
                rule.weight = 0.2

    def calculate_entry_point(self, price: float, analysis: MarketAnalysis, balance: float) -> Optional[EntryPoint]:
        """Calculate optimal entry point based on current conditions"""
        try:
            # Calculate base position size
            position_size = self._calculate_position_size(price, balance, analysis)
            
            # Calculate stop loss and take profit levels
            stop_loss = self._calculate_stop_loss(price, analysis)
            take_profit_levels = self._calculate_take_profit_levels(price, analysis)
            
            # Calculate entry confidence
            confidence = self._calculate_entry_confidence(price, analysis)
            
            if confidence < self.settings.SIGNAL_THRESHOLD:
                return None
                
            return EntryPoint(
                price=price,
                size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit_levels,
                confidence=confidence,
                entry_type=self._determine_entry_type(analysis),
                timeframe=analysis.timeframe
            )
            
        except Exception as e:
            logger.error(f"Error calculating entry point: {str(e)}")
            return None

    def calculate_exit_point(self, position: Any, analysis: MarketAnalysis) -> Optional[ExitPoint]:
        """Calculate optimal exit point for a position"""
        try:
            exit_scores = self._evaluate_exit_rules(position, analysis)
            if not exit_scores:
                return None

            max_score = max(exit_scores.values())
            if max_score < self.settings.EXIT_THRESHOLD:
                return None

            urgency = self._determine_exit_urgency(exit_scores, analysis)
            size_to_exit = self._calculate_exit_size(position, urgency)

            return ExitPoint(
                price=analysis.current_price,
                size=size_to_exit,
                reason=max(exit_scores.items(), key=lambda x: x[1])[0],
                urgency=urgency,
                confidence=max_score
            )

        except Exception as e:
            logger.error(f"Error calculating exit point: {str(e)}")
            return None

    def _calculate_position_size(self, price: float, balance: float, analysis: MarketAnalysis) -> float:
        """Calculate adaptive position size"""
        try:
            # Base size from balance
            base_size = balance * self.settings.BASE_POSITION_SIZE
            
            # Adjust for regime
            regime_factor = self._get_regime_size_factor()
            
            # Adjust for volatility
            volatility_factor = 1 - (analysis.volatility * 0.5)
            
            # Final size
            size = base_size * regime_factor * volatility_factor
            
            # Apply limits
            return min(size, self.settings.MAX_POSITION_SIZE)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return float(self.settings.MIN_POSITION_SIZE)

    def _calculate_stop_loss(self, price: float, analysis: MarketAnalysis) -> float:
        """Calculate adaptive stop loss level"""
        try:
            # Base stop loss percentage
            base_stop = self.settings.STOP_LOSS_PERCENTAGE
            
            # Adjust for volatility
            volatility_factor = 1 + (analysis.volatility * 0.5)
            
            # Adjust for regime
            regime_factor = self._get_regime_stop_factor()
            
            # Calculate final stop distance
            stop_distance = base_stop * volatility_factor * regime_factor
            
            # Apply stop loss
            return price * (1 - stop_distance) if analysis.regime == MarketRegime.TRENDING_UP else price * (1 + stop_distance)
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return price * (1 - self.settings.STOP_LOSS_PERCENTAGE)

    def _calculate_take_profit_levels(self, price: float, analysis: MarketAnalysis) -> List[Dict[str, float]]:
        """Calculate multiple take profit levels"""
        try:
            base_levels = self.settings.TAKE_PROFIT_LEVELS
            trend_strength = analysis.trend_strength
            
            levels = []
            for i, base_level in enumerate(base_levels):
                # Adjust level based on trend strength
                adjusted_level = base_level * (1 + trend_strength)
                
                # Calculate target price
                target_price = price * (1 + adjusted_level)
                
                # Calculate size to exit at this level
                size_percentage = 0.4 if i == 0 else 0.3 if i == 1 else 0.3
                
                levels.append({
                    'price': target_price,
                    'size_percentage': size_percentage
                })
                
            return levels
            
        except Exception as e:
            logger.error(f"Error calculating take profit levels: {str(e)}")
            return []

    def _evaluate_exit_rules(self, position: Any, analysis: MarketAnalysis) -> Dict[str, float]:
        """Evaluate all exit rules"""
        exit_scores = {}
        
        for rule in self.exit_rules:
            if rule.is_active:
                score = self._evaluate_exit_rule(rule, position, analysis)
                if score > 0:
                    exit_scores[rule.name] = score
                    
        return exit_scores

    def _determine_exit_urgency(self, exit_scores: Dict[str, float], analysis: MarketAnalysis) -> str:
        """Determine how urgent the exit is"""
        max_score = max(exit_scores.values())
        
        if max_score > 0.8 or analysis.regime == MarketRegime.VOLATILE:
            return 'high'
        elif max_score > 0.6:
            return 'medium'
        return 'low'

    def _calculate_exit_size(self, position: Any, urgency: str) -> float:
        """Calculate how much of the position to exit"""
        urgency_factors = {
            'low': 0.3,
            'medium': 0.6,
            'high': 1.0
        }
        return position.size * urgency_factors.get(urgency, 0.3)
    

    def _get_regime_size_factor(self) -> float:
        """Get position size factor based on regime"""
        if self.current_regime is None:
            return 1.0

        factors = {
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 0.8,
            MarketRegime.VOLATILE: 0.6,
            MarketRegime.RANGING: 1.0,
            MarketRegime.ACCUMULATION: 1.1,
            MarketRegime.DISTRIBUTION: 0.7
        }
        return factors[self.current_regime] if self.current_regime in factors else 1.0


    def _get_regime_stop_factor(self) -> float:
        """Get stop loss factor based on regime"""
        if self.current_regime is None:
            return 1.0

        factors = {
            MarketRegime.TRENDING_UP: 0.8,
            MarketRegime.TRENDING_DOWN: 0.8,
            MarketRegime.VOLATILE: 1.5,
            MarketRegime.RANGING: 1.2,
            MarketRegime.ACCUMULATION: 1.0,
            MarketRegime.DISTRIBUTION: 1.0
        }
        return factors[self.current_regime] if self.current_regime in factors else 1.0

    def _calculate_entry_confidence(self, price: float, analysis: MarketAnalysis) -> float:
        """Calculate confidence score for entry"""
        try:
            # Evaluate all active entry rules
            rule_scores = []
            total_weight = 0
            
            for rule in self.entry_rules:
                if rule.is_active:
                    score = self._evaluate_entry_rule(rule, price, analysis)
                    rule_scores.append(score * rule.weight)
                    total_weight += rule.weight
                    
            if total_weight == 0:
                return 0.0
                
            # Calculate weighted average
            confidence = sum(rule_scores) / total_weight
            
            # Adjust for market conditions
            confidence *= (1 + analysis.trend_strength)
            confidence *= (1 - analysis.volatility * 0.5)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating entry confidence: {str(e)}")
            return 0.0

    def _determine_entry_type(self, analysis: MarketAnalysis) -> str:
        """Determine entry type based on market conditions"""
        if analysis.regime == MarketRegime.TRENDING_UP:
            return 'trend_following'
        elif analysis.regime == MarketRegime.RANGING:
            return 'range_bound'
        elif analysis.regime == MarketRegime.VOLATILE:
            return 'breakout'
        elif analysis.regime == MarketRegime.ACCUMULATION:
            return 'accumulation'
        return 'standard'