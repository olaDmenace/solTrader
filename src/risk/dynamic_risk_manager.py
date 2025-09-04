#!/usr/bin/env python3
"""
Dynamic Risk Manager - Enhanced Risk Management 🛡️

This module provides:
- Dynamic position sizing based on signal confidence
- Daily loss limits with automatic trading pause
- Portfolio heat tracking
- Signal confidence assessment
- Risk-adjusted position sizing
- Automatic trading suspension

Integrates with Enhanced Exit Manager and Position Tracker for comprehensive risk control.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SignalConfidence:
    """Signal confidence assessment"""
    overall_confidence: float  # 0.0 to 1.0
    momentum_strength: float
    liquidity_confidence: float
    volume_confidence: float
    technical_confidence: float
    risk_factors: List[str]
    confidence_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'

@dataclass
class DynamicPositionSize:
    """Dynamic position sizing calculation"""
    base_size: float
    confidence_adjusted_size: float
    risk_adjusted_size: float
    final_position_size: float
    size_reduction_factors: List[str]
    max_position_cap: float

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment"""
    portfolio_risk_score: float  # 0.0 to 10.0
    daily_loss_percentage: float
    position_concentration: float
    volatility_risk: float
    liquidity_risk: float
    correlation_risk: float
    overall_risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    risk_recommendations: List[str]

class DynamicRiskManager:
    """Advanced dynamic risk management system"""
    
    def __init__(self, settings):
        self.settings = settings
        self.daily_pnl_tracker = 0.0
        self.trading_paused = False
        self.pause_reason = ""
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.position_correlation_history = {}
        self.volatility_history = {}
        
        # Risk management parameters
        self.BASE_POSITION_SIZE = 0.02  # 2% base position size
        self.MAX_POSITION_SIZE = settings.MAX_POSITION_SIZE  # From settings (5%)
        self.CONFIDENCE_MULTIPLIER = 2.0  # How much confidence affects sizing
        self.VOLATILITY_PENALTY = 0.5  # Reduce size for high volatility
        self.CORRELATION_PENALTY = 0.3  # Reduce size for correlated positions
        
        logger.info("Dynamic Risk Manager initialized with enhanced controls")
    
    def reset_daily_tracking_if_needed(self):
        """Reset daily tracking at midnight"""
        now = datetime.now()
        if now.date() > self.daily_reset_time.date():
            self.daily_pnl_tracker = 0.0
            self.trading_paused = False
            self.pause_reason = ""
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            logger.info("Daily risk tracking reset")
    
    def update_daily_pnl(self, pnl_change: float):
        """Update daily P&L and check limits"""
        self.reset_daily_tracking_if_needed()
        self.daily_pnl_tracker += pnl_change
        
        # Check daily loss limit
        daily_loss_pct = self.daily_pnl_tracker / self.settings.PORTFOLIO_VALUE if self.settings.PORTFOLIO_VALUE > 0 else 0
        
        if daily_loss_pct <= -self.settings.MAX_DAILY_LOSS:
            self.pause_trading("Daily loss limit exceeded")
            logger.critical(f"TRADING PAUSED: Daily loss {daily_loss_pct:.2%} exceeds limit {-self.settings.MAX_DAILY_LOSS:.2%}")
    
    def pause_trading(self, reason: str):
        """Pause all trading activities"""
        self.trading_paused = True
        self.pause_reason = reason
        logger.critical(f"TRADING PAUSED: {reason}")
    
    def resume_trading(self):
        """Resume trading activities (manual override)"""
        self.trading_paused = False
        self.pause_reason = ""
        logger.info("Trading resumed manually")
    
    def assess_signal_confidence(self, signal_data: Dict) -> SignalConfidence:
        """Assess confidence level of trading signal"""
        
        # Extract signal components
        momentum_strength = signal_data.get('momentum_strength', 0.5)
        liquidity = signal_data.get('liquidity', 0)
        volume_24h = signal_data.get('volume_24h', 0)
        price_change = signal_data.get('price_change_percentage', 0)
        technical_indicators = signal_data.get('technical_score', 0.5)
        
        # Assess each component
        momentum_confidence = min(abs(momentum_strength) / 0.1, 1.0)  # Normalize to 1.0
        
        # Liquidity confidence (higher is better)
        liquidity_confidence = min(liquidity / self.settings.MIN_LIQUIDITY, 1.0)
        
        # Volume confidence
        volume_confidence = min(volume_24h / self.settings.MIN_VOLUME_24H, 1.0)
        
        # Technical confidence
        technical_confidence = technical_indicators
        
        # Risk factors assessment
        risk_factors = []
        if liquidity < self.settings.MIN_LIQUIDITY * 2:
            risk_factors.append("Low liquidity")
        if volume_24h < self.settings.MIN_VOLUME_24H * 2:
            risk_factors.append("Low volume")
        if abs(price_change) > 50:  # Very high volatility
            risk_factors.append("Extreme volatility")
        
        # Calculate overall confidence
        confidence_components = [
            momentum_confidence * 0.3,
            liquidity_confidence * 0.25,
            volume_confidence * 0.25,
            technical_confidence * 0.2
        ]
        
        overall_confidence = sum(confidence_components)
        
        # Apply risk factor penalties
        for risk_factor in risk_factors:
            overall_confidence *= 0.9  # 10% penalty per risk factor
        
        # Determine confidence level
        if overall_confidence >= 0.8:
            confidence_level = "VERY_HIGH"
        elif overall_confidence >= 0.65:
            confidence_level = "HIGH"
        elif overall_confidence >= 0.45:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        return SignalConfidence(
            overall_confidence=overall_confidence,
            momentum_strength=momentum_confidence,
            liquidity_confidence=liquidity_confidence,
            volume_confidence=volume_confidence,
            technical_confidence=technical_confidence,
            risk_factors=risk_factors,
            confidence_level=confidence_level
        )
    
    def calculate_dynamic_position_size(self, signal_confidence: SignalConfidence, 
                                      current_positions: List[Dict]) -> DynamicPositionSize:
        """Calculate dynamic position size based on multiple factors"""
        
        # Start with base position size
        base_size = self.BASE_POSITION_SIZE
        
        # Adjust for signal confidence
        confidence_multiplier = 1.0 + (signal_confidence.overall_confidence - 0.5) * self.CONFIDENCE_MULTIPLIER
        confidence_adjusted_size = base_size * confidence_multiplier
        
        # Risk adjustments
        risk_adjusted_size = confidence_adjusted_size
        size_reduction_factors = []
        
        # Volatility penalty
        if "Extreme volatility" in signal_confidence.risk_factors:
            risk_adjusted_size *= (1 - self.VOLATILITY_PENALTY)
            size_reduction_factors.append("High volatility penalty")
        
        # Portfolio concentration check
        current_exposure = sum(pos.get('current_value', 0) for pos in current_positions)
        portfolio_exposure = current_exposure / self.settings.PORTFOLIO_VALUE if self.settings.PORTFOLIO_VALUE > 0 else 0
        
        if portfolio_exposure > 0.15:  # Already 15% exposed
            risk_adjusted_size *= 0.7  # Reduce by 30%
            size_reduction_factors.append("High portfolio exposure")
        
        # Correlation penalty (simplified - would need more sophisticated implementation)
        if len(current_positions) >= 2:  # Multiple positions might be correlated
            risk_adjusted_size *= (1 - self.CORRELATION_PENALTY)
            size_reduction_factors.append("Position correlation risk")
        
        # Apply hard caps
        final_position_size = min(risk_adjusted_size, self.MAX_POSITION_SIZE)
        if final_position_size < risk_adjusted_size:
            size_reduction_factors.append(f"Hard cap at {self.MAX_POSITION_SIZE:.1%}")
        
        # Ensure minimum viable size
        min_viable_size = 0.005  # 0.5% minimum
        final_position_size = max(final_position_size, min_viable_size)
        
        return DynamicPositionSize(
            base_size=base_size,
            confidence_adjusted_size=confidence_adjusted_size,
            risk_adjusted_size=risk_adjusted_size,
            final_position_size=final_position_size,
            size_reduction_factors=size_reduction_factors,
            max_position_cap=self.MAX_POSITION_SIZE
        )
    
    def assess_portfolio_risk(self, positions: List[Dict]) -> RiskAssessment:
        """Assess comprehensive portfolio risk"""
        
        self.reset_daily_tracking_if_needed()
        
        if not positions:
            return RiskAssessment(
                portfolio_risk_score=0.0,
                daily_loss_percentage=self.daily_pnl_tracker / self.settings.PORTFOLIO_VALUE if self.settings.PORTFOLIO_VALUE > 0 else 0,
                position_concentration=0.0,
                volatility_risk=0.0,
                liquidity_risk=0.0,
                correlation_risk=0.0,
                overall_risk_level="LOW",
                risk_recommendations=["Portfolio is empty - ready for new positions"]
            )
        
        # Calculate risk components
        total_value = sum(pos.get('current_value', 0) for pos in positions)
        largest_position = max(pos.get('current_value', 0) for pos in positions)
        
        # Position concentration risk
        position_concentration = largest_position / self.settings.PORTFOLIO_VALUE if self.settings.PORTFOLIO_VALUE > 0 else 0
        
        # Daily loss percentage
        daily_loss_pct = self.daily_pnl_tracker / self.settings.PORTFOLIO_VALUE if self.settings.PORTFOLIO_VALUE > 0 else 0
        
        # Volatility risk (based on position P&Ls)
        pnl_percentages = [pos.get('unrealized_pnl_percentage', 0) for pos in positions]
        volatility_risk = np.std(pnl_percentages) if len(pnl_percentages) > 1 else 0
        
        # Liquidity risk (average of all positions)
        liquidities = [pos.get('liquidity', self.settings.MIN_LIQUIDITY) for pos in positions]
        avg_liquidity = sum(liquidities) / len(liquidities)
        liquidity_risk = max(0, 1 - (avg_liquidity / (self.settings.MIN_LIQUIDITY * 2)))  # Risk if below 2x minimum
        
        # Correlation risk (simplified - assumes some correlation between meme tokens)
        correlation_risk = min(len(positions) * 0.1, 1.0)  # Higher with more positions
        
        # Calculate overall portfolio risk score (0-10 scale)
        risk_components = [
            position_concentration * 3,  # Weight: 30%
            abs(daily_loss_pct) * 4,     # Weight: 40%
            volatility_risk * 2,         # Weight: 20%  
            liquidity_risk * 0.5,        # Weight: 5%
            correlation_risk * 0.5       # Weight: 5%
        ]
        
        portfolio_risk_score = sum(risk_components)
        
        # Determine overall risk level and recommendations
        risk_recommendations = []
        
        if portfolio_risk_score <= 2:
            overall_risk_level = "LOW"
            risk_recommendations.append("Risk levels are acceptable")
        elif portfolio_risk_score <= 4:
            overall_risk_level = "MEDIUM"
            risk_recommendations.append("Monitor positions closely")
        elif portfolio_risk_score <= 7:
            overall_risk_level = "HIGH"
            risk_recommendations.append("Consider reducing position sizes")
            risk_recommendations.append("Review stop-loss levels")
        else:
            overall_risk_level = "CRITICAL"
            risk_recommendations.append("URGENT: Reduce positions immediately")
            risk_recommendations.append("Review risk management parameters")
        
        # Specific recommendations
        if position_concentration > 0.08:  # > 8%
            risk_recommendations.append(f"Largest position too big: {position_concentration:.1%}")
        
        if abs(daily_loss_pct) > 0.03:  # > 3%
            risk_recommendations.append(f"Daily loss approaching limit: {daily_loss_pct:.1%}")
        
        if volatility_risk > 0.15:  # > 15% volatility
            risk_recommendations.append("High portfolio volatility detected")
        
        return RiskAssessment(
            portfolio_risk_score=portfolio_risk_score,
            daily_loss_percentage=daily_loss_pct,
            position_concentration=position_concentration,
            volatility_risk=volatility_risk,
            liquidity_risk=liquidity_risk,
            correlation_risk=correlation_risk,
            overall_risk_level=overall_risk_level,
            risk_recommendations=risk_recommendations
        )
    
    def should_allow_new_position(self, signal_data: Dict, current_positions: List[Dict]) -> Tuple[bool, str]:
        """Determine if new position should be allowed"""
        
        # Check if trading is paused
        if self.trading_paused:
            return False, f"Trading paused: {self.pause_reason}"
        
        # Assess signal confidence
        confidence = self.assess_signal_confidence(signal_data)
        
        # Reject very low confidence signals
        if confidence.overall_confidence < 0.3:
            return False, f"Signal confidence too low: {confidence.confidence_level}"
        
        # Check portfolio risk
        risk_assessment = self.assess_portfolio_risk(current_positions)
        
        # Block new positions if risk is critical
        if risk_assessment.overall_risk_level == "CRITICAL":
            return False, "Portfolio risk level CRITICAL - no new positions allowed"
        
        # Check daily loss limit
        if risk_assessment.daily_loss_percentage <= -self.settings.MAX_DAILY_LOSS * 0.8:  # 80% of limit
            return False, f"Approaching daily loss limit: {risk_assessment.daily_loss_percentage:.2%}"
        
        # Check position concentration
        if risk_assessment.position_concentration > 0.08:  # > 8%
            return False, f"Position concentration too high: {risk_assessment.position_concentration:.1%}"
        
        # Check maximum positions
        if len(current_positions) >= self.settings.MAX_POSITIONS:
            return False, f"Maximum positions reached: {len(current_positions)}/{self.settings.MAX_POSITIONS}"
        
        return True, "Position approved"
    
    def get_risk_dashboard_data(self, positions: List[Dict]) -> Dict:
        """Get risk dashboard data"""
        risk_assessment = self.assess_portfolio_risk(positions)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'trading_paused': self.trading_paused,
            'pause_reason': self.pause_reason,
            'risk_assessment': {
                'overall_risk_level': risk_assessment.overall_risk_level,
                'portfolio_risk_score': risk_assessment.portfolio_risk_score,
                'daily_loss_percentage': risk_assessment.daily_loss_percentage,
                'position_concentration': risk_assessment.position_concentration,
                'recommendations': risk_assessment.risk_recommendations
            },
            'limits': {
                'max_daily_loss': self.settings.MAX_DAILY_LOSS,
                'max_positions': self.settings.MAX_POSITIONS,
                'max_position_size': self.MAX_POSITION_SIZE
            }
        }

async def create_dynamic_risk_manager(settings):
    """Factory function to create dynamic risk manager"""
    return DynamicRiskManager(settings)

# Example usage
if __name__ == "__main__":
    async def test_risk_manager():
        from src.config.settings import load_settings
        
        settings = load_settings()
        risk_manager = DynamicRiskManager(settings)
        
        # Test signal confidence assessment
        signal_data = {
            'momentum_strength': 0.08,
            'liquidity': 150.0,
            'volume_24h': 75.0,
            'price_change_percentage': 25.0,
            'technical_score': 0.7
        }
        
        confidence = risk_manager.assess_signal_confidence(signal_data)
        print(f"Signal confidence: {confidence.confidence_level} ({confidence.overall_confidence:.2f})")
        
        # Test position sizing
        position_size = risk_manager.calculate_dynamic_position_size(confidence, [])
        print(f"Recommended position size: {position_size.final_position_size:.1%}")
        
        # Test position approval
        allowed, reason = risk_manager.should_allow_new_position(signal_data, [])
        print(f"Position allowed: {allowed} - {reason}")
    
    if __name__ == "__main__":
        asyncio.run(test_risk_manager())