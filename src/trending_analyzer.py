"""
Trending Analyzer for Birdeye API integration
Provides scoring, validation, and signal enhancement for trending tokens
"""
import logging
import math
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from .birdeye_client import TrendingToken

logger = logging.getLogger(__name__)

class TrendingAnalyzer:
    """Analyzes trending token data and provides scoring/validation"""
    
    def __init__(self, settings):
        self.settings = settings
        
        # Scoring weights (must sum to 1.0)
        self.rank_weight = 0.4      # Higher weight for trending rank
        self.momentum_weight = 0.3   # Price momentum importance
        self.volume_weight = 0.2     # Volume growth importance  
        self.liquidity_weight = 0.1  # Liquidity factor
        
    def calculate_trending_score(self, token: TrendingToken) -> float:
        """
        Calculate comprehensive trending score (0-100)
        Higher scores indicate stronger trending potential
        """
        try:
            # 1. Rank Score (40% weight) - Lower rank = higher score
            rank_score = self._calculate_rank_score(token.rank)
            
            # 2. Momentum Score (30% weight) - Price change momentum
            momentum_score = self._calculate_momentum_score(token.price_24h_change_percent)
            
            # 3. Volume Score (20% weight) - Volume growth
            volume_score = self._calculate_volume_score(token.volume_24h_change_percent, token.volume_24h_usd)
            
            # 4. Liquidity Score (10% weight) - Liquidity adequacy
            liquidity_score = self._calculate_liquidity_score(token.liquidity)
            
            # Weighted composite score
            final_score = (
                rank_score * self.rank_weight +
                momentum_score * self.momentum_weight +
                volume_score * self.volume_weight +
                liquidity_score * self.liquidity_weight
            )
            
            # Apply bonus multipliers
            final_score = self._apply_bonus_multipliers(token, final_score)
            
            # Ensure score stays within 0-100 range
            final_score = max(0, min(100, final_score))
            
            logger.debug(f"[TRENDING] Score breakdown for {token.symbol}:")
            logger.debug(f"  Rank: {rank_score:.1f} | Momentum: {momentum_score:.1f} | Volume: {volume_score:.1f} | Liquidity: {liquidity_score:.1f}")
            logger.debug(f"  Final Score: {final_score:.1f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"[TRENDING] Error calculating score for {token.symbol}: {e}")
            return 0.0
    
    def _calculate_rank_score(self, rank: int) -> float:
        """Calculate score based on trending rank (1-100 scale)"""
        # Exponential decay for rank - top 10 get high scores
        if rank <= 10:
            return 100 - (rank - 1) * 5  # 100, 95, 90, 85, etc.
        elif rank <= 25:
            return 50 - (rank - 10) * 2  # 50 to 20
        elif rank <= 50:
            return 20 - (rank - 25) * 0.8  # 20 to 0
        else:
            return 0
    
    def _calculate_momentum_score(self, price_change_24h: float) -> float:
        """Calculate score based on 24h price momentum"""
        # Positive momentum gets exponential scoring
        if price_change_24h >= 100:  # 100%+ gains
            return 100
        elif price_change_24h >= 50:  # 50-100% gains
            return 80 + (price_change_24h - 50) * 0.4
        elif price_change_24h >= 20:  # 20-50% gains (target range)
            return 60 + (price_change_24h - 20) * 0.67
        elif price_change_24h >= 10:  # 10-20% gains
            return 40 + (price_change_24h - 10) * 2
        elif price_change_24h >= 0:   # 0-10% gains
            return price_change_24h * 4
        else:  # Negative momentum
            # Penalize negative momentum heavily
            return max(0, 20 + price_change_24h * 2)
    
    def _calculate_volume_score(self, volume_change_24h: float, volume_24h_usd: float) -> float:
        """Calculate score based on volume growth and absolute volume"""
        # Volume change score
        if volume_change_24h >= 200:  # 200%+ volume increase
            change_score = 100
        elif volume_change_24h >= 100:  # 100-200% increase
            change_score = 70 + (volume_change_24h - 100) * 0.3
        elif volume_change_24h >= 50:   # 50-100% increase
            change_score = 40 + (volume_change_24h - 50) * 0.6
        elif volume_change_24h >= 10:   # 10-50% increase (minimum threshold)
            change_score = 20 + (volume_change_24h - 10) * 0.5
        elif volume_change_24h >= 0:    # 0-10% increase
            change_score = volume_change_24h * 2
        else:  # Decreasing volume
            change_score = max(0, 10 + volume_change_24h * 0.5)
        
        # Absolute volume score (bonus for high volume)
        if volume_24h_usd >= 1000000:  # $1M+ daily volume
            volume_bonus = 20
        elif volume_24h_usd >= 500000:  # $500K+ daily volume
            volume_bonus = 15
        elif volume_24h_usd >= 100000:  # $100K+ daily volume
            volume_bonus = 10
        elif volume_24h_usd >= 50000:   # $50K+ daily volume
            volume_bonus = 5
        else:
            volume_bonus = 0
        
        return min(100, change_score + volume_bonus)
    
    def _calculate_liquidity_score(self, liquidity: float) -> float:
        """Calculate score based on liquidity adequacy"""
        # Convert liquidity to SOL (assuming $150 SOL price)
        liquidity_sol = liquidity / 150
        
        if liquidity_sol >= 2000:    # High liquidity
            return 100
        elif liquidity_sol >= 1000:  # Good liquidity
            return 80 + (liquidity_sol - 1000) * 0.02
        elif liquidity_sol >= 500:   # Adequate liquidity (minimum requirement)
            return 60 + (liquidity_sol - 500) * 0.04
        elif liquidity_sol >= 100:   # Low liquidity
            return 20 + (liquidity_sol - 100) * 0.1
        else:  # Very low liquidity
            return liquidity_sol * 0.2
    
    def _apply_bonus_multipliers(self, token: TrendingToken, base_score: float) -> float:
        """Apply bonus multipliers for exceptional conditions"""
        multiplier = 1.0
        
        # Meme token bonus (often have explosive momentum)
        if self._is_meme_token(token):
            multiplier *= 1.1
            logger.debug(f"[TRENDING] Meme token bonus applied to {token.symbol}")
        
        # New token bonus (within 24 hours, high momentum)
        if token.price_24h_change_percent >= 100 and token.rank <= 20:
            multiplier *= 1.15
            logger.debug(f"[TRENDING] High momentum new token bonus applied to {token.symbol}")
        
        # Top 5 trending bonus
        if token.rank <= 5:
            multiplier *= 1.1
            logger.debug(f"[TRENDING] Top 5 trending bonus applied to {token.symbol}")
        
        return base_score * multiplier
    
    def _is_meme_token(self, token: TrendingToken) -> bool:
        """Check if token appears to be a meme token"""
        meme_indicators = ['dog', 'cat', 'pepe', 'moon', 'baby', 'inu', 'shib', 'meme', 'doge', 'wojak', 'chad']
        name_lower = token.name.lower()
        symbol_lower = token.symbol.lower()
        
        return any(indicator in name_lower or indicator in symbol_lower for indicator in meme_indicators)
    
    def meets_trending_criteria(self, token: TrendingToken) -> Tuple[bool, str]:
        """
        Check if token meets all trending validation criteria
        Returns (passes, reason) tuple
        """
        try:
            # 1. Rank requirement
            if token.rank > self.settings.MAX_TRENDING_RANK:
                return False, f"Rank too low: #{token.rank} > #{self.settings.MAX_TRENDING_RANK}"
            
            # 2. Price momentum requirement
            if token.price_24h_change_percent < self.settings.MIN_PRICE_CHANGE_24H:
                return False, f"Insufficient momentum: {token.price_24h_change_percent:.1f}% < {self.settings.MIN_PRICE_CHANGE_24H}%"
            
            # 3. Volume growth requirement
            if token.volume_24h_change_percent < self.settings.MIN_VOLUME_CHANGE_24H:
                return False, f"Insufficient volume growth: {token.volume_24h_change_percent:.1f}% < {self.settings.MIN_VOLUME_CHANGE_24H}%"
            
            # 4. Minimum volume requirement (absolute)
            min_volume_usd = 10000  # $10K minimum daily volume
            if token.volume_24h_usd < min_volume_usd:
                return False, f"Volume too low: ${token.volume_24h_usd:,.0f} < ${min_volume_usd:,.0f}"
            
            # 5. Calculate trending score
            trending_score = self.calculate_trending_score(token)
            if trending_score < self.settings.MIN_TRENDING_SCORE:
                return False, f"Trending score too low: {trending_score:.1f} < {self.settings.MIN_TRENDING_SCORE}"
            
            return True, f"All criteria met (score: {trending_score:.1f})"
            
        except Exception as e:
            logger.error(f"[TRENDING] Error validating criteria for {token.symbol}: {e}")
            return False, f"Validation error: {e}"
    
    def enhance_signal_strength(self, base_signal: float, token: TrendingToken) -> float:
        """
        Enhance existing signal strength for trending tokens
        Returns boosted signal (capped at 100)
        """
        try:
            trending_score = self.calculate_trending_score(token)
            
            # Calculate boost factor based on trending score and settings
            boost_factor = (trending_score / 100) * self.settings.TRENDING_SIGNAL_BOOST
            
            # Apply boost
            enhanced_signal = base_signal * (1 + boost_factor)
            
            # Cap at 100
            enhanced_signal = min(100, enhanced_signal)
            
            logger.info(f"[TRENDING] Signal enhanced for {token.symbol}: {base_signal:.1f} → {enhanced_signal:.1f} (boost: +{(enhanced_signal - base_signal):.1f})")
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"[TRENDING] Error enhancing signal for {token.symbol}: {e}")
            return base_signal
    
    def get_trending_summary(self, tokens: list) -> Dict[str, Any]:
        """Generate summary statistics for trending tokens"""
        if not tokens:
            return {'error': 'No trending tokens available'}
        
        try:
            # Calculate statistics
            total_tokens = len(tokens)
            avg_rank = sum(t.rank for t in tokens) / total_tokens
            avg_momentum = sum(t.price_24h_change_percent for t in tokens) / total_tokens
            avg_volume_change = sum(t.volume_24h_change_percent for t in tokens) / total_tokens
            
            # Find top performers
            top_momentum = max(tokens, key=lambda t: t.price_24h_change_percent)
            top_volume = max(tokens, key=lambda t: t.volume_24h_change_percent)
            
            # Calculate passing criteria
            passing_tokens = [t for t in tokens if self.meets_trending_criteria(t)[0]]
            pass_rate = len(passing_tokens) / total_tokens * 100
            
            return {
                'total_tokens': total_tokens,
                'avg_rank': avg_rank,
                'avg_momentum_24h': avg_momentum,
                'avg_volume_change_24h': avg_volume_change,
                'top_momentum_token': {
                    'symbol': top_momentum.symbol,
                    'momentum': top_momentum.price_24h_change_percent
                },
                'top_volume_token': {
                    'symbol': top_volume.symbol,
                    'volume_change': top_volume.volume_24h_change_percent
                },
                'tokens_passing_criteria': len(passing_tokens),
                'criteria_pass_rate': pass_rate
            }
            
        except Exception as e:
            logger.error(f"[TRENDING] Error generating summary: {e}")
            return {'error': str(e)}