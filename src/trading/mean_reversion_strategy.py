"""
Mean Reversion Trading Strategy

Implements comprehensive mean reversion trading with:
- RSI-based oversold/overbought detection
- Z-score price deviation analysis  
- Liquidity health checks
- Dynamic position sizing
- Multi-timeframe analysis

Strategy Logic:
- Buy when RSI < oversold_threshold AND Z-score < z_threshold AND liquidity is healthy
- Sell when RSI > overbought_threshold OR target profit reached OR stop loss triggered
- Position sizing based on signal confidence and market conditions
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .signals import Signal, MarketCondition
from .technical_indicators import RSICalculator, BollingerBands, MovingAverage, ATRCalculator
from ..config.settings import Settings

logger = logging.getLogger(__name__)

class MeanReversionSignalType(Enum):
    """Mean reversion signal types"""
    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    Z_SCORE_EXTREME = "z_score_extreme"
    BOLLINGER_REVERSAL = "bollinger_reversal"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_REJECTION = "resistance_rejection"

@dataclass
class LiquidityHealthCheck:
    """Liquidity health assessment for mean reversion"""
    min_liquidity_usd: float = 25000  # $25k minimum liquidity
    min_volume_24h_usd: float = 5000  # $5k minimum 24h volume
    max_bid_ask_spread: float = 0.05  # 5% maximum spread
    whale_dump_lookback_hours: int = 4
    
    def is_healthy(self, market_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Assess if liquidity is healthy for mean reversion trading"""
        checks = []
        reasons = []
        
        # Liquidity check
        liquidity = market_data.get('liquidity_usd', 0)
        if liquidity >= self.min_liquidity_usd:
            checks.append(True)
        else:
            checks.append(False)
            reasons.append(f"Low liquidity: ${liquidity:,.0f} < ${self.min_liquidity_usd:,.0f}")
        
        # Volume check
        volume_24h = market_data.get('volume_24h_usd', 0)
        if volume_24h >= self.min_volume_24h_usd:
            checks.append(True)
        else:
            checks.append(False)
            reasons.append(f"Low volume: ${volume_24h:,.0f} < ${self.min_volume_24h_usd:,.0f}")
        
        # Spread check
        bid_ask_spread = market_data.get('bid_ask_spread', 1.0)
        if bid_ask_spread <= self.max_bid_ask_spread:
            checks.append(True)
        else:
            checks.append(False)
            reasons.append(f"Wide spread: {bid_ask_spread:.1%} > {self.max_bid_ask_spread:.1%}")
        
        # Whale activity check (simplified)
        whale_activity = market_data.get('recent_whale_activity', False)
        if not whale_activity:
            checks.append(True)
        else:
            checks.append(False)
            reasons.append("Recent whale dump detected")
        
        # Require at least 3/4 conditions to pass
        healthy = sum(checks) >= 3
        reason = "Healthy" if healthy else "; ".join(reasons)
        
        return healthy, reason

@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion strategy"""
    
    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: float = 20
    rsi_overbought: float = 80
    rsi_extreme_oversold: float = 15  # Extra oversold for higher confidence
    rsi_extreme_overbought: float = 85
    
    # Z-score parameters
    z_score_window: int = 20
    z_score_buy_threshold: float = -2.0  # Buy when price is 2 std devs below mean
    z_score_sell_threshold: float = 2.0
    z_score_extreme_threshold: float = -2.5  # Higher confidence threshold
    
    # Position sizing parameters
    base_position_size: float = 0.1  # 10% of balance for base position
    max_position_size: float = 0.25  # 25% maximum for high confidence
    confidence_multiplier: float = 1.5  # Multiply position size by confidence
    
    # Risk management
    stop_loss_pct: float = 0.15  # 15% stop loss
    take_profit_pct: float = 0.25  # 25% take profit
    max_hold_time_hours: int = 24  # Maximum hold time
    
    # ATR-based risk management
    atr_period: int = 14  # ATR calculation period
    atr_stop_multiplier: float = 2.0  # Stop loss = current_price +/- (ATR * multiplier)
    atr_take_profit_multiplier: float = 3.0  # Take profit = current_price +/- (ATR * multiplier)
    use_atr_risk_management: bool = True  # Enable ATR-based stops/targets
    
    # Timeframes for analysis
    analysis_timeframes: List[str] = field(default_factory=lambda: ['5m', '15m', '1h'])
    primary_timeframe: str = '15m'

@dataclass  
class MeanReversionSignal:
    """Enhanced signal for mean reversion strategy"""
    token_address: str
    signal_type: MeanReversionSignalType
    price: float
    confidence: float
    
    # Technical indicators
    rsi_value: float
    z_score: float
    bollinger_position: float  # -1 to 1, where -1 is lower band, 1 is upper band
    
    # Market conditions
    liquidity_healthy: bool
    liquidity_reason: str
    volume_profile: Dict[str, float]
    
    # Position parameters
    suggested_size: float
    stop_loss: float
    take_profit: float
    max_hold_time: datetime
    
    # Optional fields with defaults must come after required fields
    atr_value: Optional[float] = None  # ATR for volatility-based risk management
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_signal(self) -> Signal:
        """Convert to standard Signal object"""
        return Signal(
            token_address=self.token_address,
            price=self.price,
            strength=self.confidence,
            market_data={
                'rsi': self.rsi_value,
                'z_score': self.z_score,
                'bollinger_position': self.bollinger_position,
                'atr': self.atr_value,
                'liquidity_healthy': self.liquidity_healthy,
                'volume_profile': self.volume_profile
            },
            signal_type=self.signal_type.value,
            timestamp=self.timestamp
        )

class MeanReversionStrategy:
    """Comprehensive mean reversion trading strategy"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.config = MeanReversionConfig()
        self.liquidity_checker = LiquidityHealthCheck()
        
        # Technical indicators
        self.rsi_calculator = RSICalculator(period=self.config.rsi_period)
        self.bollinger_bands = BollingerBands(period=20, std_dev=2.0)
        self.moving_average = MovingAverage(period=self.config.z_score_window)
        self.atr_calculator = ATRCalculator(period=self.config.atr_period)
        
        # Price history storage for calculations
        self.price_history: Dict[str, Dict[str, List[float]]] = {}  # token -> timeframe -> prices
        self.volume_history: Dict[str, Dict[str, List[float]]] = {}
        
        logger.info("[MEAN_REVERSION] Strategy initialized")
    
    def update_price_data(self, token_address: str, timeframe: str, 
                         price: float, volume: float, timestamp: datetime = None):
        """Update price and volume data for analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Initialize storage
        if token_address not in self.price_history:
            self.price_history[token_address] = {}
            self.volume_history[token_address] = {}
        
        if timeframe not in self.price_history[token_address]:
            self.price_history[token_address][timeframe] = []
            self.volume_history[token_address][timeframe] = []
        
        # Add new data
        self.price_history[token_address][timeframe].append(price)
        self.volume_history[token_address][timeframe].append(volume)
        
        # Maintain reasonable history size (keep last 100 points)
        max_history = 100
        if len(self.price_history[token_address][timeframe]) > max_history:
            self.price_history[token_address][timeframe] = \
                self.price_history[token_address][timeframe][-max_history:]
            self.volume_history[token_address][timeframe] = \
                self.volume_history[token_address][timeframe][-max_history:]
    
    def calculate_rsi(self, prices: List[float]) -> Optional[float]:
        """Calculate RSI for given prices"""
        if len(prices) < self.config.rsi_period + 1:
            return None
            
        try:
            # Calculate price changes
            price_changes = np.diff(prices)
            
            # Separate gains and losses
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)
            
            # Calculate average gains and losses
            avg_gain = np.mean(gains[-self.config.rsi_period:])
            avg_loss = np.mean(losses[-self.config.rsi_period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
        except Exception as e:
            logger.error(f"[MEAN_REVERSION] RSI calculation error: {e}")
            return None
    
    def calculate_z_score(self, prices: List[float]) -> Optional[float]:
        """Calculate Z-score for current price vs historical mean"""
        if len(prices) < self.config.z_score_window:
            return None
            
        try:
            current_price = prices[-1]
            window_prices = prices[-self.config.z_score_window:]
            
            mean_price = np.mean(window_prices[:-1])  # Exclude current price from mean
            std_price = np.std(window_prices[:-1])
            
            if std_price == 0:
                return 0.0
            
            z_score = (current_price - mean_price) / std_price
            return float(z_score)
        except Exception as e:
            logger.error(f"[MEAN_REVERSION] Z-score calculation error: {e}")
            return None
    
    def calculate_bollinger_position(self, prices: List[float]) -> Optional[float]:
        """Calculate position relative to Bollinger Bands (-1 to 1)"""
        if len(prices) < 20:
            return None
            
        try:
            current_price = prices[-1]
            window_prices = prices[-20:]
            
            mean_price = np.mean(window_prices)
            std_price = np.std(window_prices)
            
            upper_band = mean_price + (2 * std_price)
            lower_band = mean_price - (2 * std_price)
            
            if upper_band == lower_band:
                return 0.0
            
            # Normalize position: -1 = lower band, 0 = mean, 1 = upper band
            position = (current_price - mean_price) / (std_price * 2)
            return max(-1.0, min(1.0, float(position)))
        except Exception as e:
            logger.error(f"[MEAN_REVERSION] Bollinger position calculation error: {e}")
            return None
    
    def calculate_confidence(self, rsi: float, z_score: float, 
                           bollinger_pos: float, volume_trend: float) -> float:
        """Calculate signal confidence based on multiple factors"""
        confidence = 0.0
        
        # RSI contribution (0-40 points)
        if rsi <= self.config.rsi_extreme_oversold:
            confidence += 40
        elif rsi <= self.config.rsi_oversold:
            confidence += 30
        elif rsi >= self.config.rsi_extreme_overbought:
            confidence += 40  # For sell signals
        elif rsi >= self.config.rsi_overbought:
            confidence += 30
        
        # Z-score contribution (0-35 points)
        if abs(z_score) >= abs(self.config.z_score_extreme_threshold):
            confidence += 35
        elif abs(z_score) >= abs(self.config.z_score_buy_threshold):
            confidence += 25
        
        # Bollinger bands contribution (0-15 points)
        if abs(bollinger_pos) >= 0.8:  # Near bands
            confidence += 15
        elif abs(bollinger_pos) >= 0.6:
            confidence += 10
        
        # Volume confirmation (0-10 points)
        if volume_trend > 1.2:  # 20% above average
            confidence += 10
        elif volume_trend > 1.1:
            confidence += 5
        
        return min(100.0, confidence) / 100.0
    
    def analyze_token(self, token_address: str, market_data: Dict[str, Any], 
                     timeframe: str = None) -> Optional[MeanReversionSignal]:
        """Analyze token for mean reversion opportunities"""
        if timeframe is None:
            timeframe = self.config.primary_timeframe
        
        # Check if we have enough price history
        if (token_address not in self.price_history or 
            timeframe not in self.price_history[token_address] or
            len(self.price_history[token_address][timeframe]) < self.config.z_score_window):
            return None
        
        prices = self.price_history[token_address][timeframe]
        volumes = self.volume_history[token_address].get(timeframe, [])
        current_price = prices[-1]
        
        # Calculate technical indicators
        rsi = self.calculate_rsi(prices)
        z_score = self.calculate_z_score(prices)
        bollinger_pos = self.calculate_bollinger_position(prices)
        
        # Calculate ATR for volatility-based risk management
        atr = self.atr_calculator.calculate_from_prices(prices)
        
        if rsi is None or z_score is None or bollinger_pos is None:
            return None
        
        # Calculate volume trend
        volume_trend = 1.0
        if len(volumes) >= 10:
            recent_volume = np.mean(volumes[-5:])
            avg_volume = np.mean(volumes[-20:-5]) if len(volumes) >= 20 else np.mean(volumes[:-5])
            if avg_volume > 0:
                volume_trend = recent_volume / avg_volume
        
        # Check liquidity health
        liquidity_healthy, liquidity_reason = self.liquidity_checker.is_healthy(market_data)
        
        # Determine signal type and direction
        signal_type = None
        is_buy_signal = False
        
        # Buy signal conditions
        if (rsi <= self.config.rsi_oversold and 
            z_score <= self.config.z_score_buy_threshold and
            liquidity_healthy):
            
            if rsi <= self.config.rsi_extreme_oversold:
                signal_type = MeanReversionSignalType.RSI_OVERSOLD
            else:
                signal_type = MeanReversionSignalType.Z_SCORE_EXTREME
            is_buy_signal = True
        
        # Sell signal conditions (for existing positions)
        elif (rsi >= self.config.rsi_overbought or 
              z_score >= self.config.z_score_sell_threshold):
            
            if rsi >= self.config.rsi_extreme_overbought:
                signal_type = MeanReversionSignalType.RSI_OVERBOUGHT
            else:
                signal_type = MeanReversionSignalType.Z_SCORE_EXTREME
            is_buy_signal = False
        
        # Bollinger band signals
        elif bollinger_pos <= -0.8 and liquidity_healthy:
            signal_type = MeanReversionSignalType.BOLLINGER_REVERSAL
            is_buy_signal = True
        elif bollinger_pos >= 0.8:
            signal_type = MeanReversionSignalType.BOLLINGER_REVERSAL
            is_buy_signal = False
        
        if signal_type is None:
            return None
        
        # Calculate confidence
        confidence = self.calculate_confidence(rsi, z_score, bollinger_pos, volume_trend)
        
        # Calculate position sizing
        base_size = self.config.base_position_size
        confidence_adjusted_size = base_size * (1 + confidence * self.config.confidence_multiplier)
        suggested_size = min(confidence_adjusted_size, self.config.max_position_size)
        
        # Calculate stop loss and take profit with ATR-based risk management
        if self.config.use_atr_risk_management and atr is not None and atr > 0:
            # Use ATR-based stop loss and take profit
            atr_stop_distance = atr * self.config.atr_stop_multiplier
            atr_profit_distance = atr * self.config.atr_take_profit_multiplier
            
            if is_buy_signal:
                stop_loss = current_price - atr_stop_distance
                take_profit = current_price + atr_profit_distance
            else:
                stop_loss = current_price + atr_stop_distance
                take_profit = current_price - atr_profit_distance
            
            logger.info(f"[MEAN_REVERSION] ATR Risk Management - ATR: {atr:.6f}, "
                       f"Stop: {atr_stop_distance:.6f}, Profit: {atr_profit_distance:.6f}")
        else:
            # Fallback to percentage-based stops
            if is_buy_signal:
                stop_loss = current_price * (1 - self.config.stop_loss_pct)
                take_profit = current_price * (1 + self.config.take_profit_pct)
            else:
                stop_loss = current_price * (1 + self.config.stop_loss_pct)
                take_profit = current_price * (1 - self.config.take_profit_pct)
        
        # Calculate max hold time
        max_hold_time = datetime.now() + timedelta(hours=self.config.max_hold_time_hours)
        
        return MeanReversionSignal(
            token_address=token_address,
            signal_type=signal_type,
            price=current_price,
            confidence=confidence,
            rsi_value=rsi,
            z_score=z_score,
            bollinger_position=bollinger_pos,
            atr_value=atr,
            liquidity_healthy=liquidity_healthy,
            liquidity_reason=liquidity_reason,
            volume_profile={'trend': volume_trend, 'current': volumes[-1] if volumes else 0},
            suggested_size=suggested_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold_time=max_hold_time
        )
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get current strategy statistics"""
        total_tokens = len(self.price_history)
        tokens_with_signals = 0
        
        stats = {
            'tokens_tracked': total_tokens,
            'tokens_with_sufficient_data': 0,
            'config': {
                'rsi_oversold': self.config.rsi_oversold,
                'rsi_overbought': self.config.rsi_overbought,
                'z_score_threshold': self.config.z_score_buy_threshold,
                'min_liquidity': self.liquidity_checker.min_liquidity_usd,
                'position_size_range': f"{self.config.base_position_size:.1%} - {self.config.max_position_size:.1%}",
                'atr_risk_management': self.config.use_atr_risk_management,
                'atr_stop_multiplier': self.config.atr_stop_multiplier,
                'atr_profit_multiplier': self.config.atr_take_profit_multiplier
            }
        }
        
        # Count tokens with sufficient data
        for token_address in self.price_history:
            for timeframe in self.price_history[token_address]:
                if len(self.price_history[token_address][timeframe]) >= self.config.z_score_window:
                    stats['tokens_with_sufficient_data'] += 1
                    break
        
        return stats

# Factory function for easy integration
def create_mean_reversion_strategy(settings: Settings) -> MeanReversionStrategy:
    """Create and configure mean reversion strategy"""
    return MeanReversionStrategy(settings)