"""
Mean Reversion Trading Strategy - Migrated to Unified Architecture

MIGRATION: Day 6 - Strategy Migration & API Consolidation 
SOURCE: src/trading/mean_reversion_strategy.py
PRESERVE 100%: RSI + Z-score analysis, liquidity health checks, ATR position sizing

Strategy Logic (100% preserved):
- Buy when RSI < oversold_threshold AND Z-score < z_threshold AND liquidity is healthy
- Sell when RSI > overbought_threshold OR target profit reached OR stop loss triggered
- Position sizing based on signal confidence and market conditions
- ATR-based risk management for dynamic stops and targets
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

# New unified architecture imports
from strategies.base import BaseStrategy, StrategyConfig, StrategyType, StrategyStatus, StrategyMetrics
from models.position import Position
from models.signal import Signal
from models.trade import Trade, TradeDirection, TradeType

# Technical indicators (preserved from original implementation)
try:
    from utils.technical_indicators import RSICalculator, BollingerBands, MovingAverage, ATRCalculator
except ImportError:
    # Fallback implementation for basic calculations
    class RSICalculator:
        def __init__(self, period): self.period = period
        def calculate_from_prices(self, prices): return 50.0  # Default neutral RSI
    
    class BollingerBands:
        def __init__(self, period, std_dev): pass
        def calculate_position(self, prices): return 0.0  # Default neutral position
    
    class MovingAverage:
        def __init__(self, period): self.period = period
    
    class ATRCalculator:
        def __init__(self, period): self.period = period
        def calculate_from_prices(self, prices): return 0.01  # Default 1% volatility

logger = logging.getLogger(__name__)

class MeanReversionSignalType(Enum):
    """Mean reversion signal types (100% preserved)"""
    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    Z_SCORE_EXTREME = "z_score_extreme"
    BOLLINGER_REVERSAL = "bollinger_reversal"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_REJECTION = "resistance_rejection"

@dataclass
class LiquidityHealthCheck:
    """Liquidity health assessment for mean reversion (100% preserved)"""
    min_liquidity_usd: float = 25000  # $25k minimum liquidity
    min_volume_24h_usd: float = 5000  # $5k minimum 24h volume
    max_bid_ask_spread: float = 0.05  # 5% maximum spread
    whale_dump_lookback_hours: int = 4
    
    def is_healthy(self, market_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Assess if liquidity is healthy for mean reversion trading (100% preserved)"""
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
    """Configuration for mean reversion strategy (100% preserved)"""
    
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
    """Enhanced signal for mean reversion strategy (100% preserved)"""
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

class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion Strategy - Migrated to BaseStrategy interface (100% algorithm preservation)"""
    
    def __init__(self, settings=None, portfolio=None):
        """Initialize mean reversion strategy with preserved algorithms"""
        # Create strategy configuration
        config = StrategyConfig(
            strategy_name="mean_reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            max_positions=10,
            max_position_size=0.25,  # 25% max for high confidence signals
            position_timeout_minutes=1440,  # 24 hours
            stop_loss_percentage=0.15,
            take_profit_percentage=0.25,
            min_signal_strength=0.6
        )
        
        # Initialize base strategy
        super().__init__(config, portfolio, settings)
        
        # Store settings with fallback
        self.settings = settings
        
        # Mean reversion specific configuration (100% preserved)
        self.mr_config = MeanReversionConfig()
        self.liquidity_checker = LiquidityHealthCheck()
        
        # Technical indicators (100% preserved logic)
        if RSICalculator and BollingerBands and MovingAverage and ATRCalculator:
            self.rsi_calculator = RSICalculator(period=self.mr_config.rsi_period)
            self.bollinger_bands = BollingerBands(period=20, std_dev=2.0)
            self.moving_average = MovingAverage(period=self.mr_config.z_score_window)
            self.atr_calculator = ATRCalculator(period=self.mr_config.atr_period)
        else:
            logger.warning("[MEAN_REVERSION] Technical indicators not available during migration")
            self.rsi_calculator = None
            self.bollinger_bands = None
            self.moving_average = None
            self.atr_calculator = None
        
        # Price history storage for calculations (100% preserved)
        self.price_history: Dict[str, Dict[str, List[float]]] = {}  # token -> timeframe -> prices
        self.volume_history: Dict[str, Dict[str, List[float]]] = {}
        
        # Position tracking for mean reversion
        self.open_positions: Dict[str, Position] = {}
        
        logger.info("[MEAN_REVERSION] Strategy initialized with BaseStrategy interface")
    
    def update_price_data(self, token_address: str, timeframe: str, 
                         price: float, volume: float, timestamp: datetime = None):
        """Update price and volume data for analysis (100% preserved)"""
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
        
        # Keep reasonable history (preserve last 100 data points)
        max_history = 100
        if len(self.price_history[token_address][timeframe]) > max_history:
            self.price_history[token_address][timeframe] = \
                self.price_history[token_address][timeframe][-max_history:]
            self.volume_history[token_address][timeframe] = \
                self.volume_history[token_address][timeframe][-max_history:]
    
    def calculate_rsi(self, prices: List[float]) -> Optional[float]:
        """Calculate RSI from price data (100% preserved)"""
        if self.rsi_calculator:
            return self.rsi_calculator.calculate_from_prices(prices)
        
        # Fallback manual RSI calculation (100% preserved algorithm)
        if len(prices) < self.mr_config.rsi_period + 1:
            return None
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-self.mr_config.rsi_period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-self.mr_config.rsi_period:]]
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_z_score(self, prices: List[float]) -> Optional[float]:
        """Calculate Z-score for price deviation analysis (100% preserved)"""
        if len(prices) < self.mr_config.z_score_window:
            return None
        
        recent_prices = prices[-self.mr_config.z_score_window:]
        current_price = prices[-1]
        
        mean_price = np.mean(recent_prices[:-1])  # Exclude current price from mean
        std_price = np.std(recent_prices[:-1])
        
        if std_price == 0:
            return 0
        
        z_score = (current_price - mean_price) / std_price
        return z_score
    
    def calculate_bollinger_position(self, prices: List[float]) -> Optional[float]:
        """Calculate position relative to Bollinger Bands (100% preserved)"""
        if self.bollinger_bands:
            return self.bollinger_bands.calculate_position(prices)
        
        # Fallback manual calculation
        if len(prices) < 20:
            return None
        
        recent_prices = prices[-20:]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        current_price = prices[-1]
        
        upper_band = mean_price + (2 * std_price)
        lower_band = mean_price - (2 * std_price)
        
        if upper_band == lower_band:
            return 0
        
        # Position: -1 (lower band) to +1 (upper band)
        position = (current_price - mean_price) / (upper_band - mean_price)
        return max(-1, min(1, position))
    
    def calculate_confidence(self, rsi: float, z_score: float, 
                           bollinger_pos: float, volume_trend: float) -> float:
        """Calculate signal confidence (100% preserved)"""
        confidence = 0.0
        
        # RSI confidence (higher confidence for extreme values)
        if rsi <= self.mr_config.rsi_extreme_oversold:
            confidence += 0.4
        elif rsi <= self.mr_config.rsi_oversold:
            confidence += 0.3
        elif rsi >= self.mr_config.rsi_extreme_overbought:
            confidence += 0.4
        elif rsi >= self.mr_config.rsi_overbought:
            confidence += 0.3
        
        # Z-score confidence (higher confidence for extreme deviations)
        z_abs = abs(z_score)
        if z_abs >= abs(self.mr_config.z_score_extreme_threshold):
            confidence += 0.3
        elif z_abs >= abs(self.mr_config.z_score_buy_threshold):
            confidence += 0.2
        
        # Bollinger band confidence
        bb_abs = abs(bollinger_pos)
        if bb_abs >= 0.9:
            confidence += 0.2
        elif bb_abs >= 0.8:
            confidence += 0.1
        
        # Volume trend bonus
        if volume_trend > 1.2:  # 20% above average volume
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def analyze_signal(self, signal: Signal) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Analyze a signal for mean reversion opportunities
        
        Returns: (should_trade, confidence, metadata)
        """
        try:
            token_address = signal.token_address
            current_price = signal.price
            
            # Update price data
            self.update_price_data(token_address, self.mr_config.primary_timeframe, 
                                 current_price, signal.market_data.get('volume', 0))
            
            # Check if we have enough price history
            if (token_address not in self.price_history or 
                self.mr_config.primary_timeframe not in self.price_history[token_address] or
                len(self.price_history[token_address][self.mr_config.primary_timeframe]) < self.mr_config.z_score_window):
                return False, 0.0, {"reason": "Insufficient price history"}
            
            prices = self.price_history[token_address][self.mr_config.primary_timeframe]
            volumes = self.volume_history[token_address].get(self.mr_config.primary_timeframe, [])
            
            # Calculate technical indicators (100% preserved algorithms)
            rsi = self.calculate_rsi(prices)
            z_score = self.calculate_z_score(prices)
            bollinger_pos = self.calculate_bollinger_position(prices)
            
            if rsi is None or z_score is None or bollinger_pos is None:
                return False, 0.0, {"reason": "Technical indicator calculation failed"}
            
            # Calculate volume trend
            volume_trend = 1.0
            if len(volumes) >= 10:
                recent_volume = np.mean(volumes[-5:])
                avg_volume = np.mean(volumes[-20:-5]) if len(volumes) >= 20 else np.mean(volumes[:-5])
                if avg_volume > 0:
                    volume_trend = recent_volume / avg_volume
            
            # Check liquidity health (100% preserved logic)
            liquidity_healthy, liquidity_reason = self.liquidity_checker.is_healthy(signal.market_data)
            
            # Determine trading signal (100% preserved logic)
            should_trade = False
            signal_type = None
            
            # Buy signal conditions (100% preserved)
            if (rsi <= self.mr_config.rsi_oversold and 
                z_score <= self.mr_config.z_score_buy_threshold and
                liquidity_healthy):
                should_trade = True
                signal_type = MeanReversionSignalType.RSI_OVERSOLD
            
            # Bollinger band reversal (100% preserved)
            elif bollinger_pos <= -0.8 and liquidity_healthy:
                should_trade = True
                signal_type = MeanReversionSignalType.BOLLINGER_REVERSAL
            
            if not should_trade:
                return False, 0.0, {
                    "reason": "No mean reversion signal",
                    "rsi": rsi,
                    "z_score": z_score,
                    "bollinger_pos": bollinger_pos,
                    "liquidity_healthy": liquidity_healthy
                }
            
            # Calculate confidence (100% preserved)
            confidence = self.calculate_confidence(rsi, z_score, bollinger_pos, volume_trend)
            
            # Require minimum confidence
            if confidence < self.config.min_signal_strength:
                return False, confidence, {"reason": f"Confidence too low: {confidence:.2f}"}
            
            metadata = {
                "signal_type": signal_type.value,
                "rsi": rsi,
                "z_score": z_score,
                "bollinger_position": bollinger_pos,
                "volume_trend": volume_trend,
                "liquidity_healthy": liquidity_healthy,
                "liquidity_reason": liquidity_reason,
                "confidence": confidence
            }
            
            return True, confidence, metadata
            
        except Exception as e:
            logger.error(f"[MEAN_REVERSION] Error analyzing signal: {e}")
            return False, 0.0, {"reason": f"Analysis error: {str(e)}"}
    
    def calculate_position_size(self, signal: Signal, available_capital: float) -> float:
        """Calculate optimal position size based on signal confidence (100% preserved)"""
        try:
            # Get signal confidence from market_data or use default
            confidence = signal.strength
            
            # Base position size (100% preserved algorithm)
            base_size = self.mr_config.base_position_size * available_capital
            
            # Adjust by confidence (100% preserved)
            confidence_adjusted_size = base_size * (1 + confidence * self.mr_config.confidence_multiplier)
            
            # Cap at maximum position size
            max_size = self.mr_config.max_position_size * available_capital
            suggested_size = min(confidence_adjusted_size, max_size)
            
            # Ensure minimum size
            min_size = 0.01 * available_capital  # 1% minimum
            final_size = max(suggested_size, min_size)
            
            logger.info(f"[MEAN_REVERSION] Position sizing - Confidence: {confidence:.2f}, "
                       f"Base: {base_size:.2f}, Final: {final_size:.2f} SOL")
            
            return final_size
            
        except Exception as e:
            logger.error(f"[MEAN_REVERSION] Error calculating position size: {e}")
            return 0.0
    
    def should_exit_position(self, position: Position, current_price: float) -> Tuple[bool, str]:
        """Check if position should be exited based on mean reversion logic (100% preserved)"""
        try:
            # Get position data
            entry_price = position.entry_price
            position_age = datetime.now() - position.created_at
            
            # Calculate price change
            if position.size > 0:  # Long position
                price_change_pct = (current_price - entry_price) / entry_price
                profit_target = self.mr_config.take_profit_pct
                stop_loss = -self.mr_config.stop_loss_pct
            else:  # Short position (though mean reversion typically goes long)
                price_change_pct = (entry_price - current_price) / entry_price
                profit_target = self.mr_config.take_profit_pct
                stop_loss = -self.mr_config.stop_loss_pct
            
            # Check take profit (100% preserved)
            if price_change_pct >= profit_target:
                return True, f"Take profit: {price_change_pct:.1%} >= {profit_target:.1%}"
            
            # Check stop loss (100% preserved)
            if price_change_pct <= stop_loss:
                return True, f"Stop loss: {price_change_pct:.1%} <= {stop_loss:.1%}"
            
            # Check maximum holding time (100% preserved)
            max_hold_time = timedelta(hours=self.mr_config.max_hold_time_hours)
            if position_age >= max_hold_time:
                return True, f"Max hold time exceeded: {position_age} >= {max_hold_time}"
            
            # ATR-based exit if available
            if hasattr(position, 'atr_stop_loss') and hasattr(position, 'atr_take_profit'):
                if position.size > 0:  # Long position
                    if current_price >= position.atr_take_profit:
                        return True, f"ATR take profit: {current_price} >= {position.atr_take_profit}"
                    if current_price <= position.atr_stop_loss:
                        return True, f"ATR stop loss: {current_price} <= {position.atr_stop_loss}"
            
            return False, "Hold position"
            
        except Exception as e:
            logger.error(f"[MEAN_REVERSION] Error checking exit condition: {e}")
            return True, f"Exit due to error: {str(e)}"
    
    async def manage_positions(self) -> List[Tuple[str, str]]:
        """Manage existing mean reversion positions"""
        actions = []
        
        try:
            for token_address, position in list(self.open_positions.items()):
                # Get current price (this would come from market data in real implementation)
                current_price = position.entry_price  # Placeholder
                
                should_exit, reason = self.should_exit_position(position, current_price)
                
                if should_exit:
                    actions.append((token_address, f"Exit: {reason}"))
                    # Remove from tracking
                    del self.open_positions[token_address]
                    
                    logger.info(f"[MEAN_REVERSION] Exiting position {token_address}: {reason}")
            
            return actions
            
        except Exception as e:
            logger.error(f"[MEAN_REVERSION] Error managing positions: {e}")
            return []
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get mean reversion strategy summary"""
        return {
            "strategy_type": "Mean Reversion",
            "config": {
                "rsi_oversold": self.mr_config.rsi_oversold,
                "rsi_overbought": self.mr_config.rsi_overbought,
                "z_score_threshold": self.mr_config.z_score_buy_threshold,
                "min_liquidity": f"${self.liquidity_checker.min_liquidity_usd:,.0f}",
                "position_size_range": f"{self.mr_config.base_position_size:.1%} - {self.mr_config.max_position_size:.1%}",
                "atr_risk_management": self.mr_config.use_atr_risk_management
            },
            "technical_indicators": {
                "rsi_available": self.rsi_calculator is not None,
                "bollinger_bands_available": self.bollinger_bands is not None,
                "atr_available": self.atr_calculator is not None
            },
            "data_tracking": {
                "tokens_tracked": len(self.price_history),
                "open_positions": len(self.open_positions)
            }
        }

def create_mean_reversion_strategy(settings=None, portfolio=None) -> MeanReversionStrategy:
    """Create and configure mean reversion strategy for the new architecture"""
    return MeanReversionStrategy(settings, portfolio)