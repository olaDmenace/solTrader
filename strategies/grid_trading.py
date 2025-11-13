"""
Grid Trading Strategy - Migrated to Unified Architecture

MIGRATION: Day 6 - Strategy Migration & API Consolidation 
SOURCE: src/trading/grid_trading_strategy.py
PRESERVE 100%: Range detection, dynamic grid levels, sideways market optimization

Strategy Logic (100% preserved):
- Detects ranging/sideways markets automatically
- Creates dynamic grid levels based on volatility
- Captures systematic profits in both directions
- Manages risk with position limits
- Optimized for sideways markets where momentum strategies struggle
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

# New unified architecture imports
from strategies.base import BaseStrategy, StrategyConfig, StrategyType, StrategyStatus, StrategyMetrics
from models.position import Position
from models.signal import Signal
from models.trade import Trade, TradeDirection, TradeType

logger = logging.getLogger(__name__)

@dataclass
class GridLevel:
    """Individual grid level definition (100% preserved)"""
    price: float
    level_id: str
    grid_type: str  # 'buy' or 'sell'
    position_size: float
    status: str  # 'pending', 'filled', 'cancelled'
    created_at: datetime
    filled_at: Optional[datetime] = None

@dataclass
class GridConfiguration:
    """Grid trading configuration (100% preserved)"""
    center_price: float
    grid_spacing: float
    grid_count: int
    position_size_per_level: float
    upper_boundary: float
    lower_boundary: float
    total_capital_allocated: float
    
@dataclass
class MarketRange:
    """Detected market range information (100% preserved)"""
    support_level: float
    resistance_level: float
    range_width: float
    range_confidence: float  # 0.0 to 1.0
    range_duration_hours: float
    volatility: float
    is_valid_for_grid: bool

class GridTradingStrategy(BaseStrategy):
    """Grid Trading Strategy - Migrated to BaseStrategy interface (100% algorithm preservation)"""
    
    def __init__(self, settings=None, portfolio=None):
        """Initialize grid trading strategy with preserved algorithms"""
        # Create strategy configuration
        config = StrategyConfig(
            strategy_name="grid_trading",
            strategy_type=StrategyType.GRID_TRADING,
            max_positions=20,  # Grid trading can have many positions
            max_position_size=0.05,  # 5% per grid level
            position_timeout_minutes=2160,  # 36 hours for grid positions
            stop_loss_percentage=0.10,  # 10% stop loss for entire grid
            take_profit_percentage=0.02,  # 2% per grid level
            min_signal_strength=0.7  # Higher confidence for range detection
        )
        
        # Initialize base strategy
        super().__init__(config, portfolio, settings)
        
        # Store settings with fallback
        self.settings = settings
        
        # Grid trading specific data (100% preserved)
        self.active_grids = {}  # token_address -> GridConfiguration
        self.grid_positions = {}  # token_address -> List[GridLevel]
        self.market_ranges = {}  # token_address -> MarketRange
        
        # Grid trading parameters (100% preserved)
        self.MIN_RANGE_WIDTH = 0.05  # 5% minimum range width
        self.MAX_RANGE_WIDTH = 0.30  # 30% maximum range width  
        self.MIN_RANGE_CONFIDENCE = 0.7  # 70% confidence required
        self.GRID_PROFIT_TARGET = 0.02  # 2% profit target per level
        self.MAX_GRID_LEVELS = 10  # Maximum grid levels
        self.MIN_GRID_LEVELS = 3   # Minimum grid levels
        self.MIN_GRID_SPACING = 0.01  # 1% minimum spacing
        
        # Price history storage for range detection
        self.price_history: Dict[str, List[float]] = {}
        self.timestamp_history: Dict[str, List[datetime]] = {}
        
        logger.info("[GRID_TRADING] Strategy initialized with BaseStrategy interface")
    
    def update_price_data(self, token_address: str, price: float, timestamp: datetime = None):
        """Update price data for range detection"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if token_address not in self.price_history:
            self.price_history[token_address] = []
            self.timestamp_history[token_address] = []
        
        self.price_history[token_address].append(price)
        self.timestamp_history[token_address].append(timestamp)
        
        # Keep reasonable history (last 200 data points)
        max_history = 200
        if len(self.price_history[token_address]) > max_history:
            self.price_history[token_address] = self.price_history[token_address][-max_history:]
            self.timestamp_history[token_address] = self.timestamp_history[token_address][-max_history:]
    
    def detect_market_range(self, price_history: List[float], timestamps: List[datetime]) -> Optional[MarketRange]:
        """Detect if market is in a ranging/sideways pattern (100% preserved algorithm)"""
        
        if len(price_history) < 50:  # Need sufficient data
            return None
        
        prices = np.array(price_history)
        
        # Calculate support and resistance levels (100% preserved)
        support_level = np.percentile(prices, 10)  # 10th percentile
        resistance_level = np.percentile(prices, 90)  # 90th percentile
        
        range_width = (resistance_level - support_level) / support_level
        
        # Check if range is valid for grid trading (100% preserved)
        if range_width < self.MIN_RANGE_WIDTH or range_width > self.MAX_RANGE_WIDTH:
            return None
        
        # Calculate range confidence based on how well prices respect levels (100% preserved)
        price_touches_support = np.sum((prices >= support_level) & (prices <= support_level * 1.02))
        price_touches_resistance = np.sum((prices <= resistance_level) & (prices >= resistance_level * 0.98))
        total_touches = price_touches_support + price_touches_resistance
        
        range_confidence = min(total_touches / len(prices) * 5, 1.0)  # Normalize to 1.0
        
        # Calculate volatility (100% preserved)
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(24)  # Annualized
        
        # Calculate range duration (100% preserved)
        range_duration = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # Hours
        
        # Determine if valid for grid trading (100% preserved)
        is_valid = (
            range_confidence >= self.MIN_RANGE_CONFIDENCE and
            range_width >= self.MIN_RANGE_WIDTH and
            range_width <= self.MAX_RANGE_WIDTH and
            range_duration >= 2.0  # At least 2 hours of ranging
        )
        
        return MarketRange(
            support_level=support_level,
            resistance_level=resistance_level,
            range_width=range_width,
            range_confidence=range_confidence,
            range_duration_hours=range_duration,
            volatility=volatility,
            is_valid_for_grid=is_valid
        )
    
    def calculate_optimal_grid_spacing(self, market_range: MarketRange, current_price: float) -> Tuple[float, int]:
        """Calculate optimal grid spacing based on market conditions (100% preserved)"""
        
        # Base spacing on volatility and range width (100% preserved algorithm)
        volatility_factor = min(market_range.volatility * 2, 0.1)  # Cap at 10%
        range_factor = market_range.range_width / 10  # Scale range width
        
        # Combine factors (100% preserved)
        optimal_spacing = max(
            self.MIN_GRID_SPACING,  # Minimum spacing
            volatility_factor + range_factor  # Dynamic calculation
        )
        
        # Calculate number of grid levels that fit in the range (100% preserved)
        range_height = market_range.resistance_level - market_range.support_level
        max_levels_in_range = int(range_height / (current_price * optimal_spacing))
        
        # Constrain to limits (100% preserved)
        grid_levels = max(
            self.MIN_GRID_LEVELS,
            min(max_levels_in_range, self.MAX_GRID_LEVELS)
        )
        
        return optimal_spacing, grid_levels
    
    def create_grid_configuration(self, token_address: str, current_price: float, 
                                market_range: MarketRange, available_capital: float) -> GridConfiguration:
        """Create grid configuration for a token (100% preserved)"""
        
        # Calculate optimal spacing and levels (100% preserved)
        grid_spacing, grid_count = self.calculate_optimal_grid_spacing(market_range, current_price)
        
        # Calculate position size per level (100% preserved logic, adapted for available capital)
        total_capital_for_grid = available_capital * 0.1  # 10% max for grid trading
        position_size_per_level = total_capital_for_grid / (grid_count * 2)  # Buy and sell levels
        
        # Set boundaries (100% preserved)
        upper_boundary = min(market_range.resistance_level, current_price * 1.15)  # Cap at 15% above
        lower_boundary = max(market_range.support_level, current_price * 0.85)   # Cap at 15% below
        
        return GridConfiguration(
            center_price=current_price,
            grid_spacing=grid_spacing,
            grid_count=grid_count,
            position_size_per_level=position_size_per_level,
            upper_boundary=upper_boundary,
            lower_boundary=lower_boundary,
            total_capital_allocated=total_capital_for_grid
        )
    
    def generate_grid_levels(self, config: GridConfiguration) -> List[GridLevel]:
        """Generate buy and sell grid levels (100% preserved algorithm)"""
        
        grid_levels = []
        current_time = datetime.now()
        
        # Generate buy levels (below center price) - 100% preserved
        for i in range(1, config.grid_count + 1):
            buy_price = config.center_price * (1 - config.grid_spacing * i)
            
            if buy_price >= config.lower_boundary:
                grid_levels.append(GridLevel(
                    price=buy_price,
                    level_id=f"buy_{i}",
                    grid_type='buy',
                    position_size=config.position_size_per_level,
                    status='pending',
                    created_at=current_time
                ))
        
        # Generate sell levels (above center price) - 100% preserved
        for i in range(1, config.grid_count + 1):
            sell_price = config.center_price * (1 + config.grid_spacing * i)
            
            if sell_price <= config.upper_boundary:
                grid_levels.append(GridLevel(
                    price=sell_price,
                    level_id=f"sell_{i}",
                    grid_type='sell',
                    position_size=config.position_size_per_level,
                    status='pending',
                    created_at=current_time
                ))
        
        return grid_levels
    
    async def setup_grid_trading(self, token_address: str, current_price: float, available_capital: float) -> bool:
        """Setup grid trading for a token (100% preserved logic)"""
        
        # Check if we have sufficient price history
        if (token_address not in self.price_history or 
            len(self.price_history[token_address]) < 50):
            return False
        
        price_history = self.price_history[token_address]
        timestamps = self.timestamp_history[token_address]
        
        # Detect market range (100% preserved)
        market_range = self.detect_market_range(price_history, timestamps)
        
        if not market_range or not market_range.is_valid_for_grid:
            logger.debug(f"[GRID_TRADING] Market range not suitable for grid trading: {token_address}")
            return False
        
        # Create grid configuration (100% preserved)
        config = self.create_grid_configuration(token_address, current_price, market_range, available_capital)
        
        # Generate grid levels (100% preserved)
        grid_levels = self.generate_grid_levels(config)
        
        if len(grid_levels) < self.MIN_GRID_LEVELS:
            logger.warning(f"[GRID_TRADING] Insufficient grid levels generated: {len(grid_levels)}")
            return False
        
        # Store grid setup (100% preserved)
        self.active_grids[token_address] = config
        self.grid_positions[token_address] = grid_levels
        self.market_ranges[token_address] = market_range
        
        logger.info(f"[GRID_TRADING] Grid setup for {token_address}: {len(grid_levels)} levels, "
                   f"{config.grid_spacing:.1%} spacing, range {market_range.range_width:.1%}")
        
        return True
    
    def check_grid_triggers(self, token_address: str, current_price: float) -> List[GridLevel]:
        """Check which grid levels should be triggered (100% preserved)"""
        
        if token_address not in self.grid_positions:
            return []
        
        triggered_levels = []
        grid_levels = self.grid_positions[token_address]
        
        for level in grid_levels:
            if level.status != 'pending':
                continue
            
            # Check buy level trigger (price at or below buy level) - 100% preserved
            if level.grid_type == 'buy' and current_price <= level.price:
                triggered_levels.append(level)
            
            # Check sell level trigger (price at or above sell level) - 100% preserved
            elif level.grid_type == 'sell' and current_price >= level.price:
                triggered_levels.append(level)
        
        return triggered_levels
    
    async def analyze_signal(self, signal: Signal) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Analyze signal for grid trading opportunities
        
        Returns: (should_trade, confidence, metadata)
        """
        try:
            token_address = signal.token_address
            current_price = signal.price
            
            # Update price data
            self.update_price_data(token_address, current_price)
            
            # Check if we already have an active grid
            if token_address in self.active_grids:
                # Check for grid level triggers
                triggered_levels = self.check_grid_triggers(token_address, current_price)
                
                if triggered_levels:
                    confidence = 0.8  # High confidence for grid triggers
                    metadata = {
                        "reason": "Grid level triggered",
                        "triggered_levels": len(triggered_levels),
                        "grid_active": True
                    }
                    return True, confidence, metadata
                
                # Check if price is still within grid range
                market_range = self.market_ranges.get(token_address)
                if market_range:
                    if (current_price < market_range.support_level * 0.95 or 
                        current_price > market_range.resistance_level * 1.05):
                        # Price broke out of range - consider closing grid
                        return False, 0.0, {"reason": "Price broke out of grid range"}
                
                return False, 0.0, {"reason": "No grid triggers"}
            
            # Check if we can setup new grid trading
            if len(self.price_history.get(token_address, [])) < 50:
                return False, 0.0, {"reason": "Insufficient price history for range detection"}
            
            # Detect market range
            price_history = self.price_history[token_address]
            timestamps = self.timestamp_history[token_address]
            market_range = self.detect_market_range(price_history, timestamps)
            
            if not market_range or not market_range.is_valid_for_grid:
                return False, 0.0, {
                    "reason": "Market not suitable for grid trading",
                    "range_detected": market_range is not None,
                    "range_valid": market_range.is_valid_for_grid if market_range else False
                }
            
            # Signal for new grid setup
            confidence = market_range.range_confidence
            metadata = {
                "reason": "New grid trading opportunity",
                "range_width": market_range.range_width,
                "range_confidence": market_range.range_confidence,
                "range_duration": market_range.range_duration_hours,
                "support_level": market_range.support_level,
                "resistance_level": market_range.resistance_level
            }
            
            return True, confidence, metadata
            
        except Exception as e:
            logger.error(f"[GRID_TRADING] Error analyzing signal: {e}")
            return False, 0.0, {"reason": f"Analysis error: {str(e)}"}
    
    def calculate_position_size(self, signal: Signal, available_capital: float) -> float:
        """Calculate position size for grid trading (adapted for BaseStrategy interface)"""
        try:
            token_address = signal.token_address
            
            # If setting up new grid
            if token_address not in self.active_grids:
                # Total capital allocation for grid (10% of available)
                return available_capital * 0.1
            
            # For grid level execution, use the configured per-level size
            config = self.active_grids[token_address]
            return config.position_size_per_level
            
        except Exception as e:
            logger.error(f"[GRID_TRADING] Error calculating position size: {e}")
            return 0.0
    
    def should_exit_position(self, position: Position, current_price: float) -> Tuple[bool, str]:
        """Check if position should be exited (adapted for grid trading)"""
        try:
            token_address = position.token_address
            
            # For grid trading, positions are managed differently
            # Check if entire grid should be closed
            if token_address in self.active_grids:
                should_close, reason = self.should_close_grid(token_address, current_price)
                if should_close:
                    return True, f"Close entire grid: {reason}"
            
            # Individual position management
            position_age = datetime.now() - position.created_at
            
            # Check for individual grid level profit target
            if position.size > 0:  # Long position
                profit_pct = (current_price - position.entry_price) / position.entry_price
                if profit_pct >= self.GRID_PROFIT_TARGET:
                    return True, f"Grid level profit target: {profit_pct:.1%}"
            
            # Check maximum hold time (longer for grid trading)
            max_hold_time = timedelta(hours=36)  # 36 hours for grid positions
            if position_age >= max_hold_time:
                return True, f"Max hold time exceeded: {position_age}"
            
            return False, "Hold grid position"
            
        except Exception as e:
            logger.error(f"[GRID_TRADING] Error checking exit condition: {e}")
            return True, f"Exit due to error: {str(e)}"
    
    def should_close_grid(self, token_address: str, current_price: float) -> Tuple[bool, str]:
        """Check if entire grid should be closed (100% preserved logic)"""
        try:
            if token_address not in self.market_ranges:
                return True, "No market range data"
            
            market_range = self.market_ranges[token_address]
            
            # Close if price breaks significantly out of range (100% preserved)
            if current_price < market_range.support_level * 0.90:
                return True, f"Price broke below support: {current_price} < {market_range.support_level * 0.90}"
            
            if current_price > market_range.resistance_level * 1.10:
                return True, f"Price broke above resistance: {current_price} > {market_range.resistance_level * 1.10}"
            
            # Check grid age
            if token_address in self.active_grids:
                config = self.active_grids[token_address]
                grid_age = datetime.now() - self.grid_positions[token_address][0].created_at
                
                if grid_age > timedelta(hours=48):  # 48 hours max
                    return True, f"Grid too old: {grid_age}"
            
            return False, "Grid conditions still valid"
            
        except Exception as e:
            logger.error(f"[GRID_TRADING] Error checking grid closure: {e}")
            return True, f"Close due to error: {str(e)}"
    
    async def manage_positions(self) -> List[Tuple[str, str]]:
        """Manage grid trading positions"""
        actions = []
        
        try:
            # Check each active grid
            for token_address in list(self.active_grids.keys()):
                # This would be called with real current price in production
                current_price = self.price_history[token_address][-1] if token_address in self.price_history else 0
                
                if current_price <= 0:
                    continue
                
                # Check if grid should be closed
                should_close, reason = self.should_close_grid(token_address, current_price)
                if should_close:
                    actions.append((token_address, f"Close grid: {reason}"))
                    # Clean up grid data
                    self.active_grids.pop(token_address, None)
                    self.grid_positions.pop(token_address, None)
                    self.market_ranges.pop(token_address, None)
                    continue
                
                # Check for triggered grid levels
                triggered_levels = self.check_grid_triggers(token_address, current_price)
                for level in triggered_levels:
                    actions.append((token_address, f"Execute {level.grid_type} at {level.price}"))
                    level.status = 'filled'
                    level.filled_at = datetime.now()
            
            return actions
            
        except Exception as e:
            logger.error(f"[GRID_TRADING] Error managing positions: {e}")
            return []
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get grid trading strategy summary"""
        return {
            "strategy_type": "Grid Trading",
            "active_grids": len(self.active_grids),
            "config": {
                "min_range_width": f"{self.MIN_RANGE_WIDTH:.1%}",
                "max_range_width": f"{self.MAX_RANGE_WIDTH:.1%}",
                "min_range_confidence": f"{self.MIN_RANGE_CONFIDENCE:.1%}",
                "grid_levels": f"{self.MIN_GRID_LEVELS}-{self.MAX_GRID_LEVELS}",
                "profit_target": f"{self.GRID_PROFIT_TARGET:.1%}"
            },
            "grids_summary": {
                token_address: {
                    "levels": len(self.grid_positions.get(token_address, [])),
                    "spacing": f"{config.grid_spacing:.1%}",
                    "range": f"{self.market_ranges[token_address].range_width:.1%}" if token_address in self.market_ranges else "N/A"
                }
                for token_address, config in self.active_grids.items()
            }
        }

def create_grid_trading_strategy(settings=None, portfolio=None) -> GridTradingStrategy:
    """Create and configure grid trading strategy for the new architecture"""
    return GridTradingStrategy(settings, portfolio)