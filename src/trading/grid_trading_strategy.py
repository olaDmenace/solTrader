#!/usr/bin/env python3
"""
Grid Trading Strategy - Phase 2 Implementation 📈

This module implements a sophisticated grid trading system that:
- Detects ranging/sideways markets automatically
- Creates dynamic grid levels based on volatility
- Captures systematic profits in both directions
- Manages risk with position limits
- Coordinates with momentum strategies

Perfect for sideways markets where momentum strategies struggle.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import talib

logger = logging.getLogger(__name__)

@dataclass
class GridLevel:
    """Individual grid level definition"""
    price: float
    level_id: int
    grid_type: str  # 'buy' or 'sell'
    position_size: float
    status: str  # 'pending', 'filled', 'cancelled'
    created_at: datetime
    filled_at: Optional[datetime] = None

@dataclass
class GridConfiguration:
    """Grid trading configuration"""
    center_price: float
    grid_spacing: float
    grid_count: int
    position_size_per_level: float
    upper_boundary: float
    lower_boundary: float
    total_capital_allocated: float
    
@dataclass
class MarketRange:
    """Detected market range information"""
    support_level: float
    resistance_level: float
    range_width: float
    range_confidence: float  # 0.0 to 1.0
    range_duration_hours: float
    volatility: float
    is_valid_for_grid: bool

class GridTradingStrategy:
    """Advanced Grid Trading Strategy implementation"""
    
    def __init__(self, settings, jupiter_client=None, wallet=None, mode=None, analytics=None):
        self.settings = settings
        self.jupiter_client = jupiter_client
        self.wallet = wallet
        self.mode = mode
        self.analytics = analytics  # Analytics integration
        self.active_grids = {}  # token_address -> GridConfiguration
        self.grid_positions = {}  # token_address -> List[GridLevel]
        self.market_ranges = {}  # token_address -> MarketRange
        
        # Grid trading parameters - configurable
        self.MIN_RANGE_WIDTH = 0.05  # 5% minimum range width
        self.MAX_RANGE_WIDTH = 0.30  # 30% maximum range width  
        self.MIN_RANGE_CONFIDENCE = 0.7  # 70% confidence required
        self.GRID_PROFIT_TARGET = settings.GRID_TAKE_PROFIT  # From settings (2%)
        self.MAX_GRID_LEVELS = settings.MAX_GRID_LEVELS  # From settings (10)
        self.MIN_GRID_LEVELS = settings.MIN_GRID_LEVELS  # From settings (3)
        self.MIN_GRID_SPACING = settings.MIN_GRID_SPACING  # From settings (1%)
        
        logger.info("Grid Trading Strategy initialized")
    
    def detect_market_range(self, price_history: List[float], timestamps: List[datetime]) -> Optional[MarketRange]:
        """Detect if market is in a ranging/sideways pattern"""
        
        if len(price_history) < 50:  # Need sufficient data
            return None
        
        prices = np.array(price_history)
        
        # Calculate support and resistance levels
        support_level = np.percentile(prices, 10)  # 10th percentile
        resistance_level = np.percentile(prices, 90)  # 90th percentile
        
        range_width = (resistance_level - support_level) / support_level
        
        # Check if range is valid for grid trading
        if range_width < self.MIN_RANGE_WIDTH or range_width > self.MAX_RANGE_WIDTH:
            return None
        
        # Calculate range confidence based on how well prices respect levels
        price_touches_support = np.sum((prices >= support_level) & (prices <= support_level * 1.02))
        price_touches_resistance = np.sum((prices <= resistance_level) & (prices >= resistance_level * 0.98))
        total_touches = price_touches_support + price_touches_resistance
        
        range_confidence = min(total_touches / len(prices) * 5, 1.0)  # Normalize to 1.0
        
        # Calculate volatility
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(24)  # Annualized
        
        # Calculate range duration
        range_duration = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # Hours
        
        # Determine if valid for grid trading
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
        """Calculate optimal grid spacing based on market conditions"""
        
        # Base spacing on volatility and range width
        volatility_factor = min(market_range.volatility * 2, 0.1)  # Cap at 10%
        range_factor = market_range.range_width / 10  # Scale range width
        
        # Combine factors
        optimal_spacing = max(
            self.MIN_GRID_SPACING,  # Minimum spacing
            volatility_factor + range_factor  # Dynamic calculation
        )
        
        # Calculate number of grid levels that fit in the range
        range_height = market_range.resistance_level - market_range.support_level
        max_levels_in_range = int(range_height / (current_price * optimal_spacing))
        
        # Constrain to settings limits
        grid_levels = max(
            self.MIN_GRID_LEVELS,
            min(max_levels_in_range, self.MAX_GRID_LEVELS)
        )
        
        return optimal_spacing, grid_levels
    
    def create_grid_configuration(self, token_address: str, current_price: float, 
                                market_range: MarketRange) -> GridConfiguration:
        """Create grid configuration for a token"""
        
        # Calculate optimal spacing and levels
        grid_spacing, grid_count = self.calculate_optimal_grid_spacing(market_range, current_price)
        
        # Calculate position size per level
        total_capital_for_grid = self.settings.PORTFOLIO_VALUE * 0.1  # 10% max for grid trading
        position_size_per_level = total_capital_for_grid / (grid_count * 2)  # Buy and sell levels
        
        # Set boundaries
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
        """Generate buy and sell grid levels"""
        
        grid_levels = []
        current_time = datetime.now()
        
        # Generate buy levels (below center price)
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
        
        # Generate sell levels (above center price)  
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
    
    async def setup_grid_trading(self, token_address: str, price_history: List[float], 
                                timestamps: List[datetime], current_price: float) -> bool:
        """Setup grid trading for a token"""
        
        # Detect market range
        market_range = self.detect_market_range(price_history, timestamps)
        
        if not market_range or not market_range.is_valid_for_grid:
            logger.debug(f"Market range not suitable for grid trading: {token_address}")
            return False
        
        # Create grid configuration
        config = self.create_grid_configuration(token_address, current_price, market_range)
        
        # Generate grid levels
        grid_levels = self.generate_grid_levels(config)
        
        if len(grid_levels) < self.MIN_GRID_LEVELS:
            logger.warning(f"Insufficient grid levels generated: {len(grid_levels)}")
            return False
        
        # Store grid setup
        self.active_grids[token_address] = config
        self.grid_positions[token_address] = grid_levels
        self.market_ranges[token_address] = market_range
        
        logger.info(f"Grid trading setup for {token_address}: {len(grid_levels)} levels, "
                   f"{config.grid_spacing:.1%} spacing, range {market_range.range_width:.1%}")
        
        return True
    
    def check_grid_triggers(self, token_address: str, current_price: float) -> List[GridLevel]:
        """Check which grid levels should be triggered"""
        
        if token_address not in self.grid_positions:
            return []
        
        triggered_levels = []
        grid_levels = self.grid_positions[token_address]
        
        for level in grid_levels:
            if level.status != 'pending':
                continue
            
            # Check buy level trigger (price at or below buy level)
            if level.grid_type == 'buy' and current_price <= level.price:
                triggered_levels.append(level)
            
            # Check sell level trigger (price at or above sell level)
            elif level.grid_type == 'sell' and current_price >= level.price:
                triggered_levels.append(level)
        
        return triggered_levels
    
    def execute_grid_level(self, token_address: str, level: GridLevel, token_symbol: str = "UNKNOWN") -> Dict:
        """Execute a grid level trade"""
        
        # Mark level as filled
        level.status = 'filled'
        level.filled_at = datetime.now()
        
        # Create trade order data
        trade_data = {
            'token_address': token_address,
            'trade_type': level.grid_type,
            'price': level.price,
            'quantity': level.position_size / level.price,
            'strategy': 'grid_trading',
            'grid_level_id': level.level_id,
            'timestamp': level.filled_at
        }
        
        # Record trade with analytics if available
        if self.analytics:
            try:
                if level.grid_type == 'buy':
                    # Record buy trade entry
                    trade_id = self.analytics.record_trade_entry(
                        token_address=token_address,
                        token_symbol=token_symbol,
                        entry_price=level.price,
                        quantity=trade_data['quantity'],
                        gas_fees=0.005,  # Estimated gas fee
                        discovery_source="grid_trading"
                    )
                    trade_data['analytics_trade_id'] = trade_id
                    logger.info(f"[GRID_ANALYTICS] Recorded buy entry: {trade_id}")
                
                else:  # sell
                    # For grid trading, we need to find the corresponding buy trade
                    # For now, record as a simple trade completion
                    # TODO: Implement proper buy/sell pairing for grid trades
                    logger.info(f"[GRID_ANALYTICS] Grid sell executed at ${level.price:.6f}")
                    
            except Exception as e:
                logger.warning(f"[GRID_ANALYTICS] Failed to record trade: {e}")
        
        logger.info(f"Grid level executed: {level.grid_type} {token_address} at ${level.price:.6f}")
        
        return trade_data
    
    def calculate_grid_profit(self, token_address: str) -> Dict:
        """Calculate current grid trading profit"""
        
        if token_address not in self.grid_positions:
            return {'total_profit': 0, 'realized_profit': 0, 'unrealized_profit': 0}
        
        grid_levels = self.grid_positions[token_address]
        config = self.active_grids[token_address]
        
        realized_profit = 0
        unrealized_profit = 0
        
        # Calculate profit from filled levels
        filled_buy_levels = [l for l in grid_levels if l.grid_type == 'buy' and l.status == 'filled']
        filled_sell_levels = [l for l in grid_levels if l.grid_type == 'sell' and l.status == 'filled']
        
        # Each filled buy-sell pair generates profit
        profit_per_pair = config.position_size_per_level * config.grid_spacing * 2  # Approximate
        completed_pairs = min(len(filled_buy_levels), len(filled_sell_levels))
        
        realized_profit = completed_pairs * profit_per_pair
        
        return {
            'total_profit': realized_profit + unrealized_profit,
            'realized_profit': realized_profit,
            'unrealized_profit': unrealized_profit,
            'completed_pairs': completed_pairs,
            'active_levels': len([l for l in grid_levels if l.status == 'pending'])
        }
    
    def should_close_grid(self, token_address: str, current_price: float) -> Tuple[bool, str]:
        """Determine if grid should be closed"""
        
        if token_address not in self.market_ranges:
            return False, ""
        
        market_range = self.market_ranges[token_address]
        config = self.active_grids[token_address]
        
        # Close if price breaks out of range significantly
        if current_price > market_range.resistance_level * 1.05:  # 5% breakout above
            return True, "Upward breakout detected"
        
        if current_price < market_range.support_level * 0.95:  # 5% breakout below
            return True, "Downward breakout detected"
        
        # Close if grid has been running too long without profit
        grid_age = datetime.now() - config.created_at if hasattr(config, 'created_at') else timedelta(0)
        if grid_age > timedelta(hours=12):  # 12 hours max
            profit_data = self.calculate_grid_profit(token_address)
            if profit_data['completed_pairs'] == 0:
                return True, "No profit after 12 hours"
        
        return False, ""
    
    async def close_grid_trading(self, token_address: str, reason: str) -> Dict:
        """Close grid trading for a token"""
        
        if token_address not in self.active_grids:
            return {'status': 'not_found'}
        
        # Calculate final profit
        final_profit = self.calculate_grid_profit(token_address)
        
        # Cancel all pending levels
        if token_address in self.grid_positions:
            for level in self.grid_positions[token_address]:
                if level.status == 'pending':
                    level.status = 'cancelled'
        
        # Clean up
        del self.active_grids[token_address]
        if token_address in self.grid_positions:
            del self.grid_positions[token_address]
        if token_address in self.market_ranges:
            del self.market_ranges[token_address]
        
        logger.info(f"Grid trading closed for {token_address}: {reason} | "
                   f"Profit: {final_profit['realized_profit']:.2f}")
        
        return {
            'status': 'closed',
            'reason': reason,
            'final_profit': final_profit,
            'timestamp': datetime.now()
        }
    
    def get_active_grids_summary(self) -> Dict:
        """Get summary of all active grids"""
        
        summary = {
            'total_active_grids': len(self.active_grids),
            'total_allocated_capital': sum(config.total_capital_allocated for config in self.active_grids.values()),
            'grids': []
        }
        
        for token_address, config in self.active_grids.items():
            profit_data = self.calculate_grid_profit(token_address)
            
            grid_summary = {
                'token_address': token_address,
                'center_price': config.center_price,
                'grid_spacing': config.grid_spacing,
                'active_levels': len([l for l in self.grid_positions[token_address] if l.status == 'pending']),
                'completed_pairs': profit_data['completed_pairs'],
                'realized_profit': profit_data['realized_profit'],
                'capital_allocated': config.total_capital_allocated
            }
            
            summary['grids'].append(grid_summary)
        
        return summary

async def create_grid_trading_strategy(settings):
    """Factory function to create grid trading strategy"""
    return GridTradingStrategy(settings)

# Example usage and testing
if __name__ == "__main__":
    async def test_grid_strategy():
        from src.config.settings import load_settings
        
        settings = load_settings()
        grid_strategy = GridTradingStrategy(settings)
        
        # Generate sample price data (ranging market)
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
        base_price = 0.001
        
        # Create ranging price pattern
        price_history = []
        for i, ts in enumerate(timestamps):
            # Simulate ranging between 0.0008 and 0.0012
            cycle_position = (i % 20) / 20  # 20-hour cycle
            range_factor = 0.2  # 20% range
            noise = np.random.normal(0, 0.02)  # 2% noise
            
            if cycle_position < 0.5:  # Rising phase
                price = base_price * (0.9 + range_factor * cycle_position * 2)
            else:  # Falling phase  
                price = base_price * (1.1 - range_factor * (cycle_position - 0.5) * 2)
            
            price += price * noise
            price_history.append(max(price, 0.0001))  # Minimum price floor
        
        current_price = price_history[-1]
        
        # Test grid setup
        success = await grid_strategy.setup_grid_trading(
            'TestToken123',
            price_history, 
            timestamps,
            current_price
        )
        
        print(f"Grid setup successful: {success}")
        
        if success:
            # Test grid triggers
            triggered = grid_strategy.check_grid_triggers('TestToken123', current_price * 0.98)  # 2% lower
            print(f"Triggered levels: {len(triggered)}")
            
            # Get summary
            summary = grid_strategy.get_active_grids_summary()
            print(f"Active grids: {summary['total_active_grids']}")
    
    if __name__ == "__main__":
        asyncio.run(test_grid_strategy())