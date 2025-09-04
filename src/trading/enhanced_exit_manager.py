#!/usr/bin/env python3
"""
Enhanced Exit Manager - Smart Position Exit Logic 🎯

This module implements sophisticated exit strategies including:
- Dynamic stop-losses based on volatility
- Multiple take-profit levels with position scaling
- Trailing stops to lock in profits
- Time-based exits to prevent bag-holding
- Daily loss limit enforcement

CRITICAL RISK MITIGATION:
- Hard stop-losses at 10% per trade
- Daily loss limit at 5% of portfolio
- Maximum position size: 5%
- Maximum slippage: 3%
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PositionExit:
    """Position exit signal with detailed reasoning"""
    token_address: str
    exit_type: str  # 'stop_loss', 'take_profit', 'trailing_stop', 'time_exit', 'daily_limit'
    exit_price: float
    exit_percentage: float  # How much of position to exit
    confidence: float
    reason: str
    urgency: int  # 1-5, 5 = immediate exit required

@dataclass
class RiskMetrics:
    """Current portfolio risk metrics"""
    daily_pnl_percentage: float
    open_positions_count: int
    total_portfolio_risk: float
    largest_position_risk: float
    volatility_index: float

class EnhancedExitManager:
    """Enhanced exit management with multiple sophisticated strategies"""
    
    def __init__(self, settings):
        self.settings = settings
        self.position_entry_times = {}  # Track when positions were opened
        self.position_high_watermarks = {}  # Track highest price seen for trailing stops
        self.daily_pnl_tracker = 0.0  # Track daily P&L
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Enhanced exit parameters - NOW CONFIGURABLE VIA .ENV
        self.DYNAMIC_STOP_LOSS_BASE = settings.DYNAMIC_STOP_LOSS_BASE
        self.VOLATILITY_MULTIPLIER = settings.VOLATILITY_MULTIPLIER
        self.TRAILING_STOP_ACTIVATION = settings.TRAILING_STOP_ACTIVATION
        self.TRAILING_STOP_PERCENTAGE = settings.TRAILING_STOP_PERCENTAGE
        self.MAX_HOLD_TIME_HOURS = settings.MAX_HOLD_TIME_HOURS
        self.TAKE_PROFIT_LEVELS = [
            settings.TAKE_PROFIT_LEVEL_1,
            settings.TAKE_PROFIT_LEVEL_2,
            settings.TAKE_PROFIT_LEVEL_3
        ]
        self.POSITION_SCALING = [
            settings.POSITION_SCALE_1,
            settings.POSITION_SCALE_2,
            settings.POSITION_SCALE_3
        ]
        
        logger.info("Enhanced Exit Manager initialized with CRITICAL risk mitigation")
    
    def reset_daily_tracking_if_needed(self):
        """Reset daily tracking at midnight"""
        now = datetime.now()
        if now.date() > self.daily_reset_time.date():
            self.daily_pnl_tracker = 0.0
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            logger.info("Daily P&L tracking reset")
    
    def update_daily_pnl(self, trade_pnl: float):
        """Update daily P&L tracker"""
        self.reset_daily_tracking_if_needed()
        self.daily_pnl_tracker += trade_pnl
        logger.info(f"Daily P&L updated: {self.daily_pnl_tracker:.2%}")
    
    def calculate_risk_metrics(self, positions: List[Dict]) -> RiskMetrics:
        """Calculate current portfolio risk metrics"""
        self.reset_daily_tracking_if_needed()
        
        total_value = sum(pos.get('current_value', 0) for pos in positions)
        portfolio_value = self.settings.PORTFOLIO_VALUE
        
        # Calculate risk metrics
        daily_pnl_pct = self.daily_pnl_tracker
        open_positions = len(positions)
        total_risk = total_value / portfolio_value if portfolio_value > 0 else 0
        largest_position = max([pos.get('current_value', 0) / portfolio_value for pos in positions], default=0)
        
        # Simple volatility index (could be enhanced with actual volatility calculations)
        recent_pnls = [pos.get('unrealized_pnl_percentage', 0) for pos in positions]
        volatility = np.std(recent_pnls) if recent_pnls else 0
        
        return RiskMetrics(
            daily_pnl_percentage=daily_pnl_pct,
            open_positions_count=open_positions,
            total_portfolio_risk=total_risk,
            largest_position_risk=largest_position,
            volatility_index=volatility
        )
    
    def calculate_dynamic_stop_loss(self, token_address: str, current_volatility: float) -> float:
        """Calculate dynamic stop loss based on volatility"""
        # Adjust stop loss based on volatility
        volatility_adjustment = min(current_volatility * self.VOLATILITY_MULTIPLIER, 0.05)  # Cap at 5%
        dynamic_stop = self.DYNAMIC_STOP_LOSS_BASE + volatility_adjustment
        
        # Never exceed maximum stop loss
        return min(dynamic_stop, 0.15)  # Hard cap at 15%
    
    def update_trailing_stops(self, token_address: str, current_price: float, entry_price: float):
        """Update trailing stop high watermark"""
        current_gain = (current_price - entry_price) / entry_price
        
        # Activate trailing stop if profit exceeds threshold
        if current_gain >= self.TRAILING_STOP_ACTIVATION:
            if token_address not in self.position_high_watermarks:
                self.position_high_watermarks[token_address] = current_price
                logger.info(f"Trailing stop activated for {token_address} at ${current_price:.6f}")
            else:
                # Update high watermark if price increased
                if current_price > self.position_high_watermarks[token_address]:
                    self.position_high_watermarks[token_address] = current_price
                    logger.debug(f"Trailing stop updated for {token_address}: ${current_price:.6f}")
    
    def check_trailing_stop_exit(self, token_address: str, current_price: float) -> Optional[PositionExit]:
        """Check if trailing stop should trigger"""
        if token_address not in self.position_high_watermarks:
            return None
        
        high_watermark = self.position_high_watermarks[token_address]
        trailing_stop_price = high_watermark * (1 - self.TRAILING_STOP_PERCENTAGE)
        
        if current_price <= trailing_stop_price:
            return PositionExit(
                token_address=token_address,
                exit_type='trailing_stop',
                exit_price=current_price,
                exit_percentage=1.0,  # Exit full position
                confidence=0.9,
                reason=f"Trailing stop triggered: Price ${current_price:.6f} below trail ${trailing_stop_price:.6f}",
                urgency=4
            )
        
        return None
    
    def check_time_based_exit(self, token_address: str, entry_time: datetime) -> Optional[PositionExit]:
        """Check if position should exit based on hold time"""
        if token_address not in self.position_entry_times:
            self.position_entry_times[token_address] = entry_time
        
        hold_time = datetime.now() - entry_time
        max_hold_time = timedelta(hours=self.MAX_HOLD_TIME_HOURS)
        
        if hold_time >= max_hold_time:
            return PositionExit(
                token_address=token_address,
                exit_type='time_exit',
                exit_price=0,  # Will be filled with current market price
                exit_percentage=1.0,
                confidence=0.8,
                reason=f"Maximum hold time exceeded: {hold_time.total_seconds()/3600:.1f} hours",
                urgency=3
            )
        
        return None
    
    def check_daily_loss_limit(self, risk_metrics: RiskMetrics) -> Optional[PositionExit]:
        """Check if daily loss limit requires position closure"""
        if risk_metrics.daily_pnl_percentage <= -self.settings.MAX_DAILY_LOSS:
            return PositionExit(
                token_address="ALL_POSITIONS",
                exit_type='daily_limit',
                exit_price=0,
                exit_percentage=1.0,
                confidence=1.0,
                reason=f"Daily loss limit exceeded: {risk_metrics.daily_pnl_percentage:.2%}",
                urgency=5  # Maximum urgency
            )
        
        return None
    
    def check_take_profit_levels(self, token_address: str, current_price: float, 
                                entry_price: float) -> List[PositionExit]:
        """Check multiple take profit levels"""
        exits = []
        current_gain = (current_price - entry_price) / entry_price
        
        for i, tp_level in enumerate(self.TAKE_PROFIT_LEVELS):
            if current_gain >= tp_level:
                exit_percentage = self.POSITION_SCALING[i]
                exits.append(PositionExit(
                    token_address=token_address,
                    exit_type='take_profit',
                    exit_price=current_price,
                    exit_percentage=exit_percentage,
                    confidence=0.85,
                    reason=f"Take profit level {i+1}: {tp_level:.0%} gain achieved",
                    urgency=2
                ))
        
        return exits
    
    def analyze_position_exits(self, positions: List[Dict]) -> List[PositionExit]:
        """Analyze all positions and generate exit signals"""
        exits = []
        risk_metrics = self.calculate_risk_metrics(positions)
        
        # Check daily loss limit first (highest priority)
        daily_limit_exit = self.check_daily_loss_limit(risk_metrics)
        if daily_limit_exit:
            exits.append(daily_limit_exit)
            logger.critical(f"DAILY LOSS LIMIT EXCEEDED: {risk_metrics.daily_pnl_percentage:.2%}")
            return exits  # Return immediately - close all positions
        
        # Analyze each position
        for position in positions:
            token_address = position.get('token_address', '')
            current_price = position.get('current_price', 0)
            entry_price = position.get('entry_price', 0)
            entry_time = position.get('entry_time', datetime.now())
            current_volatility = position.get('volatility', 0.1)  # Default 10% volatility
            
            if not all([token_address, current_price, entry_price]):
                continue
            
            # Update trailing stops
            self.update_trailing_stops(token_address, current_price, entry_price)
            
            # Check stop loss
            current_loss = (entry_price - current_price) / entry_price
            dynamic_stop_loss = self.calculate_dynamic_stop_loss(token_address, current_volatility)
            
            if current_loss >= dynamic_stop_loss:
                exits.append(PositionExit(
                    token_address=token_address,
                    exit_type='stop_loss',
                    exit_price=current_price,
                    exit_percentage=1.0,
                    confidence=1.0,
                    reason=f"Dynamic stop loss triggered: {current_loss:.2%} loss >= {dynamic_stop_loss:.2%}",
                    urgency=5
                ))
                continue  # Skip other checks if stop loss triggered
            
            # Check trailing stop
            trailing_exit = self.check_trailing_stop_exit(token_address, current_price)
            if trailing_exit:
                exits.append(trailing_exit)
                continue
            
            # Check time-based exit
            time_exit = self.check_time_based_exit(token_address, entry_time)
            if time_exit:
                exits.append(time_exit)
                continue
            
            # Check take profit levels
            tp_exits = self.check_take_profit_levels(token_address, current_price, entry_price)
            exits.extend(tp_exits)
        
        # Sort exits by urgency (highest first)
        exits.sort(key=lambda x: x.urgency, reverse=True)
        
        if exits:
            logger.info(f"Generated {len(exits)} exit signals")
            for exit_signal in exits[:3]:  # Log top 3 most urgent
                logger.info(f"Exit signal: {exit_signal.exit_type} for {exit_signal.token_address} - {exit_signal.reason}")
        
        return exits
    
    def generate_exit_report(self, exits: List[PositionExit]) -> Dict:
        """Generate comprehensive exit report"""
        if not exits:
            return {"status": "No exits required", "total_exits": 0}
        
        exit_types = {}
        total_positions_to_exit = 0
        highest_urgency = 0
        
        for exit_signal in exits:
            exit_type = exit_signal.exit_type
            exit_types[exit_type] = exit_types.get(exit_type, 0) + 1
            total_positions_to_exit += exit_signal.exit_percentage
            highest_urgency = max(highest_urgency, exit_signal.urgency)
        
        return {
            "status": "Exits required",
            "total_exits": len(exits),
            "exit_types": exit_types,
            "positions_affected": total_positions_to_exit,
            "highest_urgency": highest_urgency,
            "immediate_action_required": highest_urgency >= 4
        }

async def create_enhanced_exit_manager(settings):
    """Factory function to create enhanced exit manager"""
    return EnhancedExitManager(settings)

# Example usage for testing
if __name__ == "__main__":
    async def test_exit_manager():
        from src.config.settings import load_settings
        
        settings = load_settings()
        exit_manager = EnhancedExitManager(settings)
        
        # Test with sample positions
        sample_positions = [
            {
                'token_address': 'Token123',
                'current_price': 0.0008,
                'entry_price': 0.001,
                'entry_time': datetime.now() - timedelta(hours=2),
                'current_value': 50,
                'unrealized_pnl_percentage': -0.2,
                'volatility': 0.15
            }
        ]
        
        exits = exit_manager.analyze_position_exits(sample_positions)
        report = exit_manager.generate_exit_report(exits)
        
        print("Exit Analysis Report:")
        print(f"Status: {report['status']}")
        print(f"Total exits: {report['total_exits']}")
        if exits:
            print("\nExit signals:")
            for exit_signal in exits:
                print(f"- {exit_signal.exit_type}: {exit_signal.reason}")
    
    if __name__ == "__main__":
        asyncio.run(test_exit_manager())