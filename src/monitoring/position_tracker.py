#!/usr/bin/env python3
"""
Real-Time Position Tracking System 📊

This module provides:
- Individual trade notifications (email/Telegram)
- Live P&L dashboard for all open positions
- Exit signal detection (momentum reversal, volume decline)
- Portfolio heat tracking
- Real-time risk monitoring

Integration with Enhanced Exit Manager for comprehensive position management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class TradeNotification:
    """Trade notification data structure"""
    trade_id: str
    token_address: str
    token_symbol: str
    notification_type: str  # 'entry', 'exit', 'alert', 'stop_loss', 'take_profit'
    price: float
    quantity: float
    value_usd: float
    timestamp: datetime
    message: str
    urgency: int  # 1-5, 5 = critical

@dataclass
class PositionSnapshot:
    """Real-time position snapshot"""
    token_address: str
    token_symbol: str
    entry_price: float
    current_price: float
    quantity: float
    entry_value: float
    current_value: float
    unrealized_pnl: float
    unrealized_pnl_percentage: float
    hold_time: timedelta
    strategy: str
    exit_signals: List[str]
    risk_level: int  # 1-5, 5 = high risk

@dataclass
class PortfolioHeat:
    """Portfolio risk heat map"""
    total_positions: int
    total_exposure: float
    largest_position_pct: float
    daily_pnl_pct: float
    open_risk_score: float
    heat_level: str  # 'COOL', 'WARM', 'HOT', 'CRITICAL'
    risk_alerts: List[str]

class RealTimePositionTracker:
    """Real-time position tracking and notification system"""
    
    def __init__(self, settings):
        self.settings = settings
        self.active_positions = {}  # Track all active positions
        self.notification_history = []  # Track sent notifications
        self.position_alerts = {}  # Track position-specific alerts
        
        # Notification cooldowns to prevent spam
        self.notification_cooldowns = {
            'entry': timedelta(seconds=0),     # No cooldown for entries
            'exit': timedelta(seconds=0),      # No cooldown for exits
            'alert': timedelta(minutes=5),     # 5min cooldown for alerts
            'stop_loss': timedelta(seconds=0), # No cooldown for stop losses
            'take_profit': timedelta(seconds=0) # No cooldown for take profits
        }
        
        logger.info("Real-Time Position Tracker initialized")
    
    async def track_new_position(self, position_data: Dict):
        """Track a new position entry"""
        token_address = position_data.get('token_address', '')
        
        # Create position snapshot
        position = PositionSnapshot(
            token_address=token_address,
            token_symbol=position_data.get('token_symbol', 'UNKNOWN'),
            entry_price=position_data.get('entry_price', 0),
            current_price=position_data.get('entry_price', 0),  # Same at entry
            quantity=position_data.get('quantity', 0),
            entry_value=position_data.get('entry_value', 0),
            current_value=position_data.get('entry_value', 0),
            unrealized_pnl=0.0,
            unrealized_pnl_percentage=0.0,
            hold_time=timedelta(seconds=0),
            strategy=position_data.get('strategy', 'momentum'),
            exit_signals=[],
            risk_level=2  # Default medium-low risk
        )
        
        self.active_positions[token_address] = {
            'snapshot': position,
            'entry_time': datetime.now(),
            'last_update': datetime.now(),
            'alerts_sent': []
        }
        
        # Send entry notification
        notification = TradeNotification(
            trade_id=f"entry_{token_address}_{int(datetime.now().timestamp())}",
            token_address=token_address,
            token_symbol=position.token_symbol,
            notification_type='entry',
            price=position.entry_price,
            quantity=position.quantity,
            value_usd=position.entry_value,
            timestamp=datetime.now(),
            message=f"🚀 POSITION OPENED: {position.token_symbol} at ${position.entry_price:.6f}",
            urgency=3
        )
        
        await self.send_notification(notification)
        logger.info(f"New position tracked: {position.token_symbol} at ${position.entry_price:.6f}")
    
    async def update_position(self, token_address: str, current_price: float, current_volume: float = 0):
        """Update existing position with current market data"""
        if token_address not in self.active_positions:
            logger.warning(f"Attempted to update non-existent position: {token_address}")
            return
        
        position_data = self.active_positions[token_address]
        snapshot = position_data['snapshot']
        entry_time = position_data['entry_time']
        
        # Update position metrics
        snapshot.current_price = current_price
        snapshot.current_value = snapshot.quantity * current_price
        snapshot.unrealized_pnl = snapshot.current_value - snapshot.entry_value
        snapshot.unrealized_pnl_percentage = (snapshot.unrealized_pnl / snapshot.entry_value) if snapshot.entry_value > 0 else 0
        snapshot.hold_time = datetime.now() - entry_time
        
        # Update risk level based on P&L and hold time
        snapshot.risk_level = self.calculate_position_risk_level(snapshot)
        
        # Check for exit signals
        exit_signals = await self.detect_exit_signals(snapshot, current_volume)
        snapshot.exit_signals = exit_signals
        
        # Update timestamp
        position_data['last_update'] = datetime.now()
        
        # Send alerts if needed
        await self.check_position_alerts(snapshot, position_data)
    
    def calculate_position_risk_level(self, snapshot: PositionSnapshot) -> int:
        """Calculate position risk level (1-5)"""
        risk_score = 1
        
        # Risk based on unrealized P&L
        if snapshot.unrealized_pnl_percentage <= -0.05:  # -5%
            risk_score += 1
        if snapshot.unrealized_pnl_percentage <= -0.08:  # -8%
            risk_score += 1
        if snapshot.unrealized_pnl_percentage <= -0.10:  # -10%
            risk_score += 2  # Major risk
        
        # Risk based on hold time (meme coins decay quickly)
        hold_hours = snapshot.hold_time.total_seconds() / 3600
        if hold_hours > 2:
            risk_score += 1
        if hold_hours > 4:
            risk_score += 1
        if hold_hours > 6:
            risk_score += 1
        
        return min(risk_score, 5)  # Cap at 5
    
    async def detect_exit_signals(self, snapshot: PositionSnapshot, current_volume: float) -> List[str]:
        """Detect potential exit signals"""
        signals = []
        
        # Momentum reversal detection (simple implementation)
        if snapshot.unrealized_pnl_percentage < -0.03:  # -3% momentum loss
            signals.append("momentum_reversal")
        
        # Volume decline detection (if volume drops significantly)
        # This would need historical volume data for proper implementation
        if current_volume > 0:  # Placeholder for volume analysis
            # signals.append("volume_decline")
            pass
        
        # Time-based signal
        hold_hours = snapshot.hold_time.total_seconds() / 3600
        if hold_hours >= self.settings.MAX_HOLD_TIME_HOURS:
            signals.append("max_hold_time")
        
        # Profit protection signal
        if snapshot.unrealized_pnl_percentage >= 0.15:  # 15% profit
            signals.append("profit_protection_active")
        
        return signals
    
    async def check_position_alerts(self, snapshot: PositionSnapshot, position_data: Dict):
        """Check if position needs alerts"""
        alerts_sent = position_data.get('alerts_sent', [])
        
        # Stop loss approaching alert
        if (snapshot.unrealized_pnl_percentage <= -0.08 and 
            'stop_loss_warning' not in alerts_sent):
            
            notification = TradeNotification(
                trade_id=f"alert_{snapshot.token_address}_{int(datetime.now().timestamp())}",
                token_address=snapshot.token_address,
                token_symbol=snapshot.token_symbol,
                notification_type='alert',
                price=snapshot.current_price,
                quantity=snapshot.quantity,
                value_usd=snapshot.current_value,
                timestamp=datetime.now(),
                message=f"⚠️ STOP LOSS WARNING: {snapshot.token_symbol} down {snapshot.unrealized_pnl_percentage:.1%}",
                urgency=4
            )
            
            await self.send_notification(notification)
            alerts_sent.append('stop_loss_warning')
        
        # Take profit opportunity alert
        if (snapshot.unrealized_pnl_percentage >= 0.20 and 
            'take_profit_opportunity' not in alerts_sent):
            
            notification = TradeNotification(
                trade_id=f"alert_{snapshot.token_address}_{int(datetime.now().timestamp())}",
                token_address=snapshot.token_address,
                token_symbol=snapshot.token_symbol,
                notification_type='alert',
                price=snapshot.current_price,
                quantity=snapshot.quantity,
                value_usd=snapshot.current_value,
                timestamp=datetime.now(),
                message=f"🎯 TAKE PROFIT: {snapshot.token_symbol} up {snapshot.unrealized_pnl_percentage:.1%}",
                urgency=3
            )
            
            await self.send_notification(notification)
            alerts_sent.append('take_profit_opportunity')
    
    async def close_position(self, token_address: str, exit_price: float, exit_reason: str):
        """Track position closure"""
        if token_address not in self.active_positions:
            logger.warning(f"Attempted to close non-existent position: {token_address}")
            return
        
        position_data = self.active_positions[token_address]
        snapshot = position_data['snapshot']
        
        # Calculate final P&L
        final_value = snapshot.quantity * exit_price
        realized_pnl = final_value - snapshot.entry_value
        realized_pnl_pct = (realized_pnl / snapshot.entry_value) if snapshot.entry_value > 0 else 0
        
        # Send exit notification
        notification = TradeNotification(
            trade_id=f"exit_{token_address}_{int(datetime.now().timestamp())}",
            token_address=token_address,
            token_symbol=snapshot.token_symbol,
            notification_type='exit',
            price=exit_price,
            quantity=snapshot.quantity,
            value_usd=final_value,
            timestamp=datetime.now(),
            message=f"📊 POSITION CLOSED: {snapshot.token_symbol} | P&L: {realized_pnl_pct:+.1%} | Reason: {exit_reason}",
            urgency=2 if realized_pnl >= 0 else 3
        )
        
        await self.send_notification(notification)
        
        # Remove from active positions
        del self.active_positions[token_address]
        
        logger.info(f"Position closed: {snapshot.token_symbol} | P&L: {realized_pnl_pct:+.1%}")
    
    def get_portfolio_heat(self) -> PortfolioHeat:
        """Calculate current portfolio risk heat"""
        if not self.active_positions:
            return PortfolioHeat(
                total_positions=0,
                total_exposure=0,
                largest_position_pct=0,
                daily_pnl_pct=0,
                open_risk_score=0,
                heat_level='COOL',
                risk_alerts=[]
            )
        
        # Calculate metrics
        total_positions = len(self.active_positions)
        total_current_value = sum(pos['snapshot'].current_value for pos in self.active_positions.values())
        total_entry_value = sum(pos['snapshot'].entry_value for pos in self.active_positions.values())
        
        largest_position_value = max(pos['snapshot'].current_value for pos in self.active_positions.values())
        largest_position_pct = (largest_position_value / self.settings.PORTFOLIO_VALUE) if self.settings.PORTFOLIO_VALUE > 0 else 0
        
        daily_pnl = total_current_value - total_entry_value
        daily_pnl_pct = (daily_pnl / self.settings.PORTFOLIO_VALUE) if self.settings.PORTFOLIO_VALUE > 0 else 0
        
        # Calculate risk score
        risk_scores = [pos['snapshot'].risk_level for pos in self.active_positions.values()]
        avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        
        # Determine heat level
        heat_level = 'COOL'
        risk_alerts = []
        
        if avg_risk_score >= 3 or abs(daily_pnl_pct) >= 0.03:
            heat_level = 'WARM'
        if avg_risk_score >= 4 or abs(daily_pnl_pct) >= 0.05:
            heat_level = 'HOT'
            risk_alerts.append('High risk positions detected')
        if avg_risk_score >= 5 or daily_pnl_pct <= -0.05:
            heat_level = 'CRITICAL'
            risk_alerts.append('CRITICAL: Immediate attention required')
        
        if largest_position_pct > 0.10:  # Position too large
            risk_alerts.append(f'Large position detected: {largest_position_pct:.1%}')
        
        return PortfolioHeat(
            total_positions=total_positions,
            total_exposure=total_current_value / self.settings.PORTFOLIO_VALUE if self.settings.PORTFOLIO_VALUE > 0 else 0,
            largest_position_pct=largest_position_pct,
            daily_pnl_pct=daily_pnl_pct,
            open_risk_score=avg_risk_score,
            heat_level=heat_level,
            risk_alerts=risk_alerts
        )
    
    def get_live_dashboard_data(self) -> Dict:
        """Get data for live P&L dashboard"""
        portfolio_heat = self.get_portfolio_heat()
        
        positions = []
        for token_address, position_data in self.active_positions.items():
            snapshot = position_data['snapshot']
            positions.append({
                'token_symbol': snapshot.token_symbol,
                'entry_price': snapshot.entry_price,
                'current_price': snapshot.current_price,
                'unrealized_pnl_pct': snapshot.unrealized_pnl_percentage,
                'hold_time_hours': snapshot.hold_time.total_seconds() / 3600,
                'risk_level': snapshot.risk_level,
                'exit_signals': snapshot.exit_signals
            })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_heat': {
                'total_positions': portfolio_heat.total_positions,
                'total_exposure_pct': portfolio_heat.total_exposure,
                'daily_pnl_pct': portfolio_heat.daily_pnl_pct,
                'heat_level': portfolio_heat.heat_level,
                'risk_alerts': portfolio_heat.risk_alerts
            },
            'positions': positions
        }
    
    async def send_notification(self, notification: TradeNotification):
        """Send notification via configured channels"""
        # Check cooldown
        if self.is_notification_in_cooldown(notification):
            return
        
        message = f"[{notification.timestamp.strftime('%H:%M:%S')}] {notification.message}"
        
        try:
            # Email notification
            if self.settings.EMAIL_ENABLED and self.settings.EMAIL_TO:
                await self.send_email_notification(notification, message)
            
            # Telegram notification (if configured)
            if self.settings.TELEGRAM_BOT_TOKEN and self.settings.TELEGRAM_CHAT_ID:
                await self.send_telegram_notification(notification, message)
            
            # Discord notification (if configured)
            if self.settings.DISCORD_WEBHOOK_URL:
                await self.send_discord_notification(notification, message)
            
            # Store notification history
            self.notification_history.append(notification)
            
            # Limit history size
            if len(self.notification_history) > 1000:
                self.notification_history = self.notification_history[-500:]
                
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def is_notification_in_cooldown(self, notification: TradeNotification) -> bool:
        """Check if notification type is in cooldown period"""
        cooldown = self.notification_cooldowns.get(notification.notification_type, timedelta(0))
        if cooldown.total_seconds() == 0:
            return False
        
        # Check recent notifications of same type for same token
        cutoff_time = datetime.now() - cooldown
        recent_notifications = [
            n for n in self.notification_history
            if (n.token_address == notification.token_address and
                n.notification_type == notification.notification_type and
                n.timestamp > cutoff_time)
        ]
        
        return len(recent_notifications) > 0
    
    async def send_email_notification(self, notification: TradeNotification, message: str):
        """Send email notification (implementation would use SMTP)"""
        # Placeholder for email implementation
        logger.info(f"EMAIL: {message}")
    
    async def send_telegram_notification(self, notification: TradeNotification, message: str):
        """Send Telegram notification (implementation would use Telegram Bot API)"""
        # Placeholder for Telegram implementation  
        logger.info(f"TELEGRAM: {message}")
    
    async def send_discord_notification(self, notification: TradeNotification, message: str):
        """Send Discord notification (implementation would use Discord webhook)"""
        # Placeholder for Discord implementation
        logger.info(f"DISCORD: {message}")

async def create_position_tracker(settings):
    """Factory function to create position tracker"""
    return RealTimePositionTracker(settings)

# Example usage
if __name__ == "__main__":
    async def test_position_tracker():
        from src.config.settings import load_settings
        
        settings = load_settings()
        tracker = RealTimePositionTracker(settings)
        
        # Test position tracking
        position_data = {
            'token_address': 'TestToken123',
            'token_symbol': 'TEST',
            'entry_price': 0.001,
            'quantity': 1000,
            'entry_value': 1.0,
            'strategy': 'momentum'
        }
        
        await tracker.track_new_position(position_data)
        await tracker.update_position('TestToken123', 0.0011)  # +10% price
        
        dashboard_data = tracker.get_live_dashboard_data()
        print("Dashboard Data:", json.dumps(dashboard_data, indent=2))
    
    if __name__ == "__main__":
        asyncio.run(test_position_tracker())