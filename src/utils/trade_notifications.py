#!/usr/bin/env python3
"""
Real-Time Trade Notifications
Multi-channel notification system for trade events and bot status updates
"""
import json
import logging
import smtplib
import requests
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
from collections import deque
import time

logger = logging.getLogger(__name__)

@dataclass
class TradeNotification:
    """Trade notification data structure"""
    notification_id: str
    timestamp: datetime
    event_type: str  # 'trade_open', 'trade_close', 'position_update', 'alert', 'system'
    title: str
    message: str
    priority: str  # 'low', 'medium', 'high', 'critical'
    data: Dict[str, Any]
    channels: List[str]  # ['email', 'webhook', 'desktop', 'log']
    sent: bool = False
    retry_count: int = 0

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    name: str
    enabled: bool
    config: Dict[str, Any]
    rate_limit: int = 10  # messages per minute
    last_sent: Optional[datetime] = None
    sent_count: int = 0

class TradeNotificationSystem:
    """Real-time trade notification system"""
    
    def __init__(self):
        self.notifications_dir = Path("notifications")
        self.notifications_dir.mkdir(exist_ok=True)
        
        self.config_file = self.notifications_dir / "notification_config.json"
        self.history_file = self.notifications_dir / "notification_history.json"
        
        # Notification queue and history
        self.notification_queue: deque = deque(maxlen=1000)
        self.notification_history: List[TradeNotification] = []
        self.pending_notifications: Dict[str, TradeNotification] = {}
        
        # Channels
        self.channels: Dict[str, NotificationChannel] = {}
        self._initialize_channels()
        
        # Rate limiting
        self.rate_limits = {
            'email': {'count': 0, 'last_reset': datetime.now()},
            'webhook': {'count': 0, 'last_reset': datetime.now()},
            'desktop': {'count': 0, 'last_reset': datetime.now()}
        }
        
        # Background processing
        self.processing = False
        self.process_thread: Optional[threading.Thread] = None
        
        # Load existing configuration
        self._load_config()
        
    def _initialize_channels(self):
        """Initialize default notification channels"""
        self.channels = {
            'email': NotificationChannel(
                name='email',
                enabled=False,
                config={
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'recipient': '',
                    'sender_name': 'SolTrader Bot'
                }
            ),
            'webhook': NotificationChannel(
                name='webhook',
                enabled=False,
                config={
                    'url': '',
                    'method': 'POST',
                    'headers': {'Content-Type': 'application/json'},
                    'timeout': 10
                }
            ),
            'desktop': NotificationChannel(
                name='desktop',
                enabled=True,
                config={
                    'show_system_tray': True,
                    'show_console': True,
                    'sound_enabled': False
                }
            ),
            'log': NotificationChannel(
                name='log',
                enabled=True,
                config={
                    'log_level': 'INFO',
                    'include_data': True
                }
            )
        }
    
    def _load_config(self):
        """Load notification configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    
                for channel_name, channel_data in config_data.get('channels', {}).items():
                    if channel_name in self.channels:
                        self.channels[channel_name].enabled = channel_data.get('enabled', False)
                        self.channels[channel_name].config.update(channel_data.get('config', {}))
                        
        except Exception as e:
            logger.warning(f"Failed to load notification config: {e}")
    
    def save_config(self):
        """Save notification configuration to file"""
        try:
            config_data = {
                'channels': {},
                'last_updated': datetime.now().isoformat()
            }
            
            for channel_name, channel in self.channels.items():
                config_data['channels'][channel_name] = {
                    'enabled': channel.enabled,
                    'config': channel.config
                }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save notification config: {e}")
    
    def configure_email(self, smtp_server: str, smtp_port: int, username: str, 
                       password: str, recipient: str, sender_name: str = "SolTrader Bot"):
        """Configure email notifications"""
        self.channels['email'].config.update({
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'recipient': recipient,
            'sender_name': sender_name
        })
        self.channels['email'].enabled = True
        self.save_config()
        logger.info("Email notifications configured")
    
    def configure_webhook(self, url: str, method: str = "POST", 
                         headers: Optional[Dict[str, str]] = None):
        """Configure webhook notifications"""
        self.channels['webhook'].config.update({
            'url': url,
            'method': method.upper(),
            'headers': headers or {'Content-Type': 'application/json'}
        })
        self.channels['webhook'].enabled = True
        self.save_config()
        logger.info(f"Webhook notifications configured: {url}")
    
    def disable_channel(self, channel_name: str):
        """Disable a notification channel"""
        if channel_name in self.channels:
            self.channels[channel_name].enabled = False
            self.save_config()
            logger.info(f"Disabled {channel_name} notifications")
    
    def enable_channel(self, channel_name: str):
        """Enable a notification channel"""
        if channel_name in self.channels:
            self.channels[channel_name].enabled = True
            self.save_config()
            logger.info(f"Enabled {channel_name} notifications")
    
    def send_trade_notification(self, event_type: str, title: str, message: str,
                              priority: str = "medium", data: Optional[Dict[str, Any]] = None,
                              channels: Optional[List[str]] = None):
        """Send a trade notification"""
        notification_id = f"{event_type}_{int(time.time() * 1000)}"
        
        notification = TradeNotification(
            notification_id=notification_id,
            timestamp=datetime.now(),
            event_type=event_type,
            title=title,
            message=message,
            priority=priority,
            data=data or {},
            channels=channels or ['desktop', 'log'],
            sent=False,
            retry_count=0
        )
        
        self.notification_queue.append(notification)
        self.pending_notifications[notification_id] = notification
        
        # Process immediately for high/critical priority
        if priority in ['high', 'critical']:
            self._process_notification(notification)
        
        return notification_id
    
    def send_trade_opened(self, token_address: str, symbol: str, entry_price: float,
                         quantity: float, trade_type: str = "buy"):
        """Send trade opened notification"""
        title = f"Trade Opened: {symbol}"
        message = f"Opened {trade_type} position in {symbol}\nPrice: ${entry_price:.6f}\nQuantity: {quantity:,.2f}"
        
        data = {
            'token_address': token_address,
            'symbol': symbol,
            'entry_price': entry_price,
            'quantity': quantity,
            'trade_type': trade_type,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.send_trade_notification(
            event_type='trade_open',
            title=title,
            message=message,
            priority='medium',
            data=data,
            channels=['email', 'webhook', 'desktop', 'log']
        )
    
    def send_trade_closed(self, token_address: str, symbol: str, exit_price: float,
                         pnl: float, pnl_percentage: float, duration_minutes: float):
        """Send trade closed notification"""
        pnl_status = "PROFIT" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"
        
        title = f"Trade Closed: {symbol} - {pnl_status}"
        message = (f"Closed position in {symbol}\n"
                  f"Exit Price: ${exit_price:.6f}\n"
                  f"P&L: ${pnl:.6f} ({pnl_percentage:+.2f}%)\n"
                  f"Duration: {duration_minutes:.1f} minutes")
        
        data = {
            'token_address': token_address,
            'symbol': symbol,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'duration_minutes': duration_minutes,
            'timestamp': datetime.now().isoformat()
        }
        
        priority = 'high' if abs(pnl_percentage) > 10 else 'medium'
        
        return self.send_trade_notification(
            event_type='trade_close',
            title=title,
            message=message,
            priority=priority,
            data=data,
            channels=['email', 'webhook', 'desktop', 'log']
        )
    
    def send_alert_notification(self, alert_type: str, message: str, 
                              severity: str = "warning"):
        """Send system alert notification"""
        title = f"SolTrader Alert: {alert_type}"
        
        priority_map = {
            'info': 'low',
            'warning': 'medium', 
            'error': 'high',
            'critical': 'critical'
        }
        
        priority = priority_map.get(severity, 'medium')
        
        data = {
            'alert_type': alert_type,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        channels = ['desktop', 'log']
        if priority in ['high', 'critical']:
            channels.extend(['email', 'webhook'])
        
        return self.send_trade_notification(
            event_type='alert',
            title=title,
            message=message,
            priority=priority,
            data=data,
            channels=channels
        )
    
    def send_system_status(self, status: str, details: str = ""):
        """Send system status notification"""
        title = f"SolTrader Status: {status.upper()}"
        message = f"Bot status changed to: {status}"
        if details:
            message += f"\n{details}"
        
        data = {
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.send_trade_notification(
            event_type='system',
            title=title,
            message=message,
            priority='medium',
            data=data,
            channels=['desktop', 'log', 'webhook']
        )
    
    def _process_notification(self, notification: TradeNotification):
        """Process a single notification through all enabled channels"""
        try:
            for channel_name in notification.channels:
                if channel_name in self.channels and self.channels[channel_name].enabled:
                    if self._check_rate_limit(channel_name):
                        self._send_to_channel(notification, channel_name)
            
            notification.sent = True
            self.notification_history.append(notification)
            
            # Remove from pending
            if notification.notification_id in self.pending_notifications:
                del self.pending_notifications[notification.notification_id]
                
        except Exception as e:
            logger.error(f"Failed to process notification {notification.notification_id}: {e}")
            notification.retry_count += 1
    
    def _check_rate_limit(self, channel_name: str) -> bool:
        """Check if channel is within rate limits"""
        now = datetime.now()
        rate_info = self.rate_limits.get(channel_name, {'count': 0, 'last_reset': now})
        
        # Reset count if more than 1 minute has passed
        if now - rate_info['last_reset'] > timedelta(minutes=1):
            rate_info['count'] = 0
            rate_info['last_reset'] = now
        
        # Check limit
        channel = self.channels.get(channel_name)
        if channel and rate_info['count'] >= channel.rate_limit:
            return False
        
        rate_info['count'] += 1
        return True
    
    def _send_to_channel(self, notification: TradeNotification, channel_name: str):
        """Send notification to specific channel"""
        try:
            if channel_name == 'email':
                self._send_email(notification)
            elif channel_name == 'webhook':
                self._send_webhook(notification)
            elif channel_name == 'desktop':
                self._send_desktop(notification)
            elif channel_name == 'log':
                self._send_log(notification)
                
        except Exception as e:
            logger.error(f"Failed to send to {channel_name}: {e}")
    
    def _send_email(self, notification: TradeNotification):
        """Send email notification"""
        config = self.channels['email'].config
        
        if not all([config.get('username'), config.get('password'), config.get('recipient')]):
            return
        
        msg = MIMEMultipart()
        msg['From'] = f"{config['sender_name']} <{config['username']}>"
        msg['To'] = config['recipient']
        msg['Subject'] = notification.title
        
        # Create email body
        body = notification.message
        if notification.data:
            body += "\n\nDetails:\n"
            for key, value in notification.data.items():
                body += f"{key}: {value}\n"
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
        
        logger.info(f"Email sent: {notification.title}")
    
    def _send_webhook(self, notification: TradeNotification):
        """Send webhook notification"""
        config = self.channels['webhook'].config
        
        if not config.get('url'):
            return
        
        payload = {
            'notification_id': notification.notification_id,
            'timestamp': notification.timestamp.isoformat(),
            'event_type': notification.event_type,
            'title': notification.title,
            'message': notification.message,
            'priority': notification.priority,
            'data': notification.data
        }
        
        response = requests.request(
            method=config['method'],
            url=config['url'],
            json=payload,
            headers=config['headers'],
            timeout=config.get('timeout', 10)
        )
        
        response.raise_for_status()
        logger.info(f"Webhook sent: {notification.title}")
    
    def _send_desktop(self, notification: TradeNotification):
        """Send desktop notification"""
        config = self.channels['desktop'].config
        
        if config.get('show_console', True):
            priority_indicator = {
                'low': '[INFO]',
                'medium': '[TRADE]',
                'high': '[ALERT]',
                'critical': '[CRITICAL]'
            }.get(notification.priority, '[TRADE]')
            
            print(f"\n{priority_indicator} {notification.title}")
            print(f"{notification.message}")
            print(f"Time: {notification.timestamp.strftime('%H:%M:%S')}")
        
        # Try to show system notification (Windows)
        try:
            import win10toast
            toaster = win10toast.ToastNotifier()
            toaster.show_toast(
                title=notification.title,
                msg=notification.message[:100] + "..." if len(notification.message) > 100 else notification.message,
                duration=10,
                threaded=True
            )
        except:
            pass  # Silent fail if win10toast not available
    
    def _send_log(self, notification: TradeNotification):
        """Send log notification"""
        config = self.channels['log'].config
        
        log_level = config.get('log_level', 'INFO')
        log_message = f"[{notification.event_type.upper()}] {notification.title}: {notification.message}"
        
        if config.get('include_data', True) and notification.data:
            log_message += f" | Data: {json.dumps(notification.data)}"
        
        if log_level == 'DEBUG':
            logger.debug(log_message)
        elif log_level == 'INFO':
            logger.info(log_message)
        elif log_level == 'WARNING':
            logger.warning(log_message)
        elif log_level == 'ERROR':
            logger.error(log_message)
    
    def start_processing(self):
        """Start background notification processing"""
        if self.processing:
            return
        
        self.processing = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        logger.info("Notification processing started")
    
    def stop_processing(self):
        """Stop background notification processing"""
        self.processing = False
        if self.process_thread:
            self.process_thread.join(timeout=5)
        logger.info("Notification processing stopped")
    
    def _process_loop(self):
        """Background processing loop"""
        while self.processing:
            try:
                # Process pending notifications
                pending_copy = list(self.pending_notifications.values())
                for notification in pending_copy:
                    if not notification.sent and notification.retry_count < 3:
                        self._process_notification(notification)
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in notification processing loop: {e}")
                time.sleep(5)
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification system statistics"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_notifications = [n for n in self.notification_history if n.timestamp > last_24h]
        
        stats = {
            'total_notifications': len(self.notification_history),
            'pending_notifications': len(self.pending_notifications),
            'last_24h_count': len(recent_notifications),
            'channels_enabled': {name: channel.enabled for name, channel in self.channels.items()},
            'last_notification': self.notification_history[-1].timestamp.isoformat() if self.notification_history else None,
            'processing': self.processing
        }
        
        # Count by event type in last 24h
        event_counts = {}
        for notification in recent_notifications:
            event_counts[notification.event_type] = event_counts.get(notification.event_type, 0) + 1
        
        stats['last_24h_by_type'] = event_counts
        
        return stats

# Global notification system instance
trade_notifications = TradeNotificationSystem()

# Convenience functions
def send_trade_opened(token_address: str, symbol: str, entry_price: float, quantity: float, trade_type: str = "buy"):
    """Send trade opened notification"""
    return trade_notifications.send_trade_opened(token_address, symbol, entry_price, quantity, trade_type)

def send_trade_closed(token_address: str, symbol: str, exit_price: float, pnl: float, pnl_percentage: float, duration_minutes: float):
    """Send trade closed notification"""
    return trade_notifications.send_trade_closed(token_address, symbol, exit_price, pnl, pnl_percentage, duration_minutes)

def send_alert(alert_type: str, message: str, severity: str = "warning"):
    """Send alert notification"""
    return trade_notifications.send_alert_notification(alert_type, message, severity)

def send_status_update(status: str, details: str = ""):
    """Send system status notification"""
    return trade_notifications.send_system_status(status, details)

def configure_email_notifications(smtp_server: str, smtp_port: int, username: str, password: str, recipient: str):
    """Configure email notifications"""
    trade_notifications.configure_email(smtp_server, smtp_port, username, password, recipient)

def configure_webhook_notifications(url: str, method: str = "POST", headers: Optional[Dict[str, str]] = None):
    """Configure webhook notifications"""
    trade_notifications.configure_webhook(url, method, headers)

def start_notifications():
    """Start notification processing"""
    trade_notifications.start_processing()

def stop_notifications():
    """Stop notification processing"""
    trade_notifications.stop_processing()

def get_notification_stats():
    """Get notification statistics"""
    return trade_notifications.get_notification_stats()