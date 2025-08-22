#!/usr/bin/env python3
"""
Production Alerting System for SolTrader
Sends email and SMS notifications for critical bot failures
"""
import smtplib
import logging
import asyncio
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import json

# For SMS (multiple providers supported)
import aiohttp
import requests

logger = logging.getLogger(__name__)

@dataclass
class AlertConfig:
    """Configuration for alert destinations and preferences"""
    # Email settings
    email_enabled: bool = True
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""  # Use app password for Gmail
    from_email: str = ""
    to_emails: List[str] = None
    
    # SMS settings
    sms_enabled: bool = False
    sms_provider: str = "twilio"  # twilio, textbelt, or custom
    
    # Twilio settings
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    to_phone_numbers: List[str] = None
    
    # TextBelt settings (simple API)
    textbelt_api_key: str = ""  # Get from textbelt.com
    
    # Alert thresholds and timing
    min_alert_interval: int = 300  # 5 minutes between same alerts
    max_alerts_per_hour: int = 10
    quiet_hours_start: int = 23  # 11 PM
    quiet_hours_end: int = 7     # 7 AM
    respect_quiet_hours: bool = True
    
    def __post_init__(self):
        if self.to_emails is None:
            self.to_emails = []
        if self.to_phone_numbers is None:
            self.to_phone_numbers = []

@dataclass  
class AlertMessage:
    """Represents an alert message to be sent"""
    title: str
    message: str
    severity: str
    component: str
    timestamp: datetime
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

class ProductionAlerter:
    """Production alerting system for critical bot failures"""
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or self._load_config_from_env()
        
        # Alert tracking
        self.sent_alerts: List[Dict[str, Any]] = []
        self.alert_counts = {"hour": 0, "last_hour_reset": time.time()}
        
        # Rate limiting
        self.last_alert_times: Dict[str, float] = {}
        
        # Status tracking
        self.system_status = {
            "email_working": None,
            "sms_working": None,
            "last_test": None
        }
    
    def _load_config_from_env(self) -> AlertConfig:
        """Load alert configuration from environment variables"""
        return AlertConfig(
            # Email from env
            email_enabled=os.getenv('ALERT_EMAIL_ENABLED', 'true').lower() == 'true',
            smtp_host=os.getenv('ALERT_SMTP_HOST', 'smtp.gmail.com'),
            smtp_port=int(os.getenv('ALERT_SMTP_PORT', '587')),
            smtp_username=os.getenv('ALERT_SMTP_USERNAME', ''),
            smtp_password=os.getenv('ALERT_SMTP_PASSWORD', ''),
            from_email=os.getenv('ALERT_FROM_EMAIL', ''),
            to_emails=os.getenv('ALERT_TO_EMAILS', '').split(',') if os.getenv('ALERT_TO_EMAILS') else [],
            
            # SMS from env
            sms_enabled=os.getenv('ALERT_SMS_ENABLED', 'false').lower() == 'true',
            sms_provider=os.getenv('ALERT_SMS_PROVIDER', 'twilio'),
            twilio_account_sid=os.getenv('ALERT_TWILIO_SID', ''),
            twilio_auth_token=os.getenv('ALERT_TWILIO_TOKEN', ''),
            twilio_from_number=os.getenv('ALERT_TWILIO_FROM', ''),
            to_phone_numbers=os.getenv('ALERT_TO_PHONES', '').split(',') if os.getenv('ALERT_TO_PHONES') else [],
            textbelt_api_key=os.getenv('ALERT_TEXTBELT_KEY', ''),
            
            # Timing from env
            min_alert_interval=int(os.getenv('ALERT_MIN_INTERVAL', '300')),
            max_alerts_per_hour=int(os.getenv('ALERT_MAX_PER_HOUR', '10')),
            quiet_hours_start=int(os.getenv('ALERT_QUIET_START', '23')),
            quiet_hours_end=int(os.getenv('ALERT_QUIET_END', '7')),
            respect_quiet_hours=os.getenv('ALERT_RESPECT_QUIET', 'true').lower() == 'true'
        )
    
    def _should_send_alert(self, component: str, severity: str) -> bool:
        """Check if we should send an alert based on rate limiting and quiet hours"""
        
        # Check hourly rate limit
        current_time = time.time()
        if current_time - self.alert_counts["last_hour_reset"] > 3600:
            self.alert_counts["hour"] = 0
            self.alert_counts["last_hour_reset"] = current_time
        
        if self.alert_counts["hour"] >= self.config.max_alerts_per_hour:
            logger.warning(f"Alert rate limit reached ({self.config.max_alerts_per_hour}/hour)")
            return False
        
        # Check minimum interval between same component alerts
        alert_key = f"{component}:{severity}"
        last_alert_time = self.last_alert_times.get(alert_key, 0)
        if current_time - last_alert_time < self.config.min_alert_interval:
            logger.debug(f"Alert suppressed due to min interval: {alert_key}")
            return False
        
        # Check quiet hours
        if self.config.respect_quiet_hours:
            current_hour = datetime.now().hour
            if self.config.quiet_hours_start < self.config.quiet_hours_end:
                # Same day quiet hours (e.g., 23:00 to 07:00)
                is_quiet = (current_hour >= self.config.quiet_hours_start or 
                           current_hour < self.config.quiet_hours_end)
            else:
                # Overnight quiet hours (e.g., 23:00 to 07:00)  
                is_quiet = (current_hour >= self.config.quiet_hours_start or
                           current_hour < self.config.quiet_hours_end)
            
            if is_quiet and severity.lower() not in ['critical', 'high']:
                logger.info(f"Alert suppressed due to quiet hours: {alert_key}")
                return False
        
        return True
    
    async def send_email_alert(self, alert: AlertMessage) -> bool:
        """Send email alert"""
        if not self.config.email_enabled or not self.config.to_emails:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.from_email
            msg['To'] = ', '.join(self.config.to_emails)
            msg['Subject'] = f"SolTrader Alert: {alert.title}"
            
            # Create email body
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port)
            server.starttls()
            server.login(self.config.smtp_username, self.config.smtp_password)
            
            text = msg.as_string()
            server.sendmail(self.config.from_email, self.config.to_emails, text)
            server.quit()
            
            logger.info(f"Email alert sent successfully to {len(self.config.to_emails)} recipients")
            self.system_status["email_working"] = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            self.system_status["email_working"] = False
            return False
    
    async def send_sms_alert(self, alert: AlertMessage) -> bool:
        """Send SMS alert using configured provider"""
        if not self.config.sms_enabled or not self.config.to_phone_numbers:
            return False
        
        try:
            if self.config.sms_provider == "twilio":
                return await self._send_twilio_sms(alert)
            elif self.config.sms_provider == "textbelt":
                return await self._send_textbelt_sms(alert)
            else:
                logger.error(f"Unknown SMS provider: {self.config.sms_provider}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")
            self.system_status["sms_working"] = False
            return False
    
    async def _send_twilio_sms(self, alert: AlertMessage) -> bool:
        """Send SMS via Twilio"""
        from twilio.rest import Client
        
        if not all([self.config.twilio_account_sid, self.config.twilio_auth_token, self.config.twilio_from_number]):
            logger.error("Twilio credentials not configured")
            return False
        
        try:
            client = Client(self.config.twilio_account_sid, self.config.twilio_auth_token)
            
            sms_body = self._format_sms_body(alert)
            success_count = 0
            
            for phone_number in self.config.to_phone_numbers:
                try:
                    message = client.messages.create(
                        body=sms_body,
                        from_=self.config.twilio_from_number,
                        to=phone_number
                    )
                    logger.info(f"SMS sent to {phone_number}: {message.sid}")
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to send SMS to {phone_number}: {e}")
            
            self.system_status["sms_working"] = success_count > 0
            return success_count > 0
            
        except ImportError:
            logger.error("Twilio library not installed. Install with: pip install twilio")
            return False
        except Exception as e:
            logger.error(f"Twilio SMS failed: {e}")
            return False
    
    async def _send_textbelt_sms(self, alert: AlertMessage) -> bool:
        """Send SMS via TextBelt (simple HTTP API)"""
        if not self.config.textbelt_api_key:
            logger.error("TextBelt API key not configured")
            return False
        
        sms_body = self._format_sms_body(alert)
        success_count = 0
        
        for phone_number in self.config.to_phone_numbers:
            try:
                response = requests.post('https://textbelt.com/text', {
                    'phone': phone_number,
                    'message': sms_body,
                    'key': self.config.textbelt_api_key,
                })
                
                result = response.json()
                if result.get('success'):
                    logger.info(f"TextBelt SMS sent to {phone_number}")
                    success_count += 1
                else:
                    logger.error(f"TextBelt SMS failed for {phone_number}: {result.get('error')}")
                    
            except Exception as e:
                logger.error(f"TextBelt SMS error for {phone_number}: {e}")
        
        self.system_status["sms_working"] = success_count > 0
        return success_count > 0
    
    def _format_email_body(self, alert: AlertMessage) -> str:
        """Format email alert body with HTML styling"""
        severity_colors = {
            'critical': '#dc3545',
            'high': '#fd7e14', 
            'medium': '#ffc107',
            'low': '#28a745'
        }
        
        color = severity_colors.get(alert.severity.lower(), '#6c757d')
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <div style="border-left: 5px solid {color}; padding: 15px; background-color: #f8f9fa;">
                <h2 style="color: {color}; margin-top: 0;">SolTrader Alert</h2>
                <p><strong>Component:</strong> {alert.component}</p>
                <p><strong>Severity:</strong> <span style="color: {color}; font-weight: bold;">{alert.severity.upper()}</span></p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Message:</strong></p>
                <div style="background-color: white; padding: 10px; border-radius: 4px; border: 1px solid #dee2e6;">
                    {alert.message}
                </div>
        """
        
        if alert.details:
            html_body += """
                <p><strong>Details:</strong></p>
                <div style="background-color: #f1f3f4; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 12px;">
            """
            for key, value in alert.details.items():
                html_body += f"<br><strong>{key}:</strong> {value}"
            html_body += "</div>"
        
        html_body += """
            </div>
            <hr style="margin: 20px 0;">
            <p style="color: #6c757d; font-size: 12px;">
                This alert was generated by your SolTrader production monitoring system.<br>
                To disable alerts, update your environment variables or contact support.
            </p>
        </body>
        </html>
        """
        
        return html_body
    
    def _format_sms_body(self, alert: AlertMessage) -> str:
        """Format SMS alert body (keep it short for SMS)"""
        return (f"SolTrader ALERT\n"
                f"{alert.severity.upper()}: {alert.component}\n"
                f"{alert.message[:100]}{'...' if len(alert.message) > 100 else ''}\n"
                f"Time: {alert.timestamp.strftime('%H:%M:%S')}")
    
    async def send_alert(self, title: str, message: str, severity: str, component: str, details: Dict[str, Any] = None) -> bool:
        """Send alert via all configured channels"""
        
        # Check if we should send this alert
        if not self._should_send_alert(component, severity):
            return False
        
        # Create alert message
        alert = AlertMessage(
            title=title,
            message=message,
            severity=severity,
            component=component,
            timestamp=datetime.now(),
            details=details or {}
        )
        
        # Update tracking
        alert_key = f"{component}:{severity}"
        self.last_alert_times[alert_key] = time.time()
        self.alert_counts["hour"] += 1
        
        # Send via all channels
        email_sent = False
        sms_sent = False
        
        if self.config.email_enabled:
            email_sent = await self.send_email_alert(alert)
        
        if self.config.sms_enabled:
            sms_sent = await self.send_sms_alert(alert)
        
        # Record the alert
        self.sent_alerts.append({
            'timestamp': alert.timestamp.isoformat(),
            'component': component,
            'severity': severity,
            'title': title,
            'message': message,
            'email_sent': email_sent,
            'sms_sent': sms_sent
        })
        
        # Keep only last 100 alerts in memory
        if len(self.sent_alerts) > 100:
            self.sent_alerts = self.sent_alerts[-100:]
        
        success = email_sent or sms_sent
        if success:
            logger.info(f"Alert sent successfully: {title} (Email: {email_sent}, SMS: {sms_sent})")
        else:
            logger.error(f"Failed to send alert: {title}")
        
        return success
    
    async def test_alert_system(self) -> Dict[str, bool]:
        """Test the alert system to ensure it's working"""
        logger.info("Testing alert system...")
        
        test_alert = AlertMessage(
            title="Alert System Test",
            message="This is a test message to verify your SolTrader alerting system is working correctly.",
            severity="medium",
            component="alert_system",
            timestamp=datetime.now(),
            details={
                "test_type": "system_test",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        results = {}
        
        # Test email
        if self.config.email_enabled:
            results['email'] = await self.send_email_alert(test_alert)
        else:
            results['email'] = None
        
        # Test SMS  
        if self.config.sms_enabled:
            results['sms'] = await self.send_sms_alert(test_alert)
        else:
            results['sms'] = None
        
        self.system_status["last_test"] = datetime.now().isoformat()
        
        return results
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alerting system statistics"""
        recent_alerts = [a for a in self.sent_alerts if 
                        datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=24)]
        
        return {
            "total_alerts_sent": len(self.sent_alerts),
            "alerts_last_24h": len(recent_alerts),
            "current_hour_count": self.alert_counts["hour"],
            "max_per_hour": self.config.max_alerts_per_hour,
            "email_enabled": self.config.email_enabled,
            "sms_enabled": self.config.sms_enabled,
            "system_status": self.system_status,
            "last_alert_times": {k: datetime.fromtimestamp(v).isoformat() 
                               for k, v in self.last_alert_times.items()}
        }

# Global alerter instance
production_alerter = ProductionAlerter()

async def send_critical_alert(component: str, message: str, details: Dict[str, Any] = None):
    """Quick function to send critical alerts"""
    await production_alerter.send_alert(
        title=f"CRITICAL: {component}",
        message=message,
        severity="critical", 
        component=component,
        details=details
    )

async def send_error_alert(component: str, message: str, details: Dict[str, Any] = None):
    """Quick function to send error alerts"""
    await production_alerter.send_alert(
        title=f"ERROR: {component}",
        message=message,
        severity="high",
        component=component, 
        details=details
    )

async def test_alerting():
    """Test the alerting system"""
    return await production_alerter.test_alert_system()