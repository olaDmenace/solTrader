import smtplib
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import json
import time
from dataclasses import dataclass

from ..config.settings import Settings

logger = logging.getLogger(__name__)

@dataclass
class EmailAlert:
    subject: str
    message: str
    alert_type: str
    priority: str
    data: Optional[Dict] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class EmailNotificationSystem:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.EMAIL_ENABLED and all([
            settings.EMAIL_USER,
            settings.EMAIL_PASSWORD,
            settings.EMAIL_TO
        ])
        
        # Rate limiting to prevent spam
        self.last_email_time = {}
        self.min_intervals = {
            'critical': 60,      # 1 minute for critical alerts
            'daily': 86400,      # 24 hours for daily reports
            'performance': 3600, # 1 hour for performance alerts
            'opportunity': 300   # 5 minutes for opportunities
        }
        
        # Email queue for batch sending
        self.email_queue = asyncio.Queue()
        self.is_running = False
        self.email_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'emails_sent': 0,
            'emails_failed': 0,
            'last_sent': None,
            'alerts_by_type': {
                'critical': 0,
                'daily': 0,
                'performance': 0,
                'opportunity': 0
            }
        }
        
        if self.enabled:
            logger.info("Email notification system initialized")
        else:
            logger.warning("Email notifications disabled - missing configuration")

    async def start(self):
        """Start the email notification system"""
        if not self.enabled:
            logger.warning("Cannot start email system - not properly configured")
            return
            
        if self.is_running:
            logger.warning("Email system already running")
            return
            
        self.is_running = True
        self.email_task = asyncio.create_task(self._email_worker())
        logger.info("Email notification system started")

    async def stop(self):
        """Stop the email notification system"""
        if not self.is_running:
            return
            
        logger.info("Stopping email notification system...")
        self.is_running = False
        
        if self.email_task:
            self.email_task.cancel()
            try:
                await self.email_task
            except asyncio.CancelledError:
                pass
        
        # Process any remaining emails in queue
        while not self.email_queue.empty():
            try:
                alert = await asyncio.wait_for(self.email_queue.get(), timeout=1.0)
                await self._send_email_sync(alert)
            except asyncio.TimeoutError:
                break
        
        logger.info("Email notification system stopped")

    async def _email_worker(self):
        """Background worker to process email queue"""
        while self.is_running:
            try:
                # Wait for email with timeout
                alert = await asyncio.wait_for(self.email_queue.get(), timeout=1.0)
                await self._send_email_sync(alert)
                await asyncio.sleep(1)  # Small delay between emails
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in email worker: {e}")
                await asyncio.sleep(5)

    async def send_alert(self, subject: str, message: str, alert_type: str = "info", 
                        priority: str = "normal", data: Dict = None, is_critical: bool = False):
        """Queue an email alert for sending"""
        if not self.enabled:
            logger.debug(f"Email disabled, skipping alert: {subject}")
            return
        
        # Override alert_type and priority if is_critical is True
        if is_critical:
            alert_type = "critical"
            priority = "critical"
            
        # Check rate limiting (bypass for critical alerts)
        if not is_critical and not self._can_send_email(alert_type):
            logger.debug(f"Rate limit hit for {alert_type}, skipping: {subject}")
            return
        
        alert = EmailAlert(
            subject=subject,
            message=message,
            alert_type=alert_type,
            priority=priority,
            data=data
        )
        
        await self.email_queue.put(alert)
        logger.debug(f"Queued email alert: {subject}")

    def _can_send_email(self, alert_type: str) -> bool:
        """Check if we can send an email of this type based on rate limiting"""
        now = time.time()
        last_sent = self.last_email_time.get(alert_type, 0)
        min_interval = self.min_intervals.get(alert_type, 300)
        
        return (now - last_sent) >= min_interval

    async def _send_email_sync(self, alert: EmailAlert):
        """Send email synchronously"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[SolTrader] {alert.subject}"
            msg['From'] = self.settings.EMAIL_USER
            msg['To'] = self.settings.EMAIL_TO
            
            # Create HTML content
            html_content = self._create_html_email(alert)
            text_content = self._create_text_email(alert)
            
            # Attach both HTML and text versions
            part1 = MIMEText(text_content, 'plain')
            part2 = MIMEText(html_content, 'html')
            
            msg.attach(part1)
            msg.attach(part2)
            
            # Send email
            with smtplib.SMTP(self.settings.EMAIL_SMTP_SERVER, self.settings.EMAIL_PORT) as server:
                server.starttls()
                server.login(self.settings.EMAIL_USER, self.settings.EMAIL_PASSWORD)
                text = msg.as_string()
                server.sendmail(self.settings.EMAIL_USER, self.settings.EMAIL_TO, text)
            
            # Update statistics
            self.stats['emails_sent'] += 1
            self.stats['last_sent'] = datetime.now()
            self.stats['alerts_by_type'][alert.alert_type] += 1
            self.last_email_time[alert.alert_type] = time.time()
            
            logger.info(f"Email sent successfully: {alert.subject}")
            
        except Exception as e:
            self.stats['emails_failed'] += 1
            logger.error(f"Failed to send email '{alert.subject}': {e}")

    def _create_html_email(self, alert: EmailAlert) -> str:
        """Create HTML email content"""
        priority_colors = {
            'critical': '#FF6B6B',
            'high': '#FFA726',
            'normal': '#66BB6A',
            'low': '#42A5F5'
        }
        
        color = priority_colors.get(alert.priority, '#666666')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>SolTrader Alert</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, {color}, {color}88); color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 30px; }}
                .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
                .alert-type {{ background-color: {color}22; color: {color}; padding: 5px 15px; border-radius: 20px; display: inline-block; font-size: 12px; font-weight: bold; text-transform: uppercase; }}
                .timestamp {{ color: #666; font-size: 14px; margin-top: 10px; }}
                .data-section {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px; }}
                .data-table {{ width: 100%; border-collapse: collapse; }}
                .data-table th, .data-table td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                .data-table th {{ background-color: #e9ecef; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 SolTrader Alert</h1>
                    <div class="alert-type">{alert.alert_type}</div>
                </div>
                <div class="content">
                    <h2>{alert.subject}</h2>
                    <p>{alert.message.replace(chr(10), '<br>')}</p>
                    <div class="timestamp">
                        📅 {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
                    </div>
        """
        
        # Add data section if available
        if alert.data:
            html += """
                    <div class="data-section">
                        <h3>📊 Details</h3>
                        <table class="data-table">
            """
            for key, value in alert.data.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, indent=2)
                html += f"<tr><th>{key.replace('_', ' ').title()}</th><td>{value}</td></tr>"
            html += """
                        </table>
                    </div>
            """
        
        html += """
                </div>
                <div class="footer">
                    <p>SolTrader Bot - Automated Solana Trading System</p>
                    <p>This is an automated message. Do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

    def _create_text_email(self, alert: EmailAlert) -> str:
        """Create plain text email content"""
        text = f"""
SolTrader Alert - {alert.alert_type.upper()}

{alert.subject}

{alert.message}

Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
        """
        
        if alert.data:
            text += "\n\nDetails:\n"
            for key, value in alert.data.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, indent=2)
                text += f"{key.replace('_', ' ').title()}: {value}\n"
        
        text += "\n---\nSolTrader Bot - Automated Solana Trading System\nThis is an automated message."
        
        return text

    # Specific alert methods
    async def send_critical_alert(self, subject: str, message: str, data: Dict = None):
        """Send critical system alert"""
        await self.send_alert(subject, message, "critical", "critical", data)

    async def send_daily_report(self, stats: Dict):
        """Send daily performance report"""
        subject = f"Daily Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Check if we're in paper trading mode
        is_paper_trading = stats.get('paper_trading_mode', True)
        mode_indicator = "📝 PAPER TRADING" if is_paper_trading else "💰 LIVE TRADING"
        
        message = f"""
Daily Trading Summary - {mode_indicator}

📊 PERFORMANCE METRICS
• Tokens Scanned: {stats.get('tokens_scanned', 0)}
• Tokens Approved: {stats.get('tokens_approved', 0)} ({stats.get('approval_rate', 0):.1f}% rate)
• Trades Executed: {stats.get('trades_executed', 0)}
• Win Rate: {stats.get('win_rate', 0):.1f}%
• Total P&L: ${stats.get('total_pnl', 0):.2f}{' (simulated)' if is_paper_trading else ''}

💰 TRADING STATS
• Best Trade: +{stats.get('best_trade', 0):.1f}%
• Worst Trade: {stats.get('worst_trade', 0):.1f}%
• Average Hold Time: {stats.get('avg_hold_time', 0):.1f} minutes
• Gas Fees: ${stats.get('gas_fees', 0):.2f}{' (simulated)' if is_paper_trading else ''}

🔍 DISCOVERY ANALYTICS
• API Requests Used: {stats.get('api_requests_used', 0)}/333
• High Momentum Bypasses: {stats.get('high_momentum_bypasses', 0)}
• Source Effectiveness: {stats.get('source_breakdown', {})}

📈 PORTFOLIO STATUS
• Current Value: ${stats.get('portfolio_value', 0):.2f}{' (paper)' if is_paper_trading else ''}
• Active Positions: {stats.get('active_positions', 0)}
• Available Balance: ${stats.get('available_balance', 0):.2f}{' (paper)' if is_paper_trading else ''}

{('🧪 PAPER TRADING NOTES:' + chr(10) + '• All trades are simulated - no real funds at risk' + chr(10) + '• Performance metrics indicate strategy effectiveness' + chr(10) + '• Ready for live trading when you decide to switch modes') if is_paper_trading else ('⚠️ LIVE TRADING ACTIVE:' + chr(10) + '• Real funds at risk - monitor positions closely' + chr(10) + '• All P&L represents actual gains/losses' + chr(10) + '• Ensure risk management settings are appropriate')}
        """
        
        await self.send_alert(subject, message, "daily", "normal", stats)

    async def send_opportunity_alert(self, token_symbol: str, gain_percent: float, data: Dict = None):
        """Send high-profit opportunity alert"""
        if gain_percent < 100:  # Only for >100% gains
            return
            
        subject = f"🚀 High Gain Opportunity: {token_symbol} (+{gain_percent:.1f}%)"
        message = f"""
Explosive token movement detected!

Token: {token_symbol}
Gain: +{gain_percent:.1f}%
Status: {'CAPTURED' if data and data.get('traded') else 'MISSED'}

This token shows exceptional momentum and may represent a significant opportunity.
        """
        
        await self.send_alert(subject, message, "opportunity", "high", data)

    async def send_performance_alert(self, alert_type: str, message: str, data: Dict = None):
        """Send performance-related alert"""
        subjects = {
            'risk_breach': '⚠️ Risk Limit Breach',
            'api_limit': '🔄 API Rate Limit Warning',
            'system_health': '🔧 System Health Alert',
            'position_stuck': '🔒 Position Management Alert'
        }
        
        subject = subjects.get(alert_type, f"Performance Alert: {alert_type}")
        await self.send_alert(subject, message, "performance", "high", data)

    async def send_weekly_report(self, stats: Dict):
        """Send comprehensive weekly report"""
        subject = f"Weekly Performance Report - {datetime.now().strftime('%Y-W%U')}"
        
        message = f"""
Weekly Trading Performance Summary:

📊 WEEKLY OVERVIEW
• Trading Days: {stats.get('trading_days', 0)}
• Total Trades: {stats.get('total_trades', 0)}
• Cumulative P&L: ${stats.get('cumulative_pnl', 0):.2f} ({stats.get('return_percentage', 0):.1f}% return)
• Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}
• Maximum Drawdown: {stats.get('max_drawdown', 0):.1f}%

🎯 TRADING PATTERNS
• Average Trades/Day: {stats.get('avg_trades_per_day', 0):.1f}
• Longest Winning Streak: {stats.get('longest_win_streak', 0)}
• Longest Losing Streak: {stats.get('longest_loss_streak', 0)}
• Most Profitable Category: {stats.get('best_category', 'N/A')}
• Best Trading Hour: {stats.get('best_hour', 'N/A')}

🔍 DISCOVERY INSIGHTS
• Tokens Discovered: {stats.get('total_discovered', 0)}
• Discovery Sources: {stats.get('source_breakdown', {})}
• Average Discovery Age: {stats.get('avg_discovery_age', 0):.1f} minutes
• Filter Optimization Impact: {stats.get('filter_impact', 'N/A')}

📈 PERFORMANCE TRENDS
• Week-over-week Growth: {stats.get('wow_growth', 0):.1f}%
• Risk-adjusted Returns: {stats.get('risk_adjusted_return', 0):.2f}
• Portfolio Efficiency: {stats.get('efficiency_score', 0):.1f}/10
        """
        
        await self.send_alert(subject, message, "weekly", "normal", stats)

    def get_stats(self) -> Dict:
        """Get email system statistics"""
        return {
            'enabled': self.enabled,
            'emails_sent': self.stats['emails_sent'],
            'emails_failed': self.stats['emails_failed'],
            'success_rate': (self.stats['emails_sent'] / (self.stats['emails_sent'] + self.stats['emails_failed']) * 100) 
                          if (self.stats['emails_sent'] + self.stats['emails_failed']) > 0 else 0,
            'last_sent': self.stats['last_sent'].isoformat() if self.stats['last_sent'] else None,
            'queue_size': self.email_queue.qsize(),
            'alerts_by_type': self.stats['alerts_by_type'].copy(),
            'rate_limits': {
                alert_type: {
                    'min_interval': interval,
                    'last_sent': self.last_email_time.get(alert_type, 0),
                    'can_send_now': self._can_send_email(alert_type)
                }
                for alert_type, interval in self.min_intervals.items()
            }
        }

    async def test_email(self):
        """Send a test email to verify configuration"""
        if not self.enabled:
            return False, "Email system not enabled or configured"
        
        try:
            await self.send_alert(
                subject="Test Email - System Check",
                message="This is a test email to verify that the SolTrader email notification system is working correctly.",
                alert_type="info",
                priority="low",
                data={
                    'test_timestamp': datetime.now().isoformat(),
                    'system_status': 'operational',
                    'configuration': 'valid'
                }
            )
            return True, "Test email queued successfully"
        except Exception as e:
            return False, f"Failed to queue test email: {e}"