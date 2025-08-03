"""
Health Monitoring System for SolTrader Bot
Monitors bot performance, API health, and system resources with auto-recovery capabilities
"""

import asyncio
import json
import logging
import os
import psutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class RecoveryAction(Enum):
    """Recovery action types"""
    NONE = "none"
    SOFT_RECOVERY = "soft_recovery"
    MEDIUM_RECOVERY = "medium_recovery"
    HARD_RECOVERY = "hard_recovery"
    MANUAL_INTERVENTION = "manual_intervention"

@dataclass
class HealthMetric:
    """Individual health metric tracking"""
    name: str
    current_value: Any
    threshold_warning: Any
    threshold_critical: Any
    status: HealthStatus = HealthStatus.UNKNOWN
    last_updated: datetime = field(default_factory=datetime.now)
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, value: Any) -> HealthStatus:
        """Update metric value and determine status"""
        self.current_value = value
        self.last_updated = datetime.now()
        self.history.append((datetime.now(), value))
        
        # Determine status based on thresholds
        if self._is_critical(value):
            self.status = HealthStatus.CRITICAL
        elif self._is_warning(value):
            self.status = HealthStatus.WARNING
        else:
            self.status = HealthStatus.HEALTHY
            
        return self.status
    
    def _is_critical(self, value: Any) -> bool:
        """Check if value is at critical level"""
        if isinstance(value, (int, float)) and isinstance(self.threshold_critical, (int, float)):
            if self.name in ['cpu_usage', 'memory_usage', 'disk_usage', 'api_error_rate']:
                return value >= self.threshold_critical
            elif self.name in ['token_discovery_rate', 'approval_rate', 'trade_execution_rate']:
                return value <= self.threshold_critical
        return False
    
    def _is_warning(self, value: Any) -> bool:
        """Check if value is at warning level"""
        if isinstance(value, (int, float)) and isinstance(self.threshold_warning, (int, float)):
            if self.name in ['cpu_usage', 'memory_usage', 'disk_usage', 'api_error_rate']:
                return value >= self.threshold_warning
            elif self.name in ['token_discovery_rate', 'approval_rate', 'trade_execution_rate']:
                return value <= self.threshold_warning
        return False

@dataclass
class HealthReport:
    """Complete health report"""
    timestamp: datetime
    overall_status: HealthStatus
    metrics: Dict[str, HealthMetric]
    issues: List[str]
    recovery_actions: List[RecoveryAction]
    uptime: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_status': self.overall_status.value,
            'metrics': {
                name: {
                    'current_value': metric.current_value,
                    'status': metric.status.value,
                    'last_updated': metric.last_updated.isoformat()
                } for name, metric in self.metrics.items()
            },
            'issues': self.issues,
            'recovery_actions': [action.value for action in self.recovery_actions],
            'uptime': self.uptime
        }

class HealthMonitor:
    """Comprehensive health monitoring system"""
    
    def __init__(self, bot_instance=None, settings=None):
        """Initialize health monitor"""
        self.bot = bot_instance
        self.settings = settings
        self.start_time = datetime.now()
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Health metrics
        self.metrics: Dict[str, HealthMetric] = {}
        self.last_report: Optional[HealthReport] = None
        self.recovery_attempts = deque(maxlen=50)
        
        # Configuration
        self.config = self._load_config()
        self._initialize_metrics()
        
        # State tracking
        self.last_log_check = datetime.now()
        self.consecutive_issues = 0
        self.last_recovery_time = None
        
        logger.info("Health monitoring system initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            'monitoring_interval': 60,  # seconds
            'log_file_path': 'logs/trading.log',
            'dashboard_file': 'bot_data.json',
            'max_recovery_attempts_per_hour': 5,
            'recovery_cooldown_minutes': 5,
            'thresholds': {
                'token_discovery_rate': {'warning': 100, 'critical': 50},
                'approval_rate': {'warning': 20, 'critical': 15},
                'api_error_rate': {'warning': 10, 'critical': 20},
                'trade_execution_rate': {'warning': 5, 'critical': 0},
                'cpu_usage': {'warning': 80, 'critical': 95},
                'memory_usage': {'warning': 80, 'critical': 90},
                'disk_usage': {'warning': 85, 'critical': 95},
                'api_response_time': {'warning': 10, 'critical': 30}
            }
        }
        
        # Load custom config if available
        config_file = Path('health_monitor_config.json')
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                    default_config.update(custom_config)
            except Exception as e:
                logger.warning(f"Failed to load custom config: {e}")
        
        return default_config
    
    def _initialize_metrics(self):
        """Initialize health metrics"""
        thresholds = self.config['thresholds']
        
        for metric_name, threshold in thresholds.items():
            self.metrics[metric_name] = HealthMetric(
                name=metric_name,
                current_value=0,
                threshold_warning=threshold['warning'],
                threshold_critical=threshold['critical']
            )
    
    async def start_monitoring(self):
        """Start the health monitoring loop"""
        if self.is_monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop the health monitoring loop"""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect all health metrics
                await self._collect_metrics()
                
                # Generate health report
                report = self._generate_health_report()
                self.last_report = report
                
                # Check for issues and trigger recovery if needed
                if report.overall_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                    await self._handle_health_issues(report)
                
                # Save health report
                await self._save_health_report(report)
                
                # Wait for next check interval
                await asyncio.sleep(self.config['monitoring_interval'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def _collect_metrics(self):
        """Collect all health metrics"""
        # Performance metrics
        await self._collect_performance_metrics()
        
        # API health metrics
        await self._collect_api_health_metrics()
        
        # System resource metrics
        await self._collect_system_metrics()
        
        # Trade execution metrics
        await self._collect_trade_metrics()
        
        # Log activity metrics
        await self._collect_log_metrics()
    
    async def _collect_performance_metrics(self):
        """Collect bot performance metrics"""
        try:
            # Check dashboard data for recent performance
            dashboard_file = self.config['dashboard_file']
            if os.path.exists(dashboard_file):
                with open(dashboard_file, 'r') as f:
                    data = json.load(f)
                
                # Token discovery rate
                activity = data.get('activity', [])
                recent_scans = [
                    a for a in activity 
                    if a.get('type') == 'scan_completed' and
                    datetime.fromisoformat(a.get('timestamp', '2024-01-01')) > datetime.now() - timedelta(hours=1)
                ]
                
                if recent_scans:
                    avg_tokens = sum(scan.get('data', {}).get('tokens_found', 0) for scan in recent_scans) / len(recent_scans)
                    self.metrics['token_discovery_rate'].update(avg_tokens)
                else:
                    self.metrics['token_discovery_rate'].update(0)
                
                # Trade execution rate
                trades = data.get('trades', [])
                recent_trades = [
                    t for t in trades 
                    if datetime.fromisoformat(t.get('timestamp', '2024-01-01')) > datetime.now() - timedelta(hours=1)
                ]
                self.metrics['trade_execution_rate'].update(len(recent_trades))
                
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    async def _collect_api_health_metrics(self):
        """Collect API health metrics"""
        try:
            # Check API response times and error rates
            if self.bot and hasattr(self.bot, 'enhanced_scanner'):
                scanner = self.bot.enhanced_scanner
                
                # Get recent API stats from scanner
                if hasattr(scanner, 'daily_stats'):
                    stats = scanner.daily_stats
                    error_rate = stats.get('api_errors', 0) / max(stats.get('api_requests_used', 1), 1) * 100
                    self.metrics['api_error_rate'].update(error_rate)
                
                # Test API connectivity
                start_time = time.time()
                try:
                    if hasattr(scanner, 'solana_tracker'):
                        # Simple connectivity test
                        await asyncio.wait_for(
                            scanner.solana_tracker._test_connection(),
                            timeout=10
                        )
                    response_time = time.time() - start_time
                    self.metrics['api_response_time'].update(response_time)
                except asyncio.TimeoutError:
                    self.metrics['api_response_time'].update(30)  # Max timeout
                except Exception:
                    self.metrics['api_response_time'].update(30)
                    
        except Exception as e:
            logger.error(f"Error collecting API health metrics: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['cpu_usage'].update(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics['memory_usage'].update(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics['disk_usage'].update(disk_percent)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_trade_metrics(self):
        """Collect trading-specific metrics"""
        try:
            if self.bot and hasattr(self.bot, 'strategy'):
                strategy = self.bot.strategy
                
                # Calculate approval rate if scanner available
                if hasattr(self.bot, 'enhanced_scanner'):
                    scanner = self.bot.enhanced_scanner
                    if hasattr(scanner, 'daily_stats'):
                        stats = scanner.daily_stats
                        approval_rate = stats.get('approval_rate', 0)
                        self.metrics['approval_rate'].update(approval_rate)
                
        except Exception as e:
            logger.error(f"Error collecting trade metrics: {e}")
    
    async def _collect_log_metrics(self):
        """Collect log activity metrics"""
        try:
            log_file = Path(self.config['log_file_path'])
            if log_file.exists():
                # Check for recent log entries
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(minutes=10)
                
                recent_activity = False
                try:
                    # Read last few lines of log file
                    with open(log_file, 'r') as f:
                        lines = f.readlines()[-50:]  # Last 50 lines
                    
                    for line in lines:
                        if len(line) > 19:  # Minimum length for timestamp
                            try:
                                timestamp_str = line[:19]
                                log_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                                if log_time > cutoff_time:
                                    recent_activity = True
                                    break
                            except ValueError:
                                continue
                                
                except Exception as e:
                    logger.warning(f"Error reading log file: {e}")
                
                # Update metric based on recent activity
                self.metrics['log_activity'] = HealthMetric(
                    name='log_activity',
                    current_value=recent_activity,
                    threshold_warning=False,
                    threshold_critical=False
                )
                self.metrics['log_activity'].update(recent_activity)
                
        except Exception as e:
            logger.error(f"Error collecting log metrics: {e}")
    
    def _generate_health_report(self) -> HealthReport:
        """Generate comprehensive health report"""
        current_time = datetime.now()
        uptime = (current_time - self.start_time).total_seconds()
        
        # Determine overall status
        critical_count = sum(1 for m in self.metrics.values() if m.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for m in self.metrics.values() if m.status == HealthStatus.WARNING)
        
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_count > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Collect issues
        issues = []
        recovery_actions = []
        
        for metric in self.metrics.values():
            if metric.status == HealthStatus.CRITICAL:
                issues.append(f"CRITICAL: {metric.name} = {metric.current_value} (threshold: {metric.threshold_critical})")
                recovery_actions.append(self._determine_recovery_action(metric))
            elif metric.status == HealthStatus.WARNING:
                issues.append(f"WARNING: {metric.name} = {metric.current_value} (threshold: {metric.threshold_warning})")
        
        return HealthReport(
            timestamp=current_time,
            overall_status=overall_status,
            metrics=self.metrics.copy(),
            issues=issues,
            recovery_actions=list(set(recovery_actions)),
            uptime=uptime
        )
    
    def _determine_recovery_action(self, metric: HealthMetric) -> RecoveryAction:
        """Determine appropriate recovery action for a critical metric"""
        if metric.name in ['api_error_rate', 'api_response_time']:
            return RecoveryAction.SOFT_RECOVERY
        elif metric.name in ['token_discovery_rate', 'approval_rate']:
            return RecoveryAction.MEDIUM_RECOVERY
        elif metric.name in ['trade_execution_rate']:
            return RecoveryAction.MEDIUM_RECOVERY
        elif metric.name in ['cpu_usage', 'memory_usage']:
            return RecoveryAction.HARD_RECOVERY
        elif metric.name == 'disk_usage':
            return RecoveryAction.MANUAL_INTERVENTION
        else:
            return RecoveryAction.SOFT_RECOVERY
    
    async def _handle_health_issues(self, report: HealthReport):
        """Handle health issues with appropriate recovery actions"""
        if not report.recovery_actions:
            return
        
        # Check if we're in cooldown period
        if self.last_recovery_time:
            cooldown_minutes = self.config['recovery_cooldown_minutes']
            if datetime.now() < self.last_recovery_time + timedelta(minutes=cooldown_minutes):
                logger.info("In recovery cooldown period, skipping recovery actions")
                return
        
        # Check recovery attempt limits
        recent_attempts = [
            attempt for attempt in self.recovery_attempts
            if attempt['timestamp'] > datetime.now() - timedelta(hours=1)
        ]
        
        if len(recent_attempts) >= self.config['max_recovery_attempts_per_hour']:
            logger.warning("Maximum recovery attempts per hour reached")
            await self._send_alert(
                "Recovery Limit Reached",
                f"Maximum recovery attempts ({self.config['max_recovery_attempts_per_hour']}) reached in the last hour. Manual intervention may be required.",
                is_critical=True
            )
            return
        
        # Execute recovery actions
        for action in report.recovery_actions:
            try:
                success = await self._execute_recovery_action(action, report)
                
                # Record recovery attempt
                self.recovery_attempts.append({
                    'timestamp': datetime.now(),
                    'action': action.value,
                    'success': success,
                    'issues': report.issues
                })
                
                if success:
                    self.last_recovery_time = datetime.now()
                    self.consecutive_issues = 0
                    logger.info(f"Recovery action {action.value} completed successfully")
                    await self._send_alert(
                        "Auto-Recovery Successful",
                        f"Successfully executed {action.value} to resolve health issues.",
                        is_critical=False
                    )
                    break  # Stop after first successful recovery
                else:
                    logger.error(f"Recovery action {action.value} failed")
                    
            except Exception as e:
                logger.error(f"Error executing recovery action {action.value}: {e}")
        
        self.consecutive_issues += 1
        
        # Escalate if too many consecutive issues
        if self.consecutive_issues >= 3:
            await self._send_alert(
                "Critical: Multiple Recovery Failures",
                f"Health issues persist after {self.consecutive_issues} recovery attempts. Manual intervention required.",
                is_critical=True
            )
    
    async def _execute_recovery_action(self, action: RecoveryAction, report: HealthReport) -> bool:
        """Execute specific recovery action"""
        logger.info(f"Executing recovery action: {action.value}")
        
        try:
            if action == RecoveryAction.SOFT_RECOVERY:
                return await self._soft_recovery()
            elif action == RecoveryAction.MEDIUM_RECOVERY:
                return await self._medium_recovery()
            elif action == RecoveryAction.HARD_RECOVERY:
                return await self._hard_recovery()
            elif action == RecoveryAction.MANUAL_INTERVENTION:
                await self._request_manual_intervention(report)
                return False
            else:
                return False
                
        except Exception as e:
            logger.error(f"Recovery action {action.value} failed with error: {e}")
            return False
    
    async def _soft_recovery(self) -> bool:
        """Soft recovery: Connection resets, cache clearing"""
        try:
            logger.info("Executing soft recovery...")
            
            if self.bot and hasattr(self.bot, 'enhanced_scanner'):
                scanner = self.bot.enhanced_scanner
                
                # Reset API connections
                if hasattr(scanner, 'solana_tracker'):
                    await scanner.solana_tracker.close_session()
                    await asyncio.sleep(2)
                    await scanner.solana_tracker.start_session()
                
                # Clear any cached data
                if hasattr(scanner, 'discovered_tokens'):
                    scanner.discovered_tokens.clear()
                
                logger.info("Soft recovery completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Soft recovery failed: {e}")
            return False
        
        return False
    
    async def _medium_recovery(self) -> bool:
        """Medium recovery: Service restart with monitoring"""
        try:
            logger.info("Executing medium recovery...")
            
            # Stop and restart key components
            if self.bot:
                # Stop scanner
                if hasattr(self.bot, 'enhanced_scanner'):
                    await self.bot.enhanced_scanner.stop()
                    await asyncio.sleep(3)
                    await self.bot.enhanced_scanner.start()
                
                # Reset strategy if available
                if hasattr(self.bot, 'strategy'):
                    if hasattr(self.bot.strategy, 'state'):
                        # Clear any stuck state
                        self.bot.strategy.state.pending_orders.clear()
                
                logger.info("Medium recovery completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Medium recovery failed: {e}")
            return False
        
        return False
    
    async def _hard_recovery(self) -> bool:
        """Hard recovery: Full system restart if needed"""
        try:
            logger.info("Executing hard recovery...")
            
            # This would typically restart the entire bot process
            # For now, we'll do a comprehensive reset
            
            if self.bot:
                # Stop all components
                try:
                    await self.bot.stop()
                    await asyncio.sleep(5)
                    await self.bot.startup()
                    
                    logger.info("Hard recovery completed successfully")
                    return True
                except Exception as e:
                    logger.error(f"Hard recovery restart failed: {e}")
                    return False
            
        except Exception as e:
            logger.error(f"Hard recovery failed: {e}")
            return False
        
        return False
    
    async def _request_manual_intervention(self, report: HealthReport):
        """Request manual intervention for critical issues"""
        logger.critical("Manual intervention required for critical health issues")
        
        await self._send_alert(
            "CRITICAL: Manual Intervention Required",
            f"Critical health issues detected that require manual intervention:\n\n" +
            "\n".join(report.issues) +
            f"\n\nBot uptime: {report.uptime:.0f} seconds",
            is_critical=True
        )
    
    async def _send_alert(self, subject: str, message: str, is_critical: bool = False):
        """Send alert notification"""
        try:
            # Use bot's email system if available
            if self.bot and hasattr(self.bot, 'email_system'):
                await self.bot.email_system.send_alert(
                    subject=f"{'[CRITICAL] ' if is_critical else '[ALERT] '}{subject}",
                    message=message,
                    is_critical=is_critical
                )
            else:
                # Log the alert
                log_level = logging.CRITICAL if is_critical else logging.WARNING
                logger.log(log_level, f"ALERT: {subject} - {message}")
                
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    async def _save_health_report(self, report: HealthReport):
        """Save health report to file"""
        try:
            health_dir = Path('health_reports')
            health_dir.mkdir(exist_ok=True)
            
            # Save latest report
            latest_file = health_dir / 'latest_health_report.json'
            with open(latest_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            
            # Save historical report
            timestamp_str = report.timestamp.strftime('%Y%m%d_%H%M%S')
            historical_file = health_dir / f'health_report_{timestamp_str}.json'
            with open(historical_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            
            # Cleanup old reports (keep last 100)
            all_reports = sorted(health_dir.glob('health_report_*.json'))
            if len(all_reports) > 100:
                for old_report in all_reports[:-100]:
                    old_report.unlink()
                    
        except Exception as e:
            logger.error(f"Failed to save health report: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current health status"""
        if self.last_report:
            return self.last_report.to_dict()
        else:
            return {
                'status': 'unknown',
                'message': 'No health report available yet'
            }
    
    async def force_health_check(self) -> HealthReport:
        """Force an immediate health check"""
        await self._collect_metrics()
        report = self._generate_health_report()
        self.last_report = report
        await self._save_health_report(report)
        return report