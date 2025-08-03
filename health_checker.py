#!/usr/bin/env python3
"""
External Health Checker for SolTrader Bot
Independent monitoring script that can restart the bot if needed
Run via cron or systemd for continuous monitoring
"""

import argparse
import json
import logging
import os
import psutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Setup logging
def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/health_checker.log'),
            logging.StreamHandler()
        ]
    )

class ExternalHealthChecker:
    """External health monitoring and recovery system"""
    
    def __init__(self, config_file: str = "health_checker_config.json"):
        """Initialize external health checker"""
        self.config = self._load_config(config_file)
        self.logger = logging.getLogger(__name__)
        self.bot_process = None
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        default_config = {
            "bot_script": "main.py",
            "python_executable": "./venv/Scripts/python.exe",
            "working_directory": ".",
            "service_name": "soltrader",
            "use_systemd": False,
            "max_restarts_per_hour": 3,
            "checks": {
                "process_running": True,
                "log_activity": True,
                "performance_metrics": True,
                "resource_usage": True,
                "api_connectivity": True
            },
            "thresholds": {
                "max_cpu_usage": 95,
                "max_memory_usage": 90,
                "max_log_age_minutes": 15,
                "min_tokens_per_scan": 50,
                "min_approval_rate": 15
            },
            "files": {
                "log_file": "logs/trading.log",
                "dashboard_file": "bot_data.json",
                "health_report": "health_reports/latest_health_report.json",
                "restart_log": "logs/restart_log.json"
            },
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": None,
                "password": None,
                "to_address": None
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                    default_config.update(custom_config)
            except Exception as e:
                print(f"Warning: Failed to load config file {config_file}: {e}")
        
        return default_config
    
    def check_bot_health(self) -> Tuple[bool, List[str]]:
        """Perform comprehensive bot health check"""
        issues = []
        
        # Check 1: Process running
        if self.config["checks"]["process_running"]:
            if not self._is_bot_process_running():
                issues.append("Bot process is not running")
        
        # Check 2: Log activity
        if self.config["checks"]["log_activity"]:
            if not self._check_log_activity():
                issues.append("No recent log activity detected")
        
        # Check 3: Performance metrics
        if self.config["checks"]["performance_metrics"]:
            perf_issues = self._check_performance_metrics()
            issues.extend(perf_issues)
        
        # Check 4: Resource usage
        if self.config["checks"]["resource_usage"]:
            resource_issues = self._check_resource_usage()
            issues.extend(resource_issues)
        
        # Check 5: API connectivity (if bot is running)
        if self.config["checks"]["api_connectivity"] and not issues:
            api_issues = self._check_api_connectivity()
            issues.extend(api_issues)
        
        is_healthy = len(issues) == 0
        return is_healthy, issues
    
    def _is_bot_process_running(self) -> bool:
        """Check if bot process is running"""
        try:
            if self.config["use_systemd"]:
                result = subprocess.run(
                    ["systemctl", "is-active", self.config["service_name"]],
                    capture_output=True,
                    text=True
                )
                return result.returncode == 0 and result.stdout.strip() == "active"
            else:
                # Check for Python process running the bot script
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['name'] and 'python' in proc.info['name'].lower():
                            cmdline = proc.info['cmdline'] or []
                            if any(self.config["bot_script"] in arg for arg in cmdline):
                                self.bot_process = proc
                                return True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                return False
        except Exception as e:
            self.logger.error(f"Error checking process status: {e}")
            return False
    
    def _check_log_activity(self) -> bool:
        """Check for recent log activity"""
        try:
            log_file = Path(self.config["files"]["log_file"])
            if not log_file.exists():
                return False
            
            # Get file modification time
            mod_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            max_age = timedelta(minutes=self.config["thresholds"]["max_log_age_minutes"])
            
            return datetime.now() - mod_time < max_age
            
        except Exception as e:
            self.logger.error(f"Error checking log activity: {e}")
            return False
    
    def _check_performance_metrics(self) -> List[str]:
        """Check bot performance metrics"""
        issues = []
        
        try:
            # Check dashboard data
            dashboard_file = Path(self.config["files"]["dashboard_file"])
            if dashboard_file.exists():
                with open(dashboard_file, 'r') as f:
                    data = json.load(f)
                
                # Check recent scan activity
                activity = data.get('activity', [])
                recent_scans = [
                    a for a in activity 
                    if a.get('type') == 'scan_completed' and
                    datetime.fromisoformat(a.get('timestamp', '2024-01-01')) > datetime.now() - timedelta(hours=1)
                ]
                
                if recent_scans:
                    avg_tokens = sum(scan.get('data', {}).get('tokens_found', 0) for scan in recent_scans) / len(recent_scans)
                    if avg_tokens < self.config["thresholds"]["min_tokens_per_scan"]:
                        issues.append(f"Low token discovery rate: {avg_tokens:.1f} < {self.config['thresholds']['min_tokens_per_scan']}")
                else:
                    issues.append("No recent scan activity found")
                
                # Check trade execution
                trades = data.get('trades', [])
                performance = data.get('performance', {})
                open_positions = performance.get('open_positions', 0)
                
                # If we have many approved tokens but no trades or positions, that's an issue
                if avg_tokens > 100 and len(trades) == 0 and open_positions == 0:
                    issues.append("High token discovery but no trades executed")
            
            # Check health reports
            health_report_file = Path(self.config["files"]["health_report"])
            if health_report_file.exists():
                with open(health_report_file, 'r') as f:
                    report = json.load(f)
                
                if report.get('overall_status') == 'critical':
                    issues.append("Health monitor reports critical status")
                
                # Check specific metrics
                metrics = report.get('metrics', {})
                approval_rate = metrics.get('approval_rate', {}).get('current_value', 0)
                if approval_rate < self.config["thresholds"]["min_approval_rate"]:
                    issues.append(f"Low approval rate: {approval_rate:.1f}% < {self.config['thresholds']['min_approval_rate']}%")
        
        except Exception as e:
            self.logger.error(f"Error checking performance metrics: {e}")
            issues.append("Failed to check performance metrics")
        
        return issues
    
    def _check_resource_usage(self) -> List[str]:
        """Check system resource usage"""
        issues = []
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.config["thresholds"]["max_cpu_usage"]:
                issues.append(f"High CPU usage: {cpu_percent:.1f}% > {self.config['thresholds']['max_cpu_usage']}%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.config["thresholds"]["max_memory_usage"]:
                issues.append(f"High memory usage: {memory.percent:.1f}% > {self.config['thresholds']['max_memory_usage']}%")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 90:  # Hard-coded high threshold
                issues.append(f"High disk usage: {disk_percent:.1f}% > 90%")
        
        except Exception as e:
            self.logger.error(f"Error checking resource usage: {e}")
            issues.append("Failed to check resource usage")
        
        return issues
    
    def _check_api_connectivity(self) -> List[str]:
        """Check API connectivity (basic test)"""
        issues = []
        
        try:
            # This is a simple connectivity check
            # The actual API health is monitored by the internal health monitor
            import urllib.request
            import socket
            
            # Test internet connectivity
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=10)
            except OSError:
                issues.append("No internet connectivity")
            
        except Exception as e:
            self.logger.error(f"Error checking API connectivity: {e}")
            issues.append("Failed to check API connectivity")
        
        return issues
    
    def restart_bot(self) -> bool:
        """Restart the bot"""
        try:
            self.logger.info("Attempting to restart bot...")
            
            # Log restart attempt
            self._log_restart_attempt("restart_requested")
            
            # Check restart limits
            if not self._can_restart():
                self.logger.warning("Restart limit exceeded")
                self._send_alert("Restart Limit Exceeded", 
                               "Bot restart limit exceeded. Manual intervention required.")
                return False
            
            # Stop bot
            if not self._stop_bot():
                self.logger.error("Failed to stop bot")
                return False
            
            # Wait a moment
            time.sleep(5)
            
            # Start bot
            if not self._start_bot():
                self.logger.error("Failed to start bot")
                return False
            
            self.logger.info("Bot restart completed successfully")
            self._log_restart_attempt("restart_successful")
            self._send_alert("Bot Restarted", "Bot has been successfully restarted by health checker.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restarting bot: {e}")
            self._log_restart_attempt("restart_failed", str(e))
            return False
    
    def _stop_bot(self) -> bool:
        """Stop the bot"""
        try:
            if self.config["use_systemd"]:
                result = subprocess.run(
                    ["systemctl", "stop", self.config["service_name"]],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                return result.returncode == 0
            else:
                # Find and terminate the bot process
                if self.bot_process and self.bot_process.is_running():
                    self.bot_process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        self.bot_process.wait(timeout=15)
                    except psutil.TimeoutExpired:
                        # Force kill if needed
                        self.bot_process.kill()
                        self.bot_process.wait(timeout=5)
                    
                    return True
                return True  # Already stopped
                
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
            return False
    
    def _start_bot(self) -> bool:
        """Start the bot"""
        try:
            if self.config["use_systemd"]:
                result = subprocess.run(
                    ["systemctl", "start", self.config["service_name"]],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                return result.returncode == 0
            else:
                # Start bot process
                cmd = [
                    self.config["python_executable"],
                    self.config["bot_script"]
                ]
                
                # Change to working directory
                cwd = self.config["working_directory"]
                
                # Start process in background
                process = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
                
                # Wait a moment and check if it's still running
                time.sleep(3)
                if process.poll() is None:
                    return True
                else:
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
            return False
    
    def _can_restart(self) -> bool:
        """Check if bot can be restarted (within limits)"""
        try:
            restart_log_file = Path(self.config["files"]["restart_log"])
            if not restart_log_file.exists():
                return True
            
            with open(restart_log_file, 'r') as f:
                restart_log = json.load(f)
            
            # Count restarts in the last hour
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_restarts = [
                entry for entry in restart_log
                if datetime.fromisoformat(entry['timestamp']) > one_hour_ago
                and entry['action'] in ['restart_successful', 'restart_requested']
            ]
            
            return len(recent_restarts) < self.config["max_restarts_per_hour"]
            
        except Exception as e:
            self.logger.error(f"Error checking restart limits: {e}")
            return True  # Allow restart if we can't check
    
    def _log_restart_attempt(self, action: str, details: str = ""):
        """Log restart attempt"""
        try:
            restart_log_file = Path(self.config["files"]["restart_log"])
            
            # Load existing log
            if restart_log_file.exists():
                with open(restart_log_file, 'r') as f:
                    restart_log = json.load(f)
            else:
                restart_log = []
            
            # Add new entry
            restart_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'details': details
            })
            
            # Keep only last 100 entries
            restart_log = restart_log[-100:]
            
            # Save log
            restart_log_file.parent.mkdir(exist_ok=True)
            with open(restart_log_file, 'w') as f:
                json.dump(restart_log, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging restart attempt: {e}")
    
    def _send_alert(self, subject: str, message: str):
        """Send email alert if configured"""
        try:
            if not self.config["email"]["enabled"]:
                return
            
            import smtplib
            from email.mime.text import MimeText
            from email.mime.multipart import MimeMultipart
            
            username = self.config["email"]["username"]
            password = self.config["email"]["password"]
            to_address = self.config["email"]["to_address"]
            
            if not all([username, password, to_address]):
                self.logger.warning("Email configuration incomplete, skipping alert")
                return
            
            msg = MimeMultipart()
            msg['From'] = username
            msg['To'] = to_address
            msg['Subject'] = f"[SolTrader Health] {subject}"
            
            body = f"""
Health Checker Alert

{message}

Timestamp: {datetime.now().isoformat()}
Server: {os.uname().nodename if hasattr(os, 'uname') else 'Windows'}

This is an automated message from the SolTrader health monitoring system.
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.config["email"]["smtp_server"], self.config["email"]["smtp_port"])
            server.starttls()
            server.login(username, password)
            text = msg.as_string()
            server.sendmail(username, to_address, text)
            server.quit()
            
            self.logger.info(f"Alert email sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def run_check(self, auto_restart: bool = True) -> Dict:
        """Run health check and optionally restart if needed"""
        self.logger.info("Starting health check...")
        
        is_healthy, issues = self.check_bot_health()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'healthy': is_healthy,
            'issues': issues,
            'restart_attempted': False,
            'restart_successful': False
        }
        
        if not is_healthy:
            self.logger.warning(f"Health issues detected: {issues}")
            
            if auto_restart:
                self.logger.info("Attempting auto-restart...")
                result['restart_attempted'] = True
                
                restart_success = self.restart_bot()
                result['restart_successful'] = restart_success
                
                if restart_success:
                    self.logger.info("Auto-restart successful")
                    # Wait and recheck
                    time.sleep(10)
                    is_healthy_after, issues_after = self.check_bot_health()
                    result['healthy_after_restart'] = is_healthy_after
                    result['issues_after_restart'] = issues_after
                else:
                    self.logger.error("Auto-restart failed")
                    self._send_alert("Bot Restart Failed", 
                                   f"Failed to restart bot. Issues: {', '.join(issues)}")
            else:
                self._send_alert("Bot Health Issues", 
                               f"Health issues detected: {', '.join(issues)}")
        else:
            self.logger.info("Bot is healthy")
        
        return result

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SolTrader Bot Health Checker")
    parser.add_argument('--config', default='health_checker_config.json',
                       help='Config file path')
    parser.add_argument('--no-restart', action='store_true',
                       help='Check health but do not restart')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as daemon (continuous monitoring)')
    parser.add_argument('--interval', type=int, default=300,
                       help='Check interval in seconds (daemon mode)')
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        checker = ExternalHealthChecker(args.config)
        
        if args.daemon:
            logger.info(f"Starting daemon mode with {args.interval}s interval")
            while True:
                try:
                    result = checker.run_check(auto_restart=not args.no_restart)
                    logger.info(f"Health check result: {result['healthy']}")
                    time.sleep(args.interval)
                except KeyboardInterrupt:
                    logger.info("Daemon mode interrupted")
                    break
                except Exception as e:
                    logger.error(f"Error in daemon loop: {e}")
                    time.sleep(60)  # Wait before retry
        else:
            result = checker.run_check(auto_restart=not args.no_restart)
            print(json.dumps(result, indent=2))
            
            # Exit with error code if unhealthy
            sys.exit(0 if result['healthy'] else 1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()