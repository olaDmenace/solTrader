import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.database.db_manager import DatabaseManager


@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    uptime: float


class SystemMonitor:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        self.is_monitoring = False
        self.start_time = datetime.now()
        self.metrics_history: List[SystemMetrics] = []
        
        # Alert thresholds
        self.cpu_threshold = 80.0
        self.memory_threshold = 85.0
        self.disk_threshold = 90.0
        
    async def initialize(self):
        """Initialize system monitoring"""
        try:
            self.logger.info("System monitor initialized")
            
        except Exception as e:
            self.logger.error(f"System monitor initialization failed: {e}")
            raise
            
    async def start_monitoring(self):
        """Start system monitoring loop"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.start_time = datetime.now()
        
        # Start monitoring task
        asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("System monitoring started")
        
    async def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        self.logger.info("System monitoring stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                    
                # Log metrics to database
                await self._log_metrics(metrics)
                
                # Check for alerts
                await self._check_alert_conditions(metrics)
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
                
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Active connections
            connections = len(psutil.net_connections())
            
            # Uptime
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                active_connections=connections,
                uptime=uptime
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            
            # Return default metrics on error
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                active_connections=0,
                uptime=0.0
            )
            
    async def _log_metrics(self, metrics: SystemMetrics):
        """Log metrics to database"""
        try:
            await self.db_manager.log_metric("cpu_usage", metrics.cpu_usage)
            await self.db_manager.log_metric("memory_usage", metrics.memory_usage)
            await self.db_manager.log_metric("disk_usage", metrics.disk_usage)
            await self.db_manager.log_metric("active_connections", metrics.active_connections)
            await self.db_manager.log_metric("uptime", metrics.uptime)
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
            
    async def _check_alert_conditions(self, metrics: SystemMetrics):
        """Check for system alert conditions"""
        try:
            alerts = []
            
            if metrics.cpu_usage > self.cpu_threshold:
                alerts.append({
                    'type': 'high_cpu',
                    'message': f'High CPU usage: {metrics.cpu_usage:.1f}%',
                    'severity': 'warning'
                })
                
            if metrics.memory_usage > self.memory_threshold:
                alerts.append({
                    'type': 'high_memory',
                    'message': f'High memory usage: {metrics.memory_usage:.1f}%',
                    'severity': 'warning'
                })
                
            if metrics.disk_usage > self.disk_threshold:
                alerts.append({
                    'type': 'high_disk',
                    'message': f'High disk usage: {metrics.disk_usage:.1f}%',
                    'severity': 'critical'
                })
                
            # Log alerts
            for alert in alerts:
                await self.log_system_event('system_alert', alert, alert['severity'])
                self.logger.warning(f"SYSTEM ALERT: {alert['message']}")
                
        except Exception as e:
            self.logger.error(f"Alert check failed: {e}")
            
    async def log_system_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Log a custom system metric"""
        try:
            await self.db_manager.log_metric(metric_name, value, metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to log system metric: {e}")
            
    async def log_system_event(self, event_type: str, event_data: Dict[str, Any], severity: str = 'info'):
        """Log a system event"""
        try:
            await self.db_manager.log_event(event_type, event_data, severity)
            
        except Exception as e:
            self.logger.error(f"Failed to log system event: {e}")
            
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get most recent system metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
        
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get summary of metrics for the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                return {}
                
            cpu_values = [m.cpu_usage for m in recent_metrics]
            memory_values = [m.memory_usage for m in recent_metrics]
            disk_values = [m.disk_usage for m in recent_metrics]
            
            return {
                'period_hours': hours,
                'data_points': len(recent_metrics),
                'cpu_usage': {
                    'avg': sum(cpu_values) / len(cpu_values),
                    'max': max(cpu_values),
                    'min': min(cpu_values)
                },
                'memory_usage': {
                    'avg': sum(memory_values) / len(memory_values),
                    'max': max(memory_values),
                    'min': min(memory_values)
                },
                'disk_usage': {
                    'avg': sum(disk_values) / len(disk_values),
                    'max': max(disk_values),
                    'min': min(disk_values)
                },
                'uptime_hours': recent_metrics[-1].uptime / 3600 if recent_metrics else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics summary: {e}")
            return {}
            
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            current_metrics = self.get_current_metrics()
            
            if not current_metrics:
                return {
                    'status': 'UNKNOWN',
                    'message': 'No metrics available'
                }
                
            # Determine health status
            issues = []
            
            if current_metrics.cpu_usage > self.cpu_threshold:
                issues.append(f'High CPU: {current_metrics.cpu_usage:.1f}%')
                
            if current_metrics.memory_usage > self.memory_threshold:
                issues.append(f'High Memory: {current_metrics.memory_usage:.1f}%')
                
            if current_metrics.disk_usage > self.disk_threshold:
                issues.append(f'High Disk: {current_metrics.disk_usage:.1f}%')
                
            if issues:
                status = 'WARNING' if len(issues) == 1 else 'CRITICAL'
                message = ', '.join(issues)
            else:
                status = 'HEALTHY'
                message = 'All systems normal'
                
            return {
                'status': status,
                'message': message,
                'metrics': {
                    'cpu_usage': current_metrics.cpu_usage,
                    'memory_usage': current_metrics.memory_usage,
                    'disk_usage': current_metrics.disk_usage,
                    'active_connections': current_metrics.active_connections,
                    'uptime_hours': current_metrics.uptime / 3600
                },
                'timestamp': current_metrics.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health status check failed: {e}")
            return {
                'status': 'ERROR',
                'message': f'Health check failed: {e}'
            }
            
    async def shutdown(self):
        """Shutdown system monitor"""
        try:
            await self.stop_monitoring()
            self.logger.info("System monitor shutdown")
            
        except Exception as e:
            self.logger.error(f"System monitor shutdown error: {e}")