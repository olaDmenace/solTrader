#!/usr/bin/env python3
"""
UNIFIED SYSTEM MANAGER - Day 12 Implementation
Professional health monitoring via Prometheus with automated health checks

This system manager provides:
1. Comprehensive system health monitoring
2. Prometheus metrics integration for professional monitoring
3. Automated health checks and alerting
4. Performance monitoring across all managers
5. Resource utilization tracking
6. Error tracking and alerting
7. Grafana dashboard integration support
8. System recovery and maintenance automation
"""

import asyncio
import logging
import time
import json
import os
import psutil
import sqlite3
import aiosqlite
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import threading
import traceback

# Prometheus integration
try:
    from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, Info
    from prometheus_client import start_http_server, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available - install with: pip install prometheus_client")

# Email notifications for critical alerts
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_available_gb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    active_connections: int

@dataclass
class ComponentHealth:
    """Individual component health status"""
    component_name: str
    status: HealthStatus
    last_check: datetime
    error_count: int = 0
    last_error: Optional[str] = None
    response_time_ms: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """System alert structure"""
    id: str
    level: AlertLevel
    title: str
    message: str
    component: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class UnifiedSystemManager:
    """Comprehensive system monitoring and health management"""
    
    def __init__(self, settings: Any):
        self.settings = settings
        
        # Component registry
        self.components: Dict[str, ComponentHealth] = {}
        self.manager_registry: Dict[str, Any] = {}
        
        # Metrics tracking
        self.system_metrics_history: deque = deque(maxsize=1000)  # Keep last 1000 readings
        self.performance_baselines: Dict[str, float] = {}
        
        # Alert system
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_callbacks: List[callable] = []
        
        # Health check configuration
        self.health_check_interval = getattr(settings, 'HEALTH_CHECK_INTERVAL', 30)  # seconds
        self.metrics_collection_interval = getattr(settings, 'METRICS_COLLECTION_INTERVAL', 10)  # seconds
        self.alert_thresholds = {
            'cpu_percent': getattr(settings, 'CPU_ALERT_THRESHOLD', 80.0),
            'memory_percent': getattr(settings, 'MEMORY_ALERT_THRESHOLD', 85.0),
            'disk_percent': getattr(settings, 'DISK_ALERT_THRESHOLD', 90.0),
            'component_error_rate': getattr(settings, 'COMPONENT_ERROR_THRESHOLD', 0.1),
            'response_time_ms': getattr(settings, 'RESPONSE_TIME_THRESHOLD', 5000.0)
        }
        
        # Prometheus integration
        self.prometheus_enabled = PROMETHEUS_AVAILABLE and getattr(settings, 'PROMETHEUS_ENABLED', True)
        self.prometheus_port = getattr(settings, 'PROMETHEUS_PORT', 8000)
        
        if self.prometheus_enabled:
            self._initialize_prometheus_metrics()
        
        # Database for persistent monitoring data
        self.db_path = "data/system_monitoring.db"
        self.db_initialized = False
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Email notifications
        self.email_enabled = EMAIL_AVAILABLE and hasattr(settings, 'EMAIL_SMTP_SERVER')
        
        logger.info("[UNIFIED_SYSTEM] System manager configuration loaded")
    
    async def initialize(self):
        """Initialize the unified system manager"""
        try:
            logger.info("[UNIFIED_SYSTEM] Initializing Unified System Manager...")
            
            # Initialize database
            await self._initialize_database()
            
            # Setup Prometheus metrics server
            if self.prometheus_enabled:
                await self._setup_prometheus_server()
            
            # Establish performance baselines
            await self._establish_baselines()
            
            # Start background monitoring tasks
            await self._start_monitoring_tasks()
            
            logger.info("[UNIFIED_SYSTEM] Unified System Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"[UNIFIED_SYSTEM] System manager initialization failed: {e}")
            raise
    
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics collectors"""
        if not self.prometheus_enabled:
            return
        
        self.prometheus_registry = CollectorRegistry()
        
        # System metrics
        self.prom_cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage', registry=self.prometheus_registry)
        self.prom_memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage', registry=self.prometheus_registry)
        self.prom_disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage', registry=self.prometheus_registry)
        self.prom_network_sent = Counter('system_network_sent_bytes_total', 'Network bytes sent', registry=self.prometheus_registry)
        self.prom_network_recv = Counter('system_network_recv_bytes_total', 'Network bytes received', registry=self.prometheus_registry)
        
        # Component health metrics
        self.prom_component_health = Gauge('component_health_status', 'Component health status (1=healthy, 0=unhealthy)', ['component'], registry=self.prometheus_registry)
        self.prom_component_response_time = Histogram('component_response_time_seconds', 'Component response time', ['component'], registry=self.prometheus_registry)
        self.prom_component_errors = Counter('component_errors_total', 'Component error count', ['component'], registry=self.prometheus_registry)
        
        # Trading metrics
        self.prom_active_orders = Gauge('trading_active_orders', 'Number of active orders', registry=self.prometheus_registry)
        self.prom_total_trades = Counter('trading_trades_total', 'Total number of trades', ['strategy'], registry=self.prometheus_registry)
        self.prom_trade_success_rate = Gauge('trading_success_rate', 'Trade success rate', ['strategy'], registry=self.prometheus_registry)
        self.prom_portfolio_value = Gauge('portfolio_total_value_usd', 'Total portfolio value in USD', registry=self.prometheus_registry)
        
        # Risk metrics  
        self.prom_risk_level = Gauge('risk_current_level', 'Current risk level (0-1)', registry=self.prometheus_registry)
        self.prom_daily_pnl = Gauge('trading_daily_pnl_usd', 'Daily PnL in USD', registry=self.prometheus_registry)
        self.prom_position_count = Gauge('trading_active_positions', 'Number of active positions', registry=self.prometheus_registry)
        
        # Alert metrics
        self.prom_active_alerts = Gauge('system_active_alerts', 'Number of active alerts', ['level'], registry=self.prometheus_registry)
        
        logger.info("[UNIFIED_SYSTEM] Prometheus metrics initialized")
    
    async def _initialize_database(self):
        """Initialize SQLite database for monitoring data"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            async with aiosqlite.connect(self.db_path) as db:
                # System metrics table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        cpu_percent REAL NOT NULL,
                        memory_percent REAL NOT NULL,
                        memory_used_gb REAL NOT NULL,
                        memory_available_gb REAL NOT NULL,
                        disk_percent REAL NOT NULL,
                        disk_used_gb REAL NOT NULL,
                        disk_available_gb REAL NOT NULL,
                        network_sent_mb REAL NOT NULL,
                        network_recv_mb REAL NOT NULL,
                        process_count INTEGER NOT NULL,
                        active_connections INTEGER NOT NULL
                    )
                """)
                
                # Component health table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS component_health (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        component_name TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        status TEXT NOT NULL,
                        error_count INTEGER NOT NULL,
                        last_error TEXT,
                        response_time_ms REAL NOT NULL,
                        custom_metrics TEXT
                    )
                """)
                
                # Alert history table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS alert_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT NOT NULL,
                        level TEXT NOT NULL,
                        title TEXT NOT NULL,
                        message TEXT NOT NULL,
                        component TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        resolved BOOLEAN NOT NULL,
                        resolution_time REAL,
                        metadata TEXT
                    )
                """)
                
                await db.commit()
                
            self.db_initialized = True
            logger.info("[UNIFIED_SYSTEM] Monitoring database initialized")
            
        except Exception as e:
            logger.error(f"[UNIFIED_SYSTEM] Database initialization failed: {e}")
            raise
    
    async def _setup_prometheus_server(self):
        """Setup Prometheus metrics HTTP server"""
        if not self.prometheus_enabled:
            return
        
        try:
            def run_prometheus_server():
                start_http_server(self.prometheus_port, registry=self.prometheus_registry)
                logger.info(f"[UNIFIED_SYSTEM] Prometheus metrics server started on port {self.prometheus_port}")
            
            # Start Prometheus server in separate thread
            prometheus_thread = threading.Thread(target=run_prometheus_server, daemon=True)
            prometheus_thread.start()
            
        except Exception as e:
            logger.error(f"[UNIFIED_SYSTEM] Prometheus server setup failed: {e}")
            self.prometheus_enabled = False
    
    async def _establish_baselines(self):
        """Establish performance baselines for anomaly detection"""
        try:
            # Collect initial system metrics for baseline
            initial_metrics = await self._collect_system_metrics()
            
            self.performance_baselines = {
                'cpu_baseline': initial_metrics.cpu_percent,
                'memory_baseline': initial_metrics.memory_percent,
                'disk_baseline': initial_metrics.disk_percent,
                'response_time_baseline': 100.0  # 100ms baseline
            }
            
            logger.info(f"[UNIFIED_SYSTEM] Performance baselines established: {self.performance_baselines}")
            
        except Exception as e:
            logger.warning(f"[UNIFIED_SYSTEM] Baseline establishment failed: {e}")
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        try:
            # System metrics collection task
            self.background_tasks.append(
                asyncio.create_task(self._system_metrics_loop())
            )
            
            # Component health check task
            self.background_tasks.append(
                asyncio.create_task(self._health_check_loop())
            )
            
            # Alert processing task
            self.background_tasks.append(
                asyncio.create_task(self._alert_processing_loop())
            )
            
            # Database maintenance task
            self.background_tasks.append(
                asyncio.create_task(self._database_maintenance_loop())
            )
            
            # Performance analysis task
            self.background_tasks.append(
                asyncio.create_task(self._performance_analysis_loop())
            )
            
            logger.info("[UNIFIED_SYSTEM] Background monitoring tasks started")
            
        except Exception as e:
            logger.error(f"[UNIFIED_SYSTEM] Background task startup failed: {e}")
            raise
    
    def register_component(self, name: str, component: Any, health_check_callback: Optional[callable] = None):
        """Register a component for monitoring"""
        try:
            self.components[name] = ComponentHealth(
                component_name=name,
                status=HealthStatus.UNKNOWN,
                last_check=datetime.now()
            )
            
            self.manager_registry[name] = {
                'component': component,
                'health_check': health_check_callback
            }
            
            logger.info(f"[UNIFIED_SYSTEM] Component registered: {name}")
            
        except Exception as e:
            logger.error(f"[UNIFIED_SYSTEM] Component registration failed: {e}")
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024**3)
            disk_available_gb = disk.free / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024**2)
            network_recv_mb = network.bytes_recv / (1024**2)
            
            # Process metrics
            process_count = len(psutil.pids())
            active_connections = len(psutil.net_connections())
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_available_gb=memory_available_gb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_available_gb=disk_available_gb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                process_count=process_count,
                active_connections=active_connections
            )
            
        except Exception as e:
            logger.error(f"[UNIFIED_SYSTEM] System metrics collection failed: {e}")
            raise
    
    async def _system_metrics_loop(self):
        """Background system metrics collection"""
        while not self.shutdown_event.is_set():
            try:
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                self.system_metrics_history.append(metrics)
                
                # Update Prometheus metrics
                if self.prometheus_enabled:
                    self.prom_cpu_usage.set(metrics.cpu_percent)
                    self.prom_memory_usage.set(metrics.memory_percent)
                    self.prom_disk_usage.set(metrics.disk_percent)
                    self.prom_network_sent.inc(metrics.network_sent_mb * 1024 * 1024)
                    self.prom_network_recv.inc(metrics.network_recv_mb * 1024 * 1024)
                
                # Store to database
                if self.db_initialized:
                    await self._store_system_metrics(metrics)
                
                # Check for system alerts
                await self._check_system_alerts(metrics)
                
                await asyncio.sleep(self.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"[UNIFIED_SYSTEM] System metrics loop error: {e}")
                await asyncio.sleep(self.metrics_collection_interval)
    
    async def _health_check_loop(self):
        """Background component health checking"""
        while not self.shutdown_event.is_set():
            try:
                # Check health of all registered components
                for name, component_info in self.manager_registry.items():
                    await self._check_component_health(name, component_info)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"[UNIFIED_SYSTEM] Health check loop error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_component_health(self, name: str, component_info: Dict):
        """Check health of individual component"""
        start_time = time.time()
        
        try:
            component = component_info['component']
            health_check = component_info.get('health_check')
            
            # Default health check - try to get component status
            if health_check:
                try:
                    if asyncio.iscoroutinefunction(health_check):
                        health_result = await health_check()
                    else:
                        health_result = health_check()
                    
                    status = HealthStatus.HEALTHY if health_result else HealthStatus.WARNING
                    error_msg = None
                    
                except Exception as e:
                    status = HealthStatus.CRITICAL
                    error_msg = str(e)
                    
            else:
                # Basic health check - ensure component has essential methods
                try:
                    if hasattr(component, 'get_health_status'):
                        health_status = await component.get_health_status()
                        status = HealthStatus.HEALTHY if health_status.get('status') == 'healthy' else HealthStatus.WARNING
                        error_msg = health_status.get('error')
                    else:
                        # Component exists and is accessible
                        status = HealthStatus.HEALTHY
                        error_msg = None
                        
                except Exception as e:
                    status = HealthStatus.CRITICAL
                    error_msg = str(e)
            
            # Update component health
            component_health = self.components[name]
            component_health.status = status
            component_health.last_check = datetime.now()
            component_health.response_time_ms = (time.time() - start_time) * 1000
            
            if error_msg:
                component_health.error_count += 1
                component_health.last_error = error_msg
            
            # Update Prometheus metrics
            if self.prometheus_enabled:
                self.prom_component_health.labels(component=name).set(1 if status == HealthStatus.HEALTHY else 0)
                self.prom_component_response_time.labels(component=name).observe(component_health.response_time_ms / 1000)
                if error_msg:
                    self.prom_component_errors.labels(component=name).inc()
            
            # Store to database
            if self.db_initialized:
                await self._store_component_health(component_health)
            
            # Check for component alerts
            await self._check_component_alerts(component_health)
            
        except Exception as e:
            logger.error(f"[UNIFIED_SYSTEM] Component health check failed for {name}: {e}")
            
            # Mark component as critical
            if name in self.components:
                self.components[name].status = HealthStatus.CRITICAL
                self.components[name].last_error = str(e)
                self.components[name].error_count += 1
    
    async def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics for alert conditions"""
        try:
            alerts_to_create = []
            
            # CPU usage alert
            if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
                alerts_to_create.append(Alert(
                    id=f"cpu_high_{int(time.time())}",
                    level=AlertLevel.WARNING if metrics.cpu_percent < 95 else AlertLevel.CRITICAL,
                    title="High CPU Usage",
                    message=f"CPU usage is {metrics.cpu_percent:.1f}% (threshold: {self.alert_thresholds['cpu_percent']:.1f}%)",
                    component="system",
                    timestamp=datetime.now(),
                    metadata={'cpu_percent': metrics.cpu_percent}
                ))
            
            # Memory usage alert
            if metrics.memory_percent > self.alert_thresholds['memory_percent']:
                alerts_to_create.append(Alert(
                    id=f"memory_high_{int(time.time())}",
                    level=AlertLevel.WARNING if metrics.memory_percent < 95 else AlertLevel.CRITICAL,
                    title="High Memory Usage",
                    message=f"Memory usage is {metrics.memory_percent:.1f}% (threshold: {self.alert_thresholds['memory_percent']:.1f}%)",
                    component="system",
                    timestamp=datetime.now(),
                    metadata={'memory_percent': metrics.memory_percent, 'memory_used_gb': metrics.memory_used_gb}
                ))
            
            # Disk usage alert
            if metrics.disk_percent > self.alert_thresholds['disk_percent']:
                alerts_to_create.append(Alert(
                    id=f"disk_high_{int(time.time())}",
                    level=AlertLevel.CRITICAL,
                    title="High Disk Usage",
                    message=f"Disk usage is {metrics.disk_percent:.1f}% (threshold: {self.alert_thresholds['disk_percent']:.1f}%)",
                    component="system",
                    timestamp=datetime.now(),
                    metadata={'disk_percent': metrics.disk_percent, 'disk_used_gb': metrics.disk_used_gb}
                ))
            
            # Create alerts
            for alert in alerts_to_create:
                await self._create_alert(alert)
                
        except Exception as e:
            logger.error(f"[UNIFIED_SYSTEM] System alert check failed: {e}")
    
    async def _check_component_alerts(self, component_health: ComponentHealth):
        """Check component health for alert conditions"""
        try:
            # Component status alert
            if component_health.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                alert_level = AlertLevel.WARNING if component_health.status == HealthStatus.WARNING else AlertLevel.CRITICAL
                
                alert = Alert(
                    id=f"component_{component_health.component_name}_{int(time.time())}",
                    level=alert_level,
                    title=f"Component Health Issue: {component_health.component_name}",
                    message=f"Component {component_health.component_name} is {component_health.status.value}. Last error: {component_health.last_error}",
                    component=component_health.component_name,
                    timestamp=datetime.now(),
                    metadata={
                        'status': component_health.status.value,
                        'error_count': component_health.error_count,
                        'response_time_ms': component_health.response_time_ms
                    }
                )
                await self._create_alert(alert)
            
            # Response time alert
            if component_health.response_time_ms > self.alert_thresholds['response_time_ms']:
                alert = Alert(
                    id=f"response_time_{component_health.component_name}_{int(time.time())}",
                    level=AlertLevel.WARNING,
                    title=f"Slow Response Time: {component_health.component_name}",
                    message=f"Component {component_health.component_name} response time is {component_health.response_time_ms:.1f}ms (threshold: {self.alert_thresholds['response_time_ms']:.1f}ms)",
                    component=component_health.component_name,
                    timestamp=datetime.now(),
                    metadata={'response_time_ms': component_health.response_time_ms}
                )
                await self._create_alert(alert)
                
        except Exception as e:
            logger.error(f"[UNIFIED_SYSTEM] Component alert check failed: {e}")
    
    async def _create_alert(self, alert: Alert):
        """Create and process new alert"""
        try:
            # Check if similar alert already exists
            existing_alert_key = f"{alert.component}_{alert.level.value}"
            if existing_alert_key in self.active_alerts:
                # Update existing alert timestamp
                self.active_alerts[existing_alert_key].timestamp = alert.timestamp
                return
            
            # Add to active alerts
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
            
            # Update Prometheus alert metrics
            if self.prometheus_enabled:
                self.prom_active_alerts.labels(level=alert.level.value).inc()
            
            # Store to database
            if self.db_initialized:
                await self._store_alert(alert)
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.warning(f"[UNIFIED_SYSTEM] Alert callback failed: {e}")
            
            # Send email notification for critical alerts
            if alert.level == AlertLevel.CRITICAL and self.email_enabled:
                await self._send_email_alert(alert)
            
            logger.warning(f"[UNIFIED_SYSTEM] Alert created: {alert.title} - {alert.message}")
            
        except Exception as e:
            logger.error(f"[UNIFIED_SYSTEM] Alert creation failed: {e}")
    
    async def _alert_processing_loop(self):
        """Background alert processing and resolution"""
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                alerts_to_resolve = []
                
                # Check for alerts that can be auto-resolved
                for alert_id, alert in self.active_alerts.items():
                    # Auto-resolve alerts older than 1 hour if conditions are normal
                    if (current_time - alert.timestamp).seconds > 3600:
                        if await self._check_alert_resolution(alert):
                            alerts_to_resolve.append(alert_id)
                
                # Resolve alerts
                for alert_id in alerts_to_resolve:
                    await self.resolve_alert(alert_id)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"[UNIFIED_SYSTEM] Alert processing loop error: {e}")
                await asyncio.sleep(300)
    
    async def _check_alert_resolution(self, alert: Alert) -> bool:
        """Check if alert conditions have been resolved"""
        try:
            if alert.component == "system":
                # Check latest system metrics
                if self.system_metrics_history:
                    latest_metrics = self.system_metrics_history[-1]
                    
                    if "cpu" in alert.message.lower():
                        return latest_metrics.cpu_percent < self.alert_thresholds['cpu_percent']
                    elif "memory" in alert.message.lower():
                        return latest_metrics.memory_percent < self.alert_thresholds['memory_percent']
                    elif "disk" in alert.message.lower():
                        return latest_metrics.disk_percent < self.alert_thresholds['disk_percent']
            
            else:
                # Check component health
                if alert.component in self.components:
                    component_health = self.components[alert.component]
                    return component_health.status == HealthStatus.HEALTHY
            
            return False
            
        except Exception as e:
            logger.warning(f"[UNIFIED_SYSTEM] Alert resolution check failed: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str):
        """Manually or automatically resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_time = datetime.now()
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                # Update Prometheus metrics
                if self.prometheus_enabled:
                    self.prom_active_alerts.labels(level=alert.level.value).dec()
                
                # Update database
                if self.db_initialized:
                    await self._update_alert_resolution(alert)
                
                logger.info(f"[UNIFIED_SYSTEM] Alert resolved: {alert.title}")
                
        except Exception as e:
            logger.error(f"[UNIFIED_SYSTEM] Alert resolution failed: {e}")
    
    async def _database_maintenance_loop(self):
        """Background database maintenance"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                if self.db_initialized:
                    # Clean old data (keep last 7 days)
                    cutoff_time = time.time() - (7 * 24 * 3600)
                    
                    async with aiosqlite.connect(self.db_path) as db:
                        await db.execute("DELETE FROM system_metrics WHERE timestamp < ?", (cutoff_time,))
                        await db.execute("DELETE FROM component_health WHERE timestamp < ?", (cutoff_time,))
                        
                        # Keep resolved alerts for 30 days
                        alert_cutoff = time.time() - (30 * 24 * 3600)
                        await db.execute("DELETE FROM alert_history WHERE resolved = 1 AND resolution_time < ?", (alert_cutoff,))
                        
                        await db.commit()
                    
                    logger.info("[UNIFIED_SYSTEM] Database maintenance completed")
                
            except Exception as e:
                logger.error(f"[UNIFIED_SYSTEM] Database maintenance error: {e}")
    
    async def _performance_analysis_loop(self):
        """Background performance analysis and optimization"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                # Analyze system performance trends
                if len(self.system_metrics_history) >= 10:
                    await self._analyze_performance_trends()
                
                # Update component performance scores
                await self._update_component_scores()
                
            except Exception as e:
                logger.error(f"[UNIFIED_SYSTEM] Performance analysis error: {e}")
    
    async def _analyze_performance_trends(self):
        """Analyze system performance trends and predict issues"""
        try:
            # Get recent metrics
            recent_metrics = list(self.system_metrics_history)[-60:]  # Last hour
            
            if len(recent_metrics) < 10:
                return
            
            # Calculate trends
            cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
            memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
            
            # Predict potential issues
            if cpu_trend > 1.0:  # CPU increasing by 1% per reading
                logger.warning("[UNIFIED_SYSTEM] CPU usage trend indicates potential issue")
                
            if memory_trend > 0.5:  # Memory increasing by 0.5% per reading
                logger.warning("[UNIFIED_SYSTEM] Memory usage trend indicates potential issue")
                
        except Exception as e:
            logger.warning(f"[UNIFIED_SYSTEM] Performance trend analysis failed: {e}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values"""
        try:
            if len(values) < 2:
                return 0.0
            
            n = len(values)
            x_values = list(range(n))
            
            # Simple linear regression slope calculation
            x_mean = sum(x_values) / n
            y_mean = sum(values) / n
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            return numerator / denominator if denominator != 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def _update_component_scores(self):
        """Update component performance scores"""
        try:
            for name, health in self.components.items():
                # Calculate performance score based on multiple factors
                score = 100.0  # Start with perfect score
                
                # Deduct points for errors
                if health.error_count > 0:
                    score -= min(health.error_count * 5, 30)  # Max 30 points for errors
                
                # Deduct points for slow response
                if health.response_time_ms > 1000:
                    score -= min((health.response_time_ms - 1000) / 100, 20)  # Max 20 points for slowness
                
                # Deduct points for poor health status
                if health.status == HealthStatus.WARNING:
                    score -= 25
                elif health.status == HealthStatus.CRITICAL:
                    score -= 50
                elif health.status == HealthStatus.UNKNOWN:
                    score -= 10
                
                health.custom_metrics['performance_score'] = max(score, 0)
                
        except Exception as e:
            logger.warning(f"[UNIFIED_SYSTEM] Component score update failed: {e}")
    
    # Database storage methods
    async def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO system_metrics 
                    (timestamp, cpu_percent, memory_percent, memory_used_gb, memory_available_gb,
                     disk_percent, disk_used_gb, disk_available_gb, network_sent_mb, network_recv_mb,
                     process_count, active_connections)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp.timestamp(),
                    metrics.cpu_percent,
                    metrics.memory_percent,
                    metrics.memory_used_gb,
                    metrics.memory_available_gb,
                    metrics.disk_percent,
                    metrics.disk_used_gb,
                    metrics.disk_available_gb,
                    metrics.network_sent_mb,
                    metrics.network_recv_mb,
                    metrics.process_count,
                    metrics.active_connections
                ))
                await db.commit()
        except Exception as e:
            logger.warning(f"[UNIFIED_SYSTEM] System metrics storage failed: {e}")
    
    async def _store_component_health(self, health: ComponentHealth):
        """Store component health to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO component_health 
                    (component_name, timestamp, status, error_count, last_error, response_time_ms, custom_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    health.component_name,
                    health.last_check.timestamp(),
                    health.status.value,
                    health.error_count,
                    health.last_error,
                    health.response_time_ms,
                    json.dumps(health.custom_metrics)
                ))
                await db.commit()
        except Exception as e:
            logger.warning(f"[UNIFIED_SYSTEM] Component health storage failed: {e}")
    
    async def _store_alert(self, alert: Alert):
        """Store alert to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO alert_history 
                    (alert_id, level, title, message, component, timestamp, resolved, resolution_time, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.id,
                    alert.level.value,
                    alert.title,
                    alert.message,
                    alert.component,
                    alert.timestamp.timestamp(),
                    alert.resolved,
                    alert.resolution_time.timestamp() if alert.resolution_time else None,
                    json.dumps(alert.metadata)
                ))
                await db.commit()
        except Exception as e:
            logger.warning(f"[UNIFIED_SYSTEM] Alert storage failed: {e}")
    
    async def _update_alert_resolution(self, alert: Alert):
        """Update alert resolution in database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE alert_history 
                    SET resolved = ?, resolution_time = ?
                    WHERE alert_id = ?
                """, (
                    alert.resolved,
                    alert.resolution_time.timestamp() if alert.resolution_time else None,
                    alert.id
                ))
                await db.commit()
        except Exception as e:
            logger.warning(f"[UNIFIED_SYSTEM] Alert resolution update failed: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send email notification for critical alerts"""
        if not self.email_enabled:
            return
        
        try:
            # Email configuration from settings
            smtp_server = getattr(self.settings, 'EMAIL_SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = getattr(self.settings, 'EMAIL_SMTP_PORT', 587)
            sender_email = getattr(self.settings, 'EMAIL_SENDER', '')
            sender_password = getattr(self.settings, 'EMAIL_PASSWORD', '')
            recipient_emails = getattr(self.settings, 'EMAIL_RECIPIENTS', [])
            
            if not sender_email or not sender_password or not recipient_emails:
                logger.warning("[UNIFIED_SYSTEM] Email configuration incomplete")
                return
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ", ".join(recipient_emails)
            msg['Subject'] = f"CRITICAL ALERT: {alert.title}"
            
            body = f"""
            SolTrader System Alert

            Alert Level: {alert.level.value.upper()}
            Component: {alert.component}
            Title: {alert.title}
            Message: {alert.message}
            Timestamp: {alert.timestamp}

            Metadata: {json.dumps(alert.metadata, indent=2)}

            Please check the system immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            logger.info(f"[UNIFIED_SYSTEM] Critical alert email sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"[UNIFIED_SYSTEM] Email alert failed: {e}")
    
    # Public API methods
    def add_alert_callback(self, callback: callable):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: callable):
        """Remove alert callback"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Latest system metrics
            latest_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
            
            # Component health summary
            component_summary = {}
            for name, health in self.components.items():
                component_summary[name] = {
                    'status': health.status.value,
                    'last_check': health.last_check.isoformat(),
                    'error_count': health.error_count,
                    'response_time_ms': health.response_time_ms,
                    'performance_score': health.custom_metrics.get('performance_score', 0)
                }
            
            # Alert summary
            alert_summary = {
                'active_alerts': len(self.active_alerts),
                'critical_alerts': len([a for a in self.active_alerts.values() if a.level == AlertLevel.CRITICAL]),
                'warning_alerts': len([a for a in self.active_alerts.values() if a.level == AlertLevel.WARNING]),
                'recent_alerts': [
                    {
                        'id': a.id,
                        'level': a.level.value,
                        'title': a.title,
                        'component': a.component,
                        'timestamp': a.timestamp.isoformat()
                    }
                    for a in sorted(self.alert_history[-10:], key=lambda x: x.timestamp, reverse=True)
                ]
            }
            
            # Overall system health
            overall_health = "healthy"
            if any(h.status == HealthStatus.CRITICAL for h in self.components.values()):
                overall_health = "critical"
            elif any(h.status == HealthStatus.WARNING for h in self.components.values()):
                overall_health = "warning"
            
            return {
                'status': overall_health,
                'timestamp': datetime.now().isoformat(),
                'system_metrics': asdict(latest_metrics) if latest_metrics else None,
                'components': component_summary,
                'alerts': alert_summary,
                'prometheus_enabled': self.prometheus_enabled,
                'prometheus_port': self.prometheus_port if self.prometheus_enabled else None,
                'uptime_seconds': (datetime.now() - self.startup_time).total_seconds() if hasattr(self, 'startup_time') else 0
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def get_performance_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter metrics by time period
            recent_metrics = [
                m for m in self.system_metrics_history 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {'error': 'No metrics available for specified time period'}
            
            # Calculate statistics
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]
            disk_values = [m.disk_percent for m in recent_metrics]
            
            return {
                'time_period_hours': hours,
                'metrics_count': len(recent_metrics),
                'cpu': {
                    'average': sum(cpu_values) / len(cpu_values),
                    'maximum': max(cpu_values),
                    'minimum': min(cpu_values)
                },
                'memory': {
                    'average': sum(memory_values) / len(memory_values),
                    'maximum': max(memory_values),
                    'minimum': min(memory_values)
                },
                'disk': {
                    'average': sum(disk_values) / len(disk_values),
                    'maximum': max(disk_values),
                    'minimum': min(disk_values)
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def shutdown(self):
        """Graceful shutdown of system manager"""
        try:
            logger.info("[UNIFIED_SYSTEM] Shutting down Unified System Manager...")
            
            # Signal shutdown to background tasks
            self.shutdown_event.set()
            
            # Wait for background tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Save final state
            if self.db_initialized:
                # Final database cleanup/optimization
                pass
            
            logger.info("[UNIFIED_SYSTEM] Unified System Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"[UNIFIED_SYSTEM] Shutdown error: {e}")

# Initialize startup time
UnifiedSystemManager.startup_time = datetime.now()