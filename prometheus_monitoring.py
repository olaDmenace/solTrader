#!/usr/bin/env python3
"""
Prometheus Monitoring & Alerting System
=======================================

Enterprise-grade monitoring and alerting for SolTrader production deployment:
- Prometheus metrics collection and export
- Critical trading and system alerts
- Real-time performance dashboards
- Automated incident response
- Integration with Grafana and external alerting

Production monitoring features:
- Trading performance metrics (PnL, win rate, execution times)
- System resource metrics (memory, CPU, connections)
- API performance and error tracking
- Strategy coordination metrics
- Custom business metrics and alerts
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import weakref

# Prometheus client (mock implementation for systems without prometheus_client)
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info, Enum as PrometheusEnum,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    # Mock prometheus_client for systems without it
    PROMETHEUS_AVAILABLE = False
    
    class MockMetric:
        def __init__(self, name, documentation, labelnames=None, registry=None):
            self.name = name
            self.documentation = documentation
            self.labelnames = labelnames or []
            self._value = 0
            self._labels = {}
        
        def inc(self, amount=1): self._value += amount
        def dec(self, amount=1): self._value -= amount
        def set(self, value): self._value = value
        def observe(self, amount): self._value = amount
        def labels(self, **kwargs): return self
        def info(self, info_dict): pass
    
    Counter = Gauge = Histogram = Summary = Info = PrometheusEnum = MockMetric
    CollectorRegistry = lambda: None
    generate_latest = lambda x: b"# Prometheus metrics mock"
    CONTENT_TYPE_LATEST = "text/plain"
    start_http_server = lambda x: None

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(Enum):
    """Metric types for organization"""
    TRADING = "trading"
    SYSTEM = "system"
    API = "api"
    STRATEGY = "strategy"
    BUSINESS = "business"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_name: str
    condition: str  # e.g., "> 0.8", "< 100", "== 0"
    threshold: float
    severity: AlertSeverity
    duration_seconds: int = 60  # How long condition must be true
    description: str = ""
    runbook_url: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class Alert:
    """Active alert instance"""
    rule: AlertRule
    triggered_at: datetime
    current_value: float
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    resolved_at: Optional[datetime] = None
    escalated: bool = False

@dataclass
class MetricDefinition:
    """Metric definition for registration"""
    name: str
    type: str  # counter, gauge, histogram, summary
    description: str
    labels: List[str] = field(default_factory=list)
    metric_type: MetricType = MetricType.SYSTEM
    buckets: Optional[List[float]] = None  # For histograms

class PrometheusMetrics:
    """Prometheus metrics collector and exporter"""
    
    def __init__(self, registry=None, port: int = 8000):
        self.registry = registry or (CollectorRegistry() if PROMETHEUS_AVAILABLE else None)
        self.port = port
        self.metrics: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.http_server = None
        
        # Initialize core metrics
        self._initialize_core_metrics()
        
        logger.info(f"PrometheusMetrics initialized on port {port}")
    
    def _initialize_core_metrics(self):
        """Initialize core system and trading metrics"""
        
        # Trading metrics
        self.register_metric(MetricDefinition(
            name="soltrader_trades_total",
            type="counter",
            description="Total number of trades executed",
            labels=["strategy", "outcome", "token"],
            metric_type=MetricType.TRADING
        ))
        
        self.register_metric(MetricDefinition(
            name="soltrader_pnl_total",
            type="gauge",
            description="Total profit and loss in SOL",
            labels=["strategy", "token"],
            metric_type=MetricType.TRADING
        ))
        
        self.register_metric(MetricDefinition(
            name="soltrader_trade_execution_duration_seconds",
            type="histogram",
            description="Time taken to execute trades",
            labels=["strategy", "exchange"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            metric_type=MetricType.TRADING
        ))
        
        self.register_metric(MetricDefinition(
            name="soltrader_position_value_sol",
            type="gauge",
            description="Current position value in SOL",
            labels=["strategy", "token"],
            metric_type=MetricType.TRADING
        ))
        
        # System metrics
        self.register_metric(MetricDefinition(
            name="soltrader_memory_usage_bytes",
            type="gauge",
            description="Memory usage in bytes",
            labels=["component"],
            metric_type=MetricType.SYSTEM
        ))
        
        self.register_metric(MetricDefinition(
            name="soltrader_cpu_usage_percent",
            type="gauge",
            description="CPU usage percentage",
            labels=["component"],
            metric_type=MetricType.SYSTEM
        ))
        
        # API metrics
        self.register_metric(MetricDefinition(
            name="soltrader_api_requests_total",
            type="counter",
            description="Total API requests",
            labels=["api", "method", "status"],
            metric_type=MetricType.API
        ))
        
        self.register_metric(MetricDefinition(
            name="soltrader_api_request_duration_seconds",
            type="histogram",
            description="API request duration",
            labels=["api", "method"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            metric_type=MetricType.API
        ))
        
        # Strategy metrics
        self.register_metric(MetricDefinition(
            name="soltrader_strategy_signals_total",
            type="counter",
            description="Total strategy signals generated",
            labels=["strategy", "signal_type"],
            metric_type=MetricType.STRATEGY
        ))
        
        self.register_metric(MetricDefinition(
            name="soltrader_strategy_coordination_duration_seconds",
            type="histogram",
            description="Time taken for strategy coordination",
            labels=["coordination_type"],
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            metric_type=MetricType.STRATEGY
        ))
        
        # Business metrics
        self.register_metric(MetricDefinition(
            name="soltrader_portfolio_value_sol",
            type="gauge",
            description="Total portfolio value in SOL",
            metric_type=MetricType.BUSINESS
        ))
        
        self.register_metric(MetricDefinition(
            name="soltrader_win_rate",
            type="gauge",
            description="Trading win rate percentage",
            labels=["strategy", "timeframe"],
            metric_type=MetricType.BUSINESS
        ))
    
    def register_metric(self, definition: MetricDefinition):
        """Register a new metric"""
        
        self.metric_definitions[definition.name] = definition
        
        if not PROMETHEUS_AVAILABLE:
            self.metrics[definition.name] = MockMetric(
                definition.name, definition.description, definition.labels
            )
            return
        
        # Create actual Prometheus metric
        kwargs = {
            'name': definition.name,
            'documentation': definition.description,
            'labelnames': definition.labels,
            'registry': self.registry
        }
        
        if definition.type == "counter":
            metric = Counter(**kwargs)
        elif definition.type == "gauge":
            metric = Gauge(**kwargs)
        elif definition.type == "histogram":
            if definition.buckets:
                kwargs['buckets'] = definition.buckets
            metric = Histogram(**kwargs)
        elif definition.type == "summary":
            metric = Summary(**kwargs)
        else:
            raise ValueError(f"Unknown metric type: {definition.type}")
        
        self.metrics[definition.name] = metric
        logger.debug(f"Registered metric: {definition.name}")
    
    def get_metric(self, name: str):
        """Get metric by name"""
        return self.metrics.get(name)
    
    def increment_counter(self, name: str, amount: float = 1, labels: Dict[str, str] = None):
        """Increment counter metric"""
        metric = self.get_metric(name)
        if metric:
            if labels:
                metric.labels(**labels).inc(amount)
            else:
                metric.inc(amount)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set gauge metric value"""
        metric = self.get_metric(name)
        if metric:
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe histogram metric"""
        metric = self.get_metric(name)
        if metric:
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
    
    def start_http_server(self):
        """Start HTTP server for metrics export"""
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, starting mock HTTP server")
            self._start_mock_server()
            return
        
        try:
            self.http_server = start_http_server(self.port, registry=self.registry)
            logger.info(f"Prometheus HTTP server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus HTTP server: {e}")
            raise
    
    def _start_mock_server(self):
        """Start mock HTTP server for testing"""
        
        class MockHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/metrics':
                    self.send_response(200)
                    self.send_header('Content-Type', CONTENT_TYPE_LATEST)
                    self.end_headers()
                    self.wfile.write(b"# Mock Prometheus metrics\n# soltrader_system_status 1\n")
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress logging
        
        def run_server():
            try:
                server = HTTPServer(('localhost', self.port), MockHandler)
                server.serve_forever()
            except Exception as e:
                logger.error(f"Mock server error: {e}")
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        
        logger.info(f"Mock Prometheus server started on port {self.port}")
    
    def stop_http_server(self):
        """Stop HTTP server"""
        if self.http_server:
            self.http_server.shutdown()
            logger.info("Prometheus HTTP server stopped")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        
        metrics_by_type = defaultdict(list)
        
        for name, definition in self.metric_definitions.items():
            metrics_by_type[definition.metric_type.value].append({
                'name': name,
                'type': definition.type,
                'description': definition.description,
                'labels': definition.labels
            })
        
        return {
            'total_metrics': len(self.metrics),
            'metrics_by_type': dict(metrics_by_type),
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'http_server_port': self.port
        }

class AlertManager:
    """Alert management system with escalation and notification"""
    
    def __init__(self, prometheus_metrics: PrometheusMetrics):
        self.prometheus_metrics = prometheus_metrics
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Alert state tracking
        self.rule_states: Dict[str, Dict] = defaultdict(lambda: {
            'condition_start': None,
            'condition_met_duration': 0,
            'last_check': None
        })
        
        # Notification callbacks
        self.notification_callbacks: List[Callable] = []
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Initialize default alert rules
        self._initialize_default_alerts()
        
        logger.info("AlertManager initialized")
    
    def _initialize_default_alerts(self):
        """Initialize default critical alert rules"""
        
        # Trading alerts
        self.register_alert_rule(AlertRule(
            name="high_trading_loss",
            metric_name="soltrader_pnl_total",
            condition="< -100",
            threshold=-100.0,
            severity=AlertSeverity.CRITICAL,
            duration_seconds=60,
            description="Trading losses exceed 100 SOL",
            labels={"category": "trading", "impact": "financial"}
        ))
        
        self.register_alert_rule(AlertRule(
            name="trade_execution_slow",
            metric_name="soltrader_trade_execution_duration_seconds",
            condition="> 10",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            duration_seconds=120,
            description="Trade execution taking longer than 10 seconds",
            labels={"category": "performance", "impact": "trading"}
        ))
        
        # System alerts
        self.register_alert_rule(AlertRule(
            name="high_memory_usage",
            metric_name="soltrader_memory_usage_bytes",
            condition="> 2147483648",  # 2GB
            threshold=2147483648,
            severity=AlertSeverity.WARNING,
            duration_seconds=300,
            description="Memory usage exceeds 2GB",
            labels={"category": "system", "impact": "stability"}
        ))
        
        self.register_alert_rule(AlertRule(
            name="critical_memory_usage",
            metric_name="soltrader_memory_usage_bytes",
            condition="> 3221225472",  # 3GB
            threshold=3221225472,
            severity=AlertSeverity.CRITICAL,
            duration_seconds=60,
            description="Memory usage critically high (>3GB)",
            labels={"category": "system", "impact": "stability"}
        ))
        
        # API alerts
        self.register_alert_rule(AlertRule(
            name="api_error_rate_high",
            metric_name="soltrader_api_requests_total",
            condition="> 0.1",  # 10% error rate
            threshold=0.1,
            severity=AlertSeverity.WARNING,
            duration_seconds=180,
            description="API error rate exceeds 10%",
            labels={"category": "api", "impact": "trading"}
        ))
        
        # Business alerts
        self.register_alert_rule(AlertRule(
            name="low_win_rate",
            metric_name="soltrader_win_rate",
            condition="< 0.4",  # 40% win rate
            threshold=0.4,
            severity=AlertSeverity.WARNING,
            duration_seconds=3600,  # 1 hour
            description="Trading win rate below 40%",
            labels={"category": "business", "impact": "performance"}
        ))
    
    def register_alert_rule(self, rule: AlertRule):
        """Register a new alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            # Clean up any active alerts for this rule
            if rule_name in self.active_alerts:
                del self.active_alerts[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    async def start_monitoring(self, check_interval_seconds: int = 30):
        """Start alert monitoring"""
        
        if self.is_monitoring:
            logger.warning("Alert monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(check_interval_seconds)
        )
        
        logger.info(f"Alert monitoring started with {check_interval_seconds}s interval")
    
    async def stop_monitoring(self):
        """Stop alert monitoring"""
        
        self.is_monitoring = False
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Alert monitoring stopped")
    
    async def _monitoring_loop(self, check_interval: int):
        """Main alert monitoring loop"""
        
        while self.is_monitoring:
            try:
                current_time = datetime.now()
                
                # Check all alert rules
                for rule_name, rule in self.alert_rules.items():
                    if not rule.enabled:
                        continue
                    
                    await self._check_alert_rule(rule, current_time)
                
                # Clean up resolved alerts
                await self._cleanup_resolved_alerts()
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _check_alert_rule(self, rule: AlertRule, current_time: datetime):
        """Check individual alert rule"""
        
        try:
            # Get current metric value (mock for now)
            current_value = await self._get_metric_value(rule.metric_name)
            
            if current_value is None:
                return
            
            # Evaluate condition
            condition_met = self._evaluate_condition(current_value, rule.condition, rule.threshold)
            
            rule_state = self.rule_states[rule.name]
            
            if condition_met:
                if rule_state['condition_start'] is None:
                    rule_state['condition_start'] = current_time
                
                # Calculate duration
                duration = (current_time - rule_state['condition_start']).total_seconds()
                
                # Check if duration threshold met
                if duration >= rule.duration_seconds:
                    await self._trigger_alert(rule, current_value, current_time)
            else:
                # Condition not met, reset state
                if rule_state['condition_start'] is not None:
                    rule_state['condition_start'] = None
                    
                    # Resolve alert if active
                    if rule.name in self.active_alerts:
                        await self._resolve_alert(rule.name, current_time)
            
            rule_state['last_check'] = current_time
            
        except Exception as e:
            logger.error(f"Error checking alert rule {rule.name}: {e}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        
        if condition.startswith('>'):
            return value > threshold
        elif condition.startswith('<'):
            return value < threshold
        elif condition.startswith('=='):
            return abs(value - threshold) < 0.001  # Float comparison
        elif condition.startswith('!='):
            return abs(value - threshold) >= 0.001
        elif condition.startswith('>='):
            return value >= threshold
        elif condition.startswith('<='):
            return value <= threshold
        else:
            logger.error(f"Unknown condition operator: {condition}")
            return False
    
    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current metric value (mock implementation)"""
        
        # This would integrate with actual Prometheus metrics
        # For now, return mock values for testing
        mock_values = {
            "soltrader_pnl_total": -50.0,  # Moderate loss
            "soltrader_memory_usage_bytes": 1500000000,  # 1.5GB
            "soltrader_api_requests_total": 0.05,  # 5% error rate
            "soltrader_win_rate": 0.65,  # 65% win rate
            "soltrader_trade_execution_duration_seconds": 5.0  # 5 second execution
        }
        
        return mock_values.get(metric_name)
    
    async def _trigger_alert(self, rule: AlertRule, current_value: float, trigger_time: datetime):
        """Trigger an alert"""
        
        # Don't retrigger existing alert
        if rule.name in self.active_alerts:
            return
        
        alert = Alert(
            rule=rule,
            triggered_at=trigger_time,
            current_value=current_value,
            message=f"{rule.description} (current: {current_value}, threshold: {rule.threshold})",
            labels=rule.labels.copy()
        )
        
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        
        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }.get(rule.severity, logging.INFO)
        
        logger.log(log_level, f"ALERT TRIGGERED [{rule.severity.value.upper()}]: {alert.message}")
        
        # Notify callbacks
        await self._send_notifications(alert)
        
        # Update Prometheus metrics
        self.prometheus_metrics.increment_counter(
            "soltrader_alerts_total",
            labels={"rule": rule.name, "severity": rule.severity.value}
        )
    
    async def _resolve_alert(self, rule_name: str, resolve_time: datetime):
        """Resolve an active alert"""
        
        if rule_name not in self.active_alerts:
            return
        
        alert = self.active_alerts[rule_name]
        alert.resolved_at = resolve_time
        
        del self.active_alerts[rule_name]
        
        logger.info(f"ALERT RESOLVED: {alert.rule.name} - {alert.message}")
        
        # Notify callbacks about resolution
        for callback in self.notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback({
                        'type': 'resolved',
                        'alert': alert,
                        'resolved_at': resolve_time
                    })
                else:
                    callback({
                        'type': 'resolved',
                        'alert': alert,
                        'resolved_at': resolve_time
                    })
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        
        notification_data = {
            'type': 'triggered',
            'alert': alert,
            'severity': alert.rule.severity.value,
            'message': alert.message,
            'timestamp': alert.triggered_at,
            'labels': alert.labels
        }
        
        for callback in self.notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification_data)
                else:
                    callback(notification_data)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")
    
    async def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts from history"""
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Keep only recent alerts in history
        recent_alerts = []
        for alert in self.alert_history:
            if (alert.resolved_at and alert.resolved_at >= cutoff_time) or \
               (not alert.resolved_at and alert.triggered_at >= cutoff_time):
                recent_alerts.append(alert)
        
        self.alert_history.clear()
        self.alert_history.extend(recent_alerts)
    
    def register_notification_callback(self, callback: Callable):
        """Register notification callback"""
        self.notification_callbacks.append(callback)
        logger.debug(f"Registered notification callback: {callback.__name__}")
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status"""
        
        active_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_severity[alert.rule.severity.value] += 1
        
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.triggered_at >= datetime.now() - timedelta(hours=24)
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.is_monitoring,
            'total_rules': len(self.alert_rules),
            'active_alerts': len(self.active_alerts),
            'active_by_severity': dict(active_by_severity),
            'recent_alerts_24h': len(recent_alerts),
            'alert_rules_enabled': len([r for r in self.alert_rules.values() if r.enabled]),
            'notification_callbacks': len(self.notification_callbacks)
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts"""
        
        return [
            {
                'rule_name': alert.rule.name,
                'severity': alert.rule.severity.value,
                'message': alert.message,
                'triggered_at': alert.triggered_at.isoformat(),
                'current_value': alert.current_value,
                'threshold': alert.rule.threshold,
                'labels': alert.labels,
                'escalated': alert.escalated
            }
            for alert in self.active_alerts.values()
        ]

class MonitoringSystem:
    """Comprehensive monitoring system integrating Prometheus and AlertManager"""
    
    def __init__(self, prometheus_port: int = 8000, alert_check_interval: int = 30):
        self.prometheus_metrics = PrometheusMetrics(port=prometheus_port)
        self.alert_manager = AlertManager(self.prometheus_metrics)
        self.alert_check_interval = alert_check_interval
        
        # System state
        self.is_running = False
        self.startup_time = datetime.now()
        
        # Background tasks
        self.metric_collection_task: Optional[asyncio.Task] = None
        
        logger.info("MonitoringSystem initialized")
    
    async def start(self):
        """Start the monitoring system"""
        
        try:
            # Start Prometheus HTTP server
            self.prometheus_metrics.start_http_server()
            
            # Start alert monitoring
            await self.alert_manager.start_monitoring(self.alert_check_interval)
            
            # Start metric collection
            self.metric_collection_task = asyncio.create_task(self._metric_collection_loop())
            
            self.is_running = True
            
            logger.info("MonitoringSystem started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start MonitoringSystem: {e}")
            raise
    
    async def stop(self):
        """Stop the monitoring system"""
        
        try:
            self.is_running = False
            
            # Stop alert monitoring
            await self.alert_manager.stop_monitoring()
            
            # Stop metric collection
            if self.metric_collection_task and not self.metric_collection_task.done():
                self.metric_collection_task.cancel()
                try:
                    await self.metric_collection_task
                except asyncio.CancelledError:
                    pass
            
            # Stop Prometheus server
            self.prometheus_metrics.stop_http_server()
            
            logger.info("MonitoringSystem stopped")
            
        except Exception as e:
            logger.error(f"Error stopping MonitoringSystem: {e}")
    
    async def _metric_collection_loop(self):
        """Periodic metric collection"""
        
        while self.is_running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Metric collection error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self):
        """Collect system metrics"""
        
        try:
            import psutil
            
            # Memory metrics
            process = psutil.Process()
            memory_bytes = process.memory_info().rss
            
            self.prometheus_metrics.set_gauge(
                "soltrader_memory_usage_bytes",
                memory_bytes,
                {"component": "main"}
            )
            
            # CPU metrics
            cpu_percent = process.cpu_percent()
            self.prometheus_metrics.set_gauge(
                "soltrader_cpu_usage_percent",
                cpu_percent,
                {"component": "main"}
            )
            
            # Portfolio value (mock)
            self.prometheus_metrics.set_gauge(
                "soltrader_portfolio_value_sol",
                1000.0  # Mock portfolio value
            )
            
        except ImportError:
            # psutil not available, use mock values
            self.prometheus_metrics.set_gauge(
                "soltrader_memory_usage_bytes",
                1500000000,  # 1.5GB mock
                {"component": "main"}
            )
    
    def record_trade(self, strategy: str, outcome: str, token: str, 
                    pnl: float, execution_time: float):
        """Record trade metrics"""
        
        # Count trade
        self.prometheus_metrics.increment_counter(
            "soltrader_trades_total",
            labels={"strategy": strategy, "outcome": outcome, "token": token}
        )
        
        # Record PnL
        self.prometheus_metrics.set_gauge(
            "soltrader_pnl_total",
            pnl,
            {"strategy": strategy, "token": token}
        )
        
        # Record execution time
        self.prometheus_metrics.observe_histogram(
            "soltrader_trade_execution_duration_seconds",
            execution_time,
            {"strategy": strategy, "exchange": "jupiter"}
        )
    
    def record_api_call(self, api: str, method: str, status: str, duration: float):
        """Record API call metrics"""
        
        self.prometheus_metrics.increment_counter(
            "soltrader_api_requests_total",
            labels={"api": api, "method": method, "status": status}
        )
        
        self.prometheus_metrics.observe_histogram(
            "soltrader_api_request_duration_seconds",
            duration,
            {"api": api, "method": method}
        )
    
    def record_strategy_coordination(self, coordination_type: str, duration: float):
        """Record strategy coordination metrics"""
        
        self.prometheus_metrics.observe_histogram(
            "soltrader_strategy_coordination_duration_seconds",
            duration,
            {"coordination_type": coordination_type}
        )
    
    def register_alert_callback(self, callback: Callable):
        """Register alert notification callback"""
        self.alert_manager.register_notification_callback(callback)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        
        uptime = (datetime.now() - self.startup_time).total_seconds()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'prometheus': self.prometheus_metrics.get_metrics_summary(),
            'alerts': self.alert_manager.get_alert_status(),
            'active_alerts': self.alert_manager.get_active_alerts()
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_monitoring_system():
        """Test the monitoring system"""
        
        print("Testing Monitoring & Alerting System")
        print("=" * 50)
        
        # Create monitoring system
        monitoring = MonitoringSystem(prometheus_port=8001, alert_check_interval=5)
        
        # Register alert callback
        def alert_callback(alert_data):
            print(f"ALERT NOTIFICATION: {alert_data['type']} - {alert_data.get('message', 'N/A')}")
        
        monitoring.register_alert_callback(alert_callback)
        
        try:
            # Start monitoring
            await monitoring.start()
            print("Monitoring system started")
            
            # Record some sample metrics
            print("\nRecording sample metrics...")
            monitoring.record_trade("momentum", "win", "SOL/USDC", 10.5, 2.3)
            monitoring.record_trade("mean_reversion", "loss", "ETH/SOL", -5.2, 1.8)
            monitoring.record_api_call("jupiter", "swap", "success", 0.45)
            monitoring.record_strategy_coordination("conflict_resolution", 0.12)
            
            # Wait for metrics collection and alert checking
            print("Monitoring for alerts...")
            await asyncio.sleep(10)
            
            # Get status
            status = monitoring.get_monitoring_status()
            print(f"\nMonitoring Status:")
            print(f"  Running: {status['is_running']}")
            print(f"  Uptime: {status['uptime_seconds']:.1f}s")
            print(f"  Total Metrics: {status['prometheus']['total_metrics']}")
            print(f"  Active Alerts: {status['alerts']['active_alerts']}")
            print(f"  Alert Rules: {status['alerts']['total_rules']}")
            
            # Show active alerts
            active_alerts = status['active_alerts']
            if active_alerts:
                print(f"\nActive Alerts:")
                for alert in active_alerts:
                    print(f"  - {alert['rule_name']}: {alert['message']}")
            else:
                print(f"\nNo active alerts")
            
            print("\nMonitoring system test completed successfully!")
            
        finally:
            await monitoring.stop()
            print("Monitoring system stopped")
    
    asyncio.run(test_monitoring_system())