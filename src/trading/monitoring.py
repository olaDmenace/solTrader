# src/trading/monitoring.py

import logging
from typing import Dict, Any, Optional, List, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"

@dataclass
class Metric:
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    severity: str
    message: str
    metric: str
    threshold: float
    current_value: float
    timestamp: datetime

class MonitoringSystem:
    def __init__(self, alert_system: Any, settings: Any):
        self.alert_system = alert_system
        self.settings = settings
        self.metrics: Dict[str, Deque[Metric]] = {}
        self.alerts: Deque[Alert] = deque(maxlen=1000)
        self._monitor_task: Optional[asyncio.Task] = None
        self.window_size = timedelta(hours=24)
        self.update_interval = 60  # seconds
        
        # Initialize metric storage
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize metric collectors"""
        metric_configs = {
            'trade_count': MetricType.COUNTER,
            'position_count': MetricType.GAUGE,
            'portfolio_value': MetricType.GAUGE,
            'unrealized_pnl': MetricType.GAUGE,
            'realized_pnl': MetricType.GAUGE,
            'win_rate': MetricType.GAUGE,
            'avg_trade_duration': MetricType.HISTOGRAM,
            'trade_frequency': MetricType.GAUGE,
            'drawdown': MetricType.GAUGE,
            'volatility': MetricType.GAUGE,
            'api_latency': MetricType.HISTOGRAM,
            'error_rate': MetricType.GAUGE,
            'slippage': MetricType.HISTOGRAM
        }
        
        for name, type_ in metric_configs.items():
            self.metrics[name] = deque(maxlen=1000)

    async def start_monitoring(self) -> None:
        """Start the monitoring system"""
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                await asyncio.sleep(5)

    async def _collect_metrics(self) -> None:
        """Collect current system metrics"""
        timestamp = datetime.now()
        
        # Update metrics safely - check if position manager exists
        position_manager = getattr(self.settings, 'position_manager', None)
        if position_manager and hasattr(position_manager, 'positions'):
            self.update_metric('position_count', len(position_manager.positions))
        else:
            self.update_metric('position_count', 0)
            
        self.update_metric('portfolio_value', await self._get_portfolio_value())
        self.update_metric('error_rate', self._calculate_error_rate())
        self.update_metric('drawdown', self._calculate_drawdown())
        self.update_metric('volatility', self._calculate_volatility())

    async def _check_alerts(self) -> None:
        """Check for alert conditions"""
        checks = {
            'drawdown': (self.settings.MAX_DRAWDOWN, 'high'),
            'error_rate': (0.1, 'high'),
            'portfolio_value': (self.settings.MIN_PORTFOLIO_VALUE, 'low'),
            'volatility': (self.settings.MAX_VOLATILITY, 'high'),
        }
        
        for metric, (threshold, condition) in checks.items():
            current = self.get_current_value(metric)
            if current is None:
                continue
                
            if condition == 'high' and current > threshold:
                await self._create_alert('warning', metric, threshold, current)
            elif condition == 'low' and current < threshold:
                await self._create_alert('warning', metric, threshold, current)

    async def _create_alert(self, severity: str, metric: str, 
                          threshold: float, current: float) -> None:
        """Create and record alert"""
        alert = Alert(
            severity=severity,
            message=f"{metric} threshold breached",
            metric=metric,
            threshold=threshold,
            current_value=current,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        await self.alert_system.emit_alert(
            level=severity,
            type="threshold_breach",
            message=alert.message,
            data={
                "metric": metric,
                "threshold": threshold,
                "current_value": current,
                "timestamp": alert.timestamp.isoformat()
            }
        )

    def update_metric(self, name: str, value: float, 
                     labels: Optional[Dict[str, str]] = None) -> None:
        """Update metric value"""
        if name not in self.metrics:
            return
            
        metric = Metric(
            name=name,
            type=MetricType.GAUGE,  # Default to gauge
            value=value,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        self.metrics[name].append(metric)

    def get_current_value(self, metric_name: str) -> Optional[float]:
        """Get current value for metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        return self.metrics[metric_name][-1].value

    def get_metric_history(self, metric_name: str, 
                     window: Optional[timedelta] = None) -> List[Metric]:
        if metric_name not in self.metrics:
            return []
            
        window = window or self.window_size
        cutoff = datetime.now() - window
        return [m for m in self.metrics[metric_name] if m.timestamp > cutoff]

    async def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        value = 0.0
        position_manager = getattr(self.settings, 'position_manager', None)
        if position_manager and hasattr(position_manager, 'positions'):
            for position in position_manager.positions.values():
                value += position.size * position.current_price
        return value

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        window = timedelta(minutes=5)
        total_ops = 0
        errors = 0
        
        cutoff = datetime.now() - window
        for metric in self.metrics.get('error_count', []):
            if metric.timestamp > cutoff:
                errors += metric.value
                
        for metric in self.metrics.get('operation_count', []):
            if metric.timestamp > cutoff:
                total_ops += metric.value
                
        return errors / total_ops if total_ops > 0 else 0.0

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        values = [m.value for m in self.metrics.get('portfolio_value', [])]
        if not values:
            return 0.0
            
        peak = max(values)
        current = values[-1]
        return (peak - current) / peak if peak > 0 else 0.0

    def _calculate_volatility(self) -> float:
        """Calculate portfolio volatility"""
        values = [m.value for m in self.metrics.get('portfolio_value', [])]
        if len(values) < 2:
            return 0.0
            
        returns = np.diff(values) / values[:-1]
        return float(np.std(returns) * np.sqrt(365))

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            'portfolio': {
                'value': self.get_current_value('portfolio_value'),
                'drawdown': self.get_current_value('drawdown'),
                'volatility': self.get_current_value('volatility')
            },
            'trading': {
                'win_rate': self.get_current_value('win_rate'),
                'trade_count': self.get_current_value('trade_count'),
                'avg_trade_duration': self.get_current_value('avg_trade_duration')
            },
            'risk': {
                'error_rate': self.get_current_value('error_rate'),
                'api_latency': self.get_current_value('api_latency'),
                'slippage': self.get_current_value('slippage')
            }
        }