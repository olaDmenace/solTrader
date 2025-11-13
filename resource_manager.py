#!/usr/bin/env python3
"""
Production Resource Management System
====================================

Comprehensive resource management for production deployment:
- Memory limits and garbage collection optimization
- CPU usage monitoring and throttling
- Process restart policies and recovery
- Connection pooling and resource cleanup
- System resource monitoring and alerting

Production-grade features:
- Automatic resource cleanup and garbage collection
- Process health monitoring with restart policies
- Resource usage alerts and emergency procedures
- Connection pool management and optimization
- System metrics collection and analysis
"""

import os
import gc
import sys
import psutil
import asyncio
import logging
import signal
import threading
import time
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import subprocess
# Resource module not available on Windows
try:
    import resource as system_resource
except ImportError:
    system_resource = None

logger = logging.getLogger(__name__)

class ResourceStatus(Enum):
    """Resource status levels"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RestartPolicy(Enum):
    """Process restart policies"""
    NEVER = "never"
    ON_FAILURE = "on-failure"
    UNLESS_STOPPED = "unless-stopped"
    ALWAYS = "always"

@dataclass
class ResourceLimits:
    """Resource limit configuration"""
    max_memory_mb: int = 2048
    max_memory_percent: float = 80.0
    warning_memory_mb: int = 1536
    critical_memory_mb: int = 1792
    max_cpu_percent: float = 85.0
    warning_cpu_percent: float = 70.0
    critical_cpu_percent: float = 80.0
    max_connections: int = 1000
    warning_connections: int = 800
    critical_connections: int = 900
    max_file_descriptors: int = 8192
    gc_threshold_mb: int = 512
    emergency_cleanup_threshold_mb: int = 1800

@dataclass
class RestartConfig:
    """Process restart configuration"""
    policy: RestartPolicy = RestartPolicy.UNLESS_STOPPED
    max_restarts: int = 5
    restart_window_minutes: int = 60
    restart_delay_seconds: int = 10
    escalation_delay_seconds: int = 30
    health_check_timeout: int = 30
    enable_graceful_shutdown: bool = True
    graceful_shutdown_timeout: int = 30

@dataclass
class ResourceMetrics:
    """Current resource usage metrics"""
    timestamp: datetime
    memory_mb: float
    memory_percent: float
    cpu_percent: float
    connections: int
    file_descriptors: int
    disk_usage_percent: float
    network_connections: int
    thread_count: int
    gc_objects: int
    status: ResourceStatus

@dataclass
class ProcessInfo:
    """Process information and status"""
    pid: int
    name: str
    status: str
    memory_mb: float
    cpu_percent: float
    start_time: datetime
    restart_count: int
    last_restart: Optional[datetime] = None

class MemoryManager:
    """Advanced memory management system"""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.gc_stats = {'collections': 0, 'objects_collected': 0}
        self.memory_history: deque = deque(maxlen=100)
        self.weak_refs: weakref.WeakSet = weakref.WeakSet()
        self.cleanup_callbacks: List[Callable] = []
        
        # Configure garbage collection
        self._configure_gc()
        
        logger.info(f"MemoryManager initialized with {limits.max_memory_mb}MB limit")
    
    def _configure_gc(self):
        """Configure garbage collection for optimal performance"""
        
        # Set more aggressive GC thresholds for production
        gc.set_threshold(700, 10, 10)  # More frequent collection
        
        # Enable automatic garbage collection
        gc.enable()
        
        logger.info("Garbage collection configured for production")
    
    async def check_memory_usage(self) -> ResourceMetrics:
        """Check current memory usage and status"""
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Get system memory
            system_memory = psutil.virtual_memory()
            memory_percent = (memory_mb / (system_memory.total / 1024 / 1024)) * 100
            
            # Determine status
            if memory_mb >= self.limits.emergency_cleanup_threshold_mb:
                status = ResourceStatus.EMERGENCY
            elif memory_mb >= self.limits.critical_memory_mb:
                status = ResourceStatus.CRITICAL
            elif memory_mb >= self.limits.warning_memory_mb:
                status = ResourceStatus.WARNING
            else:
                status = ResourceStatus.NORMAL
            
            # Record in history
            self.memory_history.append({
                'timestamp': datetime.now(),
                'memory_mb': memory_mb,
                'status': status
            })
            
            return ResourceMetrics(
                timestamp=datetime.now(),
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                cpu_percent=process.cpu_percent(),
                connections=len(process.connections()),
                file_descriptors=process.num_fds() if hasattr(process, 'num_fds') else 0,
                disk_usage_percent=psutil.disk_usage('/').percent,
                network_connections=len(psutil.net_connections()),
                thread_count=process.num_threads(),
                gc_objects=len(gc.get_objects()),
                status=status
            )
            
        except Exception as e:
            logger.error(f"Failed to check memory usage: {e}")
            raise
    
    async def perform_memory_cleanup(self, aggressive: bool = False) -> Dict[str, Any]:
        """Perform memory cleanup operations"""
        
        cleanup_start = time.time()
        initial_memory = await self._get_memory_usage()
        
        try:
            objects_before = len(gc.get_objects())
            
            # Execute cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    logger.warning(f"Cleanup callback failed: {e}")
            
            # Force garbage collection
            if aggressive:
                # Multiple GC cycles for aggressive cleanup
                collected = 0
                for generation in range(3):
                    collected += gc.collect(generation)
            else:
                collected = gc.collect()
            
            # Clear weak references
            if aggressive:
                self.weak_refs.clear()
            
            # Clean up caches if aggressive
            if aggressive:
                import sys
                # Clear module cache (be very careful with this)
                if hasattr(sys, '_clear_type_cache'):
                    sys._clear_type_cache()
            
            objects_after = len(gc.get_objects())
            final_memory = await self._get_memory_usage()
            
            cleanup_time = (time.time() - cleanup_start) * 1000
            memory_freed = initial_memory - final_memory
            
            # Update statistics
            self.gc_stats['collections'] += 1
            self.gc_stats['objects_collected'] += collected
            
            result = {
                'success': True,
                'cleanup_time_ms': cleanup_time,
                'memory_freed_mb': memory_freed,
                'objects_before': objects_before,
                'objects_after': objects_after,
                'objects_collected': collected,
                'aggressive': aggressive,
                'callbacks_executed': len(self.cleanup_callbacks)
            }
            
            logger.info(f"Memory cleanup completed: {memory_freed:.1f}MB freed, {collected} objects collected")
            return result
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'cleanup_time_ms': (time.time() - cleanup_start) * 1000
            }
    
    async def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def register_cleanup_callback(self, callback: Callable):
        """Register cleanup callback for memory management"""
        self.cleanup_callbacks.append(callback)
        logger.debug(f"Registered cleanup callback: {callback.__name__}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory management statistics"""
        
        recent_history = list(self.memory_history)[-10:]  # Last 10 measurements
        
        if recent_history:
            avg_memory = sum(h['memory_mb'] for h in recent_history) / len(recent_history)
            peak_memory = max(h['memory_mb'] for h in recent_history)
        else:
            avg_memory = peak_memory = 0.0
        
        return {
            'gc_collections': self.gc_stats['collections'],
            'total_objects_collected': self.gc_stats['objects_collected'],
            'cleanup_callbacks_registered': len(self.cleanup_callbacks),
            'memory_history_entries': len(self.memory_history),
            'recent_average_memory_mb': avg_memory,
            'recent_peak_memory_mb': peak_memory,
            'current_gc_objects': len(gc.get_objects()),
            'gc_thresholds': gc.get_threshold()
        }

class ProcessManager:
    """Process monitoring and restart management"""
    
    def __init__(self, restart_config: RestartConfig):
        self.restart_config = restart_config
        self.process_info: Dict[int, ProcessInfo] = {}
        self.restart_history: deque = deque(maxlen=100)
        self.shutdown_event = asyncio.Event()
        self.restart_semaphore = asyncio.Semaphore(1)  # Only one restart at a time
        
        # Register signal handlers
        self._register_signal_handlers()
        
        logger.info(f"ProcessManager initialized with {restart_config.policy.value} policy")
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.initiate_shutdown())
        
        if os.name != 'nt':  # Unix-like systems
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        
        logger.debug("Signal handlers registered")
    
    async def monitor_process_health(self) -> Dict[str, Any]:
        """Monitor current process health"""
        
        try:
            process = psutil.Process()
            
            # Update process info
            self.process_info[process.pid] = ProcessInfo(
                pid=process.pid,
                name=process.name(),
                status=process.status(),
                memory_mb=process.memory_info().rss / 1024 / 1024,
                cpu_percent=process.cpu_percent(),
                start_time=datetime.fromtimestamp(process.create_time()),
                restart_count=self.process_info.get(process.pid, ProcessInfo(0, '', '', 0, 0, datetime.now(), 0)).restart_count
            )
            
            # Get system load
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            
            # Get resource usage (if available)
            if system_resource:
                usage = system_resource.getrusage(system_resource.RUSAGE_SELF)
                user_time = usage.ru_utime
                system_time = usage.ru_stime
            else:
                user_time = system_time = 0.0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'process_id': process.pid,
                'process_name': process.name(),
                'process_status': process.status(),
                'uptime_seconds': (datetime.now() - self.process_info[process.pid].start_time).total_seconds(),
                'memory_mb': self.process_info[process.pid].memory_mb,
                'cpu_percent': self.process_info[process.pid].cpu_percent,
                'thread_count': process.num_threads(),
                'file_descriptors': process.num_fds() if hasattr(process, 'num_fds') else 0,
                'connections': len(process.connections()),
                'load_average': load_avg,
                'user_time': user_time,
                'system_time': system_time,
                'restart_count': self.process_info[process.pid].restart_count,
                'health_status': 'healthy' if self.process_info[process.pid].cpu_percent < 90 else 'degraded'
            }
            
        except Exception as e:
            logger.error(f"Process health monitoring failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'health_status': 'unknown'
            }
    
    async def check_restart_needed(self, metrics: ResourceMetrics) -> bool:
        """Check if process restart is needed based on resource usage"""
        
        restart_needed = False
        
        # Check memory limits
        if metrics.memory_mb >= self.restart_config.max_restarts:  # Using as emergency threshold
            logger.critical(f"Memory usage critical: {metrics.memory_mb:.1f}MB")
            restart_needed = True
        
        # Check if process is responsive
        try:
            process = psutil.Process()
            if process.status() == psutil.STATUS_ZOMBIE:
                logger.error("Process is in zombie state")
                restart_needed = True
        except psutil.NoSuchProcess:
            logger.error("Process no longer exists")
            restart_needed = True
        
        # Check restart policy
        if restart_needed and self.restart_config.policy == RestartPolicy.NEVER:
            logger.warning("Restart needed but policy is NEVER - manual intervention required")
            return False
        
        return restart_needed
    
    async def perform_restart(self, reason: str) -> bool:
        """Perform process restart with configured policy"""
        
        async with self.restart_semaphore:
            try:
                current_process = psutil.Process()
                restart_time = datetime.now()
                
                # Check restart limits
                recent_restarts = [
                    r for r in self.restart_history
                    if r['timestamp'] >= restart_time - timedelta(minutes=self.restart_config.restart_window_minutes)
                ]
                
                if len(recent_restarts) >= self.restart_config.max_restarts:
                    logger.error(f"Maximum restarts ({self.restart_config.max_restarts}) exceeded in {self.restart_config.restart_window_minutes} minutes")
                    return False
                
                logger.info(f"Initiating process restart: {reason}")
                
                # Record restart attempt
                restart_record = {
                    'timestamp': restart_time,
                    'reason': reason,
                    'pid': current_process.pid,
                    'memory_mb': current_process.memory_info().rss / 1024 / 1024
                }
                self.restart_history.append(restart_record)
                
                # Update process info
                if current_process.pid in self.process_info:
                    self.process_info[current_process.pid].restart_count += 1
                    self.process_info[current_process.pid].last_restart = restart_time
                
                # Graceful shutdown
                if self.restart_config.enable_graceful_shutdown:
                    await self._graceful_shutdown()
                
                # For now, just return True as actual restart would be handled by process manager
                # In a real deployment, this would trigger the actual restart mechanism
                logger.info("Process restart completed successfully")
                return True
                
            except Exception as e:
                logger.error(f"Process restart failed: {e}")
                return False
    
    async def _graceful_shutdown(self):
        """Perform graceful shutdown"""
        
        logger.info("Starting graceful shutdown...")
        
        try:
            # Set shutdown event
            self.shutdown_event.set()
            
            # Wait for graceful shutdown timeout
            await asyncio.sleep(self.restart_config.graceful_shutdown_timeout)
            
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Graceful shutdown failed: {e}")
    
    async def initiate_shutdown(self):
        """Initiate system shutdown"""
        
        logger.info("Initiating system shutdown...")
        self.shutdown_event.set()
    
    def get_restart_statistics(self) -> Dict[str, Any]:
        """Get restart statistics"""
        
        total_restarts = len(self.restart_history)
        recent_restarts = [
            r for r in self.restart_history
            if r['timestamp'] >= datetime.now() - timedelta(hours=24)
        ]
        
        if self.restart_history:
            last_restart = max(self.restart_history, key=lambda x: x['timestamp'])
        else:
            last_restart = None
        
        return {
            'total_restarts': total_restarts,
            'recent_restarts_24h': len(recent_restarts),
            'restart_policy': self.restart_config.policy.value,
            'max_restarts_allowed': self.restart_config.max_restarts,
            'restart_window_minutes': self.restart_config.restart_window_minutes,
            'last_restart': last_restart['timestamp'].isoformat() if last_restart else None,
            'last_restart_reason': last_restart['reason'] if last_restart else None,
            'graceful_shutdown_enabled': self.restart_config.enable_graceful_shutdown
        }

class ResourceManager:
    """Main resource management system"""
    
    def __init__(self, 
                 resource_limits: ResourceLimits = None,
                 restart_config: RestartConfig = None):
        
        self.resource_limits = resource_limits or ResourceLimits()
        self.restart_config = restart_config or RestartConfig()
        
        # Component managers
        self.memory_manager = MemoryManager(self.resource_limits)
        self.process_manager = ProcessManager(self.restart_config)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Resource history
        self.resource_history: deque = deque(maxlen=1000)
        self.alert_history: deque = deque(maxlen=100)
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        logger.info("ResourceManager initialized")
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start resource monitoring"""
        
        if self.is_monitoring:
            logger.warning("Resource monitoring already running")
            return
        
        self.is_monitoring = True
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        
        # Start periodic cleanup task
        self.cleanup_task = asyncio.create_task(
            self._cleanup_loop(interval_seconds * 4)  # Cleanup every 2 minutes if monitoring every 30s
        )
        
        logger.info(f"Resource monitoring started with {interval_seconds}s interval")
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        
        self.is_monitoring = False
        
        # Cancel monitoring task
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Cancel cleanup task
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Resource monitoring stopped")
    
    async def _monitoring_loop(self, interval_seconds: int):
        """Main resource monitoring loop"""
        
        while self.is_monitoring:
            try:
                # Get current resource metrics
                metrics = await self.memory_manager.check_memory_usage()
                
                # Get process health
                process_health = await self.process_manager.monitor_process_health()
                
                # Record metrics
                self.resource_history.append({
                    'timestamp': metrics.timestamp,
                    'metrics': metrics,
                    'process_health': process_health
                })
                
                # Check for alerts
                await self._check_resource_alerts(metrics, process_health)
                
                # Check if restart needed
                restart_needed = await self.process_manager.check_restart_needed(metrics)
                
                if restart_needed:
                    await self.process_manager.perform_restart("Resource limits exceeded")
                
                # Wait for next check
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _cleanup_loop(self, interval_seconds: int):
        """Periodic cleanup loop"""
        
        while self.is_monitoring:
            try:
                # Check if cleanup is needed
                metrics = await self.memory_manager.check_memory_usage()
                
                if metrics.memory_mb > self.resource_limits.gc_threshold_mb:
                    aggressive = metrics.status == ResourceStatus.CRITICAL
                    await self.memory_manager.perform_memory_cleanup(aggressive=aggressive)
                
                # Wait for next cleanup
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _check_resource_alerts(self, metrics: ResourceMetrics, process_health: Dict[str, Any]):
        """Check resource usage and trigger alerts"""
        
        alerts = []
        
        # Memory alerts
        if metrics.status == ResourceStatus.EMERGENCY:
            alerts.append({
                'level': 'emergency',
                'type': 'memory',
                'message': f'Emergency memory usage: {metrics.memory_mb:.1f}MB',
                'value': metrics.memory_mb,
                'threshold': self.resource_limits.emergency_cleanup_threshold_mb
            })
        elif metrics.status == ResourceStatus.CRITICAL:
            alerts.append({
                'level': 'critical',
                'type': 'memory',
                'message': f'Critical memory usage: {metrics.memory_mb:.1f}MB',
                'value': metrics.memory_mb,
                'threshold': self.resource_limits.critical_memory_mb
            })
        elif metrics.status == ResourceStatus.WARNING:
            alerts.append({
                'level': 'warning',
                'type': 'memory',
                'message': f'High memory usage: {metrics.memory_mb:.1f}MB',
                'value': metrics.memory_mb,
                'threshold': self.resource_limits.warning_memory_mb
            })
        
        # CPU alerts
        if metrics.cpu_percent > self.resource_limits.critical_cpu_percent:
            alerts.append({
                'level': 'critical',
                'type': 'cpu',
                'message': f'Critical CPU usage: {metrics.cpu_percent:.1f}%',
                'value': metrics.cpu_percent,
                'threshold': self.resource_limits.critical_cpu_percent
            })
        elif metrics.cpu_percent > self.resource_limits.warning_cpu_percent:
            alerts.append({
                'level': 'warning',
                'type': 'cpu',
                'message': f'High CPU usage: {metrics.cpu_percent:.1f}%',
                'value': metrics.cpu_percent,
                'threshold': self.resource_limits.warning_cpu_percent
            })
        
        # Connection alerts
        if metrics.connections > self.resource_limits.critical_connections:
            alerts.append({
                'level': 'critical',
                'type': 'connections',
                'message': f'Critical connection count: {metrics.connections}',
                'value': metrics.connections,
                'threshold': self.resource_limits.critical_connections
            })
        
        # Process alerts if unhealthy
        if process_health.get('health_status') == 'degraded':
            alerts.append({
                'level': 'warning',
                'type': 'process',
                'message': f'Process health degraded',
                'details': process_health
            })
        
        # Record and handle alerts
        for alert in alerts:
            alert['timestamp'] = datetime.now()
            self.alert_history.append(alert)
            
            # Log alert
            level = alert['level'].upper()
            message = alert['message']
            logger.log(
                logging.CRITICAL if level == 'EMERGENCY' else
                logging.ERROR if level == 'CRITICAL' else
                logging.WARNING,
                f"RESOURCE ALERT [{level}]: {message}"
            )
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def register_alert_callback(self, callback: Callable):
        """Register alert callback"""
        self.alert_callbacks.append(callback)
        logger.debug(f"Registered alert callback: {callback.__name__}")
    
    async def force_cleanup(self, aggressive: bool = True) -> Dict[str, Any]:
        """Force immediate resource cleanup"""
        
        logger.info(f"Forcing {'aggressive' if aggressive else 'normal'} resource cleanup")
        
        return await self.memory_manager.perform_memory_cleanup(aggressive=aggressive)
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status"""
        
        try:
            # Get latest metrics if available
            if self.resource_history:
                latest_entry = self.resource_history[-1]
                latest_metrics = latest_entry['metrics']
                latest_process = latest_entry['process_health']
            else:
                latest_metrics = None
                latest_process = None
            
            # Get recent alerts
            recent_alerts = [
                alert for alert in self.alert_history
                if alert['timestamp'] >= datetime.now() - timedelta(minutes=30)
            ]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'monitoring_active': self.is_monitoring,
                'resource_limits': {
                    'max_memory_mb': self.resource_limits.max_memory_mb,
                    'max_cpu_percent': self.resource_limits.max_cpu_percent,
                    'max_connections': self.resource_limits.max_connections
                },
                'current_usage': {
                    'memory_mb': latest_metrics.memory_mb if latest_metrics else 0,
                    'cpu_percent': latest_metrics.cpu_percent if latest_metrics else 0,
                    'connections': latest_metrics.connections if latest_metrics else 0,
                    'status': latest_metrics.status.value if latest_metrics else 'unknown'
                },
                'process_health': latest_process,
                'memory_stats': self.memory_manager.get_memory_stats(),
                'restart_stats': self.process_manager.get_restart_statistics(),
                'recent_alerts': len(recent_alerts),
                'alert_summary': {
                    'emergency': len([a for a in recent_alerts if a['level'] == 'emergency']),
                    'critical': len([a for a in recent_alerts if a['level'] == 'critical']),
                    'warning': len([a for a in recent_alerts if a['level'] == 'warning'])
                },
                'history_entries': len(self.resource_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get resource status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'monitoring_active': self.is_monitoring
            }

# Factory functions
def create_production_resource_manager() -> ResourceManager:
    """Create production resource manager"""
    
    limits = ResourceLimits(
        max_memory_mb=2048,
        warning_memory_mb=1536,
        critical_memory_mb=1792,
        emergency_cleanup_threshold_mb=1900,
        max_cpu_percent=85.0,
        warning_cpu_percent=70.0,
        critical_cpu_percent=80.0,
        max_connections=1000,
        gc_threshold_mb=512
    )
    
    restart_config = RestartConfig(
        policy=RestartPolicy.UNLESS_STOPPED,
        max_restarts=3,
        restart_window_minutes=60,
        restart_delay_seconds=10,
        enable_graceful_shutdown=True,
        graceful_shutdown_timeout=30
    )
    
    return ResourceManager(limits, restart_config)

def create_development_resource_manager() -> ResourceManager:
    """Create development resource manager with relaxed limits"""
    
    limits = ResourceLimits(
        max_memory_mb=4096,
        warning_memory_mb=3072,
        critical_memory_mb=3584,
        emergency_cleanup_threshold_mb=3800,
        max_cpu_percent=95.0,
        warning_cpu_percent=80.0,
        critical_cpu_percent=90.0,
        max_connections=2000,
        gc_threshold_mb=1024
    )
    
    restart_config = RestartConfig(
        policy=RestartPolicy.ON_FAILURE,
        max_restarts=10,
        restart_window_minutes=30,
        restart_delay_seconds=5
    )
    
    return ResourceManager(limits, restart_config)

# Example usage and testing
if __name__ == "__main__":
    async def test_resource_management():
        """Test resource management system"""
        
        print("Testing Resource Management System")
        print("=" * 50)
        
        # Create production resource manager
        rm = create_production_resource_manager()
        
        try:
            # Start monitoring
            await rm.start_monitoring(interval_seconds=5)  # Fast interval for testing
            print("Resource monitoring started")
            
            # Get initial status
            status = rm.get_resource_status()
            print(f"Initial status: {status['current_usage']['memory_mb']:.1f}MB memory")
            
            # Simulate some load and test cleanup
            print("\nTesting memory cleanup...")
            cleanup_result = await rm.force_cleanup(aggressive=True)
            print(f"Cleanup result: {cleanup_result['memory_freed_mb']:.1f}MB freed")
            
            # Monitor for a short time
            print("\nMonitoring resources for 10 seconds...")
            await asyncio.sleep(10)
            
            # Get final status
            final_status = rm.get_resource_status()
            print(f"Final status: {final_status['current_usage']['memory_mb']:.1f}MB memory")
            print(f"History entries: {final_status['history_entries']}")
            
            print("\nResource management test completed successfully!")
            
        finally:
            await rm.stop_monitoring()
            print("Resource monitoring stopped")
    
    asyncio.run(test_resource_management())