#!/usr/bin/env python3
"""
Robust API utilities with comprehensive error handling and retry logic
Provides decorators and classes for production-ready API interactions
"""
import asyncio
import logging
import functools
import time
from typing import Any, Callable, Dict, Optional, Union, List, Type
from dataclasses import dataclass
from enum import Enum
import aiohttp
import json

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels for different types of failures"""
    LOW = "low"           # Temporary glitches, safe to retry
    MEDIUM = "medium"     # Concerning but recoverable
    HIGH = "high"         # Serious issues requiring attention
    CRITICAL = "critical" # System-threatening, requires immediate action

@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0  # Initial delay in seconds
    max_delay: float = 60.0  # Maximum delay between retries
    exponential_base: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to prevent thundering herd
    
    # Which errors should trigger retries
    retryable_status_codes: List[int] = None
    retryable_exceptions: List[Type[Exception]] = None
    
    def __post_init__(self):
        if self.retryable_status_codes is None:
            self.retryable_status_codes = [429, 500, 502, 503, 504, 520, 521, 522, 523, 524]
        
        if self.retryable_exceptions is None:
            self.retryable_exceptions = [
                aiohttp.ClientTimeout,
                aiohttp.ClientConnectionError,
                aiohttp.ClientPayloadError,
                asyncio.TimeoutError,
                ConnectionError,
            ]

@dataclass
class ErrorEvent:
    """Represents an error event for tracking and alerting"""
    timestamp: float
    component: str
    error_type: str
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    retry_attempt: int
    resolved: bool = False

class ErrorTracker:
    """Tracks errors and triggers alerts when thresholds are exceeded"""
    
    def __init__(self):
        self.errors: List[ErrorEvent] = []
        self.error_counts: Dict[str, int] = {}
        self.last_alert_time: Dict[str, float] = {}
        
        # Alert thresholds
        self.thresholds = {
            ErrorSeverity.LOW: 10,      # Alert after 10 low-severity errors
            ErrorSeverity.MEDIUM: 5,    # Alert after 5 medium-severity errors  
            ErrorSeverity.HIGH: 2,      # Alert after 2 high-severity errors
            ErrorSeverity.CRITICAL: 1,  # Alert immediately for critical errors
        }
        
        # Minimum time between alerts for same component (minutes)
        self.alert_cooldown = {
            ErrorSeverity.LOW: 30,
            ErrorSeverity.MEDIUM: 15,
            ErrorSeverity.HIGH: 5,
            ErrorSeverity.CRITICAL: 0,  # No cooldown for critical
        }
    
    def record_error(self, component: str, error: Exception, severity: ErrorSeverity, 
                    retry_attempt: int = 0, details: Dict[str, Any] = None) -> ErrorEvent:
        """Record an error event"""
        event = ErrorEvent(
            timestamp=time.time(),
            component=component,
            error_type=type(error).__name__,
            severity=severity,
            message=str(error),
            details=details or {},
            retry_attempt=retry_attempt
        )
        
        self.errors.append(event)
        
        # Update error counts
        key = f"{component}:{severity.value}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # Check if we need to trigger an alert
        self._check_alert_threshold(component, severity)
        
        logger.error(f"[{component}] {severity.value.upper()} error (attempt {retry_attempt}): {error}")
        if details:
            logger.error(f"[{component}] Error details: {details}")
        
        return event
    
    def record_recovery(self, component: str, retry_attempt: int):
        """Record successful recovery after errors"""
        logger.info(f"[{component}] Recovered successfully after {retry_attempt} retries")
        
        # Mark recent errors as resolved
        current_time = time.time()
        for event in reversed(self.errors):
            if (event.component == component and 
                not event.resolved and 
                current_time - event.timestamp < 300):  # Last 5 minutes
                event.resolved = True
    
    def _check_alert_threshold(self, component: str, severity: ErrorSeverity):
        """Check if error count exceeds threshold and should trigger alert"""
        key = f"{component}:{severity.value}"
        error_count = self.error_counts.get(key, 0)
        threshold = self.thresholds[severity]
        
        if error_count >= threshold:
            cooldown_minutes = self.alert_cooldown[severity]
            last_alert = self.last_alert_time.get(key, 0)
            
            if time.time() - last_alert > cooldown_minutes * 60:
                self._trigger_alert(component, severity, error_count)
                self.last_alert_time[key] = time.time()
                # Reset counter after alert
                self.error_counts[key] = 0
    
    def _trigger_alert(self, component: str, severity: ErrorSeverity, error_count: int):
        """Trigger alert with integrated email/SMS notifications"""
        message = f"ALERT: {component} - {severity.value.upper()} errors: {error_count}"
        logger.critical(message)
        
        # Console alert for immediate visibility
        print(f"\n{'='*60}")
        print(f"PRODUCTION ALERT")
        print(f"Component: {component}")
        print(f"Severity: {severity.value.upper()}")
        print(f"Error Count: {error_count}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Send email/SMS alert asynchronously
        import asyncio
        try:
            # Import here to avoid circular imports
            from .alerting_system import production_alerter
            
            # Get recent error details
            recent_errors = [e for e in self.errors[-5:] if e.component == component]
            error_details = {
                "error_count": error_count,
                "recent_errors": [
                    {
                        "error_type": e.error_type,
                        "message": e.message,
                        "timestamp": time.strftime('%H:%M:%S', time.localtime(e.timestamp))
                    } for e in recent_errors
                ],
                "component": component,
                "severity": severity.value
            }
            
            # Create async task to send alert
            alert_message = (f"{component} has encountered {error_count} {severity.value} severity errors. "
                           f"Recent errors: {', '.join([e.error_type for e in recent_errors[-3:]])}")
            
            # Try to send the alert in the current event loop, or create a new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule as a task in the running loop
                    asyncio.create_task(
                        production_alerter.send_alert(
                            title=f"{severity.value.upper()}: {component} Errors",
                            message=alert_message,
                            severity=severity.value,
                            component=component,
                            details=error_details
                        )
                    )
                else:
                    # Run in the existing loop
                    loop.run_until_complete(
                        production_alerter.send_alert(
                            title=f"{severity.value.upper()}: {component} Errors", 
                            message=alert_message,
                            severity=severity.value,
                            component=component,
                            details=error_details
                        )
                    )
            except RuntimeError:
                # No event loop, create a new one
                asyncio.run(
                    production_alerter.send_alert(
                        title=f"{severity.value.upper()}: {component} Errors",
                        message=alert_message,
                        severity=severity.value,
                        component=component,
                        details=error_details
                    )
                )
                
        except Exception as e:
            # Don't let alert failures break the main system
            logger.error(f"Failed to send alert notification: {e}")
            pass

# Global error tracker instance
error_tracker = ErrorTracker()

def robust_api_call(config: RetryConfig = None, component: str = "api"):
    """
    Decorator for robust API calls with comprehensive error handling and retries
    
    Args:
        config: RetryConfig instance, uses defaults if None
        component: Component name for error tracking
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    # Call the original function
                    result = await func(*args, **kwargs)
                    
                    # Record recovery if this wasn't the first attempt
                    if attempt > 0:
                        error_tracker.record_recovery(component, attempt)
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Determine error severity and if we should retry
                    severity = _determine_error_severity(e)
                    should_retry = _should_retry(e, config, attempt)
                    
                    # Record the error
                    error_tracker.record_error(
                        component=component,
                        error=e,
                        severity=severity,
                        retry_attempt=attempt + 1,
                        details={
                            'function': func.__name__,
                            'args_count': len(args),
                            'kwargs_keys': list(kwargs.keys()),
                            'should_retry': should_retry,
                            'final_attempt': attempt == config.max_attempts - 1
                        }
                    )
                    
                    # If this is the last attempt or we shouldn't retry, raise
                    if not should_retry or attempt == config.max_attempts - 1:
                        logger.error(f"[{component}] Failed after {attempt + 1} attempts: {e}")
                        raise
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
                    
                    logger.warning(f"[{component}] Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator

def _determine_error_severity(error: Exception) -> ErrorSeverity:
    """Determine the severity of an error"""
    
    # Critical errors - these threaten system stability
    if isinstance(error, (MemoryError, SystemExit, KeyboardInterrupt)):
        return ErrorSeverity.CRITICAL
    
    # High severity - authentication, authorization, configuration issues
    if isinstance(error, aiohttp.ClientResponseError):
        if error.status in [401, 403]:  # Auth issues
            return ErrorSeverity.HIGH
        elif error.status in [400, 404, 422]:  # Client errors
            return ErrorSeverity.MEDIUM
        elif error.status in [429]:  # Rate limiting
            return ErrorSeverity.LOW
        elif error.status >= 500:  # Server errors
            return ErrorSeverity.MEDIUM
    
    # Connection and timeout issues - usually temporary
    if isinstance(error, (
        aiohttp.ClientTimeout,
        aiohttp.ClientConnectionError,
        asyncio.TimeoutError,
        ConnectionError
    )):
        return ErrorSeverity.LOW
    
    # JSON/Data parsing errors - might indicate API changes
    if isinstance(error, (json.JSONDecodeError, ValueError, KeyError)):
        return ErrorSeverity.MEDIUM
    
    # Default to medium for unknown errors
    return ErrorSeverity.MEDIUM

def _should_retry(error: Exception, config: RetryConfig, attempt: int) -> bool:
    """Determine if an error should trigger a retry"""
    
    # Don't retry if we've hit max attempts
    if attempt >= config.max_attempts - 1:
        return False
    
    # Check if it's a retryable exception type
    if any(isinstance(error, exc_type) for exc_type in config.retryable_exceptions):
        return True
    
    # Check HTTP status codes
    if isinstance(error, aiohttp.ClientResponseError):
        return error.status in config.retryable_status_codes
    
    # Don't retry authentication errors
    if isinstance(error, aiohttp.ClientResponseError) and error.status in [401, 403]:
        return False
    
    return False

class RobustHTTPClient:
    """Enhanced HTTP client with built-in error handling and retry logic"""
    
    def __init__(self, 
                 base_url: str,
                 component_name: str,
                 headers: Dict[str, str] = None,
                 timeout: float = 30.0,
                 retry_config: RetryConfig = None):
        
        self.base_url = base_url.rstrip('/')
        self.component_name = component_name
        self.headers = headers or {}
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_config = retry_config or RetryConfig()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def ensure_session(self):
        """Ensure HTTP session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=self.timeout
            )
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    @robust_api_call()
    async def get(self, endpoint: str, params: Dict = None, **kwargs) -> Dict[str, Any]:
        """Robust GET request"""
        await self.ensure_session()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        async with self.session.get(url, params=params, **kwargs) as response:
            response.raise_for_status()
            return await response.json()
    
    @robust_api_call()
    async def post(self, endpoint: str, data: Dict = None, json_data: Dict = None, **kwargs) -> Dict[str, Any]:
        """Robust POST request"""
        await self.ensure_session()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        async with self.session.post(url, data=data, json=json_data, **kwargs) as response:
            response.raise_for_status()
            return await response.json()

def get_error_summary(hours: int = 24) -> Dict[str, Any]:
    """Get summary of errors from the last N hours"""
    cutoff_time = time.time() - (hours * 3600)
    recent_errors = [e for e in error_tracker.errors if e.timestamp > cutoff_time]
    
    summary = {
        'total_errors': len(recent_errors),
        'by_component': {},
        'by_severity': {},
        'unresolved_count': 0,
        'critical_errors': []
    }
    
    for error in recent_errors:
        # Count by component
        comp = summary['by_component'].setdefault(error.component, 0)
        summary['by_component'][error.component] = comp + 1
        
        # Count by severity
        sev = summary['by_severity'].setdefault(error.severity.value, 0)
        summary['by_severity'][error.severity.value] = sev + 1
        
        # Count unresolved
        if not error.resolved:
            summary['unresolved_count'] += 1
        
        # Track critical errors
        if error.severity == ErrorSeverity.CRITICAL:
            summary['critical_errors'].append({
                'timestamp': error.timestamp,
                'component': error.component,
                'message': error.message
            })
    
    return summary