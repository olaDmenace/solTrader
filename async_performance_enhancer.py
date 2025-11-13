#!/usr/bin/env python3
"""
Async Performance Enhancement System
====================================

Advanced asyncio optimization for SolTrader production deployment:
- Intelligent task coordination and batching
- Concurrent strategy execution with conflict prevention
- API rate limiting and connection pooling
- Memory-optimized async operations
- Performance monitoring and optimization

Targets:
- 40% reduction in API call latency through batching
- 60% improvement in concurrent strategy execution
- 25% reduction in memory usage through async optimization
"""

import asyncio
import aiohttp
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import weakref
import gc
from contextlib import asynccontextmanager
import json
import traceback

logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    """Performance optimization levels"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class TaskPriority(Enum):
    """Task priority levels for scheduling"""
    CRITICAL = 1    # Trading executions
    HIGH = 2        # Position monitoring
    MEDIUM = 3      # Data updates
    LOW = 4         # Analytics
    BACKGROUND = 5  # Cleanup tasks

@dataclass
class AsyncTaskConfig:
    """Configuration for async task optimization"""
    max_concurrent_tasks: int = 20
    task_timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: float = 0.1
    batch_size: int = 10
    batch_timeout_seconds: float = 2.0
    enable_task_monitoring: bool = True
    memory_cleanup_interval: int = 300  # 5 minutes

@dataclass
class APIBatchConfig:
    """Configuration for API request batching"""
    max_batch_size: int = 50
    batch_timeout_ms: int = 100
    rate_limit_per_second: int = 30
    concurrent_connections: int = 10
    connection_timeout: int = 10
    read_timeout: int = 15

@dataclass
class TaskMetrics:
    """Metrics for task performance monitoring"""
    task_id: str
    task_type: str
    priority: TaskPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[float] = None
    memory_usage_mb: float = 0.0
    result_size_bytes: int = 0
    error_count: int = 0
    retry_count: int = 0
    status: str = "pending"

@dataclass
class BatchRequestItem:
    """Individual request in a batch"""
    id: str
    url: str
    method: str = "GET"
    headers: Optional[Dict] = None
    data: Optional[Dict] = None
    callback: Optional[Callable] = None
    priority: TaskPriority = TaskPriority.MEDIUM

class AsyncPerformanceEnhancer:
    """Production-grade async performance optimization system"""
    
    def __init__(self, config: AsyncTaskConfig = None, api_config: APIBatchConfig = None):
        self.config = config or AsyncTaskConfig()
        self.api_config = api_config or APIBatchConfig()
        
        # Task management
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.priority_queues: Dict[TaskPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in TaskPriority
        }
        
        # Performance monitoring
        self.task_metrics: Dict[str, TaskMetrics] = {}
        self.performance_history: deque = deque(maxlen=1000)
        self.rate_limiter: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Batching system
        self.batch_queues: Dict[str, List[BatchRequestItem]] = defaultdict(list)
        self.batch_timers: Dict[str, asyncio.Task] = {}
        
        # Connection management
        self.session: Optional[aiohttp.ClientSession] = None
        self.connection_semaphore = asyncio.Semaphore(self.api_config.concurrent_connections)
        
        # Memory management
        self.memory_monitor_task: Optional[asyncio.Task] = None
        self.cleanup_interval = self.config.memory_cleanup_interval
        
        # Performance optimization flags
        self.optimization_enabled = True
        self.batch_optimization_enabled = True
        self.memory_optimization_enabled = True
        
        logger.info(f"AsyncPerformanceEnhancer initialized with {self.config.max_concurrent_tasks} max concurrent tasks")
    
    async def initialize(self):
        """Initialize the async performance system"""
        try:
            # Create optimized HTTP session
            timeout = aiohttp.ClientTimeout(
                connect=self.api_config.connection_timeout,
                total=self.api_config.read_timeout
            )
            
            connector = aiohttp.TCPConnector(
                limit=self.api_config.concurrent_connections,
                limit_per_host=self.api_config.concurrent_connections // 2,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"User-Agent": "SolTrader/1.0 AsyncOptimized"}
            )
            
            # Start background tasks
            if self.memory_optimization_enabled:
                self.memory_monitor_task = asyncio.create_task(self._memory_monitor())
            
            # Start task scheduler
            asyncio.create_task(self._task_scheduler())
            
            logger.info("AsyncPerformanceEnhancer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AsyncPerformanceEnhancer: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            # Cancel all active tasks
            for task_id, task in list(self.active_tasks.items()):
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Clean up batch timers
            for timer in self.batch_timers.values():
                if not timer.done():
                    timer.cancel()
            
            # Clean up memory monitor
            if self.memory_monitor_task and not self.memory_monitor_task.done():
                self.memory_monitor_task.cancel()
                try:
                    await self.memory_monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Close HTTP session
            if self.session:
                await self.session.close()
            
            logger.info("AsyncPerformanceEnhancer cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during AsyncPerformanceEnhancer cleanup: {e}")
    
    async def execute_optimized_task(self, 
                                   task_func: Callable,
                                   task_id: str = None,
                                   priority: TaskPriority = TaskPriority.MEDIUM,
                                   timeout: Optional[int] = None,
                                   **kwargs) -> Any:
        """Execute a task with optimization and monitoring"""
        
        if not task_id:
            task_id = f"task_{int(time.time() * 1000)}"
        
        # Create task metrics
        metrics = TaskMetrics(
            task_id=task_id,
            task_type=task_func.__name__,
            priority=priority,
            created_at=datetime.now()
        )
        self.task_metrics[task_id] = metrics
        
        try:
            # Apply rate limiting
            await self._apply_rate_limit(task_func.__name__)
            
            # Execute with timeout and monitoring
            timeout = timeout or self.config.task_timeout_seconds
            
            metrics.started_at = datetime.now()
            metrics.status = "running"
            
            # Memory tracking
            initial_memory = self._get_memory_usage()
            
            # Execute task with timeout
            result = await asyncio.wait_for(task_func(**kwargs), timeout=timeout)
            
            # Update metrics
            metrics.completed_at = datetime.now()
            metrics.execution_time_ms = (metrics.completed_at - metrics.started_at).total_seconds() * 1000
            metrics.memory_usage_mb = self._get_memory_usage() - initial_memory
            metrics.result_size_bytes = len(str(result)) if result else 0
            metrics.status = "completed"
            
            # Store performance history
            self.performance_history.append({
                'task_id': task_id,
                'execution_time_ms': metrics.execution_time_ms,
                'memory_usage_mb': metrics.memory_usage_mb,
                'timestamp': datetime.now()
            })
            
            return result
            
        except asyncio.TimeoutError:
            metrics.status = "timeout"
            logger.warning(f"Task {task_id} timed out after {timeout}s")
            raise
        except Exception as e:
            metrics.error_count += 1
            metrics.status = "error"
            logger.error(f"Task {task_id} failed: {e}")
            raise
        finally:
            # Cleanup task reference
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def batch_api_requests(self, 
                                requests: List[BatchRequestItem],
                                batch_id: str = None) -> List[Dict]:
        """Execute multiple API requests in optimized batches"""
        
        if not batch_id:
            batch_id = f"batch_{int(time.time() * 1000)}"
        
        results = []
        
        try:
            # Sort requests by priority
            sorted_requests = sorted(requests, key=lambda x: x.priority.value)
            
            # Process in batches
            for i in range(0, len(sorted_requests), self.api_config.max_batch_size):
                batch = sorted_requests[i:i + self.api_config.max_batch_size]
                
                # Execute batch concurrently
                batch_results = await self._execute_request_batch(batch)
                results.extend(batch_results)
                
                # Apply rate limiting between batches
                if i + self.api_config.max_batch_size < len(sorted_requests):
                    await asyncio.sleep(1.0 / self.api_config.rate_limit_per_second)
            
            logger.info(f"Batch {batch_id} completed: {len(requests)} requests, {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Batch {batch_id} failed: {e}")
            raise
    
    async def _execute_request_batch(self, batch: List[BatchRequestItem]) -> List[Dict]:
        """Execute a single batch of API requests"""
        
        if not self.session:
            raise RuntimeError("HTTP session not initialized")
        
        async def execute_single_request(request: BatchRequestItem) -> Dict:
            async with self.connection_semaphore:
                try:
                    async with self.session.request(
                        method=request.method,
                        url=request.url,
                        headers=request.headers,
                        json=request.data
                    ) as response:
                        result = {
                            'id': request.id,
                            'status_code': response.status,
                            'data': await response.json() if response.content_type == 'application/json' else await response.text(),
                            'success': 200 <= response.status < 300
                        }
                        
                        # Execute callback if provided
                        if request.callback:
                            try:
                                await request.callback(result)
                            except Exception as callback_error:
                                logger.warning(f"Callback failed for request {request.id}: {callback_error}")
                        
                        return result
                        
                except Exception as e:
                    return {
                        'id': request.id,
                        'error': str(e),
                        'success': False
                    }
        
        # Execute all requests in the batch concurrently
        tasks = [execute_single_request(request) for request in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'id': batch[i].id,
                    'error': str(result),
                    'success': False
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def optimize_strategy_coordination(self, strategies: List[Dict]) -> Dict:
        """Optimize coordination between multiple strategies"""
        
        optimization_start = time.time()
        
        try:
            # Group strategies by compatibility
            compatible_groups = self._group_compatible_strategies(strategies)
            
            # Execute compatible strategies concurrently
            coordination_results = []
            
            for group in compatible_groups:
                if len(group) > 1:
                    # Execute strategies in this group concurrently
                    group_results = await self._execute_strategy_group_concurrent(group)
                else:
                    # Single strategy - execute normally
                    group_results = await self._execute_strategy_single(group[0])
                
                coordination_results.extend(group_results if isinstance(group_results, list) else [group_results])
            
            optimization_time = (time.time() - optimization_start) * 1000
            
            return {
                'success': True,
                'strategy_count': len(strategies),
                'concurrent_groups': len(compatible_groups),
                'optimization_time_ms': optimization_time,
                'results': coordination_results
            }
            
        except Exception as e:
            logger.error(f"Strategy coordination optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'strategy_count': len(strategies)
            }
    
    def _group_compatible_strategies(self, strategies: List[Dict]) -> List[List[Dict]]:
        """Group strategies that can be executed concurrently without conflicts"""
        
        # Simple compatibility check based on token overlap
        groups = []
        ungrouped = strategies.copy()
        
        while ungrouped:
            current_group = [ungrouped.pop(0)]
            current_tokens = set(current_group[0].get('target_tokens', []))
            
            # Find compatible strategies
            i = 0
            while i < len(ungrouped):
                strategy = ungrouped[i]
                strategy_tokens = set(strategy.get('target_tokens', []))
                
                # Check for token conflicts
                if not current_tokens.intersection(strategy_tokens):
                    current_group.append(ungrouped.pop(i))
                    current_tokens.update(strategy_tokens)
                else:
                    i += 1
            
            groups.append(current_group)
        
        return groups
    
    async def _execute_strategy_group_concurrent(self, strategy_group: List[Dict]) -> List[Dict]:
        """Execute a group of compatible strategies concurrently"""
        
        async def execute_strategy(strategy: Dict) -> Dict:
            try:
                # Simulate strategy execution (replace with actual strategy logic)
                await asyncio.sleep(0.1)  # Simulate processing time
                return {
                    'strategy_id': strategy.get('id'),
                    'success': True,
                    'execution_time_ms': 100
                }
            except Exception as e:
                return {
                    'strategy_id': strategy.get('id'),
                    'success': False,
                    'error': str(e)
                }
        
        # Execute all strategies in the group concurrently
        tasks = [execute_strategy(strategy) for strategy in strategy_group]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_strategy_single(self, strategy: Dict) -> Dict:
        """Execute a single strategy"""
        try:
            # Simulate strategy execution
            await asyncio.sleep(0.1)
            return {
                'strategy_id': strategy.get('id'),
                'success': True,
                'execution_time_ms': 100
            }
        except Exception as e:
            return {
                'strategy_id': strategy.get('id'),
                'success': False,
                'error': str(e)
            }
    
    async def _task_scheduler(self):
        """Background task scheduler for priority-based task execution"""
        
        while self.optimization_enabled:
            try:
                # Process tasks by priority
                for priority in TaskPriority:
                    queue = self.priority_queues[priority]
                    
                    # Process all available tasks for this priority level
                    while not queue.empty() and len(self.active_tasks) < self.config.max_concurrent_tasks:
                        try:
                            task_item = queue.get_nowait()
                            task = asyncio.create_task(task_item['coro'])
                            self.active_tasks[task_item['id']] = task
                        except asyncio.QueueEmpty:
                            break
                
                # Clean up completed tasks
                completed_tasks = [
                    task_id for task_id, task in self.active_tasks.items()
                    if task.done()
                ]
                for task_id in completed_tasks:
                    del self.active_tasks[task_id]
                
                # Wait before next scheduling cycle
                await asyncio.sleep(0.01)  # 10ms scheduling interval
                
            except Exception as e:
                logger.error(f"Task scheduler error: {e}")
                await asyncio.sleep(1.0)
    
    async def _memory_monitor(self):
        """Background memory monitoring and cleanup"""
        
        while self.memory_optimization_enabled:
            try:
                # Get current memory usage
                current_memory = self._get_memory_usage()
                
                # Trigger cleanup if memory usage is high
                if current_memory > 1500:  # 1.5GB threshold
                    await self._perform_memory_cleanup()
                
                # Regular cleanup every interval
                await asyncio.sleep(self.cleanup_interval)
                await self._perform_memory_cleanup()
                
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _perform_memory_cleanup(self):
        """Perform memory cleanup operations"""
        try:
            # Clean up old task metrics
            cutoff_time = datetime.now() - timedelta(hours=1)
            old_metrics = [
                task_id for task_id, metrics in self.task_metrics.items()
                if metrics.completed_at and metrics.completed_at < cutoff_time
            ]
            for task_id in old_metrics:
                del self.task_metrics[task_id]
            
            # Clean up rate limiter history
            for endpoint, history in self.rate_limiter.items():
                cutoff = time.time() - 60  # Keep last minute
                while history and history[0] < cutoff:
                    history.popleft()
            
            # Force garbage collection
            collected = gc.collect()
            
            logger.debug(f"Memory cleanup completed: {collected} objects collected, {len(old_metrics)} old metrics removed")
            
        except Exception as e:
            logger.error(f"Memory cleanup error: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    async def _apply_rate_limit(self, endpoint: str):
        """Apply rate limiting for API endpoints"""
        current_time = time.time()
        history = self.rate_limiter[endpoint]
        
        # Clean old requests
        cutoff = current_time - 1.0  # 1 second window
        while history and history[0] < cutoff:
            history.popleft()
        
        # Check if we need to wait
        if len(history) >= self.api_config.rate_limit_per_second:
            wait_time = 1.0 - (current_time - history[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Record this request
        history.append(current_time)
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        
        active_task_count = len(self.active_tasks)
        completed_tasks = [m for m in self.task_metrics.values() if m.status == "completed"]
        failed_tasks = [m for m in self.task_metrics.values() if m.status == "error"]
        
        avg_execution_time = 0.0
        if completed_tasks:
            avg_execution_time = sum(m.execution_time_ms for m in completed_tasks if m.execution_time_ms) / len(completed_tasks)
        
        avg_memory_usage = 0.0
        if completed_tasks:
            avg_memory_usage = sum(m.memory_usage_mb for m in completed_tasks) / len(completed_tasks)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'active_tasks': active_task_count,
            'completed_tasks': len(completed_tasks),
            'failed_tasks': len(failed_tasks),
            'success_rate': len(completed_tasks) / max(1, len(completed_tasks) + len(failed_tasks)),
            'average_execution_time_ms': avg_execution_time,
            'average_memory_usage_mb': avg_memory_usage,
            'current_memory_mb': self._get_memory_usage(),
            'optimization_enabled': self.optimization_enabled,
            'batch_optimization_enabled': self.batch_optimization_enabled,
            'memory_optimization_enabled': self.memory_optimization_enabled
        }

@asynccontextmanager
async def create_performance_enhancer(config: AsyncTaskConfig = None, api_config: APIBatchConfig = None):
    """Context manager for AsyncPerformanceEnhancer"""
    enhancer = AsyncPerformanceEnhancer(config, api_config)
    try:
        await enhancer.initialize()
        yield enhancer
    finally:
        await enhancer.cleanup()

# Integration with strategy coordinator
class EnhancedStrategyCoordinator:
    """Enhanced strategy coordinator with async performance optimization"""
    
    def __init__(self, base_coordinator, performance_enhancer: AsyncPerformanceEnhancer):
        self.base_coordinator = base_coordinator
        self.performance_enhancer = performance_enhancer
        
    async def coordinate_strategies_optimized(self, strategies: List[Dict]) -> Dict:
        """Coordinate strategies with async performance optimization"""
        
        # Use performance enhancer for optimal coordination
        return await self.performance_enhancer.optimize_strategy_coordination(strategies)
    
    async def execute_batch_operations(self, operations: List[Dict]) -> List[Dict]:
        """Execute multiple operations in optimized batches"""
        
        # Convert operations to batch requests
        batch_requests = []
        for i, op in enumerate(operations):
            batch_requests.append(BatchRequestItem(
                id=f"op_{i}",
                url=op.get('url', ''),
                method=op.get('method', 'GET'),
                data=op.get('data'),
                priority=TaskPriority.HIGH if op.get('critical', False) else TaskPriority.MEDIUM
            ))
        
        # Execute in optimized batches
        results = await self.performance_enhancer.batch_api_requests(batch_requests)
        return results

# Example usage and testing
if __name__ == "__main__":
    async def test_async_enhancement():
        """Test the async performance enhancement system"""
        
        config = AsyncTaskConfig(
            max_concurrent_tasks=15,
            task_timeout_seconds=10,
            batch_size=20
        )
        
        api_config = APIBatchConfig(
            max_batch_size=25,
            rate_limit_per_second=20,
            concurrent_connections=8
        )
        
        async with create_performance_enhancer(config, api_config) as enhancer:
            # Test optimized task execution
            async def sample_task(delay=0.1):
                await asyncio.sleep(delay)
                return f"Task completed at {datetime.now()}"
            
            # Execute multiple tasks concurrently
            tasks = []
            for i in range(10):
                task = enhancer.execute_optimized_task(
                    sample_task,
                    task_id=f"test_task_{i}",
                    priority=TaskPriority.HIGH,
                    delay=0.05
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            print(f"Executed {len(results)} tasks concurrently")
            
            # Test strategy coordination
            test_strategies = [
                {'id': 'momentum_1', 'target_tokens': ['token_a']},
                {'id': 'mean_reversion_1', 'target_tokens': ['token_b']},
                {'id': 'grid_1', 'target_tokens': ['token_c']},
            ]
            
            coordination_result = await enhancer.optimize_strategy_coordination(test_strategies)
            print(f"Strategy coordination: {coordination_result['success']}")
            
            # Get performance metrics
            metrics = enhancer.get_performance_metrics()
            print(f"Performance metrics: {metrics}")
    
    asyncio.run(test_async_enhancement())