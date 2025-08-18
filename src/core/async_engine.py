"""
Async Performance Framework
High-performance asyncio-based command execution engine for Monk CLI
"""

import asyncio
import time
import sys
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import importlib
import inspect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CommandResult:
    """Result of command execution"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    startup_time: float = 0.0
    command_execution_time: float = 0.0
    memory_usage: float = 0.0
    cache_hit_ratio: float = 0.0
    concurrent_operations: int = 0

class AsyncCommandRouter:
    """
    Asynchronous command router with lazy loading and performance monitoring
    """
    
    def __init__(self):
        self.commands = {}
        self.command_modules = {}
        self.metrics = PerformanceMetrics()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.coordinator_instance = None  # Add coordinator instance reference
        
        # Register core commands
        self._register_core_commands()
        
        logger.info(f"AsyncCommandRouter initialized in {time.time() - time.time():.3f}s")
    
    def set_coordinator_instance(self, coordinator):
        """Set the coordinator instance to pass to command functions"""
        self.coordinator_instance = coordinator
        logger.info("Coordinator instance set in AsyncCommandRouter")
    
    def _register_core_commands(self):
        """Register core commands with lazy loading paths"""
        self.command_modules.update({
            'analyze': 'src.analyzers.analyzer_coordinator',
            'github': 'src.analyzers.analyzer_coordinator',  # Changed from github_analyzer
            'docker': 'src.analyzers.analyzer_coordinator',  # Changed from docker_optimizer
            'npm': 'src.analyzers.analyzer_coordinator',     # Changed from npm_manager
            'git': 'src.analyzers.analyzer_coordinator',     # Changed from git_analyzer
            'ai': 'src.analyzers.analyzer_coordinator',      # New AI command
            'workspace': 'src.workspace.workspace_manager',
            'plugin': 'src.plugins.plugin_manager',
            'config': 'src.core.config_manager',
            'cache': 'src.core.cache_manager',
            'help': 'src.core.help_system'
        })
    
    async def execute_command(self, command: str, args: List[str] = None, **kwargs) -> CommandResult:
        """
        Execute command asynchronously with performance monitoring
        """
        start_time = time.time()
        args = args or []
        
        try:
            # Lazy load command if not already loaded
            if command not in self.commands:
                await self._lazy_load_command(command)
            
            if command not in self.commands:
                return CommandResult(
                    success=False,
                    error=f"Unknown command: {command}",
                    execution_time=time.time() - start_time
                )
            
            # Execute command
            command_func = self.commands[command]
            
            # Add coordinator instance to kwargs if available
            if self.coordinator_instance is not None:
                kwargs['coordinator_instance'] = self.coordinator_instance
            
            # Handle both sync and async functions
            if inspect.iscoroutinefunction(command_func):
                result = await command_func(args, **kwargs)
            else:
                # Run sync functions in thread pool to avoid blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, command_func, args, **kwargs
                )
            
            execution_time = time.time() - start_time
            self.metrics.command_execution_time = execution_time
            
            return CommandResult(
                success=True,
                data=result,
                execution_time=execution_time,
                metadata={'command': command, 'args': args}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Command '{command}' failed: {e}")
            
            return CommandResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={'command': command, 'args': args, 'exception_type': type(e).__name__}
            )
    
    async def _lazy_load_command(self, command: str):
        """Lazy load command module to optimize startup time"""
        if command not in self.command_modules:
            return
        
        module_path = self.command_modules[command]
        try:
            # Import module dynamically
            module = importlib.import_module(module_path)
            
            # Look for command function (convention: command_<name> or main)
            command_func = None
            for func_name in [f'command_{command}', 'main', 'execute', command]:
                if hasattr(module, func_name):
                    command_func = getattr(module, func_name)
                    break
            
            if command_func:
                self.commands[command] = command_func
                logger.debug(f"Lazy loaded command: {command}")
            else:
                logger.warning(f"No suitable function found in {module_path} for command: {command}")
                
        except ImportError as e:
            logger.error(f"Failed to lazy load command '{command}': {e}")
    
    async def execute_concurrent_commands(self, commands: List[Dict[str, Any]]) -> List[CommandResult]:
        """
        Execute multiple commands concurrently for improved performance
        """
        self.metrics.concurrent_operations = len(commands)
        
        tasks = []
        for cmd_config in commands:
            command = cmd_config.get('command')
            args = cmd_config.get('args', [])
            kwargs = cmd_config.get('kwargs', {})
            
            task = self.execute_command(command, args, **kwargs)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to CommandResult objects
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(CommandResult(
                    success=False,
                    error=str(result),
                    metadata={'exception_type': type(result).__name__}
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self.metrics
    
    def register_command(self, name: str, func: Callable):
        """Register a new command function"""
        self.commands[name] = func
        logger.debug(f"Registered command: {name}")
    
    async def shutdown(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        logger.info("AsyncCommandRouter shutdown complete")

class ConnectionPool:
    """
    Connection pool for external APIs to improve performance
    """
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.pools: Dict[str, asyncio.Queue] = {}
        self.active_connections: Dict[str, int] = {}
    
    async def get_connection(self, service: str, connection_factory: Callable) -> Any:
        """Get connection from pool or create new one"""
        if service not in self.pools:
            self.pools[service] = asyncio.Queue(maxsize=self.max_connections)
            self.active_connections[service] = 0
        
        pool = self.pools[service]
        
        try:
            # Try to get existing connection
            connection = pool.get_nowait()
            return connection
        except asyncio.QueueEmpty:
            # Create new connection if under limit
            if self.active_connections[service] < self.max_connections:
                connection = await connection_factory()
                self.active_connections[service] += 1
                return connection
            else:
                # Wait for available connection
                return await pool.get()
    
    async def return_connection(self, service: str, connection: Any):
        """Return connection to pool"""
        if service in self.pools:
            try:
                self.pools[service].put_nowait(connection)
            except asyncio.QueueFull:
                # Pool is full, close connection
                if hasattr(connection, 'close'):
                    await connection.close()
                self.active_connections[service] -= 1
    
    async def close_all_connections(self):
        """Close all pooled connections"""
        for service, pool in self.pools.items():
            while not pool.empty():
                try:
                    connection = pool.get_nowait()
                    if hasattr(connection, 'close'):
                        await connection.close()
                except asyncio.QueueEmpty:
                    break
        
        self.pools.clear()
        self.active_connections.clear()

class MemoryOptimizer:
    """
    Memory-efficient data structures and optimization utilities
    """
    
    @staticmethod
    def create_efficient_dict(data: Dict[str, Any], max_size: int = 1000) -> Dict[str, Any]:
        """Create memory-efficient dictionary with size limits"""
        if len(data) <= max_size:
            return data
        
        # Keep most recently accessed items
        sorted_items = sorted(data.items(), key=lambda x: hash(x[0]))
        return dict(sorted_items[:max_size])
    
    @staticmethod
    def compress_large_strings(text: str, threshold: int = 1000) -> str:
        """Compress large strings to save memory"""
        if len(text) <= threshold:
            return text
        
        # Simple compression: remove extra whitespace and truncate
        compressed = ' '.join(text.split())
        if len(compressed) > threshold:
            compressed = compressed[:threshold-3] + '...'
        
        return compressed
    
    @staticmethod
    async def process_large_dataset_chunks(data: List[Any], chunk_size: int = 100) -> List[Any]:
        """Process large datasets in chunks to manage memory"""
        results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            # Process chunk
            processed_chunk = [item for item in chunk if item is not None]
            results.extend(processed_chunk)
            
            # Yield control to event loop
            await asyncio.sleep(0)
        
        return results

class PerformanceProfiler:
    """
    Performance monitoring and profiling utilities
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing and record duration"""
        if operation not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        del self.start_times[operation]
        
        return duration
    
    def get_average_time(self, operation: str) -> float:
        """Get average execution time for operation"""
        if operation not in self.metrics or not self.metrics[operation]:
            return 0.0
        
        return sum(self.metrics[operation]) / len(self.metrics[operation])
    
    def get_performance_report(self) -> Dict[str, Dict[str, float]]:
        """Generate performance report"""
        report = {}
        
        for operation, times in self.metrics.items():
            if times:
                report[operation] = {
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times),
                    'total': sum(times)
                }
        
        return report

# Global instances
async_router = AsyncCommandRouter()
connection_pool = ConnectionPool()
memory_optimizer = MemoryOptimizer()
performance_profiler = PerformanceProfiler()

# Performance monitoring decorator
def monitor_performance(operation_name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                performance_profiler.start_timer(operation_name)
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = performance_profiler.end_timer(operation_name)
                    logger.debug(f"{operation_name} completed in {duration:.3f}s")
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                performance_profiler.start_timer(operation_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = performance_profiler.end_timer(operation_name)
                    logger.debug(f"{operation_name} completed in {duration:.3f}s")
            return sync_wrapper
    return decorator

# Example usage
if __name__ == "__main__":
    async def example_usage():
        # Test async command execution
        result = await async_router.execute_command('help', ['analyze'])
        print(f"Command result: {result}")
        
        # Test concurrent execution
        commands = [
            {'command': 'github', 'args': ['status']},
            {'command': 'docker', 'args': ['analyze']},
            {'command': 'npm', 'args': ['audit']}
        ]
        
        results = await async_router.execute_concurrent_commands(commands)
        print(f"Concurrent results: {len(results)} commands executed")
        
        # Get performance metrics
        metrics = async_router.get_performance_metrics()
        print(f"Startup time: {metrics.startup_time:.3f}s")
        
        # Cleanup
        await async_router.shutdown()
        await connection_pool.close_all_connections()
    
    # Run example
    asyncio.run(example_usage())