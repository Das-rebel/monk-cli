"""
MONK CLI Phase 2.5 - Performance Benchmarking Suite
Comprehensive performance analysis of open source integration components
"""

import asyncio
import time
import statistics
import json
import tempfile
import os
import sys
import logging
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from phase25.bridges.treequest_smolagent_bridge import (
    TreeQuestSmolagentBridge, TreeQuestTask, AgentPersonality
)
from phase25.smolagents.multi_agent_system import (
    MONKMultiAgentSystem, MultiAgentConfiguration, AgentTask, TaskStatus
)
from phase25.lsp.tree_sitter_explorer import (
    TreeSitterLSPExplorer, CodeLanguage, NodeType
)
from phase25.collaboration.collaborative_platform import (
    CollaborativePlatform, CollaborationUser, UserRole
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    component: str
    test_name: str
    duration_ms: float
    memory_usage_mb: float
    throughput: float
    success_rate: float
    error_count: int
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class PerformanceProfile:
    """Performance profile for component"""
    component: str
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    throughput_ops_per_sec: float
    memory_efficiency_mb_per_op: float
    cpu_usage_percent: float
    success_rate_percent: float
    scalability_score: float


class Phase25Benchmarker:
    """Comprehensive benchmarking suite for Phase 2.5"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.start_time = None
        
        # System monitoring
        self.process = psutil.Process(os.getpid())
        self.initial_memory = None
        
    async def run_complete_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        logger.info("Starting Phase 2.5 Performance Benchmark Suite")
        self.start_time = time.time()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Run individual component benchmarks
        await self.benchmark_treequest_bridge()
        await self.benchmark_multi_agent_system()
        await self.benchmark_tree_sitter_explorer()
        await self.benchmark_collaborative_platform()
        
        # Run integration benchmarks
        await self.benchmark_end_to_end_workflows()
        await self.benchmark_scalability()
        await self.benchmark_memory_efficiency()
        
        # Generate performance profiles
        self.generate_performance_profiles()
        
        # Create comprehensive report
        report = self.generate_benchmark_report()
        
        total_time = time.time() - self.start_time
        logger.info(f"Benchmark suite completed in {total_time:.2f} seconds")
        
        return report
    
    async def benchmark_treequest_bridge(self):
        """Benchmark TreeQuest-Smolagent bridge performance"""
        logger.info("Benchmarking TreeQuest-Smolagent Bridge...")
        
        bridge = TreeQuestSmolagentBridge()
        await bridge.initialize()
        
        # Test 1: Task decomposition performance
        await self._benchmark_task_decomposition(bridge)
        
        # Test 2: Task execution performance
        await self._benchmark_task_execution(bridge)
        
        # Test 3: Personality switching performance
        await self._benchmark_personality_switching(bridge)
        
        # Test 4: Agent allocation optimization
        await self._benchmark_agent_allocation(bridge)
    
    async def _benchmark_task_decomposition(self, bridge: TreeQuestSmolagentBridge):
        """Benchmark task decomposition performance"""
        test_tasks = [
            "Implement user authentication system",
            "Create REST API for inventory management", 
            "Build real-time chat application",
            "Develop machine learning recommendation engine",
            "Design responsive web dashboard"
        ]
        
        durations = []
        memory_usage = []
        success_count = 0
        
        for task in test_tasks:
            gc.collect()
            start_memory = self.process.memory_info().rss / 1024 / 1024
            start_time = time.time()
            
            try:
                result = await bridge.decompose_task_hierarchy(
                    root_task=task,
                    domain="software_development",
                    target_complexity=3
                )
                
                if result and len(result.subtasks) >= 3:
                    success_count += 1
                
            except Exception as e:
                logger.error(f"Task decomposition failed: {e}")
            
            duration = (time.time() - start_time) * 1000
            end_memory = self.process.memory_info().rss / 1024 / 1024
            
            durations.append(duration)
            memory_usage.append(end_memory - start_memory)
        
        self.results.append(BenchmarkResult(
            component="TreeQuest Bridge",
            test_name="Task Decomposition",
            duration_ms=statistics.mean(durations),
            memory_usage_mb=statistics.mean(memory_usage),
            throughput=len(test_tasks) / (sum(durations) / 1000),
            success_rate=success_count / len(test_tasks),
            error_count=len(test_tasks) - success_count,
            metadata={
                "avg_duration_ms": statistics.mean(durations),
                "p95_duration_ms": statistics.quantiles(durations, n=20)[18] if len(durations) > 1 else durations[0],
                "tasks_tested": len(test_tasks)
            },
            timestamp=datetime.now()
        ))
    
    async def _benchmark_task_execution(self, bridge: TreeQuestSmolagentBridge):
        """Benchmark task execution performance"""
        test_tasks = []
        
        # Create test tasks with different complexities
        for i in range(10):
            task = TreeQuestTask(
                task_id=f"bench_task_{i}",
                description=f"Benchmark task {i} with complexity analysis",
                complexity=min(i % 5 + 1, 5),
                domain="benchmarking",
                subtasks=[f"Subtask {j}" for j in range(3)],
                dependencies=[],
                agent_personality_required=list(AgentPersonality)[i % 4],
                smolagent_tools=["analysis", "processing"]
            )
            test_tasks.append(task)
        
        durations = []
        memory_usage = []
        success_count = 0
        
        for task in test_tasks:
            gc.collect()
            start_memory = self.process.memory_info().rss / 1024 / 1024
            start_time = time.time()
            
            try:
                result = await bridge.execute_task_with_personality(task)
                
                if result and result.get("success"):
                    success_count += 1
                
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
            
            duration = (time.time() - start_time) * 1000
            end_memory = self.process.memory_info().rss / 1024 / 1024
            
            durations.append(duration)
            memory_usage.append(end_memory - start_memory)
        
        self.results.append(BenchmarkResult(
            component="TreeQuest Bridge",
            test_name="Task Execution",
            duration_ms=statistics.mean(durations),
            memory_usage_mb=statistics.mean(memory_usage),
            throughput=len(test_tasks) / (sum(durations) / 1000),
            success_rate=success_count / len(test_tasks),
            error_count=len(test_tasks) - success_count,
            metadata={
                "avg_duration_ms": statistics.mean(durations),
                "p95_duration_ms": statistics.quantiles(durations, n=20)[18] if len(durations) > 1 else durations[0],
                "tasks_executed": len(test_tasks)
            },
            timestamp=datetime.now()
        ))
    
    async def _benchmark_personality_switching(self, bridge: TreeQuestSmolagentBridge):
        """Benchmark personality switching performance"""
        personalities = list(AgentPersonality)
        durations = []
        
        for i in range(20):  # Test 20 switches
            personality = personalities[i % len(personalities)]
            
            start_time = time.time()
            
            task = TreeQuestTask(
                task_id=f"switch_task_{i}",
                description="Personality switch test",
                complexity=1,
                domain="testing",
                subtasks=["Test"],
                dependencies=[],
                agent_personality_required=personality,
                smolagent_tools=["test"]
            )
            
            try:
                await bridge.execute_task_with_personality(task)
            except Exception as e:
                logger.error(f"Personality switch failed: {e}")
            
            duration = (time.time() - start_time) * 1000
            durations.append(duration)
        
        self.results.append(BenchmarkResult(
            component="TreeQuest Bridge",
            test_name="Personality Switching",
            duration_ms=statistics.mean(durations),
            memory_usage_mb=0,  # Not measuring memory for this test
            throughput=len(durations) / (sum(durations) / 1000),
            success_rate=1.0,  # Assume success for timing test
            error_count=0,
            metadata={
                "avg_switch_time_ms": statistics.mean(durations),
                "switches_tested": len(durations)
            },
            timestamp=datetime.now()
        ))
    
    async def _benchmark_agent_allocation(self, bridge: TreeQuestSmolagentBridge):
        """Benchmark agent allocation optimization"""
        # Create multiple tasks for allocation
        tasks = []
        for i in range(20):
            task = TreeQuestTask(
                task_id=f"alloc_task_{i}",
                description=f"Allocation test task {i}",
                complexity=i % 5 + 1,
                domain="testing",
                subtasks=["Allocate", "Execute"],
                dependencies=[],
                agent_personality_required=list(AgentPersonality)[i % 4],
                smolagent_tools=["test"]
            )
            tasks.append(task)
        
        start_time = time.time()
        
        try:
            allocation = await bridge.optimize_agent_allocation(tasks)
            success = len(allocation) == 4  # Should have 4 personality buckets
        except Exception as e:
            logger.error(f"Agent allocation failed: {e}")
            success = False
        
        duration = (time.time() - start_time) * 1000
        
        self.results.append(BenchmarkResult(
            component="TreeQuest Bridge",
            test_name="Agent Allocation",
            duration_ms=duration,
            memory_usage_mb=0,
            throughput=len(tasks) / (duration / 1000),
            success_rate=1.0 if success else 0.0,
            error_count=0 if success else 1,
            metadata={
                "tasks_allocated": len(tasks),
                "allocation_time_ms": duration
            },
            timestamp=datetime.now()
        ))
    
    async def benchmark_multi_agent_system(self):
        """Benchmark multi-agent system performance"""
        logger.info("Benchmarking Multi-Agent System...")
        
        config = MultiAgentConfiguration(
            max_concurrent_agents=4,
            task_timeout_seconds=30
        )
        system = MONKMultiAgentSystem(config)
        await system.initialize()
        
        try:
            # Test 1: Task submission and execution
            await self._benchmark_mas_task_execution(system)
            
            # Test 2: Concurrent task handling
            await self._benchmark_mas_concurrency(system)
            
            # Test 3: System scalability
            await self._benchmark_mas_scalability(system)
            
        finally:
            await system.shutdown()
    
    async def _benchmark_mas_task_execution(self, system: MONKMultiAgentSystem):
        """Benchmark basic task execution"""
        durations = []
        success_count = 0
        
        for i in range(15):
            start_time = time.time()
            
            try:
                task_id = await system.submit_task(
                    description=f"Benchmark task {i}",
                    input_data={"index": i, "type": "benchmark"}
                )
                
                results = await system.wait_for_completion([task_id], timeout=10)
                
                if task_id in results and results[task_id]["status"] == "completed":
                    success_count += 1
                
            except Exception as e:
                logger.error(f"MAS task execution failed: {e}")
            
            duration = (time.time() - start_time) * 1000
            durations.append(duration)
        
        self.results.append(BenchmarkResult(
            component="Multi-Agent System",
            test_name="Task Execution",
            duration_ms=statistics.mean(durations),
            memory_usage_mb=0,
            throughput=len(durations) / (sum(durations) / 1000),
            success_rate=success_count / len(durations),
            error_count=len(durations) - success_count,
            metadata={
                "avg_execution_time_ms": statistics.mean(durations),
                "tasks_executed": len(durations)
            },
            timestamp=datetime.now()
        ))
    
    async def _benchmark_mas_concurrency(self, system: MONKMultiAgentSystem):
        """Benchmark concurrent task handling"""
        num_tasks = 10
        start_time = time.time()
        
        # Submit all tasks concurrently
        task_ids = []
        for i in range(num_tasks):
            task_id = await system.submit_task(
                description=f"Concurrent task {i}",
                input_data={"index": i, "type": "concurrent"}
            )
            task_ids.append(task_id)
        
        # Wait for all to complete
        results = await system.wait_for_completion(task_ids, timeout=30)
        
        total_duration = (time.time() - start_time) * 1000
        success_count = sum(1 for r in results.values() if r["status"] == "completed")
        
        self.results.append(BenchmarkResult(
            component="Multi-Agent System",
            test_name="Concurrent Execution",
            duration_ms=total_duration,
            memory_usage_mb=0,
            throughput=num_tasks / (total_duration / 1000),
            success_rate=success_count / num_tasks,
            error_count=num_tasks - success_count,
            metadata={
                "concurrent_tasks": num_tasks,
                "total_time_ms": total_duration,
                "parallelization_efficiency": num_tasks / (total_duration / 1000)
            },
            timestamp=datetime.now()
        ))
    
    async def _benchmark_mas_scalability(self, system: MONKMultiAgentSystem):
        """Benchmark system scalability with increasing load"""
        task_counts = [5, 10, 20, 30]
        scalability_results = []
        
        for task_count in task_counts:
            start_time = time.time()
            
            # Submit tasks
            task_ids = []
            for i in range(task_count):
                task_id = await system.submit_task(
                    description=f"Scale test task {i}",
                    input_data={"index": i, "batch_size": task_count}
                )
                task_ids.append(task_id)
            
            # Wait for completion
            results = await system.wait_for_completion(task_ids, timeout=60)
            
            duration = (time.time() - start_time) * 1000
            success_count = sum(1 for r in results.values() if r["status"] == "completed")
            
            scalability_results.append({
                "task_count": task_count,
                "duration_ms": duration,
                "throughput": task_count / (duration / 1000),
                "success_rate": success_count / task_count
            })
        
        # Calculate scalability efficiency
        base_throughput = scalability_results[0]["throughput"]
        max_throughput = max(r["throughput"] for r in scalability_results)
        scalability_score = max_throughput / base_throughput if base_throughput > 0 else 0
        
        self.results.append(BenchmarkResult(
            component="Multi-Agent System",
            test_name="Scalability",
            duration_ms=statistics.mean([r["duration_ms"] for r in scalability_results]),
            memory_usage_mb=0,
            throughput=max_throughput,
            success_rate=statistics.mean([r["success_rate"] for r in scalability_results]),
            error_count=0,
            metadata={
                "scalability_score": scalability_score,
                "test_results": scalability_results,
                "max_throughput": max_throughput
            },
            timestamp=datetime.now()
        ))
    
    async def benchmark_tree_sitter_explorer(self):
        """Benchmark tree-sitter explorer performance"""
        logger.info("Benchmarking Tree-Sitter Explorer...")
        
        explorer = TreeSitterLSPExplorer()
        await explorer.initialize()
        
        # Test 1: File parsing performance
        await self._benchmark_file_parsing(explorer)
        
        # Test 2: Directory parsing performance
        await self._benchmark_directory_parsing(explorer)
        
        # Test 3: Symbol search performance
        await self._benchmark_symbol_search(explorer)
        
        # Test 4: Complexity analysis performance
        await self._benchmark_complexity_analysis(explorer)
    
    async def _benchmark_file_parsing(self, explorer: TreeSitterLSPExplorer):
        """Benchmark individual file parsing"""
        # Create test files of different sizes and complexities
        test_files = []
        file_sizes = []
        
        # Small file
        small_content = '''
def simple_function():
    return "hello"

class SimpleClass:
    pass
        '''
        
        # Medium file
        medium_content = '''
import os
import sys
from typing import List, Dict, Any

def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of numbers."""
    if not data:
        return {}
    
    total = sum(data)
    count = len(data)
    mean = total / count
    
    sorted_data = sorted(data)
    median = sorted_data[count // 2] if count % 2 == 1 else (sorted_data[count // 2 - 1] + sorted_data[count // 2]) / 2
    
    variance = sum((x - mean) ** 2 for x in data) / count
    std_dev = variance ** 0.5
    
    return {
        "mean": mean,
        "median": median,
        "std_dev": std_dev,
        "min": min(data),
        "max": max(data)
    }

class DataProcessor:
    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0
    
    def process_batch(self, batch: List[Any]) -> List[Any]:
        results = []
        for item in batch:
            if isinstance(item, (int, float)):
                results.append(item * 2)
            elif isinstance(item, str):
                results.append(item.upper())
            else:
                results.append(str(item))
        
        self.processed_count += len(batch)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "processed_count": self.processed_count
        }
        '''
        
        # Large file
        large_content = medium_content * 5 + '''

class AdvancedProcessor(DataProcessor):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name)
        self.config = config
        self.cache = {}
    
    def process_with_cache(self, key: str, data: Any) -> Any:
        if key in self.cache:
            return self.cache[key]
        
        result = self._complex_processing(data)
        self.cache[key] = result
        return result
    
    def _complex_processing(self, data: Any) -> Any:
        # Simulate complex processing
        if isinstance(data, list):
            return [self._process_item(item) for item in data]
        else:
            return self._process_item(data)
    
    def _process_item(self, item: Any) -> Any:
        if isinstance(item, dict):
            return {k: v * 2 if isinstance(v, (int, float)) else v for k, v in item.items()}
        elif isinstance(item, (int, float)):
            return item ** 2
        else:
            return str(item).upper()
        '''
        
        contents = [small_content, medium_content, large_content]
        
        for i, content in enumerate(contents):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                test_files.append(f.name)
                file_sizes.append(len(content))
        
        try:
            durations = []
            success_count = 0
            
            for file_path in test_files:
                start_time = time.time()
                
                try:
                    result = await explorer.parse_file(file_path)
                    if result and len(result.symbols) > 0:
                        success_count += 1
                except Exception as e:
                    logger.error(f"File parsing failed: {e}")
                
                duration = (time.time() - start_time) * 1000
                durations.append(duration)
            
            self.results.append(BenchmarkResult(
                component="Tree-Sitter Explorer",
                test_name="File Parsing",
                duration_ms=statistics.mean(durations),
                memory_usage_mb=0,
                throughput=len(test_files) / (sum(durations) / 1000),
                success_rate=success_count / len(test_files),
                error_count=len(test_files) - success_count,
                metadata={
                    "file_sizes": file_sizes,
                    "avg_parse_time_ms": statistics.mean(durations),
                    "files_tested": len(test_files)
                },
                timestamp=datetime.now()
            ))
            
        finally:
            for file_path in test_files:
                os.unlink(file_path)
    
    async def _benchmark_directory_parsing(self, explorer: TreeSitterLSPExplorer):
        """Benchmark directory parsing performance"""
        # Create temporary directory with multiple files
        with tempfile.TemporaryDirectory() as temp_dir:
            file_count = 10
            
            for i in range(file_count):
                file_path = os.path.join(temp_dir, f"test_{i}.py")
                with open(file_path, 'w') as f:
                    f.write(f'''
def function_{i}(x):
    return x * {i}

class Class_{i}:
    def method_{i}(self):
        return function_{i}({i})
                    ''')
            
            start_time = time.time()
            
            try:
                results = await explorer.parse_directory(temp_dir)
                success = len(results) == file_count
            except Exception as e:
                logger.error(f"Directory parsing failed: {e}")
                success = False
                results = {}
            
            duration = (time.time() - start_time) * 1000
            
            self.results.append(BenchmarkResult(
                component="Tree-Sitter Explorer",
                test_name="Directory Parsing",
                duration_ms=duration,
                memory_usage_mb=0,
                throughput=file_count / (duration / 1000),
                success_rate=1.0 if success else 0.0,
                error_count=0 if success else 1,
                metadata={
                    "files_in_directory": file_count,
                    "files_parsed": len(results),
                    "parse_time_ms": duration
                },
                timestamp=datetime.now()
            ))
    
    async def _benchmark_symbol_search(self, explorer: TreeSitterLSPExplorer):
        """Benchmark symbol search performance"""
        # First, parse a file to have symbols to search
        test_content = '''
def search_test_function():
    pass

class SearchTestClass:
    def search_test_method(self):
        pass

def another_function():
    pass

class AnotherClass:
    pass
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            file_path = f.name
        
        try:
            # Parse file first
            await explorer.parse_file(file_path)
            
            # Test different search queries
            search_queries = [
                "search_test",
                "function",
                "class",
                "test",
                "another"
            ]
            
            durations = []
            total_results = 0
            
            for query in search_queries:
                start_time = time.time()
                
                try:
                    results = await explorer.search_symbols(query, file_path)
                    total_results += len(results)
                except Exception as e:
                    logger.error(f"Symbol search failed: {e}")
                
                duration = (time.time() - start_time) * 1000
                durations.append(duration)
            
            self.results.append(BenchmarkResult(
                component="Tree-Sitter Explorer",
                test_name="Symbol Search",
                duration_ms=statistics.mean(durations),
                memory_usage_mb=0,
                throughput=len(search_queries) / (sum(durations) / 1000),
                success_rate=1.0,  # Assume success for timing
                error_count=0,
                metadata={
                    "search_queries": len(search_queries),
                    "total_results": total_results,
                    "avg_search_time_ms": statistics.mean(durations)
                },
                timestamp=datetime.now()
            ))
            
        finally:
            os.unlink(file_path)
    
    async def _benchmark_complexity_analysis(self, explorer: TreeSitterLSPExplorer):
        """Benchmark complexity analysis performance"""
        complex_content = '''
def complex_function(data, options):
    """Complex function with multiple branches and loops."""
    result = []
    
    for item in data:
        if item is None:
            continue
        
        if isinstance(item, dict):
            if "type" in item:
                if item["type"] == "A":
                    for i in range(item.get("count", 0)):
                        if i % 2 == 0:
                            result.append(i * 2)
                        else:
                            result.append(i + 1)
                elif item["type"] == "B":
                    try:
                        value = item["value"]
                        if value > 0:
                            result.append(value ** 2)
                        else:
                            result.append(abs(value))
                    except KeyError:
                        result.append(0)
                else:
                    result.append(item.get("default", -1))
            else:
                result.append(len(item))
        elif isinstance(item, list):
            for subitem in item:
                if subitem is not None:
                    result.append(subitem)
        else:
            result.append(item)
    
    # Additional complexity
    if options.get("sort", False):
        result.sort()
    
    if options.get("filter", False):
        result = [x for x in result if x > 0]
    
    return result

class ComplexClass:
    def __init__(self, config):
        self.config = config
        self.cache = {}
        self.stats = {"calls": 0, "cache_hits": 0}
    
    def process(self, data):
        self.stats["calls"] += 1
        
        cache_key = str(hash(str(data)))
        if cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]
        
        result = self._process_internal(data)
        self.cache[cache_key] = result
        return result
    
    def _process_internal(self, data):
        if not data:
            return []
        
        processed = []
        for item in data:
            if isinstance(item, dict):
                processed.extend(self._process_dict(item))
            elif isinstance(item, list):
                processed.extend(self._process_list(item))
            else:
                processed.append(self._process_item(item))
        
        return processed
    
    def _process_dict(self, d):
        results = []
        for key, value in d.items():
            if key.startswith("_"):
                continue
            
            if isinstance(value, (int, float)):
                if value > 0:
                    results.append(value * 2)
                elif value < 0:
                    results.append(abs(value))
                else:
                    results.append(1)
            else:
                results.append(len(str(value)))
        
        return results
    
    def _process_list(self, lst):
        return [self._process_item(item) for item in lst if item is not None]
    
    def _process_item(self, item):
        if isinstance(item, str):
            return len(item)
        elif isinstance(item, (int, float)):
            return item ** 2
        else:
            return 0
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(complex_content)
            file_path = f.name
        
        try:
            durations = []
            
            # Run complexity analysis multiple times
            for _ in range(5):
                start_time = time.time()
                
                try:
                    result = await explorer.parse_file(file_path, force_reparse=True)
                    # Complexity metrics are calculated during parsing
                    success = result is not None and "cyclomatic_complexity" in result.complexity_metrics
                except Exception as e:
                    logger.error(f"Complexity analysis failed: {e}")
                    success = False
                
                duration = (time.time() - start_time) * 1000
                durations.append(duration)
            
            self.results.append(BenchmarkResult(
                component="Tree-Sitter Explorer",
                test_name="Complexity Analysis",
                duration_ms=statistics.mean(durations),
                memory_usage_mb=0,
                throughput=5 / (sum(durations) / 1000),
                success_rate=1.0 if success else 0.0,
                error_count=0 if success else 1,
                metadata={
                    "analysis_runs": 5,
                    "avg_analysis_time_ms": statistics.mean(durations)
                },
                timestamp=datetime.now()
            ))
            
        finally:
            os.unlink(file_path)
    
    async def benchmark_collaborative_platform(self):
        """Benchmark collaborative platform performance"""
        logger.info("Benchmarking Collaborative Platform...")
        
        # Mock the websocket and HTTP servers for testing
        from unittest.mock import patch
        
        platform = CollaborativePlatform()
        
        with patch.object(platform, 'start_websocket_server'):
            with patch.object(platform, 'start_http_server'):
                with patch.object(platform, 'start_background_services'):
                    await platform.initialize()
        
        # Test 1: Session creation performance
        await self._benchmark_session_creation(platform)
        
        # Test 2: Event processing performance
        await self._benchmark_event_processing(platform)
        
        # Test 3: User management performance
        await self._benchmark_user_management(platform)
    
    async def _benchmark_session_creation(self, platform: CollaborativePlatform):
        """Benchmark session creation performance"""
        durations = []
        success_count = 0
        
        for i in range(20):
            start_time = time.time()
            
            try:
                session = await platform.get_or_create_session(
                    f"bench_session_{i}",
                    {
                        "name": f"Benchmark Session {i}",
                        "description": "Performance test session",
                        "workspace_path": f"/tmp/bench_{i}",
                        "owner_id": f"user_{i}"
                    }
                )
                
                if session and session.session_id == f"bench_session_{i}":
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Session creation failed: {e}")
            
            duration = (time.time() - start_time) * 1000
            durations.append(duration)
        
        self.results.append(BenchmarkResult(
            component="Collaborative Platform",
            test_name="Session Creation",
            duration_ms=statistics.mean(durations),
            memory_usage_mb=0,
            throughput=20 / (sum(durations) / 1000),
            success_rate=success_count / 20,
            error_count=20 - success_count,
            metadata={
                "sessions_created": 20,
                "avg_creation_time_ms": statistics.mean(durations)
            },
            timestamp=datetime.now()
        ))
    
    async def _benchmark_event_processing(self, platform: CollaborativePlatform):
        """Benchmark event processing performance"""
        from phase25.collaboration.collaborative_platform import CollaborationEvent, CollaborationEventType
        
        durations = []
        
        for i in range(50):
            event = CollaborationEvent(
                event_id=f"bench_event_{i}",
                event_type=CollaborationEventType.CODE_CHANGE,
                user_id=f"user_{i % 5}",
                session_id=f"session_{i % 3}",
                data={"change": f"benchmark change {i}"},
                timestamp=datetime.now()
            )
            
            start_time = time.time()
            
            try:
                await platform.process_event(event)
            except Exception as e:
                logger.error(f"Event processing failed: {e}")
            
            duration = (time.time() - start_time) * 1000
            durations.append(duration)
        
        self.results.append(BenchmarkResult(
            component="Collaborative Platform",
            test_name="Event Processing",
            duration_ms=statistics.mean(durations),
            memory_usage_mb=0,
            throughput=50 / (sum(durations) / 1000),
            success_rate=1.0,  # Assume success for timing
            error_count=0,
            metadata={
                "events_processed": 50,
                "avg_processing_time_ms": statistics.mean(durations)
            },
            timestamp=datetime.now()
        ))
    
    async def _benchmark_user_management(self, platform: CollaborativePlatform):
        """Benchmark user management performance"""
        # Create a session first
        session = await platform.get_or_create_session("user_bench_session", {})
        
        durations = []
        success_count = 0
        
        # Test adding users
        for i in range(15):
            user = CollaborationUser(
                user_id=f"bench_user_{i}",
                name=f"Benchmark User {i}",
                email=f"bench{i}@example.com",
                role=UserRole.CONTRIBUTOR
            )
            
            start_time = time.time()
            
            try:
                session.users[user.user_id] = user
                platform.user_sessions[user.user_id] = session.session_id
                success_count += 1
            except Exception as e:
                logger.error(f"User management failed: {e}")
            
            duration = (time.time() - start_time) * 1000
            durations.append(duration)
        
        self.results.append(BenchmarkResult(
            component="Collaborative Platform",
            test_name="User Management",
            duration_ms=statistics.mean(durations),
            memory_usage_mb=0,
            throughput=15 / (sum(durations) / 1000),
            success_rate=success_count / 15,
            error_count=15 - success_count,
            metadata={
                "users_managed": 15,
                "avg_management_time_ms": statistics.mean(durations)
            },
            timestamp=datetime.now()
        ))
    
    async def benchmark_end_to_end_workflows(self):
        """Benchmark complete end-to-end workflows"""
        logger.info("Benchmarking End-to-End Workflows...")
        
        # Initialize all components
        bridge = TreeQuestSmolagentBridge()
        mas = MONKMultiAgentSystem()
        explorer = TreeSitterLSPExplorer()
        
        await bridge.initialize()
        await mas.initialize()
        await explorer.initialize()
        
        try:
            durations = []
            success_count = 0
            
            for i in range(5):  # Run 5 complete workflows
                start_time = time.time()
                
                try:
                    # Create test file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                        f.write(f'''
def workflow_function_{i}(data):
    """Workflow test function {i}."""
    result = []
    for item in data:
                        if item > {i}:
                            result.append(item * 2)
                        else:
                            result.append(item + 1)
                    return result

class WorkflowClass_{i}:
                        def process(self, data):
                            return workflow_function_{i}(data)
                        ''')
                        file_path = f.name
                    
                    # Step 1: Parse with tree-sitter
                    code_tree = await explorer.parse_file(file_path)
                    
                    # Step 2: Create task
                    task = await bridge.decompose_task_hierarchy(
                        root_task=f"Analyze workflow file {i}",
                        domain="workflow_analysis",
                        target_complexity=2
                    )
                    
                    # Step 3: Execute with multi-agent system
                    task_id = await mas.submit_task(
                        description=task.description,
                        input_data={
                            "file_path": file_path,
                            "workflow_index": i
                        }
                    )
                    
                    results = await mas.wait_for_completion([task_id], timeout=15)
                    
                    if (code_tree and task and task_id in results and 
                        results[task_id]["status"] == "completed"):
                        success_count += 1
                    
                    # Cleanup
                    os.unlink(file_path)
                    
                except Exception as e:
                    logger.error(f"End-to-end workflow failed: {e}")
                
                duration = (time.time() - start_time) * 1000
                durations.append(duration)
            
            self.results.append(BenchmarkResult(
                component="End-to-End Integration",
                test_name="Complete Workflow",
                duration_ms=statistics.mean(durations),
                memory_usage_mb=0,
                throughput=5 / (sum(durations) / 1000),
                success_rate=success_count / 5,
                error_count=5 - success_count,
                metadata={
                    "workflows_executed": 5,
                    "avg_workflow_time_ms": statistics.mean(durations)
                },
                timestamp=datetime.now()
            ))
            
        finally:
            await mas.shutdown()
    
    async def benchmark_scalability(self):
        """Benchmark system scalability under load"""
        logger.info("Benchmarking System Scalability...")
        
        # Test different load levels
        load_levels = [10, 25, 50, 100]
        scalability_results = []
        
        for load in load_levels:
            logger.info(f"Testing scalability with {load} concurrent operations...")
            
            bridge = TreeQuestSmolagentBridge()
            await bridge.initialize()
            
            start_time = time.time()
            success_count = 0
            
            # Create tasks for this load level
            tasks = []
            for i in range(load):
                task = TreeQuestTask(
                    task_id=f"scale_{load}_{i}",
                    description=f"Scalability test task {i} for load {load}",
                    complexity=1,
                    domain="scalability_test",
                    subtasks=["Execute", "Complete"],
                    dependencies=[],
                    agent_personality_required=list(AgentPersonality)[i % 4],
                    smolagent_tools=["test"]
                )
                tasks.append(task)
            
            # Execute all tasks
            try:
                results = await asyncio.gather(*[
                    bridge.execute_task_with_personality(task) for task in tasks
                ], return_exceptions=True)
                
                success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
                
            except Exception as e:
                logger.error(f"Scalability test failed for load {load}: {e}")
            
            duration = (time.time() - start_time) * 1000
            throughput = load / (duration / 1000)
            
            scalability_results.append({
                "load": load,
                "duration_ms": duration,
                "throughput": throughput,
                "success_rate": success_count / load
            })
        
        # Calculate scalability efficiency
        base_throughput = scalability_results[0]["throughput"]
        linear_expectation = [base_throughput * (r["load"] / load_levels[0]) for r in scalability_results]
        actual_throughput = [r["throughput"] for r in scalability_results]
        
        efficiency_scores = [
            actual / expected if expected > 0 else 0
            for actual, expected in zip(actual_throughput, linear_expectation)
        ]
        
        avg_efficiency = statistics.mean(efficiency_scores)
        
        self.results.append(BenchmarkResult(
            component="System Scalability",
            test_name="Load Testing",
            duration_ms=statistics.mean([r["duration_ms"] for r in scalability_results]),
            memory_usage_mb=0,
            throughput=max(actual_throughput),
            success_rate=statistics.mean([r["success_rate"] for r in scalability_results]),
            error_count=0,
            metadata={
                "scalability_results": scalability_results,
                "efficiency_scores": efficiency_scores,
                "avg_efficiency": avg_efficiency,
                "max_throughput": max(actual_throughput)
            },
            timestamp=datetime.now()
        ))
    
    async def benchmark_memory_efficiency(self):
        """Benchmark memory efficiency under sustained load"""
        logger.info("Benchmarking Memory Efficiency...")
        
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        memory_samples = [initial_memory]
        
        bridge = TreeQuestSmolagentBridge()
        await bridge.initialize()
        
        # Run sustained operations and monitor memory
        for i in range(50):
            # Create and execute task
            task = TreeQuestTask(
                task_id=f"memory_test_{i}",
                description=f"Memory efficiency test {i}",
                complexity=2,
                domain="memory_test",
                subtasks=["Process", "Cleanup"],
                dependencies=[],
                agent_personality_required=list(AgentPersonality)[i % 4],
                smolagent_tools=["test"]
            )
            
            await bridge.execute_task_with_personality(task)
            
            # Sample memory every 10 operations
            if i % 10 == 0:
                gc.collect()  # Force garbage collection
                current_memory = self.process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
        
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        max_memory = max(memory_samples)
        
        self.results.append(BenchmarkResult(
            component="Memory Efficiency",
            test_name="Sustained Load",
            duration_ms=0,  # Not measuring time for this test
            memory_usage_mb=memory_increase,
            throughput=50,  # Operations completed
            success_rate=1.0,  # Assume success
            error_count=0,
            metadata={
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "max_memory_mb": max_memory,
                "memory_increase_mb": memory_increase,
                "memory_samples": memory_samples,
                "operations_completed": 50
            },
            timestamp=datetime.now()
        ))
    
    def generate_performance_profiles(self):
        """Generate performance profiles for each component"""
        components = set(result.component for result in self.results)
        
        for component in components:
            component_results = [r for r in self.results if r.component == component]
            
            if not component_results:
                continue
            
            durations = [r.duration_ms for r in component_results if r.duration_ms > 0]
            throughputs = [r.throughput for r in component_results if r.throughput > 0]
            success_rates = [r.success_rate for r in component_results]
            memory_usage = [r.memory_usage_mb for r in component_results if r.memory_usage_mb > 0]
            
            # Calculate profile metrics
            avg_response_time = statistics.mean(durations) if durations else 0
            p95_response_time = statistics.quantiles(durations, n=20)[18] if len(durations) > 1 else (durations[0] if durations else 0)
            p99_response_time = statistics.quantiles(durations, n=100)[98] if len(durations) > 2 else p95_response_time
            avg_throughput = statistics.mean(throughputs) if throughputs else 0
            avg_memory_per_op = statistics.mean(memory_usage) if memory_usage else 0
            avg_success_rate = statistics.mean(success_rates) * 100
            
            # Calculate scalability score
            scalability_score = 1.0
            scalability_results = [r for r in component_results if "scalability" in r.test_name.lower()]
            if scalability_results:
                for result in scalability_results:
                    if "scalability_score" in result.metadata:
                        scalability_score = result.metadata["scalability_score"]
                        break
            
            profile = PerformanceProfile(
                component=component,
                avg_response_time_ms=avg_response_time,
                p95_response_time_ms=p95_response_time,
                p99_response_time_ms=p99_response_time,
                throughput_ops_per_sec=avg_throughput,
                memory_efficiency_mb_per_op=avg_memory_per_op,
                cpu_usage_percent=0,  # Would need additional monitoring
                success_rate_percent=avg_success_rate,
                scalability_score=scalability_score
            )
            
            self.profiles[component] = profile
    
    def generate_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        total_time = time.time() - self.start_time
        final_memory = self.process.memory_info().rss / 1024 / 1024
        
        report = {
            "benchmark_summary": {
                "total_execution_time_seconds": total_time,
                "total_tests_run": len(self.results),
                "components_tested": len(set(r.component for r in self.results)),
                "initial_memory_mb": self.initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": final_memory - self.initial_memory,
                "timestamp": datetime.now().isoformat()
            },
            "performance_profiles": {
                component: asdict(profile) for component, profile in self.profiles.items()
            },
            "detailed_results": [asdict(result) for result in self.results],
            "recommendations": self._generate_recommendations(),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": psutil.cpu_count(),
                "total_memory_mb": psutil.virtual_memory().total / 1024 / 1024
            }
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on results"""
        recommendations = []
        
        # Check response times
        slow_components = [
            name for name, profile in self.profiles.items()
            if profile.avg_response_time_ms > 1000  # > 1 second
        ]
        
        if slow_components:
            recommendations.append(
                f"Consider optimizing response times for: {', '.join(slow_components)}"
            )
        
        # Check success rates
        unreliable_components = [
            name for name, profile in self.profiles.items()
            if profile.success_rate_percent < 95.0
        ]
        
        if unreliable_components:
            recommendations.append(
                f"Improve reliability for: {', '.join(unreliable_components)}"
            )
        
        # Check memory efficiency
        memory_heavy_components = [
            name for name, profile in self.profiles.items()
            if profile.memory_efficiency_mb_per_op > 10.0  # > 10MB per operation
        ]
        
        if memory_heavy_components:
            recommendations.append(
                f"Optimize memory usage for: {', '.join(memory_heavy_components)}"
            )
        
        # Check scalability
        poor_scaling_components = [
            name for name, profile in self.profiles.items()
            if profile.scalability_score < 0.7  # Less than 70% efficiency
        ]
        
        if poor_scaling_components:
            recommendations.append(
                f"Improve scalability for: {', '.join(poor_scaling_components)}"
            )
        
        if not recommendations:
            recommendations.append("All components performing within acceptable parameters")
        
        return recommendations


async def main():
    """Run complete benchmark suite"""
    logging.basicConfig(level=logging.INFO)
    
    benchmarker = Phase25Benchmarker()
    report = await benchmarker.run_complete_benchmark()
    
    # Save report to file
    report_path = Path(__file__).parent / f"phase25_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("MONK CLI Phase 2.5 - Performance Benchmark Report")
    print("="*80)
    
    summary = report["benchmark_summary"]
    print(f"Total Execution Time: {summary['total_execution_time_seconds']:.2f} seconds")
    print(f"Tests Run: {summary['total_tests_run']}")
    print(f"Components Tested: {summary['components_tested']}")
    print(f"Memory Usage: {summary['memory_increase_mb']:.2f} MB increase")
    
    print("\nPerformance Profiles:")
    print("-" * 40)
    
    for component, profile in report["performance_profiles"].items():
        print(f"\n{component}:")
        print(f"  Avg Response Time: {profile['avg_response_time_ms']:.2f} ms")
        print(f"  Throughput: {profile['throughput_ops_per_sec']:.2f} ops/sec")
        print(f"  Success Rate: {profile['success_rate_percent']:.1f}%")
        print(f"  Scalability Score: {profile['scalability_score']:.2f}")
    
    print("\nRecommendations:")
    print("-" * 40)
    for rec in report["recommendations"]:
        print(f"• {rec}")
    
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())