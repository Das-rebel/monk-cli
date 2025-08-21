"""
MONK CLI Phase 2.5 - Open Source Integration Performance Tests
Comprehensive testing and validation of all Phase 2.5 components
"""

import asyncio
import pytest
import time
import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any, Optional
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from phase25.bridges.treequest_smolagent_bridge import (
    TreeQuestSmolagentBridge, TreeQuestTask, AgentPersonality
)
from phase25.smolagents.multi_agent_system import (
    MONKMultiAgentSystem, MultiAgentConfiguration, AgentTask, TaskStatus
)
from phase25.lsp.tree_sitter_explorer import (
    TreeSitterLSPExplorer, CodeLanguage, NodeType, CodeSymbol
)
from phase25.collaboration.collaborative_platform import (
    CollaborativePlatform, CollaborationUser, UserRole, CollaborationEventType
)

logger = logging.getLogger(__name__)


class TestTreeQuestSmolagentBridge:
    """Test TreeQuest-Smolagent bridge functionality"""
    
    @pytest.fixture
    async def bridge(self):
        """Create bridge instance for testing"""
        bridge = TreeQuestSmolagentBridge()
        await bridge.initialize()
        return bridge
    
    @pytest.mark.asyncio
    async def test_bridge_initialization(self, bridge):
        """Test bridge initializes correctly"""
        assert bridge is not None
        assert len(bridge.personality_agent_mapping) == 4
        assert AgentPersonality.ANALYTICAL in bridge.personality_agent_mapping
        assert AgentPersonality.CREATIVE in bridge.personality_agent_mapping
        assert AgentPersonality.DETAIL_ORIENTED in bridge.personality_agent_mapping
        assert AgentPersonality.COLLABORATIVE in bridge.personality_agent_mapping
    
    @pytest.mark.asyncio
    async def test_task_decomposition(self, bridge):
        """Test hierarchical task decomposition"""
        task = await bridge.decompose_task_hierarchy(
            root_task="Implement a REST API for user management",
            domain="web_development",
            target_complexity=3
        )
        
        assert task is not None
        assert task.description == "Implement a REST API for user management"
        assert task.domain == "web_development"
        assert task.complexity <= 5
        assert len(task.subtasks) >= 3
        assert task.agent_personality_required in [
            AgentPersonality.ANALYTICAL,
            AgentPersonality.CREATIVE,
            AgentPersonality.DETAIL_ORIENTED,
            AgentPersonality.COLLABORATIVE
        ]
    
    @pytest.mark.asyncio
    async def test_personality_execution(self, bridge):
        """Test task execution with different personalities"""
        # Test analytical personality
        analytical_task = TreeQuestTask(
            task_id="test_analytical",
            description="Analyze code complexity",
            complexity=2,
            domain="code_analysis",
            subtasks=["Parse code", "Calculate metrics", "Generate report"],
            dependencies=[],
            agent_personality_required=AgentPersonality.ANALYTICAL,
            smolagent_tools=["python", "analysis"]
        )
        
        result = await bridge.execute_task_with_personality(analytical_task)
        
        assert result["success"] is True
        assert result["task_id"] == "test_analytical"
        assert "execution_time_ms" in result
        assert result["personality_used"] == AgentPersonality.ANALYTICAL.value
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, bridge):
        """Test performance metrics collection"""
        # Execute multiple tasks
        tasks = []
        for i in range(3):
            task = TreeQuestTask(
                task_id=f"perf_test_{i}",
                description=f"Performance test task {i}",
                complexity=1,
                domain="testing",
                subtasks=["Execute", "Validate"],
                dependencies=[],
                agent_personality_required=AgentPersonality.ANALYTICAL,
                smolagent_tools=["test"]
            )
            tasks.append(task)
        
        # Execute tasks
        results = []
        for task in tasks:
            result = await bridge.execute_task_with_personality(task)
            results.append(result)
        
        # Get metrics
        metrics = await bridge.get_agent_performance_metrics()
        
        assert metrics["total_tasks_executed"] >= 3
        assert "agent_usage" in metrics
        assert "personality_performance" in metrics
        assert metrics["average_execution_time_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_agent_allocation_optimization(self, bridge):
        """Test agent allocation optimization"""
        # Create multiple tasks with different requirements
        tasks = [
            TreeQuestTask(
                task_id="opt_test_1",
                description="Complex analysis task",
                complexity=4,
                domain="analysis",
                subtasks=["Analyze", "Report"],
                dependencies=[],
                agent_personality_required=AgentPersonality.ANALYTICAL,
                smolagent_tools=["python"]
            ),
            TreeQuestTask(
                task_id="opt_test_2",
                description="Creative design task", 
                complexity=3,
                domain="design",
                subtasks=["Design", "Iterate"],
                dependencies=[],
                agent_personality_required=AgentPersonality.CREATIVE,
                smolagent_tools=["design"]
            )
        ]
        
        allocation = await bridge.optimize_agent_allocation(tasks)
        
        assert isinstance(allocation, dict)
        assert len(allocation) == 4  # One entry per personality
        
        # Check that tasks are allocated
        total_allocated = sum(len(task_list) for task_list in allocation.values())
        assert total_allocated == len(tasks)


class TestMultiAgentSystem:
    """Test multi-agent system functionality"""
    
    @pytest.fixture
    async def multi_agent_system(self):
        """Create multi-agent system for testing"""
        config = MultiAgentConfiguration(
            max_concurrent_agents=2,
            task_timeout_seconds=30,
            enable_agent_communication=True
        )
        system = MONKMultiAgentSystem(config)
        await system.initialize()
        return system
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, multi_agent_system):
        """Test multi-agent system initialization"""
        status = await multi_agent_system.get_system_status()
        
        assert status["status"] == "ready"
        assert status["agents"]["total"] >= 4  # At least one per personality
        assert status["tasks"]["pending"] == 0
        assert status["tasks"]["running"] == 0
    
    @pytest.mark.asyncio
    async def test_task_submission_and_execution(self, multi_agent_system):
        """Test task submission and execution"""
        task_id = await multi_agent_system.submit_task(
            description="Test task execution",
            input_data={"test_data": "value"},
            required_personality=AgentPersonality.ANALYTICAL,
            priority=3
        )
        
        assert task_id is not None
        
        # Wait for completion
        results = await multi_agent_system.wait_for_completion([task_id], timeout=10)
        
        assert task_id in results
        assert results[task_id]["status"] in ["completed", "timeout"]
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, multi_agent_system):
        """Test concurrent task execution"""
        # Submit multiple tasks
        task_ids = []
        for i in range(3):
            task_id = await multi_agent_system.submit_task(
                description=f"Concurrent test task {i}",
                input_data={"task_index": i},
                priority=2
            )
            task_ids.append(task_id)
        
        # Wait for all to complete
        start_time = time.time()
        results = await multi_agent_system.wait_for_completion(task_ids, timeout=15)
        execution_time = time.time() - start_time
        
        # Verify results
        assert len(results) == 3
        for task_id in task_ids:
            assert task_id in results
        
        # Should be faster than sequential execution due to concurrency
        assert execution_time < 10  # Reasonable upper bound
    
    @pytest.mark.asyncio
    async def test_task_dependencies(self, multi_agent_system):
        """Test task dependency management"""
        # Submit parent task
        parent_task_id = await multi_agent_system.submit_task(
            description="Parent task",
            input_data={"type": "parent"}
        )
        
        # Submit dependent task
        dependent_task_id = await multi_agent_system.submit_task(
            description="Dependent task",
            input_data={"type": "dependent"},
            dependencies=[parent_task_id]
        )
        
        # Wait for completion
        results = await multi_agent_system.wait_for_completion(
            [parent_task_id, dependent_task_id], 
            timeout=20
        )
        
        assert len(results) == 2
        assert results[parent_task_id]["status"] == "completed"
        assert results[dependent_task_id]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_system_performance_metrics(self, multi_agent_system):
        """Test performance metrics collection"""
        # Execute some tasks to generate metrics
        task_ids = []
        for i in range(5):
            task_id = await multi_agent_system.submit_task(
                description=f"Metrics test task {i}",
                input_data={"index": i}
            )
            task_ids.append(task_id)
        
        await multi_agent_system.wait_for_completion(task_ids, timeout=15)
        
        status = await multi_agent_system.get_system_status()
        
        assert status["performance"]["total_tasks"] >= 5
        assert status["performance"]["completed_tasks"] >= 0
        assert "agent_utilization" in status["performance"]


class TestTreeSitterExplorer:
    """Test tree-sitter code exploration functionality"""
    
    @pytest.fixture
    async def explorer(self):
        """Create tree-sitter explorer for testing"""
        explorer = TreeSitterLSPExplorer()
        await explorer.initialize()
        return explorer
    
    @pytest.fixture
    def sample_python_file(self):
        """Create sample Python file for testing"""
        content = '''
def calculate_fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class Calculator:
    """Simple calculator class."""
    
    def __init__(self):
        self.result = 0
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        result = 0
        for i in range(b):
            result = self.add(result, a)
        return result

import math
from typing import List
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            return f.name
    
    @pytest.mark.asyncio
    async def test_file_parsing(self, explorer, sample_python_file):
        """Test file parsing functionality"""
        code_tree = await explorer.parse_file(sample_python_file)
        
        assert code_tree is not None
        assert code_tree.language == CodeLanguage.PYTHON
        assert len(code_tree.symbols) > 0
        
        # Check for expected symbols
        symbol_names = [symbol.name for symbol in code_tree.symbols]
        assert "calculate_fibonacci" in symbol_names
        assert "Calculator" in symbol_names
        
        # Check function symbols
        function_symbols = [s for s in code_tree.symbols if s.symbol_type == NodeType.FUNCTION_DEF]
        assert len(function_symbols) >= 3  # calculate_fibonacci, add, multiply
        
        # Check class symbols
        class_symbols = [s for s in code_tree.symbols if s.symbol_type == NodeType.CLASS_DEF]
        assert len(class_symbols) >= 1  # Calculator
        
        # Check import symbols
        import_symbols = [s for s in code_tree.symbols if s.symbol_type == NodeType.IMPORT]
        assert len(import_symbols) >= 2  # math, typing imports
    
    @pytest.mark.asyncio
    async def test_complexity_metrics(self, explorer, sample_python_file):
        """Test complexity metrics calculation"""
        code_tree = await explorer.parse_file(sample_python_file)
        
        assert "cyclomatic_complexity" in code_tree.complexity_metrics
        assert "function_count" in code_tree.complexity_metrics
        assert "class_count" in code_tree.complexity_metrics
        assert "lines_of_code" in code_tree.complexity_metrics
        
        metrics = code_tree.complexity_metrics
        assert metrics["function_count"] >= 3
        assert metrics["class_count"] >= 1
        assert metrics["lines_of_code"] > 20
        assert metrics["maintainability_index"] >= 0
    
    @pytest.mark.asyncio
    async def test_symbol_search(self, explorer, sample_python_file):
        """Test symbol search functionality"""
        await explorer.parse_file(sample_python_file)
        
        # Search for function
        results = await explorer.search_symbols("fibonacci")
        assert len(results) >= 1
        
        # Search by symbol type
        function_results = await explorer.search_symbols(
            "add", 
            symbol_types=[NodeType.FUNCTION_DEF]
        )
        assert len(function_results) >= 1
        
        # Search by language
        python_results = await explorer.search_symbols(
            "Calculator",
            language=CodeLanguage.PYTHON
        )
        assert len(python_results) >= 1
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, explorer, sample_python_file):
        """Test parsing performance metrics"""
        # Parse file multiple times to generate metrics
        for _ in range(3):
            await explorer.parse_file(sample_python_file)
        
        metrics = await explorer.get_metrics_summary()
        
        assert "parsing_performance" in metrics
        assert "code_analysis" in metrics
        assert "system_status" in metrics
        
        perf_metrics = metrics["parsing_performance"]
        assert perf_metrics["files_parsed"] >= 3
        assert perf_metrics["average_parse_time_ms"] > 0
        
        code_metrics = metrics["code_analysis"]
        assert code_metrics["total_symbols"] > 0
        assert "symbol_types" in code_metrics
        assert "languages" in code_metrics
    
    @pytest.mark.asyncio 
    async def test_directory_parsing(self, explorer):
        """Test directory parsing functionality"""
        # Create temporary directory with sample files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create Python file
            py_file = os.path.join(temp_dir, "test.py")
            with open(py_file, 'w') as f:
                f.write("def test_function():\n    pass\n")
            
            # Create JavaScript file
            js_file = os.path.join(temp_dir, "test.js")
            with open(js_file, 'w') as f:
                f.write("function testFunction() {\n    return true;\n}\n")
            
            # Parse directory
            parsed_files = await explorer.parse_directory(temp_dir, recursive=False)
            
            assert len(parsed_files) >= 2
            
            # Check that both files were parsed
            file_paths = list(parsed_files.keys())
            assert any("test.py" in path for path in file_paths)
            assert any("test.js" in path for path in file_paths)


class TestCollaborativePlatform:
    """Test collaborative development platform"""
    
    @pytest.fixture
    async def platform(self):
        """Create collaborative platform for testing"""
        platform = CollaborativePlatform()
        
        # Mock websocket server to avoid port conflicts in tests
        with patch.object(platform, 'start_websocket_server'):
            with patch.object(platform, 'start_http_server'):
                with patch.object(platform, 'start_background_services'):
                    await platform.initialize()
        
        return platform
    
    @pytest.mark.asyncio
    async def test_platform_initialization(self, platform):
        """Test platform initialization"""
        assert platform is not None
        assert len(platform.sessions) == 0
        assert len(platform.websocket_connections) == 0
        assert platform.metrics["active_sessions"] == 0
    
    @pytest.mark.asyncio
    async def test_session_creation(self, platform):
        """Test collaboration session creation"""
        session_config = {
            "name": "Test Session",
            "description": "Testing collaboration",
            "workspace_path": "/tmp/test",
            "owner_id": "user123"
        }
        
        session = await platform.get_or_create_session("test_session", session_config)
        
        assert session is not None
        assert session.session_id == "test_session"
        assert session.name == "Test Session"
        assert session.owner_id == "user123"
        assert len(session.users) == 0
    
    @pytest.mark.asyncio
    async def test_user_management(self, platform):
        """Test user join/leave functionality"""
        # Create session
        session = await platform.get_or_create_session("user_test", {})
        
        # Simulate user joining
        user_data = {
            "session_id": "user_test",
            "user": {
                "user_id": "user123",
                "name": "Test User",
                "email": "test@example.com",
                "role": "contributor"
            }
        }
        
        # Mock connection
        mock_connection = Mock()
        platform.websocket_connections["conn123"] = mock_connection
        
        await platform.handle_join_session("conn123", user_data)
        
        # Verify user was added
        assert "user123" in session.users
        assert session.users["user123"].name == "Test User"
        assert platform.user_sessions["user123"] == "user_test"
    
    @pytest.mark.asyncio
    async def test_event_processing(self, platform):
        """Test collaboration event processing"""
        from phase25.collaboration.collaborative_platform import CollaborationEvent, CollaborationEventType
        
        event = CollaborationEvent(
            event_id="test_event",
            event_type=CollaborationEventType.CODE_CHANGE,
            user_id="user123",
            session_id="test_session",
            data={"test": "data"},
            timestamp=time.time()
        )
        
        await platform.process_event(event)
        
        # Verify event was processed
        assert len(platform.event_history) >= 1
        assert platform.event_history[-1].event_id == "test_event"
        assert platform.metrics["events_processed"] >= 1


class TestIntegrationPerformance:
    """Performance and integration tests for all Phase 2.5 components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Initialize all components
        bridge = TreeQuestSmolagentBridge()
        multi_agent = MONKMultiAgentSystem()
        explorer = TreeSitterLSPExplorer()
        
        await bridge.initialize()
        await multi_agent.initialize()
        await explorer.initialize()
        
        # Create sample code file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def complex_algorithm(data):
    """Complex algorithm implementation."""
    result = []
    for item in data:
        if item % 2 == 0:
            result.append(item * 2)
        else:
            result.append(item + 1)
    return result

class DataProcessor:
    def __init__(self):
        self.processed_count = 0
    
    def process(self, data):
        result = complex_algorithm(data)
        self.processed_count += len(result)
        return result
            ''')
            file_path = f.name
        
        try:
            # Step 1: Parse code with tree-sitter
            start_time = time.time()
            code_tree = await explorer.parse_file(file_path)
            parse_time = time.time() - start_time
            
            assert code_tree is not None
            assert parse_time < 1.0  # Should parse quickly
            
            # Step 2: Create task for code analysis
            task = await bridge.decompose_task_hierarchy(
                root_task=f"Analyze and optimize code in {file_path}",
                domain="code_optimization",
                target_complexity=3
            )
            
            assert task is not None
            assert len(task.subtasks) >= 3
            
            # Step 3: Execute task with multi-agent system
            start_time = time.time()
            task_id = await multi_agent.submit_task(
                description=task.description,
                input_data={
                    "file_path": file_path,
                    "code_tree": code_tree.__dict__,
                    "analysis_type": "optimization"
                },
                required_personality=task.agent_personality_required
            )
            
            results = await multi_agent.wait_for_completion([task_id], timeout=30)
            execution_time = time.time() - start_time
            
            assert task_id in results
            assert results[task_id]["status"] == "completed"
            assert execution_time < 30  # Should complete within timeout
            
            # Step 4: Verify performance metrics
            bridge_metrics = await bridge.get_agent_performance_metrics()
            system_status = await multi_agent.get_system_status()
            explorer_metrics = await explorer.get_metrics_summary()
            
            assert bridge_metrics["total_tasks_executed"] >= 1
            assert system_status["performance"]["total_tasks"] >= 1
            assert explorer_metrics["parsing_performance"]["files_parsed"] >= 1
            
        finally:
            # Cleanup
            os.unlink(file_path)
            await multi_agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_scalability_metrics(self):
        """Test system scalability with multiple concurrent operations"""
        # Initialize systems
        bridge = TreeQuestSmolagentBridge()
        multi_agent = MONKMultiAgentSystem()
        explorer = TreeSitterLSPExplorer()
        
        await bridge.initialize()
        await multi_agent.initialize()
        await explorer.initialize()
        
        # Create multiple test files
        test_files = []
        for i in range(5):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(f'''
def function_{i}(x):
    """Function {i} implementation."""
    return x * {i} + {i}

class Class_{i}:
    def method_{i}(self):
        return function_{i}({i})
                ''')
                test_files.append(f.name)
        
        try:
            # Test concurrent parsing
            start_time = time.time()
            parse_tasks = [explorer.parse_file(file_path) for file_path in test_files]
            parse_results = await asyncio.gather(*parse_tasks)
            parse_time = time.time() - start_time
            
            assert len(parse_results) == 5
            assert all(result is not None for result in parse_results)
            assert parse_time < 5.0  # Should handle concurrent parsing efficiently
            
            # Test concurrent task execution
            start_time = time.time()
            task_ids = []
            for i, file_path in enumerate(test_files):
                task_id = await multi_agent.submit_task(
                    description=f"Process file {i}",
                    input_data={"file_path": file_path, "index": i},
                    priority=1
                )
                task_ids.append(task_id)
            
            execution_results = await multi_agent.wait_for_completion(task_ids, timeout=60)
            execution_time = time.time() - start_time
            
            assert len(execution_results) == 5
            assert execution_time < 60  # Should handle concurrent execution
            
            # Verify all tasks completed successfully
            completed_count = sum(
                1 for result in execution_results.values() 
                if result["status"] == "completed"
            )
            assert completed_count >= 4  # Allow for some potential timeouts
            
        finally:
            # Cleanup
            for file_path in test_files:
                os.unlink(file_path)
            await multi_agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_and_resource_usage(self):
        """Test memory efficiency and resource usage"""
        import psutil
        import gc
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize systems
        bridge = TreeQuestSmolagentBridge()
        multi_agent = MONKMultiAgentSystem()
        explorer = TreeSitterLSPExplorer()
        
        await bridge.initialize()
        await multi_agent.initialize()
        await explorer.initialize()
        
        init_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform intensive operations
        for i in range(10):
            # Create and parse file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(f'''
def intensive_function_{i}():
    data = [x for x in range(1000)]
    result = []
    for item in data:
        if item % 2 == 0:
            result.append(item ** 2)
    return result

class IntensiveClass_{i}:
    def __init__(self):
        self.data = list(range(100))
    
    def process(self):
        return [x * 2 for x in self.data]
                ''')
                file_path = f.name
            
            try:
                # Parse and analyze
                code_tree = await explorer.parse_file(file_path)
                
                # Submit task
                task_id = await multi_agent.submit_task(
                    description=f"Intensive task {i}",
                    input_data={"iteration": i, "file_path": file_path}
                )
                
                results = await multi_agent.wait_for_completion([task_id], timeout=10)
                
            finally:
                os.unlink(file_path)
            
            # Force garbage collection
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory usage should be reasonable
        memory_increase = final_memory - initial_memory
        assert memory_increase < 500  # Less than 500MB increase
        
        # Cleanup
        await multi_agent.shutdown()
        
        logger.info(f"Memory usage: Initial={initial_memory:.1f}MB, "
                   f"After init={init_memory:.1f}MB, Final={final_memory:.1f}MB")


@pytest.mark.asyncio
async def test_complete_integration_suite():
    """Run complete integration test suite"""
    logger.info("Starting complete Phase 2.5 integration test suite")
    
    # Test individual components
    logger.info("Testing TreeQuest-Smolagent Bridge...")
    bridge_test = TestTreeQuestSmolagentBridge()
    bridge = TreeQuestSmolagentBridge()
    await bridge.initialize()
    
    await bridge_test.test_bridge_initialization(bridge)
    await bridge_test.test_task_decomposition(bridge)
    await bridge_test.test_personality_execution(bridge)
    
    logger.info("Testing Multi-Agent System...")
    mas_test = TestMultiAgentSystem()
    mas = MONKMultiAgentSystem()
    await mas.initialize()
    
    await mas_test.test_system_initialization(mas)
    await mas_test.test_task_submission_and_execution(mas)
    
    logger.info("Testing Tree-Sitter Explorer...")
    explorer_test = TestTreeSitterExplorer()
    explorer = TreeSitterLSPExplorer()
    await explorer.initialize()
    
    # Create sample file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('''
def test_function():
    return "test"

class TestClass:
    pass
        ''')
        sample_file = f.name
    
    try:
        await explorer_test.test_file_parsing(explorer, sample_file)
    finally:
        os.unlink(sample_file)
    
    logger.info("Testing Integration Performance...")
    perf_test = TestIntegrationPerformance()
    await perf_test.test_end_to_end_workflow()
    
    # Cleanup
    await mas.shutdown()
    
    logger.info("Phase 2.5 integration test suite completed successfully!")


if __name__ == "__main__":
    # Run basic integration test
    asyncio.run(test_complete_integration_suite())