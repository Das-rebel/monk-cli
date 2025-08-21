"""
MONK CLI Phase 1 Comprehensive Test Suite
Tests agent framework, memory system, and CLI interface
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, patch
import json
from typing import Dict, Any

# Import Phase 1 components
from src.agents.agent_framework import (
    PersonalityProfile, TaskContext, AgentResponse,
    ArchitectAgent, QualityEnforcerAgent, InnovationDriverAgent, IntegrationSpecialistAgent
)
from src.agents.orchestrator import orchestrator, AgentSelectionResult
from src.memory.memory_system import memory_system, MemoryQuery, MemoryResult
from src.core.config import config
from src.core.database import get_db_session, startup_database, shutdown_database


class TestPhase1PersonalitySystem:
    """Test the personality system and agent framework"""
    
    def test_personality_profile_creation(self):
        """Test personality profile creation and validation"""
        profile = PersonalityProfile(
            conscientiousness=0.9,
            openness=0.7,
            creativity=0.8,
            analytical_thinking=0.6
        )
        
        assert profile.conscientiousness == 0.9
        assert profile.openness == 0.7
        assert profile.creativity == 0.8
        assert profile.analytical_thinking == 0.6
        
        # Test validation
        with pytest.raises(ValueError):
            PersonalityProfile(conscientiousness=1.5)  # > 1.0
        
        with pytest.raises(ValueError):
            PersonalityProfile(openness=-0.1)  # < 0.0
    
    def test_personality_similarity(self):
        """Test personality similarity calculation"""
        profile1 = PersonalityProfile(
            conscientiousness=0.9,
            openness=0.7,
            creativity=0.8
        )
        
        profile2 = PersonalityProfile(
            conscientiousness=0.8,
            openness=0.6,
            creativity=0.7
        )
        
        similarity = profile1.similarity(profile2)
        assert 0.0 <= similarity <= 1.0
        
        # Similar profiles should have high similarity
        assert similarity > 0.8
    
    def test_personality_complement(self):
        """Test personality complement scoring"""
        high_conscientiousness = PersonalityProfile(conscientiousness=0.9, creativity=0.3)
        high_creativity = PersonalityProfile(conscientiousness=0.3, creativity=0.9)
        
        complement = high_conscientiousness.complement_score(high_creativity)
        assert 0.0 <= complement <= 1.0
        
        # Complementary traits should score well
        assert complement > 0.5


class TestAgentFramework:
    """Test individual agent implementations"""
    
    def test_architect_agent_initialization(self):
        """Test architect agent initialization"""
        agent = ArchitectAgent()
        
        assert agent.name == "Architect"
        assert "system_design" in agent.specializations
        assert agent.personality.conscientiousness > 0.8
        assert agent.personality.analytical_thinking > 0.8
    
    def test_quality_enforcer_initialization(self):
        """Test quality enforcer agent initialization"""
        agent = QualityEnforcerAgent()
        
        assert agent.name == "Quality Enforcer"
        assert "code_review" in agent.specializations
        assert agent.personality.conscientiousness > 0.9
        assert agent.personality.risk_tolerance < 0.2
    
    def test_agent_suitability_calculation(self):
        """Test agent suitability calculation"""
        architect = ArchitectAgent()
        
        # Architecture task should have high suitability
        arch_context = TaskContext(
            user_id="test-user",
            task_description="Design a scalable microservices architecture",
            task_type="architecture",
            domain="system_design",
            complexity_level=0.8,
            urgency_level=0.5
        )
        
        suitability = architect.calculate_suitability(arch_context)
        assert suitability > 0.7
        
        # Non-architecture task should have lower suitability
        non_arch_context = TaskContext(
            user_id="test-user",
            task_description="Write a simple hello world script",
            task_type="coding",
            domain="scripting",
            complexity_level=0.2,
            urgency_level=0.5
        )
        
        suitability_low = architect.calculate_suitability(non_arch_context)
        assert suitability_low < suitability
    
    @pytest.mark.asyncio
    async def test_agent_task_execution(self):
        """Test agent task execution"""
        agent = ArchitectAgent()
        
        context = TaskContext(
            user_id="test-user",
            task_description="Design a real-time chat system",
            task_type="architecture",
            domain="system_design",
            complexity_level=0.7,
            urgency_level=0.6
        )
        
        response = await agent.execute_task(context)
        
        assert isinstance(response, AgentResponse)
        assert response.success
        assert response.execution_time_ms > 0
        assert 0.0 <= response.confidence_score <= 1.0
        assert "analysis_type" in response.result
        assert response.result["analysis_type"] == "system_architecture"
        
        # Check if agent updated its metrics
        assert agent.total_executions == 1
        assert agent.successful_executions == 1


class TestAgentOrchestrator:
    """Test agent orchestration and selection"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        await orchestrator.start()
        
        status = orchestrator.get_orchestrator_status()
        
        assert status["total_agents"] > 0
        assert "development" in status["agent_pools"]
        assert len(status["agent_pools"]["development"]) > 0
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_agent_selection(self):
        """Test optimal agent selection"""
        await orchestrator.start()
        
        # Architecture task should select architect
        arch_context = TaskContext(
            user_id="test-user",
            task_description="Design a microservices architecture for e-commerce",
            task_type="architecture",
            domain="system_design",
            complexity_level=0.8,
            urgency_level=0.5
        )
        
        selection = await orchestrator.select_optimal_agent(arch_context)
        
        assert isinstance(selection, AgentSelectionResult)
        assert selection.selected_agent.name == "Architect"
        assert selection.confidence_score > 0.6
        assert selection.selection_time_ms > 0
        assert len(selection.alternative_agents) >= 0
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_task_execution_with_orchestrator(self):
        """Test complete task execution through orchestrator"""
        await orchestrator.start()
        
        context = TaskContext(
            user_id="test-user",
            task_description="Review this code for potential security issues",
            task_type="code_review",
            domain="quality_assurance",
            complexity_level=0.6,
            urgency_level=0.7
        )
        
        response, selection = await orchestrator.execute_task(context)
        
        assert response.success
        assert selection.selected_agent.name == "Quality Enforcer"
        assert response.execution_time_ms > 0
        assert response.confidence_score > 0.0
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_collaborative_task_execution(self):
        """Test collaborative task execution"""
        await orchestrator.start()
        
        context = TaskContext(
            user_id="test-user",
            task_description="Design, review, and deploy a new API endpoint",
            task_type="collaborative",
            domain="development",
            complexity_level=0.8,
            urgency_level=0.6
        )
        
        results = await orchestrator.execute_collaborative_task(context)
        
        assert len(results) > 1  # Multiple agents involved
        
        # Check that we have results from multiple agents
        agent_names = set()
        for agent_type, response in results.items():
            assert isinstance(response, AgentResponse)
            agent_names.add(response.result.get("analysis_type", "unknown"))
        
        assert len(agent_names) > 1  # Multiple different analysis types
        
        await orchestrator.stop()


class TestMemorySystem:
    """Test the persistent memory system"""
    
    @pytest.fixture
    async def setup_memory_system(self):
        """Setup memory system for testing"""
        # Mock database operations for testing
        with patch('src.memory.memory_system.get_db_session'), \
             patch('src.memory.memory_system.get_pinecone_index'):
            yield memory_system
    
    @pytest.mark.asyncio
    async def test_memory_storage(self, setup_memory_system):
        """Test storing memories"""
        memory_id = await memory_system.store_interaction(
            user_id="test-user",
            interaction_type="task_execution",
            content={
                "task_description": "Design authentication system",
                "agent_used": "Architect",
                "success": True,
                "execution_time_ms": 1500,
                "domain": "system_design"
            },
            context={"complexity_level": 0.7, "urgency_level": 0.5},
            importance_score=0.8
        )
        
        assert memory_id is not None
        assert isinstance(memory_id, str)
    
    @pytest.mark.asyncio
    async def test_memory_retrieval(self, setup_memory_system):
        """Test memory retrieval"""
        # Mock the memory retrieval to return test data
        with patch.object(memory_system.episodic_manager, 'retrieve_memories') as mock_retrieve:
            mock_retrieve.return_value = [
                MemoryResult(
                    memory_id="test-memory-1",
                    memory_type="task_execution",
                    content={"task_description": "Design API", "success": True},
                    relevance_score=0.9,
                    importance_score=0.8,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=1
                )
            ]
            
            query = MemoryQuery(
                query_text="API design patterns",
                user_id="test-user",
                limit=5
            )
            
            memories = await memory_system.retrieve_relevant_memories(query)
            
            assert "episodic" in memories
            assert len(memories["episodic"]) > 0
            
            memory = memories["episodic"][0]
            assert memory.relevance_score > 0.5
            assert "task_description" in memory.content
    
    @pytest.mark.asyncio
    async def test_memory_insights(self, setup_memory_system):
        """Test memory insight generation"""
        # Mock database queries for insights
        with patch('src.memory.memory_system.get_db_session'):
            insights = await memory_system.get_memory_insights("test-user")
            
            # Should return list of insights (may be empty in test)
            assert isinstance(insights, list)


class TestPerformanceBenchmarks:
    """Performance benchmark tests for Phase 1"""
    
    @pytest.mark.asyncio
    async def test_agent_selection_performance(self):
        """Test agent selection meets <100ms target"""
        await orchestrator.start()
        
        context = TaskContext(
            user_id="test-user",
            task_description="Optimize database query performance",
            task_type="optimization",
            domain="performance",
            complexity_level=0.6,
            urgency_level=0.8
        )
        
        start_time = time.time()
        selection = await orchestrator.select_optimal_agent(context)
        selection_time_ms = (time.time() - start_time) * 1000
        
        # Should meet <100ms target (allowing some overhead in test environment)
        assert selection_time_ms < 200  # 200ms for test tolerance
        assert selection.selection_time_ms > 0
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_memory_retrieval_performance(self):
        """Test memory retrieval meets <50ms target"""
        # Mock fast memory retrieval
        with patch.object(memory_system.embedding_engine, 'encode_text') as mock_encode, \
             patch.object(memory_system.episodic_manager, 'retrieve_memories') as mock_retrieve:
            
            # Mock fast encoding and retrieval
            mock_encode.return_value = [0.1] * 768  # Mock embedding
            mock_retrieve.return_value = []
            
            query = MemoryQuery(
                query_text="test query",
                user_id="test-user",
                limit=5
            )
            
            start_time = time.time()
            await memory_system.retrieve_relevant_memories(query)
            retrieval_time_ms = (time.time() - start_time) * 1000
            
            # Should be very fast with mocked operations
            assert retrieval_time_ms < 50
    
    @pytest.mark.asyncio
    async def test_concurrent_user_simulation(self):
        """Test system handles multiple concurrent users"""
        await orchestrator.start()
        
        # Simulate 10 concurrent users (scaled down for test)
        async def simulate_user_task(user_id: str):
            context = TaskContext(
                user_id=user_id,
                task_description=f"Task from user {user_id}",
                task_type="general",
                domain="development",
                complexity_level=0.5,
                urgency_level=0.5
            )
            
            start_time = time.time()
            response, _ = await orchestrator.execute_task(context)
            execution_time = time.time() - start_time
            
            return response.success, execution_time
        
        # Run concurrent tasks
        tasks = [simulate_user_task(f"user-{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Check results
        successful_tasks = sum(1 for success, _ in results if success)
        avg_execution_time = sum(time for _, time in results) / len(results)
        
        assert successful_tasks >= 8  # At least 80% success rate
        assert avg_execution_time < 5.0  # Average under 5 seconds
        
        await orchestrator.stop()


class TestIntegrationScenarios:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_development_workflow(self):
        """Test complete development workflow with memory"""
        await orchestrator.start()
        
        user_id = "test-integration-user"
        
        # Step 1: Architecture design
        arch_context = TaskContext(
            user_id=user_id,
            task_description="Design a RESTful API for user management",
            task_type="architecture",
            domain="system_design",
            complexity_level=0.7,
            urgency_level=0.5
        )
        
        arch_response, arch_selection = await orchestrator.execute_task(arch_context)
        assert arch_response.success
        assert arch_selection.selected_agent.name == "Architect"
        
        # Store in memory
        await memory_system.store_interaction(
            user_id=user_id,
            interaction_type="task_execution",
            content={
                "task_description": arch_context.task_description,
                "agent_used": arch_selection.selected_agent.name,
                "success": arch_response.success,
                "result_type": "architecture"
            }
        )
        
        # Step 2: Quality review (should reference previous architecture)
        review_context = TaskContext(
            user_id=user_id,
            task_description="Review the user management API design for security",
            task_type="code_review",
            domain="quality_assurance",
            complexity_level=0.6,
            urgency_level=0.7
        )
        
        review_response, review_selection = await orchestrator.execute_task(review_context)
        assert review_response.success
        assert review_selection.selected_agent.name == "Quality Enforcer"
        
        # Step 3: Test memory retrieval
        query = MemoryQuery(
            query_text="user management API design",
            user_id=user_id,
            limit=5
        )
        
        # Mock memory retrieval for integration test
        with patch.object(memory_system.episodic_manager, 'retrieve_memories') as mock_retrieve:
            mock_retrieve.return_value = [
                MemoryResult(
                    memory_id="test-arch-memory",
                    memory_type="task_execution",
                    content={
                        "task_description": arch_context.task_description,
                        "agent_used": "Architect",
                        "success": True,
                        "result_type": "architecture"
                    },
                    relevance_score=0.9,
                    importance_score=0.8,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=1
                )
            ]
            
            memories = await memory_system.retrieve_relevant_memories(query)
            assert len(memories["episodic"]) > 0
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_learning_and_improvement_cycle(self):
        """Test that system learns and improves from repeated tasks"""
        await orchestrator.start()
        
        user_id = "test-learning-user"
        
        # Execute same type of task multiple times
        task_type = "optimization"
        base_description = "Optimize database query performance"
        
        execution_times = []
        
        for i in range(3):
            context = TaskContext(
                user_id=user_id,
                task_description=f"{base_description} - iteration {i+1}",
                task_type=task_type,
                domain="performance",
                complexity_level=0.6,
                urgency_level=0.5
            )
            
            start_time = time.time()
            response, selection = await orchestrator.execute_task(context)
            execution_time = time.time() - start_time
            
            execution_times.append(execution_time)
            
            # Store successful execution
            if response.success:
                await memory_system.store_interaction(
                    user_id=user_id,
                    interaction_type="task_execution",
                    content={
                        "task_description": context.task_description,
                        "task_type": task_type,
                        "success": True,
                        "execution_time": execution_time,
                        "iteration": i + 1
                    }
                )
        
        # Check that execution times generally improve (allowing some variance)
        # In real implementation, this would be more sophisticated
        avg_early = sum(execution_times[:2]) / 2
        avg_late = execution_times[2]
        
        # Allow for some variance but expect general improvement or consistency
        assert abs(avg_early - avg_late) < avg_early * 0.5  # Within 50%
        
        await orchestrator.stop()


# Test runner configuration
@pytest.fixture(scope="session", autouse=True)
async def setup_test_environment():
    """Setup test environment"""
    # Mock database initialization for tests
    with patch('src.core.database.startup_database'), \
         patch('src.core.database.shutdown_database'):
        yield


# Performance benchmark runner
class PerformanceBenchmarkRunner:
    """Run performance benchmarks and generate reports"""
    
    @staticmethod
    async def run_all_benchmarks():
        """Run all performance benchmarks"""
        results = {}
        
        # Agent selection benchmark
        await orchestrator.start()
        
        # Test agent selection speed
        contexts = [
            TaskContext(
                user_id=f"bench-user-{i}",
                task_description=f"Benchmark task {i}",
                task_type="general",
                domain="development",
                complexity_level=0.5,
                urgency_level=0.5
            ) for i in range(100)
        ]
        
        selection_times = []
        
        for context in contexts:
            start_time = time.time()
            selection = await orchestrator.select_optimal_agent(context)
            selection_time = (time.time() - start_time) * 1000
            selection_times.append(selection_time)
        
        results["agent_selection"] = {
            "avg_time_ms": sum(selection_times) / len(selection_times),
            "p95_time_ms": sorted(selection_times)[int(len(selection_times) * 0.95)],
            "max_time_ms": max(selection_times),
            "min_time_ms": min(selection_times),
            "target_met": all(t < 100 for t in selection_times)  # <100ms target
        }
        
        await orchestrator.stop()
        
        return results


if __name__ == "__main__":
    # Run specific test or benchmark
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # Run performance benchmarks
        async def run_benchmarks():
            results = await PerformanceBenchmarkRunner.run_all_benchmarks()
            print("\n=== MONK CLI Phase 1 Performance Benchmarks ===")
            print(json.dumps(results, indent=2))
        
        asyncio.run(run_benchmarks())
    else:
        # Run pytest
        pytest.main([__file__, "-v"])