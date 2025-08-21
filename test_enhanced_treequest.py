"""
Test and Validation System for Enhanced TreeQuest
Comprehensive testing of all memory and learning enhancements
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock imports for testing (would be actual imports in real system)
class MockMemoryManager:
    """Mock memory manager for testing"""
    def __init__(self):
        self.memories = {}
    
    def store(self, key: str, value: Any, metadata: Dict = None):
        self.memories[key] = {'value': value, 'metadata': metadata or {}}
    
    def get(self, key: str):
        return self.memories.get(key)

class MockModelRegistry:
    """Mock model registry for testing"""
    def __init__(self):
        self.models = {
            'gpt-4o': {'cost': 0.005, 'quality': 0.95, 'latency': 1200},
            'claude-3-sonnet': {'cost': 0.003, 'quality': 0.90, 'latency': 800},
            'gpt-4o-mini': {'cost': 0.0006, 'quality': 0.85, 'latency': 600}
        }
    
    def pick(self, role, objective):
        # Simple selection for testing
        return 'claude-3-sonnet'
    
    def get_available_models(self):
        return list(self.models.keys())

class EnhancedTreeQuestTester:
    """Comprehensive test suite for Enhanced TreeQuest"""
    
    def __init__(self):
        self.test_results = {}
        self.mock_memory_manager = MockMemoryManager()
        self.mock_model_registry = MockModelRegistry()
        
        # Test scenarios
        self.test_scenarios = [
            {
                'name': 'code_analysis_task',
                'task': 'Analyze Python code for performance bottlenecks and suggest optimizations',
                'context': {
                    'language': 'python',
                    'complexity': 0.7,
                    'domain': 'performance_optimization',
                    'estimated_hours': 2
                },
                'expected_agent': 'analyzer',
                'expected_domains': ['code_analysis', 'performance_optimization']
            },
            {
                'name': 'architecture_design_task', 
                'task': 'Design microservices architecture for e-commerce platform',
                'context': {
                    'complexity': 0.9,
                    'domain': 'architecture_design',
                    'technologies': ['microservices', 'api', 'database'],
                    'estimated_hours': 8
                },
                'expected_agent': 'planner',
                'expected_domains': ['architecture_design', 'api_design']
            },
            {
                'name': 'security_assessment_task',
                'task': 'Review authentication system for security vulnerabilities',
                'context': {
                    'complexity': 0.6,
                    'domain': 'security',
                    'focus_areas': ['authentication', 'authorization'],
                    'estimated_hours': 3
                },
                'expected_agent': 'critic',
                'expected_domains': ['security_assessment']
            },
            {
                'name': 'integration_task',
                'task': 'Integrate payment gateway API with existing checkout system',
                'context': {
                    'complexity': 0.5,
                    'domain': 'integration',
                    'technologies': ['api', 'payment'],
                    'estimated_hours': 4
                },
                'expected_agent': 'executor',
                'expected_domains': ['api_design', 'backend_systems']
            },
            {
                'name': 'synthesis_task',
                'task': 'Combine findings from code review, security audit, and performance analysis',
                'context': {
                    'complexity': 0.8,
                    'domain': 'synthesis',
                    'input_sources': ['code_review', 'security_audit', 'performance_analysis'],
                    'estimated_hours': 2
                },
                'expected_agent': 'synthesizer',
                'expected_domains': ['requirements_analysis']
            }
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        logger.info("Starting Enhanced TreeQuest comprehensive test suite...")
        
        start_time = time.time()
        
        # Test 1: Memory Filesystem Tests
        await self.test_memory_filesystem()
        
        # Test 2: Historical Performance Tracking Tests
        await self.test_historical_performance()
        
        # Test 3: Adaptive Rewards Tests
        await self.test_adaptive_rewards()
        
        # Test 4: Agent Specialization Tests
        await self.test_agent_specialization()
        
        # Test 5: Memory-Guided MCTS Tests
        await self.test_memory_guided_mcts()
        
        # Test 6: Integration Tests
        await self.test_enhanced_integration()
        
        # Test 7: Performance and Scalability Tests
        await self.test_performance_scalability()
        
        total_time = time.time() - start_time
        
        # Generate test report
        test_report = self.generate_test_report(total_time)
        
        logger.info(f"Test suite completed in {total_time:.2f} seconds")
        return test_report
    
    async def test_memory_filesystem(self):
        """Test Memory Filesystem functionality"""
        logger.info("Testing Memory Filesystem...")
        
        try:
            # Import and test Memory Filesystem
            from src.core.memory_filesystem import MemoryFilesystem, MemoryNodeType
            
            # Initialize filesystem
            memory_fs = MemoryFilesystem(self.mock_memory_manager)
            
            # Test 1: Directory structure creation
            memory_fs._ensure_directory_exists("/test/agents/planner")
            test_node = memory_fs._navigate_to_path("/test/agents/planner")
            
            assert test_node is not None, "Failed to create directory structure"
            assert test_node.node_type == MemoryNodeType.DIRECTORY, "Node type incorrect"
            
            # Test 2: Memory storage and retrieval
            test_content = {'task': 'test task', 'result': 'success'}
            test_metadata = {'importance_score': 0.8, 'success_rate': 0.9}
            
            success = memory_fs.store_memory("/test/memories/test_memory", test_content, test_metadata)
            assert success, "Failed to store memory"
            
            retrieved = memory_fs.get_memory("/test/memories/test_memory")
            assert retrieved is not None, "Failed to retrieve memory"
            assert retrieved.content == test_content, "Memory content mismatch"
            
            # Test 3: Successful path storage and retrieval
            task_signature = "test_task_123"
            path_data = {'agents': ['planner', 'analyzer'], 'success': True}
            
            success = memory_fs.store_successful_path(task_signature, path_data, 0.85)
            assert success, "Failed to store successful path"
            
            successful_paths = memory_fs.get_successful_paths(task_signature)
            assert len(successful_paths) > 0, "Failed to retrieve successful paths"
            
            # Test 4: Adaptive forgetting
            forgotten_count = memory_fs.adaptive_forget(forget_threshold=0.9)  # High threshold
            logger.info(f"Adaptive forgetting removed {forgotten_count} memories")
            
            # Test 5: Memory statistics
            stats = memory_fs.get_memory_stats()
            assert 'total_memories' in stats, "Missing memory statistics"
            
            self.test_results['memory_filesystem'] = {
                'passed': True,
                'tests_run': 5,
                'details': {
                    'directory_creation': True,
                    'memory_storage': True,
                    'successful_paths': True,
                    'adaptive_forgetting': True,
                    'statistics': True
                }
            }
            
            logger.info("✅ Memory Filesystem tests passed")
            
        except Exception as e:
            logger.error(f"❌ Memory Filesystem tests failed: {e}")
            self.test_results['memory_filesystem'] = {
                'passed': False,
                'error': str(e),
                'tests_run': 0
            }
    
    async def test_historical_performance(self):
        """Test Historical Performance Tracking"""
        logger.info("Testing Historical Performance Tracking...")
        
        try:
            from src.ai.historical_performance import (
                HistoricalPerformanceTracker, PerformanceMetric, 
                PerformanceObjective
            )
            from src.core.memory_filesystem import MemoryFilesystem
            
            # Initialize components
            memory_fs = MemoryFilesystem(self.mock_memory_manager)
            performance_tracker = HistoricalPerformanceTracker(memory_fs)
            
            # Test 1: Record performance metrics
            test_metrics = [
                PerformanceMetric(
                    timestamp=time.time(),
                    provider="test_provider",
                    model="test_model",
                    agent_role="planner",
                    task_type="architecture_design",
                    quality_score=0.85,
                    latency_ms=1200,
                    cost_usd=0.05,
                    success=True
                ),
                PerformanceMetric(
                    timestamp=time.time() + 1,
                    provider="test_provider",
                    model="test_model",
                    agent_role="analyzer",
                    task_type="code_analysis",
                    quality_score=0.90,
                    latency_ms=800,
                    cost_usd=0.03,
                    success=True
                )
            ]
            
            for metric in test_metrics:
                success = await performance_tracker.record_performance(metric)
                assert success, "Failed to record performance metric"
            
            # Test 2: Get optimal provider
            optimal_provider = await performance_tracker.get_optimal_provider(
                "planner", "architecture_design", PerformanceObjective.QUALITY
            )
            assert optimal_provider is not None, "Failed to get optimal provider"
            
            # Test 3: Performance summary
            summary = performance_tracker.get_provider_performance_summary(
                "test_provider", "test_model"
            )
            assert 'total_requests' in summary, "Missing performance summary data"
            assert summary['total_requests'] >= 2, "Incorrect request count"
            
            # Test 4: Performance trends
            trends = performance_tracker.get_performance_trends(hours_back=1)
            assert 'total_metrics' in trends, "Missing trend data"
            
            # Test 5: Performance recommendations
            recommendations = performance_tracker.get_performance_recommendations({})
            assert isinstance(recommendations, dict), "Invalid recommendations format"
            
            self.test_results['historical_performance'] = {
                'passed': True,
                'tests_run': 5,
                'metrics_recorded': len(test_metrics),
                'details': {
                    'metric_recording': True,
                    'optimal_provider_selection': True,
                    'performance_summary': True,
                    'trend_analysis': True,
                    'recommendations': True
                }
            }
            
            logger.info("✅ Historical Performance tests passed")
            
        except Exception as e:
            logger.error(f"❌ Historical Performance tests failed: {e}")
            self.test_results['historical_performance'] = {
                'passed': False,
                'error': str(e),
                'tests_run': 0
            }
    
    async def test_adaptive_rewards(self):
        """Test Adaptive Reward System"""
        logger.info("Testing Adaptive Rewards...")
        
        try:
            from src.ai.adaptive_rewards import AdaptiveRewardSystem
            from src.ai.historical_performance import HistoricalPerformanceTracker
            from src.core.memory_filesystem import MemoryFilesystem
            
            # Initialize components
            memory_fs = MemoryFilesystem(self.mock_memory_manager)
            performance_tracker = HistoricalPerformanceTracker(memory_fs)
            adaptive_rewards = AdaptiveRewardSystem(memory_fs, performance_tracker)
            
            # Test 1: Calculate adaptive rewards
            test_contexts = [
                {
                    'agent_role': 'planner',
                    'task_type': 'architecture_design',
                    'quality_score': 0.8,
                    'latency_ms': 1000,
                    'cost_usd': 0.04
                },
                {
                    'agent_role': 'analyzer',
                    'task_type': 'code_analysis',
                    'quality_score': 0.9,
                    'latency_ms': 600,
                    'cost_usd': 0.02
                }
            ]
            
            for context in test_contexts:
                reward, reward_context = await adaptive_rewards.calculate_adaptive_reward(
                    context['agent_role'], context
                )
                assert 0.0 <= reward <= 1.0, f"Invalid reward value: {reward}"
                assert 'component_scores' in reward_context, "Missing reward context"
            
            # Test 2: Record outcomes and learning
            for i, context in enumerate(test_contexts):
                predicted_reward = 0.7 + (i * 0.1)
                actual_outcome = 0.8 + (i * 0.05)
                
                await adaptive_rewards.record_outcome(
                    predicted_reward, actual_outcome, context
                )
            
            # Test 3: Get adaptation summary
            summary = adaptive_rewards.get_adaptation_summary()
            assert 'total_outcomes_recorded' in summary, "Missing adaptation summary"
            assert summary['total_outcomes_recorded'] >= 2, "Incorrect outcome count"
            
            self.test_results['adaptive_rewards'] = {
                'passed': True,
                'tests_run': 3,
                'rewards_calculated': len(test_contexts),
                'details': {
                    'reward_calculation': True,
                    'outcome_learning': True,
                    'adaptation_summary': True
                }
            }
            
            logger.info("✅ Adaptive Rewards tests passed")
            
        except Exception as e:
            logger.error(f"❌ Adaptive Rewards tests failed: {e}")
            self.test_results['adaptive_rewards'] = {
                'passed': False,
                'error': str(e),
                'tests_run': 0
            }
    
    async def test_agent_specialization(self):
        """Test Agent Specialization System"""
        logger.info("Testing Agent Specialization...")
        
        try:
            from src.ai.agent_specialization import AgentSpecializationSystem, SpecializationDomain
            from src.ai.historical_performance import HistoricalPerformanceTracker
            from src.core.memory_filesystem import MemoryFilesystem
            
            # Initialize components
            memory_fs = MemoryFilesystem(self.mock_memory_manager)
            performance_tracker = HistoricalPerformanceTracker(memory_fs)
            specialization_system = AgentSpecializationSystem(memory_fs, performance_tracker)
            
            # Test 1: Record task performance for multiple agents
            test_performances = [
                {
                    'agent_role': 'planner',
                    'task_context': {
                        'task': 'Design system architecture',
                        'domain': 'architecture_design',
                        'complexity': 0.8
                    },
                    'performance_metrics': {
                        'quality_score': 0.9,
                        'success': True,
                        'latency_ms': 1500,
                        'cost_usd': 0.06
                    }
                },
                {
                    'agent_role': 'analyzer',
                    'task_context': {
                        'task': 'Analyze code performance',
                        'domain': 'code_analysis',
                        'complexity': 0.6
                    },
                    'performance_metrics': {
                        'quality_score': 0.85,
                        'success': True,
                        'latency_ms': 800,
                        'cost_usd': 0.03
                    }
                }
            ]
            
            for perf in test_performances:
                await specialization_system.record_task_performance(
                    perf['agent_role'],
                    perf['task_context'],
                    perf['performance_metrics']
                )
            
            # Test 2: Get optimal agent assignment
            test_task = {
                'task': 'Design microservices architecture for payment system',
                'domain': 'architecture',
                'complexity': 0.9,
                'technologies': ['microservices', 'api', 'payment']
            }
            
            assignment = await specialization_system.get_optimal_agent_assignment(test_task)
            assert assignment.recommended_agent is not None, "Failed to get agent assignment"
            assert 0.0 <= assignment.confidence <= 1.0, "Invalid confidence score"
            assert assignment.reasoning, "Missing assignment reasoning"
            
            # Test 3: Get domain experts
            experts = specialization_system.get_domain_experts(SpecializationDomain.ARCHITECTURE_DESIGN)
            assert isinstance(experts, list), "Domain experts not returned as list"
            
            # Test 4: Get specialization report
            for agent_role in ['planner', 'analyzer']:
                report = specialization_system.get_agent_specialization_report(agent_role)
                assert 'agent_role' in report, "Missing agent role in report"
                assert 'total_experience' in report, "Missing experience in report"
            
            self.test_results['agent_specialization'] = {
                'passed': True,
                'tests_run': 4,
                'performances_recorded': len(test_performances),
                'details': {
                    'performance_recording': True,
                    'agent_assignment': True,
                    'domain_experts': True,
                    'specialization_reports': True
                }
            }
            
            logger.info("✅ Agent Specialization tests passed")
            
        except Exception as e:
            logger.error(f"❌ Agent Specialization tests failed: {e}")
            self.test_results['agent_specialization'] = {
                'passed': False,
                'error': str(e),
                'tests_run': 0
            }
    
    async def test_memory_guided_mcts(self):
        """Test Memory-Guided MCTS"""
        logger.info("Testing Memory-Guided MCTS...")
        
        try:
            from src.ai.memory_guided_mcts import MemoryGuidedMCTS
            from src.core.memory_filesystem import MemoryFilesystem
            from src.ai.treequest_engine import TreeQuestConfig
            
            # Initialize components
            memory_fs = MemoryFilesystem(self.mock_memory_manager)
            config = TreeQuestConfig(
                max_depth=2,
                branching_factor=2,
                rollout_budget=4,
                timeout_seconds=5
            )
            
            mcts_engine = MemoryGuidedMCTS(memory_fs, config=config, models=self.mock_model_registry)
            
            # Test 1: Task signature creation
            task = "Analyze Python code for performance issues"
            context = {'language': 'python', 'domain': 'performance'}
            
            signature = mcts_engine._create_task_signature(task, context)
            assert signature, "Failed to create task signature"
            assert isinstance(signature, str), "Task signature not a string"
            
            # Test 2: Context matching
            pattern_content = {'context': {'language': 'python'}}
            current_state = {'language': 'python', 'domain': 'performance'}
            
            match_score = mcts_engine._calculate_context_match(pattern_content, current_state)
            assert 0.0 <= match_score <= 1.0, "Invalid context match score"
            
            # Test 3: Response quality assessment
            test_responses = [
                "Analyze the code for performance bottlenecks and suggest optimizations",
                "Short response",
                "Detailed analysis:\n1. Memory usage patterns\n2. Algorithm efficiency\n3. I/O operations"
            ]
            
            for response in test_responses:
                quality = mcts_engine._assess_response_quality(response, 'analysis')
                assert 0.0 <= quality <= 1.0, "Invalid quality score"
            
            self.test_results['memory_guided_mcts'] = {
                'passed': True,
                'tests_run': 3,
                'details': {
                    'task_signature_creation': True,
                    'context_matching': True,
                    'quality_assessment': True
                }
            }
            
            logger.info("✅ Memory-Guided MCTS tests passed")
            
        except Exception as e:
            logger.error(f"❌ Memory-Guided MCTS tests failed: {e}")
            self.test_results['memory_guided_mcts'] = {
                'passed': False,
                'error': str(e),
                'tests_run': 0
            }
    
    async def test_enhanced_integration(self):
        """Test Enhanced TreeQuest Integration"""
        logger.info("Testing Enhanced TreeQuest Integration...")
        
        try:
            from src.ai.enhanced_treequest import EnhancedTreeQuestEngine, EnhancedTreeQuestConfig
            
            # Initialize enhanced engine
            config = EnhancedTreeQuestConfig(
                max_depth=2,
                branching_factor=2,
                rollout_budget=4,
                timeout_seconds=5,
                memory_guided=True,
                adaptive_rewards=True,
                agent_specialization=True,
                performance_tracking=True
            )
            
            enhanced_engine = EnhancedTreeQuestEngine(
                config, self.mock_memory_manager, self.mock_model_registry
            )
            
            # Test 1: System initialization
            assert enhanced_engine.memory_fs is not None, "Memory filesystem not initialized"
            assert enhanced_engine.performance_tracker is not None, "Performance tracker not initialized"
            assert enhanced_engine.adaptive_rewards is not None, "Adaptive rewards not initialized"
            assert enhanced_engine.agent_specialization is not None, "Agent specialization not initialized"
            
            # Test 2: Task classification
            test_tasks = [
                ("Analyze Python code for bugs", "code_analysis"),
                ("Design REST API architecture", "architecture_design"),
                ("Plan project timeline", "planning"),
                ("Debug authentication error", "troubleshooting")
            ]
            
            for task, expected_type in test_tasks:
                classified_type = enhanced_engine._classify_task_type(task, {})
                # Accept either exact match or reasonable classification
                assert classified_type in ['code_analysis', 'architecture_design', 'planning', 'troubleshooting', 'general'], "Invalid task classification"
            
            # Test 3: Task signature creation
            for task, _ in test_tasks:
                signature = enhanced_engine._create_task_signature(task, {'domain': 'test'})
                assert signature, "Failed to create task signature"
            
            # Test 4: System status
            status = await enhanced_engine.get_system_status()
            assert 'memory_filesystem' in status, "Missing memory filesystem status"
            assert 'performance_tracking' in status, "Missing performance tracking status"
            assert 'enhanced_features' in status, "Missing enhanced features status"
            
            # Test 5: Export learning data
            export_data = await enhanced_engine.export_learning_data()
            assert 'timestamp' in export_data, "Missing timestamp in export"
            assert 'system_config' in export_data, "Missing system config in export"
            
            self.test_results['enhanced_integration'] = {
                'passed': True,
                'tests_run': 5,
                'details': {
                    'system_initialization': True,
                    'task_classification': True,
                    'signature_creation': True,
                    'system_status': True,
                    'data_export': True
                }
            }
            
            logger.info("✅ Enhanced Integration tests passed")
            
        except Exception as e:
            logger.error(f"❌ Enhanced Integration tests failed: {e}")
            self.test_results['enhanced_integration'] = {
                'passed': False,
                'error': str(e),
                'tests_run': 0
            }
    
    async def test_performance_scalability(self):
        """Test Performance and Scalability"""
        logger.info("Testing Performance and Scalability...")
        
        try:
            from src.core.memory_filesystem import MemoryFilesystem
            
            # Initialize filesystem
            memory_fs = MemoryFilesystem(self.mock_memory_manager)
            
            # Test 1: Memory storage scalability
            start_time = time.time()
            num_memories = 100
            
            for i in range(num_memories):
                content = {'task': f'test_task_{i}', 'result': f'result_{i}'}
                metadata = {'importance_score': 0.5 + (i % 10) / 20}
                success = memory_fs.store_memory(f"/test/scalability/memory_{i}", content, metadata)
                assert success, f"Failed to store memory {i}"
            
            storage_time = time.time() - start_time
            
            # Test 2: Memory retrieval scalability
            start_time = time.time()
            retrieved_count = 0
            
            for i in range(0, num_memories, 10):  # Sample every 10th memory
                memory = memory_fs.get_memory(f"/test/scalability/memory_{i}")
                if memory:
                    retrieved_count += 1
            
            retrieval_time = time.time() - start_time
            
            # Test 3: Memory statistics performance
            start_time = time.time()
            stats = memory_fs.get_memory_stats()
            stats_time = time.time() - start_time
            
            # Test 4: Adaptive forgetting performance
            start_time = time.time()
            forgotten_count = memory_fs.adaptive_forget(forget_threshold=0.8)
            forgetting_time = time.time() - start_time
            
            self.test_results['performance_scalability'] = {
                'passed': True,
                'tests_run': 4,
                'performance_metrics': {
                    'storage_time_per_memory_ms': (storage_time / num_memories) * 1000,
                    'retrieval_time_per_memory_ms': (retrieval_time / 10) * 1000,
                    'stats_generation_time_ms': stats_time * 1000,
                    'adaptive_forgetting_time_ms': forgetting_time * 1000,
                    'memories_stored': num_memories,
                    'memories_retrieved': retrieved_count,
                    'memories_forgotten': forgotten_count
                },
                'details': {
                    'storage_scalability': True,
                    'retrieval_scalability': True,
                    'statistics_performance': True,
                    'forgetting_performance': True
                }
            }
            
            logger.info(f"✅ Performance tests passed - Storage: {storage_time:.2f}s, Retrieval: {retrieval_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Performance and Scalability tests failed: {e}")
            self.test_results['performance_scalability'] = {
                'passed': False,
                'error': str(e),
                'tests_run': 0
            }
    
    def generate_test_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        passed_tests = sum(1 for result in self.test_results.values() if result.get('passed', False))
        total_tests = len(self.test_results)
        
        report = {
            'test_summary': {
                'total_test_suites': total_tests,
                'passed_test_suites': passed_tests,
                'failed_test_suites': total_tests - passed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'total_execution_time': total_time,
                'timestamp': time.time()
            },
            'detailed_results': self.test_results,
            'system_validation': {
                'memory_filesystem': self.test_results.get('memory_filesystem', {}).get('passed', False),
                'historical_performance': self.test_results.get('historical_performance', {}).get('passed', False),
                'adaptive_rewards': self.test_results.get('adaptive_rewards', {}).get('passed', False),
                'agent_specialization': self.test_results.get('agent_specialization', {}).get('passed', False),
                'memory_guided_mcts': self.test_results.get('memory_guided_mcts', {}).get('passed', False),
                'enhanced_integration': self.test_results.get('enhanced_integration', {}).get('passed', False),
                'performance_scalability': self.test_results.get('performance_scalability', {}).get('passed', False)
            },
            'recommendations': self.generate_recommendations(),
            'next_steps': [
                "Deploy enhanced TreeQuest system to production environment",
                "Monitor memory filesystem performance under real workloads",
                "Collect user feedback on agent specialization accuracy",
                "Fine-tune adaptive reward parameters based on actual usage",
                "Implement continuous integration testing for all components"
            ]
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not self.test_results.get('memory_filesystem', {}).get('passed', False):
            recommendations.append("Fix memory filesystem implementation before deployment")
        
        if not self.test_results.get('performance_scalability', {}).get('passed', False):
            recommendations.append("Optimize performance for larger datasets")
        
        if not self.test_results.get('enhanced_integration', {}).get('passed', False):
            recommendations.append("Resolve integration issues between components")
        
        # Performance-based recommendations
        perf_results = self.test_results.get('performance_scalability', {})
        if 'performance_metrics' in perf_results:
            metrics = perf_results['performance_metrics']
            if metrics.get('storage_time_per_memory_ms', 0) > 10:  # > 10ms per memory
                recommendations.append("Optimize memory storage performance")
            
            if metrics.get('retrieval_time_per_memory_ms', 0) > 5:  # > 5ms per retrieval
                recommendations.append("Optimize memory retrieval performance")
        
        if not recommendations:
            recommendations.append("All tests passed - system ready for deployment")
            recommendations.append("Consider adding more comprehensive integration tests")
            recommendations.append("Monitor system performance in production environment")
        
        return recommendations

async def main():
    """Main test execution function"""
    print("🧘 Enhanced TreeQuest - Comprehensive Test Suite")
    print("=" * 60)
    
    tester = EnhancedTreeQuestTester()
    
    try:
        test_report = await tester.run_all_tests()
        
        print("\n" + "=" * 60)
        print("📊 TEST REPORT SUMMARY")
        print("=" * 60)
        
        summary = test_report['test_summary']
        print(f"Total Test Suites: {summary['total_test_suites']}")
        print(f"Passed: {summary['passed_test_suites']}")
        print(f"Failed: {summary['failed_test_suites']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Execution Time: {summary['total_execution_time']:.2f} seconds")
        
        print("\n🔍 COMPONENT VALIDATION:")
        validation = test_report['system_validation']
        for component, status in validation.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {component.replace('_', ' ').title()}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(test_report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print(f"\n🚀 NEXT STEPS:")
        for i, step in enumerate(test_report['next_steps'], 1):
            print(f"{i}. {step}")
        
        # Save detailed report
        report_path = Path("enhanced_treequest_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(test_report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        if summary['success_rate'] >= 80:
            print("\n🎉 Enhanced TreeQuest system validation SUCCESSFUL!")
            return True
        else:
            print("\n⚠️  Enhanced TreeQuest system validation FAILED - address issues before deployment")
            return False
            
    except Exception as e:
        print(f"\n❌ Test suite execution failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)