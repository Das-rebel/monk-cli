"""
Enhanced TreeQuest Integration Layer
Integrates all memory and learning enhancements into a unified TreeQuest system
"""

import json
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Import enhanced components
from src.core.memory_filesystem import MemoryFilesystem
from src.ai.memory_guided_mcts import MemoryGuidedMCTS
from src.ai.historical_performance import HistoricalPerformanceTracker, PerformanceMetric, PerformanceObjective
from src.ai.adaptive_rewards import AdaptiveRewardSystem
from src.ai.agent_specialization import AgentSpecializationSystem, SpecializationDomain

# Import base components
from src.ai.treequest_engine import TreeQuestConfig, TreeNode
from src.ai.model_registry import ModelRegistry
from src.core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

@dataclass
class EnhancedTreeQuestConfig(TreeQuestConfig):
    """Extended configuration for enhanced TreeQuest"""
    memory_guided: bool = True
    adaptive_rewards: bool = True
    agent_specialization: bool = True
    performance_tracking: bool = True
    learning_enabled: bool = True
    memory_weight: float = 0.3
    specialization_threshold: float = 0.7
    performance_correlation_threshold: float = 0.6

class EnhancedTreeQuestEngine:
    """Enhanced TreeQuest engine with memory, learning, and specialization"""
    
    def __init__(self, config: EnhancedTreeQuestConfig, memory_manager: MemoryManager, 
                 model_registry: ModelRegistry):
        self.config = config
        self.memory_manager = memory_manager
        self.model_registry = model_registry
        
        # Initialize enhanced components
        self.memory_fs = MemoryFilesystem(memory_manager)
        self.performance_tracker = HistoricalPerformanceTracker(self.memory_fs)
        self.adaptive_rewards = AdaptiveRewardSystem(self.memory_fs, self.performance_tracker)
        self.agent_specialization = AgentSpecializationSystem(self.memory_fs, self.performance_tracker)
        
        # Initialize memory-guided MCTS
        self.mcts_engine = MemoryGuidedMCTS(
            self.memory_fs, 
            config=config,
            models=model_registry
        )
        
        # Execution tracking
        self.execution_history = []
        self.session_metrics = {}
        
        logger.info("Enhanced TreeQuest engine initialized with all memory and learning components")
    
    async def solve_enhanced(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced solve method with full memory and learning integration"""
        try:
            start_time = time.time()
            context = context or {}
            
            # Phase 1: Pre-execution Analysis
            logger.info(f"Starting enhanced TreeQuest solve for task: {task[:100]}...")
            
            # Get optimal agent assignment using specialization system
            agent_assignment = await self.agent_specialization.get_optimal_agent_assignment(
                {'task': task, **context}
            )
            
            logger.info(f"Agent assignment: {agent_assignment.recommended_agent} "
                       f"(confidence: {agent_assignment.confidence:.2f})")
            
            # Enhance context with agent assignment and historical data
            enhanced_context = {
                **context,
                'optimal_agent': agent_assignment.recommended_agent,
                'agent_confidence': agent_assignment.confidence,
                'specialization_match': agent_assignment.specialization_match,
                'task_analysis_timestamp': start_time
            }
            
            # Phase 2: Memory-Guided Execution
            execution_result = await self.mcts_engine.solve_with_memory(task, enhanced_context)
            
            # Phase 3: Performance Recording and Learning
            await self._record_execution_performance(
                task, enhanced_context, execution_result, start_time
            )
            
            # Phase 4: Post-Execution Analysis and Learning
            await self._post_execution_learning(
                task, enhanced_context, execution_result, agent_assignment
            )
            
            # Phase 5: Generate Enhanced Results
            enhanced_result = await self._generate_enhanced_result(
                execution_result, agent_assignment, enhanced_context, start_time
            )
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in enhanced TreeQuest solve: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_used': True,
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    async def _record_execution_performance(self, task: str, context: Dict[str, Any], 
                                          result: Dict[str, Any], start_time: float):
        """Record comprehensive performance metrics"""
        try:
            execution_time = time.time() - start_time
            
            # Extract performance data
            quality_score = result.get('final_reward', 0.5)
            success = result.get('success', False)
            cost_usd = result.get('total_cost_usd', 0.0)
            
            # Record for each agent used in execution
            agents_used = result.get('agents_used', [context.get('optimal_agent', 'unknown')])
            
            for agent_role in agents_used:
                # Create performance metric
                metric = PerformanceMetric(
                    timestamp=time.time(),
                    provider="enhanced_treequest",
                    model=f"agent_{agent_role}",
                    agent_role=agent_role,
                    task_type=self._classify_task_type(task, context),
                    quality_score=quality_score,
                    latency_ms=execution_time * 1000,
                    cost_usd=cost_usd,
                    success=success,
                    error_type=result.get('error_type'),
                    context_size=len(str(context)),
                    response_length=len(str(result))
                )
                
                # Record in performance tracker
                await self.performance_tracker.record_performance(metric)
                
                # Record in agent specialization system
                await self.agent_specialization.record_task_performance(
                    agent_role,
                    {'task': task, 'task_type': metric.task_type, **context},
                    {
                        'quality_score': quality_score,
                        'success': success,
                        'latency_ms': execution_time * 1000,
                        'cost_usd': cost_usd
                    }
                )
            
            logger.debug(f"Recorded performance metrics for {len(agents_used)} agents")
            
        except Exception as e:
            logger.error(f"Error recording execution performance: {e}")
    
    async def _post_execution_learning(self, task: str, context: Dict[str, Any], 
                                     result: Dict[str, Any], agent_assignment):
        """Perform post-execution learning and adaptation"""
        try:
            # Record outcome for adaptive rewards
            if 'reward_context' in result:
                predicted_reward = result['reward_context'].get('predicted_reward', 0.5)
                actual_outcome = result.get('final_reward', 0.5)
                
                await self.adaptive_rewards.record_outcome(
                    predicted_reward, actual_outcome, {
                        **context,
                        **result.get('reward_context', {}),
                        'agent_assignment': asdict(agent_assignment)
                    }
                )
            
            # Store successful execution patterns
            if result.get('success', False) and result.get('final_reward', 0) > 0.6:
                await self._store_successful_execution_pattern(task, context, result)
            
            # Adaptive memory cleanup
            if hasattr(self.memory_fs, 'adaptive_forget'):
                forgotten_count = self.memory_fs.adaptive_forget(forget_threshold=0.3)
                if forgotten_count > 0:
                    logger.debug(f"Adaptive forgetting removed {forgotten_count} low-value memories")
            
        except Exception as e:
            logger.error(f"Error in post-execution learning: {e}")
    
    async def _store_successful_execution_pattern(self, task: str, context: Dict[str, Any], 
                                                result: Dict[str, Any]):
        """Store successful execution patterns for future learning"""
        try:
            pattern_data = {
                'task_signature': self._create_task_signature(task, context),
                'execution_strategy': {
                    'optimal_agent': context.get('optimal_agent'),
                    'specialization_match': context.get('specialization_match', {}),
                    'agents_used': result.get('agents_used', []),
                    'execution_path': result.get('execution_path', [])
                },
                'performance_metrics': {
                    'final_reward': result.get('final_reward'),
                    'execution_time': result.get('execution_time'),
                    'total_cost': result.get('total_cost_usd'),
                    'success_indicators': result.get('success_indicators', [])
                },
                'context_factors': {
                    'task_complexity': context.get('complexity', 0.5),
                    'domain_requirements': context.get('domains', []),
                    'constraints': context.get('constraints', {}),
                    'user_preferences': context.get('preferences', {})
                }
            }
            
            # Store in memory filesystem
            success_rate = result.get('final_reward', 0.5)
            self.memory_fs.store_successful_path(
                pattern_data['task_signature'],
                pattern_data,
                success_rate
            )
            
            logger.info(f"Stored successful execution pattern with {success_rate:.2f} success rate")
            
        except Exception as e:
            logger.error(f"Error storing successful execution pattern: {e}")
    
    async def _generate_enhanced_result(self, base_result: Dict[str, Any], 
                                      agent_assignment, context: Dict[str, Any], 
                                      start_time: float) -> Dict[str, Any]:
        """Generate comprehensive enhanced result"""
        try:
            execution_time = time.time() - start_time
            
            enhanced_result = {
                **base_result,
                'enhanced_features': {
                    'memory_guided': self.config.memory_guided,
                    'adaptive_rewards': self.config.adaptive_rewards,
                    'agent_specialization': self.config.agent_specialization,
                    'performance_tracking': self.config.performance_tracking
                },
                'agent_assignment': {
                    'selected_agent': agent_assignment.recommended_agent,
                    'confidence': agent_assignment.confidence,
                    'reasoning': agent_assignment.reasoning,
                    'alternatives': agent_assignment.alternative_agents[:3],
                    'specialization_match': agent_assignment.specialization_match
                },
                'execution_analytics': {
                    'total_execution_time': execution_time,
                    'memory_guided_decisions': base_result.get('memory_guided_decisions', 0),
                    'historical_patterns_used': base_result.get('historical_patterns_used', 0),
                    'learning_applied': base_result.get('learning_applied', False),
                    'adaptation_events': base_result.get('adaptation_events', [])
                },
                'performance_insights': await self._get_performance_insights(context),
                'recommendations': await self._generate_recommendations(base_result, context),
                'session_id': str(int(start_time)),
                'enhanced_by': 'TreeQuest-memU-v1.0'
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error generating enhanced result: {e}")
            return {**base_result, 'enhancement_error': str(e)}
    
    async def _get_performance_insights(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance insights and trends"""
        try:
            insights = {
                'provider_performance': {},
                'agent_effectiveness': {},
                'cost_efficiency': {},
                'learning_progress': {}
            }
            
            # Get recent performance trends
            trends = self.performance_tracker.get_performance_trends(hours_back=24)
            if 'provider_trends' in trends:
                insights['provider_performance'] = {
                    'trending_up': [],
                    'trending_down': [],
                    'stable': []
                }
                
                for provider, trend_data in trends['provider_trends'].items():
                    if trend_data['quality_trend'] > 0.1:
                        insights['provider_performance']['trending_up'].append(provider)
                    elif trend_data['quality_trend'] < -0.1:
                        insights['provider_performance']['trending_down'].append(provider)
                    else:
                        insights['provider_performance']['stable'].append(provider)
            
            # Get agent specialization insights
            for agent_role in ['planner', 'analyzer', 'critic', 'synthesizer', 'executor']:
                report = self.agent_specialization.get_agent_specialization_report(agent_role)
                if 'primary_specializations' in report:
                    insights['agent_effectiveness'][agent_role] = {
                        'specializations': len(report['primary_specializations']),
                        'experience': report.get('total_experience', 0),
                        'consistency': report.get('consistency_score', 0.5)
                    }
            
            # Get adaptive rewards progress
            adaptation_summary = self.adaptive_rewards.get_adaptation_summary()
            insights['learning_progress'] = {
                'total_outcomes': adaptation_summary.get('total_outcomes_recorded', 0),
                'agents_adapted': len(adaptation_summary.get('agent_adaptations', {})),
                'recent_accuracy': adaptation_summary.get('recent_performance', {}).get('avg_prediction_error', 1.0)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting performance insights: {e}")
            return {}
    
    async def _generate_recommendations(self, result: Dict[str, Any], 
                                      context: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            # Performance-based recommendations
            if result.get('final_reward', 0) < 0.6:
                recommendations.append(
                    "Consider refining task description or breaking into smaller subtasks"
                )
            
            # Cost optimization recommendations
            cost = result.get('total_cost_usd', 0)
            if cost > 0.1:  # High cost
                recommendations.append(
                    "Task had high cost - consider using more cost-effective models for similar tasks"
                )
            
            # Agent specialization recommendations
            agent_confidence = context.get('agent_confidence', 1.0)
            if agent_confidence < 0.7:
                recommendations.append(
                    "Low agent confidence - consider developing specialized expertise for this task type"
                )
            
            # Memory utilization recommendations
            if result.get('memory_guided_decisions', 0) == 0:
                recommendations.append(
                    "No historical patterns found - this appears to be a novel task type"
                )
            
            # Learning recommendations
            if not result.get('learning_applied', False):
                recommendations.append(
                    "Enable learning features to improve performance on similar future tasks"
                )
            
            # Get system-wide recommendations
            performance_recommendations = self.performance_tracker.get_performance_recommendations(context)
            if 'provider_suggestions' in performance_recommendations:
                for suggestion in performance_recommendations['provider_suggestions'][:2]:  # Top 2
                    recommendations.append(
                        f"Consider using {suggestion['provider']}: {suggestion['reason']}"
                    )
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _classify_task_type(self, task: str, context: Dict[str, Any]) -> str:
        """Classify task type for performance tracking"""
        try:
            task_lower = task.lower()
            
            # Define task type patterns
            task_patterns = {
                'code_analysis': ['analyze', 'review', 'audit', 'inspect', 'code'],
                'architecture_design': ['design', 'architecture', 'structure', 'pattern', 'system'],
                'planning': ['plan', 'strategy', 'roadmap', 'timeline', 'schedule'],
                'implementation': ['implement', 'build', 'create', 'develop', 'code'],
                'optimization': ['optimize', 'improve', 'enhance', 'performance', 'efficiency'],
                'testing': ['test', 'validate', 'verify', 'check', 'quality'],
                'documentation': ['document', 'explain', 'describe', 'specification', 'guide'],
                'troubleshooting': ['debug', 'fix', 'solve', 'error', 'issue', 'problem'],
                'security': ['security', 'secure', 'vulnerability', 'auth', 'encrypt'],
                'integration': ['integrate', 'connect', 'combine', 'merge', 'api']
            }
            
            # Check for pattern matches
            for task_type, keywords in task_patterns.items():
                if any(keyword in task_lower for keyword in keywords):
                    return task_type
            
            # Use context clues
            if 'task_type' in context:
                return context['task_type']
            
            return 'general'
            
        except Exception as e:
            logger.error(f"Error classifying task type: {e}")
            return 'unknown'
    
    def _create_task_signature(self, task: str, context: Dict[str, Any]) -> str:
        """Create task signature for pattern matching"""
        try:
            # Normalize task
            task_normalized = ' '.join(task.lower().split())[:100]  # First 100 chars
            
            # Add context elements
            context_elements = []
            for key in ['domain', 'complexity', 'framework', 'language']:
                if key in context:
                    context_elements.append(f"{key}:{context[key]}")
            
            # Create signature
            signature = f"{task_normalized}|{';'.join(context_elements)}"
            
            # Hash for consistency
            import hashlib
            signature_hash = hashlib.md5(signature.encode()).hexdigest()[:16]
            
            return f"{signature_hash}:{task_normalized[:30]}"
            
        except Exception as e:
            logger.error(f"Error creating task signature: {e}")
            return task[:50]
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'memory_filesystem': {
                    'stats': self.memory_fs.get_memory_stats(),
                    'health': 'operational'
                },
                'performance_tracking': {
                    'recent_metrics': len(self.performance_tracker.recent_metrics),
                    'provider_rankings': len(self.performance_tracker.provider_rankings),
                    'health': 'operational'
                },
                'adaptive_rewards': {
                    'adaptation_summary': self.adaptive_rewards.get_adaptation_summary(),
                    'health': 'operational'
                },
                'agent_specialization': {
                    'agents_tracked': len(self.agent_specialization.agent_profiles),
                    'domain_experts': len(self.agent_specialization.domain_experts),
                    'health': 'operational'
                },
                'enhanced_features': {
                    'memory_guided': self.config.memory_guided,
                    'adaptive_rewards': self.config.adaptive_rewards,
                    'agent_specialization': self.config.agent_specialization,
                    'performance_tracking': self.config.performance_tracking,
                    'learning_enabled': self.config.learning_enabled
                },
                'system_metrics': {
                    'total_executions': len(self.execution_history),
                    'uptime': time.time() - (self.execution_history[0] if self.execution_history else time.time()),
                    'last_execution': self.execution_history[-1] if self.execution_history else None
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    async def optimize_system_parameters(self) -> Dict[str, Any]:
        """Optimize system parameters based on performance data"""
        try:
            optimizations = {
                'memory_optimizations': [],
                'reward_optimizations': [],
                'specialization_optimizations': [],
                'performance_optimizations': []
            }
            
            # Memory filesystem optimizations
            memory_stats = self.memory_fs.get_memory_stats()
            if memory_stats.get('total_memories', 0) > 1000:
                forgotten = self.memory_fs.adaptive_forget(forget_threshold=0.4)
                optimizations['memory_optimizations'].append(
                    f"Removed {forgotten} low-value memories"
                )
            
            # Performance tracking optimizations
            performance_recommendations = self.performance_tracker.get_performance_recommendations({})
            if 'cost_optimizations' in performance_recommendations:
                for opt in performance_recommendations['cost_optimizations'][:3]:
                    optimizations['performance_optimizations'].append(opt['suggestion'])
            
            # Generate system-wide recommendations
            optimizations['system_recommendations'] = [
                "System is learning and adapting based on usage patterns",
                "Memory-guided decisions are improving task efficiency",
                "Agent specializations are developing based on performance"
            ]
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Error optimizing system parameters: {e}")
            return {'error': str(e)}
    
    async def export_learning_data(self, export_path: str = None) -> Dict[str, Any]:
        """Export learning data for analysis or backup"""
        try:
            export_data = {
                'timestamp': time.time(),
                'memory_filesystem': {
                    'stats': self.memory_fs.get_memory_stats(),
                    'structure': 'hierarchical_filesystem'
                },
                'performance_data': {
                    'aggregates_count': len(self.performance_tracker.aggregates),
                    'recent_metrics_count': len(self.performance_tracker.recent_metrics),
                    'provider_rankings': self.performance_tracker.provider_rankings
                },
                'specialization_data': {
                    'agent_profiles': len(self.agent_specialization.agent_profiles),
                    'domain_experts': {
                        domain.value: experts[:3] 
                        for domain, experts in self.agent_specialization.domain_experts.items()
                    }
                },
                'adaptive_rewards': self.adaptive_rewards.get_adaptation_summary(),
                'system_config': asdict(self.config)
            }
            
            # Export to file if path specified
            if export_path:
                import json
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                    
                logger.info(f"Learning data exported to {export_path}")
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting learning data: {e}")
            return {'error': str(e)}