"""
Memory-Guided Monte Carlo Tree Search
Enhances TreeQuest MCTS with historical path learning and memory-driven exploration
"""

import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import logging
import numpy as np
from dataclasses import dataclass

from src.ai.treequest_engine import TreeQuestEngine, TreeNode, TreeQuestConfig
from src.core.memory_filesystem import MemoryFilesystem, MemoryNodeType

logger = logging.getLogger(__name__)

@dataclass
class PathPattern:
    """Represents a successful path pattern in MCTS"""
    task_signature: str
    agent_sequence: List[str]
    node_depths: List[int]
    rewards: List[float]
    success_rate: float
    usage_count: int
    created_at: float

class MemoryGuidedMCTS(TreeQuestEngine):
    """Enhanced TreeQuest engine with memory-guided exploration"""
    
    def __init__(self, memory_filesystem: MemoryFilesystem, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_fs = memory_filesystem
        self.path_patterns = {}  # task_signature -> List[PathPattern]
        self.current_execution_path = []  # Track current path for learning
        self.exploration_bonus = 0.2  # Bonus for exploring new paths
        self.memory_weight = 0.3  # Weight for memory-guided decisions
        
        # Load existing path patterns
        asyncio.create_task(self._load_path_patterns())
    
    async def solve_with_memory(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced solve method with memory-guided exploration"""
        try:
            # Create task signature for memory lookup
            task_signature = self._create_task_signature(task, context)
            logger.info(f"Solving task with memory guidance: {task_signature}")
            
            # Get historical successful paths
            historical_paths = self.memory_fs.get_successful_paths(
                task_signature, min_success_rate=0.6
            )
            
            # Initialize memory-guided context
            memory_context = {
                **context,
                'task_signature': task_signature,
                'historical_paths': historical_paths,
                'memory_guided': True
            }
            
            # Reset execution tracking
            self.current_execution_path = []
            
            # Run memory-guided MCTS
            result = await super().solve(task, memory_context)
            
            # Store successful execution path
            if result.get('success', False):
                await self._store_successful_execution(
                    task_signature, 
                    self.current_execution_path, 
                    result.get('final_reward', 0.0)
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in memory-guided solve: {e}")
            return await super().solve(task, context)  # Fallback to standard solve
    
    async def _expand_node_with_memory(self, node: TreeNode, initial_state: Dict[str, Any]) -> TreeNode:
        """Expand node using memory-guided strategy"""
        try:
            task_signature = initial_state.get('task_signature', '')
            historical_paths = initial_state.get('historical_paths', [])
            
            # Check if we have successful historical patterns for this depth
            memory_guided_child = await self._create_memory_guided_child(
                node, initial_state, historical_paths
            )
            
            if memory_guided_child:
                logger.debug(f"Created memory-guided child at depth {node.depth}")
                return memory_guided_child
            
            # Fallback to standard expansion with exploration bonus
            child = await self._standard_expand_node(node, initial_state)
            if child:
                # Add exploration bonus for new paths
                child.total_reward += self.exploration_bonus
                logger.debug(f"Added exploration bonus to new path at depth {node.depth}")
            
            return child
            
        except Exception as e:
            logger.error(f"Error in memory-guided expansion: {e}")
            return await self._standard_expand_node(node, initial_state)
    
    async def _create_memory_guided_child(self, parent: TreeNode, initial_state: Dict[str, Any], 
                                        historical_paths: List[Dict[str, Any]]) -> Optional[TreeNode]:
        """Create child node based on historical successful patterns"""
        if not historical_paths:
            return None
        
        try:
            # Find best historical path for current depth
            current_depth = parent.depth
            best_pattern = None
            best_score = 0.0
            
            for path_data in historical_paths:
                path_content = path_data.get('content', {})
                path_sequence = path_content.get('path_data', {}).get('agent_sequence', [])
                
                if len(path_sequence) > current_depth:
                    # Calculate pattern score based on success rate and usage
                    pattern_score = (
                        0.6 * path_data.get('success_rate', 0.5) +
                        0.2 * min(1.0, path_data.get('usage_count', 0) / 10.0) +
                        0.2 * self._calculate_context_match(path_content, initial_state)
                    )
                    
                    if pattern_score > best_score:
                        best_score = pattern_score
                        best_pattern = path_content
            
            if best_pattern and best_score > 0.6:  # Threshold for using memory pattern
                return await self._create_child_from_pattern(parent, initial_state, best_pattern)
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating memory-guided child: {e}")
            return None
    
    async def _create_child_from_pattern(self, parent: TreeNode, initial_state: Dict[str, Any], 
                                       pattern: Dict[str, Any]) -> TreeNode:
        """Create child node from successful pattern"""
        try:
            path_data = pattern.get('path_data', {})
            agent_sequence = path_data.get('agent_sequence', [])
            current_depth = parent.depth
            
            if len(agent_sequence) > current_depth:
                # Get suggested agent for this depth
                suggested_agent = agent_sequence[current_depth]
                
                # Create child state with memory guidance
                child_state = {
                    **initial_state,
                    'suggested_agent': suggested_agent,
                    'memory_guided': True,
                    'pattern_confidence': pattern.get('success_rate', 0.5)
                }
                
                # Create child node
                child_hash = self._generate_state_hash(child_state)
                child = TreeNode(
                    state=child_state,
                    parent=parent,
                    depth=parent.depth + 1,
                    hash=child_hash
                )
                
                # Set suggested agent
                child.agent_role = suggested_agent
                
                # Add memory-guided reward bonus
                memory_bonus = 0.1 * pattern.get('success_rate', 0.5)
                child.total_reward = memory_bonus
                child.visits = 0
                
                parent.children.append(child)
                return child
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating child from pattern: {e}")
            return None
    
    async def _standard_expand_node(self, node: TreeNode, initial_state: Dict[str, Any]) -> TreeNode:
        """Standard node expansion with memory tracking"""
        try:
            # Create child state
            child_state = self._create_child_state(node, initial_state)
            
            # Generate child hash
            child_hash = self._generate_state_hash(child_state)
            
            # Create child node
            child = TreeNode(
                state=child_state,
                parent=node,
                depth=node.depth + 1,
                hash=child_hash
            )
            
            # Add to parent's children
            node.children.append(child)
            
            return child
            
        except Exception as e:
            logger.error(f"Error in standard expansion: {e}")
            return None
    
    def _create_child_state(self, parent: TreeNode, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create child state from parent state"""
        return {
            **initial_state,
            'parent_hash': parent.hash,
            'depth': parent.depth + 1,
            'exploration_path': initial_state.get('exploration_path', []) + [parent.hash]
        }
    
    async def _simulate_with_memory_tracking(self, node: TreeNode, initial_state: Dict[str, Any]) -> float:
        """Simulate with memory tracking for learning"""
        try:
            # Track agent selection in execution path
            agent_role = node.agent_role or self._select_agent_for_simulation(node, initial_state)
            
            execution_step = {
                'depth': node.depth,
                'agent': agent_role,
                'state_hash': node.hash,
                'timestamp': time.time()
            }
            self.current_execution_path.append(execution_step)
            
            # Get memory-guided context if available
            suggested_agent = initial_state.get('suggested_agent')
            if suggested_agent and suggested_agent == agent_role:
                # Bonus for following memory guidance
                memory_guidance_bonus = 0.05
                logger.debug(f"Following memory guidance for agent {agent_role}")
            else:
                memory_guidance_bonus = 0.0
            
            # Execute simulation
            reward = await self._execute_simulation(agent_role, node, initial_state)
            
            # Apply memory guidance bonus
            final_reward = reward + memory_guidance_bonus
            
            return final_reward
            
        except Exception as e:
            logger.error(f"Error in memory-tracked simulation: {e}")
            return 0.0
    
    async def _execute_simulation(self, agent_role: str, node: TreeNode, 
                                initial_state: Dict[str, Any]) -> float:
        """Execute agent simulation with enhanced context"""
        try:
            # Build enhanced prompt with historical context
            base_prompt = self._build_agent_prompt(agent_role, initial_state)
            
            # Add memory context if available
            historical_insights = await self._get_agent_historical_insights(agent_role, initial_state)
            if historical_insights:
                enhanced_prompt = f"{base_prompt}\n\nHistorical Insights:\n{historical_insights}"
            else:
                enhanced_prompt = base_prompt
            
            # Select model for agent
            model_name = self.models.pick(self._get_model_role(agent_role), self.config.objective)
            
            # Execute with appropriate handler
            if agent_role == "planner":
                return await self._handle_planning_task_with_memory(enhanced_prompt, model_name, node)
            elif agent_role == "analyzer":
                return await self._handle_analyzer_task_with_memory(enhanced_prompt, model_name, node)
            elif agent_role == "critic":
                return await self._handle_critic_task_with_memory(enhanced_prompt, model_name, node)
            elif agent_role == "synthesizer":
                return await self._handle_synthesizer_task_with_memory(enhanced_prompt, model_name, node)
            else:  # executor
                return await self._handle_executor_task_with_memory(enhanced_prompt, model_name, node)
                
        except Exception as e:
            logger.error(f"Error executing simulation for {agent_role}: {e}")
            return 0.0
    
    async def _get_agent_historical_insights(self, agent_role: str, 
                                           initial_state: Dict[str, Any]) -> Optional[str]:
        """Get historical insights for specific agent"""
        try:
            agent_insights_path = f"/agents/{agent_role}"
            agent_node = self.memory_fs.get_memory(agent_insights_path)
            
            if agent_node and agent_node.children:
                # Get top 3 most successful insights
                insights = []
                for child_name, child_node in agent_node.children.items():
                    if child_node.metadata.success_rate > 0.6:
                        insights.append({
                            'content': child_node.content,
                            'success_rate': child_node.metadata.success_rate,
                            'access_count': child_node.metadata.access_count
                        })
                
                # Sort by success rate and usage
                insights.sort(key=lambda x: (x['success_rate'], x['access_count']), reverse=True)
                
                if insights:
                    formatted_insights = []
                    for insight in insights[:3]:
                        formatted_insights.append(
                            f"- {insight['content'].get('insight', '')} "
                            f"(Success: {insight['success_rate']:.2f})"
                        )
                    return "\n".join(formatted_insights)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting historical insights for {agent_role}: {e}")
            return None
    
    async def _store_successful_execution(self, task_signature: str, 
                                        execution_path: List[Dict[str, Any]], 
                                        final_reward: float):
        """Store successful execution path for future learning"""
        try:
            if final_reward < 0.6:  # Only store reasonably successful paths
                return
            
            # Extract agent sequence
            agent_sequence = [step['agent'] for step in execution_path]
            depths = [step['depth'] for step in execution_path]
            
            # Create path pattern
            pattern = PathPattern(
                task_signature=task_signature,
                agent_sequence=agent_sequence,
                node_depths=depths,
                rewards=[final_reward],  # Can be extended to track per-step rewards
                success_rate=final_reward,
                usage_count=1,
                created_at=time.time()
            )
            
            # Store in memory filesystem
            path_data = {
                'agent_sequence': agent_sequence,
                'depths': depths,
                'final_reward': final_reward,
                'execution_steps': execution_path
            }
            
            success = self.memory_fs.store_successful_path(
                task_signature, path_data, final_reward
            )
            
            if success:
                logger.info(f"Stored successful path pattern for task: {task_signature}")
                
                # Also store agent-specific insights
                await self._store_agent_insights(execution_path, final_reward)
            
        except Exception as e:
            logger.error(f"Error storing successful execution: {e}")
    
    async def _store_agent_insights(self, execution_path: List[Dict[str, Any]], 
                                  final_reward: float):
        """Store agent-specific insights from successful execution"""
        try:
            for step in execution_path:
                agent = step['agent']
                depth = step['depth']
                
                # Create insight content
                insight_content = {
                    'insight': f"Successful {agent} execution at depth {depth}",
                    'depth': depth,
                    'reward_contribution': final_reward,
                    'execution_context': {
                        'timestamp': step['timestamp'],
                        'state_hash': step['state_hash']
                    }
                }
                
                # Store insight
                insight_path = f"/agents/{agent}/insight_{int(time.time())}"
                metadata = {
                    'importance_score': min(final_reward + 0.1, 1.0),
                    'success_rate': final_reward,
                    'tags': [agent, 'insight', f'depth_{depth}'],
                    'agent_annotations': {'agent': agent, 'depth': depth},
                    'performance_metrics': {'reward': final_reward}
                }
                
                self.memory_fs.store_memory(insight_path, insight_content, metadata)
            
        except Exception as e:
            logger.error(f"Error storing agent insights: {e}")
    
    def _create_task_signature(self, task: str, context: Dict[str, Any]) -> str:
        """Create standardized task signature for memory lookup"""
        try:
            # Normalize task text
            task_normalized = ' '.join(task.lower().split())
            
            # Extract key context elements
            context_keys = []
            for key in ['project_type', 'language', 'framework', 'complexity']:
                if key in context:
                    context_keys.append(f"{key}:{context[key]}")
            
            # Create signature
            signature_parts = [task_normalized] + context_keys
            signature = ' | '.join(signature_parts)
            
            # Create hash for consistent lookup
            signature_hash = hashlib.md5(signature.encode()).hexdigest()[:16]
            
            return f"{signature_hash}:{task_normalized[:50]}"
            
        except Exception as e:
            logger.error(f"Error creating task signature: {e}")
            return task[:50]  # Fallback
    
    def _calculate_context_match(self, pattern_content: Dict[str, Any], 
                               current_state: Dict[str, Any]) -> float:
        """Calculate how well a pattern matches current context"""
        try:
            pattern_context = pattern_content.get('context', {})
            
            if not pattern_context:
                return 0.5  # Neutral match if no context
            
            matches = 0
            total_keys = 0
            
            for key in ['project_type', 'language', 'framework', 'complexity']:
                if key in pattern_context:
                    total_keys += 1
                    if key in current_state and current_state[key] == pattern_context[key]:
                        matches += 1
            
            if total_keys == 0:
                return 0.5
                
            return matches / total_keys
            
        except Exception as e:
            logger.error(f"Error calculating context match: {e}")
            return 0.0
    
    def _build_agent_prompt(self, agent_role: str, initial_state: Dict[str, Any]) -> str:
        """Build agent-specific prompt"""
        task = initial_state.get('task', '')
        context = initial_state.get('context', {})
        
        base_prompts = {
            'planner': f"As a strategic planner, create a comprehensive plan for: {task}",
            'analyzer': f"As a data analyzer, analyze the following task in detail: {task}",
            'critic': f"As a quality critic, evaluate and provide feedback on: {task}",
            'synthesizer': f"As a synthesizer, integrate insights and create unified recommendations for: {task}",
            'executor': f"As an execution specialist, create actionable implementation steps for: {task}"
        }
        
        prompt = base_prompts.get(agent_role, f"Process the following task: {task}")
        
        if context:
            prompt += f"\n\nContext: {json.dumps(context, indent=2)}"
        
        return prompt
    
    async def _load_path_patterns(self):
        """Load existing path patterns from memory"""
        try:
            patterns_node = self.memory_fs.get_memory("/patterns/successful_paths")
            if patterns_node and patterns_node.children:
                logger.info(f"Loaded {len(patterns_node.children)} path patterns from memory")
            
        except Exception as e:
            logger.error(f"Error loading path patterns: {e}")
    
    # Enhanced agent handlers with memory integration
    async def _handle_planning_task_with_memory(self, prompt: str, model_name: str, node: TreeNode) -> float:
        """Enhanced planning with memory integration"""
        try:
            # Add strategic depth bonus for planner
            base_reward = 0.8
            if node.depth == 0:
                base_reward += 0.1  # Root planning bonus
            
            # Execute planning
            response = await self._execute_llm_request(prompt, model_name)
            
            # Calculate quality-based reward
            quality_score = self._assess_response_quality(response, 'planning')
            final_reward = base_reward * quality_score
            
            return final_reward
            
        except Exception as e:
            logger.error(f"Error in planning task: {e}")
            return 0.3
    
    async def _handle_analyzer_task_with_memory(self, prompt: str, model_name: str, node: TreeNode) -> float:
        """Enhanced analysis with memory integration"""
        try:
            base_reward = 0.7
            depth_bonus = min(0.2, node.depth * 0.05)  # Bonus for deeper analysis
            
            response = await self._execute_llm_request(prompt, model_name)
            quality_score = self._assess_response_quality(response, 'analysis')
            
            return (base_reward + depth_bonus) * quality_score
            
        except Exception as e:
            logger.error(f"Error in analyzer task: {e}")
            return 0.4
    
    async def _handle_critic_task_with_memory(self, prompt: str, model_name: str, node: TreeNode) -> float:
        """Enhanced criticism with memory integration"""
        try:
            base_reward = 0.6
            quality_bonus = 0.15  # Bonus for thorough critique
            
            response = await self._execute_llm_request(prompt, model_name)
            quality_score = self._assess_response_quality(response, 'critique')
            
            return (base_reward + quality_bonus) * quality_score
            
        except Exception as e:
            logger.error(f"Error in critic task: {e}")
            return 0.35
    
    async def _handle_synthesizer_task_with_memory(self, prompt: str, model_name: str, node: TreeNode) -> float:
        """Enhanced synthesis with memory integration"""
        try:
            base_reward = 0.75
            integration_bonus = 0.2 if node.depth >= 2 else 0.1  # Bonus for deep synthesis
            
            response = await self._execute_llm_request(prompt, model_name)
            quality_score = self._assess_response_quality(response, 'synthesis')
            
            return (base_reward + integration_bonus) * quality_score
            
        except Exception as e:
            logger.error(f"Error in synthesizer task: {e}")
            return 0.4
    
    async def _handle_executor_task_with_memory(self, prompt: str, model_name: str, node: TreeNode) -> float:
        """Enhanced execution planning with memory integration"""
        try:
            base_reward = 0.65
            feasibility_bonus = 0.15  # Bonus for practical execution plans
            
            response = await self._execute_llm_request(prompt, model_name)
            quality_score = self._assess_response_quality(response, 'execution')
            
            return (base_reward + feasibility_bonus) * quality_score
            
        except Exception as e:
            logger.error(f"Error in executor task: {e}")
            return 0.3
    
    def _assess_response_quality(self, response: str, task_type: str) -> float:
        """Assess response quality based on task type"""
        if not response or len(response.strip()) < 50:
            return 0.3  # Low quality for very short responses
        
        # Basic quality indicators
        quality_score = 0.5  # Base score
        
        # Length bonus (up to 0.2)
        length_bonus = min(0.2, len(response) / 1000)
        quality_score += length_bonus
        
        # Structure bonus (up to 0.15)
        if '\n' in response and ('•' in response or '-' in response or '1.' in response):
            quality_score += 0.15
        
        # Task-specific bonuses
        task_indicators = {
            'planning': ['plan', 'steps', 'strategy', 'approach', 'timeline'],
            'analysis': ['analyze', 'data', 'pattern', 'insight', 'finding'],
            'critique': ['issue', 'problem', 'improve', 'recommend', 'concern'],
            'synthesis': ['combine', 'integrate', 'overall', 'summary', 'conclusion'],
            'execution': ['implement', 'action', 'execute', 'deploy', 'process']
        }
        
        indicators = task_indicators.get(task_type, [])
        matches = sum(1 for indicator in indicators if indicator in response.lower())
        indicator_bonus = min(0.15, matches * 0.03)
        quality_score += indicator_bonus
        
        return min(1.0, quality_score)
    
    async def _execute_llm_request(self, prompt: str, model_name: str) -> str:
        """Execute LLM request with error handling"""
        try:
            # This would integrate with actual LLM providers
            # For now, return placeholder response
            return f"Simulated response for prompt: {prompt[:100]}..."
            
        except Exception as e:
            logger.error(f"Error executing LLM request: {e}")
            return "Error in LLM execution"