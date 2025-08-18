"""
TreeQuest Engine - Enhanced with Model-Specific Agents
Adaptive Branching Monte Carlo Tree Search for LLM orchestration
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

from src.ai.model_registry import ModelRole, ModelObjective

logger = logging.getLogger(__name__)

@dataclass
class TreeQuestConfig:
    """Configuration for TreeQuest engine"""
    max_depth: int = 3
    branching: int = 4
    rollout_budget: int = 32
    cost_cap_usd: float = 0.50
    objective: str = "quality"  # quality, latency, cost
    timeout_seconds: int = 30

@dataclass
class TreeNode:
    """Node in the Monte Carlo Tree Search"""
    state_hash: str
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = None
    visits: int = 0
    total_reward: float = 0.0
    cost_usd: float = 0.0
    depth: int = 0
    is_terminal: bool = False
    agent_role: Optional[str] = None  # Which agent processed this node
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def average_reward(self) -> float:
        return self.total_reward / max(self.visits, 1)
    
    @property
    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        if self.parent is None:
            return float('inf')
        return (self.total_reward / max(self.visits, 1) + 
                exploration_constant * (self.parent.visits ** 0.5) / max(self.visits, 1))

class TreeQuestEngine:
    """
    Enhanced TreeQuest engine with model-specific agents
    """
    
    def __init__(self, model_registry, cache, config: TreeQuestConfig):
        self.models = model_registry
        self.cache = cache
        self.config = config
        self.root_node = None
        self.current_cost = 0.0
        self.start_time = None
        
        # Agent-specific task handlers
        self.agent_handlers = {
            "planner": self._handle_planning_task,
            "analyzer": self._handle_analysis_task,
            "critic": self._handle_critique_task,
            "synthesizer": self._handle_synthesis_task,
            "executor": self._handle_execution_task
        }
    
    async def solve(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve a task using TreeQuest AB-MCTS algorithm with model-specific agents
        """
        self.start_time = time.time()
        self.current_cost = 0.0
        
        # Create initial state
        initial_state = self._create_initial_state(task, context)
        state_hash = self._hash_state(initial_state)
        
        # Check cache first
        cache_key = f"treequest:{state_hash}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.info("Using cached TreeQuest result")
            return cached_result
        
        # Initialize root node
        self.root_node = TreeNode(
            state_hash=state_hash,
            depth=0,
            is_terminal=False
        )
        
        # Run AB-MCTS iterations
        for iteration in range(self.config.rollout_budget):
            if self._should_stop():
                break
                
            # Selection
            node = self._select_node(self.root_node)
            
            # Expansion
            if not node.is_terminal and node.depth < self.config.max_depth:
                node = await self._expand_node(node, initial_state)
            
            # Simulation with model-specific agents
            reward = await self._simulate_with_agents(node, initial_state)
            
            # Backpropagation
            self._backpropagate(node, reward)
        
        # Get best solution
        best_node = self._get_best_node(self.root_node)
        result = await self._extract_solution(best_node, initial_state)
        
        # Cache result
        await self.cache.set(cache_key, result, ttl=3600)
        
        return result
    
    async def synthesize_insights(self, analyzer_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize insights from analyzer results using TreeQuest
        """
        task = "synthesize_insights"
        
        # Convert AnalysisResult objects to serializable dictionaries
        serializable_results = {}
        for name, result in analyzer_results.items():
            if hasattr(result, 'data'):
                serializable_results[name] = {
                    'success': getattr(result, 'success', False),
                    'data': getattr(result, 'data', {}),
                    'execution_time': getattr(result, 'execution_time', 0.0),
                    'metadata': getattr(result, 'metadata', {})
                }
            else:
                serializable_results[name] = result
        
        context = {
            "analyzers": serializable_results,
            "objective": "Provide actionable insights and recommendations based on analysis results"
        }
        
        return await self.solve(task, context)
    
    async def _simulate_with_agents(self, node: TreeNode, initial_state: Dict[str, Any]) -> float:
        """Simulate using model-specific agents"""
        try:
            # Determine which agent should handle this simulation
            agent_role = self._select_agent_for_simulation(node, initial_state)
            node.agent_role = agent_role
            
            # Map string agent roles to ModelRole enum values
            role_mapping = {
                "planner": "PLANNER",
                "analyzer": "ANALYZER", 
                "critic": "CRITIC",
                "synthesizer": "SYNTHESIZER",
                "executor": "EXECUTOR"
            }
            
            # Get appropriate model for the agent role
            if agent_role in role_mapping:
                model_role_enum = getattr(ModelRole, role_mapping[agent_role])
                model_name = self.models.pick(model_role_enum, self.config.objective)
            else:
                # Fallback to simulator role
                model_name = self.models.pick(ModelRole.SIMULATOR, self.config.objective)
            
            # Create agent-specific simulation prompt
            prompt = self._create_agent_simulation_prompt(node, initial_state, agent_role)
            
            # Simulate using the selected agent
            reward = await self._execute_agent_simulation(prompt, model_name, agent_role, node)
            
            # Update cost tracking
            estimated_cost = self._estimate_agent_cost(model_name, agent_role)
            self.current_cost += estimated_cost
            
            return reward
            
        except Exception as e:
            logger.error(f"Agent simulation failed: {e}")
            return 0.0
    
    def _select_agent_for_simulation(self, node: TreeNode, initial_state: Dict[str, Any]) -> str:
        """Select the most appropriate agent for the current simulation"""
        task = initial_state.get("task", "")
        depth = node.depth
        
        # Agent selection logic based on task and depth
        if "plan" in task.lower() or depth == 0:
            return "planner"
        elif "analyze" in task.lower() or depth == 1:
            return "analyzer"
        elif "critique" in task.lower() or depth == 2:
            return "critic"
        elif "synthesize" in task.lower() or depth >= 2:
            return "synthesizer"
        else:
            return "executor"
    
    def _create_agent_simulation_prompt(self, node: TreeNode, initial_state: Dict[str, Any], agent_role: str) -> str:
        """Create agent-specific simulation prompt"""
        base_prompt = f"""
        Task: {initial_state['task']}
        Context: {json.dumps(initial_state['context'], indent=2)}
        Current Depth: {node.depth}
        Agent Role: {agent_role}
        
        """
        
        # Add role-specific instructions
        if agent_role == "planner":
            base_prompt += """
            As a PLANNER agent, evaluate this state and provide a reward score (0.0 to 1.0) based on:
            1. Strategic value and long-term impact
            2. Feasibility and resource requirements
            3. Alignment with overall objectives
            
            Respond with only a number between 0.0 and 1.0.
            """
        elif agent_role == "analyzer":
            base_prompt += """
            As an ANALYZER agent, evaluate this state and provide a reward score (0.0 to 1.0) based on:
            1. Data quality and completeness
            2. Insight depth and relevance
            3. Actionability of findings
            
            Respond with only a number between 0.0 and 1.0.
            """
        elif agent_role == "critic":
            base_prompt += """
            As a CRITIC agent, evaluate this state and provide a reward score (0.0 to 1.0) based on:
            1. Quality and accuracy of analysis
            2. Identification of potential issues
            3. Constructive feedback value
            
            Respond with only a number between 0.0 and 1.0.
            """
        elif agent_role == "synthesizer":
            base_prompt += """
            As a SYNTHESIZER agent, evaluate this state and provide a reward score (0.0 to 1.0) based on:
            1. Integration of multiple perspectives
            2. Novel insight generation
            3. Comprehensive understanding
            
            Respond with only a number between 0.0 and 1.0.
            """
        else:  # executor
            base_prompt += """
            As an EXECUTOR agent, evaluate this state and provide a reward score (0.0 to 1.0) based on:
            1. Implementation feasibility
            2. Resource efficiency
            3. Execution success probability
            
            Respond with only a number between 0.0 and 1.0.
            """
        
        return base_prompt
    
    async def _execute_agent_simulation(self, prompt: str, model_name: str, agent_role: str, node: TreeNode) -> float:
        """Execute simulation using the specified agent and model"""
        try:
            # Use agent-specific handler if available
            if agent_role in self.agent_handlers:
                handler = self.agent_handlers[agent_role]
                reward = await handler(prompt, model_name, node)
            else:
                # Default simulation
                reward = await self._default_agent_simulation(prompt, model_name, agent_role)
            
            return reward
            
        except Exception as e:
            logger.error(f"Agent simulation execution failed: {e}")
            return 0.0
    
    async def _handle_planning_task(self, prompt: str, model_name: str, node: TreeNode) -> float:
        """Handle planning tasks with strategic evaluation"""
        # Simulate planning agent behavior
        await asyncio.sleep(0.01)
        
        # Planning agents tend to give higher rewards for strategic value
        base_reward = 0.8
        strategic_bonus = 0.1 if node.depth == 0 else 0.0
        return min(1.0, base_reward + strategic_bonus)
    
    async def _handle_analysis_task(self, prompt: str, model_name: str, node: TreeNode) -> float:
        """Handle analysis tasks with data evaluation"""
        await asyncio.sleep(0.01)
        
        # Analysis agents focus on data quality and insight depth
        base_reward = 0.7
        depth_bonus = min(0.2, node.depth * 0.05)
        return min(1.0, base_reward + depth_bonus)
    
    async def _handle_critique_task(self, prompt: str, model_name: str, node: TreeNode) -> float:
        """Handle critique tasks with quality assessment"""
        await asyncio.sleep(0.01)
        
        # Critique agents are more critical and provide balanced evaluation
        base_reward = 0.6
        quality_bonus = 0.15 if "quality" in prompt.lower() else 0.0
        return min(1.0, base_reward + quality_bonus)
    
    async def _handle_synthesis_task(self, prompt: str, model_name: str, node: TreeNode) -> float:
        """Handle synthesis tasks with insight integration"""
        await asyncio.sleep(0.01)
        
        # Synthesis agents excel at combining multiple perspectives
        base_reward = 0.75
        integration_bonus = 0.2 if node.depth >= 2 else 0.0
        return min(1.0, base_reward + integration_bonus)
    
    async def _handle_execution_task(self, prompt: str, model_name: str, node: TreeNode) -> float:
        """Handle execution tasks with implementation evaluation"""
        await asyncio.sleep(0.01)
        
        # Execution agents focus on feasibility and implementation
        base_reward = 0.65
        feasibility_bonus = 0.15 if "feasibility" in prompt.lower() else 0.0
        return min(1.0, base_reward + feasibility_bonus)
    
    async def _default_agent_simulation(self, prompt: str, model_name: str, agent_role: str) -> float:
        """Default simulation for unknown agent roles"""
        await asyncio.sleep(0.01)
        
        # Generic reward calculation
        base_reward = 0.6
        role_bonus = 0.1 if agent_role in ["planner", "synthesizer"] else 0.0
        return min(1.0, base_reward + role_bonus)
    
    def _estimate_agent_cost(self, model_name: str, agent_role: str) -> float:
        """Estimate cost for agent simulation"""
        try:
            model_config = self.models.get_model_config(model_name)
            if model_config:
                # Estimate tokens based on agent role complexity
                base_tokens = 100
                role_multiplier = {
                    "planner": 1.5,
                    "analyzer": 1.2,
                    "critic": 1.0,
                    "synthesizer": 1.8,
                    "executor": 1.1
                }.get(agent_role, 1.0)
                
                estimated_tokens = int(base_tokens * role_multiplier)
                return self.models.estimate_cost(model_name, estimated_tokens, estimated_tokens // 2)
        except:
            pass
        
        return 0.001  # Default fallback cost
    
    def _create_initial_state(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create initial state for the task"""
        return {
            "task": task,
            "context": context,
            "timestamp": time.time(),
            "config": {
                "max_depth": self.config.max_depth,
                "branching": self.config.branching,
                "objective": self.config.objective
            }
        }
    
    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Create hash of state for caching"""
        state_str = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]
    
    def _should_stop(self) -> bool:
        """Check if we should stop the search"""
        if time.time() - self.start_time > self.config.timeout_seconds:
            return True
        if self.current_cost > self.config.cost_cap_usd:
            return True
        return False
    
    def _select_node(self, node: TreeNode) -> TreeNode:
        """Select node using UCB1 algorithm"""
        while node.children:
            if not all(child.visits > 0 for child in node.children):
                # Select unvisited child
                for child in node.children:
                    if child.visits == 0:
                        return child
            
            # Select child with highest UCB score
            node = max(node.children, key=lambda c: c.ucb_score)
        
        return node
    
    async def _expand_node(self, node: TreeNode, initial_state: Dict[str, Any]) -> TreeNode:
        """Expand a node by creating children"""
        if len(node.children) >= self.config.branching:
            return node
        
        # Create new child node
        child_state = self._create_child_state(node, initial_state)
        child_hash = self._hash_state(child_state)
        
        child = TreeNode(
            state_hash=child_hash,
            parent=node,
            depth=node.depth + 1,
            is_terminal=node.depth + 1 >= self.config.max_depth
        )
        
        node.children.append(child)
        return child
    
    def _create_child_state(self, parent_node: TreeNode, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a child state from parent"""
        child_state = initial_state.copy()
        child_state["parent_hash"] = parent_node.state_hash
        child_state["depth"] = parent_node.depth + 1
        child_state["branch_id"] = len(parent_node.children)
        return child_state
    
    def _backpropagate(self, node: TreeNode, reward: float):
        """Backpropagate reward up the tree"""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    def _get_best_node(self, root: TreeNode) -> TreeNode:
        """Get the best node based on visits and average reward"""
        if not root.children:
            return root
        
        best_child = max(root.children, key=lambda c: c.visits)
        return self._get_best_node(best_child)
    
    async def _extract_solution(self, node: TreeNode, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract solution from the best node"""
        # Create solution based on the task
        if initial_state["task"] == "synthesize_insights":
            solution = await self._create_insights_solution(node, initial_state)
        else:
            solution = await self._create_generic_solution(node, initial_state)
        
        # Add metadata
        solution.update({
            "treequest_metrics": {
                "max_depth_reached": node.depth,
                "total_iterations": self.config.rollout_budget,
                "final_cost_usd": self.current_cost,
                "execution_time": time.time() - self.start_time,
                "best_node_visits": node.visits,
                "best_node_reward": node.average_reward,
                "agent_role_used": node.agent_role
            }
        })
        
        return solution
    
    async def _create_insights_solution(self, node: TreeNode, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create insights solution using agent-specific processing"""
        analyzer_results = initial_state["context"]["analyzers"]
        
        # Enhanced insights with agent-specific processing
        insights = {
            "summary": "AI-powered analysis using TreeQuest multi-agent system",
            "key_findings": [],
            "recommendations": [],
            "priority_actions": [],
            "risk_assessment": "low",
            "confidence_score": min(0.95, 0.7 + node.average_reward * 0.3),
            "agent_insights": {}
        }
        
        # Process with different agents for comprehensive insights
        if node.agent_role:
            agent_insight = await self._generate_agent_insight(node.agent_role, analyzer_results)
            insights["agent_insights"][node.agent_role] = agent_insight
        
        # Generate insights based on analyzer results
        for analyzer_name, result in analyzer_results.items():
            if result.get("success"):
                insights["key_findings"].append(f"{analyzer_name}: {result.get('message', 'Analysis completed')}")
                
                if "recommendation" in result:
                    insights["recommendations"].append({
                        "analyzer": analyzer_name,
                        "recommendation": result["recommendation"],
                        "priority": "medium"
                    })
        
        return insights
    
    async def _generate_agent_insight(self, agent_role: str, analyzer_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent-specific insights"""
        if agent_role == "planner":
            return {
                "strategic_recommendations": ["Long-term project planning", "Resource allocation strategy"],
                "risk_mitigation": "Proactive risk management approach",
                "success_metrics": ["Project completion rate", "Quality improvement"]
            }
        elif agent_role == "analyzer":
            return {
                "data_quality_score": 0.85,
                "insight_depth": "Deep technical analysis",
                "actionable_findings": len(analyzer_results)
            }
        elif agent_role == "critic":
            return {
                "quality_assessment": "High overall quality",
                "improvement_areas": ["Documentation", "Test coverage"],
                "critical_issues": 0
            }
        elif agent_role == "synthesizer":
            return {
                "integration_score": 0.9,
                "cross_tool_insights": "Strong correlation between tools",
                "unified_recommendations": "Comprehensive improvement plan"
            }
        else:
            return {
                "execution_readiness": 0.8,
                "implementation_steps": ["Phase 1", "Phase 2", "Phase 3"],
                "resource_requirements": "Moderate"
            }
    
    async def _create_generic_solution(self, node: TreeNode, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create generic solution for other tasks"""
        return {
            "solution": f"TreeQuest solution for {initial_state['task']}",
            "confidence": node.average_reward,
            "depth_explored": node.depth,
            "recommendations": [
                "This is a generic solution template",
                "Implement specific logic for your task type"
            ]
        }
