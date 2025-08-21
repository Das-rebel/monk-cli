"""
MONK CLI Agent Orchestrator
Manages agent selection, coordination, and execution
"""
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import logging

from .agent_framework import (
    BaseAgent, TaskContext, AgentResponse, PersonalityProfile,
    create_agent, AGENT_REGISTRY
)
from ..core.config import config
from ..core.database import get_db_session, cache
from ..core.models import Agent, AgentExecution, User

logger = logging.getLogger(__name__)


@dataclass
class AgentSelectionResult:
    """Result of agent selection process"""
    selected_agent: BaseAgent
    confidence_score: float
    selection_reasoning: str
    alternative_agents: List[Tuple[BaseAgent, float]]
    selection_time_ms: int


class AgentOrchestrator:
    """
    Orchestrates agent selection and execution
    Implements supervisor pattern for managing multiple agents
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_pools: Dict[str, List[BaseAgent]] = defaultdict(list)
        self.task_queue = asyncio.Queue()
        self.running = False
        self.max_concurrent_tasks = config.agents.max_concurrent_agents
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.successful_tasks = 0
        self.average_selection_time = 0.0
        
        # Initialize agent pools
        self._initialize_agent_pools()
    
    def _initialize_agent_pools(self):
        """Initialize agent pools for different stacks"""
        # Development stack agents
        development_agents = [
            create_agent("architect"),
            create_agent("quality_enforcer"),
            create_agent("innovation_driver"),
            create_agent("integration_specialist")
        ]
        
        for agent in development_agents:
            self.agents[agent.agent_id] = agent
            self.agent_pools["development"].extend([agent])
        
        # Content stack (Phase 2 - placeholder for now)
        self.agent_pools["content"] = []
        
        # Business stack (Phase 2 - placeholder for now)
        self.agent_pools["business"] = []
        
        # Security stack (Phase 2 - placeholder for now)
        self.agent_pools["security"] = []
        
        logger.info(f"Initialized {len(self.agents)} agents across {len(self.agent_pools)} stacks")
    
    async def select_optimal_agent(self, context: TaskContext) -> AgentSelectionResult:
        """
        Select the optimal agent for a given task using multiple criteria:
        1. Domain specialization match
        2. Personality fit for task type
        3. Current agent availability and load
        4. Historical performance on similar tasks
        """
        start_time = time.time()
        
        # Get available agents
        available_agents = [agent for agent in self.agents.values() if not agent.is_busy]
        if not available_agents:
            raise RuntimeError("No available agents")
        
        # Score each agent
        agent_scores = []
        for agent in available_agents:
            if agent.can_handle_task(context):
                score = await self._calculate_agent_score(agent, context)
                agent_scores.append((agent, score))
        
        if not agent_scores:
            raise RuntimeError("No suitable agents for this task")
        
        # Sort by score (highest first)
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_agent, best_score = agent_scores[0]
        alternatives = agent_scores[1:5]  # Top 5 alternatives
        
        selection_time_ms = int((time.time() - start_time) * 1000)
        
        # Generate reasoning
        reasoning = self._generate_selection_reasoning(selected_agent, context, best_score)
        
        # Update selection time metrics
        self._update_selection_metrics(selection_time_ms)
        
        return AgentSelectionResult(
            selected_agent=selected_agent,
            confidence_score=best_score,
            selection_reasoning=reasoning,
            alternative_agents=alternatives,
            selection_time_ms=selection_time_ms
        )
    
    async def _calculate_agent_score(self, agent: BaseAgent, context: TaskContext) -> float:
        """Calculate comprehensive agent score for task"""
        # Base suitability from agent
        base_score = agent.calculate_suitability(context)
        
        # Performance modifier based on historical success
        performance_modifier = self._calculate_performance_modifier(agent)
        
        # Load balancing modifier (prefer less busy agents)
        load_modifier = self._calculate_load_modifier(agent)
        
        # Personality fit modifier
        personality_modifier = await self._calculate_personality_fit(agent, context)
        
        # Combine scores with weights
        final_score = (
            base_score * 0.4 +
            performance_modifier * 0.2 +
            load_modifier * 0.15 +
            personality_modifier * 0.25
        )
        
        return min(1.0, final_score)
    
    def _calculate_performance_modifier(self, agent: BaseAgent) -> float:
        """Calculate performance modifier based on agent's historical success"""
        if agent.total_executions == 0:
            return 0.5  # Neutral score for new agents
        
        success_rate = agent.successful_executions / agent.total_executions
        
        # Scale success rate to modifier range (0.0 to 1.0)
        return success_rate
    
    def _calculate_load_modifier(self, agent: BaseAgent) -> float:
        """Calculate load modifier - prefer agents with lighter load"""
        # For now, simple busy check (Phase 2 will have more sophisticated load balancing)
        return 1.0 if not agent.is_busy else 0.0
    
    async def _calculate_personality_fit(self, agent: BaseAgent, context: TaskContext) -> float:
        """Calculate how well agent personality fits the task"""
        personality = agent.personality
        
        # Task complexity preference
        complexity_fit = 0.5
        if context.complexity_level > 0.7:
            # Complex tasks favor high conscientiousness and analytical thinking
            complexity_fit = (personality.conscientiousness + personality.analytical_thinking) / 2
        elif context.complexity_level < 0.3:
            # Simple tasks favor efficiency (lower conscientiousness can be faster)
            complexity_fit = 1.0 - (personality.conscientiousness * 0.3)
        
        # Urgency preference
        urgency_fit = 0.5
        if context.urgency_level > 0.8:
            # Urgent tasks favor low neuroticism and high extraversion
            urgency_fit = (1.0 - personality.neuroticism + personality.extraversion) / 2
        
        # Creative tasks preference
        creative_fit = 0.5
        creative_keywords = ["creative", "innovative", "design", "brainstorm"]
        if any(keyword in context.task_description.lower() for keyword in creative_keywords):
            creative_fit = (personality.creativity + personality.openness) / 2
        
        # Average personality fit components
        return (complexity_fit + urgency_fit + creative_fit) / 3
    
    def _generate_selection_reasoning(self, agent: BaseAgent, context: TaskContext, score: float) -> str:
        """Generate human-readable reasoning for agent selection"""
        reasons = []
        
        # Domain specialization
        if context.domain in agent.specializations:
            reasons.append(f"Expert in {context.domain}")
        
        # Personality traits
        personality = agent.personality
        if personality.conscientiousness > 0.8:
            reasons.append("High attention to detail")
        if personality.creativity > 0.8:
            reasons.append("Highly creative approach")
        if personality.analytical_thinking > 0.8:
            reasons.append("Strong analytical skills")
        
        # Performance history
        if agent.total_executions > 10:
            success_rate = agent.successful_executions / agent.total_executions
            if success_rate > 0.9:
                reasons.append("Excellent track record")
            elif success_rate > 0.8:
                reasons.append("Good performance history")
        
        # Task fit
        if score > 0.8:
            reasons.append("Excellent task fit")
        elif score > 0.6:
            reasons.append("Good task alignment")
        
        return f"Selected {agent.name}: {', '.join(reasons[:3])} (confidence: {score:.2f})"
    
    async def execute_task(self, context: TaskContext) -> Tuple[AgentResponse, AgentSelectionResult]:
        """Execute a task by selecting optimal agent and running it"""
        try:
            # Select optimal agent
            selection_result = await self.select_optimal_agent(context)
            selected_agent = selection_result.selected_agent
            
            logger.info(f"Selected {selected_agent.name} for task: {context.task_description[:100]}...")
            
            # Execute task with selected agent
            response = await selected_agent.execute_task(context)
            
            # Store execution record in database
            await self._store_execution_record(context, selected_agent, response)
            
            # Update orchestrator metrics
            self.total_tasks_processed += 1
            if response.success:
                self.successful_tasks += 1
            
            return response, selection_result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise
    
    async def _store_execution_record(self, context: TaskContext, agent: BaseAgent, response: AgentResponse):
        """Store agent execution record in database"""
        async with get_db_session() as session:
            # Get or create user
            user = await session.get(User, context.user_id)
            if not user:
                logger.warning(f"User {context.user_id} not found, skipping execution record")
                return
            
            # Create execution record
            execution = AgentExecution(
                user_id=context.user_id,
                agent_id=agent.agent_id,
                task_description=context.task_description,
                task_type=context.task_type,
                execution_context=context.to_dict(),
                result=response.to_dict(),
                success=response.success,
                error_message=response.error_message,
                execution_time_ms=response.execution_time_ms,
                tokens_used=response.tokens_used,
                memory_queries_made=response.memory_queries_made
            )
            
            session.add(execution)
            await session.commit()
    
    def _update_selection_metrics(self, selection_time_ms: int):
        """Update agent selection performance metrics"""
        if self.total_tasks_processed == 0:
            self.average_selection_time = selection_time_ms
        else:
            self.average_selection_time = (
                (self.average_selection_time * self.total_tasks_processed + selection_time_ms)
                / (self.total_tasks_processed + 1)
            )
    
    async def execute_collaborative_task(
        self, 
        context: TaskContext, 
        required_agents: Optional[List[str]] = None
    ) -> Dict[str, AgentResponse]:
        """
        Execute a task that requires collaboration between multiple agents
        Phase 1: Basic sequential collaboration
        Phase 2: Will add parallel and dynamic collaboration patterns
        """
        if required_agents is None:
            # Auto-select complementary agents based on task
            required_agents = await self._select_collaborative_agents(context)
        
        results = {}
        
        # Sequential collaboration for Phase 1
        for agent_type in required_agents:
            if agent_type not in self.agents:
                logger.warning(f"Agent type {agent_type} not available for collaboration")
                continue
            
            agent = self.agents[agent_type]
            if agent.can_handle_task(context):
                # Update context with previous results
                context.context_data["previous_agent_results"] = results
                
                # Execute with current agent
                response = await agent.execute_task(context)
                results[agent_type] = response
                
                logger.info(f"Collaborative agent {agent.name} completed with success: {response.success}")
        
        return results
    
    async def _select_collaborative_agents(self, context: TaskContext) -> List[str]:
        """Select agents for collaborative task based on task characteristics"""
        # Simple heuristics for Phase 1
        # Phase 2 will use ML-based agent selection
        
        selected_agents = []
        
        # Always start with architect for complex tasks
        if context.complexity_level > 0.6:
            selected_agents.append("architect")
        
        # Add quality enforcer for review tasks
        review_keywords = ["review", "check", "validate", "test"]
        if any(keyword in context.task_description.lower() for keyword in review_keywords):
            selected_agents.append("quality_enforcer")
        
        # Add innovation driver for optimization tasks
        optimize_keywords = ["optimize", "improve", "performance", "faster"]
        if any(keyword in context.task_description.lower() for keyword in optimize_keywords):
            selected_agents.append("innovation_driver")
        
        # Add integration specialist for deployment tasks
        deploy_keywords = ["deploy", "integrate", "api", "service"]
        if any(keyword in context.task_description.lower() for keyword in deploy_keywords):
            selected_agents.append("integration_specialist")
        
        # Ensure at least one agent
        if not selected_agents:
            selected_agents = ["architect"]  # Default to architect
        
        return selected_agents
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status and metrics"""
        agent_statuses = {agent_id: agent.get_status() for agent_id, agent in self.agents.items()}
        
        success_rate = self.successful_tasks / max(self.total_tasks_processed, 1)
        
        return {
            "total_agents": len(self.agents),
            "agents": agent_statuses,
            "agent_pools": {pool: len(agents) for pool, agents in self.agent_pools.items()},
            "total_tasks_processed": self.total_tasks_processed,
            "successful_tasks": self.successful_tasks,
            "success_rate": success_rate,
            "average_selection_time_ms": self.average_selection_time,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "running": self.running
        }
    
    async def start(self):
        """Start the orchestrator (for future background task processing)"""
        self.running = True
        logger.info("Agent Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator"""
        self.running = False
        logger.info("Agent Orchestrator stopped")


# Global orchestrator instance
orchestrator = AgentOrchestrator()