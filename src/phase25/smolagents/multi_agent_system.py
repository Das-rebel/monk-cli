"""
MONK CLI Phase 2.5 - Multi-Agent System with Smolagents
Enhanced multi-agent orchestration using Hugging Face smolagents
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from smolagents import Agent, CodeAgent, ToolBox, HfApiModel
    from smolagents.types import AgentOutput
    from smolagents.tools import Tool
    import torch
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False

from ...core.config import config
from ...agents.agent_selector import PersonalityProfile, AgentPersonality, AgentType
from ...memory.memory_system import MemorySystem, MemoryQuery, MemoryResult
from ..bridges.treequest_smolagent_bridge import TreeQuestTask, TreeQuestSmolagentBridge
from ...core.database import get_db_session

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class AgentRole(Enum):
    """Agent roles in multi-agent system"""
    COORDINATOR = "coordinator"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    OPTIMIZER = "optimizer"
    MONITOR = "monitor"


@dataclass
class AgentTask:
    """Task assigned to specific agent"""
    task_id: str
    agent_id: str
    description: str
    input_data: Dict[str, Any]
    dependencies: List[str]
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 1  # 1=low, 5=high


@dataclass
class MultiAgentConfiguration:
    """Configuration for multi-agent system"""
    max_concurrent_agents: int = 4
    task_timeout_seconds: int = 300
    enable_agent_communication: bool = True
    load_balancing_strategy: str = "round_robin"  # round_robin, least_loaded, capability_based
    result_aggregation_method: str = "voting"  # voting, weighted, first_success
    failure_handling: str = "retry_with_fallback"  # retry, retry_with_fallback, fail_fast


@dataclass
class AgentCapability:
    """Agent capability definition"""
    agent_id: str
    personality: AgentPersonality
    roles: List[AgentRole]
    specializations: List[str]
    max_concurrent_tasks: int
    current_load: int
    tools: List[str]
    model_name: str
    performance_score: float = 1.0
    availability: bool = True


class MONKMultiAgentSystem:
    """Multi-agent system orchestrator with smolagents integration"""
    
    def __init__(self, config: MultiAgentConfiguration = None):
        self.config = config or MultiAgentConfiguration()
        self.smolagents_available = SMOLAGENTS_AVAILABLE
        
        # Agent management
        self.agents: Dict[str, Any] = {}  # agent_id -> smolagent instance
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.agent_toolboxes: Dict[str, ToolBox] = {}
        
        # Task management
        self.tasks: Dict[str, AgentTask] = {}
        self.task_queue: List[str] = []
        self.completed_tasks: Dict[str, AgentTask] = {}
        self.failed_tasks: Dict[str, AgentTask] = {}
        
        # System state
        self.system_status = "initializing"
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_agents)
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0,
            "agent_utilization": {}
        }
        
        # Integration with existing MONK systems
        self.memory_system = MemorySystem()
        self.treequest_bridge = TreeQuestSmolagentBridge()
        
    async def initialize(self):
        """Initialize multi-agent system"""
        try:
            logger.info("Initializing MONK Multi-Agent System with smolagents")
            
            # Initialize TreeQuest bridge
            await self.treequest_bridge.initialize()
            
            # Initialize memory system
            await self.memory_system.initialize()
            
            # Setup default agents
            await self._setup_default_agents()
            
            # Initialize monitoring
            self._start_monitoring()
            
            self.system_status = "ready"
            logger.info("Multi-Agent System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-agent system: {e}")
            self.system_status = "error"
            return False
    
    async def _setup_default_agents(self):
        """Setup default agents for each personality type"""
        if not self.smolagents_available:
            logger.warning("Smolagents not available - using mock agents")
            await self._setup_mock_agents()
            return
        
        default_agent_configs = [
            {
                "personality": AgentPersonality.ANALYTICAL,
                "roles": [AgentRole.EXECUTOR, AgentRole.VALIDATOR],
                "specializations": ["code_analysis", "data_processing", "problem_solving"],
                "tools": ["python", "file_system", "web_search", "calculator"],
                "model": "microsoft/DialoGPT-medium"
            },
            {
                "personality": AgentPersonality.CREATIVE,
                "roles": [AgentRole.EXECUTOR, AgentRole.OPTIMIZER],
                "specializations": ["design", "brainstorming", "innovation"],
                "tools": ["web_search", "image_generator", "file_system"],
                "model": "microsoft/DialoGPT-medium"
            },
            {
                "personality": AgentPersonality.DETAIL_ORIENTED,
                "roles": [AgentRole.EXECUTOR, AgentRole.VALIDATOR],
                "specializations": ["testing", "documentation", "quality_assurance"],
                "tools": ["python", "file_system", "calculator", "validator"],
                "model": "microsoft/DialoGPT-medium"
            },
            {
                "personality": AgentPersonality.COLLABORATIVE,
                "roles": [AgentRole.COORDINATOR, AgentRole.MONITOR],
                "specializations": ["project_management", "communication", "coordination"],
                "tools": ["web_search", "file_system", "email", "calendar"],
                "model": "microsoft/DialoGPT-medium"
            }
        ]
        
        for agent_config in default_agent_configs:
            await self._create_agent(agent_config)
    
    async def _setup_mock_agents(self):
        """Setup mock agents when smolagents unavailable"""
        mock_personalities = [
            AgentPersonality.ANALYTICAL,
            AgentPersonality.CREATIVE, 
            AgentPersonality.DETAIL_ORIENTED,
            AgentPersonality.COLLABORATIVE
        ]
        
        for personality in mock_personalities:
            agent_id = f"mock_{personality.value}_{uuid.uuid4().hex[:8]}"
            
            capability = AgentCapability(
                agent_id=agent_id,
                personality=personality,
                roles=[AgentRole.EXECUTOR],
                specializations=[personality.value],
                max_concurrent_tasks=2,
                current_load=0,
                tools=["mock_tool"],
                model_name="mock_model"
            )
            
            self.agent_capabilities[agent_id] = capability
            self.agents[agent_id] = None  # Mock agent
            
            logger.info(f"Created mock agent: {agent_id} ({personality.value})")
    
    async def _create_agent(self, agent_config: Dict[str, Any]):
        """Create smolagent with specified configuration"""
        try:
            agent_id = f"{agent_config['personality'].value}_{uuid.uuid4().hex[:8]}"
            
            # Create toolbox for agent
            toolbox = ToolBox()
            # In practice, add specific tools based on agent_config['tools']
            self.agent_toolboxes[agent_id] = toolbox
            
            # Create HF model
            model = HfApiModel(model_id=agent_config['model'])
            
            # Create smolagent
            agent = Agent(
                model=model,
                toolbox=toolbox,
                max_iterations=20,
                planning_interval=5
            )
            
            # Store agent and its capabilities
            self.agents[agent_id] = agent
            
            capability = AgentCapability(
                agent_id=agent_id,
                personality=agent_config['personality'],
                roles=agent_config['roles'],
                specializations=agent_config['specializations'],
                max_concurrent_tasks=3,
                current_load=0,
                tools=agent_config['tools'],
                model_name=agent_config['model']
            )
            
            self.agent_capabilities[agent_id] = capability
            
            logger.info(f"Created smolagent: {agent_id} ({agent_config['personality'].value})")
            
        except Exception as e:
            logger.error(f"Failed to create agent {agent_config.get('personality', 'unknown')}: {e}")
    
    async def submit_task(self, description: str, input_data: Dict[str, Any] = None,
                         required_personality: AgentPersonality = None,
                         required_role: AgentRole = None,
                         priority: int = 1,
                         dependencies: List[str] = None) -> str:
        """Submit task to multi-agent system"""
        
        if input_data is None:
            input_data = {}
        if dependencies is None:
            dependencies = []
        
        task_id = uuid.uuid4().hex
        
        # Select appropriate agent
        selected_agent_id = await self._select_agent(
            required_personality=required_personality,
            required_role=required_role,
            task_complexity=input_data.get("complexity", 1)
        )
        
        if not selected_agent_id:
            raise ValueError("No suitable agent available for task")
        
        # Create task
        task = AgentTask(
            task_id=task_id,
            agent_id=selected_agent_id,
            description=description,
            input_data=input_data,
            dependencies=dependencies,
            status=TaskStatus.PENDING,
            priority=priority
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        
        # Update agent load
        self.agent_capabilities[selected_agent_id].current_load += 1
        
        logger.info(f"Submitted task {task_id} to agent {selected_agent_id}")
        
        # Schedule execution if dependencies are met
        if await self._dependencies_met(task_id):
            await self._schedule_task_execution(task_id)
        
        return task_id
    
    async def _select_agent(self, required_personality: AgentPersonality = None,
                          required_role: AgentRole = None,
                          task_complexity: int = 1) -> Optional[str]:
        """Select best available agent for task"""
        
        available_agents = [
            (agent_id, cap) for agent_id, cap in self.agent_capabilities.items()
            if cap.availability and cap.current_load < cap.max_concurrent_tasks
        ]
        
        if not available_agents:
            return None
        
        # Filter by requirements
        if required_personality:
            available_agents = [
                (agent_id, cap) for agent_id, cap in available_agents
                if cap.personality == required_personality
            ]
        
        if required_role:
            available_agents = [
                (agent_id, cap) for agent_id, cap in available_agents
                if required_role in cap.roles
            ]
        
        if not available_agents:
            return None
        
        # Select based on load balancing strategy
        if self.config.load_balancing_strategy == "least_loaded":
            selected = min(available_agents, key=lambda x: x[1].current_load)
        elif self.config.load_balancing_strategy == "capability_based":
            selected = max(available_agents, key=lambda x: x[1].performance_score)
        else:  # round_robin
            selected = available_agents[len(self.completed_tasks) % len(available_agents)]
        
        return selected[0]
    
    async def _dependencies_met(self, task_id: str) -> bool:
        """Check if task dependencies are met"""
        task = self.tasks[task_id]
        
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        return True
    
    async def _schedule_task_execution(self, task_id: str):
        """Schedule task for execution"""
        if task_id in self.running_tasks:
            return  # Already running
        
        task = self.tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        # Create execution coroutine
        execution_coro = self._execute_task(task_id)
        execution_task = asyncio.create_task(execution_coro)
        self.running_tasks[task_id] = execution_task
        
        logger.info(f"Scheduled task {task_id} for execution")
    
    async def _execute_task(self, task_id: str):
        """Execute task with assigned agent"""
        task = self.tasks[task_id]
        
        try:
            # Get agent
            agent = self.agents[task.agent_id]
            agent_capability = self.agent_capabilities[task.agent_id]
            
            logger.info(f"Executing task {task_id} with agent {task.agent_id}")
            
            # Prepare execution context
            context = {
                "task_description": task.description,
                "input_data": task.input_data,
                "agent_personality": agent_capability.personality.value,
                "agent_specializations": agent_capability.specializations,
                "dependencies_results": await self._get_dependency_results(task)
            }
            
            # Execute based on agent availability
            if agent is None:  # Mock agent
                result = await self._execute_mock_task(task, context)
            else:  # Real smolagent
                result = await self._execute_smolagent_task(agent, task, context)
            
            # Process result
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Move to completed tasks
            self.completed_tasks[task_id] = task
            del self.tasks[task_id]
            
            # Update agent load
            agent_capability.current_load = max(0, agent_capability.current_load - 1)
            
            # Update performance metrics
            self._update_performance_metrics(task, success=True)
            
            # Schedule dependent tasks
            await self._schedule_dependent_tasks(task_id)
            
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            # Handle retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                
                # Reschedule with delay
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                await self._schedule_task_execution(task_id)
            else:
                # Move to failed tasks
                self.failed_tasks[task_id] = task
                del self.tasks[task_id]
                
                # Update agent load
                agent_capability = self.agent_capabilities[task.agent_id]
                agent_capability.current_load = max(0, agent_capability.current_load - 1)
                
                # Update performance metrics
                self._update_performance_metrics(task, success=False)
        
        finally:
            # Clean up running task
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def _get_dependency_results(self, task: AgentTask) -> Dict[str, Any]:
        """Get results from dependency tasks"""
        dependency_results = {}
        
        for dep_id in task.dependencies:
            if dep_id in self.completed_tasks:
                dep_task = self.completed_tasks[dep_id]
                dependency_results[dep_id] = dep_task.result
        
        return dependency_results
    
    async def _execute_mock_task(self, task: AgentTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using mock agent"""
        # Simulate execution time
        execution_time = min(max(len(task.description) * 0.1, 1.0), 10.0)
        await asyncio.sleep(execution_time)
        
        return {
            "success": True,
            "result": f"Mock execution of: {task.description}",
            "agent_used": task.agent_id,
            "execution_time_seconds": execution_time,
            "context": context,
            "artifacts": []
        }
    
    async def _execute_smolagent_task(self, agent: Any, task: AgentTask, 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using real smolagent"""
        # Create execution prompt
        prompt = f"""
        Task: {task.description}
        
        Input Data: {json.dumps(task.input_data, indent=2)}
        
        Context: {json.dumps(context, indent=2, default=str)}
        
        Please execute this task and provide detailed output including:
        1. Step-by-step execution
        2. Results and artifacts
        3. Success/failure status
        4. Any recommendations or insights
        """
        
        # Execute with smolagent
        start_time = datetime.now()
        result = agent.run(prompt)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Process smolagent result
        if hasattr(result, 'content'):
            content = result.content
        elif hasattr(result, 'output'):
            content = result.output
        else:
            content = str(result)
        
        return {
            "success": True,
            "result": content,
            "agent_used": task.agent_id,
            "execution_time_seconds": execution_time,
            "context": context,
            "artifacts": self._extract_artifacts_from_result(content)
        }
    
    def _extract_artifacts_from_result(self, content: str) -> List[Dict[str, Any]]:
        """Extract artifacts from smolagent result"""
        artifacts = []
        
        # Extract code blocks
        import re
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL)
        for i, (lang, code) in enumerate(code_blocks):
            artifacts.append({
                "type": "code",
                "language": lang or "text",
                "content": code.strip(),
                "index": i
            })
        
        return artifacts
    
    async def _schedule_dependent_tasks(self, completed_task_id: str):
        """Schedule tasks that depend on completed task"""
        for task_id, task in list(self.tasks.items()):
            if completed_task_id in task.dependencies and await self._dependencies_met(task_id):
                await self._schedule_task_execution(task_id)
    
    def _update_performance_metrics(self, task: AgentTask, success: bool):
        """Update system performance metrics"""
        self.performance_metrics["total_tasks"] += 1
        
        if success:
            self.performance_metrics["completed_tasks"] += 1
        else:
            self.performance_metrics["failed_tasks"] += 1
        
        # Update execution time average
        if task.started_at and task.completed_at:
            execution_time = (task.completed_at - task.started_at).total_seconds()
            current_avg = self.performance_metrics["average_execution_time"]
            total_tasks = self.performance_metrics["total_tasks"]
            
            self.performance_metrics["average_execution_time"] = (
                (current_avg * (total_tasks - 1) + execution_time) / total_tasks
            )
        
        # Update agent utilization
        agent_id = task.agent_id
        if agent_id not in self.performance_metrics["agent_utilization"]:
            self.performance_metrics["agent_utilization"][agent_id] = {
                "tasks_handled": 0,
                "success_rate": 0,
                "average_time": 0
            }
        
        agent_metrics = self.performance_metrics["agent_utilization"][agent_id]
        agent_metrics["tasks_handled"] += 1
        
        # Recalculate success rate for agent
        agent_tasks = [
            t for t in list(self.completed_tasks.values()) + list(self.failed_tasks.values())
            if t.agent_id == agent_id
        ]
        
        successful_agent_tasks = [t for t in agent_tasks if t.status == TaskStatus.COMPLETED]
        agent_metrics["success_rate"] = len(successful_agent_tasks) / len(agent_tasks) if agent_tasks else 0
        
        # Store execution history
        self.execution_history.append({
            "task_id": task.task_id,
            "agent_id": task.agent_id,
            "description": task.description,
            "success": success,
            "execution_time": execution_time if task.started_at and task.completed_at else 0,
            "timestamp": datetime.now().isoformat()
        })
    
    def _start_monitoring(self):
        """Start system monitoring thread"""
        def monitor():
            while self.system_status in ["ready", "running"]:
                try:
                    # Check for stuck tasks
                    current_time = datetime.now()
                    timeout_threshold = timedelta(seconds=self.config.task_timeout_seconds)
                    
                    for task_id, task in self.tasks.items():
                        if (task.status == TaskStatus.RUNNING and 
                            task.started_at and 
                            current_time - task.started_at > timeout_threshold):
                            
                            logger.warning(f"Task {task_id} timed out, cancelling")
                            task.status = TaskStatus.CANCELLED
                            task.error = "Task timeout"
                            
                            # Cancel running task
                            if task_id in self.running_tasks:
                                self.running_tasks[task_id].cancel()
                    
                    # Log system status
                    pending_count = len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING])
                    running_count = len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING])
                    
                    if pending_count > 0 or running_count > 0:
                        logger.debug(f"System status - Pending: {pending_count}, Running: {running_count}")
                    
                    # Check agent availability
                    for agent_id, capability in self.agent_capabilities.items():
                        if capability.current_load >= capability.max_concurrent_tasks:
                            logger.debug(f"Agent {agent_id} at capacity ({capability.current_load}/{capability.max_concurrent_tasks})")
                    
                except Exception as e:
                    logger.error(f"Error in monitoring thread: {e}")
                
                # Sleep for monitoring interval
                asyncio.sleep(10)
        
        monitoring_thread = threading.Thread(target=monitor, daemon=True)
        monitoring_thread.start()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "status": self.system_status,
            "smolagents_available": self.smolagents_available,
            "agents": {
                "total": len(self.agents),
                "available": len([c for c in self.agent_capabilities.values() if c.availability]),
                "busy": len([c for c in self.agent_capabilities.values() if c.current_load > 0])
            },
            "tasks": {
                "pending": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
                "running": len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks)
            },
            "performance": self.performance_metrics,
            "config": asdict(self.config)
        }
    
    async def wait_for_completion(self, task_ids: List[str] = None, timeout: int = None) -> Dict[str, Dict[str, Any]]:
        """Wait for specific tasks or all tasks to complete"""
        if task_ids is None:
            # Wait for all pending and running tasks
            task_ids = list(self.tasks.keys())
        
        start_time = datetime.now()
        results = {}
        
        while task_ids:
            if timeout and (datetime.now() - start_time).total_seconds() > timeout:
                break
            
            completed_in_batch = []
            for task_id in task_ids:
                if task_id in self.completed_tasks:
                    results[task_id] = {
                        "status": "completed",
                        "result": self.completed_tasks[task_id].result
                    }
                    completed_in_batch.append(task_id)
                elif task_id in self.failed_tasks:
                    results[task_id] = {
                        "status": "failed",
                        "error": self.failed_tasks[task_id].error
                    }
                    completed_in_batch.append(task_id)
            
            # Remove completed tasks from waiting list
            for task_id in completed_in_batch:
                task_ids.remove(task_id)
            
            if task_ids:
                await asyncio.sleep(1)  # Poll every second
        
        # Handle remaining tasks (timed out)
        for task_id in task_ids:
            results[task_id] = {
                "status": "timeout",
                "error": "Task did not complete within timeout"
            }
        
        return results
    
    async def shutdown(self):
        """Gracefully shutdown multi-agent system"""
        logger.info("Shutting down multi-agent system")
        
        self.system_status = "shutting_down"
        
        # Cancel running tasks
        for task_id, running_task in self.running_tasks.items():
            running_task.cancel()
        
        # Wait for tasks to complete or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.running_tasks.values(), return_exceptions=True),
                timeout=30
            )
        except asyncio.TimeoutError:
            logger.warning("Some tasks did not complete during shutdown")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.system_status = "shutdown"
        logger.info("Multi-agent system shutdown complete")


# Global instance
multi_agent_system = MONKMultiAgentSystem()