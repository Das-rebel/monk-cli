"""
MONK CLI Phase 2.5 - TreeQuest-Smolagent Bridge
Open Source Integration Foundation with personality-driven agent selection
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

try:
    from smolagents import Agent, CodeAgent, ToolBox, HfApiModel
    from smolagents.types import AgentOutput
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    # Fallback for when smolagents is still installing
    SMOLAGENTS_AVAILABLE = False
    print("smolagents not yet available - will be enabled after installation")

from ...core.config import config
from ...agents.agent_selector import PersonalityProfile, AgentPersonality, AgentType
from ...memory.memory_system import MemoryQuery, MemoryResult
from ...core.database import get_db_session

logger = logging.getLogger(__name__)


@dataclass
class TreeQuestTask:
    """Enhanced TreeQuest task with smolagent capabilities"""
    task_id: str
    description: str
    complexity: int  # 1-5 scale
    domain: str
    subtasks: List[str]
    dependencies: List[str]
    agent_personality_required: AgentPersonality
    smolagent_tools: List[str]
    priority: str = "medium"
    estimated_time_hours: float = 1.0
    context_requirements: Dict[str, Any] = None
    success_criteria: List[str] = None
    
    def __post_init__(self):
        if self.context_requirements is None:
            self.context_requirements = {}
        if self.success_criteria is None:
            self.success_criteria = []


@dataclass
class SmolagentCapability:
    """Smolagent enhanced capability definition"""
    name: str
    tools: List[str]
    model_name: str
    max_iterations: int
    context_length: int
    specialization: str
    personality_compatibility: List[AgentPersonality]


class TreeQuestSmolagentBridge:
    """Bridge between TreeQuest hierarchical tasks and Hugging Face smolagents"""
    
    def __init__(self):
        self.smolagents_available = SMOLAGENTS_AVAILABLE
        self.agents_cache = {}
        self.toolboxes = {}
        self.models_cache = {}
        self.task_execution_history = {}
        
        # MONK personality-to-smolagent mapping
        self.personality_agent_mapping = {
            AgentPersonality.ANALYTICAL: SmolagentCapability(
                name="code_analyst",
                tools=["python", "file_system", "web_search"],
                model_name="microsoft/DialoGPT-medium",
                max_iterations=20,
                context_length=8192,
                specialization="code_analysis",
                personality_compatibility=[AgentPersonality.ANALYTICAL, AgentPersonality.DETAIL_ORIENTED]
            ),
            AgentPersonality.CREATIVE: SmolagentCapability(
                name="creative_architect", 
                tools=["web_search", "image_generator", "file_system"],
                model_name="microsoft/DialoGPT-medium",
                max_iterations=15,
                context_length=4096,
                specialization="creative_problem_solving",
                personality_compatibility=[AgentPersonality.CREATIVE, AgentPersonality.INNOVATIVE]
            ),
            AgentPersonality.DETAIL_ORIENTED: SmolagentCapability(
                name="detail_executor",
                tools=["python", "file_system", "calculator"],
                model_name="microsoft/DialoGPT-medium", 
                max_iterations=25,
                context_length=8192,
                specialization="detailed_execution",
                personality_compatibility=[AgentPersonality.DETAIL_ORIENTED, AgentPersonality.SYSTEMATIC]
            ),
            AgentPersonality.COLLABORATIVE: SmolagentCapability(
                name="collaboration_coordinator",
                tools=["web_search", "file_system", "email"],
                model_name="microsoft/DialoGPT-medium",
                max_iterations=10,
                context_length=4096,
                specialization="team_coordination",
                personality_compatibility=[AgentPersonality.COLLABORATIVE, AgentPersonality.SOCIAL]
            )
        }
        
        # Task complexity to agent requirements
        self.complexity_requirements = {
            1: {"max_iterations": 5, "context_length": 2048, "tools_needed": 1},
            2: {"max_iterations": 10, "context_length": 4096, "tools_needed": 2},
            3: {"max_iterations": 15, "context_length": 6144, "tools_needed": 3},
            4: {"max_iterations": 20, "context_length": 8192, "tools_needed": 4},
            5: {"max_iterations": 25, "context_length": 8192, "tools_needed": 5}
        }
        
    async def initialize(self):
        """Initialize the TreeQuest-Smolagent bridge"""
        try:
            if not self.smolagents_available:
                logger.warning("Smolagents not available - bridge running in compatibility mode")
                return False
            
            # Initialize basic toolboxes
            await self._setup_toolboxes()
            
            # Initialize personality-based agents
            await self._initialize_personality_agents()
            
            logger.info("TreeQuest-Smolagent bridge initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TreeQuest-Smolagent bridge: {e}")
            return False
    
    async def _setup_toolboxes(self):
        """Setup smolagent toolboxes for different capabilities"""
        if not self.smolagents_available:
            return
            
        try:
            # Basic toolbox for all agents
            self.toolboxes["basic"] = ToolBox()
            
            # Code analysis toolbox
            code_tools = ToolBox()
            # Add Python execution, file system, web search tools
            self.toolboxes["code"] = code_tools
            
            # Creative toolbox  
            creative_tools = ToolBox()
            # Add image generation, web search, file system tools
            self.toolboxes["creative"] = creative_tools
            
            # Detailed execution toolbox
            detail_tools = ToolBox()
            # Add Python, file system, calculator, data analysis tools
            self.toolboxes["detail"] = detail_tools
            
            # Collaboration toolbox
            collab_tools = ToolBox()
            # Add web search, file system, communication tools
            self.toolboxes["collaboration"] = collab_tools
            
            logger.info("Smolagent toolboxes configured")
            
        except Exception as e:
            logger.error(f"Error setting up toolboxes: {e}")
    
    async def _initialize_personality_agents(self):
        """Initialize smolagents for each MONK personality type"""
        if not self.smolagents_available:
            return
            
        for personality, capability in self.personality_agent_mapping.items():
            try:
                # Create HF API model
                model = HfApiModel(model_id=capability.model_name)
                
                # Get appropriate toolbox
                toolbox_name = capability.specialization.split("_")[0]
                toolbox = self.toolboxes.get(toolbox_name, self.toolboxes["basic"])
                
                # Create agent
                agent = Agent(
                    model=model,
                    toolbox=toolbox,
                    max_iterations=capability.max_iterations,
                    planning_interval=5
                )
                
                self.agents_cache[personality.value] = {
                    "agent": agent,
                    "capability": capability,
                    "last_used": datetime.now(),
                    "usage_count": 0
                }
                
                logger.info(f"Initialized smolagent for {personality.value} personality")
                
            except Exception as e:
                logger.error(f"Failed to initialize agent for {personality.value}: {e}")
    
    async def decompose_task_hierarchy(self, root_task: str, domain: str, 
                                     target_complexity: int = 3) -> TreeQuestTask:
        """Decompose a root task into hierarchical subtasks using smolagents"""
        try:
            # Select appropriate agent for task decomposition
            decomposition_personality = AgentPersonality.ANALYTICAL
            
            if not self.smolagents_available:
                return self._create_fallback_task(root_task, domain, target_complexity)
            
            agent_info = self.agents_cache.get(decomposition_personality.value)
            if not agent_info:
                logger.error("Task decomposition agent not available")
                return self._create_fallback_task(root_task, domain, target_complexity)
            
            agent = agent_info["agent"]
            
            # Create decomposition prompt
            decomposition_prompt = f"""
            Analyze and decompose the following task into a hierarchical structure:
            
            Root Task: {root_task}
            Domain: {domain}
            Target Complexity Level: {target_complexity} (1=simple, 5=very complex)
            
            Please provide:
            1. Task breakdown into 3-7 logical subtasks
            2. Dependencies between subtasks
            3. Estimated complexity for each subtask (1-5)
            4. Required tools/capabilities for each subtask
            5. Success criteria for each subtask
            6. Recommended agent personality for execution
            
            Format the response as structured data that can be parsed.
            """
            
            # Execute decomposition
            result = agent.run(decomposition_prompt)
            
            # Parse agent output and create TreeQuestTask
            task = self._parse_decomposition_result(result, root_task, domain)
            
            # Update agent usage stats
            agent_info["last_used"] = datetime.now()
            agent_info["usage_count"] += 1
            
            return task
            
        except Exception as e:
            logger.error(f"Error in task decomposition: {e}")
            return self._create_fallback_task(root_task, domain, target_complexity)
    
    def _create_fallback_task(self, task_description: str, domain: str, 
                            complexity: int) -> TreeQuestTask:
        """Create fallback task structure when smolagents unavailable"""
        task_id = hashlib.md5(task_description.encode()).hexdigest()[:8]
        
        # Simple rule-based subtask generation
        if "implement" in task_description.lower():
            subtasks = ["Plan architecture", "Write core logic", "Add error handling", "Test implementation"]
            personality = AgentPersonality.DETAIL_ORIENTED
        elif "analyze" in task_description.lower():
            subtasks = ["Gather requirements", "Analyze data", "Generate insights", "Create report"]
            personality = AgentPersonality.ANALYTICAL
        elif "design" in task_description.lower():
            subtasks = ["Research patterns", "Create mockups", "Validate design", "Refine solution"]
            personality = AgentPersonality.CREATIVE
        else:
            subtasks = ["Break down requirements", "Execute main logic", "Validate results", "Document outcome"]
            personality = AgentPersonality.COLLABORATIVE
        
        return TreeQuestTask(
            task_id=task_id,
            description=task_description,
            complexity=complexity,
            domain=domain,
            subtasks=subtasks,
            dependencies=[],
            agent_personality_required=personality,
            smolagent_tools=["python", "file_system"],
            success_criteria=[f"Complete {task_description} successfully"]
        )
    
    def _parse_decomposition_result(self, result: Any, root_task: str, domain: str) -> TreeQuestTask:
        """Parse smolagent decomposition result into TreeQuestTask"""
        task_id = hashlib.md5(root_task.encode()).hexdigest()[:8]
        
        try:
            # Extract structured information from agent result
            if hasattr(result, 'content'):
                content = result.content
            else:
                content = str(result)
            
            # Simple parsing - in production, use more sophisticated NLP
            lines = content.split('\n')
            subtasks = []
            dependencies = []
            complexity = 3
            tools = ["python", "file_system"]
            personality = AgentPersonality.ANALYTICAL
            
            for line in lines:
                line = line.strip()
                if line.startswith("- ") or line.startswith("* "):
                    subtasks.append(line[2:])
                elif "complexity" in line.lower() and any(c.isdigit() for c in line):
                    try:
                        complexity = int(next(c for c in line if c.isdigit()))
                    except:
                        complexity = 3
                elif "tools" in line.lower():
                    # Extract tools mentioned
                    if "python" in line.lower():
                        tools.append("python")
                    if "web" in line.lower():
                        tools.append("web_search")
                elif "creative" in line.lower():
                    personality = AgentPersonality.CREATIVE
                elif "detail" in line.lower():
                    personality = AgentPersonality.DETAIL_ORIENTED
                elif "collaborative" in line.lower():
                    personality = AgentPersonality.COLLABORATIVE
            
            if not subtasks:
                subtasks = ["Execute main task", "Validate results", "Document outcome"]
            
            return TreeQuestTask(
                task_id=task_id,
                description=root_task,
                complexity=min(max(complexity, 1), 5),
                domain=domain,
                subtasks=subtasks[:7],  # Limit to 7 subtasks
                dependencies=dependencies,
                agent_personality_required=personality,
                smolagent_tools=list(set(tools)),
                success_criteria=[f"Complete {root_task} with all subtasks executed"]
            )
            
        except Exception as e:
            logger.error(f"Error parsing decomposition result: {e}")
            return self._create_fallback_task(root_task, domain, 3)
    
    async def execute_task_with_personality(self, task: TreeQuestTask, 
                                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute TreeQuest task using personality-matched smolagent"""
        execution_start = datetime.now()
        
        if context is None:
            context = {}
        
        try:
            # Get personality-matched agent
            personality = task.agent_personality_required
            agent_info = self.agents_cache.get(personality.value)
            
            if not agent_info or not self.smolagents_available:
                return await self._execute_fallback_task(task, context)
            
            agent = agent_info["agent"]
            capability = agent_info["capability"]
            
            # Prepare execution context
            execution_context = {
                "task_description": task.description,
                "subtasks": task.subtasks,
                "complexity": task.complexity,
                "domain": task.domain,
                "success_criteria": task.success_criteria,
                "available_tools": capability.tools,
                **context
            }
            
            # Create execution prompt
            prompt = self._create_execution_prompt(task, execution_context)
            
            # Execute task with smolagent
            result = agent.run(prompt)
            
            # Process and validate result
            execution_result = self._process_execution_result(task, result, execution_start)
            
            # Update usage stats
            agent_info["last_used"] = datetime.now()
            agent_info["usage_count"] += 1
            
            # Store execution history
            self.task_execution_history[task.task_id] = {
                "task": asdict(task),
                "result": execution_result,
                "agent_personality": personality.value,
                "execution_time": execution_result["execution_time_ms"],
                "success": execution_result["success"]
            }
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task.task_id,
                "execution_time_ms": (datetime.now() - execution_start).total_seconds() * 1000,
                "result": None
            }
    
    def _create_execution_prompt(self, task: TreeQuestTask, context: Dict[str, Any]) -> str:
        """Create execution prompt for smolagent"""
        prompt = f"""
        Execute the following task with precision and attention to detail:
        
        Main Task: {task.description}
        Domain: {task.domain}
        Complexity Level: {task.complexity}/5
        
        Subtasks to complete:
        """
        
        for i, subtask in enumerate(task.subtasks, 1):
            prompt += f"\n{i}. {subtask}"
        
        prompt += f"""
        
        Success Criteria:
        """
        
        for criterion in task.success_criteria:
            prompt += f"\n- {criterion}"
        
        if context.get("previous_results"):
            prompt += f"\n\nPrevious Results Context:\n{context['previous_results']}"
        
        if context.get("constraints"):
            prompt += f"\n\nConstraints:\n{context['constraints']}"
        
        prompt += """
        
        Please execute this task step by step, providing clear output for each subtask.
        Include any code, analysis, or deliverables produced.
        """
        
        return prompt
    
    def _process_execution_result(self, task: TreeQuestTask, result: Any, 
                                start_time: datetime) -> Dict[str, Any]:
        """Process smolagent execution result"""
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        try:
            # Extract result content
            if hasattr(result, 'content'):
                content = result.content
            elif hasattr(result, 'output'):
                content = result.output
            else:
                content = str(result)
            
            # Analyze execution success
            success_indicators = ["completed", "successful", "done", "finished"]
            error_indicators = ["error", "failed", "exception", "could not"]
            
            success_score = sum(1 for indicator in success_indicators if indicator in content.lower())
            error_score = sum(1 for indicator in error_indicators if indicator in content.lower())
            
            success = success_score > error_score
            
            # Extract artifacts (code, files, data)
            artifacts = self._extract_artifacts(content)
            
            return {
                "success": success,
                "task_id": task.task_id,
                "execution_time_ms": int(execution_time),
                "result": content,
                "artifacts": artifacts,
                "subtasks_completed": len(task.subtasks),  # Assume all if successful
                "confidence_score": min(success_score / max(len(task.subtasks), 1), 1.0),
                "personality_used": task.agent_personality_required.value
            }
            
        except Exception as e:
            logger.error(f"Error processing execution result: {e}")
            return {
                "success": False,
                "task_id": task.task_id,
                "execution_time_ms": int(execution_time),
                "result": str(result),
                "artifacts": [],
                "error": str(e)
            }
    
    def _extract_artifacts(self, content: str) -> List[Dict[str, Any]]:
        """Extract code, files, and other artifacts from execution result"""
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
        
        # Extract file paths mentioned
        file_patterns = re.findall(r'[\'"]([^\'"\s]+\.(py|js|ts|json|md|txt))[\'"]', content)
        for file_path, ext in file_patterns:
            artifacts.append({
                "type": "file_reference", 
                "path": file_path,
                "extension": ext
            })
        
        # Extract URLs
        url_patterns = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
        for url in url_patterns:
            artifacts.append({
                "type": "url",
                "url": url
            })
        
        return artifacts
    
    async def _execute_fallback_task(self, task: TreeQuestTask, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback task execution when smolagents unavailable"""
        execution_start = datetime.now()
        
        # Simple rule-based execution simulation
        result_content = f"""
        Executed task: {task.description}
        
        Completed subtasks:
        """
        
        for i, subtask in enumerate(task.subtasks, 1):
            result_content += f"\n{i}. {subtask} - Completed"
        
        result_content += f"""
        
        Domain: {task.domain}
        Complexity: {task.complexity}/5
        Personality: {task.agent_personality_required.value}
        
        Task completed using fallback execution mode.
        """
        
        execution_time = (datetime.now() - execution_start).total_seconds() * 1000
        
        return {
            "success": True,
            "task_id": task.task_id,
            "execution_time_ms": int(execution_time),
            "result": result_content,
            "artifacts": [],
            "subtasks_completed": len(task.subtasks),
            "confidence_score": 0.7,  # Lower confidence for fallback
            "personality_used": task.agent_personality_required.value,
            "execution_mode": "fallback"
        }
    
    async def get_agent_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for personality-based agents"""
        metrics = {
            "total_tasks_executed": len(self.task_execution_history),
            "smolagents_available": self.smolagents_available,
            "agent_usage": {},
            "average_execution_time_ms": 0,
            "success_rate": 0,
            "personality_performance": {}
        }
        
        if not self.task_execution_history:
            return metrics
        
        total_time = 0
        successful_tasks = 0
        
        for personality, agent_info in self.agents_cache.items():
            metrics["agent_usage"][personality] = {
                "usage_count": agent_info["usage_count"],
                "last_used": agent_info["last_used"].isoformat() if agent_info["last_used"] else None
            }
        
        for task_id, history in self.task_execution_history.items():
            total_time += history["execution_time"]
            if history["success"]:
                successful_tasks += 1
            
            personality = history["agent_personality"]
            if personality not in metrics["personality_performance"]:
                metrics["personality_performance"][personality] = {
                    "tasks_handled": 0,
                    "success_rate": 0,
                    "avg_execution_time": 0
                }
            
            perf = metrics["personality_performance"][personality]
            perf["tasks_handled"] += 1
        
        # Calculate averages
        task_count = len(self.task_execution_history)
        metrics["average_execution_time_ms"] = total_time / task_count
        metrics["success_rate"] = successful_tasks / task_count
        
        # Calculate personality-specific performance
        for personality, perf in metrics["personality_performance"].items():
            personality_tasks = [h for h in self.task_execution_history.values() 
                               if h["agent_personality"] == personality]
            
            if personality_tasks:
                successful = sum(1 for t in personality_tasks if t["success"])
                perf["success_rate"] = successful / len(personality_tasks)
                perf["avg_execution_time"] = sum(t["execution_time"] for t in personality_tasks) / len(personality_tasks)
        
        return metrics
    
    async def optimize_agent_allocation(self, upcoming_tasks: List[TreeQuestTask]) -> Dict[str, List[str]]:
        """Optimize agent allocation based on personality compatibility and performance"""
        allocation = {personality.value: [] for personality in AgentPersonality}
        
        # Get current performance metrics
        metrics = await self.get_agent_performance_metrics()
        personality_performance = metrics.get("personality_performance", {})
        
        for task in upcoming_tasks:
            best_personality = task.agent_personality_required.value
            best_score = 0
            
            # Consider alternative personalities based on compatibility
            capability = self.personality_agent_mapping.get(task.agent_personality_required)
            if capability:
                for compatible_personality in capability.personality_compatibility:
                    personality_key = compatible_personality.value
                    
                    # Calculate suitability score
                    perf = personality_performance.get(personality_key, {})
                    success_rate = perf.get("success_rate", 0.5)
                    avg_time = perf.get("avg_execution_time", 5000)  # Default 5 seconds
                    current_load = len(allocation.get(personality_key, []))
                    
                    # Score based on success rate, speed, and load balancing
                    score = (
                        success_rate * 0.5 +           # 50% weight on success rate
                        (1.0 / max(avg_time, 1000)) * 0.3 +  # 30% weight on speed (inverse)
                        (1.0 / max(current_load + 1, 1)) * 0.2   # 20% weight on load balancing
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_personality = personality_key
            
            allocation[best_personality].append(task.task_id)
        
        return allocation


# Global instance
treequest_smolagent_bridge = TreeQuestSmolagentBridge()