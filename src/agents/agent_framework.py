"""
MONK CLI Agent Framework
Core agent system with personality-driven specialization
"""
import asyncio
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..core.config import config
from ..core.database import get_db_session, cache
from ..core.models import Agent, AgentStack, AgentExecution, User

logger = logging.getLogger(__name__)


class PersonalityTrait(Enum):
    """Big Five personality traits plus AI-specific dimensions"""
    CONSCIENTIOUSNESS = "conscientiousness"
    OPENNESS = "openness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"
    # AI-specific traits
    CREATIVITY = "creativity"
    ANALYTICAL_THINKING = "analytical_thinking"
    RISK_TOLERANCE = "risk_tolerance"


@dataclass
class PersonalityProfile:
    """Agent personality profile with Big Five + AI traits"""
    conscientiousness: float = 0.5  # 0.0 to 1.0
    openness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    
    # AI-specific traits
    creativity: float = 0.5
    analytical_thinking: float = 0.5
    risk_tolerance: float = 0.5
    
    def __post_init__(self):
        """Validate trait values"""
        for field_name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Trait {field_name} must be between 0.0 and 1.0, got {value}")
    
    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "PersonalityProfile":
        return cls(**data)
    
    def similarity(self, other: "PersonalityProfile") -> float:
        """Calculate personality similarity with another profile"""
        traits = list(self.__dict__.keys())
        differences = [abs(getattr(self, trait) - getattr(other, trait)) for trait in traits]
        return 1.0 - (sum(differences) / len(differences))
    
    def complement_score(self, other: "PersonalityProfile") -> float:
        """Calculate how well this personality complements another"""
        # High complement when traits balance each other
        complement_pairs = [
            ("conscientiousness", "creativity"),
            ("analytical_thinking", "openness"),
            ("agreeableness", "risk_tolerance"),
            ("extraversion", "neuroticism")
        ]
        
        scores = []
        for trait1, trait2 in complement_pairs:
            val1 = getattr(self, trait1)
            val2 = getattr(other, trait2)
            # High complement when one is high and other is low
            complement = abs(val1 - val2) / 1.0
            scores.append(complement)
        
        return sum(scores) / len(scores)


@dataclass
class TaskContext:
    """Context information for agent task execution"""
    user_id: str
    task_description: str
    task_type: str
    domain: str
    complexity_level: float  # 0.0 to 1.0
    urgency_level: float     # 0.0 to 1.0
    context_data: Dict[str, Any] = field(default_factory=dict)
    memory_context: List[Dict] = field(default_factory=list)
    previous_attempts: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "task_description": self.task_description,
            "task_type": self.task_type,
            "domain": self.domain,
            "complexity_level": self.complexity_level,
            "urgency_level": self.urgency_level,
            "context_data": self.context_data,
            "memory_context": self.memory_context,
            "previous_attempts": self.previous_attempts
        }


@dataclass
class AgentResponse:
    """Response from agent execution"""
    success: bool
    result: Dict[str, Any]
    execution_time_ms: int
    confidence_score: float
    tokens_used: int = 0
    memory_queries_made: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "result": self.result,
            "execution_time_ms": self.execution_time_ms,
            "confidence_score": self.confidence_score,
            "tokens_used": self.tokens_used,
            "memory_queries_made": self.memory_queries_made,
            "error_message": self.error_message
        }


class BaseAgent(ABC):
    """Base class for all MONK agents"""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        personality: PersonalityProfile,
        specializations: List[str],
        tools: List[str]
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.personality = personality
        self.specializations = specializations
        self.tools = tools
        self.is_busy = False
        self.current_task = None
        
        # Performance tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.average_execution_time = 0.0
    
    @abstractmethod
    async def execute_task(self, context: TaskContext) -> AgentResponse:
        """Execute a task with given context"""
        pass
    
    @abstractmethod
    def calculate_suitability(self, context: TaskContext) -> float:
        """Calculate how suitable this agent is for the given task (0.0 to 1.0)"""
        pass
    
    def can_handle_task(self, context: TaskContext) -> bool:
        """Check if agent can handle the given task"""
        if self.is_busy:
            return False
        
        # Check domain specialization
        if context.domain and context.domain not in self.specializations:
            return False
        
        # Check complexity vs agent capabilities
        suitability = self.calculate_suitability(context)
        return suitability >= 0.3  # Minimum threshold
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        success_rate = self.successful_executions / max(self.total_executions, 1)
        
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "is_busy": self.is_busy,
            "current_task": self.current_task,
            "total_executions": self.total_executions,
            "success_rate": success_rate,
            "average_execution_time_ms": self.average_execution_time,
            "personality": self.personality.to_dict(),
            "specializations": self.specializations,
            "tools": self.tools
        }
    
    async def _update_performance_metrics(self, response: AgentResponse):
        """Update agent performance metrics"""
        self.total_executions += 1
        if response.success:
            self.successful_executions += 1
        
        # Update average execution time
        if self.total_executions == 1:
            self.average_execution_time = response.execution_time_ms
        else:
            self.average_execution_time = (
                (self.average_execution_time * (self.total_executions - 1) + response.execution_time_ms)
                / self.total_executions
            )


class ArchitectAgent(BaseAgent):
    """Architect Agent - System design and scalability analysis"""
    
    def __init__(self):
        personality = PersonalityProfile(
            conscientiousness=0.9,
            openness=0.7,
            extraversion=0.4,
            agreeableness=0.6,
            neuroticism=0.2,
            creativity=0.6,
            analytical_thinking=0.9,
            risk_tolerance=0.3
        )
        
        super().__init__(
            agent_id="architect-001",
            name="Architect",
            description="System design, architecture analysis, and scalability planning",
            personality=personality,
            specializations=["system_design", "architecture", "scalability", "tech_debt_analysis"],
            tools=["architecture_analyzer", "dependency_mapper", "scalability_assessor"]
        )
    
    def calculate_suitability(self, context: TaskContext) -> float:
        """Calculate suitability based on task characteristics and personality fit"""
        base_score = 0.5
        
        # Domain match
        if context.domain in self.specializations:
            base_score += 0.3
        
        # Task type bonuses
        architecture_keywords = ["design", "architecture", "system", "scalability", "structure"]
        if any(keyword in context.task_description.lower() for keyword in architecture_keywords):
            base_score += 0.2
        
        # Complexity preference (high conscientiousness likes complex tasks)
        complexity_bonus = self.personality.conscientiousness * context.complexity_level * 0.2
        base_score += complexity_bonus
        
        # Analytical thinking bonus
        if context.complexity_level > 0.6:
            base_score += self.personality.analytical_thinking * 0.1
        
        return min(1.0, base_score)
    
    async def execute_task(self, context: TaskContext) -> AgentResponse:
        """Execute architecture analysis task"""
        start_time = time.time()
        self.is_busy = True
        self.current_task = context.task_description[:100] + "..."
        
        try:
            # Simulate architecture analysis with personality influence
            confidence = 0.7 + (self.personality.conscientiousness * 0.2)
            
            # Mock result - in real implementation, this would call actual AI models
            result = {
                "analysis_type": "system_architecture",
                "recommendations": [
                    "Implement microservices architecture for scalability",
                    "Use event-driven patterns for loose coupling",
                    "Consider caching strategy for performance"
                ],
                "complexity_assessment": context.complexity_level,
                "confidence": confidence,
                "agent_personality_influence": {
                    "thoroughness": self.personality.conscientiousness,
                    "innovation_level": self.personality.creativity,
                    "risk_assessment": 1.0 - self.personality.risk_tolerance
                }
            }
            
            # Simulate processing time based on complexity
            processing_time = 1.0 + (context.complexity_level * 2.0)
            await asyncio.sleep(processing_time)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            response = AgentResponse(
                success=True,
                result=result,
                execution_time_ms=execution_time,
                confidence_score=confidence,
                tokens_used=500,  # Mock value
                memory_queries_made=len(context.memory_context)
            )
            
            await self._update_performance_metrics(response)
            
            return response
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Architect agent execution failed: {e}")
            
            return AgentResponse(
                success=False,
                result={},
                execution_time_ms=execution_time,
                confidence_score=0.0,
                error_message=str(e)
            )
        
        finally:
            self.is_busy = False
            self.current_task = None


class QualityEnforcerAgent(BaseAgent):
    """Quality Enforcer Agent - Code review and testing strategy"""
    
    def __init__(self):
        personality = PersonalityProfile(
            conscientiousness=0.95,
            openness=0.4,
            extraversion=0.3,
            agreeableness=0.4,
            neuroticism=0.3,
            creativity=0.3,
            analytical_thinking=0.8,
            risk_tolerance=0.1
        )
        
        super().__init__(
            agent_id="quality-001",
            name="Quality Enforcer",
            description="Code review, testing strategy, and quality assurance",
            personality=personality,
            specializations=["code_review", "testing_strategy", "quality_assurance", "security_analysis"],
            tools=["code_quality_scanner", "test_coverage_analyzer", "security_vulnerability_detector"]
        )
    
    def calculate_suitability(self, context: TaskContext) -> float:
        """Calculate suitability for quality/testing tasks"""
        base_score = 0.5
        
        # Domain match
        if context.domain in self.specializations:
            base_score += 0.3
        
        # Quality/testing keywords
        quality_keywords = ["review", "test", "quality", "security", "bug", "validation"]
        if any(keyword in context.task_description.lower() for keyword in quality_keywords):
            base_score += 0.3
        
        # High conscientiousness loves detailed work
        base_score += self.personality.conscientiousness * 0.2
        
        # Low risk tolerance perfect for quality work
        base_score += (1.0 - self.personality.risk_tolerance) * 0.1
        
        return min(1.0, base_score)
    
    async def execute_task(self, context: TaskContext) -> AgentResponse:
        """Execute quality analysis task"""
        start_time = time.time()
        self.is_busy = True
        self.current_task = context.task_description[:100] + "..."
        
        try:
            # Quality analysis with personality influence
            confidence = 0.8 + (self.personality.conscientiousness * 0.15)
            
            result = {
                "analysis_type": "quality_review",
                "quality_score": 0.85,  # Mock score
                "issues_found": [
                    {"type": "code_style", "severity": "low", "message": "Inconsistent naming convention"},
                    {"type": "security", "severity": "medium", "message": "Potential SQL injection vulnerability"}
                ],
                "recommendations": [
                    "Add unit tests for edge cases",
                    "Implement input validation",
                    "Consider adding integration tests"
                ],
                "confidence": confidence,
                "agent_thoroughness": self.personality.conscientiousness
            }
            
            # Processing time - quality agents are thorough
            processing_time = 2.0 + (context.complexity_level * 1.5)
            await asyncio.sleep(processing_time)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            response = AgentResponse(
                success=True,
                result=result,
                execution_time_ms=execution_time,
                confidence_score=confidence,
                tokens_used=600,
                memory_queries_made=len(context.memory_context)
            )
            
            await self._update_performance_metrics(response)
            return response
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Quality Enforcer agent execution failed: {e}")
            
            return AgentResponse(
                success=False,
                result={},
                execution_time_ms=execution_time,
                confidence_score=0.0,
                error_message=str(e)
            )
        
        finally:
            self.is_busy = False
            self.current_task = None


class InnovationDriverAgent(BaseAgent):
    """Innovation Driver Agent - Emerging tech and optimization"""
    
    def __init__(self):
        personality = PersonalityProfile(
            conscientiousness=0.6,
            openness=0.95,
            extraversion=0.7,
            agreeableness=0.6,
            neuroticism=0.3,
            creativity=0.9,
            analytical_thinking=0.7,
            risk_tolerance=0.8
        )
        
        super().__init__(
            agent_id="innovation-001",
            name="Innovation Driver",
            description="Emerging technology analysis, optimization, and creative solutions",
            personality=personality,
            specializations=["emerging_tech", "optimization", "creative_solutions", "performance_tuning"],
            tools=["technology_trend_analyzer", "performance_optimizer", "solution_generator"]
        )
    
    def calculate_suitability(self, context: TaskContext) -> float:
        """Calculate suitability for innovation/optimization tasks"""
        base_score = 0.5
        
        # Domain match
        if context.domain in self.specializations:
            base_score += 0.3
        
        # Innovation keywords
        innovation_keywords = ["optimize", "improve", "innovative", "creative", "new", "experimental"]
        if any(keyword in context.task_description.lower() for keyword in innovation_keywords):
            base_score += 0.3
        
        # High openness and creativity boost
        base_score += (self.personality.openness + self.personality.creativity) * 0.15
        
        # Risk tolerance helps with experimental solutions
        base_score += self.personality.risk_tolerance * 0.1
        
        return min(1.0, base_score)
    
    async def execute_task(self, context: TaskContext) -> AgentResponse:
        """Execute innovation/optimization task"""
        start_time = time.time()
        self.is_busy = True
        self.current_task = context.task_description[:100] + "..."
        
        try:
            # Innovation analysis with personality influence
            confidence = 0.6 + (self.personality.creativity * 0.3)
            
            result = {
                "analysis_type": "innovation_optimization",
                "creative_solutions": [
                    "Implement AI-driven caching strategy",
                    "Use machine learning for predictive optimization",
                    "Consider blockchain for data integrity"
                ],
                "optimization_opportunities": [
                    {"area": "algorithm_efficiency", "potential_gain": "30%"},
                    {"area": "memory_usage", "potential_gain": "20%"}
                ],
                "emerging_tech_suggestions": [
                    "WebAssembly for performance",
                    "Edge computing for latency"
                ],
                "confidence": confidence,
                "innovation_level": self.personality.creativity
            }
            
            # Faster execution due to high creativity and risk tolerance
            processing_time = 0.8 + (context.complexity_level * 1.0)
            await asyncio.sleep(processing_time)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            response = AgentResponse(
                success=True,
                result=result,
                execution_time_ms=execution_time,
                confidence_score=confidence,
                tokens_used=450,
                memory_queries_made=len(context.memory_context)
            )
            
            await self._update_performance_metrics(response)
            return response
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Innovation Driver agent execution failed: {e}")
            
            return AgentResponse(
                success=False,
                result={},
                execution_time_ms=execution_time,
                confidence_score=0.0,
                error_message=str(e)
            )
        
        finally:
            self.is_busy = False
            self.current_task = None


class IntegrationSpecialistAgent(BaseAgent):
    """Integration Specialist Agent - API integration and deployment"""
    
    def __init__(self):
        personality = PersonalityProfile(
            conscientiousness=0.7,
            openness=0.6,
            extraversion=0.5,
            agreeableness=0.9,
            neuroticism=0.2,
            creativity=0.5,
            analytical_thinking=0.75,
            risk_tolerance=0.4
        )
        
        super().__init__(
            agent_id="integration-001",
            name="Integration Specialist",
            description="API integration, service orchestration, and deployment",
            personality=personality,
            specializations=["api_integration", "service_orchestration", "deployment", "devops"],
            tools=["api_compatibility_checker", "integration_tester", "deployment_orchestrator"]
        )
    
    def calculate_suitability(self, context: TaskContext) -> float:
        """Calculate suitability for integration/deployment tasks"""
        base_score = 0.5
        
        # Domain match
        if context.domain in self.specializations:
            base_score += 0.3
        
        # Integration keywords
        integration_keywords = ["integrate", "deploy", "api", "service", "orchestrate", "connect"]
        if any(keyword in context.task_description.lower() for keyword in integration_keywords):
            base_score += 0.3
        
        # High agreeableness helps with service integration
        base_score += self.personality.agreeableness * 0.15
        
        # Analytical thinking for complex integrations
        base_score += self.personality.analytical_thinking * 0.1
        
        return min(1.0, base_score)
    
    async def execute_task(self, context: TaskContext) -> AgentResponse:
        """Execute integration/deployment task"""
        start_time = time.time()
        self.is_busy = True
        self.current_task = context.task_description[:100] + "..."
        
        try:
            # Integration analysis with personality influence
            confidence = 0.75 + (self.personality.agreeableness * 0.2)
            
            result = {
                "analysis_type": "integration_deployment",
                "integration_plan": [
                    "Assess API compatibility",
                    "Design integration architecture",
                    "Implement connection layer",
                    "Test integration points",
                    "Deploy with monitoring"
                ],
                "compatibility_assessment": "98%",
                "deployment_strategy": "blue_green_deployment",
                "risk_factors": [
                    {"factor": "API version changes", "mitigation": "Version pinning"},
                    {"factor": "Network latency", "mitigation": "Circuit breakers"}
                ],
                "confidence": confidence,
                "collaboration_approach": self.personality.agreeableness
            }
            
            # Processing time based on integration complexity
            processing_time = 1.5 + (context.complexity_level * 1.2)
            await asyncio.sleep(processing_time)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            response = AgentResponse(
                success=True,
                result=result,
                execution_time_ms=execution_time,
                confidence_score=confidence,
                tokens_used=550,
                memory_queries_made=len(context.memory_context)
            )
            
            await self._update_performance_metrics(response)
            return response
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Integration Specialist agent execution failed: {e}")
            
            return AgentResponse(
                success=False,
                result={},
                execution_time_ms=execution_time,
                confidence_score=0.0,
                error_message=str(e)
            )
        
        finally:
            self.is_busy = False
            self.current_task = None


# Agent registry for easy access
AGENT_REGISTRY = {
    "architect": ArchitectAgent,
    "quality_enforcer": QualityEnforcerAgent,
    "innovation_driver": InnovationDriverAgent,
    "integration_specialist": IntegrationSpecialistAgent
}


def create_agent(agent_type: str) -> BaseAgent:
    """Factory function to create agent instances"""
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return AGENT_REGISTRY[agent_type]()