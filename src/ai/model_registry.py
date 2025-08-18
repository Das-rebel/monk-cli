"""
Model Registry - Manages LLM providers and model selection for TreeQuest
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ModelRole(Enum):
    """Different roles for models in TreeQuest"""
    PLANNER = "planner"      # High-level planning and strategy
    ANALYZER = "analyzer"     # Data analysis and insight generation
    CODER = "coder"          # Code generation and analysis
    CRITIC = "critic"        # Evaluation and quality assessment
    SIMULATOR = "simulator"  # Simulation and rollout evaluation
    SYNTHESIZER = "synthesizer"  # Combining and summarizing insights
    EXECUTOR = "executor"    # Implementation and execution planning

class ModelObjective(Enum):
    """Different optimization objectives"""
    QUALITY = "quality"      # Prioritize output quality
    LATENCY = "latency"      # Prioritize speed
    COST = "cost"           # Prioritize cost efficiency

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    provider: str
    api_key_env: str
    cost_per_1k_tokens_input: float
    cost_per_1k_tokens_output: float
    max_tokens: int
    capabilities: List[ModelRole]
    latency_ms: float
    quality_score: float  # 0.0 to 1.0
    
    @property
    def is_available(self) -> bool:
        """Check if model is available (has API key)"""
        return bool(os.getenv(self.api_key_env))

class ModelRegistry:
    """
    Registry for managing different LLM models and providers
    """
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize with default model configurations"""
        default_models = [
            ModelConfig(
                name="gpt-4o",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                cost_per_1k_tokens_input=0.005,
                cost_per_1k_tokens_output=0.015,
                max_tokens=128000,
                capabilities=[ModelRole.PLANNER, ModelRole.ANALYZER, ModelRole.CODER, ModelRole.CRITIC, ModelRole.SYNTHESIZER, ModelRole.EXECUTOR, ModelRole.SIMULATOR],
                latency_ms=2000,
                quality_score=0.95
            ),
            ModelConfig(
                name="gpt-4o-mini",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                cost_per_1k_tokens_input=0.00015,
                cost_per_1k_tokens_output=0.0006,
                max_tokens=128000,
                capabilities=[ModelRole.PLANNER, ModelRole.ANALYZER, ModelRole.CODER, ModelRole.CRITIC, ModelRole.EXECUTOR, ModelRole.SIMULATOR],
                latency_ms=1000,
                quality_score=0.85
            ),
            ModelConfig(
                name="claude-3-opus",
                provider="anthropic",
                api_key_env="ANTHROPIC_API_KEY",
                cost_per_1k_tokens_input=0.015,
                cost_per_1k_tokens_output=0.075,
                max_tokens=200000,
                capabilities=[ModelRole.PLANNER, ModelRole.ANALYZER, ModelRole.CODER, ModelRole.CRITIC, ModelRole.SYNTHESIZER, ModelRole.EXECUTOR, ModelRole.SIMULATOR],
                latency_ms=3000,
                quality_score=0.98
            ),
            ModelConfig(
                name="claude-3-sonnet",
                provider="anthropic",
                api_key_env="ANTHROPIC_API_KEY",
                cost_per_1k_tokens_input=0.003,
                cost_per_1k_tokens_output=0.015,
                max_tokens=200000,
                capabilities=[ModelRole.PLANNER, ModelRole.ANALYZER, ModelRole.CODER, ModelRole.CRITIC, ModelRole.EXECUTOR, ModelRole.SIMULATOR],
                latency_ms=1500,
                quality_score=0.90
            ),
            ModelConfig(
                name="mistral-large",
                provider="mistral",
                api_key_env="MISTRAL_API_KEY",
                cost_per_1k_tokens_input=0.007,
                cost_per_1k_tokens_output=0.024,
                max_tokens=32768,
                capabilities=[ModelRole.PLANNER, ModelRole.ANALYZER, ModelRole.CODER, ModelRole.CRITIC, ModelRole.EXECUTOR, ModelRole.SIMULATOR],
                latency_ms=2500,
                quality_score=0.88
            ),
            ModelConfig(
                name="gemini-pro",
                provider="google",
                api_key_env="GOOGLE_API_KEY",
                cost_per_1k_tokens_input=0.0005,
                cost_per_1k_tokens_output=0.0015,
                max_tokens=1000000,
                capabilities=[ModelRole.PLANNER, ModelRole.ANALYZER, ModelRole.CODER, ModelRole.CRITIC, ModelRole.EXECUTOR, ModelRole.SIMULATOR],
                latency_ms=1800,
                quality_score=0.87
            )
        ]
        
        for model in default_models:
            self.register_model(model)
    
    def register_model(self, model: ModelConfig):
        """Register a new model"""
        self.models[model.name] = model
        logger.info(f"Registered model: {model.name} ({model.provider})")
    
    def get_available_models(self) -> List[ModelConfig]:
        """Get list of available models (with API keys)"""
        return [model for model in self.models.values() if model.is_available]
    
    def pick(self, role: ModelRole, objective: ModelObjective = ModelObjective.QUALITY) -> str:
        """
        Pick the best model for a given role and objective
        
        Args:
            role: The role the model needs to fulfill
            objective: Optimization objective (quality, latency, cost)
            
        Returns:
            Model name string
        """
        available_models = [
            model for model in self.get_available_models()
            if role in model.capabilities
        ]
        
        if not available_models:
            logger.warning(f"No available models for role {role}")
            return "gpt-4o-mini"  # Fallback
        
        # Score models based on objective
        scored_models = []
        for model in available_models:
            if objective == ModelObjective.QUALITY:
                score = model.quality_score
            elif objective == ModelObjective.LATENCY:
                score = 1.0 / (model.latency_ms / 1000.0)  # Higher score for lower latency
            elif objective == ModelObjective.COST:
                score = 1.0 / (model.cost_per_1k_tokens_input + model.cost_per_1k_tokens_output)
            else:
                score = model.quality_score
            
            scored_models.append((model, score))
        
        # Sort by score (descending) and return best
        scored_models.sort(key=lambda x: x[1], reverse=True)
        best_model = scored_models[0][0]
        
        logger.debug(f"Selected model {best_model.name} for role {role} with objective {objective}")
        return best_model.name
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return self.models.get(model_name)
    
    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a model call"""
        model = self.get_model_config(model_name)
        if not model:
            return 0.0
        
        input_cost = (input_tokens / 1000.0) * model.cost_per_1k_tokens_input
        output_cost = (output_tokens / 1000.0) * model.cost_per_1k_tokens_output
        return input_cost + output_cost
    
    def get_models_by_capability(self, capability: ModelRole) -> List[ModelConfig]:
        """Get all models that support a specific capability"""
        return [
            model for model in self.models.values()
            if capability in model.capabilities and model.is_available
        ]
    
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Get cost analysis for all available models"""
        analysis = {
            "total_models": len(self.models),
            "available_models": len(self.get_available_models()),
            "models_by_provider": {},
            "cost_ranges": {
                "lowest_cost": None,
                "highest_cost": None,
                "average_cost": 0.0
            }
        }
        
        # Group by provider
        for model in self.models.values():
            provider = model.provider
            if provider not in analysis["models_by_provider"]:
                analysis["models_by_provider"][provider] = []
            analysis["models_by_provider"][provider].append({
                "name": model.name,
                "available": model.is_available,
                "cost_per_1k_input": model.cost_per_1k_tokens_input,
                "cost_per_1k_output": model.cost_per_1k_tokens_output,
                "quality": model.quality_score
            })
        
        # Calculate cost ranges
        available_models = self.get_available_models()
        if available_models:
            costs = [model.cost_per_1k_tokens_input + model.cost_per_1k_tokens_output 
                    for model in available_models]
            analysis["cost_ranges"]["lowest_cost"] = min(costs)
            analysis["cost_ranges"]["highest_cost"] = max(costs)
            analysis["cost_ranges"]["average_cost"] = sum(costs) / len(costs)
        
        return analysis
