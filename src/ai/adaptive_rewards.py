"""
Adaptive Reward Functions with Historical Learning
Dynamic reward systems that evolve based on performance outcomes
"""

import json
import time
import math
import statistics
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import logging
import numpy as np

logger = logging.getLogger(__name__)

class RewardDimension(Enum):
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    NOVELTY = "novelty"
    CONSISTENCY = "consistency"
    COST_EFFECTIVENESS = "cost_effectiveness"

@dataclass
class RewardOutcome:
    """Records the outcome of a reward decision"""
    timestamp: float
    agent_role: str
    task_type: str
    predicted_reward: float
    actual_outcome: float
    context_factors: Dict[str, Any]
    learning_applied: bool
    adjustment_magnitude: float

@dataclass
class RewardParameters:
    """Parameters for adaptive reward function"""
    base_reward: float
    quality_weight: float
    efficiency_weight: float
    novelty_weight: float
    consistency_weight: float
    cost_weight: float
    learning_rate: float
    decay_factor: float
    adaptation_threshold: float

class AdaptiveRewardSystem:
    """Adaptive reward system that learns from historical outcomes"""
    
    def __init__(self, memory_filesystem, historical_performance_tracker):
        self.memory_fs = memory_filesystem
        self.performance_tracker = historical_performance_tracker
        
        # Reward parameters for each agent role
        self.agent_reward_params = {}
        self._initialize_default_parameters()
        
        # Learning history
        self.reward_outcomes = []
        self.adaptation_history = {}
        
        # Configuration
        self.max_learning_history = 500
        self.min_samples_for_adaptation = 10
        self.performance_correlation_threshold = 0.7
        
        # Load existing learning data
        asyncio.create_task(self._load_adaptive_data())
    
    def _initialize_default_parameters(self):
        """Initialize default reward parameters for each agent"""
        default_params = {
            "planner": RewardParameters(
                base_reward=0.8,
                quality_weight=0.4,
                efficiency_weight=0.2,
                novelty_weight=0.15,
                consistency_weight=0.15,
                cost_weight=0.1,
                learning_rate=0.05,
                decay_factor=0.95,
                adaptation_threshold=0.1
            ),
            "analyzer": RewardParameters(
                base_reward=0.7,
                quality_weight=0.5,
                efficiency_weight=0.25,
                novelty_weight=0.1,
                consistency_weight=0.1,
                cost_weight=0.05,
                learning_rate=0.04,
                decay_factor=0.96,
                adaptation_threshold=0.12
            ),
            "critic": RewardParameters(
                base_reward=0.6,
                quality_weight=0.35,
                efficiency_weight=0.15,
                novelty_weight=0.2,
                consistency_weight=0.25,
                cost_weight=0.05,
                learning_rate=0.06,
                decay_factor=0.94,
                adaptation_threshold=0.08
            ),
            "synthesizer": RewardParameters(
                base_reward=0.75,
                quality_weight=0.3,
                efficiency_weight=0.2,
                novelty_weight=0.3,
                consistency_weight=0.15,
                cost_weight=0.05,
                learning_rate=0.05,
                decay_factor=0.95,
                adaptation_threshold=0.1
            ),
            "executor": RewardParameters(
                base_reward=0.65,
                quality_weight=0.25,
                efficiency_weight=0.4,
                novelty_weight=0.1,
                consistency_weight=0.2,
                cost_weight=0.05,
                learning_rate=0.04,
                decay_factor=0.97,
                adaptation_threshold=0.15
            )
        }
        
        self.agent_reward_params = default_params
    
    async def calculate_adaptive_reward(self, agent_role: str, context: Dict[str, Any], 
                                      performance_data: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, Any]]:
        """Calculate reward using adaptive parameters"""
        try:
            if agent_role not in self.agent_reward_params:
                logger.warning(f"Unknown agent role: {agent_role}")
                return 0.5, {}
            
            params = self.agent_reward_params[agent_role]
            
            # Calculate base reward components
            quality_score = self._calculate_quality_component(context, performance_data)
            efficiency_score = self._calculate_efficiency_component(context, performance_data)
            novelty_score = self._calculate_novelty_component(agent_role, context)
            consistency_score = self._calculate_consistency_component(agent_role, context)
            cost_effectiveness = self._calculate_cost_effectiveness_component(context, performance_data)
            
            # Apply adaptive weights
            weighted_score = (
                params.quality_weight * quality_score +
                params.efficiency_weight * efficiency_score +
                params.novelty_weight * novelty_score +
                params.consistency_weight * consistency_score +
                params.cost_weight * cost_effectiveness
            )
            
            # Apply base reward and normalization
            final_reward = params.base_reward * (0.5 + 0.5 * weighted_score)
            
            # Apply historical learning adjustments
            historical_adjustment = await self._get_historical_adjustment(agent_role, context)
            final_reward = final_reward * (1.0 + historical_adjustment)
            
            # Ensure reward is in valid range
            final_reward = max(0.0, min(1.0, final_reward))
            
            # Record prediction for later learning
            reward_context = {
                'agent_role': agent_role,
                'task_type': context.get('task_type', 'unknown'),
                'context_hash': self._hash_context(context),
                'component_scores': {
                    'quality': quality_score,
                    'efficiency': efficiency_score,
                    'novelty': novelty_score,
                    'consistency': consistency_score,
                    'cost_effectiveness': cost_effectiveness
                },
                'parameters_used': asdict(params),
                'historical_adjustment': historical_adjustment
            }
            
            return final_reward, reward_context
            
        except Exception as e:
            logger.error(f"Error calculating adaptive reward: {e}")
            return 0.5, {}
    
    async def record_outcome(self, predicted_reward: float, actual_outcome: float, 
                           context: Dict[str, Any]):
        """Record actual outcome for learning"""
        try:
            outcome = RewardOutcome(
                timestamp=time.time(),
                agent_role=context.get('agent_role', 'unknown'),
                task_type=context.get('task_type', 'unknown'),
                predicted_reward=predicted_reward,
                actual_outcome=actual_outcome,
                context_factors=context,
                learning_applied=False,
                adjustment_magnitude=0.0
            )
            
            self.reward_outcomes.append(outcome)
            
            # Keep only recent outcomes
            if len(self.reward_outcomes) > self.max_learning_history:
                self.reward_outcomes = self.reward_outcomes[-self.max_learning_history:]
            
            # Trigger adaptation if conditions are met
            await self._maybe_adapt_parameters(outcome)
            
            # Store outcome in memory filesystem
            await self._store_outcome_to_memory(outcome)
            
        except Exception as e:
            logger.error(f"Error recording outcome: {e}")
    
    async def _maybe_adapt_parameters(self, latest_outcome: RewardOutcome):
        """Adapt parameters if learning conditions are met"""
        try:
            agent_role = latest_outcome.agent_role
            task_type = latest_outcome.task_type
            
            # Get recent outcomes for this agent/task combination
            relevant_outcomes = [
                outcome for outcome in self.reward_outcomes
                if (outcome.agent_role == agent_role and 
                    outcome.task_type == task_type)
            ]
            
            if len(relevant_outcomes) < self.min_samples_for_adaptation:
                return
            
            # Calculate prediction accuracy
            prediction_errors = [
                abs(outcome.predicted_reward - outcome.actual_outcome)
                for outcome in relevant_outcomes
            ]
            
            avg_error = statistics.mean(prediction_errors)
            error_trend = self._calculate_error_trend(relevant_outcomes)
            
            # Adapt if error is significant or trend is negative
            if (avg_error > self.agent_reward_params[agent_role].adaptation_threshold or
                error_trend > 0.05):
                
                await self._adapt_agent_parameters(agent_role, task_type, relevant_outcomes)
                
                logger.info(f"Adapted reward parameters for {agent_role}:{task_type} "
                           f"(avg_error: {avg_error:.3f}, trend: {error_trend:.3f})")
            
        except Exception as e:
            logger.error(f"Error in parameter adaptation: {e}")
    
    async def _adapt_agent_parameters(self, agent_role: str, task_type: str, 
                                    outcomes: List[RewardOutcome]):
        """Adapt reward parameters based on outcomes"""
        try:
            params = self.agent_reward_params[agent_role]
            learning_rate = params.learning_rate
            
            # Calculate adjustment based on prediction errors
            recent_outcomes = outcomes[-20:]  # Use last 20 outcomes
            
            # Find patterns in over/under-prediction
            over_predictions = [o for o in recent_outcomes if o.predicted_reward > o.actual_outcome]
            under_predictions = [o for o in recent_outcomes if o.predicted_reward < o.actual_outcome]
            
            # Adjust base reward if consistent bias
            if len(over_predictions) > len(under_predictions) * 1.5:
                # Consistent over-prediction: reduce base reward
                adjustment = -learning_rate * 0.1
                params.base_reward = max(0.1, params.base_reward + adjustment)
            elif len(under_predictions) > len(over_predictions) * 1.5:
                # Consistent under-prediction: increase base reward
                adjustment = learning_rate * 0.1
                params.base_reward = min(1.0, params.base_reward + adjustment)
            
            # Analyze component correlations
            await self._adapt_component_weights(params, recent_outcomes, learning_rate)
            
            # Record adaptation
            adaptation_record = {
                'timestamp': time.time(),
                'agent_role': agent_role,
                'task_type': task_type,
                'old_params': asdict(params),
                'outcomes_analyzed': len(recent_outcomes),
                'adaptation_trigger': 'prediction_error'
            }
            
            if agent_role not in self.adaptation_history:
                self.adaptation_history[agent_role] = []
            self.adaptation_history[agent_role].append(adaptation_record)
            
            # Mark outcomes as having learning applied
            for outcome in recent_outcomes:
                outcome.learning_applied = True
                outcome.adjustment_magnitude = abs(adjustment if 'adjustment' in locals() else 0.0)
            
        except Exception as e:
            logger.error(f"Error adapting agent parameters: {e}")
    
    async def _adapt_component_weights(self, params: RewardParameters, 
                                     outcomes: List[RewardOutcome], learning_rate: float):
        """Adapt component weights based on correlation with actual outcomes"""
        try:
            if len(outcomes) < 5:
                return
            
            # Extract component scores and actual outcomes
            component_data = {
                'quality': [],
                'efficiency': [],
                'novelty': [],
                'consistency': [],
                'cost_effectiveness': []
            }
            actual_outcomes = []
            
            for outcome in outcomes:
                if 'component_scores' in outcome.context_factors:
                    scores = outcome.context_factors['component_scores']
                    for component, score in scores.items():
                        if component in component_data:
                            component_data[component].append(score)
                    actual_outcomes.append(outcome.actual_outcome)
            
            # Calculate correlations and adjust weights
            total_weight_adjustment = 0.0
            weight_adjustments = {}
            
            for component, scores in component_data.items():
                if len(scores) == len(actual_outcomes) and len(scores) > 3:
                    correlation = self._calculate_correlation(scores, actual_outcomes)
                    
                    # Adjust weight based on correlation
                    if correlation > 0.3:  # Positive correlation
                        adjustment = learning_rate * 0.02 * correlation
                        weight_adjustments[component] = adjustment
                        total_weight_adjustment += adjustment
                    elif correlation < -0.3:  # Negative correlation
                        adjustment = -learning_rate * 0.02 * abs(correlation)
                        weight_adjustments[component] = adjustment
                        total_weight_adjustment += abs(adjustment)
            
            # Apply weight adjustments while maintaining sum = 1.0
            if weight_adjustments:
                # Apply adjustments
                params.quality_weight += weight_adjustments.get('quality', 0.0)
                params.efficiency_weight += weight_adjustments.get('efficiency', 0.0)
                params.novelty_weight += weight_adjustments.get('novelty', 0.0)
                params.consistency_weight += weight_adjustments.get('consistency', 0.0)
                params.cost_weight += weight_adjustments.get('cost_effectiveness', 0.0)
                
                # Normalize weights to sum to 1.0
                total_weight = (params.quality_weight + params.efficiency_weight + 
                              params.novelty_weight + params.consistency_weight + params.cost_weight)
                
                if total_weight > 0:
                    params.quality_weight /= total_weight
                    params.efficiency_weight /= total_weight
                    params.novelty_weight /= total_weight
                    params.consistency_weight /= total_weight
                    params.cost_weight /= total_weight
            
        except Exception as e:
            logger.error(f"Error adapting component weights: {e}")
    
    def _calculate_quality_component(self, context: Dict[str, Any], 
                                   performance_data: Optional[Dict[str, Any]]) -> float:
        """Calculate quality component score"""
        try:
            base_quality = context.get('quality_score', 0.5)
            
            # Adjust based on response characteristics
            response_length = context.get('response_length', 0)
            if response_length > 0:
                # Bonus for adequate length
                length_bonus = min(0.2, response_length / 2000)  # Up to 0.2 bonus
                base_quality += length_bonus
            
            # Structure bonus
            if context.get('has_structure', False):
                base_quality += 0.1
            
            # Domain expertise indicators
            if context.get('domain_expertise_shown', False):
                base_quality += 0.15
            
            return max(0.0, min(1.0, base_quality))
            
        except Exception as e:
            logger.error(f"Error calculating quality component: {e}")
            return 0.5
    
    def _calculate_efficiency_component(self, context: Dict[str, Any], 
                                      performance_data: Optional[Dict[str, Any]]) -> float:
        """Calculate efficiency component score"""
        try:
            base_efficiency = 0.5
            
            # Latency factor
            latency_ms = context.get('latency_ms', 2000)
            if latency_ms > 0:
                # Normalize latency (lower is better)
                # Assume 100ms = 1.0, 5000ms = 0.0
                latency_score = max(0.0, (5000 - latency_ms) / 4900)
                base_efficiency = 0.3 * base_efficiency + 0.7 * latency_score
            
            # Token efficiency
            token_count = context.get('token_count', 0)
            response_quality = context.get('quality_score', 0.5)
            if token_count > 0 and response_quality > 0:
                # Quality per token (higher is better)
                token_efficiency = response_quality / (token_count / 1000)
                efficiency_bonus = min(0.3, token_efficiency * 0.1)
                base_efficiency += efficiency_bonus
            
            return max(0.0, min(1.0, base_efficiency))
            
        except Exception as e:
            logger.error(f"Error calculating efficiency component: {e}")
            return 0.5
    
    def _calculate_novelty_component(self, agent_role: str, context: Dict[str, Any]) -> float:
        """Calculate novelty/creativity component score"""
        try:
            base_novelty = 0.5
            
            # Check for novel approaches
            if context.get('novel_approach_detected', False):
                base_novelty += 0.3
            
            # Check against historical patterns
            task_signature = context.get('task_signature', '')
            if task_signature:
                historical_paths = self.memory_fs.get_successful_paths(task_signature)
                if len(historical_paths) == 0:
                    # Completely new task type
                    base_novelty += 0.4
                elif len(historical_paths) < 3:
                    # Rarely seen task type
                    base_novelty += 0.2
            
            # Agent-specific novelty bonuses
            if agent_role == "synthesizer" and context.get('cross_domain_synthesis', False):
                base_novelty += 0.2
            elif agent_role == "critic" and context.get('unique_perspective_offered', False):
                base_novelty += 0.15
            
            return max(0.0, min(1.0, base_novelty))
            
        except Exception as e:
            logger.error(f"Error calculating novelty component: {e}")
            return 0.5
    
    def _calculate_consistency_component(self, agent_role: str, context: Dict[str, Any]) -> float:
        """Calculate consistency component score"""
        try:
            base_consistency = 0.5
            
            # Get agent's historical performance
            agent_insights = self.memory_fs.get_memory(f"/agents/{agent_role}")
            if agent_insights and agent_insights.children:
                # Calculate consistency from historical success rates
                success_rates = []
                for child_node in agent_insights.children.values():
                    if child_node.metadata.success_rate > 0:
                        success_rates.append(child_node.metadata.success_rate)
                
                if success_rates:
                    # Higher consistency = lower variance in success rates
                    variance = statistics.variance(success_rates) if len(success_rates) > 1 else 0
                    consistency_score = max(0.0, 1.0 - variance)
                    base_consistency = 0.3 * base_consistency + 0.7 * consistency_score
            
            # Current task consistency indicators
            if context.get('follows_established_pattern', False):
                base_consistency += 0.2
            
            if context.get('response_coherence_score', 0) > 0.7:
                base_consistency += 0.1
            
            return max(0.0, min(1.0, base_consistency))
            
        except Exception as e:
            logger.error(f"Error calculating consistency component: {e}")
            return 0.5
    
    def _calculate_cost_effectiveness_component(self, context: Dict[str, Any], 
                                              performance_data: Optional[Dict[str, Any]]) -> float:
        """Calculate cost effectiveness component score"""
        try:
            base_cost_effectiveness = 0.5
            
            cost = context.get('cost_usd', 0.0)
            quality = context.get('quality_score', 0.5)
            
            if cost > 0 and quality > 0:
                # Quality per dollar (higher is better)
                quality_per_cost = quality / cost
                
                # Normalize (assume $0.001/quality point is excellent, $0.1/quality point is poor)
                normalized_efficiency = max(0.0, min(1.0, (0.1 - quality_per_cost) / 0.099))
                base_cost_effectiveness = normalized_efficiency
            
            # Budget efficiency bonus
            if context.get('under_budget', True):
                base_cost_effectiveness += 0.1
            
            return max(0.0, min(1.0, base_cost_effectiveness))
            
        except Exception as e:
            logger.error(f"Error calculating cost effectiveness component: {e}")
            return 0.5
    
    async def _get_historical_adjustment(self, agent_role: str, context: Dict[str, Any]) -> float:
        """Get historical learning adjustment factor"""
        try:
            task_type = context.get('task_type', 'unknown')
            
            # Get recent outcomes for this agent/task combination
            relevant_outcomes = [
                outcome for outcome in self.reward_outcomes[-50:]  # Last 50 outcomes
                if (outcome.agent_role == agent_role and outcome.task_type == task_type)
            ]
            
            if len(relevant_outcomes) < 3:
                return 0.0
            
            # Calculate average prediction error
            prediction_errors = [
                outcome.predicted_reward - outcome.actual_outcome
                for outcome in relevant_outcomes
            ]
            
            avg_error = statistics.mean(prediction_errors)
            
            # Return adjustment that counteracts systematic bias
            # Positive error means over-prediction, so reduce reward
            # Negative error means under-prediction, so increase reward
            adjustment = -avg_error * 0.2  # 20% of average error
            
            return max(-0.3, min(0.3, adjustment))  # Cap adjustment at 30%
            
        except Exception as e:
            logger.error(f"Error calculating historical adjustment: {e}")
            return 0.0
    
    def _calculate_error_trend(self, outcomes: List[RewardOutcome]) -> float:
        """Calculate trend in prediction errors"""
        try:
            if len(outcomes) < 5:
                return 0.0
            
            # Sort by timestamp
            outcomes.sort(key=lambda x: x.timestamp)
            
            # Calculate errors over time
            errors = [abs(outcome.predicted_reward - outcome.actual_outcome) 
                     for outcome in outcomes]
            
            # Calculate trend using simple linear regression
            n = len(errors)
            x_values = list(range(n))
            
            sum_x = sum(x_values)
            sum_y = sum(errors)
            sum_xy = sum(x * y for x, y in zip(x_values, errors))
            sum_xx = sum(x * x for x in x_values)
            
            if n * sum_xx - sum_x * sum_x == 0:
                return 0.0
                
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            return slope
            
        except Exception as e:
            logger.error(f"Error calculating error trend: {e}")
            return 0.0
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        try:
            if len(x_values) != len(y_values) or len(x_values) < 2:
                return 0.0
            
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_xx = sum(x * x for x in x_values)
            sum_yy = sum(y * y for y in y_values)
            
            denominator = math.sqrt((n * sum_xx - sum_x ** 2) * (n * sum_yy - sum_y ** 2))
            
            if denominator == 0:
                return 0.0
            
            correlation = (n * sum_xy - sum_x * sum_y) / denominator
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create hash of context for consistent identification"""
        try:
            # Extract key context elements
            key_elements = {
                'task_type': context.get('task_type', ''),
                'complexity': context.get('complexity', ''),
                'domain': context.get('domain', ''),
                'framework': context.get('framework', '')
            }
            
            context_str = json.dumps(key_elements, sort_keys=True)
            return str(hash(context_str))
            
        except Exception as e:
            logger.error(f"Error hashing context: {e}")
            return "unknown"
    
    async def _store_outcome_to_memory(self, outcome: RewardOutcome):
        """Store outcome to memory filesystem for persistence"""
        try:
            timestamp_str = str(int(outcome.timestamp))
            memory_path = f"/insights/adaptive_rewards/{outcome.agent_role}_{timestamp_str}"
            
            content = {
                'outcome_data': asdict(outcome),
                'stored_at': time.time()
            }
            
            metadata = {
                'importance_score': 0.4 + (0.3 * abs(outcome.predicted_reward - outcome.actual_outcome)),
                'success_rate': 1.0 if abs(outcome.predicted_reward - outcome.actual_outcome) < 0.2 else 0.5,
                'tags': ['adaptive_rewards', 'learning', outcome.agent_role],
                'agent_annotations': {
                    'agent_role': outcome.agent_role,
                    'task_type': outcome.task_type,
                    'prediction_error': abs(outcome.predicted_reward - outcome.actual_outcome)
                },
                'performance_metrics': {
                    'predicted_reward': outcome.predicted_reward,
                    'actual_outcome': outcome.actual_outcome,
                    'error': abs(outcome.predicted_reward - outcome.actual_outcome)
                }
            }
            
            self.memory_fs.store_memory(memory_path, content, metadata)
            
        except Exception as e:
            logger.error(f"Error storing outcome to memory: {e}")
    
    async def _load_adaptive_data(self):
        """Load existing adaptive learning data"""
        try:
            # Load from memory filesystem
            rewards_insights = self.memory_fs.get_memory("/insights/adaptive_rewards")
            if rewards_insights and rewards_insights.children:
                loaded_outcomes = []
                
                for child_node in rewards_insights.children.values():
                    if 'outcome_data' in child_node.content:
                        outcome_data = child_node.content['outcome_data']
                        outcome = RewardOutcome(**outcome_data)
                        loaded_outcomes.append(outcome)
                
                # Sort by timestamp and keep recent ones
                loaded_outcomes.sort(key=lambda x: x.timestamp)
                self.reward_outcomes = loaded_outcomes[-self.max_learning_history:]
                
                logger.info(f"Loaded {len(self.reward_outcomes)} reward learning outcomes")
            
        except Exception as e:
            logger.error(f"Error loading adaptive data: {e}")
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptive learning status"""
        try:
            summary = {
                'total_outcomes_recorded': len(self.reward_outcomes),
                'agent_adaptations': {},
                'recent_performance': {},
                'parameter_evolution': {}
            }
            
            # Summarize by agent
            for agent_role in self.agent_reward_params.keys():
                agent_outcomes = [o for o in self.reward_outcomes if o.agent_role == agent_role]
                
                if agent_outcomes:
                    recent_outcomes = agent_outcomes[-20:]  # Last 20
                    avg_error = statistics.mean([
                        abs(o.predicted_reward - o.actual_outcome) 
                        for o in recent_outcomes
                    ])
                    
                    adaptations_count = len(self.adaptation_history.get(agent_role, []))
                    
                    summary['agent_adaptations'][agent_role] = {
                        'total_outcomes': len(agent_outcomes),
                        'recent_avg_error': avg_error,
                        'adaptations_count': adaptations_count,
                        'current_parameters': asdict(self.agent_reward_params[agent_role])
                    }
            
            # Recent overall performance
            if self.reward_outcomes:
                recent_outcomes = self.reward_outcomes[-50:]
                summary['recent_performance'] = {
                    'avg_prediction_error': statistics.mean([
                        abs(o.predicted_reward - o.actual_outcome) 
                        for o in recent_outcomes
                    ]),
                    'outcomes_with_learning': sum(1 for o in recent_outcomes if o.learning_applied),
                    'time_span_hours': (recent_outcomes[-1].timestamp - recent_outcomes[0].timestamp) / 3600
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting adaptation summary: {e}")
            return {'error': str(e)}