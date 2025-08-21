"""
Historical Performance Tracking System
Tracks and analyzes AI provider and agent performance over time
"""

import json
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceObjective(Enum):
    QUALITY = "quality"
    LATENCY = "latency" 
    COST = "cost"
    RELIABILITY = "reliability"

@dataclass
class PerformanceMetric:
    """Individual performance measurement"""
    timestamp: float
    provider: str
    model: str
    agent_role: str
    task_type: str
    quality_score: float
    latency_ms: float
    cost_usd: float
    success: bool
    error_type: Optional[str] = None
    context_size: int = 0
    response_length: int = 0

@dataclass
class PerformanceAggregate:
    """Aggregated performance statistics"""
    provider: str
    model: str
    agent_role: str
    task_type: str
    total_requests: int
    success_rate: float
    avg_quality_score: float
    avg_latency_ms: float
    avg_cost_usd: float
    p95_latency_ms: float
    cost_per_quality_point: float
    last_updated: float
    trend_direction: str  # "improving", "declining", "stable"

class HistoricalPerformanceTracker:
    """Tracks and analyzes historical performance of AI providers and agents"""
    
    def __init__(self, memory_filesystem, storage_path: Optional[Path] = None):
        self.memory_fs = memory_filesystem
        self.storage_path = storage_path or Path.home() / ".monk-memory" / "performance"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory performance data
        self.recent_metrics = []  # Recent measurements for trend analysis
        self.aggregates = {}  # provider:model:agent:task -> PerformanceAggregate
        self.provider_rankings = {}  # objective -> ranked list of providers
        
        # Configuration
        self.max_recent_metrics = 1000  # Keep last 1000 metrics in memory
        self.min_samples_for_ranking = 5  # Minimum samples before ranking
        self.trend_window_hours = 24  # Hours to look back for trend analysis
        
        # Load existing data
        asyncio.create_task(self._load_historical_data())
    
    async def record_performance(self, metric: PerformanceMetric) -> bool:
        """Record a new performance measurement"""
        try:
            # Add to recent metrics
            self.recent_metrics.append(metric)
            
            # Keep only recent metrics in memory
            if len(self.recent_metrics) > self.max_recent_metrics:
                self.recent_metrics = self.recent_metrics[-self.max_recent_metrics:]
            
            # Update aggregates
            await self._update_aggregates(metric)
            
            # Store to memory filesystem
            await self._store_metric_to_memory(metric)
            
            # Update provider rankings
            await self._update_provider_rankings()
            
            # Persist to disk periodically
            if len(self.recent_metrics) % 50 == 0:  # Every 50 metrics
                await self._persist_performance_data()
            
            logger.debug(f"Recorded performance: {metric.provider}:{metric.model} - "
                        f"Quality: {metric.quality_score:.2f}, Latency: {metric.latency_ms}ms")
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording performance metric: {e}")
            return False
    
    async def get_optimal_provider(self, agent_role: str, task_type: str, 
                                 objective: PerformanceObjective = PerformanceObjective.QUALITY,
                                 context_constraints: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get optimal provider based on historical performance"""
        try:
            # Find all providers that have handled this agent/task combination
            candidates = []
            
            for key, aggregate in self.aggregates.items():
                provider, model, agg_agent, agg_task = key.split(':', 3)
                
                if (agg_agent == agent_role and 
                    (agg_task == task_type or self._is_similar_task_type(agg_task, task_type)) and
                    aggregate.total_requests >= self.min_samples_for_ranking):
                    
                    # Calculate score based on objective
                    score = self._calculate_objective_score(aggregate, objective)
                    
                    # Apply context constraints if any
                    if context_constraints:
                        score *= self._apply_context_constraints(aggregate, context_constraints)
                    
                    candidates.append({
                        'provider': provider,
                        'model': model,
                        'score': score,
                        'aggregate': aggregate
                    })
            
            if not candidates:
                logger.warning(f"No performance data for {agent_role}:{task_type}")
                return None
            
            # Sort by score and return best
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best = candidates[0]
            
            logger.info(f"Selected optimal provider: {best['provider']}:{best['model']} "
                       f"(score: {best['score']:.3f}) for {agent_role}:{task_type}")
            
            return f"{best['provider']}:{best['model']}"
            
        except Exception as e:
            logger.error(f"Error getting optimal provider: {e}")
            return None
    
    def get_provider_performance_summary(self, provider: str, model: str, 
                                       agent_role: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for a specific provider/model"""
        try:
            matching_aggregates = []
            
            for key, aggregate in self.aggregates.items():
                agg_provider, agg_model, agg_agent, agg_task = key.split(':', 3)
                
                if (agg_provider == provider and agg_model == model and
                    (agent_role is None or agg_agent == agent_role)):
                    matching_aggregates.append(aggregate)
            
            if not matching_aggregates:
                return {'error': 'No performance data found'}
            
            # Calculate overall statistics
            total_requests = sum(agg.total_requests for agg in matching_aggregates)
            avg_success_rate = statistics.mean(agg.success_rate for agg in matching_aggregates)
            avg_quality = statistics.mean(agg.avg_quality_score for agg in matching_aggregates)
            avg_latency = statistics.mean(agg.avg_latency_ms for agg in matching_aggregates)
            avg_cost = statistics.mean(agg.avg_cost_usd for agg in matching_aggregates)
            
            # Determine overall trend
            trends = [agg.trend_direction for agg in matching_aggregates]
            improving_count = trends.count("improving")
            declining_count = trends.count("declining")
            
            if improving_count > declining_count:
                overall_trend = "improving"
            elif declining_count > improving_count:
                overall_trend = "declining"
            else:
                overall_trend = "stable"
            
            return {
                'provider': provider,
                'model': model,
                'agent_role': agent_role,
                'total_requests': total_requests,
                'avg_success_rate': avg_success_rate,
                'avg_quality_score': avg_quality,
                'avg_latency_ms': avg_latency,
                'avg_cost_usd': avg_cost,
                'cost_efficiency': avg_quality / max(avg_cost, 0.001),
                'overall_trend': overall_trend,
                'task_specializations': [agg.task_type for agg in matching_aggregates],
                'last_updated': max(agg.last_updated for agg in matching_aggregates)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def get_performance_trends(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get performance trends over specified time period"""
        try:
            cutoff_time = time.time() - (hours_back * 3600)
            recent_metrics = [m for m in self.recent_metrics if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                return {'error': 'No recent metrics available'}
            
            # Group by provider:model
            provider_groups = {}
            for metric in recent_metrics:
                key = f"{metric.provider}:{metric.model}"
                if key not in provider_groups:
                    provider_groups[key] = []
                provider_groups[key].append(metric)
            
            trends = {}
            for provider_key, metrics in provider_groups.items():
                if len(metrics) < 3:  # Need minimum data for trend
                    continue
                
                # Sort by timestamp
                metrics.sort(key=lambda m: m.timestamp)
                
                # Calculate trends
                quality_trend = self._calculate_trend([m.quality_score for m in metrics])
                latency_trend = self._calculate_trend([m.latency_ms for m in metrics])
                success_trend = self._calculate_trend([float(m.success) for m in metrics])
                
                trends[provider_key] = {
                    'quality_trend': quality_trend,
                    'latency_trend': latency_trend,  # Lower is better, so negative trend is good
                    'success_trend': success_trend,
                    'sample_count': len(metrics),
                    'time_span_hours': (metrics[-1].timestamp - metrics[0].timestamp) / 3600
                }
            
            return {
                'time_period_hours': hours_back,
                'total_metrics': len(recent_metrics),
                'provider_trends': trends,
                'generated_at': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return {'error': str(e)}
    
    async def _update_aggregates(self, metric: PerformanceMetric):
        """Update aggregate statistics with new metric"""
        try:
            key = f"{metric.provider}:{metric.model}:{metric.agent_role}:{metric.task_type}"
            
            if key in self.aggregates:
                # Update existing aggregate
                agg = self.aggregates[key]
                
                # Calculate new averages
                total_requests = agg.total_requests + 1
                new_success_rate = ((agg.success_rate * agg.total_requests) + 
                                  (1.0 if metric.success else 0.0)) / total_requests
                
                new_avg_quality = ((agg.avg_quality_score * agg.total_requests) + 
                                 metric.quality_score) / total_requests
                
                new_avg_latency = ((agg.avg_latency_ms * agg.total_requests) + 
                                 metric.latency_ms) / total_requests
                
                new_avg_cost = ((agg.avg_cost_usd * agg.total_requests) + 
                              metric.cost_usd) / total_requests
                
                # Update aggregate
                agg.total_requests = total_requests
                agg.success_rate = new_success_rate
                agg.avg_quality_score = new_avg_quality
                agg.avg_latency_ms = new_avg_latency
                agg.avg_cost_usd = new_avg_cost
                agg.cost_per_quality_point = new_avg_cost / max(new_avg_quality, 0.001)
                agg.last_updated = time.time()
                
                # Update trend
                agg.trend_direction = await self._calculate_trend_direction(key, metric)
                
                # Update P95 latency (simplified)
                recent_latencies = [m.latency_ms for m in self.recent_metrics 
                                  if (f"{m.provider}:{m.model}:{m.agent_role}:{m.task_type}" == key)]
                if recent_latencies:
                    recent_latencies.sort()
                    p95_index = int(len(recent_latencies) * 0.95)
                    agg.p95_latency_ms = recent_latencies[p95_index]
            
            else:
                # Create new aggregate
                self.aggregates[key] = PerformanceAggregate(
                    provider=metric.provider,
                    model=metric.model,
                    agent_role=metric.agent_role,
                    task_type=metric.task_type,
                    total_requests=1,
                    success_rate=1.0 if metric.success else 0.0,
                    avg_quality_score=metric.quality_score,
                    avg_latency_ms=metric.latency_ms,
                    avg_cost_usd=metric.cost_usd,
                    p95_latency_ms=metric.latency_ms,
                    cost_per_quality_point=metric.cost_usd / max(metric.quality_score, 0.001),
                    last_updated=time.time(),
                    trend_direction="stable"
                )
            
        except Exception as e:
            logger.error(f"Error updating aggregates: {e}")
    
    async def _store_metric_to_memory(self, metric: PerformanceMetric):
        """Store metric to memory filesystem"""
        try:
            # Create memory path
            timestamp_str = str(int(metric.timestamp))
            memory_path = f"/insights/provider_performance/{metric.provider}_{metric.model}_{timestamp_str}"
            
            # Prepare content
            content = {
                'metric_data': asdict(metric),
                'recorded_at': time.time()
            }
            
            # Prepare metadata
            metadata = {
                'importance_score': 0.3 + (0.4 * metric.quality_score),  # Higher for better quality
                'success_rate': 1.0 if metric.success else 0.0,
                'tags': ['performance', 'provider', metric.provider, metric.agent_role],
                'agent_annotations': {
                    'provider': metric.provider,
                    'model': metric.model,
                    'agent_role': metric.agent_role,
                    'task_type': metric.task_type
                },
                'performance_metrics': {
                    'quality': metric.quality_score,
                    'latency': metric.latency_ms,
                    'cost': metric.cost_usd
                }
            }
            
            # Store in memory filesystem
            self.memory_fs.store_memory(memory_path, content, metadata)
            
        except Exception as e:
            logger.error(f"Error storing metric to memory: {e}")
    
    async def _update_provider_rankings(self):
        """Update provider rankings for different objectives"""
        try:
            for objective in PerformanceObjective:
                rankings = []
                
                # Group by provider:model
                provider_scores = {}
                for key, aggregate in self.aggregates.items():
                    provider, model, agent_role, task_type = key.split(':', 3)
                    provider_key = f"{provider}:{model}"
                    
                    if aggregate.total_requests >= self.min_samples_for_ranking:
                        score = self._calculate_objective_score(aggregate, objective)
                        
                        if provider_key not in provider_scores:
                            provider_scores[provider_key] = []
                        provider_scores[provider_key].append(score)
                
                # Calculate average scores per provider
                for provider_key, scores in provider_scores.items():
                    avg_score = statistics.mean(scores)
                    rankings.append({
                        'provider': provider_key,
                        'score': avg_score,
                        'sample_count': len(scores)
                    })
                
                # Sort by score
                rankings.sort(key=lambda x: x['score'], reverse=True)
                
                # Store rankings
                self.provider_rankings[objective.value] = rankings
                
        except Exception as e:
            logger.error(f"Error updating provider rankings: {e}")
    
    def _calculate_objective_score(self, aggregate: PerformanceAggregate, 
                                 objective: PerformanceObjective) -> float:
        """Calculate score based on optimization objective"""
        try:
            if objective == PerformanceObjective.QUALITY:
                # Quality score with success rate penalty
                return aggregate.avg_quality_score * aggregate.success_rate
                
            elif objective == PerformanceObjective.LATENCY:
                # Lower latency is better, normalize to 0-1 scale
                # Assume 5000ms is worst case (score=0), 100ms is best case (score=1)
                max_latency = 5000
                min_latency = 100
                normalized = (max_latency - aggregate.avg_latency_ms) / (max_latency - min_latency)
                return max(0.0, min(1.0, normalized)) * aggregate.success_rate
                
            elif objective == PerformanceObjective.COST:
                # Lower cost per quality point is better
                if aggregate.cost_per_quality_point <= 0:
                    return 0.0
                # Normalize: assume $0.001/point is best, $1.0/point is worst
                max_cost_per_quality = 1.0
                min_cost_per_quality = 0.001
                normalized = (max_cost_per_quality - aggregate.cost_per_quality_point) / (max_cost_per_quality - min_cost_per_quality)
                return max(0.0, min(1.0, normalized)) * aggregate.success_rate
                
            elif objective == PerformanceObjective.RELIABILITY:
                # Success rate with consistency bonus
                trend_bonus = 0.1 if aggregate.trend_direction == "improving" else 0.0
                return aggregate.success_rate + trend_bonus
                
            else:
                return aggregate.avg_quality_score * aggregate.success_rate
                
        except Exception as e:
            logger.error(f"Error calculating objective score: {e}")
            return 0.0
    
    def _apply_context_constraints(self, aggregate: PerformanceAggregate, 
                                 constraints: Dict[str, Any]) -> float:
        """Apply context constraints to scoring"""
        multiplier = 1.0
        
        try:
            # Latency constraint
            if 'max_latency_ms' in constraints:
                if aggregate.avg_latency_ms > constraints['max_latency_ms']:
                    multiplier *= 0.5  # Penalize high latency
            
            # Cost constraint  
            if 'max_cost_usd' in constraints:
                if aggregate.avg_cost_usd > constraints['max_cost_usd']:
                    multiplier *= 0.3  # Heavy penalty for exceeding budget
            
            # Minimum quality constraint
            if 'min_quality' in constraints:
                if aggregate.avg_quality_score < constraints['min_quality']:
                    multiplier *= 0.2  # Heavy penalty for low quality
            
            # Reliability constraint
            if 'min_success_rate' in constraints:
                if aggregate.success_rate < constraints['min_success_rate']:
                    multiplier *= 0.1  # Very heavy penalty for unreliability
                    
        except Exception as e:
            logger.error(f"Error applying context constraints: {e}")
        
        return multiplier
    
    async def _calculate_trend_direction(self, aggregate_key: str, latest_metric: PerformanceMetric) -> str:
        """Calculate trend direction for an aggregate"""
        try:
            cutoff_time = time.time() - (self.trend_window_hours * 3600)
            
            # Get recent metrics for this provider/agent/task combo
            recent_metrics = [
                m for m in self.recent_metrics
                if (f"{m.provider}:{m.model}:{m.agent_role}:{m.task_type}" == aggregate_key and
                    m.timestamp > cutoff_time)
            ]
            
            if len(recent_metrics) < 5:  # Need minimum data for trend
                return "stable"
            
            # Sort by timestamp
            recent_metrics.sort(key=lambda m: m.timestamp)
            
            # Calculate quality trend over time
            quality_scores = [m.quality_score for m in recent_metrics]
            trend_slope = self._calculate_trend(quality_scores)
            
            if trend_slope > 0.05:
                return "improving"
            elif trend_slope < -0.05:
                return "declining"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating trend direction: {e}")
            return "stable"
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope using simple linear regression"""
        if len(values) < 2:
            return 0.0
            
        try:
            n = len(values)
            x_values = list(range(n))
            
            # Calculate slope using least squares
            sum_x = sum(x_values)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(x_values, values))
            sum_xx = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            return slope
            
        except (ZeroDivisionError, ValueError):
            return 0.0
    
    def _is_similar_task_type(self, task1: str, task2: str) -> bool:
        """Check if two task types are similar enough to share performance data"""
        # Simple similarity check - can be enhanced
        task1_words = set(task1.lower().split())
        task2_words = set(task2.lower().split())
        
        if not task1_words or not task2_words:
            return False
            
        intersection = task1_words & task2_words
        union = task1_words | task2_words
        
        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity >= 0.3  # 30% similarity threshold
    
    async def _persist_performance_data(self):
        """Persist performance data to disk"""
        try:
            # Save aggregates
            aggregates_file = self.storage_path / "performance_aggregates.json"
            aggregates_data = {
                key: asdict(aggregate) 
                for key, aggregate in self.aggregates.items()
            }
            
            with open(aggregates_file, 'w') as f:
                json.dump(aggregates_data, f, indent=2, default=str)
            
            # Save recent metrics (keep last 500)
            metrics_file = self.storage_path / "recent_metrics.json"
            recent_data = [
                asdict(metric) 
                for metric in self.recent_metrics[-500:]
            ]
            
            with open(metrics_file, 'w') as f:
                json.dump(recent_data, f, indent=2, default=str)
                
            logger.debug("Persisted performance data to disk")
            
        except Exception as e:
            logger.error(f"Error persisting performance data: {e}")
    
    async def _load_historical_data(self):
        """Load historical performance data from disk"""
        try:
            # Load aggregates
            aggregates_file = self.storage_path / "performance_aggregates.json"
            if aggregates_file.exists():
                with open(aggregates_file, 'r') as f:
                    aggregates_data = json.load(f)
                
                for key, data in aggregates_data.items():
                    self.aggregates[key] = PerformanceAggregate(**data)
                
                logger.info(f"Loaded {len(self.aggregates)} performance aggregates")
            
            # Load recent metrics
            metrics_file = self.storage_path / "recent_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                self.recent_metrics = [PerformanceMetric(**data) for data in metrics_data]
                logger.info(f"Loaded {len(self.recent_metrics)} recent metrics")
            
            # Update rankings
            await self._update_provider_rankings()
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def get_performance_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance optimization recommendations"""
        try:
            recommendations = {
                'provider_suggestions': [],
                'cost_optimizations': [],
                'performance_alerts': [],
                'trend_insights': []
            }
            
            # Analyze current performance patterns
            recent_trends = self.get_performance_trends(hours_back=24)
            
            if 'provider_trends' in recent_trends:
                for provider_key, trend_data in recent_trends['provider_trends'].items():
                    
                    # Provider suggestions based on trends
                    if trend_data['quality_trend'] > 0.1:
                        recommendations['provider_suggestions'].append({
                            'provider': provider_key,
                            'reason': f"Quality improving (trend: +{trend_data['quality_trend']:.3f})",
                            'confidence': 0.8
                        })
                    
                    # Performance alerts
                    if trend_data['success_trend'] < -0.1:
                        recommendations['performance_alerts'].append({
                            'provider': provider_key,
                            'alert': f"Success rate declining (trend: {trend_data['success_trend']:.3f})",
                            'severity': 'high'
                        })
                    
                    # Cost optimization
                    provider_summary = self.get_provider_performance_summary(
                        *provider_key.split(':', 1)
                    )
                    if 'cost_efficiency' in provider_summary:
                        if provider_summary['cost_efficiency'] < 500:  # Low efficiency
                            recommendations['cost_optimizations'].append({
                                'provider': provider_key,
                                'suggestion': 'Consider switching to more cost-effective provider',
                                'current_efficiency': provider_summary['cost_efficiency']
                            })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting performance recommendations: {e}")
            return {'error': str(e)}