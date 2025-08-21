"""
Agent Specialization System
Enables agents to develop specialized capabilities based on performance patterns
"""

import json
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import logging
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)

class SpecializationDomain(Enum):
    """Domains that agents can specialize in"""
    CODE_ANALYSIS = "code_analysis"
    ARCHITECTURE_DESIGN = "architecture_design"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_ASSESSMENT = "security_assessment"
    DATABASE_DESIGN = "database_design"
    FRONTEND_DEVELOPMENT = "frontend_development"
    BACKEND_SYSTEMS = "backend_systems"
    DEVOPS_AUTOMATION = "devops_automation"
    API_DESIGN = "api_design"
    TESTING_STRATEGIES = "testing_strategies"
    PROJECT_PLANNING = "project_planning"
    REQUIREMENTS_ANALYSIS = "requirements_analysis"

class SpecializationLevel(Enum):
    """Levels of specialization"""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"

@dataclass
class SpecializationMetrics:
    """Metrics for a specific specialization"""
    domain: SpecializationDomain
    level: SpecializationLevel
    proficiency_score: float  # 0.0 to 1.0
    task_success_rate: float
    avg_quality_score: float
    total_tasks_handled: int
    learning_velocity: float  # Rate of improvement
    confidence_score: float
    last_updated: float
    specialization_evidence: List[str]  # Evidence of specialization

@dataclass
class AgentProfile:
    """Profile of an agent's capabilities and specializations"""
    agent_role: str
    primary_specializations: List[SpecializationMetrics]
    secondary_specializations: List[SpecializationMetrics]
    learning_preferences: Dict[str, float]
    collaboration_patterns: Dict[str, float]  # How well works with other agents
    adaptation_rate: float
    consistency_score: float
    total_experience: int
    created_at: float
    last_performance_update: float

@dataclass
class TaskAssignmentRecommendation:
    """Recommendation for task assignment"""
    recommended_agent: str
    confidence: float
    reasoning: str
    alternative_agents: List[Tuple[str, float]]  # agent, confidence
    specialization_match: Dict[str, float]

class AgentSpecializationSystem:
    """System for developing and managing agent specializations"""
    
    def __init__(self, memory_filesystem, historical_performance_tracker):
        self.memory_fs = memory_filesystem
        self.performance_tracker = historical_performance_tracker
        
        # Agent profiles and specializations
        self.agent_profiles = {}  # agent_role -> AgentProfile
        self.domain_experts = {}  # domain -> List[agent_role] ranked by expertise
        self.task_patterns = {}  # task_signature -> specialization requirements
        
        # Configuration
        self.min_tasks_for_specialization = 10
        self.specialization_threshold = 0.7  # Performance threshold for specialization
        self.expertise_decay_rate = 0.95  # Daily decay if not used
        self.cross_training_bonus = 0.1  # Bonus for working across domains
        
        # Initialize base agent profiles
        self._initialize_base_profiles()
        
        # Load existing specialization data
        asyncio.create_task(self._load_specialization_data())
    
    def _initialize_base_profiles(self):
        """Initialize base profiles for standard agents"""
        base_agents = ["planner", "analyzer", "critic", "synthesizer", "executor"]
        
        for agent_role in base_agents:
            profile = AgentProfile(
                agent_role=agent_role,
                primary_specializations=[],
                secondary_specializations=[],
                learning_preferences=self._get_default_learning_preferences(agent_role),
                collaboration_patterns=self._get_default_collaboration_patterns(agent_role),
                adaptation_rate=0.1,
                consistency_score=0.5,
                total_experience=0,
                created_at=time.time(),
                last_performance_update=time.time()
            )
            
            self.agent_profiles[agent_role] = profile
    
    def _get_default_learning_preferences(self, agent_role: str) -> Dict[str, float]:
        """Get default learning preferences for agent role"""
        preferences = {
            "planner": {
                "architecture_design": 0.9,
                "project_planning": 1.0,
                "requirements_analysis": 0.8,
                "performance_optimization": 0.6
            },
            "analyzer": {
                "code_analysis": 1.0,
                "performance_optimization": 0.9,
                "testing_strategies": 0.7,
                "database_design": 0.6
            },
            "critic": {
                "security_assessment": 0.9,
                "code_analysis": 0.8,
                "testing_strategies": 0.8,
                "performance_optimization": 0.7
            },
            "synthesizer": {
                "api_design": 0.8,
                "architecture_design": 0.7,
                "requirements_analysis": 0.8,
                "project_planning": 0.6
            },
            "executor": {
                "devops_automation": 0.9,
                "backend_systems": 0.8,
                "frontend_development": 0.7,
                "database_design": 0.6
            }
        }
        
        return preferences.get(agent_role, {})
    
    def _get_default_collaboration_patterns(self, agent_role: str) -> Dict[str, float]:
        """Get default collaboration effectiveness with other agents"""
        patterns = {
            "planner": {
                "analyzer": 0.8,
                "critic": 0.6,
                "synthesizer": 0.9,
                "executor": 0.7
            },
            "analyzer": {
                "planner": 0.8,
                "critic": 0.9,
                "synthesizer": 0.7,
                "executor": 0.6
            },
            "critic": {
                "planner": 0.6,
                "analyzer": 0.9,
                "synthesizer": 0.8,
                "executor": 0.7
            },
            "synthesizer": {
                "planner": 0.9,
                "analyzer": 0.7,
                "critic": 0.8,
                "executor": 0.8
            },
            "executor": {
                "planner": 0.7,
                "analyzer": 0.6,
                "critic": 0.7,
                "synthesizer": 0.8
            }
        }
        
        return patterns.get(agent_role, {})
    
    async def record_task_performance(self, agent_role: str, task_context: Dict[str, Any], 
                                    performance_metrics: Dict[str, float]):
        """Record task performance and update specializations"""
        try:
            if agent_role not in self.agent_profiles:
                logger.warning(f"Unknown agent role: {agent_role}")
                return
            
            profile = self.agent_profiles[agent_role]
            
            # Identify task domain(s)
            task_domains = self._identify_task_domains(task_context)
            
            # Update experience
            profile.total_experience += 1
            profile.last_performance_update = time.time()
            
            # Update specializations
            for domain in task_domains:
                await self._update_domain_performance(profile, domain, performance_metrics)
            
            # Update collaboration patterns if applicable
            collaborating_agents = task_context.get('collaborating_agents', [])
            if collaborating_agents:
                await self._update_collaboration_patterns(profile, collaborating_agents, 
                                                       performance_metrics.get('quality_score', 0.5))
            
            # Check for new specializations
            await self._evaluate_specialization_development(profile)
            
            # Update domain expert rankings
            await self._update_domain_rankings()
            
            # Store specialization data
            await self._store_specialization_update(agent_role, task_domains, performance_metrics)
            
        except Exception as e:
            logger.error(f"Error recording task performance: {e}")
    
    async def get_optimal_agent_assignment(self, task_context: Dict[str, Any], 
                                         available_agents: List[str] = None) -> TaskAssignmentRecommendation:
        """Get optimal agent assignment based on specializations"""
        try:
            if available_agents is None:
                available_agents = list(self.agent_profiles.keys())
            
            # Identify task domains
            task_domains = self._identify_task_domains(task_context)
            task_complexity = self._estimate_task_complexity(task_context)
            
            # Score each agent
            agent_scores = []
            
            for agent_role in available_agents:
                if agent_role not in self.agent_profiles:
                    continue
                
                profile = self.agent_profiles[agent_role]
                score = await self._calculate_agent_task_score(profile, task_domains, 
                                                             task_complexity, task_context)
                
                if score > 0:
                    agent_scores.append((agent_role, score))
            
            if not agent_scores:
                return TaskAssignmentRecommendation(
                    recommended_agent="planner",  # Default fallback
                    confidence=0.1,
                    reasoning="No specialized agent found, using default planner",
                    alternative_agents=[],
                    specialization_match={}
                )
            
            # Sort by score
            agent_scores.sort(key=lambda x: x[1], reverse=True)
            
            best_agent, best_score = agent_scores[0]
            alternatives = agent_scores[1:4]  # Top 3 alternatives
            
            # Calculate specialization match details
            specialization_match = await self._get_specialization_match_details(
                self.agent_profiles[best_agent], task_domains
            )
            
            # Generate reasoning
            reasoning = await self._generate_assignment_reasoning(
                best_agent, task_domains, best_score, specialization_match
            )
            
            return TaskAssignmentRecommendation(
                recommended_agent=best_agent,
                confidence=min(1.0, best_score),
                reasoning=reasoning,
                alternative_agents=alternatives,
                specialization_match=specialization_match
            )
            
        except Exception as e:
            logger.error(f"Error getting optimal agent assignment: {e}")
            return TaskAssignmentRecommendation(
                recommended_agent="planner",
                confidence=0.1,
                reasoning=f"Error in assignment: {str(e)}",
                alternative_agents=[],
                specialization_match={}
            )
    
    def _identify_task_domains(self, task_context: Dict[str, Any]) -> List[SpecializationDomain]:
        """Identify specialization domains relevant to task"""
        domains = []
        
        try:
            task_description = task_context.get('task', '').lower()
            task_type = task_context.get('task_type', '').lower()
            keywords = task_context.get('keywords', [])
            
            # Combine all text for analysis
            text_content = f"{task_description} {task_type} {' '.join(keywords)}".lower()
            
            # Domain keyword mappings
            domain_keywords = {
                SpecializationDomain.CODE_ANALYSIS: [
                    'code', 'analyze', 'review', 'refactor', 'debug', 'syntax', 'logic'
                ],
                SpecializationDomain.ARCHITECTURE_DESIGN: [
                    'architecture', 'design', 'pattern', 'system', 'structure', 'component'
                ],
                SpecializationDomain.PERFORMANCE_OPTIMIZATION: [
                    'performance', 'optimize', 'speed', 'memory', 'efficiency', 'benchmark'
                ],
                SpecializationDomain.SECURITY_ASSESSMENT: [
                    'security', 'vulnerability', 'auth', 'encryption', 'secure', 'attack'
                ],
                SpecializationDomain.DATABASE_DESIGN: [
                    'database', 'sql', 'query', 'schema', 'table', 'index', 'data'
                ],
                SpecializationDomain.FRONTEND_DEVELOPMENT: [
                    'frontend', 'ui', 'react', 'vue', 'angular', 'css', 'html', 'browser'
                ],
                SpecializationDomain.BACKEND_SYSTEMS: [
                    'backend', 'server', 'api', 'microservice', 'service', 'endpoint'
                ],
                SpecializationDomain.DEVOPS_AUTOMATION: [
                    'devops', 'deploy', 'ci/cd', 'docker', 'kubernetes', 'automation'
                ],
                SpecializationDomain.API_DESIGN: [
                    'api', 'rest', 'graphql', 'endpoint', 'interface', 'contract'
                ],
                SpecializationDomain.TESTING_STRATEGIES: [
                    'test', 'testing', 'unit', 'integration', 'qa', 'validation'
                ],
                SpecializationDomain.PROJECT_PLANNING: [
                    'plan', 'planning', 'roadmap', 'timeline', 'milestone', 'project'
                ],
                SpecializationDomain.REQUIREMENTS_ANALYSIS: [
                    'requirements', 'specification', 'analysis', 'stakeholder', 'needs'
                ]
            }
            
            # Check for domain matches
            for domain, keywords_list in domain_keywords.items():
                keyword_matches = sum(1 for keyword in keywords_list if keyword in text_content)
                if keyword_matches >= 1:  # At least one keyword match
                    domains.append(domain)
            
            # If no specific domains identified, try to infer from context
            if not domains:
                # Default based on task type patterns
                if any(word in text_content for word in ['create', 'build', 'implement']):
                    domains.append(SpecializationDomain.BACKEND_SYSTEMS)
                elif any(word in text_content for word in ['analyze', 'review', 'check']):
                    domains.append(SpecializationDomain.CODE_ANALYSIS)
                else:
                    domains.append(SpecializationDomain.PROJECT_PLANNING)  # Default
            
            return domains
            
        except Exception as e:
            logger.error(f"Error identifying task domains: {e}")
            return [SpecializationDomain.PROJECT_PLANNING]
    
    def _estimate_task_complexity(self, task_context: Dict[str, Any]) -> float:
        """Estimate task complexity (0.0 to 1.0)"""
        try:
            complexity_indicators = {
                'task_length': len(task_context.get('task', '')),
                'num_requirements': len(task_context.get('requirements', [])),
                'num_technologies': len(task_context.get('technologies', [])),
                'estimated_time': task_context.get('estimated_hours', 1)
            }
            
            # Normalize and weight indicators
            normalized_complexity = 0.0
            
            # Task description length (longer = more complex)
            if complexity_indicators['task_length'] > 0:
                length_complexity = min(1.0, complexity_indicators['task_length'] / 1000)
                normalized_complexity += 0.2 * length_complexity
            
            # Number of requirements
            req_complexity = min(1.0, complexity_indicators['num_requirements'] / 10)
            normalized_complexity += 0.3 * req_complexity
            
            # Number of technologies involved
            tech_complexity = min(1.0, complexity_indicators['num_technologies'] / 5)
            normalized_complexity += 0.2 * tech_complexity
            
            # Estimated time
            time_complexity = min(1.0, complexity_indicators['estimated_time'] / 40)  # 40 hours = very complex
            normalized_complexity += 0.3 * time_complexity
            
            return max(0.1, min(1.0, normalized_complexity))
            
        except Exception as e:
            logger.error(f"Error estimating task complexity: {e}")
            return 0.5
    
    async def _calculate_agent_task_score(self, profile: AgentProfile, 
                                        task_domains: List[SpecializationDomain],
                                        task_complexity: float,
                                        task_context: Dict[str, Any]) -> float:
        """Calculate how well an agent matches a task"""
        try:
            base_score = 0.3  # Base score for any agent
            
            # Specialization match score
            specialization_score = 0.0
            max_specialization_bonus = 0.6
            
            for domain in task_domains:
                domain_proficiency = self._get_domain_proficiency(profile, domain)
                specialization_score += domain_proficiency
            
            if task_domains:
                specialization_score /= len(task_domains)  # Average proficiency
            
            specialization_bonus = specialization_score * max_specialization_bonus
            
            # Experience factor
            experience_factor = min(1.0, profile.total_experience / 100)  # Cap at 100 tasks
            experience_bonus = experience_factor * 0.2
            
            # Consistency bonus
            consistency_bonus = profile.consistency_score * 0.1
            
            # Complexity match - some agents better with complex tasks
            complexity_match = self._calculate_complexity_match(profile.agent_role, task_complexity)
            complexity_bonus = complexity_match * 0.1
            
            # Recent performance factor
            recent_performance = await self._get_recent_performance_factor(profile)
            performance_bonus = recent_performance * 0.1
            
            total_score = (base_score + specialization_bonus + experience_bonus + 
                         consistency_bonus + complexity_bonus + performance_bonus)
            
            return max(0.0, min(1.0, total_score))
            
        except Exception as e:
            logger.error(f"Error calculating agent task score: {e}")
            return 0.3
    
    def _get_domain_proficiency(self, profile: AgentProfile, 
                              domain: SpecializationDomain) -> float:
        """Get agent's proficiency in specific domain"""
        try:
            # Check primary specializations
            for spec in profile.primary_specializations:
                if spec.domain == domain:
                    return spec.proficiency_score
            
            # Check secondary specializations
            for spec in profile.secondary_specializations:
                if spec.domain == domain:
                    return spec.proficiency_score * 0.7  # Secondary gets 70% weight
            
            # Check learning preferences
            domain_name = domain.value
            if domain_name in profile.learning_preferences:
                return profile.learning_preferences[domain_name] * 0.3  # Preference gets 30% weight
            
            return 0.1  # Minimal proficiency if no specialization
            
        except Exception as e:
            logger.error(f"Error getting domain proficiency: {e}")
            return 0.1
    
    def _calculate_complexity_match(self, agent_role: str, task_complexity: float) -> float:
        """Calculate how well agent handles specific complexity level"""
        # Agent complexity preferences
        complexity_preferences = {
            "planner": 0.8,      # Prefers complex strategic tasks
            "analyzer": 0.6,     # Good with medium complexity
            "critic": 0.7,       # Good with moderately complex reviews
            "synthesizer": 0.9,  # Excels at complex integration tasks
            "executor": 0.5      # Better with straightforward implementation
        }
        
        preferred_complexity = complexity_preferences.get(agent_role, 0.5)
        
        # Calculate match (closer to preferred = better)
        complexity_diff = abs(preferred_complexity - task_complexity)
        match_score = max(0.0, 1.0 - complexity_diff)
        
        return match_score
    
    async def _get_recent_performance_factor(self, profile: AgentProfile) -> float:
        """Get recent performance factor for agent"""
        try:
            # Get recent performance data from performance tracker
            recent_summary = self.performance_tracker.get_provider_performance_summary(
                "agent", profile.agent_role  # Treating agent as provider
            )
            
            if 'avg_success_rate' in recent_summary:
                return recent_summary['avg_success_rate']
            
            return 0.5  # Neutral if no recent data
            
        except Exception as e:
            logger.error(f"Error getting recent performance factor: {e}")
            return 0.5
    
    async def _update_domain_performance(self, profile: AgentProfile, 
                                       domain: SpecializationDomain,
                                       performance_metrics: Dict[str, float]):
        """Update performance for specific domain"""
        try:
            quality_score = performance_metrics.get('quality_score', 0.5)
            success = performance_metrics.get('success', True)
            
            # Find existing specialization or create new one
            specialization = None
            for spec in profile.primary_specializations + profile.secondary_specializations:
                if spec.domain == domain:
                    specialization = spec
                    break
            
            if not specialization:
                # Create new specialization
                specialization = SpecializationMetrics(
                    domain=domain,
                    level=SpecializationLevel.NOVICE,
                    proficiency_score=0.3,
                    task_success_rate=0.0,
                    avg_quality_score=0.0,
                    total_tasks_handled=0,
                    learning_velocity=0.0,
                    confidence_score=0.0,
                    last_updated=time.time(),
                    specialization_evidence=[]
                )
                
                # Add to secondary specializations initially
                profile.secondary_specializations.append(specialization)
            
            # Update metrics
            old_tasks = specialization.total_tasks_handled
            new_tasks = old_tasks + 1
            
            # Update averages
            old_quality = specialization.avg_quality_score
            new_quality = ((old_quality * old_tasks) + quality_score) / new_tasks
            
            old_success_rate = specialization.task_success_rate
            new_success_rate = ((old_success_rate * old_tasks) + (1.0 if success else 0.0)) / new_tasks
            
            # Calculate learning velocity (improvement rate)
            old_proficiency = specialization.proficiency_score
            new_proficiency = (new_quality * 0.7) + (new_success_rate * 0.3)
            learning_velocity = new_proficiency - old_proficiency
            
            # Update specialization
            specialization.total_tasks_handled = new_tasks
            specialization.avg_quality_score = new_quality
            specialization.task_success_rate = new_success_rate
            specialization.proficiency_score = new_proficiency
            specialization.learning_velocity = learning_velocity
            specialization.confidence_score = min(1.0, new_tasks / 20)  # Confidence grows with experience
            specialization.last_updated = time.time()
            
            # Update specialization level
            await self._update_specialization_level(specialization)
            
            # Add evidence of specialization
            if quality_score > 0.7:
                evidence = f"High quality task completion ({quality_score:.2f}) on {time.strftime('%Y-%m-%d')}"
                specialization.specialization_evidence.append(evidence)
                # Keep only recent evidence
                if len(specialization.specialization_evidence) > 10:
                    specialization.specialization_evidence = specialization.specialization_evidence[-10:]
            
        except Exception as e:
            logger.error(f"Error updating domain performance: {e}")
    
    async def _update_specialization_level(self, specialization: SpecializationMetrics):
        """Update specialization level based on performance"""
        try:
            proficiency = specialization.proficiency_score
            tasks_handled = specialization.total_tasks_handled
            success_rate = specialization.task_success_rate
            
            # Determine level based on metrics
            if (proficiency >= 0.9 and tasks_handled >= 50 and success_rate >= 0.85):
                specialization.level = SpecializationLevel.MASTER
            elif (proficiency >= 0.8 and tasks_handled >= 30 and success_rate >= 0.8):
                specialization.level = SpecializationLevel.EXPERT
            elif (proficiency >= 0.7 and tasks_handled >= 20 and success_rate >= 0.75):
                specialization.level = SpecializationLevel.ADVANCED
            elif (proficiency >= 0.6 and tasks_handled >= 10 and success_rate >= 0.7):
                specialization.level = SpecializationLevel.INTERMEDIATE
            else:
                specialization.level = SpecializationLevel.NOVICE
                
        except Exception as e:
            logger.error(f"Error updating specialization level: {e}")
    
    async def _evaluate_specialization_development(self, profile: AgentProfile):
        """Evaluate if specializations should be promoted/demoted"""
        try:
            # Move high-performing secondary specializations to primary
            to_promote = []
            for spec in profile.secondary_specializations:
                if (spec.proficiency_score >= self.specialization_threshold and
                    spec.total_tasks_handled >= self.min_tasks_for_specialization):
                    to_promote.append(spec)
            
            for spec in to_promote:
                profile.secondary_specializations.remove(spec)
                profile.primary_specializations.append(spec)
                logger.info(f"Promoted {spec.domain.value} to primary specialization for {profile.agent_role}")
            
            # Limit number of primary specializations (max 3)
            if len(profile.primary_specializations) > 3:
                # Keep top 3 by proficiency
                profile.primary_specializations.sort(
                    key=lambda s: s.proficiency_score, reverse=True
                )
                demoted = profile.primary_specializations[3:]
                profile.primary_specializations = profile.primary_specializations[:3]
                profile.secondary_specializations.extend(demoted)
            
        except Exception as e:
            logger.error(f"Error evaluating specialization development: {e}")
    
    async def _update_domain_rankings(self):
        """Update rankings of agents by domain expertise"""
        try:
            # Clear existing rankings
            self.domain_experts = {}
            
            # Build rankings for each domain
            for domain in SpecializationDomain:
                domain_rankings = []
                
                for agent_role, profile in self.agent_profiles.items():
                    proficiency = self._get_domain_proficiency(profile, domain)
                    
                    if proficiency > 0.3:  # Only include agents with some proficiency
                        domain_rankings.append((agent_role, proficiency))
                
                # Sort by proficiency
                domain_rankings.sort(key=lambda x: x[1], reverse=True)
                
                # Store top experts (up to 5)
                self.domain_experts[domain] = [agent for agent, _ in domain_rankings[:5]]
            
        except Exception as e:
            logger.error(f"Error updating domain rankings: {e}")
    
    def get_domain_experts(self, domain: SpecializationDomain, top_n: int = 3) -> List[str]:
        """Get top experts for a specific domain"""
        try:
            experts = self.domain_experts.get(domain, [])
            return experts[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting domain experts: {e}")
            return []
    
    def get_agent_specialization_report(self, agent_role: str) -> Dict[str, Any]:
        """Get detailed specialization report for agent"""
        try:
            if agent_role not in self.agent_profiles:
                return {'error': 'Agent not found'}
            
            profile = self.agent_profiles[agent_role]
            
            report = {
                'agent_role': agent_role,
                'total_experience': profile.total_experience,
                'consistency_score': profile.consistency_score,
                'adaptation_rate': profile.adaptation_rate,
                'primary_specializations': [],
                'secondary_specializations': [],
                'learning_preferences': profile.learning_preferences,
                'collaboration_effectiveness': profile.collaboration_patterns,
                'domain_rankings': {},
                'development_recommendations': []
            }
            
            # Add specialization details
            for spec in profile.primary_specializations:
                report['primary_specializations'].append({
                    'domain': spec.domain.value,
                    'level': spec.level.value,
                    'proficiency_score': spec.proficiency_score,
                    'task_success_rate': spec.task_success_rate,
                    'total_tasks': spec.total_tasks_handled,
                    'confidence': spec.confidence_score,
                    'learning_velocity': spec.learning_velocity,
                    'recent_evidence': spec.specialization_evidence[-3:]  # Last 3 pieces of evidence
                })
            
            for spec in profile.secondary_specializations:
                report['secondary_specializations'].append({
                    'domain': spec.domain.value,
                    'level': spec.level.value,
                    'proficiency_score': spec.proficiency_score,
                    'tasks_to_promotion': max(0, self.min_tasks_for_specialization - spec.total_tasks_handled)
                })
            
            # Add domain rankings
            for domain, experts in self.domain_experts.items():
                if agent_role in experts:
                    rank = experts.index(agent_role) + 1
                    report['domain_rankings'][domain.value] = rank
            
            # Generate development recommendations
            report['development_recommendations'] = self._generate_development_recommendations(profile)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating specialization report: {e}")
            return {'error': str(e)}
    
    def _generate_development_recommendations(self, profile: AgentProfile) -> List[str]:
        """Generate development recommendations for agent"""
        recommendations = []
        
        try:
            # Recommend promoting promising secondary specializations
            for spec in profile.secondary_specializations:
                if (spec.proficiency_score >= 0.6 and 
                    spec.total_tasks_handled >= self.min_tasks_for_specialization // 2):
                    tasks_needed = self.min_tasks_for_specialization - spec.total_tasks_handled
                    recommendations.append(
                        f"Focus on {spec.domain.value} tasks - need {tasks_needed} more tasks for promotion"
                    )
            
            # Recommend new learning areas
            weak_preferences = {k: v for k, v in profile.learning_preferences.items() if v < 0.4}
            if weak_preferences:
                top_weak = sorted(weak_preferences.items(), key=lambda x: x[1])[0]
                recommendations.append(
                    f"Consider developing {top_weak[0]} skills to broaden expertise"
                )
            
            # Collaboration recommendations
            weak_collaborations = {k: v for k, v in profile.collaboration_patterns.items() if v < 0.6}
            if weak_collaborations:
                recommendations.append(
                    "Improve collaboration with " + ", ".join(weak_collaborations.keys())
                )
            
            # Experience-based recommendations
            if profile.total_experience < 20:
                recommendations.append("Gain more diverse task experience to develop broader skills")
            
            return recommendations[:5]  # Top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating development recommendations: {e}")
            return []
    
    async def _store_specialization_update(self, agent_role: str, domains: List[SpecializationDomain],
                                         performance_metrics: Dict[str, float]):
        """Store specialization update to memory filesystem"""
        try:
            timestamp_str = str(int(time.time()))
            memory_path = f"/specializations/domains/{agent_role}_{timestamp_str}"
            
            content = {
                'agent_role': agent_role,
                'domains_involved': [d.value for d in domains],
                'performance_metrics': performance_metrics,
                'profile_snapshot': asdict(self.agent_profiles[agent_role]),
                'update_timestamp': time.time()
            }
            
            metadata = {
                'importance_score': 0.5 + (0.3 * performance_metrics.get('quality_score', 0.5)),
                'success_rate': 1.0 if performance_metrics.get('success', True) else 0.0,
                'tags': ['specialization', 'learning', agent_role] + [d.value for d in domains],
                'agent_annotations': {
                    'agent_role': agent_role,
                    'primary_domain': domains[0].value if domains else 'unknown',
                    'performance_quality': performance_metrics.get('quality_score', 0.5)
                },
                'performance_metrics': performance_metrics
            }
            
            self.memory_fs.store_memory(memory_path, content, metadata)
            
        except Exception as e:
            logger.error(f"Error storing specialization update: {e}")
    
    async def _load_specialization_data(self):
        """Load existing specialization data from memory filesystem"""
        try:
            specializations_node = self.memory_fs.get_memory("/specializations/domains")
            if specializations_node and specializations_node.children:
                
                for child_node in specializations_node.children.values():
                    if 'profile_snapshot' in child_node.content:
                        profile_data = child_node.content['profile_snapshot']
                        agent_role = profile_data.get('agent_role')
                        
                        if agent_role and agent_role in self.agent_profiles:
                            # Update with stored profile data
                            stored_profile = AgentProfile(**profile_data)
                            self.agent_profiles[agent_role] = stored_profile
                
                # Update domain rankings
                await self._update_domain_rankings()
                
                logger.info(f"Loaded specialization data for {len(self.agent_profiles)} agents")
            
        except Exception as e:
            logger.error(f"Error loading specialization data: {e}")
    
    async def _get_specialization_match_details(self, profile: AgentProfile, 
                                              task_domains: List[SpecializationDomain]) -> Dict[str, float]:
        """Get detailed specialization match information"""
        try:
            match_details = {}
            
            for domain in task_domains:
                proficiency = self._get_domain_proficiency(profile, domain)
                match_details[domain.value] = proficiency
            
            return match_details
            
        except Exception as e:
            logger.error(f"Error getting specialization match details: {e}")
            return {}
    
    async def _generate_assignment_reasoning(self, agent_role: str, 
                                           task_domains: List[SpecializationDomain],
                                           score: float,
                                           specialization_match: Dict[str, float]) -> str:
        """Generate human-readable reasoning for agent assignment"""
        try:
            profile = self.agent_profiles[agent_role]
            
            reasons = []
            
            # Specialization reasons
            strong_domains = [domain for domain, score in specialization_match.items() if score > 0.7]
            if strong_domains:
                reasons.append(f"Strong expertise in {', '.join(strong_domains)}")
            
            # Experience reasons
            if profile.total_experience > 50:
                reasons.append("Highly experienced")
            elif profile.total_experience > 20:
                reasons.append("Good experience level")
            
            # Performance reasons
            if profile.consistency_score > 0.8:
                reasons.append("Consistently high performance")
            
            # Primary specializations
            if profile.primary_specializations:
                primary_domains = [spec.domain.value for spec in profile.primary_specializations]
                matching_primary = [d for d in primary_domains if d in [td.value for td in task_domains]]
                if matching_primary:
                    reasons.append(f"Primary specialization in {', '.join(matching_primary)}")
            
            if not reasons:
                reasons.append("Best available option based on overall capabilities")
            
            reasoning = f"Selected {agent_role} - " + "; ".join(reasons)
            reasoning += f" (Confidence: {score:.2f})"
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Error generating assignment reasoning: {e}")
            return f"Selected {agent_role} based on overall fit"
    
    async def _update_collaboration_patterns(self, profile: AgentProfile, 
                                           collaborating_agents: List[str],
                                           quality_score: float):
        """Update collaboration effectiveness patterns"""
        try:
            for other_agent in collaborating_agents:
                if other_agent != profile.agent_role and other_agent in profile.collaboration_patterns:
                    # Update collaboration score using exponential moving average
                    old_score = profile.collaboration_patterns[other_agent]
                    new_score = 0.8 * old_score + 0.2 * quality_score
                    profile.collaboration_patterns[other_agent] = new_score
            
        except Exception as e:
            logger.error(f"Error updating collaboration patterns: {e}")