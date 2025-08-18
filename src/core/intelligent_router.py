"""
Intelligent Router
Routes queries to optimal AI providers based on query type and context
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from src.core.conversation_manager import conversation_manager

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries for routing decisions"""
    CODE_ANALYSIS = "code_analysis"
    CODE_GENERATION = "code_generation"
    FILE_OPERATION = "file_operation"
    GIT_OPERATION = "git_operation"
    GENERAL_QUESTION = "general_question"
    PROJECT_ANALYSIS = "project_analysis"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    SYSTEM_COMMAND = "system_command"
    COMPLEX_TASKS = "complex_tasks"

@dataclass
class RoutingDecision:
    """Result of routing analysis"""
    primary_provider: str
    fallback_providers: List[str]
    confidence: float
    reasoning: str
    query_type: QueryType
    metadata: Dict[str, Any]

class IntelligentRouter:
    """
    Routes queries to optimal AI providers based on analysis
    """
    
    def __init__(self):
        # Provider capabilities mapping
        self.provider_capabilities = {
            'claude_code': {
                'strengths': ['code_analysis', 'file_operation', 'debugging', 'documentation'],
                'performance': 0.95,
                'availability': self._check_claude_availability
            },
            'treequest': {
                'strengths': ['code_generation', 'project_analysis', 'complex_tasks'],
                'performance': 0.90,
                'availability': self._check_treequest_availability
            },
            'gemma': {
                'strengths': ['general_question', 'documentation', 'explanation'],
                'performance': 0.85,
                'availability': self._check_gemma_availability
            },
            'opendia': {
                'strengths': ['general_question', 'conversation'],
                'performance': 0.80,
                'availability': self._check_opendia_availability
            },
            'mcp': {
                'strengths': ['system_command', 'tool_integration'],
                'performance': 0.90,
                'availability': self._check_mcp_availability
            }
        }
        
        # Query patterns for classification
        self.query_patterns = {
            QueryType.CODE_ANALYSIS: [
                r'analyze.*code',
                r'review.*file',
                r'check.*syntax',
                r'find.*bugs?',
                r'lint',
                r'static.*analysis'
            ],
            QueryType.CODE_GENERATION: [
                r'write.*function',
                r'create.*class',
                r'generate.*code',
                r'implement.*',
                r'build.*feature'
            ],
            QueryType.FILE_OPERATION: [
                r'/cat',
                r'/ls',
                r'/tree',
                r'show.*file',
                r'read.*file',
                r'list.*directory'
            ],
            QueryType.GIT_OPERATION: [
                r'/git',
                r'git.*status',
                r'git.*diff',
                r'commit',
                r'push',
                r'pull'
            ],
            QueryType.GENERAL_QUESTION: [
                r'what.*is',
                r'how.*does',
                r'why.*do',
                r'explain',
                r'describe',
                r'tell.*me'
            ],
            QueryType.PROJECT_ANALYSIS: [
                r'analyze.*project',
                r'project.*overview',
                r'codebase.*analysis',
                r'architecture.*review',
                r'performance.*analysis'
            ],
            QueryType.DEBUGGING: [
                r'debug',
                r'error.*fix',
                r'bug.*fix',
                r'troubleshoot',
                r'issue.*resolution'
            ],
            QueryType.DOCUMENTATION: [
                r'document',
                r'readme',
                r'api.*docs',
                r'code.*comments',
                r'write.*docs'
            ],
            QueryType.SYSTEM_COMMAND: [
                r'/help',
                r'/clear',
                r'/settings',
                r'/providers',
                r'/history'
            ],
            QueryType.COMPLEX_TASKS: [
                r'optimize.*project',
                r'security.*audit',
                r'performance.*optimization',
                r'code.*refactoring',
                r'architecture.*redesign',
                r'strategic.*planning',
                r'comprehensive.*analysis',
                r'cross.*tool.*integration'
            ]
        }
    
    async def route_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Route query to optimal AI provider based on analysis
        """
        if context is None:
            context = {}
        
        # Classify query type
        query_type = self._classify_query(query)
        
        # Score providers based on query type and context
        provider_scores = {}
        
        for provider, capabilities in self.provider_capabilities.items():
            score = 0.0
            
            # Base score from performance
            score += capabilities['performance']
            
            # Query type matching
            if query_type.value in capabilities['strengths']:
                score += 0.3  # Significant boost for matching strengths
            
            # Special handling for TreeQuest
            if provider == 'treequest' and query_type in [
                QueryType.PROJECT_ANALYSIS, 
                QueryType.CODE_GENERATION, 
                QueryType.COMPLEX_TASKS
            ]:
                score += 0.4  # Extra boost for TreeQuest's core strengths
            
            # Provider-specific bonuses
            if query_type == QueryType.CODE_ANALYSIS and provider == 'claude_code':
                score += 0.2
            elif query_type == QueryType.SYSTEM_COMMAND and provider == 'mcp':
                score += 0.2
            elif query_type == QueryType.GENERAL_QUESTION and provider == 'gemma':
                score += 0.15
            
            # Context-based adjustments
            if context.get('project_type') == 'Python' and provider in ['claude_code', 'treequest']:
                score += 0.1
            
            # TreeQuest priority when available
            if provider == 'treequest' and context.get('treequest_enabled', False):
                score += 0.5  # Significant boost when TreeQuest is explicitly enabled
            
            # Availability check (async)
            try:
                is_available = await capabilities['availability']()
                if not is_available:
                    score *= 0.3  # Penalize unavailable providers
            except:
                score *= 0.5  # Penalize providers with availability check errors
            
            provider_scores[provider] = score
        
        # Sort providers by score
        sorted_providers = sorted(provider_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_providers:
            # Fallback to default
            return RoutingDecision(
                primary_provider='claude_code',
                fallback_providers=['gemma', 'opendia'],
                confidence=0.5,
                reasoning="Default fallback - no providers scored",
                query_type=query_type,
                metadata={}
            )
        
        primary_provider, primary_score = sorted_providers[0]
        fallback_providers = [p for p, _ in sorted_providers[1:4]]  # Top 3 alternatives
        
        # Generate reasoning
        reasoning = f"Selected {primary_provider} for {query_type.value} (score: {primary_score:.2f})"
        if query_type.value in self.provider_capabilities[primary_provider]['strengths']:
            reasoning += f" - matches provider strengths"
        
        return RoutingDecision(
            primary_provider=primary_provider,
            fallback_providers=fallback_providers,
            confidence=min(primary_score, 1.0),
            reasoning=reasoning,
            query_type=query_type,
            metadata={
                'scores': provider_scores,
                'context_used': bool(context)
            }
        )
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify query type based on patterns"""
        
        # Check for slash commands first (highest priority)
        if query.startswith('/'):
            if any(cmd in query for cmd in ['help', 'clear', 'settings', 'workspace', 'providers']):
                return QueryType.SYSTEM_COMMAND
            elif any(cmd in query for cmd in ['cat', 'ls', 'tree']):
                return QueryType.FILE_OPERATION
            elif any(cmd in query for cmd in ['git', 'status', 'diff']):
                return QueryType.GIT_OPERATION
            elif 'analyze' in query:
                return QueryType.PROJECT_ANALYSIS
        
        # Pattern matching for other queries
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return query_type
        
        # Default to general question
        return QueryType.GENERAL_QUESTION
    
    async def _select_optimal_provider(self, query_type: QueryType, query: str, context: Dict[str, Any]) -> RoutingDecision:
        """Select the best provider for the query type"""
        
        # Score providers based on capabilities
        provider_scores = {}
        
        for provider, capabilities in self.provider_capabilities.items():
            score = 0.0
            
            # Base score from query type match
            if query_type.value in capabilities['strengths']:
                score += capabilities['performance'] * 0.8
            
            # Bonus for specific query patterns
            if query_type == QueryType.CODE_ANALYSIS and provider == 'claude_code':
                score += 0.2
            elif query_type == QueryType.CODE_GENERATION and provider == 'treequest':
                score += 0.2
            elif query_type == QueryType.SYSTEM_COMMAND and provider == 'mcp':
                score += 0.2
            elif query_type == QueryType.GENERAL_QUESTION and provider == 'gemma':
                score += 0.15
            
            # Context-based adjustments
            if context.get('project_type') == 'Python' and provider in ['claude_code', 'treequest']:
                score += 0.1
            
            # Availability check (async)
            try:
                is_available = await capabilities['availability']()
                if not is_available:
                    score *= 0.3  # Penalize unavailable providers
            except:
                score *= 0.5  # Penalize providers with availability check errors
            
            provider_scores[provider] = score
        
        # Sort providers by score
        sorted_providers = sorted(provider_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_providers:
            # Fallback to default
            return RoutingDecision(
                primary_provider='claude_code',
                fallback_providers=['gemma', 'opendia'],
                confidence=0.5,
                reasoning="Default fallback - no providers scored",
                query_type=query_type,
                metadata={}
            )
        
        primary_provider, primary_score = sorted_providers[0]
        fallback_providers = [p for p, _ in sorted_providers[1:4]]  # Top 3 alternatives
        
        # Generate reasoning
        reasoning = f"Selected {primary_provider} for {query_type.value} (score: {primary_score:.2f})"
        if query_type.value in self.provider_capabilities[primary_provider]['strengths']:
            reasoning += f" - matches provider strengths"
        
        return RoutingDecision(
            primary_provider=primary_provider,
            fallback_providers=fallback_providers,
            confidence=min(primary_score, 1.0),
            reasoning=reasoning,
            query_type=query_type,
            metadata={
                'scores': provider_scores,
                'context_used': bool(context)
            }
        )
    
    # Provider availability checks
    async def _check_claude_availability(self) -> bool:
        """Check if Claude Code is available"""
        try:
            import subprocess
            result = subprocess.run(['which', 'claude'], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    async def _check_treequest_availability(self) -> bool:
        """Check if TreeQuest is available"""
        try:
            # Check if treequest modules exist
            from src.ai.treequest_engine import TreeQuestEngine
            return True
        except ImportError:
            return False
    
    async def _check_gemma_availability(self) -> bool:
        """Check if Gemma is available"""
        # TODO: Implement actual availability check
        return True
    
    async def _check_opendia_availability(self) -> bool:
        """Check if OpenDia is available"""
        # TODO: Implement actual availability check
        return True
    
    async def _check_mcp_availability(self) -> bool:
        """Check if MCP tools are available"""
        # TODO: Implement actual availability check
        return True
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        status = {}
        
        for provider, capabilities in self.provider_capabilities.items():
            status[provider] = {
                'strengths': capabilities['strengths'],
                'performance': capabilities['performance'],
                'available': None  # Would be populated by async check
            }
        
        return status
    
    def explain_routing(self, decision: RoutingDecision) -> str:
        """Provide human-readable explanation of routing decision"""
        explanation = f"""🧠 **Routing Decision**:
- **Selected Provider**: {decision.primary_provider}
- **Confidence**: {decision.confidence:.1%}
- **Query Type**: {decision.query_type.value}
- **Reasoning**: {decision.reasoning}"""
        
        if decision.fallback_providers:
            fallbacks = ", ".join(decision.fallback_providers)
            explanation += f"\n- **Fallback Options**: {fallbacks}"
        
        return explanation

# Global intelligent router
intelligent_router = IntelligentRouter()
