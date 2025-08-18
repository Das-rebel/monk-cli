"""
Enhanced Analyzer Coordinator with TreeQuest Integration
Orchestrates all analyzers with AI agent intelligence
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

from src.core.cache_manager import cache_manager
from src.core.async_engine import monitor_performance

# Import TreeQuest components
from src.ai.treequest_engine import TreeQuestEngine, TreeQuestConfig
from src.ai.model_registry import ModelRegistry, ModelRole, ModelObjective

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Result from a single analyzer"""
    analyzer_name: str
    success: bool
    data: Dict[str, Any]
    execution_time: float
    metadata: Dict[str, Any]

@dataclass
class CrossToolInsight:
    """Insight derived from multiple analyzers"""
    insight_type: str
    description: str
    confidence: float
    related_analyzers: List[str]
    recommendations: List[str]
    severity: str

@dataclass
class TreeQuestAnalysisResult:
    """Enhanced analysis result with TreeQuest insights"""
    analyzer_results: Dict[str, AnalysisResult]
    cross_tool_insights: List[CrossToolInsight]
    treequest_insights: Dict[str, Any]
    overall_score: float
    priority_actions: List[str]
    risk_assessment: str
    confidence_score: float

class EnhancedAnalyzerCoordinator:
    """
    Enhanced coordinator with TreeQuest AI agent integration
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.analyzers = {}
            self.cross_tool_insights = []
            self.analysis_history = []
            self.treequest_engine = None
            self.model_registry = None
            self._initialized = True
        
    async def initialize(self):
        """Initialize all analyzers and TreeQuest engine"""
        logger.info("Starting enhanced analyzer initialization...")
        
        from src.analyzers.github_analyzer import GitHubAnalyzer
        from src.analyzers.docker_optimizer import DockerOptimizer
        from src.analyzers.npm_manager import NPMManager
        from src.analyzers.git_analyzer import GitAnalyzer
        
        self.analyzers = {
            'github': GitHubAnalyzer(),
            'docker': DockerOptimizer(),
            'npm': NPMManager(),
            'git': GitAnalyzer()
        }
        
        # Initialize all analyzers
        for name, analyzer in self.analyzers.items():
            try:
                await analyzer.initialize()
                logger.info(f"Initialized {name} analyzer")
            except Exception as e:
                logger.error(f"Failed to initialize {name} analyzer: {e}")
                del self.analyzers[name]
        
        # Initialize TreeQuest engine
        try:
            self.model_registry = ModelRegistry()
            treequest_config = TreeQuestConfig(
                max_depth=3,
                branching=4,
                rollout_budget=32,
                cost_cap_usd=0.50,
                objective="quality"
            )
            
            self.treequest_engine = TreeQuestEngine(
                self.model_registry, 
                cache_manager, 
                treequest_config
            )
            
            logger.info("TreeQuest engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TreeQuest engine: {e}")
    
    @monitor_performance("enhanced_comprehensive_analysis")
    async def comprehensive_analysis(self, project_path: Path, options: Dict[str, Any] = None) -> TreeQuestAnalysisResult:
        """Run comprehensive analysis using all available tools with TreeQuest intelligence"""
        
        options = options or {}
        start_time = time.time()
        
        logger.info(f"Starting comprehensive analysis of {project_path}")
        
        # Run all analyzers in parallel
        analysis_tasks = []
        for name, analyzer in self.analyzers.items():
            if options.get('analyzers') and name not in options['analyzers']:
                continue
            
            task = self._run_analyzer_with_retry(name, analyzer, project_path, options)
            analysis_tasks.append(task)
        
        # Execute all analyzers
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Process results and create AnalysisResult objects
        analyzer_results = {}
        for i, (name, analyzer) in enumerate(self.analyzers.items()):
            if i < len(results):
                result = results[i]
                if isinstance(result, Exception):
                    analyzer_results[name] = AnalysisResult(
                        analyzer_name=name,
                        success=False,
                        data={"error": str(result)},
                        execution_time=0.0,
                        metadata={"exception": True}
                    )
                else:
                    analyzer_results[name] = result
        
        # Use TreeQuest to synthesize insights
        treequest_insights = await self._synthesize_with_treequest(analyzer_results, project_path)
        
        # Generate cross-tool insights
        cross_tool_insights = await self._generate_cross_tool_insights(analyzer_results)
        
        # Calculate overall metrics
        overall_score = self._calculate_overall_score(analyzer_results, treequest_insights)
        priority_actions = self._extract_priority_actions(analyzer_results, treequest_insights)
        risk_assessment = self._assess_overall_risk(analyzer_results, treequest_insights)
        confidence_score = treequest_insights.get("confidence_score", 0.7)
        
        # Create enhanced result
        result = TreeQuestAnalysisResult(
            analyzer_results=analyzer_results,
            cross_tool_insights=cross_tool_insights,
            treequest_insights=treequest_insights,
            overall_score=overall_score,
            priority_actions=priority_actions,
            risk_assessment=risk_assessment,
            confidence_score=confidence_score
        )
        
        # Cache results
        cache_key = f"comprehensive_analysis:{project_path.absolute()}:{hash(str(options))}"
        await cache_manager.set(cache_key, result, ttl=1800)
        
        execution_time = time.time() - start_time
        logger.info(f"Comprehensive analysis completed in {execution_time:.2f}s")
        
        return result
    
    async def _synthesize_with_treequest(self, analyzer_results: Dict[str, AnalysisResult], project_path: Path) -> Dict[str, Any]:
        """Use TreeQuest to synthesize insights from analyzer results"""
        if not self.treequest_engine:
            return {"error": "TreeQuest engine not available"}
        
        try:
            # Prepare context for TreeQuest
            context = {
                "analyzers": {},
                "project_path": str(project_path),
                "analysis_timestamp": time.time()
            }
            
            # Convert analyzer results to serializable format
            for name, result in analyzer_results.items():
                context["analyzers"][name] = {
                    "success": result.success,
                    "data": result.data,
                    "execution_time": result.execution_time,
                    "metadata": result.metadata
                }
            
            # Execute TreeQuest synthesis
            synthesis_result = await self.treequest_engine.synthesize_insights(context)
            
            return synthesis_result
            
        except Exception as e:
            logger.error(f"TreeQuest synthesis failed: {e}")
            return {"error": str(e)}
    
    async def _generate_cross_tool_insights(self, analyzer_results: Dict[str, AnalysisResult]) -> List[CrossToolInsight]:
        """Generate insights by correlating findings across multiple analyzers"""
        insights = []
        
        # Security correlation
        security_insights = self._correlate_security_findings(analyzer_results)
        if security_insights:
            insights.extend(security_insights)
        
        # Performance correlation
        performance_insights = self._correlate_performance_findings(analyzer_results)
        if performance_insights:
            insights.extend(performance_insights)
        
        # Quality correlation
        quality_insights = self._correlate_quality_findings(analyzer_results)
        if quality_insights:
            insights.extend(quality_insights)
        
        return insights
    
    def _correlate_security_findings(self, analyzer_results: Dict[str, AnalysisResult]) -> List[CrossToolInsight]:
        """Correlate security findings across analyzers"""
        security_insights = []
        
        # Collect security-related findings
        security_findings = {}
        for name, result in analyzer_results.items():
            if result.success and "vulnerabilities" in result.data:
                security_findings[name] = result.data["vulnerabilities"]
        
        if len(security_findings) > 1:
            # Look for patterns across tools
            common_vulnerabilities = self._find_common_security_issues(security_findings)
            
            if common_vulnerabilities:
                insight = CrossToolInsight(
                    insight_type="security_correlation",
                    description=f"Found {len(common_vulnerabilities)} security issues across multiple tools",
                    confidence=0.85,
                    related_analyzers=list(security_findings.keys()),
                    recommendations=["Address cross-tool security issues", "Implement security scanning in CI/CD"],
                    severity="high"
                )
                security_insights.append(insight)
        
        return security_insights
    
    def _correlate_performance_findings(self, analyzer_results: Dict[str, AnalysisResult]) -> List[CrossToolInsight]:
        """Correlate performance findings across analyzers"""
        performance_insights = []
        
        # Collect performance-related findings
        performance_findings = {}
        for name, result in analyzer_results.items():
            if result.success:
                if "performance" in result.data:
                    performance_findings[name] = result.data["performance"]
                elif "optimization" in result.data:
                    performance_findings[name] = result.data["optimization"]
        
        if len(performance_findings) > 1:
            insight = CrossToolInsight(
                insight_type="performance_correlation",
                description="Performance optimization opportunities identified across tools",
                confidence=0.80,
                related_analyzers=list(performance_findings.keys()),
                recommendations=["Implement performance monitoring", "Optimize resource usage"],
                severity="medium"
            )
            performance_insights.append(insight)
        
        return performance_insights
    
    def _correlate_quality_findings(self, analyzer_results: Dict[str, AnalysisResult]) -> List[CrossToolInsight]:
        """Correlate quality findings across analyzers"""
        quality_insights = []
        
        # Collect quality-related findings
        quality_findings = {}
        for name, result in analyzer_results.items():
            if result.success:
                if "quality" in result.data:
                    quality_findings[name] = result.data["quality"]
                elif "recommendations" in result.data:
                    quality_findings[name] = result.data["recommendations"]
        
        if len(quality_findings) > 1:
            insight = CrossToolInsight(
                insight_type="quality_correlation",
                description="Code quality improvements identified across tools",
                confidence=0.75,
                related_analyzers=list(quality_findings.keys()),
                recommendations=["Implement code quality gates", "Add automated testing"],
                severity="medium"
            )
            quality_insights.append(insight)
        
        return quality_insights
    
    def _find_common_security_issues(self, security_findings: Dict[str, Any]) -> List[str]:
        """Find common security issues across analyzers"""
        common_issues = []
        
        # This is a simplified implementation
        # In practice, you'd want more sophisticated pattern matching
        for analyzer_name, findings in security_findings.items():
            if isinstance(findings, list):
                for finding in findings:
                    if isinstance(finding, dict) and "title" in finding:
                        common_issues.append(finding["title"])
        
        return list(set(common_issues))  # Remove duplicates
    
    def _calculate_overall_score(self, analyzer_results: Dict[str, AnalysisResult], treequest_insights: Dict[str, Any]) -> float:
        """Calculate overall project health score"""
        base_score = 100.0
        
        # Penalty for failed analyzers
        failed_analyzers = sum(1 for result in analyzer_results.values() if not result.success)
        base_score -= failed_analyzers * 10
        
        # Bonus for TreeQuest insights
        if treequest_insights and "confidence_score" in treequest_insights:
            confidence_bonus = treequest_insights["confidence_score"] * 20
            base_score += confidence_bonus
        
        # Penalty for errors in TreeQuest insights
        if treequest_insights and "error" in treequest_insights:
            base_score -= 15
        
        return max(0.0, min(100.0, base_score))
    
    def _extract_priority_actions(self, analyzer_results: Dict[str, AnalysisResult], treequest_insights: Dict[str, Any]) -> List[str]:
        """Extract priority actions from analysis results"""
        actions = []
        
        # Extract from TreeQuest insights
        if treequest_insights and "insights" in treequest_insights:
            insights = treequest_insights["insights"]
            if "priority_actions" in insights:
                actions.extend(insights["priority_actions"])
        
        # Extract from analyzer results
        for name, result in analyzer_results.items():
            if result.success and "recommendations" in result.data:
                recommendations = result.data["recommendations"]
                if isinstance(recommendations, list):
                    actions.extend(recommendations[:2])  # Top 2 recommendations per analyzer
        
        return list(set(actions))[:10]  # Remove duplicates and limit to 10
    
    def _assess_overall_risk(self, analyzer_results: Dict[str, AnalysisResult], treequest_insights: Dict[str, Any]) -> str:
        """Assess overall project risk level"""
        risk_score = 0
        
        # Risk from failed analyzers
        failed_analyzers = sum(1 for result in analyzer_results.values() if not result.success)
        risk_score += failed_analyzers * 2
        
        # Risk from TreeQuest insights
        if treequest_insights and "insights" in treequest_insights:
            insights = treequest_insights["insights"]
            if insights.get("risk_assessment") == "high":
                risk_score += 5
            elif insights.get("risk_assessment") == "medium":
                risk_score += 3
        
        # Determine risk level
        if risk_score >= 8:
            return "high"
        elif risk_score >= 4:
            return "medium"
        else:
            return "low"
    
    async def _run_analyzer_with_retry(self, name: str, analyzer: Any, project_path: Path, options: Dict[str, Any]) -> AnalysisResult:
        """Run analyzer with retry logic"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                result = await analyzer.analyze(project_path, options)
                execution_time = time.time() - start_time
                
                return AnalysisResult(
                    analyzer_name=name,
                    success=True,
                    data=result,
                    execution_time=execution_time,
                    metadata={"attempt": attempt + 1}
                )
                
            except Exception as e:
                logger.warning(f"Analyzer {name} attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return AnalysisResult(
                        analyzer_name=name,
                        success=False,
                        data={"error": str(e)},
                        execution_time=0.0,
                        metadata={"attempts": max_retries, "final_error": str(e)}
                    )
    
    async def get_analysis_summary(self, project_path: Path) -> Dict[str, Any]:
        """Get a summary of analysis results"""
        try:
            # Run comprehensive analysis
            result = await self.comprehensive_analysis(project_path)
            
            summary = {
                "project_path": str(project_path),
                "overall_score": result.overall_score,
                "risk_level": result.risk_assessment,
                "confidence": result.confidence_score,
                "analyzers_used": list(result.analyzer_results.keys()),
                "successful_analyzers": sum(1 for r in result.analyzer_results.values() if r.success),
                "total_analyzers": len(result.analyzer_results),
                "key_insights": len(result.cross_tool_insights),
                "priority_actions": len(result.priority_actions)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get analysis summary: {e}")
            return {"error": str(e)}
    
    async def get_analyzer_status(self) -> Dict[str, Any]:
        """Get status of all analyzers"""
        status = {
            "total_analyzers": len(self.analyzers),
            "available_analyzers": [],
            "unavailable_analyzers": []
        }
        
        for name, analyzer in self.analyzers.items():
            try:
                analyzer_status = await analyzer.get_status()
                if analyzer_status.get("status") == "ready":
                    status["available_analyzers"].append({
                        "name": name,
                        "status": "ready",
                        "details": analyzer_status
                    })
                else:
                    status["unavailable_analyzers"].append({
                        "name": name,
                        "status": "not_ready",
                        "details": analyzer_status
                    })
            except Exception as e:
                status["unavailable_analyzers"].append({
                    "name": name,
                    "status": "error",
                    "error": str(e)
                })
        
        return status

# Global enhanced analyzer coordinator
analyzer_coordinator = EnhancedAnalyzerCoordinator()

# Command functions for CLI
async def command_analyze(args: List[str] = None, coordinator_instance=None, **kwargs) -> Dict[str, Any]:
    """CLI command to analyze a project"""
    args = args or []
    
    if not args:
        return {
            'success': False,
            'error': 'Please provide a project path to analyze'
        }
    
    project_path = args[0]
    options = {}
    
    # Parse additional options
    for arg in args[1:]:
        if arg.startswith('--'):
            if '=' in arg:
                key, value = arg[2:].split('=', 1)
                options[key] = value
            else:
                options[arg[2:]] = True
    
    try:
        # Use the passed coordinator instance
        if coordinator_instance is None:
            return {
                'success': False,
                'error': 'No coordinator instance provided'
            }
        
        result = await coordinator_instance.analyze_project(project_path, options)
        return {
            'success': True,
            'data': result
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

async def command_github(args: List[str] = None, coordinator_instance=None, **kwargs) -> Dict[str, Any]:
    """CLI command for GitHub analysis"""
    args = args or []
    
    if not args:
        return {
            'success': False,
            'error': 'Please provide a GitHub action (analyze, metrics, quality)'
        }
    
    action = args[0]
    project_path = args[1] if len(args) > 1 else '.'
    
    try:
        if coordinator_instance is None:
            return {
                'success': False,
                'error': 'No coordinator instance provided'
            }
        
        if action == 'analyze':
            result = await coordinator_instance.analyzers['github'].analyze(Path(project_path))
            return {
                'success': True,
                'data': result
            }
        else:
            return {
                'success': False,
                'error': f'Unknown GitHub action: {action}'
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

async def command_docker(args: List[str] = None, coordinator_instance=None, **kwargs) -> Dict[str, Any]:
    """CLI command for Docker analysis"""
    args = args or []
    
    if not args:
        return {
            'success': False,
            'error': 'Please provide a Docker action (analyze, optimize)'
        }
    
    action = args[0]
    project_path = args[1] if len(args) > 1 else '.'
    
    try:
        if coordinator_instance is None:
            return {
                'success': False,
                'error': 'No coordinator instance provided'
            }
        
        if action == 'analyze':
            result = await coordinator_instance.analyzers['docker'].analyze(Path(project_path))
            return {
                'success': True,
                'data': result
            }
        elif action == 'optimize':
            result = await coordinator_instance.analyzers['docker'].analyze(Path(project_path), {'--optimize': True})
            return {
                'success': True,
                'data': result
            }
        else:
            return {
                'success': False,
                'error': f'Unknown Docker action: {action}'
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

async def command_npm(args: List[str] = None, coordinator_instance=None, **kwargs) -> Dict[str, Any]:
    """CLI command for NPM analysis"""
    args = args or []
    
    if not args:
        return {
            'success': False,
            'error': 'Please provide an NPM action (analyze, audit, update)'
        }
    
    action = args[0]
    project_path = args[1] if len(args) > 1 else '.'
    
    try:
        if coordinator_instance is None:
            return {
                'success': False,
                'error': 'No coordinator instance provided'
            }
        
        if action == 'analyze':
            options = {}
            if '--security-audit' in args:
                options['--security-audit'] = True
            if '--check-updates' in args:
                options['--check-updates'] = True
            
            result = await coordinator_instance.analyzers['npm'].analyze(Path(project_path), options)
            return {
                'success': True,
                'data': result
            }
        else:
            return {
                'success': False,
                'error': f'Unknown NPM action: {action}'
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

async def command_git(args: List[str] = None, coordinator_instance=None, **kwargs) -> Dict[str, Any]:
    """CLI command for Git analysis"""
    args = args or []
    
    if not args:
        return {
            'success': False,
            'error': 'Please provide a Git action (analyze, workflow, collaboration)'
        }
    
    action = args[0]
    project_path = args[1] if len(args) > 1 else '.'
    
    try:
        if coordinator_instance is None:
            return {
                'success': False,
                'error': 'No coordinator instance provided'
            }
        
        if action == 'analyze':
            result = await coordinator_instance.analyzers['git'].analyze(Path(project_path))
            return {
                'success': True,
                'data': result
            }
        else:
            return {
                'success': False,
                'error': f'Unknown Git action: {action}'
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Keep the old function names for backward compatibility
async def analyze(args: List[str] = None, **kwargs) -> Dict[str, Any]:
    """CLI command to analyze a project"""
    return await command_analyze(args, **kwargs)

async def github(args: List[str] = None, **kwargs) -> Dict[str, Any]:
    """CLI command for GitHub analysis"""
    return await command_github(args, **kwargs)

async def docker(args: List[str] = None, **kwargs) -> Dict[str, Any]:
    """CLI command for Docker analysis"""
    return await command_docker(args, **kwargs)

async def npm(args: List[str] = None, **kwargs) -> Dict[str, Any]:
    """CLI command for NPM analysis"""
    return await command_npm(args, **kwargs)

async def git(args: List[str] = None, **kwargs) -> Dict[str, Any]:
    """CLI command for Git analysis"""
    return await command_git(args, **kwargs)

# New AI-powered commands
async def command_ai(args: List[str] = None, coordinator_instance=None, **kwargs) -> Dict[str, Any]:
    """CLI command for AI-powered analysis and recommendations"""
    args = args or []
    
    if not args:
        return {
            'success': False,
            'error': 'Please provide an AI action (plan, fix, explain, models)'
        }
    
    action = args[0]
    project_path = args[1] if len(args) > 1 else '.'
    
    try:
        if coordinator_instance is None:
            return {
                'success': False,
                'error': 'No coordinator instance provided'
            }
        
        if action == 'plan':
            # Generate AI-powered action plan
            options = {'ai-recommendations': True}
            result = await coordinator_instance.analyze_project(project_path, options)
            
            # Extract AI insights for planning
            ai_insights = result.get('ai_insights', {})
            if ai_insights.get('error'):
                return {
                    'success': False,
                    'error': f'AI planning failed: {ai_insights["error"]}'
                }
            
            return {
                'success': True,
                'data': {
                    'action_plan': ai_insights.get('recommendations', []),
                    'priority_actions': ai_insights.get('priority_actions', []),
                    'risk_assessment': ai_insights.get('risk_assessment', 'unknown'),
                    'confidence_score': ai_insights.get('confidence_score', 0.0),
                    'treequest_metrics': ai_insights.get('treequest_metrics', {})
                }
            }
            
        elif action == 'fix':
            # Generate AI-powered fixes and recommendations
            options = {'ai-recommendations': True}
            result = await coordinator_instance.analyze_project(project_path, options)
            
            ai_insights = result.get('ai_insights', {})
            if ai_insights.get('error'):
                return {
                    'success': False,
                    'error': f'AI fix generation failed: {ai_insights["error"]}'
                }
            
            # Generate actionable fixes
            fixes = []
            for rec in ai_insights.get('recommendations', []):
                if isinstance(rec, dict) and 'recommendation' in rec:
                    fixes.append({
                        'analyzer': rec.get('analyzer', 'unknown'),
                        'issue': rec.get('recommendation', ''),
                        'fix_type': 'recommendation',
                        'priority': rec.get('priority', 'medium'),
                        'actionable': True
                    })
            
            return {
                'success': True,
                'data': {
                    'fixes': fixes,
                    'total_fixes': len(fixes),
                    'confidence_score': ai_insights.get('confidence_score', 0.0),
                    'treequest_metrics': ai_insights.get('treequest_metrics', {})
                }
            }
            
        elif action == 'explain':
            # Generate AI-powered explanation and summary
            options = {'ai-recommendations': True}
            result = await coordinator_instance.analyze_project(project_path, options)
            
            ai_insights = result.get('ai_insights', {})
            if ai_insights.get('error'):
                return {
                    'success': False,
                    'error': f'AI explanation failed: {ai_insights["error"]}'
                }
            
            return {
                'success': True,
                'data': {
                    'summary': ai_insights.get('summary', ''),
                    'key_findings': ai_insights.get('key_findings', []),
                    'executive_summary': f"Project health assessment with {ai_insights.get('confidence_score', 0.0):.1%} confidence",
                    'confidence_score': ai_insights.get('confidence_score', 0.0),
                    'treequest_metrics': ai_insights.get('treequest_metrics', {})
                }
            }
            
        elif action == 'models':
            # Show available AI models and their status
            if coordinator_instance.model_registry:
                cost_analysis = coordinator_instance.model_registry.get_cost_analysis()
                available_models = coordinator_instance.model_registry.get_available_models()
                
                # Parse format options
                format_type = 'table'  # default
                for arg in args[1:]:
                    if arg.startswith('--format='):
                        format_type = arg.split('=', 1)[1]
                    elif arg == '--format':
                        # Handle --format table format
                        format_type = 'table'
                
                # Format the output based on requested format
                if format_type == 'table':
                    # Create formatted table data
                    formatted_data = {
                        'format': 'table',
                        'title': 'AI Models Status',
                        'headers': ['Model', 'Provider', 'Status', 'Cost/1K Input', 'Cost/1K Output', 'Quality', 'Capabilities'],
                        'rows': []
                    }
                    
                    for model in available_models:
                        status = "✅ Available" if model.is_available else "❌ Unavailable"
                        cost_input = f"${model.cost_per_1k_tokens_input:.6f}"
                        cost_output = f"${model.cost_per_1k_tokens_output:.6f}"
                        quality = f"{model.quality_score:.2f}"
                        capabilities = ", ".join([cap.value for cap in model.capabilities])
                        
                        formatted_data['rows'].append([
                            model.name,
                            model.provider,
                            status,
                            cost_input,
                            cost_output,
                            quality,
                            capabilities
                        ])
                    
                    # Add summary rows
                    formatted_data['summary'] = {
                        'available_models': len(available_models),
                        'total_models': cost_analysis['total_models'],
                        'cost_ranges': cost_analysis['cost_ranges']
                    }
                    
                    return {
                        'success': True,
                        'data': formatted_data
                    }
                else:
                    # Return raw data for other formats
                    return {
                        'success': True,
                        'data': {
                            'available_models': len(available_models),
                            'total_models': cost_analysis['total_models'],
                            'models_by_provider': cost_analysis['models_by_provider'],
                            'cost_ranges': cost_analysis['cost_ranges'],
                            'model_details': [
                                {
                                    'name': model.name,
                                    'provider': model.provider,
                                    'capabilities': [cap.value for cap in model.capabilities],
                                    'quality_score': model.quality_score,
                                    'latency_ms': model.latency_ms,
                                    'cost_per_1k_input': model.cost_per_1k_tokens_input,
                                    'cost_per_1k_output': model.cost_per_1k_tokens_output
                                }
                                for model in available_models
                            ]
                        }
                    }
            else:
                return {
                    'success': False,
                    'error': 'Model registry not available'
                }
        
        else:
            return {
                'success': False,
                'error': f'Unknown AI action: {action}. Available actions: plan, fix, explain, models'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

async def ai(args: List[str] = None, **kwargs) -> Dict[str, Any]:
    """CLI command for AI-powered analysis and recommendations"""
    return await command_ai(args, **kwargs)
