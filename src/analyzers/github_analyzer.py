"""
GitHub Analyzer
Analyzes GitHub repositories for code quality, security, and best practices
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import aiohttp
import subprocess
from datetime import datetime, timedelta

from src.core.cache_manager import cache_manager
from src.core.async_engine import monitor_performance

logger = logging.getLogger(__name__)

class GitHubAnalyzer:
    """
    Analyzes GitHub repositories for various metrics and insights
    """
    
    def __init__(self):
        self.github_token = None
        self.session = None
        self.base_url = "https://api.github.com"
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = None
        
    async def initialize(self):
        """Initialize the analyzer"""
        # Try to get GitHub token from environment
        import os
        self.github_token = os.getenv('GITHUB_TOKEN')
        
        # Initialize HTTP session
        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        logger.info("GitHub Analyzer initialized")
    
    @monitor_performance("github_analysis")
    async def analyze(self, project_path: Path, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze GitHub repository for various metrics
        """
        options = options or {}
        
        # Check if this is a git repository
        git_dir = project_path / '.git'
        if not git_dir.exists():
            return {
                'error': 'Not a git repository',
                'recommendation': 'Initialize git repository or run from git project directory'
            }
        
        # Get repository information
        repo_info = await self._get_repository_info(project_path)
        if not repo_info:
            return {
                'error': 'Could not determine GitHub repository',
                'recommendation': 'Ensure remote origin points to GitHub repository'
            }
        
        # Check cache for this repository
        cache_key = f"github_analysis:{repo_info['full_name']}:{repo_info.get('default_branch', 'main')}"
        cached_result = await cache_manager.get(cache_key)
        if cached_result and not options.get('--no-cache'):
            return cached_result
        
        # Perform analysis
        analysis_results = {}
        
        # Repository metrics
        if repo_info.get('id'):
            repo_metrics = await self._analyze_repository_metrics(repo_info)
            analysis_results['repository'] = repo_metrics
        
        # Code quality analysis
        code_quality = await self._analyze_code_quality(project_path)
        analysis_results['code_quality'] = code_quality
        
        # Security analysis
        security_analysis = await self._analyze_security(project_path, repo_info)
        analysis_results['security'] = security_analysis
        
        # Workflow analysis
        workflow_analysis = await self._analyze_workflows(project_path)
        analysis_results['workflows'] = workflow_analysis
        
        # Community health
        if repo_info.get('id'):
            community_health = await self._analyze_community_health(repo_info)
            analysis_results['community'] = community_health
        
        # Compile results
        final_result = {
            'repository_info': repo_info,
            'analysis': analysis_results,
            'timestamp': time.time(),
            'recommendations': self._generate_recommendations(analysis_results)
        }
        
        # Cache results
        await cache_manager.set(cache_key, final_result, ttl=1800)  # 30 minutes
        
        return final_result
    
    async def _get_repository_info(self, project_path: Path) -> Optional[Dict[str, Any]]:
        """Get GitHub repository information from git remote"""
        try:
            # Get remote origin URL
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                cwd=project_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            remote_url = result.stdout.strip()
            
            # Parse GitHub URL
            if 'github.com' in remote_url:
                # Extract owner/repo from URL
                if remote_url.startswith('https://github.com/'):
                    parts = remote_url.replace('https://github.com/', '').split('/')
                elif remote_url.startswith('git@github.com:'):
                    parts = remote_url.replace('git@github.com:', '').split('/')
                else:
                    return None
                
                if len(parts) >= 2:
                    owner = parts[0]
                    repo = parts[1].replace('.git', '')
                    
                    # Get additional info from GitHub API
                    repo_info = await self._fetch_repository_info(owner, repo)
                    if repo_info:
                        repo_info['full_name'] = f"{owner}/{repo}"
                        repo_info['owner'] = owner
                        repo_info['name'] = repo
                        return repo_info
                    
                    # Fallback to basic info
                    return {
                        'full_name': f"{owner}/{repo}",
                        'owner': owner,
                        'name': repo,
                        'html_url': f"https://github.com/{owner}/{repo}"
                    }
        
        except subprocess.CalledProcessError:
            logger.warning("Could not get git remote origin")
        except Exception as e:
            logger.error(f"Error getting repository info: {e}")
        
        return None
    
    async def _fetch_repository_info(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Fetch repository information from GitHub API"""
        if not self.session:
            return None
        
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check rate limiting
                    self._update_rate_limit_info(response.headers)
                    
                    return {
                        'id': data.get('id'),
                        'full_name': data.get('full_name'),
                        'description': data.get('description'),
                        'language': data.get('language'),
                        'stars': data.get('stargazers_count', 0),
                        'forks': data.get('forks_count', 0),
                        'open_issues': data.get('open_issues_count', 0),
                        'default_branch': data.get('default_branch', 'main'),
                        'created_at': data.get('created_at'),
                        'updated_at': data.get('updated_at'),
                        'size': data.get('size', 0),
                        'license': data.get('license', {}).get('name') if data.get('license') else None,
                        'topics': data.get('topics', []),
                        'visibility': data.get('visibility', 'public'),
                        'archived': data.get('archived', False),
                        'disabled': data.get('disabled', False)
                    }
                elif response.status == 404:
                    logger.warning(f"Repository {owner}/{repo} not found")
                elif response.status == 403:
                    logger.warning("Rate limited or access denied")
                    self._update_rate_limit_info(response.headers)
                else:
                    logger.warning(f"GitHub API returned status {response.status}")
        
        except Exception as e:
            logger.error(f"Error fetching repository info: {e}")
        
        return None
    
    async def _analyze_repository_metrics(self, repo_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze repository-level metrics"""
        metrics = {
            'age_days': 0,
            'activity_score': 0,
            'popularity_score': 0,
            'health_score': 0
        }
        
        # Calculate repository age
        if repo_info.get('created_at'):
            created_date = datetime.fromisoformat(repo_info['created_at'].replace('Z', '+00:00'))
            age_days = (datetime.now(created_date.tzinfo) - created_date).days
            metrics['age_days'] = age_days
        
        # Calculate activity score based on recent updates
        if repo_info.get('updated_at'):
            updated_date = datetime.fromisoformat(repo_info['updated_at'].replace('Z', '+00:00'))
            days_since_update = (datetime.now(updated_date.tzinfo) - updated_date).days
            
            if days_since_update <= 7:
                metrics['activity_score'] = 100
            elif days_since_update <= 30:
                metrics['activity_score'] = 80
            elif days_since_update <= 90:
                metrics['activity_score'] = 60
            elif days_since_update <= 365:
                metrics['activity_score'] = 40
            else:
                metrics['activity_score'] = 20
        
        # Calculate popularity score
        stars = repo_info.get('stars', 0)
        forks = repo_info.get('forks', 0)
        
        if stars >= 1000:
            metrics['popularity_score'] = 100
        elif stars >= 100:
            metrics['popularity_score'] = 80
        elif stars >= 10:
            metrics['popularity_score'] = 60
        elif stars >= 1:
            metrics['popularity_score'] = 40
        else:
            metrics['popularity_score'] = 20
        
        # Calculate overall health score
        metrics['health_score'] = (
            metrics['activity_score'] * 0.4 +
            metrics['popularity_score'] * 0.3 +
            (100 if not repo_info.get('archived', False) else 0) * 0.2 +
            (100 if not repo_info.get('disabled', False) else 0) * 0.1
        )
        
        return metrics
    
    async def _analyze_code_quality(self, project_path: Path) -> Dict[str, Any]:
        """Analyze code quality metrics"""
        quality_metrics = {
            'file_count': 0,
            'line_count': 0,
            'language_distribution': {},
            'documentation_coverage': 0,
            'test_coverage': 0
        }
        
        try:
            # Count files and lines
            total_lines = 0
            file_types = {}
            
            for file_path in project_path.rglob('*'):
                if file_path.is_file() and not self._is_ignored_file(file_path):
                    quality_metrics['file_count'] += 1
                    
                    # Get file extension
                    suffix = file_path.suffix.lower()
                    if suffix:
                        file_types[suffix] = file_types.get(suffix, 0) + 1
                    
                    # Count lines (for text files)
                    if suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.md', '.txt']:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = len(f.readlines())
                                total_lines += lines
                        except:
                            pass
            
            quality_metrics['line_count'] = total_lines
            quality_metrics['language_distribution'] = file_types
            
            # Estimate documentation coverage
            doc_files = sum(1 for ext in file_types.keys() if ext in ['.md', '.txt', '.rst'])
            quality_metrics['documentation_coverage'] = min(100, (doc_files / max(1, quality_metrics['file_count'])) * 100)
            
            # Estimate test coverage
            test_files = sum(1 for ext in file_types.keys() if ext in ['.py', '.js', '.ts', '.java'] and 'test' in str(ext))
            quality_metrics['test_coverage'] = min(100, (test_files / max(1, quality_metrics['file_count'])) * 100)
        
        except Exception as e:
            logger.error(f"Error analyzing code quality: {e}")
        
        return quality_metrics
    
    async def _analyze_security(self, project_path: Path, repo_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security aspects"""
        security_metrics = {
            'dependencies_checked': False,
            'vulnerabilities_found': 0,
            'security_files_present': [],
            'recommendations': []
        }
        
        # Check for security-related files
        security_files = [
            'SECURITY.md', 'SECURITY', 'security.md',
            '.github/SECURITY.md', '.github/security.md',
            'docs/SECURITY.md', 'docs/security.md'
        ]
        
        for sec_file in security_files:
            if (project_path / sec_file).exists():
                security_metrics['security_files_present'].append(sec_file)
        
        # Check for dependency files
        dependency_files = ['requirements.txt', 'package.json', 'Pipfile', 'poetry.lock', 'Cargo.toml']
        for dep_file in dependency_files:
            if (project_path / dep_file).exists():
                security_metrics['dependencies_checked'] = True
                break
        
        # Generate security recommendations
        if not security_metrics['security_files_present']:
            security_metrics['recommendations'].append("Add SECURITY.md file for vulnerability reporting")
        
        if not security_metrics['dependencies_checked']:
            security_metrics['recommendations'].append("Implement dependency scanning for security vulnerabilities")
        
        if repo_info.get('visibility') == 'public':
            security_metrics['recommendations'].append("Review public repository for sensitive information exposure")
        
        return security_metrics
    
    async def _analyze_workflows(self, project_path: Path) -> Dict[str, Any]:
        """Analyze CI/CD workflows"""
        workflow_metrics = {
            'workflows_present': False,
            'workflow_count': 0,
            'workflow_types': [],
            'automation_score': 0
        }
        
        # Check for GitHub Actions
        github_actions_dir = project_path / '.github' / 'workflows'
        if github_actions_dir.exists():
            workflow_metrics['workflows_present'] = True
            workflow_files = list(github_actions_dir.glob('*.yml')) + list(github_actions_dir.glob('*.yaml'))
            workflow_metrics['workflow_count'] = len(workflow_files)
            
            for workflow_file in workflow_files:
                try:
                    with open(workflow_file, 'r') as f:
                        content = f.read()
                        if 'on:' in content:
                            workflow_metrics['workflow_types'].append('github-actions')
                except:
                    pass
        
        # Check for other CI/CD files
        ci_files = [
            '.travis.yml', '.circleci/config.yml', '.gitlab-ci.yml',
            'Jenkinsfile', 'azure-pipelines.yml', '.drone.yml'
        ]
        
        for ci_file in ci_files:
            if (project_path / ci_file).exists():
                workflow_metrics['workflows_present'] = True
                workflow_metrics['workflow_count'] += 1
                workflow_metrics['workflow_types'].append(ci_file.split('.')[1])
        
        # Calculate automation score
        if workflow_metrics['workflow_count'] >= 3:
            workflow_metrics['automation_score'] = 100
        elif workflow_metrics['workflow_count'] >= 2:
            workflow_metrics['automation_score'] = 80
        elif workflow_metrics['workflow_count'] >= 1:
            workflow_metrics['automation_score'] = 60
        else:
            workflow_metrics['automation_score'] = 20
        
        return workflow_metrics
    
    async def _analyze_community_health(self, repo_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze community health metrics"""
        community_metrics = {
            'contributor_count': 0,
            'issue_response_time': 0,
            'pr_merge_time': 0,
            'community_score': 0
        }
        
        # Basic community score based on available metrics
        score = 0
        
        # Stars and forks indicate community interest
        if repo_info.get('stars', 0) > 0:
            score += 20
        if repo_info.get('forks', 0) > 0:
            score += 20
        
        # Open issues indicate active maintenance
        if repo_info.get('open_issues', 0) > 0:
            score += 20
        
        # Recent updates indicate active development
        if repo_info.get('updated_at'):
            updated_date = datetime.fromisoformat(repo_info['updated_at'].replace('Z', '+00:00'))
            days_since_update = (datetime.now(updated_date.tzinfo) - updated_date).days
            
            if days_since_update <= 30:
                score += 40
            elif days_since_update <= 90:
                score += 20
        
        community_metrics['community_score'] = min(100, score)
        
        return community_metrics
    
    def _is_ignored_file(self, file_path: Path) -> bool:
        """Check if file should be ignored in analysis"""
        ignored_patterns = [
            '.git', '__pycache__', '.pytest_cache', '.mypy_cache',
            'node_modules', '.venv', 'venv', 'env', '.env',
            '.DS_Store', '*.pyc', '*.pyo', '*.pyd',
            '*.so', '*.dll', '*.dylib', '*.exe'
        ]
        
        for pattern in ignored_patterns:
            if pattern in str(file_path):
                return True
        
        return False
    
    def _update_rate_limit_info(self, headers):
        """Update rate limit information from response headers"""
        if 'X-RateLimit-Remaining' in headers:
            self.rate_limit_remaining = int(headers['X-RateLimit-Remaining'])
        
        if 'X-RateLimit-Reset' in headers:
            self.rate_limit_reset = int(headers['X-RateLimit-Reset'])
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Repository recommendations
        if 'repository' in analysis_results:
            repo_metrics = analysis_results['repository']
            if repo_metrics.get('health_score', 0) < 70:
                recommendations.append("Improve repository health by addressing activity and maintenance issues")
        
        # Code quality recommendations
        if 'code_quality' in analysis_results:
            code_quality = analysis_results['code_quality']
            if code_quality.get('documentation_coverage', 0) < 50:
                recommendations.append("Improve documentation coverage by adding README and code comments")
            
            if code_quality.get('test_coverage', 0) < 30:
                recommendations.append("Increase test coverage by adding unit and integration tests")
        
        # Security recommendations
        if 'security' in analysis_results:
            security = analysis_results['security']
            recommendations.extend(security.get('recommendations', []))
        
        # Workflow recommendations
        if 'workflows' in analysis_results:
            workflows = analysis_results['workflows']
            if workflows.get('automation_score', 0) < 60:
                recommendations.append("Implement CI/CD workflows for automated testing and deployment")
        
        return recommendations
    
    async def get_status(self) -> Dict[str, Any]:
        """Get analyzer status"""
        return {
            'status': 'ready' if self.session else 'not_initialized',
            'github_token_configured': bool(self.github_token),
            'rate_limit_remaining': self.rate_limit_remaining,
            'rate_limit_reset': self.rate_limit_reset,
            'last_check': time.time()
        }
    
    async def shutdown(self):
        """Shutdown the analyzer"""
        if self.session:
            await self.session.close()
            logger.info("GitHub Analyzer shutdown complete")
