"""
Git Analyzer
Analyzes Git repositories and workflows for best practices and optimization
"""

import asyncio
import json
import time
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.core.cache_manager import cache_manager
from src.core.async_engine import monitor_performance

logger = logging.getLogger(__name__)

@dataclass
class CommitInfo:
    """Information about a Git commit"""
    hash: str
    author: str
    date: str
    message: str
    files_changed: int
    insertions: int
    deletions: int
    branch: str

class GitAnalyzer:
    """
    Analyzes Git repositories and workflows
    """
    
    def __init__(self):
        self.git_available = False
        self.git_version = None
        
    async def initialize(self):
        """Initialize the Git analyzer"""
        try:
            result = subprocess.run(
                ['git', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            
            self.git_available = True
            self.git_version = result.stdout.strip()
            logger.info(f"Git Analyzer initialized: {self.git_version}")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.git_available = False
            logger.warning("Git not available - analyzer will run in limited mode")
    
    @monitor_performance("git_analysis")
    async def analyze(self, project_path: Path, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze Git repository and workflow"""
        options = options or {}
        
        if not self.git_available:
            return {
                'error': 'Git not available',
                'recommendation': 'Install Git to enable repository analysis'
            }
        
        git_dir = project_path / '.git'
        if not git_dir.exists():
            return {
                'error': 'Not a Git repository',
                'recommendation': 'Initialize Git repository with git init'
            }
        
        # Check cache
        cache_key = f"git_analysis:{project_path.absolute()}:{hash(str(options))}"
        cached_result = await cache_manager.get(cache_key)
        if cached_result and not options.get('--no-cache'):
            return cached_result
        
        # Perform analysis
        analysis_results = {
            'repository_info': {},
            'branch_analysis': {},
            'commit_analysis': {},
            'workflow_analysis': {},
            'overall_score': 0,
            'recommendations': []
        }
        
        # Repository information
        repo_info = await self._get_repository_info(project_path)
        analysis_results['repository_info'] = repo_info
        
        # Branch analysis
        branch_analysis = await self._analyze_branches(project_path)
        analysis_results['branch_analysis'] = branch_analysis
        
        # Commit analysis
        commit_analysis = await self._analyze_commits(project_path, options)
        analysis_results['commit_analysis'] = commit_analysis
        
        # Workflow analysis
        workflow_analysis = await self._analyze_workflow(project_path)
        analysis_results['workflow_analysis'] = workflow_analysis
        
        # Calculate overall score
        analysis_results['overall_score'] = self._calculate_overall_score(analysis_results)
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_recommendations(analysis_results)
        
        # Cache results
        await cache_manager.set(cache_key, analysis_results, ttl=1800)
        
        return analysis_results
    
    async def _get_repository_info(self, project_path: Path) -> Dict[str, Any]:
        """Get basic repository information"""
        try:
            # Get remote origin
            origin_result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                cwd=project_path,
                capture_output=True,
                text=True,
                check=True
            )
            origin_url = origin_result.stdout.strip()
            
            # Get current branch
            branch_result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=project_path,
                capture_output=True,
                text=True,
                check=True
            )
            current_branch = branch_result.stdout.strip()
            
            return {
                'origin_url': origin_url,
                'current_branch': current_branch,
                'is_bare': '.git' in str(project_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting repository info: {e}")
            return {'error': str(e)}
    
    async def _analyze_branches(self, project_path: Path) -> Dict[str, Any]:
        """Analyze Git branches"""
        try:
            # Get all branches
            branches_result = subprocess.run(
                ['git', 'branch', '-a', '--format=%(refname:short)\t%(upstream:short)\t%(upstream:track)'],
                cwd=project_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            branches = []
            current_branch = None
            
            for line in branches_result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        branch_name = parts[0]
                        upstream = parts[1] if parts[1] else None
                        tracking = parts[2] if parts[2] else None
                        
                        # Check if this is the current branch
                        is_current = branch_name.startswith('*')
                        if is_current:
                            current_branch = branch_name[1:].strip()
                            branch_name = current_branch
                        
                        # Parse tracking information
                        ahead = 0
                        behind = 0
                        if tracking:
                            if 'ahead' in tracking:
                                ahead_match = tracking.split('ahead')[1].split(',')[0].strip()
                                ahead = int(ahead_match) if ahead_match.isdigit() else 0
                            
                            if 'behind' in tracking:
                                behind_match = tracking.split('behind')[1].split(']')[0].strip()
                                behind = int(behind_match) if behind_match.isdigit() else 0
                        
                        branches.append({
                            'name': branch_name,
                            'upstream': upstream,
                            'ahead': ahead,
                            'behind': behind,
                            'is_remote': branch_name.startswith('remotes/'),
                            'is_current': is_current
                        })
            
            return {
                'branches': branches,
                'current_branch': current_branch,
                'total_branches': len(branches),
                'local_branches': len([b for b in branches if not b['is_remote']]),
                'remote_branches': len([b for b in branches if b['is_remote']])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing branches: {e}")
            return {'error': str(e)}
    
    async def _analyze_commits(self, project_path: Path, options: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Git commits"""
        try:
            # Get recent commits
            limit = options.get('--commit-limit', 100)
            
            commits_result = subprocess.run(
                ['git', 'log', f'--max-count={limit}', '--format=%H\t%an\t%aI\t%s\t%h'],
                cwd=project_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = []
            authors = {}
            
            for line in commits_result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 5:
                        commit_hash = parts[0]
                        author = parts[1]
                        date = parts[2]
                        message = parts[3]
                        short_hash = parts[4]
                        
                        # Count author contributions
                        authors[author] = authors.get(author, 0) + 1
                        
                        commits.append({
                            'hash': short_hash,
                            'full_hash': commit_hash,
                            'author': author,
                            'date': date,
                            'message': message
                        })
            
            # Calculate commit statistics
            total_commits = len(commits)
            
            return {
                'commits': commits,
                'total_commits': total_commits,
                'authors': authors,
                'top_contributors': sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing commits: {e}")
            return {'error': str(e)}
    
    async def _analyze_workflow(self, project_path: Path) -> Dict[str, Any]:
        """Analyze Git workflow and practices"""
        try:
            workflow_metrics = {
                'has_remote': False,
                'has_upstream': False,
                'uses_feature_branches': False,
                'has_tags': False,
                'workflow_score': 0
            }
            
            # Check for remote
            try:
                remote_result = subprocess.run(
                    ['git', 'remote', '-v'],
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                if remote_result.stdout.strip():
                    workflow_metrics['has_remote'] = True
                    
                    # Check for upstream
                    if 'origin' in remote_result.stdout:
                        workflow_metrics['has_upstream'] = True
            except:
                pass
            
            # Check for feature branches
            try:
                branch_result = subprocess.run(
                    ['git', 'branch', '--list'],
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                branches = [b.strip().replace('* ', '') for b in branch_result.stdout.split('\n') if b.strip()]
                feature_branches = [b for b in branches if any(pattern in b.lower() for pattern in ['feature', 'bugfix', 'hotfix', 'develop'])]
                
                if len(feature_branches) > 0:
                    workflow_metrics['uses_feature_branches'] = True
            except:
                pass
            
            # Check for tags
            try:
                tag_result = subprocess.run(
                    ['git', 'tag', '--list'],
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                if tag_result.stdout.strip():
                    workflow_metrics['has_tags'] = True
            except:
                pass
            
            # Calculate workflow score
            score = 0
            if workflow_metrics['has_remote']:
                score += 20
            if workflow_metrics['has_upstream']:
                score += 20
            if workflow_metrics['uses_feature_branches']:
                score += 25
            if workflow_metrics['has_tags']:
                score += 20
            
            workflow_metrics['workflow_score'] = score
            
            return workflow_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing workflow: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall Git repository health score"""
        base_score = 100.0
        
        # Workflow score (40%)
        workflow_score = analysis_results.get('workflow_analysis', {}).get('workflow_score', 0)
        workflow_contribution = (workflow_score / 100) * 40
        
        # Branch management (30%)
        branch_analysis = analysis_results.get('branch_analysis', {})
        branch_score = 0
        if branch_analysis.get('local_branches', 0) > 1:
            branch_score += 30
        branch_contribution = (branch_score / 30) * 30
        
        # Commit activity (30%)
        commit_analysis = analysis_results.get('commit_analysis', {})
        commit_score = 0
        if commit_analysis.get('total_commits', 0) > 0:
            commit_score += 30
        commit_contribution = (commit_score / 30) * 30
        
        final_score = workflow_contribution + branch_contribution + commit_contribution
        
        return max(0.0, min(100.0, final_score))
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Workflow recommendations
        workflow = analysis_results.get('workflow_analysis', {})
        if not workflow.get('has_remote'):
            recommendations.append("Set up a remote repository for collaboration")
        
        if not workflow.get('uses_feature_branches'):
            recommendations.append("Use feature branches for new development")
        
        if not workflow.get('has_tags'):
            recommendations.append("Use Git tags for version releases")
        
        # Branch recommendations
        branch_analysis = analysis_results.get('branch_analysis', {})
        if branch_analysis.get('local_branches', 0) <= 1:
            recommendations.append("Create feature branches for new work")
        
        # Commit recommendations
        commit_analysis = analysis_results.get('commit_analysis', {})
        if commit_analysis.get('total_commits', 0) == 0:
            recommendations.append("Make your first commit to start tracking changes")
        
        if not recommendations:
            recommendations.append("Git repository is well-configured - continue good practices")
        
        return recommendations
    
    async def get_status(self) -> Dict[str, Any]:
        """Get analyzer status"""
        return {
            'status': 'ready' if self.git_available else 'not_available',
            'git_version': self.git_version,
            'git_available': self.git_available,
            'last_check': time.time()
        }
    
    async def shutdown(self):
        """Shutdown the analyzer"""
        logger.info("Git Analyzer shutdown complete")
