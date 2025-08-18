"""
NPM Manager
Analyzes Node.js projects and package dependencies for security, updates, and best practices
"""

import asyncio
import json
import time
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import aiohttp
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.core.cache_manager import cache_manager
from src.core.async_engine import monitor_performance

logger = logging.getLogger(__name__)

@dataclass
class PackageInfo:
    """Information about an NPM package"""
    name: str
    version: str
    latest_version: str
    outdated: bool
    vulnerabilities: List[Dict[str, Any]]
    license: str
    dependencies: int
    dev_dependencies: int

@dataclass
class SecurityVulnerability:
    """Security vulnerability information"""
    package_name: str
    severity: str
    title: str
    description: str
    cwe: str
    cve: str
    fixed_in: str

class NPMManager:
    """
    Analyzes Node.js projects and NPM packages
    """
    
    def __init__(self):
        self.npm_available = False
        self.npm_version = None
        self.session = None
        
    async def initialize(self):
        """Initialize the NPM analyzer"""
        try:
            # Check if NPM is available
            result = subprocess.run(
                ['npm', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            
            self.npm_available = True
            self.npm_version = result.stdout.strip()
            logger.info(f"NPM Manager initialized: {self.npm_version}")
            
            # Initialize HTTP session for registry queries
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.npm_available = False
            logger.warning("NPM not available - analyzer will run in limited mode")
    
    @monitor_performance("npm_analysis")
    async def analyze(self, project_path: Path, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze Node.js project and NPM packages
        """
        options = options or {}
        
        if not self.npm_available:
            return {
                'error': 'NPM not available',
                'recommendation': 'Install Node.js and NPM to enable package analysis'
            }
        
        # Check if this is a Node.js project
        package_json = project_path / 'package.json'
        if not package_json.exists():
            return {
                'message': 'No package.json found',
                'recommendation': 'Initialize Node.js project with npm init'
            }
        
        # Check cache
        cache_key = f"npm_analysis:{project_path.absolute()}:{hash(str(options))}"
        cached_result = await cache_manager.get(cache_key)
        if cached_result and not options.get('--no-cache'):
            return cached_result
        
        # Perform analysis
        analysis_results = {
            'project_info': {},
            'dependencies': [],
            'dev_dependencies': [],
            'outdated_packages': [],
            'vulnerabilities': [],
            'audit_results': {},
            'scripts': {},
            'engines': {},
            'overall_score': 0,
            'recommendations': []
        }
        
        # Analyze package.json
        package_analysis = await self._analyze_package_json(package_json)
        analysis_results['project_info'] = package_analysis
        
        # Get dependency information
        dependencies = await self._get_dependencies(project_path)
        analysis_results['dependencies'] = dependencies.get('dependencies', [])
        analysis_results['dev_dependencies'] = dependencies.get('dev_dependencies', [])
        
        # Check for outdated packages
        if options.get('--check-updates'):
            outdated = await self._check_outdated_packages(project_path)
            analysis_results['outdated_packages'] = outdated
        
        # Security audit
        if options.get('--security-audit'):
            audit_results = await self._security_audit(project_path)
            analysis_results['audit_results'] = audit_results
            analysis_results['vulnerabilities'] = audit_results.get('vulnerabilities', [])
        
        # Calculate overall score
        analysis_results['overall_score'] = self._calculate_overall_score(analysis_results)
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_recommendations(analysis_results)
        
        # Cache results
        await cache_manager.set(cache_key, analysis_results, ttl=1800)  # 30 minutes
        
        return analysis_results
    
    async def _analyze_package_json(self, package_json_path: Path) -> Dict[str, Any]:
        """Analyze package.json file"""
        try:
            with open(package_json_path, 'r') as f:
                package_data = json.load(f)
            
            analysis = {
                'name': package_data.get('name', 'Unknown'),
                'version': package_data.get('version', '0.0.0'),
                'description': package_data.get('description', ''),
                'main': package_data.get('main', ''),
                'scripts': package_data.get('scripts', {}),
                'engines': package_data.get('engines', {}),
                'license': package_data.get('license', ''),
                'repository': package_data.get('repository', {}),
                'keywords': package_data.get('keywords', []),
                'author': package_data.get('author', ''),
                'homepage': package_data.get('homepage', ''),
                'bugs': package_data.get('bugs', {}),
                'dependencies_count': len(package_data.get('dependencies', {})),
                'dev_dependencies_count': len(package_data.get('devDependencies', {})),
                'peer_dependencies_count': len(package_data.get('peerDependencies', {})),
                'optional_dependencies_count': len(package_data.get('optionalDependencies', {})),
                'scripts_count': len(package_data.get('scripts', {}))
            }
            
            # Check for common issues
            issues = []
            if not analysis['description']:
                issues.append("Missing project description")
            
            if not analysis['license']:
                issues.append("Missing license information")
            
            if not analysis['repository']:
                issues.append("Missing repository information")
            
            if not analysis['main']:
                issues.append("Missing main entry point")
            
            analysis['issues'] = issues
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing package.json: {e}")
            return {
                'error': str(e),
                'issues': [f"Failed to parse package.json: {str(e)}"]
            }
    
    async def _get_dependencies(self, project_path: Path) -> Dict[str, List[PackageInfo]]:
        """Get detailed dependency information"""
        dependencies = {'dependencies': [], 'dev_dependencies': []}
        
        try:
            # Get dependency tree
            result = subprocess.run(
                ['npm', 'list', '--json', '--depth=0'],
                cwd=project_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            dep_tree = json.loads(result.stdout)
            
            # Process dependencies
            if 'dependencies' in dep_tree:
                for name, info in dep_tree['dependencies'].items():
                    if isinstance(info, dict):
                        package_info = PackageInfo(
                            name=name,
                            version=info.get('version', 'Unknown'),
                            latest_version=info.get('version', 'Unknown'),  # Will be updated later
                            outdated=False,
                            vulnerabilities=[],
                            license=info.get('license', 'Unknown'),
                            dependencies=len(info.get('dependencies', {})),
                            dev_dependencies=0
                        )
                        dependencies['dependencies'].append(package_info)
            
            # Get dev dependencies separately
            try:
                dev_result = subprocess.run(
                    ['npm', 'list', '--json', '--depth=0', '--dev'],
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                dev_tree = json.loads(dev_result.stdout)
                
                if 'dependencies' in dev_tree:
                    for name, info in dev_tree['dependencies'].items():
                        if isinstance(info, dict):
                            package_info = PackageInfo(
                                name=name,
                                version=info.get('version', 'Unknown'),
                                latest_version=info.get('version', 'Unknown'),
                                outdated=False,
                                vulnerabilities=[],
                                license=info.get('license', 'Unknown'),
                                dependencies=len(info.get('dependencies', {})),
                                dev_dependencies=0
                            )
                            dependencies['dev_dependencies'].append(package_info)
            
            except subprocess.CalledProcessError:
                logger.warning("Could not get dev dependencies")
        
        except Exception as e:
            logger.error(f"Error getting dependencies: {e}")
        
        return dependencies
    
    async def _check_outdated_packages(self, project_path: Path) -> List[Dict[str, Any]]:
        """Check for outdated packages"""
        outdated_packages = []
        
        try:
            result = subprocess.run(
                ['npm', 'outdated', '--json'],
                cwd=project_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            outdated_data = json.loads(result.stdout)
            
            for package_name, info in outdated_data.items():
                if isinstance(info, dict):
                    outdated_packages.append({
                        'name': package_name,
                        'current': info.get('current', 'Unknown'),
                        'wanted': info.get('wanted', 'Unknown'),
                        'latest': info.get('latest', 'Unknown'),
                        'location': info.get('location', ''),
                        'dependency_type': info.get('dependencyType', 'Unknown')
                    })
        
        except subprocess.CalledProcessError:
            # No outdated packages or error
            pass
        except Exception as e:
            logger.error(f"Error checking outdated packages: {e}")
        
        return outdated_packages
    
    async def _security_audit(self, project_path: Path) -> Dict[str, Any]:
        """Perform security audit"""
        audit_results = {
            'vulnerabilities': [],
            'summary': {},
            'metadata': {}
        }
        
        try:
            # Run npm audit
            result = subprocess.run(
                ['npm', 'audit', '--json'],
                cwd=project_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            audit_data = json.loads(result.stdout)
            
            # Extract vulnerabilities
            if 'vulnerabilities' in audit_data:
                for package_name, vuln_info in audit_data['vulnerabilities'].items():
                    if isinstance(vuln_info, dict):
                        for vuln_id, details in vuln_info.items():
                            if isinstance(details, dict):
                                vulnerability = SecurityVulnerability(
                                    package_name=package_name,
                                    severity=details.get('severity', 'Unknown'),
                                    title=details.get('title', ''),
                                    description=details.get('description', ''),
                                    cwe=details.get('cwe', ''),
                                    cve=details.get('cve', ''),
                                    fixed_in=details.get('fixedIn', 'Not fixed')
                                )
                                audit_results['vulnerabilities'].append(vulnerability)
            
            # Extract summary
            if 'metadata' in audit_data:
                audit_results['metadata'] = audit_data['metadata']
                
                if 'vulnerabilities' in audit_data['metadata']:
                    audit_results['summary'] = audit_data['metadata']['vulnerabilities']
        
        except subprocess.CalledProcessError as e:
            # No vulnerabilities found or error
            if e.returncode == 1:  # NPM audit returns 1 when vulnerabilities found
                # Try to parse the error output for vulnerability info
                try:
                    error_data = json.loads(e.stdout)
                    if 'vulnerabilities' in error_data:
                        audit_results['vulnerabilities'] = error_data['vulnerabilities']
                except:
                    pass
            else:
                logger.warning(f"NPM audit failed: {e}")
        
        except Exception as e:
            logger.error(f"Error during security audit: {e}")
        
        return audit_results
    
    async def _get_package_info_from_registry(self, package_name: str) -> Optional[Dict[str, Any]]:
        """Get package information from NPM registry"""
        if not self.session:
            return None
        
        try:
            url = f"https://registry.npmjs.org/{package_name}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Get latest version
                    versions = data.get('versions', {})
                    if versions:
                        latest_version = list(versions.keys())[-1]
                        latest_info = versions[latest_version]
                        
                        return {
                            'latest_version': latest_version,
                            'license': latest_info.get('license', 'Unknown'),
                            'description': latest_info.get('description', ''),
                            'homepage': latest_info.get('homepage', ''),
                            'repository': latest_info.get('repository', {}),
                            'keywords': latest_info.get('keywords', []),
                            'author': latest_info.get('author', ''),
                            'maintainers': latest_info.get('maintainers', []),
                            'publish_time': latest_info.get('time', {}).get(latest_version, ''),
                            'dependencies': len(latest_info.get('dependencies', {})),
                            'dev_dependencies': len(latest_info.get('devDependencies', {}))
                        }
        
        except Exception as e:
            logger.error(f"Error fetching package info for {package_name}: {e}")
        
        return None
    
    def _calculate_overall_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall project health score"""
        base_score = 100.0
        
        # Penalty for vulnerabilities
        vulnerabilities = analysis_results.get('vulnerabilities', [])
        critical_vulns = len([v for v in vulnerabilities if v.severity == 'critical'])
        high_vulns = len([v for v in vulnerabilities if v.severity == 'high'])
        medium_vulns = len([v for v in vulnerabilities if v.severity == 'medium'])
        
        base_score -= critical_vulns * 20
        base_score -= high_vulns * 10
        base_score -= medium_vulns * 5
        
        # Penalty for outdated packages
        outdated_count = len(analysis_results.get('outdated_packages', []))
        base_score -= min(outdated_count * 2, 20)
        
        # Penalty for missing project info
        project_info = analysis_results.get('project_info', {})
        issues = project_info.get('issues', [])
        base_score -= len(issues) * 5
        
        # Bonus for good practices
        if project_info.get('scripts_count', 0) >= 3:
            base_score += 5
        
        if project_info.get('license'):
            base_score += 5
        
        if project_info.get('repository'):
            base_score += 5
        
        return max(0.0, min(100.0, base_score))
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Security recommendations
        vulnerabilities = analysis_results.get('vulnerabilities', [])
        if vulnerabilities:
            critical_count = len([v for v in vulnerabilities if v.severity == 'critical'])
            high_count = len([v for v in vulnerabilities if v.severity == 'high'])
            
            if critical_count > 0:
                recommendations.append(f"Address {critical_count} critical security vulnerabilities immediately")
            
            if high_count > 0:
                recommendations.append(f"Fix {high_count} high-severity security vulnerabilities")
            
            recommendations.append("Run 'npm audit fix' to automatically fix vulnerabilities")
        
        # Update recommendations
        outdated_packages = analysis_results.get('outdated_packages', [])
        if outdated_packages:
            recommendations.append(f"Update {len(outdated_packages)} outdated packages")
            recommendations.append("Run 'npm update' to update packages within version ranges")
            recommendations.append("Review breaking changes before major version updates")
        
        # Project quality recommendations
        project_info = analysis_results.get('project_info', {})
        issues = project_info.get('issues', [])
        
        for issue in issues:
            if "Missing project description" in issue:
                recommendations.append("Add a clear project description in package.json")
            
            if "Missing license" in issue:
                recommendations.append("Specify a license for your project")
            
            if "Missing repository" in issue:
                recommendations.append("Add repository information to package.json")
            
            if "Missing main entry point" in issue:
                recommendations.append("Specify the main entry point for your package")
        
        # General recommendations
        if analysis_results.get('overall_score', 0) < 70:
            recommendations.append("Overall project health needs improvement - address the above issues")
        
        if not recommendations:
            recommendations.append("Project appears to be in good health - continue monitoring")
        
        return recommendations
    
    async def get_status(self) -> Dict[str, Any]:
        """Get analyzer status"""
        return {
            'status': 'ready' if self.npm_available else 'not_available',
            'npm_version': self.npm_version,
            'npm_available': self.npm_available,
            'last_check': time.time()
        }
    
    async def shutdown(self):
        """Shutdown the analyzer"""
        if self.session:
            await self.session.close()
        logger.info("NPM Manager shutdown complete")
