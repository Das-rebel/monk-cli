"""
Docker Optimizer
Analyzes Docker configurations and provides optimization recommendations
"""

import asyncio
import json
import time
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import re
from dataclasses import dataclass

from src.core.cache_manager import cache_manager
from src.core.async_engine import monitor_performance

logger = logging.getLogger(__name__)

@dataclass
class DockerImageInfo:
    """Information about a Docker image"""
    name: str
    tag: str
    size: str
    layers: int
    created: str
    digest: str

@dataclass
class DockerfileAnalysis:
    """Analysis of a Dockerfile"""
    base_image: str
    layers_count: int
    multi_stage: bool
    optimization_score: float
    issues: List[str]
    recommendations: List[str]

class DockerOptimizer:
    """
    Analyzes Docker configurations and provides optimization recommendations
    """
    
    def __init__(self):
        self.docker_available = False
        self.docker_version = None
        
    async def initialize(self):
        """Initialize the Docker analyzer"""
        try:
            # Check if Docker is available
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            
            self.docker_available = True
            self.docker_version = result.stdout.strip()
            logger.info(f"Docker Optimizer initialized: {self.docker_version}")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.docker_available = False
            logger.warning("Docker not available - analyzer will run in limited mode")
    
    @monitor_performance("docker_analysis")
    async def analyze(self, project_path: Path, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze Docker configurations in the project
        """
        options = options or {}
        
        if not self.docker_available:
            return {
                'error': 'Docker not available',
                'recommendation': 'Install Docker to enable full Docker analysis'
            }
        
        # Check cache
        cache_key = f"docker_analysis:{project_path.absolute()}:{hash(str(options))}"
        cached_result = await cache_manager.get(cache_key)
        if cached_result and not options.get('--no-cache'):
            return cached_result
        
        # Find Dockerfiles
        dockerfiles = list(project_path.rglob('Dockerfile*'))
        docker_compose_files = list(project_path.rglob('docker-compose*.yml')) + list(project_path.rglob('docker-compose*.yaml'))
        
        if not dockerfiles and not docker_compose_files:
            return {
                'message': 'No Docker configurations found',
                'recommendation': 'Consider containerizing your application for better deployment consistency'
            }
        
        analysis_results = {
            'dockerfiles': [],
            'docker_compose': [],
            'images': [],
            'security_scan': [],
            'optimization_score': 0,
            'recommendations': []
        }
        
        # Analyze Dockerfiles
        for dockerfile in dockerfiles:
            dockerfile_analysis = await self._analyze_dockerfile(dockerfile)
            analysis_results['dockerfiles'].append(dockerfile_analysis)
        
        # Analyze docker-compose files
        for compose_file in docker_compose_files:
            compose_analysis = await self._analyze_docker_compose(compose_file)
            analysis_results['docker_compose'].append(compose_analysis)
        
        # Analyze existing images
        if options.get('--scan-images'):
            images = await self._get_docker_images()
            analysis_results['images'] = images
        
        # Security scan
        if options.get('--security-scan'):
            security_results = await self._security_scan(project_path)
            analysis_results['security_scan'] = security_results
        
        # Calculate overall optimization score
        analysis_results['optimization_score'] = self._calculate_optimization_score(analysis_results)
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_recommendations(analysis_results)
        
        # Cache results
        await cache_manager.set(cache_key, analysis_results, ttl=3600)  # 1 hour
        
        return analysis_results
    
    async def _analyze_dockerfile(self, dockerfile_path: Path) -> DockerfileAnalysis:
        """Analyze a single Dockerfile"""
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Analyze base image
            base_image = self._extract_base_image(lines)
            
            # Count layers
            layers_count = self._count_layers(lines)
            
            # Check for multi-stage builds
            multi_stage = 'FROM' in content and content.count('FROM') > 1
            
            # Analyze optimization opportunities
            issues = []
            recommendations = []
            
            # Check for common issues
            if not base_image:
                issues.append("No base image specified")
                recommendations.append("Specify a base image using FROM instruction")
            
            if 'latest' in base_image:
                issues.append("Using 'latest' tag")
                recommendations.append("Use specific version tags for reproducible builds")
            
            if layers_count > 20:
                issues.append(f"High number of layers ({layers_count})")
                recommendations.append("Combine RUN commands to reduce layers")
            
            # Check for optimization patterns
            if 'COPY . .' in content:
                issues.append("Copying entire directory")
                recommendations.append("Use .dockerignore and copy only necessary files")
            
            if 'RUN apt-get update' in content and 'RUN apt-get install' in content:
                if not any('&&' in line for line in lines if 'RUN apt-get' in line):
                    issues.append("Separate apt-get commands")
                    recommendations.append("Combine apt-get update and install in single RUN command")
            
            # Calculate optimization score
            optimization_score = max(0, 100 - len(issues) * 15)
            
            return DockerfileAnalysis(
                base_image=base_image or "Unknown",
                layers_count=layers_count,
                multi_stage=multi_stage,
                optimization_score=optimization_score,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing Dockerfile {dockerfile_path}: {e}")
            return DockerfileAnalysis(
                base_image="Error",
                layers_count=0,
                multi_stage=False,
                optimization_score=0,
                issues=[f"Analysis failed: {str(e)}"],
                recommendations=["Fix Dockerfile syntax errors"]
            )
    
    async def _analyze_docker_compose(self, compose_path: Path) -> Dict[str, Any]:
        """Analyze a docker-compose file"""
        try:
            with open(compose_path, 'r') as f:
                content = f.read()
            
            # Basic analysis
            services_count = content.count('services:')
            volumes_count = content.count('volumes:')
            networks_count = content.count('networks:')
            
            # Check for common issues
            issues = []
            recommendations = []
            
            if 'latest' in content:
                issues.append("Using 'latest' tags")
                recommendations.append("Specify version tags for all services")
            
            if 'build: .' in content:
                issues.append("Building from current directory")
                recommendations.append("Use specific Dockerfile paths for better control")
            
            if 'restart: always' in content:
                issues.append("Always restart policy")
                recommendations.append("Consider using 'unless-stopped' instead of 'always'")
            
            return {
                'file_path': str(compose_path),
                'services_count': services_count,
                'volumes_count': volumes_count,
                'networks_count': networks_count,
                'issues': issues,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing docker-compose file {compose_path}: {e}")
            return {
                'file_path': str(compose_path),
                'error': str(e),
                'issues': [f"Analysis failed: {str(e)}"],
                'recommendations': ["Fix docker-compose file syntax errors"]
            }
    
    async def _get_docker_images(self) -> List[DockerImageInfo]:
        """Get information about Docker images"""
        try:
            result = subprocess.run(
                ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}\t{{.Digest}}'],
                capture_output=True,
                text=True,
                check=True
            )
            
            images = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        name_tag = parts[0]
                        size = parts[1]
                        created = parts[2]
                        digest = parts[3]
                        
                        if ':' in name_tag:
                            name, tag = name_tag.rsplit(':', 1)
                        else:
                            name, tag = name_tag, 'latest'
                        
                        # Count layers (approximate)
                        layers = await self._count_image_layers(name_tag)
                        
                        images.append(DockerImageInfo(
                            name=name,
                            tag=tag,
                            size=size,
                            layers=layers,
                            created=created,
                            digest=digest
                        ))
            
            return images
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting Docker images: {e}")
            return []
    
    async def _count_image_layers(self, image_name: str) -> int:
        """Count the number of layers in a Docker image"""
        try:
            result = subprocess.run(
                ['docker', 'history', '--format', '{{.CreatedBy}}', image_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Count non-empty lines
            layers = [line for line in result.stdout.split('\n') if line.strip()]
            return len(layers)
            
        except subprocess.CalledProcessError:
            return 0
    
    async def _security_scan(self, project_path: Path) -> List[Dict[str, Any]]:
        """Perform security scan on Docker images"""
        security_results = []
        
        try:
            # Find Dockerfiles to identify potential images
            dockerfiles = list(project_path.rglob('Dockerfile*'))
            
            for dockerfile in dockerfiles:
                try:
                    # Try to build and scan the image
                    image_name = f"temp-scan-{int(time.time())}"
                    
                    # Build image
                    build_result = subprocess.run(
                        ['docker', 'build', '-t', image_name, str(dockerfile.parent)],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minutes timeout
                    )
                    
                    if build_result.returncode == 0:
                        # Scan for vulnerabilities (using Trivy if available)
                        scan_result = await self._scan_image_vulnerabilities(image_name)
                        
                        security_results.append({
                            'dockerfile': str(dockerfile),
                            'image_name': image_name,
                            'vulnerabilities': scan_result,
                            'scan_success': True
                        })
                        
                        # Clean up
                        subprocess.run(['docker', 'rmi', image_name], capture_output=True)
                    else:
                        security_results.append({
                            'dockerfile': str(dockerfile),
                            'build_error': build_result.stderr,
                            'scan_success': False
                        })
                        
                except subprocess.TimeoutExpired:
                    security_results.append({
                        'dockerfile': str(dockerfile),
                        'build_error': 'Build timeout',
                        'scan_success': False
                    })
                except Exception as e:
                    security_results.append({
                        'dockerfile': str(dockerfile),
                        'build_error': str(e),
                        'scan_success': False
                    })
        
        except Exception as e:
            logger.error(f"Error during security scan: {e}")
        
        return security_results
    
    async def _scan_image_vulnerabilities(self, image_name: str) -> List[Dict[str, Any]]:
        """Scan a Docker image for vulnerabilities"""
        vulnerabilities = []
        
        try:
            # Try to use Trivy if available
            result = subprocess.run(
                ['trivy', 'image', '--format', 'json', image_name],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                try:
                    scan_data = json.loads(result.stdout)
                    if 'Results' in scan_data:
                        for result_item in scan_data['Results']:
                            if 'Vulnerabilities' in result_item:
                                for vuln in result_item['Vulnerabilities']:
                                    vulnerabilities.append({
                                        'vulnerability_id': vuln.get('VulnerabilityID', 'Unknown'),
                                        'package_name': vuln.get('PkgName', 'Unknown'),
                                        'severity': vuln.get('Severity', 'Unknown'),
                                        'description': vuln.get('Description', 'No description'),
                                        'fixed_version': vuln.get('FixedVersion', 'Not fixed')
                                    })
                except json.JSONDecodeError:
                    pass
            
            # Fallback: check for common base image vulnerabilities
            if not vulnerabilities:
                vulnerabilities = await self._check_common_vulnerabilities(image_name)
        
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Trivy not available, use fallback
            vulnerabilities = await self._check_common_vulnerabilities(image_name)
        
        return vulnerabilities
    
    async def _check_common_vulnerabilities(self, image_name: str) -> List[Dict[str, Any]]:
        """Check for common vulnerabilities in base images"""
        vulnerabilities = []
        
        try:
            # Get base image info
            result = subprocess.run(
                ['docker', 'inspect', image_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            inspect_data = json.loads(result.stdout)
            if inspect_data:
                image_info = inspect_data[0]
                
                # Check for old base images
                created = image_info.get('Created', '')
                if created:
                    # Simple check for very old images
                    if '2020' in created or '2019' in created:
                        vulnerabilities.append({
                            'vulnerability_id': 'OLD_BASE_IMAGE',
                            'package_name': 'base_image',
                            'severity': 'MEDIUM',
                            'description': 'Base image is quite old and may contain security vulnerabilities',
                            'fixed_version': 'Update to newer base image'
                        })
                
                # Check for root user
                config = image_info.get('Config', {})
                user = config.get('User', '')
                if user == '' or user == '0':
                    vulnerabilities.append({
                        'vulnerability_id': 'ROOT_USER',
                        'package_name': 'user_config',
                        'severity': 'MEDIUM',
                        'description': 'Container runs as root user',
                        'fixed_version': 'Use non-root user in Dockerfile'
                    })
        
        except Exception as e:
            logger.error(f"Error checking common vulnerabilities: {e}")
        
        return vulnerabilities
    
    def _extract_base_image(self, lines: List[str]) -> Optional[str]:
        """Extract base image from Dockerfile lines"""
        for line in lines:
            line = line.strip()
            if line.startswith('FROM '):
                # Remove FROM and get image
                image = line[5:].strip()
                # Remove AS alias if present
                if ' AS ' in image:
                    image = image.split(' AS ')[0]
                return image
        return None
    
    def _count_layers(self, lines: List[str]) -> int:
        """Count the number of layers in Dockerfile"""
        layer_instructions = ['FROM', 'RUN', 'COPY', 'ADD', 'WORKDIR', 'ENV', 'EXPOSE', 'VOLUME', 'USER', 'LABEL']
        layer_count = 0
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                for instruction in layer_instructions:
                    if line.startswith(instruction):
                        layer_count += 1
                        break
        
        return layer_count
    
    def _calculate_optimization_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall Docker optimization score"""
        if not analysis_results['dockerfiles']:
            return 0.0
        
        # Calculate average Dockerfile optimization score
        dockerfile_scores = [df['optimization_score'] for df in analysis_results['dockerfiles']]
        avg_dockerfile_score = sum(dockerfile_scores) / len(dockerfile_scores)
        
        # Bonus for multi-stage builds
        multi_stage_bonus = 10 if any(df['multi_stage'] for df in analysis_results['dockerfiles']) else 0
        
        # Penalty for security issues
        security_penalty = len(analysis_results['security_scan']) * 5
        
        # Penalty for high layer counts
        layer_penalty = 0
        for df in analysis_results['dockerfiles']:
            if df['layers_count'] > 15:
                layer_penalty += (df['layers_count'] - 15) * 2
        
        final_score = avg_dockerfile_score + multi_stage_bonus - security_penalty - layer_penalty
        return max(0.0, min(100.0, final_score))
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Dockerfile recommendations
        for dockerfile in analysis_results['dockerfiles']:
            recommendations.extend(dockerfile.recommendations)
        
        # General recommendations
        if analysis_results['optimization_score'] < 70:
            recommendations.append("Overall Docker optimization needs improvement")
        
        if not any(df['multi_stage'] for df in analysis_results['dockerfiles']):
            recommendations.append("Consider using multi-stage builds to reduce final image size")
        
        if analysis_results['security_scan']:
            recommendations.append("Address security vulnerabilities in Docker images")
        
        # Performance recommendations
        total_layers = sum(df['layers_count'] for df in analysis_results['dockerfiles'])
        if total_layers > 30:
            recommendations.append("Reduce total layers across all Dockerfiles for better performance")
        
        return list(set(recommendations))  # Remove duplicates
    
    async def get_status(self) -> Dict[str, Any]:
        """Get analyzer status"""
        return {
            'status': 'ready' if self.docker_available else 'not_available',
            'docker_version': self.docker_version,
            'docker_available': self.docker_available,
            'last_check': time.time()
        }
    
    async def shutdown(self):
        """Shutdown the analyzer"""
        logger.info("Docker Optimizer shutdown complete")
