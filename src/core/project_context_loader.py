"""
Project Context Loader
Automatically detects project type and loads relevant context
"""

import asyncio
import json
import os
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

from src.core.conversation_manager import ProjectContext, conversation_manager

logger = logging.getLogger(__name__)

class ProjectContextLoader:
    """
    Automatically loads and maintains project context
    """
    
    def __init__(self):
        self.project_type_indicators = {
            'Python': {
                'files': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile'],
                'dirs': ['venv', '.venv', '__pycache__'],
                'extensions': ['.py']
            },
            'Node.js': {
                'files': ['package.json', 'yarn.lock', 'package-lock.json'],
                'dirs': ['node_modules', '.npm'],
                'extensions': ['.js', '.ts', '.jsx', '.tsx']
            },
            'Java': {
                'files': ['pom.xml', 'build.gradle', 'gradle.properties'],
                'dirs': ['target', 'build', '.gradle'],
                'extensions': ['.java']
            },
            'Rust': {
                'files': ['Cargo.toml', 'Cargo.lock'],
                'dirs': ['target'],
                'extensions': ['.rs']
            },
            'Go': {
                'files': ['go.mod', 'go.sum'],
                'dirs': ['vendor'],
                'extensions': ['.go']
            },
            'C++': {
                'files': ['CMakeLists.txt', 'Makefile', 'configure.ac'],
                'dirs': ['build', 'cmake-build-debug'],
                'extensions': ['.cpp', '.cc', '.cxx', '.c', '.h', '.hpp']
            },
            'Web': {
                'files': ['index.html', 'webpack.config.js', '.babelrc'],
                'dirs': ['dist', 'build', 'public'],
                'extensions': ['.html', '.css', '.js', '.ts']
            }
        }
        
        self.important_files = [
            'README.md', 'README.rst', 'README.txt',
            'CHANGELOG.md', 'CHANGELOG.txt',
            'LICENSE', 'LICENSE.md', 'LICENSE.txt',
            'CONTRIBUTING.md',
            '.gitignore', '.dockerignore',
            'Dockerfile', 'docker-compose.yml',
            '.env', '.env.example'
        ]
    
    async def load_project_context(self, project_path: Optional[str] = None) -> ProjectContext:
        """Load complete project context"""
        project_path = project_path or str(Path.cwd())
        project_dir = Path(project_path)
        
        if not project_dir.exists() or not project_dir.is_dir():
            raise ValueError(f"Invalid project path: {project_path}")
        
        logger.info(f"Loading project context for: {project_path}")
        
        # Detect project type
        project_type = self._detect_project_type(project_dir)
        
        # Get git status
        git_status = await self._get_git_status(project_dir)
        
        # Find relevant files
        relevant_files = self._find_relevant_files(project_dir, project_type)
        
        # Extract dependencies
        dependencies = await self._extract_dependencies(project_dir, project_type)
        
        # Generate project summary
        summary = await self._generate_project_summary(project_dir, project_type, relevant_files)
        
        context = ProjectContext(
            project_path=project_path,
            project_type=project_type,
            git_status=git_status,
            relevant_files=relevant_files,
            dependencies=dependencies,
            last_updated=datetime.now().timestamp(),
            summary=summary
        )
        
        # Set in conversation manager
        conversation_manager.set_project_context(context)
        
        return context
    
    def _detect_project_type(self, project_dir: Path) -> str:
        """Detect project type based on files and structure"""
        type_scores = {}
        
        for project_type, indicators in self.project_type_indicators.items():
            score = 0
            
            # Check for indicator files
            for file_name in indicators['files']:
                if (project_dir / file_name).exists():
                    score += 3
            
            # Check for indicator directories
            for dir_name in indicators['dirs']:
                if (project_dir / dir_name).exists():
                    score += 2
            
            # Check for file extensions
            extension_count = 0
            for ext in indicators['extensions']:
                extension_files = list(project_dir.rglob(f'*{ext}'))
                extension_count += len(extension_files[:10])  # Limit to avoid excessive scanning
            
            score += min(extension_count, 10)  # Cap extension score
            
            type_scores[project_type] = score
        
        # Return type with highest score, or 'Unknown' if no clear match
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] >= 3:  # Minimum confidence threshold
                return best_type
        
        return 'Unknown'
    
    async def _get_git_status(self, project_dir: Path) -> Dict[str, Any]:
        """Get git repository status"""
        git_status = {'is_repo': False, 'branch': None, 'clean': True, 'files': []}
        
        try:
            # Check if it's a git repository
            result = await self._run_command(['git', 'rev-parse', '--git-dir'], project_dir)
            if result['returncode'] != 0:
                return git_status
            
            git_status['is_repo'] = True
            
            # Get current branch
            branch_result = await self._run_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], project_dir)
            if branch_result['returncode'] == 0:
                git_status['branch'] = branch_result['stdout'].strip()
            
            # Get status
            status_result = await self._run_command(['git', 'status', '--porcelain'], project_dir)
            if status_result['returncode'] == 0:
                status_lines = status_result['stdout'].strip().split('\n')
                if status_lines != ['']:
                    git_status['clean'] = False
                    git_status['files'] = [line.strip() for line in status_lines if line.strip()]
            
            # Get remote info
            remote_result = await self._run_command(['git', 'remote', '-v'], project_dir)
            if remote_result['returncode'] == 0:
                git_status['remotes'] = remote_result['stdout'].strip()
            
        except Exception as e:
            logger.debug(f"Error getting git status: {e}")
        
        return git_status
    
    def _find_relevant_files(self, project_dir: Path, project_type: str) -> List[str]:
        """Find relevant files for the project"""
        relevant_files = []
        
        # Add important files that exist
        for important_file in self.important_files:
            file_path = project_dir / important_file
            if file_path.exists():
                relevant_files.append(str(file_path.relative_to(project_dir)))
        
        # Add type-specific files
        if project_type in self.project_type_indicators:
            indicators = self.project_type_indicators[project_type]
            
            for file_name in indicators['files']:
                file_path = project_dir / file_name
                if file_path.exists():
                    relevant_files.append(str(file_path.relative_to(project_dir)))
        
        # Add main source files (limited to avoid clutter)
        source_patterns = {
            'Python': ['main.py', 'app.py', '__init__.py', 'setup.py'],
            'Node.js': ['index.js', 'server.js', 'app.js', 'main.js'],
            'Java': ['Main.java', 'Application.java'],
            'Rust': ['main.rs', 'lib.rs'],
            'Go': ['main.go'],
            'C++': ['main.cpp', 'main.c']
        }
        
        if project_type in source_patterns:
            for pattern in source_patterns[project_type]:
                matches = list(project_dir.rglob(pattern))
                for match in matches[:3]:  # Limit to first 3 matches
                    relevant_files.append(str(match.relative_to(project_dir)))
        
        return list(set(relevant_files))  # Remove duplicates
    
    async def _extract_dependencies(self, project_dir: Path, project_type: str) -> List[str]:
        """Extract project dependencies"""
        dependencies = []
        
        try:
            if project_type == 'Python':
                # Try requirements.txt
                req_file = project_dir / 'requirements.txt'
                if req_file.exists():
                    content = req_file.read_text()
                    deps = [line.split('==')[0].split('>=')[0].split('~=')[0].strip() 
                           for line in content.split('\n') 
                           if line.strip() and not line.startswith('#')]
                    dependencies.extend(deps[:20])  # Limit to first 20
                
                # Try pyproject.toml
                pyproject_file = project_dir / 'pyproject.toml'
                if pyproject_file.exists():
                    # Basic parsing - would need proper TOML parser for production
                    content = pyproject_file.read_text()
                    if 'dependencies' in content:
                        dependencies.extend(['pyproject.toml dependencies'])
            
            elif project_type == 'Node.js':
                package_file = project_dir / 'package.json'
                if package_file.exists():
                    try:
                        package_data = json.loads(package_file.read_text())
                        deps = list(package_data.get('dependencies', {}).keys())
                        dev_deps = list(package_data.get('devDependencies', {}).keys())
                        dependencies.extend(deps[:15] + dev_deps[:10])
                    except json.JSONDecodeError:
                        dependencies.append('package.json (parse error)')
            
            elif project_type == 'Rust':
                cargo_file = project_dir / 'Cargo.toml'
                if cargo_file.exists():
                    content = cargo_file.read_text()
                    # Basic TOML parsing for dependencies section
                    if '[dependencies]' in content:
                        dependencies.extend(['Cargo.toml dependencies'])
            
            elif project_type == 'Go':
                go_mod = project_dir / 'go.mod'
                if go_mod.exists():
                    content = go_mod.read_text()
                    for line in content.split('\n'):
                        if line.strip() and not line.startswith('module') and not line.startswith('go '):
                            if '/' in line:  # Likely a dependency
                                dep = line.strip().split()[0]
                                dependencies.append(dep)
                                if len(dependencies) >= 15:
                                    break
            
        except Exception as e:
            logger.debug(f"Error extracting dependencies: {e}")
        
        return dependencies
    
    async def _generate_project_summary(self, project_dir: Path, project_type: str, relevant_files: List[str]) -> str:
        """Generate a concise project summary"""
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"{project_type} project")
        
        # File count
        try:
            # Limit search depth and add timeout to prevent hanging
            file_count = 0
            max_files = 1000  # Limit to prevent hanging
            search_depth = 3   # Limit directory depth
            
            for depth in range(search_depth):
                if file_count >= max_files:
                    break
                for item in project_dir.iterdir():
                    if item.is_file() and not any(part.startswith('.') for part in item.parts):
                        file_count += 1
                        if file_count >= max_files:
                            break
                    elif item.is_dir() and depth < search_depth - 1:
                        # Recursively count files in subdirectories (limited depth)
                        try:
                            for subitem in item.rglob('*'):
                                if subitem.is_file() and not any(part.startswith('.') for part in subitem.parts):
                                    file_count += 1
                                    if file_count >= max_files:
                                        break
                            if file_count >= max_files:
                                break
                        except:
                            pass  # Skip problematic directories
            
            if file_count >= max_files:
                summary_parts.append(f"{file_count}+ files")
            else:
                summary_parts.append(f"{file_count} files")
        except:
            summary_parts.append("files (count unavailable)")
        
        # Key characteristics
        if 'README.md' in relevant_files:
            summary_parts.append("documented")
        
        if any('test' in f.lower() for f in relevant_files):
            summary_parts.append("with tests")
        
        if 'Dockerfile' in relevant_files:
            summary_parts.append("containerized")
        
        return ", ".join(summary_parts)
    
    async def _run_command(self, cmd: List[str], cwd: Path) -> Dict[str, Any]:
        """Run command and return result"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Add timeout to prevent hanging
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10.0)
            except asyncio.TimeoutError:
                process.terminate()
                return {
                    'returncode': -1,
                    'stdout': '',
                    'stderr': 'Command timed out'
                }
            
            return {
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8'),
                'stderr': stderr.decode('utf-8')
            }
        except Exception as e:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e)
            }
    
    async def monitor_project_changes(self, project_path: str, callback=None):
        """Monitor project for changes and update context"""
        # This would integrate with file system watching
        # For now, just a placeholder for future implementation
        logger.info(f"Monitoring {project_path} for changes")
        pass
    
    def get_project_insights(self, context: ProjectContext) -> List[str]:
        """Generate insights about the project"""
        insights = []
        
        # Git insights
        if context.git_status['is_repo']:
            if not context.git_status['clean']:
                insights.append(f"🔄 Working directory has {len(context.git_status['files'])} changed files")
            else:
                insights.append("✅ Working directory is clean")
        else:
            insights.append("⚠️ Not a git repository")
        
        # Dependency insights
        if len(context.dependencies) > 20:
            insights.append(f"📦 Large project with {len(context.dependencies)}+ dependencies")
        elif len(context.dependencies) > 0:
            insights.append(f"📦 {len(context.dependencies)} dependencies")
        
        # File insights
        if 'README.md' in context.relevant_files:
            insights.append("📚 Well documented")
        else:
            insights.append("📝 Consider adding README documentation")
        
        return insights

# Global project context loader
project_context_loader = ProjectContextLoader()


# Example usage
if __name__ == "__main__":
    async def test_context_loader():
        loader = ProjectContextLoader()
        context = await loader.load_project_context('.')
        
        print(f"Project Type: {context.project_type}")
        print(f"Git Status: {context.git_status}")
        print(f"Relevant Files: {context.relevant_files}")
        print(f"Dependencies: {context.dependencies[:10]}")
        print(f"Summary: {context.summary}")
        
        insights = loader.get_project_insights(context)
        print(f"Insights: {insights}")
    
    asyncio.run(test_context_loader())
