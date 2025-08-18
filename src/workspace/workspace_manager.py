"""
Workspace Manager
Multi-project workspace support with context switching and configuration management
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import shutil
from datetime import datetime

from src.core.cache_manager import cache_manager
from src.core.async_engine import monitor_performance

logger = logging.getLogger(__name__)

class WorkspaceManager:
    """
    Manages multiple project workspaces with context switching
    """
    
    def __init__(self, workspace_root: str = None):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.home() / '.monk-workspaces'
        self.workspace_root.mkdir(exist_ok=True)
        self.current_workspace = None
        self.workspaces = {}
        self.templates = {}
        
        # Load existing workspaces
        self._load_workspaces()
        self._load_templates()
    
    def _load_workspaces(self):
        """Load existing workspaces from disk"""
        workspaces_file = self.workspace_root / 'workspaces.json'
        if workspaces_file.exists():
            try:
                with open(workspaces_file, 'r') as f:
                    self.workspaces = json.load(f)
            except Exception as e:
                logger.error(f"Error loading workspaces: {e}")
                self.workspaces = {}
    
    def _save_workspaces(self):
        """Save workspaces to disk"""
        workspaces_file = self.workspace_root / 'workspaces.json'
        try:
            with open(workspaces_file, 'w') as f:
                json.dump(self.workspaces, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving workspaces: {e}")
    
    def _load_templates(self):
        """Load workspace templates"""
        templates_dir = self.workspace_root / 'templates'
        templates_dir.mkdir(exist_ok=True)
        
        # Load built-in templates
        self.templates = {
            'python': {
                'name': 'Python Project',
                'description': 'Standard Python project structure',
                'files': {
                    'requirements.txt': '# Python dependencies\n',
                    'README.md': '# Python Project\n\nProject description here.\n',
                    '.gitignore': '# Python\n__pycache__/\n*.pyc\n*.pyo\n*.pyd\n.Python\nenv/\nvenv/\n.env\n',
                    'src/__init__.py': '# Package initialization\n',
                    'tests/__init__.py': '# Test package\n'
                },
                'directories': ['src', 'tests', 'docs']
            },
            'nodejs': {
                'name': 'Node.js Project',
                'description': 'Standard Node.js project structure',
                'files': {
                    'package.json': '{\n  "name": "project-name",\n  "version": "1.0.0",\n  "description": "Project description",\n  "main": "index.js",\n  "scripts": {\n    "test": "echo \\"Error: no test specified\\" && exit 1"\n  }\n}\n',
                    'README.md': '# Node.js Project\n\nProject description here.\n',
                    '.gitignore': '# Node.js\nnode_modules/\nnpm-debug.log*\nyarn-debug.log*\nyarn-error.log*\n.env\n',
                    'src/index.js': '// Main entry point\nconsole.log("Hello, World!");\n',
                    'tests/index.test.js': '// Test file\n'
                },
                'directories': ['src', 'tests', 'docs']
            },
            'web': {
                'name': 'Web Project',
                'description': 'Standard web project structure',
                'files': {
                    'index.html': '<!DOCTYPE html>\n<html>\n<head>\n    <title>Web Project</title>\n</head>\n<body>\n    <h1>Hello, World!</h1>\n</body>\n</html>\n',
                    'README.md': '# Web Project\n\nProject description here.\n',
                    '.gitignore': '# Web\n.DS_Store\nnode_modules/\n.env\n*.log\n',
                    'css/style.css': '/* Styles */\nbody {\n    font-family: Arial, sans-serif;\n}\n',
                    'js/main.js': '// Main JavaScript file\nconsole.log("Web project loaded");\n'
                },
                'directories': ['css', 'js', 'images', 'docs']
            }
        }
        
        # Load custom templates
        custom_templates_dir = self.workspace_root / 'custom-templates'
        if custom_templates_dir.exists():
            for template_dir in custom_templates_dir.iterdir():
                if template_dir.is_dir():
                    template_file = template_dir / 'template.json'
                    if template_file.exists():
                        try:
                            with open(template_file, 'r') as f:
                                template_data = json.load(f)
                                self.templates[template_dir.name] = template_data
                        except Exception as e:
                            logger.error(f"Error loading custom template {template_dir.name}: {e}")
    
    async def create_workspace(self, name: str, path: str, template: str = None, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a new workspace
        """
        options = options or {}
        workspace_path = Path(path)
        
        if workspace_path.exists():
            return {
                'success': False,
                'error': f'Path already exists: {path}'
            }
        
        try:
            # Create workspace directory
            workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Apply template if specified
            if template and template in self.templates:
                await self._apply_template(workspace_path, template, options)
            
            # Create workspace configuration
            workspace_config = {
                'name': name,
                'path': str(workspace_path.absolute()),
                'template': template,
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'options': options,
                'metadata': {}
            }
            
            # Save workspace config
            config_file = workspace_path / '.monk-workspace.json'
            with open(config_file, 'w') as f:
                json.dump(workspace_config, f, indent=2)
            
            # Add to workspace list
            self.workspaces[name] = workspace_config
            self._save_workspaces()
            
            logger.info(f"Created workspace: {name} at {path}")
            
            return {
                'success': True,
                'workspace': workspace_config,
                'message': f'Workspace "{name}" created successfully'
            }
            
        except Exception as e:
            logger.error(f"Error creating workspace: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _apply_template(self, workspace_path: Path, template_name: str, options: Dict[str, Any]):
        """Apply a template to a workspace"""
        template = self.templates[template_name]
        
        # Create directories
        for directory in template.get('directories', []):
            (workspace_path / directory).mkdir(exist_ok=True)
        
        # Create files
        for file_path, content in template.get('files', {}).items():
            full_path = workspace_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process template variables
            processed_content = self._process_template_content(content, options)
            
            with open(full_path, 'w') as f:
                f.write(processed_content)
    
    def _process_template_content(self, content: str, options: Dict[str, Any]) -> str:
        """Process template content with variables"""
        # Simple variable substitution
        for key, value in options.items():
            placeholder = f"{{{{{key}}}}}"
            content = content.replace(placeholder, str(value))
        
        return content
    
    async def open_workspace(self, name: str) -> Dict[str, Any]:
        """
        Open a workspace and set it as current
        """
        if name not in self.workspaces:
            return {
                'success': False,
                'error': f'Workspace not found: {name}'
            }
        
        workspace = self.workspaces[name]
        workspace_path = Path(workspace['path'])
        
        if not workspace_path.exists():
            return {
                'success': False,
                'error': f'Workspace path does not exist: {workspace_path}'
            }
        
        # Update last accessed time
        workspace['last_accessed'] = datetime.now().isoformat()
        self.current_workspace = name
        self._save_workspaces()
        
        logger.info(f"Opened workspace: {name}")
        
        return {
            'success': True,
            'workspace': workspace,
            'message': f'Workspace "{name}" opened successfully'
        }
    
    async def close_workspace(self) -> Dict[str, Any]:
        """
        Close the current workspace
        """
        if not self.current_workspace:
            return {
                'success': False,
                'error': 'No workspace is currently open'
            }
        
        workspace_name = self.current_workspace
        self.current_workspace = None
        
        logger.info(f"Closed workspace: {workspace_name}")
        
        return {
            'success': True,
            'message': f'Workspace "{workspace_name}" closed'
        }
    
    async def list_workspaces(self) -> Dict[str, Any]:
        """
        List all available workspaces
        """
        return {
            'success': True,
            'workspaces': self.workspaces,
            'current_workspace': self.current_workspace,
            'total_workspaces': len(self.workspaces)
        }
    
    async def get_workspace_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a specific workspace
        """
        if name not in self.workspaces:
            return {
                'success': False,
                'error': f'Workspace not found: {name}'
            }
        
        workspace = self.workspaces[name]
        workspace_path = Path(workspace['path'])
        
        # Get additional workspace statistics
        stats = await self._get_workspace_stats(workspace_path)
        
        return {
            'success': True,
            'workspace': workspace,
            'stats': stats
        }
    
    async def _get_workspace_stats(self, workspace_path: Path) -> Dict[str, Any]:
        """Get workspace statistics"""
        try:
            stats = {
                'file_count': 0,
                'directory_count': 0,
                'total_size': 0,
                'last_modified': None
            }
            
            for item in workspace_path.rglob('*'):
                if item.is_file():
                    stats['file_count'] += 1
                    try:
                        stats['total_size'] += item.stat().st_size
                        mtime = item.stat().st_mtime
                        if not stats['last_modified'] or mtime > stats['last_modified']:
                            stats['last_modified'] = mtime
                    except:
                        pass
                elif item.is_dir():
                    stats['directory_count'] += 1
            
            # Convert timestamp to readable format
            if stats['last_modified']:
                stats['last_modified'] = datetime.fromtimestamp(stats['last_modified']).isoformat()
            
            # Convert size to human readable
            stats['total_size_mb'] = stats['total_size'] / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting workspace stats: {e}")
            return {}
    
    async def delete_workspace(self, name: str, delete_files: bool = False) -> Dict[str, Any]:
        """
        Delete a workspace
        """
        if name not in self.workspaces:
            return {
                'success': False,
                'error': f'Workspace not found: {name}'
            }
        
        workspace = self.workspaces[name]
        workspace_path = Path(workspace['path'])
        
        # Check if this is the current workspace
        if self.current_workspace == name:
            await self.close_workspace()
        
        try:
            # Delete files if requested
            if delete_files and workspace_path.exists():
                shutil.rmtree(workspace_path)
            
            # Remove from workspace list
            del self.workspaces[name]
            self._save_workspaces()
            
            logger.info(f"Deleted workspace: {name}")
            
            return {
                'success': True,
                'message': f'Workspace "{name}" deleted successfully'
            }
            
        except Exception as e:
            logger.error(f"Error deleting workspace: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def list_templates(self) -> Dict[str, Any]:
        """
        List available workspace templates
        """
        return {
            'success': True,
            'templates': self.templates,
            'total_templates': len(self.templates)
        }
    
    async def create_template(self, name: str, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a custom workspace template
        """
        try:
            # Validate template data
            required_fields = ['name', 'description', 'files']
            for field in required_fields:
                if field not in template_data:
                    return {
                        'success': False,
                        'error': f'Missing required field: {field}'
                    }
            
            # Save template
            self.templates[name] = template_data
            
            # Save to disk
            custom_templates_dir = self.workspace_root / 'custom-templates' / name
            custom_templates_dir.mkdir(parents=True, exist_ok=True)
            
            template_file = custom_templates_dir / 'template.json'
            with open(template_file, 'w') as f:
                json.dump(template_data, f, indent=2)
            
            logger.info(f"Created template: {name}")
            
            return {
                'success': True,
                'message': f'Template "{name}" created successfully'
            }
            
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_current_workspace(self) -> Dict[str, Any]:
        """
        Get information about the current workspace
        """
        if not self.current_workspace:
            return {
                'success': False,
                'error': 'No workspace is currently open'
            }
        
        return await self.get_workspace_info(self.current_workspace)
    
    async def switch_workspace(self, name: str) -> Dict[str, Any]:
        """
        Switch to a different workspace
        """
        # Close current workspace if any
        if self.current_workspace:
            await self.close_workspace()
        
        # Open new workspace
        return await self.open_workspace(name)
    
    async def export_workspace(self, name: str, export_path: str) -> Dict[str, Any]:
        """
        Export a workspace to a different location
        """
        if name not in self.workspaces:
            return {
                'success': False,
                'error': f'Workspace not found: {name}'
            }
        
        workspace = self.workspaces[name]
        source_path = Path(workspace['path'])
        target_path = Path(export_path)
        
        if not source_path.exists():
            return {
                'success': False,
                'error': f'Source workspace does not exist: {source_path}'
            }
        
        try:
            # Copy workspace
            if target_path.exists():
                shutil.rmtree(target_path)
            
            shutil.copytree(source_path, target_path)
            
            # Remove workspace-specific files
            workspace_config = target_path / '.monk-workspace.json'
            if workspace_config.exists():
                workspace_config.unlink()
            
            logger.info(f"Exported workspace {name} to {export_path}")
            
            return {
                'success': True,
                'message': f'Workspace "{name}" exported to {export_path}'
            }
            
        except Exception as e:
            logger.error(f"Error exporting workspace: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get workspace manager status"""
        return {
            'status': 'ready',
            'current_workspace': self.current_workspace,
            'total_workspaces': len(self.workspaces),
            'total_templates': len(self.templates),
            'workspace_root': str(self.workspace_root),
            'last_check': time.time()
        }
    
    async def shutdown(self):
        """Shutdown the workspace manager"""
        self._save_workspaces()
        logger.info("Workspace Manager shutdown complete")

# Global workspace manager instance
workspace_manager = WorkspaceManager()
