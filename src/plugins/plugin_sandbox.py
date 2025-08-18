"""
Plugin Sandbox
Security sandbox for plugin execution and permission management
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import logging
import os
import sys
import importlib

from src.plugins.plugin_base import BasePlugin

logger = logging.getLogger(__name__)

class PluginSandbox:
    """
    Security sandbox for plugin execution and permission management
    """
    
    def __init__(self):
        self.allowed_modules = {
            'os': ['path', 'environ', 'getcwd', 'listdir', 'stat'],
            'pathlib': ['Path'],
            'json': ['loads', 'dumps', 'load', 'dump'],
            'time': ['time', 'sleep'],
            'datetime': ['datetime', 'timedelta'],
            'asyncio': ['sleep', 'gather', 'create_task'],
            'logging': ['getLogger', 'info', 'warning', 'error', 'debug'],
            'typing': ['Dict', 'List', 'Any', 'Optional', 'Union'],
            'dataclasses': ['dataclass', 'field'],
            'abc': ['ABC', 'abstractmethod']
        }
        
        self.restricted_modules = {
            'subprocess',
            'multiprocessing',
            'threading',
            'socket',
            'urllib',
            'requests',
            'sqlite3',
            'pickle',
            'marshal',
            'ctypes',
            'sys',
            'builtins'
        }
        
        self.allowed_file_operations = {
            'read': True,
            'write': False,  # Restrictive by default
            'delete': False,
            'execute': False,
            'network': False
        }
        
        self.plugin_permissions: Dict[str, Set[str]] = {}
        self.execution_history: Dict[str, List[Dict[str, Any]]] = {}
        
    def check_permissions(self, plugin: BasePlugin) -> bool:
        """
        Check if a plugin has the required permissions to run
        """
        plugin_name = plugin.metadata.name
        
        # Check basic permissions
        required_permissions = self._get_required_permissions(plugin)
        granted_permissions = set(plugin.config.permissions)
        
        # Check if plugin has all required permissions
        missing_permissions = required_permissions - granted_permissions
        if missing_permissions:
            logger.warning(f"Plugin {plugin_name} missing permissions: {missing_permissions}")
            return False
        
        # Check if plugin has any restricted permissions
        restricted_permissions = granted_permissions.intersection(self._get_restricted_permissions())
        if restricted_permissions:
            logger.warning(f"Plugin {plugin_name} has restricted permissions: {restricted_permissions}")
            return False
        
        return True
    
    def _get_required_permissions(self, plugin: BasePlugin) -> Set[str]:
        """Get permissions required by the plugin based on its code"""
        required = set()
        
        # Check plugin commands
        if plugin.commands:
            required.add('execute_commands')
        
        # Check API endpoints
        if plugin.api_endpoints:
            required.add('expose_api')
        
        # Check file operations
        if hasattr(plugin, 'file_operations'):
            required.add('file_access')
        
        # Check network operations
        if hasattr(plugin, 'network_operations'):
            required.add('network_access')
        
        return required
    
    def _get_restricted_permissions(self) -> Set[str]:
        """Get list of restricted permissions"""
        return {
            'system_access',
            'network_access',
            'file_write',
            'file_delete',
            'execute_files',
            'modify_system',
            'access_environment'
        }
    
    def grant_permission(self, plugin_name: str, permission: str) -> bool:
        """
        Grant a permission to a plugin
        """
        if permission in self._get_restricted_permissions():
            logger.warning(f"Cannot grant restricted permission {permission} to plugin {plugin_name}")
            return False
        
        if plugin_name not in self.plugin_permissions:
            self.plugin_permissions[plugin_name] = set()
        
        self.plugin_permissions[plugin_name].add(permission)
        logger.info(f"Granted permission {permission} to plugin {plugin_name}")
        return True
    
    def revoke_permission(self, plugin_name: str, permission: str) -> bool:
        """
        Revoke a permission from a plugin
        """
        if plugin_name in self.plugin_permissions and permission in self.plugin_permissions[plugin_name]:
            self.plugin_permissions[plugin_name].remove(permission)
            logger.info(f"Revoked permission {permission} from plugin {plugin_name}")
            return True
        return False
    
    def check_module_access(self, plugin: BasePlugin, module_name: str) -> bool:
        """
        Check if a plugin can access a specific module
        """
        plugin_name = plugin.metadata.name
        
        # Check if module is completely restricted
        if module_name in self.restricted_modules:
            logger.warning(f"Plugin {plugin_name} attempted to access restricted module: {module_name}")
            return False
        
        # Check if module is allowed with restrictions
        if module_name in self.allowed_modules:
            return True
        
        # Check if plugin has explicit permission
        if plugin_name in self.plugin_permissions:
            if 'unrestricted_module_access' in self.plugin_permissions[plugin_name]:
                return True
        
        # Deny access by default
        logger.warning(f"Plugin {plugin_name} attempted to access unknown module: {module_name}")
        return False
    
    def check_file_operation(self, plugin: BasePlugin, operation: str, file_path: str = None) -> bool:
        """
        Check if a plugin can perform a file operation
        """
        plugin_name = plugin.metadata.name
        
        # Check if operation is allowed globally
        if not self.allowed_file_operations.get(operation, False):
            logger.warning(f"Plugin {plugin_name} attempted restricted file operation: {operation}")
            return False
        
        # Check if plugin has explicit file permissions
        if plugin_name in self.plugin_permissions:
            if f'file_{operation}' in self.plugin_permissions[plugin_name]:
                return True
        
        # Check if file path is in allowed directories
        if file_path:
            allowed_dirs = self._get_allowed_directories(plugin)
            file_path_obj = Path(file_path)
            
            for allowed_dir in allowed_dirs:
                try:
                    file_path_obj.relative_to(allowed_dir)
                    return True
                except ValueError:
                    continue
            
            logger.warning(f"Plugin {plugin_name} attempted file operation on restricted path: {file_path}")
            return False
        
        return False
    
    def _get_allowed_directories(self, plugin: BasePlugin) -> List[Path]:
        """Get directories the plugin is allowed to access"""
        allowed_dirs = []
        
        # Plugin's own directory
        if plugin.plugin_path:
            allowed_dirs.append(plugin.plugin_path)
        
        # Current working directory (if plugin has permission)
        if plugin.has_permission('access_working_directory'):
            allowed_dirs.append(Path.cwd())
        
        # Temporary directory
        import tempfile
        allowed_dirs.append(Path(tempfile.gettempdir()))
        
        return allowed_dirs
    
    def check_network_access(self, plugin: BasePlugin, host: str = None, port: int = None) -> bool:
        """
        Check if a plugin can access the network
        """
        plugin_name = plugin.metadata.name
        
        # Check if network access is globally allowed
        if not self.allowed_file_operations.get('network', False):
            logger.warning(f"Plugin {plugin_name} attempted network access (globally restricted)")
            return False
        
        # Check if plugin has network permission
        if plugin.has_permission('network_access'):
            # Check for restricted hosts/ports
            if host and self._is_restricted_host(host):
                logger.warning(f"Plugin {plugin_name} attempted access to restricted host: {host}")
                return False
            
            if port and self._is_restricted_port(port):
                logger.warning(f"Plugin {plugin_name} attempted access to restricted port: {port}")
                return False
            
            return True
        
        logger.warning(f"Plugin {plugin_name} attempted network access without permission")
        return False
    
    def _is_restricted_host(self, host: str) -> bool:
        """Check if a host is restricted"""
        restricted_hosts = {
            'localhost',
            '127.0.0.1',
            '::1',
            '0.0.0.0'
        }
        
        return host in restricted_hosts
    
    def _is_restricted_port(self, port: int) -> bool:
        """Check if a port is restricted"""
        restricted_ports = {
            22,    # SSH
            23,    # Telnet
            25,    # SMTP
            53,    # DNS
            80,    # HTTP
            443,   # HTTPS
            3306,  # MySQL
            5432,  # PostgreSQL
            6379,  # Redis
            27017  # MongoDB
        }
        
        return port in restricted_ports
    
    def log_execution(self, plugin: BasePlugin, operation: str, details: Dict[str, Any] = None):
        """
        Log a plugin execution for audit purposes
        """
        plugin_name = plugin.metadata.name
        timestamp = time.time()
        
        if plugin_name not in self.execution_history:
            self.execution_history[plugin_name] = []
        
        log_entry = {
            'timestamp': timestamp,
            'operation': operation,
            'details': details or {},
            'plugin_version': plugin.metadata.version,
            'permissions': list(plugin.config.permissions)
        }
        
        self.execution_history[plugin_name].append(log_entry)
        
        # Keep only last 100 entries per plugin
        if len(self.execution_history[plugin_name]) > 100:
            self.execution_history[plugin_name] = self.execution_history[plugin_name][-100:]
        
        logger.debug(f"Plugin {plugin_name} executed operation: {operation}")
    
    def get_execution_history(self, plugin_name: str = None) -> Dict[str, Any]:
        """
        Get execution history for plugins
        """
        if plugin_name:
            if plugin_name in self.execution_history:
                return {
                    'success': True,
                    'history': self.execution_history[plugin_name]
                }
            else:
                return {
                    'success': False,
                    'error': f'No execution history for plugin {plugin_name}'
                }
        
        return {
            'success': True,
            'history': self.execution_history
        }
    
    def get_plugin_permissions(self, plugin_name: str = None) -> Dict[str, Any]:
        """
        Get permission information for plugins
        """
        if plugin_name:
            if plugin_name in self.plugin_permissions:
                return {
                    'success': True,
                    'permissions': list(self.plugin_permissions[plugin_name])
                }
            else:
                return {
                    'success': False,
                    'error': f'No permissions found for plugin {plugin_name}'
                }
        
        return {
            'success': True,
            'permissions': self.plugin_permissions
        }
    
    def reset_plugin_permissions(self, plugin_name: str) -> bool:
        """
        Reset all permissions for a plugin
        """
        if plugin_name in self.plugin_permissions:
            del self.plugin_permissions[plugin_name]
            logger.info(f"Reset permissions for plugin {plugin_name}")
            return True
        return False
    
    def get_sandbox_status(self) -> Dict[str, Any]:
        """Get sandbox status and configuration"""
        return {
            'allowed_modules': self.allowed_modules,
            'restricted_modules': list(self.restricted_modules),
            'allowed_file_operations': self.allowed_file_operations,
            'total_plugins_with_permissions': len(self.plugin_permissions),
            'total_execution_logs': sum(len(logs) for logs in self.execution_history.values()),
            'last_check': time.time()
        }
    
    def update_sandbox_config(self, config: Dict[str, Any]) -> bool:
        """
        Update sandbox configuration
        """
        try:
            if 'allowed_modules' in config:
                self.allowed_modules.update(config['allowed_modules'])
            
            if 'restricted_modules' in config:
                self.restricted_modules.update(config['restricted_modules'])
            
            if 'allowed_file_operations' in config:
                self.allowed_file_operations.update(config['allowed_file_operations'])
            
            logger.info("Sandbox configuration updated")
            return True
            
        except Exception as e:
            logger.error(f"Error updating sandbox config: {e}")
            return False
    
    def cleanup(self):
        """Cleanup sandbox resources"""
        # Clear execution history older than 24 hours
        cutoff_time = time.time() - (24 * 60 * 60)
        
        for plugin_name in list(self.execution_history.keys()):
            self.execution_history[plugin_name] = [
                entry for entry in self.execution_history[plugin_name]
                if entry['timestamp'] > cutoff_time
            ]
            
            # Remove empty history
            if not self.execution_history[plugin_name]:
                del self.execution_history[plugin_name]
        
        logger.info("Plugin sandbox cleanup completed")
