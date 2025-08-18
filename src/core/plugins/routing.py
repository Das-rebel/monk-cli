"""
Plugin Routing for Monk CLI

Handles routing of commands to appropriate plugins and manages plugin execution.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging

from .base import PluginBase, PluginType
from .registry import PluginRegistry

logger = logging.getLogger(__name__)


class PluginRouter:
    """
    Routes commands to appropriate plugins and manages execution
    
    Handles:
    - Command routing logic
    - Plugin execution coordination
    - Fallback mechanisms
    """
    
    def __init__(self, plugin_registry: PluginRegistry):
        self.plugin_registry = plugin_registry
        self.command_routes: Dict[str, str] = {}  # command -> plugin_name
        self.plugin_commands: Dict[str, List[str]] = {}  # plugin_name -> [commands]
        self._build_routes()
    
    def _build_routes(self) -> None:
        """Build command routing table"""
        try:
            # Clear existing routes
            self.command_routes.clear()
            self.plugin_commands.clear()
            
            # Get all enabled plugins
            enabled_plugins = self.plugin_registry.get_enabled_plugins()
            
            for plugin in enabled_plugins:
                plugin_name = plugin.__class__.__name__
                commands = plugin.get_commands()
                
                if commands:
                    self.plugin_commands[plugin_name] = commands
                    
                    # Register each command
                    for command in commands:
                        if command in self.command_routes:
                            logger.warning(f"Command {command} already registered by {self.command_routes[command]}, overriding with {plugin_name}")
                        
                        self.command_routes[command] = plugin_name
                        logger.debug(f"Registered command {command} -> {plugin_name}")
            
            logger.info(f"Built routing table with {len(self.command_routes)} commands across {len(self.plugin_commands)} plugins")
            
        except Exception as e:
            logger.error(f"Error building command routes: {e}")
    
    def route_command(self, command: str) -> Optional[Tuple[str, PluginBase]]:
        """
        Route a command to the appropriate plugin
        
        Args:
            command: Command to route
            
        Returns:
            Tuple of (plugin_name, plugin_instance) or None if not found
        """
        if command not in self.command_routes:
            return None
        
        plugin_name = self.command_routes[command]
        plugin = self.plugin_registry.get_plugin(plugin_name)
        
        if plugin:
            return (plugin_name, plugin)
        else:
            logger.error(f"Plugin {plugin_name} not found for command {command}")
            return None
    
    def get_available_commands(self) -> List[str]:
        """Get list of all available commands"""
        return list(self.command_routes.keys())
    
    def get_plugin_commands(self, plugin_name: str) -> List[str]:
        """Get commands provided by a specific plugin"""
        return self.plugin_commands.get(plugin_name, [])
    
    def get_command_help(self, command: str) -> Optional[str]:
        """
        Get help text for a command
        
        Args:
            command: Command name
            
        Returns:
            Help text or None if command not found
        """
        route = self.route_command(command)
        if route:
            plugin_name, plugin = route
            return plugin.get_help()
        return None
    
    def execute_command(self, command: str, args: List[str], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a command through the appropriate plugin
        
        Args:
            command: Command to execute
            args: Command arguments
            kwargs: Command options
            
        Returns:
            Command execution result
        """
        route = self.route_command(command)
        if not route:
            raise ValueError(f"Unknown command: {command}")
        
        plugin_name, plugin = route
        
        try:
            # Update plugin status
            plugin.update_status(plugin.status.EXECUTING)
            
            # Execute command
            result = plugin.execute(command, args, kwargs)
            
            # Update status
            plugin.update_status(plugin.status.COMPLETED)
            
            logger.info(f"Executed command {command} through plugin {plugin_name}")
            return result
            
        except Exception as e:
            # Update status
            plugin.update_status(plugin.status.ERROR)
            logger.error(f"Error executing command {command} through plugin {plugin_name}: {e}")
            raise
    
    def suggest_commands(self, partial_command: str) -> List[str]:
        """
        Suggest commands based on partial input
        
        Args:
            partial_command: Partial command string
            
        Returns:
            List of matching commands
        """
        suggestions = []
        partial_lower = partial_command.lower()
        
        for command in self.command_routes.keys():
            if command.lower().startswith(partial_lower):
                suggestions.append(command)
        
        return sorted(suggestions)
    
    def get_command_usage(self, command: str) -> Optional[str]:
        """
        Get usage information for a command
        
        Args:
            command: Command name
            
        Returns:
            Usage string or None if command not found
        """
        route = self.route_command(command)
        if route:
            plugin_name, plugin = route
            if hasattr(plugin, 'command_usage'):
                return plugin.command_usage
            elif hasattr(plugin, 'get_help'):
                return plugin.get_help()
        return None
    
    def reload_routes(self) -> None:
        """Reload command routing table"""
        logger.info("Reloading plugin command routes")
        self._build_routes()
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            'total_commands': len(self.command_routes),
            'total_plugins': len(self.plugin_commands),
            'commands_per_plugin': {
                plugin: len(commands) 
                for plugin, commands in self.plugin_commands.items()
            },
            'route_conflicts': self._detect_route_conflicts()
        }
    
    def _detect_route_conflicts(self) -> List[Dict[str, str]]:
        """Detect command routing conflicts"""
        conflicts = []
        command_plugins = {}
        
        for command, plugin_name in self.command_routes.items():
            if command in command_plugins:
                conflicts.append({
                    'command': command,
                    'primary_plugin': command_plugins[command],
                    'conflicting_plugin': plugin_name
                })
            else:
                command_plugins[command] = plugin_name
        
        return conflicts
    
    def validate_routes(self) -> Dict[str, List[str]]:
        """
        Validate all command routes
        
        Returns:
            Dictionary mapping plugin names to validation issues
        """
        issues = {}
        
        for plugin_name, commands in self.plugin_commands.items():
            plugin_issues = []
            plugin = self.plugin_registry.get_plugin(plugin_name)
            
            if not plugin:
                plugin_issues.append("Plugin not found in registry")
                issues[plugin_name] = plugin_issues
                continue
            
            # Check if plugin is enabled
            if plugin.status.value == "disabled":
                plugin_issues.append("Plugin is disabled")
            
            # Check command implementations
            for command in commands:
                try:
                    # Test command execution
                    test_result = plugin.execute(command, [], {})
                    if not isinstance(test_result, str):
                        plugin_issues.append(f"Command {command} does not return string")
                except Exception as e:
                    plugin_issues.append(f"Command {command} execution failed: {e}")
            
            if plugin_issues:
                issues[plugin_name] = plugin_issues
        
        return issues
    
    def get_plugin_dependencies(self, plugin_name: str) -> List[str]:
        """
        Get dependencies for a plugin
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            List of dependency names
        """
        plugin = self.plugin_registry.get_plugin(plugin_name)
        if plugin and plugin.metadata:
            return plugin.metadata.dependencies
        return []
    
    def check_plugin_compatibility(self, plugin_name: str) -> Dict[str, Any]:
        """
        Check plugin compatibility with current Monk CLI version
        
        Args:
            plugin_name: Name of the plugin to check
            
        Returns:
            Compatibility information
        """
        plugin = self.plugin_registry.get_plugin(plugin_name)
        if not plugin:
            return {'compatible': False, 'error': 'Plugin not found'}
        
        compatibility_info = {
            'plugin_name': plugin_name,
            'compatible': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check required methods
            required_methods = ['initialize', 'execute', 'cleanup']
            for method in required_methods:
                if not hasattr(plugin, method):
                    compatibility_info['errors'].append(f"Missing required method: {method}")
                    compatibility_info['compatible'] = False
                elif not callable(getattr(plugin, method)):
                    compatibility_info['errors'].append(f"Required method not callable: {method}")
                    compatibility_info['compatible'] = False
            
            # Check metadata
            if not hasattr(plugin, 'metadata') or not plugin.metadata:
                compatibility_info['warnings'].append("No metadata available")
            
            # Check dependencies
            missing_deps = plugin.validate_dependencies()
            if missing_deps:
                compatibility_info['warnings'].append(f"Missing dependencies: {', '.join(missing_deps)}")
            
        except Exception as e:
            compatibility_info['errors'].append(f"Compatibility check failed: {e}")
            compatibility_info['compatible'] = False
        
        return compatibility_info
