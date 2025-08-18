"""
Plugin Registry for Monk CLI

Manages plugin discovery, loading, validation, and lifecycle management.
"""

import os
import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
import logging
import json
import time

from .base import PluginBase, PluginMetadata, PluginStatus, PluginType
from .validator import PluginValidator

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Central registry for managing all Monk CLI plugins
    
    Handles:
    - Plugin discovery and loading
    - Plugin validation and security
    - Plugin lifecycle management
    - Plugin dependency resolution
    """
    
    def __init__(self):
        self.plugins: Dict[str, PluginBase] = {}
        self.metadata: Dict[str, PluginMetadata] = {}
        self.plugin_paths: List[str] = []
        self.validator = PluginValidator()
        self.discovery_paths = self._get_default_discovery_paths()
        
        # Plugin state tracking
        self.enabled_plugins: List[str] = []
        self.disabled_plugins: List[str] = []
        self.error_plugins: List[str] = []
        
        # Performance tracking
        self.load_times: Dict[str, float] = {}
        self.last_discovery: float = 0
        
    def _get_default_discovery_paths(self) -> List[str]:
        """Get default plugin discovery paths"""
        paths = []
        
        # User's home directory
        home = Path.home()
        paths.append(str(home / ".monk" / "plugins"))
        
        # Current project directory
        paths.append(str(Path.cwd() / "plugins"))
        
        # Monk CLI installation directory
        try:
            import monk
            monk_path = Path(monk.__file__).parent / "plugins"
            paths.append(str(monk_path))
        except ImportError:
            pass
            
        # System-wide plugins
        paths.extend([
            "/usr/local/lib/monk/plugins",
            "/usr/lib/monk/plugins"
        ])
        
        return paths
    
    def add_discovery_path(self, path: str) -> None:
        """Add a custom plugin discovery path"""
        if path not in self.discovery_paths:
            self.discovery_paths.append(path)
            logger.info(f"Added plugin discovery path: {path}")
    
    def discover_plugins(self, force_refresh: bool = False) -> List[str]:
        """
        Discover available plugins in discovery paths
        
        Args:
            force_refresh: Force refresh even if recently discovered
            
        Returns:
            List of discovered plugin names
        """
        current_time = time.time()
        if not force_refresh and current_time - self.last_discovery < 300:  # 5 min cache
            return list(self.plugins.keys())
        
        discovered = []
        
        for path in self.discovery_paths:
            if not os.path.exists(path):
                continue
                
            try:
                path_plugins = self._discover_plugins_in_path(path)
                discovered.extend(path_plugins)
                logger.info(f"Discovered {len(path_plugins)} plugins in {path}")
            except Exception as e:
                logger.error(f"Error discovering plugins in {path}: {e}")
        
        self.last_discovery = current_time
        return discovered
    
    def _discover_plugins_in_path(self, path: str) -> List[str]:
        """Discover plugins in a specific directory"""
        plugins = []
        path_obj = Path(path)
        
        # Look for Python files and directories
        for item in path_obj.iterdir():
            if item.is_file() and item.suffix == '.py':
                if self._is_plugin_file(item):
                    plugins.append(item.stem)
            elif item.is_dir() and (item / "__init__.py").exists():
                if self._is_plugin_package(item):
                    plugins.append(item.name)
        
        return plugins
    
    def _is_plugin_file(self, file_path: Path) -> bool:
        """Check if a Python file contains a plugin"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                # Look for plugin class definitions
                return any(keyword in content for keyword in [
                    'class', 'PluginBase', 'PluginCommand', 'PluginAnalyzer'
                ])
        except Exception:
            return False
    
    def _is_plugin_package(self, dir_path: Path) -> bool:
        """Check if a directory contains a plugin package"""
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            return False
            
        try:
            with open(init_file, 'r') as f:
                content = f.read()
                return any(keyword in content for keyword in [
                    'PluginBase', 'PluginCommand', 'PluginAnalyzer'
                ])
        except Exception:
            return False
    
    def load_plugin(self, plugin_name: str, plugin_path: Optional[str] = None) -> bool:
        """
        Load a specific plugin
        
        Args:
            plugin_name: Name of the plugin to load
            plugin_path: Optional path to the plugin
            
        Returns:
            True if plugin loaded successfully
        """
        if plugin_name in self.plugins:
            logger.info(f"Plugin {plugin_name} already loaded")
            return True
        
        try:
            start_time = time.time()
            
            if plugin_path:
                plugin = self._load_plugin_from_path(plugin_name, plugin_path)
            else:
                plugin = self._load_plugin_by_name(plugin_name)
            
            if plugin and self.validator.validate_plugin(plugin):
                self.plugins[plugin_name] = plugin
                if plugin.metadata:
                    self.metadata[plugin_name] = plugin.metadata
                
                self.load_times[plugin_name] = time.time() - start_time
                logger.info(f"Successfully loaded plugin: {plugin_name}")
                return True
            else:
                logger.error(f"Failed to load plugin: {plugin_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return False
    
    def _load_plugin_from_path(self, plugin_name: str, plugin_path: str) -> Optional[PluginBase]:
        """Load plugin from a specific file path"""
        try:
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for plugin classes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, PluginBase) and 
                        attr != PluginBase):
                        return attr()
                        
        except Exception as e:
            logger.error(f"Error loading plugin from path {plugin_path}: {e}")
        
        return None
    
    def _load_plugin_by_name(self, plugin_name: str) -> Optional[PluginBase]:
        """Load plugin by importing its module"""
        try:
            module = importlib.import_module(plugin_name)
            
            # Look for plugin classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, PluginBase) and 
                    attr != PluginBase):
                    return attr()
                    
        except ImportError:
            # Try with common prefixes
            for prefix in ['monk_', 'monk.plugins.']:
                try:
                    full_name = f"{prefix}{plugin_name}"
                    module = importlib.import_module(full_name)
                    
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, PluginBase) and 
                            attr != PluginBase):
                            return attr()
                except ImportError:
                    continue
                    
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
        
        return None
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if plugin unloaded successfully
        """
        if plugin_name not in self.plugins:
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            plugin.cleanup()
            
            del self.plugins[plugin_name]
            if plugin_name in self.metadata:
                del self.metadata[plugin_name]
            if plugin_name in self.load_times:
                del self.load_times[plugin_name]
            
            logger.info(f"Successfully unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin"""
        if plugin_name not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_name]
        plugin.update_status(PluginStatus.ENABLED)
        
        if plugin_name not in self.enabled_plugins:
            self.enabled_plugins.append(plugin_name)
        if plugin_name in self.disabled_plugins:
            self.disabled_plugins.remove(plugin_name)
        if plugin_name in self.error_plugins:
            self.error_plugins.remove(plugin_name)
        
        logger.info(f"Enabled plugin: {plugin_name}")
        return True
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin"""
        if plugin_name not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_name]
        plugin.update_status(PluginStatus.DISABLED)
        
        if plugin_name in self.enabled_plugins:
            self.enabled_plugins.remove(plugin_name)
        if plugin_name not in self.disabled_plugins:
            self.disabled_plugins.append(plugin_name)
        
        logger.info(f"Disabled plugin: {plugin_name}")
        return True
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginBase]:
        """Get a plugin by name"""
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginBase]:
        """Get all plugins of a specific type"""
        return [
            plugin for plugin in self.plugins.values()
            if plugin.metadata and plugin.metadata.plugin_type == plugin_type
        ]
    
    def get_enabled_plugins(self) -> List[PluginBase]:
        """Get all enabled plugins"""
        return [self.plugins[name] for name in self.enabled_plugins if name in self.plugins]
    
    def get_plugin_commands(self) -> Dict[str, PluginBase]:
        """Get all commands provided by plugins"""
        commands = {}
        for plugin in self.get_enabled_plugins():
            for cmd in plugin.get_commands():
                commands[cmd] = plugin
        return commands
    
    def execute_plugin_command(self, command: str, args: List[str], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a command through the appropriate plugin
        
        Args:
            command: Command to execute
            args: Command arguments
            kwargs: Command options
            
        Returns:
            Command execution result
        """
        plugin_commands = self.get_plugin_commands()
        
        if command not in plugin_commands:
            raise ValueError(f"Unknown plugin command: {command}")
        
        plugin = plugin_commands[command]
        plugin.update_status(PluginStatus.EXECUTING)
        
        try:
            result = plugin.execute(command, args, kwargs)
            plugin.update_status(PluginStatus.COMPLETED)
            return result
        except Exception as e:
            plugin.update_status(PluginStatus.ERROR)
            logger.error(f"Error executing plugin command {command}: {e}")
            raise
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a plugin"""
        if plugin_name not in self.plugins:
            return None
        
        plugin = self.plugins[plugin_name]
        info = {
            'name': plugin_name,
            'status': plugin.status.value,
            'performance': plugin.get_performance_metrics()
        }
        
        if plugin.metadata:
            info['metadata'] = plugin.metadata.to_dict()
        
        return info
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of plugin registry state"""
        return {
            'total_plugins': len(self.plugins),
            'enabled_plugins': len(self.enabled_plugins),
            'disabled_plugins': len(self.disabled_plugins),
            'error_plugins': len(self.error_plugins),
            'plugin_types': {
                pt.value: len(self.get_plugins_by_type(pt))
                for pt in PluginType
            },
            'discovery_paths': self.discovery_paths,
            'last_discovery': self.last_discovery
        }
    
    def reload_all_plugins(self) -> Dict[str, bool]:
        """Reload all plugins and return success status"""
        results = {}
        plugin_names = list(self.plugins.keys())
        
        for name in plugin_names:
            success = self.unload_plugin(name)
            if success:
                success = self.load_plugin(name)
            results[name] = success
        
        return results
    
    def cleanup(self) -> None:
        """Clean up all plugins and registry"""
        for plugin_name in list(self.plugins.keys()):
            self.unload_plugin(plugin_name)
        
        self.plugins.clear()
        self.metadata.clear()
        self.load_times.clear()
        logger.info("Plugin registry cleaned up")
