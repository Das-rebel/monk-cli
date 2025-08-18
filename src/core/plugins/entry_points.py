"""
Plugin Entry Points for Monk CLI

Handles plugin registration and discovery through setuptools entry points.
"""

import pkg_resources
from typing import Dict, List, Any, Optional
import logging

from .base import PluginBase, PluginType

logger = logging.getLogger(__name__)


class PluginEntryPoint:
    """
    Manages plugin entry points for Monk CLI
    
    Handles:
    - Entry point discovery
    - Plugin registration
    - Entry point validation
    """
    
    ENTRY_POINT_GROUP = "monk.plugins"
    
    def __init__(self):
        self.entry_points: Dict[str, Any] = {}
        self.discovered_plugins: List[str] = []
    
    def discover_entry_points(self) -> List[str]:
        """
        Discover all available plugin entry points
        
        Returns:
            List of discovered plugin names
        """
        try:
            discovered = []
            
            for entry_point in pkg_resources.iter_entry_points(self.ENTRY_POINT_GROUP):
                try:
                    plugin_class = entry_point.load()
                    if self._validate_entry_point(entry_point, plugin_class):
                        self.entry_points[entry_point.name] = {
                            'entry_point': entry_point,
                            'plugin_class': plugin_class,
                            'distribution': entry_point.dist
                        }
                        discovered.append(entry_point.name)
                        logger.info(f"Discovered plugin entry point: {entry_point.name}")
                    else:
                        logger.warning(f"Invalid plugin entry point: {entry_point.name}")
                        
                except Exception as e:
                    logger.error(f"Error loading entry point {entry_point.name}: {e}")
            
            self.discovered_plugins = discovered
            return discovered
            
        except Exception as e:
            logger.error(f"Error discovering entry points: {e}")
            return []
    
    def _validate_entry_point(self, entry_point: pkg_resources.EntryPoint, 
                            plugin_class: type) -> bool:
        """
        Validate a plugin entry point
        
        Args:
            entry_point: The entry point to validate
            plugin_class: The plugin class to validate
            
        Returns:
            True if entry point is valid
        """
        try:
            # Check if it's a valid plugin class
            if not issubclass(plugin_class, PluginBase):
                logger.warning(f"Entry point {entry_point.name} does not inherit from PluginBase")
                return False
            
            # Check if it can be instantiated
            try:
                plugin_instance = plugin_class()
                if not hasattr(plugin_instance, 'metadata'):
                    logger.warning(f"Plugin {entry_point.name} has no metadata")
                    return False
            except Exception as e:
                logger.warning(f"Plugin {entry_point.name} cannot be instantiated: {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating entry point {entry_point.name}: {e}")
            return False
    
    def get_plugin_instance(self, plugin_name: str) -> Optional[PluginBase]:
        """
        Get a plugin instance from an entry point
        
        Args:
            plugin_name: Name of the plugin to instantiate
            
        Returns:
            Plugin instance or None if not found
        """
        if plugin_name not in self.entry_points:
            return None
        
        try:
            entry_point_info = self.entry_points[plugin_name]
            plugin_class = entry_point_info['plugin_class']
            plugin_instance = plugin_class()
            
            # Set distribution info if available
            if hasattr(plugin_instance, 'distribution'):
                plugin_instance.distribution = entry_point_info['distribution']
            
            return plugin_instance
            
        except Exception as e:
            logger.error(f"Error instantiating plugin {plugin_name}: {e}")
            return None
    
    def get_entry_point_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an entry point
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Entry point information or None if not found
        """
        if plugin_name not in self.entry_points:
            return None
        
        entry_point_info = self.entry_points[plugin_name]
        distribution = entry_point_info['distribution']
        
        return {
            'name': plugin_name,
            'module': entry_point_info['entry_point'].module_name,
            'object': entry_point_info['entry_point'].attrs[0],
            'distribution': {
                'name': distribution.project_name,
                'version': distribution.version,
                'location': distribution.location
            }
        }
    
    def list_entry_points(self) -> List[Dict[str, Any]]:
        """
        List all discovered entry points with information
        
        Returns:
            List of entry point information
        """
        entry_points = []
        
        for plugin_name, entry_point_info in self.entry_points.items():
            info = self.get_entry_point_info(plugin_name)
            if info:
                entry_points.append(info)
        
        return entry_points
    
    def get_plugin_distribution(self, plugin_name: str) -> Optional[pkg_resources.Distribution]:
        """
        Get the distribution that provides a plugin
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Distribution object or None if not found
        """
        if plugin_name not in self.entry_points:
            return None
        
        return self.entry_points[plugin_name]['distribution']
    
    def reload_entry_points(self) -> List[str]:
        """
        Reload all entry points
        
        Returns:
            List of discovered plugin names
        """
        # Clear existing entry points
        self.entry_points.clear()
        self.discovered_plugins.clear()
        
        # Rediscover
        return self.discover_entry_points()
    
    def validate_all_entry_points(self) -> Dict[str, List[str]]:
        """
        Validate all discovered entry points
        
        Returns:
            Dictionary mapping plugin names to validation issues
        """
        validation_results = {}
        
        for plugin_name in self.discovered_plugins:
            issues = []
            
            try:
                entry_point_info = self.entry_points[plugin_name]
                plugin_class = entry_point_info['plugin_class']
                
                # Check inheritance
                if not issubclass(plugin_class, PluginBase):
                    issues.append("Does not inherit from PluginBase")
                
                # Check instantiation
                try:
                    plugin_instance = plugin_class()
                    
                    # Check required attributes
                    if not hasattr(plugin_instance, 'metadata'):
                        issues.append("Missing metadata attribute")
                    if not hasattr(plugin_instance, 'initialize'):
                        issues.append("Missing initialize method")
                    if not hasattr(plugin_instance, 'execute'):
                        issues.append("Missing execute method")
                    if not hasattr(plugin_instance, 'cleanup'):
                        issues.append("Missing cleanup method")
                    
                except Exception as e:
                    issues.append(f"Cannot instantiate: {e}")
                
            except Exception as e:
                issues.append(f"Validation error: {e}")
            
            if issues:
                validation_results[plugin_name] = issues
        
        return validation_results


# Global entry point manager
entry_point_manager = PluginEntryPoint()
