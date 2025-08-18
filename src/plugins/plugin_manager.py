"""
Plugin Manager
Manages plugin loading, security, and lifecycle
"""

import asyncio
import json
import time
import importlib
import importlib.util
from typing import Dict, List, Any, Optional, Type, Callable
from pathlib import Path
import logging
import shutil
from datetime import datetime

from src.plugins.plugin_base import BasePlugin, PluginMetadata, PluginConfig
from src.plugins.plugin_sandbox import PluginSandbox

logger = logging.getLogger(__name__)

class PluginManager:
    """
    Manages plugin loading, security, and lifecycle
    """
    
    def __init__(self, plugins_dir: str = None):
        self.plugins_dir = Path(plugins_dir) if plugins_dir else Path.home() / '.monk-plugins'
        self.plugins_dir.mkdir(exist_ok=True)
        
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.plugin_configs: Dict[str, PluginConfig] = {}
        self.loaded_plugins: Dict[str, BasePlugin] = {}
        self.enabled_plugins: Dict[str, BasePlugin] = {}
        
        self.sandbox = PluginSandbox()
        self.plugin_hooks: Dict[str, List[BasePlugin]] = {}
        
        # Load plugin configurations
        self._load_plugin_configs()
    
    def _load_plugin_configs(self):
        """Load plugin configurations from disk"""
        config_file = self.plugins_dir / 'plugin_configs.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    configs = json.load(f)
                    for plugin_name, config_data in configs.items():
                        self.plugin_configs[plugin_name] = PluginConfig(**config_data)
            except Exception as e:
                logger.error(f"Error loading plugin configs: {e}")
    
    def _save_plugin_configs(self):
        """Save plugin configurations to disk"""
        config_file = self.plugins_dir / 'plugin_configs.json'
        try:
            configs = {}
            for plugin_name, config in self.plugin_configs.items():
                configs[plugin_name] = config.__dict__
            
            with open(config_file, 'w') as f:
                json.dump(configs, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving plugin configs: {e}")
    
    async def discover_plugins(self) -> Dict[str, Any]:
        """
        Discover available plugins in the plugins directory
        """
        discovered_plugins = {}
        
        try:
            for plugin_dir in self.plugins_dir.iterdir():
                if plugin_dir.is_dir() and not plugin_dir.name.startswith('.'):
                    plugin_info = await self._inspect_plugin_directory(plugin_dir)
                    if plugin_info:
                        discovered_plugins[plugin_dir.name] = plugin_info
            
            logger.info(f"Discovered {len(discovered_plugins)} plugins")
            return {
                'success': True,
                'plugins': discovered_plugins,
                'total': len(discovered_plugins)
            }
            
        except Exception as e:
            logger.error(f"Error discovering plugins: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _inspect_plugin_directory(self, plugin_dir: Path) -> Optional[Dict[str, Any]]:
        """Inspect a plugin directory for metadata and structure"""
        try:
            # Check for plugin.json
            plugin_json = plugin_dir / 'plugin.json'
            if not plugin_json.exists():
                return None
            
            # Load metadata
            with open(plugin_json, 'r') as f:
                metadata_dict = json.load(f)
            
            # Check for main plugin file
            main_file = None
            for possible_main in ['plugin.py', 'main.py', '__init__.py']:
                if (plugin_dir / possible_main).exists():
                    main_file = possible_main
                    break
            
            if not main_file:
                return None
            
            # Check for requirements
            requirements_file = plugin_dir / 'requirements.txt'
            requirements = []
            if requirements_file.exists():
                with open(requirements_file, 'r') as f:
                    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            # Check for README
            readme_file = plugin_dir / 'README.md'
            readme = ""
            if readme_file.exists():
                with open(readme_file, 'r') as f:
                    readme = f.read()
            
            return {
                'metadata': metadata_dict,
                'main_file': main_file,
                'requirements': requirements,
                'readme': readme,
                'directory': str(plugin_dir),
                'last_modified': plugin_dir.stat().st_mtime
            }
            
        except Exception as e:
            logger.error(f"Error inspecting plugin directory {plugin_dir}: {e}")
            return None
    
    async def load_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """
        Load a specific plugin
        """
        if plugin_name in self.loaded_plugins:
            return {
                'success': False,
                'error': f'Plugin {plugin_name} is already loaded'
            }
        
        try:
            plugin_dir = self.plugins_dir / plugin_name
            if not plugin_dir.exists():
                return {
                    'success': False,
                    'error': f'Plugin directory not found: {plugin_name}'
                }
            
            # Inspect plugin
            plugin_info = await self._inspect_plugin_directory(plugin_dir)
            if not plugin_info:
                return {
                    'success': False,
                    'error': f'Invalid plugin structure: {plugin_name}'
                }
            
            # Load plugin module
            plugin_module = await self._load_plugin_module(plugin_dir, plugin_info['main_file'])
            if not plugin_module:
                return {
                    'success': False,
                    'error': f'Failed to load plugin module: {plugin_name}'
                }
            
            # Find plugin class
            plugin_class = self._find_plugin_class(plugin_module)
            if not plugin_class:
                return {
                    'success': False,
                    'error': f'No plugin class found in module: {plugin_name}'
                }
            
            # Create plugin instance
            plugin_instance = plugin_class(str(plugin_dir))
            
            # Initialize plugin
            if not await plugin_instance.initialize():
                return {
                    'success': False,
                    'error': f'Plugin initialization failed: {plugin_name}'
                }
            
            # Store plugin
            self.plugins[plugin_name] = plugin_instance
            self.plugin_metadata[plugin_name] = PluginMetadata(**plugin_info['metadata'])
            
            # Load configuration
            if plugin_name not in self.plugin_configs:
                self.plugin_configs[plugin_name] = PluginConfig()
            
            # Call load hook
            await plugin_instance.on_load(self)
            
            # Mark as loaded
            self.loaded_plugins[plugin_name] = plugin_instance
            
            # Auto-enable if configured
            if self.plugin_configs[plugin_name].auto_load:
                await self.enable_plugin(plugin_name)
            
            logger.info(f"Plugin {plugin_name} loaded successfully")
            
            return {
                'success': True,
                'plugin': plugin_instance,
                'metadata': self.plugin_metadata[plugin_name],
                'message': f'Plugin {plugin_name} loaded successfully'
            }
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _load_plugin_module(self, plugin_dir: Path, main_file: str) -> Optional[Any]:
        """Load a plugin module from disk"""
        try:
            # Create module spec
            spec = importlib.util.spec_from_file_location(
                f"plugin_{plugin_dir.name}",
                plugin_dir / main_file
            )
            
            if not spec or not spec.loader:
                return None
            
            # Load module
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return module
            
        except Exception as e:
            logger.error(f"Error loading plugin module: {e}")
            return None
    
    def _find_plugin_class(self, module) -> Optional[Type[BasePlugin]]:
        """Find a plugin class in a module"""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, BasePlugin) and 
                attr != BasePlugin):
                return attr
        return None
    
    async def unload_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """
        Unload a specific plugin
        """
        if plugin_name not in self.loaded_plugins:
            return {
                'success': False,
                'error': f'Plugin {plugin_name} is not loaded'
            }
        
        try:
            plugin = self.loaded_plugins[plugin_name]
            
            # Disable if enabled
            if plugin_name in self.enabled_plugins:
                await self.disable_plugin(plugin_name)
            
            # Call unload hook
            await plugin.on_unload()
            
            # Cleanup plugin
            await plugin.cleanup()
            
            # Remove from loaded plugins
            del self.loaded_plugins[plugin_name]
            
            # Remove from plugins dict
            if plugin_name in self.plugins:
                del self.plugins[plugin_name]
            
            logger.info(f"Plugin {plugin_name} unloaded successfully")
            
            return {
                'success': True,
                'message': f'Plugin {plugin_name} unloaded successfully'
            }
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def enable_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """
        Enable a loaded plugin
        """
        if plugin_name not in self.loaded_plugins:
            return {
                'success': False,
                'error': f'Plugin {plugin_name} is not loaded'
            }
        
        if plugin_name in self.enabled_plugins:
            return {
                'success': False,
                'error': f'Plugin {plugin_name} is already enabled'
            }
        
        try:
            plugin = self.loaded_plugins[plugin_name]
            
            # Check permissions
            if not self.sandbox.check_permissions(plugin):
                return {
                    'success': False,
                    'error': f'Plugin {plugin_name} does not have required permissions'
                }
            
            # Enable plugin
            await plugin.on_enable()
            self.enabled_plugins[plugin_name] = plugin
            
            # Update config
            self.plugin_configs[plugin_name].enabled = True
            self._save_plugin_configs()
            
            logger.info(f"Plugin {plugin_name} enabled successfully")
            
            return {
                'success': True,
                'message': f'Plugin {plugin_name} enabled successfully'
            }
            
        except Exception as e:
            logger.error(f"Error enabling plugin {plugin_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def disable_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """
        Disable an enabled plugin
        """
        if plugin_name not in self.enabled_plugins:
            return {
                'success': False,
                'error': f'Plugin {plugin_name} is not enabled'
            }
        
        try:
            plugin = self.enabled_plugins[plugin_name]
            
            # Disable plugin
            await plugin.on_disable()
            del self.enabled_plugins[plugin_name]
            
            # Update config
            self.plugin_configs[plugin_name].enabled = False
            self._save_plugin_configs()
            
            logger.info(f"Plugin {plugin_name} disabled successfully")
            
            return {
                'success': True,
                'message': f'Plugin {plugin_name} disabled successfully'
            }
            
        except Exception as e:
            logger.error(f"Error disabling plugin {plugin_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def reload_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """
        Reload a plugin
        """
        if plugin_name in self.loaded_plugins:
            await self.unload_plugin(plugin_name)
        
        return await self.load_plugin(plugin_name)
    
    async def install_plugin(self, plugin_source: str, plugin_name: str = None) -> Dict[str, Any]:
        """
        Install a plugin from source
        """
        try:
            # For now, assume plugin_source is a local path
            # In a real implementation, this could handle git repos, package managers, etc.
            source_path = Path(plugin_source)
            
            if not source_path.exists():
                return {
                    'success': False,
                    'error': f'Plugin source not found: {plugin_source}'
                }
            
            # Determine plugin name
            if not plugin_name:
                plugin_name = source_path.name
            
            # Check if plugin already exists
            target_path = self.plugins_dir / plugin_name
            if target_path.exists():
                return {
                    'success': False,
                    'error': f'Plugin {plugin_name} already exists'
                }
            
            # Copy plugin to plugins directory
            if source_path.is_dir():
                shutil.copytree(source_path, target_path)
            else:
                target_path.mkdir()
                shutil.copy2(source_path, target_path)
            
            logger.info(f"Plugin {plugin_name} installed successfully")
            
            return {
                'success': True,
                'message': f'Plugin {plugin_name} installed successfully'
            }
            
        except Exception as e:
            logger.error(f"Error installing plugin: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def uninstall_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """
        Uninstall a plugin
        """
        try:
            # Unload if loaded
            if plugin_name in self.loaded_plugins:
                await self.unload_plugin(plugin_name)
            
            # Remove plugin directory
            plugin_path = self.plugins_dir / plugin_name
            if plugin_path.exists():
                shutil.rmtree(plugin_path)
            
            # Remove from configs
            if plugin_name in self.plugin_configs:
                del self.plugin_configs[plugin_name]
                self._save_plugin_configs()
            
            logger.info(f"Plugin {plugin_name} uninstalled successfully")
            
            return {
                'success': True,
                'message': f'Plugin {plugin_name} uninstalled successfully'
            }
            
        except Exception as e:
            logger.error(f"Error uninstalling plugin {plugin_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_plugin_commands(self) -> Dict[str, Callable]:
        """Get all available plugin commands"""
        commands = {}
        for plugin_name, plugin in self.enabled_plugins.items():
            for command_name, command_func in plugin.commands.items():
                commands[f"{plugin_name}.{command_name}"] = command_func
        return commands
    
    def get_plugin_api_endpoints(self) -> Dict[str, Callable]:
        """Get all available plugin API endpoints"""
        endpoints = {}
        for plugin_name, plugin in self.enabled_plugins.items():
            for endpoint, endpoint_func in plugin.api_endpoints.items():
                endpoints[f"{plugin_name}.{endpoint}"] = endpoint_func
        return endpoints
    
    async def call_plugin_hook(self, hook_name: str, *args, **kwargs):
        """Call a hook on all enabled plugins"""
        for plugin_name, plugin in self.enabled_plugins.items():
            try:
                await plugin.call_hook(hook_name, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error calling hook {hook_name} on plugin {plugin_name}: {e}")
    
    def get_plugin_status(self, plugin_name: str = None) -> Dict[str, Any]:
        """Get status of plugins"""
        if plugin_name:
            if plugin_name in self.loaded_plugins:
                return {
                    'success': True,
                    'status': self.loaded_plugins[plugin_name].get_status()
                }
            else:
                return {
                    'success': False,
                    'error': f'Plugin {plugin_name} not found'
                }
        
        # Return status of all plugins
        status = {}
        for name, plugin in self.loaded_plugins.items():
            status[name] = plugin.get_status()
        
        return {
            'success': True,
            'plugins': status,
            'total_loaded': len(self.loaded_plugins),
            'total_enabled': len(self.enabled_plugins)
        }
    
    def get_plugin_help(self, plugin_name: str = None) -> Dict[str, Any]:
        """Get help information for plugins"""
        if plugin_name:
            if plugin_name in self.loaded_plugins:
                return {
                    'success': True,
                    'help': self.loaded_plugins[plugin_name].get_help()
                }
            else:
                return {
                    'success': False,
                    'error': f'Plugin {plugin_name} not found'
                }
        
        # Return help for all plugins
        help_info = {}
        for name, plugin in self.loaded_plugins.items():
            help_info[name] = plugin.get_help()
        
        return {
            'success': True,
            'help': help_info
        }
    
    async def shutdown(self):
        """Shutdown the plugin manager"""
        try:
            # Disable all plugins
            for plugin_name in list(self.enabled_plugins.keys()):
                await self.disable_plugin(plugin_name)
            
            # Unload all plugins
            for plugin_name in list(self.loaded_plugins.keys()):
                await self.unload_plugin(plugin_name)
            
            # Save configurations
            self._save_plugin_configs()
            
            logger.info("Plugin Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during plugin manager shutdown: {e}")

# Global plugin manager instance
plugin_manager = PluginManager()
