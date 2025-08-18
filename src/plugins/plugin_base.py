"""
Base Plugin Class
Base class that all plugins must inherit from
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class PluginMetadata:
    """Plugin metadata information"""
    name: str
    version: str
    description: str
    author: str
    license: str
    homepage: str = ""
    repository: str = ""
    keywords: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    python_version: str = ">=3.8"
    created_at: str = field(default_factory=lambda: time.time())
    updated_at: str = field(default_factory=lambda: time.time())

@dataclass
class PluginConfig:
    """Plugin configuration"""
    enabled: bool = True
    auto_load: bool = False
    priority: int = 100
    settings: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)

class BasePlugin(ABC):
    """
    Base class for all plugins
    """
    
    def __init__(self, plugin_path: str = None):
        self.plugin_path = Path(plugin_path) if plugin_path else None
        self.metadata: Optional[PluginMetadata] = None
        self.config: Optional[PluginConfig] = None
        self.manager = None
        self.is_loaded = False
        self.is_enabled = False
        self.load_time = None
        self.error_count = 0
        self.last_error = None
        
        # Plugin hooks
        self.hooks: Dict[str, List[Callable]] = {}
        self.commands: Dict[str, Callable] = {}
        self.api_endpoints: Dict[str, Callable] = {}
        
        # Initialize plugin
        self._load_metadata()
        self._load_config()
        self._register_default_hooks()
    
    def _load_metadata(self):
        """Load plugin metadata"""
        if self.plugin_path and (self.plugin_path / 'plugin.json').exists():
            try:
                with open(self.plugin_path / 'plugin.json', 'r') as f:
                    metadata_dict = json.load(f)
                    self.metadata = PluginMetadata(**metadata_dict)
            except Exception as e:
                logger.error(f"Error loading plugin metadata: {e}")
                self._create_default_metadata()
        else:
            self._create_default_metadata()
    
    def _create_default_metadata(self):
        """Create default metadata for the plugin"""
        self.metadata = PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description="Plugin description",
            author="Unknown",
            license="MIT"
        )
    
    def _load_config(self):
        """Load plugin configuration"""
        if self.plugin_path and (self.plugin_path / 'config.json').exists():
            try:
                with open(self.plugin_path / 'config.json', 'r') as f:
                    config_dict = json.load(f)
                    self.config = PluginConfig(**config_dict)
            except Exception as e:
                logger.error(f"Error loading plugin config: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration for the plugin"""
        self.config = PluginConfig(
            enabled=True,
            auto_load=False,
            priority=100,
            settings={},
            permissions=[]
        )
    
    def _register_default_hooks(self):
        """Register default plugin hooks"""
        self.register_hook('on_load', self.on_load)
        self.register_hook('on_enable', self.on_enable)
        self.register_hook('on_disable', self.on_disable)
        self.register_hook('on_unload', self.on_unload)
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the plugin. Must be implemented by subclasses.
        Returns True if initialization was successful.
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """
        Cleanup plugin resources. Must be implemented by subclasses.
        Returns True if cleanup was successful.
        """
        pass
    
    async def on_load(self, manager) -> None:
        """
        Called when the plugin is loaded by the plugin manager.
        Override this method to perform custom load logic.
        """
        self.manager = manager
        self.is_loaded = True
        self.load_time = time.time()
        logger.info(f"Plugin {self.metadata.name} loaded successfully")
    
    async def on_enable(self) -> None:
        """
        Called when the plugin is enabled.
        Override this method to perform custom enable logic.
        """
        self.is_enabled = True
        logger.info(f"Plugin {self.metadata.name} enabled")
    
    async def on_disable(self) -> None:
        """
        Called when the plugin is disabled.
        Override this method to perform custom disable logic.
        """
        self.is_enabled = False
        logger.info(f"Plugin {self.metadata.name} disabled")
    
    async def on_unload(self) -> None:
        """
        Called when the plugin is unloaded by the plugin manager.
        Override this method to perform custom unload logic.
        """
        self.is_loaded = False
        self.is_enabled = False
        self.manager = None
        logger.info(f"Plugin {self.metadata.name} unloaded")
    
    def register_hook(self, hook_name: str, callback: Callable):
        """Register a hook callback"""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)
    
    def unregister_hook(self, hook_name: str, callback: Callable):
        """Unregister a hook callback"""
        if hook_name in self.hooks and callback in self.hooks[hook_name]:
            self.hooks[hook_name].remove(callback)
    
    async def call_hook(self, hook_name: str, *args, **kwargs):
        """Call all registered hooks for a given hook name"""
        if hook_name in self.hooks:
            for callback in self.hooks[hook_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in plugin hook {hook_name}: {e}")
                    self.error_count += 1
                    self.last_error = str(e)
    
    def register_command(self, command_name: str, callback: Callable):
        """Register a command that can be called from the CLI"""
        self.commands[command_name] = callback
        logger.debug(f"Registered command: {command_name}")
    
    def unregister_command(self, command_name: str):
        """Unregister a command"""
        if command_name in self.commands:
            del self.commands[command_name]
            logger.debug(f"Unregistered command: {command_name}")
    
    def register_api_endpoint(self, endpoint: str, callback: Callable):
        """Register an API endpoint"""
        self.api_endpoints[endpoint] = callback
        logger.debug(f"Registered API endpoint: {endpoint}")
    
    def unregister_api_endpoint(self, endpoint: str):
        """Unregister an API endpoint"""
        if endpoint in self.api_endpoints:
            del self.api_endpoints[endpoint]
            logger.debug(f"Unregistered API endpoint: {endpoint}")
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a plugin setting"""
        return self.config.settings.get(key, default)
    
    def set_setting(self, key: str, value: Any):
        """Set a plugin setting"""
        self.config.settings[key] = value
        self._save_config()
    
    def has_permission(self, permission: str) -> bool:
        """Check if plugin has a specific permission"""
        return permission in self.config.permissions
    
    def add_permission(self, permission: str):
        """Add a permission to the plugin"""
        if permission not in self.config.permissions:
            self.config.permissions.append(permission)
            self._save_config()
    
    def remove_permission(self, permission: str):
        """Remove a permission from the plugin"""
        if permission in self.config.permissions:
            self.config.permissions.remove(permission)
            self._save_config()
    
    def _save_config(self):
        """Save plugin configuration to disk"""
        if self.plugin_path:
            try:
                config_file = self.plugin_path / 'config.json'
                with open(config_file, 'w') as f:
                    json.dump(self.config.__dict__, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving plugin config: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin status information"""
        return {
            'name': self.metadata.name,
            'version': self.metadata.version,
            'enabled': self.is_enabled,
            'loaded': self.is_loaded,
            'load_time': self.load_time,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'hooks_count': len(self.hooks),
            'commands_count': len(self.commands),
            'api_endpoints_count': len(self.api_endpoints)
        }
    
    def get_help(self) -> str:
        """Get plugin help information"""
        help_text = f"""
Plugin: {self.metadata.name} v{self.metadata.version}
Description: {self.metadata.description}
Author: {self.metadata.author}
License: {self.metadata.license}

Commands: {', '.join(self.commands.keys()) if self.commands else 'None'}
API Endpoints: {', '.join(self.api_endpoints.keys()) if self.api_endpoints else 'None'}
        """
        return help_text.strip()
    
    def __str__(self):
        return f"Plugin({self.metadata.name} v{self.metadata.version})"
    
    def __repr__(self):
        return self.__str__()
