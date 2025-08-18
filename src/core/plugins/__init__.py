"""
Plugin System for Monk CLI

This module provides a comprehensive plugin architecture that allows:
- Dynamic loading of community plugins
- Secure plugin execution environments
- Integration with TreeQuest AI agents
- Memory-aware plugin operations
"""

from .base import PluginBase, PluginContext, PluginMetadata
from .registry import PluginRegistry
from .validator import PluginValidator
from .entry_points import PluginEntryPoint
from .routing import PluginRouter
from .treequest_integration import TreeQuestPluginIntegration

__all__ = [
    'PluginBase',
    'PluginContext', 
    'PluginMetadata',
    'PluginRegistry',
    'PluginValidator',
    'PluginEntryPoint',
    'PluginRouter',
    'TreeQuestPluginIntegration'
]

# Global plugin registry instance
plugin_registry = PluginRegistry()
