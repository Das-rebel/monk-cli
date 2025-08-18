"""
Plugins Package
Secure plugin manager with sandboxing, API, and marketplace integration
"""

from .plugin_manager import PluginManager
from .plugin_base import BasePlugin
from .plugin_sandbox import PluginSandbox

__all__ = [
    'PluginManager',
    'BasePlugin',
    'PluginSandbox'
]
