"""
Example Plugin for Monk CLI

A command plugin that adds new CLI functionality.
"""

from typing import List, Dict, Any
from src.core.plugins.base import PluginCommand, PluginMetadata, PluginType, PluginContext


class ExamplePlugin(PluginCommand):
    """Example command plugin for Monk CLI"""
    
    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="example",
            version="0.1.0",
            description="Adds example functionality to Monk CLI",
            author="Your Name",
            plugin_type=PluginType.COMMAND,
            dependencies=[],
            tags=["example", "command", "monk"],
            memory_access=["*"]  # Access to all memory
        )
        
        self.command_name = "example"
        self.command_help = "Execute example functionality"
        self.command_usage = "/example [options]"
    
    def initialize(self, context: PluginContext) -> bool:
        """Initialize the plugin with execution context"""
        try:
            self._context = context
            logger.info(f"Initialized example plugin")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize example plugin: {e}")
            return False
    
    def execute(self, command: str, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Execute the example command"""
        try:
            # Access memory context if needed
            project_info = self._context.get_memory("project_info", "No project info")
            
            # Your plugin logic here
            result = f"Executed example command with args: {args}"
            if kwargs:
                result += f" and kwargs: {kwargs}"
            
            # Example: Access project context
            project_path = self._context.project_path
            result += f"\nProject: {project_path}"
            
            return result
            
        except Exception as e:
            return f"Error executing example: {e}"
    
    def cleanup(self) -> None:
        """Clean up plugin resources"""
        self._context = None
        logger.info(f"Cleaned up example plugin")


# Plugin instance
example_plugin = ExamplePlugin()
