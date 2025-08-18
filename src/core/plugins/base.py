"""
Base Plugin Classes for Monk CLI

Defines the core interfaces and base classes that all plugins must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import time
import json


class PluginType(Enum):
    """Types of plugins supported by Monk CLI"""
    COMMAND = "command"           # Adds new CLI commands
    ANALYZER = "analyzer"         # Project analysis capabilities
    INTEGRATION = "integration"   # External service integration
    AI_AGENT = "ai_agent"        # Custom AI agent implementation
    WORKFLOW = "workflow"         # Automated workflow steps
    UI_ENHANCEMENT = "ui"        # Interface improvements


class PluginStatus(Enum):
    """Plugin execution and lifecycle status"""
    DISABLED = "disabled"
    ENABLED = "enabled"
    LOADING = "loading"
    ERROR = "error"
    EXECUTING = "executing"
    COMPLETED = "completed"


@dataclass
class PluginMetadata:
    """Metadata describing a plugin"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    memory_access: List[str] = field(default_factory=list)  # Memory keys this plugin can access
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'plugin_type': self.plugin_type.value,
            'dependencies': self.dependencies,
            'tags': self.tags,
            'memory_access': self.memory_access,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginMetadata':
        """Create metadata from dictionary"""
        data['plugin_type'] = PluginType(data['plugin_type'])
        return cls(**data)


@dataclass
class PluginContext:
    """Context provided to plugins during execution"""
    project_path: str
    project_context: Dict[str, Any]
    memory_context: Dict[str, Any]
    user_preferences: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    plugin_registry: 'PluginRegistry'
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """Safely access memory context"""
        return self.memory_context.get(key, default)
    
    def get_project_info(self, key: str, default: Any = None) -> Any:
        """Safely access project context"""
        return self.project_context.get(key, default)
    
    def log_activity(self, message: str, level: str = "info") -> None:
        """Log plugin activity"""
        # This will be implemented to integrate with Monk's logging system
        pass


class PluginBase(ABC):
    """
    Base class for all Monk CLI plugins
    
    Plugins must implement this interface to be loaded and executed
    by the plugin system.
    """
    
    def __init__(self):
        self.metadata: Optional[PluginMetadata] = None
        self.status: PluginStatus = PluginStatus.DISABLED
        self.last_executed: Optional[float] = None
        self.execution_count: int = 0
        self.error_count: int = 0
        self._context: Optional[PluginContext] = None
    
    @abstractmethod
    def initialize(self, context: PluginContext) -> bool:
        """
        Initialize the plugin with execution context
        
        Args:
            context: Plugin execution context
            
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute(self, command: str, args: List[str], kwargs: Dict[str, Any]) -> Any:
        """
        Execute the plugin's main functionality
        
        Args:
            command: Command to execute
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Plugin execution result
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources"""
        pass
    
    def get_help(self) -> str:
        """Get help text for the plugin"""
        if self.metadata:
            return f"{self.metadata.name} v{self.metadata.version}\n{self.metadata.description}"
        return "No help available"
    
    def get_commands(self) -> List[str]:
        """Get list of commands this plugin provides"""
        return []
    
    def can_access_memory(self, key: str) -> bool:
        """Check if plugin can access specific memory key"""
        if not self.metadata or not self.metadata.memory_access:
            return False
        return key in self.metadata.memory_access or "*" in self.metadata.memory_access
    
    def validate_dependencies(self) -> List[str]:
        """Validate plugin dependencies and return missing ones"""
        missing = []
        for dep in self.metadata.dependencies if self.metadata else []:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        return missing
    
    def update_status(self, status: PluginStatus) -> None:
        """Update plugin status"""
        self.status = status
        if status == PluginStatus.COMPLETED:
            self.last_executed = time.time()
            self.execution_count += 1
        elif status == PluginStatus.ERROR:
            self.error_count += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get plugin performance metrics"""
        return {
            'execution_count': self.execution_count,
            'error_count': self.error_count,
            'last_executed': self.last_executed,
            'status': self.status.value,
            'uptime': time.time() - (self.metadata.created_at if self.metadata else time.time())
        }


class PluginCommand(PluginBase):
    """Base class for command plugins that add new CLI commands"""
    
    def __init__(self):
        super().__init__()
        self.command_name: str = ""
        self.command_help: str = ""
        self.command_usage: str = ""
    
    @abstractmethod
    def execute(self, command: str, args: List[str], kwargs: Dict[str, Any]) -> str:
        """
        Execute the command plugin
        
        Args:
            command: Command name
            args: Command arguments
            kwargs: Command options
            
        Returns:
            Command output as string
        """
        pass
    
    def get_help(self) -> str:
        """Get command help text"""
        return f"{self.command_name}: {self.command_help}\nUsage: {self.command_usage}"
    
    def get_commands(self) -> List[str]:
        """Get command names this plugin provides"""
        return [self.command_name] if self.command_name else []


class PluginAnalyzer(PluginBase):
    """Base class for analyzer plugins that provide project analysis capabilities"""
    
    def __init__(self):
        super().__init__()
        self.analysis_types: List[str] = []
    
    @abstractmethod
    def analyze(self, analysis_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform analysis of specified type
        
        Args:
            analysis_type: Type of analysis to perform
            context: Analysis context
            
        Returns:
            Analysis results
        """
        pass
    
    def get_analysis_types(self) -> List[str]:
        """Get supported analysis types"""
        return self.analysis_types
    
    def can_analyze(self, analysis_type: str) -> bool:
        """Check if plugin can perform specific analysis"""
        return analysis_type in self.analysis_types


class PluginIntegration(PluginBase):
    """Base class for integration plugins that connect to external services"""
    
    def __init__(self):
        super().__init__()
        self.service_name: str = ""
        self.api_endpoints: List[str] = []
    
    @abstractmethod
    def connect(self, credentials: Dict[str, Any]) -> bool:
        """
        Connect to external service
        
        Args:
            credentials: Service connection credentials
            
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from external service"""
        pass
    
    def is_connected(self) -> bool:
        """Check if plugin is connected to service"""
        return self.status == PluginStatus.ENABLED
