"""
Plugin Scaffolding Tool for Monk CLI

Generates plugin templates and project structure for rapid plugin development.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PluginScaffolder:
    """
    Generates plugin project structure and templates
    
    Creates:
    - Plugin directory structure
    - Base plugin classes
    - Setup.py and requirements.txt
    - README and documentation
    - Test files
    """
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.templates_dir = Path(__file__).parent / "templates"
    
    def create_plugin(self, plugin_name: str, plugin_type: str = "command", 
                     author: str = "Unknown", description: str = "", 
                     dependencies: Optional[list] = None) -> bool:
        """
        Create a complete plugin project structure
        
        Args:
            plugin_name: Name of the plugin
            plugin_type: Type of plugin (command, analyzer, integration, etc.)
            author: Plugin author name
            description: Plugin description
            dependencies: List of Python dependencies
            
        Returns:
            True if plugin created successfully
        """
        try:
            # Validate plugin name
            if not self._validate_plugin_name(plugin_name):
                logger.error(f"Invalid plugin name: {plugin_name}")
                return False
            
            # Create plugin directory
            plugin_dir = self.output_dir / plugin_name
            if plugin_dir.exists():
                logger.error(f"Plugin directory already exists: {plugin_dir}")
                return False
            
            plugin_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate plugin structure
            self._create_plugin_structure(plugin_dir, plugin_name, plugin_type, 
                                        author, description, dependencies or [])
            
            # Copy template files
            self._copy_template_files(plugin_dir, plugin_name, plugin_type)
            
            # Generate setup.py
            self._generate_setup_py(plugin_dir, plugin_name, plugin_type, 
                                  author, description, dependencies or [])
            
            # Generate README
            self._generate_readme(plugin_dir, plugin_name, plugin_type, description)
            
            # Generate requirements.txt
            self._generate_requirements(plugin_dir, dependencies or [])
            
            # Generate test files
            self._generate_test_files(plugin_dir, plugin_name)
            
            logger.info(f"Successfully created plugin: {plugin_name}")
            logger.info(f"Plugin directory: {plugin_dir}")
            logger.info(f"Next steps:")
            logger.info(f"  1. cd {plugin_name}")
            logger.info(f"  2. pip install -e .")
            logger.info(f"  3. python -m pytest tests/")
            logger.info(f"  4. monk plugin test {plugin_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating plugin {plugin_name}: {e}")
            return False
    
    def _validate_plugin_name(self, name: str) -> bool:
        """Validate plugin name format"""
        import re
        # Must be valid Python identifier and follow naming conventions
        if not re.match(r'^[a-z][a-z0-9_]*$', name):
            return False
        # Allow common names that might be useful for demos and examples
        if name in ['test']:
            return False
        return True
    
    def _create_plugin_structure(self, plugin_dir: Path, plugin_name: str, 
                                plugin_type: str, author: str, description: str, 
                                dependencies: list) -> None:
        """Create the plugin directory structure"""
        
        # Create subdirectories
        (plugin_dir / "src" / plugin_name).mkdir(parents=True, exist_ok=True)
        (plugin_dir / "tests").mkdir(exist_ok=True)
        (plugin_dir / "docs").mkdir(exist_ok=True)
        
        # Create __init__.py files
        (plugin_dir / "src" / "__init__.py").touch()
        (plugin_dir / "src" / plugin_name / "__init__.py").touch()
        (plugin_dir / "tests" / "__init__.py").touch()
    
    def _copy_template_files(self, plugin_dir: Path, plugin_name: str, plugin_type: str) -> None:
        """Copy template files based on plugin type"""
        
        # Copy base plugin template
        plugin_file = plugin_dir / "src" / plugin_name / "plugin.py"
        self._generate_plugin_file(plugin_file, plugin_name, plugin_type)
        
        # Copy __init__.py with plugin registration
        init_file = plugin_dir / "src" / plugin_name / "__init__.py"
        self._generate_init_file(init_file, plugin_name, plugin_type)
    
    def _generate_plugin_file(self, plugin_file: Path, plugin_name: str, plugin_type: str) -> None:
        """Generate the main plugin implementation file"""
        
        if plugin_type == "command":
            template = self._get_command_plugin_template(plugin_name)
        elif plugin_type == "analyzer":
            template = self._get_analyzer_plugin_template(plugin_name)
        elif plugin_type == "integration":
            template = self._get_integration_plugin_template(plugin_name)
        else:
            template = self._get_base_plugin_template(plugin_name, plugin_type)
        
        plugin_file.write_text(template)
    
    def _get_command_plugin_template(self, plugin_name: str) -> str:
        """Get template for command plugin"""
        return f'''"""
{plugin_name.title()} Plugin for Monk CLI

A command plugin that adds new CLI functionality.
"""

from typing import List, Dict, Any
from src.core.plugins.base import PluginCommand, PluginMetadata, PluginType, PluginContext


class {plugin_name.title()}Plugin(PluginCommand):
    """{plugin_name.title()} command plugin for Monk CLI"""
    
    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="{plugin_name}",
            version="0.1.0",
            description="Adds {plugin_name} functionality to Monk CLI",
            author="Your Name",
            plugin_type=PluginType.COMMAND,
            dependencies=[],
            tags=["{plugin_name}", "command", "monk"],
            memory_access=["*"]  # Access to all memory
        )
        
        self.command_name = "{plugin_name}"
        self.command_help = "Execute {plugin_name} functionality"
        self.command_usage = "/{plugin_name} [options]"
    
    def initialize(self, context: PluginContext) -> bool:
        """Initialize the plugin with execution context"""
        try:
            self._context = context
            logger.info(f"Initialized {plugin_name} plugin")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {plugin_name} plugin: {{e}}")
            return False
    
    def execute(self, command: str, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Execute the {plugin_name} command"""
        try:
            # Access memory context if needed
            project_info = self._context.get_memory("project_info", "No project info")
            
            # Your plugin logic here
            result = f"Executed {plugin_name} command with args: {{args}}"
            if kwargs:
                result += f" and kwargs: {{kwargs}}"
            
            # Example: Access project context
            project_path = self._context.project_path
            result += f"\\nProject: {{project_path}}"
            
            return result
            
        except Exception as e:
            return f"Error executing {plugin_name}: {{e}}"
    
    def cleanup(self) -> None:
        """Clean up plugin resources"""
        self._context = None
        logger.info(f"Cleaned up {plugin_name} plugin")


# Plugin instance
{plugin_name}_plugin = {plugin_name.title()}Plugin()
'''
    
    def _get_analyzer_plugin_template(self, plugin_name: str) -> str:
        """Get template for analyzer plugin"""
        return f'''"""
{plugin_name.title()} Analyzer Plugin for Monk CLI

An analyzer plugin that provides project analysis capabilities.
"""

from typing import List, Dict, Any
from src.core.plugins.base import PluginAnalyzer, PluginMetadata, PluginType, PluginContext


class {plugin_name.title()}Analyzer(PluginAnalyzer):
    """{plugin_name.title()} analyzer plugin for Monk CLI"""
    
    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="{plugin_name}_analyzer",
            version="0.1.0",
            description="Provides {plugin_name} analysis for projects",
            author="Your Name",
            plugin_type=PluginType.ANALYZER,
            dependencies=[],
            tags=["{plugin_name}", "analyzer", "monk"],
            memory_access=["project_*", "analysis_*"]
        )
        
        self.analysis_types = ["{plugin_name}", "basic", "detailed"]
    
    def initialize(self, context: PluginContext) -> bool:
        """Initialize the analyzer plugin"""
        try:
            self._context = context
            logger.info(f"Initialized {plugin_name} analyzer plugin")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {plugin_name} analyzer: {{e}}")
            return False
    
    def execute(self, command: str, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Execute analyzer functionality"""
        try:
            analysis_type = kwargs.get("type", "basic")
            if analysis_type not in self.analysis_types:
                return f"Unknown analysis type: {{analysis_type}}. Available: {{', '.join(self.analysis_types)}}"
            
            # Perform analysis
            result = self.analyze(analysis_type, {{"args": args, "kwargs": kwargs}})
            
            return f"Analysis result for {{analysis_type}}: {{result}}"
            
        except Exception as e:
            return f"Error in {plugin_name} analysis: {{e}}"
    
    def analyze(self, analysis_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the specified analysis"""
        if analysis_type == "{plugin_name}":
            return self._analyze_{plugin_name}(context)
        elif analysis_type == "basic":
            return self._analyze_basic(context)
        elif analysis_type == "detailed":
            return self._analyze_detailed(context)
        else:
            return {{"error": f"Unknown analysis type: {{analysis_type}}"}}
    
    def _analyze_{plugin_name}(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform {plugin_name}-specific analysis"""
        return {{
            "type": "{plugin_name}",
            "status": "completed",
            "findings": ["Sample finding 1", "Sample finding 2"],
            "recommendations": ["Recommendation 1", "Recommendation 2"]
        }}
    
    def _analyze_basic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic analysis"""
        return {{
            "type": "basic",
            "status": "completed",
            "summary": "Basic analysis completed"
        }}
    
    def _analyze_detailed(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed analysis"""
        return {{
            "type": "detailed",
            "status": "completed",
            "details": "Detailed analysis with comprehensive results"
        }}
    
    def cleanup(self) -> None:
        """Clean up analyzer resources"""
        self._context = None
        logger.info(f"Cleaned up {plugin_name} analyzer plugin")


# Plugin instance
{plugin_name}_analyzer = {plugin_name.title()}Analyzer()
'''
    
    def _get_integration_plugin_template(self, plugin_name: str) -> str:
        """Get template for integration plugin"""
        return f'''"""
{plugin_name.title()} Integration Plugin for Monk CLI

An integration plugin that connects to external services.
"""

from typing import List, Dict, Any
from src.core.plugins.base import PluginIntegration, PluginMetadata, PluginType, PluginContext


class {plugin_name.title()}Integration(PluginIntegration):
    """{plugin_name.title()} integration plugin for Monk CLI"""
    
    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="{plugin_name}_integration",
            version="0.1.0",
            description="Integrates with {plugin_name} service",
            author="Your Name",
            plugin_type=PluginType.INTEGRATION,
            dependencies=[],
            tags=["{plugin_name}", "integration", "monk"],
            memory_access=["credentials_*", "config_*"]
        )
        
        self.service_name = "{plugin_name}"
        self.api_endpoints = ["/api/v1/endpoint1", "/api/v1/endpoint2"]
        self._connected = False
    
    def initialize(self, context: PluginContext) -> bool:
        """Initialize the integration plugin"""
        try:
            self._context = context
            logger.info(f"Initialized {plugin_name} integration plugin")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {plugin_name} integration: {{e}}")
            return False
    
    def execute(self, command: str, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Execute integration functionality"""
        try:
            if command == "connect":
                credentials = kwargs.get("credentials", {{}})
                if self.connect(credentials):
                    return f"Successfully connected to {{self.service_name}}"
                else:
                    return f"Failed to connect to {{self.service_name}}"
            
            elif command == "disconnect":
                self.disconnect()
                return f"Disconnected from {{self.service_name}}"
            
            elif command == "status":
                return f"Connection status: {{'Connected' if self.is_connected() else 'Disconnected'}}"
            
            else:
                return f"Unknown command: {{command}}. Available: connect, disconnect, status"
                
        except Exception as e:
            return f"Error in {plugin_name} integration: {{e}}"
    
    def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to {plugin_name} service"""
        try:
            # Your connection logic here
            # Example: API key validation, service health check
            api_key = credentials.get("api_key")
            if not api_key:
                logger.error("No API key provided")
                return False
            
            # Simulate connection
            self._connected = True
            logger.info(f"Connected to {{self.service_name}}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {{self.service_name}}: {{e}}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from {plugin_name} service"""
        self._connected = False
        logger.info(f"Disconnected from {{self.service_name}}")
    
    def is_connected(self) -> bool:
        """Check if connected to service"""
        return self._connected
    
    def cleanup(self) -> None:
        """Clean up integration resources"""
        self.disconnect()
        self._context = None
        logger.info(f"Cleaned up {plugin_name} integration plugin")


# Plugin instance
{plugin_name}_integration = {plugin_name.title()}Integration()
'''
    
    def _get_base_plugin_template(self, plugin_name: str, plugin_type: str) -> str:
        """Get template for base plugin"""
        return f'''"""
{plugin_name.title()} Plugin for Monk CLI

A {plugin_type} plugin that extends Monk CLI functionality.
"""

from typing import List, Dict, Any
from src.core.plugins.base import PluginBase, PluginMetadata, PluginType, PluginContext


class {plugin_name.title()}Plugin(PluginBase):
    """{plugin_name.title()} {plugin_type} plugin for Monk CLI"""
    
    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="{plugin_name}",
            version="0.1.0",
            description="Adds {plugin_name} functionality to Monk CLI",
            author="Your Name",
            plugin_type=PluginType.{plugin_type.upper()},
            dependencies=[],
            tags=["{plugin_name}", "{plugin_type}", "monk"],
            memory_access=["*"]
        )
    
    def initialize(self, context: PluginContext) -> bool:
        """Initialize the plugin with execution context"""
        try:
            self._context = context
            logger.info(f"Initialized {plugin_name} plugin")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {plugin_name} plugin: {{e}}")
            return False
    
    def execute(self, command: str, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Execute the plugin functionality"""
        try:
            # Your plugin logic here
            result = f"Executed {plugin_name} {{plugin_type}} plugin with args: {{args}}"
            if kwargs:
                result += f" and kwargs: {{kwargs}}"
            
            return result
            
        except Exception as e:
            return f"Error executing {plugin_name}: {{e}}"
    
    def cleanup(self) -> None:
        """Clean up plugin resources"""
        self._context = None
        logger.info(f"Cleaned up {plugin_name} plugin")


# Plugin instance
{plugin_name}_plugin = {plugin_name.title()}Plugin()
'''
    
    def _generate_init_file(self, init_file: Path, plugin_name: str, plugin_type: str) -> None:
        """Generate the plugin __init__.py file"""
        
        if plugin_type == "command":
            content = f'''"""
{plugin_name.title()} Plugin Package

This package provides {plugin_name} functionality for Monk CLI.
"""

from .plugin import {plugin_name}_plugin

__all__ = ["{plugin_name}_plugin"]
'''
        elif plugin_type == "analyzer":
            content = f'''"""
{plugin_name.title()} Analyzer Plugin Package

This package provides {plugin_name} analysis capabilities for Monk CLI.
"""

from .plugin import {plugin_name}_analyzer

__all__ = ["{plugin_name}_analyzer"]
'''
        elif plugin_type == "integration":
            content = f'''"""
{plugin_name.title()} Integration Plugin Package

This package provides {plugin_name} integration for Monk CLI.
"""

from .plugin import {plugin_name}_integration

__all__ = ["{plugin_name}_integration"]
'''
        else:
            content = f'''"""
{plugin_name.title()} Plugin Package

This package provides {plugin_name} {plugin_type} functionality for Monk CLI.
"""

from .plugin import {plugin_name}_plugin

__all__ = ["{plugin_name}_plugin"]
'''
        
        init_file.write_text(content)
    
    def _generate_setup_py(self, plugin_dir: Path, plugin_name: str, plugin_type: str, 
                          author: str, description: str, dependencies: list) -> None:
        """Generate setup.py for the plugin"""
        
        setup_content = f'''"""
Setup script for {plugin_name} plugin
"""

from setuptools import setup, find_packages

setup(
    name="monk-{plugin_name}",
    version="0.1.0",
    description="{description or f'{plugin_name.title()} plugin for Monk CLI'}",
    author="{author}",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/monk-{plugin_name}",
    packages=find_packages(where="src"),
    package_dir={{"": "src"}},
    python_requires=">=3.8",
    install_requires={dependencies},
    extras_require={{
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800"
        ]
    }},
    entry_points={{
        "monk.plugins": [
            "{plugin_name} = {plugin_name}.plugin:{plugin_name}_plugin"
        ]
    }},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="monk, cli, plugin, {plugin_type}",
    project_urls={{
        "Bug Reports": "https://github.com/yourusername/monk-{plugin_name}/issues",
        "Source": "https://github.com/yourusername/monk-{plugin_name}",
        "Documentation": "https://github.com/yourusername/monk-{plugin_name}/blob/main/README.md",
    }}
)
'''
        
        setup_file = plugin_dir / "setup.py"
        setup_file.write_text(setup_content)
    
    def _generate_readme(self, plugin_dir: Path, plugin_name: str, plugin_type: str, description: str) -> None:
        """Generate README.md for the plugin"""
        
        readme_content = f'''# Monk {plugin_name.title()} Plugin

{description or f'A {plugin_type} plugin that extends Monk CLI with {plugin_name} functionality.'}

## Features

- **{plugin_type.title()} Integration**: Seamlessly integrates with Monk CLI
- **Memory Access**: Can access and modify Monk CLI memory context
- **TreeQuest Compatible**: Works with Monk's AI agent system
- **Easy Installation**: Simple pip install process

## Installation

```bash
# Install the plugin
pip install -e .

# Verify installation
monk plugin list
```

## Usage

```bash
# Use the plugin (specific commands depend on plugin type)
monk {plugin_name} [options]
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Plugin Type: {plugin_type.title()}

This is a **{plugin_type}** plugin that:

- Implements the required PluginBase interface
- Provides specific functionality for Monk CLI
- Integrates with Monk's memory and context systems
- Follows Monk CLI plugin development guidelines

## Configuration

The plugin can be configured through Monk CLI settings or environment variables.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For support and questions:
- Open an issue on GitHub
- Check Monk CLI documentation
- Join the Monk CLI community
'''
        
        readme_file = plugin_dir / "README.md"
        readme_file.write_text(readme_content)
    
    def _generate_requirements(self, plugin_dir: Path, dependencies: list) -> None:
        """Generate requirements.txt for the plugin"""
        
        requirements_content = "# Plugin dependencies\n"
        for dep in dependencies:
            requirements_content += f"{dep}\n"
        
        requirements_file = plugin_dir / "requirements.txt"
        requirements_file.write_text(requirements_content)
    
    def _generate_test_files(self, plugin_dir: Path, plugin_name: str) -> None:
        """Generate test files for the plugin"""
        
        # Basic test file
        test_content = f'''"""
Tests for {plugin_name} plugin
"""

import pytest
from unittest.mock import Mock, patch
from src.{plugin_name}.plugin import {plugin_name}_plugin


class Test{plugin_name.title()}Plugin:
    """Test cases for {plugin_name.title()}Plugin"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.plugin = {plugin_name}_plugin
        self.mock_context = Mock()
        self.mock_context.project_path = "/test/project"
        self.mock_context.project_context = {{"name": "test-project"}}
        self.mock_context.memory_context = {{"test": "value"}}
        self.mock_context.user_preferences = {{}}
        self.mock_context.conversation_history = []
        self.mock_context.plugin_registry = Mock()
    
    def test_plugin_initialization(self):
        """Test plugin initialization"""
        assert self.plugin.metadata is not None
        assert self.plugin.metadata.name == "{plugin_name}"
        assert self.plugin.metadata.plugin_type.value in ["command", "analyzer", "integration"]
    
    def test_plugin_initialize(self):
        """Test plugin initialization with context"""
        result = self.plugin.initialize(self.mock_context)
        assert result is True
        assert self.plugin._context == self.mock_context
    
    def test_plugin_execute(self):
        """Test plugin execution"""
        # Initialize first
        self.plugin.initialize(self.mock_context)
        
        # Test execution
        result = self.plugin.execute("test", ["arg1"], {{"key": "value"}})
        assert isinstance(result, str)
        assert "{plugin_name}" in result
    
    def test_plugin_cleanup(self):
        """Test plugin cleanup"""
        self.plugin.initialize(self.mock_context)
        self.plugin.cleanup()
        assert self.plugin._context is None
    
    def test_plugin_help(self):
        """Test plugin help text"""
        help_text = self.plugin.get_help()
        assert isinstance(help_text, str)
        assert "{plugin_name}" in help_text
    
    def test_plugin_commands(self):
        """Test plugin command list"""
        commands = self.plugin.get_commands()
        assert isinstance(commands, list)
    
    def test_plugin_performance_metrics(self):
        """Test plugin performance metrics"""
        metrics = self.plugin.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert "execution_count" in metrics
        assert "error_count" in metrics
        assert "status" in metrics


if __name__ == "__main__":
    pytest.main([__file__])
'''
        
        test_file = plugin_dir / "tests" / f"test_{plugin_name}.py"
        test_file.write_text(test_content)
    
    def create_sample_plugin(self, plugin_name: str = "example") -> bool:
        """Create a sample plugin for demonstration"""
        return self.create_plugin(
            plugin_name=plugin_name,
            plugin_type="command",
            author="Monk CLI Team",
            description="A sample plugin demonstrating Monk CLI plugin capabilities",
            dependencies=["requests>=2.25.0"]
        )
