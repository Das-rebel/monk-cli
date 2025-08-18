"""
Tests for example plugin
"""

import pytest
from unittest.mock import Mock, patch
from src.example.plugin import example_plugin


class TestExamplePlugin:
    """Test cases for ExamplePlugin"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.plugin = example_plugin
        self.mock_context = Mock()
        self.mock_context.project_path = "/test/project"
        self.mock_context.project_context = {"name": "test-project"}
        self.mock_context.memory_context = {"test": "value"}
        self.mock_context.user_preferences = {}
        self.mock_context.conversation_history = []
        self.mock_context.plugin_registry = Mock()
    
    def test_plugin_initialization(self):
        """Test plugin initialization"""
        assert self.plugin.metadata is not None
        assert self.plugin.metadata.name == "example"
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
        result = self.plugin.execute("test", ["arg1"], {"key": "value"})
        assert isinstance(result, str)
        assert "example" in result
    
    def test_plugin_cleanup(self):
        """Test plugin cleanup"""
        self.plugin.initialize(self.mock_context)
        self.plugin.cleanup()
        assert self.plugin._context is None
    
    def test_plugin_help(self):
        """Test plugin help text"""
        help_text = self.plugin.get_help()
        assert isinstance(help_text, str)
        assert "example" in help_text
    
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
