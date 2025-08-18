"""
Plugin Validator for Monk CLI

Validates plugins for security, compatibility, and proper implementation.
"""

import ast
import inspect
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .base import PluginBase, PluginMetadata, PluginType

logger = logging.getLogger(__name__)


class PluginValidator:
    """
    Validates plugins for security and compatibility
    
    Performs:
    - Code security analysis
    - Interface compliance checking
    - Dependency validation
    - Performance impact assessment
    """
    
    def __init__(self):
        self.security_checks = [
            self._check_dangerous_imports,
            self._check_file_operations,
            self._check_network_access,
            self._check_system_commands,
            self._check_code_evaluation
        ]
        
        self.compatibility_checks = [
            self._check_interface_compliance,
            self._check_metadata_validity,
            self._check_dependency_safety
        ]
        
        self.performance_checks = [
            self._check_import_complexity,
            self._check_method_complexity
        ]
    
    def validate_plugin(self, plugin: PluginBase) -> bool:
        """
        Comprehensive plugin validation
        
        Args:
            plugin: Plugin instance to validate
            
        Returns:
            True if plugin passes all validations
        """
        try:
            # Security validation
            security_issues = self._validate_security(plugin)
            if security_issues:
                logger.error(f"Plugin {plugin.__class__.__name__} failed security validation: {security_issues}")
                return False
            
            # Compatibility validation
            compatibility_issues = self._validate_compatibility(plugin)
            if compatibility_issues:
                logger.error(f"Plugin {plugin.__class__.__name__} failed compatibility validation: {compatibility_issues}")
                return False
            
            # Performance validation
            performance_issues = self._validate_performance(plugin)
            if performance_issues:
                logger.warning(f"Plugin {plugin.__class__.__name__} has performance concerns: {performance_issues}")
                # Performance issues are warnings, not failures
            
            logger.info(f"Plugin {plugin.__class__.__name__} passed all validations")
            return True
            
        except Exception as e:
            logger.error(f"Error validating plugin {plugin.__class__.__name__}: {e}")
            return False
    
    def _validate_security(self, plugin: PluginBase) -> List[str]:
        """Validate plugin security"""
        issues = []
        
        try:
            # Get plugin source code
            source = inspect.getsource(plugin.__class__)
            tree = ast.parse(source)
            
            for check in self.security_checks:
                check_issues = check(tree, plugin)
                issues.extend(check_issues)
                
        except Exception as e:
            issues.append(f"Could not analyze source code: {e}")
        
        return issues
    
    def _validate_compatibility(self, plugin: PluginBase) -> List[str]:
        """Validate plugin compatibility"""
        issues = []
        
        for check in self.compatibility_checks:
            check_issues = check(plugin)
            issues.extend(check_issues)
        
        return issues
    
    def _validate_performance(self, plugin: PluginBase) -> List[str]:
        """Validate plugin performance characteristics"""
        issues = []
        
        try:
            source = inspect.getsource(plugin.__class__)
            tree = ast.parse(source)
            
            for check in self.performance_checks:
                check_issues = check(tree, plugin)
                issues.extend(check_issues)
                
        except Exception as e:
            issues.append(f"Could not analyze performance: {e}")
        
        return issues
    
    def _check_dangerous_imports(self, tree: ast.AST, plugin: PluginBase) -> List[str]:
        """Check for dangerous module imports"""
        issues = []
        dangerous_modules = {
            'os', 'subprocess', 'sys', 'builtins', 'ctypes',
            'pickle', 'marshal', 'shelve', 'dbm', 'sqlite3'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in dangerous_modules:
                        issues.append(f"Dangerous import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module in dangerous_modules:
                    issues.append(f"Dangerous import from: {node.module}")
        
        return issues
    
    def _check_file_operations(self, tree: ast.AST, plugin: PluginBase) -> List[str]:
        """Check for potentially dangerous file operations"""
        issues = []
        dangerous_functions = {
            'open', 'file', 'remove', 'unlink', 'rmdir', 'removedirs',
            'rename', 'renames', 'replace', 'copy', 'copy2', 'copytree',
            'move', 'shutil.rmtree', 'shutil.copy', 'shutil.move'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in dangerous_functions:
                        issues.append(f"Potentially dangerous file operation: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    if f"{node.func.value.id}.{node.func.attr}" in dangerous_functions:
                        issues.append(f"Potentially dangerous file operation: {node.func.value.id}.{node.func.attr}")
        
        return issues
    
    def _check_network_access(self, tree: ast.AST, plugin: PluginBase) -> List[str]:
        """Check for network access capabilities"""
        issues = []
        network_modules = {
            'urllib', 'requests', 'httpx', 'aiohttp', 'socket',
            'ftplib', 'smtplib', 'poplib', 'imaplib', 'telnetlib'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(network in alias.name for network in network_modules):
                        issues.append(f"Network access capability: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if any(network in node.module for network in network_modules):
                    issues.append(f"Network access capability: {node.module}")
        
        return issues
    
    def _check_system_commands(self, tree: ast.AST, plugin: PluginBase) -> List[str]:
        """Check for system command execution"""
        issues = []
        system_functions = {
            'os.system', 'os.popen', 'subprocess.call', 'subprocess.Popen',
            'subprocess.run', 'subprocess.check_call', 'subprocess.check_output'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    func_name = f"{node.func.value.id}.{node.func.attr}"
                    if func_name in system_functions:
                        issues.append(f"System command execution: {func_name}")
        
        return issues
    
    def _check_code_evaluation(self, tree: ast.AST, plugin: PluginBase) -> List[str]:
        """Check for code evaluation capabilities"""
        issues = []
        eval_functions = {'eval', 'exec', 'compile'}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in eval_functions:
                        issues.append(f"Code evaluation capability: {node.func.id}")
        
        return issues
    
    def _check_interface_compliance(self, plugin: PluginBase) -> List[str]:
        """Check if plugin implements required interface"""
        issues = []
        
        # Check required methods
        required_methods = ['initialize', 'execute', 'cleanup']
        for method in required_methods:
            if not hasattr(plugin, method):
                issues.append(f"Missing required method: {method}")
            elif not callable(getattr(plugin, method)):
                issues.append(f"Required method not callable: {method}")
        
        # Check if it's a proper subclass
        if not isinstance(plugin, PluginBase):
            issues.append("Plugin must inherit from PluginBase")
        
        return issues
    
    def _check_metadata_validity(self, plugin: PluginBase) -> List[str]:
        """Check plugin metadata validity"""
        issues = []
        
        if not hasattr(plugin, 'metadata') or plugin.metadata is None:
            issues.append("Plugin must have metadata")
            return issues
        
        metadata = plugin.metadata
        
        # Check required fields
        required_fields = ['name', 'version', 'description', 'author', 'plugin_type']
        for field in required_fields:
            if not hasattr(metadata, field) or getattr(metadata, field) is None:
                issues.append(f"Missing required metadata field: {field}")
        
        # Check field types
        if hasattr(metadata, 'plugin_type') and not isinstance(metadata.plugin_type, PluginType):
            issues.append("plugin_type must be a PluginType enum value")
        
        if hasattr(metadata, 'dependencies') and not isinstance(metadata.dependencies, list):
            issues.append("dependencies must be a list")
        
        if hasattr(metadata, 'tags') and not isinstance(metadata.tags, list):
            issues.append("tags must be a list")
        
        return issues
    
    def _check_dependency_safety(self, plugin: PluginBase) -> List[str]:
        """Check plugin dependencies for safety"""
        issues = []
        
        if not hasattr(plugin, 'metadata') or not plugin.metadata:
            return issues
        
        metadata = plugin.metadata
        if not hasattr(metadata, 'dependencies'):
            return issues
        
        dangerous_dependencies = {
            'pickle', 'marshal', 'shelve', 'dbm', 'sqlite3',
            'ctypes', 'mmap', 'signal', 'pwd', 'grp'
        }
        
        for dep in metadata.dependencies:
            if dep in dangerous_dependencies:
                issues.append(f"Potentially dangerous dependency: {dep}")
        
        return issues
    
    def _check_import_complexity(self, tree: ast.AST, plugin: PluginBase) -> List[str]:
        """Check import complexity"""
        issues = []
        import_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_count += 1
        
        if import_count > 20:
            issues.append(f"High import complexity: {import_count} imports")
        
        return issues
    
    def _check_method_complexity(self, tree: ast.AST, plugin: PluginBase) -> List[str]:
        """Check method complexity"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count lines in method
                line_count = len(node.body)
                if line_count > 100:
                    issues.append(f"Complex method {node.name}: {line_count} lines")
                
                # Count nested blocks
                nested_blocks = 0
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                        nested_blocks += 1
                
                if nested_blocks > 10:
                    issues.append(f"Deeply nested method {node.name}: {nested_blocks} blocks")
        
        return issues
    
    def get_validation_report(self, plugin: PluginBase) -> Dict[str, Any]:
        """Get detailed validation report for a plugin"""
        report = {
            'plugin_name': plugin.__class__.__name__,
            'passed': False,
            'security_issues': [],
            'compatibility_issues': [],
            'performance_issues': [],
            'recommendations': []
        }
        
        try:
            # Security validation
            security_issues = self._validate_security(plugin)
            report['security_issues'] = security_issues
            
            # Compatibility validation
            compatibility_issues = self._validate_compatibility(plugin)
            report['compatibility_issues'] = compatibility_issues
            
            # Performance validation
            performance_issues = self._validate_performance(plugin)
            report['performance_issues'] = performance_issues
            
            # Overall result
            report['passed'] = len(security_issues) == 0 and len(compatibility_issues) == 0
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(
                security_issues, compatibility_issues, performance_issues
            )
            
        except Exception as e:
            report['error'] = str(e)
        
        return report
    
    def _generate_recommendations(self, security_issues: List[str], 
                                compatibility_issues: List[str], 
                                performance_issues: List[str]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if security_issues:
            recommendations.append("Review and restrict dangerous operations")
            recommendations.append("Implement proper input validation")
            recommendations.append("Use sandboxed execution environment")
        
        if compatibility_issues:
            recommendations.append("Implement all required interface methods")
            recommendations.append("Provide complete and valid metadata")
            recommendations.append("Review dependency requirements")
        
        if performance_issues:
            recommendations.append("Optimize import statements")
            recommendations.append("Refactor complex methods")
            recommendations.append("Consider lazy loading for heavy operations")
        
        if not recommendations:
            recommendations.append("Plugin meets all quality standards")
        
        return recommendations
