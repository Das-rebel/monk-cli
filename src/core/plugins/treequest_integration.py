"""
TreeQuest Integration for Monk CLI Plugins

Allows plugins to participate in TreeQuest AI agent optimization and decision-making.
"""

from typing import Dict, List, Any, Optional, Callable
import logging
import json

from .base import PluginBase, PluginType, PluginContext
from .registry import PluginRegistry

logger = logging.getLogger(__name__)


class TreeQuestPluginIntegration:
    """
    Integrates plugins with TreeQuest AI agent system
    
    Provides:
    - Plugin function registration for TreeQuest
    - Plugin performance tracking
    - Plugin optimization within AB-MCTS
    - Plugin function composition
    """
    
    def __init__(self, plugin_registry: PluginRegistry):
        self.plugin_registry = plugin_registry
        self.registered_functions: Dict[str, Dict[str, Any]] = {}
        self.plugin_performance: Dict[str, Dict[str, Any]] = {}
        self.optimization_weights: Dict[str, float] = {}
    
    def register_plugin_function(self, plugin_name: str, function_name: str, 
                               function: Callable, description: str = "",
                               input_schema: Optional[Dict] = None,
                               output_schema: Optional[Dict] = None,
                               complexity: float = 1.0) -> bool:
        """
        Register a plugin function for TreeQuest optimization
        
        Args:
            plugin_name: Name of the plugin
            function_name: Name of the function
            function: Callable function
            description: Function description
            input_schema: Expected input schema
            output_schema: Expected output schema
            complexity: Function complexity (1.0 = simple, 5.0 = complex)
            
        Returns:
            True if registration successful
        """
        try:
            if not callable(function):
                logger.error(f"Function {function_name} from plugin {plugin_name} is not callable")
                return False
            
            function_key = f"{plugin_name}.{function_name}"
            
            self.registered_functions[function_key] = {
                'plugin_name': plugin_name,
                'function_name': function_name,
                'function': function,
                'description': description,
                'input_schema': input_schema or {},
                'output_schema': output_schema or {},
                'complexity': max(0.1, min(10.0, complexity)),  # Clamp between 0.1 and 10.0
                'call_count': 0,
                'total_time': 0.0,
                'success_count': 0,
                'error_count': 0
            }
            
            # Initialize performance tracking
            if plugin_name not in self.plugin_performance:
                self.plugin_performance[plugin_name] = {
                    'total_functions': 0,
                    'total_calls': 0,
                    'total_time': 0.0,
                    'success_rate': 1.0,
                    'avg_complexity': 0.0
                }
            
            self.plugin_performance[plugin_name]['total_functions'] += 1
            
            logger.info(f"Registered plugin function: {function_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering plugin function {function_name} from {plugin_name}: {e}")
            return False
    
    def unregister_plugin_function(self, plugin_name: str, function_name: str) -> bool:
        """
        Unregister a plugin function
        
        Args:
            plugin_name: Name of the plugin
            function_name: Name of the function
            
        Returns:
            True if unregistration successful
        """
        function_key = f"{plugin_name}.{function_name}"
        
        if function_key in self.registered_functions:
            del self.registered_functions[function_key]
            logger.info(f"Unregistered plugin function: {function_key}")
            return True
        
        return False
    
    def execute_plugin_function(self, function_key: str, *args, **kwargs) -> Any:
        """
        Execute a registered plugin function with performance tracking
        
        Args:
            function_key: Key of the function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function execution result
        """
        if function_key not in self.registered_functions:
            raise ValueError(f"Unknown plugin function: {function_key}")
        
        function_info = self.registered_functions[function_key]
        plugin_name = function_info['plugin_name']
        function = function_info['function']
        
        import time
        start_time = time.time()
        
        try:
            # Execute function
            result = function(*args, **kwargs)
            
            # Update performance metrics
            execution_time = time.time() - start_time
            function_info['call_count'] += 1
            function_info['total_time'] += execution_time
            function_info['success_count'] += 1
            
            # Update plugin performance
            if plugin_name in self.plugin_performance:
                self.plugin_performance[plugin_name]['total_calls'] += 1
                self.plugin_performance[plugin_name]['total_time'] += execution_time
                self.plugin_performance[plugin_name]['success_rate'] = (
                    self.plugin_performance[plugin_name]['success_count'] / 
                    self.plugin_performance[plugin_name]['total_calls']
                )
            
            logger.debug(f"Executed plugin function {function_key} in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            # Update error metrics
            execution_time = time.time() - start_time
            function_info['call_count'] += 1
            function_info['total_time'] += execution_time
            function_info['error_count'] += 1
            
            # Update plugin performance
            if plugin_name in self.plugin_performance:
                self.plugin_performance[plugin_name]['total_calls'] += 1
                self.plugin_performance[plugin_name]['total_time'] += execution_time
                self.plugin_performance[plugin_name]['success_rate'] = (
                    self.plugin_performance[plugin_name]['success_count'] / 
                    self.plugin_performance[plugin_name]['total_calls']
                )
            
            logger.error(f"Error executing plugin function {function_key}: {e}")
            raise
    
    def get_available_functions(self) -> List[Dict[str, Any]]:
        """
        Get list of all available plugin functions
        
        Returns:
            List of function information dictionaries
        """
        functions = []
        
        for function_key, function_info in self.registered_functions.items():
            functions.append({
                'key': function_key,
                'plugin_name': function_info['plugin_name'],
                'function_name': function_info['function_name'],
                'description': function_info['description'],
                'complexity': function_info['complexity'],
                'call_count': function_info['call_count'],
                'avg_time': (
                    function_info['total_time'] / function_info['call_count']
                    if function_info['call_count'] > 0 else 0.0
                ),
                'success_rate': (
                    function_info['success_count'] / function_info['call_count']
                    if function_info['call_count'] > 0 else 1.0
                )
            })
        
        return functions
    
    def get_plugin_performance(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for a specific plugin
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Performance metrics or None if not found
        """
        return self.plugin_performance.get(plugin_name)
    
    def get_optimization_recommendations(self, task_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get optimization recommendations for TreeQuest based on plugin capabilities
        
        Args:
            task_context: Context of the task to optimize
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Analyze task context
        task_complexity = task_context.get('complexity', 1.0)
        task_type = task_context.get('type', 'general')
        required_capabilities = task_context.get('capabilities', [])
        
        # Find suitable plugin functions
        for function_key, function_info in self.registered_functions.items():
            score = 0.0
            reasons = []
            
            # Complexity matching
            complexity_diff = abs(function_info['complexity'] - task_complexity)
            if complexity_diff <= 1.0:
                score += 2.0
                reasons.append("Complexity well-matched")
            elif complexity_diff <= 2.0:
                score += 1.0
                reasons.append("Complexity acceptable")
            
            # Performance consideration
            if function_info['call_count'] > 0:
                success_rate = function_info['success_count'] / function_info['call_count']
                if success_rate >= 0.9:
                    score += 1.5
                    reasons.append("High success rate")
                elif success_rate >= 0.7:
                    score += 0.5
                    reasons.append("Acceptable success rate")
            
            # Capability matching
            if required_capabilities:
                # Simple keyword matching - could be enhanced with semantic analysis
                function_desc = function_info['description'].lower()
                for capability in required_capabilities:
                    if capability.lower() in function_desc:
                        score += 1.0
                        reasons.append(f"Provides {capability} capability")
            
            if score > 0:
                recommendations.append({
                    'function_key': function_key,
                    'plugin_name': function_info['plugin_name'],
                    'score': score,
                    'reasons': reasons,
                    'complexity': function_info['complexity'],
                    'avg_time': (
                        function_info['total_time'] / function_info['call_count']
                        if function_info['call_count'] > 0 else 0.0
                    )
                })
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations
    
    def compose_plugin_functions(self, function_keys: List[str], 
                               composition_strategy: str = "sequential") -> Callable:
        """
        Compose multiple plugin functions into a single callable
        
        Args:
            function_keys: List of function keys to compose
            composition_strategy: How to compose functions ('sequential', 'parallel', 'conditional')
            
        Returns:
            Composed function
        """
        if not function_keys:
            raise ValueError("No functions to compose")
        
        if composition_strategy == "sequential":
            return self._compose_sequential(function_keys)
        elif composition_strategy == "parallel":
            return self._compose_parallel(function_keys)
        elif composition_strategy == "conditional":
            return self._compose_conditional(function_keys)
        else:
            raise ValueError(f"Unknown composition strategy: {composition_strategy}")
    
    def _compose_sequential(self, function_keys: List[str]) -> Callable:
        """Compose functions to execute sequentially"""
        def composed_function(*args, **kwargs):
            result = None
            for function_key in function_keys:
                result = self.execute_plugin_function(function_key, *args, **kwargs)
            return result
        return composed_function
    
    def _compose_parallel(self, function_keys: List[str]) -> Callable:
        """Compose functions to execute in parallel"""
        import asyncio
        import concurrent.futures
        
        def composed_function(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.execute_plugin_function, function_key, *args, **kwargs)
                    for function_key in function_keys
                ]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            return results
        
        return composed_function
    
    def _compose_conditional(self, function_keys: List[str]) -> Callable:
        """Compose functions with conditional execution"""
        def composed_function(*args, **kwargs):
            for function_key in function_keys:
                try:
                    result = self.execute_plugin_function(function_key, *args, **kwargs)
                    if result:  # If function returns truthy value, consider it successful
                        return result
                except Exception as e:
                    logger.warning(f"Function {function_key} failed, trying next: {e}")
                    continue
            return None
        
        return composed_function
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get TreeQuest integration statistics"""
        total_functions = len(self.registered_functions)
        total_calls = sum(f['call_count'] for f in self.registered_functions.values())
        total_time = sum(f['total_time'] for f in self.registered_functions.values())
        
        return {
            'total_registered_functions': total_functions,
            'total_function_calls': total_calls,
            'total_execution_time': total_time,
            'avg_call_time': total_time / total_calls if total_calls > 0 else 0.0,
            'plugins_with_functions': len(set(f['plugin_name'] for f in self.registered_functions.values())),
            'performance_metrics': self.plugin_performance
        }
    
    def reset_performance_metrics(self) -> None:
        """Reset all performance metrics"""
        for function_info in self.registered_functions.values():
            function_info['call_count'] = 0
            function_info['total_time'] = 0.0
            function_info['success_count'] = 0
            function_info['error_count'] = 0
        
        self.plugin_performance.clear()
        logger.info("Reset all plugin performance metrics")
    
    def export_function_registry(self) -> Dict[str, Any]:
        """Export function registry for external use"""
        return {
            'functions': self.registered_functions,
            'performance': self.plugin_performance,
            'stats': self.get_integration_stats()
        }
    
    def import_function_registry(self, registry_data: Dict[str, Any]) -> bool:
        """Import function registry from external source"""
        try:
            if 'functions' in registry_data:
                self.registered_functions.update(registry_data['functions'])
            if 'performance' in registry_data:
                self.plugin_performance.update(registry_data['performance'])
            
            logger.info("Imported function registry data")
            return True
            
        except Exception as e:
            logger.error(f"Error importing function registry: {e}")
            return False
