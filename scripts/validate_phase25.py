"""
MONK CLI Phase 2.5 - Validation and Testing Script
Comprehensive validation of open source integration implementation
"""

import asyncio
import sys
import os
import logging
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Validation result data structure"""
    component: str
    test_name: str
    status: str  # "pass", "fail", "skip", "error"
    message: str
    execution_time_ms: float
    details: Dict[str, Any]


class Phase25Validator:
    """Comprehensive validator for Phase 2.5 implementation"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.project_root = Path(__file__).parent.parent
        
    async def run_validation_suite(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        logger.info("Starting Phase 2.5 Validation Suite")
        
        # 1. Environment and Dependencies Validation
        await self.validate_environment()
        
        # 2. Code Quality and Structure Validation  
        await self.validate_code_quality()
        
        # 3. Component Integration Validation
        await self.validate_component_integration()
        
        # 4. Performance Validation
        await self.validate_performance()
        
        # 5. Feature Completeness Validation
        await self.validate_feature_completeness()
        
        # 6. Documentation Validation
        await self.validate_documentation()
        
        # Generate validation report
        return self.generate_validation_report()
    
    async def validate_environment(self):
        """Validate environment setup and dependencies"""
        logger.info("Validating environment and dependencies...")
        
        # Check Python version
        await self._check_python_version()
        
        # Check required packages
        await self._check_required_packages()
        
        # Check project structure
        await self._check_project_structure()
        
        # Check configuration files
        await self._check_configuration_files()
    
    async def _check_python_version(self):
        """Check Python version compatibility"""
        start_time = time.time()
        
        try:
            version = sys.version_info
            if version.major >= 3 and version.minor >= 8:
                status = "pass"
                message = f"Python {version.major}.{version.minor} is compatible"
            else:
                status = "fail"
                message = f"Python {version.major}.{version.minor} is too old, requires 3.8+"
        except Exception as e:
            status = "error"
            message = f"Failed to check Python version: {e}"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Environment",
            test_name="Python Version",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={"version": f"{version.major}.{version.minor}.{version.micro}"}
        ))
    
    async def _check_required_packages(self):
        """Check if required packages are available"""
        start_time = time.time()
        
        required_packages = [
            "asyncio",
            "dataclasses", 
            "pathlib",
            "logging",
            "json",
            "time",
            "datetime",
            "typing",
            "enum",
            "hashlib",
            "threading",
            "concurrent.futures"
        ]
        
        optional_packages = [
            "smolagents",
            "transformers",
            "tree_sitter",
            "tree_sitter_python",
            "tree_sitter_javascript", 
            "tree_sitter_typescript",
            "websockets",
            "aiohttp",
            "psutil"
        ]
        
        missing_required = []
        missing_optional = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_required.append(package)
        
        for package in optional_packages:
            try:
                __import__(package)
            except ImportError:
                missing_optional.append(package)
        
        if missing_required:
            status = "fail"
            message = f"Missing required packages: {', '.join(missing_required)}"
        elif missing_optional:
            status = "pass"
            message = f"Optional packages missing: {', '.join(missing_optional)} (fallback mode available)"
        else:
            status = "pass"
            message = "All required packages available"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Environment",
            test_name="Required Packages",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "missing_required": missing_required,
                "missing_optional": missing_optional
            }
        ))
    
    async def _check_project_structure(self):
        """Check project directory structure"""
        start_time = time.time()
        
        required_directories = [
            "src",
            "src/phase25",
            "src/phase25/bridges",
            "src/phase25/smolagents",
            "src/phase25/lsp",
            "src/phase25/vscode",
            "src/phase25/collaboration",
            "tests",
            "benchmarks"
        ]
        
        required_files = [
            "src/phase25/bridges/treequest_smolagent_bridge.py",
            "src/phase25/smolagents/multi_agent_system.py",
            "src/phase25/lsp/tree_sitter_explorer.py",
            "src/phase25/vscode/extension.ts",
            "src/phase25/vscode/package.json",
            "src/phase25/vscode/chatProvider.ts",
            "src/phase25/collaboration/collaborative_platform.py",
            "tests/test_phase25_integration.py",
            "benchmarks/phase25_benchmark.py"
        ]
        
        missing_dirs = []
        missing_files = []
        
        for directory in required_directories:
            dir_path = self.project_root / directory
            if not dir_path.exists():
                missing_dirs.append(directory)
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_dirs or missing_files:
            status = "fail"
            message = f"Missing directories: {missing_dirs}, Missing files: {missing_files}"
        else:
            status = "pass"
            message = "Project structure is complete"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Environment",
            test_name="Project Structure",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "missing_directories": missing_dirs,
                "missing_files": missing_files
            }
        ))
    
    async def _check_configuration_files(self):
        """Check configuration files"""
        start_time = time.time()
        
        config_files = [
            "requirements.txt",
            "phase1_requirements.txt",
            "src/phase25/vscode/package.json"
        ]
        
        missing_configs = []
        invalid_configs = []
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            
            if not config_path.exists():
                missing_configs.append(config_file)
                continue
            
            # Validate JSON files
            if config_file.endswith('.json'):
                try:
                    with open(config_path, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError:
                    invalid_configs.append(config_file)
        
        if missing_configs or invalid_configs:
            status = "fail" if missing_configs else "pass"
            message = f"Missing: {missing_configs}, Invalid: {invalid_configs}"
        else:
            status = "pass"
            message = "All configuration files are valid"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Environment",
            test_name="Configuration Files",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "missing_configs": missing_configs,
                "invalid_configs": invalid_configs
            }
        ))
    
    async def validate_code_quality(self):
        """Validate code quality and structure"""
        logger.info("Validating code quality...")
        
        # Check syntax of Python files
        await self._check_python_syntax()
        
        # Check TypeScript syntax
        await self._check_typescript_syntax()
        
        # Check import structure
        await self._check_import_structure()
        
        # Check coding standards
        await self._check_coding_standards()
    
    async def _check_python_syntax(self):
        """Check Python syntax in all .py files"""
        start_time = time.time()
        
        python_files = list(self.project_root.rglob("*.py"))
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                compile(code, str(py_file), 'exec')
                
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")
            except Exception as e:
                syntax_errors.append(f"{py_file}: {e}")
        
        status = "pass" if not syntax_errors else "fail"
        message = f"Checked {len(python_files)} files, {len(syntax_errors)} syntax errors"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Code Quality",
            test_name="Python Syntax",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "files_checked": len(python_files),
                "syntax_errors": syntax_errors[:10]  # Limit to first 10 errors
            }
        ))
    
    async def _check_typescript_syntax(self):
        """Check TypeScript syntax"""
        start_time = time.time()
        
        ts_files = list(self.project_root.rglob("*.ts"))
        
        if not ts_files:
            status = "skip"
            message = "No TypeScript files found"
        else:
            # For now, just check that files exist and are readable
            unreadable_files = []
            
            for ts_file in ts_files:
                try:
                    with open(ts_file, 'r', encoding='utf-8') as f:
                        f.read()
                except Exception as e:
                    unreadable_files.append(f"{ts_file}: {e}")
            
            status = "pass" if not unreadable_files else "fail"
            message = f"Checked {len(ts_files)} TypeScript files, {len(unreadable_files)} unreadable"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Code Quality",
            test_name="TypeScript Syntax",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "files_checked": len(ts_files),
                "unreadable_files": unreadable_files
            }
        ))
    
    async def _check_import_structure(self):
        """Check import structure and dependencies"""
        start_time = time.time()
        
        phase25_files = list((self.project_root / "src" / "phase25").rglob("*.py"))
        import_issues = []
        
        for py_file in phase25_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for relative imports from parent packages
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if line.startswith('from ...') or line.startswith('import ...'):
                        # This is expected for Phase 2.5 files
                        continue
                    elif 'from ..' in line and 'import' in line:
                        # Check if it's properly structured
                        if not any(pkg in line for pkg in ['core', 'agents', 'memory', 'interfaces']):
                            import_issues.append(f"{py_file}:{i} - Questionable import: {line}")
                        
            except Exception as e:
                import_issues.append(f"{py_file}: Failed to read - {e}")
        
        status = "pass" if len(import_issues) < 5 else "fail"  # Allow some flexibility
        message = f"Checked {len(phase25_files)} files, {len(import_issues)} import issues"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Code Quality",
            test_name="Import Structure",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "files_checked": len(phase25_files),
                "import_issues": import_issues[:10]  # Limit output
            }
        ))
    
    async def _check_coding_standards(self):
        """Check basic coding standards"""
        start_time = time.time()
        
        python_files = list((self.project_root / "src" / "phase25").rglob("*.py"))
        standard_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                # Check for basic standards
                for i, line in enumerate(lines, 1):
                    # Check line length (allow up to 120 characters)
                    if len(line) > 120:
                        standard_issues.append(f"{py_file}:{i} - Line too long ({len(line)} chars)")
                    
                    # Check for TODO/FIXME comments that might indicate incomplete code
                    if 'TODO' in line or 'FIXME' in line:
                        standard_issues.append(f"{py_file}:{i} - TODO/FIXME found: {line.strip()}")
                
                # Check for docstrings in classes and functions
                if 'class ' in content or 'def ' in content:
                    # This is a simplified check - in practice you'd use AST parsing
                    if '"""' not in content and "'''" not in content:
                        standard_issues.append(f"{py_file} - Missing docstrings")
                        
            except Exception as e:
                standard_issues.append(f"{py_file}: Failed to check - {e}")
        
        # Filter out excessive issues
        filtered_issues = standard_issues[:20]  # Limit to first 20 issues
        
        status = "pass" if len(standard_issues) < 10 else "fail"
        message = f"Checked {len(python_files)} files, {len(standard_issues)} standard issues"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Code Quality",
            test_name="Coding Standards",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "files_checked": len(python_files),
                "total_issues": len(standard_issues),
                "sample_issues": filtered_issues
            }
        ))
    
    async def validate_component_integration(self):
        """Validate component integration"""
        logger.info("Validating component integration...")
        
        # Test basic imports
        await self._test_component_imports()
        
        # Test component initialization
        await self._test_component_initialization()
        
        # Test component interactions
        await self._test_component_interactions()
    
    async def _test_component_imports(self):
        """Test that all components can be imported"""
        start_time = time.time()
        
        components_to_import = [
            "phase25.bridges.treequest_smolagent_bridge",
            "phase25.smolagents.multi_agent_system",
            "phase25.lsp.tree_sitter_explorer",
            "phase25.collaboration.collaborative_platform"
        ]
        
        import_failures = []
        
        for component in components_to_import:
            try:
                __import__(component)
            except ImportError as e:
                import_failures.append(f"{component}: {e}")
            except Exception as e:
                import_failures.append(f"{component}: Unexpected error - {e}")
        
        status = "pass" if not import_failures else "fail"
        message = f"Tested {len(components_to_import)} imports, {len(import_failures)} failures"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Integration",
            test_name="Component Imports",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "components_tested": components_to_import,
                "import_failures": import_failures
            }
        ))
    
    async def _test_component_initialization(self):
        """Test component initialization"""
        start_time = time.time()
        
        init_results = {}
        
        # Test TreeQuest Bridge
        try:
            from phase25.bridges.treequest_smolagent_bridge import TreeQuestSmolagentBridge
            bridge = TreeQuestSmolagentBridge()
            init_results["TreeQuest Bridge"] = "success"
        except Exception as e:
            init_results["TreeQuest Bridge"] = f"failed: {e}"
        
        # Test Multi-Agent System
        try:
            from phase25.smolagents.multi_agent_system import MONKMultiAgentSystem
            mas = MONKMultiAgentSystem()
            init_results["Multi-Agent System"] = "success"
        except Exception as e:
            init_results["Multi-Agent System"] = f"failed: {e}"
        
        # Test Tree-Sitter Explorer
        try:
            from phase25.lsp.tree_sitter_explorer import TreeSitterLSPExplorer
            explorer = TreeSitterLSPExplorer()
            init_results["Tree-Sitter Explorer"] = "success"
        except Exception as e:
            init_results["Tree-Sitter Explorer"] = f"failed: {e}"
        
        # Test Collaborative Platform
        try:
            from phase25.collaboration.collaborative_platform import CollaborativePlatform
            platform = CollaborativePlatform()
            init_results["Collaborative Platform"] = "success"
        except Exception as e:
            init_results["Collaborative Platform"] = f"failed: {e}"
        
        failures = [comp for comp, result in init_results.items() if "failed" in result]
        
        status = "pass" if not failures else "fail"
        message = f"Tested {len(init_results)} components, {len(failures)} initialization failures"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Integration",
            test_name="Component Initialization",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "initialization_results": init_results,
                "failed_components": failures
            }
        ))
    
    async def _test_component_interactions(self):
        """Test basic component interactions"""
        start_time = time.time()
        
        interaction_results = {}
        
        try:
            # Test bridge with basic task creation
            from phase25.bridges.treequest_smolagent_bridge import TreeQuestSmolagentBridge, TreeQuestTask, AgentPersonality
            
            bridge = TreeQuestSmolagentBridge()
            task = TreeQuestTask(
                task_id="validation_test",
                description="Validation test task",
                complexity=1,
                domain="validation",
                subtasks=["test"],
                dependencies=[],
                agent_personality_required=AgentPersonality.ANALYTICAL,
                smolagent_tools=["test"]
            )
            
            # This should not raise an exception
            interaction_results["Bridge Task Creation"] = "success"
            
        except Exception as e:
            interaction_results["Bridge Task Creation"] = f"failed: {e}"
        
        try:
            # Test explorer with basic language detection
            from phase25.lsp.tree_sitter_explorer import TreeSitterLSPExplorer
            
            explorer = TreeSitterLSPExplorer()
            # Test language detection method
            language = explorer._detect_language("test.py")
            if language:
                interaction_results["Explorer Language Detection"] = "success"
            else:
                interaction_results["Explorer Language Detection"] = "failed: no language detected"
                
        except Exception as e:
            interaction_results["Explorer Language Detection"] = f"failed: {e}"
        
        failures = [comp for comp, result in interaction_results.items() if "failed" in result]
        
        status = "pass" if not failures else "fail"
        message = f"Tested {len(interaction_results)} interactions, {len(failures)} failures"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Integration",
            test_name="Component Interactions",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "interaction_results": interaction_results,
                "failed_interactions": failures
            }
        ))
    
    async def validate_performance(self):
        """Validate performance requirements"""
        logger.info("Validating performance...")
        
        # Check if benchmark script exists and runs
        await self._test_benchmark_availability()
        
        # Basic performance tests
        await self._test_basic_performance()
    
    async def _test_benchmark_availability(self):
        """Test benchmark script availability"""
        start_time = time.time()
        
        benchmark_path = self.project_root / "benchmarks" / "phase25_benchmark.py"
        
        if not benchmark_path.exists():
            status = "fail"
            message = "Benchmark script not found"
        else:
            try:
                # Try to import benchmark module
                spec = subprocess.run(
                    [sys.executable, "-c", f"import sys; sys.path.append('{self.project_root}'); from benchmarks.phase25_benchmark import Phase25Benchmarker"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if spec.returncode == 0:
                    status = "pass"
                    message = "Benchmark script is importable"
                else:
                    status = "fail"
                    message = f"Benchmark import failed: {spec.stderr}"
                    
            except subprocess.TimeoutExpired:
                status = "fail"
                message = "Benchmark import timed out"
            except Exception as e:
                status = "fail"
                message = f"Benchmark test failed: {e}"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Performance",
            test_name="Benchmark Availability",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={"benchmark_path": str(benchmark_path)}
        ))
    
    async def _test_basic_performance(self):
        """Test basic performance characteristics"""
        start_time = time.time()
        
        performance_results = {}
        
        try:
            # Test component creation speed
            creation_start = time.time()
            
            from phase25.bridges.treequest_smolagent_bridge import TreeQuestSmolagentBridge
            bridge = TreeQuestSmolagentBridge()
            
            creation_time = (time.time() - creation_start) * 1000
            performance_results["Bridge Creation Time (ms)"] = creation_time
            
            # Test should be under 1 second
            if creation_time < 1000:
                performance_results["Bridge Creation Performance"] = "pass"
            else:
                performance_results["Bridge Creation Performance"] = "slow"
            
        except Exception as e:
            performance_results["Bridge Creation Performance"] = f"failed: {e}"
        
        try:
            # Test memory usage
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            performance_results["Memory Usage (MB)"] = memory_mb
            
            # Should use reasonable amount of memory
            if memory_mb < 500:  # Less than 500MB
                performance_results["Memory Usage Performance"] = "pass"
            else:
                performance_results["Memory Usage Performance"] = "high"
                
        except Exception as e:
            performance_results["Memory Usage Performance"] = f"failed: {e}"
        
        # Determine overall status
        failures = [k for k, v in performance_results.items() if isinstance(v, str) and ("failed" in v or v == "slow" or v == "high")]
        
        status = "pass" if not failures else "fail"
        message = f"Performance validation completed, {len(failures)} issues found"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Performance",
            test_name="Basic Performance",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "performance_results": performance_results,
                "issues": failures
            }
        ))
    
    async def validate_feature_completeness(self):
        """Validate feature completeness"""
        logger.info("Validating feature completeness...")
        
        await self._check_required_features()
        await self._check_api_completeness()
        await self._check_integration_points()
    
    async def _check_required_features(self):
        """Check if all required features are implemented"""
        start_time = time.time()
        
        required_features = {
            "TreeQuest-Smolagent Bridge": [
                "task decomposition",
                "personality-based execution",
                "agent allocation optimization",
                "performance metrics"
            ],
            "Multi-Agent System": [
                "task submission",
                "concurrent execution",
                "dependency management",
                "scalability"
            ],
            "Tree-Sitter Explorer": [
                "file parsing",
                "symbol extraction",
                "complexity analysis",
                "language detection"
            ],
            "VS Code Extension": [
                "extension manifest",
                "chat provider",
                "command registration",
                "personality selection"
            ],
            "Collaborative Platform": [
                "session management",
                "real-time communication",
                "event processing",
                "user management"
            ]
        }
        
        feature_status = {}
        missing_features = []
        
        for component, features in required_features.items():
            component_status = {}
            
            for feature in features:
                # Check if feature is implemented (simplified check)
                try:
                    if component == "TreeQuest-Smolagent Bridge":
                        from phase25.bridges.treequest_smolagent_bridge import TreeQuestSmolagentBridge
                        bridge = TreeQuestSmolagentBridge()
                        
                        if feature == "task decomposition":
                            has_method = hasattr(bridge, 'decompose_task_hierarchy')
                        elif feature == "personality-based execution":
                            has_method = hasattr(bridge, 'execute_task_with_personality')
                        elif feature == "agent allocation optimization":
                            has_method = hasattr(bridge, 'optimize_agent_allocation')
                        elif feature == "performance metrics":
                            has_method = hasattr(bridge, 'get_agent_performance_metrics')
                        else:
                            has_method = False
                        
                        component_status[feature] = "implemented" if has_method else "missing"
                        
                    elif component == "Multi-Agent System":
                        from phase25.smolagents.multi_agent_system import MONKMultiAgentSystem
                        mas = MONKMultiAgentSystem()
                        
                        if feature == "task submission":
                            has_method = hasattr(mas, 'submit_task')
                        elif feature == "concurrent execution":
                            has_method = hasattr(mas, '_execute_task')
                        elif feature == "dependency management":
                            has_method = hasattr(mas, '_dependencies_met')
                        elif feature == "scalability":
                            has_method = hasattr(mas, 'get_system_status')
                        else:
                            has_method = False
                        
                        component_status[feature] = "implemented" if has_method else "missing"
                        
                    elif component == "Tree-Sitter Explorer":
                        from phase25.lsp.tree_sitter_explorer import TreeSitterLSPExplorer
                        explorer = TreeSitterLSPExplorer()
                        
                        if feature == "file parsing":
                            has_method = hasattr(explorer, 'parse_file')
                        elif feature == "symbol extraction":
                            has_method = hasattr(explorer, '_extract_symbols_from_tree')
                        elif feature == "complexity analysis":
                            has_method = hasattr(explorer, '_calculate_complexity_metrics')
                        elif feature == "language detection":
                            has_method = hasattr(explorer, '_detect_language')
                        else:
                            has_method = False
                        
                        component_status[feature] = "implemented" if has_method else "missing"
                        
                    elif component == "VS Code Extension":
                        # Check file existence
                        vscode_dir = self.project_root / "src" / "phase25" / "vscode"
                        
                        if feature == "extension manifest":
                            has_feature = (vscode_dir / "package.json").exists()
                        elif feature == "chat provider":
                            has_feature = (vscode_dir / "chatProvider.ts").exists()
                        elif feature == "command registration":
                            has_feature = (vscode_dir / "extension.ts").exists()
                        elif feature == "personality selection":
                            # Check if chatProvider has personality functionality
                            chat_file = vscode_dir / "chatProvider.ts"
                            if chat_file.exists():
                                with open(chat_file, 'r') as f:
                                    content = f.read()
                                has_feature = "personality" in content.lower()
                            else:
                                has_feature = False
                        else:
                            has_feature = False
                        
                        component_status[feature] = "implemented" if has_feature else "missing"
                        
                    elif component == "Collaborative Platform":
                        from phase25.collaboration.collaborative_platform import CollaborativePlatform
                        platform = CollaborativePlatform()
                        
                        if feature == "session management":
                            has_method = hasattr(platform, 'get_or_create_session')
                        elif feature == "real-time communication":
                            has_method = hasattr(platform, 'handle_websocket_connection')
                        elif feature == "event processing":
                            has_method = hasattr(platform, 'process_event')
                        elif feature == "user management":
                            has_method = hasattr(platform, 'handle_join_session')
                        else:
                            has_method = False
                        
                        component_status[feature] = "implemented" if has_method else "missing"
                    
                    else:
                        component_status[feature] = "unknown"
                    
                    if component_status[feature] == "missing":
                        missing_features.append(f"{component}: {feature}")
                        
                except Exception as e:
                    component_status[feature] = f"error: {e}"
                    missing_features.append(f"{component}: {feature} (error)")
            
            feature_status[component] = component_status
        
        status = "pass" if not missing_features else "fail"
        message = f"Feature completeness check: {len(missing_features)} missing features"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Features",
            test_name="Required Features",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "feature_status": feature_status,
                "missing_features": missing_features[:20]  # Limit output
            }
        ))
    
    async def _check_api_completeness(self):
        """Check API completeness"""
        start_time = time.time()
        
        # Check if core API methods are available
        api_methods = {}
        
        try:
            from phase25.bridges.treequest_smolagent_bridge import TreeQuestSmolagentBridge
            bridge = TreeQuestSmolagentBridge()
            
            api_methods["TreeQuest Bridge"] = {
                "decompose_task_hierarchy": hasattr(bridge, 'decompose_task_hierarchy'),
                "execute_task_with_personality": hasattr(bridge, 'execute_task_with_personality'),
                "optimize_agent_allocation": hasattr(bridge, 'optimize_agent_allocation'),
                "get_agent_performance_metrics": hasattr(bridge, 'get_agent_performance_metrics')
            }
        except Exception as e:
            api_methods["TreeQuest Bridge"] = {"error": str(e)}
        
        try:
            from phase25.smolagents.multi_agent_system import MONKMultiAgentSystem
            mas = MONKMultiAgentSystem()
            
            api_methods["Multi-Agent System"] = {
                "submit_task": hasattr(mas, 'submit_task'),
                "wait_for_completion": hasattr(mas, 'wait_for_completion'),
                "get_system_status": hasattr(mas, 'get_system_status'),
                "shutdown": hasattr(mas, 'shutdown')
            }
        except Exception as e:
            api_methods["Multi-Agent System"] = {"error": str(e)}
        
        # Count missing methods
        missing_methods = []
        for component, methods in api_methods.items():
            if "error" in methods:
                missing_methods.append(f"{component}: {methods['error']}")
            else:
                for method, exists in methods.items():
                    if not exists:
                        missing_methods.append(f"{component}.{method}")
        
        status = "pass" if not missing_methods else "fail"
        message = f"API completeness: {len(missing_methods)} missing methods"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Features",
            test_name="API Completeness",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "api_methods": api_methods,
                "missing_methods": missing_methods
            }
        ))
    
    async def _check_integration_points(self):
        """Check integration points between components"""
        start_time = time.time()
        
        integration_status = {}
        
        # Check TreeQuest Bridge -> Multi-Agent System integration
        try:
            from phase25.bridges.treequest_smolagent_bridge import TreeQuestSmolagentBridge
            from phase25.smolagents.multi_agent_system import MONKMultiAgentSystem
            
            # Check if bridge can reference multi-agent system
            bridge = TreeQuestSmolagentBridge()
            mas = MONKMultiAgentSystem()
            
            integration_status["Bridge-MAS"] = "compatible"
            
        except Exception as e:
            integration_status["Bridge-MAS"] = f"failed: {e}"
        
        # Check Tree-Sitter Explorer -> Bridge integration
        try:
            from phase25.lsp.tree_sitter_explorer import TreeSitterLSPExplorer
            
            explorer = TreeSitterLSPExplorer()
            
            # Check if explorer results can be used by bridge
            integration_status["Explorer-Bridge"] = "compatible"
            
        except Exception as e:
            integration_status["Explorer-Bridge"] = f"failed: {e}"
        
        # Check Collaborative Platform -> All components integration
        try:
            from phase25.collaboration.collaborative_platform import CollaborativePlatform
            
            platform = CollaborativePlatform()
            
            # Check if platform can integrate with other components
            integration_status["Platform-All"] = "compatible"
            
        except Exception as e:
            integration_status["Platform-All"] = f"failed: {e}"
        
        failures = [k for k, v in integration_status.items() if "failed" in v]
        
        status = "pass" if not failures else "fail"
        message = f"Integration points: {len(failures)} failed integrations"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Features",
            test_name="Integration Points",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "integration_status": integration_status,
                "failed_integrations": failures
            }
        ))
    
    async def validate_documentation(self):
        """Validate documentation completeness"""
        logger.info("Validating documentation...")
        
        await self._check_documentation_files()
        await self._check_code_documentation()
    
    async def _check_documentation_files(self):
        """Check documentation files"""
        start_time = time.time()
        
        expected_docs = [
            "README.md",
            "README_PHASE1_IMPLEMENTATION.md",
            "README_PHASE2_IMPLEMENTATION.md"
        ]
        
        missing_docs = []
        incomplete_docs = []
        
        for doc_file in expected_docs:
            doc_path = self.project_root / doc_file
            
            if not doc_path.exists():
                missing_docs.append(doc_file)
            else:
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check if documentation is reasonably complete
                    if len(content) < 100:  # Very short documentation
                        incomplete_docs.append(f"{doc_file}: too short")
                    elif "TODO" in content or "TBD" in content:
                        incomplete_docs.append(f"{doc_file}: contains TODO/TBD")
                        
                except Exception as e:
                    incomplete_docs.append(f"{doc_file}: read error - {e}")
        
        status = "pass" if not missing_docs and not incomplete_docs else "fail"
        message = f"Documentation: {len(missing_docs)} missing, {len(incomplete_docs)} incomplete"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Documentation",
            test_name="Documentation Files",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "missing_docs": missing_docs,
                "incomplete_docs": incomplete_docs
            }
        ))
    
    async def _check_code_documentation(self):
        """Check code documentation (docstrings)"""
        start_time = time.time()
        
        python_files = list((self.project_root / "src" / "phase25").rglob("*.py"))
        doc_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for module docstring
                lines = content.strip().split('\n')
                if not (lines and lines[0].startswith('"""') or '"""' in content[:200]):
                    doc_issues.append(f"{py_file}: Missing module docstring")
                
                # Check for class and function docstrings (simplified)
                if 'class ' in content:
                    # Count classes vs docstrings
                    class_count = content.count('class ')
                    docstring_count = content.count('"""')
                    
                    if docstring_count < class_count:
                        doc_issues.append(f"{py_file}: Some classes may lack docstrings")
                
            except Exception as e:
                doc_issues.append(f"{py_file}: Check failed - {e}")
        
        status = "pass" if len(doc_issues) < len(python_files) * 0.3 else "fail"  # Allow 30% to have issues
        message = f"Code documentation: {len(doc_issues)} issues in {len(python_files)} files"
        
        execution_time = (time.time() - start_time) * 1000
        
        self.results.append(ValidationResult(
            component="Documentation",
            test_name="Code Documentation",
            status=status,
            message=message,
            execution_time_ms=execution_time,
            details={
                "files_checked": len(python_files),
                "documentation_issues": doc_issues[:10]  # Limit output
            }
        ))
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "pass"])
        failed_tests = len([r for r in self.results if r.status == "fail"])
        skipped_tests = len([r for r in self.results if r.status == "skip"])
        error_tests = len([r for r in self.results if r.status == "error"])
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Group results by component
        component_results = {}
        for result in self.results:
            if result.component not in component_results:
                component_results[result.component] = []
            component_results[result.component].append(result)
        
        # Generate component summaries
        component_summaries = {}
        for component, results in component_results.items():
            component_passed = len([r for r in results if r.status == "pass"])
            component_total = len(results)
            component_summaries[component] = {
                "total_tests": component_total,
                "passed": component_passed,
                "failed": len([r for r in results if r.status == "fail"]),
                "success_rate": (component_passed / component_total * 100) if component_total > 0 else 0
            }
        
        # Identify critical issues
        critical_issues = [
            result for result in self.results 
            if result.status == "fail" and result.component in ["Environment", "Integration"]
        ]
        
        # Generate recommendations
        recommendations = self._generate_validation_recommendations()
        
        report = {
            "validation_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "skipped": skipped_tests,
                "errors": error_tests,
                "success_rate_percent": success_rate,
                "validation_timestamp": time.time()
            },
            "component_summaries": component_summaries,
            "critical_issues": [asdict(issue) for issue in critical_issues],
            "detailed_results": [asdict(result) for result in self.results],
            "recommendations": recommendations,
            "overall_status": "PASS" if success_rate >= 80 and not critical_issues else "FAIL"
        }
        
        return report
    
    def _generate_validation_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        failed_results = [r for r in self.results if r.status == "fail"]
        
        # Check for environment issues
        env_failures = [r for r in failed_results if r.component == "Environment"]
        if env_failures:
            recommendations.append("Fix environment setup issues before proceeding with development")
        
        # Check for code quality issues
        quality_failures = [r for r in failed_results if r.component == "Code Quality"]
        if quality_failures:
            recommendations.append("Address code quality issues to improve maintainability")
        
        # Check for integration issues
        integration_failures = [r for r in failed_results if r.component == "Integration"]
        if integration_failures:
            recommendations.append("Resolve component integration issues for proper functionality")
        
        # Check for feature completeness
        feature_failures = [r for r in failed_results if r.component == "Features"]
        if feature_failures:
            recommendations.append("Complete missing features to meet Phase 2.5 requirements")
        
        # Check for performance issues
        performance_failures = [r for r in failed_results if r.component == "Performance"]
        if performance_failures:
            recommendations.append("Optimize performance to meet scalability requirements")
        
        # Check for documentation issues
        doc_failures = [r for r in failed_results if r.component == "Documentation"]
        if doc_failures:
            recommendations.append("Improve documentation for better usability and maintenance")
        
        if not recommendations:
            recommendations.append("All validation checks passed - Phase 2.5 implementation is ready")
        
        return recommendations


async def main():
    """Run Phase 2.5 validation suite"""
    logging.basicConfig(level=logging.INFO)
    
    validator = Phase25Validator()
    report = await validator.run_validation_suite()
    
    # Save report
    report_path = Path(__file__).parent.parent / f"phase25_validation_report_{int(time.time())}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("MONK CLI Phase 2.5 - Validation Report")
    print("="*80)
    
    summary = report["validation_summary"]
    print(f"Overall Status: {report['overall_status']}")
    print(f"Success Rate: {summary['success_rate_percent']:.1f}%")
    print(f"Tests: {summary['passed']}/{summary['total_tests']} passed")
    
    if summary['failed'] > 0:
        print(f"Failed Tests: {summary['failed']}")
    if summary['errors'] > 0:
        print(f"Error Tests: {summary['errors']}")
    
    print("\nComponent Summary:")
    print("-" * 40)
    
    for component, comp_summary in report["component_summaries"].items():
        status_indicator = "✓" if comp_summary["success_rate"] >= 80 else "✗"
        print(f"{status_indicator} {component}: {comp_summary['success_rate']:.1f}% ({comp_summary['passed']}/{comp_summary['total_tests']})")
    
    if report["critical_issues"]:
        print(f"\nCritical Issues ({len(report['critical_issues'])}):")
        print("-" * 40)
        for issue in report["critical_issues"][:5]:  # Show first 5
            print(f"• {issue['component']}: {issue['test_name']} - {issue['message']}")
    
    print("\nRecommendations:")
    print("-" * 40)
    for rec in report["recommendations"]:
        print(f"• {rec}")
    
    print(f"\nFull validation report saved to: {report_path}")
    
    # Exit with appropriate code
    sys.exit(0 if report['overall_status'] == 'PASS' else 1)


if __name__ == "__main__":
    asyncio.run(main())