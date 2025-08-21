"""
MONK CLI Phase 2.5 - Tree-sitter + LSP Code Exploration
Incremental code parsing and AST analysis with LSP integration
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import hashlib
import os
import subprocess

try:
    import tree_sitter
    from tree_sitter import Language, Parser, Node, Tree
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_typescript
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("tree-sitter modules not yet available - will be enabled after installation")

from ...core.config import config
from ...memory.memory_system import MemorySystem, MemoryQuery
from ..bridges.treequest_smolagent_bridge import TreeQuestTask
from ...core.database import get_db_session

logger = logging.getLogger(__name__)


class CodeLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript" 
    TYPESCRIPT = "typescript"
    JSON = "json"
    MARKDOWN = "markdown"
    YAML = "yaml"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CPP = "cpp"


class NodeType(Enum):
    """AST node types of interest"""
    FUNCTION_DEF = "function_definition"
    CLASS_DEF = "class_definition"
    VARIABLE = "variable"
    IMPORT = "import_statement"
    COMMENT = "comment"
    STRING = "string"
    CALL = "function_call"
    CONDITIONAL = "if_statement"
    LOOP = "for_statement"
    ERROR = "error"


@dataclass
class CodeLocation:
    """Code location information"""
    file_path: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    start_byte: int
    end_byte: int


@dataclass
class CodeSymbol:
    """Code symbol representation"""
    name: str
    symbol_type: NodeType
    language: CodeLanguage
    location: CodeLocation
    parent_name: Optional[str] = None
    docstring: Optional[str] = None
    parameters: List[str] = None
    return_type: Optional[str] = None
    complexity_score: int = 1
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.dependencies is None:
            self.dependencies = []


@dataclass  
class CodeTree:
    """Parsed code tree representation"""
    file_path: str
    language: CodeLanguage
    last_modified: datetime
    symbols: List[CodeSymbol]
    imports: List[str]
    errors: List[Dict[str, Any]]
    complexity_metrics: Dict[str, float]
    tree_hash: str


@dataclass
class LSPCapability:
    """LSP server capability definition"""
    language: CodeLanguage
    server_command: List[str]
    initialization_options: Dict[str, Any]
    supported_features: Set[str]  # hover, completion, definition, references, etc.
    workspace_folders: List[str]


class TreeSitterLSPExplorer:
    """Tree-sitter + LSP based code exploration system"""
    
    def __init__(self):
        self.tree_sitter_available = TREE_SITTER_AVAILABLE
        self.parsers: Dict[CodeLanguage, Parser] = {}
        self.languages: Dict[CodeLanguage, Language] = {}
        self.parsed_trees: Dict[str, CodeTree] = {}
        self.file_watchers: Dict[str, Any] = {}
        
        # LSP client management
        self.lsp_servers: Dict[CodeLanguage, Any] = {}
        self.lsp_capabilities: Dict[CodeLanguage, LSPCapability] = {}
        
        # Code analysis cache
        self.symbol_cache: Dict[str, List[CodeSymbol]] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.complexity_cache: Dict[str, Dict[str, float]] = {}
        
        # Memory system integration
        self.memory_system = MemorySystem()
        
        # Performance metrics
        self.parse_metrics = {
            "files_parsed": 0,
            "total_parse_time_ms": 0,
            "average_parse_time_ms": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
    async def initialize(self):
        """Initialize tree-sitter parsers and LSP servers"""
        try:
            logger.info("Initializing Tree-sitter + LSP code exploration system")
            
            # Initialize memory system
            await self.memory_system.initialize()
            
            # Setup tree-sitter parsers
            await self._setup_tree_sitter_parsers()
            
            # Setup LSP capabilities (without starting servers yet)
            self._setup_lsp_capabilities()
            
            logger.info("Tree-sitter + LSP explorer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Tree-sitter + LSP explorer: {e}")
            return False
    
    async def _setup_tree_sitter_parsers(self):
        """Setup tree-sitter parsers for supported languages"""
        if not self.tree_sitter_available:
            logger.warning("Tree-sitter not available - using fallback parsing")
            return
        
        try:
            # Python parser
            python_lang = Language(tree_sitter_python.language())
            python_parser = Parser(python_lang)
            self.languages[CodeLanguage.PYTHON] = python_lang
            self.parsers[CodeLanguage.PYTHON] = python_parser
            
            # JavaScript parser  
            js_lang = Language(tree_sitter_javascript.language())
            js_parser = Parser(js_lang)
            self.languages[CodeLanguage.JAVASCRIPT] = js_lang
            self.parsers[CodeLanguage.JAVASCRIPT] = js_parser
            
            # TypeScript parser
            ts_lang = Language(tree_sitter_typescript.language())
            ts_parser = Parser(ts_lang)
            self.languages[CodeLanguage.TYPESCRIPT] = ts_lang
            self.parsers[CodeLanguage.TYPESCRIPT] = ts_parser
            
            logger.info("Tree-sitter parsers initialized for Python, JavaScript, TypeScript")
            
        except Exception as e:
            logger.error(f"Error setting up tree-sitter parsers: {e}")
    
    def _setup_lsp_capabilities(self):
        """Setup LSP server capabilities for different languages"""
        # Python LSP (pylsp)
        self.lsp_capabilities[CodeLanguage.PYTHON] = LSPCapability(
            language=CodeLanguage.PYTHON,
            server_command=["pylsp"],
            initialization_options={},
            supported_features={"hover", "completion", "definition", "references", "formatting"},
            workspace_folders=[]
        )
        
        # TypeScript/JavaScript LSP  
        self.lsp_capabilities[CodeLanguage.TYPESCRIPT] = LSPCapability(
            language=CodeLanguage.TYPESCRIPT,
            server_command=["typescript-language-server", "--stdio"],
            initialization_options={},
            supported_features={"hover", "completion", "definition", "references", "formatting"},
            workspace_folders=[]
        )
        
        self.lsp_capabilities[CodeLanguage.JAVASCRIPT] = LSPCapability(
            language=CodeLanguage.JAVASCRIPT,
            server_command=["typescript-language-server", "--stdio"],
            initialization_options={},
            supported_features={"hover", "completion", "definition", "references"},
            workspace_folders=[]
        )
    
    async def parse_file(self, file_path: str, force_reparse: bool = False) -> Optional[CodeTree]:
        """Parse single file using tree-sitter"""
        start_time = datetime.now()
        
        try:
            # Normalize path
            file_path = os.path.abspath(file_path)
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            # Detect language
            language = self._detect_language(file_path)
            if not language:
                logger.warning(f"Unsupported language for file: {file_path}")
                return None
            
            # Check cache
            if not force_reparse and file_path in self.parsed_trees:
                cached_tree = self.parsed_trees[file_path]
                file_stat = os.stat(file_path)
                file_modified = datetime.fromtimestamp(file_stat.st_mtime)
                
                if cached_tree.last_modified >= file_modified:
                    self.parse_metrics["cache_hits"] += 1
                    return cached_tree
            
            self.parse_metrics["cache_misses"] += 1
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
            
            # Parse with tree-sitter or fallback
            if self.tree_sitter_available and language in self.parsers:
                tree = await self._parse_with_tree_sitter(source_code, language)
                symbols = await self._extract_symbols_from_tree(tree, source_code, file_path, language)
            else:
                tree = None
                symbols = await self._fallback_parse_symbols(source_code, file_path, language)
            
            # Calculate complexity metrics
            complexity_metrics = await self._calculate_complexity_metrics(symbols, source_code)
            
            # Extract imports
            imports = await self._extract_imports(symbols, source_code, language)
            
            # Create code tree
            file_stat = os.stat(file_path)
            tree_hash = hashlib.md5(source_code.encode()).hexdigest()
            
            code_tree = CodeTree(
                file_path=file_path,
                language=language,
                last_modified=datetime.fromtimestamp(file_stat.st_mtime),
                symbols=symbols,
                imports=imports,
                errors=[],  # Will be populated if there are parse errors
                complexity_metrics=complexity_metrics,
                tree_hash=tree_hash
            )
            
            # Cache the parsed tree
            self.parsed_trees[file_path] = code_tree
            self.symbol_cache[file_path] = symbols
            
            # Update metrics
            parse_time = (datetime.now() - start_time).total_seconds() * 1000
            self.parse_metrics["files_parsed"] += 1
            self.parse_metrics["total_parse_time_ms"] += parse_time
            self.parse_metrics["average_parse_time_ms"] = (
                self.parse_metrics["total_parse_time_ms"] / self.parse_metrics["files_parsed"]
            )
            
            logger.debug(f"Parsed {file_path} in {parse_time:.2f}ms, found {len(symbols)} symbols")
            
            # Store in memory system for future retrieval
            await self._store_in_memory_system(code_tree)
            
            return code_tree
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return None
    
    def _detect_language(self, file_path: str) -> Optional[CodeLanguage]:
        """Detect programming language from file extension"""
        extension = Path(file_path).suffix.lower()
        
        extension_map = {
            '.py': CodeLanguage.PYTHON,
            '.js': CodeLanguage.JAVASCRIPT,
            '.ts': CodeLanguage.TYPESCRIPT,
            '.tsx': CodeLanguage.TYPESCRIPT,
            '.json': CodeLanguage.JSON,
            '.md': CodeLanguage.MARKDOWN,
            '.yml': CodeLanguage.YAML,
            '.yaml': CodeLanguage.YAML,
            '.go': CodeLanguage.GO,
            '.rs': CodeLanguage.RUST,
            '.java': CodeLanguage.JAVA,
            '.cpp': CodeLanguage.CPP,
            '.cc': CodeLanguage.CPP,
            '.cxx': CodeLanguage.CPP,
        }
        
        return extension_map.get(extension)
    
    async def _parse_with_tree_sitter(self, source_code: str, language: CodeLanguage) -> Tree:
        """Parse source code using tree-sitter"""
        if language not in self.parsers:
            raise ValueError(f"No parser available for {language}")
        
        parser = self.parsers[language]
        tree = parser.parse(bytes(source_code, "utf8"))
        
        return tree
    
    async def _extract_symbols_from_tree(self, tree: Tree, source_code: str, 
                                       file_path: str, language: CodeLanguage) -> List[CodeSymbol]:
        """Extract symbols from parsed tree-sitter tree"""
        symbols = []
        source_lines = source_code.split('\n')
        
        def traverse_node(node: Node, parent_name: str = None):
            node_type = node.type
            
            # Map tree-sitter node types to our NodeType enum
            if node_type in ['function_def', 'function_definition', 'method_definition']:
                symbol = self._create_function_symbol(node, source_lines, file_path, language, parent_name)
                if symbol:
                    symbols.append(symbol)
                    # Set parent name for nested functions
                    new_parent = symbol.name
                    for child in node.children:
                        traverse_node(child, new_parent)
                    return  # Don't traverse children again
                    
            elif node_type in ['class_def', 'class_definition', 'class_declaration']:
                symbol = self._create_class_symbol(node, source_lines, file_path, language, parent_name)
                if symbol:
                    symbols.append(symbol)
                    # Set parent name for class members
                    new_parent = symbol.name
                    for child in node.children:
                        traverse_node(child, new_parent)
                    return
                    
            elif node_type in ['import_statement', 'import_from_statement']:
                symbol = self._create_import_symbol(node, source_lines, file_path, language)
                if symbol:
                    symbols.append(symbol)
                    
            elif node_type in ['variable_declaration', 'assignment']:
                symbol = self._create_variable_symbol(node, source_lines, file_path, language, parent_name)
                if symbol:
                    symbols.append(symbol)
            
            # Continue traversing children
            for child in node.children:
                traverse_node(child, parent_name)
        
        # Start traversal from root
        traverse_node(tree.root_node)
        
        return symbols
    
    def _create_function_symbol(self, node: Node, source_lines: List[str], 
                              file_path: str, language: CodeLanguage, 
                              parent_name: str = None) -> Optional[CodeSymbol]:
        """Create function symbol from tree-sitter node"""
        try:
            # Extract function name
            name_node = None
            for child in node.children:
                if child.type in ['identifier', 'name']:
                    name_node = child
                    break
            
            if not name_node:
                return None
            
            start_point = name_node.start_point
            end_point = node.end_point
            
            function_name = source_lines[start_point[0]][start_point[1]:start_point[1] + (name_node.end_point[1] - name_node.start_point[1])]
            
            # Extract parameters
            parameters = []
            for child in node.children:
                if child.type in ['parameters', 'parameter_list']:
                    parameters = self._extract_parameters(child, source_lines)
                    break
            
            # Extract docstring
            docstring = self._extract_docstring(node, source_lines)
            
            # Calculate complexity
            complexity = self._calculate_node_complexity(node)
            
            location = CodeLocation(
                file_path=file_path,
                start_line=start_point[0] + 1,
                end_line=end_point[0] + 1,
                start_column=start_point[1],
                end_column=end_point[1],
                start_byte=node.start_byte,
                end_byte=node.end_byte
            )
            
            return CodeSymbol(
                name=function_name,
                symbol_type=NodeType.FUNCTION_DEF,
                language=language,
                location=location,
                parent_name=parent_name,
                docstring=docstring,
                parameters=parameters,
                complexity_score=complexity
            )
            
        except Exception as e:
            logger.error(f"Error creating function symbol: {e}")
            return None
    
    def _create_class_symbol(self, node: Node, source_lines: List[str],
                           file_path: str, language: CodeLanguage,
                           parent_name: str = None) -> Optional[CodeSymbol]:
        """Create class symbol from tree-sitter node"""
        try:
            # Extract class name
            name_node = None
            for child in node.children:
                if child.type in ['identifier', 'name']:
                    name_node = child
                    break
            
            if not name_node:
                return None
            
            start_point = name_node.start_point
            end_point = node.end_point
            
            class_name = source_lines[start_point[0]][start_point[1]:start_point[1] + (name_node.end_point[1] - name_node.start_point[1])]
            
            # Extract docstring
            docstring = self._extract_docstring(node, source_lines)
            
            location = CodeLocation(
                file_path=file_path,
                start_line=start_point[0] + 1,
                end_line=end_point[0] + 1,
                start_column=start_point[1],
                end_column=end_point[1],
                start_byte=node.start_byte,
                end_byte=node.end_byte
            )
            
            return CodeSymbol(
                name=class_name,
                symbol_type=NodeType.CLASS_DEF,
                language=language,
                location=location,
                parent_name=parent_name,
                docstring=docstring,
                complexity_score=2  # Classes have base complexity of 2
            )
            
        except Exception as e:
            logger.error(f"Error creating class symbol: {e}")
            return None
    
    def _create_import_symbol(self, node: Node, source_lines: List[str],
                            file_path: str, language: CodeLanguage) -> Optional[CodeSymbol]:
        """Create import symbol from tree-sitter node"""
        try:
            start_point = node.start_point
            end_point = node.end_point
            
            # Extract import statement text
            import_text = ""
            for line_idx in range(start_point[0], end_point[0] + 1):
                if line_idx < len(source_lines):
                    line = source_lines[line_idx]
                    if line_idx == start_point[0] and line_idx == end_point[0]:
                        import_text = line[start_point[1]:end_point[1]]
                    elif line_idx == start_point[0]:
                        import_text = line[start_point[1]:]
                    elif line_idx == end_point[0]:
                        import_text += "\n" + line[:end_point[1]]
                    else:
                        import_text += "\n" + line
            
            location = CodeLocation(
                file_path=file_path,
                start_line=start_point[0] + 1,
                end_line=end_point[0] + 1,
                start_column=start_point[1],
                end_column=end_point[1],
                start_byte=node.start_byte,
                end_byte=node.end_byte
            )
            
            return CodeSymbol(
                name=import_text.strip(),
                symbol_type=NodeType.IMPORT,
                language=language,
                location=location,
                complexity_score=1
            )
            
        except Exception as e:
            logger.error(f"Error creating import symbol: {e}")
            return None
    
    def _create_variable_symbol(self, node: Node, source_lines: List[str],
                              file_path: str, language: CodeLanguage,
                              parent_name: str = None) -> Optional[CodeSymbol]:
        """Create variable symbol from tree-sitter node"""
        try:
            # This is a simplified implementation
            # In practice, you'd extract the actual variable name and type information
            
            start_point = node.start_point
            end_point = node.end_point
            
            # Extract variable name (simplified)
            var_line = source_lines[start_point[0]] if start_point[0] < len(source_lines) else ""
            var_name = var_line.strip().split('=')[0].strip() if '=' in var_line else "variable"
            
            location = CodeLocation(
                file_path=file_path,
                start_line=start_point[0] + 1,
                end_line=end_point[0] + 1,
                start_column=start_point[1],
                end_column=end_point[1],
                start_byte=node.start_byte,
                end_byte=node.end_byte
            )
            
            return CodeSymbol(
                name=var_name,
                symbol_type=NodeType.VARIABLE,
                language=language,
                location=location,
                parent_name=parent_name,
                complexity_score=1
            )
            
        except Exception as e:
            logger.error(f"Error creating variable symbol: {e}")
            return None
    
    def _extract_parameters(self, param_node: Node, source_lines: List[str]) -> List[str]:
        """Extract function parameters from parameter node"""
        parameters = []
        
        for child in param_node.children:
            if child.type in ['identifier', 'parameter']:
                start_point = child.start_point
                end_point = child.end_point
                
                if start_point[0] < len(source_lines):
                    param_text = source_lines[start_point[0]][start_point[1]:end_point[1]]
                    parameters.append(param_text.strip())
        
        return parameters
    
    def _extract_docstring(self, node: Node, source_lines: List[str]) -> Optional[str]:
        """Extract docstring from function/class node"""
        # Look for string literals immediately after function/class definition
        for child in node.children:
            if child.type in ['string', 'string_literal']:
                start_point = child.start_point
                end_point = child.end_point
                
                docstring = ""
                for line_idx in range(start_point[0], end_point[0] + 1):
                    if line_idx < len(source_lines):
                        line = source_lines[line_idx]
                        if line_idx == start_point[0] and line_idx == end_point[0]:
                            docstring = line[start_point[1]:end_point[1]]
                        elif line_idx == start_point[0]:
                            docstring = line[start_point[1]:]
                        elif line_idx == end_point[0]:
                            docstring += "\n" + line[:end_point[1]]
                        else:
                            docstring += "\n" + line
                
                return docstring.strip().strip('"\'')
        
        return None
    
    def _calculate_node_complexity(self, node: Node) -> int:
        """Calculate cyclomatic complexity of a node"""
        complexity = 1  # Base complexity
        
        def traverse_for_complexity(n: Node):
            nonlocal complexity
            
            # Add complexity for control structures
            if n.type in ['if_statement', 'elif_clause', 'while_statement', 
                         'for_statement', 'try_statement', 'except_clause',
                         'with_statement', 'case_clause']:
                complexity += 1
            elif n.type in ['and', 'or']:  # Boolean operators
                complexity += 1
            
            for child in n.children:
                traverse_for_complexity(child)
        
        traverse_for_complexity(node)
        return complexity
    
    async def _fallback_parse_symbols(self, source_code: str, file_path: str, 
                                    language: CodeLanguage) -> List[CodeSymbol]:
        """Fallback symbol extraction using regex when tree-sitter unavailable"""
        symbols = []
        source_lines = source_code.split('\n')
        
        if language == CodeLanguage.PYTHON:
            symbols.extend(self._parse_python_fallback(source_lines, file_path))
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            symbols.extend(self._parse_js_fallback(source_lines, file_path, language))
        
        return symbols
    
    def _parse_python_fallback(self, source_lines: List[str], file_path: str) -> List[CodeSymbol]:
        """Fallback Python parsing using regex"""
        import re
        symbols = []
        
        for line_num, line in enumerate(source_lines):
            line_stripped = line.strip()
            
            # Function definitions
            func_match = re.match(r'^def\s+(\w+)\s*\((.*?)\):', line_stripped)
            if func_match:
                func_name = func_match.group(1)
                params_str = func_match.group(2)
                params = [p.strip() for p in params_str.split(',') if p.strip()]
                
                location = CodeLocation(
                    file_path=file_path,
                    start_line=line_num + 1,
                    end_line=line_num + 1,
                    start_column=0,
                    end_column=len(line),
                    start_byte=0,
                    end_byte=0
                )
                
                symbols.append(CodeSymbol(
                    name=func_name,
                    symbol_type=NodeType.FUNCTION_DEF,
                    language=CodeLanguage.PYTHON,
                    location=location,
                    parameters=params,
                    complexity_score=2
                ))
            
            # Class definitions
            class_match = re.match(r'^class\s+(\w+).*?:', line_stripped)
            if class_match:
                class_name = class_match.group(1)
                
                location = CodeLocation(
                    file_path=file_path,
                    start_line=line_num + 1,
                    end_line=line_num + 1,
                    start_column=0,
                    end_column=len(line),
                    start_byte=0,
                    end_byte=0
                )
                
                symbols.append(CodeSymbol(
                    name=class_name,
                    symbol_type=NodeType.CLASS_DEF,
                    language=CodeLanguage.PYTHON,
                    location=location,
                    complexity_score=2
                ))
            
            # Import statements
            import_match = re.match(r'^(import\s+.+|from\s+.+\s+import\s+.+)', line_stripped)
            if import_match:
                import_stmt = import_match.group(1)
                
                location = CodeLocation(
                    file_path=file_path,
                    start_line=line_num + 1,
                    end_line=line_num + 1,
                    start_column=0,
                    end_column=len(line),
                    start_byte=0,
                    end_byte=0
                )
                
                symbols.append(CodeSymbol(
                    name=import_stmt,
                    symbol_type=NodeType.IMPORT,
                    language=CodeLanguage.PYTHON,
                    location=location,
                    complexity_score=1
                ))
        
        return symbols
    
    def _parse_js_fallback(self, source_lines: List[str], file_path: str, 
                          language: CodeLanguage) -> List[CodeSymbol]:
        """Fallback JavaScript/TypeScript parsing using regex"""
        import re
        symbols = []
        
        for line_num, line in enumerate(source_lines):
            line_stripped = line.strip()
            
            # Function declarations
            func_matches = [
                re.match(r'^function\s+(\w+)\s*\((.*?)\)', line_stripped),
                re.match(r'^(\w+)\s*:\s*function\s*\((.*?)\)', line_stripped),
                re.match(r'^(\w+)\s*=\s*function\s*\((.*?)\)', line_stripped),
                re.match(r'^(\w+)\s*=\s*\((.*?)\)\s*=>', line_stripped),
                re.match(r'^const\s+(\w+)\s*=\s*\((.*?)\)\s*=>', line_stripped),
            ]
            
            for func_match in func_matches:
                if func_match:
                    func_name = func_match.group(1)
                    params_str = func_match.group(2)
                    params = [p.strip() for p in params_str.split(',') if p.strip()]
                    
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=line_num + 1,
                        end_line=line_num + 1,
                        start_column=0,
                        end_column=len(line),
                        start_byte=0,
                        end_byte=0
                    )
                    
                    symbols.append(CodeSymbol(
                        name=func_name,
                        symbol_type=NodeType.FUNCTION_DEF,
                        language=language,
                        location=location,
                        parameters=params,
                        complexity_score=2
                    ))
                    break
            
            # Class declarations
            class_match = re.match(r'^class\s+(\w+)', line_stripped)
            if class_match:
                class_name = class_match.group(1)
                
                location = CodeLocation(
                    file_path=file_path,
                    start_line=line_num + 1,
                    end_line=line_num + 1,
                    start_column=0,
                    end_column=len(line),
                    start_byte=0,
                    end_byte=0
                )
                
                symbols.append(CodeSymbol(
                    name=class_name,
                    symbol_type=NodeType.CLASS_DEF,
                    language=language,
                    location=location,
                    complexity_score=2
                ))
            
            # Import statements
            import_matches = [
                re.match(r'^import\s+.+', line_stripped),
                re.match(r'^const\s+.+\s*=\s*require\(.+\)', line_stripped),
            ]
            
            for import_match in import_matches:
                if import_match:
                    import_stmt = import_match.group(0)
                    
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=line_num + 1,
                        end_line=line_num + 1,
                        start_column=0,
                        end_column=len(line),
                        start_byte=0,
                        end_byte=0
                    )
                    
                    symbols.append(CodeSymbol(
                        name=import_stmt,
                        symbol_type=NodeType.IMPORT,
                        language=language,
                        location=location,
                        complexity_score=1
                    ))
                    break
        
        return symbols
    
    async def _calculate_complexity_metrics(self, symbols: List[CodeSymbol], 
                                          source_code: str) -> Dict[str, float]:
        """Calculate various complexity metrics"""
        metrics = {
            "cyclomatic_complexity": 0,
            "cognitive_complexity": 0,
            "lines_of_code": len(source_code.split('\n')),
            "function_count": 0,
            "class_count": 0,
            "import_count": 0,
            "average_function_complexity": 0,
            "maintainability_index": 0
        }
        
        function_complexities = []
        
        for symbol in symbols:
            if symbol.symbol_type == NodeType.FUNCTION_DEF:
                metrics["function_count"] += 1
                metrics["cyclomatic_complexity"] += symbol.complexity_score
                function_complexities.append(symbol.complexity_score)
            elif symbol.symbol_type == NodeType.CLASS_DEF:
                metrics["class_count"] += 1
            elif symbol.symbol_type == NodeType.IMPORT:
                metrics["import_count"] += 1
        
        if function_complexities:
            metrics["average_function_complexity"] = sum(function_complexities) / len(function_complexities)
        
        # Simple maintainability index calculation
        loc = metrics["lines_of_code"]
        cc = metrics["cyclomatic_complexity"]
        
        if loc > 0:
            # Simplified maintainability index formula
            metrics["maintainability_index"] = max(0, (171 - 5.2 * cc - 0.23 * loc) / 171 * 100)
        
        return metrics
    
    async def _extract_imports(self, symbols: List[CodeSymbol], 
                             source_code: str, language: CodeLanguage) -> List[str]:
        """Extract import statements from symbols"""
        imports = []
        
        for symbol in symbols:
            if symbol.symbol_type == NodeType.IMPORT:
                imports.append(symbol.name)
        
        return imports
    
    async def _store_in_memory_system(self, code_tree: CodeTree):
        """Store parsed code tree in memory system for retrieval"""
        try:
            # Create memory entries for symbols
            for symbol in code_tree.symbols:
                memory_data = {
                    "symbol_name": symbol.name,
                    "symbol_type": symbol.symbol_type.value,
                    "file_path": symbol.location.file_path,
                    "language": symbol.language.value,
                    "start_line": symbol.location.start_line,
                    "complexity": symbol.complexity_score,
                    "parent": symbol.parent_name,
                    "docstring": symbol.docstring,
                    "parameters": symbol.parameters,
                }
                
                await self.memory_system.store_memory(
                    content=f"Code symbol: {symbol.name}",
                    memory_type="code_symbol",
                    metadata=memory_data,
                    context_tags=[
                        f"lang:{symbol.language.value}",
                        f"type:{symbol.symbol_type.value}",
                        f"file:{Path(symbol.location.file_path).name}"
                    ]
                )
            
            # Store file-level metadata
            file_metadata = {
                "file_path": code_tree.file_path,
                "language": code_tree.language.value,
                "symbol_count": len(code_tree.symbols),
                "complexity_metrics": code_tree.complexity_metrics,
                "imports": code_tree.imports,
                "tree_hash": code_tree.tree_hash
            }
            
            await self.memory_system.store_memory(
                content=f"Code file: {Path(code_tree.file_path).name}",
                memory_type="code_file",
                metadata=file_metadata,
                context_tags=[
                    f"lang:{code_tree.language.value}",
                    f"file:{Path(code_tree.file_path).name}"
                ]
            )
            
        except Exception as e:
            logger.error(f"Error storing code tree in memory system: {e}")
    
    async def parse_directory(self, directory_path: str, 
                            recursive: bool = True,
                            file_patterns: List[str] = None) -> Dict[str, CodeTree]:
        """Parse all files in a directory"""
        if file_patterns is None:
            file_patterns = ["*.py", "*.js", "*.ts", "*.tsx"]
        
        parsed_files = {}
        
        try:
            directory_path = os.path.abspath(directory_path)
            
            if not os.path.exists(directory_path):
                logger.error(f"Directory not found: {directory_path}")
                return parsed_files
            
            # Collect files to parse
            files_to_parse = []
            
            if recursive:
                for root, dirs, files in os.walk(directory_path):
                    # Skip common ignored directories
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'dist', 'build']]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        if self._should_parse_file(file_path, file_patterns):
                            files_to_parse.append(file_path)
            else:
                for file in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, file)
                    if os.path.isfile(file_path) and self._should_parse_file(file_path, file_patterns):
                        files_to_parse.append(file_path)
            
            logger.info(f"Found {len(files_to_parse)} files to parse in {directory_path}")
            
            # Parse files concurrently (with limit to avoid overwhelming system)
            semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent parses
            
            async def parse_with_semaphore(file_path):
                async with semaphore:
                    return await self.parse_file(file_path)
            
            tasks = [parse_with_semaphore(file_path) for file_path in files_to_parse]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for file_path, result in zip(files_to_parse, results):
                if isinstance(result, Exception):
                    logger.error(f"Error parsing {file_path}: {result}")
                elif result:
                    parsed_files[file_path] = result
            
            logger.info(f"Successfully parsed {len(parsed_files)} files")
            
        except Exception as e:
            logger.error(f"Error parsing directory {directory_path}: {e}")
        
        return parsed_files
    
    def _should_parse_file(self, file_path: str, patterns: List[str]) -> bool:
        """Check if file should be parsed based on patterns"""
        import fnmatch
        
        file_name = os.path.basename(file_path)
        
        for pattern in patterns:
            if fnmatch.fnmatch(file_name, pattern):
                return True
        
        return False
    
    async def search_symbols(self, query: str, file_path: str = None,
                           symbol_types: List[NodeType] = None,
                           language: CodeLanguage = None) -> List[CodeSymbol]:
        """Search for symbols across parsed code"""
        results = []
        
        # Search in specific file or all parsed files
        files_to_search = [file_path] if file_path else list(self.symbol_cache.keys())
        
        for file in files_to_search:
            if file not in self.symbol_cache:
                continue
            
            symbols = self.symbol_cache[file]
            
            for symbol in symbols:
                # Apply filters
                if symbol_types and symbol.symbol_type not in symbol_types:
                    continue
                
                if language and symbol.language != language:
                    continue
                
                # Search in symbol name and docstring
                if query.lower() in symbol.name.lower():
                    results.append(symbol)
                elif symbol.docstring and query.lower() in symbol.docstring.lower():
                    results.append(symbol)
        
        return results
    
    async def get_symbol_references(self, symbol_name: str, file_path: str = None) -> List[CodeLocation]:
        """Find all references to a symbol (simplified implementation)"""
        references = []
        
        # This is a simplified implementation
        # In practice, you'd use LSP or more sophisticated analysis
        
        files_to_search = [file_path] if file_path else list(self.parsed_trees.keys())
        
        for file in files_to_search:
            if file not in self.parsed_trees:
                continue
            
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for line_num, line in enumerate(lines):
                    if symbol_name in line:
                        # Found potential reference
                        col_start = line.find(symbol_name)
                        references.append(CodeLocation(
                            file_path=file,
                            start_line=line_num + 1,
                            end_line=line_num + 1,
                            start_column=col_start,
                            end_column=col_start + len(symbol_name),
                            start_byte=0,
                            end_byte=0
                        ))
                        
            except Exception as e:
                logger.error(f"Error searching references in {file}: {e}")
        
        return references
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        total_symbols = sum(len(symbols) for symbols in self.symbol_cache.values())
        
        symbol_type_counts = {}
        language_counts = {}
        complexity_distribution = {"low": 0, "medium": 0, "high": 0}
        
        for symbols in self.symbol_cache.values():
            for symbol in symbols:
                # Count by symbol type
                symbol_type = symbol.symbol_type.value
                symbol_type_counts[symbol_type] = symbol_type_counts.get(symbol_type, 0) + 1
                
                # Count by language
                language = symbol.language.value
                language_counts[language] = language_counts.get(language, 0) + 1
                
                # Complexity distribution
                if symbol.complexity_score <= 2:
                    complexity_distribution["low"] += 1
                elif symbol.complexity_score <= 5:
                    complexity_distribution["medium"] += 1
                else:
                    complexity_distribution["high"] += 1
        
        return {
            "parsing_performance": self.parse_metrics,
            "code_analysis": {
                "total_files_parsed": len(self.parsed_trees),
                "total_symbols": total_symbols,
                "symbol_types": symbol_type_counts,
                "languages": language_counts,
                "complexity_distribution": complexity_distribution
            },
            "system_status": {
                "tree_sitter_available": self.tree_sitter_available,
                "active_parsers": len(self.parsers),
                "cached_trees": len(self.parsed_trees)
            }
        }


# Global instance
tree_sitter_explorer = TreeSitterLSPExplorer()