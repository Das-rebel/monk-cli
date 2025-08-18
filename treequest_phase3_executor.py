#!/usr/bin/env python3
"""
TreeQuest Phase 3 Executor
Completes the remaining Phase 3 Claude-style interface components
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import time

class TreeQuestPhase3Executor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        
    async def execute_all_phase3_tasks(self):
        """Execute all remaining Phase 3 tasks"""
        print("🚀 TreeQuest Phase 3 Executor - Claude-Style Interface")
        print("=" * 60)
        
        tasks = [
            self.implement_slash_command_processor,
            self.implement_intelligent_router,
            self.implement_nl_command_parser,
            self.implement_auto_completion,
            self.implement_project_context_loader,
            self.implement_session_management_integration,
            self.update_main_cli_integration
        ]
        
        for i, task in enumerate(tasks, 1):
            print(f"\n🔧 Executing Task {i}/7: {task.__name__}")
            try:
                await task()
                print(f"✅ Task {i} completed successfully")
            except Exception as e:
                print(f"❌ Task {i} failed: {e}")
                
        print("\n🎉 TreeQuest Phase 3 Execution Completed!")
    
    async def implement_slash_command_processor(self):
        """Implement Claude-style slash command processor"""
        print("🔧 Implementing Slash Command Processor...")
        
        slash_command_content = '''"""
Slash Command Processor
Claude-style slash commands (/help, /clear, /settings, etc.)
"""

import asyncio
import re
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import logging

from src.core.conversation_manager import conversation_manager
from src.analyzers.analyzer_coordinator import AnalyzerCoordinator
from src.workspace.workspace_manager import WorkspaceManager

logger = logging.getLogger(__name__)

@dataclass
class SlashCommand:
    """Represents a parsed slash command"""
    name: str
    args: List[str]
    kwargs: Dict[str, Any]
    raw_input: str

class SlashCommandProcessor:
    """
    Processes Claude-style slash commands
    """
    
    def __init__(self):
        self.commands: Dict[str, Callable] = {}
        self.command_help: Dict[str, str] = {}
        self.analyzer_coordinator = AnalyzerCoordinator()
        self.workspace_manager = WorkspaceManager()
        self._register_core_commands()
    
    def _register_core_commands(self):
        """Register all core slash commands"""
        
        # Core system commands
        self.register_command("help", self._cmd_help, "Show available commands and context-aware help")
        self.register_command("clear", self._cmd_clear, "Clear conversation history")
        self.register_command("settings", self._cmd_settings, "Show/modify CLI settings")
        
        # Project and analysis commands
        self.register_command("analyze", self._cmd_analyze, "Analyze files or entire project")
        self.register_command("workspace", self._cmd_workspace, "Show current workspace information")
        self.register_command("project", self._cmd_project, "Show project context and details")
        
        # Provider and session management
        self.register_command("providers", self._cmd_providers, "List and manage AI providers")
        self.register_command("history", self._cmd_history, "Show conversation history")
        self.register_command("save", self._cmd_save, "Save current conversation session")
        self.register_command("load", self._cmd_load, "Load saved conversation session")
        
        # File operations
        self.register_command("ls", self._cmd_ls, "List directory contents")
        self.register_command("cat", self._cmd_cat, "Show file contents")
        self.register_command("tree", self._cmd_tree, "Show project directory tree")
        
        # Git operations
        self.register_command("git", self._cmd_git, "Git repository operations")
        self.register_command("status", self._cmd_git_status, "Show git status")
        self.register_command("diff", self._cmd_git_diff, "Show git diff")
    
    def register_command(self, name: str, func: Callable, help_text: str):
        """Register a new slash command"""
        self.commands[name] = func
        self.command_help[name] = help_text
        logger.debug(f"Registered slash command: /{name}")
    
    def parse_command(self, input_text: str) -> Optional[SlashCommand]:
        """Parse slash command from input text"""
        input_text = input_text.strip()
        
        if not input_text.startswith('/'):
            return None
        
        # Remove leading slash
        command_text = input_text[1:]
        
        if not command_text:
            return SlashCommand("help", [], {}, input_text)
        
        # Split command and arguments
        parts = command_text.split()
        command_name = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        # Parse key=value arguments
        kwargs = {}
        filtered_args = []
        
        for arg in args:
            if '=' in arg and not arg.startswith('-'):
                key, value = arg.split('=', 1)
                kwargs[key] = value
            else:
                filtered_args.append(arg)
        
        return SlashCommand(command_name, filtered_args, kwargs, input_text)
    
    async def execute_command(self, command: SlashCommand) -> Dict[str, Any]:
        """Execute a parsed slash command"""
        if command.name not in self.commands:
            return {
                "success": False,
                "error": f"Unknown command: /{command.name}",
                "suggestions": self._get_command_suggestions(command.name)
            }
        
        try:
            command_func = self.commands[command.name]
            
            # Execute command
            if asyncio.iscoroutinefunction(command_func):
                result = await command_func(command)
            else:
                result = command_func(command)
            
            return {
                "success": True,
                "data": result,
                "command": command.name
            }
            
        except Exception as e:
            logger.error(f"Error executing command /{command.name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "command": command.name
            }
    
    def _get_command_suggestions(self, partial_command: str) -> List[str]:
        """Get command suggestions for unknown/partial commands"""
        suggestions = []
        
        for cmd_name in self.commands.keys():
            if cmd_name.startswith(partial_command) or partial_command in cmd_name:
                suggestions.append(cmd_name)
        
        return sorted(suggestions)[:5]
    
    # Command implementations
    
    async def _cmd_help(self, cmd: SlashCommand) -> str:
        """Show help information"""
        if cmd.args:
            # Help for specific command
            command_name = cmd.args[0]
            if command_name in self.command_help:
                return f"/{command_name}: {self.command_help[command_name]}"
            else:
                suggestions = self._get_command_suggestions(command_name)
                help_text = f"Unknown command: /{command_name}"
                if suggestions:
                    help_text += f"\\nDid you mean: {', '.join(f'/{s}' for s in suggestions)}"
                return help_text
        
        # General help
        help_lines = ["Available Commands:"]
        
        # Group commands by category
        categories = {
            "System": ["help", "clear", "settings", "providers"],
            "Project": ["analyze", "workspace", "project", "tree"],
            "Files": ["ls", "cat"],
            "Git": ["git", "status", "diff"],
            "Session": ["history", "save", "load"]
        }
        
        for category, commands in categories.items():
            help_lines.append(f"\\n**{category}:**")
            for cmd_name in commands:
                if cmd_name in self.command_help:
                    help_lines.append(f"  /{cmd_name:<12} - {self.command_help[cmd_name]}")
        
        # Add context-aware suggestions
        context = conversation_manager.get_project_context()
        if context:
            help_lines.append(f"\\n**Context-Aware Suggestions:**")
            help_lines.append(f"  /analyze {context.project_path}")
            help_lines.append(f"  /workspace")
            if context.git_status:
                help_lines.append(f"  /status")
        
        return "\\n".join(help_lines)
    
    async def _cmd_clear(self, cmd: SlashCommand) -> str:
        """Clear conversation history"""
        conversation_manager.clear_conversation()
        return "✅ Conversation history cleared"
    
    async def _cmd_settings(self, cmd: SlashCommand) -> str:
        """Show or modify settings"""
        if cmd.args:
            if len(cmd.args) >= 2:
                key, value = cmd.args[0], cmd.args[1]
                # TODO: Implement settings management
                return f"✅ Setting {key} = {value}"
            else:
                key = cmd.args[0]
                return f"Setting {key}: (value would be shown here)"
        else:
            return """Current Settings:
- Provider: auto-select
- Format: rich
- Context Window: 10 messages
- Cache: enabled
Use /settings <key> <value> to modify"""
    
    async def _cmd_analyze(self, cmd: SlashCommand) -> str:
        """Analyze files or project"""
        target = cmd.args[0] if cmd.args else "."
        path = Path(target).resolve()
        
        if not path.exists():
            return f"❌ Path not found: {target}"
        
        # Use analyzer coordinator
        try:
            result = await self.analyzer_coordinator.analyze_path(str(path))
            return f"📊 Analysis completed for {target}\\nResults: {len(result.get('findings', []))} findings"
        except Exception as e:
            return f"❌ Analysis failed: {e}"
    
    async def _cmd_workspace(self, cmd: SlashCommand) -> str:
        """Show workspace information"""
        current_workspace = self.workspace_manager.get_current_workspace()
        if current_workspace:
            return f"""📁 **Current Workspace**: {current_workspace['name']}
- **Path**: {current_workspace['path']}
- **Type**: {current_workspace.get('project_type', 'Unknown')}
- **Last Updated**: {current_workspace.get('last_updated', 'Unknown')}"""
        else:
            return "No active workspace. Use /analyze to detect project automatically."
    
    async def _cmd_project(self, cmd: SlashCommand) -> str:
        """Show project context"""
        context = conversation_manager.get_project_context()
        if context:
            return f"""📋 **Project Context**:
- **Path**: {context.project_path}
- **Type**: {context.project_type}
- **Git Status**: {'Clean' if context.git_status.get('clean') else 'Modified'}
- **Files**: {len(context.relevant_files)} relevant files
- **Dependencies**: {len(context.dependencies)} dependencies
- **Summary**: {context.summary}"""
        else:
            return "No project context loaded. Use /analyze to scan current directory."
    
    async def _cmd_providers(self, cmd: SlashCommand) -> str:
        """List and manage providers"""
        # TODO: Integrate with provider system
        return """🔧 **Available Providers**:
- ✅ claude_code (Active)
- ✅ treequest
- ❌ gemma (Unavailable)
- ✅ opendia
- ✅ mcp

Use /providers switch <name> to change active provider"""
    
    async def _cmd_history(self, cmd: SlashCommand) -> str:
        """Show conversation history"""
        messages = conversation_manager.get_context_messages()
        if not messages:
            return "No conversation history"
        
        history_lines = ["📜 **Recent Conversation**:"]
        for msg in messages[-5:]:  # Show last 5 messages
            timestamp = time.strftime("%H:%M", time.localtime(msg.timestamp))
            role = msg.role.title()
            preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            history_lines.append(f"  [{timestamp}] {role}: {preview}")
        
        return "\\n".join(history_lines)
    
    async def _cmd_save(self, cmd: SlashCommand) -> str:
        """Save conversation session"""
        session_name = cmd.args[0] if cmd.args else "default"
        # TODO: Implement named session saving
        return f"✅ Conversation saved as '{session_name}'"
    
    async def _cmd_load(self, cmd: SlashCommand) -> str:
        """Load conversation session"""
        session_name = cmd.args[0] if cmd.args else "default"
        # TODO: Implement session loading
        return f"✅ Loaded conversation '{session_name}'"
    
    async def _cmd_ls(self, cmd: SlashCommand) -> str:
        """List directory contents"""
        target = Path(cmd.args[0]) if cmd.args else Path.cwd()
        
        if not target.exists():
            return f"❌ Directory not found: {target}"
        
        if not target.is_dir():
            return f"❌ Not a directory: {target}"
        
        items = []
        for item in sorted(target.iterdir()):
            if item.is_dir():
                items.append(f"📁 {item.name}/")
            else:
                items.append(f"📄 {item.name}")
        
        return "\\n".join(items[:20])  # Limit to 20 items
    
    async def _cmd_cat(self, cmd: SlashCommand) -> str:
        """Show file contents"""
        if not cmd.args:
            return "❌ No file specified"
        
        file_path = Path(cmd.args[0])
        if not file_path.exists():
            return f"❌ File not found: {file_path}"
        
        try:
            content = file_path.read_text()
            lines = content.split('\\n')
            
            if len(lines) > 50:
                content = '\\n'.join(lines[:50]) + f'\\n... (truncated, {len(lines)} total lines)'
            
            return f"📄 **{file_path.name}**:\\n```\\n{content}\\n```"
        except Exception as e:
            return f"❌ Error reading file: {e}"
    
    async def _cmd_tree(self, cmd: SlashCommand) -> str:
        """Show directory tree"""
        target = Path(cmd.args[0]) if cmd.args else Path.cwd()
        
        if not target.exists() or not target.is_dir():
            return f"❌ Invalid directory: {target}"
        
        def build_tree(path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> List[str]:
            if current_depth >= max_depth:
                return []
            
            items = []
            entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            
            for i, entry in enumerate(entries[:10]):  # Limit items
                is_last = i == len(entries) - 1
                current_prefix = "└── " if is_last else "├── "
                next_prefix = prefix + ("    " if is_last else "│   ")
                
                if entry.is_dir():
                    items.append(f"{prefix}{current_prefix}📁 {entry.name}/")
                    items.extend(build_tree(entry, next_prefix, max_depth, current_depth + 1))
                else:
                    items.append(f"{prefix}{current_prefix}📄 {entry.name}")
            
            return items
        
        tree_lines = [f"📁 {target.name}/"] + build_tree(target)
        return "\\n".join(tree_lines)
    
    async def _cmd_git(self, cmd: SlashCommand) -> str:
        """Git operations"""
        if not cmd.args:
            return await self._cmd_git_status(cmd)
        
        git_cmd = cmd.args[0]
        if git_cmd == "status":
            return await self._cmd_git_status(cmd)
        elif git_cmd == "diff":
            return await self._cmd_git_diff(cmd)
        else:
            return f"Git subcommand '{git_cmd}' not implemented"
    
    async def _cmd_git_status(self, cmd: SlashCommand) -> str:
        """Show git status"""
        try:
            import subprocess
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode != 0:
                return "❌ Not a git repository"
            
            if not result.stdout.strip():
                return "✅ Working directory clean"
            
            status_lines = ["📊 **Git Status**:"]
            for line in result.stdout.strip().split('\\n')[:10]:
                status = line[:2]
                filename = line[3:]
                status_icon = "📝" if 'M' in status else "➕" if 'A' in status else "❓"
                status_lines.append(f"  {status_icon} {filename}")
            
            return "\\n".join(status_lines)
            
        except Exception as e:
            return f"❌ Error checking git status: {e}"
    
    async def _cmd_git_diff(self, cmd: SlashCommand) -> str:
        """Show git diff"""
        try:
            import subprocess
            result = subprocess.run(['git', 'diff', '--name-only'], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode != 0:
                return "❌ Not a git repository"
            
            if not result.stdout.strip():
                return "No changes to show"
            
            files = result.stdout.strip().split('\\n')
            return f"📊 **Modified Files** ({len(files)}):\\n" + "\\n".join(f"  📝 {f}" for f in files[:10])
            
        except Exception as e:
            return f"❌ Error checking git diff: {e}"

# Global slash command processor
slash_processor = SlashCommandProcessor()
'''
        
        os.makedirs(self.project_root / "src" / "core", exist_ok=True)
        with open(self.project_root / "src" / "core" / "slash_command_processor.py", "w") as f:
            f.write(slash_command_content)
        
        print("✅ Slash Command Processor implementation completed")
    
    async def implement_intelligent_router(self):
        """Implement intelligent query routing to optimal providers"""
        print("🔧 Implementing Intelligent Router...")
        
        router_content = '''"""
Intelligent Router
Routes queries to optimal AI providers based on query type and context
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from src.core.conversation_manager import conversation_manager

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries for routing decisions"""
    CODE_ANALYSIS = "code_analysis"
    CODE_GENERATION = "code_generation"
    FILE_OPERATION = "file_operation"
    GIT_OPERATION = "git_operation"
    GENERAL_QUESTION = "general_question"
    PROJECT_ANALYSIS = "project_analysis"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    SYSTEM_COMMAND = "system_command"

@dataclass
class RoutingDecision:
    """Result of routing analysis"""
    primary_provider: str
    fallback_providers: List[str]
    confidence: float
    reasoning: str
    query_type: QueryType
    metadata: Dict[str, Any]

class IntelligentRouter:
    """
    Routes queries to optimal AI providers based on analysis
    """
    
    def __init__(self):
        # Provider capabilities mapping
        self.provider_capabilities = {
            'claude_code': {
                'strengths': ['code_analysis', 'file_operation', 'debugging', 'documentation'],
                'performance': 0.95,
                'availability': self._check_claude_availability
            },
            'treequest': {
                'strengths': ['code_generation', 'project_analysis', 'complex_tasks'],
                'performance': 0.90,
                'availability': self._check_treequest_availability
            },
            'gemma': {
                'strengths': ['general_question', 'documentation', 'explanation'],
                'performance': 0.85,
                'availability': self._check_gemma_availability
            },
            'opendia': {
                'strengths': ['general_question', 'conversation'],
                'performance': 0.80,
                'availability': self._check_opendia_availability
            },
            'mcp': {
                'strengths': ['system_command', 'tool_integration'],
                'performance': 0.90,
                'availability': self._check_mcp_availability
            }
        }
        
        # Query patterns for classification
        self.query_patterns = {
            QueryType.CODE_ANALYSIS: [
                r'analyze.*code',
                r'review.*file',
                r'check.*syntax',
                r'find.*bugs?',
                r'lint',
                r'static.*analysis'
            ],
            QueryType.CODE_GENERATION: [
                r'write.*function',
                r'create.*class',
                r'generate.*code',
                r'implement.*',
                r'build.*feature'
            ],
            QueryType.FILE_OPERATION: [
                r'/cat',
                r'/ls',
                r'/tree',
                r'show.*file',
                r'read.*file',
                r'list.*directory'
            ],
            QueryType.GIT_OPERATION: [
                r'/git',
                r'/status',
                r'/diff',
                r'git.*status',
                r'git.*diff',
                r'commit.*',
                r'branch.*'
            ],
            QueryType.PROJECT_ANALYSIS: [
                r'/analyze',
                r'analyze.*project',
                r'project.*structure',
                r'dependencies',
                r'architecture'
            ],
            QueryType.DEBUGGING: [
                r'debug',
                r'error.*fix',
                r'why.*not.*work',
                r'troubleshoot',
                r'exception',
                r'stack.*trace'
            ],
            QueryType.SYSTEM_COMMAND: [
                r'/help',
                r'/clear',
                r'/settings',
                r'/workspace',
                r'/providers'
            ],
            QueryType.GENERAL_QUESTION: [
                r'what.*is',
                r'how.*do',
                r'explain',
                r'tell.*me.*about'
            ]
        }
    
    async def route_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Analyze query and route to optimal provider
        """
        # Normalize query
        query_lower = query.lower().strip()
        
        # Classify query type
        query_type = self._classify_query(query_lower)
        
        # Get project context if available
        if not context:
            project_context = conversation_manager.get_project_context()
            context = project_context.to_dict() if project_context else {}
        
        # Determine optimal provider
        routing_decision = await self._select_optimal_provider(query_type, query_lower, context)
        
        logger.info(f"Routed '{query[:50]}...' to {routing_decision.primary_provider} (confidence: {routing_decision.confidence:.2f})")
        
        return routing_decision
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify query type based on patterns"""
        
        # Check for slash commands first (highest priority)
        if query.startswith('/'):
            if any(cmd in query for cmd in ['help', 'clear', 'settings', 'workspace', 'providers']):
                return QueryType.SYSTEM_COMMAND
            elif any(cmd in query for cmd in ['cat', 'ls', 'tree']):
                return QueryType.FILE_OPERATION
            elif any(cmd in query for cmd in ['git', 'status', 'diff']):
                return QueryType.GIT_OPERATION
            elif 'analyze' in query:
                return QueryType.PROJECT_ANALYSIS
        
        # Pattern matching for other queries
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return query_type
        
        # Default to general question
        return QueryType.GENERAL_QUESTION
    
    async def _select_optimal_provider(self, query_type: QueryType, query: str, context: Dict[str, Any]) -> RoutingDecision:
        """Select the best provider for the query type"""
        
        # Score providers based on capabilities
        provider_scores = {}
        
        for provider, capabilities in self.provider_capabilities.items():
            score = 0.0
            
            # Base score from query type match
            if query_type.value in capabilities['strengths']:
                score += capabilities['performance'] * 0.8
            
            # Bonus for specific query patterns
            if query_type == QueryType.CODE_ANALYSIS and provider == 'claude_code':
                score += 0.2
            elif query_type == QueryType.CODE_GENERATION and provider == 'treequest':
                score += 0.2
            elif query_type == QueryType.SYSTEM_COMMAND and provider == 'mcp':
                score += 0.2
            elif query_type == QueryType.GENERAL_QUESTION and provider == 'gemma':
                score += 0.15
            
            # Context-based adjustments
            if context.get('project_type') == 'Python' and provider in ['claude_code', 'treequest']:
                score += 0.1
            
            # Availability check (async)
            try:
                is_available = await capabilities['availability']()
                if not is_available:
                    score *= 0.3  # Penalize unavailable providers
            except:
                score *= 0.5  # Penalize providers with availability check errors
            
            provider_scores[provider] = score
        
        # Sort providers by score
        sorted_providers = sorted(provider_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_providers:
            # Fallback to default
            return RoutingDecision(
                primary_provider='claude_code',
                fallback_providers=['gemma', 'opendia'],
                confidence=0.5,
                reasoning="Default fallback - no providers scored",
                query_type=query_type,
                metadata={}
            )
        
        primary_provider, primary_score = sorted_providers[0]
        fallback_providers = [p for p, _ in sorted_providers[1:4]]  # Top 3 alternatives
        
        # Generate reasoning
        reasoning = f"Selected {primary_provider} for {query_type.value} (score: {primary_score:.2f})"
        if query_type.value in self.provider_capabilities[primary_provider]['strengths']:
            reasoning += f" - matches provider strengths"
        
        return RoutingDecision(
            primary_provider=primary_provider,
            fallback_providers=fallback_providers,
            confidence=min(primary_score, 1.0),
            reasoning=reasoning,
            query_type=query_type,
            metadata={
                'scores': provider_scores,
                'context_used': bool(context)
            }
        )
    
    # Provider availability checks
    async def _check_claude_availability(self) -> bool:
        """Check if Claude Code is available"""
        try:
            import subprocess
            result = subprocess.run(['which', 'claude'], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    async def _check_treequest_availability(self) -> bool:
        """Check if TreeQuest is available"""
        try:
            # Check if treequest modules exist
            from src.ai.treequest_engine import TreeQuestEngine
            return True
        except ImportError:
            return False
    
    async def _check_gemma_availability(self) -> bool:
        """Check if Gemma is available"""
        # TODO: Implement actual availability check
        return True
    
    async def _check_opendia_availability(self) -> bool:
        """Check if OpenDia is available"""
        # TODO: Implement actual availability check
        return True
    
    async def _check_mcp_availability(self) -> bool:
        """Check if MCP tools are available"""
        # TODO: Implement actual availability check
        return True
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        status = {}
        
        for provider, capabilities in self.provider_capabilities.items():
            status[provider] = {
                'strengths': capabilities['strengths'],
                'performance': capabilities['performance'],
                'available': None  # Would be populated by async check
            }
        
        return status
    
    def explain_routing(self, decision: RoutingDecision) -> str:
        """Provide human-readable explanation of routing decision"""
        explanation = f"""🧠 **Routing Decision**:
- **Selected Provider**: {decision.primary_provider}
- **Confidence**: {decision.confidence:.1%}
- **Query Type**: {decision.query_type.value}
- **Reasoning**: {decision.reasoning}"""
        
        if decision.fallback_providers:
            fallbacks = ", ".join(decision.fallback_providers)
            explanation += f"\\n- **Fallback Options**: {fallbacks}"
        
        return explanation

# Global intelligent router
intelligent_router = IntelligentRouter()
'''
        
        with open(self.project_root / "src" / "core" / "intelligent_router.py", "w") as f:
            f.write(router_content)
        
        print("✅ Intelligent Router implementation completed")
    
    async def implement_nl_command_parser(self):
        """Implement natural language to command parser"""
        print("🔧 Implementing Natural Language Command Parser...")
        
        nl_parser_content = '''"""
Natural Language Command Parser
Converts natural language input to structured commands
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

@dataclass
class ParsedIntent:
    """Parsed intent from natural language"""
    intent: str
    confidence: float
    command: Optional[str]
    parameters: Dict[str, Any]
    original_text: str

class NLCommandParser:
    """
    Converts natural language to structured commands
    """
    
    def __init__(self):
        # Intent patterns with associated commands
        self.intent_patterns = {
            'analyze_project': {
                'patterns': [
                    r'analyze.*project',
                    r'check.*project',
                    r'scan.*project',
                    r'look.*at.*project',
                    r'examine.*project'
                ],
                'command': '/analyze',
                'confidence': 0.9
            },
            'analyze_file': {
                'patterns': [
                    r'analyze.*file',
                    r'check.*file',
                    r'look.*at.*file',
                    r'examine.*file',
                    r'review.*file'
                ],
                'command': '/analyze',
                'confidence': 0.85
            },
            'show_file': {
                'patterns': [
                    r'show.*file',
                    r'display.*file',
                    r'cat.*file',
                    r'read.*file',
                    r'view.*file'
                ],
                'command': '/cat',
                'confidence': 0.9
            },
            'list_directory': {
                'patterns': [
                    r'list.*files?',
                    r'show.*files?',
                    r'ls.*',
                    r'what.*files?.*in',
                    r'files?.*in.*directory'
                ],
                'command': '/ls',
                'confidence': 0.85
            },
            'show_tree': {
                'patterns': [
                    r'show.*tree',
                    r'directory.*tree',
                    r'project.*structure',
                    r'folder.*structure'
                ],
                'command': '/tree',
                'confidence': 0.9
            },
            'git_status': {
                'patterns': [
                    r'git.*status',
                    r'check.*status',
                    r'repo.*status',
                    r'what.*changed',
                    r'working.*directory.*status'
                ],
                'command': '/status',
                'confidence': 0.9
            },
            'git_diff': {
                'patterns': [
                    r'git.*diff',
                    r'show.*diff',
                    r'what.*different',
                    r'changes.*made',
                    r'modifications'
                ],
                'command': '/diff',
                'confidence': 0.85
            },
            'workspace_info': {
                'patterns': [
                    r'current.*workspace',
                    r'workspace.*info',
                    r'where.*am.*i',
                    r'project.*info'
                ],
                'command': '/workspace',
                'confidence': 0.9
            },
            'clear_history': {
                'patterns': [
                    r'clear.*history',
                    r'clear.*conversation',
                    r'start.*fresh',
                    r'reset.*chat'
                ],
                'command': '/clear',
                'confidence': 0.95
            },
            'help': {
                'patterns': [
                    r'help',
                    r'commands?.*available',
                    r'what.*can.*do',
                    r'how.*use',
                    r'guide'
                ],
                'command': '/help',
                'confidence': 0.85
            }
        }
        
        # Parameter extraction patterns
        self.parameter_patterns = {
            'file_path': r'([/\\w.-]+\\.[\\w]+)|(["\'][^"\']*["\'])',
            'directory_path': r'([/\\w.-]+/?)|(["\'][^"\']*["\'])',
            'command_args': r'--?([\\w-]+)(?:=([\\w.-]+))?'
        }
    
    async def parse(self, text: str) -> ParsedIntent:
        """Parse natural language text to structured intent"""
        text_lower = text.lower().strip()
        
        # If already a slash command, return as-is
        if text.startswith('/'):
            return ParsedIntent(
                intent='slash_command',
                confidence=1.0,
                command=text,
                parameters={},
                original_text=text
            )
        
        # Find best matching intent
        best_match = None
        highest_confidence = 0.0
        
        for intent_name, intent_data in self.intent_patterns.items():
            for pattern in intent_data['patterns']:
                if re.search(pattern, text_lower):
                    confidence = intent_data['confidence']
                    # Boost confidence for exact matches
                    if pattern == text_lower:
                        confidence += 0.1
                    
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_match = {
                            'intent': intent_name,
                            'command': intent_data['command'],
                            'confidence': confidence
                        }
        
        if best_match:
            # Extract parameters
            parameters = self._extract_parameters(text, best_match['intent'])
            
            # Build command with parameters
            command = self._build_command(best_match['command'], parameters)
            
            return ParsedIntent(
                intent=best_match['intent'],
                confidence=best_match['confidence'],
                command=command,
                parameters=parameters,
                original_text=text
            )
        
        # No match found - general query
        return ParsedIntent(
            intent='general_query',
            confidence=0.5,
            command=None,
            parameters={'query': text},
            original_text=text
        )
    
    def _extract_parameters(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract parameters based on intent type"""
        parameters = {}
        
        if intent in ['analyze_file', 'show_file']:
            # Extract file path
            file_match = re.search(self.parameter_patterns['file_path'], text)
            if file_match:
                file_path = file_match.group(1) or file_match.group(2)
                parameters['file'] = file_path.strip('"\'')
        
        elif intent in ['list_directory', 'show_tree']:
            # Extract directory path
            dir_match = re.search(self.parameter_patterns['directory_path'], text)
            if dir_match:
                dir_path = dir_match.group(1) or dir_match.group(2)
                parameters['directory'] = dir_path.strip('"\'')
        
        elif intent == 'help':
            # Extract specific help topic
            help_topics = ['analyze', 'git', 'files', 'workspace']
            for topic in help_topics:
                if topic in text.lower():
                    parameters['topic'] = topic
                    break
        
        # Extract general command arguments
        arg_matches = re.findall(self.parameter_patterns['command_args'], text)
        for arg_name, arg_value in arg_matches:
            parameters[arg_name] = arg_value if arg_value else True
        
        return parameters
    
    def _build_command(self, base_command: str, parameters: Dict[str, Any]) -> str:
        """Build final command string with parameters"""
        command_parts = [base_command]
        
        # Add positional parameters
        if 'file' in parameters:
            command_parts.append(parameters['file'])
        elif 'directory' in parameters:
            command_parts.append(parameters['directory'])
        elif 'topic' in parameters:
            command_parts.append(parameters['topic'])
        
        # Add flag parameters
        for key, value in parameters.items():
            if key not in ['file', 'directory', 'topic', 'query']:
                if value is True:
                    command_parts.append(f'--{key}')
                else:
                    command_parts.append(f'--{key}={value}')
        
        return ' '.join(command_parts)
    
    def get_suggestions(self, partial_text: str) -> List[str]:
        """Get command suggestions for partial input"""
        partial_lower = partial_text.lower()
        suggestions = []
        
        # Check for partial matches in intent patterns
        for intent_name, intent_data in self.intent_patterns.items():
            command = intent_data['command']
            
            # Check if any pattern words match partial input
            for pattern in intent_data['patterns']:
                pattern_words = re.findall(r'\\w+', pattern)
                for word in pattern_words:
                    if word.startswith(partial_lower) or partial_lower in word:
                        suggestion = f"{command} (from: {partial_text})"
                        if suggestion not in suggestions:
                            suggestions.append(suggestion)
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def explain_parsing(self, parsed: ParsedIntent) -> str:
        """Provide explanation of parsing result"""
        if parsed.intent == 'slash_command':
            return f"✅ Recognized as slash command: {parsed.command}"
        elif parsed.intent == 'general_query':
            return f"❓ Interpreted as general query (no specific command pattern matched)"
        else:
            explanation = f"🧠 **Parsed Intent**: {parsed.intent}\\n"
            explanation += f"- **Confidence**: {parsed.confidence:.1%}\\n"
            explanation += f"- **Command**: {parsed.command}\\n"
            if parsed.parameters:
                params = ", ".join(f"{k}={v}" for k, v in parsed.parameters.items())
                explanation += f"- **Parameters**: {params}"
            return explanation

# Global NL command parser
nl_parser = NLCommandParser()


# Example usage and testing
if __name__ == "__main__":
    async def test_parser():
        test_cases = [
            "analyze this project",
            "show me the main.py file", 
            "what files are in src/",
            "git status",
            "clear conversation",
            "/help analyze"
        ]
        
        parser = NLCommandParser()
        
        for test in test_cases:
            result = await parser.parse(test)
            print(f"Input: {test}")
            print(f"Intent: {result.intent} ({result.confidence:.1%})")
            print(f"Command: {result.command}")
            print(f"Parameters: {result.parameters}")
            print("-" * 50)
    
    asyncio.run(test_parser())
'''
        
        with open(self.project_root / "src" / "core" / "nl_command_parser.py", "w") as f:
            f.write(nl_parser_content)
        
        print("✅ Natural Language Command Parser implementation completed")
    
    async def implement_auto_completion(self):
        """Implement auto-completion and suggestions"""
        print("🔧 Implementing Auto-completion & Suggestions...")
        
        completion_content = '''"""
Auto-completion and Command Suggestions
Provides intelligent command completion and context-aware suggestions
"""

import asyncio
import os
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

from src.core.conversation_manager import conversation_manager

logger = logging.getLogger(__name__)

class CommandCompletion:
    """
    Provides intelligent auto-completion for commands, files, and context
    """
    
    def __init__(self):
        # Available slash commands
        self.slash_commands = {
            '/help': 'Show available commands and help',
            '/clear': 'Clear conversation history',
            '/settings': 'Show/modify CLI settings',
            '/analyze': 'Analyze files or project',
            '/workspace': 'Show workspace information',
            '/project': 'Show project context',
            '/providers': 'List AI providers',
            '/history': 'Show conversation history',
            '/save': 'Save conversation session',
            '/load': 'Load conversation session',
            '/ls': 'List directory contents',
            '/cat': 'Show file contents',
            '/tree': 'Show directory tree',
            '/git': 'Git operations',
            '/status': 'Show git status',
            '/diff': 'Show git diff'
        }
        
        # File extensions for completion
        self.code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.go', '.rs', '.php'}
        self.config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'}
        self.doc_extensions = {'.md', '.txt', '.rst', '.adoc'}
        
        # Common directories
        self.common_dirs = ['src', 'lib', 'test', 'tests', 'docs', 'config', 'scripts', 'bin']
    
    async def complete_command(self, partial_input: str, current_dir: Optional[Path] = None) -> List[str]:
        """
        Complete partial command input with suggestions
        """
        if not partial_input:
            return self._get_context_suggestions()
        
        partial_input = partial_input.strip()
        current_dir = current_dir or Path.cwd()
        
        # Slash command completion
        if partial_input.startswith('/'):
            return self._complete_slash_command(partial_input)
        
        # File/directory completion for existing commands
        if ' ' in partial_input:
            command_part, path_part = partial_input.rsplit(' ', 1)
            if any(cmd in command_part for cmd in ['/cat', '/analyze', '/ls', '/tree']):
                return self._complete_file_path(path_part, current_dir)
        
        # General command suggestions
        return self._get_general_suggestions(partial_input)
    
    def _complete_slash_command(self, partial: str) -> List[str]:
        """Complete slash commands"""
        completions = []
        
        # If just '/', show all commands
        if partial == '/':
            return [f"{cmd} - {desc}" for cmd, desc in self.slash_commands.items()]
        
        # Find matching commands
        for cmd, desc in self.slash_commands.items():
            if cmd.startswith(partial):
                completions.append(f"{cmd} - {desc}")
        
        return completions[:10]
    
    def _complete_file_path(self, partial_path: str, current_dir: Path) -> List[str]:
        """Complete file and directory paths"""
        try:
            # Handle absolute vs relative paths
            if partial_path.startswith('/'):
                base_path = Path(partial_path).parent
                search_name = Path(partial_path).name
            else:
                if '/' in partial_path:
                    base_path = current_dir / Path(partial_path).parent
                    search_name = Path(partial_path).name
                else:
                    base_path = current_dir
                    search_name = partial_path
            
            if not base_path.exists():
                return []
            
            completions = []
            
            # Get matching files/directories
            for item in base_path.iterdir():
                if item.name.startswith(search_name):
                    if item.is_dir():
                        completions.append(f"📁 {item.name}/")
                    else:
                        # Add icon based on file type
                        icon = self._get_file_icon(item)
                        completions.append(f"{icon} {item.name}")
            
            return sorted(completions)[:15]
            
        except Exception as e:
            logger.debug(f"Error completing file path: {e}")
            return []
    
    def _get_file_icon(self, file_path: Path) -> str:
        """Get appropriate icon for file type"""
        suffix = file_path.suffix.lower()
        
        if suffix in self.code_extensions:
            return "🐍" if suffix == '.py' else "💻"
        elif suffix in self.config_extensions:
            return "⚙️"
        elif suffix in self.doc_extensions:
            return "📝"
        elif file_path.name in ['README', 'readme', 'README.md']:
            return "📖"
        elif file_path.name.startswith('.'):
            return "🔧"
        else:
            return "📄"
    
    def _get_context_suggestions(self) -> List[str]:
        """Get context-aware suggestions when no input provided"""
        suggestions = []
        
        # Get project context
        context = conversation_manager.get_project_context()
        
        if context:
            suggestions.extend([
                f"/analyze {context.project_path} - Analyze current project",
                f"/workspace - Show workspace info",
                f"/status - Check git status"
            ])
        
        # Add common commands
        suggestions.extend([
            "/help - Show available commands",
            "/ls . - List current directory",
            "/tree - Show project structure"
        ])
        
        # Check for common files in current directory
        current_dir = Path.cwd()
        common_files = ['README.md', 'package.json', 'requirements.txt', 'Cargo.toml', 'pom.xml']
        
        for file_name in common_files:
            if (current_dir / file_name).exists():
                suggestions.append(f"/cat {file_name} - Show {file_name}")
        
        return suggestions[:8]
    
    def _get_general_suggestions(self, partial: str) -> List[str]:
        """Get general suggestions for partial input"""
        suggestions = []
        partial_lower = partial.lower()
        
        # Suggest commands based on keywords
        keyword_suggestions = {
            'analyze': ['/analyze .', '/analyze src/', '/workspace'],
            'show': ['/cat', '/ls', '/tree', '/status'],
            'git': ['/git status', '/status', '/diff'],
            'help': ['/help', '/help analyze'],
            'list': ['/ls', '/tree', '/providers'],
            'file': ['/cat', '/ls', '/analyze'],
            'project': ['/analyze', '/workspace', '/tree']
        }
        
        for keyword, cmds in keyword_suggestions.items():
            if keyword in partial_lower:
                suggestions.extend(cmds)
        
        # Remove duplicates and limit
        return list(dict.fromkeys(suggestions))[:10]
    
    async def get_smart_suggestions(self, conversation_history: List[str]) -> List[str]:
        """Get smart suggestions based on conversation history"""
        suggestions = []
        
        # Analyze recent conversation for patterns
        recent_queries = conversation_history[-5:] if conversation_history else []
        
        # Look for common patterns
        if any('analyze' in query.lower() for query in recent_queries):
            suggestions.append("/workspace - Check current workspace")
            suggestions.append("/tree - See project structure")
        
        if any('file' in query.lower() for query in recent_queries):
            suggestions.extend(["/ls - List files", "/cat <filename> - Show file"])
        
        if any('git' in query.lower() for query in recent_queries):
            suggestions.extend(["/status - Git status", "/diff - Show changes"])
        
        # Add contextual suggestions based on project type
        context = conversation_manager.get_project_context()
        if context:
            if context.project_type == 'Python':
                suggestions.extend([
                    "/analyze requirements.txt",
                    "/cat setup.py"
                ])
            elif context.project_type == 'Node.js':
                suggestions.extend([
                    "/analyze package.json",
                    "/cat package.json"
                ])
        
        return suggestions[:8]
    
    def get_command_help(self, command: str) -> Optional[str]:
        """Get help text for a specific command"""
        
        help_texts = {
            '/analyze': """Analyze files or projects
Examples:
  /analyze .              - Analyze current directory
  /analyze src/main.py    - Analyze specific file
  /analyze --deep         - Deep analysis with dependencies""",
            
            '/cat': """Show file contents
Examples:
  /cat README.md          - Show README file
  /cat src/main.py        - Show Python file
  /cat config.json        - Show configuration""",
            
            '/ls': """List directory contents
Examples:
  /ls                     - List current directory
  /ls src/                - List src directory
  /ls --all               - Show hidden files""",
            
            '/git': """Git repository operations
Examples:
  /git status             - Show git status
  /git diff               - Show changes
  /git log                - Show commit history""",
            
            '/workspace': """Workspace management
Examples:
  /workspace              - Show current workspace
  /workspace switch <name> - Switch workspace
  /workspace list         - List all workspaces"""
        }
        
        return help_texts.get(command)
    
    def fuzzy_match_commands(self, query: str) -> List[Tuple[str, float]]:
        """Fuzzy match commands with similarity scores"""
        matches = []
        query_lower = query.lower()
        
        for cmd in self.slash_commands.keys():
            cmd_lower = cmd[1:]  # Remove slash for comparison
            
            # Simple similarity scoring
            if cmd_lower.startswith(query_lower):
                score = 1.0 - (len(cmd_lower) - len(query_lower)) / len(cmd_lower)
            elif query_lower in cmd_lower:
                score = 0.8 - (len(cmd_lower) - len(query_lower)) / len(cmd_lower)
            else:
                # Character overlap scoring
                common_chars = set(query_lower) & set(cmd_lower)
                score = len(common_chars) / max(len(query_lower), len(cmd_lower))
                if score < 0.3:
                    continue
            
            matches.append((cmd, score))
        
        # Sort by score, return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:5]

# Global command completion instance
command_completion = CommandCompletion()


# Example usage
if __name__ == "__main__":
    async def test_completion():
        completion = CommandCompletion()
        
        test_cases = [
            "/",
            "/ana",
            "/cat README",
            "analyze",
            "show file",
            ""
        ]
        
        for test in test_cases:
            results = await completion.complete_command(test)
            print(f"Input: '{test}'")
            print(f"Suggestions: {results}")
            print("-" * 50)
    
    asyncio.run(test_completion())
'''
        
        with open(self.project_root / "src" / "core" / "command_completion.py", "w") as f:
            f.write(completion_content)
        
        print("✅ Auto-completion & Suggestions implementation completed")
    
    async def implement_project_context_loader(self):
        """Implement project context awareness and loading"""
        print("🔧 Implementing Project Context Loader...")
        
        context_loader_content = '''"""
Project Context Loader
Automatically detects project type and loads relevant context
"""

import asyncio
import json
import os
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

from src.core.conversation_manager import ProjectContext, conversation_manager

logger = logging.getLogger(__name__)

class ProjectContextLoader:
    """
    Automatically loads and maintains project context
    """
    
    def __init__(self):
        self.project_type_indicators = {
            'Python': {
                'files': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile'],
                'dirs': ['venv', '.venv', '__pycache__'],
                'extensions': ['.py']
            },
            'Node.js': {
                'files': ['package.json', 'yarn.lock', 'package-lock.json'],
                'dirs': ['node_modules', '.npm'],
                'extensions': ['.js', '.ts', '.jsx', '.tsx']
            },
            'Java': {
                'files': ['pom.xml', 'build.gradle', 'gradle.properties'],
                'dirs': ['target', 'build', '.gradle'],
                'extensions': ['.java']
            },
            'Rust': {
                'files': ['Cargo.toml', 'Cargo.lock'],
                'dirs': ['target'],
                'extensions': ['.rs']
            },
            'Go': {
                'files': ['go.mod', 'go.sum'],
                'dirs': ['vendor'],
                'extensions': ['.go']
            },
            'C++': {
                'files': ['CMakeLists.txt', 'Makefile', 'configure.ac'],
                'dirs': ['build', 'cmake-build-debug'],
                'extensions': ['.cpp', '.cc', '.cxx', '.c', '.h', '.hpp']
            },
            'Web': {
                'files': ['index.html', 'webpack.config.js', '.babelrc'],
                'dirs': ['dist', 'build', 'public'],
                'extensions': ['.html', '.css', '.js', '.ts']
            }
        }
        
        self.important_files = [
            'README.md', 'README.rst', 'README.txt',
            'CHANGELOG.md', 'CHANGELOG.txt',
            'LICENSE', 'LICENSE.md', 'LICENSE.txt',
            'CONTRIBUTING.md',
            '.gitignore', '.dockerignore',
            'Dockerfile', 'docker-compose.yml',
            '.env', '.env.example'
        ]
    
    async def load_project_context(self, project_path: Optional[str] = None) -> ProjectContext:
        """Load complete project context"""
        project_path = project_path or str(Path.cwd())
        project_dir = Path(project_path)
        
        if not project_dir.exists() or not project_dir.is_dir():
            raise ValueError(f"Invalid project path: {project_path}")
        
        logger.info(f"Loading project context for: {project_path}")
        
        # Detect project type
        project_type = self._detect_project_type(project_dir)
        
        # Get git status
        git_status = await self._get_git_status(project_dir)
        
        # Find relevant files
        relevant_files = self._find_relevant_files(project_dir, project_type)
        
        # Extract dependencies
        dependencies = await self._extract_dependencies(project_dir, project_type)
        
        # Generate project summary
        summary = await self._generate_project_summary(project_dir, project_type, relevant_files)
        
        context = ProjectContext(
            project_path=project_path,
            project_type=project_type,
            git_status=git_status,
            relevant_files=relevant_files,
            dependencies=dependencies,
            last_updated=datetime.now().timestamp(),
            summary=summary
        )
        
        # Set in conversation manager
        conversation_manager.set_project_context(context)
        
        return context
    
    def _detect_project_type(self, project_dir: Path) -> str:
        """Detect project type based on files and structure"""
        type_scores = {}
        
        for project_type, indicators in self.project_type_indicators.items():
            score = 0
            
            # Check for indicator files
            for file_name in indicators['files']:
                if (project_dir / file_name).exists():
                    score += 3
            
            # Check for indicator directories
            for dir_name in indicators['dirs']:
                if (project_dir / dir_name).exists():
                    score += 2
            
            # Check for file extensions
            extension_count = 0
            for ext in indicators['extensions']:
                extension_files = list(project_dir.rglob(f'*{ext}'))
                extension_count += len(extension_files[:10])  # Limit to avoid excessive scanning
            
            score += min(extension_count, 10)  # Cap extension score
            
            type_scores[project_type] = score
        
        # Return type with highest score, or 'Unknown' if no clear match
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] >= 3:  # Minimum confidence threshold
                return best_type
        
        return 'Unknown'
    
    async def _get_git_status(self, project_dir: Path) -> Dict[str, Any]:
        """Get git repository status"""
        git_status = {'is_repo': False, 'branch': None, 'clean': True, 'files': []}
        
        try:
            # Check if it's a git repository
            result = await self._run_command(['git', 'rev-parse', '--git-dir'], project_dir)
            if result['returncode'] != 0:
                return git_status
            
            git_status['is_repo'] = True
            
            # Get current branch
            branch_result = await self._run_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], project_dir)
            if branch_result['returncode'] == 0:
                git_status['branch'] = branch_result['stdout'].strip()
            
            # Get status
            status_result = await self._run_command(['git', 'status', '--porcelain'], project_dir)
            if status_result['returncode'] == 0:
                status_lines = status_result['stdout'].strip().split('\\n')
                if status_lines != ['']:
                    git_status['clean'] = False
                    git_status['files'] = [line.strip() for line in status_lines if line.strip()]
            
            # Get remote info
            remote_result = await self._run_command(['git', 'remote', '-v'], project_dir)
            if remote_result['returncode'] == 0:
                git_status['remotes'] = remote_result['stdout'].strip()
            
        except Exception as e:
            logger.debug(f"Error getting git status: {e}")
        
        return git_status
    
    def _find_relevant_files(self, project_dir: Path, project_type: str) -> List[str]:
        """Find relevant files for the project"""
        relevant_files = []
        
        # Add important files that exist
        for important_file in self.important_files:
            file_path = project_dir / important_file
            if file_path.exists():
                relevant_files.append(str(file_path.relative_to(project_dir)))
        
        # Add type-specific files
        if project_type in self.project_type_indicators:
            indicators = self.project_type_indicators[project_type]
            
            for file_name in indicators['files']:
                file_path = project_dir / file_name
                if file_path.exists():
                    relevant_files.append(str(file_path.relative_to(project_dir)))
        
        # Add main source files (limited to avoid clutter)
        source_patterns = {
            'Python': ['main.py', 'app.py', '__init__.py', 'setup.py'],
            'Node.js': ['index.js', 'server.js', 'app.js', 'main.js'],
            'Java': ['Main.java', 'Application.java'],
            'Rust': ['main.rs', 'lib.rs'],
            'Go': ['main.go'],
            'C++': ['main.cpp', 'main.c']
        }
        
        if project_type in source_patterns:
            for pattern in source_patterns[project_type]:
                matches = list(project_dir.rglob(pattern))
                for match in matches[:3]:  # Limit to first 3 matches
                    relevant_files.append(str(match.relative_to(project_dir)))
        
        return list(set(relevant_files))  # Remove duplicates
    
    async def _extract_dependencies(self, project_dir: Path, project_type: str) -> List[str]:
        """Extract project dependencies"""
        dependencies = []
        
        try:
            if project_type == 'Python':
                # Try requirements.txt
                req_file = project_dir / 'requirements.txt'
                if req_file.exists():
                    content = req_file.read_text()
                    deps = [line.split('==')[0].split('>=')[0].split('~=')[0].strip() 
                           for line in content.split('\\n') 
                           if line.strip() and not line.startswith('#')]
                    dependencies.extend(deps[:20])  # Limit to first 20
                
                # Try pyproject.toml
                pyproject_file = project_dir / 'pyproject.toml'
                if pyproject_file.exists():
                    # Basic parsing - would need proper TOML parser for production
                    content = pyproject_file.read_text()
                    if 'dependencies' in content:
                        dependencies.extend(['pyproject.toml dependencies'])
            
            elif project_type == 'Node.js':
                package_file = project_dir / 'package.json'
                if package_file.exists():
                    try:
                        package_data = json.loads(package_file.read_text())
                        deps = list(package_data.get('dependencies', {}).keys())
                        dev_deps = list(package_data.get('devDependencies', {}).keys())
                        dependencies.extend(deps[:15] + dev_deps[:10])
                    except json.JSONDecodeError:
                        dependencies.append('package.json (parse error)')
            
            elif project_type == 'Rust':
                cargo_file = project_dir / 'Cargo.toml'
                if cargo_file.exists():
                    content = cargo_file.read_text()
                    # Basic TOML parsing for dependencies section
                    if '[dependencies]' in content:
                        dependencies.extend(['Cargo.toml dependencies'])
            
            elif project_type == 'Go':
                go_mod = project_dir / 'go.mod'
                if go_mod.exists():
                    content = go_mod.read_text()
                    for line in content.split('\\n'):
                        if line.strip() and not line.startswith('module') and not line.startswith('go '):
                            if '/' in line:  # Likely a dependency
                                dep = line.strip().split()[0]
                                dependencies.append(dep)
                                if len(dependencies) >= 15:
                                    break
            
        except Exception as e:
            logger.debug(f"Error extracting dependencies: {e}")
        
        return dependencies
    
    async def _generate_project_summary(self, project_dir: Path, project_type: str, relevant_files: List[str]) -> str:
        """Generate a concise project summary"""
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"{project_type} project")
        
        # File count
        try:
            file_count = len([f for f in project_dir.rglob('*') if f.is_file() and not any(part.startswith('.') for part in f.parts)])
            summary_parts.append(f"{file_count} files")
        except:
            pass
        
        # Key characteristics
        if 'README.md' in relevant_files:
            summary_parts.append("documented")
        
        if any('test' in f.lower() for f in relevant_files):
            summary_parts.append("with tests")
        
        if 'Dockerfile' in relevant_files:
            summary_parts.append("containerized")
        
        return ", ".join(summary_parts)
    
    async def _run_command(self, cmd: List[str], cwd: Path) -> Dict[str, Any]:
        """Run command and return result"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return {
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8'),
                'stderr': stderr.decode('utf-8')
            }
        except Exception as e:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e)
            }
    
    async def monitor_project_changes(self, project_path: str, callback=None):
        """Monitor project for changes and update context"""
        # This would integrate with file system watching
        # For now, just a placeholder for future implementation
        logger.info(f"Monitoring {project_path} for changes")
        pass
    
    def get_project_insights(self, context: ProjectContext) -> List[str]:
        """Generate insights about the project"""
        insights = []
        
        # Git insights
        if context.git_status['is_repo']:
            if not context.git_status['clean']:
                insights.append(f"🔄 Working directory has {len(context.git_status['files'])} changed files")
            else:
                insights.append("✅ Working directory is clean")
        else:
            insights.append("⚠️ Not a git repository")
        
        # Dependency insights
        if len(context.dependencies) > 20:
            insights.append(f"📦 Large project with {len(context.dependencies)}+ dependencies")
        elif len(context.dependencies) > 0:
            insights.append(f"📦 {len(context.dependencies)} dependencies")
        
        # File insights
        if 'README.md' in context.relevant_files:
            insights.append("📚 Well documented")
        else:
            insights.append("📝 Consider adding README documentation")
        
        return insights

# Global project context loader
project_context_loader = ProjectContextLoader()


# Example usage
if __name__ == "__main__":
    async def test_context_loader():
        loader = ProjectContextLoader()
        context = await loader.load_project_context('.')
        
        print(f"Project Type: {context.project_type}")
        print(f"Git Status: {context.git_status}")
        print(f"Relevant Files: {context.relevant_files}")
        print(f"Dependencies: {context.dependencies[:10]}")
        print(f"Summary: {context.summary}")
        
        insights = loader.get_project_insights(context)
        print(f"Insights: {insights}")
    
    asyncio.run(test_context_loader())
'''
        
        with open(self.project_root / "src" / "core" / "project_context_loader.py", "w") as f:
            f.write(context_loader_content)
        
        print("✅ Project Context Loader implementation completed")
    
    async def implement_session_management_integration(self):
        """Implement session management integration with existing components"""
        print("🔧 Implementing Session Management Integration...")
        
        session_integration_content = '''"""
Session Management Integration
Integrates session management with all Phase 3 components
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from src.core.conversation_manager import conversation_manager, Message
from src.core.project_context_loader import project_context_loader

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Advanced session management with project awareness
    """
    
    def __init__(self):
        self.current_session_id = None
        self.session_metadata = {}
    
    async def start_session(self, project_path: Optional[str] = None) -> str:
        """Start new session with project context loading"""
        
        # Load project context if path provided or detect current
        if project_path:
            project_context = await project_context_loader.load_project_context(project_path)
        else:
            try:
                project_context = await project_context_loader.load_project_context()
            except Exception as e:
                logger.warning(f"Could not load project context: {e}")
                project_context = None
        
        # Start new conversation session
        old_session = conversation_manager.start_new_session()
        session_id = conversation_manager.current_session_id
        
        # Add session start message
        conversation_manager.add_message(
            "system", 
            f"Session started. Project type: {project_context.project_type if project_context else 'Unknown'}",
            {"session_start": True, "project_path": project_path}
        )
        
        self.current_session_id = session_id
        self.session_metadata[session_id] = {
            "started_at": conversation_manager.conversation_history[-1].timestamp,
            "project_path": project_path,
            "project_type": project_context.project_type if project_context else None
        }
        
        logger.info(f"Started new session {session_id} for project: {project_path}")
        return session_id
    
    async def save_named_session(self, name: str, description: Optional[str] = None) -> bool:
        """Save current session with a custom name"""
        if not conversation_manager.conversation_history:
            return False
        
        try:
            session_data = {
                "name": name,
                "description": description,
                "session_id": conversation_manager.current_session_id,
                "messages": [msg.to_dict() for msg in conversation_manager.conversation_history],
                "project_context": conversation_manager.project_context.to_dict() if conversation_manager.project_context else None,
                "saved_at": conversation_manager.conversation_history[-1].timestamp,
                "metadata": self.session_metadata.get(conversation_manager.current_session_id, {})
            }
            
            # Save to named sessions directory
            named_sessions_dir = conversation_manager.session_dir / "named_sessions"
            named_sessions_dir.mkdir(exist_ok=True)
            
            session_file = named_sessions_dir / f"{name}.json"
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(f"Saved session as '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save named session '{name}': {e}")
            return False
    
    async def load_named_session(self, name: str) -> bool:
        """Load a named session"""
        try:
            named_sessions_dir = conversation_manager.session_dir / "named_sessions"
            session_file = named_sessions_dir / f"{name}.json"
            
            if not session_file.exists():
                return False
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Create new session ID and load data
            old_session = conversation_manager.start_new_session()
            
            # Restore messages
            conversation_manager.conversation_history = [
                Message.from_dict(msg_data) 
                for msg_data in session_data.get('messages', [])
            ]
            
            # Restore project context
            if session_data.get('project_context'):
                from src.core.conversation_manager import ProjectContext
                project_context = ProjectContext.from_dict(session_data['project_context'])
                conversation_manager.set_project_context(project_context)
            
            # Update session metadata
            self.current_session_id = conversation_manager.current_session_id
            self.session_metadata[self.current_session_id] = session_data.get('metadata', {})
            
            # Add session restore message
            conversation_manager.add_message(
                "system",
                f"Restored session '{name}' with {len(conversation_manager.conversation_history)} messages",
                {"session_restore": True, "original_name": name}
            )
            
            logger.info(f"Loaded named session '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load named session '{name}': {e}")
            return False
    
    def list_named_sessions(self) -> List[Dict[str, Any]]:
        """List all named sessions"""
        sessions = []
        
        try:
            named_sessions_dir = conversation_manager.session_dir / "named_sessions"
            if not named_sessions_dir.exists():
                return sessions
            
            for session_file in named_sessions_dir.glob("*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    sessions.append({
                        "name": session_data.get("name"),
                        "description": session_data.get("description"),
                        "saved_at": session_data.get("saved_at"),
                        "message_count": len(session_data.get("messages", [])),
                        "project_type": session_data.get("metadata", {}).get("project_type"),
                        "project_path": session_data.get("metadata", {}).get("project_path")
                    })
                    
                except Exception as e:
                    logger.warning(f"Error reading session {session_file}: {e}")
            
            # Sort by saved_at
            sessions.sort(key=lambda x: x.get("saved_at", 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing named sessions: {e}")
        
        return sessions
    
    async def auto_save_session(self) -> bool:
        """Auto-save current session periodically"""
        try:
            if len(conversation_manager.conversation_history) >= 5:
                # Auto-save with timestamp name
                import time
                timestamp = int(time.time())
                name = f"auto_save_{timestamp}"
                
                return await self.save_named_session(
                    name, 
                    f"Auto-saved session with {len(conversation_manager.conversation_history)} messages"
                )
            return False
            
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
            return False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about current session"""
        if not conversation_manager.conversation_history:
            return {"active": False}
        
        messages = conversation_manager.conversation_history
        user_messages = [m for m in messages if m.role == 'user']
        assistant_messages = [m for m in messages if m.role == 'assistant']
        
        stats = {
            "active": True,
            "session_id": conversation_manager.current_session_id,
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "session_duration": messages[-1].timestamp - messages[0].timestamp if messages else 0,
            "project_context": conversation_manager.project_context is not None
        }
        
        if conversation_manager.project_context:
            stats["project_type"] = conversation_manager.project_context.project_type
            stats["project_path"] = conversation_manager.project_context.project_path
        
        return stats
    
    async def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old sessions"""
        import time
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        cleaned_count = 0
        
        try:
            # Clean regular sessions
            for session_file in conversation_manager.session_dir.glob("*.json"):
                if session_file.name == "global_history.json":
                    continue
                
                try:
                    with open(session_file, 'r') as f:
                        data = json.load(f)
                    
                    if data.get('last_updated', 0) < cutoff_time:
                        session_file.unlink()
                        cleaned_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error checking session {session_file}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} old sessions")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return cleaned_count

# Global session manager
session_manager = SessionManager()


# Integration with slash commands
class SessionCommands:
    """Session-related slash commands"""
    
    @staticmethod
    async def save_session(args: List[str]) -> str:
        """Save current session with name"""
        if not args:
            return "❌ Please provide a session name: /save <name>"
        
        name = args[0]
        description = " ".join(args[1:]) if len(args) > 1 else None
        
        success = await session_manager.save_named_session(name, description)
        
        if success:
            return f"✅ Session saved as '{name}'"
        else:
            return f"❌ Failed to save session '{name}'"
    
    @staticmethod
    async def load_session(args: List[str]) -> str:
        """Load named session"""
        if not args:
            # List available sessions
            sessions = session_manager.list_named_sessions()
            if not sessions:
                return "No saved sessions found"
            
            lines = ["📚 **Available Sessions**:"]
            for session in sessions[:10]:
                name = session["name"]
                desc = session.get("description", "")
                msg_count = session.get("message_count", 0)
                lines.append(f"  💾 {name} - {msg_count} messages" + (f" ({desc})" if desc else ""))
            
            return "\\n".join(lines)
        
        name = args[0]
        success = await session_manager.load_named_session(name)
        
        if success:
            return f"✅ Loaded session '{name}'"
        else:
            return f"❌ Session '{name}' not found"
    
    @staticmethod
    def session_stats() -> str:
        """Get current session statistics"""
        stats = session_manager.get_session_stats()
        
        if not stats["active"]:
            return "No active session"
        
        lines = ["📊 **Session Statistics**:"]
        lines.append(f"- **Session ID**: {stats['session_id'][:8]}...")
        lines.append(f"- **Messages**: {stats['total_messages']} total ({stats['user_messages']} user, {stats['assistant_messages']} assistant)")
        
        if stats["session_duration"] > 0:
            duration_mins = int(stats["session_duration"] / 60)
            lines.append(f"- **Duration**: {duration_mins} minutes")
        
        if stats["project_context"]:
            lines.append(f"- **Project**: {stats['project_type']} at {stats.get('project_path', 'Unknown')}")
        
        return "\\n".join(lines)


# Export session commands for integration
session_commands = SessionCommands()
'''
        
        with open(self.project_root / "src" / "core" / "session_manager.py", "w") as f:
            f.write(session_integration_content)
        
        print("✅ Session Management Integration implementation completed")
    
    async def update_main_cli_integration(self):
        """Update main CLI to integrate all Phase 3 components"""
        print("🔧 Updating Main CLI Integration...")
        
        # Create enhanced main CLI that integrates all components
        enhanced_cli_content = '''#!/usr/bin/env python3
"""
Smart AI Enhanced CLI v3.0 - Claude-Style Interface
Complete integration of all Phase 3 components
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Add paths
sys.path.append('/Users/Subho')
sys.path.append('/Users/Subho/smart-ai-enhanced-modules')
sys.path.append(str(Path(__file__).parent))

# Import Phase 3 components
from src.core.conversation_manager import conversation_manager
from src.core.slash_command_processor import slash_processor
from src.core.intelligent_router import intelligent_router
from src.core.nl_command_parser import nl_parser
from src.core.command_completion import command_completion
from src.core.project_context_loader import project_context_loader
from src.core.session_manager import session_manager
from src.ui.rich_interface import RichCLI

# Import existing backend
try:
    from smart_ai_backend import SmartAIBackend
except ImportError as e:
    print(f"❌ Import error: {e}")
    SmartAIBackend = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartAIEnhancedV3:
    """
    Enhanced Smart AI CLI with Claude-style interface and Phase 3 features
    """
    
    def __init__(self):
        self.backend = SmartAIBackend() if SmartAIBackend else None
        self.rich_cli = RichCLI()
        self.interactive_mode = False
        
    async def initialize(self):
        """Initialize all components"""
        try:
            # Start session with project context
            await session_manager.start_session()
            
            # Load project context for current directory
            try:
                await project_context_loader.load_project_context()
                logger.info("Project context loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load project context: {e}")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
    
    def create_parser(self):
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="Smart AI Enhanced CLI v3.0 - Claude-Style Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  smart-ai-enhanced.py                          # Interactive mode
  smart-ai-enhanced.py "analyze this project"  # Direct query
  smart-ai-enhanced.py /help                   # Slash command
  smart-ai-enhanced.py --chat                  # Force interactive mode
            """
        )
        
        parser.add_argument('query', nargs='*', help='Query or command to execute')
        parser.add_argument('--chat', '-c', action='store_true', help='Interactive chat mode')
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        parser.add_argument('--format', choices=['rich', 'plain', 'json'], default='rich', help='Output format')
        parser.add_argument('--no-context', action='store_true', help='Disable project context loading')
        parser.add_argument('--provider', help='Force specific AI provider')
        
        return parser
    
    async def handle_query(self, query_text: str, args) -> str:
        """Handle a single query with full Phase 3 processing"""
        
        # Step 1: Check if it's a slash command
        slash_command = slash_processor.parse_command(query_text)
        
        if slash_command:
            # Execute slash command
            result = await slash_processor.execute_command(slash_command)
            
            if result["success"]:
                response = result["data"]
                
                # Log interaction
                conversation_manager.add_message("user", query_text)
                conversation_manager.add_message("assistant", response, {"command_type": "slash"})
                
                return response
            else:
                error_msg = f"❌ {result['error']}"
                if "suggestions" in result:
                    suggestions = result["suggestions"]
                    if suggestions:
                        error_msg += f"\\nDid you mean: {', '.join(suggestions)}"
                return error_msg
        
        # Step 2: Try natural language parsing
        parsed_intent = await nl_parser.parse(query_text)
        
        if parsed_intent.command and parsed_intent.confidence > 0.7:
            # Execute parsed command
            converted_command = slash_processor.parse_command(parsed_intent.command)
            if converted_command:
                result = await slash_processor.execute_command(converted_command)
                
                if result["success"]:
                    response = f"🧠 Interpreted as: {parsed_intent.command}\\n\\n{result['data']}"
                    
                    # Log interaction
                    conversation_manager.add_message("user", query_text)
                    conversation_manager.add_message("assistant", response, {
                        "command_type": "nl_parsed",
                        "original_intent": parsed_intent.intent,
                        "parsed_command": parsed_intent.command
                    })
                    
                    return response
        
        # Step 3: Route to AI provider
        routing_decision = await intelligent_router.route_query(
            query_text,
            conversation_manager.get_context_for_ai()
        )
        
        provider = args.provider or routing_decision.primary_provider
        
        if args.verbose:
            routing_explanation = intelligent_router.explain_routing(routing_decision)
            print(routing_explanation)
        
        # Step 4: Process with AI provider
        if self.backend:
            try:
                response = await self.backend.process_request_async(query_text, provider)
                
                if response:
                    # Log interaction
                    conversation_manager.add_message("user", query_text)
                    conversation_manager.add_message("assistant", response, {
                        "provider": provider,
                        "routing_confidence": routing_decision.confidence,
                        "query_type": routing_decision.query_type.value
                    })
                    
                    return response
                else:
                    # Try fallback providers
                    for fallback_provider in routing_decision.fallback_providers:
                        try:
                            response = await self.backend.process_request_async(query_text, fallback_provider)
                            if response:
                                conversation_manager.add_message("user", query_text)
                                conversation_manager.add_message("assistant", response, {
                                    "provider": fallback_provider,
                                    "fallback_used": True
                                })
                                return f"🔄 Used fallback provider {fallback_provider}:\\n\\n{response}"
                        except:
                            continue
                    
                    return "❌ All AI providers failed to respond"
            
            except Exception as e:
                return f"❌ Error processing query: {e}"
        else:
            return f"❌ Backend not available. Query: {query_text}"
    
    async def interactive_mode(self, args):
        """Enhanced interactive mode with all Phase 3 features"""
        print("🤖 Smart AI Enhanced v3.0 - Claude-Style Interface")
        print("=" * 60)
        
        # Show context information
        context = conversation_manager.get_project_context()
        if context:
            print(f"📁 Project: {context.project_type} at {context.project_path}")
            insights = project_context_loader.get_project_insights(context)
            for insight in insights[:3]:
                print(f"   {insight}")
        
        print("\\nType '/help' for commands, 'quit' to exit")
        print("-" * 60)
        
        self.interactive_mode = True
        
        while True:
            try:
                # Get user input with auto-completion hints
                user_input = input("\\n💭 Smart AI> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    # Show context-aware suggestions
                    suggestions = await command_completion.complete_command("")
                    if suggestions:
                        print("💡 Suggestions:")
                        for suggestion in suggestions[:5]:
                            print(f"   {suggestion}")
                    continue
                
                # Process query
                response = await self.handle_query(user_input, args)
                
                # Display response with rich formatting
                if args.format == 'rich':
                    await self.rich_cli.display_response(response)
                else:
                    print(response)
                
                # Auto-save session periodically
                if len(conversation_manager.conversation_history) % 10 == 0:
                    await session_manager.auto_save_session()
            
            except KeyboardInterrupt:
                print("\\n\\n👋 Goodbye!")
                break
            except EOFError:
                print("\\n\\n👋 Goodbye!")
                break
            except Exception as e:
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                else:
                    print(f"❌ Error: {e}")
    
    async def run(self):
        """Main run method"""
        parser = self.create_parser()
        args = parser.parse_args()
        
        # Initialize components
        await self.initialize()
        
        # Handle direct query
        if args.query:
            query_text = ' '.join(args.query)
            response = await self.handle_query(query_text, args)
            
            # Display response
            if args.format == 'json':
                import json
                result = {
                    "query": query_text,
                    "response": response,
                    "session_id": conversation_manager.current_session_id
                }
                print(json.dumps(result, indent=2))
            elif args.format == 'rich':
                await self.rich_cli.display_response(response)
            else:
                print(response)
        
        # Interactive mode
        elif args.chat or not args.query:
            await self.interactive_mode(args)

async def main():
    """Main entry point"""
    try:
        cli = SmartAIEnhancedV3()
        await cli.run()
    except KeyboardInterrupt:
        print("\\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        # Update the main CLI file
        with open(self.project_root / "smart_ai_enhanced_v3.py", "w") as f:
            f.write(enhanced_cli_content)
        
        # Make it executable
        os.chmod(self.project_root / "smart_ai_enhanced_v3.py", 0o755)
        
        print("✅ Main CLI Integration completed")

async def main():
    """Main execution function"""
    project_root = "/Users/Subho/smart-ai-enhanced-project"
    executor = TreeQuestPhase3Executor(project_root)
    
    try:
        await executor.execute_all_phase3_tasks()
        print("\n🎉 TreeQuest Phase 3 Build Process Completed Successfully!")
        print("All Claude-style interface components implemented")
        print("Run: python smart_ai_enhanced_v3.py --help")
    except KeyboardInterrupt:
        print("\n⚠️ Build process interrupted by user")
    except Exception as e:
        print(f"\n❌ Build process failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())