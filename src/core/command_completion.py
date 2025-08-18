"""
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
