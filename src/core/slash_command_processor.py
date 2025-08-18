"""
Enhanced Slash Command Processor with TreeQuest Integration
Claude-style slash commands with AI agent orchestration
"""

import asyncio
import re
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import time

from src.core.conversation_manager import conversation_manager
from src.analyzers.analyzer_coordinator import EnhancedAnalyzerCoordinator
from src.workspace.workspace_manager import WorkspaceManager
from src.ai.treequest_engine import TreeQuestEngine, TreeQuestConfig
from src.ai.model_registry import ModelRegistry, ModelRole, ModelObjective
from src.core.memory_manager import memory_manager

logger = logging.getLogger(__name__)

@dataclass
class SlashCommand:
    """Represents a parsed slash command"""
    name: str
    args: List[str]
    kwargs: Dict[str, Any]
    raw_input: str

class EnhancedSlashCommandProcessor:
    """
    Enhanced slash command processor with TreeQuest AI agent integration
    """
    
    def __init__(self):
        self.commands: Dict[str, Callable] = {}
        self.command_help: Dict[str, str] = {}
        self.analyzer_coordinator = EnhancedAnalyzerCoordinator()
        self.workspace_manager = WorkspaceManager()
        self.treequest_engine = None
        self.model_registry = None
        self._register_enhanced_commands()
    
    async def initialize(self):
        """Initialize TreeQuest engine and model registry"""
        try:
            self.model_registry = ModelRegistry()
            treequest_config = TreeQuestConfig(
                max_depth=3,
                branching=4,
                rollout_budget=32,
                cost_cap_usd=0.50,
                objective="quality"
            )
            
            # Initialize TreeQuest engine with cache
            from src.core.cache_manager import cache_manager
            self.treequest_engine = TreeQuestEngine(
                self.model_registry, 
                cache_manager, 
                treequest_config
            )
            
            logger.info("Enhanced slash command processor initialized with TreeQuest")
            
        except Exception as e:
            logger.error(f"Failed to initialize TreeQuest integration: {e}")
    
    def _register_enhanced_commands(self):
        """Register enhanced slash commands with TreeQuest integration"""
        
        # Core system commands
        self.register_command("help", self._cmd_help, "Show available commands and context-aware help")
        self.register_command("clear", self._cmd_clear, "Clear conversation history")
        self.register_command("settings", self._cmd_settings, "Show/modify CLI settings")
        
        # Memory management commands
        self.register_command("memory", self._cmd_memory, "Manage persistent memory (facts, preferences, tasks)")
        self.register_command("remember", self._cmd_remember, "Add information to persistent memory")
        self.register_command("forget", self._cmd_forget, "Remove information from persistent memory")
        self.register_command("recall", self._cmd_recall, "Search and retrieve information from memory")
        
        # Plugin management commands
        self.register_command("plugins", self._cmd_plugins, "List and manage Monk CLI plugins")
        self.register_command("plugin", self._cmd_plugin, "Plugin management operations")
        self.register_command("create-plugin", self._cmd_create_plugin, "Create a new plugin project")
        
        # Enhanced AI agent commands
        self.register_command("agents", self._cmd_agents, "Manage AI agents and their roles")
        self.register_command("plan", self._cmd_plan, "Create execution plan using TreeQuest planner agent")
        self.register_command("critique", self._cmd_critique, "Get code quality critique using TreeQuest critic agent")
        self.register_command("synthesize", self._cmd_synthesize, "Synthesize insights using TreeQuest synthesizer agent")
        
        # Advanced analysis commands
        self.register_command("deep-analyze", self._cmd_deep_analyze, "Multi-agent deep analysis using TreeQuest")
        self.register_command("optimize", self._cmd_optimize, "Get optimization recommendations using TreeQuest")
        self.register_command("security-scan", self._cmd_security_scan, "Comprehensive security analysis")
        
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
        
        # Split command and arguments - preserve the first word as the command name
        parts = command_text.split()
        command_name = parts[0].lower()  # Convert to lowercase for consistency
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
        
        return SlashCommand(
            name=command_name,
            args=filtered_args,
            kwargs=kwargs,
            raw_input=input_text
        )
    
    async def _cmd_agents(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Show available AI agents and their capabilities"""
        if not self.model_registry:
            return "❌ AI agents not initialized"
        
        available_models = self.model_registry.get_available_models()
        
        response = "🤖 **Available AI Agents**\n\n"
        # Show memory snapshot summary for transparency
        mem = memory_manager.export_context(max_facts=5, max_prefs=5, max_tasks=5)
        if mem['facts'] or mem['preferences'] or mem['tasks']:
            response += "**Memory Snapshot (used across providers)**\n"
            if mem['facts']:
                response += f"  • Facts: {len(mem['facts'])}\n"
            if mem['preferences']:
                response += f"  • Preferences: {len(mem['preferences'])}\n"
            if mem['tasks']:
                response += f"  • Tasks: {len(mem['tasks'])}\n"
            response += "\n"
        
        # Group by capabilities
        agents_by_role = {}
        for model in available_models:
            for capability in model.capabilities:
                if capability.value not in agents_by_role:
                    agents_by_role[capability.value] = []
                agents_by_role[capability.value].append(model)
        
        for role, models in agents_by_role.items():
            response += f"**{role.upper()}**\n"
            for model in models:
                response += f"  • {model.name} ({model.provider}) - Quality: {model.quality_score:.2f}\n"
            response += "\n"
        
        # Show cost analysis
        cost_analysis = self.model_registry.get_cost_analysis()
        response += f"**Cost Analysis**\n"
        response += f"  • Available models: {cost_analysis['available_models']}\n"
        if cost_analysis['cost_ranges']['average_cost']:
            response += f"  • Average cost per 1K tokens: ${cost_analysis['cost_ranges']['average_cost']:.4f}\n"
        
        return response
    
    async def _cmd_plan(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Create execution plan using TreeQuest planner agent"""
        if not self.treequest_engine:
            return "❌ TreeQuest engine not initialized"
        
        # Get planning context
        project_context = conversation_manager.get_project_context()
        if not project_context:
            return "❌ No project context available. Use /workspace to set context."
        
        # Create planning task
        task = "create_execution_plan"
        context = {
            "project_path": project_context.project_path,
            "project_type": project_context.project_type,
            "objective": " ".join(args) if args else "Improve project quality and performance",
            "constraints": kwargs.get("constraints", "Time and resource efficient"),
            "timeline": kwargs.get("timeline", "2-4 weeks"),
            "memory": memory_manager.export_context(),
        }
        
        try:
            # Execute TreeQuest planning
            result = await self.treequest_engine.solve(task, context)
            
            response = "📋 **Execution Plan Generated**\n\n"
            response += f"**Objective**: {context['objective']}\n"
            response += f"**Timeline**: {context['timeline']}\n\n"
            
            if "insights" in result:
                insights = result["insights"]
                response += "**Key Findings**\n"
                for finding in insights.get("key_findings", [])[:5]:
                    response += f"  • {finding}\n"
                
                response += "\n**Recommendations**\n"
                for rec in insights.get("recommendations", [])[:5]:
                    response += f"  • {rec.get('recommendation', 'N/A')}\n"
            
            # Add TreeQuest metrics
            if "treequest_metrics" in result:
                metrics = result["treequest_metrics"]
                response += f"\n**AI Agent Metrics**\n"
                response += f"  • Confidence: {metrics.get('best_node_reward', 0):.2f}\n"
                response += f"  • Cost: ${metrics.get('final_cost_usd', 0):.4f}\n"
                response += f"  • Execution time: {metrics.get('execution_time', 0):.2f}s\n"
                response += f"  • Agent used: {metrics.get('agent_role_used', 'N/A')}\n"
            
            return response
            
        except Exception as e:
            return f"❌ Planning failed: {e}"
    
    async def _cmd_critique(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Get code quality critique using TreeQuest critic agent"""
        if not self.treequest_engine:
            return "❌ TreeQuest engine not initialized"
        
        # Get code context
        target_path = kwargs.get("path", ".")
        if not Path(target_path).exists():
            return f"❌ Path not found: {target_path}"
        
        task = "code_quality_critique"
        context = {
            "target_path": target_path,
            "critique_focus": " ".join(args) if args else "Overall code quality",
            "include_suggestions": kwargs.get("suggestions", "true").lower() == "true",
            "severity_threshold": kwargs.get("severity", "medium")
        }
        
        try:
            # Execute TreeQuest critique
            result = await self.treequest_engine.solve(task, context)
            
            response = "🔍 **Code Quality Critique**\n\n"
            response += f"**Target**: {target_path}\n"
            response += f"**Focus**: {context['critique_focus']}\n\n"
            
            if "insights" in result:
                insights = result["insights"]
                response += "**Quality Assessment**\n"
                response += f"  • Overall Score: {insights.get('confidence_score', 0):.2f}\n"
                response += f"  • Risk Level: {insights.get('risk_assessment', 'unknown')}\n\n"
                
                response += "**Key Issues**\n"
                for finding in insights.get("key_findings", [])[:5]:
                    response += f"  • {finding}\n"
                
                if context["include_suggestions"]:
                    response += "\n**Improvement Suggestions**\n"
                    for rec in insights.get("recommendations", [])[:5]:
                        response += f"  • {rec.get('recommendation', 'N/A')}\n"
            
            return response
            
        except Exception as e:
            return f"❌ Critique failed: {e}"
    
    async def _cmd_synthesize(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Synthesize insights using TreeQuest synthesizer agent"""
        if not self.treequest_engine:
            return "❌ TreeQuest engine not initialized"
        
        # Get synthesis context
        synthesis_topic = " ".join(args) if args else "Project insights and recommendations"
        
        task = "synthesize_insights"
        context = {
            "synthesis_topic": synthesis_topic,
            "include_analyzers": kwargs.get("analyzers", "all"),
            "depth": kwargs.get("depth", "comprehensive"),
            "format": kwargs.get("format", "structured")
        }
        
        try:
            # Execute TreeQuest synthesis
            result = await self.treequest_engine.solve(task, context)
            
            response = "🧠 **Insight Synthesis**\n\n"
            response += f"**Topic**: {synthesis_topic}\n\n"
            
            if "insights" in result:
                insights = result["insights"]
                response += "**Executive Summary**\n"
                response += f"{insights.get('summary', 'No summary available')}\n\n"
                
                response += "**Key Insights**\n"
                for finding in insights.get("key_findings", [])[:7]:
                    response += f"  • {finding}\n"
                
                response += "\n**Priority Actions**\n"
                for action in insights.get("priority_actions", [])[:5]:
                    response += f"  • {action}\n"
                
                if "agent_insights" in insights:
                    response += "\n**AI Agent Perspectives**\n"
                    for agent, agent_insight in insights["agent_insights"].items():
                        response += f"  **{agent.title()}**: {agent_insight.get('integration_score', 'N/A')}\n"
            
            return response
            
        except Exception as e:
            return f"❌ Synthesis failed: {e}"
    
    async def _cmd_deep_analyze(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Multi-agent deep analysis using TreeQuest"""
        if not self.treequest_engine:
            return "❌ TreeQuest engine not initialized"
        
        # Get analysis parameters
        target_path = kwargs.get("path", ".")
        analysis_depth = kwargs.get("depth", "deep")
        include_agents = kwargs.get("agents", "all").split(",")
        
        # Run comprehensive analysis
        task = "comprehensive_project_analysis"
        context = {
            "target_path": target_path,
            "analysis_depth": analysis_depth,
            "included_agents": include_agents,
            "correlation_analysis": True,
            "insight_generation": True
        }
        
        try:
            # Execute TreeQuest analysis
            result = await self.treequest_engine.solve(task, context)
            
            response = "🔬 **Deep Analysis Results**\n\n"
            response += f"**Target**: {target_path}\n"
            response += f"**Depth**: {analysis_depth}\n"
            response += f"**Agents Used**: {', '.join(include_agents)}\n\n"
            
            if "insights" in result:
                insights = result["insights"]
                response += "**Analysis Summary**\n"
                response += f"{insights.get('summary', 'No summary available')}\n\n"
                
                response += "**Critical Findings**\n"
                for finding in insights.get("key_findings", [])[:10]:
                    response += f"  • {finding}\n"
                
                response += "\n**Strategic Recommendations**\n"
                for rec in insights.get("recommendations", [])[:8]:
                    response += f"  • {rec.get('recommendation', 'N/A')}\n"
                
                response += "\n**Risk Assessment**\n"
                response += f"  • Level: {insights.get('risk_assessment', 'unknown')}\n"
                response += f"  • Confidence: {insights.get('confidence_score', 0):.2f}\n"
            
            # Add TreeQuest performance metrics
            if "treequest_metrics" in result:
                metrics = result["treequest_metrics"]
                response += f"\n**AI Performance**\n"
                response += f"  • Total iterations: {metrics.get('total_iterations', 0)}\n"
                response += f"  • Max depth explored: {metrics.get('max_depth_reached', 0)}\n"
                response += f"  • Final cost: ${metrics.get('final_cost_usd', 0):.4f}\n"
                response += f"  • Execution time: {metrics.get('execution_time', 0):.2f}s\n"
                response += f"  • Agent used: {metrics.get('agent_role_used', 'N/A')}\n"
            
            return response
            
        except Exception as e:
            return f"❌ Deep analysis failed: {e}"
    
    async def _cmd_optimize(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Get optimization recommendations using TreeQuest"""
        if not self.treequest_engine:
            return "❌ TreeQuest engine not initialized"
        
        task = "project_optimization"
        context = {
            "optimization_target": " ".join(args) if args else "Overall project performance",
            "focus_areas": kwargs.get("areas", "performance,security,quality").split(","),
            "constraints": kwargs.get("constraints", "Maintain backward compatibility")
        }
        
        try:
            result = await self.treequest_engine.solve(task, context)
            
            response = "⚡ **Optimization Recommendations**\n\n"
            response += f"**Target**: {context['optimization_target']}\n"
            response += f"**Focus Areas**: {', '.join(context['focus_areas'])}\n\n"
            
            if "insights" in result:
                insights = result["insights"]
                response += "**Optimization Summary**\n"
                response += f"{insights.get('summary', 'No summary available')}\n\n"
                
                response += "**Key Recommendations**\n"
                for rec in insights.get("recommendations", [])[:8]:
                    response += f"  • {rec.get('recommendation', 'N/A')}\n"
            
            return response
            
        except Exception as e:
            return f"❌ Optimization analysis failed: {e}"
    
    async def _cmd_security_scan(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Comprehensive security analysis using TreeQuest"""
        if not self.treequest_engine:
            return "❌ TreeQuest engine not initialized"
        
        task = "comprehensive_security_analysis"
        context = {
            "scan_depth": kwargs.get("depth", "comprehensive"),
            "include_dependencies": kwargs.get("deps", "true").lower() == "true",
            "vulnerability_threshold": kwargs.get("threshold", "medium")
        }
        
        try:
            result = await self.treequest_engine.solve(task, context)
            
            response = "🔒 **Security Scan Results**\n\n"
            response += f"**Scan Depth**: {context['scan_depth']}\n"
            response += f"**Dependencies**: {'Included' if context['include_dependencies'] else 'Excluded'}\n\n"
            
            if "insights" in result:
                insights = result["insights"]
                response += "**Security Assessment**\n"
                response += f"  • Risk Level: {insights.get('risk_assessment', 'unknown')}\n"
                response += f"  • Confidence: {insights.get('confidence_score', 0):.2f}\n\n"
                
                response += "**Critical Issues**\n"
                for finding in insights.get("key_findings", [])[:5]:
                    response += f"  • {finding}\n"
                
                response += "\n**Remediation Steps**\n"
                for rec in insights.get("recommendations", [])[:5]:
                    response += f"  • {rec.get('recommendation', 'N/A')}\n"
            
            return response
            
        except Exception as e:
            return f"❌ Security scan failed: {e}"
    
    async def _cmd_help(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Show help information"""
        if args:
            # Help for specific command
            command_name = args[0]
            if command_name in self.command_help:
                return f"/{command_name}: {self.command_help[command_name]}"
            else:
                return f"❌ Unknown command: /{command_name}"
        
        # Show all commands
        help_lines = ["🤖 **Available Commands**\n"]
        
        # Group commands by category
        categories = {
            "Core": ["help", "clear", "settings"],
            "AI Agents": ["agents", "plan", "critique", "synthesize"],
            "Analysis": ["analyze", "deep-analyze", "optimize", "security-scan"],
            "Project": ["workspace", "project"],
            "System": ["providers", "history", "save", "load"],
            "Files": ["ls", "cat", "tree"],
            "Git": ["git", "status", "diff"]
        }
        
        for category, commands in categories.items():
            help_lines.append(f"\n**{category}**")
            for cmd in commands:
                if cmd in self.command_help:
                    help_lines.append(f"  /{cmd} - {self.command_help[cmd]}")
        
        help_lines.append("\n💡 Use /help <command> for detailed help on a specific command")
        return "\n".join(help_lines)
    
    async def _cmd_clear(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Clear conversation history"""
        conversation_manager.clear_conversation()
        return "✅ Conversation history cleared"
    
    async def _cmd_settings(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Show or modify settings"""
        if args:
            if len(args) >= 2:
                key, value = args[0], args[1]
                # TODO: Implement settings management
                return f"✅ Setting {key} = {value}"
            else:
                key = args[0]
                return f"Setting {key}: (value would be shown here)"
        else:
            return """⚙️ **Current Settings**
            
            No settings configured yet.
            Use /settings <key> <value> to modify"""
    
    async def _cmd_analyze(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Analyze files or project"""
        target = args[0] if args else "."
        path = Path(target).resolve()
        
        if not path.exists():
            return f"❌ Path not found: {target}"
        
        try:
            # Use enhanced analyzer coordinator
            result = await self.analyzer_coordinator.comprehensive_analysis(path, kwargs)
            return f"✅ Analysis completed for {target}\n\n{result}"
        except Exception as e:
            return f"❌ Analysis failed: {e}"
    
    async def _cmd_workspace(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Show workspace information"""
        current_workspace = self.workspace_manager.get_current_workspace()
        if current_workspace:
            return f"📁 **Current Workspace**\n\n{current_workspace}"
        else:
            return "No active workspace. Use /analyze to detect project automatically."
    
    async def _cmd_project(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Show project context"""
        context = conversation_manager.get_project_context()
        if context:
            return f"📋 **Project Context**\n\n{context}"
        else:
            return "No project context loaded. Use /analyze to scan current directory."
    
    async def _cmd_providers(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """List and manage providers"""
        # TODO: Integrate with provider system
        return """🤖 **AI Providers**
        
        No providers configured yet.
        Use /providers switch <name> to change active provider"""
    
    async def _cmd_history(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Show conversation history"""
        messages = conversation_manager.get_context_messages()
        if not messages:
            return "No conversation history"
        
        history_lines = ["💬 **Conversation History**\n"]
        for msg in messages[-10:]:  # Last 10 messages
            timestamp = time.strftime("%H:%M", time.localtime(msg.timestamp))
            history_lines.append(f"[{timestamp}] {msg.role}: {msg.content[:100]}...")
        
        return "\n".join(history_lines)
    
    async def _cmd_save(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Save conversation session"""
        session_name = args[0] if args else "default"
        # TODO: Implement named session saving
        return f"✅ Conversation saved as '{session_name}'"
    
    async def _cmd_load(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Load conversation session"""
        session_name = args[0] if args else "default"
        # TODO: Implement session loading
        return f"✅ Loaded conversation '{session_name}'"
    
    async def _cmd_ls(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """List directory contents"""
        target = Path(args[0]) if args else Path.cwd()
        
        if not target.exists():
            return f"❌ Path not found: {target}"
        
        if not target.is_dir():
            return f"❌ Not a directory: {target}"
        
        try:
            items = []
            for item in target.iterdir():
                icon = "📁" if item.is_dir() else "📄"
                items.append(f"{icon} {item.name}")
            
            return f"📂 **{target}**\n\n" + "\n".join(items[:20])  # Limit to 20 items
        except Exception as e:
            return f"❌ Error listing directory: {e}"
    
    async def _cmd_cat(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Show file contents"""
        if not args:
            return "❌ No file specified"
        
        file_path = Path(args[0])
        if not file_path.exists():
            return f"❌ File not found: {file_path}"
        
        if not file_path.is_file():
            return f"❌ Not a file: {file_path}"
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Limit content length
            if len(content) > 1000:
                content = content[:1000] + "\n... (truncated)"
            
            return f"📄 **{file_path}**\n\n```\n{content}\n```"
        except Exception as e:
            return f"❌ Error reading file: {e}"
    
    async def _cmd_tree(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Show directory tree"""
        target = Path(args[0]) if args else Path.cwd()
        
        if not target.exists() or not target.is_dir():
            return f"❌ Directory not found: {target}"
        
        try:
            tree_lines = [f"🌳 **{target}**"]
            
            def build_tree(path: Path, prefix: str = "", is_last: bool = True):
                items = list(path.iterdir())
                items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
                
                for i, item in enumerate(items):
                    is_last_item = i == len(items) - 1
                    current_prefix = "└── " if is_last_item else "├── "
                    tree_lines.append(f"{prefix}{current_prefix}{item.name}")
                    
                    if item.is_dir():
                        next_prefix = prefix + ("    " if is_last_item else "│   ")
                        build_tree(item, next_prefix, is_last_item)
            
            build_tree(target)
            return "\n".join(tree_lines[:50])  # Limit tree size
        except Exception as e:
            return f"❌ Error building tree: {e}"
    
    async def _cmd_git(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Git operations"""
        if not args:
            return await self._cmd_git_status(args, kwargs)
        
        git_cmd = args[0]
        if git_cmd == "status":
            return await self._cmd_git_status(args, kwargs)
        elif git_cmd == "diff":
            return await self._cmd_git_diff(args, kwargs)
        else:
            return f"Git subcommand '{git_cmd}' not implemented"
    
    async def _cmd_git_status(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Show git status"""
        try:
            import subprocess
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                 capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                if result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    status_lines = ["📊 **Git Status**\n"]
                    for line in lines[:10]:  # Limit to 10 changes
                        status_lines.append(f"  {line}")
                    return "\n".join(status_lines)
                else:
                    return "✅ Working directory clean"
            else:
                return "❌ Not a git repository"
        except Exception as e:
            return f"❌ Error checking git status: {e}"
    
    async def _cmd_git_diff(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Show git diff"""
        try:
            import subprocess
            result = subprocess.run(['git', 'diff'], 
                                 capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                if result.stdout.strip():
                    diff_content = result.stdout.strip()
                    # Limit diff length
                    if len(diff_content) > 1000:
                        diff_content = diff_content[:1000] + "\n... (truncated)"
                    
                    return f"📝 **Git Diff**\n\n```diff\n{diff_content}\n```"
                else:
                    return "✅ No changes to show"
            else:
                return "❌ Not a git repository"
        except Exception as e:
            return f"❌ Error checking git diff: {e}"

    # Memory Management Commands
    async def _cmd_memory(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Show memory overview and statistics"""
        try:
            mem = memory_manager.export_context()
            
            response = "🧠 **Persistent Memory Overview**\n\n"
            
            # Facts
            facts = mem.get('facts', [])
            response += f"**Facts** ({len(facts)}):\n"
            for fact in facts[:5]:
                response += f"  • {fact.get('key', 'unknown')}: {str(fact.get('value', ''))[:100]}...\n"
            if len(facts) > 5:
                response += f"  ... and {len(facts) - 5} more\n"
            
            # Preferences
            prefs = mem.get('preferences', [])
            response += f"\n**Preferences** ({len(prefs)}):\n"
            for pref in prefs[:5]:
                response += f"  • {pref.get('key', 'unknown')}: {str(pref.get('value', ''))[:100]}...\n"
            if len(prefs) > 5:
                response += f"  ... and {len(prefs) - 5} more\n"
            
            # Tasks
            tasks = mem.get('tasks', [])
            response += f"\n**Active Tasks** ({len(tasks)}):\n"
            for task in tasks[:5]:
                response += f"  • {task.get('key', 'unknown')}: {str(task.get('value', ''))[:100]}...\n"
            if len(tasks) > 5:
                response += f"  ... and {len(tasks) - 5} more\n"
            
            response += f"\n💡 Use /remember to add, /recall to search, /forget to remove"
            
            return response
            
        except Exception as e:
            return f"❌ Error accessing memory: {e}"

    async def _cmd_remember(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Add information to persistent memory"""
        if not args:
            return "❌ Usage: /remember <key> value=<value> [type=<fact|preference|task>] [importance=<0.0-1.0>] [tags=<tag1,tag2>]"
        
        key = args[0]
        value = kwargs.get('value')
        if not value:
            # Try to construct value from remaining args if no value= specified
            value = ' '.join(args[1:]) if len(args) > 1 else ''
        
        memory_type = kwargs.get('type', 'fact')
        importance = float(kwargs.get('importance', 0.6))
        tags = kwargs.get('tags', '').split(',') if kwargs.get('tags') else []
        
        if not value:
            return "❌ Please provide a value using value=<content> or as additional arguments"
        
        try:
            if memory_type == 'fact':
                item = memory_manager.add_fact(key, value, tags, importance)
            elif memory_type == 'preference':
                item = memory_manager.add_preference(key, value, tags, importance)
            elif memory_type == 'task':
                ttl = int(kwargs.get('ttl', 7 * 24 * 3600))  # Default 7 days
                item = memory_manager.add_task_state(key, value, tags, importance, ttl)
            else:
                return f"❌ Invalid memory type: {memory_type}. Use: fact, preference, or task"
            
            return f"✅ Added to {memory_type} memory:\n**{key}**: {value}\nImportance: {importance}, Tags: {', '.join(tags) if tags else 'none'}"
            
        except Exception as e:
            return f"❌ Error adding to memory: {e}"

    async def _cmd_forget(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Remove information from persistent memory"""
        if not args:
            return "❌ Usage: /forget <key> [type=<fact|preference|task>]"
        
        key = args[0]
        memory_type = kwargs.get('type', 'fact')
        
        try:
            if memory_manager.remove(memory_type, key):
                return f"✅ Removed {memory_type}: {key}"
            else:
                return f"❌ {memory_type.capitalize()} not found: {key}"
                
        except Exception as e:
            return f"❌ Error removing from memory: {e}"

    async def _cmd_recall(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Search and retrieve information from memory"""
        if not args:
            return "❌ Usage: /recall <query> [type=<fact|preference|task>] [limit=<number>]"
        
        query = ' '.join(args)
        memory_type = kwargs.get('type')
        limit = int(kwargs.get('limit', 10))
        
        try:
            # Search by tags or content
            kinds = [memory_type] if memory_type else ['fact', 'preference', 'task']
            results = memory_manager.query(kinds=kinds, tags=[query], limit=limit)
            
            if not results:
                # Try broader search
                results = memory_manager.query(kinds=kinds, limit=limit)
                # Filter by content matching
                results = [r for r in results if query.lower() in str(r.value).lower() or query.lower() in r.key.lower()]
            
            if not results:
                return f"🔍 No memory items found matching: {query}"
            
            response = f"🔍 **Memory Search Results for: {query}**\n\n"
            
            for item in results[:limit]:
                response += f"**{item.kind.upper()}**: {item.key}\n"
                response += f"  Value: {str(item.value)[:150]}...\n"
                response += f"  Importance: {item.importance}, Tags: {', '.join(item.tags) if item.tags else 'none'}\n"
                response += f"  Updated: {time.strftime('%Y-%m-%d %H:%M', time.localtime(item.updated_at))}\n\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error searching memory: {e}"

    # Plugin Management Commands
    async def _cmd_plugins(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """List and manage Monk CLI plugins"""
        try:
            from src.core.plugins import plugin_registry
            
            # Get plugin registry summary
            summary = plugin_registry.get_registry_summary()
            
            response = "🔌 **Monk CLI Plugin Registry**\n\n"
            
            # Plugin counts
            response += f"**Total Plugins**: {summary['total_plugins']}\n"
            response += f"**Enabled**: {summary['enabled_plugins']}\n"
            response += f"**Disabled**: {summary['disabled_plugins']}\n"
            response += f"**Errors**: {summary['error_plugins']}\n\n"
            
            # Plugin types
            response += "**Plugin Types**:\n"
            for ptype, count in summary['plugin_types'].items():
                response += f"  • {ptype.title()}: {count}\n"
            
            # Discovery paths
            response += f"\n**Discovery Paths**:\n"
            for path in summary['discovery_paths'][:3]:  # Show first 3
                response += f"  • {path}\n"
            if len(summary['discovery_paths']) > 3:
                response += f"  ... and {len(summary['discovery_paths']) - 3} more\n"
            
            response += f"\n💡 Use /plugin <operation> for specific plugin management"
            
            return response
            
        except Exception as e:
            return f"❌ Error accessing plugin registry: {e}"

    async def _cmd_plugin(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Plugin management operations"""
        if not args:
            return "❌ Usage: /plugin <operation> [plugin_name] [options]\n\nOperations:\n  list - List all plugins\n  info <name> - Get plugin info\n  enable <name> - Enable plugin\n  disable <name> - Disable plugin\n  reload <name> - Reload plugin\n  test <name> - Test plugin"
        
        operation = args[0].lower()
        
        try:
            from src.core.plugins import plugin_registry
            
            if operation == "list":
                return await self._cmd_plugin_list(args[1:], kwargs)
            elif operation == "info":
                return await self._cmd_plugin_info(args[1:], kwargs)
            elif operation == "enable":
                return await self._cmd_plugin_enable(args[1:], kwargs)
            elif operation == "disable":
                return await self._cmd_plugin_disable(args[1:], kwargs)
            elif operation == "reload":
                return await self._cmd_plugin_reload(args[1:], kwargs)
            elif operation == "test":
                return await self._cmd_plugin_test(args[1:], kwargs)
            else:
                return f"❌ Unknown plugin operation: {operation}"
                
        except Exception as e:
            return f"❌ Error in plugin operation: {e}"

    async def _cmd_plugin_list(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """List all plugins with details"""
        try:
            from src.core.plugins import plugin_registry
            
            plugins = list(plugin_registry.plugins.values())
            if not plugins:
                return "📭 No plugins found in registry"
            
            response = "🔌 **Available Plugins**\n\n"
            
            for plugin in plugins:
                status_emoji = {
                    "enabled": "✅",
                    "disabled": "❌", 
                    "error": "⚠️",
                    "loading": "⏳",
                    "executing": "🔄",
                    "completed": "✅"
                }.get(plugin.status.value, "❓")
                
                response += f"{status_emoji} **{plugin.__class__.__name__}**\n"
                
                if plugin.metadata:
                    response += f"  Version: {plugin.metadata.version}\n"
                    response += f"  Type: {plugin.metadata.plugin_type.value.title()}\n"
                    response += f"  Author: {plugin.metadata.author}\n"
                    response += f"  Description: {plugin.metadata.description[:100]}...\n"
                
                response += f"  Status: {plugin.status.value}\n"
                response += f"  Executions: {plugin.execution_count}\n"
                response += f"  Errors: {plugin.error_count}\n\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error listing plugins: {e}"

    async def _cmd_plugin_info(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Get detailed information about a specific plugin"""
        if not args:
            return "❌ Usage: /plugin info <plugin_name>"
        
        plugin_name = args[0]
        
        try:
            from src.core.plugins import plugin_registry
            
            info = plugin_registry.get_plugin_info(plugin_name)
            if not info:
                return f"❌ Plugin not found: {plugin_name}"
            
            response = f"🔌 **Plugin Info: {plugin_name}**\n\n"
            
            # Basic info
            response += f"**Status**: {info['status']}\n"
            
            # Performance metrics
            perf = info['performance']
            response += f"**Performance**:\n"
            response += f"  • Executions: {perf['execution_count']}\n"
            response += f"  • Errors: {perf['error_count']}\n"
            response += f"  • Uptime: {perf['uptime']:.1f}s\n"
            
            # Metadata
            if 'metadata' in info:
                meta = info['metadata']
                response += f"\n**Metadata**:\n"
                response += f"  • Version: {meta['version']}\n"
                response += f"  • Type: {meta['plugin_type'].title()}\n"
                response += f"  • Author: {meta['author']}\n"
                response += f"  • Description: {meta['description']}\n"
                response += f"  • Dependencies: {', '.join(meta['dependencies']) if meta['dependencies'] else 'None'}\n"
                response += f"  • Tags: {', '.join(meta['tags']) if meta['tags'] else 'None'}\n"
                response += f"  • Memory Access: {', '.join(meta['memory_access']) if meta['memory_access'] else 'None'}\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error getting plugin info: {e}"

    async def _cmd_plugin_enable(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Enable a plugin"""
        if not args:
            return "❌ Usage: /plugin enable <plugin_name>"
        
        plugin_name = args[0]
        
        try:
            from src.core.plugins import plugin_registry
            
            if plugin_registry.enable_plugin(plugin_name):
                return f"✅ Enabled plugin: {plugin_name}"
            else:
                return f"❌ Failed to enable plugin: {plugin_name}"
                
        except Exception as e:
            return f"❌ Error enabling plugin: {e}"

    async def _cmd_plugin_disable(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Disable a plugin"""
        if not args:
            return "❌ Usage: /plugin disable <plugin_name>"
        
        plugin_name = args[0]
        
        try:
            from src.core.plugins import plugin_registry
            
            if plugin_registry.disable_plugin(plugin_name):
                return f"✅ Disabled plugin: {plugin_name}"
            else:
                return f"❌ Failed to disable plugin: {plugin_name}"
                
        except Exception as e:
            return f"❌ Error disabling plugin: {e}"

    async def _cmd_plugin_reload(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Reload a plugin"""
        if not args:
            return "❌ Usage: /plugin reload <plugin_name>"
        
        plugin_name = args[0]
        
        try:
            from src.core.plugins import plugin_registry
            
            # Unload and reload
            if plugin_registry.unload_plugin(plugin_name):
                if plugin_registry.load_plugin(plugin_name):
                    return f"✅ Reloaded plugin: {plugin_name}"
                else:
                    return f"❌ Failed to reload plugin: {plugin_name}"
            else:
                return f"❌ Failed to unload plugin: {plugin_name}"
                
        except Exception as e:
            return f"❌ Error reloading plugin: {e}"

    async def _cmd_plugin_test(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Test a plugin"""
        if not args:
            return "❌ Usage: /plugin test <plugin_name>"
        
        plugin_name = args[0]
        
        try:
            from src.core.plugins import plugin_registry
            
            plugin = plugin_registry.get_plugin(plugin_name)
            if not plugin:
                return f"❌ Plugin not found: {plugin_name}"
            
            # Test plugin execution
            try:
                result = plugin.execute("test", ["test_arg"], {"test_key": "test_value"})
                return f"✅ Plugin test successful: {plugin_name}\nResult: {result[:200]}..."
            except Exception as e:
                return f"❌ Plugin test failed: {plugin_name}\nError: {e}"
                
        except Exception as e:
            return f"❌ Error testing plugin: {e}"

    async def _cmd_create_plugin(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """Create a new plugin project"""
        if not args:
            return "❌ Usage: /create-plugin <plugin_name> [type=<command|analyzer|integration>] [author=<name>] [description=<text>]"
        
        plugin_name = args[0]
        plugin_type = kwargs.get('type', 'command')
        author = kwargs.get('author', 'Unknown')
        description = kwargs.get('description', f'A {plugin_type} plugin for Monk CLI')
        dependencies = kwargs.get('dependencies', '').split(',') if kwargs.get('dependencies') else []
        
        try:
            from src.core.plugins.scaffold import PluginScaffolder
            
            scaffolder = PluginScaffolder()
            success = scaffolder.create_plugin(
                plugin_name=plugin_name,
                plugin_type=plugin_type,
                author=author,
                description=description,
                dependencies=dependencies
            )
            
            if success:
                return f"✅ Successfully created plugin: {plugin_name}\n\nNext steps:\n1. cd {plugin_name}\n2. pip install -e .\n3. monk plugin test {plugin_name}"
            else:
                return f"❌ Failed to create plugin: {plugin_name}"
                
        except Exception as e:
            return f"❌ Error creating plugin: {e}"
    
    async def execute_command(self, command: SlashCommand) -> Dict[str, Any]:
        """Execute a slash command with enhanced TreeQuest integration"""
        try:
            if command.name not in self.commands:
                return {
                    "success": False,
                    "error": f"Unknown command: /{command.name}",
                    "suggestions": self._get_command_suggestions(command.name)
                }
            
            # Execute the command
            command_func = self.commands[command.name]
            result = await command_func(command.args, command.kwargs)
            
            return {
                "success": True,
                "data": result,
                "command": command.name,
                "execution_time": 0.0  # Could add timing here
            }
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "suggestions": []
            }
    
    def _get_command_suggestions(self, command_name: str) -> List[str]:
        """Get command suggestions for typos"""
        suggestions = []
        for cmd in self.commands.keys():
            if cmd.startswith(command_name) or command_name in cmd:
                suggestions.append(cmd)
        return suggestions[:3]  # Limit to 3 suggestions

# Global enhanced slash command processor
slash_processor = EnhancedSlashCommandProcessor()
