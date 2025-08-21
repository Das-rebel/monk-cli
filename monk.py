#!/usr/bin/env python3
"""
Monk CLI - Enhanced with TreeQuest AI Agents
A high-performance, intelligent command-line interface for project analysis and development workflow enhancement.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import re

# Add paths
sys.path.append(str(Path(__file__).parent))

from src.core.conversation_manager import conversation_manager
from src.core.slash_command_processor import EnhancedSlashCommandProcessor
from src.core.intelligent_router import IntelligentRouter
from src.core.project_context_loader import ProjectContextLoader
from src.core.nl_command_parser import NLCommandParser
from src.core.memory_manager import MemoryManager

# Enhanced TreeQuest imports
try:
    from src.ai.enhanced_treequest import EnhancedTreeQuestEngine, EnhancedTreeQuestConfig
    from src.ai.model_registry import ModelRegistry
    ENHANCED_TREEQUEST_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced TreeQuest not available: {e}")
    ENHANCED_TREEQUEST_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonkCLI:
    """
    Monk CLI - Enhanced with TreeQuest AI Agents
    High-performance, intelligent command-line interface
    """
    
    def __init__(self):
        self.backend = None
        self.initialized = False
        self.enhanced_treequest = None
        self.memory_manager = None
        self.model_registry = None
    
    async def initialize(self, args):
        """Initialize Monk CLI components"""
        try:
            # Initialize core components
            self.project_context_loader = ProjectContextLoader()
            self.slash_processor = EnhancedSlashCommandProcessor()
            self.intelligent_router = IntelligentRouter()
            self.nl_parser = NLCommandParser()
            self.memory_manager = MemoryManager()
            
            # Initialize Enhanced TreeQuest if available and enabled
            if ENHANCED_TREEQUEST_AVAILABLE and (args.enhanced or args.treequest):
                logger.info("Initializing Enhanced TreeQuest system...")
                
                # Initialize model registry
                self.model_registry = ModelRegistry()
                
                # Configure enhanced TreeQuest
                config = EnhancedTreeQuestConfig(
                    max_depth=3,
                    branching_factor=4,
                    rollout_budget=32,
                    cost_cap_usd=0.50,
                    timeout_seconds=120,
                    memory_guided=True,
                    adaptive_rewards=True,
                    agent_specialization=True,
                    performance_tracking=True,
                    learning_enabled=True
                )
                
                # Initialize enhanced TreeQuest engine
                self.enhanced_treequest = EnhancedTreeQuestEngine(
                    config, self.memory_manager, self.model_registry
                )
                
                logger.info("✅ Enhanced TreeQuest system initialized with memory and learning capabilities")
                
            # Initialize TreeQuest if enabled but enhanced not available
            elif args.treequest:
                await self.slash_processor.initialize()
                logger.info("TreeQuest initialized (basic mode)")
            
            # Standard initialization
            else:
                logger.info("Monk CLI initialized in standard mode")
            
            # Load project context with timeout and fallback
            try:
                # Try to load project context with timeout
                context_task = self.project_context_loader.load_project_context()
                context = await asyncio.wait_for(context_task, timeout=15.0)
                logger.info("Project context loaded successfully")
            except asyncio.TimeoutError:
                logger.warning("Project context loading timed out, continuing without context")
                context = None
            except Exception as e:
                logger.warning(f"Could not load project context: {e}, continuing without context")
                context = None
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Monk CLI: {e}")
            return False
    
    def create_parser(self):
        """Create command line argument parser"""
        parser = argparse.ArgumentParser(
            description="🧘 Monk CLI - Enhanced with TreeQuest AI Agents",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  monk --treequest /agents                    # Show available AI agents
  monk --treequest /plan objective="Improve security"  # Create execution plan
  monk --treequest /deep-analyze path=src/   # Run comprehensive analysis
  monk --treequest --chat                     # Interactive mode with TreeQuest
  monk "How can I optimize my project?"      # Natural language query
            """
        )
        
        parser.add_argument(
            'query',
            nargs='*',
            help='Query to process (optional for interactive mode)'
        )
        
        parser.add_argument(
            '--treequest',
            action='store_true',
            help='Enable TreeQuest AI agent integration'
        )
        
        parser.add_argument(
            '--chat',
            action='store_true',
            help='Enable interactive chat mode'
        )
        
        parser.add_argument(
            '--provider',
            help='Force use of specific AI provider'
        )
        
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output'
        )
        
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug mode'
        )
        
        parser.add_argument(
            '--enhanced',
            action='store_true',
            help='Enable Enhanced TreeQuest with memory and learning (requires deployment)'
        )
        
        return parser
    
    def _is_complex_query(self, query: str) -> bool:
        """Determine if a query is complex enough to warrant TreeQuest processing"""
        complex_patterns = [
            r'\b(analyze|analyze|examine|investigate|research|study)\b',
            r'\b(plan|strategy|roadmap|timeline|approach)\b',
            r'\b(optimize|improve|enhance|refactor|restructure)\b',
            r'\b(security|performance|quality|architecture|design)\b',
            r'\b(complex|complicated|challenging|difficult)\b',
            r'\b(how\s+to|what\s+if|why\s+does|when\s+should)\b',
            r'\b(compare|evaluate|assess|review|critique)\b'
        ]
        
        query_lower = query.lower()
        complexity_score = 0
        
        for pattern in complex_patterns:
            if re.search(pattern, query_lower):
                complexity_score += 1
        
        # Consider queries with 2+ complexity indicators as complex
        return complexity_score >= 2
    
    def _is_in_project_directory(self) -> bool:
        """Check if current directory appears to be a project directory"""
        current_dir = Path.cwd()
        
        # Check for common project indicators
        project_indicators = [
            'requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile',  # Python
            'package.json', 'yarn.lock', 'package-lock.json',           # Node.js
            'pom.xml', 'build.gradle', 'gradle.properties',             # Java
            'Cargo.toml', 'Cargo.lock',                                 # Rust
            'go.mod', 'go.sum',                                         # Go
            'CMakeLists.txt', 'Makefile',                               # C++
            'index.html', 'webpack.config.js', '.babelrc',              # Web
            '.git', 'README.md', 'LICENSE'                              # General
        ]
        
        return any((current_dir / indicator).exists() for indicator in project_indicators)
    
    def _get_project_guidance(self) -> str:
        """Get guidance for users not in a project directory"""
        if not self._is_in_project_directory():
            return """
⚠️  You're not in a project directory. For best results:

📁 Navigate to your project directory:
   cd /path/to/your/project

🧘 Or run Monk CLI with specific context:
   monk --treequest /agents
   monk --treequest /plan objective="Your objective"

💡 Monk CLI works best when you're inside a project directory!
"""
        return ""
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query for enhanced processing"""
        query_lower = query.lower()
        
        patterns = {
            'code_analysis': ['analyze code', 'review code', 'code review', 'bugs', 'issues'],
            'architecture_design': ['architecture', 'design system', 'structure', 'patterns'],
            'planning': ['plan', 'strategy', 'roadmap', 'timeline', 'schedule'],
            'optimization': ['optimize', 'improve', 'performance', 'efficiency'],
            'security': ['security', 'vulnerability', 'secure', 'auth'],
            'documentation': ['document', 'explain', 'describe'],
            'troubleshooting': ['debug', 'fix', 'solve', 'error', 'problem']
        }
        
        for query_type, keywords in patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type
        
        return 'general'
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score (0.0 to 1.0)"""
        complexity_factors = {
            'length': len(query.split()) / 50,  # Normalize by 50 words
            'technical_terms': len([word for word in query.lower().split() 
                                  if word in ['api', 'database', 'architecture', 'algorithm', 'performance', 'security']]) / 10,
            'question_complexity': len([word for word in query.lower().split() 
                                      if word in ['how', 'why', 'what', 'when', 'where', 'which']]) / 5,
            'action_verbs': len([word for word in query.lower().split() 
                               if word in ['analyze', 'optimize', 'design', 'implement', 'review', 'compare']]) / 5
        }
        
        # Weighted complexity score
        complexity = (
            0.2 * min(1.0, complexity_factors['length']) +
            0.3 * min(1.0, complexity_factors['technical_terms']) +
            0.2 * min(1.0, complexity_factors['question_complexity']) +
            0.3 * min(1.0, complexity_factors['action_verbs'])
        )
        
        return max(0.1, min(1.0, complexity))
    
    def _format_enhanced_response(self, result: Dict[str, Any], verbose: bool = False) -> str:
        """Format enhanced TreeQuest response for display"""
        response_parts = []
        
        # Main response
        if 'response' in result:
            response_parts.append(f"🧠 **Enhanced TreeQuest Response**\n\n{result['response']}")
        elif 'insights' in result and 'summary' in result['insights']:
            response_parts.append(f"🧠 **Enhanced TreeQuest Response**\n\n{result['insights']['summary']}")
        else:
            response_parts.append("🧠 **Enhanced TreeQuest Response**\n\nQuery processed successfully.")
        
        # Agent assignment info
        agent_info = result.get('agent_assignment', {})
        if agent_info.get('selected_agent'):
            confidence = agent_info.get('confidence', 0)
            response_parts.append(f"\n👤 **Agent**: {agent_info['selected_agent'].title()} (Confidence: {confidence:.1%})")
            
            if verbose and agent_info.get('reasoning'):
                response_parts.append(f"**Reasoning**: {agent_info['reasoning']}")
        
        # Execution analytics (if verbose)
        if verbose:
            analytics = result.get('execution_analytics', {})
            if analytics:
                response_parts.append("\n📊 **Analytics**:")
                if 'total_execution_time' in analytics:
                    response_parts.append(f"- Execution time: {analytics['total_execution_time']:.2f}s")
                if 'memory_guided_decisions' in analytics:
                    response_parts.append(f"- Memory-guided decisions: {analytics['memory_guided_decisions']}")
                if 'learning_applied' in analytics:
                    response_parts.append(f"- Learning applied: {analytics['learning_applied']}")
        
        # Recommendations
        recommendations = result.get('recommendations', [])
        if recommendations:
            response_parts.append("\n💡 **Recommendations**:")
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                response_parts.append(f"{i}. {rec}")
        
        return '\n'.join(response_parts)
    
    async def handle_query(self, query_text: str, args) -> str:
        """Handle a single query with full Phase 3 processing and TreeQuest integration"""
        
        # Step 1: Check if it's a slash command
        slash_command = self.slash_processor.parse_command(query_text)
        
        if slash_command:
            # Execute slash command with TreeQuest integration
            result = await self.slash_processor.execute_command(slash_command)
            
            if result["success"]:
                response = result["data"]
                
                # Log interaction
                conversation_manager.add_message("user", query_text)
                conversation_manager.add_message("assistant", response, {"command_type": "slash", "treequest_enabled": args.treequest})
                
                return response
            else:
                error_msg = f"❌ {result['error']}"
                if "suggestions" in result:
                    suggestions = result["suggestions"]
                    if suggestions:
                        error_msg += f"\nDid you mean: {', '.join(suggestions)}"
                return error_msg
        
        # Step 2: Try natural language parsing
        parsed_intent = await self.nl_parser.parse(query_text)
        
        if parsed_intent.command and parsed_intent.confidence > 0.7:
            # Execute parsed command
            converted_command = self.slash_processor.parse_command(parsed_intent.command)
            if converted_command:
                result = await self.slash_processor.execute_command(converted_command)
                
                if result["success"]:
                    response = f"🧠 Interpreted as: {parsed_intent.command}\n\n{result['data']}"
                    
                    # Log interaction
                    conversation_manager.add_message("user", query_text)
                    conversation_manager.add_message("assistant", response, {
                        "command_type": "nl_parsed",
                        "original_intent": parsed_intent.intent,
                        "parsed_command": parsed_intent.command,
                        "treequest_enabled": args.treequest
                    })
                    
                    return response
        
        # Step 3: Route to Enhanced TreeQuest or standard processing
        # Use Enhanced TreeQuest for complex queries when available
        if self.enhanced_treequest and self._is_complex_query(query_text):
            try:
                # Build enhanced context
                context = {
                    "query": query_text,
                    "conversation_history": conversation_manager.get_recent_messages(5),
                    "project_context": conversation_manager.get_context_for_ai(),
                    "user_preferences": {"verbose": args.verbose, "debug": args.debug},
                    "task_type": self._classify_query_type(query_text),
                    "complexity": self._calculate_query_complexity(query_text),
                    "in_project_directory": self._is_in_project_directory()
                }
                
                logger.info("🧠 Using Enhanced TreeQuest for complex query...")
                result = await self.enhanced_treequest.solve_enhanced(query_text, context)
                
                if result.get("success", False):
                    # Format enhanced response
                    response = self._format_enhanced_response(result, args.verbose)
                else:
                    response = f"🤖 Enhanced TreeQuest encountered an issue: {result.get('error', 'Unknown error')}"
                
                # Log interaction with enhanced metadata
                conversation_manager.add_message("user", query_text)
                conversation_manager.add_message("assistant", response, {
                    "provider": "enhanced_treequest",
                    "routing_confidence": 1.0,
                    "query_type": "enhanced_treequest",
                    "agent_used": result.get('agent_assignment', {}).get('selected_agent', 'unknown'),
                    "execution_time": result.get('execution_analytics', {}).get('total_execution_time', 0),
                    "memory_guided": result.get('enhanced_features', {}).get('memory_guided', False),
                    "learning_applied": result.get('execution_analytics', {}).get('learning_applied', False)
                })
                
                return response
                
            except Exception as e:
                logger.error(f"Enhanced TreeQuest error: {e}")
                # Fall back to standard processing
                
        # Fall back to standard TreeQuest if available
        elif args.treequest and self._is_complex_query(query_text):
            # Use standard TreeQuest for complex queries
            try:
                task = "general_ai_assistance"
                context = {
                    "query": query_text,
                    "conversation_history": conversation_manager.get_recent_messages(5),
                    "project_context": conversation_manager.get_context_for_ai()
                }
                
                result = await self.slash_processor.treequest_engine.solve(task, context)
                
                if "insights" in result:
                    response = f"🤖 **TreeQuest AI Response**\n\n{result['insights'].get('summary', 'No response generated')}"
                else:
                    response = "🤖 TreeQuest processed your query but no response was generated."
                
                # Log interaction
                conversation_manager.add_message("user", query_text)
                conversation_manager.add_message("assistant", response, {
                    "provider": "treequest",
                    "routing_confidence": 1.0,
                    "query_type": "treequest_direct",
                    "treequest_enabled": args.treequest
                })
                
                return response
                
            except Exception as e:
                # Fall back to regular routing if TreeQuest fails
                pass
        
        routing_decision = await self.intelligent_router.route_query(
            query_text,
            conversation_manager.get_context_for_ai()
        )
        
        provider = args.provider or routing_decision.primary_provider
        
        if args.verbose:
            routing_explanation = self.intelligent_router.explain_routing(routing_decision)
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
                        "query_type": routing_decision.query_type.value,
                        "treequest_enabled": args.treequest
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
                                    "fallback_used": True,
                                    "treequest_enabled": args.treequest
                                })
                                return f"🔄 Used fallback provider {fallback_provider}:\n\n{response}"
                        except:
                            continue
                    
                    return "❌ All AI providers failed to respond"
            
            except Exception as e:
                return f"❌ Error processing query: {e}"
        else:
            return f"❌ Backend not available. Query: {query_text}"
    
    async def interactive_mode(self, args):
        """Enhanced interactive mode with all Phase 3 features and TreeQuest integration"""
        print("🤖 Monk CLI - Enhanced with TreeQuest AI Agents")
        print("=" * 70)
        
        # Show project guidance if not in project directory
        project_guidance = self._get_project_guidance()
        if project_guidance:
            print(project_guidance)
        
        # Show context information
        context = conversation_manager.get_project_context()
        if context:
            print(f"📁 Project: {context.project_type} at {context.project_path}")
            insights = self.project_context_loader.get_project_insights(context)
            for insight in insights[:3]:
                print(f"   {insight}")
        else:
            print("📁 No project context available")
        
        # Show TreeQuest agent information
        if hasattr(self.slash_processor, 'model_registry') and self.slash_processor.model_registry:
            available_models = self.slash_processor.model_registry.get_available_models()
            print(f"🤖 AI Agents: {len(available_models)} models available")
        
        print("\nType '/help' for commands, '/agents' for AI agent info, 'quit' to exit")
        print("-" * 70)
        
        while True:
            try:
                # Get user input with auto-completion hints
                user_input = input("\n💭 Monk> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    # Show context-aware suggestions
                    suggestions = await self.slash_processor.command_completion.complete_command("")
                    if suggestions:
                        print("💡 Suggestions:")
                        for suggestion in suggestions[:5]:
                            print(f"   {suggestion}")
                    continue
                
                # Process query
                response = await self.handle_query(user_input, args)
                
                # Display response with rich formatting
                if args.verbose:
                    print(response)
                else:
                    print(response)
                
                # Auto-save session periodically
                if len(conversation_manager.conversation_history) % 10 == 0:
                    await self.slash_processor.session_manager.auto_save_session()
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
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
        await self.initialize(args)
        
        # Handle direct query
        if args.query:
            # Join arguments properly, preserving slash commands
            if len(args.query) == 1 and args.query[0].startswith('/'):
                # Single slash command - use as-is
                query_text = args.query[0]
            else:
                # Multiple arguments or non-slash command - join with spaces
                query_text = ' '.join(args.query)
            
            response = await self.handle_query(query_text, args)
            
            # Display response
            if args.verbose:
                print(response)
            else:
                print(response)
        
        # Interactive mode
        elif args.chat or not args.query:
            await self.interactive_mode(args)

async def main():
    """Main entry point"""
    try:
        cli = MonkCLI()
        await cli.run()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
