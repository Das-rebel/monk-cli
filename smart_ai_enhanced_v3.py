#!/usr/bin/env python3
"""
Smart AI Enhanced CLI v3.0 - Claude-Style Interface with TreeQuest AI Agents
Complete integration of all Phase 3 components with enhanced TreeQuest capabilities
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import re

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
    Enhanced Smart AI CLI with Claude-style interface, Phase 3 features, and TreeQuest AI agents
    """
    
    def __init__(self):
        self.backend = SmartAIBackend() if SmartAIBackend else None
        self.rich_cli = RichCLI()
        self.is_interactive_mode = False
        
    async def initialize(self):
        """Initialize all components including TreeQuest integration"""
        try:
            # Start session with project context
            await session_manager.start_session()
            
            # Initialize enhanced slash command processor with TreeQuest
            await slash_processor.initialize()
            logger.info("Enhanced slash command processor initialized with TreeQuest")
            
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
            description="Smart AI Enhanced CLI v3.0 - Claude-Style Interface with TreeQuest AI Agents",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  smart-ai-enhanced.py                          # Interactive mode
  smart-ai-enhanced.py "analyze this project"  # Direct query
  smart-ai-enhanced.py /help                   # Slash command
  smart-ai-enhanced.py /agents                 # Show AI agents
  smart-ai-enhanced.py /plan                   # Create execution plan
  smart-ai-enhanced.py /deep-analyze           # Multi-agent analysis
  smart-ai-enhanced.py --chat                  # Force interactive mode
            """
        )
        
        parser.add_argument('query', nargs='*', help='Query or command to execute')
        parser.add_argument('--chat', '-c', action='store_true', help='Interactive chat mode')
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        parser.add_argument('--format', choices=['rich', 'plain', 'json'], default='rich', help='Output format')
        parser.add_argument('--no-context', action='store_true', help='Disable project context loading')
        parser.add_argument('--provider', help='Force specific AI provider')
        parser.add_argument('--treequest', action='store_true', help='Enable TreeQuest AI agent features')
        
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
    
    async def handle_query(self, query_text: str, args) -> str:
        """Handle a single query with full Phase 3 processing and TreeQuest integration"""
        
        # Step 1: Check if it's a slash command
        slash_command = slash_processor.parse_command(query_text)
        
        if slash_command:
            # Execute slash command with TreeQuest integration
            result = await slash_processor.execute_command(slash_command)
            
            if result["success"]:
                response = result["data"]
                
                # Log interaction
                conversation_manager.add_message("user", query_text)
                conversation_manager.add_message("assistant", response, {"command_type": "slash", "treequest_enabled": True})
                
                return response
            else:
                error_msg = f"❌ {result['error']}"
                if "suggestions" in result:
                    suggestions = result["suggestions"]
                    if suggestions:
                        error_msg += f"\nDid you mean: {', '.join(suggestions)}"
                return error_msg
        
        # Step 2: Try natural language parsing
        parsed_intent = await nl_parser.parse(query_text)
        
        if parsed_intent.command and parsed_intent.confidence > 0.7:
            # Execute parsed command
            converted_command = slash_processor.parse_command(parsed_intent.command)
            if converted_command:
                result = await slash_processor.execute_command(converted_command)
                
                if result["success"]:
                    response = f"🧠 Interpreted as: {parsed_intent.command}\n\n{result['data']}"
                    
                    # Log interaction
                    conversation_manager.add_message("user", query_text)
                    conversation_manager.add_message("assistant", response, {
                        "command_type": "nl_parsed",
                        "original_intent": parsed_intent.intent,
                        "parsed_command": parsed_intent.command,
                        "treequest_enabled": True
                    })
                    
                    return response
        
        # Step 3: Route to AI provider with TreeQuest consideration
        # When TreeQuest is enabled, prioritize it for complex queries
        if args.treequest and self._is_complex_query(query_text):
            # Use TreeQuest directly for complex queries
            try:
                task = "general_ai_assistance"
                context = {
                    "query": query_text,
                    "conversation_history": conversation_manager.get_recent_messages(5),
                    "project_context": conversation_manager.get_context_for_ai()
                }
                
                result = await slash_processor.treequest_engine.solve(task, context)
                
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
                    "treequest_enabled": True
                })
                
                return response
                
            except Exception as e:
                # Fall back to regular routing if TreeQuest fails
                pass
        
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
        print("🤖 Smart AI Enhanced v3.0 - Claude-Style Interface with TreeQuest AI Agents")
        print("=" * 70)
        
        # Show context information
        context = conversation_manager.get_project_context()
        if context:
            print(f"📁 Project: {context.project_type} at {context.project_path}")
            insights = project_context_loader.get_project_insights(context)
            for insight in insights[:3]:
                print(f"   {insight}")
        
        # Show TreeQuest agent information
        if hasattr(slash_processor, 'model_registry') and slash_processor.model_registry:
            available_models = slash_processor.model_registry.get_available_models()
            print(f"🤖 AI Agents: {len(available_models)} models available")
        
        print("\nType '/help' for commands, '/agents' for AI agent info, 'quit' to exit")
        print("-" * 70)
        
        self.is_interactive_mode = True
        
        while True:
            try:
                # Get user input with auto-completion hints
                user_input = input("\n💭 Smart AI> ").strip()
                
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
                    print(response)
                else:
                    print(response)
                
                # Auto-save session periodically
                if len(conversation_manager.conversation_history) % 10 == 0:
                    await session_manager.auto_save_session()
            
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
                    "session_id": conversation_manager.current_session_id,
                    "treequest_enabled": args.treequest
                }
                print(json.dumps(result, indent=2))
            elif args.format == 'rich':
                print(response)
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
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
