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
    
    async def initialize(self, args):
        """Initialize Monk CLI components"""
        try:
            # Initialize core components
            self.project_context_loader = ProjectContextLoader()
            self.slash_processor = EnhancedSlashCommandProcessor()
            self.intelligent_router = IntelligentRouter()
            self.nl_parser = NLCommandParser()
            
            # Initialize TreeQuest if enabled
            if args.treequest:
                await self.slash_processor.initialize()
                logger.info("Enhanced slash command processor initialized with TreeQuest")
            
            # Load project context
            await self.project_context_loader.load_project_context()
            logger.info("Project context loaded successfully")
            
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
            nargs='?',
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
        
        # Show context information
        context = conversation_manager.get_project_context()
        if context:
            print(f"📁 Project: {context.project_type} at {context.project_path}")
            insights = self.project_context_loader.get_project_insights(context)
            for insight in insights[:3]:
                print(f"   {insight}")
        
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
