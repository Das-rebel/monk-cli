#!/usr/bin/env python3
"""
Monk CLI Enhanced - Entry Point
High-performance CLI with async operations and advanced features
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.async_engine import async_router, performance_profiler
from ui.rich_interface import RichCLI
from core.cache_manager import cache_manager
from analyzers.analyzer_coordinator import coordinator
from workspace.workspace_manager import workspace_manager
from plugins.plugin_manager import plugin_manager

async def main():
    """Main CLI entry point"""
    start_time = time.time()
    
    try:
        # Initialize components
        await initialize_components()
        
        # Initialize Rich CLI
        cli = RichCLI()
        
        # Parse arguments
        if len(sys.argv) < 2:
            await cli.show_help()
            print("\n🎯 Entering interactive mode... (Press Ctrl+C to exit)")
            print("💡 Tip: Use 'Monk help' to see all commands, or run a specific command like 'Monk ai models'")
            
            # Interactive mode - wait for user input
            try:
                while True:
                    user_input = input("\n🔍 Monk> ").strip()
                    if not user_input:
                        continue
                    
                    # Parse user input
                    parts = user_input.split()
                    command = parts[0]
                    args = parts[1:] if len(parts) > 1 else []
                    
                    # Handle exit commands
                    if command.lower() in ['exit', 'quit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    # Handle help
                    if command.lower() in ['help', 'h', '?']:
                        await cli.show_help()
                        continue
                    
                    # Execute command
                    try:
                        result = await async_router.execute_command(command, args)
                        await cli.display_result(result)
                    except Exception as e:
                        print(f"❌ Error executing '{command}': {e}")
                        print("💡 Try 'help' to see available commands")
                        
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
            return
        
        command = sys.argv[1]
        args = sys.argv[2:]
        
        # Handle help commands
        if command in ['--help', '-h', 'help']:
            await cli.show_help()
            return
        
        # Execute command
        result = await async_router.execute_command(command, args)
        
        # Display result
        await cli.display_result(result)
        
        # Show performance metrics if requested
        if '--metrics' in args:
            metrics = async_router.get_performance_metrics()
            await cli.display_metrics(metrics)
    
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        await cleanup_components()
        total_time = time.time() - start_time
        print(f"\n⏱️ Total execution time: {total_time:.3f}s")

async def initialize_components():
    """Initialize all system components"""
            print("🚀 Initializing Monk CLI Enhanced...")
    
    # Initialize cache manager
    await cache_manager.initialize()
    print("✅ Cache manager initialized")
    
    # Initialize analyzer coordinator
    await coordinator.initialize()
    print("✅ Analyzer coordinator initialized")
    
    # Set coordinator instance in async router
    async_router.set_coordinator_instance(coordinator)
    print("✅ Coordinator instance set in async router")
    
    # Initialize workspace manager
    print("✅ Workspace manager initialized")
    
    # Initialize plugin manager
    print("✅ Plugin manager initialized")
    
    # Discover and load plugins
    plugins_result = await plugin_manager.discover_plugins()
    if plugins_result['success']:
        print(f"🔌 Discovered {plugins_result['total']} plugins")
    
            print("🎯 Monk CLI Enhanced ready!")

async def cleanup_components():
    """Cleanup all system components"""
    print("\n🧹 Cleaning up...")
    
    # Shutdown plugin manager
    await plugin_manager.shutdown()
    
    # Shutdown analyzer coordinator
    await coordinator.shutdown()
    
    # Shutdown async router
    await async_router.shutdown()
    
    print("✅ Cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())
