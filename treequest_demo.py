#!/usr/bin/env python3
"""
TreeQuest AI Agent Demo for Monk CLI
Demonstrates the enhanced TreeQuest integration with multi-agent capabilities
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def print_header():
    """Print demo header"""
    print("🌟 Monk CLI - TreeQuest AI Agent Demo")
    print("=" * 60)
    print("🚀 TreeQuest AI Agent Integration Demo")
    print("=" * 50)

async def demo_treequest_agents():
    """Demo TreeQuest AI agent capabilities"""
    print_header()
    
    try:
        # Import TreeQuest components
        from src.ai.model_registry import ModelRegistry
        from src.ai.treequest_engine import TreeQuestEngine, TreeQuestConfig
        from src.core.cache_manager import cache_manager
        
        print("✅ TreeQuest components imported successfully")
        
        # Initialize model registry
        print("\n🤖 Initializing AI Model Registry...")
        model_registry = ModelRegistry()
        available_models = model_registry.get_available_models()
        print(f"   Available models: {len(available_models)}")
        
        for model in available_models[:3]:  # Show first 3 models
            print(f"   • {model.name} ({model.provider}) - Quality: {model.quality_score:.2f}")
        
        # Initialize TreeQuest engine
        print("\n🧠 Initializing TreeQuest Engine...")
        treequest_config = TreeQuestConfig(
            max_depth=3,
            branching=4,
            rollout_budget=16,  # Reduced for demo
            cost_cap_usd=0.25,
            objective="quality"
        )
        
        treequest_engine = TreeQuestEngine(
            model_registry, 
            cache_manager, 
            treequest_config
        )
        print("   ✅ TreeQuest engine initialized")
        
        # Demo 1: Basic synthesis task
        print("\n📊 Demo 1: Basic Insight Synthesis")
        print("-" * 30)
        
        demo_context = {
            "analyzers": {
                "github": {
                    "success": True,
                    "data": {"code_quality": 85, "issues": 3},
                    "execution_time": 1.2,
                    "metadata": {"analyzer": "github"}
                },
                "npm": {
                    "success": True,
                    "data": {"vulnerabilities": 2, "outdated": 5},
                    "execution_time": 0.8,
                    "metadata": {"analyzer": "npm"}
                },
                "docker": {
                    "success": True,
                    "data": {"security_score": 90, "optimization": "medium"},
                    "execution_time": 1.5,
                    "metadata": {"analyzer": "docker"}
                }
            }
        }
        
        synthesis_result = await treequest_engine.synthesize_insights(demo_context)
        print("   Synthesis completed!")
        
        if "insights" in synthesis_result:
            insights = synthesis_result["insights"]
            print(f"   Summary: {insights.get('summary', 'N/A')}")
            print(f"   Confidence: {insights.get('confidence_score', 0):.2f}")
            print(f"   Risk Level: {insights.get('risk_assessment', 'N/A')}")
            
            if "agent_insights" in insights:
                print("   Agent Insights:")
                for agent, insight in insights["agent_insights"].items():
                    print(f"     {agent.title()}: {insight}")
        
        # Demo 2: Planning task
        print("\n📋 Demo 2: Execution Planning")
        print("-" * 30)
        
        planning_context = {
            "project_path": "/demo/project",
            "project_type": "web-application",
            "objective": "Improve security and performance",
            "constraints": "Maintain backward compatibility",
            "timeline": "2-3 weeks"
        }
        
        planning_result = await treequest_engine.solve("create_execution_plan", planning_context)
        print("   Planning completed!")
        
        if "treequest_metrics" in planning_result:
            metrics = planning_result["treequest_metrics"]
            print(f"   Agent Used: {metrics.get('agent_role_used', 'N/A')}")
            print(f"   Confidence: {metrics.get('best_node_reward', 0):.2f}")
            print(f"   Cost: ${metrics.get('final_cost_usd', 0):.4f}")
        
        # Demo 3: Enhanced slash commands
        print("\n⚡ Demo 3: Enhanced Slash Commands")
        print("-" * 30)
        
        from src.core.slash_command_processor import slash_processor
        
        # Initialize slash command processor
        await slash_processor.initialize()
        print("   ✅ Enhanced slash command processor initialized")
        
        # Test agents command
        print("\n   Testing /agents command...")
        agents_result = await slash_processor.execute_command(
            slash_processor.parse_command("/agents")
        )
        
        if agents_result["success"]:
            print("   ✅ /agents command working")
            # Show a snippet of the response
            response = agents_result["data"]
            lines = response.split('\n')[:5]
            for line in lines:
                print(f"     {line}")
        else:
            print(f"   ❌ /agents command failed: {agents_result['error']}")
        
        # Demo 4: Analysis coordination
        print("\n🔬 Demo 4: Enhanced Analysis Coordination")
        print("-" * 30)
        
        from src.analyzers.analyzer_coordinator import analyzer_coordinator
        
        # Initialize analyzer coordinator
        await analyzer_coordinator.initialize()
        print("   ✅ Enhanced analyzer coordinator initialized")
        
        # Get analyzer status
        status = await analyzer_coordinator.get_analyzer_status()
        print(f"   Total analyzers: {status['total_analyzers']}")
        print(f"   Available: {len(status['available_analyzers'])}")
        print(f"   Unavailable: {len(status['unavailable_analyzers'])}")
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Try these commands in the interactive CLI:")
        print("   /agents - Show available AI agents")
        print("   /plan - Create execution plan")
        print("   /deep-analyze - Multi-agent analysis")
        print("   /synthesize - Synthesize insights")
        print("   /critique - Code quality critique")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

async def demo_interactive_commands():
    """Demo interactive TreeQuest commands"""
    print("\n🎮 Interactive Commands Demo")
    print("=" * 40)
    
    try:
        from src.core.slash_command_processor import slash_processor
        await slash_processor.initialize()
        
        # Demo various commands
        commands_to_test = [
            "/help",
            "/agents",
            "/plan",
            "/deep-analyze path=. depth=deep",
            "/synthesize topic=project optimization",
            "/critique path=src/ focus=code quality"
        ]
        
        for cmd in commands_to_test:
            print(f"\n🔍 Testing: {cmd}")
            parsed_cmd = slash_processor.parse_command(cmd)
            if parsed_cmd:
                result = await slash_processor.execute_command(parsed_cmd)
                if result["success"]:
                    print(f"   ✅ Success")
                    # Show first line of response
                    response = result["data"]
                    first_line = response.split('\n')[0] if response else "No response"
                    print(f"   Response: {first_line[:60]}...")
                else:
                    print(f"   ❌ Failed: {result['error']}")
            else:
                print(f"   ❌ Failed to parse command")
        
        print("\n✅ Interactive commands demo completed!")
        
    except Exception as e:
        print(f"❌ Interactive demo failed: {e}")

async def main():
    """Main demo function"""
    print_header()
    
    # Run main demo
    await demo_treequest_agents()
    
    # Run interactive commands demo
    await demo_interactive_commands()
    
    print("\n🚀 Demo completed! Your Smart AI CLI is now enhanced with TreeQuest AI agents!")
    print("\nTo use the enhanced CLI:")
    print("   python smart_ai_enhanced_v3.py --treequest")
    print("   python smart_ai_enhanced_v3.py /agents")
    print("   python smart_ai_enhanced_v3.py /plan")

if __name__ == "__main__":
    asyncio.run(main())
