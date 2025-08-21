"""
MONK CLI Enhanced Interface
Phase 1 implementation with agent stack selection and memory-guided suggestions
"""
import asyncio
import click
import sys
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
import json
import time

from ..agents.orchestrator import orchestrator, TaskContext
from ..memory.memory_system import memory_system, MemoryQuery
from ..core.config import config
from ..core.database import startup_database, shutdown_database, get_db_session
from ..core.models import User
from .community_cli import community

console = Console()


class MONKCLIInterface:
    """Enhanced MONK CLI interface with agent and memory integration"""
    
    def __init__(self):
        self.current_user_id = None
        self.current_session = None
        self.agent_stack = "development"  # Default stack
        
    async def initialize(self):
        """Initialize the CLI interface"""
        await startup_database()
        await orchestrator.start()
        console.print("[bold green]MONK CLI initialized successfully[/bold green]")
    
    async def shutdown(self):
        """Shutdown the CLI interface"""
        await orchestrator.stop()
        await shutdown_database()
    
    async def authenticate_user(self, email: str = None) -> str:
        """Authenticate or create user"""
        if not email:
            email = Prompt.ask("Enter your email")
        
        async with get_db_session() as session:
            # Try to find existing user
            result = await session.execute(
                "SELECT id FROM users WHERE email = :email",
                {"email": email}
            )
            user_row = result.fetchone()
            
            if user_row:
                user_id = str(user_row[0])
                console.print(f"[green]Welcome back! User ID: {user_id[:8]}...[/green]")
            else:
                # Create new user
                user = User(email=email, username=email.split('@')[0])
                session.add(user)
                await session.commit()
                user_id = str(user.id)
                console.print(f"[green]New user created! User ID: {user_id[:8]}...[/green]")
        
        self.current_user_id = user_id
        return user_id
    
    async def execute_task(self, task_description: str, **kwargs):
        """Execute a task using the agent orchestrator"""
        if not self.current_user_id:
            console.print("[red]Please authenticate first using 'monk auth'[/red]")
            return
        
        # Create task context
        context = TaskContext(
            user_id=self.current_user_id,
            task_description=task_description,
            task_type=kwargs.get("task_type", "general"),
            domain=kwargs.get("domain", self.agent_stack),
            complexity_level=kwargs.get("complexity", 0.5),
            urgency_level=kwargs.get("urgency", 0.5),
            context_data=kwargs.get("context", {}),
            memory_context=await self._get_memory_context(task_description)
        )
        
        # Show task analysis
        self._display_task_analysis(context)
        
        # Execute with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_progress = progress.add_task("Analyzing and executing task...", total=None)
            
            try:
                response, selection_result = await orchestrator.execute_task(context)
                
                # Display results
                self._display_execution_results(response, selection_result, context)
                
                # Store interaction in memory
                await self._store_task_memory(context, response, selection_result)
                
                return response
                
            except Exception as e:
                console.print(f"[red]Task execution failed: {e}[/red]")
                return None
    
    async def _get_memory_context(self, task_description: str) -> List[Dict]:
        """Get relevant memory context for the task"""
        try:
            query = MemoryQuery(
                query_text=task_description,
                user_id=self.current_user_id,
                memory_types=["episodic"],
                limit=5,
                min_relevance_score=0.4
            )
            
            memories = await memory_system.retrieve_relevant_memories(query)
            return [memory.content for memory in memories.get("episodic", [])]
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not retrieve memory context: {e}[/yellow]")
            return []
    
    def _display_task_analysis(self, context: TaskContext):
        """Display task analysis panel"""
        analysis_table = Table(title="Task Analysis")
        analysis_table.add_column("Attribute", style="cyan")
        analysis_table.add_column("Value", style="white")
        
        analysis_table.add_row("Description", context.task_description[:100] + "...")
        analysis_table.add_row("Domain", context.domain)
        analysis_table.add_row("Complexity", f"{context.complexity_level:.1%}")
        analysis_table.add_row("Urgency", f"{context.urgency_level:.1%}")
        analysis_table.add_row("Memory Context", f"{len(context.memory_context)} relevant memories")
        
        console.print(analysis_table)
    
    def _display_execution_results(self, response, selection_result, context):
        """Display task execution results"""
        # Agent selection info
        selection_panel = Panel(
            f"[bold blue]Agent:[/bold blue] {selection_result.selected_agent.name}\n"
            f"[bold blue]Confidence:[/bold blue] {selection_result.confidence_score:.1%}\n"
            f"[bold blue]Selection Time:[/bold blue] {selection_result.selection_time_ms}ms\n"
            f"[bold blue]Reasoning:[/bold blue] {selection_result.selection_reasoning}",
            title="Agent Selection",
            border_style="blue"
        )
        console.print(selection_panel)
        
        # Execution results
        if response.success:
            result_panel = Panel(
                f"[bold green]Success![/bold green]\n"
                f"[bold]Execution Time:[/bold] {response.execution_time_ms}ms\n"
                f"[bold]Confidence:[/bold] {response.confidence_score:.1%}\n"
                f"[bold]Tokens Used:[/bold] {response.tokens_used}\n"
                f"[bold]Memory Queries:[/bold] {response.memory_queries_made}",
                title="Execution Results",
                border_style="green"
            )
            
            # Display actual results
            if response.result:
                console.print("\n[bold]Task Results:[/bold]")
                self._display_agent_results(response.result)
            
        else:
            result_panel = Panel(
                f"[bold red]Failed[/bold red]\n"
                f"[bold]Execution Time:[/bold] {response.execution_time_ms}ms\n"
                f"[bold]Error:[/bold] {response.error_message or 'Unknown error'}",
                title="Execution Results",
                border_style="red"
            )
        
        console.print(result_panel)
    
    def _display_agent_results(self, result: Dict[str, Any]):
        """Display agent-specific results"""
        if result.get("analysis_type") == "system_architecture":
            # Architecture analysis results
            if "recommendations" in result:
                console.print("\n[bold cyan]Architecture Recommendations:[/bold cyan]")
                for i, rec in enumerate(result["recommendations"], 1):
                    console.print(f"{i}. {rec}")
            
            if "complexity_assessment" in result:
                console.print(f"\n[bold]Complexity Assessment:[/bold] {result['complexity_assessment']:.1%}")
        
        elif result.get("analysis_type") == "quality_review":
            # Quality review results
            if "quality_score" in result:
                console.print(f"\n[bold]Quality Score:[/bold] {result['quality_score']:.1%}")
            
            if "issues_found" in result:
                console.print("\n[bold red]Issues Found:[/bold red]")
                for issue in result["issues_found"]:
                    console.print(f"• [{issue['severity'].upper()}] {issue['type']}: {issue['message']}")
            
            if "recommendations" in result:
                console.print("\n[bold cyan]Recommendations:[/bold cyan]")
                for rec in result["recommendations"]:
                    console.print(f"• {rec}")
        
        elif result.get("analysis_type") == "innovation_optimization":
            # Innovation results
            if "creative_solutions" in result:
                console.print("\n[bold magenta]Creative Solutions:[/bold magenta]")
                for solution in result["creative_solutions"]:
                    console.print(f"• {solution}")
            
            if "optimization_opportunities" in result:
                console.print("\n[bold yellow]Optimization Opportunities:[/bold yellow]")
                for opp in result["optimization_opportunities"]:
                    console.print(f"• {opp['area']}: {opp['potential_gain']} improvement")
        
        elif result.get("analysis_type") == "integration_deployment":
            # Integration results
            if "integration_plan" in result:
                console.print("\n[bold blue]Integration Plan:[/bold blue]")
                for i, step in enumerate(result["integration_plan"], 1):
                    console.print(f"{i}. {step}")
            
            if "risk_factors" in result:
                console.print("\n[bold orange]Risk Factors:[/bold orange]")
                for risk in result["risk_factors"]:
                    console.print(f"• {risk['factor']}: {risk['mitigation']}")
    
    async def _store_task_memory(self, context: TaskContext, response, selection_result):
        """Store task execution in memory system"""
        try:
            memory_content = {
                "task_description": context.task_description,
                "task_type": context.task_type,
                "domain": context.domain,
                "agent_used": selection_result.selected_agent.name,
                "agent_id": selection_result.selected_agent.agent_id,
                "success": response.success,
                "execution_time_ms": response.execution_time_ms,
                "confidence_score": response.confidence_score,
                "result_summary": self._summarize_result(response.result) if response.success else None,
                "error_message": response.error_message,
                "tokens_used": response.tokens_used,
                "memory_queries": response.memory_queries_made
            }
            
            await memory_system.store_interaction(
                user_id=self.current_user_id,
                interaction_type="task_execution",
                content=memory_content,
                context={
                    "complexity_level": context.complexity_level,
                    "urgency_level": context.urgency_level,
                    "agent_stack": self.agent_stack
                },
                importance_score=0.8 if response.success else 0.4
            )
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not store task in memory: {e}[/yellow]")
    
    def _summarize_result(self, result: Dict[str, Any]) -> str:
        """Create a brief summary of the result for memory storage"""
        if not result:
            return "No result"
        
        analysis_type = result.get("analysis_type", "unknown")
        
        if analysis_type == "system_architecture":
            rec_count = len(result.get("recommendations", []))
            return f"Architecture analysis with {rec_count} recommendations"
        
        elif analysis_type == "quality_review":
            issues_count = len(result.get("issues_found", []))
            quality_score = result.get("quality_score", 0)
            return f"Quality review: {quality_score:.1%} score, {issues_count} issues found"
        
        elif analysis_type == "innovation_optimization":
            solutions_count = len(result.get("creative_solutions", []))
            return f"Innovation analysis with {solutions_count} creative solutions"
        
        elif analysis_type == "integration_deployment":
            plan_steps = len(result.get("integration_plan", []))
            return f"Integration plan with {plan_steps} steps"
        
        return f"{analysis_type} completed successfully"
    
    async def show_memory_insights(self):
        """Display memory insights for current user"""
        if not self.current_user_id:
            console.print("[red]Please authenticate first[/red]")
            return
        
        insights = await memory_system.get_memory_insights(self.current_user_id)
        
        if not insights:
            console.print("[yellow]No memory insights available yet. Complete more tasks to build insights.[/yellow]")
            return
        
        console.print("\n[bold cyan]Memory Insights:[/bold cyan]")
        
        for insight in insights:
            insight_panel = Panel(
                f"[bold]{insight.description}[/bold]\n"
                f"Confidence: {insight.confidence_score:.1%}\n"
                f"Suggested Action: {insight.suggested_action or 'None'}",
                title=f"{insight.insight_type.replace('_', ' ').title()}",
                border_style="cyan"
            )
            console.print(insight_panel)
    
    async def show_agent_status(self):
        """Display current agent status"""
        status = orchestrator.get_orchestrator_status()
        
        # Orchestrator status
        orchestrator_table = Table(title="Agent Orchestrator Status")
        orchestrator_table.add_column("Metric", style="cyan")
        orchestrator_table.add_column("Value", style="white")
        
        orchestrator_table.add_row("Total Agents", str(status["total_agents"]))
        orchestrator_table.add_row("Tasks Processed", str(status["total_tasks_processed"]))
        orchestrator_table.add_row("Success Rate", f"{status['success_rate']:.1%}")
        orchestrator_table.add_row("Avg Selection Time", f"{status['average_selection_time_ms']:.0f}ms")
        
        console.print(orchestrator_table)
        
        # Individual agent status
        console.print("\n[bold]Individual Agent Status:[/bold]")
        agents_table = Table()
        agents_table.add_column("Agent", style="cyan")
        agents_table.add_column("Status", style="white")
        agents_table.add_column("Executions", style="white")
        agents_table.add_column("Success Rate", style="white")
        agents_table.add_column("Avg Time", style="white")
        
        for agent_id, agent_status in status["agents"].items():
            status_text = "🔴 Busy" if agent_status["is_busy"] else "🟢 Available"
            success_rate = (agent_status["total_executions"] > 0 and 
                           agent_status["success_rate"] or 0)
            
            agents_table.add_row(
                agent_status["name"],
                status_text,
                str(agent_status["total_executions"]),
                f"{success_rate:.1%}",
                f"{agent_status['average_execution_time_ms']:.0f}ms"
            )
        
        console.print(agents_table)
    
    async def set_agent_stack(self, stack_name: str):
        """Set the active agent stack"""
        available_stacks = ["development", "content", "business", "security"]
        
        if stack_name not in available_stacks:
            console.print(f"[red]Invalid stack. Available stacks: {', '.join(available_stacks)}[/red]")
            return
        
        self.agent_stack = stack_name
        console.print(f"[green]Active agent stack set to: {stack_name}[/green]")
        
        # Show available agents in stack
        status = orchestrator.get_orchestrator_status()
        stack_agents = [agent for agent_id, agent in status["agents"].items() 
                       if stack_name in agent.get("specializations", [])]
        
        if stack_agents:
            console.print(f"\n[bold]Available agents in {stack_name} stack:[/bold]")
            for agent in stack_agents:
                console.print(f"• {agent['name']}: {', '.join(agent['specializations'])}")
    
    async def search_memories(self, query_text: str, memory_type: str = None, limit: int = 5):
        """Search user memories"""
        if not self.current_user_id:
            console.print("[red]Please authenticate first[/red]")
            return
        
        memory_types = [memory_type] if memory_type else ["episodic"]
        
        query = MemoryQuery(
            query_text=query_text,
            user_id=self.current_user_id,
            memory_types=memory_types,
            limit=limit,
            min_relevance_score=0.3
        )
        
        memories = await memory_system.retrieve_relevant_memories(query)
        
        if not memories.get("episodic"):
            console.print("[yellow]No relevant memories found[/yellow]")
            return
        
        console.print(f"\n[bold cyan]Found {len(memories['episodic'])} relevant memories:[/bold cyan]")
        
        for i, memory in enumerate(memories["episodic"], 1):
            memory_panel = Panel(
                f"[bold]Type:[/bold] {memory.memory_type}\n"
                f"[bold]Relevance:[/bold] {memory.relevance_score:.1%}\n"
                f"[bold]Importance:[/bold] {memory.importance_score:.1%}\n"
                f"[bold]Created:[/bold] {memory.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                f"[bold]Content:[/bold] {self._summarize_memory_content(memory.content)}",
                title=f"Memory {i}",
                border_style="cyan"
            )
            console.print(memory_panel)
    
    def _summarize_memory_content(self, content: Dict[str, Any]) -> str:
        """Summarize memory content for display"""
        if "task_description" in content:
            return content["task_description"][:100] + "..."
        elif "description" in content:
            return content["description"][:100] + "..."
        else:
            # Try to find any text field
            for key, value in content.items():
                if isinstance(value, str) and len(value) > 10:
                    return f"{key}: {value[:80]}..."
            return "Complex content"


# CLI interface instance
cli_interface = MONKCLIInterface()


# Click command group
@click.group()
@click.version_option(version="1.0.0-phase1")
def monk():
    """MONK CLI - Phase 1 Foundation with Agent Specialization and Memory"""
    pass


# Add community intelligence commands
monk.add_command(community)


@monk.command()
@click.option("--email", help="User email for authentication")
def auth(email):
    """Authenticate or create user account"""
    async def _auth():
        await cli_interface.initialize()
        try:
            await cli_interface.authenticate_user(email)
        finally:
            await cli_interface.shutdown()
    
    asyncio.run(_auth())


@monk.command()
@click.argument("task_description")
@click.option("--stack", default="development", help="Agent stack to use")
@click.option("--complexity", default=0.5, type=float, help="Task complexity (0.0-1.0)")
@click.option("--urgency", default=0.5, type=float, help="Task urgency (0.0-1.0)")
def task(task_description, stack, complexity, urgency):
    """Execute a task using MONK agents"""
    async def _execute():
        await cli_interface.initialize()
        try:
            # Set agent stack
            await cli_interface.set_agent_stack(stack)
            
            # Execute task
            await cli_interface.execute_task(
                task_description,
                complexity=complexity,
                urgency=urgency,
                domain=stack
            )
        finally:
            await cli_interface.shutdown()
    
    asyncio.run(_execute())


@monk.command()
@click.argument("agents", nargs=-1)
@click.argument("task_description")
def collaborate(agents, task_description):
    """Execute a collaborative task with multiple agents"""
    async def _collaborate():
        await cli_interface.initialize()
        try:
            # Create context
            context = TaskContext(
                user_id=cli_interface.current_user_id or "demo-user",
                task_description=task_description,
                task_type="collaborative",
                domain="development",
                complexity_level=0.7,  # Collaborative tasks tend to be complex
                urgency_level=0.5
            )
            
            # Execute collaborative task
            results = await orchestrator.execute_collaborative_task(
                context, list(agents) if agents else None
            )
            
            # Display results
            console.print("\n[bold green]Collaborative Task Results:[/bold green]")
            for agent_type, response in results.items():
                console.print(f"\n[bold blue]{agent_type.replace('_', ' ').title()}:[/bold blue]")
                if response.success:
                    console.print(f"✅ Success ({response.execution_time_ms}ms)")
                    cli_interface._display_agent_results(response.result)
                else:
                    console.print(f"❌ Failed: {response.error_message}")
                    
        finally:
            await cli_interface.shutdown()
    
    asyncio.run(_collaborate())


@monk.command()
def status():
    """Show agent and system status"""
    async def _status():
        await cli_interface.initialize()
        try:
            await cli_interface.show_agent_status()
        finally:
            await cli_interface.shutdown()
    
    asyncio.run(_status())


@monk.command()
@click.argument("stack_name")
def stack(stack_name):
    """Set active agent stack"""
    async def _stack():
        await cli_interface.initialize()
        try:
            await cli_interface.set_agent_stack(stack_name)
        finally:
            await cli_interface.shutdown()
    
    asyncio.run(_stack())


@monk.command()
def insights():
    """Show memory insights"""
    async def _insights():
        await cli_interface.initialize()
        try:
            await cli_interface.show_memory_insights()
        finally:
            await cli_interface.shutdown()
    
    asyncio.run(_insights())


@monk.command()
@click.argument("query_text")
@click.option("--type", help="Memory type to search")
@click.option("--limit", default=5, type=int, help="Number of results")
def memory(query_text, type, limit):
    """Search memories"""
    async def _memory():
        await cli_interface.initialize()
        try:
            await cli_interface.search_memories(query_text, type, limit)
        finally:
            await cli_interface.shutdown()
    
    asyncio.run(_memory())


if __name__ == "__main__":
    monk()