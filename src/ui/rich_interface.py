"""
Rich Terminal Interface
Advanced terminal UI with progress bars, syntax highlighting, and interactive elements
"""

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from rich.syntax import Syntax
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt, Confirm
from rich.tree import Tree
import asyncio
from typing import Any, Dict, List

class RichCLI:
    def __init__(self):
        self.console = Console()
        self.current_progress = None
    
    async def show_help(self):
        """Display comprehensive help information"""
        console = Console()
        
        help_text = """
╭────────────────────────────────────────────────── Help ──────────────────────────────────────────────────╮
│                                                                                                          │
│ Monk CLI Enhanced                                                                                        │
│                                                                                                          │
│ COMMANDS:                                                                                                │
│   analyze      - Analyze project with all tools                                                          │
│   github     - GitHub repository analysis                                                                │
│   docker     - Docker optimization analysis                                                              │
│   npm        - NPM package analysis                                                                      │
│   git        - Git workflow analysis                                                                     │
│   ai         - AI-powered analysis and recommendations (TreeQuest)                                       │
│   workspace  - Workspace management                                                                      │
│                                                                                                          │
│ AI SUBCOMMANDS:                                                                                          │
│   ai plan .     - Generate AI-powered action plan                                                       │
│   ai fix .      - Generate AI-powered fixes and recommendations                                          │
│   ai explain .  - AI-powered project explanation and summary                                             │
│   ai models     - Show available AI models and their capabilities                                        │
│                                                                                                          │
│ OPTIONS:                                                                                                 │
│   --format json|markdown|table  - Output format                                                          │
│   --metrics                     - Show performance metrics                                               │
│   --cache                       - Use cached results                                                     │
│   --ai-recommendations          - Enable AI-powered insights (TreeQuest)                                 │
│   --reasoning-level {low,med,high} - Control AI reasoning depth                                          │
│   --max-depth <N>               - Maximum TreeQuest search depth                                         │
│   --rollout-budget <N>          - TreeQuest rollout iterations                                           │
│   --cost-cap <USD>              - Maximum cost for AI operations                                        │
│   --optimize {quality|latency|cost} - AI optimization objective                                          │
│                                                                                                          │
│ EXAMPLES:                                                                                                │
│   Monk analyze .                                                                                          │
│   Monk analyze . --ai-recommendations                                                                    │
│   Monk ai plan .                                                                                          │
│   Monk ai fix .                                                                                           │
│   Monk ai explain .                                                                                       │
│   Monk ai models                                                                                          │
│   Monk github --repo=owner/repo                                                                          │
│   Monk docker optimize                                                                                    │
│                                                                                                          │
│ TREEQUEST FEATURES:                                                                                      │
│   • Adaptive Branching Monte Carlo Tree Search                                                          │
│   • Multi-model orchestration (GPT-4, Claude, Mistral, Gemini)                                          │
│   • Intelligent model selection by role and objective                                                    │
│   • Cost-aware reasoning with configurable budgets                                                       │
│   • Cached results for improved performance                                                              │
│                                                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
        """
        
        console.print(help_text, style="cyan")
    
    async def display_result(self, result):
        """Display command result with rich formatting"""
        if not result.success:
            self.console.print(f"[red]Error:[/red] {result.error}")
            return
        
        # Check if this is a formatted table result (handle nested structure)
        actual_data = result.data
        if isinstance(result.data, dict) and 'data' in result.data:
            actual_data = result.data['data']
        
        if isinstance(actual_data, dict) and actual_data.get('format') == 'table':
            await self._display_formatted_table(actual_data)
            return
        
        # Create default result table
        table = Table(title="Analysis Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        if isinstance(result.data, dict):
            for key, value in result.data.items():
                status = "✅" if isinstance(value, (int, float)) and value > 70 else "⚠️"
                table.add_row(str(key), str(value), status)
        elif result.data is not None:
            # Handle non-dict data
            table.add_row("Result", str(result.data), "✅")
        
        self.console.print(table)
    
    async def _display_formatted_table(self, data):
        """Display a formatted table result"""
        table = Table(title=data.get('title', 'Results'))
        
        # Add columns based on headers
        for header in data.get('headers', []):
            table.add_column(header, style="cyan")
        
        # Add rows
        for row in data.get('rows', []):
            table.add_row(*[str(cell) for cell in row])
        
        # Display the table
        self.console.print(table)
        
        # Display summary if available
        if 'summary' in data:
            summary = data['summary']
            if isinstance(summary, dict):
                summary_table = Table(title="Summary")
                summary_table.add_column("Metric", style="cyan")
                summary_table.add_column("Value", style="green")
                
                for key, value in summary.items():
                    if key == 'cost_ranges' and isinstance(value, dict):
                        for cost_key, cost_value in value.items():
                            summary_table.add_row(f"Cost {cost_key.replace('_', ' ').title()}", str(cost_value))
                    else:
                        summary_table.add_row(key.replace('_', ' ').title(), str(value))
                
                self.console.print(summary_table)
    
    async def display_metrics(self, metrics):
        """Display performance metrics"""
        metrics_table = Table(title="Performance Metrics")
        metrics_table.add_column("Metric", style="blue")
        metrics_table.add_column("Value", style="green")
        
        metrics_table.add_row("Startup Time", f"{metrics.startup_time:.3f}s")
        metrics_table.add_row("Command Time", f"{metrics.command_execution_time:.3f}s")
        metrics_table.add_row("Memory Usage", f"{metrics.memory_usage:.1f}MB")
        metrics_table.add_row("Cache Hit Rate", f"{metrics.cache_hit_ratio:.1%}")
        
        self.console.print(metrics_table)
    
    def start_progress(self, description: str) -> TaskID:
        """Start progress bar"""
        if not self.current_progress:
            self.current_progress = Progress()
            self.current_progress.start()
        
        task_id = self.current_progress.add_task(description, total=100)
        return task_id
    
    def update_progress(self, task_id: TaskID, completed: int):
        """Update progress bar"""
        if self.current_progress:
            self.current_progress.update(task_id, completed=completed)
    
    def stop_progress(self):
        """Stop progress bar"""
        if self.current_progress:
            self.current_progress.stop()
            self.current_progress = None
