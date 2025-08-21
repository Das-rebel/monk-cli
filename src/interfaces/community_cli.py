"""
MONK CLI Community Intelligence Interface - Phase 2
CLI commands for research monitoring and capability enhancement
"""
import asyncio
import click
import json
from datetime import datetime, timedelta
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.text import Text

from ..community.intelligence_system import (
    community_intelligence, 
    ResearchFindingData, 
    CapabilityEnhancementPlan,
    SignificanceLevel,
    SourceType
)
from ..core.database import get_db_session
from ..core.models import ResearchFinding, CapabilityEnhancement

console = Console()


@click.group()
def community():
    """Community intelligence commands for research monitoring and enhancement"""
    pass


@community.command()
@click.option("--limit", "-l", default=20, help="Number of findings to show")
@click.option("--significance", "-s", type=click.Choice(["low", "medium", "high", "breakthrough"]), 
              help="Filter by significance level")
@click.option("--source", type=click.Choice(["arxiv", "github", "blog", "community"]), 
              help="Filter by source type")
@click.option("--days", "-d", default=7, help="Days back to search")
@click.option("--focus-area", "-f", help="Filter by focus area")
def research(limit: int, significance: Optional[str], source: Optional[str], 
            days: int, focus_area: Optional[str]):
    """Show latest research findings"""
    async def show_research():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Loading research findings...", total=None)
                
                # Get research findings
                async with get_db_session() as session:
                    query = """
                        SELECT * FROM research_findings 
                        WHERE discovered_at > :since_date
                    """
                    params = {"since_date": datetime.now() - timedelta(days=days)}
                    
                    # Add filters
                    if significance:
                        query += " AND significance_level = :significance"
                        params["significance"] = significance
                    
                    if source:
                        query += " AND source_type = :source"
                        params["source"] = source
                    
                    if focus_area:
                        query += " AND JSON_CONTAINS(focus_areas, :focus_area)"
                        params["focus_area"] = f'"{focus_area}"'
                    
                    query += " ORDER BY significance_score DESC, discovered_at DESC LIMIT :limit"
                    params["limit"] = limit
                    
                    result = await session.execute(query, params)
                    findings = result.fetchall()
                    
                    progress.update(task, completed=True)
            
            if not findings:
                console.print("[yellow]No research findings found with the specified criteria[/yellow]")
                return
            
            # Create table
            table = Table(title=f"🔬 Latest Research Findings ({len(findings)} results)")
            table.add_column("Significance", style="bold")
            table.add_column("Title", style="cyan", max_width=50)
            table.add_column("Source", style="green")
            table.add_column("Focus Areas", style="blue")
            table.add_column("Discovered", style="dim")
            
            for finding in findings:
                # Format significance with emoji
                sig_emoji = {
                    "breakthrough": "🚀",
                    "high": "⭐",
                    "medium": "📋",
                    "low": "📄"
                }.get(finding.significance_level, "📄")
                
                significance_text = f"{sig_emoji} {finding.significance_level}"
                if finding.significance_score:
                    significance_text += f" ({finding.significance_score:.1%})"
                
                # Format focus areas
                focus_areas = finding.focus_areas or []
                focus_text = ", ".join(focus_areas[:2])
                if len(focus_areas) > 2:
                    focus_text += f" +{len(focus_areas)-2}"
                
                # Format discovered date
                discovered = finding.discovered_at.strftime("%m/%d %H:%M")
                
                table.add_row(
                    significance_text,
                    finding.title[:50] + ("..." if len(finding.title) > 50 else ""),
                    finding.source_type,
                    focus_text,
                    discovered
                )
            
            console.print(table)
            
            # Show summary stats
            sig_counts = {}
            for finding in findings:
                sig_counts[finding.significance_level] = sig_counts.get(finding.significance_level, 0) + 1
            
            summary_text = "Summary: "
            summary_parts = []
            for level, count in sig_counts.items():
                emoji = {"breakthrough": "🚀", "high": "⭐", "medium": "📋", "low": "📄"}.get(level, "📄")
                summary_parts.append(f"{emoji} {count} {level}")
            
            console.print(f"\n[dim]{summary_text}{', '.join(summary_parts)}[/dim]")
            
        except Exception as e:
            console.print(f"[red]Error loading research findings: {e}[/red]")
    
    asyncio.run(show_research())


@community.command()
@click.argument("finding_id")
def show(finding_id: str):
    """Show detailed research finding"""
    async def show_finding():
        try:
            async with get_db_session() as session:
                finding = await session.get(ResearchFinding, finding_id)
                
                if not finding:
                    console.print(f"[red]Research finding {finding_id} not found[/red]")
                    return
                
                # Create detailed view
                layout = Layout()
                layout.split_column(
                    Layout(name="header", size=8),
                    Layout(name="content", ratio=1),
                    Layout(name="metadata", size=6)
                )
                
                # Header
                sig_emoji = {
                    "breakthrough": "🚀",
                    "high": "⭐", 
                    "medium": "📋",
                    "low": "📄"
                }.get(finding.significance_level, "📄")
                
                header_text = f"{sig_emoji} {finding.title}"
                layout["header"].update(Panel(
                    Text(header_text, style="bold cyan"),
                    title=f"Research Finding: {finding.source_type.upper()}"
                ))
                
                # Content
                content_text = finding.summary
                if finding.full_content and finding.full_content != finding.summary:
                    content_text = finding.full_content
                
                layout["content"].update(Panel(
                    Text(content_text, style="white"),
                    title="Summary & Content"
                ))
                
                # Metadata
                metadata_lines = [
                    f"🔗 Source: {finding.source_url}",
                    f"📊 Significance: {finding.significance_score:.1%} ({finding.significance_level})",
                    f"⚡ Implementation Potential: {finding.implementation_potential:.1%}",
                    f"👥 Community Interest: {finding.community_interest:.1%}",
                    f"📅 Discovered: {finding.discovered_at.strftime('%Y-%m-%d %H:%M')}",
                    f"🎯 Focus Areas: {', '.join(finding.focus_areas or [])}",
                    f"🏷️ Tags: {', '.join(finding.tags or [])}",
                    f"👤 Authors: {', '.join(finding.authors or [])}"
                ]
                
                layout["metadata"].update(Panel(
                    "\n".join(metadata_lines),
                    title="Metadata"
                ))
                
                console.print(layout)
                
                # Check if enhancement plan exists
                enhancement_result = await session.execute(
                    "SELECT id, status FROM capability_enhancements WHERE research_finding_id = :finding_id",
                    {"finding_id": finding_id}
                )
                enhancement = enhancement_result.fetchone()
                
                if enhancement:
                    console.print(f"\n[green]✅ Enhancement plan exists: {enhancement.id} (status: {enhancement.status})[/green]")
                    console.print(f"[dim]Use 'monk community enhancements --id {enhancement.id}' to view details[/dim]")
                else:
                    if finding.significance_level in ["high", "breakthrough"]:
                        console.print(f"\n[yellow]💡 This finding could benefit from an enhancement plan[/yellow]")
                        console.print(f"[dim]Use 'monk community enhance {finding_id}' to create one[/dim]")
                
        except Exception as e:
            console.print(f"[red]Error showing finding: {e}[/red]")
    
    asyncio.run(show_finding())


@community.command()
@click.option("--status", type=click.Choice(["planned", "in_progress", "testing", "deployed", "cancelled"]),
              help="Filter by status")
@click.option("--priority", type=click.Choice(["low", "medium", "high", "critical"]),
              help="Filter by priority")
@click.option("--limit", "-l", default=20, help="Number of enhancements to show")
@click.option("--id", "enhancement_id", help="Show specific enhancement by ID")
def enhancements(status: Optional[str], priority: Optional[str], limit: int, enhancement_id: Optional[str]):
    """Show capability enhancement plans"""
    async def show_enhancements():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Loading enhancement plans...", total=None)
                
                async with get_db_session() as session:
                    if enhancement_id:
                        # Show specific enhancement
                        enhancement = await session.get(CapabilityEnhancement, enhancement_id)
                        if not enhancement:
                            console.print(f"[red]Enhancement plan {enhancement_id} not found[/red]")
                            return
                        
                        progress.update(task, completed=True)
                        
                        # Show detailed enhancement view
                        await show_enhancement_detail(enhancement)
                        return
                    
                    # Build query for list view
                    query = "SELECT * FROM capability_enhancements WHERE 1=1"
                    params = {}
                    
                    if status:
                        query += " AND status = :status"
                        params["status"] = status
                    
                    if priority:
                        query += " AND priority = :priority"
                        params["priority"] = priority
                    
                    query += " ORDER BY estimated_impact DESC, created_at DESC LIMIT :limit"
                    params["limit"] = limit
                    
                    result = await session.execute(query, params)
                    enhancements = result.fetchall()
                    
                    progress.update(task, completed=True)
            
            if not enhancements:
                console.print("[yellow]No enhancement plans found with the specified criteria[/yellow]")
                return
            
            # Create table
            table = Table(title=f"🚀 Capability Enhancement Plans ({len(enhancements)} results)")
            table.add_column("Priority", style="bold")
            table.add_column("Title", style="cyan", max_width=40)
            table.add_column("Status", style="green")
            table.add_column("Impact", style="blue")
            table.add_column("Complexity", style="yellow")
            table.add_column("Time Est.", style="dim")
            table.add_column("Assigned", style="magenta")
            
            for enhancement in enhancements:
                # Format priority with emoji
                priority_emoji = {
                    "critical": "🔥",
                    "high": "⚡",
                    "medium": "📋",
                    "low": "💭"
                }.get(enhancement.priority, "📋")
                
                # Format status with emoji
                status_emoji = {
                    "planned": "📝",
                    "in_progress": "⚙️",
                    "testing": "🧪",
                    "deployed": "✅",
                    "cancelled": "❌"
                }.get(enhancement.status, "📝")
                
                table.add_row(
                    f"{priority_emoji} {enhancement.priority}",
                    enhancement.title[:40] + ("..." if len(enhancement.title) > 40 else ""),
                    f"{status_emoji} {enhancement.status}",
                    f"{enhancement.estimated_impact:.1%}",
                    f"{enhancement.implementation_complexity:.1f}",
                    f"{enhancement.development_time_days}d",
                    enhancement.assigned_to or "Unassigned"
                )
            
            console.print(table)
            
            # Show summary stats
            status_counts = {}
            for enhancement in enhancements:
                status_counts[enhancement.status] = status_counts.get(enhancement.status, 0) + 1
            
            summary_text = "Summary: "
            summary_parts = []
            for status, count in status_counts.items():
                emoji = {"planned": "📝", "in_progress": "⚙️", "testing": "🧪", "deployed": "✅", "cancelled": "❌"}.get(status, "📝")
                summary_parts.append(f"{emoji} {count} {status}")
            
            console.print(f"\n[dim]{summary_text}{', '.join(summary_parts)}[/dim]")
            
        except Exception as e:
            console.print(f"[red]Error loading enhancement plans: {e}[/red]")
    
    asyncio.run(show_enhancements())


async def show_enhancement_detail(enhancement):
    """Show detailed enhancement plan"""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=6),
        Layout(name="plan", ratio=1),
        Layout(name="status", size=8)
    )
    
    # Header
    priority_emoji = {"critical": "🔥", "high": "⚡", "medium": "📋", "low": "💭"}.get(enhancement.priority, "📋")
    status_emoji = {"planned": "📝", "in_progress": "⚙️", "testing": "🧪", "deployed": "✅", "cancelled": "❌"}.get(enhancement.status, "📝")
    
    header_text = f"{priority_emoji} {enhancement.title}\n{status_emoji} Status: {enhancement.status}"
    layout["header"].update(Panel(
        Text(header_text, style="bold cyan"),
        title="Enhancement Plan"
    ))
    
    # Plan details
    plan_text = enhancement.description
    layout["plan"].update(Panel(
        Text(plan_text, style="white"),
        title="Description & Plan"
    ))
    
    # Status and metadata
    status_lines = [
        f"📊 Estimated Impact: {enhancement.estimated_impact:.1%}",
        f"⚙️ Implementation Complexity: {enhancement.implementation_complexity:.1f}/1.0",
        f"⏱️ Estimated Time: {enhancement.development_time_days} days",
        f"👤 Assigned To: {enhancement.assigned_to or 'Unassigned'}",
        f"📅 Created: {enhancement.created_at.strftime('%Y-%m-%d %H:%M')}",
        ""
    ]
    
    if enhancement.started_at:
        status_lines.append(f"▶️ Started: {enhancement.started_at.strftime('%Y-%m-%d %H:%M')}")
    if enhancement.completed_at:
        status_lines.append(f"✅ Completed: {enhancement.completed_at.strftime('%Y-%m-%d %H:%M')}")
    if enhancement.deployed_at:
        status_lines.append(f"🚀 Deployed: {enhancement.deployed_at.strftime('%Y-%m-%d %H:%M')}")
    
    if enhancement.actual_impact_score:
        status_lines.append(f"📈 Actual Impact: {enhancement.actual_impact_score:.1%}")
    if enhancement.user_feedback_score:
        status_lines.append(f"👍 User Feedback: {enhancement.user_feedback_score:.1%}")
    
    layout["status"].update(Panel(
        "\n".join(status_lines),
        title="Status & Metrics"
    ))
    
    console.print(layout)


@community.command()
@click.argument("finding_id")
@click.option("--title", prompt="Enhancement title", help="Title for the enhancement")
@click.option("--description", prompt="Enhancement description", help="Description of the enhancement")
@click.option("--priority", type=click.Choice(["low", "medium", "high", "critical"]), 
              default="medium", help="Priority level")
def enhance(finding_id: str, title: str, description: str, priority: str):
    """Create enhancement plan from research finding"""
    async def create_enhancement():
        try:
            async with get_db_session() as session:
                # Check if finding exists
                finding = await session.get(ResearchFinding, finding_id)
                if not finding:
                    console.print(f"[red]Research finding {finding_id} not found[/red]")
                    return
                
                # Check if enhancement already exists
                existing_result = await session.execute(
                    "SELECT id FROM capability_enhancements WHERE research_finding_id = :finding_id",
                    {"finding_id": finding_id}
                )
                existing = existing_result.fetchone()
                if existing:
                    console.print(f"[yellow]Enhancement plan already exists: {existing.id}[/yellow]")
                    return
                
                # Generate enhancement plan using community intelligence
                finding_data = ResearchFindingData(
                    id=finding.id,
                    title=finding.title,
                    summary=finding.summary,
                    source_url=finding.source_url,
                    source_type=SourceType(finding.source_type),
                    discovered_at=finding.discovered_at,
                    significance_score=finding.significance_score,
                    significance_level=SignificanceLevel(finding.significance_level),
                    focus_areas=finding.focus_areas or [],
                    implementation_potential=finding.implementation_potential,
                    community_interest=finding.community_interest,
                    authors=finding.authors or [],
                    tags=finding.tags or [],
                    full_content=finding.full_content or "",
                    metadata=finding.metadata or {}
                )
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Generating enhancement plan...", total=None)
                    
                    enhancement_plan = await community_intelligence.capability_enhancer.generate_enhancement_plan(finding_data)
                    
                    progress.update(task, completed=True)
                
                if not enhancement_plan:
                    console.print("[yellow]Could not generate enhancement plan for this finding[/yellow]")
                    console.print("[dim]The finding may not meet the criteria for automatic enhancement generation[/dim]")
                    return
                
                # Override with user input
                enhancement_plan.title = title
                enhancement_plan.description = description
                
                # Create enhancement in database
                enhancement_id = f"enhancement_{finding_id}"
                enhancement = CapabilityEnhancement(
                    id=enhancement_id,
                    research_finding_id=finding_id,
                    title=title,
                    description=description,
                    implementation_complexity=enhancement_plan.implementation_complexity,
                    estimated_impact=enhancement_plan.estimated_impact,
                    development_time_days=enhancement_plan.development_time_days,
                    required_resources=enhancement_plan.required_resources,
                    implementation_plan=enhancement_plan.implementation_plan,
                    testing_strategy=enhancement_plan.testing_strategy,
                    deployment_strategy=enhancement_plan.deployment_strategy,
                    risk_assessment=enhancement_plan.risk_assessment,
                    status="planned",
                    priority=priority,
                    created_at=datetime.now()
                )
                
                session.add(enhancement)
                await session.commit()
                
                console.print(f"[green]✅ Enhancement plan created: {enhancement_id}[/green]")
                console.print(f"[dim]Use 'monk community enhancements --id {enhancement_id}' to view details[/dim]")
                
        except Exception as e:
            console.print(f"[red]Error creating enhancement plan: {e}[/red]")
    
    asyncio.run(create_enhancement())


@community.command()
@click.argument("enhancement_id")
@click.argument("status", type=click.Choice(["planned", "in_progress", "testing", "deployed", "cancelled"]))
@click.option("--assigned-to", help="Assign to person")
@click.option("--feedback-score", type=float, help="User feedback score (0.0-1.0)")
@click.option("--impact-score", type=float, help="Actual impact score (0.0-1.0)")
def update_status(enhancement_id: str, status: str, assigned_to: Optional[str], 
                 feedback_score: Optional[float], impact_score: Optional[float]):
    """Update enhancement plan status"""
    async def update_enhancement():
        try:
            async with get_db_session() as session:
                enhancement = await session.get(CapabilityEnhancement, enhancement_id)
                if not enhancement:
                    console.print(f"[red]Enhancement plan {enhancement_id} not found[/red]")
                    return
                
                old_status = enhancement.status
                enhancement.status = status
                
                # Update timestamps
                if status == "in_progress" and old_status == "planned":
                    enhancement.started_at = datetime.now()
                elif status == "deployed" and old_status in ["testing", "in_progress"]:
                    enhancement.deployed_at = datetime.now()
                    if enhancement.started_at:
                        enhancement.actual_development_time_days = (datetime.now() - enhancement.started_at).days
                elif status == "testing" and old_status == "in_progress":
                    enhancement.completed_at = datetime.now()
                
                # Update assignments and feedback
                if assigned_to:
                    enhancement.assigned_to = assigned_to
                    enhancement.assigned_at = datetime.now()
                
                if feedback_score is not None:
                    enhancement.user_feedback_score = feedback_score
                
                if impact_score is not None:
                    enhancement.actual_impact_score = impact_score
                
                await session.commit()
                
                console.print(f"[green]✅ Enhancement status updated: {old_status} → {status}[/green]")
                
                if assigned_to:
                    console.print(f"[blue]👤 Assigned to: {assigned_to}[/blue]")
                
                if feedback_score is not None:
                    console.print(f"[blue]👍 User feedback: {feedback_score:.1%}[/blue]")
                
                if impact_score is not None:
                    console.print(f"[blue]📈 Impact score: {impact_score:.1%}[/blue]")
                
        except Exception as e:
            console.print(f"[red]Error updating enhancement status: {e}[/red]")
    
    asyncio.run(update_enhancement())


@community.command()
def status():
    """Show community intelligence system status"""
    async def show_status():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Getting system status...", total=None)
                
                status_data = await community_intelligence.get_system_status()
                
                progress.update(task, completed=True)
            
            # Create status display
            layout = Layout()
            layout.split_column(
                Layout(name="system", size=8),
                Layout(name="metrics", ratio=1),
                Layout(name="sources", size=6)
            )
            
            # System status
            status_emoji = "🟢" if status_data["status"] == "active" else "🔴"
            last_update = status_data.get("last_update_cycle")
            last_update_text = last_update if last_update else "Never"
            
            system_text = f"{status_emoji} Status: {status_data['status']}\n📅 Last Update: {last_update_text}"
            
            layout["system"].update(Panel(
                Text(system_text, style="bold"),
                title="🧘 Community Intelligence System"
            ))
            
            # Metrics
            metrics_lines = [
                f"📊 Total Research Findings: {status_data['total_research_findings']}",
                f"🚀 Total Enhancement Plans: {status_data['total_enhancement_plans']}",
                f"📈 Recent Findings (7 days): {status_data['recent_findings_7_days']}",
                ""
            ]
            
            layout["metrics"].update(Panel(
                "\n".join(metrics_lines),
                title="📊 Metrics"
            ))
            
            # Active sources
            sources_text = "\n".join([f"✅ {source}" for source in status_data["research_sources"]])
            monitors_text = "\n".join([f"🔍 {monitor}" for monitor in status_data["active_monitors"]])
            
            layout["sources"].update(Panel(
                f"Research Sources:\n{sources_text}\n\nMonitors:\n{monitors_text}",
                title="🔍 Active Sources"
            ))
            
            console.print(layout)
            
        except Exception as e:
            console.print(f"[red]Error getting system status: {e}[/red]")
    
    asyncio.run(show_status())


@community.command()
def start_monitoring():
    """Start community intelligence monitoring"""
    async def start():
        try:
            if not community_intelligence.running:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Starting monitoring...", total=None)
                    
                    await community_intelligence.start_monitoring()
                    
                    progress.update(task, completed=True)
                
                console.print("[green]✅ Community intelligence monitoring started[/green]")
            else:
                console.print("[yellow]⚠️ Monitoring is already running[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error starting monitoring: {e}[/red]")
    
    asyncio.run(start())


@community.command()
def stop_monitoring():
    """Stop community intelligence monitoring"""
    async def stop():
        try:
            if community_intelligence.running:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Stopping monitoring...", total=None)
                    
                    await community_intelligence.stop_monitoring()
                    
                    progress.update(task, completed=True)
                
                console.print("[green]✅ Community intelligence monitoring stopped[/green]")
            else:
                console.print("[yellow]⚠️ Monitoring is not running[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error stopping monitoring: {e}[/red]")
    
    asyncio.run(stop())


@community.command()
def trigger_update():
    """Trigger manual research update cycle"""
    async def trigger():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Running update cycle...", total=None)
                
                await community_intelligence._run_update_cycle()
                
                progress.update(task, completed=True)
            
            console.print("[green]✅ Manual update cycle completed[/green]")
            console.print("[dim]Use 'monk community research' to see new findings[/dim]")
            
        except Exception as e:
            console.print(f"[red]Error triggering update: {e}[/red]")
    
    asyncio.run(trigger())


if __name__ == "__main__":
    community()