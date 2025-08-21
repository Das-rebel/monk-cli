"""
MONK CLI Community Intelligence API Endpoints - Phase 2
Research monitoring and capability enhancement API
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from ..core.database import get_db_session
from ..core.models import ResearchFinding, CapabilityEnhancement, CommunityIntelligence
from ..community.intelligence_system import (
    community_intelligence, 
    ResearchFindingData, 
    CapabilityEnhancementPlan,
    SignificanceLevel,
    SourceType
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/community", tags=["community"])


# Pydantic models for API requests/responses
class ResearchFindingResponse(BaseModel):
    """Response model for research findings"""
    id: str
    title: str
    summary: str
    source_url: str
    source_type: str
    discovered_at: str
    significance_score: float
    significance_level: str
    focus_areas: List[str]
    implementation_potential: float
    community_interest: float
    authors: List[str]
    tags: List[str]
    metadata: Dict[str, Any]


class CapabilityEnhancementResponse(BaseModel):
    """Response model for capability enhancements"""
    id: str
    research_finding_id: str
    title: str
    description: str
    implementation_complexity: float
    estimated_impact: float
    development_time_days: int
    required_resources: List[str]
    status: str
    priority: str
    created_at: str
    assigned_to: Optional[str] = None
    actual_impact_score: Optional[float] = None


class CommunityIntelligenceStatusResponse(BaseModel):
    """Response model for community intelligence system status"""
    status: str
    last_update_cycle: Optional[str] = None
    total_research_findings: int
    total_enhancement_plans: int
    recent_findings_7_days: int
    active_monitors: List[str]
    research_sources: List[str]
    next_scheduled_update: Optional[str] = None


class EnhancementPlanRequest(BaseModel):
    """Request to create or update enhancement plan"""
    title: str
    description: str
    priority: str = Field(default="medium", pattern="^(low|medium|high|critical)$")
    assigned_to: Optional[str] = None


class ResearchQueryRequest(BaseModel):
    """Request for research query"""
    query_text: str
    focus_areas: Optional[List[str]] = None
    significance_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    limit: int = Field(default=20, ge=1, le=100)
    source_types: Optional[List[str]] = None


# Research findings endpoints
@router.get("/research/findings", response_model=List[ResearchFindingResponse])
async def get_research_findings(
    limit: int = Query(default=50, ge=1, le=200),
    significance_level: Optional[str] = Query(default=None),
    source_type: Optional[str] = Query(default=None),
    focus_area: Optional[str] = Query(default=None),
    days_back: int = Query(default=30, ge=1, le=365)
):
    """Get research findings with optional filtering"""
    try:
        async with get_db_session() as session:
            # Build query
            query = """
                SELECT * FROM research_findings 
                WHERE discovered_at > :since_date
            """
            params = {"since_date": datetime.now() - timedelta(days=days_back)}
            
            # Add filters
            if significance_level:
                query += " AND significance_level = :significance_level"
                params["significance_level"] = significance_level
            
            if source_type:
                query += " AND source_type = :source_type"
                params["source_type"] = source_type
            
            if focus_area:
                query += " AND JSON_CONTAINS(focus_areas, :focus_area)"
                params["focus_area"] = f'"{focus_area}"'
            
            query += " ORDER BY significance_score DESC, discovered_at DESC LIMIT :limit"
            params["limit"] = limit
            
            result = await session.execute(query, params)
            rows = result.fetchall()
            
            findings = []
            for row in rows:
                finding = ResearchFindingResponse(
                    id=row.id,
                    title=row.title,
                    summary=row.summary,
                    source_url=row.source_url,
                    source_type=row.source_type,
                    discovered_at=row.discovered_at.isoformat(),
                    significance_score=row.significance_score,
                    significance_level=row.significance_level,
                    focus_areas=row.focus_areas or [],
                    implementation_potential=row.implementation_potential,
                    community_interest=row.community_interest,
                    authors=row.authors or [],
                    tags=row.tags or [],
                    metadata=row.metadata or {}
                )
                findings.append(finding)
            
            return findings
            
    except Exception as e:
        logger.error(f"Error getting research findings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/research/findings/{finding_id}", response_model=ResearchFindingResponse)
async def get_research_finding(finding_id: str):
    """Get specific research finding by ID"""
    try:
        async with get_db_session() as session:
            finding = await session.get(ResearchFinding, finding_id)
            
            if not finding:
                raise HTTPException(status_code=404, detail="Research finding not found")
            
            return ResearchFindingResponse(
                id=finding.id,
                title=finding.title,
                summary=finding.summary,
                source_url=finding.source_url,
                source_type=finding.source_type,
                discovered_at=finding.discovered_at.isoformat(),
                significance_score=finding.significance_score,
                significance_level=finding.significance_level,
                focus_areas=finding.focus_areas or [],
                implementation_potential=finding.implementation_potential,
                community_interest=finding.community_interest,
                authors=finding.authors or [],
                tags=finding.tags or [],
                metadata=finding.metadata or {}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting research finding {finding_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research/query", response_model=List[ResearchFindingResponse])
async def query_research_findings(request: ResearchQueryRequest):
    """Query research findings with semantic search"""
    try:
        # For Phase 2, implement basic text search
        # Phase 3 would add vector similarity search
        async with get_db_session() as session:
            query = """
                SELECT * FROM research_findings 
                WHERE (title ILIKE :query OR summary ILIKE :query)
                AND significance_score >= :threshold
            """
            params = {
                "query": f"%{request.query_text}%",
                "threshold": request.significance_threshold
            }
            
            # Add focus area filter
            if request.focus_areas:
                focus_conditions = []
                for i, area in enumerate(request.focus_areas):
                    condition = f"JSON_CONTAINS(focus_areas, :focus_area_{i})"
                    focus_conditions.append(condition)
                    params[f"focus_area_{i}"] = f'"{area}"'
                
                if focus_conditions:
                    query += f" AND ({' OR '.join(focus_conditions)})"
            
            # Add source type filter
            if request.source_types:
                source_conditions = " OR ".join([f"source_type = :source_{i}" for i in range(len(request.source_types))])
                query += f" AND ({source_conditions})"
                for i, source in enumerate(request.source_types):
                    params[f"source_{i}"] = source
            
            query += " ORDER BY significance_score DESC LIMIT :limit"
            params["limit"] = request.limit
            
            result = await session.execute(query, params)
            rows = result.fetchall()
            
            findings = []
            for row in rows:
                finding = ResearchFindingResponse(
                    id=row.id,
                    title=row.title,
                    summary=row.summary,
                    source_url=row.source_url,
                    source_type=row.source_type,
                    discovered_at=row.discovered_at.isoformat(),
                    significance_score=row.significance_score,
                    significance_level=row.significance_level,
                    focus_areas=row.focus_areas or [],
                    implementation_potential=row.implementation_potential,
                    community_interest=row.community_interest,
                    authors=row.authors or [],
                    tags=row.tags or [],
                    metadata=row.metadata or {}
                )
                findings.append(finding)
            
            return findings
            
    except Exception as e:
        logger.error(f"Error querying research findings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Enhancement plans endpoints
@router.get("/enhancements", response_model=List[CapabilityEnhancementResponse])
async def get_enhancement_plans(
    status: Optional[str] = Query(default=None),
    priority: Optional[str] = Query(default=None),
    assigned_to: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200)
):
    """Get capability enhancement plans with optional filtering"""
    try:
        async with get_db_session() as session:
            query = "SELECT * FROM capability_enhancements WHERE 1=1"
            params = {}
            
            # Add filters
            if status:
                query += " AND status = :status"
                params["status"] = status
            
            if priority:
                query += " AND priority = :priority"
                params["priority"] = priority
            
            if assigned_to:
                query += " AND assigned_to = :assigned_to"
                params["assigned_to"] = assigned_to
            
            query += " ORDER BY estimated_impact DESC, created_at DESC LIMIT :limit"
            params["limit"] = limit
            
            result = await session.execute(query, params)
            rows = result.fetchall()
            
            enhancements = []
            for row in rows:
                enhancement = CapabilityEnhancementResponse(
                    id=row.id,
                    research_finding_id=row.research_finding_id,
                    title=row.title,
                    description=row.description,
                    implementation_complexity=row.implementation_complexity,
                    estimated_impact=row.estimated_impact,
                    development_time_days=row.development_time_days,
                    required_resources=row.required_resources or [],
                    status=row.status,
                    priority=row.priority,
                    created_at=row.created_at.isoformat(),
                    assigned_to=row.assigned_to,
                    actual_impact_score=row.actual_impact_score
                )
                enhancements.append(enhancement)
            
            return enhancements
            
    except Exception as e:
        logger.error(f"Error getting enhancement plans: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enhancements/{enhancement_id}", response_model=Dict[str, Any])
async def get_enhancement_plan(enhancement_id: str):
    """Get detailed enhancement plan by ID"""
    try:
        async with get_db_session() as session:
            enhancement = await session.get(CapabilityEnhancement, enhancement_id)
            
            if not enhancement:
                raise HTTPException(status_code=404, detail="Enhancement plan not found")
            
            return {
                "id": enhancement.id,
                "research_finding_id": enhancement.research_finding_id,
                "title": enhancement.title,
                "description": enhancement.description,
                "implementation_complexity": enhancement.implementation_complexity,
                "estimated_impact": enhancement.estimated_impact,
                "development_time_days": enhancement.development_time_days,
                "required_resources": enhancement.required_resources or [],
                "implementation_plan": enhancement.implementation_plan or {},
                "testing_strategy": enhancement.testing_strategy or {},
                "deployment_strategy": enhancement.deployment_strategy or {},
                "risk_assessment": enhancement.risk_assessment or {},
                "status": enhancement.status,
                "priority": enhancement.priority,
                "created_at": enhancement.created_at.isoformat(),
                "started_at": enhancement.started_at.isoformat() if enhancement.started_at else None,
                "completed_at": enhancement.completed_at.isoformat() if enhancement.completed_at else None,
                "deployed_at": enhancement.deployed_at.isoformat() if enhancement.deployed_at else None,
                "assigned_to": enhancement.assigned_to,
                "assigned_at": enhancement.assigned_at.isoformat() if enhancement.assigned_at else None,
                "actual_development_time_days": enhancement.actual_development_time_days,
                "actual_impact_score": enhancement.actual_impact_score,
                "user_feedback_score": enhancement.user_feedback_score,
                "performance_impact": enhancement.performance_impact or {}
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting enhancement plan {enhancement_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enhancements/{finding_id}")
async def create_enhancement_plan(finding_id: str, request: EnhancementPlanRequest):
    """Create enhancement plan from research finding"""
    try:
        async with get_db_session() as session:
            # Check if finding exists
            finding = await session.get(ResearchFinding, finding_id)
            if not finding:
                raise HTTPException(status_code=404, detail="Research finding not found")
            
            # Check if enhancement plan already exists
            existing_result = await session.execute(
                "SELECT id FROM capability_enhancements WHERE research_finding_id = :finding_id",
                {"finding_id": finding_id}
            )
            existing = existing_result.fetchone()
            if existing:
                raise HTTPException(status_code=400, detail="Enhancement plan already exists for this finding")
            
            # Create enhancement plan
            enhancement_id = f"enhancement_{finding_id}"
            enhancement = CapabilityEnhancement(
                id=enhancement_id,
                research_finding_id=finding_id,
                title=request.title,
                description=request.description,
                implementation_complexity=0.5,  # Default, should be calculated
                estimated_impact=finding.significance_score,
                development_time_days=7,  # Default
                required_resources=["Development Team"],
                status="planned",
                priority=request.priority,
                assigned_to=request.assigned_to,
                assigned_at=datetime.now() if request.assigned_to else None,
                created_at=datetime.now()
            )
            
            session.add(enhancement)
            await session.commit()
            
            return {"enhancement_id": enhancement_id, "status": "created"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating enhancement plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/enhancements/{enhancement_id}/status")
async def update_enhancement_status(
    enhancement_id: str, 
    status: str,
    background_tasks: BackgroundTasks,
    assigned_to: Optional[str] = None,
    user_feedback_score: Optional[float] = None,
    actual_impact_score: Optional[float] = None
):
    """Update enhancement plan status"""
    try:
        valid_statuses = ["planned", "in_progress", "testing", "deployed", "cancelled"]
        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
        
        async with get_db_session() as session:
            enhancement = await session.get(CapabilityEnhancement, enhancement_id)
            if not enhancement:
                raise HTTPException(status_code=404, detail="Enhancement plan not found")
            
            # Update status and timestamps
            old_status = enhancement.status
            enhancement.status = status
            
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
            
            if user_feedback_score is not None:
                enhancement.user_feedback_score = user_feedback_score
            
            if actual_impact_score is not None:
                enhancement.actual_impact_score = actual_impact_score
            
            await session.commit()
            
            # Update community intelligence metrics in background
            background_tasks.add_task(update_community_intelligence_metrics)
            
            return {"status": "updated", "new_status": status}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating enhancement status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System status and monitoring endpoints
@router.get("/status", response_model=CommunityIntelligenceStatusResponse)
async def get_community_intelligence_status():
    """Get community intelligence system status"""
    try:
        # Get status from community intelligence system
        system_status = await community_intelligence.get_system_status()
        
        return CommunityIntelligenceStatusResponse(
            status=system_status["status"],
            last_update_cycle=system_status.get("last_update_cycle"),
            total_research_findings=system_status["total_research_findings"],
            total_enhancement_plans=system_status["total_enhancement_plans"],
            recent_findings_7_days=system_status["recent_findings_7_days"],
            active_monitors=system_status["active_monitors"],
            research_sources=system_status["research_sources"],
            next_scheduled_update=None  # Will be calculated
        )
        
    except Exception as e:
        logger.error(f"Error getting community intelligence status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/start")
async def start_monitoring():
    """Start community intelligence monitoring"""
    try:
        if not community_intelligence.running:
            await community_intelligence.start_monitoring()
            return {"status": "started", "message": "Community intelligence monitoring started"}
        else:
            return {"status": "already_running", "message": "Community intelligence monitoring is already running"}
            
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/stop")
async def stop_monitoring():
    """Stop community intelligence monitoring"""
    try:
        if community_intelligence.running:
            await community_intelligence.stop_monitoring()
            return {"status": "stopped", "message": "Community intelligence monitoring stopped"}
        else:
            return {"status": "already_stopped", "message": "Community intelligence monitoring is not running"}
            
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/trigger-update")
async def trigger_manual_update(background_tasks: BackgroundTasks):
    """Trigger manual research update cycle"""
    try:
        background_tasks.add_task(community_intelligence._run_update_cycle)
        return {"status": "triggered", "message": "Manual update cycle triggered"}
        
    except Exception as e:
        logger.error(f"Error triggering manual update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_community_metrics():
    """Get detailed community intelligence metrics"""
    try:
        async with get_db_session() as session:
            # Get various metrics
            metrics = {}
            
            # Research findings metrics
            findings_by_source = await session.execute("""
                SELECT source_type, COUNT(*) as count, AVG(significance_score) as avg_significance
                FROM research_findings 
                GROUP BY source_type
            """)
            metrics["findings_by_source"] = {row.source_type: {"count": row.count, "avg_significance": row.avg_significance} 
                                           for row in findings_by_source.fetchall()}
            
            # Enhancement metrics
            enhancement_status = await session.execute("""
                SELECT status, COUNT(*) as count, AVG(estimated_impact) as avg_impact
                FROM capability_enhancements 
                GROUP BY status
            """)
            metrics["enhancements_by_status"] = {row.status: {"count": row.count, "avg_impact": row.avg_impact} 
                                               for row in enhancement_status.fetchall()}
            
            # Recent activity
            recent_findings = await session.execute("""
                SELECT DATE(discovered_at) as date, COUNT(*) as count
                FROM research_findings 
                WHERE discovered_at > :since
                GROUP BY DATE(discovered_at)
                ORDER BY date DESC
            """, {"since": datetime.now() - timedelta(days=30)})
            metrics["daily_findings_30_days"] = {str(row.date): row.count for row in recent_findings.fetchall()}
            
            # Top focus areas
            focus_areas = await session.execute("""
                SELECT JSON_UNQUOTE(JSON_EXTRACT(focus_areas, '$[*]')) as area, COUNT(*) as count
                FROM research_findings 
                WHERE JSON_LENGTH(focus_areas) > 0
                GROUP BY area
                ORDER BY count DESC
                LIMIT 10
            """)
            metrics["top_focus_areas"] = {row.area: row.count for row in focus_areas.fetchall()}
            
            return metrics
            
    except Exception as e:
        logger.error(f"Error getting community metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def update_community_intelligence_metrics():
    """Update community intelligence metrics in background"""
    try:
        async with get_db_session() as session:
            # Get or create community intelligence record
            ci_result = await session.execute("SELECT * FROM community_intelligence LIMIT 1")
            ci_row = ci_result.fetchone()
            
            if ci_row:
                ci = await session.get(CommunityIntelligence, ci_row.id)
            else:
                ci = CommunityIntelligence()
                session.add(ci)
            
            # Update metrics
            findings_count = await session.execute("SELECT COUNT(*) FROM research_findings")
            ci.total_research_findings = findings_count.scalar()
            
            plans_count = await session.execute("SELECT COUNT(*) FROM capability_enhancements")
            ci.total_enhancement_plans = plans_count.scalar()
            
            active_count = await session.execute(
                "SELECT COUNT(*) FROM capability_enhancements WHERE status IN ('planned', 'in_progress', 'testing')"
            )
            ci.active_enhancements = active_count.scalar()
            
            completed_count = await session.execute(
                "SELECT COUNT(*) FROM capability_enhancements WHERE status = 'deployed'"
            )
            ci.completed_enhancements = completed_count.scalar()
            
            # Calculate success rate
            if ci.total_enhancement_plans > 0:
                ci.enhancement_success_rate = ci.completed_enhancements / ci.total_enhancement_plans
            
            ci.updated_at = datetime.now()
            await session.commit()
            
    except Exception as e:
        logger.error(f"Error updating community intelligence metrics: {e}")