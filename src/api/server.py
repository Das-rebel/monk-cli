"""
MONK CLI Unified Backend API Server
Phase 1 implementation with agent orchestration and memory services
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging

from ..core.config import config
from ..core.database import startup_database, shutdown_database, get_db_session
from ..core.models import User
from ..agents.orchestrator import orchestrator, TaskContext
from ..memory.memory_system import memory_system, MemoryQuery
from ..community.intelligence_system import community_intelligence
from .community_endpoints import router as community_router

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class TaskRequest(BaseModel):
    """Request model for task execution"""
    user_id: str
    task_description: str
    task_type: str = "general"
    domain: str = "development"
    complexity_level: float = Field(default=0.5, ge=0.0, le=1.0)
    urgency_level: float = Field(default=0.5, ge=0.0, le=1.0)
    context_data: Dict[str, Any] = {}
    agent_preferences: Optional[List[str]] = None


class TaskResponse(BaseModel):
    """Response model for task execution"""
    task_id: str
    success: bool
    result: Dict[str, Any]
    execution_time_ms: int
    confidence_score: float
    selected_agent: str
    agent_reasoning: str
    tokens_used: int = 0
    memory_queries_made: int = 0
    error_message: Optional[str] = None


class CollaborativeTaskRequest(BaseModel):
    """Request for collaborative task execution"""
    user_id: str
    task_description: str
    required_agents: Optional[List[str]] = None
    context_data: Dict[str, Any] = {}


class MemorySearchRequest(BaseModel):
    """Request for memory search"""
    user_id: str
    query_text: str
    memory_types: List[str] = ["episodic"]
    limit: int = Field(default=10, ge=1, le=50)
    min_relevance_score: float = Field(default=0.3, ge=0.0, le=1.0)
    context_filters: Dict[str, Any] = {}


class MemoryResponse(BaseModel):
    """Response for memory search"""
    memory_id: str
    memory_type: str
    content: Dict[str, Any]
    relevance_score: float
    importance_score: float
    created_at: str
    last_accessed: str
    access_count: int


class SystemStatusResponse(BaseModel):
    """System status response"""
    status: str
    uptime_seconds: float
    total_users: int
    total_tasks_processed: int
    success_rate: float
    agent_status: Dict[str, Any]
    memory_stats: Dict[str, Any]


class UserStatsResponse(BaseModel):
    """User statistics response"""
    user_id: str
    total_tasks_completed: int
    total_agents_used: int
    total_memory_queries: int
    favorite_agent: Optional[str] = None
    expertise_domains: List[str] = []
    success_rate: float


# Startup/shutdown lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_database()
    await orchestrator.start()
    
    # Start community intelligence monitoring
    await community_intelligence.start_monitoring()
    
    logger.info("MONK CLI API server started")
    
    yield
    
    # Shutdown
    await community_intelligence.stop_monitoring()
    await orchestrator.stop()
    await shutdown_database()
    logger.info("MONK CLI API server shut down")


# FastAPI app instance
app = FastAPI(
    title="MONK CLI API",
    description="Phase 1 Foundation API with Agent Orchestration and Memory System",
    version="1.0.0-phase1",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include community intelligence router
app.include_router(community_router)


# Dependency to get authenticated user
async def get_user(user_id: str) -> User:
    """Get user by ID"""
    async with get_db_session() as session:
        user = await session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user


# Task execution endpoints
@app.post("/api/v1/tasks/execute", response_model=TaskResponse)
async def execute_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Execute a task using agent orchestrator"""
    try:
        # Validate user
        user = await get_user(request.user_id)
        
        # Get memory context
        memory_query = MemoryQuery(
            query_text=request.task_description,
            user_id=request.user_id,
            memory_types=["episodic"],
            limit=5
        )
        
        memories = await memory_system.retrieve_relevant_memories(memory_query)
        memory_context = [memory.content for memory in memories.get("episodic", [])]
        
        # Create task context
        context = TaskContext(
            user_id=request.user_id,
            task_description=request.task_description,
            task_type=request.task_type,
            domain=request.domain,
            complexity_level=request.complexity_level,
            urgency_level=request.urgency_level,
            context_data=request.context_data,
            memory_context=memory_context
        )
        
        # Execute task
        response, selection_result = await orchestrator.execute_task(context)
        
        # Store in memory (background task)
        background_tasks.add_task(
            store_task_memory,
            request.user_id,
            context,
            response,
            selection_result
        )
        
        # Update user stats (background task)
        background_tasks.add_task(update_user_stats, user, response.success)
        
        return TaskResponse(
            task_id=str(time.time()),  # Simple ID for Phase 1
            success=response.success,
            result=response.result,
            execution_time_ms=response.execution_time_ms,
            confidence_score=response.confidence_score,
            selected_agent=selection_result.selected_agent.name,
            agent_reasoning=selection_result.selection_reasoning,
            tokens_used=response.tokens_used,
            memory_queries_made=response.memory_queries_made,
            error_message=response.error_message
        )
        
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/tasks/collaborate")
async def execute_collaborative_task(request: CollaborativeTaskRequest):
    """Execute collaborative task with multiple agents"""
    try:
        # Validate user
        await get_user(request.user_id)
        
        # Create context
        context = TaskContext(
            user_id=request.user_id,
            task_description=request.task_description,
            task_type="collaborative",
            domain="development",
            complexity_level=0.7,  # Collaborative tasks are typically complex
            urgency_level=0.5,
            context_data=request.context_data
        )
        
        # Execute collaborative task
        results = await orchestrator.execute_collaborative_task(
            context, request.required_agents
        )
        
        # Format response
        formatted_results = {}
        for agent_type, response in results.items():
            formatted_results[agent_type] = {
                "success": response.success,
                "result": response.result,
                "execution_time_ms": response.execution_time_ms,
                "confidence_score": response.confidence_score,
                "error_message": response.error_message
            }
        
        return {
            "collaborative_task_id": str(time.time()),
            "agents_involved": list(results.keys()),
            "overall_success": all(r.success for r in results.values()),
            "total_execution_time_ms": sum(r.execution_time_ms for r in results.values()),
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Collaborative task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Memory endpoints
@app.post("/api/v1/memory/search", response_model=List[MemoryResponse])
async def search_memories(request: MemorySearchRequest):
    """Search user memories"""
    try:
        # Validate user
        await get_user(request.user_id)
        
        # Create memory query
        query = MemoryQuery(
            query_text=request.query_text,
            user_id=request.user_id,
            memory_types=request.memory_types,
            limit=request.limit,
            min_relevance_score=request.min_relevance_score,
            context_filters=request.context_filters
        )
        
        # Search memories
        memories = await memory_system.retrieve_relevant_memories(query)
        
        # Format response
        response = []
        for memory_type, memory_list in memories.items():
            for memory in memory_list:
                response.append(MemoryResponse(
                    memory_id=memory.memory_id,
                    memory_type=memory.memory_type,
                    content=memory.content,
                    relevance_score=memory.relevance_score,
                    importance_score=memory.importance_score,
                    created_at=memory.created_at.isoformat(),
                    last_accessed=memory.last_accessed.isoformat(),
                    access_count=memory.access_count
                ))
        
        return response
        
    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memory/insights/{user_id}")
async def get_memory_insights(user_id: str):
    """Get memory insights for user"""
    try:
        # Validate user
        await get_user(user_id)
        
        # Get insights
        insights = await memory_system.get_memory_insights(user_id)
        
        # Format response
        formatted_insights = []
        for insight in insights:
            formatted_insights.append({
                "insight_type": insight.insight_type,
                "description": insight.description,
                "confidence_score": insight.confidence_score,
                "suggested_action": insight.suggested_action,
                "supporting_memories": insight.supporting_memories
            })
        
        return {"insights": formatted_insights}
        
    except Exception as e:
        logger.error(f"Getting memory insights failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/memory/optimize/{user_id}")
async def optimize_user_memory(user_id: str, background_tasks: BackgroundTasks):
    """Optimize user memory system"""
    try:
        # Validate user
        await get_user(user_id)
        
        # Run optimization in background
        background_tasks.add_task(memory_system.optimize_memory_performance, user_id)
        
        return {"message": "Memory optimization started"}
        
    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System status endpoints
@app.get("/api/v1/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get overall system status"""
    try:
        # Get orchestrator status
        orchestrator_status = orchestrator.get_orchestrator_status()
        
        # Get user count
        async with get_db_session() as session:
            result = await session.execute("SELECT COUNT(*) FROM users")
            total_users = result.scalar()
        
        # Calculate uptime (simple implementation)
        uptime = time.time() - getattr(get_system_status, 'start_time', time.time())
        get_system_status.start_time = getattr(get_system_status, 'start_time', time.time())
        
        return SystemStatusResponse(
            status="healthy",
            uptime_seconds=uptime,
            total_users=total_users,
            total_tasks_processed=orchestrator_status["total_tasks_processed"],
            success_rate=orchestrator_status["success_rate"],
            agent_status=orchestrator_status["agents"],
            memory_stats={
                "total_memories": "not_implemented",  # Phase 2
                "memory_efficiency": "not_implemented"
            }
        )
        
    except Exception as e:
        logger.error(f"Getting system status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agents/status")
async def get_agent_status():
    """Get detailed agent status"""
    try:
        status = orchestrator.get_orchestrator_status()
        return status
        
    except Exception as e:
        logger.error(f"Getting agent status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/users/{user_id}/stats", response_model=UserStatsResponse)
async def get_user_stats(user_id: str):
    """Get user statistics"""
    try:
        # Get user
        user = await get_user(user_id)
        
        # Calculate success rate from agent executions
        async with get_db_session() as session:
            success_result = await session.execute(
                "SELECT COUNT(*) FROM agent_executions WHERE user_id = :user_id AND success = true",
                {"user_id": user_id}
            )
            successful_tasks = success_result.scalar()
            
            total_result = await session.execute(
                "SELECT COUNT(*) FROM agent_executions WHERE user_id = :user_id",
                {"user_id": user_id}
            )
            total_tasks = total_result.scalar()
            
            # Get favorite agent
            favorite_result = await session.execute(
                """
                SELECT a.name, COUNT(*) as usage_count 
                FROM agent_executions ae 
                JOIN agents a ON ae.agent_id = a.id 
                WHERE ae.user_id = :user_id 
                GROUP BY a.id, a.name 
                ORDER BY usage_count DESC 
                LIMIT 1
                """,
                {"user_id": user_id}
            )
            favorite_row = favorite_result.fetchone()
            favorite_agent = favorite_row[0] if favorite_row else None
        
        success_rate = successful_tasks / max(total_tasks, 1)
        
        return UserStatsResponse(
            user_id=user_id,
            total_tasks_completed=user.total_tasks_completed,
            total_agents_used=user.total_agents_used,
            total_memory_queries=user.total_memory_queries,
            favorite_agent=favorite_agent,
            expertise_domains=[],  # Phase 2 - extract from memory analysis
            success_rate=success_rate
        )
        
    except Exception as e:
        logger.error(f"Getting user stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": time.time()}


# WebSocket endpoint for real-time updates (Phase 2 will expand)
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket, user_id: str):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    
    try:
        # Simple echo implementation for Phase 1
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Background task functions
async def store_task_memory(user_id: str, context: TaskContext, response, selection_result):
    """Background task to store task execution in memory"""
    try:
        memory_content = {
            "task_description": context.task_description,
            "task_type": context.task_type,
            "domain": context.domain,
            "agent_used": selection_result.selected_agent.name,
            "success": response.success,
            "execution_time_ms": response.execution_time_ms,
            "confidence_score": response.confidence_score,
            "tokens_used": response.tokens_used,
            "memory_queries": response.memory_queries_made
        }
        
        await memory_system.store_interaction(
            user_id=user_id,
            interaction_type="api_task_execution",
            content=memory_content,
            context={
                "complexity_level": context.complexity_level,
                "urgency_level": context.urgency_level,
                "interface": "api"
            },
            importance_score=0.7 if response.success else 0.3
        )
        
    except Exception as e:
        logger.error(f"Failed to store task memory: {e}")


async def update_user_stats(user: User, success: bool):
    """Background task to update user statistics"""
    try:
        async with get_db_session() as session:
            user.total_tasks_completed += 1
            if success:
                # Update success-related stats
                pass
            
            await session.commit()
            
    except Exception as e:
        logger.error(f"Failed to update user stats: {e}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.server:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.is_development,
        log_level="info"
    )