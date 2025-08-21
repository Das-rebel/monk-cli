"""
MONK CLI Core Database Models
Phase 1 Foundation Models for Users, Agents, Memory, and Sessions
"""
from datetime import datetime
from typing import Optional, Dict, List, Any
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import uuid

Base = declarative_base()


# Phase 2 Community Intelligence Models

class ResearchFinding(Base):
    """Research findings discovered by community intelligence system"""
    __tablename__ = "research_findings"
    
    id = Column(String(255), primary_key=True)  # Hash-based ID
    title = Column(String(500), nullable=False)
    summary = Column(Text, nullable=False)
    source_url = Column(String(1000), nullable=False)
    source_type = Column(String(50), nullable=False)  # arxiv, github, blog, community
    discovered_at = Column(DateTime, default=datetime.utcnow)
    
    # Significance assessment
    significance_score = Column(Float, nullable=False, default=0.0)
    significance_level = Column(String(50), nullable=False)  # low, medium, high, breakthrough
    
    # Content analysis
    focus_areas = Column(JSON, default=list)
    implementation_potential = Column(Float, default=0.0)
    community_interest = Column(Float, default=0.0)
    
    # Metadata
    authors = Column(JSON, default=list)
    tags = Column(JSON, default=list)
    full_content = Column(Text)
    metadata = Column(JSON, default=dict)
    
    # Processing status
    processed = Column(Boolean, default=False)
    enhancement_plan_generated = Column(Boolean, default=False)
    
    # Relationships
    enhancement_plans = relationship("CapabilityEnhancement", back_populates="research_finding")
    
    def __repr__(self):
        return f"<ResearchFinding {self.title[:50]}...>"


class CapabilityEnhancement(Base):
    """Capability enhancement plans generated from research findings"""
    __tablename__ = "capability_enhancements"
    
    id = Column(String(255), primary_key=True)
    research_finding_id = Column(String(255), ForeignKey("research_findings.id"), nullable=False)
    
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    
    # Assessment scores
    implementation_complexity = Column(Float, nullable=False)
    estimated_impact = Column(Float, nullable=False)
    development_time_days = Column(Integer, nullable=False)
    
    # Implementation details
    required_resources = Column(JSON, default=list)
    implementation_plan = Column(JSON, default=dict)
    testing_strategy = Column(JSON, default=dict)
    deployment_strategy = Column(JSON, default=dict)
    risk_assessment = Column(JSON, default=dict)
    
    # Status tracking
    status = Column(String(50), default="planned")  # planned, in_progress, testing, deployed, cancelled
    priority = Column(String(50), default="medium")  # low, medium, high, critical
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    deployed_at = Column(DateTime)
    
    # Assignment
    assigned_to = Column(String(255))
    assigned_at = Column(DateTime)
    
    # Results tracking
    actual_development_time_days = Column(Integer)
    actual_impact_score = Column(Float)
    user_feedback_score = Column(Float)
    performance_impact = Column(JSON, default=dict)
    
    # Relationships
    research_finding = relationship("ResearchFinding", back_populates="enhancement_plans")
    
    def __repr__(self):
        return f"<CapabilityEnhancement {self.title[:50]}...>"


class CommunityIntelligence(Base):
    """Community intelligence system status and metrics"""
    __tablename__ = "community_intelligence"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # System status
    system_status = Column(String(50), default="active")  # active, inactive, maintenance
    last_update_cycle = Column(DateTime)
    next_scheduled_update = Column(DateTime)
    
    # Metrics
    total_research_findings = Column(Integer, default=0)
    total_enhancement_plans = Column(Integer, default=0)
    active_enhancements = Column(Integer, default=0)
    completed_enhancements = Column(Integer, default=0)
    
    # Performance metrics
    average_discovery_to_plan_hours = Column(Float, default=0.0)
    average_plan_to_deployment_days = Column(Float, default=0.0)
    enhancement_success_rate = Column(Float, default=0.0)
    
    # Source monitoring status
    arxiv_last_check = Column(DateTime)
    github_last_check = Column(DateTime)
    blog_last_check = Column(DateTime)
    community_last_check = Column(DateTime)
    
    # Configuration
    monitoring_config = Column(JSON, default=dict)
    enhancement_thresholds = Column(JSON, default=dict)
    
    # Error tracking
    last_error = Column(Text)
    error_count_24h = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<CommunityIntelligence {self.system_status}>"


class User(Base):
    """User model for MONK CLI"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_active_at = Column(DateTime, default=datetime.utcnow)
    
    # User preferences and settings
    preferences = Column(JSON, default=dict)
    subscription_tier = Column(String(50), default="free")
    
    # Usage statistics
    total_tasks_completed = Column(Integer, default=0)
    total_agents_used = Column(Integer, default=0)
    total_memory_queries = Column(Integer, default=0)
    
    # Relationships
    sessions = relationship("UserSession", back_populates="user")
    memories = relationship("EpisodicMemory", back_populates="user")
    semantic_memories = relationship("SemanticMemory", back_populates="user")
    procedural_memories = relationship("ProceduralMemory", back_populates="user")
    agent_executions = relationship("AgentExecution", back_populates="user")
    
    def __repr__(self):
        return f"<User {self.email}>"


class UserSession(Base):
    """User session tracking across interfaces (CLI, VS Code, Web)"""
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    interface_type = Column(String(50), nullable=False)  # 'cli', 'vscode', 'web'
    
    # Session data
    session_data = Column(JSON, default=dict)
    context_data = Column(JSON, default=dict)
    active_agent_stack = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Performance tracking
    tasks_completed_in_session = Column(Integer, default=0)
    average_task_time = Column(Float, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    __table_args__ = (
        Index('idx_user_session_active', 'user_id', 'is_active'),
        Index('idx_session_interface', 'interface_type', 'is_active'),
    )


class AgentStack(Base):
    """Agent stack definitions (Development, Content, Business, Security)"""
    __tablename__ = "agent_stacks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    config = Column(JSON, default=dict)
    
    # Stack properties
    is_enabled = Column(Boolean, default=True)
    specialization_domains = Column(ARRAY(String), default=list)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    agents = relationship("Agent", back_populates="stack")


class Agent(Base):
    """Individual agent definitions within stacks"""
    __tablename__ = "agents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stack_id = Column(UUID(as_uuid=True), ForeignKey("agent_stacks.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Agent personality (Big Five traits)
    personality_traits = Column(JSON, default=dict)  # conscientiousness, openness, etc.
    
    # Agent capabilities
    specializations = Column(ARRAY(String), default=list)
    tools = Column(ARRAY(String), default=list)
    
    # Performance metrics
    success_rate = Column(Float, default=0.0)
    average_execution_time = Column(Float, default=0.0)
    total_executions = Column(Integer, default=0)
    
    # Configuration
    config = Column(JSON, default=dict)
    is_enabled = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    stack = relationship("AgentStack", back_populates="agents")
    executions = relationship("AgentExecution", back_populates="agent")
    
    def __repr__(self):
        return f"<Agent {self.name} in {self.stack.name if self.stack else 'Unknown'} stack>"


class AgentExecution(Base):
    """Track individual agent execution instances"""
    __tablename__ = "agent_executions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)
    
    # Task details
    task_description = Column(Text, nullable=False)
    task_type = Column(String(100), nullable=True)
    execution_context = Column(JSON, default=dict)
    
    # Results
    result = Column(JSON, default=dict)
    success = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)
    
    # Performance metrics
    execution_time_ms = Column(Integer, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    memory_queries_made = Column(Integer, default=0)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="agent_executions")
    agent = relationship("Agent", back_populates="executions")
    
    __table_args__ = (
        Index('idx_agent_execution_user', 'user_id', 'started_at'),
        Index('idx_agent_execution_agent', 'agent_id', 'success'),
    )


class EpisodicMemory(Base):
    """Episodic memories - specific events and interactions"""
    __tablename__ = "episodic_memories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=True)
    
    # Memory content
    memory_type = Column(String(50), nullable=False)  # 'interaction', 'task', 'outcome'
    content = Column(JSON, nullable=False)
    context = Column(JSON, default=dict)
    
    # Memory metadata
    importance_score = Column(Float, default=0.5)
    confidence_score = Column(Float, default=1.0)
    tags = Column(ARRAY(String), default=list)
    
    # Access patterns
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="memories")
    agent = relationship("Agent")
    
    __table_args__ = (
        Index('idx_episodic_memory_user', 'user_id', 'importance_score'),
        Index('idx_episodic_memory_type', 'memory_type', 'created_at'),
        Index('idx_episodic_memory_access', 'last_accessed', 'access_count'),
    )


class SemanticMemory(Base):
    """Semantic memories - facts, patterns, and knowledge"""
    __tablename__ = "semantic_memories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Knowledge representation (subject-predicate-object)
    knowledge_type = Column(String(100), nullable=False)  # 'fact', 'pattern', 'preference'
    subject = Column(Text, nullable=False)
    predicate = Column(Text, nullable=False)
    object = Column(Text, nullable=False)
    
    # Metadata
    confidence_score = Column(Float, default=0.5)
    source_count = Column(Integer, default=1)  # How many episodes support this
    
    # Source tracking
    source_memory_ids = Column(ARRAY(UUID), default=list)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="semantic_memories")
    
    __table_args__ = (
        Index('idx_semantic_memory_user', 'user_id', 'knowledge_type'),
        Index('idx_semantic_memory_confidence', 'confidence_score', 'source_count'),
    )


class ProceduralMemory(Base):
    """Procedural memories - learned procedures and workflows"""
    __tablename__ = "procedural_memories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Procedure details
    procedure_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    steps = Column(JSON, nullable=False)
    
    # Performance metrics
    success_rate = Column(Float, default=0.0)
    usage_frequency = Column(Integer, default=0)
    average_execution_time = Column(Float, nullable=True)
    
    # Context and triggers
    trigger_conditions = Column(JSON, default=dict)
    domain_tags = Column(ARRAY(String), default=list)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="procedural_memories")
    
    __table_args__ = (
        Index('idx_procedural_memory_user', 'user_id', 'success_rate'),
        Index('idx_procedural_memory_usage', 'usage_frequency', 'last_used'),
    )


class TaskPerformance(Base):
    """Track task performance for learning and improvement"""
    __tablename__ = "task_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Task identification
    task_type = Column(String(100), nullable=False)
    task_signature = Column(String(500), nullable=False)  # Hash of task characteristics
    
    # Performance metrics
    completion_time_seconds = Column(Float, nullable=False)
    success = Column(Boolean, nullable=False)
    quality_score = Column(Float, nullable=True)
    
    # Context
    agent_stack_used = Column(String(100), nullable=True)
    agents_involved = Column(ARRAY(String), default=list)
    memory_queries_count = Column(Integer, default=0)
    
    # Tracking
    attempt_number = Column(Integer, default=1)  # For same task repeated
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_task_performance_user', 'user_id', 'task_type'),
        Index('idx_task_performance_signature', 'task_signature', 'created_at'),
    )


class SystemMetrics(Base):
    """System-wide performance and health metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Metric identification
    metric_type = Column(String(100), nullable=False)  # 'response_time', 'memory_usage', etc.
    metric_name = Column(String(100), nullable=False)
    
    # Metric values
    value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=False)
    
    # Context
    component = Column(String(100), nullable=True)  # 'agent_orchestrator', 'memory_system', etc.
    metadata = Column(JSON, default=dict)
    
    recorded_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_system_metrics_type', 'metric_type', 'recorded_at'),
        Index('idx_system_metrics_component', 'component', 'metric_name'),
    )