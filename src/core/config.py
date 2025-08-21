"""
MONK CLI Phase 1 Configuration Management
"""
import os
from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    
    # PostgreSQL Primary Database
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT") 
    postgres_db: str = Field(default="monk_cli", env="POSTGRES_DB")
    postgres_user: str = Field(default="monk", env="POSTGRES_USER")
    postgres_password: str = Field(default="", env="POSTGRES_PASSWORD")
    
    # Redis Cache & Sessions
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Vector Database (Pinecone)
    pinecone_api_key: str = Field(default="", env="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="us-west1-gcp", env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="monk-memory", env="PINECONE_INDEX_NAME")
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


class AIModelSettings(BaseSettings):
    """AI Model configuration settings"""
    
    # API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    
    # Default Models
    default_model: str = Field(default="gpt-4-turbo-preview", env="DEFAULT_AI_MODEL")
    fast_model: str = Field(default="gpt-3.5-turbo", env="FAST_AI_MODEL")
    reasoning_model: str = Field(default="claude-3-opus-20240229", env="REASONING_AI_MODEL")
    
    # Model Limits
    max_tokens: int = Field(default=4096, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="AI_TEMPERATURE")
    
    @validator("openai_api_key", "anthropic_api_key")
    def validate_api_keys(cls, v):
        if not v:
            raise ValueError("AI API keys are required")
        return v


class AgentSettings(BaseSettings):
    """Agent system configuration"""
    
    # Agent Pool Configuration
    max_concurrent_agents: int = Field(default=100, env="MAX_CONCURRENT_AGENTS")
    agent_timeout_seconds: int = Field(default=300, env="AGENT_TIMEOUT_SECONDS")
    
    # Development Stack Agents
    architect_agent_enabled: bool = Field(default=True, env="ARCHITECT_AGENT_ENABLED")
    quality_enforcer_enabled: bool = Field(default=True, env="QUALITY_ENFORCER_ENABLED")
    innovation_driver_enabled: bool = Field(default=True, env="INNOVATION_DRIVER_ENABLED")
    integration_specialist_enabled: bool = Field(default=True, env="INTEGRATION_SPECIALIST_ENABLED")
    
    # Agent Personality Configuration
    personality_system_enabled: bool = Field(default=True, env="PERSONALITY_SYSTEM_ENABLED")
    big_five_traits_enabled: bool = Field(default=True, env="BIG_FIVE_TRAITS_ENABLED")
    
    # Performance Thresholds
    agent_selection_max_time_ms: int = Field(default=100, env="AGENT_SELECTION_MAX_TIME_MS")
    agent_task_success_rate_threshold: float = Field(default=0.95, env="AGENT_SUCCESS_THRESHOLD")


class MemorySettings(BaseSettings):
    """Memory system configuration"""
    
    # Memory Types
    episodic_memory_enabled: bool = Field(default=True, env="EPISODIC_MEMORY_ENABLED")
    semantic_memory_enabled: bool = Field(default=True, env="SEMANTIC_MEMORY_ENABLED")
    procedural_memory_enabled: bool = Field(default=True, env="PROCEDURAL_MEMORY_ENABLED")
    
    # Memory Limits
    max_memories_per_user: int = Field(default=1000000, env="MAX_MEMORIES_PER_USER")
    memory_retention_days: int = Field(default=90, env="MEMORY_RETENTION_DAYS")
    
    # Performance Targets
    memory_retrieval_max_time_ms: int = Field(default=50, env="MEMORY_RETRIEVAL_MAX_TIME_MS")
    memory_batch_size: int = Field(default=100, env="MEMORY_BATCH_SIZE")
    
    # Memory Decay Configuration
    memory_decay_enabled: bool = Field(default=True, env="MEMORY_DECAY_ENABLED")
    importance_score_threshold: float = Field(default=0.3, env="MEMORY_IMPORTANCE_THRESHOLD")


class InterfaceSettings(BaseSettings):
    """Interface configuration for CLI, IDE, Web"""
    
    # CLI Interface
    cli_max_concurrent_sessions: int = Field(default=200, env="CLI_MAX_SESSIONS")
    cli_session_timeout_seconds: int = Field(default=3600, env="CLI_SESSION_TIMEOUT")
    
    # VS Code Extension
    vscode_extension_enabled: bool = Field(default=True, env="VSCODE_EXTENSION_ENABLED")
    vscode_max_connections: int = Field(default=400, env="VSCODE_MAX_CONNECTIONS")
    
    # Web Interface (Phase 2)
    web_interface_enabled: bool = Field(default=False, env="WEB_INTERFACE_ENABLED")
    web_max_connections: int = Field(default=300, env="WEB_MAX_CONNECTIONS")
    
    # Real-time Sync
    websocket_enabled: bool = Field(default=True, env="WEBSOCKET_ENABLED")
    sync_frequency_seconds: int = Field(default=1, env="SYNC_FREQUENCY_SECONDS")


class PerformanceSettings(BaseSettings):
    """Performance and scaling configuration"""
    
    # Response Time Targets
    api_response_time_target_ms: int = Field(default=200, env="API_RESPONSE_TARGET_MS")
    memory_query_time_target_ms: int = Field(default=50, env="MEMORY_QUERY_TARGET_MS")
    agent_selection_time_target_ms: int = Field(default=100, env="AGENT_SELECTION_TARGET_MS")
    
    # Scaling Targets
    target_concurrent_users: int = Field(default=500, env="TARGET_CONCURRENT_USERS")
    target_daily_active_users: int = Field(default=2000, env="TARGET_DAILY_ACTIVE_USERS")
    
    # System Resources
    cpu_utilization_threshold: float = Field(default=0.7, env="CPU_UTILIZATION_THRESHOLD")
    memory_utilization_threshold: float = Field(default=0.8, env="MEMORY_UTILIZATION_THRESHOLD")
    
    # Uptime Target
    uptime_target_percentage: float = Field(default=99.5, env="UPTIME_TARGET_PERCENTAGE")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration"""
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    structured_logging: bool = Field(default=True, env="STRUCTURED_LOGGING")
    
    # Metrics
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=8000, env="PROMETHEUS_PORT")
    
    # Health Checks
    health_check_enabled: bool = Field(default=True, env="HEALTH_CHECK_ENABLED")
    health_check_interval_seconds: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")


class MONKConfig(BaseSettings):
    """Main MONK CLI Configuration"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8080, env="API_PORT")
    
    # Security
    secret_key: str = Field(default="", env="SECRET_KEY")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Sub-configurations
    database: DatabaseSettings = DatabaseSettings()
    ai_models: AIModelSettings = AIModelSettings()
    agents: AgentSettings = AgentSettings()
    memory: MemorySettings = MemorySettings()
    interfaces: InterfaceSettings = InterfaceSettings()
    performance: PerformanceSettings = PerformanceSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("secret_key")
    def validate_secret_key(cls, v):
        if not v and cls.environment != "development":
            raise ValueError("SECRET_KEY is required for production")
        return v or "dev-secret-key-change-in-production"
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"


# Global configuration instance
config = MONKConfig()