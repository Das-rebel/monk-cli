"""
MONK CLI Database Connection and Session Management
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
import redis.asyncio as redis
import pinecone
from .config import config
from .models import Base
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages all database connections (PostgreSQL, Redis, Pinecone)"""
    
    def __init__(self):
        self.postgres_engine = None
        self.async_session_factory = None
        self.redis_client = None
        self.pinecone_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize all database connections"""
        if self._initialized:
            return
        
        # PostgreSQL Connection
        await self._init_postgres()
        
        # Redis Connection
        await self._init_redis()
        
        # Pinecone Connection
        await self._init_pinecone()
        
        self._initialized = True
        logger.info("Database manager initialized successfully")
    
    async def _init_postgres(self):
        """Initialize PostgreSQL connection"""
        try:
            # Create async engine
            self.postgres_engine = create_async_engine(
                config.database.postgres_url,
                echo=config.is_development,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600,  # 1 hour
                poolclass=StaticPool if config.is_development else None
            )
            
            # Create session factory
            self.async_session_factory = async_sessionmaker(
                bind=self.postgres_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("PostgreSQL connection initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                config.database.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def _init_pinecone(self):
        """Initialize Pinecone vector database"""
        try:
            if not config.database.pinecone_api_key:
                logger.warning("Pinecone API key not provided, skipping initialization")
                return
            
            pinecone.init(
                api_key=config.database.pinecone_api_key,
                environment=config.database.pinecone_environment
            )
            
            # Check if index exists, create if not
            index_name = config.database.pinecone_index_name
            if index_name not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {index_name}")
                pinecone.create_index(
                    name=index_name,
                    dimension=768,  # Standard embedding size
                    metric="cosine"
                )
            
            self.pinecone_client = pinecone.Index(index_name)
            logger.info("Pinecone connection initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    async def create_tables(self):
        """Create database tables"""
        if not self.postgres_engine:
            raise RuntimeError("PostgreSQL not initialized")
        
        async with self.postgres_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
    
    async def get_session(self) -> AsyncSession:
        """Get async database session"""
        if not self.async_session_factory:
            raise RuntimeError("Database not initialized")
        
        return self.async_session_factory()
    
    async def get_redis(self) -> redis.Redis:
        """Get Redis client"""
        if not self.redis_client:
            raise RuntimeError("Redis not initialized")
        
        return self.redis_client
    
    def get_pinecone(self) -> pinecone.Index:
        """Get Pinecone index"""
        if not self.pinecone_client:
            raise RuntimeError("Pinecone not initialized")
        
        return self.pinecone_client
    
    async def close(self):
        """Close all database connections"""
        if self.postgres_engine:
            await self.postgres_engine.dispose()
        
        if self.redis_client:
            await self.redis_client.close()
        
        self._initialized = False
        logger.info("Database connections closed")
    
    async def health_check(self) -> dict:
        """Check health of all database connections"""
        health = {
            "postgres": False,
            "redis": False,
            "pinecone": False
        }
        
        # Check PostgreSQL
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
                health["postgres"] = True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
        
        # Check Redis
        try:
            redis_client = await self.get_redis()
            await redis_client.ping()
            health["redis"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
        
        # Check Pinecone
        try:
            if self.pinecone_client:
                # Just check if we can get stats
                self.pinecone_client.describe_index_stats()
                health["pinecone"] = True
        except Exception as e:
            logger.error(f"Pinecone health check failed: {e}")
        
        return health


# Global database manager instance
db_manager = DatabaseManager()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database sessions"""
    async with db_manager.get_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_redis_client() -> AsyncGenerator[redis.Redis, None]:
    """Context manager for Redis client"""
    client = await db_manager.get_redis()
    try:
        yield client
    finally:
        # Redis client is reused, no need to close
        pass


def get_pinecone_index() -> pinecone.Index:
    """Get Pinecone index (synchronous)"""
    return db_manager.get_pinecone()


# Database startup/shutdown handlers
async def startup_database():
    """Initialize database connections on startup"""
    await db_manager.initialize()
    
    if config.is_development:
        # Create tables in development
        await db_manager.create_tables()


async def shutdown_database():
    """Close database connections on shutdown"""
    await db_manager.close()


# Transaction decorator
def transactional(func):
    """Decorator to wrap function in database transaction"""
    async def wrapper(*args, **kwargs):
        async with get_db_session() as session:
            # Add session to kwargs if not present
            if 'session' not in kwargs:
                kwargs['session'] = session
            
            return await func(*args, **kwargs)
    
    return wrapper


# Cache utilities
class CacheManager:
    """Redis cache management utilities"""
    
    @staticmethod
    async def get(key: str, default=None):
        """Get value from cache"""
        async with get_redis_client() as redis_client:
            value = await redis_client.get(key)
            return value if value is not None else default
    
    @staticmethod
    async def set(key: str, value: str, expire: int = None):
        """Set value in cache with optional expiration"""
        async with get_redis_client() as redis_client:
            await redis_client.set(key, value, ex=expire)
    
    @staticmethod
    async def delete(key: str):
        """Delete key from cache"""
        async with get_redis_client() as redis_client:
            await redis_client.delete(key)
    
    @staticmethod
    async def exists(key: str) -> bool:
        """Check if key exists in cache"""
        async with get_redis_client() as redis_client:
            return bool(await redis_client.exists(key))
    
    @staticmethod
    async def increment(key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        async with get_redis_client() as redis_client:
            return await redis_client.incrby(key, amount)
    
    @staticmethod
    async def expire(key: str, seconds: int):
        """Set expiration for key"""
        async with get_redis_client() as redis_client:
            await redis_client.expire(key, seconds)


cache = CacheManager()