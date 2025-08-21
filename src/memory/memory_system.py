"""
MONK CLI Persistent Memory System
Implements episodic, semantic, and procedural memory with intelligent retrieval
"""
import asyncio
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

from sentence_transformers import SentenceTransformer
import numpy as np
from ..core.config import config
from ..core.database import get_db_session, cache, get_pinecone_index
from ..core.models import EpisodicMemory, SemanticMemory, ProceduralMemory, User

logger = logging.getLogger(__name__)


@dataclass
class MemoryQuery:
    """Query for memory retrieval"""
    query_text: str
    user_id: str
    memory_types: List[str] = None  # ['episodic', 'semantic', 'procedural']
    context_filters: Dict[str, Any] = None
    limit: int = 10
    min_relevance_score: float = 0.3
    include_metadata: bool = True


@dataclass
class MemoryResult:
    """Result from memory retrieval"""
    memory_id: str
    memory_type: str
    content: Dict[str, Any]
    relevance_score: float
    importance_score: float
    created_at: datetime
    last_accessed: datetime
    access_count: int
    metadata: Dict[str, Any] = None


@dataclass
class MemoryInsight:
    """Insight derived from memory analysis"""
    insight_type: str
    description: str
    supporting_memories: List[str]
    confidence_score: float
    suggested_action: Optional[str] = None


class MemoryEmbeddingEngine:
    """Handles text embeddings for memory storage and retrieval"""
    
    def __init__(self):
        self.model = None
        self.model_name = "all-MiniLM-L6-v2"  # Lightweight, fast model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.model = None
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to vector embedding"""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        return self.model.encode(text)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch of texts to embeddings"""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        return self.model.encode(texts)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))


class EpisodicMemoryManager:
    """Manages episodic memories - specific events and interactions"""
    
    def __init__(self, embedding_engine: MemoryEmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.pinecone_namespace = "episodic"
    
    async def store_memory(
        self, 
        user_id: str, 
        memory_type: str, 
        content: Dict[str, Any],
        context: Dict[str, Any] = None,
        importance_score: float = 0.5,
        tags: List[str] = None
    ) -> str:
        """Store an episodic memory"""
        async with get_db_session() as session:
            # Create memory record
            memory = EpisodicMemory(
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                context=context or {},
                importance_score=importance_score,
                tags=tags or []
            )
            
            session.add(memory)
            await session.flush()  # Get the memory ID
            
            # Generate embedding
            memory_text = self._extract_searchable_text(content)
            embedding = self.embedding_engine.encode_text(memory_text)
            
            # Store in vector database
            try:
                pinecone_index = get_pinecone_index()
                pinecone_index.upsert(
                    vectors=[(
                        str(memory.id),
                        embedding.tolist(),
                        {
                            "user_id": user_id,
                            "memory_type": memory_type,
                            "importance_score": importance_score,
                            "created_at": memory.created_at.isoformat(),
                            "namespace": self.pinecone_namespace
                        }
                    )],
                    namespace=self.pinecone_namespace
                )
                logger.debug(f"Stored episodic memory {memory.id} in vector database")
            except Exception as e:
                logger.error(f"Failed to store memory in vector database: {e}")
            
            await session.commit()
            return str(memory.id)
    
    def _extract_searchable_text(self, content: Dict[str, Any]) -> str:
        """Extract searchable text from memory content"""
        searchable_parts = []
        
        # Common text fields
        text_fields = ["description", "task", "outcome", "error", "message", "text", "summary"]
        
        for field in text_fields:
            if field in content and isinstance(content[field], str):
                searchable_parts.append(content[field])
        
        # Handle nested content
        for key, value in content.items():
            if isinstance(value, str) and len(value) > 10:  # Meaningful text
                searchable_parts.append(value)
            elif isinstance(value, dict):
                nested_text = self._extract_searchable_text(value)
                if nested_text:
                    searchable_parts.append(nested_text)
        
        return " ".join(searchable_parts)
    
    async def retrieve_memories(
        self, 
        query: MemoryQuery,
        user_id: str = None
    ) -> List[MemoryResult]:
        """Retrieve relevant episodic memories"""
        user_id = user_id or query.user_id
        
        # Generate query embedding
        query_embedding = self.embedding_engine.encode_text(query.query_text)
        
        try:
            # Search vector database
            pinecone_index = get_pinecone_index()
            search_results = pinecone_index.query(
                vector=query_embedding.tolist(),
                top_k=query.limit * 2,  # Get more results to filter
                include_metadata=True,
                namespace=self.pinecone_namespace,
                filter={"user_id": {"$eq": user_id}}
            )
            
            memory_ids = [match["id"] for match in search_results["matches"] 
                         if match["score"] >= query.min_relevance_score]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            memory_ids = []
        
        # Retrieve from database
        results = []
        async with get_db_session() as session:
            for memory_id in memory_ids[:query.limit]:
                memory = await session.get(EpisodicMemory, memory_id)
                if memory:
                    # Update access tracking
                    memory.access_count += 1
                    memory.last_accessed = datetime.utcnow()
                    
                    # Calculate relevance score
                    relevance_score = self._calculate_memory_relevance(
                        memory, query, query_embedding
                    )
                    
                    result = MemoryResult(
                        memory_id=str(memory.id),
                        memory_type=memory.memory_type,
                        content=memory.content,
                        relevance_score=relevance_score,
                        importance_score=memory.importance_score,
                        created_at=memory.created_at,
                        last_accessed=memory.last_accessed,
                        access_count=memory.access_count,
                        metadata={
                            "tags": memory.tags,
                            "context": memory.context
                        } if query.include_metadata else None
                    )
                    results.append(result)
            
            await session.commit()
        
        # Sort by relevance and importance
        results.sort(key=lambda r: r.relevance_score * r.importance_score, reverse=True)
        
        return results[:query.limit]
    
    def _calculate_memory_relevance(
        self, 
        memory: EpisodicMemory, 
        query: MemoryQuery, 
        query_embedding: np.ndarray
    ) -> float:
        """Calculate relevance score for a memory"""
        # Base relevance from vector similarity (already calculated in Pinecone)
        base_relevance = 0.7  # Default assumption from vector match
        
        # Temporal relevance (recent memories more relevant)
        days_old = (datetime.utcnow() - memory.created_at).days
        temporal_factor = max(0.1, 1.0 - (days_old / 30.0))  # Decay over 30 days
        
        # Access frequency boost
        frequency_factor = min(1.0, memory.access_count / 10.0)  # Boost up to 10 accesses
        
        # Context matching
        context_factor = 1.0
        if query.context_filters and memory.context:
            matches = sum(1 for key, value in query.context_filters.items()
                         if memory.context.get(key) == value)
            context_factor = 1.0 + (matches * 0.2)  # 20% boost per match
        
        # Combined relevance
        relevance = base_relevance * temporal_factor * (1 + frequency_factor * 0.2) * context_factor
        
        return min(1.0, relevance)
    
    async def cleanup_old_memories(self, user_id: str, retention_days: int = None):
        """Clean up old, low-importance memories"""
        retention_days = retention_days or config.memory.memory_retention_days
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        async with get_db_session() as session:
            # Delete memories that are old AND have low importance/access
            memories_to_delete = await session.execute(
                """
                SELECT id FROM episodic_memories 
                WHERE user_id = :user_id 
                AND created_at < :cutoff_date 
                AND importance_score < 0.3 
                AND access_count <= 1
                """,
                {"user_id": user_id, "cutoff_date": cutoff_date}
            )
            
            for (memory_id,) in memories_to_delete.fetchall():
                # Delete from vector database
                try:
                    pinecone_index = get_pinecone_index()
                    pinecone_index.delete(ids=[str(memory_id)], namespace=self.pinecone_namespace)
                except Exception as e:
                    logger.error(f"Failed to delete memory {memory_id} from vector DB: {e}")
                
                # Delete from database
                memory = await session.get(EpisodicMemory, memory_id)
                if memory:
                    await session.delete(memory)
            
            await session.commit()


class SemanticMemoryManager:
    """Manages semantic memories - facts, patterns, and knowledge"""
    
    def __init__(self):
        pass
    
    async def extract_knowledge(self, episodic_memories: List[MemoryResult]) -> List[Dict[str, Any]]:
        """Extract semantic knowledge from episodic memories"""
        knowledge_facts = []
        
        # Simple pattern extraction for Phase 1
        # Phase 2 will use more sophisticated NLP
        
        for memory in episodic_memories:
            facts = self._extract_facts_from_memory(memory)
            knowledge_facts.extend(facts)
        
        return knowledge_facts
    
    def _extract_facts_from_memory(self, memory: MemoryResult) -> List[Dict[str, Any]]:
        """Extract facts from a single memory (simplified for Phase 1)"""
        facts = []
        content = memory.content
        
        # Extract tool preferences
        if "tool_used" in content and "outcome" in content:
            fact = {
                "knowledge_type": "preference",
                "subject": "user",
                "predicate": "prefers_tool",
                "object": content["tool_used"],
                "confidence_score": 0.7 if content.get("outcome") == "success" else 0.3,
                "source_memory_id": memory.memory_id
            }
            facts.append(fact)
        
        # Extract domain expertise patterns
        if "domain" in content and "success" in content:
            fact = {
                "knowledge_type": "expertise",
                "subject": "user",
                "predicate": "has_experience_in",
                "object": content["domain"],
                "confidence_score": 0.8 if content["success"] else 0.4,
                "source_memory_id": memory.memory_id
            }
            facts.append(fact)
        
        # Extract workflow patterns
        if "workflow_steps" in content:
            fact = {
                "knowledge_type": "pattern",
                "subject": "user",
                "predicate": "follows_workflow",
                "object": json.dumps(content["workflow_steps"]),
                "confidence_score": 0.6,
                "source_memory_id": memory.memory_id
            }
            facts.append(fact)
        
        return facts
    
    async def store_knowledge(self, user_id: str, knowledge_facts: List[Dict[str, Any]]):
        """Store extracted knowledge in semantic memory"""
        async with get_db_session() as session:
            for fact in knowledge_facts:
                # Check if similar fact exists
                existing = await session.execute(
                    """
                    SELECT id, confidence_score, source_count FROM semantic_memories 
                    WHERE user_id = :user_id 
                    AND subject = :subject 
                    AND predicate = :predicate 
                    AND object = :object
                    """,
                    {
                        "user_id": user_id,
                        "subject": fact["subject"],
                        "predicate": fact["predicate"],
                        "object": fact["object"]
                    }
                )
                
                existing_fact = existing.fetchone()
                
                if existing_fact:
                    # Update existing fact
                    semantic_memory = await session.get(SemanticMemory, existing_fact[0])
                    if semantic_memory:
                        # Weighted average of confidence scores
                        old_confidence = existing_fact[1]
                        new_confidence = fact["confidence_score"]
                        source_count = existing_fact[2] + 1
                        
                        semantic_memory.confidence_score = (
                            (old_confidence * (source_count - 1) + new_confidence) / source_count
                        )
                        semantic_memory.source_count = source_count
                        
                        # Add source memory ID
                        source_ids = semantic_memory.source_memory_ids or []
                        source_ids.append(fact["source_memory_id"])
                        semantic_memory.source_memory_ids = source_ids
                        
                        semantic_memory.updated_at = datetime.utcnow()
                else:
                    # Create new semantic memory
                    semantic_memory = SemanticMemory(
                        user_id=user_id,
                        knowledge_type=fact["knowledge_type"],
                        subject=fact["subject"],
                        predicate=fact["predicate"],
                        object=fact["object"],
                        confidence_score=fact["confidence_score"],
                        source_memory_ids=[fact["source_memory_id"]]
                    )
                    session.add(semantic_memory)
            
            await session.commit()


class ProceduralMemoryManager:
    """Manages procedural memories - learned procedures and workflows"""
    
    def __init__(self):
        pass
    
    async def learn_procedure(
        self, 
        user_id: str, 
        procedure_name: str,
        steps: List[Dict[str, Any]],
        success_outcome: bool,
        execution_time: float,
        context: Dict[str, Any] = None
    ):
        """Learn or update a procedure from successful task execution"""
        async with get_db_session() as session:
            # Check if procedure already exists
            existing = await session.execute(
                """
                SELECT id FROM procedural_memories 
                WHERE user_id = :user_id AND procedure_name = :procedure_name
                """,
                {"user_id": user_id, "procedure_name": procedure_name}
            )
            
            existing_procedure = existing.fetchone()
            
            if existing_procedure:
                # Update existing procedure
                procedure = await session.get(ProceduralMemory, existing_procedure[0])
                if procedure:
                    # Update success rate
                    old_rate = procedure.success_rate
                    old_usage = procedure.usage_frequency
                    new_usage = old_usage + 1
                    
                    if success_outcome:
                        new_rate = ((old_rate * old_usage) + 1.0) / new_usage
                    else:
                        new_rate = (old_rate * old_usage) / new_usage
                    
                    procedure.success_rate = new_rate
                    procedure.usage_frequency = new_usage
                    procedure.last_used = datetime.utcnow()
                    
                    # Update average execution time
                    if procedure.average_execution_time:
                        procedure.average_execution_time = (
                            (procedure.average_execution_time * old_usage + execution_time) / new_usage
                        )
                    else:
                        procedure.average_execution_time = execution_time
            else:
                # Create new procedure
                procedure = ProceduralMemory(
                    user_id=user_id,
                    procedure_name=procedure_name,
                    steps=steps,
                    success_rate=1.0 if success_outcome else 0.0,
                    usage_frequency=1,
                    average_execution_time=execution_time,
                    trigger_conditions=context or {},
                    domain_tags=self._extract_domain_tags(steps, context)
                )
                session.add(procedure)
            
            await session.commit()
    
    def _extract_domain_tags(self, steps: List[Dict[str, Any]], context: Dict[str, Any] = None) -> List[str]:
        """Extract domain tags from procedure steps and context"""
        tags = []
        
        # Extract from steps
        for step in steps:
            if "domain" in step:
                tags.append(step["domain"])
            if "tool" in step:
                tags.append(f"tool_{step['tool']}")
            if "action" in step:
                tags.append(f"action_{step['action']}")
        
        # Extract from context
        if context:
            if "project_type" in context:
                tags.append(context["project_type"])
            if "language" in context:
                tags.append(context["language"])
        
        return list(set(tags))  # Remove duplicates
    
    async def suggest_procedure(self, user_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Suggest a procedure based on current context"""
        async with get_db_session() as session:
            # Find procedures with matching context/tags
            procedures = await session.execute(
                """
                SELECT * FROM procedural_memories 
                WHERE user_id = :user_id 
                AND success_rate > 0.6 
                ORDER BY usage_frequency DESC, success_rate DESC 
                LIMIT 10
                """,
                {"user_id": user_id}
            )
            
            best_procedure = None
            best_score = 0.0
            
            for procedure_row in procedures.fetchall():
                procedure = dict(procedure_row)
                
                # Calculate context match score
                match_score = self._calculate_context_match(
                    procedure.get("trigger_conditions", {}), 
                    context
                )
                
                # Combined score
                total_score = (
                    match_score * 0.4 +
                    procedure["success_rate"] * 0.3 +
                    min(1.0, procedure["usage_frequency"] / 10.0) * 0.3
                )
                
                if total_score > best_score:
                    best_score = total_score
                    best_procedure = procedure
            
            if best_procedure and best_score > 0.5:
                return {
                    "procedure_name": best_procedure["procedure_name"],
                    "steps": best_procedure["steps"],
                    "confidence": best_score,
                    "success_rate": best_procedure["success_rate"],
                    "avg_execution_time": best_procedure["average_execution_time"]
                }
            
            return None
    
    def _calculate_context_match(self, trigger_conditions: Dict, current_context: Dict) -> float:
        """Calculate how well trigger conditions match current context"""
        if not trigger_conditions or not current_context:
            return 0.0
        
        total_conditions = len(trigger_conditions)
        matches = 0
        
        for key, value in trigger_conditions.items():
            if key in current_context and current_context[key] == value:
                matches += 1
        
        return matches / total_conditions if total_conditions > 0 else 0.0


class MemorySystem:
    """Main memory system coordinator"""
    
    def __init__(self):
        self.embedding_engine = MemoryEmbeddingEngine()
        self.episodic_manager = EpisodicMemoryManager(self.embedding_engine)
        self.semantic_manager = SemanticMemoryManager()
        self.procedural_manager = ProceduralMemoryManager()
    
    async def store_interaction(
        self, 
        user_id: str,
        interaction_type: str,
        content: Dict[str, Any],
        context: Dict[str, Any] = None,
        importance_score: float = 0.5
    ) -> str:
        """Store an interaction in memory system"""
        # Store in episodic memory
        memory_id = await self.episodic_manager.store_memory(
            user_id=user_id,
            memory_type=interaction_type,
            content=content,
            context=context,
            importance_score=importance_score
        )
        
        # Extract knowledge for semantic memory (async background task)
        asyncio.create_task(self._process_semantic_extraction(user_id, memory_id))
        
        # Learn procedures if it's a successful task
        if content.get("success") and "workflow_steps" in content:
            asyncio.create_task(
                self.procedural_manager.learn_procedure(
                    user_id=user_id,
                    procedure_name=content.get("task_type", "unknown"),
                    steps=content["workflow_steps"],
                    success_outcome=True,
                    execution_time=content.get("execution_time", 0.0),
                    context=context
                )
            )
        
        return memory_id
    
    async def _process_semantic_extraction(self, user_id: str, memory_id: str):
        """Background task to extract semantic knowledge"""
        try:
            # Get the memory
            query = MemoryQuery(
                query_text="",  # Not used for direct ID lookup
                user_id=user_id,
                limit=1
            )
            
            # For now, we'll extract from recent memories
            # Phase 2 will have more sophisticated extraction
            recent_memories = await self.episodic_manager.retrieve_memories(query, user_id)
            
            if recent_memories:
                knowledge_facts = await self.semantic_manager.extract_knowledge(recent_memories[:5])
                if knowledge_facts:
                    await self.semantic_manager.store_knowledge(user_id, knowledge_facts)
                    
        except Exception as e:
            logger.error(f"Semantic extraction failed for memory {memory_id}: {e}")
    
    async def retrieve_relevant_memories(self, query: MemoryQuery) -> Dict[str, List[MemoryResult]]:
        """Retrieve relevant memories from all memory types"""
        results = {}
        
        memory_types = query.memory_types or ["episodic", "semantic", "procedural"]
        
        if "episodic" in memory_types:
            results["episodic"] = await self.episodic_manager.retrieve_memories(query)
        
        # Phase 2 will add semantic and procedural retrieval
        if "semantic" in memory_types:
            results["semantic"] = []  # Placeholder
        
        if "procedural" in memory_types:
            results["procedural"] = []  # Placeholder
        
        return results
    
    async def get_memory_insights(self, user_id: str, domain: str = None) -> List[MemoryInsight]:
        """Generate insights from user's memory patterns"""
        insights = []
        
        # Simple insights for Phase 1
        # Phase 2 will have ML-driven insight generation
        
        async with get_db_session() as session:
            # Tool usage patterns
            tool_usage = await session.execute(
                """
                SELECT content->>'tool_used' as tool, COUNT(*) as usage_count,
                       AVG(CASE WHEN content->>'success' = 'true' THEN 1.0 ELSE 0.0 END) as success_rate
                FROM episodic_memories 
                WHERE user_id = :user_id 
                AND content->>'tool_used' IS NOT NULL
                GROUP BY content->>'tool_used'
                HAVING COUNT(*) >= 3
                ORDER BY usage_count DESC
                """,
                {"user_id": user_id}
            )
            
            for tool, count, success_rate in tool_usage.fetchall():
                if success_rate > 0.8:
                    insight = MemoryInsight(
                        insight_type="tool_expertise",
                        description=f"You have high success rate ({success_rate:.1%}) with {tool}",
                        supporting_memories=[],  # Would populate with actual memory IDs
                        confidence_score=min(1.0, count / 10.0),
                        suggested_action=f"Consider using {tool} for similar tasks"
                    )
                    insights.append(insight)
        
        return insights
    
    async def optimize_memory_performance(self, user_id: str):
        """Optimize memory system performance for user"""
        # Clean up old memories
        await self.episodic_manager.cleanup_old_memories(user_id)
        
        # Update importance scores based on access patterns
        await self._update_importance_scores(user_id)
        
        logger.info(f"Memory optimization completed for user {user_id}")
    
    async def _update_importance_scores(self, user_id: str):
        """Update memory importance scores based on access patterns"""
        async with get_db_session() as session:
            # Boost importance for frequently accessed memories
            await session.execute(
                """
                UPDATE episodic_memories 
                SET importance_score = LEAST(1.0, importance_score + (access_count * 0.1))
                WHERE user_id = :user_id 
                AND access_count > 5
                """,
                {"user_id": user_id}
            )
            
            # Decay importance for old, unaccessed memories
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            await session.execute(
                """
                UPDATE episodic_memories 
                SET importance_score = GREATEST(0.1, importance_score * 0.8)
                WHERE user_id = :user_id 
                AND last_accessed < :cutoff_date 
                AND access_count <= 2
                """,
                {"user_id": user_id, "cutoff_date": cutoff_date}
            )
            
            await session.commit()


# Global memory system instance
memory_system = MemorySystem()