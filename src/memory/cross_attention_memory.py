"""
MONK CLI Cross-Attention Memory Networks - Phase 2
Advanced memory retrieval with cross-attention mechanisms
"""
import numpy as np
import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import pickle
import hashlib

from ..core.config import config
from ..core.database import get_db_session
from ..core.models import User, EpisodicMemory, SemanticMemory, ProceduralMemory
from .memory_system import MemoryQuery, MemoryResult

logger = logging.getLogger(__name__)


@dataclass
class AttentionWeights:
    """Attention weights for memory retrieval"""
    query_memory_attention: np.ndarray  # Shape: (query_seq_len, memory_seq_len)
    memory_importance_weights: np.ndarray  # Shape: (num_memories,)
    cross_memory_correlations: np.ndarray  # Shape: (num_memories, num_memories)
    temporal_decay_factors: np.ndarray  # Shape: (num_memories,)
    relevance_scores: np.ndarray  # Shape: (num_memories,)


@dataclass 
class CrossAttentionResult:
    """Result from cross-attention memory retrieval"""
    memories: List[MemoryResult]
    attention_weights: AttentionWeights
    retrieval_reasoning: str
    confidence_score: float
    processing_time_ms: int
    memory_correlations: Dict[str, List[str]]
    expertise_insights: List[str]


class CrossAttentionEncoder(nn.Module):
    """Cross-attention encoder for memory retrieval"""
    
    def __init__(self, hidden_size: int = 768, num_attention_heads: int = 12, 
                 num_layers: int = 6, max_seq_length: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_seq_length = max_seq_length
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # Memory importance classifier
        self.importance_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Cross-memory correlation predictor
        self.correlation_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query_embeddings: torch.Tensor, 
                memory_embeddings: torch.Tensor,
                memory_masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-attention memory retrieval
        
        Args:
            query_embeddings: [batch_size, query_seq_len, hidden_size]
            memory_embeddings: [batch_size, num_memories, memory_seq_len, hidden_size]
            memory_masks: [batch_size, num_memories, memory_seq_len]
        
        Returns:
            attended_memories: [batch_size, num_memories, hidden_size]
            attention_weights: [batch_size, query_seq_len, total_memory_seq_len]
            importance_scores: [batch_size, num_memories]
        """
        batch_size, num_memories, memory_seq_len, hidden_size = memory_embeddings.shape
        query_seq_len = query_embeddings.shape[1]
        
        # Flatten memory embeddings for attention computation
        flat_memory_embeddings = memory_embeddings.view(batch_size, num_memories * memory_seq_len, hidden_size)
        
        # Cross-attention between query and memories
        attended_output = query_embeddings
        all_attention_weights = []
        
        for i, (attention_layer, layer_norm, ffn) in enumerate(zip(
            self.attention_layers, self.layer_norms, self.ffn_layers
        )):
            # Multi-head attention
            attn_output, attn_weights = attention_layer(
                attended_output,  # query
                flat_memory_embeddings,  # key
                flat_memory_embeddings,  # value
                key_padding_mask=memory_masks.view(batch_size, -1) if memory_masks is not None else None
            )
            
            # Residual connection and layer norm
            attended_output = layer_norm(attended_output + attn_output)
            
            # Feed-forward network
            ffn_output = ffn(attended_output)
            attended_output = layer_norm(attended_output + ffn_output)
            
            all_attention_weights.append(attn_weights)
        
        # Average attention weights across layers
        final_attention_weights = torch.mean(torch.stack(all_attention_weights), dim=0)
        
        # Compute memory-level representations
        memory_representations = []
        for mem_idx in range(num_memories):
            start_idx = mem_idx * memory_seq_len
            end_idx = (mem_idx + 1) * memory_seq_len
            
            # Average attention weights for this memory
            mem_attention = final_attention_weights[:, :, start_idx:end_idx].mean(dim=-1)  # [batch, query_seq_len]
            
            # Weight memory embeddings by attention
            weighted_memory = torch.matmul(
                mem_attention.unsqueeze(1),  # [batch, 1, query_seq_len]
                attended_output  # [batch, query_seq_len, hidden_size]
            ).squeeze(1)  # [batch, hidden_size]
            
            memory_representations.append(weighted_memory)
        
        attended_memories = torch.stack(memory_representations, dim=1)  # [batch, num_memories, hidden_size]
        
        # Compute importance scores for each memory
        importance_scores = self.importance_classifier(attended_memories).squeeze(-1)  # [batch, num_memories]
        
        return attended_memories, final_attention_weights, importance_scores
    
    def compute_memory_correlations(self, memory_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute correlations between memories"""
        batch_size, num_memories, hidden_size = memory_embeddings.shape
        correlations = torch.zeros(batch_size, num_memories, num_memories)
        
        for i in range(num_memories):
            for j in range(i + 1, num_memories):
                # Concatenate memory pairs
                mem_pair = torch.cat([memory_embeddings[:, i], memory_embeddings[:, j]], dim=-1)
                
                # Predict correlation
                correlation = self.correlation_predictor(mem_pair).squeeze(-1)
                correlations[:, i, j] = correlation
                correlations[:, j, i] = correlation  # Symmetric
        
        return correlations


class CrossAttentionMemoryRetrieval:
    """Advanced memory retrieval using cross-attention mechanisms"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.base_model = None
        self.cross_attention_model = None
        self.memory_embeddings_cache = {}
        self.expertise_tracker = {}
        
        # Performance thresholds
        self.min_attention_threshold = 0.1
        self.max_memories_per_query = 50
        self.embedding_cache_size = 1000
        
    async def initialize(self):
        """Initialize the cross-attention memory system"""
        try:
            # Load pre-trained language model
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.base_model = AutoModel.from_pretrained(model_name)
            self.base_model.to(self.device)
            self.base_model.eval()
            
            # Initialize cross-attention model
            self.cross_attention_model = CrossAttentionEncoder(
                hidden_size=384,  # MiniLM embedding size
                num_attention_heads=12,
                num_layers=4,
                max_seq_length=512
            )
            self.cross_attention_model.to(self.device)
            
            # Try to load pre-trained weights if available
            try:
                checkpoint = torch.load('cross_attention_memory.pth', map_location=self.device)
                self.cross_attention_model.load_state_dict(checkpoint['model_state'])
                logger.info("Loaded pre-trained cross-attention weights")
            except FileNotFoundError:
                logger.info("No pre-trained weights found, using random initialization")
            
            logger.info("Cross-attention memory system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cross-attention memory: {e}")
            raise
    
    async def retrieve_memories_with_attention(self, query: MemoryQuery) -> CrossAttentionResult:
        """Retrieve memories using cross-attention mechanism"""
        start_time = time.time()
        
        try:
            # Get candidate memories
            candidate_memories = await self._get_candidate_memories(query)
            
            if not candidate_memories:
                return CrossAttentionResult(
                    memories=[],
                    attention_weights=AttentionWeights(
                        query_memory_attention=np.array([]),
                        memory_importance_weights=np.array([]),
                        cross_memory_correlations=np.array([]),
                        temporal_decay_factors=np.array([]),
                        relevance_scores=np.array([])
                    ),
                    retrieval_reasoning="No candidate memories found",
                    confidence_score=0.0,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    memory_correlations={},
                    expertise_insights=[]
                )
            
            # Encode query and memories
            query_embedding = await self._encode_query(query)
            memory_embeddings, memory_texts = await self._encode_memories(candidate_memories)
            
            # Apply cross-attention
            with torch.no_grad():
                query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
                memory_tensor = torch.tensor(memory_embeddings, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                attended_memories, attention_weights, importance_scores = self.cross_attention_model(
                    query_tensor.unsqueeze(1),  # Add sequence dimension
                    memory_tensor.unsqueeze(2).unsqueeze(2)  # Add sequence dimensions
                )
                
                # Compute memory correlations
                correlations = self.cross_attention_model.compute_memory_correlations(attended_memories)
                
                # Convert to numpy
                attention_weights_np = attention_weights.cpu().numpy()[0]
                importance_scores_np = importance_scores.cpu().numpy()[0]
                correlations_np = correlations.cpu().numpy()[0]
            
            # Apply temporal decay
            temporal_factors = self._compute_temporal_decay(candidate_memories)
            
            # Compute final relevance scores
            relevance_scores = self._compute_final_relevance_scores(
                importance_scores_np, temporal_factors, candidate_memories, query
            )
            
            # Select top memories
            top_memory_indices = np.argsort(relevance_scores)[::-1][:query.limit]
            selected_memories = [candidate_memories[i] for i in top_memory_indices]
            
            # Generate retrieval reasoning
            reasoning = self._generate_retrieval_reasoning(
                query, selected_memories, importance_scores_np[top_memory_indices],
                attention_weights_np, correlations_np
            )
            
            # Extract memory correlations
            memory_correlations = self._extract_memory_correlations(
                selected_memories, correlations_np, top_memory_indices
            )
            
            # Generate expertise insights
            expertise_insights = await self._generate_expertise_insights(
                query, selected_memories, attention_weights_np
            )
            
            # Create attention weights object
            attention_weights_obj = AttentionWeights(
                query_memory_attention=attention_weights_np,
                memory_importance_weights=importance_scores_np[top_memory_indices],
                cross_memory_correlations=correlations_np[np.ix_(top_memory_indices, top_memory_indices)],
                temporal_decay_factors=temporal_factors[top_memory_indices],
                relevance_scores=relevance_scores[top_memory_indices]
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Update expertise tracking
            await self._update_expertise_tracking(query.user_id, selected_memories, relevance_scores[top_memory_indices])
            
            return CrossAttentionResult(
                memories=selected_memories,
                attention_weights=attention_weights_obj,
                retrieval_reasoning=reasoning,
                confidence_score=float(np.mean(relevance_scores[top_memory_indices])),
                processing_time_ms=processing_time,
                memory_correlations=memory_correlations,
                expertise_insights=expertise_insights
            )
            
        except Exception as e:
            logger.error(f"Cross-attention memory retrieval failed: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            
            return CrossAttentionResult(
                memories=[],
                attention_weights=AttentionWeights(
                    query_memory_attention=np.array([]),
                    memory_importance_weights=np.array([]),
                    cross_memory_correlations=np.array([]),
                    temporal_decay_factors=np.array([]),
                    relevance_scores=np.array([])
                ),
                retrieval_reasoning=f"Error in cross-attention retrieval: {str(e)}",
                confidence_score=0.0,
                processing_time_ms=processing_time,
                memory_correlations={},
                expertise_insights=[]
            )
    
    async def _get_candidate_memories(self, query: MemoryQuery) -> List[MemoryResult]:
        """Get candidate memories for attention-based ranking"""
        try:
            memories = []
            expanded_limit = min(query.limit * 5, self.max_memories_per_query)  # Get more candidates
            
            async with get_db_session() as session:
                # Get episodic memories
                if "episodic" in query.memory_types:
                    episodic_query = """
                        SELECT * FROM episodic_memories 
                        WHERE user_id = :user_id
                        AND (content->>'$.task_description' LIKE :query 
                             OR content->>'$.result_summary' LIKE :query
                             OR content->>'$.domain' LIKE :domain)
                        ORDER BY created_at DESC
                        LIMIT :limit
                    """
                    
                    result = await session.execute(episodic_query, {
                        "user_id": query.user_id,
                        "query": f"%{query.query_text}%",
                        "domain": f"%{query.query_text}%",
                        "limit": expanded_limit
                    })
                    
                    for row in result.fetchall():
                        memory = MemoryResult(
                            memory_id=row.id,
                            memory_type="episodic",
                            content=row.content or {},
                            relevance_score=0.5,  # Will be updated by attention
                            importance_score=row.importance_score or 0.5,
                            created_at=row.created_at,
                            last_accessed=row.last_accessed or row.created_at,
                            access_count=row.access_count or 0
                        )
                        memories.append(memory)
                
                # Get semantic memories
                if "semantic" in query.memory_types:
                    semantic_query = """
                        SELECT * FROM semantic_memories 
                        WHERE user_id = :user_id
                        AND (concept LIKE :query OR content LIKE :query)
                        ORDER BY confidence_score DESC
                        LIMIT :limit
                    """
                    
                    result = await session.execute(semantic_query, {
                        "user_id": query.user_id,
                        "query": f"%{query.query_text}%",
                        "limit": expanded_limit // 2
                    })
                    
                    for row in result.fetchall():
                        memory = MemoryResult(
                            memory_id=row.id,
                            memory_type="semantic",
                            content={"concept": row.concept, "content": row.content},
                            relevance_score=0.5,
                            importance_score=row.confidence_score or 0.5,
                            created_at=row.created_at,
                            last_accessed=row.last_accessed or row.created_at,
                            access_count=row.access_count or 0
                        )
                        memories.append(memory)
                
                # Get procedural memories
                if "procedural" in query.memory_types:
                    procedural_query = """
                        SELECT * FROM procedural_memories 
                        WHERE user_id = :user_id
                        AND (workflow_name LIKE :query OR steps LIKE :query)
                        ORDER BY success_rate DESC
                        LIMIT :limit
                    """
                    
                    result = await session.execute(procedural_query, {
                        "user_id": query.user_id,
                        "query": f"%{query.query_text}%",
                        "limit": expanded_limit // 2
                    })
                    
                    for row in result.fetchall():
                        memory = MemoryResult(
                            memory_id=row.id,
                            memory_type="procedural",
                            content={"workflow": row.workflow_name, "steps": row.steps},
                            relevance_score=0.5,
                            importance_score=row.success_rate or 0.5,
                            created_at=row.created_at,
                            last_accessed=row.last_accessed or row.created_at,
                            access_count=row.access_count or 0
                        )
                        memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Error getting candidate memories: {e}")
            return []
    
    async def _encode_query(self, query: MemoryQuery) -> np.ndarray:
        """Encode query text to embeddings"""
        try:
            # Create query text with context
            query_text = query.query_text
            if hasattr(query, 'context_filters') and query.context_filters:
                context_items = [f"{k}: {v}" for k, v in query.context_filters.items()]
                query_text += " " + " ".join(context_items)
            
            # Tokenize and encode
            tokens = self.tokenizer(
                query_text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            
            with torch.no_grad():
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                outputs = self.base_model(**tokens)
                
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.cpu().numpy()[0]
            
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            return np.zeros(384)  # Return zero embedding on error
    
    async def _encode_memories(self, memories: List[MemoryResult]) -> Tuple[np.ndarray, List[str]]:
        """Encode memories to embeddings"""
        try:
            memory_texts = []
            embeddings = []
            
            for memory in memories:
                # Create text representation of memory
                if memory.memory_type == "episodic":
                    text = f"Task: {memory.content.get('task_description', '')} "
                    text += f"Result: {memory.content.get('result_summary', '')} "
                    text += f"Agent: {memory.content.get('agent_used', '')}"
                elif memory.memory_type == "semantic":
                    text = f"Concept: {memory.content.get('concept', '')} "
                    text += f"Content: {memory.content.get('content', '')}"
                elif memory.memory_type == "procedural":
                    text = f"Workflow: {memory.content.get('workflow', '')} "
                    text += f"Steps: {memory.content.get('steps', '')}"
                else:
                    text = json.dumps(memory.content)
                
                memory_texts.append(text)
                
                # Check cache
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in self.memory_embeddings_cache:
                    embedding = self.memory_embeddings_cache[text_hash]
                else:
                    # Encode text
                    tokens = self.tokenizer(
                        text,
                        return_tensors='pt',
                        truncation=True,
                        padding=True,
                        max_length=512
                    )
                    
                    with torch.no_grad():
                        tokens = {k: v.to(self.device) for k, v in tokens.items()}
                        outputs = self.base_model(**tokens)
                        
                        # Use mean pooling
                        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                    
                    # Cache embedding
                    if len(self.memory_embeddings_cache) < self.embedding_cache_size:
                        self.memory_embeddings_cache[text_hash] = embedding
                
                embeddings.append(embedding)
            
            return np.array(embeddings), memory_texts
            
        except Exception as e:
            logger.error(f"Error encoding memories: {e}")
            return np.zeros((len(memories), 384)), []
    
    def _compute_temporal_decay(self, memories: List[MemoryResult]) -> np.ndarray:
        """Compute temporal decay factors for memories"""
        current_time = datetime.now()
        decay_factors = []
        
        for memory in memories:
            # Time since creation
            age_days = (current_time - memory.created_at).days
            
            # Time since last access
            last_access_days = (current_time - memory.last_accessed).days
            
            # Compute decay factor (0.0 to 1.0)
            # Recent memories and recently accessed memories get higher scores
            creation_factor = np.exp(-age_days / 30.0)  # 30-day half-life
            access_factor = np.exp(-last_access_days / 14.0)  # 14-day access half-life
            
            # Combined decay factor
            decay_factor = 0.6 * creation_factor + 0.4 * access_factor
            decay_factors.append(decay_factor)
        
        return np.array(decay_factors)
    
    def _compute_final_relevance_scores(self, importance_scores: np.ndarray, 
                                       temporal_factors: np.ndarray,
                                       memories: List[MemoryResult],
                                       query: MemoryQuery) -> np.ndarray:
        """Compute final relevance scores combining all factors"""
        # Base relevance from attention mechanism
        base_scores = importance_scores
        
        # Apply temporal decay
        temporal_scores = base_scores * temporal_factors
        
        # Access frequency boost
        access_counts = np.array([memory.access_count for memory in memories])
        max_access = max(access_counts) if len(access_counts) > 0 else 1
        access_scores = access_counts / max_access if max_access > 0 else np.zeros_like(access_counts)
        
        # Importance boost from original importance scores
        importance_boost = np.array([memory.importance_score for memory in memories])
        
        # Memory type preferences (if specified in query)
        type_scores = np.ones(len(memories))
        if hasattr(query, 'memory_type_preferences'):
            for i, memory in enumerate(memories):
                if memory.memory_type in query.memory_type_preferences:
                    type_scores[i] = query.memory_type_preferences[memory.memory_type]
        
        # Combined score with weights
        final_scores = (
            0.40 * temporal_scores +           # Temporal relevance and attention
            0.20 * access_scores +             # Access frequency
            0.20 * importance_boost +          # Original importance
            0.20 * type_scores                 # Type preferences
        )
        
        return final_scores
    
    def _generate_retrieval_reasoning(self, query: MemoryQuery, memories: List[MemoryResult],
                                    importance_scores: np.ndarray, attention_weights: np.ndarray,
                                    correlations: np.ndarray) -> str:
        """Generate human-readable reasoning for memory retrieval"""
        if not memories:
            return "No relevant memories found for the query."
        
        reasoning_parts = [
            f"Retrieved {len(memories)} memories using cross-attention analysis."
        ]
        
        # Analyze attention patterns
        avg_attention = np.mean(importance_scores)
        high_attention_count = np.sum(importance_scores > avg_attention * 1.2)
        
        if high_attention_count > 0:
            reasoning_parts.append(
                f"Found {high_attention_count} memories with high attention scores (>{avg_attention * 1.2:.2f})."
            )
        
        # Analyze memory types
        memory_types = {}
        for memory in memories:
            memory_types[memory.memory_type] = memory_types.get(memory.memory_type, 0) + 1
        
        if len(memory_types) > 1:
            type_desc = ", ".join([f"{count} {mem_type}" for mem_type, count in memory_types.items()])
            reasoning_parts.append(f"Memory types: {type_desc}.")
        
        # Analyze correlations
        if correlations.size > 0:
            high_correlation_pairs = np.sum(correlations > 0.7)
            if high_correlation_pairs > 0:
                reasoning_parts.append(f"Found {high_correlation_pairs} highly correlated memory pairs.")
        
        # Temporal analysis
        if memories:
            recent_count = sum(1 for m in memories if (datetime.now() - m.created_at).days < 7)
            if recent_count > 0:
                reasoning_parts.append(f"{recent_count} memories from the past week.")
        
        return " ".join(reasoning_parts)
    
    def _extract_memory_correlations(self, memories: List[MemoryResult], 
                                   correlations: np.ndarray,
                                   indices: np.ndarray) -> Dict[str, List[str]]:
        """Extract highly correlated memory pairs"""
        correlations_dict = {}
        
        if correlations.size == 0 or len(indices) < 2:
            return correlations_dict
        
        # Find high correlation pairs
        correlation_threshold = 0.7
        
        for i, idx_i in enumerate(indices):
            memory_i = memories[idx_i]
            correlated_memories = []
            
            for j, idx_j in enumerate(indices):
                if i != j and correlations[i, j] > correlation_threshold:
                    memory_j = memories[idx_j]
                    correlated_memories.append(memory_j.memory_id)
            
            if correlated_memories:
                correlations_dict[memory_i.memory_id] = correlated_memories
        
        return correlations_dict
    
    async def _generate_expertise_insights(self, query: MemoryQuery, 
                                         memories: List[MemoryResult],
                                         attention_weights: np.ndarray) -> List[str]:
        """Generate insights about user expertise based on memory patterns"""
        insights = []
        
        if not memories:
            return insights
        
        # Analyze domain expertise
        domains = {}
        for memory in memories:
            if memory.memory_type == "episodic":
                domain = memory.content.get("domain", "unknown")
                domains[domain] = domains.get(domain, 0) + 1
        
        if domains:
            top_domain = max(domains.keys(), key=domains.get)
            if domains[top_domain] >= 3:
                insights.append(f"Strong expertise pattern in {top_domain} domain ({domains[top_domain]} related memories)")
        
        # Analyze success patterns
        successful_tasks = 0
        total_episodic = 0
        
        for memory in memories:
            if memory.memory_type == "episodic":
                total_episodic += 1
                if memory.content.get("success", False):
                    successful_tasks += 1
        
        if total_episodic > 0:
            success_rate = successful_tasks / total_episodic
            if success_rate > 0.8:
                insights.append(f"High success rate ({success_rate:.1%}) in related tasks")
            elif success_rate < 0.5:
                insights.append(f"Room for improvement in this area (success rate: {success_rate:.1%})")
        
        # Analyze agent preferences
        agent_usage = {}
        for memory in memories:
            if memory.memory_type == "episodic":
                agent = memory.content.get("agent_used", "unknown")
                agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        if agent_usage:
            preferred_agent = max(agent_usage.keys(), key=agent_usage.get)
            if agent_usage[preferred_agent] >= 2:
                insights.append(f"Frequently uses {preferred_agent} agent for similar tasks")
        
        # Analyze learning progression
        if len(memories) >= 3:
            # Sort memories by creation time
            sorted_memories = sorted(memories, key=lambda m: m.created_at)
            recent_memories = sorted_memories[-3:]
            
            # Check if recent memories show improvement
            recent_success = sum(1 for m in recent_memories 
                               if m.memory_type == "episodic" and m.content.get("success", False))
            
            if recent_success >= 2:
                insights.append("Recent tasks show consistent successful outcomes")
        
        return insights
    
    async def _update_expertise_tracking(self, user_id: str, memories: List[MemoryResult], 
                                       relevance_scores: np.ndarray):
        """Update user expertise tracking based on retrieved memories"""
        try:
            if user_id not in self.expertise_tracker:
                self.expertise_tracker[user_id] = {
                    "domains": {},
                    "agents": {},
                    "success_patterns": {},
                    "last_updated": datetime.now()
                }
            
            user_expertise = self.expertise_tracker[user_id]
            
            # Update domain expertise
            for i, memory in enumerate(memories):
                relevance = relevance_scores[i]
                
                if memory.memory_type == "episodic":
                    domain = memory.content.get("domain", "unknown")
                    agent = memory.content.get("agent_used", "unknown")
                    success = memory.content.get("success", False)
                    
                    # Weight by relevance score
                    user_expertise["domains"][domain] = user_expertise["domains"].get(domain, 0) + relevance
                    user_expertise["agents"][agent] = user_expertise["agents"].get(agent, 0) + relevance
                    
                    if success:
                        user_expertise["success_patterns"][domain] = user_expertise["success_patterns"].get(domain, 0) + relevance
            
            user_expertise["last_updated"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating expertise tracking: {e}")
    
    async def get_user_expertise_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user's expertise profile based on memory patterns"""
        if user_id not in self.expertise_tracker:
            return {"domains": [], "agents": [], "expertise_level": "novice"}
        
        expertise = self.expertise_tracker[user_id]
        
        # Sort domains by expertise score
        top_domains = sorted(expertise["domains"].items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Sort agents by usage
        top_agents = sorted(expertise["agents"].items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Calculate overall expertise level
        total_expertise = sum(expertise["domains"].values())
        if total_expertise > 20:
            expertise_level = "expert"
        elif total_expertise > 10:
            expertise_level = "intermediate"
        elif total_expertise > 5:
            expertise_level = "beginner"
        else:
            expertise_level = "novice"
        
        return {
            "domains": [{"domain": domain, "score": score} for domain, score in top_domains],
            "agents": [{"agent": agent, "usage": usage} for agent, usage in top_agents],
            "expertise_level": expertise_level,
            "total_experience": total_expertise,
            "last_updated": expertise["last_updated"].isoformat()
        }
    
    def save_model_weights(self, filepath: str = "cross_attention_memory.pth"):
        """Save cross-attention model weights"""
        try:
            if self.cross_attention_model:
                torch.save({
                    'model_state': self.cross_attention_model.state_dict(),
                    'expertise_tracker': self.expertise_tracker,
                    'timestamp': datetime.now().isoformat()
                }, filepath)
                logger.info(f"Cross-attention model weights saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model weights: {e}")
    
    def clear_cache(self):
        """Clear embedding cache to free memory"""
        self.memory_embeddings_cache.clear()
        logger.info("Embedding cache cleared")


# Global instance
cross_attention_memory = CrossAttentionMemoryRetrieval()