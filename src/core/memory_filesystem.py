"""
Memory Filesystem - Hierarchical memory management for TreeQuest AI agents
Implements memory-as-filesystem architecture with adaptive forgetting
"""

import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)

class MemoryNodeType(Enum):
    DIRECTORY = "directory"
    MEMORY = "memory"
    INSIGHT = "insight"
    PATTERN = "pattern"

@dataclass
class MemoryMetadata:
    created_at: float
    last_accessed: float
    access_count: int
    importance_score: float
    success_rate: float
    semantic_hash: str
    tags: List[str]
    agent_annotations: Dict[str, Any]
    performance_metrics: Dict[str, float]

@dataclass
class MemoryNode:
    path: str
    node_type: MemoryNodeType
    content: Any
    metadata: MemoryMetadata
    children: Dict[str, 'MemoryNode']
    parent: Optional['MemoryNode'] = None
    
    def add_child(self, name: str, child: 'MemoryNode'):
        """Add child node"""
        self.children[name] = child
        child.parent = self
    
    def get_child(self, name: str) -> Optional['MemoryNode']:
        """Get child by name"""
        return self.children.get(name)
    
    def get_path_segments(self) -> List[str]:
        """Get path as list of segments"""
        return [seg for seg in self.path.split('/') if seg]
    
    def update_access(self):
        """Update access metadata"""
        self.metadata.last_accessed = time.time()
        self.metadata.access_count += 1

class MemoryFilesystem:
    """Hierarchical memory filesystem for TreeQuest agents"""
    
    def __init__(self, memory_manager, base_path: Optional[Path] = None):
        self.memory_manager = memory_manager
        self.base_path = base_path or Path.home() / ".monk-memory" / "filesystem"
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize root node
        self.root = MemoryNode(
            path="/",
            node_type=MemoryNodeType.DIRECTORY,
            content={},
            metadata=MemoryMetadata(
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0,
                importance_score=1.0,
                success_rate=1.0,
                semantic_hash="",
                tags=["root"],
                agent_annotations={},
                performance_metrics={}
            ),
            children={}
        )
        
        # Initialize standard directory structure
        self._initialize_standard_structure()
        
        # Path success rate cache
        self.path_success_cache = {}
        
        # Load existing filesystem if available
        asyncio.create_task(self._load_filesystem())
    
    def _initialize_standard_structure(self):
        """Create standard directory structure"""
        standard_dirs = [
            "/agents/planner",
            "/agents/analyzer", 
            "/agents/critic",
            "/agents/synthesizer",
            "/agents/executor",
            "/patterns/successful_paths",
            "/patterns/failed_paths",
            "/insights/cross_agent",
            "/insights/provider_performance",
            "/session_data/trees",
            "/session_data/conversations",
            "/specializations/domains",
            "/specializations/task_types"
        ]
        
        for dir_path in standard_dirs:
            self._ensure_directory_exists(dir_path)
    
    def _ensure_directory_exists(self, path: str):
        """Ensure directory path exists"""
        segments = [seg for seg in path.split('/') if seg]
        current_node = self.root
        current_path = ""
        
        for segment in segments:
            current_path += f"/{segment}"
            if segment not in current_node.children:
                dir_node = MemoryNode(
                    path=current_path,
                    node_type=MemoryNodeType.DIRECTORY,
                    content={},
                    metadata=MemoryMetadata(
                        created_at=time.time(),
                        last_accessed=time.time(),
                        access_count=0,
                        importance_score=0.5,
                        success_rate=0.5,
                        semantic_hash="",
                        tags=["directory"],
                        agent_annotations={},
                        performance_metrics={}
                    ),
                    children={}
                )
                current_node.add_child(segment, dir_node)
            current_node = current_node.children[segment]
    
    def store_memory(self, path: str, content: Any, metadata: Dict[str, Any]) -> bool:
        """Store memory at specified path"""
        try:
            # Ensure parent directories exist
            parent_path = '/'.join(path.split('/')[:-1])
            if parent_path:
                self._ensure_directory_exists(parent_path)
            
            # Create memory metadata
            memory_metadata = MemoryMetadata(
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0,
                importance_score=metadata.get('importance_score', 0.5),
                success_rate=metadata.get('success_rate', 0.5),
                semantic_hash=self._generate_semantic_hash(content),
                tags=metadata.get('tags', []),
                agent_annotations=metadata.get('agent_annotations', {}),
                performance_metrics=metadata.get('performance_metrics', {})
            )
            
            # Create memory node
            memory_node = MemoryNode(
                path=path,
                node_type=MemoryNodeType.MEMORY,
                content=content,
                metadata=memory_metadata,
                children={}
            )
            
            # Navigate to parent and add memory
            parent_node = self._navigate_to_path(parent_path) if parent_path else self.root
            if parent_node:
                memory_name = path.split('/')[-1]
                parent_node.add_child(memory_name, memory_node)
                
                # Persist to disk
                asyncio.create_task(self._persist_memory(memory_node))
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error storing memory at {path}: {e}")
            return False
    
    def get_memory(self, path: str) -> Optional[MemoryNode]:
        """Get memory at specified path"""
        try:
            node = self._navigate_to_path(path)
            if node:
                node.update_access()
                return node
            return None
        except Exception as e:
            logger.error(f"Error getting memory at {path}: {e}")
            return None
    
    def get_successful_paths(self, task_signature: str, min_success_rate: float = 0.6) -> List[Dict[str, Any]]:
        """Get historically successful paths for similar tasks"""
        try:
            successful_paths = []
            patterns_node = self._navigate_to_path("/patterns/successful_paths")
            
            if patterns_node:
                for child_name, child_node in patterns_node.children.items():
                    if (child_node.metadata.success_rate >= min_success_rate and
                        self._is_similar_task(task_signature, child_node.content.get('task_signature', ''))):
                        
                        successful_paths.append({
                            'path': child_node.path,
                            'success_rate': child_node.metadata.success_rate,
                            'content': child_node.content,
                            'usage_count': child_node.metadata.access_count
                        })
            
            # Sort by success rate and usage count
            successful_paths.sort(
                key=lambda x: (x['success_rate'], x['usage_count']), 
                reverse=True
            )
            
            return successful_paths[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Error getting successful paths: {e}")
            return []
    
    def store_successful_path(self, task_signature: str, path_data: Dict[str, Any], 
                            success_rate: float):
        """Store a successful path pattern"""
        try:
            path_hash = hashlib.md5(task_signature.encode()).hexdigest()[:8]
            storage_path = f"/patterns/successful_paths/{path_hash}"
            
            content = {
                'task_signature': task_signature,
                'path_data': path_data,
                'recorded_at': time.time()
            }
            
            metadata = {
                'importance_score': min(success_rate + 0.2, 1.0),
                'success_rate': success_rate,
                'tags': ['successful_path', 'pattern'],
                'agent_annotations': {'pattern_type': 'successful_path'},
                'performance_metrics': {'success_rate': success_rate}
            }
            
            return self.store_memory(storage_path, content, metadata)
            
        except Exception as e:
            logger.error(f"Error storing successful path: {e}")
            return False
    
    def _navigate_to_path(self, path: str) -> Optional[MemoryNode]:
        """Navigate to specified path in filesystem"""
        if not path or path == "/":
            return self.root
            
        segments = [seg for seg in path.split('/') if seg]
        current_node = self.root
        
        for segment in segments:
            if segment in current_node.children:
                current_node = current_node.children[segment]
            else:
                return None
        
        return current_node
    
    def _generate_semantic_hash(self, content: Any) -> str:
        """Generate semantic hash for content"""
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _is_similar_task(self, task1: str, task2: str, similarity_threshold: float = 0.7) -> bool:
        """Check if two task signatures are similar"""
        # Simple similarity check - can be enhanced with embeddings
        if not task1 or not task2:
            return False
            
        task1_words = set(task1.lower().split())
        task2_words = set(task2.lower().split())
        
        if not task1_words or not task2_words:
            return False
            
        intersection = task1_words & task2_words
        union = task1_words | task2_words
        
        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity >= similarity_threshold
    
    async def _persist_memory(self, memory_node: MemoryNode):
        """Persist memory node to disk"""
        try:
            file_path = self.base_path / f"{memory_node.metadata.semantic_hash}.json"
            
            data = {
                'path': memory_node.path,
                'node_type': memory_node.node_type.value,
                'content': memory_node.content,
                'metadata': asdict(memory_node.metadata)
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error persisting memory: {e}")
    
    async def _load_filesystem(self):
        """Load existing filesystem from disk"""
        try:
            if not self.base_path.exists():
                return
                
            for file_path in self.base_path.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Recreate memory node
                    metadata = MemoryMetadata(**data['metadata'])
                    memory_node = MemoryNode(
                        path=data['path'],
                        node_type=MemoryNodeType(data['node_type']),
                        content=data['content'],
                        metadata=metadata,
                        children={}
                    )
                    
                    # Add to filesystem
                    parent_path = '/'.join(data['path'].split('/')[:-1])
                    parent_node = self._navigate_to_path(parent_path) if parent_path else self.root
                    if parent_node:
                        memory_name = data['path'].split('/')[-1]
                        parent_node.add_child(memory_name, memory_node)
                        
                except Exception as e:
                    logger.warning(f"Error loading memory file {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading filesystem: {e}")
    
    def adaptive_forget(self, forget_threshold: float = 0.3) -> int:
        """Remove low-value memories using adaptive forgetting"""
        forgotten_count = 0
        
        try:
            forgotten_count = self._recursive_forget(self.root, forget_threshold)
            logger.info(f"Adaptive forgetting removed {forgotten_count} low-value memories")
            
        except Exception as e:
            logger.error(f"Error during adaptive forgetting: {e}")
            
        return forgotten_count
    
    def _recursive_forget(self, node: MemoryNode, forget_threshold: float) -> int:
        """Recursively forget low-value memories"""
        forgotten_count = 0
        
        # Process children first
        children_to_remove = []
        for child_name, child_node in node.children.items():
            forgotten_count += self._recursive_forget(child_node, forget_threshold)
            
            # Check if child should be forgotten
            if self._should_forget_memory(child_node, forget_threshold):
                children_to_remove.append(child_name)
        
        # Remove low-value children
        for child_name in children_to_remove:
            del node.children[child_name]
            forgotten_count += 1
            
            # Also remove from disk
            child_node = node.children.get(child_name)
            if child_node:
                asyncio.create_task(self._remove_from_disk(child_node))
        
        return forgotten_count
    
    def _should_forget_memory(self, node: MemoryNode, forget_threshold: float) -> bool:
        """Determine if memory should be forgotten"""
        if node.node_type == MemoryNodeType.DIRECTORY:
            return False  # Don't forget directories
            
        # Calculate forget score based on multiple factors
        age_factor = self._calculate_age_factor(node)
        usage_factor = self._calculate_usage_factor(node)
        importance_factor = node.metadata.importance_score
        success_factor = node.metadata.success_rate
        
        # Weighted forget score (higher = more likely to forget)
        forget_score = (
            0.3 * age_factor +
            0.2 * (1 - usage_factor) +
            0.2 * (1 - importance_factor) +
            0.3 * (1 - success_factor)
        )
        
        return forget_score > forget_threshold
    
    def _calculate_age_factor(self, node: MemoryNode) -> float:
        """Calculate age factor (0=new, 1=old)"""
        age_seconds = time.time() - node.metadata.created_at
        age_days = age_seconds / (24 * 3600)
        
        # Exponential decay - memories become "old" after 30 days
        return min(1.0, age_days / 30.0)
    
    def _calculate_usage_factor(self, node: MemoryNode) -> float:
        """Calculate usage factor (0=unused, 1=heavily used)"""
        if node.metadata.access_count == 0:
            return 0.0
            
        # Normalize access count (assume max 100 accesses = heavily used)
        return min(1.0, node.metadata.access_count / 100.0)
    
    async def _remove_from_disk(self, node: MemoryNode):
        """Remove memory from disk"""
        try:
            file_path = self.base_path / f"{node.metadata.semantic_hash}.json"
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Error removing memory from disk: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory filesystem statistics"""
        stats = {
            'total_memories': 0,
            'total_directories': 0,
            'avg_importance': 0.0,
            'avg_success_rate': 0.0,
            'most_accessed_path': None,
            'oldest_memory': None,
            'newest_memory': None
        }
        
        memories = []
        self._collect_all_memories(self.root, memories)
        
        if memories:
            stats['total_memories'] = len([m for m in memories if m.node_type != MemoryNodeType.DIRECTORY])
            stats['total_directories'] = len([m for m in memories if m.node_type == MemoryNodeType.DIRECTORY])
            
            memory_nodes = [m for m in memories if m.node_type != MemoryNodeType.DIRECTORY]
            if memory_nodes:
                stats['avg_importance'] = sum(m.metadata.importance_score for m in memory_nodes) / len(memory_nodes)
                stats['avg_success_rate'] = sum(m.metadata.success_rate for m in memory_nodes) / len(memory_nodes)
                stats['most_accessed_path'] = max(memory_nodes, key=lambda m: m.metadata.access_count).path
                stats['oldest_memory'] = min(memory_nodes, key=lambda m: m.metadata.created_at).path
                stats['newest_memory'] = max(memory_nodes, key=lambda m: m.metadata.created_at).path
        
        return stats
    
    def _collect_all_memories(self, node: MemoryNode, memories: List[MemoryNode]):
        """Recursively collect all memory nodes"""
        memories.append(node)
        for child in node.children.values():
            self._collect_all_memories(child, memories)