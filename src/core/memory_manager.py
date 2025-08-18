"""
Memory Manager
Provides consistent, provider-agnostic memory for Monk CLI across sessions.

Memory types:
- facts: durable project/user facts and invariants
- preferences: user/provider/agent preferences
- tasks: orchestrator/task state snapshots

Design goals:
- Lightweight JSON persistence under ~/.monk-memory
- Simple tag-based retrieval and TTL/decay support
- Export compact context for LLMs (Gemma, TreeQuest)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    """A single memory entry."""
    kind: str  # 'fact' | 'preference' | 'task'
    key: str
    value: Any
    tags: List[str]
    importance: float = 0.5  # 0.0 - 1.0
    created_at: float = time.time()
    updated_at: float = time.time()
    ttl_seconds: Optional[int] = None  # None = no expiry

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(**data)


class MemoryManager:
    """
    Persistent memory store with simple scoring and export.
    """

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self.store_dir = store_dir or Path.home() / ".monk-memory"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.global_store_file = self.store_dir / "global_memory.json"

        self._items: List[MemoryItem] = []
        self._load()

    # CRUD
    def add(self, kind: str, key: str, value: Any, tags: Optional[List[str]] = None, importance: float = 0.5, ttl_seconds: Optional[int] = None) -> MemoryItem:
        existing = self.get(kind, key)
        if existing:
            existing.value = value
            existing.tags = list(sorted(set((existing.tags or []) + (tags or []))))
            existing.importance = max(existing.importance, importance)
            existing.updated_at = time.time()
            existing.ttl_seconds = ttl_seconds if ttl_seconds is not None else existing.ttl_seconds
            self._save()
            return existing

        item = MemoryItem(kind=kind, key=key, value=value, tags=tags or [], importance=importance, created_at=time.time(), updated_at=time.time(), ttl_seconds=ttl_seconds)
        self._items.append(item)
        self._save()
        return item

    def get(self, kind: str, key: str) -> Optional[MemoryItem]:
        for item in self._items:
            if item.kind == kind and item.key == key:
                if not self._is_expired(item):
                    return item
        return None

    def remove(self, kind: str, key: str) -> bool:
        before = len(self._items)
        self._items = [i for i in self._items if not (i.kind == kind and i.key == key)]
        changed = len(self._items) != before
        if changed:
            self._save()
        return changed

    def query(self, kinds: Optional[List[str]] = None, tags: Optional[List[str]] = None, limit: int = 20) -> List[MemoryItem]:
        results: List[MemoryItem] = []
        for item in self._items:
            if self._is_expired(item):
                continue
            if kinds and item.kind not in kinds:
                continue
            if tags and not set(tags).issubset(set(item.tags)):
                continue
            results.append(item)
        # Sort by importance desc, then recency
        results.sort(key=lambda i: (i.importance, i.updated_at), reverse=True)
        return results[:limit]

    # Convenience APIs
    def add_fact(self, key: str, value: Any, tags: Optional[List[str]] = None, importance: float = 0.6) -> MemoryItem:
        return self.add("fact", key, value, tags, importance)

    def add_preference(self, key: str, value: Any, tags: Optional[List[str]] = None, importance: float = 0.7) -> MemoryItem:
        return self.add("preference", key, value, tags, importance)

    def add_task_state(self, key: str, value: Any, tags: Optional[List[str]] = None, importance: float = 0.8, ttl_seconds: Optional[int] = 7 * 24 * 3600) -> MemoryItem:
        return self.add("task", key, value, tags, importance, ttl_seconds)

    def export_context(self, max_facts: int = 10, max_prefs: int = 10, max_tasks: int = 10) -> Dict[str, Any]:
        facts = [i.to_dict() for i in self.query(kinds=["fact"], limit=max_facts)]
        prefs = [i.to_dict() for i in self.query(kinds=["preference"], limit=max_prefs)]
        tasks = [i.to_dict() for i in self.query(kinds=["task"], limit=max_tasks)]
        return {
            "facts": facts,
            "preferences": prefs,
            "tasks": tasks,
        }

    # Internal
    def _is_expired(self, item: MemoryItem) -> bool:
        if item.ttl_seconds is None:
            return False
        return (time.time() - item.updated_at) > item.ttl_seconds

    def _load(self) -> None:
        if not self.global_store_file.exists():
            self._save()
            return
        try:
            with open(self.global_store_file, "r") as f:
                data = json.load(f)
            items_data = data.get("items", [])
            self._items = [MemoryItem.from_dict(d) for d in items_data]
        except Exception:
            self._items = []

    def _save(self) -> None:
        try:
            data = {"items": [i.to_dict() for i in self._items], "last_updated": time.time()}
            with open(self.global_store_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass


# Global memory manager instance
memory_manager = MemoryManager()


